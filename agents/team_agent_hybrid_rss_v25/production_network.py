"""Conditional throughput model for competing wafer destinations.

Only wafer dispatch is changed. Downstream flows in the LP price feasible
production; they are not actions or reservations. Current marks persist in this
explicit planning approximation. Sink forecasts and visible stock/WIP constrain
the useful output, so two Fabs cannot each claim the same downstream demand.
"""

import math
from collections import defaultdict

import numpy as np


class ProductionNetwork:
    def __init__(self, config, network, *, strength=0.0):
        if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not 0 <= strength <= 1:
            raise ValueError("production_network_strength must be in [0, 1]")
        self.strength = float(strength)
        self.network, self.config = network, config
        self.last_diagnostics = {}
        self.goods = {name: k for k, name in enumerate(network.commodity_names)}
        self.wafer = self.goods.get("wafer")
        self.pairs = tuple(map(tuple, config["layout"]["stock_slots"]))
        self.stock_rows = {pair: row for row, pair in enumerate(self.pairs)}
        self.nodes = {node: profile for node, profile in enumerate(config["static"]["instance"]["nodes"])}
        self.kinds = {k for name, k in self.goods.items() if name == "wafer" or name.startswith("chip_")}
        self.routes = tuple(route for route in network.routes if route.commodity_id in self.kinds)
        self.wafer_slots = tuple(route.slot_id for route in self.routes if route.commodity_id == self.wafer)
        self.supply_nodes = {node for node, _ in config["layout"]["supply_slots"]}
        self.fab_rows = {node: i for i, node in enumerate(config["layout"]["fabs"])}
        self.osat_rows = {node: i for i, node in enumerate(config["layout"]["osats"])}
        self.fleet = None
        if strength:
            from queue_forecast import QueueForecaster

            self.fleet = QueueForecaster(config, max_weeks=1)

    @staticmethod
    def _value(obs, key, row):
        if key not in obs or not obs[key + ".observed"][row]:
            return None
        value = float(obs[key][row])
        return value if math.isfinite(value) else None

    def apply(self, flows, obs):
        self.last_diagnostics = {"status": "disabled"}
        if not self.strength or not self.wafer_slots:
            return flows
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix

        week = int(obs["week"][0])
        left = self.config["T"] - week + 1
        live_pairs = {pair for pair in self.pairs if pair[1] in self.kinds}
        if any(self._value(obs, "stock.qty", self.stock_rows[pair]) is None for pair in live_pairs):
            self.last_diagnostics = {"status": "unknown_stock_keep_heuristic"}
            return flows
        pair_rows = {pair: i for i, pair in enumerate(sorted(live_pairs))}
        bounds, objective, incidence = [], [], []
        edge_use, production_use = defaultdict(dict), defaultdict(dict)
        current_wafer, source_use, fleet_use = {}, defaultdict(dict), defaultdict(dict)
        holding = 4.0  # visible inventory distributed over four planning weeks

        def var(cap, cost, terms):
            index = len(bounds)
            bounds.append((0, max(0.0, cap)))
            objective.append(cost)
            incidence.append(terms)
            return index

        for route in self.routes:
            times = [self._value(obs, "graph_now.tau", edge) for edge in route.edges]
            rates = [self._value(obs, "graph_now.u", edge) for edge in route.edges]
            if any(value is None for value in times + rates):
                continue
            transit, cap = sum(times), min(rates)
            if not obs["action_mask"][route.slot_id] or transit >= left:
                continue
            for pos in route.chokepoint_positions:
                opening = self._value(obs, "graph_now.open", pos)
                throughput = self._value(obs, f"graph_now.kappa.{route.pool}", pos)
                if opening is None or throughput is None:
                    cap = 0
                    break
                cap = min(cap, throughput)
            cost = 0.0
            for edge in route.edges:
                freight = self._value(obs, "graph_now.c", edge)
                tariff = self._value(obs, "graph_now.tariff", (edge, route.commodity_id))
                if freight is None or tariff is None:
                    cap = 0
                    break
                cost += freight + tariff * self.config["static"]["commodities"]["v"][route.commodity_id]
            if route.commodity_id == self.wafer:
                other = sum(float(flows[r.slot_id]) for r in self.network.routes
                            if r.edge_id == route.edge_id and r.commodity_id != self.wafer)
                cap = min(cap, max(0, rates[0] - other))
            # Positive transit charge breaks equivalent path ties and expresses
            # a soft cost of delayed output without calling it timely coverage.
            index = var(cap, cost + 20.0 * transit,
                        {(route.source_node, route.commodity_id): 1,
                         (route.destination_node, route.commodity_id): -1})
            for edge in route.edges:
                edge_use[edge][index] = 1.0
            if route.commodity_id == self.wafer:
                current_wafer[route.slot_id] = index
                source_use[route.source_node, route.commodity_id][index] = 1
                weight = sum(delta for lane, delta in self.fleet.fleet_terms.get(route.edge_id, ())
                             if lane is None or lane == route.lane_id)
                if weight:
                    fleet_use[route.pool][index] = weight
        for node, profile in self.nodes.items():
            fab, osat = profile.get("fab"), profile.get("osat")
            if fab:
                cap = self._value(obs, "graph_now.fab.cap_eff", self.fab_rows[node])
                recipes = {fab["input"]: fab["product"]}
                lead = int(fab["tau"]) + 1
            elif osat:
                cap = self._value(obs, "graph_now.osat.thr_eff", self.osat_rows[node])
                recipes, lead = osat["packages"], int(osat["tau"]) + 1
            else:
                continue
            if cap is None or lead >= left:
                continue
            for raw, output in recipes.items():
                if self.config["static"]["units"][raw] != self.config["static"]["units"][output]:
                    continue
                index = var(cap, 20.0 * lead, {(node, self.goods[raw]): 1, (node, self.goods[output]): -1})
                production_use[node][index] = 1
        # Exogenous inventory is a separate, single-use source. No pipeline
        # lot is also counted as on-hand; uncertain queue arrivals are excluded.
        stock = {pair: float(obs["stock.qty"][self.stock_rows[pair]]) for pair in live_pairs}
        for row in np.flatnonzero(obs["wip.qty.observed"]):
            if not all(obs[f"wip.{name}.observed"][row] for name in ("node", "k", "out_week")):
                continue
            due = int(obs["wip.out_week"][row])
            pair = int(obs["wip.node"][row]), int(obs["wip.k"][row])
            if pair in stock and week <= due < week + holding:
                stock[pair] += float(obs["wip.qty"][row])
        for pair, quantity in stock.items():
            cap = quantity if pair[0] in self.supply_nodes else quantity / holding
            var(cap, 0.0, {pair: -1})
        for row, pair in enumerate(map(tuple, self.config["layout"]["demands"])):
            if pair not in live_pairs:
                continue
            forecast = obs["demand_forecast.qty"][row]
            visible = obs["demand_forecast.qty.observed"][row].astype(bool)
            count = min(left, len(forecast))
            if not np.all(visible[:count]):
                continue
            rate = float(np.mean(forecast[:count]))
            backlog = self._value(obs, "backlog.qty", row)
            rate += max(0, backlog or 0) / 8.0
            var(rate, -float(self.config["static"]["sinks"]["pi"][row]), {pair: 1})

        ub, limits = [], []
        for edge, terms in edge_use.items():
            ub.append(terms)
            limits.append(float(obs["graph_now.u"][edge]))
        for node, terms in production_use.items():
            key, row = ("graph_now.fab.cap_eff", self.fab_rows[node]) if node in self.fab_rows else (
                "graph_now.osat.thr_eff", self.osat_rows[node])
            ub.append(terms)
            limits.append(float(obs[key][row]))
        for pair, terms in source_use.items():
            ub.append(terms)
            limits.append(float(obs["stock.qty"][self.stock_rows[pair]]))
        for pool, terms in fleet_use.items():
            # Do not borrow detour capacity already requested by other goods.
            spent = sum(float(flows[r.slot_id]) * sum(d for lane, d in self.fleet.fleet_terms.get(r.edge_id, ())
                                                     if lane is None or lane == r.lane_id)
                        for r in self.network.routes if r.pool == pool and r.commodity_id != self.wafer)
            ub.append(terms)
            limits.append(max(0, self.fleet.fleet_caps[pool] - spent))

        def matrix(rows, size):
            i, j, values = [], [], []
            for r, terms in enumerate(rows):
                for col, value in terms.items():
                    i.append(r)
                    j.append(col)
                    values.append(value)
            return coo_matrix((values, (i, j)), shape=(len(rows), size)).tocsr()

        balance = [dict() for _ in pair_rows]
        for col, terms in enumerate(incidence):
            for pair, coefficient in terms.items():
                balance[pair_rows[pair]][col] = coefficient
        result = linprog(np.asarray(objective) / max(1, max(abs(x) for x in objective)),
                         A_eq=matrix(balance, len(bounds)), b_eq=np.zeros(len(balance)),
                         A_ub=matrix(ub, len(bounds)), b_ub=limits, bounds=bounds, method="highs",
                         options={"time_limit": 0.5})
        if not result.success:
            self.last_diagnostics = {"status": f"solver_{result.status}_keep_heuristic"}
            return flows
        out = flows.copy()
        for slot in self.wafer_slots:
            planned = max(0.0, result.x[current_wafer[slot]]) if slot in current_wafer else 0.0
            out[slot] = (1 - self.strength) * flows[slot] + self.strength * planned
        self.last_diagnostics = {"status": "conditional_throughput", "variables": len(bounds),
                                 "changed_slots": int(np.count_nonzero(np.abs(out - flows) > 1e-8)),
                                 "strength": self.strength}
        return out
