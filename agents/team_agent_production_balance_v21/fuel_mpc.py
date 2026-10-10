"""Receding fuel plan: current marks persist as a labelled forecast, not truth."""

import math
from collections import defaultdict

import numpy as np
from network import NetworkTracker
from queue_forecast import QueueForecaster
from state import StateBuilder


class FuelMPC:
    def __init__(self, config, network, *, horizon=0, industry_bonus=0.0):
        if isinstance(horizon, bool) or not isinstance(horizon, int) or not 0 <= horizon <= 24:
            raise ValueError("fuel_mpc_horizon must be an integer in [0, 24]")
        if (isinstance(industry_bonus, bool) or not isinstance(industry_bonus, (int, float))
                or not math.isfinite(industry_bonus) or industry_bonus < 0):
            raise ValueError("industry_bonus must be finite and nonnegative")
        self.enabled = bool(horizon)
        self.last_status = "disabled"
        if not self.enabled:
            return
        self.horizon, self.industry_bonus = horizon, industry_bonus
        self.config, self.network = config, network
        self.tracker = NetworkTracker(config, network)
        fleet = QueueForecaster(config, max_weeks=horizon)
        self.queue_forecaster, self.state_builder = fleet, StateBuilder(config)
        self.fleet_terms, self.fleet_caps = fleet.fleet_terms, fleet.fleet_caps
        self.profiles = {network.node_names.index(n["id"]): n for n in config["static"]["instance"]["nodes"]}
        self.goods = {name: i for i, name in enumerate(network.commodity_names)}
        self.grids = tuple(config["layout"]["grids"])
        self.grid_rows = {node: row for row, node in enumerate(self.grids)}
        self.fuels = frozenset(self.goods[k] for node in self.grids
                               for k in self.profiles[node]["grid"]["shares"] if k in self.goods)
        self.routes = tuple(r for r in network.routes if r.commodity_id in self.fuels)
        pairs = {pair for r in self.routes for pair in ((r.source_node, r.commodity_id),
                                                       (r.destination_node, r.commodity_id))}
        self.pairs = tuple(sorted(pairs))
        self.stock_rows = {tuple(pair): row for row, pair in enumerate(config["layout"]["stock_slots"])}
        self.supply_rows = {tuple(pair): row for row, pair in enumerate(config["layout"]["supply_slots"])}
        self.storage = {pair: float(self.profiles[pair[0]]["stock"][network.commodity_names[pair[1]]]["storage"])
                        for pair in self.pairs}
        self.lane_edges = frozenset(edge for path in network.lane_edges for edge in path)
        self.psi = float(config["static"]["instance"]["params"]["psi"])
        self.chokes = {node: row for row, node in enumerate(network.chokepoints)}
        self.last_plan = None

    def _incoming(self, observation, week, horizon):
        incoming = defaultdict(float)
        for row in np.flatnonzero(observation["pipeline.qty.observed"]):
            if not all(observation[f"pipeline.{key}.observed"][row] for key in ("edge", "k", "arrival_week")):
                continue
            k = int(observation["pipeline.k"][row])
            if k not in self.fuels:
                continue
            edge = int(observation["pipeline.edge"][row])
            due = int(observation["pipeline.arrival_week"][row])
            if observation["pipeline.lane.observed"][row]:
                progress = self.network.transit_progress(edge, int(observation["pipeline.lane"][row]))
                if progress.remaining_chokepoints:
                    continue
                node, remaining = progress.destination_node, progress.remaining_edges
                if not all(observation["graph_now.tau.observed"][e] for e in remaining):
                    continue
                due += sum(int(observation["graph_now.tau"][e]) for e in remaining)
            elif edge not in self.lane_edges:
                node = self.network.edge_head[edge]
                if node in self.chokes:
                    continue
            else:
                continue
            if (node, k) in self.storage and week <= due < week + horizon:
                incoming[node, k, due - week] += float(observation["pipeline.qty"][row])
        return incoming

    def _grid(self, node, observation):
        row, grid = self.grid_rows[node], self.profiles[node]["grid"]
        values = []
        for name, nominal in (("G_bar", "deliverable"), ("y_bar", "base_load")):
            key = f"graph_now.grid.{name}"
            values.append(float(observation[key][row]) if observation[key + ".observed"][row] else float(grid[nominal]))
        return values

    def apply(self, flows, observation):
        if not self.enabled:
            return flows
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix

        snapshot = self.tracker.update(observation)
        week = snapshot.week
        horizon = min(self.horizon, self.config["T"] - week + 1)
        if not all(observation["stock.qty.observed"][self.stock_rows[pair]] for pair in self.pairs):
            self.last_status = "unobserved_stock_keep_control"
            self.last_plan = None
            return flows
        start = {pair: float(observation["stock.qty"][self.stock_rows[pair]]) for pair in self.pairs}
        incoming = self._incoming(observation, week, horizon)
        state = self.state_builder.build(observation, self.network)
        queue_forecast = self.queue_forecaster.forecast(state, observation, self.network)
        reserved_edges, reserved_chokes = defaultdict(float), defaultdict(float)
        sources = {lot.lot_id: lot for lot in (*state.pipeline, *state.queues)}
        for arrival in queue_forecast.arrivals:
            if (arrival.arrival_week is None or arrival.quantity.value is None
                    or arrival.commodity_id not in self.fuels):
                continue
            due = arrival.arrival_week - week
            if not 0 <= due < horizon:
                continue
            incoming[arrival.destination_node, arrival.commodity_id, due] += arrival.quantity.value
            source = sources[arrival.source_id]
            edge = self.network.lane_edges[source.lane_id][-1]
            departure = due - int(snapshot.fields["graph_now.tau"].values[edge])
            if departure >= 0:
                reserved_edges[edge, departure] += arrival.quantity.value
                node = self.network.edge_tail[edge]
                if node in self.chokes:
                    pool = self.config["static"]["commodities"]["pool"][arrival.commodity_id]
                    reserved_chokes[node, pool, departure] += arrival.quantity.value
        prices, bounds = [], []

        def variable(lower=0.0, upper=None, cost=0.0):
            pos = len(bounds)
            bounds.append((lower, upper))
            prices.append(cost)
            return pos

        x, stock, lift, burn, served, waste = {}, {}, {}, {}, {}, {}
        outgoing, arriving = defaultdict(dict), defaultdict(dict)
        edge_use, choke_use, fleet_use = defaultdict(dict), defaultdict(dict), defaultdict(dict)
        for r in self.routes:
            costs = sum(float(snapshot.fields["graph_now.c"].values[e]) for e in r.edges)
            delay, allowed = 0, bool(observation["action_mask"][r.slot_id])
            path = []
            for edge in r.edges:
                path.append((edge, delay))
                delay += int(snapshot.fields["graph_now.tau"].values[edge])
            for t in range(horizon):
                upper = (float(snapshot.fields["graph_now.u"].values[r.edge_id])
                         if allowed and t + delay < horizon else 0.0)
                pos = variable(upper=upper, cost=costs / 1e6)
                x[r.slot_id, t] = pos
                outgoing[r.source_node, r.commodity_id, t][pos] = 1.0
                if t + delay < horizon:
                    arriving[r.destination_node, r.commodity_id, t + delay][pos] = 1.0
                for edge, offset in path:
                    if t + offset >= horizon:
                        continue
                    edge_use[edge, t + offset][pos] = 1.0
                    node = self.network.edge_tail[edge]
                    if node in self.chokes:
                        choke_use[node, r.pool, t + offset][pos] = 1.0
                    weight = sum(delta for lane, delta in self.fleet_terms.get(edge, ())
                                 if lane is None or lane == r.lane_id)
                    if weight:
                        fleet_use[r.pool, t + offset][pos] = weight
        for pair in self.pairs:
            for t in range(horizon):
                stock[*pair, t] = variable(upper=self.storage[pair], cost=1e-5)
                if pair not in self.supply_rows:
                    penalty = (float(self.config["static"]["instance"]["params"]["upsilon"])
                               * float(self.config["static"]["commodities"]["v"][pair[1]]))
                    waste[*pair, t] = variable(cost=penalty / 1e6)
                if pair in self.supply_rows:
                    row = self.supply_rows[pair]
                    key = "graph_now.supply.avail"
                    available = float(observation[key][row]) if observation[key + ".observed"][row] else 0.0
                    lift[*pair, t] = variable(upper=available)
        grid_values = {}
        for node in self.grids:
            gbar, ybar = self._grid(node, observation)
            grid = self.profiles[node]["grid"]
            draw, profit = 0.0, 0.0
            for fi, fab_node in enumerate(self.config["layout"]["fabs"]):
                fab = self.profiles[fab_node]["fab"]
                if fab["grid"] != self.network.node_names[node]:
                    continue
                key = "graph_now.fab.alpha_bar"
                alpha = float(observation[key][fi]) if observation[key + ".observed"][fi] else 1.0
                draw += alpha * float(fab["cap0"]) * float(fab["e"])
                product = fab["product"].removesuffix("_raw")
                penalties = [float(pi) for k, pi in zip(self.config["static"]["sinks"]["k"],
                                                       self.config["static"]["sinks"]["pi"], strict=True)
                             if self.network.commodity_names[k] == product]
                profit += alpha * float(fab["cap0"]) * max(penalties, default=0.0)
            value = float(grid["voll"]) + self.industry_bonus * profit / max(1.0, ybar)
            grid_values[node] = gbar, ybar + draw, value
            for t in range(horizon):
                served[node, t] = variable(upper=ybar + draw, cost=-value / 1e6)
                for name, share in grid["shares"].items():
                    if name in self.goods and (node, self.goods[name]) in self.storage:
                        burn[node, self.goods[name], t] = variable(upper=max(0.0, float(share) * gbar))
        eq, eq_rhs, ub, ub_rhs = [], [], [], []

        def constraint(rows, rhs, terms, value):
            rows.append({index: coefficient for index, coefficient in terms.items() if coefficient})
            rhs.append(value)

        for node, k in self.pairs:
            for t in range(horizon):
                terms = {stock[node, k, t]: 1.0}
                if t:
                    terms[stock[node, k, t - 1]] = -1.0
                terms.update(outgoing[node, k, t])
                terms.update({v: -q for v, q in arriving[node, k, t].items()})
                if (node, k, t) in lift:
                    terms[lift[node, k, t]] = -1.0
                if (node, k, t) in burn:
                    terms[burn[node, k, t]] = 1.0
                if (node, k, t) in waste:
                    terms[waste[node, k, t]] = 1.0
                constraint(eq, eq_rhs, terms, (start[node, k] if not t else 0.0) + incoming[node, k, t])
                terms = dict(outgoing[node, k, t])
                if terms:
                    if t:
                        terms[stock[node, k, t - 1]] = -1.0
                    constraint(ub, ub_rhs, terms, start[node, k] if not t else 0.0)
        for node in self.grids:
            grid = self.profiles[node]["grid"]
            gbar, _load, _value = grid_values[node]
            for t in range(horizon):
                terms = {served[node, t]: 1.0}
                for k in self.fuels:
                    if (node, k, t) in burn:
                        terms[burn[node, k, t]] = -1.0
                constraint(ub, ub_rhs, terms, float(grid["shares"].get("unmodelled", 0.0)) * gbar)
                name = grid.get("rationed")
                if name in self.goods:
                    k = self.goods[name]
                    threshold = self.psi * float(grid["ibar"].get(name, 0.0))
                    if threshold > 0 and (node, k, t) in burn:
                        factor = float(grid["shares"][name]) * gbar / threshold
                        terms = {burn[node, k, t]: 1.0}
                        if t:
                            terms[stock[node, k, t - 1]] = -factor
                        constraint(ub, ub_rhs, terms, factor * start[node, k] if not t else 0.0)
        for (edge, t), terms in edge_use.items():
            constraint(ub, ub_rhs, terms,
                       max(0.0, float(snapshot.fields["graph_now.u"].values[edge]) - reserved_edges[edge, t]))
        for (node, pool, t), terms in choke_use.items():
            constraint(ub, ub_rhs, terms,
                       max(0.0, float(snapshot.fields[f"graph_now.kappa.{pool}"].values[self.chokes[node]])
                           - reserved_chokes[node, pool, t]))
        for (pool, _t), terms in fleet_use.items():
            constraint(ub, ub_rhs, terms, self.fleet_caps[pool])

        def matrix(rows):
            ri, ci, values = [], [], []
            for row, terms in enumerate(rows):
                for col, value in terms.items():
                    ri.append(row)
                    ci.append(col)
                    values.append(value)
            return coo_matrix((values, (ri, ci)), shape=(len(rows), len(bounds))).tocsr()

        solution = linprog(prices, A_ub=matrix(ub), b_ub=ub_rhs, A_eq=matrix(eq), b_eq=eq_rhs,
                           bounds=bounds, method="highs")
        if not solution.success:
            self.last_status = "forecast_infeasible_keep_control"
            self.last_plan = None
            return flows
        out = flows.copy()
        for route in self.routes:
            pos = x[route.slot_id, 0]
            out[route.slot_id] = min(bounds[pos][1], max(0.0, float(solution.x[pos])))
        self.last_status = "planned_current_marks_persist"
        self.last_plan = {"objective": float(solution.fun), "forecast_horizon": horizon,
                          "queue_issues": tuple(queue_forecast.issues)}
        return out
