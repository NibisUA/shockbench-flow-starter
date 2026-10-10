"""Shared receiver-coverage LP with public, conditional arrival estimates."""

import math
from collections import defaultdict

import numpy as np
from queue_delay import QueueDelay
from queue_outedge import OutEdgeQueueDelay


class SalesDispatch:
    def __init__(self, config, network, *, cover=0.0, discount=0.05, stage="sales", retain_dispatch=False,
                 queue_delay=False, outedge_delay=False):
        for name, value in (("cover", cover), ("discount", discount)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        self.enabled = bool(cover)
        if stage not in ("sales", "fab", "raw", "fuel"):
            raise ValueError("stage must be sales, fab, raw or fuel")
        self.stage = stage
        if not isinstance(retain_dispatch, bool):
            raise ValueError("retain_dispatch must be a boolean")
        self.retain_dispatch = retain_dispatch
        if not isinstance(queue_delay, bool):
            raise ValueError("sales_queue_delay_enabled must be a boolean")
        if not isinstance(outedge_delay, bool):
            raise ValueError("sales_outedge_delay_enabled must be a boolean")
        if outedge_delay and (not queue_delay or stage != "sales"):
            raise ValueError("sales outedge delay requires sales queue valuation")
        delay_type = OutEdgeQueueDelay if outedge_delay else QueueDelay
        self.queue_delay = delay_type(config, network, enabled=queue_delay and stage == "sales")
        self.last_delay_diagnostics = {}
        if not self.enabled:
            return
        self.cover, self.discount = cover, discount
        self.network, self.horizon = network, config["T"]
        self.stocks = {tuple(pair): row for row, pair in enumerate(config["layout"]["stock_slots"])}
        self.demands = {tuple(pair): row for row, pair in enumerate(config["layout"]["demands"])}
        nodes = {network.node_names.index(node["id"]): node for node in config["static"]["instance"]["nodes"]}
        self.nominal = {
            pair: float(nodes[pair[0]]["sink"]["demand"][network.commodity_names[pair[1]]]["dbar"])
            for pair in self.demands
        }
        self.penalties = {tuple(pair): float(pi) for pair, pi in zip(config["layout"]["demands"],
                                                                   config["static"]["sinks"]["pi"], strict=True)}
        self.targets, self.target_rates = dict(self.demands), {}
        self.production_times = {}
        self.target_stocks = {pair: (pair,) for pair in self.targets}
        self.storage = {}
        if stage != "sales":
            penalties = {}
            for (_node, k), penalty in self.penalties.items():
                penalties[k] = max(penalties.get(k, 0.0), penalty)
            goods = {name: k for k, name in enumerate(network.commodity_names)}
            self.targets, self.nominal, self.penalties = {}, {}, {}
            if stage == "fuel":
                from rebalance import StockRebalancer

                cover_model = StockRebalancer(config, network, fuel_power=1)
                self.targets = {pair: row for row, pair in enumerate(cover_model.fuel_rates)}
                self.nominal = dict(cover_model.fuel_rates)
                self.target_stocks = cover_model.fuel_consumers
                for pair, consumers in self.target_stocks.items():
                    self.storage[pair] = sum(float(nodes[n]["stock"][network.commodity_names[k]]["storage"])
                                             for n, k in consumers)
                    self.penalties[pair] = max(float(nodes[n]["grid"]["voll"])
                                              for n, _k in consumers if "grid" in nodes[n])
            elif stage == "fab":
                for row, node in enumerate(config["layout"]["fabs"]):
                    fab = nodes[node]["fab"]
                    pair = node, goods[fab["input"]]
                    self.targets[pair] = row
                    self.nominal[pair] = float(fab["cap0"])
                    self.target_rates[pair] = "graph_now.fab.cap_eff", row, 1.0
                    self.penalties[pair] = penalties.get(goods[fab["product"].removesuffix("_raw")], 0.0)
                    self.production_times[pair] = int(fab["tau"])
            else:
                for row, node in enumerate(config["layout"]["osats"]):
                    osat = nodes[node]["osat"]
                    total = sum(float(nodes[node]["stock"][raw]["storage"]) for raw in osat["packages"])
                    for raw, product in osat["packages"].items():
                        pair = node, goods[raw]
                        fraction = float(nodes[node]["stock"][raw]["storage"]) / total if total else 0.0
                        self.targets[pair] = row
                        self.nominal[pair] = float(osat["thr"]) * fraction
                        self.target_rates[pair] = "graph_now.osat.thr_eff", row, fraction
                        self.penalties[pair] = penalties.get(goods[product], 0.0)
                        self.production_times[pair] = int(osat["tau"])
            if stage != "fuel":
                self.target_stocks = {pair: (pair,) for pair in self.targets}
        self.slots = tuple(route.slot_id for route in network.routes
                           if (route.destination_node, route.commodity_id) in self.targets)
        self.lane_edges = frozenset(edge for path in network.lane_edges for edge in path)
        self.goods_value = config["static"]["commodities"]["v"]

    def _arrivals(self, observation):
        incoming = defaultdict(list)
        for row in np.flatnonzero(observation["pipeline.qty.observed"]):
            if not all(observation[f"pipeline.{name}.observed"][row] for name in ("edge", "k", "arrival_week")):
                continue
            edge = int(observation["pipeline.edge"][row])
            commodity = int(observation["pipeline.k"][row])
            due = int(observation["pipeline.arrival_week"][row])
            if observation["pipeline.lane.observed"][row]:
                progress = self.network.transit_progress(edge, int(observation["pipeline.lane"][row]))
                if progress.remaining_chokepoints:
                    continue
                destination, remaining = progress.destination_node, progress.remaining_edges
                if not all(observation["graph_now.tau.observed"][e] for e in remaining):
                    continue
                due += sum(int(observation["graph_now.tau"][e]) for e in remaining)
            elif edge not in self.lane_edges:
                destination = self.network.edge_head[edge]
                if destination in self.network.chokepoints:
                    continue
            else:
                continue
            pair = destination, commodity
            if pair in self.targets:
                incoming[pair].append((due, float(observation["pipeline.qty"][row])))
        return incoming

    def apply(self, flows, requests, observation):
        self.last_delay_diagnostics = {}
        if not self.enabled:
            return flows
        from scipy.optimize import linprog

        week = int(observation["week"][0])
        active = []
        for slot in self.slots:
            route = self.network.routes[slot]
            source = self.stocks[route.source_node, route.commodity_id]
            if (requests[slot] > 0 and observation["stock.qty.observed"][source]
                    and observation["graph_now.u.observed"][route.edge_id]):
                active.append(slot)
        if not active:
            return flows
        if self.queue_delay.enabled:
            self.queue_delay.begin(observation)
        incoming = self._arrivals(observation)
        bounds = np.asarray([requests[s] for s in active], dtype=float)
        times, objective = [], []
        sources, edges, receivers = defaultdict(list), defaultdict(list), defaultdict(list)
        for i, slot in enumerate(active):
            route = self.network.routes[slot]
            pair = route.destination_node, route.commodity_id
            sources[route.source_node, route.commodity_id].append(i)
            edges[route.edge_id].append(i)
            receivers[pair].append(i)
            transit = sum(float(observation["graph_now.tau"][e]) if observation["graph_now.tau.observed"][e]
                          else self.network.edge_transit_weeks[e] for e in route.edges)
            if self.queue_delay.enabled:
                source = self.stocks[route.source_node, route.commodity_id]
                if (float(observation["stock.qty"][source]) > 0
                        and float(observation["graph_now.u"][route.edge_id]) > 0):
                    diagnostic = self.queue_delay.route_delay(route, observation, transit)
                else:
                    diagnostic = {"delay_weeks": None, "wait_weeks": None,
                                  "status": "not_evaluated_no_current_resource"}
                diagnostic = {**diagnostic, "nominal_delay_weeks": transit,
                              "uses_champion_valuation": diagnostic["delay_weeks"] is None}
                self.last_delay_diagnostics[slot] = diagnostic
                if diagnostic["delay_weeks"] is not None:
                    transit = math.ceil(diagnostic["delay_weeks"])
            times.append(int(transit))
            freight = sum(float(observation["graph_now.c"][e]) if observation["graph_now.c.observed"][e]
                          else route.nominal_freight_per_unit / len(route.edges) for e in route.edges)
            tariff = sum(float(observation["graph_now.tariff"][e, route.commodity_id])
                         if observation["graph_now.tariff.observed"][e, route.commodity_id] else 0.0
                         for e in route.edges)
            benefit = self.penalties[pair] * math.exp(
                -min(700, self.discount * (transit + self.production_times.get(pair, 0)))
            )
            objective.append(benefit - freight - tariff * self.goods_value[route.commodity_id])
            if week + transit + self.production_times.get(pair, 0) > self.horizon:
                bounds[i] = 0
        rows, limits = [], []

        def constraint(ids, limit):
            row = np.zeros(len(active))
            row[ids] = 1
            rows.append(row)
            limits.append(max(0.0, limit))

        for pair, ids in sources.items():
            constraint(ids, float(observation["stock.qty"][self.stocks[pair]]))
        for edge, ids in edges.items():
            constraint(ids, float(observation["graph_now.u"][edge]))
        for pair, ids in receivers.items():
            stock_rows = [self.stocks[p] for p in self.target_stocks[pair]]
            if not all(observation["stock.qty.observed"][row] for row in stock_rows):
                continue
            stock = sum(float(observation["stock.qty"][row]) for row in stock_rows)
            if self.stage == "sales":
                row = self.demands[pair]
                forecast = observation["demand_forecast.qty"][row]
                seen = observation["demand_forecast.qty.observed"][row]
                if observation["backlog.qty.observed"][row]:
                    stock -= float(observation["backlog.qty"][row])
            elif self.stage == "fuel":
                forecast, seen = np.asarray([self.nominal[pair]]), np.ones(1)
            else:
                key, row, fraction = self.target_rates[pair]
                rate = (float(observation[key][row]) * fraction if observation[key + ".observed"][row]
                        else self.nominal[pair])
                forecast, seen = np.asarray([rate]), np.ones(1)
            # Bounds are forecasts, not promises that a sea queue will finish.
            for delay in sorted({times[i] for i in ids}):
                count = min(self.horizon - week + 1 - self.production_times.get(pair, 0),
                            delay + math.ceil(self.cover))
                demand = sum(float(forecast[h]) if h < len(forecast) and seen[h]
                             else float(forecast[0]) if self.stage != "sales" else self.nominal[pair]
                             for h in range(max(0, count)))
                arrived = sum(q for child in self.target_stocks[pair] for due, q in incoming[child]
                              if week <= due <= week + delay)
                if pair in self.storage:
                    demand = min(demand, self.storage[pair] + delay * self.nominal[pair])
                constraint([i for i in ids if times[i] <= delay], demand - stock - arrived)
        if not bounds.any():
            out = flows.copy()
            out[active] = 0
            return out
        objective = np.asarray(objective)
        matrix = np.asarray(rows)
        original_size = len(active)
        if self.retain_dispatch:
            prior = np.minimum(bounds, flows[active])
            bounds = np.concatenate((prior, np.maximum(0.0, bounds - prior)))
            matrix = np.concatenate((matrix, matrix), axis=1)
            objective = np.concatenate((1.25 * objective, objective))
        solution = linprog(-objective / max(1.0, float(np.abs(objective).max())),
                           A_ub=matrix, b_ub=limits,
                           bounds=np.column_stack((np.zeros(len(bounds)), bounds)), method="highs")
        if not solution.success:
            raise RuntimeError(f"sales dispatch optimization failed: {solution.message}")
        out = flows.copy()
        quantity = np.minimum(bounds, np.maximum(0.0, solution.x))
        out[active] = (quantity[:original_size] + quantity[original_size:] if self.retain_dispatch else quantity)
        return out
