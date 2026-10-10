"""Conservative last-period manufacturing input bounds, disabled by default."""

import math
from collections import defaultdict

import numpy as np


class ProductionTail:
    def __init__(self, config, network, horizon, *, weeks=0):
        if weeks == "all":
            weeks = int(config["T"])
        elif isinstance(weeks, bool) or not isinstance(weeks, int) or not 0 <= weeks <= 8:
            raise ValueError("production_tail_weeks must be an integer from 0 to 8 or 'all'")
        self.weeks, self.last_limited = weeks, ()
        if not weeks:
            return
        if not horizon.enabled or not horizon.dispatch_boundaries:
            raise ValueError("production tail requires exact dispatch boundaries")
        self.network, self.T = network, int(config["T"])
        names = {name: n for n, name in enumerate(network.node_names)}
        goods = {name: k for k, name in enumerate(network.commodity_names)}
        units = config["static"]["units"]
        alpha_max = float(config["static"]["instance"].get("params", {}).get("alpha_max", 1.0))
        if not math.isfinite(alpha_max) or alpha_max < 1:
            raise ValueError("production tail requires a finite alpha_max >= 1")
        self.stocks = {tuple(pair): row for row, pair in enumerate(config["layout"]["stock_slots"])}
        rates, storage = {}, {}
        for node in config["static"]["instance"]["nodes"]:
            fab, osat = node.get("fab"), node.get("osat")
            recipes = {fab["input"]: fab["product"]} if fab else osat["packages"] if osat else {}
            for raw, product in recipes.items():
                if units[raw] == units[product]:
                    pair = names[node["id"]], goods[raw]
                    rates[pair] = float(fab["cap0"]) * alpha_max if fab else float(osat["thr"])
                    storage[pair] = float(node.get("stock", {}).get(raw, {}).get("storage", math.inf))
        self.groups = defaultdict(list)
        self.latest = {}
        self.rates = rates
        self.storage = storage
        for slot, delay in horizon.cutoffs:
            route = network.routes[slot]
            pair = route.destination_node, route.commodity_id
            if pair in rates and math.isfinite(delay):
                self.groups[pair].append(slot)
                latest = self.T - (delay - route.nominal_transit_weeks)
                self.latest[pair] = max(self.latest.get(pair, -math.inf), latest)
        self.lane_edges = frozenset(edge for path in network.lane_edges for edge in path)

    def apply(self, flows, observation):
        self.last_limited = ()
        if not self.weeks:
            return flows
        week = int(observation["week"][0])
        incoming = defaultdict(lambda: defaultdict(float))
        fields = ("qty", "edge", "k", "arrival_week")
        keys = tuple(f"pipeline.{field}" for field in fields)
        if all(key in observation and key + ".observed" in observation for key in keys):
            seen = np.logical_and.reduce([observation[key + ".observed"].astype(bool) for key in keys])
            for row in np.flatnonzero(seen):
                edge, k, due = (int(observation[f"pipeline.{field}"][row]) for field in ("edge", "k", "arrival_week"))
                node = self.network.edge_head[edge]
                pair = node, k
                if pair not in self.groups or not week <= due <= self.latest[pair]:
                    continue
                if edge in self.lane_edges:
                    if not observation["pipeline.lane.observed"][row]:
                        continue
                    progress = self.network.transit_progress(edge, int(observation["pipeline.lane"][row]))
                    if not progress.reaches_destination:
                        continue
                quantity = float(observation["pipeline.qty"][row])
                if math.isfinite(quantity) and quantity >= 0:
                    incoming[pair][due] += quantity
        out, limited = flows.copy(), []
        for pair, slots in self.groups.items():
            remaining = max(0.0, self.latest[pair] - week + 1)
            row = self.stocks[pair]
            if remaining > self.weeks or not observation["stock.qty.observed"][row]:
                continue
            stock = float(observation["stock.qty"][row])
            if not math.isfinite(stock) or stock < 0:
                continue
            exports = list(self.network.slots_from.get(pair, ()))
            stock = max(0.0, stock - float(flows[exports].sum()))
            # Late receipts cannot cover earlier starts. Consume existing inputs in calendar order.
            available, covered = stock, 0.0
            for due in range(week, int(self.latest[pair]) + 1):
                available += incoming[pair].get(due, 0.0)
                used = min(self.rates[pair], available)
                covered += used
                # The simulator disposes excess inventory after each week's production.
                available = min(self.storage[pair], available - used)
            # Include the public recovery ceiling; future actual throughput remains unknown.
            budget = max(0.0, remaining * self.rates[pair] - covered)
            requested = float(out[slots].sum())
            if requested > budget:
                out[slots] *= budget / requested
                limited.append(pair)
        self.last_limited = tuple(limited)
        return out
