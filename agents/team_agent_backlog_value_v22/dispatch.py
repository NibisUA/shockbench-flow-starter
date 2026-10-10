"""Optional route preferences for the baseline, using only public observations."""

import math
from collections import defaultdict

import numpy as np


class RoutePreferences:
    def __init__(self, config, network, *, transit_bias=0.0, cost_bias=0.0, air_fraction=1.0,
                 commodity_cost_bias=None, queue_bias=0.0, pending_bias=0.0, commodity_transit_bias=None,
                 warning_bias=0.0, warning_memory=0.0):
        for name, value in (("transit_bias", transit_bias), ("cost_bias", cost_bias), ("air_fraction", air_fraction),
                            ("queue_bias", queue_bias), ("pending_bias", pending_bias), ("warning_bias", warning_bias),
                            ("warning_memory", warning_memory)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
        if warning_memory >= 1:
            raise ValueError("warning_memory must be below one")
        self.network = network
        self.transit_bias, self.cost_bias, self.air_fraction = transit_bias, cost_bias, air_fraction
        self.queue_bias = queue_bias
        self.pending_bias = pending_bias
        self.warning_bias = warning_bias
        self.warning_memory, self.warning_history, self.warning_week = warning_memory, {}, None
        if warning_bias:
            self.warning_units = tuple((int(unit), row) for row, (kind, unit) in enumerate(
                config["layout"]["warning_units"]) if kind == "chokepoint")
        if queue_bias:
            self.chokes = tuple(config["layout"]["chokepoints"])
            keys = config["layout"].get("lot_keys")
            self.lot_keys = None if keys is None else tuple(map(tuple, keys))
            self.pools = tuple(config["static"]["commodities"]["pool"])
            self.horizon = int(config["T"])
        for key, mapping in (("commodity_cost_bias", commodity_cost_bias),
                             ("commodity_transit_bias", commodity_transit_bias)):
            if mapping is not None and not isinstance(mapping, dict):
                raise ValueError(f"{key} must be a mapping")
            for name, value in (mapping or {}).items():
                if (not isinstance(name, str) or not name or isinstance(value, bool)
                        or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
                        or name not in network.commodity_names and value != 0):
                    raise ValueError(f"{key} contains an invalid commodity or weight")
        self.commodity_cost_bias = dict(commodity_cost_bias or {})
        self.commodity_transit_bias = dict(commodity_transit_bias or {})
        groups = defaultdict(list)
        for route in network.routes:
            groups[route.source_node, route.destination_node, route.commodity_id].append(route.slot_id)
        self.groups = tuple(tuple(slots) for slots in groups.values() if len(slots) > 1)
        self.air = tuple(
            route.slot_id for route in network.routes
            if config["static"]["edges"]["id"][route.edge_id].startswith("air.")
        )

    def apply(self, flows, observation):
        if (not self.transit_bias and not self.cost_bias and self.air_fraction == 1
                and not self.commodity_cost_bias and not self.commodity_transit_bias
                and not self.queue_bias and not self.pending_bias and not self.warning_bias):
            return flows
        flows = flows.copy()
        tau, seen_tau = observation["graph_now.tau"], observation["graph_now.tau.observed"]
        cost, seen_cost = observation["graph_now.c"], observation["graph_now.c.observed"]
        queue_delay = self._queue_delay(observation) if self.queue_bias else {}
        pending = self._pending(observation) if self.pending_bias else {}
        warnings = {}
        if self.warning_bias:
            week = int(observation["week"][0]) if self.warning_memory else None
            if self.warning_memory:
                if self.warning_week is not None and week < self.warning_week:
                    self.warning_history.clear()
                self.warning_week = week
            for node, row in self.warning_units:
                if observation["warning.score.observed"][row]:
                    signal = float(observation["warning.score"][row])
                    if self.warning_memory:
                        previous = self.warning_history.get(node)
                        if previous is not None:
                            elapsed = week - previous[0]
                            weight = self.warning_memory ** elapsed
                            signal = weight * previous[1] + (1 - weight) * signal
                        self.warning_history[node] = week, signal
                    # S is a noisy hazard signal, not a calibrated probability.
                    warnings[node] = max(0.0, min(3.0, signal))
        for slots in self.groups:
            live = [slot for slot in slots if flows[slot] > 0]
            if len(live) < 2:
                continue
            # These are conditional no-queue distances, not guaranteed ETAs.
            distance = np.asarray([
                sum(float(tau[e]) if seen_tau[e] else self.network.edge_transit_weeks[e]
                    for e in self.network.routes[slot].edges)
                for slot in live
            ])
            freight = np.asarray([
                sum(float(cost[e]) for e in self.network.routes[slot].edges)
                if all(seen_cost[e] for e in self.network.routes[slot].edges)
                else self.network.routes[slot].nominal_freight_per_unit
                for slot in live
            ])
            positive = freight[freight > 0]
            scale = float(np.median(positive)) if positive.size else 1.0
            commodity = self.network.commodity_names[self.network.routes[live[0]].commodity_id]
            time_bias = self.commodity_transit_bias.get(commodity, self.transit_bias)
            exponent = time_bias * (distance - distance.min())
            if self.queue_bias:
                delay = np.asarray([sum(queue_delay.get((node, self.network.routes[slot].pool), 0.0)
                                        for node in self.network.routes[slot].chokepoints) for slot in live])
                exponent += self.queue_bias * (delay - delay.min())
            if self.pending_bias:
                risk = np.asarray([self._pending_risk(slot, pending, observation) for slot in live])
                exponent += self.pending_bias * (risk - risk.min())
            if self.warning_bias:
                signal = np.asarray([sum(warnings.get(node, 0.0) for node in self.network.routes[slot].chokepoints)
                                     for slot in live])
                exponent += self.warning_bias * (signal - signal.min())
            bias = self.commodity_cost_bias.get(commodity, self.cost_bias)
            exponent += bias * (freight - freight.min()) / scale
            flows[live] *= np.exp(-np.minimum(exponent, 700.0))
        if self.air_fraction != 1:
            flows[list(self.air)] *= self.air_fraction
        return flows

    @staticmethod
    def _pending(observation):
        rows = observation["pending_prohibitions.edge.observed"].astype(bool).copy()
        rows &= observation["pending_prohibitions.k.observed"].astype(bool)
        rows &= observation["pending_prohibitions.effective_week.observed"].astype(bool)
        pending = {}
        for row in np.flatnonzero(rows):
            pair = int(observation["pending_prohibitions.edge"][row]), int(observation["pending_prohibitions.k"][row])
            due = int(observation["pending_prohibitions.effective_week"][row])
            pending[pair] = min(due, pending.get(pair, due))
        return pending

    def _pending_risk(self, slot, pending, observation):
        route = self.network.routes[slot]
        departure, risk = int(observation["week"][0]), 0
        for pos, edge in enumerate(route.edges):
            # The first edge dispatches now. Later legs may meet an announced ban.
            if pos and departure >= pending.get((edge, route.commodity_id), math.inf):
                risk += 1
            departure += (float(observation["graph_now.tau"][edge]) if observation["graph_now.tau.observed"][edge]
                          else self.network.edge_transit_weeks[edge])
        return risk

    def _queue_delay(self, observation):
        totals = defaultdict(float)
        qty, seen = observation["queue_lots.qty"], observation["queue_lots.qty.observed"].astype(bool)
        if self.lot_keys is None:
            for row in np.flatnonzero(seen):
                if all(observation[f"queue_lots.{field}.observed"][row] for field in ("chokepoint", "k")):
                    node, k = (int(observation[f"queue_lots.{field}"][row]) for field in ("chokepoint", "k"))
                    totals[node, self.pools[k]] += float(qty[row])
        else:
            for row, (node, k, _lane, _edge) in enumerate(self.lot_keys):
                totals[node, self.pools[k]] += float(qty[row][seen[row]].sum())
        week = int(observation["week"][0])
        for row in np.flatnonzero(observation["pipeline.qty.observed"]):
            if (all(observation[f"pipeline.{field}.observed"][row] for field in ("edge", "k", "arrival_week"))
                    and int(observation["pipeline.arrival_week"][row]) == week):
                node = self.network.edge_head[int(observation["pipeline.edge"][row])]
                if node in self.chokes:
                    commodity = int(observation["pipeline.k"][row])
                    totals[node, self.pools[commodity]] += float(observation["pipeline.qty"][row])
        delay = {}
        for (node, pool), total in totals.items():
            key, row = f"graph_now.kappa.{pool}", self.chokes.index(node)
            if observation[key + ".observed"][row]:
                capacity = float(observation[key][row])
                # A current-load proxy, not a promise about future FIFO completion.
                delay[node, pool] = min(self.horizon, total / capacity) if capacity > 0 else float(self.horizon)
        return delay
