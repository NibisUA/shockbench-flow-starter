"""Observed FIFO workload proxy with persistent current throughput, not an ETA."""

from collections import defaultdict

import numpy as np


class QueueDelay:
    """Value congestion without simulating hidden future marks or reserving capacity.

    Kappa already includes the current open fraction. Out-edge bottlenecks,
    future disruptions and future dispatches are unknown; this fluid workload
    estimate is explicitly conditional rather than a completion guarantee.
    """

    def __init__(self, config, network, *, enabled=False):
        if not isinstance(enabled, bool):
            raise ValueError("sales_queue_delay_enabled must be a boolean")
        self.enabled, self.prepared, self.prepare_calls = enabled, False, 0
        if not enabled:
            return
        self.network = network
        self.pools = tuple(config["static"]["commodities"]["pool"])
        self.chokes = {node: row for row, node in enumerate(config["layout"]["chokepoints"])}
        self.lot_keys = config["layout"].get("lot_keys")

    @staticmethod
    def _seen(obs, key, index):
        return key in obs and key + ".observed" in obs and bool(
            np.broadcast_to(obs[key + ".observed"], np.shape(obs[key]))[index])

    def begin(self, observation):
        self.observation, self.prepared, self.prepare_calls = observation, False, 0

    def _blocked(self, obs, edge, k):
        key = "graph_now.prohibited"
        if self._seen(obs, key, (edge, k)) and obs[key][edge, k]:
            return True
        return self._seen(obs, "graph_now.u", edge) and float(obs["graph_now.u"][edge]) <= 0

    def _prepare(self, obs):
        self.queues, self.arrivals = defaultdict(float), defaultdict(list)
        self.unknown_pools, self.unknown_all = set(), False
        week = int(obs["week"][0])
        if "queue_lots.qty" not in obs or "queue_lots.qty.observed" not in obs:
            self.unknown_all = True
        else:
            qty = np.asarray(obs["queue_lots.qty"])
            seen = np.broadcast_to(obs["queue_lots.qty.observed"], qty.shape).astype(bool)
            if self.lot_keys is not None:
                for row, (node, k, _lane, edge) in enumerate(self.lot_keys):
                    amount = float(qty[row][seen[row]].sum())
                    if amount > 0 and not self._blocked(obs, edge, k):
                        self.queues[node, self.pools[k]] += amount
            else:
                for row in np.flatnonzero(seen):
                    amount = float(qty[row])
                    if amount <= 0:
                        continue
                    if not self._seen(obs, "queue_lots.k", row):
                        self.unknown_all = True
                        continue
                    pool = self.pools[int(obs["queue_lots.k"][row])]
                    if self._seen(obs, "queue_lots.next_edge", row):
                        edge = int(obs["queue_lots.next_edge"][row])
                        if self._blocked(obs, edge, int(obs["queue_lots.k"][row])):
                            continue
                    if not self._seen(obs, "queue_lots.chokepoint", row):
                        self.unknown_pools.add(pool)
                    else:
                        self.queues[int(obs["queue_lots.chokepoint"][row]), pool] += amount
        if "pipeline.qty" in obs and "pipeline.qty.observed" in obs:
            for row in np.flatnonzero(obs["pipeline.qty.observed"]):
                amount = float(obs["pipeline.qty"][row])
                if amount <= 0:
                    continue
                if not self._seen(obs, "pipeline.k", row):
                    self.unknown_all = True
                    continue
                pool = self.pools[int(obs["pipeline.k"][row])]
                if not self._seen(obs, "pipeline.edge", row):
                    self.unknown_pools.add(pool)
                    continue
                node = self.network.edge_head[int(obs["pipeline.edge"][row])]
                if node not in self.chokes:
                    continue
                if self._seen(obs, "pipeline.lane", row):
                    lane = int(obs["pipeline.lane"][row])
                    if 0 <= lane < len(self.network.lane_edges):
                        path = self.network.lane_edges[lane]
                        edge = int(obs["pipeline.edge"][row])
                        if edge in path and path.index(edge) + 1 < len(path):
                            next_edge = path[path.index(edge) + 1]
                            if self._blocked(obs, next_edge, int(obs["pipeline.k"][row])):
                                continue
                if not self._seen(obs, "pipeline.arrival_week", row):
                    self.unknown_pools.add(pool)
                    continue
                due = int(obs["pipeline.arrival_week"][row]) - week
                if due >= 0:
                    self.arrivals[node, pool].append((due, amount))
        for events in self.arrivals.values():
            events.sort()
        self.prepared, self.prepare_calls = True, self.prepare_calls + 1

    @staticmethod
    def _workload_wait(quantity, events, capacity, arrival_offset):
        """Drain only after inflow appears; current release precedes later arrivals."""
        backlog, previous = quantity, 0.0
        for due, amount in events:
            if due > arrival_offset:
                break
            backlog = max(0.0, backlog - capacity * (due - previous)) + amount
            previous = due
        backlog = max(0.0, backlog - capacity * max(0.0, arrival_offset - previous))
        return backlog / capacity

    def route_delay(self, route, observation, nominal_delay):
        """Return delay=None for unknown completion; never silently use ETA zero."""
        if not self.enabled:
            return {"delay_weeks": nominal_delay, "wait_weeks": 0.0, "status": "disabled"}
        if not any(self.network.edge_head[e] in self.chokes for e in route.edges):
            return {"delay_weeks": nominal_delay, "wait_weeks": 0.0, "status": "no_chokepoints"}
        if not all(self._seen(observation, "graph_now.tau", e) for e in route.edges):
            return {"delay_weeks": None, "wait_weeks": None, "status": "unknown_route_transit"}
        if not self.prepared:
            self._prepare(observation)
        pool = self.pools[route.commodity_id]
        if self.unknown_all or pool in self.unknown_pools:
            return {"delay_weeks": None, "wait_weeks": None, "status": "unknown_queue_workload"}
        transit, waiting = 0.0, 0.0
        for edge in route.edges:
            transit += float(observation["graph_now.tau"][edge])
            node = self.network.edge_head[edge]
            if node not in self.chokes:
                continue
            key, row = "graph_now.kappa." + pool, self.chokes[node]
            if not self._seen(observation, key, row):
                return {"delay_weeks": None, "wait_weeks": None, "status": "unobserved_queue_capacity"}
            capacity = float(observation[key][row])
            if capacity <= 0:
                return {"delay_weeks": None, "wait_weeks": None, "status": "unknown_completion_zero_capacity"}
            waiting += self._workload_wait(self.queues[node, pool], self.arrivals[node, pool],
                                          capacity, transit + waiting)
        return {"delay_weeks": transit + waiting, "wait_weeks": waiting,
                "status": "conditional_observed_queue_proxy"}
