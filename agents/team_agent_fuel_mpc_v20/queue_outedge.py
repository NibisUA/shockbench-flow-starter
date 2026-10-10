"""Conditional FIFO workload with the observed positive outgoing edge limit.

The pool and outgoing edge share the same lots, so waits combine with max,
not addition. Current limits persist only as a valuation assumption. Missing
edge-arrival metadata retains the existing pool-only estimate.
"""
import math
from collections import defaultdict

import numpy as np
from queue_delay import QueueDelay


class OutEdgeQueueDelay(QueueDelay):
    def _prepare(self, obs):
        super()._prepare(obs)
        self.edge_queues, self.edge_arrivals = defaultdict(float), defaultdict(list)
        self.edge_unknown_pools, self.edge_unknown_all = set(), False
        qty = np.asarray(obs["queue_lots.qty"])
        seen = np.broadcast_to(obs["queue_lots.qty.observed"], qty.shape).astype(bool)
        if self.lot_keys is not None:
            for row, (_node, k, _lane, edge) in enumerate(self.lot_keys):
                amount = float(qty[row][seen[row]].sum())
                if amount > 0 and not self._blocked(obs, edge, k):
                    self.edge_queues[edge] += amount
        else:
            for row in np.flatnonzero(seen):
                amount = float(qty[row])
                if amount <= 0:
                    continue
                if not self._seen(obs, "queue_lots.k", row):
                    self.edge_unknown_all = True
                    continue
                k = int(obs["queue_lots.k"][row])
                if not self._seen(obs, "queue_lots.next_edge", row):
                    self.edge_unknown_pools.add(self.pools[k])
                    continue
                edge = int(obs["queue_lots.next_edge"][row])
                if not self._blocked(obs, edge, k):
                    self.edge_queues[edge] += amount
        week = int(obs["week"][0])
        for row in np.flatnonzero(obs["pipeline.qty.observed"]):
            amount = float(obs["pipeline.qty"][row])
            if amount <= 0:
                continue
            if not self._seen(obs, "pipeline.k", row):
                self.edge_unknown_all = True
                continue
            k = int(obs["pipeline.k"][row])
            pool = self.pools[k]
            if not self._seen(obs, "pipeline.edge", row):
                self.edge_unknown_pools.add(pool)
                continue
            edge = int(obs["pipeline.edge"][row])
            if self.network.edge_head[edge] not in self.chokes:
                continue
            if not (self._seen(obs, "pipeline.lane", row)
                    and self._seen(obs, "pipeline.arrival_week", row)):
                self.edge_unknown_pools.add(pool)
                continue
            lane = int(obs["pipeline.lane"][row])
            if not 0 <= lane < len(self.network.lane_edges):
                self.edge_unknown_pools.add(pool)
                continue
            path = self.network.lane_edges[lane]
            if edge not in path or path.index(edge) + 1 >= len(path):
                self.edge_unknown_pools.add(pool)
                continue
            next_edge = path[path.index(edge) + 1]
            due = int(obs["pipeline.arrival_week"][row]) - week
            if due >= 0 and not self._blocked(obs, next_edge, k):
                self.edge_arrivals[next_edge].append((due, amount))
        for events in self.edge_arrivals.values():
            events.sort()

    def route_delay(self, route, observation, nominal_delay):
        base = super().route_delay(route, observation, nominal_delay)
        if base["status"] != "conditional_observed_queue_proxy":
            return base
        pool = self.pools[route.commodity_id]
        if self.edge_unknown_all or pool in self.edge_unknown_pools:
            return base
        transit, waiting = 0.0, 0.0
        for pos, edge in enumerate(route.edges):
            transit += float(observation["graph_now.tau"][edge])
            node = self.network.edge_head[edge]
            if node not in self.chokes:
                continue
            capacity = float(observation["graph_now.kappa." + pool][self.chokes[node]])
            pool_wait = self._workload_wait(
                self.queues[node, pool], self.arrivals[node, pool], capacity, transit + waiting)
            edge_wait = 0.0
            if pos + 1 < len(route.edges):
                next_edge = route.edges[pos + 1]
                if self._seen(observation, "graph_now.u", next_edge):
                    out_capacity = float(observation["graph_now.u"][next_edge])
                    if math.isfinite(out_capacity) and out_capacity > 0:
                        edge_wait = self._workload_wait(
                            self.edge_queues[next_edge], self.edge_arrivals[next_edge],
                            out_capacity, transit + waiting)
            waiting += max(pool_wait, edge_wait)
        return {"delay_weeks": transit + waiting, "wait_weeks": waiting,
                "status": "conditional_observed_pool_and_outedge_proxy"}
