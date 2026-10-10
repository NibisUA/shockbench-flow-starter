"""Opt-in partial staging toward a future closed chokepoint, never a known ETA."""

import math


class ClosedRouteDispatch:
    def __init__(self, network, *, floor=0.0, minimum_lead=1):
        if (isinstance(floor, bool) or not isinstance(floor, (int, float))
                or not math.isfinite(floor) or not 0 <= floor <= 1):
            raise ValueError("closed_route_floor must be finite and in [0, 1]")
        if isinstance(minimum_lead, bool) or not isinstance(minimum_lead, int) or minimum_lead < 1:
            raise ValueError("closed_route_min_lead must be a positive integer")
        self.floor, self.minimum_lead, self.network = floor, minimum_lead, network
        self.prefixes = () if not floor else tuple(
            (route.slot_id, row, route.edges[:route.nodes.index(node)])
            for route in network.routes
            for node, row in zip(route.chokepoints, route.chokepoint_positions, strict=True)
            if node != route.source_node
        )

    def adjustments(self, observation):
        if not self.floor:
            return {}
        result = {}
        for slot, row, edges in self.prefixes:
            if not observation["graph_now.open.observed"][row]:
                continue
            if float(observation["graph_now.open"][row]) >= self.floor:
                continue
            # Current/no-queue lead is conditional; an earlier queue can add unknown waiting.
            lead = sum(float(observation["graph_now.tau"][edge])
                       if observation["graph_now.tau.observed"][edge]
                       else self.network.edge_transit_weeks[edge] for edge in edges)
            if lead >= self.minimum_lead:
                result[slot, row] = self.floor
        return result
