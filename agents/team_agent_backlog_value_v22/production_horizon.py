"""Optional optimistic nominal production-chain cutoff, not a queue ETA."""

import math


class ProductionHorizon:
    def __init__(self, config, network, *, enabled=False, dispatch_boundaries=False):
        if not isinstance(enabled, bool):
            raise ValueError("production_horizon_enabled must be a boolean")
        if not isinstance(dispatch_boundaries, bool):
            raise ValueError("dispatch_boundaries must be a boolean")
        self.dispatch_boundaries = dispatch_boundaries
        self.enabled, self.last_removed = enabled, ()
        if not enabled:
            return
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra

        self.horizon = int(config["T"])
        stocks = {tuple(pair): row for row, pair in enumerate(config["layout"]["stock_slots"])}
        goods = {name: k for k, name in enumerate(network.commodity_names)}
        names = {name: n for n, name in enumerate(network.node_names)}
        nodes = {names[node["id"]]: node for node in config["static"]["instance"]["nodes"]}
        edges, inputs = {}, set()

        def connect(source, target, delay):
            if source in stocks and target in stocks:
                key = stocks[source], stocks[target]
                edges[key] = min(edges.get(key, math.inf), float(delay))

        for route in network.routes:
            connect((route.source_node, route.commodity_id), (route.destination_node, route.commodity_id),
                    route.nominal_transit_weeks)
        for node, attrs in nodes.items():
            fab, osat = attrs.get("fab"), attrs.get("osat")
            recipes = ({fab["input"]: fab["product"]} if fab else osat["packages"] if osat else {})
            for raw, packed in recipes.items():
                inputs.add(goods[raw])
                # Newly completed production cannot dispatch until the following week.
                connect((node, goods[raw]), (node, goods[packed]), (fab or osat)["tau"] + int(dispatch_boundaries))
        targets = [stocks[tuple(pair)] for pair in config["layout"]["demands"] if tuple(pair) in stocks]
        if not targets or not edges or not inputs:
            self.enabled = False
            return
        graph = csr_matrix(([delay for delay in edges.values()],
                            ([source for source, _target in edges], [target for _source, target in edges])),
                           shape=(len(stocks), len(stocks)))
        distance = dijkstra(graph.T.tocsr(), directed=True, indices=targets, min_only=True)
        self.cutoffs = tuple((route.slot_id, route.nominal_transit_weeks
                              + distance[stocks[route.destination_node, route.commodity_id]])
                             for route in network.routes if route.commodity_id in inputs
                             and (route.destination_node, route.commodity_id) in stocks)

    def apply(self, flows, observation):
        self.last_removed = ()
        if not self.enabled:
            return flows
        week = int(observation["week"][0])
        removed = tuple(slot for slot, delay in self.cutoffs
                        if math.isfinite(delay) and week + delay > self.horizon and flows[slot] > 0)
        if not removed:
            return flows
        out = flows.copy()
        out[list(removed)] = 0
        self.last_removed = removed
        return out
