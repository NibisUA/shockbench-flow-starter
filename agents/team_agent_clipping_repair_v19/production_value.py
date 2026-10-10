"""Propagate public sales value backward through manufacturing and transport.

This changes scarce-source priorities, never their stock budget or route caps.
Current marks and queue loads are conditional proxies, not future knowledge.
"""
import math

import numpy as np
from queue_outedge import OutEdgeQueueDelay


class ProductionValue:
    def __init__(self, config, network):
        self.network = network
        self.stocks = {tuple(pair): row for row, pair in enumerate(config['layout']['stock_slots'])}
        self.pairs = tuple(self.stocks)
        self.indices = {pair: i for i, pair in enumerate(self.pairs)}
        self.demands = {tuple(pair): row for row, pair in enumerate(config['layout']['demands'])}
        self.penalties = {pair: float(p) for pair, p in zip(self.demands, config['static']['sinks']['pi'], strict=True)}
        names = {name: n for n, name in enumerate(network.node_names)}
        goods = {name: k for k, name in enumerate(network.commodity_names)}
        nodes = {names[node['id']]: node for node in config['static']['instance']['nodes']}
        self.nominal = {pair: float(nodes[pair[0]]['sink']['demand'][network.commodity_names[pair[1]]]['dbar'])
                        for pair in self.demands}
        self.recipes, self.inputs = [], set()
        units = config['static']['units']
        for node, attrs in nodes.items():
            fab, osat = attrs.get('fab'), attrs.get('osat')
            recipes = {fab['input']: fab['product']} if fab else osat['packages'] if osat else {}
            for raw, packed in recipes.items():
                a, b = (node, goods[raw]), (node, goods[packed])
                if a in self.indices and b in self.indices and units[raw] == units[packed]:
                    # The completed product becomes dispatchable in the next week.
                    self.recipes.append((self.indices[a], self.indices[b], int((fab or osat)['tau']) + 1))
                    self.inputs.add(a)
        self.routes = tuple(r for r in network.routes
                            if (r.source_node, r.commodity_id) in self.indices
                            and (r.destination_node, r.commodity_id) in self.indices
                            and r.commodity_id in {k for _, k in self.inputs} | {k for _, k in self.demands})
        self.values = np.asarray(config['static']['commodities']['v'], dtype=float)
        self.delay = OutEdgeQueueDelay(config, network, enabled=True)
        self.current_observation, self.prices = None, {}
        self.horizon = int(config['T'])

    def begin(self, observation):
        if self.current_observation is observation:
            return
        self.current_observation = observation
        self.delay.begin(observation)
        week = int(observation['week'][0])
        remaining = max(0, self.horizon - week + 1)
        prices = np.zeros(len(self.pairs))
        for pair, row in self.demands.items():
            horizon = min(4, remaining)
            forecast = observation['demand_forecast.qty'][row]
            mask = observation['demand_forecast.qty.observed'][row]
            demand = sum(float(forecast[h]) if h < len(forecast) and mask[h] else self.nominal[pair]
                         for h in range(horizon))
            stock_row = self.stocks[pair]
            coverage = 0.0
            if observation['stock.qty.observed'][stock_row]:
                coverage = max(0.0, float(observation['stock.qty'][stock_row]))
                if observation['backlog.qty.observed'][row]:
                    coverage -= max(0.0, float(observation['backlog.qty'][row]))
            pressure = min(1.0, max(0.25, (demand - coverage) / max(1.0, demand)))
            prices[self.indices[pair]] = self.penalties[pair] * pressure
        arcs = []
        for a, b, delay in self.recipes:
            if delay <= remaining:
                arcs.append((a, b, math.exp(-0.05 * delay), 0.0))
        for route in self.routes:
            nominal = sum(float(observation['graph_now.tau'][e]) if observation['graph_now.tau.observed'][e]
                          else self.network.edge_transit_weeks[e] for e in route.edges)
            estimate = self.delay.route_delay(route, observation, nominal)
            delay = estimate['delay_weeks'] if estimate['delay_weeks'] is not None else nominal
            if delay > remaining:
                # Avoid hard assumptions about persistent closures. A nominal path retains a soft value.
                delay, readiness = nominal, 0.25
            else:
                readiness = 1.0
            if nominal > remaining:
                continue
            freight = sum(float(observation['graph_now.c'][e]) if observation['graph_now.c.observed'][e]
                          else route.nominal_freight_per_unit / len(route.edges) for e in route.edges)
            tariff = sum(float(observation['graph_now.tariff'][e, route.commodity_id])
                         if observation['graph_now.tariff.observed'][e, route.commodity_id] else 0.0 for e in route.edges)
            # Currently unavailable routes may reopen, so reduce their proxy value rather than deleting them.
            if not observation['action_mask'][route.slot_id]:
                readiness *= 0.25
            arcs.append((self.indices[route.source_node, route.commodity_id],
                         self.indices[route.destination_node, route.commodity_id],
                         readiness * math.exp(-min(700.0, 0.05 * delay)),
                         freight + tariff * self.values[route.commodity_id]))
        # Positive costs and discount <=1 preclude profitable cycles. Bellman relaxation
        # reaches each simple path in at most len(pairs)-1 passes without a wide LP.
        for _ in range(max(0, len(self.pairs) - 1)):
            changed = False
            for a, b, discount, cost in arcs:
                value = max(0.0, prices[b] * discount - cost)
                if value > prices[a] + 1e-9:
                    prices[a], changed = value, True
            if not changed:
                break
        self.prices = {pair: float(prices[self.indices[pair]]) for pair in self.inputs}

    def adjust(self, priorities, slots, observation):
        self.begin(observation)
        targets = [(self.network.routes[s].destination_node, self.network.routes[s].commodity_id) for s in slots]
        live = [self.prices[pair] for pair in targets if pair in self.prices]
        scale = max(live, default=0.0)
        if scale <= 0:
            return priorities
        # Retain diversification and uncertainty protection while rewarding reachable economic value.
        return [float(p) * (0.25 + 0.75 * self.prices[pair] / scale) if pair in self.prices else float(p)
                for pair, p in zip(targets, priorities, strict=True)]
