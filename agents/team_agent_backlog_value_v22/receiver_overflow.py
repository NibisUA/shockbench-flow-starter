"""Bound direct manufacturing inputs using an optimistic storage balance.

Only committed, observed direct arrivals reduce headroom. Future production
uses its public maximum, so current outages cannot cause premature cuts.
Indirect queue shipments and outgoing or consumer input stocks are excluded.
"""
import math
from collections import defaultdict

import numpy as np


class ReceiverOverflow:
    def __init__(self, config, network, *, enabled=False):
        if not isinstance(enabled, bool):
            raise ValueError('receiver_overflow_enabled must be a boolean')
        self.enabled = enabled
        self.last_limited = ()
        if not enabled:
            return
        self.network, self.horizon = network, int(config['T'])
        self.stocks = {tuple(pair): row for row,pair in enumerate(config['layout']['stock_slots'])}
        demands = set(map(tuple,config['layout']['demands']))
        nodes = {name:i for i,name in enumerate(network.node_names)}
        goods = {name:i for i,name in enumerate(network.commodity_names)}
        alpha = float(config['static']['instance'].get('params',{}).get('alpha_max',1.0))
        if not math.isfinite(alpha) or alpha < 1:
            raise ValueError('receiver overflow needs a public alpha_max >= 1')
        self.inputs = {}
        for attrs in config['static']['instance']['nodes']:
            n = nodes[attrs['id']]
            fab,osat = attrs.get('fab'),attrs.get('osat')
            rates = {fab['input']: float(fab['cap0'])*alpha} if fab else {
                k:float(osat['thr']) for k in osat['packages']} if osat else {}
            for k,rate in rates.items():
                pair = n,goods[k]
                storage = float(attrs.get('stock',{}).get(k,{}).get('storage',math.inf))
                # A transfer or consumer could remove stock beyond manufacturing.
                if (pair in self.stocks and pair not in demands and not network.slots_from.get(pair)
                        and math.isfinite(storage) and storage>0 and math.isfinite(rate) and rate>=0):
                    self.inputs[pair] = storage,rate
        self.groups = defaultdict(list)
        for route in network.routes:
            pair = route.destination_node,route.commodity_id
            if (pair in self.inputs and len(route.edges)==1
                    and network.edge_head[route.edge_id]==pair[0]
                    and pair[0] not in network.chokepoints):
                delay = int(network.edge_transit_weeks[route.edge_id])
                self.groups[pair,delay].append(route.slot_id)
        self.lane_edges = frozenset(e for path in network.lane_edges for e in path)

    def apply(self, flows, observation):
        self.last_limited = ()
        if not self.enabled or not self.groups:
            return flows
        week = int(observation['week'][0])
        incoming = defaultdict(lambda: defaultdict(float))
        fields = tuple('pipeline.'+x for x in ('qty','edge','k','arrival_week'))
        if all(k in observation and k+'.observed' in observation for k in fields):
            visible = np.logical_and.reduce([observation[k+'.observed'].astype(bool) for k in fields])
            for row in np.flatnonzero(visible):
                edge = int(observation['pipeline.edge'][row])
                pair = self.network.edge_head[edge],int(observation['pipeline.k'][row])
                if pair not in self.inputs:
                    continue
                if edge in self.lane_edges:
                    if ('pipeline.lane' not in observation or 'pipeline.lane.observed' not in observation
                            or not observation['pipeline.lane.observed'][row]):
                        continue
                    lane = int(observation['pipeline.lane'][row])
                    if lane < 0 or not self.network.transit_progress(edge,lane).reaches_destination:
                        continue
                due = int(observation['pipeline.arrival_week'][row])
                q = float(observation['pipeline.qty'][row])
                if week<=due<=self.horizon and math.isfinite(q) and q>=0:
                    incoming[pair][due] += q
        out = None
        limited = []
        cohorts = defaultdict(dict)
        for (pair, delay), slots in self.groups.items():
            cohorts[pair][week + delay] = slots

        # Enforce one chronological inventory balance per receiver. Independent
        # per-ETA budgets let two routes each spend the same free storage.
        for pair, by_week in cohorts.items():
            row = self.stocks[pair]
            if not observation['stock.qty.observed'][row]:
                continue
            stock = float(observation['stock.qty'][row])
            if not math.isfinite(stock) or stock < 0:
                continue
            storage, rate = self.inputs[pair]
            for due in range(week, min(self.horizon, max(by_week)) + 1):
                committed = max(0.0, incoming[pair].get(due, 0.0))
                # At most `rate` can be consumed during this week's production;
                # the remainder must fit in storage.
                budget = max(0.0, storage + rate - stock - committed)
                slots = by_week.get(due, ())
                current = flows if out is None else out
                total = float(current[list(slots)].sum()) if slots else 0.0
                accepted = min(total, budget)
                if total > accepted + 1e-8:
                    if out is None:
                        out = flows.copy()
                    out[list(slots)] *= accepted / total if total > 0 else 0.0
                    limited.append({'pair': pair, 'arrival_week': due, 'removed': total - accepted})
                    current = out
                stock = min(storage, max(0.0, stock + committed + accepted - rate))
        self.last_limited = tuple(limited)
        return flows if out is None else out
