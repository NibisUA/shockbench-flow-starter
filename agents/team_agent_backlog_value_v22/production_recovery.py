"""Recover idle production transport without withdrawing existing dispatch.

Runtime imports are standard library, NumPy and SciPy only. Existing agent
modules supply public receiver rates, pipeline and conditional FIFO delays.
"""
import math
from collections import defaultdict

import numpy as np
from queue_outedge import OutEdgeQueueDelay
from sales_dispatch import SalesDispatch


class ProductionRecovery:
    def __init__(self, config, network, cover_model, *, enabled=False, cover=4.0):
        if not isinstance(enabled, bool):
            raise ValueError('production_recovery_enabled must be a boolean')
        if isinstance(cover, bool) or not isinstance(cover, (int, float)) or not math.isfinite(cover) or cover <= 0:
            raise ValueError('production_recovery_cover must be positive and finite')
        self.enabled, self.cover = enabled, float(cover)
        self.last_diagnostics = {}
        if not enabled:
            return
        self.network, self.horizon, self.cover_model = network, int(config['T']), cover_model
        self.stocks = {tuple(pair): row for row, pair in enumerate(config['layout']['stock_slots'])}
        self.models = [SalesDispatch(config, network, stage=stage, cover=cover) for stage in ('fab', 'raw')]
        self.targets = {pair: model for model in self.models for pair in model.targets}
        self.slots = tuple(r.slot_id for r in network.routes if (r.destination_node, r.commodity_id) in self.targets)
        self.delay = OutEdgeQueueDelay(config, network, enabled=True)
        nodes = {network.node_names.index(p['id']): p for p in config['static']['instance']['nodes']}
        self.storage = {pair: float(nodes[pair[0]]['stock'][network.commodity_names[pair[1]]]['storage'])
                        for pair in self.targets}
        self.values = np.asarray(config['static']['commodities']['v'], dtype=float)

    def apply(self, flows, requests, observation):
        self.last_diagnostics = {'enabled': self.enabled, 'added_quantity': 0.0}
        if not self.enabled:
            return flows
        from scipy.optimize import linprog

        self.delay.begin(observation)
        week = int(observation['week'][0])
        incoming = {}
        for model in self.models:
            incoming.update(model._arrivals(observation))
        active, delays, values, upper = [], [], [], []
        for slot in self.slots:
            route = self.network.routes[slot]
            source = route.source_node, route.commodity_id
            target = route.destination_node, route.commodity_id
            ids = list(self.network.slots_from[source])
            row = self.stocks[source]
            if not (observation['stock.qty.observed'][row] and observation['graph_now.u.observed'][route.edge_id]
                    and observation['stock.qty.observed'][self.stocks[target]]):
                continue
            spare_stock = max(0.0, float(observation['stock.qty'][row]) - float(flows[ids].sum()))
            edge_ids = [s for s in self.network.edges_to_slots[route.edge_id]
                        if self.network.routes[s].edge_id == route.edge_id]
            spare_edge = max(0.0, float(observation['graph_now.u'][route.edge_id]) - float(flows[edge_ids].sum()))
            cap = min(spare_stock, spare_edge, max(0.0, float(requests[slot] - flows[slot])))
            if cap <= 1e-8:
                continue
            nominal = sum(float(observation['graph_now.tau'][e]) if observation['graph_now.tau.observed'][e]
                          else self.network.edge_transit_weeks[e] for e in route.edges)
            estimate = self.delay.route_delay(route, observation, nominal)
            transit = math.ceil(estimate['delay_weeks'] if estimate['delay_weeks'] is not None else nominal)
            model = self.targets[target]
            if week + transit + model.production_times[target] > self.horizon:
                continue
            freight = sum(float(observation['graph_now.c'][e]) if observation['graph_now.c.observed'][e]
                          else route.nominal_freight_per_unit / len(route.edges) for e in route.edges)
            tariff = sum(
                float(observation['graph_now.tariff'][e, route.commodity_id])
                if observation['graph_now.tariff.observed'][e, route.commodity_id]
                else 0.0
                for e in route.edges
            )
            value = model.penalties[target] * math.exp(-min(700.0, 0.05 * (transit + model.production_times[target])))
            value -= freight + tariff * self.values[route.commodity_id]
            if value <= 0:
                continue
            active.append(slot)
            delays.append(transit)
            values.append(value)
            upper.append(cap)
        if not active:
            return flows
        sources, edges, receivers = defaultdict(list), defaultdict(list), defaultdict(list)
        for i, slot in enumerate(active):
            r = self.network.routes[slot]
            sources[r.source_node, r.commodity_id].append(i)
            edges[r.edge_id].append(i)
            receivers[r.destination_node, r.commodity_id].append(i)
        rows, limits = [], []

        def constraint(ids, amount):
            row = np.zeros(len(active))
            row[ids] = 1.0
            rows.append(row)
            limits.append(max(0.0, amount))

        for pair, ids in sources.items():
            used = float(flows[list(self.network.slots_from[pair])].sum())
            constraint(ids, float(observation['stock.qty'][self.stocks[pair]]) - used)
        for edge, ids in edges.items():
            edge_ids = [s for s in self.network.edges_to_slots[edge] if self.network.routes[s].edge_id == edge]
            constraint(ids, float(observation['graph_now.u'][edge]) - float(flows[edge_ids].sum()))
        for pair, ids in receivers.items():
            model = self.targets[pair]
            stock = max(0.0, float(observation['stock.qty'][self.stocks[pair]]))
            rate = self.cover_model.production_rate(pair, observation)
            all_target = [r.slot_id for r in self.network.routes if (r.destination_node, r.commodity_id) == pair]
            # Credit already-planned shipments only in constraints whose date
            # they can reach. Subtracting all dispatches from every prefix
            # incorrectly hides late arrivals from the earlier receiver window.
            committed_arrivals = []
            for slot in all_target:
                quantity = float(flows[slot])
                if quantity <= 0:
                    continue
                route = self.network.routes[slot]
                nominal = sum(
                    float(observation['graph_now.tau'][edge])
                    if observation['graph_now.tau.observed'][edge]
                    else self.network.edge_transit_weeks[edge]
                    for edge in route.edges
                )
                estimate = self.delay.route_delay(route, observation, nominal)
                transit = math.ceil(estimate['delay_weeks'] if estimate['delay_weeks'] is not None else nominal)
                committed_arrivals.append((week + transit, quantity))
            for delay in sorted(set(delays[i] for i in ids)):
                count = max(
                    0,
                    min(
                        self.horizon - week + 1 - model.production_times[pair],
                        delay + math.ceil(self.cover),
                    ),
                )
                need = min(rate * count, self.storage[pair] + rate * delay)
                arrived = sum(q for due, q in incoming.get(pair, ()) if week <= due <= week + delay)
                committed = sum(q for due, q in committed_arrivals if due <= week + delay)
                constraint([i for i in ids if delays[i] <= delay], need - stock - arrived - committed)
        objective = np.asarray(values)
        solution = linprog(-objective / max(1.0, float(objective.max())), A_ub=np.asarray(rows), b_ub=limits,
                           bounds=np.column_stack((np.zeros(len(active)), upper)), method='highs')
        if not solution.success:
            self.last_diagnostics['solver_status'] = int(solution.status)
            return flows
        quantities = np.minimum(upper, np.maximum(0.0, solution.x))
        # The LP cannot remove baseline cargo or borrow its source/entry capacity.
        out = flows.copy()
        out[active] += quantities
        by_commodity = defaultdict(float)
        for slot, qty in zip(active, quantities):
            by_commodity[self.network.commodity_names[self.network.routes[slot].commodity_id]] += float(qty)
        self.last_diagnostics.update(added_quantity=float(quantities.sum()), by_commodity=dict(by_commodity),
                                     active_slots=len(active), changed_slots=int(np.count_nonzero(quantities > 1e-8)))
        return out
