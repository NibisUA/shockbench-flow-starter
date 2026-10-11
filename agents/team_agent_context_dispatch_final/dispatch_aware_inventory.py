"""Free current output warehouse space before valuing additional raw inputs.

Dispatch precedes arrivals and production. Deduct only the incumbent's
requested exports that fit publicly visible source and first-edge resources.
This is a soft priority adjustment: source budgets and route caps still use
the original observed stocks and incumbent allocation rules.
"""
from collections import defaultdict

import numpy as np
from inventory_control import ProductionInventoryRebalancer


class DispatchAwareInventory(ProductionInventoryRebalancer):
    def apply(self, flows, observation):
        self.current_exports = {}
        self.last_released_headroom = 0.
        if not self.production_output_power:
            return super().apply(flows, observation)
        edge_requests = defaultdict(float)
        for r in self.network.routes:
            edge_requests[r.edge_id] += max(0., float(flows[r.slot_id]))
        for pair, _storage in set(self.production_outputs.values()):
            stock = self._stock(pair, observation)
            if stock is None or stock <= 0:
                continue
            slots = self.network.slots_from.get(pair, ())
            if any(not observation['graph_now.u.observed'][self.network.routes[s].edge_id]
                   for s in slots if flows[s] > 0):
                continue
            quantity = 0.
            for s in slots:
                r = self.network.routes[s]
                total = edge_requests[r.edge_id]
                cap = max(0., float(observation['graph_now.u'][r.edge_id]))
                quantity += max(0., float(flows[s])) * min(1., cap / total) if total > 0 else 0.
            self.current_exports[pair] = min(stock, quantity)
        self.last_released_headroom = sum(self.current_exports.values())
        return super().apply(flows, observation)

    def _output_wip(self, observation):
        pending = super()._output_wip(observation)
        adjusted = dict(pending)
        for pair, amount in getattr(self, 'current_exports', {}).items():
            adjusted[pair] = adjusted.get(pair, 0.) - amount
        return adjusted
