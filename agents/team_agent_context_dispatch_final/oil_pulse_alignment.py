"""Conservative two-week oil timing, using public stocks and scheduled arrivals.

The model considers one oil hold followed by normal local fuel dispatch. It
never increases an action, changes LNG, or imports the environment. Current
network marks are conditional estimates for next week, not future knowledge.
"""
from collections import defaultdict

import numpy as np

from causal_fuel_features import CausalFuelFeatures


class OilPulseAlignment(CausalFuelFeatures):
    def __init__(self, config, network, batch):
        super().__init__(config, network, batch)
        self.batch = batch
        self.last_hold = {}
        self.last_week = 0
        self.last_decisions = []

    def _capacity(self, obs, slots):
        edges = {self.network.routes[s].edge_id for s in slots if obs['action_mask'][s]}
        if not edges or any(self._value(obs, 'graph_now.u', edge) is None for edge in edges):
            return 0.
        if any(self.network.routes[s].chokepoints or any(
                self._value(obs, 'graph_now.tau', e) != 0 for e in self.network.routes[s].edges)
               for s in slots):
            return 0.
        return sum(float(obs['graph_now.u'][edge]) for edge in edges)

    def _evaluate(self, obs, action, node, incoming, planned, hold):
        grid, row = self.nodes[node]['grid'], self.grid_rows[node]
        generation = self._value(obs, 'graph_now.grid.G_bar', row)
        load = self._value(obs, 'graph_now.grid.y_bar', row)
        if generation is None or load is None or generation <= 0:
            return None
        origin, slots, gas_origin, gas_slots = self.controls[node]
        oil, gas = self.goods['crude'], self.goods['lng']
        fuel_ids = [self.goods[name] for name in grid['shares'] if name in self.goods]
        stocks = {k: self._stock(obs, (node, k)) for k in fuel_ids}
        terminals = {oil: self._stock(obs, (origin, oil)), gas: self._stock(obs, (gas_origin, gas))}
        if any(value is None for value in (*stocks.values(), *terminals.values())):
            return None
        origins, local_slots = {oil: origin, gas: gas_origin}, {oil: slots, gas: gas_slots}
        caps = {k: self._capacity(obs, local_slots[k]) for k in (oil, gas)}
        rates = {k: generation * float(grid['shares'][self.network.commodity_names[k]]) for k in fuel_ids}
        reserve = self.psi * float(grid['ibar'].get('lng', 0))
        free = generation * sum(float(share) for name, share in grid['shares'].items() if name not in self.goods)
        members = []
        for fab_row, pair, _nominal, energy, penalty in self.members[node]:
            capacity = self._value(obs, 'graph_now.fab.cap_eff', fab_row)
            restoration = self._value(obs, 'graph_now.fab.R', fab_row)
            wafer = self._stock(obs, pair)
            if capacity is None or restoration is None or wafer is None:
                return None
            members.append([pair, capacity if restoration > 0 else 0.,
                            energy / restoration if restoration > 0 else 0.,
                            penalty / float(grid['voll']), wafer])
        score = 0.
        produced_value = 0.
        gas_ratios = []
        for t in range(2):
            shipment = {}
            for k in (oil, gas):
                if t == 0:
                    amount = 0. if hold and k == oil else float(action['flows'][list(local_slots[k])].sum())
                else:
                    amount = min(caps[k], terminals[k])
                    stock = stocks[k]
                    due = incoming[node, k, t]
                    threshold = reserve if k == gas else 0.
                    target = (threshold + min(1., stock / threshold) * rates[k]
                              if threshold > 0 else rates[k])
                    if threshold > 0 and stock >= threshold:
                        if self.batch.complete_pulse and rates[k] <= stock + due + amount < threshold + rates[k]:
                            amount = min(amount, max(0., rates[k] - stock - due))
                    elif stock + due + amount + 1e-9 < target:
                        amount = 0.
                if amount > terminals[k] + 1e-6 or amount > caps[k] + 1e-6:
                    return None
                shipment[k] = amount
            ratio = min(1., stocks[gas] / reserve) if reserve > 0 else 1.
            gas_ratios.append(ratio)
            available = {k: stocks[k] + incoming[node, k, t] + shipment.get(k, planned[node, k] if t == 0 else 0.)
                         for k in fuel_ids}
            segments = {k: min(available[k], rates[k] * (ratio if k == gas else 1.)) for k in fuel_ids}
            power = free + sum(segments.values())
            base = min(load, power)
            gross = []
            for pair, capacity, density, price, wafer in members:
                inputs = wafer + incoming[*pair, t] + (planned[pair] if t == 0 else 0.)
                gross.append(min(capacity, inputs))
            draw = sum(q * member[2] for q, member in zip(gross, members))
            factor = min(1., max(0., power - base) / draw) if draw > 0 else 1.
            produced = [q * (factor if member[2] > 0 else 1.) for q, member in zip(gross, members)]
            value = sum(q * member[3] for q, member in zip(produced, members))
            score += base + value
            produced_value += value
            utilization = (base + draw * factor) / power if power > 0 else 0.
            for k in fuel_ids:
                storage = float(self.nodes[node]['stock'][self.network.commodity_names[k]]['storage'])
                stocks[k] = min(storage, max(0., available[k] - segments[k] * utilization))
            for k in (oil, gas):
                storage = float(self.nodes[origins[k]]['stock'][self.network.commodity_names[k]]['storage'])
                terminals[k] = min(storage, max(0., terminals[k] - shipment[k] + incoming[origins[k], k, t]))
            for member, quantity in zip(members, produced):
                pair = member[0]
                member[4] += incoming[*pair, t] + (planned[pair] if t == 0 else 0.) - quantity
        # Only half of unused controllable fuel is credited after the horizon.
        score += .5 * sum(stocks[k] + terminals[k] for k in (oil, gas))
        return score, produced_value, gas_ratios

    def apply(self, action, obs):
        week = int(obs['week'][0])
        if week < self.last_week:
            self.last_hold.clear()
        self.last_week = week
        self.last_decisions = []
        if not obs['action_mask.observed'][0] or week >= self.T:
            return action
        incoming = self._incoming(obs, week)
        planned = defaultdict(float)
        for route in self.network.routes:
            if (not route.chokepoints and all(self._value(obs, 'graph_now.tau', e) == 0 for e in route.edges)):
                planned[route.destination_node, route.commodity_id] += float(action['flows'][route.slot_id])
        held = []
        for node, (_origin, slots, _gas_origin, _gas_slots) in self.controls.items():
            last = self.last_hold.get(node, -10)
            if last != week and week - last < 2:
                continue
            if week + 1 > self.batch.latest_week.get(node, self.T):
                continue
            if float(action['flows'][list(slots)].sum()) <= 1e-6:
                continue
            baseline = self._evaluate(obs, action, node, incoming, planned, False)
            alternative = self._evaluate(obs, action, node, incoming, planned, True)
            if baseline is None or alternative is None:
                continue
            improvement = alternative[0] - baseline[0]
            # Require a visible next-week gas reserve recovery and material
            # industrial benefit; never hold merely for terminal stock credit.
            if (alternative[2][0] < .95 and alternative[2][1] >= .95
                    and alternative[1] > baseline[1] + 1e-6
                    and improvement > .05 * max(1., alternative[1] - baseline[1])):
                held.extend(slots)
                self.last_hold[node] = week
                self.last_decisions.append({'grid': self.network.node_names[node],
                                            'estimated_benefit_energy_units': improvement})
        if not held:
            return action
        flows = action['flows'].copy()
        flows[held] = 0.
        return {**action, 'flows': flows}
