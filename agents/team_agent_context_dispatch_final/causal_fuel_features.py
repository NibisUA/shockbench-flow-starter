"""Public pre-action features for an offline-trained, bounded fuel selector.

This module has no simulator, scenario, reference-cost, or future-data access.
Scheduled incoming shipments and today's baseline action are public inputs.
"""
import math
from collections import defaultdict

import numpy as np


FEATURE_NAMES = (
    'remaining_fraction', 'generation_ratio', 'base_load_ratio', 'oil_share',
    'gas_share', 'unmodelled_share', 'planned_oil_cover', 'grid_oil_cover',
    'terminal_oil_cover', 'gas_reserve_ratio', 'grid_gas_cover',
    'terminal_gas_cover', 'incoming_grid_gas_cover', 'incoming_terminal_gas_cover',
    'nominal_power_margin', 'planned_power_margin', 'held_oil_power_margin',
    'fab_capacity_ratio', 'input_cover', 'industry_draw_ratio',
    'industry_value_ratio', 'previous_shed_ratio', 'incoming_grid_oil_cover',
)


class CausalFuelFeatures:
    def __init__(self, config, network, batch):
        self.network, self.T = network, int(config['T'])
        self.stocks = {tuple(pair): row for row, pair in enumerate(config['layout']['stock_slots'])}
        self.goods = {name: k for k, name in enumerate(network.commodity_names)}
        names = {name: n for n, name in enumerate(network.node_names)}
        self.nodes = {names[node['id']]: node for node in config['static']['instance']['nodes']}
        self.grid_rows = {node: row for row, node in enumerate(config['layout']['grids'])}
        self.psi = float(config['static']['instance']['params']['psi'])
        penalties = {}
        for node in self.nodes.values():
            for name, demand in node.get('sink', {}).get('demand', {}).items():
                penalties[name] = max(penalties.get(name, 0.), float(demand['pi']))
        self.members = defaultdict(list)
        for row, node in enumerate(config['layout']['fabs']):
            fab = self.nodes[node]['fab']
            if fab['grid'] is not None:
                self.members[names[fab['grid']]].append((
                    row, (node, self.goods[fab['input']]), float(fab['cap0']),
                    float(fab['e']), penalties.get(fab['product'].removesuffix('_raw'), 0.),
                ))
        self.controls = {}
        oil, gas = self.goods.get('crude'), self.goods.get('lng')
        if oil is None or gas is None:
            return
        for (node, commodity), slots in batch.groups.items():
            if commodity != oil or (node, gas) not in batch.groups or not self.members[node]:
                continue
            origins = {network.routes[s].source_node for s in slots}
            gas_slots = batch.groups[node, gas]
            gas_origins = {network.routes[s].source_node for s in gas_slots}
            if len(origins) != 1 or len(gas_origins) != 1:
                continue
            # Raising no flow: holding oil is feasible even with shared capacity.
            self.controls[node] = (next(iter(origins)), tuple(slots), next(iter(gas_origins)), tuple(gas_slots))

    @staticmethod
    def _value(obs, key, row):
        values, seen = obs.get(key), obs.get(key + '.observed')
        if values is None or seen is None or row >= len(values) or not seen[row]:
            return None
        value = float(values[row])
        return value if math.isfinite(value) and value >= 0 else None

    def _stock(self, obs, pair):
        row = self.stocks.get(pair)
        return None if row is None else self._value(obs, 'stock.qty', row)

    def _incoming(self, obs, week):
        incoming = defaultdict(float)
        for row in np.flatnonzero(obs['pipeline.qty.observed']):
            if not all(obs[f'pipeline.{key}.observed'][row] for key in ('edge', 'k', 'arrival_week')):
                continue
            edge, commodity = int(obs['pipeline.edge'][row]), int(obs['pipeline.k'][row])
            due = int(obs['pipeline.arrival_week'][row]) - week
            qty = float(obs['pipeline.qty'][row])
            if not (0 <= edge < len(self.network.edge_names) and 0 <= commodity < len(self.goods)
                    and math.isfinite(qty) and qty >= 0 and 0 <= due <= 1):
                continue
            node = self.network.edge_head[edge]
            if node not in self.network.chokepoints and (node, commodity) in self.stocks:
                incoming[node, commodity, due] += qty
        return incoming

    def rows(self, obs, action):
        week = int(obs['week'][0])
        if not obs['action_mask.observed'][0] or not 1 <= week <= self.T:
            return ()
        incoming = self._incoming(obs, week)
        planned_inputs = defaultdict(float)
        for route in self.network.routes:
            if (not route.chokepoints and route.nominal_transit_weeks == 0
                    and action['flows'][route.slot_id] > 0):
                planned_inputs[route.destination_node, route.commodity_id] += float(action['flows'][route.slot_id])
        oil, gas = self.goods['crude'], self.goods['lng']
        result = []
        for node, (origin, slots, gas_origin, gas_slots) in self.controls.items():
            oil_action = float(action['flows'][list(slots)].sum())
            if oil_action <= 1e-6:
                continue
            grid, row = self.nodes[node]['grid'], self.grid_rows[node]
            generation = self._value(obs, 'graph_now.grid.G_bar', row)
            load = self._value(obs, 'graph_now.grid.y_bar', row)
            oil_stock, oil_terminal = self._stock(obs, (node, oil)), self._stock(obs, (origin, oil))
            gas_stock, gas_terminal = self._stock(obs, (node, gas)), self._stock(obs, (gas_origin, gas))
            if any(value is None for value in (generation, load, oil_stock, oil_terminal, gas_stock, gas_terminal)):
                continue
            nominal, voll = float(grid['deliverable']), float(grid['voll'])
            if min(generation, nominal, voll) <= 0 or grid['priority'] != 'base_first':
                continue
            oil_share, gas_share = float(grid['shares']['crude']), float(grid['shares']['lng'])
            oil_rate, gas_rate = generation * oil_share, generation * gas_share
            if min(oil_rate, gas_rate) <= 0:
                continue
            reserve = self.psi * float(grid['ibar'].get('lng', 0)) if grid.get('rationed') == 'lng' else 0.
            gas_ratio = min(1., gas_stock / reserve) if reserve > 0 else 1.
            draw = gross = nominal_capacity = current_capacity = inputs = industry_value = 0.
            valid = True
            for fab_row, pair, cap0, energy, penalty in self.members[node]:
                capacity = self._value(obs, 'graph_now.fab.cap_eff', fab_row)
                restoration = self._value(obs, 'graph_now.fab.R', fab_row)
                stock = self._stock(obs, pair)
                if any(value is None for value in (capacity, restoration, stock)):
                    valid = False
                    break
                available = stock + incoming[*pair, 0] + planned_inputs[pair]
                output = min(available, capacity) if restoration > 0 else 0.
                draw += output * energy / restoration if restoration > 0 else 0.
                industry_value += output * penalty / voll
                gross += output
                current_capacity += capacity
                nominal_capacity += cap0
                inputs += available
            if not valid:
                continue
            unmodelled = sum(float(share) for name, share in grid['shares'].items() if name not in self.goods)
            power = generation * unmodelled
            oil_segment = oil_held_segment = 0.
            for name, share in grid['shares'].items():
                if name not in self.goods:
                    continue
                commodity = self.goods[name]
                stock = self._stock(obs, (node, commodity))
                if stock is None:
                    valid = False
                    break
                rate = generation * float(share)
                available = stock + incoming[node, commodity, 0] + planned_inputs[node, commodity]
                limit = rate
                if name == grid.get('rationed'):
                    threshold = self.psi * float(grid['ibar'].get(name, 0))
                    limit *= min(1., stock / threshold) if threshold > 0 else 1.
                segment = min(available, limit)
                power += segment
                if commodity == oil:
                    oil_segment = segment
                    oil_held_segment = min(stock + incoming[node, commodity, 0], limit)
            if not valid:
                continue
            shed = self._value(obs, 'last_week.shed.qty', row)
            if shed is None:
                shed = 0.
            features = (
                (self.T - week + 1) / self.T, generation / nominal, load / generation,
                oil_share, gas_share, unmodelled, oil_action / oil_rate, oil_stock / oil_rate,
                oil_terminal / oil_rate, gas_ratio, gas_stock / gas_rate, gas_terminal / gas_rate,
                (incoming[node, gas, 0] + incoming[node, gas, 1]) / gas_rate,
                (incoming[gas_origin, gas, 0] + incoming[gas_origin, gas, 1]) / gas_rate,
                (generation - load - draw) / generation, (power - load - draw) / generation,
                (power - oil_segment + oil_held_segment - load - draw) / generation,
                current_capacity / nominal_capacity if nominal_capacity else 0.,
                inputs / current_capacity if current_capacity else 0., draw / generation,
                industry_value / generation, shed / generation,
                (incoming[node, oil, 0] + incoming[node, oil, 1]) / oil_rate,
            )
            if len(features) == len(FEATURE_NAMES) and all(math.isfinite(value) for value in features):
                result.append({'node': node, 'slots': slots, 'features': features,
                               'normalization_usd': nominal * voll})
        return tuple(result)
