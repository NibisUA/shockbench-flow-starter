"""Use final-week spare transport when terminal credits and holding justify it.

Only unconsumed manufacturing outputs are eligible. Production inputs, fuel,
and demand stocks stay under the existing controller. No future marks are read.
"""
import math
from collections import defaultdict

import numpy as np


class TerminalHolding:
    def __init__(self, config, network, *, enabled=False):
        if not isinstance(enabled, bool):
            raise ValueError('terminal_holding_enabled must be a boolean')
        self.enabled, self.last_diagnostics = enabled, {}
        if not enabled:
            return
        self.horizon, self.network = int(config['T']), network
        self.stocks = {tuple(pair): row for row,pair in enumerate(config['layout']['stock_slots'])}
        nodes = {network.node_names.index(n['id']):n for n in config['static']['instance']['nodes']}
        goods = {good:k for k,good in enumerate(network.commodity_names)}
        self.output = {}
        for node,attrs in nodes.items():
            fab, osat = attrs.get('fab'), attrs.get('osat')
            recipes = {fab['input']:fab['product']} if fab else osat['packages'] if osat else {}
            inputs = set(recipes)
            for product in set(recipes.values()) - inputs:
                pair = node,goods[product]
                # No transport can replenish this output stock during the final week.
                if any(network.edge_head[e] == node and pair[1] in commodities
                       for e,commodities in enumerate(config['static']['edges']['K'])):
                    continue
                stock = attrs['stock'][product]
                capacity = float(fab['cap0'] if fab else osat['thr'])
                self.output[pair] = {'salvage':float(stock.get('salvage',0)),
                    'holding':float(stock.get('holding_cost',0)),
                    'storage':float(stock['storage']),'completion_upper':capacity}
        # Protect unusual initial WIP too; this is public config, not an observation of future disruptions.
        initial = config['static']['instance'].get('initial_state',{})
        for name in ('fab_wip','osat_wip'):
            totals = defaultdict(float)
            for item in initial.get(name,()):
                if item.get('out_week') == self.horizon and item.get('node') in network.node_names:
                    node = network.node_names.index(item['node'])
                    for pair in self.output:
                        if pair[0] == node:
                            totals[pair] += float(item.get('qty',0))
            for pair,qty in totals.items():
                self.output[pair]['completion_upper'] = max(self.output[pair]['completion_upper'],qty)
        self.slots = tuple(r.slot_id for r in network.routes
                           if (r.source_node,r.commodity_id) in self.output)
        self.nodes, self.goods_value = nodes, np.asarray(config['static']['commodities']['v'],dtype=float)

    def apply(self, flows, observation):
        self.last_diagnostics = {'enabled':self.enabled,'changed_slots':0}
        if not self.enabled or int(observation['week'][0]) != self.horizon:
            return flows
        if not observation.get('action_mask.observed',np.array([0]))[0]:
            return flows
        from scipy.optimize import linprog

        active, gains, upper = [], [], []
        for slot in self.slots:
            route = self.network.routes[slot]
            pair = route.source_node,route.commodity_id
            row,edge = self.stocks[pair],route.edge_id
            required = [('stock.qty',row),('graph_now.u',edge),('graph_now.c',edge)]
            if any(key not in observation or key+'.observed' not in observation
                   or not observation[key+'.observed'][index] for key,index in required):
                continue
            if not observation['action_mask'][slot]:
                continue
            if not observation['graph_now.tariff.observed'][edge,route.commodity_id]:
                continue
            budget = float(observation['stock.qty'][row])-float(flows[list(self.network.slots_from[pair])].sum())
            attrs = self.output[pair]
            # With no incoming transport and even a full-capacity completion, no disposal can erase holding savings.
            holding = attrs['holding'] if budget+attrs['completion_upper'] <= attrs['storage']+1e-8 else 0.0
            head = self.network.edge_head[edge]
            target = self.nodes[head]
            target_credit = 0.0 if target['type'] == 'source' else float(
                target.get('stock',{}).get(self.network.commodity_names[pair[1]],{}).get('salvage',0))
            gain = target_credit-attrs['salvage']+holding-float(observation['graph_now.c'][edge])
            gain -= float(observation['graph_now.tariff'][edge,pair[1]])*self.goods_value[pair[1]]
            ids = [s for s in self.network.edges_to_slots[edge] if self.network.routes[s].edge_id == edge]
            capacity = float(observation['graph_now.u'][edge])-float(flows[ids].sum())
            if gain > 1e-8 and min(budget,capacity) > 1e-8:
                active.append(slot)
                gains.append(gain)
                upper.append(min(budget,capacity))
        if not active:
            return flows
        sources,edges = defaultdict(list),defaultdict(list)
        for i,slot in enumerate(active):
            route = self.network.routes[slot]
            sources[route.source_node,route.commodity_id].append(i)
            edges[route.edge_id].append(i)
        rows,limits = [],[]
        def bound(ids,limit):
            row = np.zeros(len(active))
            row[ids] = 1
            rows.append(row)
            limits.append(max(0.0,limit))
        for pair,ids in sources.items():
            bound(ids,float(observation['stock.qty'][self.stocks[pair]])-
                  float(flows[list(self.network.slots_from[pair])].sum()))
        for edge,ids in edges.items():
            slots = [s for s in self.network.edges_to_slots[edge] if self.network.routes[s].edge_id == edge]
            bound(ids,float(observation['graph_now.u'][edge])-float(flows[slots].sum()))
        gain = np.asarray(gains)
        solution = linprog(-gain/max(1.0,float(gain.max())),A_ub=np.asarray(rows),b_ub=limits,
                           bounds=np.column_stack((np.zeros(len(active)),upper)),method='highs')
        if not solution.success:
            self.last_diagnostics['solver_status'] = int(solution.status)
            return flows
        out = flows.copy()
        addition = np.minimum(upper,np.maximum(0.0,solution.x))
        out[active] += addition
        self.last_diagnostics.update(changed_slots=int(np.count_nonzero(addition > 1e-8)),
            conservative_expected_saving_usd=float(gain@addition))
        return out
