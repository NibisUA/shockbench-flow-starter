"""Bounded local fuel search with conditional persistent marks, disabled by default."""

import math
from collections import Counter, defaultdict

import numpy as np


class FuelLookahead:
    def __init__(self, config, network, batch, *, horizon=0, industry_weight=1.0, stock_value=0.25,
                 preserve_rationed=False):
        if isinstance(horizon, bool) or not isinstance(horizon, int) or not 0 <= horizon <= 6:
            raise ValueError("fuel_lookahead_horizon must be an integer in [0, 6]")
        for name, value in (("industry_weight", industry_weight), ("stock_value", stock_value)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if not isinstance(preserve_rationed, bool):
            raise ValueError("preserve_rationed must be a boolean")
        self.preserve_rationed = preserve_rationed
        self.enabled = bool(horizon and batch.enabled)
        self.last_status, self.last_evaluations = "disabled", 0
        if not self.enabled:
            return
        self.horizon, self.industry_weight, self.stock_value = horizon, industry_weight, stock_value
        self.network, self.batch, self.T = network, batch, int(config["T"])
        self.goods = {name: k for k, name in enumerate(network.commodity_names)}
        self.stocks = {tuple(pair): row for row, pair in enumerate(config["layout"]["stock_slots"])}
        names = {name: n for n, name in enumerate(network.node_names)}
        self.nodes = {names[node["id"]]: node for node in config["static"]["instance"]["nodes"]}
        self.grid_rows = {node: row for row, node in enumerate(config["layout"]["grids"])}
        penalties = {}
        for node in self.nodes.values():
            for name, demand in node.get("sink", {}).get("demand", {}).items():
                penalties[name] = max(penalties.get(name, 0.0), float(demand["pi"]))
        self.members = defaultdict(list)
        for row, node in enumerate(config["layout"]["fabs"]):
            fab = self.nodes[node]["fab"]
            if fab["grid"] is not None:
                self.members[names[fab["grid"]]].append(
                    (row, (node, self.goods[fab["input"]]), float(fab["cap0"]), float(fab["e"]),
                     penalties.get(fab["product"].removesuffix("_raw"), 0.0))
                )
        sources = Counter()
        for pair, slots in batch.groups.items():
            for source in {network.routes[s].source_node for s in slots}:
                sources[source, pair[1]] += 1
        self.controls = defaultdict(dict)
        for pair, slots in batch.groups.items():
            origins = {network.routes[s].source_node for s in slots}
            if len(origins) == 1:
                source = next(iter(origins))
                if sources[source, pair[1]] == 1:
                    self.controls[pair[0]][pair[1]] = (source, tuple(slots))

    def _incoming(self, obs, week, horizon):
        incoming = defaultdict(float)
        for row in np.flatnonzero(obs["pipeline.qty.observed"]):
            if not all(obs[f"pipeline.{key}.observed"][row] for key in ("edge", "k", "arrival_week")):
                continue
            node = self.network.edge_head[int(obs["pipeline.edge"][row])]
            pair = node, int(obs["pipeline.k"][row])
            due = int(obs["pipeline.arrival_week"][row]) - week
            # Only the scheduled final edge into a local stock slot is counted.
            if node not in self.network.chokepoints and pair in self.stocks and 0 <= due < horizon:
                incoming[*pair, due] += float(obs["pipeline.qty"][row])
        return incoming

    def apply(self, flows, requests, obs):
        self.last_evaluations = 0
        if not self.enabled:
            return flows
        self.last_status = "conditional_search"
        if not obs["action_mask.observed"][0]:
            self.last_status = "unobserved_mask_keep_control"
            return flows
        week, out = int(obs["week"][0]), flows.copy()
        horizon = min(self.horizon, self.T - week + 1)
        incoming = self._incoming(obs, week, horizon)
        fab_inputs = {pair for members in self.members.values() for _r, pair, _c, _e, _p in members}
        for route in self.network.routes:
            pair = route.destination_node, route.commodity_id
            if (pair in fab_inputs and requests[route.slot_id] > 0 and not route.chokepoints
                    and all(obs["graph_now.tau.observed"][e] for e in route.edges)):
                delay = sum(int(obs["graph_now.tau"][e]) for e in route.edges)
                if 0 <= delay < horizon:
                    incoming[*pair, delay] += float(requests[route.slot_id])
        for node, controls in self.controls.items():
            grid = self.nodes[node]["grid"]
            if grid["priority"] != "base_first" or not self.members[node]:
                continue
            fuels = tuple(self.goods[name] for name in grid["shares"] if name in self.goods)
            members = self.members[node]
            pairs = [(node, k) for k in fuels] + [(source, k) for k, (source, _s) in controls.items()]
            pairs += [pair for _row, pair, _cap, _energy, _pi in members]
            if not all(obs["stock.qty.observed"][self.stocks[pair]] for pair in pairs):
                continue
            row = self.grid_rows[node]
            if not all(obs[key + ".observed"][row] for key in ("graph_now.grid.G_bar", "graph_now.grid.y_bar")):
                continue
            if any(not obs["graph_now.u.observed"][self.network.routes[s].edge_id]
                   for _source, slots in controls.values() for s in slots):
                continue
            quantity = self._plan(node, fuels, controls, members, flows, requests, obs, incoming, week, horizon)
            for k, amount in quantity.items():
                _source, slots = controls[k]
                total = float(requests[list(slots)].sum())
                out[list(slots)] = requests[list(slots)] * (amount / total if total > 0 else 0.0)
        return out

    def _plan(self, node, fuels, controls, members, flows, requests, obs, incoming, week, horizon):
        grid, row = self.nodes[node]["grid"], self.grid_rows[node]
        generation, load = float(obs["graph_now.grid.G_bar"][row]), float(obs["graph_now.grid.y_bar"][row])
        rates = np.asarray([float(grid["shares"][self.network.commodity_names[k]]) * generation for k in fuels])
        reserve = np.asarray([self.batch.grids.get((node, k), (0, 0, 0, 0, 0))[3] for k in fuels])
        storage = np.asarray([float(self.nodes[node]["stock"][self.network.commodity_names[k]]["storage"])
                              for k in fuels])
        controlled = tuple(controls)
        positions = [fuels.index(k) for k in controlled]
        caps = np.asarray([float(requests[list(controls[k][1])].sum()) for k in controlled])
        future_caps = np.asarray([sum(float(obs["graph_now.u"][self.network.routes[s].edge_id])
                                     for s in controls[k][1] if obs["action_mask"][s]) for k in controlled])
        prior = np.asarray([float(flows[list(controls[k][1])].sum()) for k in controlled])
        inventory = np.asarray([[float(obs["stock.qty"][self.stocks[node, k]]) for k in fuels]])
        terminal = np.asarray([[float(obs["stock.qty"][self.stocks[controls[k][0], k]]) for k in controlled]])
        wafer = np.asarray([[float(obs["stock.qty"][self.stocks[pair]]) for _r, pair, _c, _e, _p in members]])
        capacity, restoration = [], []
        for fab_row, _pair, nominal, _energy, _penalty in members:
            key = "graph_now.fab.R"
            r = float(obs[key][fab_row]) if obs[key + ".observed"][fab_row] else 1.0
            restoration.append(r)
            key = "graph_now.fab.cap_eff"
            capacity.append(float(obs[key][fab_row]) if obs[key + ".observed"][fab_row] else nominal * r)
        energy = np.asarray([member[3] for member in members])
        restoration = np.asarray(restoration)
        density = np.divide(energy, restoration, out=np.zeros_like(energy), where=restoration > 0)
        prices = np.asarray([member[4] / float(grid["voll"]) for member in members])
        free = sum(float(share) * generation for name, share in grid["shares"].items() if name not in self.goods)
        score, first = np.zeros(1), np.zeros((1, len(controlled)))
        for t in range(horizon):
            choices = []
            current_caps = caps if t == 0 else future_caps
            for pos, cap, original in zip(positions, current_caps, prior, strict=True):
                budget = np.minimum(terminal[:, len(choices)], cap)
                due = incoming[node, fuels[pos], t]
                stock = inventory[:, pos]
                ration = np.minimum(1, stock / reserve[pos]) if reserve[pos] > 0 else np.ones(len(stock))
                target = reserve[pos] + ration * rates[pos] if reserve[pos] > 0 else rates[pos]
                choices.append(np.column_stack((np.minimum(budget, original), np.zeros(len(stock)),
                               np.minimum(budget, np.maximum(0, rates[pos] - stock - due)),
                               np.minimum(budget, np.maximum(0, target - stock - due)), budget)))
                if self.preserve_rationed and reserve[pos] > 0:
                    if t == 0:
                        locked = np.minimum(budget, original)
                    else:
                        available = stock + due + budget
                        locked = np.where((stock >= reserve[pos]) | (available + 1e-9 >= target), budget, 0.0)
                        if self.batch.complete_pulse:
                            partial = ((stock >= reserve[pos]) & (available >= rates[pos])
                                       & (available < reserve[pos] + rates[pos]))
                            locked = np.where(partial, np.minimum(budget, np.maximum(0, rates[pos] - stock - due)),
                                              locked)
                        if week + t > self.batch.latest_week.get(node, self.T):
                            locked = budget
                    choices[-1][:] = locked[:, None]
            combinations = np.asarray(list(np.ndindex(*(5 for _k in controlled))), dtype=int)
            parent = np.repeat(np.arange(len(score)), len(combinations))
            selected = np.tile(combinations, (len(score), 1))
            shipment = np.column_stack([choice[parent, selected[:, k]] for k, choice in enumerate(choices)])
            current = inventory[parent].copy()
            available = current + np.asarray([incoming[node, k, t] for k in fuels])
            available[:, positions] += shipment
            if t == 0:
                for route in self.network.routes:
                    if (route.destination_node == node and route.commodity_id in fuels
                            and route.commodity_id not in controls and not route.chokepoints
                            and route.nominal_transit_weeks == 0):
                        available[:, fuels.index(route.commodity_id)] += float(flows[route.slot_id])
            segment = np.minimum(available, rates)
            for pos in np.flatnonzero(reserve > 0):
                rationed = rates[pos] * np.minimum(1, current[:, pos] / reserve[pos])
                segment[:, pos] = np.minimum(segment[:, pos], rationed)
            power = segment.sum(axis=1) + free
            base = np.minimum(load, power)
            inputs = wafer[parent] + np.asarray([incoming[*pair, t] for _r, pair, _c, _e, _p in members])
            gross = np.minimum(inputs, capacity)
            draw = (gross * density).sum(axis=1)
            served = np.minimum(draw, np.maximum(0, power - base))
            factor = np.divide(served, draw, out=np.ones_like(served), where=draw > 0)
            produced = gross * np.where(energy > 0, factor[:, None], 1)
            utilization = np.divide(base + served, power, out=np.zeros_like(power), where=power > 0)
            inventory = np.minimum(storage, np.maximum(0, available - segment * utilization[:, None]))
            terminal = terminal[parent] - shipment
            terminal += np.asarray([incoming[controls[k][0], k, t] for k in controlled])
            terminal_capacity = [float(self.nodes[controls[k][0]]["stock"][self.network.commodity_names[k]]["storage"])
                                 for k in controlled]
            terminal = np.minimum(terminal, terminal_capacity)
            wafer = np.maximum(0, inputs - produced)
            premium = self.industry_weight if week + t <= self.batch.latest_week.get(node, self.T) else 0.0
            score = score[parent] + 0.95 ** t * (base + premium * (produced * prices).sum(axis=1))
            first = shipment.copy() if t == 0 else first[parent]
            rank = score + self.stock_value * (inventory.sum(axis=1) + terminal.sum(axis=1))
            keep = np.argsort(-rank, kind="stable")[:32]
            self.last_evaluations += len(rank)
            inventory, terminal, wafer, score, first = (
                value[keep] for value in (inventory, terminal, wafer, score, first))
        winner = int(np.argmax(score + self.stock_value * (inventory.sum(axis=1) + terminal.sum(axis=1))))
        return dict(zip(controlled, first[winner], strict=True))
