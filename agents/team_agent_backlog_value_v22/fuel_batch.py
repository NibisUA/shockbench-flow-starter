"""Conditional fuel batches for base-first grids; disabled by default."""

import math
from collections import defaultdict

import numpy as np


class FuelBatch:
    def __init__(self, config, network, *, lng=0.0, crude=0.0, complete_pulse=False,
                 power_margin_only=False, end_aware=False, sync_crude=False, sync_margin_only=False,
                 load_aware=False, end_buffer=0):
        for name, value in (("lng", lng), ("crude", crude)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} batch factor must be finite and nonnegative")
        self.enabled = bool(lng or crude)
        if not isinstance(complete_pulse, bool):
            raise ValueError("complete_pulse must be a boolean")
        self.complete_pulse = complete_pulse
        if not isinstance(power_margin_only, bool) or not isinstance(end_aware, bool):
            raise ValueError("power_margin_only and end_aware must be booleans")
        self.power_margin_only = power_margin_only
        if not isinstance(sync_crude, bool) or not isinstance(sync_margin_only, bool):
            raise ValueError("sync_crude and sync_margin_only must be booleans")
        self.sync_crude, self.sync_margin_only = sync_crude, sync_margin_only
        if not isinstance(load_aware, bool):
            raise ValueError("load_aware must be a boolean")
        self.load_aware = load_aware
        if isinstance(end_buffer, bool) or not isinstance(end_buffer, int) or not 0 <= end_buffer <= 26:
            raise ValueError("end_buffer must be an integer between 0 and 26 weeks")
        self.last_held = ()
        if not self.enabled:
            return
        self.network = network
        self.stocks = {tuple(pair): row for row, pair in enumerate(config["layout"]["stock_slots"])}
        names = {name: n for n, name in enumerate(network.node_names)}
        goods = {name: k for k, name in enumerate(network.commodity_names)}
        nodes = {names[node["id"]]: node for node in config["static"]["instance"]["nodes"]}
        industry = {names[node["fab"]["grid"]] for node in nodes.values()
                    if "fab" in node and node["fab"]["grid"] is not None}
        self.grids = {}
        self.members = defaultdict(list)
        if load_aware:
            for row, node in enumerate(config["layout"].get("fabs", ())):
                fab = nodes[node]["fab"]
                if fab["grid"] is not None:
                    self.members[names[fab["grid"]]].append(
                        (row, (node, goods[fab["input"]]), float(fab["cap0"]), float(fab["e"]))
                    )
        self.rationed = {}
        self.crude_id = goods.get("crude")
        self.base_load, self.latest_week = {}, {}
        self.groups = defaultdict(list)
        psi = float(config["static"]["instance"]["params"]["psi"])
        for row, n in enumerate(config["layout"]["grids"]):
            if n not in industry:
                continue
            grid = nodes[n]["grid"]
            self.base_load[n] = float(grid.get("base_load", 0))
            rationed = grid.get("rationed")
            if rationed in goods and psi * float(grid["ibar"].get(rationed, 0)) > 0:
                self.rationed[n] = goods[rationed], psi * float(grid["ibar"][rationed]), float(grid["shares"][rationed])
            if end_aware:
                lead_times = []
                for fab_node in nodes.values():
                    fab = fab_node.get("fab")
                    if not fab or fab["grid"] is None or names[fab["grid"]] != n:
                        continue
                    for route in network.routes:
                        if route.source_node != names[fab_node["id"]] or route.commodity_id != goods[fab["product"]]:
                            continue
                        osat = nodes[route.destination_node].get("osat")
                        if osat is None or fab["product"] not in osat["packages"]:
                            continue
                        packed = goods[osat["packages"][fab["product"]]]
                        for delivery in network.routes:
                            if (delivery.source_node == route.destination_node and delivery.commodity_id == packed
                                    and "sink" in nodes[delivery.destination_node]):
                                lead_times.append(int(fab["tau"]) + route.nominal_transit_weeks
                                                  + int(osat["tau"]) + delivery.nominal_transit_weeks + 2)
                if lead_times:
                    self.latest_week[n] = int(config["T"]) - min(lead_times) - end_buffer
            for name, factor in (("lng", lng), ("crude", crude)):
                if name not in goods or not factor or not grid["shares"].get(name):
                    continue
                k = goods[name]
                reserve = psi * float(grid["ibar"].get(name, 0)) if grid.get("rationed") == name else 0.0
                self.grids[n, k] = row, float(grid["deliverable"]), float(grid["shares"][name]), reserve, factor
        for route in network.routes:
            pair = route.destination_node, route.commodity_id
            if (pair in self.grids and nodes[route.source_node].get("type") == "terminal"
                    and not route.chokepoints and route.nominal_transit_weeks == 0):
                self.groups[pair].append(route.slot_id)
        self.enabled = bool(self.groups)

    def apply(self, flows, observation):
        self.last_held = ()
        if not self.enabled:
            return flows
        out = flows.copy()
        week = int(observation["week"][0])
        incoming = defaultdict(float)
        for row in np.flatnonzero(observation["pipeline.qty.observed"]):
            if (all(observation[f"pipeline.{field}.observed"][row] for field in ("edge", "k", "arrival_week"))
                    and int(observation["pipeline.arrival_week"][row]) == week):
                pair = (self.network.edge_head[int(observation["pipeline.edge"][row])],
                        int(observation["pipeline.k"][row]))
                incoming[pair] += float(observation["pipeline.qty"][row])
        held = []
        for pair, slots in self.groups.items():
            if week > self.latest_week.get(pair[0], math.inf):
                continue
            stock_row = self.stocks[pair]
            if not observation["stock.qty.observed"][stock_row]:
                continue
            row, nominal, share, reserve, factor = self.grids[pair]
            generation = (float(observation["graph_now.grid.G_bar"][row])
                          if observation["graph_now.grid.G_bar.observed"][row] else nominal)
            if self.sync_crude and pair[1] == self.crude_id and pair[0] in self.rationed:
                gas, threshold, share_gas = self.rationed[pair[0]]
                gas_row = self.stocks[pair[0], gas]
                if observation["stock.qty.observed"][gas_row]:
                    ratio = min(1.0, float(observation["stock.qty"][gas_row]) / threshold)
                    key = "graph_now.grid.y_bar"
                    load = (float(observation[key][row])
                            if key in observation and observation[key + ".observed"][row] else self.base_load[pair[0]])
                    upper_power = generation * (1 - share_gas + share_gas * ratio)
                    if ratio < 1 and (not self.sync_margin_only or upper_power <= load):
                        out[slots] = 0
                        held.append(pair)
                        continue
            if self.power_margin_only:
                key = "graph_now.grid.y_bar"
                load = (float(observation[key][row]) if key in observation and observation[key + ".observed"][row]
                        else self.base_load[pair[0]])
                if generation <= load:
                    continue
            rate = generation * share
            if self.load_aware:
                rate *= self._load_fraction(pair[0], row, generation, incoming, observation)
            stock = float(observation["stock.qty"][stock_row])
            if reserve:
                ratio = min(1.0, stock / reserve)
                target = reserve * factor + ratio * rate
                healthy = stock >= reserve * factor
            else:
                target = rate * factor
                healthy = False
            available = stock + incoming[pair]
            requested = defaultdict(float)
            for slot in slots:
                route = self.network.routes[slot]
                source = self.stocks[route.source_node, pair[1]]
                if not observation["stock.qty.observed"][source]:
                    break
                requested[source] += float(out[slot])
            else:
                available += sum(min(amount, float(observation["stock.qty"][source]))
                                 for source, amount in requested.items())
                if healthy:
                    if self.complete_pulse and rate <= available < reserve + rate:
                        quantity = float(out[slots].sum())
                        if quantity:
                            out[slots] *= min(1.0, max(0.0, rate - stock - incoming[pair]) / quantity)
                    continue
                if available + 1e-9 < target:
                    out[slots] = 0
                    held.append(pair)
        self.last_held = tuple(held)
        return out

    def _load_fraction(self, node, grid_row, generation, incoming, observation):
        if generation <= 0:
            return 0.0
        key = "graph_now.grid.y_bar"
        load = (float(observation[key][grid_row]) if observation[key + ".observed"][grid_row]
                else self.base_load[node])
        for row, pair, nominal_cap, energy in self.members[node]:
            key = "graph_now.fab.R"
            restoration = float(observation[key][row]) if observation[key + ".observed"][row] else 1.0
            if restoration <= 0:
                continue
            key = "graph_now.fab.cap_eff"
            capacity = (float(observation[key][row]) if observation[key + ".observed"][row]
                        else nominal_cap * restoration)
            stock_row = self.stocks[pair]
            if observation["stock.qty.observed"][stock_row]:
                capacity = min(capacity, float(observation["stock.qty"][stock_row]) + incoming.get(pair, 0.0))
            # Conditional current-week demand, not a future production guarantee.
            load += energy * capacity / restoration
        return min(1.0, max(0.0, load / generation))
