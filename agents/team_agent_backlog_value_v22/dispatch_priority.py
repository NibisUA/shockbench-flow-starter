"""Opt-in economic request weights, grounded in public node/commodity tables."""

import math

import numpy as np


class DispatchPriority:
    def __init__(self, config, network, *, skip_zero_demand=False, product_priority_power=0.0,
                 sink_demand_power=0.0, fuel_region_weights=None):
        if not isinstance(skip_zero_demand, bool):
            raise ValueError("skip_zero_demand must be a boolean")
        for name, value in (("product_priority_power", product_priority_power),
                            ("sink_demand_power", sink_demand_power)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
        if fuel_region_weights is not None and not isinstance(fuel_region_weights, dict):
            raise ValueError("fuel_region_weights must be a mapping")
        weights = fuel_region_weights or {}
        for region, value in weights.items():
            if not isinstance(region, str):
                raise ValueError("fuel_region_weights keys must be region names")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("fuel_region_weights must contain finite nonnegative numbers")
        self.enabled = skip_zero_demand or bool(product_priority_power or sink_demand_power or weights)
        self.weights = np.ones(len(network.routes))
        self.zero_rows = []
        if not self.enabled:
            return
        profiles = {node["id"]: node for node in config["static"]["instance"]["nodes"]}
        penalties, sales = {}, {}
        for profile in profiles.values():
            for name, attrs in profile.get("sink", {}).get("demand", {}).items():
                penalties[name] = max(penalties.get(name, 0.0), float(attrs["pi"]))
            sales.update(profile.get("osat", {}).get("packages", {}))
        largest_penalty = max(penalties.values(), default=0.0)
        demand_rows = {tuple(pair): row for row, pair in enumerate(config["layout"]["demands"])}
        amounts = {}
        fuels = {name for node in profiles.values() for name in node.get("grid", {}).get("shares", {})}
        for route in network.routes:
            name = network.commodity_names[route.commodity_id]
            source = profiles[network.node_names[route.source_node]]
            target = profiles[network.node_names[route.destination_node]]
            product = target.get("fab", {}).get("product") if name == target.get("fab", {}).get("input") else name
            product = sales.get(product, product)
            if product_priority_power and largest_penalty and product in penalties:
                self.weights[route.slot_id] *= (penalties[product] / largest_penalty) ** product_priority_power
            attrs = target.get("sink", {}).get("demand", {}).get(name)
            if attrs is not None:
                amount = max(float(attrs["dbar"]), 0.0)
                amounts[route.slot_id] = amount
                row = demand_rows.get((route.destination_node, route.commodity_id))
                if skip_zero_demand and amount == 0 and row is not None:
                    self.zero_rows.append((route.slot_id, row, bool(attrs["backlog"])))
            if name in fuels and source["region"] != target["region"]:
                self.weights[route.slot_id] *= weights.get(target["region"], 1.0)
        if sink_demand_power:
            for slots in network.slots_from.values():
                largest = max((amounts.get(slot, 0.0) for slot in slots), default=0.0)
                if largest:
                    for slot in slots:
                        if amounts.get(slot, 0.0) > 0:
                            self.weights[slot] *= (amounts[slot] / largest) ** sink_demand_power

    def apply(self, flows, observation):
        if not self.enabled:
            return flows
        result = flows * self.weights
        for slot, row, backlog_enabled in self.zero_rows:
            visible = observation["demand_forecast.qty.observed"][row].astype(bool)
            if not visible.any() or np.any(observation["demand_forecast.qty"][row][visible] > 0):
                continue
            if backlog_enabled and (not observation["backlog.qty.observed"][row]
                                    or observation["backlog.qty"][row] > 0):
                continue
            result[slot] = 0.0
        return result
