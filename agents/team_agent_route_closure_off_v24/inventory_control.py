"""Visible WIP minus recent exports is a soft congestion proxy, never an ETA."""

import math

import numpy as np
from rebalance import StockRebalancer


class ProductionInventoryRebalancer(StockRebalancer):
    def __init__(self, config, network, *, net_wip_horizon=0,
                 production_queue_discount=0.0, economic_value=False, **options):
        if not isinstance(economic_value, bool):
            raise ValueError("economic_value must be a boolean")
        self.economic_model = None
        if economic_value:
            from production_value import ProductionValue
            self.economic_model = ProductionValue(config, network)
        if (isinstance(production_queue_discount, bool)
                or not isinstance(production_queue_discount, (int, float))
                or not math.isfinite(production_queue_discount) or production_queue_discount < 0):
            raise ValueError("production_queue_discount must be finite and nonnegative")
        self.production_queue_discount = float(production_queue_discount)
        self.production_queue_delay = None
        if isinstance(net_wip_horizon, bool) or not isinstance(net_wip_horizon, int) or not 0 <= net_wip_horizon <= 4:
            raise ValueError("net_wip_horizon must be an integer from 0 to 4")
        self.net_wip_horizon = net_wip_horizon
        if net_wip_horizon:
            if options.get("production_wip_horizon", 0):
                raise ValueError("net WIP and gross WIP modes cannot be combined")
            options["production_wip_horizon"] = net_wip_horizon
        super().__init__(config, network, **options)
        if self.production_queue_discount:
            from queue_outedge import OutEdgeQueueDelay

            self.production_queue_delay = OutEdgeQueueDelay(config, network, enabled=True)

    def _output_wip(self, observation):
        if not self.net_wip_horizon:
            return super()._output_wip(observation)
        keys = tuple(f"wip.{field}" for field in ("qty", "node", "k", "out_week"))
        exports = "last_week.clip.executed"
        if not all(key in observation and key + ".observed" in observation for key in (*keys, exports)):
            return {}
        live = observation["wip.qty.observed"].astype(bool)
        if any(not np.all(observation[key + ".observed"][live]) for key in keys):
            return {}
        completions = super()._output_wip(observation)
        result = {}
        for output, _storage in self.production_outputs.values():
            slots = list(self.network.slots_from.get(output, ()))
            if not slots or not np.all(observation[exports + ".observed"][slots]):
                continue
            values = observation[exports][slots]
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                continue
            quantity = float(values.sum()) * self.net_wip_horizon
            completed = completions.get(output, 0.0)
            if math.isfinite(completed) and math.isfinite(quantity):
                # This extrapolates past exports, not future capacity or guaranteed sales.
                result[output] = completed - quantity
        return result

    def apply(self, flows, observation):
        if self.production_queue_delay is not None:
            self.production_queue_delay.begin(observation)
        return super().apply(flows, observation)

    def _priorities(self, slots, caps, observation):
        weights, active = super()._priorities(slots, caps, observation)
        if not active:
            return weights, active
        weights = list(weights)
        if self.production_queue_delay is not None:
            for i, slot in enumerate(slots):
                route = self.network.routes[slot]
                pair = route.destination_node, route.commodity_id
                if caps[slot] <= 0 or pair not in self.production_rates:
                    continue
                coverage, _power = self._coverage(route, observation)
                if coverage is None:
                    continue
                transit = sum(float(observation["graph_now.tau"][edge])
                              if observation["graph_now.tau.observed"][edge]
                              else self.network.edge_transit_weeks[edge] for edge in route.edges)
                estimate = self.production_queue_delay.route_delay(route, observation, transit)
                wait = estimate["wait_weeks"]
                if wait is not None and wait > 0:
                    weights[i] *= math.exp(-min(700.0, self.production_queue_discount * wait))
        if self.economic_model is not None:
            weights = self.economic_model.adjust(weights, slots, observation)
        return weights, active
