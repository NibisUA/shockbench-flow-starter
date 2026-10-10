"""Public-state features for the marginal value of OSAT raw-chip receipts."""

import json
import math
from pathlib import Path

import numpy as np


FEATURES = (
    "input_cover", "other_input_cover", "output_fill", "other_output_fill",
    "capacity_quote", "near_wip_cover", "export_capacity", "export_net_value",
    "market_cover", "market_demand", "input_export", "output_export",
    "remaining_fraction", "export_value_capacity",
)


class RawUtility:
    def __init__(self, config, network, *, strength=0.0, model_path=None):
        self.network = network
        self.strength = float(strength)
        self.weights = None
        self.horizon = int(config["T"])
        self.stock_rows = {tuple(p): r for r, p in enumerate(config["layout"]["stock_slots"])}
        self.demands = {tuple(p): r for r, p in enumerate(config["layout"]["demands"])}
        self.goods_value = np.asarray(config["static"]["commodities"]["v"], dtype=float)
        goods = {name: k for k, name in enumerate(network.commodity_names)}
        nodes = {network.node_names.index(n["id"]): n for n in config["static"]["instance"]["nodes"]}
        self.penalties = {tuple(p): float(v) for p, v in zip(
            config["layout"]["demands"], config["static"]["sinks"]["pi"], strict=True)}
        self.value_scale = max(self.penalties.values(), default=1.0)
        self.nominal_demands = {
            p: float(nodes[p[0]]["sink"]["demand"][network.commodity_names[p[1]]]["dbar"])
            for p in self.demands
        }
        self.receivers = {}
        for row, node in enumerate(config["layout"].get("osats", ())):
            osat = nodes[node]["osat"]
            products = {goods[raw]: goods[packed] for raw, packed in osat["packages"].items()}
            storage = {goods[name]: float(a["storage"]) for name, a in nodes[node]["stock"].items()}
            total = sum(storage[k] for k in products)
            for raw, packed in products.items():
                rate = float(osat["thr"]) * storage[raw] / total if total else 0.0
                if rate > 0:
                    self.receivers[node, raw] = (row, packed, rate, float(osat["thr"]), products, storage)
        if model_path is not None and self.strength > 0:
            try:
                model = json.loads(Path(model_path).read_text())
                weights = np.asarray(model["weights"], dtype=float)
                if tuple(model["features"]) == FEATURES and weights.shape == (len(FEATURES),) and np.isfinite(weights).all():
                    self.weights = weights
            except (OSError, ValueError, KeyError, TypeError):
                pass

    @staticmethod
    def _seen(observation, key, index):
        try:
            if not observation[key + ".observed"][index]:
                return None
            value = float(observation[key][index])
            return value if math.isfinite(value) and value >= 0 else None
        except (KeyError, IndexError, TypeError, ValueError):
            return None

    def _stock(self, pair, observation):
        row = self.stock_rows.get(pair)
        return None if row is None else self._seen(observation, "stock.qty", row)

    def features(self, pair, observation):
        attrs = self.receivers.get(pair)
        if attrs is None:
            return None
        node, raw = pair
        row, packed, rate, nominal, products, storage = attrs
        own = self._stock(pair, observation)
        out = self._stock((node, packed), observation)
        capacity = self._seen(observation, "graph_now.osat.thr_eff", row)
        others = [(self._stock((node, k), observation), self._stock((node, v), observation),
                   nominal * storage[k] / sum(storage[i] for i in products), storage[v])
                  for k, v in products.items() if k != raw]
        if own is None or out is None or capacity is None or any(a is None or b is None for a, b, _, _ in others):
            return None
        own_cover = min(12.0, own / rate) / 6.0
        other_cover = min(12.0, sum(a / max(r, 1e-9) for a, _, r, _ in others) / max(1, len(others))) / 6.0
        output_fill = min(2.0, out / max(storage[packed], 1e-9))
        other_fill = sum(min(2.0, b / max(s, 1e-9)) for _, b, _, s in others) / max(1, len(others))
        week = int(observation["week"][0])
        wip = 0.0
        keys = tuple("wip." + name for name in ("qty", "node", "k", "out_week"))
        if all(key in observation and key + ".observed" in observation for key in keys):
            seen = np.logical_and.reduce([np.asarray(observation[key + ".observed"], dtype=bool) for key in keys])
            for i in np.flatnonzero(seen):
                values = [self._seen(observation, key, i) for key in keys]
                if all(v is not None for v in values):
                    qty, destination, commodity, due = values
                    if destination == node and commodity == packed and week <= due <= week + 2:
                        wip += qty
        export_capacity, weighted_value, weighted_cover, weighted_demand = 0.0, 0.0, 0.0, 0.0
        edges = set()
        for slot in self.network.slots_from.get((node, packed), ()):
            route = self.network.routes[slot]
            target = route.destination_node, packed
            if target not in self.demands or not observation["action_mask"][slot]:
                continue
            cap = self._seen(observation, "graph_now.u", route.edge_id)
            stocks = self._stock(target, observation)
            if cap is None or stocks is None:
                return None
            if route.edge_id in edges:
                continue
            edges.add(route.edge_id)
            demand_row = self.demands[target]
            forecast = [self._seen(observation, "demand_forecast.qty", (demand_row, h))
                        for h in range(min(4, observation["demand_forecast.qty"].shape[1]))]
            demand = sum(v for v in forecast if v is not None) / max(1, sum(v is not None for v in forecast))
            if not any(v is not None for v in forecast):
                demand = self.nominal_demands[target]
            cost = 0.0
            for edge in route.edges:
                freight = self._seen(observation, "graph_now.c", edge)
                tariff = self._seen(observation, "graph_now.tariff", (edge, packed))
                if freight is None or tariff is None:
                    return None
                cost += freight + tariff * self.goods_value[packed]
            value = max(0.0, self.penalties[target] - cost) / self.value_scale
            export_capacity += cap
            weighted_value += cap * value
            weighted_cover += cap * min(12.0, stocks / max(1.0, demand)) / 6.0
            weighted_demand += cap * demand / rate
        export = min(4.0, export_capacity / rate)
        value = weighted_value / export_capacity if export_capacity else 0.0
        cover = weighted_cover / export_capacity if export_capacity else 0.0
        demand = min(4.0, weighted_demand / export_capacity) if export_capacity else 0.0
        result = np.asarray((own_cover, other_cover, output_fill, other_fill,
                             min(2.0, capacity / nominal), min(12.0, wip / rate) / 6.0,
                             export, value, cover, demand, own_cover * export,
                             output_fill * export, (self.horizon - week + 1) / self.horizon,
                             export * value), dtype=float)
        return result if np.isfinite(result).all() else None

    def adjust(self, priorities, slots, caps, observation):
        if self.weights is None:
            return priorities
        bounds = np.asarray(caps, dtype=float)[slots]
        if not np.isfinite(bounds).all() or not np.isfinite(priorities).all() or np.any(bounds < 0):
            return priorities
        pairs = [(self.network.routes[s].destination_node, self.network.routes[s].commodity_id) for s in slots]
        features = {p: self.features(p, observation) for p in set(pairs)}
        if len(features) < 2 or any(x is None for x in features.values()):
            return priorities
        values = np.asarray([float(features[p] @ self.weights) for p in pairs])
        center = np.average(values, weights=np.maximum(1e-12, bounds))
        scale = np.exp(self.strength * np.clip(values - center, -1.0, 1.0))
        return (np.asarray(priorities) * scale).tolist()
