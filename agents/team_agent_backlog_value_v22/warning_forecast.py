"""Shared public-observation features for optional choke warning calibration."""

import json
import math
from pathlib import Path

import numpy as np


FEATURES = ("intercept", "warning", "warning_ema", "warning_change", "region_warning",
            "closed_fraction", "closure_age")


class WarningForecast:
    def __init__(self, config, network):
        self.chokes = tuple(config["layout"].get("chokepoints", ()))
        units = config["layout"].get("warning_units", ())
        self.warning_rows = {int(unit): row for row, (kind, unit) in enumerate(units)
                             if kind == "chokepoint"}
        self.region_rows = {int(unit): row for row, (kind, unit) in enumerate(units) if kind == "region"}
        regions = config["static"]["nodes"].get("region", ())
        self.regions = {node: regions[node] for node in self.chokes if node < len(regions)}
        self.history = {}
        self.week = None
        self.cached = None
        self.model = None
        path = Path(__file__).parent / "warning_forecast.json"
        if path.is_file():
            try:
                model = json.loads(path.read_text(encoding="utf-8"))
                coefficients = np.asarray(model["coefficients"], dtype=float)
                if (model.get("features") == list(FEATURES) and model.get("horizons") == list(range(1, 7))
                        and coefficients.shape == (6, len(FEATURES)) and np.isfinite(coefficients).all()):
                    self.model = coefficients
            except (OSError, ValueError, TypeError, KeyError):
                pass

    @staticmethod
    def _observed(observation, key, row):
        if key not in observation or key + ".observed" not in observation:
            return None
        values, seen = observation[key], observation[key + ".observed"]
        if row < 0 or row >= len(values) or not seen[row]:
            return None
        value = float(values[row])
        return value if math.isfinite(value) else None

    def features(self, observation):
        week = int(observation["week"][0])
        if self.week == week and self.cached is not None:
            return self.cached
        if self.week is not None and week < self.week:
            self.history.clear()
        result = {}
        for row, node in enumerate(self.chokes):
            warning_row = self.warning_rows.get(node)
            signal = (None if warning_row is None else
                      self._observed(observation, "warning.score", warning_row))
            opening = self._observed(observation, "graph_now.open", row)
            if signal is None or opening is None:
                continue
            signal = float(np.clip(signal, -3.0, 3.0))
            opening = float(np.clip(opening, 0.0, 1.0))
            previous = self.history.get(node)
            elapsed = week - previous[0] if previous is not None else 0
            weight = 0.75 ** elapsed if previous is not None else 0.0
            average = weight * previous[1] + (1.0 - weight) * signal if previous else signal
            change = signal - previous[2] if previous else 0.0
            age = (previous[3] + elapsed if previous and previous[4] < 1.0 else 1) if opening < 1.0 else 0
            region_row = self.region_rows.get(self.regions.get(node))
            regional = (None if region_row is None else
                        self._observed(observation, "warning.score", region_row))
            regional = float(np.clip(regional, -3.0, 3.0)) if regional is not None else 0.0
            result[node] = np.asarray((1.0, signal, average, change, regional, 1.0 - opening,
                                       min(6, age) / 6.0))
            self.history[node] = week, average, signal, age, opening
        self.week, self.cached = week, result
        return result

    def probabilities(self, observation):
        if self.model is None:
            return {}
        return {node: 1.0 / (1.0 + np.exp(-np.clip(self.model @ values, -40.0, 40.0)))
                for node, values in self.features(observation).items()}
