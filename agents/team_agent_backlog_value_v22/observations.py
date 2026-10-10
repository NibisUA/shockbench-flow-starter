"""Mask-aware access to the public flat observation, without simulator imports."""

import math
from operator import index

import numpy as np


if __package__:
    from .contracts import Quantity
else:
    from contracts import Quantity


class ObservationReader:
    def __init__(self, observation):
        self.observation = observation
        self._fields = {}

    def field(self, key):
        if key not in self._fields:
            values = np.asarray(self.observation[key])
            seen = np.asarray(self.observation[f"{key}.observed"])
            if values.shape != seen.shape or not np.all(np.isin(seen, (0, 1))):
                raise ValueError(f"{key}: invalid observed mask")
            self._fields[key] = values, seen.astype(bool)
        return self._fields[key]

    def number(self, key, position, *, upper=None):
        values, seen = self.field(key)
        if not seen[position]:
            return None
        value = values[position]
        if isinstance(value, (bool, np.bool_, str, bytes)) or np.iscomplexobj(value):
            raise ValueError(f"{key}: expected a nonnegative real number")
        value = float(value)
        if not math.isfinite(value) or value < 0 or (upper is not None and value > upper):
            raise ValueError(f"{key}: invalid observed number")
        return value

    def integer(self, key, position, *, low=0, high=None):
        values, seen = self.field(key)
        if not seen[position]:
            return None
        value = values[position]
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{key}: expected an integer")
        try:
            value = index(value)
        except TypeError as exc:
            raise ValueError(f"{key}: expected an integer") from exc
        if value < low or (high is not None and value > high):
            raise ValueError(f"{key}: index/week outside its range")
        return value

    def quantity(self, key, position):
        value = self.number(key, position)
        return Quantity(value, "unknown" if value is None else "observed")

    def live(self, key):
        values, seen = self.field(key)
        visible = values[seen]
        if values.dtype.kind not in "iuf" or np.any(~np.isfinite(visible)) or np.any(visible < 0):
            raise ValueError(f"{key}: invalid observed quantities")
        return np.argwhere(seen & (values > 0))
