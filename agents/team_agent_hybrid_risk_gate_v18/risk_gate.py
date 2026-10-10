"""Conservative gate for invoking need-priority repair during visible stress.

Only observed closures, observed route prohibitions, and observed sink backlog
are used. Warning scores are deliberately excluded until an out-of-sample
calibration supports using them.
"""

import math

import numpy as np


class RiskGate:
    def __init__(
        self,
        config,
        network,
        *,
        min_closure_fraction=0.05,
        min_blocked_fraction=0.02,
        backlog_trigger=True,
    ):
        for name, value in (
            ("min_closure_fraction", min_closure_fraction),
            ("min_blocked_fraction", min_blocked_fraction),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number in [0, 1]")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite number in [0, 1]")
        if not isinstance(backlog_trigger, bool):
            raise ValueError("backlog_trigger must be a boolean")
        self.network = network
        self.min_closure_fraction = float(min_closure_fraction)
        self.min_blocked_fraction = float(min_blocked_fraction)
        self.backlog_trigger = backlog_trigger
        self.slot_capacity = np.asarray(
            [network.edge_capacity[route.edge_id] for route in network.routes], dtype=float
        )
        self.total_capacity = float(self.slot_capacity.sum())
        self.last_features = {}
        self.last_reasons = ()

    def _closure_fraction(self, observation):
        values = observation.get("graph_now.open")
        visible = observation.get("graph_now.open.observed")
        if values is None or visible is None or self.total_capacity <= 0:
            return 0.0
        values = np.asarray(values, dtype=float).reshape(-1)
        visible = np.asarray(visible).reshape(-1).astype(bool)
        if values.shape != visible.shape:
            return 0.0
        exposed = 0.0
        for route, capacity in zip(self.network.routes, self.slot_capacity, strict=True):
            positions = route.chokepoint_positions
            if not positions or any(position >= len(visible) or not visible[position] for position in positions):
                continue
            open_fraction = np.clip(values[list(positions)], 0.0, 1.0)
            exposed += float(capacity) * (1.0 - float(np.prod(open_fraction)))
        return min(1.0, max(0.0, exposed / self.total_capacity))

    def _blocked_capacity_fraction(self, observation):
        mask = observation.get("action_mask")
        visible = observation.get("action_mask.observed")
        if mask is None or visible is None or self.total_capacity <= 0:
            return 0.0
        mask = np.asarray(mask).reshape(-1)
        visible = np.asarray(visible).reshape(-1)
        if visible.size != 1 or not visible[0] or mask.shape != self.slot_capacity.shape:
            return 0.0
        blocked = np.isfinite(mask) & (mask <= 0)
        return min(1.0, max(0.0, float(self.slot_capacity[blocked].sum()) / self.total_capacity))

    @staticmethod
    def _has_observed_backlog(observation):
        quantity = observation.get("backlog.qty")
        visible = observation.get("backlog.qty.observed")
        if quantity is None or visible is None:
            return False
        quantity = np.asarray(quantity, dtype=float).reshape(-1)
        visible = np.asarray(visible).reshape(-1).astype(bool)
        if quantity.shape != visible.shape:
            return False
        return bool(np.any(visible & np.isfinite(quantity) & (quantity > 1e-10)))

    def should_apply(self, observation):
        closure_fraction = self._closure_fraction(observation)
        blocked_fraction = self._blocked_capacity_fraction(observation)
        backlog = self._has_observed_backlog(observation)
        reasons = []
        if closure_fraction >= self.min_closure_fraction and closure_fraction > 0:
            reasons.append("observed_closure")
        if blocked_fraction >= self.min_blocked_fraction and blocked_fraction > 0:
            reasons.append("observed_route_prohibition")
        if self.backlog_trigger and backlog:
            reasons.append("observed_backlog")
        self.last_features = {
            "closure_capacity_fraction": closure_fraction,
            "blocked_capacity_fraction": blocked_fraction,
            "observed_backlog": backlog,
        }
        self.last_reasons = tuple(reasons)
        return bool(reasons)
