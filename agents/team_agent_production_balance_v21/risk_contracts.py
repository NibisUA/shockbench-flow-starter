"""Preserve analytics' experimental RiskState independently of V1 contracts.

The legacy RiskAnalyzer is not enabled in the integrated allocation pipeline.
Its scores are signals, not calibrated probabilities.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RiskState:
    week: int
    chokepoint_risk: dict[int, float] = field(default_factory=dict)
    edge_risk: dict[int, float] = field(default_factory=dict)
    region_risk: dict[int, float] = field(default_factory=dict)
    global_risk: float = 0.0
    active_threat_ids: set[int] = field(default_factory=set)
    pending_sanctions: list[dict[str, Any]] = field(default_factory=list)
    closure_forecasts: dict[int, int | None] = field(default_factory=dict)
