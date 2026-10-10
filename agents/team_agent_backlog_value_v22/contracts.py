"""V1 handoff contracts only: no observation parsing, forecasting or allocation.

Node/commodity/edge/lane IDs index config['static']; slot IDs index action_slots.
Weeks are absolute and 1-based. Quantities use static.units[commodities.id[k]].
Higher priority wins; confidence is optional, not an uncalibrated risk score.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Mapping, Protocol, Sequence

import numpy as np


if TYPE_CHECKING:
    from network import StaticNetwork


CONTRACT_VERSION = 1
DataSource = Literal["observed", "estimated", "unknown"]
StockKey = tuple[int, int]
Observation = Mapping[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class Quantity:
    """Native commodity units; unknown is None, never a fabricated zero."""

    value: float | None
    source: DataSource
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class PipelineLot:
    """A live source row/group, not a forecast computed by this contract.

    edge_arrival_week refers to the current edge's head, not final delivery.
    lane_status distinguishes a known off-lane shipment from hidden lane data.
    """

    lot_id: str
    edge_id: int | None
    commodity_id: int
    lane_id: int | None
    lane_status: Literal["known", "off_lane", "unknown"]
    quantity: Quantity
    edge_arrival_week: int | None
    destination_node: int | None
    remaining_edges: tuple[int, ...] = ()
    remaining_route_weeks: int | None = None


@dataclass(frozen=True, slots=True)
class QueueLot:
    lot_id: str
    chokepoint_node: int
    commodity_id: int
    lane_id: int | None
    lane_status: Literal["known", "off_lane", "unknown"]
    next_edge_id: int | None
    entered_week: int
    quantity: Quantity


@dataclass(frozen=True, slots=True)
class ExpectedArrival:
    """One final-stock arrival; IDs prevent counting one shipment twice.

    arrival_week is None when final timing cannot be estimated. source describes
    timing provenance, independently of quantity.source. WIP uses source_kind
    'wip'. These records do not imply capacity reservations on future edges.
    """

    arrival_id: str
    source_id: str
    source_kind: Literal["pipeline", "queue", "wip", "supply"]
    destination_node: int
    commodity_id: int
    quantity: Quantity
    arrival_week: int | None
    source: DataSource


class StateSnapshot(Protocol):
    """Implemented by the state owner; stock is separate from future arrivals.

    available_stock is dispatchable stock under the documented weekly event
    order, not automatically stock.qty at t-1. The owner must specify which
    same-week supply/arrivals are included. No state computation lives here.
    """

    week: int
    horizon: int
    available_stock: Mapping[StockKey, Quantity]
    backlog: Mapping[StockKey, Quantity]
    pipeline: Sequence[PipelineLot]
    queues: Sequence[QueueLot]
    arrivals: Sequence[ExpectedArrival]


class DeliveryNeed(Protocol):
    """Implemented/produced by the needs owner. Overdue weeks remain overdue.

    quantity is a requested amount in the commodity's native units.
    due_week is the absolute receipt deadline at this destination. For an
    upstream production need it is back-planned from downstream demand by the
    relevant production lead time and, where known, current source-to-consumer
    transit ETA. The allocator evaluates each candidate's actual ETA against
    this receipt deadline, which implies a route-specific latest order week.
    shortage_cost_per_unit_usd is marginal USD per native unit for one week,
    or None if unknown; it is not a total cost or a tariff percentage.
    confidence is None unless a [0, 1] forecast confidence has a stated meaning.
    """

    need_id: str
    destination_node: int
    commodity_id: int
    quantity: float
    due_week: int
    priority: float
    reason: str
    shortage_cost_per_unit_usd: float | None
    confidence: float | None


def need_order_key(need: DeliveryNeed) -> tuple[float, int, str]:
    """Descending priority, earlier due week, then stable ID for ties."""
    return -need.priority, need.due_week, need.need_id


@dataclass(frozen=True, slots=True)
class UnmetNeed:
    need_id: str
    remaining_quantity: float
    reason: str


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    """Current dispatch/release usage, never a future lane reservation.

    resource_index: stock -> layout.stock_slots row, edge -> static.edges row,
    chokepoint_pool -> layout.chokepoints position (NOT its node ID).
    unit must state native quantity units. limit_source refers to the limit;
    limit=None and source='unknown' means there is no confirmed limit.
    """

    kind: Literal["stock", "edge", "chokepoint_pool"]
    resource_index: int
    unit: str
    used: float
    limit: float | None
    limit_source: DataSource
    pool: Literal["tb", "ct"] | None = None


@dataclass(frozen=True, slots=True)
class DecisionReason:
    code: str
    message: str
    need_id: str | None = None
    slot_id: int | None = None


@dataclass(frozen=True, slots=True)
class AllocationResult:
    """flows is float64 in action-slot order, never a capacity fraction.

    Optional release arrays use override_slots/layout.release_pairs order.
    Omission preserves the simulator's default release, not an explicit hold.
    Diagnostic tuples stay local; only the three action arrays go to the scorer.
    """

    flows: np.ndarray
    unmet_needs: tuple[UnmetNeed, ...] = ()
    resource_usage: tuple[ResourceUsage, ...] = ()
    reasons: tuple[DecisionReason, ...] = ()
    override_qty: np.ndarray | None = None
    release_mode: np.ndarray | None = None


class StateBuilder(Protocol):
    def build(self, observation: Observation, network: "StaticNetwork") -> StateSnapshot: ...


class NeedPlanner(Protocol):
    def plan(
        self, state: StateSnapshot, observation: Observation, network: "StaticNetwork"
    ) -> Sequence[DeliveryNeed]: ...


class Allocator(Protocol):
    def allocate(
        self, state: StateSnapshot, needs: Sequence[DeliveryNeed], observation: Observation, network: "StaticNetwork"
    ) -> AllocationResult: ...
