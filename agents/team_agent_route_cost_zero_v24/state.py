"""Own state at t-1; dispatch precedes non-chokepoint arrivals and production."""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from contracts import ExpectedArrival, PipelineLot, Quantity, QueueLot
from observations import ObservationReader


@dataclass(frozen=True, slots=True)
class WorkInProgress:
    lot_id: str
    node_id: int
    commodity_id: int
    gross_quantity: Quantity
    out_week: int | None


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    week: int
    horizon: int
    available_stock: Mapping[tuple[int, int], Quantity]
    backlog: Mapping[tuple[int, int], Quantity]
    pipeline: tuple[PipelineLot, ...]
    queues: tuple[QueueLot, ...]
    arrivals: tuple[ExpectedArrival, ...]
    wip: tuple[WorkInProgress, ...] = ()
    supply_availability: Mapping[tuple[int, int], Quantity] | None = None
    queue_forecast: object = None
    issues: tuple[str, ...] = ()
    availability_mode: str = "pre_dispatch_stock_t_minus_1"

    @property
    def arrival_calendar(self):
        calendar = defaultdict(list)
        for arrival in self.arrivals:
            key = arrival.destination_node, arrival.commodity_id, arrival.arrival_week
            calendar[key].append(arrival)
        return MappingProxyType({key: tuple(entries) for key, entries in calendar.items()})


class StateBuilder:
    def __init__(self, config, *, queue_forecaster=None):
        self.config = config
        self.layout = config["layout"]
        self.horizon = config["T"]
        self.queue_forecaster = queue_forecaster
        self.lane_edges = frozenset(edge for path in config["static"]["lanes"]["edges"] for edge in path)

    def _mapping(self, reader, field, pairs):
        values, _seen = reader.field(field)
        if values.shape != (len(pairs),):
            raise ValueError(f"{field}: shape does not match layout")
        return MappingProxyType({tuple(pair): reader.quantity(field, i) for i, pair in enumerate(pairs)})

    def _lane(self, reader, field, row, network, edge):
        lane = reader.integer(field, row, high=len(network.lane_names) - 1)
        if lane is not None:
            return lane, "known"
        # A hidden lane on an edge used by a lane is ambiguous, not lane 0.
        if edge is not None and edge not in self.lane_edges:
            if (
                network.edge_tail[edge] not in network.chokepoints
                and network.edge_head[edge] not in network.chokepoints
            ):
                return None, "off_lane"
        return None, "unknown"

    def build(self, observation, network):
        reader = ObservationReader(observation)
        week = reader.integer("week", 0, low=1, high=self.horizon)
        if week is None:
            raise ValueError("week must be observed")
        stock = self._mapping(reader, "stock.qty", self.layout["stock_slots"])
        backlog = self._mapping(reader, "backlog.qty", self.layout["demands"])
        supply = self._mapping(reader, "graph_now.supply.avail", self.layout["supply_slots"])
        issues = []
        pipeline, queues, wip, arrivals = [], [], [], []
        for (row,) in reader.live("pipeline.qty"):
            edge = reader.integer("pipeline.edge", row, high=len(network.edge_names) - 1)
            commodity = reader.integer("pipeline.k", row, high=len(network.commodity_names) - 1)
            if commodity is None:
                issues.append(f"pipeline:{row}:unknown_commodity")
                continue
            if edge is not None and commodity not in self.config["static"]["edges"]["K"][edge]:
                raise ValueError("pipeline: commodity is incompatible with its edge")
            lane, status = self._lane(reader, "pipeline.lane", row, network, edge)
            due = reader.integer("pipeline.arrival_week", row, low=week)
            progress = None
            if edge is not None and status != "unknown":
                progress = network.transit_progress(edge, lane)
            destination = None if progress is None else progress.destination_node
            remaining_edges = () if progress is None else progress.remaining_edges
            remaining_times = []
            if progress is not None:
                for remaining_edge in remaining_edges:
                    transit = reader.number("graph_now.tau", remaining_edge)
                    if transit is None or transit != int(transit):
                        remaining_times = []
                        break
                    remaining_times.append(int(transit))
            remaining_weeks = (
                sum(remaining_times) if progress is not None and len(remaining_times) == len(remaining_edges) else None
            )
            lot_id = f"pipeline:{edge}:{commodity}:{status}:{lane}:{due}:{row}"
            qty = reader.quantity("pipeline.qty", row)
            pipeline.append(
                PipelineLot(
                    lot_id, edge, commodity, lane, status, qty, due, destination, remaining_edges, remaining_weeks
                )
            )
            if edge is None or status == "unknown" or due is None:
                issues.append(f"{lot_id}:incomplete_route_or_time")
            if destination is not None:
                final_week = due if progress.reaches_destination else None
                final_source = "observed" if final_week is not None else "unknown"
                if (
                    due is not None
                    and final_week is None
                    and remaining_weeks is not None
                    and not any(
                        network.edge_tail[remaining_edge] in network.chokepoints for remaining_edge in remaining_edges
                    )
                ):
                    final_week = due + remaining_weeks
                    final_source = "estimated"
                arrivals.append(
                    ExpectedArrival(
                        f"arrival:{lot_id}",
                        lot_id,
                        "pipeline",
                        destination,
                        commodity,
                        qty,
                        final_week,
                        final_source,
                    )
                )

        queue_values, _seen = reader.field("queue_lots.qty")
        dense = queue_values.ndim == 2
        if dense and queue_values.shape != (len(self.layout["lot_keys"]), self.horizon):
            raise ValueError("queue_lots.qty: shape does not match layout")
        for position in reader.live("queue_lots.qty"):
            if dense:
                row, column = map(int, position)
                c, commodity, lane, nxt = self.layout["lot_keys"][row]
                entered = column + 1
                lot_id = f"queue:{c}:{commodity}:{lane}:{nxt}:{entered}"
                status = "known"
                qty = reader.quantity("queue_lots.qty", (row, column))
            else:
                (row,) = position
                c = reader.integer("queue_lots.chokepoint", row, high=len(network.node_names) - 1)
                commodity = reader.integer("queue_lots.k", row, high=len(network.commodity_names) - 1)
                lane = reader.integer("queue_lots.lane", row, high=len(network.lane_names) - 1)
                nxt = reader.integer("queue_lots.next_edge", row, high=len(network.edge_names) - 1)
                entered = reader.integer("queue_lots.arrival_week", row, high=week - 1)
                original_id = reader.integer("queue_lots.lot_id", row)
                lot_id = f"queue:{original_id}" if original_id is not None else f"queue:row:{row}"
                status = "unknown" if lane is None else "known"
                qty = reader.quantity("queue_lots.qty", row)
                if c is None or commodity is None or entered is None:
                    issues.append(f"{lot_id}:missing_queue_metadata")
                    continue
            if c not in network.chokepoints or entered >= week:
                raise ValueError("queue: invalid chokepoint or future arrival cohort")
            destination = None
            if lane is not None and nxt is not None:
                if network.edge_tail[nxt] != c:
                    raise ValueError("queue next edge does not leave its chokepoint")
                destination = network.transit_progress(nxt, lane).destination_node
            else:
                issues.append(f"{lot_id}:incomplete_queue_route")
            queues.append(QueueLot(lot_id, c, commodity, lane, status, nxt, entered, qty))
            if destination is not None:
                arrivals.append(
                    ExpectedArrival(
                        f"arrival:{lot_id}",
                        lot_id,
                        "queue",
                        destination,
                        commodity,
                        qty,
                        None,
                        "unknown",
                    )
                )

        for (row,) in reader.live("wip.qty"):
            node = reader.integer("wip.node", row, high=len(network.node_names) - 1)
            commodity = reader.integer("wip.k", row, high=len(network.commodity_names) - 1)
            due = reader.integer("wip.out_week", row, low=week)
            if node is None or commodity is None:
                issues.append(f"wip:{row}:missing_output_metadata")
                continue
            if (node, commodity) not in stock:
                raise ValueError("wip: output does not have a stock slot")
            lot_id = f"wip:{node}:{commodity}:{due}:{row}"
            qty = reader.quantity("wip.qty", row)
            wip.append(WorkInProgress(lot_id, node, commodity, qty, due))
            # OSAT package WIP has a known completion week and is not subject
            # to the Fab scrap process, so observed lots are confirmed dated
            # supply. Fab WIP remains estimated because future scrap can
            # reduce its gross amount before it matures.
            confirmed_osat_output = (
                node in self.layout["osats"] and due is not None and qty.source == "observed"
            )
            arrival_qty = qty if confirmed_osat_output else Quantity(qty.value, "estimated")
            arrivals.append(
                ExpectedArrival(
                    f"arrival:{lot_id}",
                    lot_id,
                    "wip",
                    node,
                    commodity,
                    arrival_qty,
                    due,
                    "observed" if confirmed_osat_output else "estimated" if due is not None else "unknown",
                )
            )
        state = StateSnapshot(
            week,
            self.horizon,
            stock,
            backlog,
            tuple(pipeline),
            tuple(queues),
            tuple(arrivals),
            tuple(wip),
            supply,
            issues=tuple(issues),
        )
        if self.queue_forecaster is not None:
            forecast = self.queue_forecaster.forecast(state, observation, network)
            replaced = {arrival.source_id for arrival in forecast.arrivals}
            kept = tuple(arrival for arrival in state.arrivals if arrival.source_id not in replaced)
            state = replace(
                state,
                arrivals=kept + forecast.arrivals,
                queue_forecast=forecast,
                issues=tuple(dict.fromkeys(state.issues + forecast.issues)),
            )
        return state
