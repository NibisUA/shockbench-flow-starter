"""V3 conditional FIFO forecast, not a promise about unseen future disruptions.

Current observed rates/bans persist; default release, no future dispatches or
overrides. Known inbound shipments are advanced through every remaining queue.
Kappa already includes open fraction. Cohorts share next-edge caps then pool
throughput, with duplicate fleet slack last, as in the public simulator.
"""

import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace


if __package__:
    from .contracts import ExpectedArrival, Quantity
    from .observations import ObservationReader
else:
    from contracts import ExpectedArrival, Quantity
    from observations import ObservationReader


ASSUMPTIONS = (
    "observed edge caps, pool throughput and prohibitions persist",
    "default FIFO release; no future overrides, holds or new dispatches",
    "only observed in-flight cargo and explicitly proposed pipeline are future inflow",
    "unknown future disruptions and announcements are not a known reopening schedule",
)


@dataclass(frozen=True, slots=True)
class QueueVisit:
    source_id: str
    chokepoint_node: int
    pool: str
    arrival_week: int
    evaluated_week: int
    work_ahead: Quantity
    same_cohort_competition: Quantity
    first_release_week: int | None
    completion_release_week: int | None


@dataclass(frozen=True, slots=True)
class QueueForecast:
    arrivals: tuple[ExpectedArrival, ...]
    visits: tuple[QueueVisit, ...]
    completion_weeks: Mapping[str, int | None]
    assumptions: tuple[str, ...] = ASSUMPTIONS
    issues: tuple[str, ...] = ()


@dataclass(slots=True)
class _Lot:
    source: str
    node: int
    commodity: int
    lane: int
    next_edge: int
    entered: int
    quantity: float


class QueueForecaster:
    def __init__(self, config, *, max_weeks=None):
        if max_weeks is not None and (not isinstance(max_weeks, int) or isinstance(max_weeks, bool) or max_weeks < 1):
            raise ValueError("max_weeks must be a positive integer or None")
        self.config = config
        self.max_weeks = max_weeks
        static = config["static"]
        self.pools = tuple(static["commodities"]["pool"])
        self.allowed = tuple(frozenset(goods) for goods in static["edges"]["K"])
        self.fleet_terms = self._fleet_terms(static)
        params = static["instance"]["params"]
        self.fleet_caps = {pool: params["fleet_share"][pool] * params["fleet_measure"][pool] for pool in ("tb", "ct")}

    @staticmethod
    def _fleet_terms(static):
        edges, lanes = static["edges"], static["lanes"]
        terms = defaultdict(list)
        tau = edges["tau0"]
        for edge, ref in enumerate(edges["alt_of"]):
            if edges["mode"][edge] == "sea" and ref is not None:
                replaced = tau[ref["edge"]] if "edge" in ref else sum(tau[e] for e in lanes["edges"][ref["lane"]])
                terms[edge].append((None, tau[edge] - replaced))
        for lane, ref in enumerate(lanes["alt_of"]):
            path = lanes["edges"][lane]
            if ref is None or any(edges["mode"][edge] != "sea" for edge in path):
                continue
            if "lane" in ref:
                replaced_path = lanes["edges"][ref["lane"]]
                divergence = next(edge for edge in path if edge not in replaced_path)
                delta = sum(tau[edge] for edge in path) - sum(tau[edge] for edge in replaced_path)
            else:
                original = ref["edge"]
                divergence = next(
                    edge for edge in path if edges["tail"][edge] == edges["tail"][original] and edge != original
                )
                delta = sum(tau[edge] for edge in path[path.index(divergence) :]) - tau[original]
            term = (lane if divergence == path[0] else None, delta)
            if term not in terms[divergence]:
                terms[divergence].append(term)
        return dict(terms)

    def _fleet_weights(self, lot):
        return sum(delta for lane, delta in self.fleet_terms.get(lot.next_edge, ()) if lane is None or lane == lot.lane)

    def _rates(self, reader, network, sources):
        edges, pools = set(), set()
        for lane, commodity in sorted({(source["lane"], source["commodity"]) for source in sources.values()}):
            for edge in network.lane_edges[lane]:
                node = network.edge_tail[edge]
                if node in network.chokepoints:
                    edges.add(edge)
                    pools.add((node, self.pools[commodity]))
        capacities, throughput, banned = {}, {}, {}
        positions = {node: row for row, node in enumerate(network.chokepoints)}
        for edge in edges:
            capacity = reader.number("graph_now.u", edge)
            if capacity is None:
                return None
            capacities[edge] = capacity
            for commodity in self.allowed[edge]:
                flag = reader.integer("graph_now.prohibited", (edge, commodity), high=1)
                if flag is None:
                    return None
                banned[edge, commodity] = bool(flag)
        for node, pool in pools:
            position = positions[node]
            rate = reader.number(f"graph_now.kappa.{pool}", position)
            opened = reader.number("graph_now.open", position, upper=1)
            if rate is None or opened is None:
                return None
            throughput[node, pool] = 0.0 if opened == 0 else rate
        return capacities, throughput, banned

    def forecast(self, state, observation, network, *, proposed_pipeline=()):
        reader = ObservationReader(observation)
        sources, events, book = {}, defaultdict(list), []
        incomplete = [issue for issue in state.issues if issue.startswith(("pipeline:", "queue:"))]
        # Queue completion depends on live cargo, edge/pool rates, permissions,
        # and transit times. Unrelated on-hand inventory is not an input to the
        # no-future-dispatch FIFO projection; hiding a stock slot must not erase
        # valid ETAs for cargo already in transit.
        for cargo in tuple(state.pipeline) + tuple(proposed_pipeline):
            if cargo.edge_id is None or cargo.lane_status == "unknown" or cargo.edge_arrival_week is None:
                incomplete.append(f"queue_forecast:{cargo.lot_id}:unknown_inbound")
                continue
            progress = network.transit_progress(cargo.edge_id, cargo.lane_id)
            if progress.reaches_destination:
                continue  # Its final-edge arrival is already observed, no queue forecast needed.
            if cargo.quantity.value is None or cargo.lane_id is None:
                incomplete.append(f"queue_forecast:{cargo.lot_id}:unknown_quantity_or_lane")
                continue
            if cargo.lot_id in sources:
                raise ValueError("duplicate forecast source_id")
            sources[cargo.lot_id] = dict(
                kind="pipeline",
                commodity=cargo.commodity_id,
                lane=cargo.lane_id,
                destination=progress.destination_node,
                quantity=cargo.quantity.value,
            )
            events[cargo.edge_arrival_week].append((cargo.lot_id, cargo.edge_id, cargo.quantity.value))
        for cargo in state.queues:
            if cargo.lane_id is None or cargo.next_edge_id is None or cargo.quantity.value is None:
                incomplete.append(f"queue_forecast:{cargo.lot_id}:unknown_queue_route")
                continue
            progress = network.transit_progress(cargo.next_edge_id, cargo.lane_id)
            if cargo.lot_id in sources:
                raise ValueError("duplicate forecast source_id")
            sources[cargo.lot_id] = dict(
                kind="queue",
                commodity=cargo.commodity_id,
                lane=cargo.lane_id,
                destination=progress.destination_node,
                quantity=cargo.quantity.value,
            )
            book.append(
                _Lot(
                    cargo.lot_id,
                    cargo.chokepoint_node,
                    cargo.commodity_id,
                    cargo.lane_id,
                    cargo.next_edge_id,
                    cargo.entered_week,
                    cargo.quantity.value,
                )
            )
        # A source's remaining legs use the live observed edge transit times.
        # Static tau0 is not a valid ETA when the network currently reports a
        # changed duration. Unknown times keep completion unknown.
        live_tau = list(network.edge_transit_weeks)
        active_edges = set()
        for cargo in tuple(state.pipeline) + tuple(proposed_pipeline):
            if cargo.edge_id is not None and cargo.lane_status != "unknown":
                active_edges.update(network.transit_progress(cargo.edge_id, cargo.lane_id).remaining_edges)
        for cargo in state.queues:
            if cargo.next_edge_id is not None and cargo.lane_id is not None:
                active_edges.add(cargo.next_edge_id)
                active_edges.update(network.transit_progress(cargo.next_edge_id, cargo.lane_id).remaining_edges)
        for edge in sorted(active_edges):
            weeks = reader.number("graph_now.tau", edge)
            if weeks is None or weeks != int(weeks):
                incomplete.append(f"queue_forecast:unknown_transit_time:{edge}")
                continue
            live_tau[edge] = int(weeks)
        if incomplete:
            return QueueForecast((), (), MappingProxyType({key: None for key in sources}), issues=tuple(incomplete))
        network = SimpleNamespace(
            edge_transit_weeks=tuple(live_tau),
            lane_edges=network.lane_edges,
            chokepoints=network.chokepoints,
            edge_head=network.edge_head,
            edge_tail=network.edge_tail,
            transit_progress=network.transit_progress,
        )
        rates = self._rates(reader, network, sources)
        if rates is None:
            return QueueForecast(
                (),
                (),
                MappingProxyType({key: None for key in sources}),
                issues=("queue_forecast:unknown_capacity_or_prohibition",),
            )
        capacities, throughput, banned = rates
        delivered, visits = defaultdict(float), {}
        end = state.horizon if self.max_weeks is None else min(state.horizon, state.week + self.max_weeks - 1)
        for week in range(state.week, end + 1):
            for source, edge, quantity in events.pop(week, ()):
                info = sources[source]
                node = network.edge_head[edge]
                if node == info["destination"]:
                    delivered[source, week] += quantity
                    continue
                path = network.lane_edges[info["lane"]]
                nxt = path[path.index(edge) + 1]
                if node not in network.chokepoints:
                    raise ValueError("forecast lane has an unsupported non-chokepoint intermediate node")
                book.append(_Lot(source, node, info["commodity"], info["lane"], nxt, week, quantity))
            # Work ahead is measured on the projected book AT arrival. Initial
            # queues have no historical book, so their evaluated week is today.
            cohort_totals, source_totals = defaultdict(float), defaultdict(float)
            for lot in book:
                pool = self.pools[lot.commodity]
                cohort_totals[lot.node, pool, lot.entered] += lot.quantity
                source_totals[lot.source, lot.node, lot.entered] += lot.quantity
            ahead, running = {}, defaultdict(float)
            for (node, pool, entered), qty in sorted(cohort_totals.items()):
                ahead[node, pool, entered] = running[node, pool]
                running[node, pool] += qty
            for (source, node, entered), qty in source_totals.items():
                key = source, node, entered
                pool = self.pools[sources[source]["commodity"]]
                if key not in visits:
                    visits[key] = dict(
                        ahead=ahead[node, pool, entered],
                        cohort=cohort_totals[node, pool, entered] - qty,
                        first=None,
                        last=None,
                        entered=0.0,
                        released=0.0,
                        evaluated=week,
                    )
                visit = visits[key]
                visit["entered"] = max(visit["entered"], qty + visit["released"])
            remaining_edges, remaining_pools = dict(capacities), dict(throughput)
            cohorts = defaultdict(list)
            for lot in book:
                if lot.quantity > 0:
                    cohorts[lot.node, lot.entered, self.pools[lot.commodity]].append(lot)
            tentative = []
            pool_order = {"tb": 0, "ct": 1}
            for (node, _entered, pool), cohort in sorted(
                cohorts.items(),
                key=lambda item: (
                    item[0][0],
                    item[0][1],
                    pool_order[item[0][2]],
                ),
            ):
                eligible = [
                    lot
                    for lot in cohort
                    if lot.commodity in self.allowed[lot.next_edge] and not banned[lot.next_edge, lot.commodity]
                ]
                totals = defaultdict(float)
                for lot in eligible:
                    totals[lot.next_edge] += lot.quantity
                edge_scale = {edge: min(1.0, remaining_edges[edge] / qty) for edge, qty in totals.items()}
                total = math.fsum(edge_scale[lot.next_edge] * lot.quantity for lot in eligible)
                scale = min(1.0, remaining_pools[node, pool] / total) if total > 0 else 0.0
                for lot in eligible:
                    qty = min(lot.quantity, scale * edge_scale[lot.next_edge] * lot.quantity)
                    remaining_edges[lot.next_edge] = max(0.0, remaining_edges[lot.next_edge] - qty)
                    tentative.append((lot, qty))
                remaining_pools[node, pool] = max(0.0, remaining_pools[node, pool] - scale * total)
            fleet_totals = defaultdict(float)
            for lot, qty in tentative:
                fleet_totals[self.pools[lot.commodity]] += self._fleet_weights(lot) * qty
            for lot, qty in tentative:
                pool = self.pools[lot.commodity]
                if self._fleet_weights(lot) and fleet_totals[pool] > self.fleet_caps[pool]:
                    qty *= self.fleet_caps[pool] / fleet_totals[pool]
                if qty <= 0:
                    continue
                lot.quantity = max(0.0, lot.quantity - qty)
                visit = visits[lot.source, lot.node, lot.entered]
                visit["released"] += qty
                visit["first"] = week if visit["first"] is None else visit["first"]
                visit["last"] = week
                arrival = week + network.edge_transit_weeks[lot.next_edge]
                if arrival == week:
                    if network.edge_head[lot.next_edge] != sources[lot.source]["destination"]:
                        raise ValueError("zero-time intermediate queue edge")
                    delivered[lot.source, week] += qty
                else:
                    events[arrival].append((lot.source, lot.next_edge, qty))
            book = [lot for lot in book if lot.quantity > 1e-12]
            if not book and not events:
                break
        arrivals, completion = [], {}
        pieces_by_source = defaultdict(list)
        for (source, week), qty in delivered.items():
            pieces_by_source[source].append((week, qty))
        for source, info in sources.items():
            pieces = sorted(pieces_by_source[source])
            total = math.fsum(qty for _week, qty in pieces)
            complete = total >= info["quantity"] - 1e-9 * max(1.0, info["quantity"])
            completion[source] = pieces[-1][0] if complete and pieces else None
            for week, qty in pieces:
                arrivals.append(
                    ExpectedArrival(
                        f"v3:{source}:{week}",
                        source,
                        info["kind"],
                        info["destination"],
                        info["commodity"],
                        Quantity(qty, "estimated"),
                        week,
                        "estimated",
                    )
                )
            residual = max(0.0, info["quantity"] - total)
            if not complete:
                arrivals.append(
                    ExpectedArrival(
                        f"v3:{source}:unknown",
                        source,
                        info["kind"],
                        info["destination"],
                        info["commodity"],
                        Quantity(residual, "estimated"),
                        None,
                        "unknown",
                    )
                )
        records = []
        for (source, node, entered), visit in visits.items():
            complete = visit["released"] >= visit["entered"] - 1e-9 * max(1.0, visit["entered"])
            records.append(
                QueueVisit(
                    source,
                    node,
                    self.pools[sources[source]["commodity"]],
                    entered,
                    visit["evaluated"],
                    Quantity(visit["ahead"], "estimated"),
                    Quantity(visit["cohort"], "estimated"),
                    visit["first"],
                    visit["last"] if complete else None,
                )
            )
        return QueueForecast(tuple(arrivals), tuple(records), MappingProxyType(completion))


def retrospective_backtest(predictions, actual_snapshots):
    """Evaluate archived forecasts using later observations; never called by Agent.

    Inputs are (origin_week, QueueForecast) pairs and later StateSnapshots.
    Grouped cargo has no stable cross-week ID, so evaluation is aggregate by
    (destination, commodity, arrival week). Returns JSON-ready error records.
    Unknown actual timing is excluded, never coerced to zero.
    """
    actual = defaultdict(float)
    for snapshot in actual_snapshots:
        for arrival in snapshot.arrivals:
            if (
                arrival.arrival_week == snapshot.week
                and arrival.quantity.value is not None
                and arrival.source == "observed"
            ):
                actual[arrival.destination_node, arrival.commodity_id, snapshot.week] += arrival.quantity.value
    predicted = {}
    for origin_week, forecast in predictions:
        for arrival in forecast.arrivals:
            if (
                arrival.arrival_week is not None
                and arrival.arrival_week > origin_week
                and arrival.quantity.value is not None
            ):
                key = arrival.destination_node, arrival.commodity_id, arrival.arrival_week
                previous = predicted.get(key)
                # Forecasts are refreshed weekly; compare the last available
                # forecast before arrival, never sum repeated forecasts.
                if previous is None or origin_week > previous[0]:
                    predicted[key] = origin_week, arrival.quantity.value
    keys = sorted(set(predicted) | set(actual))
    return tuple(
        {
            "destination_node": node,
            "commodity_id": commodity,
            "arrival_week": week,
            "predicted_quantity": (
                predicted[(node, commodity, week)][1] if (node, commodity, week) in predicted else None
            ),
            "actual_quantity": actual.get((node, commodity, week)),
            "absolute_error": (
                abs(predicted[(node, commodity, week)][1] - actual[(node, commodity, week)])
                if (node, commodity, week) in actual and (node, commodity, week) in predicted
                else None
            ),
            "cargo_identity": "aggregate; grouped lots have no stable IDs",
        }
        for node, commodity, week in keys
    )
