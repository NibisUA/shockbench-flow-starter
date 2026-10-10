"""Allocator V1: deadline-aware greedy using the unchanged team contracts.

Future lane edges/pools are not reserved. Current dispatch uses pre-dispatch
stock; same-week ordinary arrivals, supply and production are unavailable.
"""

import math
from collections import defaultdict

import numpy as np


if __package__:
    from .contracts import AllocationResult, DecisionReason, ResourceUsage, UnmetNeed
    from .delivery import DeliveryEvaluator
    from .delivery_eta import CandidateETA
    from .network import NetworkTracker, StaticNetwork
    from .queue_forecast import QueueForecaster
else:
    from contracts import AllocationResult, DecisionReason, ResourceUsage, UnmetNeed
    from delivery import DeliveryEvaluator
    from delivery_eta import CandidateETA
    from network import NetworkTracker, StaticNetwork
    from queue_forecast import QueueForecaster


class Allocator:
    def __init__(self, config, network=None, *, queue_eta_enabled=False):
        if not isinstance(queue_eta_enabled, bool):
            raise ValueError("queue_eta_enabled must be a boolean")
        self.config = config
        self.network = network if network is not None else StaticNetwork(config)
        self.tracker = NetworkTracker(config, self.network)
        self.delivery = DeliveryEvaluator(config, self.network)
        # Reuse the state owner's tested public-instance detour interpretation.
        fleet = QueueForecaster(config)
        self.fleet_terms, self.fleet_caps = fleet.fleet_terms, fleet.fleet_caps
        self.queue_forecaster = QueueForecaster(config, max_weeks=CandidateETA.MAX_WEEKS) if queue_eta_enabled else None
        self.stock_indices = {tuple(pair): i for i, pair in enumerate(config["layout"]["stock_slots"])}
        self.pools = tuple(config["static"]["commodities"]["pool"])
        self.allowed = tuple(config["static"]["edges"]["K"])
        self.last_snapshot = None
        self._snapshot_input = None
        # Local diagnostic hook; absent during normal Agent execution. Does
        # not alter the team handoff or keep an unbounded runtime trace.
        self.trace_callback = None

    def _trace(self, need, rank, week, slot, stage, reason, **details):
        if self.trace_callback is not None:
            self.trace_callback(
                dict(
                    week=week,
                    need_id=need.need_id,
                    need_rank=rank,
                    priority=need.priority,
                    due_week=need.due_week,
                    destination_node=need.destination_node,
                    commodity_id=need.commodity_id,
                    slot_id=slot,
                    stage=stage,
                    reason=reason,
                    **details,
                )
            )

    def _snapshot(self, observation):
        week = int(observation["week"][0])
        keys = ("week",) + tuple(name for field in self.tracker.nominal for name in (field, field + ".observed"))
        if self.last_snapshot is not None and week == self.last_snapshot.week:
            if any(not np.array_equal(observation[key], self._snapshot_input[key], equal_nan=True) for key in keys):
                raise ValueError("network observations changed within a previously evaluated week")
            return self.last_snapshot
        snapshot = self.tracker.update(observation)
        self._snapshot_input = {key: np.array(observation[key], copy=True) for key in keys}
        self.last_snapshot = snapshot
        return snapshot

    @staticmethod
    def _candidate_key(option, need, quantity):
        # ETA is for the assigned quantity, not a nominal lower bound. The V3
        # unit probe's batch correction is conservative for current throughput.
        eta = option.estimated_completion_week
        if eta is not None and option.quantity != quantity:
            eta += max(0, math.ceil(quantity / option.snapshot_throughput) - 1)
        late = None if eta is None else max(0, eta - need.due_week)
        cost = option.transport_cost_per_unit
        penalty = need.shortage_cost_per_unit_usd
        economic = cost + (late * penalty if late is not None and penalty is not None else 0)
        return (
            2 if eta is None else int(late > 0),
            economic,
            eta if eta is not None else math.inf,
            -quantity,
            option.slot_id,
        ), eta

    @staticmethod
    def _need_order_key(need):
        """Order shared-resource requests by VOLL, deadline, then policy rank.

        The planner's class priority remains the fallback when marginal
        shortage cost is unknown. When it is known, its USD/native-unit value
        is the primary signal, so low-value forecast/safety requests cannot
        consume stock, edge capacity, or fleet ahead of a critical shortage.
        Earlier receipt deadlines win ties; the policy rank remains a stable
        tie-breaker (for example, backlog before a forecast at the same date).
        """
        priority = float(need.priority)
        due = int(need.due_week)
        penalty = need.shortage_cost_per_unit_usd
        if penalty is None:
            return (1, -priority, due, need.need_id)
        penalty = float(penalty)
        if not math.isfinite(penalty) or penalty < 0:
            raise ValueError("shortage_cost_per_unit_usd must be finite and nonnegative")
        return (0, -penalty, due, -priority, need.need_id)

    def _weight(self, edge, lane):
        return sum(delta for match, delta in self.fleet_terms.get(edge, ()) if match is None or match == lane)

    def _release_fleet_bound(self, state, observation, snapshot):
        """Upper bound for today's automatic releases, not a future reservation.

        With complete visible cargo, group eligible releases by (edge, pool)
        and cap by edge/throughput. With incomplete cargo metadata, bound each
        chokepoint detour by its full current estimated capacity. This can
        over-reserve fleet; it never fabricates an empty queue during blackout.
        """
        net, fields = self.network, snapshot.fields
        incomplete = any(issue.startswith(("pipeline:", "queue:")) for issue in getattr(state, "issues", ()))
        # Cargo masks mark live rows/cohorts; zero masks also mark padding.
        # The parsed state's quality/metadata and own-stock visibility identify
        # incomplete own state, rather than declaring every empty row unknown.
        if any(qty.source != "observed" for qty in state.available_stock.values()):
            incomplete = True
        groups = defaultdict(list)
        positions = {node: p for p, node in enumerate(net.chokepoints)}

        def add(edge, commodity, lane, quantity):
            nonlocal incomplete
            if edge is None or lane is None or quantity.value is None:
                incomplete = True
                return
            if commodity not in self.allowed[edge] or fields["graph_now.prohibited"].values[edge, commodity]:
                return
            weight = self._weight(edge, lane)
            if weight:
                groups[edge, self.pools[commodity]].append((quantity.value, weight))

        for lot in state.queues:
            add(lot.next_edge_id, lot.commodity_id, lot.lane_id, lot.quantity)
        for lot in state.pipeline:
            if lot.edge_id is None or lot.edge_arrival_week is None:
                incomplete = True
                continue
            if lot.edge_arrival_week != state.week or net.edge_head[lot.edge_id] not in positions:
                continue
            if lot.lane_status != "known" or lot.lane_id is None:
                incomplete = True
                continue
            path = net.lane_edges[lot.lane_id]
            add(path[path.index(lot.edge_id) + 1], lot.commodity_id, lot.lane_id, lot.quantity)
        if incomplete:
            groups.clear()
            for edge, terms in self.fleet_terms.items():
                if net.edge_tail[edge] not in positions:
                    continue
                # Sum matching terms; a wildcard can coexist with lane terms.
                lanes = {lane for lane, _delta in terms} | {None}
                weight = max(self._weight(edge, lane) for lane in lanes)
                for pool in {self.pools[k] for k in self.allowed[edge]}:
                    groups[edge, pool].append((math.inf, weight))
        bound = {pool: 0.0 for pool in self.fleet_caps}
        for (edge, pool), lots in groups.items():
            position = positions[net.edge_tail[edge]]
            capacity = min(
                float(fields["graph_now.u"].values[edge]),
                float(fields[f"graph_now.kappa.{pool}"].values[position]),
            )
            if fields["graph_now.open"].values[position] == 0:
                capacity = 0.0
            bound[pool] += min(sum(qty for qty, _ in lots), capacity) * max(weight for _, weight in lots)
        return {pool: min(cap, bound[pool]) for pool, cap in self.fleet_caps.items()}

    def allocate(self, state, needs, observation, network=None):
        if network is not None and network is not self.network:
            raise ValueError("allocator must share the configured StaticNetwork")
        if getattr(state, "availability_mode", "pre_dispatch_stock_t_minus_1") != "pre_dispatch_stock_t_minus_1":
            raise ValueError("allocator requires pre-dispatch stock; same-week supply/arrivals are not dispatchable")
        if state.week != int(observation["week"][0]) or state.horizon != self.config["T"]:
            raise ValueError("state week/horizon does not match observation/config")
        snapshot = self._snapshot(observation)
        net = self.network
        predictor = (
            CandidateETA(self.queue_forecaster, state, observation, net, snapshot)
            if self.queue_forecaster is not None
            else None
        )
        # Unknown/estimated stock is not spent as if physically available.
        stock_start = {
            key: float(qty.value) if qty.source == "observed" and qty.value is not None else 0.0
            for key, qty in state.available_stock.items()
        }
        stocks = dict(stock_start)
        edge_start = snapshot.fields["graph_now.u"].values
        edges = edge_start.copy()
        release_bound = self._release_fleet_bound(state, observation, snapshot)
        fleet_start = {pool: max(0.0, cap - release_bound[pool]) for pool, cap in self.fleet_caps.items()}
        fleet = dict(fleet_start)
        flows = np.zeros(len(net.routes), dtype=np.float64)
        unmet, reasons, cache = [], [], {}
        for need_rank, need in enumerate(sorted(needs, key=self._need_order_key), 1):
            key = need.destination_node, need.commodity_id
            if key not in cache:
                cache[key] = self.delivery.options(snapshot, *key)
            if not cache[key]:
                slots = net.slots_to.get(key, ())
                code = "all_delivery_slots_prohibited" if slots else "no_action_slot_to_destination"
                reasons.append(
                    DecisionReason(
                        code,
                        f"Existing delivery slots {tuple(slots)} were excluded by current permission estimates."
                        if slots
                        else "StaticNetwork has no action slot ending at this destination for this commodity; "
                        "check upstream need semantics. No route was invented.",
                        need.need_id,
                    )
                )
                if self.trace_callback is not None:
                    self._trace(
                        need,
                        need_rank,
                        state.week,
                        None,
                        "route_lookup",
                        code,
                        static_slots=list(slots),
                        permission_observed=(
                            all(snapshot.routes[slot].permission_observed for slot in slots) if slots else None
                        ),
                        commodity_destinations=sorted(
                            {r.destination_node for r in net.routes if r.commodity_id == need.commodity_id}
                        ),
                        incoming_commodities=sorted(
                            {r.commodity_id for r in net.routes if r.destination_node == need.destination_node}
                        ),
                        need_reason=need.reason,
                    )
            remaining = float(need.quantity)
            blocked = set()
            options = cache[key]
            if predictor is not None:
                # Forecast scheduling only: routes that can still meet the
                # deadline without queue delay go first. This lower bound is
                # never promoted to a known/on-time completion estimate.
                options = sorted(
                    options,
                    key=lambda option: (
                        int(option.no_wait_arrival_week > need.due_week),
                        option.transport_cost_per_unit,
                        option.no_wait_arrival_week,
                        option.slot_id,
                    ),
                )
            while remaining > 0:
                candidates = []
                for option in options:
                    route = net.routes[option.slot_id]
                    origin = route.source_node, route.commodity_id
                    status = snapshot.routes[route.slot_id]
                    if status.closed_chokepoints or status.zero_capacity_edges or status.snapshot_throughput <= 0:
                        blocked.add("currently_blocked_delivery_route")
                        self._trace(
                            need, need_rank, state.week, route.slot_id, "prefilter", "currently_blocked_delivery_route"
                        )
                        continue
                    if not option.permission_observed:
                        blocked.add("unconfirmed_permission")
                        self._trace(need, need_rank, state.week, route.slot_id, "prefilter", "unconfirmed_permission")
                        continue
                    weight = self._weight(route.edge_id, route.lane_id)
                    quantity = min(remaining, stocks.get(origin, 0.0), float(edges[route.edge_id]))
                    if weight:
                        quantity = min(quantity, fleet[route.pool] / weight)
                    if quantity <= 0:
                        blocked.add("current_resources_exhausted_or_unknown")
                        code = (
                            "no_source_stock"
                            if stocks.get(origin, 0.0) <= 0
                            else "entry_capacity_exhausted"
                            if edges[route.edge_id] <= 0
                            else "fleet_budget_exhausted"
                        )
                        self._trace(
                            need,
                            need_rank,
                            state.week,
                            route.slot_id,
                            "prefilter",
                            code,
                            source_stock=stocks.get(origin, 0.0),
                            entry_capacity=float(edges[route.edge_id]),
                            fleet_remaining=fleet[route.pool],
                            fleet_weight=weight,
                            quantity=quantity,
                        )
                        continue
                    if predictor is not None:
                        # Even an empty FIFO queue cannot beat observed transit
                        # under the same persistence assumptions as the forecast.
                        # Hidden transit is not a known lower bound.
                        if option.no_wait_arrival_week > state.horizon and np.all(
                            snapshot.fields["graph_now.tau"].observed[list(route.edges)]
                        ):
                            blocked.add("estimated_arrival_beyond_horizon")
                            self._trace(
                                need,
                                need_rank,
                                state.week,
                                route.slot_id,
                                "prefilter",
                                "estimated_arrival_beyond_horizon",
                                no_wait_arrival=option.no_wait_arrival_week,
                            )
                            continue
                        calls_before = predictor.calls
                        cached_before = (option.slot_id, quantity) in predictor.cache
                        option = predictor.evaluate(option, quantity)
                        self._trace(
                            need,
                            need_rank,
                            state.week,
                            route.slot_id,
                            "eta",
                            next((flag for flag in option.delay_flags if flag.startswith("queue_eta_")), "estimated"),
                            source_stock=stocks.get(origin, 0.0),
                            entry_capacity=float(edges[route.edge_id]),
                            permission_observed=option.permission_observed,
                            fleet_remaining=fleet[route.pool],
                            fleet_weight=weight,
                            quantity=quantity,
                            calls_before=calls_before,
                            calls_after=predictor.calls,
                            cache_hit=cached_before,
                            eta=option.estimated_completion_week,
                            no_wait_arrival=option.no_wait_arrival_week,
                        )
                    rank, eta = self._candidate_key(option, need, quantity)
                    if eta is None:
                        blocked.add("delivery_eta_unknown")
                        blocked.update(flag for flag in option.delay_flags if flag.startswith("queue_eta_"))
                        continue
                    if eta > state.horizon:
                        blocked.add("estimated_arrival_beyond_horizon")
                        continue
                    candidates.append((rank, option, quantity, eta))
                if not candidates:
                    break
                _rank, option, qty, eta = min(candidates, key=lambda candidate: candidate[0])
                route = net.routes[option.slot_id]
                origin = route.source_node, route.commodity_id
                weight = self._weight(route.edge_id, route.lane_id)
                flows[route.slot_id] += qty
                stocks[origin] -= qty
                edges[route.edge_id] -= qty
                fleet[route.pool] = max(0.0, fleet[route.pool] - weight * qty)
                remaining = max(0.0, remaining - qty)
                self._trace(
                    need,
                    need_rank,
                    state.week,
                    route.slot_id,
                    "assignment",
                    "eta_late" if eta > need.due_week else "eta_on_time_estimate",
                    quantity=qty,
                    eta=eta,
                    late_weeks=max(0, eta - need.due_week),
                    overdue_at_dispatch=need.due_week < state.week,
                    no_wait_arrival=option.no_wait_arrival_week,
                )
                if predictor is not None:
                    predictor.accept(option, qty, eta)
                reasons.append(
                    DecisionReason(
                        "allocated_current_resources",
                        f"Assigned {qty:g} {route.unit}; full-route transport estimate "
                        f"{option.transport_cost_per_unit:g} USD/unit; conditional ETA week {eta}, "
                        f"due week {need.due_week}; later edges/pools are not reserved.",
                        need.need_id,
                        route.slot_id,
                    )
                )
                reasons.append(
                    DecisionReason(
                        "eta_late" if eta > need.due_week else "eta_on_time_estimate",
                        f"Conditional completion week {eta}; "
                        + (
                            "arrives after this need's deadline."
                            if eta > need.due_week
                            else "not a guaranteed arrival."
                        ),
                        need.need_id,
                        route.slot_id,
                    )
                )
                if option.uncertain_fields:
                    reasons.append(
                        DecisionReason(
                            "delivery_estimate_uncertain",
                            "Current capacity/price estimates or missing queue forecast make completion uncertain.",
                            need.need_id,
                            route.slot_id,
                        )
                    )
                if "conditional_fifo_forecast" in option.delay_flags:
                    reasons.append(
                        DecisionReason(
                            "queue_eta_estimate",
                            f"Joint FIFO estimate for {qty:g} {route.unit}: completion week {eta}; "
                            "current rates/transit persist, selected cargo is included; "
                            "future dispatches/disruptions are unknown; no future resources reserved.",
                            need.need_id,
                            route.slot_id,
                        )
                    )
            if remaining > 0:
                reason = (
                    ",".join(sorted(blocked)) or "current_resources_exhausted_or_unknown"
                    if cache[key]
                    else "no_permitted_delivery_slot"
                )
                unmet.append(UnmetNeed(need.need_id, remaining, reason))
                self._trace(need, need_rank, state.week, None, "unmet", reason, quantity=remaining)
        usage = []
        for key, start in stock_start.items():
            used = start - stocks[key]
            if used > 0:
                unit = net.routes[next(iter(net.slots_from[key]))].unit
                usage.append(ResourceUsage("stock", self.stock_indices[key], unit, used, start, "observed"))
        for edge in sorted({route.edge_id for route in net.routes}):
            used = float(edge_start[edge] - edges[edge])
            if used > 0:
                commodity = self.allowed[edge][0]
                unit = self.config["static"]["units"][net.commodity_names[commodity]]
                source = "observed" if snapshot.fields["graph_now.u"].observed[edge] else "estimated"
                usage.append(ResourceUsage("edge", edge, unit, used, float(edge_start[edge]), source))
        for pool in ("tb", "ct"):
            used = fleet_start[pool] - fleet[pool]
            if used > 0:
                reasons.append(
                    DecisionReason(
                        "fleet_budget_usage",
                        f"Used {used:g} of {fleet_start[pool]:g} residual {pool} native-unit weeks.",
                    )
                )
            if release_bound[pool] > 0:
                reasons.append(
                    DecisionReason(
                        "automatic_release_fleet_bound",
                        f"Reserved up to {release_bound[pool]:g} {pool} unit-weeks for today's automatic releases.",
                    )
                )
        if predictor is not None:
            reasons.append(
                DecisionReason(
                    "queue_eta_forecast_usage",
                    f"Ran {predictor.calls}/{predictor.MAX_FORECASTS} joint forecasts; "
                    f"lookahead at most {predictor.MAX_WEEKS} weeks. "
                    + ("Rejected estimates: " + ",".join(sorted(predictor.rejections)) if predictor.rejections else ""),
                )
            )
        return AllocationResult(flows, tuple(unmet), tuple(usage), tuple(reasons))
