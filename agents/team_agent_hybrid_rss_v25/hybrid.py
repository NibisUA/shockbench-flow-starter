"""Opt-in need priorities and residual dispatch after the complete heuristic.

Heuristic stages produce requests; they do not spend inventory independently.
This module commits their combined request once, then shares the same ledger
with residual additions. Only today's entry edge, source stock and detour fleet
are spendable. A downstream edge or today's ordinary arrival is never reserved
as a current resource. Unknown delivery time remains unknown.
"""

import math
from collections import defaultdict

import numpy as np

if __package__:
    from .allocation import Allocator
    from .delivery_eta import CandidateETA
else:
    from allocation import Allocator
    from delivery_eta import CandidateETA


class IncompleteResources(ValueError):
    """The optional adjustment cannot certify resources from this observation."""


class DispatchLedger:
    """One account for final heuristic requests and all residual reservations."""

    def __init__(self, allocator, state, observation, snapshot, action):
        self.network, self.allocator = allocator.network, allocator
        self.state, self.snapshot = state, snapshot
        self.stock_start = {}
        for key, quantity in state.available_stock.items():
            if quantity.source != "observed" or quantity.value is None:
                raise IncompleteResources("source_stock_unknown")
            self.stock_start[key] = float(quantity.value)
        self.edge_start = np.asarray(snapshot.fields["graph_now.u"].values, dtype=float).copy()
        entries = {route.edge_id for route in self.network.routes}
        if any(not snapshot.fields["graph_now.u"].observed[e] for e in entries):
            raise IncompleteResources("entry_capacity_unknown")
        if not observation["action_mask.observed"][0]:
            raise IncompleteResources("dispatch_permission_unknown")
        self.release_bound = self._release_bound(state, snapshot, action)
        self.fleet_start = {
            pool: max(0.0, cap - self.release_bound[pool]) for pool, cap in allocator.fleet_caps.items()
        }
        self.stocks = dict(self.stock_start)
        self.edges = self.edge_start.copy()
        self.fleet = dict(self.fleet_start)
        self.flows = np.zeros(len(self.network.routes), dtype=float)
        self.arrivals = []
        self.reservations = []

    def _release_bound(self, state, snapshot, action):
        """Bound both overridden and default releases before new detour cargo.

        The simulator clips releases before dispatch, then applies one fleet
        factor to both. An upper bound is intentional: it protects every
        release chosen by the heuristic without claiming to predict its FIFO.
        Entry edges cannot leave chokepoints, so release edge/throughput budgets
        do not overlap action entries. Unknown live cargo consumes all fleet.
        """
        a, net = self.allocator, self.network
        fields = snapshot.fields
        if any(issue.startswith(("pipeline:", "queue:")) for issue in getattr(state, "issues", ())):
            return dict(a.fleet_caps)
        modes = dict(zip(map(tuple, a.config["layout"].get("release_pairs", ())),
                         action.get("release_mode", ())))
        release_modes = a.config.get("release_modes", {"default": 0, "override": 1, "hold": 2})
        default_mode, override_mode = release_modes.get("default", 0), release_modes.get("override", 1)
        content, groups = defaultdict(float), defaultdict(list)
        positions = {node: p for p, node in enumerate(net.chokepoints)}

        def cargo(node, commodity, lane, edge, quantity):
            if edge is None or lane is None or quantity.value is None:
                raise IncompleteResources("release_cargo_unknown")
            content[node, commodity] += float(quantity.value)
            if modes.get((node, commodity), default_mode) != default_mode:
                return
            add(node, commodity, lane, edge, float(quantity.value))

        def add(node, commodity, lane, edge, quantity):
            weight = a._weight(edge, lane)
            if weight <= 0 or quantity <= 0:
                return
            if commodity not in a.allowed[edge]:
                return
            if not fields["graph_now.prohibited"].observed[edge, commodity]:
                raise IncompleteResources("release_permission_unknown")
            if fields["graph_now.prohibited"].values[edge, commodity]:
                return
            pool = a.pools[commodity]
            position = positions[node]
            if not (fields["graph_now.u"].observed[edge]
                    and fields[f"graph_now.kappa.{pool}"].observed[position]):
                raise IncompleteResources("release_capacity_unknown")
            groups[node, edge, pool].append((quantity, weight))

        try:
            for lot in state.queues:
                cargo(lot.chokepoint_node, lot.commodity_id, lot.lane_id, lot.next_edge_id, lot.quantity)
            for lot in state.pipeline:
                if lot.edge_id is None or lot.edge_arrival_week is None:
                    raise IncompleteResources("release_inbound_unknown")
                node = net.edge_head[lot.edge_id]
                if node not in positions or lot.edge_arrival_week != state.week:
                    continue
                progress = net.transit_progress(lot.edge_id, lot.lane_id)
                cargo(node, lot.commodity_id, lot.lane_id, progress.remaining_edges[0], lot.quantity)
            overrides = a.config["static"].get("override_slots", {})
            for slot, quantity in enumerate(action.get("override_qty", ())):
                node, commodity = overrides["chokepoint"][slot], overrides["k"][slot]
                if modes.get((node, commodity), default_mode) == override_mode:
                    add(node, commodity, overrides["lane"][slot], overrides["out_edge"][slot],
                        min(float(quantity), content[node, commodity]))
        except IncompleteResources:
            return dict(a.fleet_caps)
        bound = {pool: 0.0 for pool in a.fleet_caps}
        self.release_resources = []
        for (node, edge, pool), lots in groups.items():
            quantity = min(sum(q for q, _weight in lots), float(fields["graph_now.u"].values[edge]),
                           float(fields[f"graph_now.kappa.{pool}"].values[positions[node]]))
            weight = max(weight for _quantity, weight in lots)
            bound[pool] += quantity * weight
            self.release_resources.append(dict(node=node, edge=edge, pool=pool, quantity_upper_bound=quantity))
        return {pool: min(a.fleet_caps[pool], bound[pool]) for pool in bound}

    def available(self, slot):
        route = self.network.routes[slot]
        weight = self.allocator._weight(route.edge_id, route.lane_id)
        quantity = min(self.stocks.get((route.source_node, route.commodity_id), 0.0), self.edges[route.edge_id])
        if weight > 0:
            quantity = min(quantity, self.fleet[route.pool] / weight)
        return max(0.0, quantity)

    def reserve(self, slot, quantity, stage, *, need_id=None):
        quantity = max(0.0, min(float(quantity), self.available(slot)))
        if quantity <= 0:
            return 0.0
        route = self.network.routes[slot]
        source = route.source_node, route.commodity_id
        weight = self.allocator._weight(route.edge_id, route.lane_id)
        self.stocks[source] = max(0.0, self.stocks[source] - quantity)
        self.edges[route.edge_id] = max(0.0, self.edges[route.edge_id] - quantity)
        self.fleet[route.pool] = max(0.0, self.fleet[route.pool] - weight * quantity)
        self.flows[slot] += quantity
        self.reservations.append(dict(stage=stage, slot_id=slot, quantity=quantity, need_id=need_id))
        return quantity

    def project(self, requests):
        """Match the simulator's edge -> source pro-rata clipping order.

        Fleet uses the reserved release bound, so it is deliberately more
        conservative than joint fleet clipping. Actual data remain untouched.
        """
        flows = np.asarray(requests, dtype=float).copy()
        for edge in {route.edge_id for route in self.network.routes}:
            slots = [r.slot_id for r in self.network.routes if r.edge_id == edge]
            total = float(flows[slots].sum())
            if total > self.edge_start[edge]:
                flows[slots] *= max(0.0, self.edge_start[edge]) / total
        for source, slots in self.network.slots_from.items():
            indices = list(slots)
            total = float(flows[indices].sum())
            if total > self.stock_start.get(source, 0.0):
                flows[indices] *= self.stock_start.get(source, 0.0) / total
        for pool, cap in self.fleet_start.items():
            slots = [r.slot_id for r in self.network.routes
                     if r.pool == pool and self.allocator._weight(r.edge_id, r.lane_id) > 0]
            total = sum(flows[s] * self.allocator._weight(self.network.routes[s].edge_id,
                                                        self.network.routes[s].lane_id) for s in slots)
            if total > cap:
                flows[slots] *= cap / total
        return flows

    def report(self):
        return dict(
            stock=[dict(node=k[0], commodity=k[1], limit=v, used=v - self.stocks[k])
                   for k, v in self.stock_start.items()],
            entry=[dict(edge=e, limit=float(self.edge_start[e]), used=float(self.edge_start[e] - self.edges[e]))
                   for e in sorted({r.edge_id for r in self.network.routes})],
            fleet=[dict(pool=p, limit=self.allocator.fleet_caps[p], reserved_release=self.release_bound[p],
                        used=self.fleet_start[p] - self.fleet[p]) for p in self.fleet],
            arrivals=list(self.arrivals), reservations=list(self.reservations),
        )


class HybridController:
    def __init__(self, config, network, pipeline, *, priority_enabled=False, residual_enabled=False,
                 clipping_repair_enabled=False):
        self.config, self.network, self.pipeline = config, network, pipeline
        self.priority_enabled, self.residual_enabled = priority_enabled, residual_enabled
        self.clipping_repair_enabled = clipping_repair_enabled
        self.allocator = pipeline.allocator
        self.last_needs, self.last_trace, self.last_ledger = (), (), None
        self.stock_attrs = {}
        for node, attrs in enumerate(config["static"]["instance"]["nodes"]):
            for commodity, name in enumerate(network.commodity_names):
                if name in attrs.get("stock", {}):
                    self.stock_attrs[node, commodity] = attrs["stock"][name]

    @staticmethod
    def marginal_value(need, option, week, holding=0.0):
        """USD/native unit of one covered shortage week, less full-route costs.

        No confidence is manufactured for an unknown ETA. Backlog can still be
        useful when late; perishable/forecast demand cannot be retroactively
        served. Unknown VOLL is distinct from the explicit, unprofitable zero.
        """
        penalty = need.shortage_cost_per_unit_usd
        if penalty is None:
            return None
        if not math.isfinite(float(penalty)) or penalty < 0:
            raise ValueError("shortage_cost_per_unit_usd must be finite and nonnegative")
        eta = option.estimated_completion_week
        if eta is None:
            return None
        late = max(0, eta - need.due_week)
        backlog = "backlog" in need.reason
        avoided = 1.0 if late == 0 or backlog else 0.0
        # Earlier stock is held until the dated shortage; explicit confidence
        # describes forecast reliability only when the planner supplied it.
        if need.confidence is not None:
            avoided *= need.confidence
        wait = max(0, need.due_week - eta)
        return float(penalty) * avoided - option.transport_cost_per_unit - float(holding) * wait

    def _options(self, snapshot, needs):
        options = {}
        for need in needs:
            pair = need.destination_node, need.commodity_id
            if pair not in options:
                options[pair] = self.allocator.delivery.options(snapshot, *pair)
        return options

    def apply(self, action, observation, *, receiver_overflow=None):
        state = self.pipeline.state_builder.build(observation, self.network)
        needs = tuple(self.pipeline.need_planner.plan(state, observation, self.network))
        self.last_needs = needs
        trace = []
        snapshot = self.allocator._snapshot(observation)
        try:
            ledger = DispatchLedger(self.allocator, state, observation, snapshot, action)
        except IncompleteResources as exc:
            self.last_trace = (dict(stage="hybrid", reason=str(exc), week=state.week, changed_slots=[]),)
            self.last_ledger = None
            return action
        requests = np.asarray(action["flows"], dtype=float)
        projected = ledger.project(requests)
        options = self._options(snapshot, needs)
        if self.clipping_repair_enabled:
            return self._repair(
                action, state, snapshot, needs, options, ledger, projected, observation, receiver_overflow
            )
        slot_options = {option.slot_id: option for group in options.values() for option in group}
        remaining = {need.need_id: float(need.quantity) for need in needs}

        # Criticality affects only competition inside the existing requests.
        # Nonbinding requests and every preceding heuristic stage are retained.
        if self.priority_enabled and not np.allclose(projected, requests, rtol=1e-12, atol=1e-10):
            candidates = []
            for need in needs:
                pair = need.destination_node, need.commodity_id
                for option in options[pair]:
                    if requests[option.slot_id] <= 0:
                        continue
                    holding = self.stock_attrs.get(pair, {}).get("holding", 0.0)
                    value = self.marginal_value(need, option, state.week, holding)
                    if value is not None and value > 0:
                        candidates.append((-value, need.due_week, -need.priority, option.slot_id, need.need_id, need))
            need_left = dict(remaining)
            for _score, _due, _priority, slot, _id, need in sorted(candidates, key=lambda row: row[:5]):
                amount = min(need_left[need.need_id], requests[slot] - ledger.flows[slot])
                used = ledger.reserve(slot, amount, "need_priority", need_id=need.need_id)
                need_left[need.need_id] -= used
            # Keep unmeasured/zero-value heuristic requests with a common
            # pro-rata ratio within each remaining resource constraint.
            residual_requests = np.maximum(0.0, requests - ledger.flows)
            for slot in np.argsort(-projected, kind="stable"):
                ledger.reserve(int(slot), residual_requests[slot], "heuristic_remainder")
        else:
            for slot, quantity in enumerate(projected):
                ledger.reserve(slot, quantity, "heuristic")
        baseline = ledger.flows.copy()

        predictor = None
        if self.allocator.queue_forecaster is not None:
            predictor = CandidateETA(self.allocator.queue_forecaster, state, observation, self.network, snapshot)
            # All heuristic sea entries compete with residual cargo in every
            # forecast. Unknown completion is never certified as an arrival.
            for slot, quantity in enumerate(baseline):
                if quantity > 0 and self.network.routes[slot].chokepoints:
                    predictor.proposals.append(predictor._proposal(slot, float(quantity)))
        incoming = defaultdict(list)
        for slot, quantity in enumerate(baseline):
            if quantity <= 0:
                continue
            route = self.network.routes[slot]
            option = slot_options.get(slot)
            if option is None:
                route_options = self.allocator.delivery.options(snapshot, route.destination_node, route.commodity_id)
                option = next((item for item in route_options if item.slot_id == slot), None)
            eta = None if option is None else option.estimated_completion_week
            lower = (state.week + route.nominal_transit_weeks if option is None else option.no_wait_arrival_week)
            arrival = dict(slot_id=slot, node=route.destination_node, commodity=route.commodity_id,
                           quantity=float(quantity), eta=eta, physical_lower_bound=lower,
                           status="conditional_estimate" if eta is not None else "pending_unknown_eta")
            ledger.arrivals.append(arrival)
            incoming[route.destination_node, route.commodity_id].append([eta, lower, float(quantity)])
        # Consume each newly committed shipment once. Existing stock/WIP and
        # in-transit were already consumed by NeedPlanner, never added again.
        for need in sorted(needs, key=lambda item: (item.due_week, -item.priority, item.need_id)):
            for arrival in incoming[need.destination_node, need.commodity_id]:
                eta, lower, quantity = arrival
                receipt = lower if eta is None else eta
                if receipt > need.due_week and "backlog" not in need.reason:
                    continue
                used = min(remaining[need.need_id], quantity)
                if used <= 0:
                    continue
                arrival[2] -= used
                remaining[need.need_id] -= used
                trace.append(dict(stage="heuristic_need_reservation", need_id=need.need_id, quantity=used,
                                  eta=eta, physical_lower_bound=lower,
                                  reason="pending_eta_unknown" if eta is None else "dated_coverage"))

        if self.residual_enabled:
            self._residual(state, observation, snapshot, needs, remaining, options, ledger, predictor, action, trace)
        final = ledger.flows
        for slot in np.flatnonzero(~np.isclose(final, requests, rtol=1e-10, atol=1e-8)):
            trace.append(dict(stage="changed_action", slot_id=int(slot), week=state.week,
                              requested=float(requests[slot]), projected=float(projected[slot]),
                              heuristic_reserved=float(baseline[slot]), final=float(final[slot])))
        self.last_trace, self.last_ledger = tuple(trace), ledger
        return {**action, "flows": final}

    def _repair(self, action, state, snapshot, needs, options, ledger, projected, observation, receiver_overflow):
        """Recover profitable clipped requests without displacing executed cargo.

        The old hybrid greedily reassigned unpriced remainder even when no
        timely positive-value Need existed. Preserve the heuristic exactly in
        that case. Start with all projected commitments, then spend only slack
        on the *same* source/destination/commodity request, including an
        observed, timely alternative route. Queue ETA must be known.
        """
        requests = np.asarray(action["flows"], dtype=float)
        self.last_needs, self.last_ledger = needs, ledger
        if np.allclose(projected, requests, rtol=1e-12, atol=1e-10):
            self.last_trace = ({"stage": "repair", "reason": "no_binding_limit"},)
            return action
        for slot, quantity in enumerate(projected):
            ledger.reserve(slot, quantity, "protected_heuristic")
        displaced = defaultdict(float)
        for route in self.network.routes:
            key = route.source_node, route.destination_node, route.commodity_id
            displaced[key] += max(0.0, requests[route.slot_id] - projected[route.slot_id])
        candidates = []
        storage_certified = {}
        storage_unknown = []
        remaining = {need.need_id: need.quantity for need in needs}
        # Consume baseline receipts exactly once before considering additions.
        incoming = defaultdict(list)
        for pair, choices in options.items():
            for option in choices:
                quantity = projected[option.slot_id]
                if quantity > 0:
                    incoming[pair].append([option.estimated_completion_week, quantity])
        for need in sorted(needs, key=lambda n: (n.due_week, n.need_id)):
            pair = need.destination_node, need.commodity_id
            if pair not in storage_certified:
                storage_certified[pair] = self._receiver_storage_observed(pair, state, receiver_overflow)
                if not storage_certified[pair]:
                    storage_unknown.append(pair)
            if not storage_certified[pair]:
                continue
            for arrival in incoming[pair]:
                eta, quantity = arrival
                # Unknown existing cargo stays reserved, conservatively avoiding
                # another order for the same deficit; it is not labelled timely.
                if eta is not None and eta > need.due_week:
                    continue
                used = min(remaining[need.need_id], quantity)
                remaining[need.need_id] -= used
                arrival[1] -= used
            for option in options[pair]:
                route = self.network.routes[option.slot_id]
                key = route.source_node, route.destination_node, route.commodity_id
                eta = option.estimated_completion_week
                if (displaced[key] <= 1e-10 or eta is None or eta > need.due_week
                        or not option.cost_observed or not option.permission_observed
                        or not option.dispatchable_now
                        or not np.all(snapshot.fields["graph_now.tau"].observed[list(route.edges)])):
                    continue
                value = self.marginal_value(need, option, state.week, self.stock_attrs.get(pair, {}).get("holding", 0))
                if value is not None and value > 0:
                    candidates.append((-value, need.due_week, option.slot_id, need.need_id, key))
        trace, added = [], 0.0
        for pair in dict.fromkeys(storage_unknown):
            trace.append(dict(stage="clipping_recovery_prefilter", pair=pair,
                              reason="receiver_storage_unknown"))
        for value, due, slot, need_id, key in sorted(candidates):
            amount = min(displaced[key], remaining[need_id])
            route = self.network.routes[slot]
            if receiver_overflow is not None and amount > 1e-10:
                protected = ledger.flows.copy()
                proposed = protected.copy()
                proposed[slot] += amount
                previous_limits = receiver_overflow.last_limited
                try:
                    limited = receiver_overflow.apply(proposed, observation)
                    limited = np.asarray(limited, dtype=float)
                    if limited.shape != protected.shape or not np.all(np.isfinite(limited)):
                        amount = 0.0
                        trace.append(dict(stage="clipping_recovery_storage_guard", slot_id=slot,
                                          reason="invalid_projection"))
                    elif np.any(limited < protected - 1e-8):
                        amount = 0.0
                        trace.append(dict(stage="clipping_recovery_storage_guard", slot_id=slot,
                                          reason="would_trim_protected_flow"))
                    else:
                        amount = min(amount, max(0.0, float(limited[slot] - protected[slot])))
                except Exception as exc:
                    amount = 0.0
                    trace.append(dict(stage="clipping_recovery_storage_guard",
                                      slot_id=slot, reason=f"guard_error:{type(exc).__name__}"))
                finally:
                    receiver_overflow.last_limited = previous_limits
            used = ledger.reserve(slot, amount, "clipping_recovery", need_id=need_id)
            if used <= 1e-10:
                continue
            displaced[key] -= used
            remaining[need_id] -= used
            added += used
            option = next(item for item in options[(route.destination_node, route.commodity_id)]
                          if item.slot_id == slot)
            ledger.arrivals.append(dict(
                slot_id=slot, node=route.destination_node, commodity=route.commodity_id, quantity=used,
                eta=option.estimated_completion_week, physical_lower_bound=option.no_wait_arrival_week,
                status="conditional_estimate",
            ))
            trace.append(dict(stage="clipping_recovery", slot_id=slot, need_id=need_id,
                              quantity=used, due_week=due, marginal_value_usd_per_unit=-value))
        if added <= 1e-10:
            self.last_trace = ({"stage": "repair", "reason": "no_certified_profitable_recovery"},)
            return action
        for slot in np.flatnonzero(~np.isclose(ledger.flows, requests, rtol=1e-10, atol=1e-8)):
            trace.append(dict(stage="changed_action", slot_id=int(slot), week=state.week,
                              requested=float(requests[slot]), projected=float(projected[slot]),
                              final=float(ledger.flows[slot])))
        self.last_trace = tuple(trace)
        return {**action, "flows": ledger.flows}

    @staticmethod
    def _receiver_storage_observed(pair, state, receiver_overflow):
        """Do not add to a bounded receiver when its storage balance is unknown."""
        if receiver_overflow is None or not receiver_overflow.enabled or pair not in receiver_overflow.inputs:
            return True
        stock = state.available_stock.get(pair)
        if stock is None or stock.source != "observed" or stock.value is None:
            return False
        if any(issue.startswith(("pipeline:", "queue:")) for issue in getattr(state, "issues", ())):
            return False
        for arrival in state.arrivals:
            if (arrival.destination_node, arrival.commodity_id) != pair:
                continue
            if arrival.source in ("pipeline", "queue") and (
                arrival.quantity.value is None or arrival.arrival_week is None
            ):
                return False
        return True

    def _residual(self, state, observation, snapshot, needs, remaining, options, ledger, predictor, action, trace):
        candidates = []
        for need in needs:
            if remaining[need.need_id] <= 1e-10:
                continue
            pair = need.destination_node, need.commodity_id
            for option in options[pair]:
                route = self.network.routes[option.slot_id]
                physical_known = bool(np.all(snapshot.fields["graph_now.tau"].observed[list(route.edges)]))
                latest = need.due_week - (option.no_wait_arrival_week - state.week) if physical_known else None
                reason = None
                if not option.permission_observed:
                    reason = "permission_unknown"
                elif physical_known and option.no_wait_arrival_week > need.due_week:
                    reason = "created_too_late"
                elif not physical_known:
                    reason = "physical_eta_unknown"
                elif not option.dispatchable_now:
                    reason = "resource_capacity"
                if reason:
                    trace.append(dict(stage="residual_prefilter", need_id=need.need_id, slot_id=option.slot_id,
                                      reason=reason, latest_feasible_dispatch_week=latest,
                                      physical_lower_bound=option.no_wait_arrival_week if physical_known else None))
                    continue
                penalty = need.shortage_cost_per_unit_usd
                if penalty is None:
                    trace.append(dict(stage="residual_prefilter", need_id=need.need_id,
                                      slot_id=option.slot_id, reason="shortage_value_unknown"))
                    continue
                # Global resource ordering includes all commodities; final
                # candidate values are recomputed after quantity/ETA bounds.
                optimistic = float(penalty) - option.transport_cost_per_unit
                candidates.append((-optimistic, need.due_week, option.slot_id, need.need_id, need, option))
        nondefault_release = bool(np.any(action.get("release_mode", np.zeros(0, dtype=int)) != 0))
        for _score, _due, slot, _id, need, option in sorted(candidates, key=lambda item: item[:4]):
            route = self.network.routes[slot]
            amount = min(remaining[need.need_id], ledger.available(slot))
            if amount <= 1e-10:
                continue
            pair = need.destination_node, need.commodity_id
            attrs = self.stock_attrs.get(pair, {})
            if "storage" in attrs:
                stock = state.available_stock.get(pair)
                on_hand = 0.0 if stock is None else float(stock.value)
                pending = sum(float(arrival.quantity.value) for arrival in state.arrivals
                              if (arrival.destination_node, arrival.commodity_id) == pair
                              and arrival.quantity.value is not None)
                proposed = sum(item["quantity"] for item in ledger.arrivals
                               if (item["node"], item["commodity"]) == pair)
                amount = min(amount, max(0.0, float(attrs["storage"]) - on_hand - pending - proposed))
            if amount <= 1e-10:
                continue
            if route.chokepoints:
                # Current heuristic overrides are not represented by the FIFO
                # model. Do not certify an incompatible release forecast.
                option = predictor.evaluate(option, amount) if predictor is not None and not nondefault_release else option
            eta = option.estimated_completion_week
            if eta is None or eta > need.due_week or eta > state.horizon:
                trace.append(dict(stage="residual_prefilter", need_id=need.need_id, slot_id=slot,
                                  reason="eta_unknown" if eta is None else "late_arrival", eta=eta))
                continue
            value = self.marginal_value(need, option, state.week, attrs.get("holding", 0.0))
            if value is None or value <= 0:
                trace.append(dict(stage="residual_prefilter", need_id=need.need_id, slot_id=slot,
                                  reason="nonpositive_marginal_value", marginal_value_usd=value))
                continue
            used = ledger.reserve(slot, amount, "residual", need_id=need.need_id)
            remaining[need.need_id] -= used
            ledger.arrivals.append(dict(slot_id=slot, node=pair[0], commodity=pair[1], quantity=used,
                                        eta=eta, physical_lower_bound=option.no_wait_arrival_week,
                                        status="conditional_estimate"))
            if predictor is not None:
                predictor.accept(option, used, eta)
            trace.append(dict(stage="residual_assignment", need_id=need.need_id, slot_id=slot, quantity=used,
                              eta=eta, due_week=need.due_week, marginal_value_usd=value))
