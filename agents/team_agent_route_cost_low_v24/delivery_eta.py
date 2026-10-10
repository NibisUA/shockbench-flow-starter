"""Bounded candidate FIFO forecasts using the state owner's existing engine.

Proposals are hypothetical pipeline, never current stock or future reservations.
Each allocator call owns its cache and deterministic forecast budget.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np


if __package__:
    from .contracts import PipelineLot, Quantity
else:
    from contracts import PipelineLot, Quantity


class CandidateETA:
    MAX_FORECASTS = 16
    MAX_WEEKS = 24

    def __init__(self, forecaster, state, observation, network, snapshot):
        self.forecaster = forecaster
        self.state, self.observation = state, observation
        self.network, self.snapshot = network, snapshot
        self.proposals, self.completions = [], {}
        self.cache = {}
        self.calls = 0
        self.rejections = set()
        # Topology is static; future transit persists at observed current tau.
        self.forecast_network = SimpleNamespace(
            edge_transit_weeks=tuple(int(tau) for tau in snapshot.fields["graph_now.tau"].values),
            lane_edges=network.lane_edges,
            chokepoints=network.chokepoints,
            edge_head=network.edge_head,
            edge_tail=network.edge_tail,
            transit_progress=network.transit_progress,
        )

    def _proposal(self, slot, quantity):
        route = self.network.routes[slot]
        return PipelineLot(
            f"allocation:{len(self.proposals)}:{slot}",
            route.edge_id,
            route.commodity_id,
            route.lane_id,
            "known",
            Quantity(quantity, "estimated"),
            self.state.week + self.forecast_network.edge_transit_weeks[route.edge_id],
            route.destination_node,
            remaining_edges=route.edges[1:],
            remaining_route_weeks=sum(self.network.edge_transit_weeks[edge] for edge in route.edges[1:]),
        )

    def evaluate(self, option, quantity):
        route = self.network.routes[option.slot_id]
        if not route.chokepoints:
            return option
        key = option.slot_id, quantity
        if key in self.cache:
            return self.cache[key]
        reason, eta = None, None
        if self.calls >= self.MAX_FORECASTS:
            reason = "queue_eta_forecast_budget_exhausted"
        else:
            # Every future leg in the joint forecast needs known transit.
            lanes = {route.lane_id}
            lanes.update(lot.lane_id for lot in (*self.state.pipeline, *self.state.queues, *self.proposals))
            edges = sorted({edge for lane in lanes if lane is not None for edge in self.network.lane_edges[lane]})
            if not np.all(self.snapshot.fields["graph_now.tau"].observed[edges]):
                reason = "queue_eta_transit_unknown"
            else:
                proposal = self._proposal(option.slot_id, quantity)
                self.calls += 1
                forecast = self.forecaster.forecast(
                    self.state,
                    self.observation,
                    self.forecast_network,
                    proposed_pipeline=(*self.proposals, proposal),
                )
                eta = forecast.completion_weeks.get(proposal.lot_id)
                if forecast.issues:
                    reason = "queue_eta_inputs_unknown"
                    eta = None
                elif eta is None:
                    reason = "queue_eta_completion_unresolved"
                elif any(
                    forecast.completion_weeks.get(source) is None or forecast.completion_weeks[source] > previous
                    for source, previous in self.completions.items()
                ):
                    # A lower-priority addition must not silently invalidate
                    # an earlier selected shipment's reported completion.
                    reason = "queue_eta_delays_selected_shipment"
                    eta = None
        if reason:
            self.rejections.add(reason)
        flags = tuple(flag for flag in option.delay_flags if flag != "queue_work_unknown")
        delay = None if eta is None else max(0, eta - option.no_wait_arrival_week)
        result = replace(
            option,
            quantity=quantity,
            transport_cost=option.transport_cost_per_unit * quantity,
            estimated_completion_week=eta,
            queue_delay_weeks=delay,
            beyond_horizon=eta is not None and eta > self.state.horizon,
            delay_flags=flags + ((reason,) if reason else ("conditional_fifo_forecast",)),
        )
        self.cache[key] = result
        return result

    def accept(self, option, quantity, completion):
        if self.network.routes[option.slot_id].chokepoints:
            proposal = self._proposal(option.slot_id, quantity)
            self.proposals.append(proposal)
            self.completions[proposal.lot_id] = completion
            # All later forecasts must include this newly selected cargo.
            self.cache.clear()
