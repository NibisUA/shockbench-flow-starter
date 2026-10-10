"""V3: price and time estimates for existing delivery slots, without allocation."""

import math
from dataclasses import dataclass

import numpy as np


if __package__:
    from .network import StaticNetwork, _checked_index
else:  # Helpers are top-level modules beside agent.py in a submission ZIP.
    from network import StaticNetwork, _checked_index


@dataclass(frozen=True, slots=True)
class DeliveryOption:
    slot_id: int
    source_node: int
    destination_node: int
    commodity_id: int
    unit: str
    quantity: float
    freight_per_unit: float
    tariff_per_unit: float
    war_risk_per_unit: float
    transport_cost_per_unit: float
    transport_cost: float
    cost_observed: bool
    permission_observed: bool
    dispatchable_now: bool
    entry_capacity: float
    snapshot_throughput: float
    no_wait_arrival_week: int
    estimated_completion_week: int | None
    queue_delay_weeks: int | None
    late_weeks: int | None
    beyond_horizon: bool
    delay_flags: tuple[str, ...]
    uncertain_fields: tuple[str, ...]


class DeliveryEvaluator:
    """Evaluate each existing slot ending at (destination, commodity).

    Future prices/capacities persist at snapshot values for these estimates.
    No stock, fleet slack, future events or demand forecasts are inferred here.
    Queue work must be supplied by the state/needs module, per (node, pool),
    estimated at our arrival before that week's release. Missing queues leave
    completion unknown. Delay flags are conditions, not calibrated probabilities.
    """

    def __init__(self, config, network=None):
        self.network = network if network is not None else StaticNetwork(config)
        static = config["static"]
        self.customs_value = tuple(static["commodities"]["v"])
        self.war_cost = {}
        for c in self.network.chokepoints:
            costs = static["instance"]["nodes"][c]["chokepoint"]["war_risk_cost"]
            for k, name in enumerate(self.network.commodity_names):
                self.war_cost[c, k] = tuple(costs.get(name, (0.0, 0.0, 0.0)))

    def options(
        self,
        snapshot,
        destination_node,
        commodity_id,
        quantity=1.0,
        due_week=None,
        source_nodes=None,
        queue_work_at_arrival=None,
    ):
        """Return legal or labelled estimated-legal options, cheapest cost first.

        Accepts primitive DeliveryNeed fields until the shared contract is fixed.
        queue_work_at_arrival may be None, or a map (chokepoint node, pool) ->
        quantity ahead of us; an explicit zero establishes the no-queue assumption.
        """
        net = self.network
        destination_node = _checked_index(destination_node, len(net.node_names), "destination")
        commodity_id = _checked_index(commodity_id, len(net.commodity_names), "commodity")
        quantity = float(quantity)
        if not math.isfinite(quantity) or quantity <= 0:
            raise ValueError("quantity must be finite and positive")
        if due_week is not None:
            due_week = _checked_index(due_week - 1, snapshot.horizon, "due week") + 1
        sources = (
            None
            if source_nodes is None
            else {_checked_index(node, len(net.node_names), "source") for node in source_nodes}
        )
        candidates = []
        for slot in net.slots_to.get((destination_node, commodity_id), ()):
            route, status = net.routes[slot], snapshot.routes[slot]
            if not status.sanction_allowed or (sources is not None and route.source_node not in sources):
                continue
            es, k, ps = list(route.edges), route.commodity_id, list(route.chokepoint_positions)
            fields = snapshot.fields
            freight = math.fsum(float(fields["graph_now.c"].values[e]) for e in es)
            tariff = math.fsum(float(fields["graph_now.tariff"].values[e, k]) * self.customs_value[k] for e in es)
            premium = math.fsum(
                self.war_cost[c, k][int(fields["graph_now.war_risk"].values[p])] for c, p in zip(route.chokepoints, ps)
            )
            cost_observed = bool(
                np.all(fields["graph_now.c"].observed[es])
                and np.all(fields["graph_now.tariff"].observed[es, k])
                and np.all(fields["graph_now.war_risk"].observed[ps])
            )
            no_wait = snapshot.week + sum(int(fields["graph_now.tau"].values[e]) for e in es)
            flags = []
            if status.zero_capacity_edges:
                flags.append("zero_edge_capacity")
            if status.closed_chokepoints:
                flags.append("currently_closed_chokepoint")
            if any(fields[f"graph_now.kappa.{route.pool}"].values[p] <= 0 for p in ps):
                flags.append("zero_chokepoint_throughput")
            if due_week is not None and no_wait > due_week:
                flags.append("no_wait_arrival_after_due")
            if status.uncertain_fields:
                flags.append("estimated_network_data")
            if not status.permission_observed:
                flags.append("unconfirmed_permission")
            queue_delay = 0
            for c, p in zip(route.chokepoints, ps):
                if queue_work_at_arrival is None or (c, route.pool) not in queue_work_at_arrival:
                    queue_delay = None
                    flags.append("queue_work_unknown")
                    break
                ahead = float(queue_work_at_arrival[c, route.pool])
                if not math.isfinite(ahead) or ahead < 0:
                    raise ValueError("queue work must be finite and nonnegative")
                capacity = float(fields[f"graph_now.kappa.{route.pool}"].values[p])
                if capacity <= 0:
                    queue_delay = None
                    flags.append("zero_chokepoint_throughput")
                    break
                # Approximate extra completion weeks beyond our own batch's
                # service time, assuming no other arrivals and fixed throughput.
                queue_delay += max(0, math.ceil((ahead + quantity) / capacity) - math.ceil(quantity / capacity))
            completion = None
            if status.snapshot_throughput > 0 and queue_delay is not None:
                batch_delay = max(0, math.ceil(quantity / status.snapshot_throughput) - 1)
                completion = no_wait + batch_delay + queue_delay
                if batch_delay:
                    flags.append("multiple_dispatch_or_service_weeks")
            if queue_delay:
                flags.append("queue_delay_estimate")
            beyond = (completion if completion is not None else no_wait) > snapshot.horizon
            if beyond:
                flags.append("beyond_episode_horizon")
            total = freight + tariff + premium
            candidates.append(
                DeliveryOption(
                    slot_id=slot,
                    source_node=route.source_node,
                    destination_node=destination_node,
                    commodity_id=k,
                    unit=route.unit,
                    quantity=quantity,
                    freight_per_unit=freight,
                    tariff_per_unit=tariff,
                    war_risk_per_unit=premium,
                    transport_cost_per_unit=total,
                    transport_cost=total * quantity,
                    cost_observed=cost_observed,
                    permission_observed=status.permission_observed,
                    dispatchable_now=status.entry_capacity > 0,
                    entry_capacity=status.entry_capacity,
                    snapshot_throughput=status.snapshot_throughput,
                    no_wait_arrival_week=no_wait,
                    estimated_completion_week=completion,
                    queue_delay_weeks=queue_delay,
                    late_weeks=None if due_week is None or completion is None else max(0, completion - due_week),
                    beyond_horizon=beyond,
                    delay_flags=tuple(dict.fromkeys(flags)),
                    uncertain_fields=status.uncertain_fields,
                )
            )
        return tuple(
            sorted(
                candidates,
                key=lambda option: (option.transport_cost_per_unit, option.no_wait_arrival_week, option.slot_id),
            )
        )
