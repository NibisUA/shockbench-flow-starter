"""Sequential stock projection; each existing arrival and backlog is consumed once."""

from collections import defaultdict
from dataclasses import dataclass, replace
from math import isfinite

from contracts import need_order_key
from observations import ObservationReader


@dataclass(frozen=True, slots=True)
class DeliveryNeed:
    need_id: str
    destination_node: int
    commodity_id: int
    quantity: float
    due_week: int
    priority: float
    reason: str
    shortage_cost_per_unit_usd: float | None
    confidence: float | None = None
    quantity_source: str = "estimated"
    assumptions: tuple[str, ...] = ()
    physical_min_transit_weeks: int | None = None
    latest_feasible_dispatch_week: int | None = None
    timing_status: str = "unknown"


def _nominal_fab_inputs(output_quantity: float) -> float:
    """Return nominal raw input for the current one-to-one Fab BOM proxy.

    Fab ``w_scr`` is the scrap observation window in weeks; it is not a
    fractional yield loss. ``tau`` is a duration as well. Neither belongs in
    the nominal material ratio. Any yield loss or reserve needs its own
    explicitly modeled policy parameter.
    """
    return output_quantity


@dataclass(frozen=True, slots=True)
class SafetyBufferPolicy:
    """Optional fractional buffer for each nominal BOM input.

    It is zero by default and separate from nominal BOM quantities. Existing
    grid ``ibar`` reserves remain controlled independently by ``safety_stock``.
    """

    input_buffer_fraction: float = 0.0

    def __post_init__(self):
        fraction = self.input_buffer_fraction
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not isfinite(fraction)
            or fraction < 0
        ):
            raise ValueError("input_buffer_fraction must be a finite nonnegative number")

    def apply(self, nominal_input: float) -> float:
        """Apply the explicit policy to one nominal input requirement."""
        return nominal_input * (1.0 + self.input_buffer_fraction)


class NeedPlanner:
    def __init__(
        self,
        config,
        *,
        production_horizon=8,
        include_estimated_arrivals=False,
        include_queue_forecast_arrivals=False,
        safety_stock=True,
        safety_buffer_policy: SafetyBufferPolicy | None = None,
        production_enabled=True,
        shortage_cost_model=False,
        fuel_replenishment_enabled=True,
        trace_enabled=False,
    ):
        if not isinstance(production_horizon, int) or isinstance(production_horizon, bool) or production_horizon < 1:
            raise ValueError("production_horizon must be a positive integer")
        self.config = config
        self.production_horizon = production_horizon
        self.include_estimated_arrivals = include_estimated_arrivals
        if not isinstance(include_queue_forecast_arrivals, bool):
            raise ValueError("include_queue_forecast_arrivals must be a boolean")
        self.include_queue_forecast_arrivals = include_queue_forecast_arrivals
        self.safety_stock = safety_stock
        self.safety_buffer_policy = safety_buffer_policy or SafetyBufferPolicy()
        self.production_enabled = bool(production_enabled)
        if not isinstance(shortage_cost_model, bool):
            raise ValueError("shortage_cost_model must be a boolean")
        self.shortage_cost_model = bool(shortage_cost_model)
        if not isinstance(fuel_replenishment_enabled, bool):
            raise ValueError("fuel_replenishment_enabled must be a boolean")
        self.fuel_replenishment_enabled = fuel_replenishment_enabled
        if not isinstance(trace_enabled, bool):
            raise ValueError("trace_enabled must be a boolean")
        self.trace_enabled = trace_enabled
        self.last_trace = ()
        self._trace_rows = []
        static = config["static"]
        self.commodities = {name: i for i, name in enumerate(static["commodities"]["id"])}
        self.nodes = {name: i for i, name in enumerate(static["nodes"]["id"])}
        self.profiles = {self.nodes[node["id"]]: node for node in static["instance"]["nodes"]}
        sinks = static["sinks"]
        self.penalties = {(n, k): float(pi) for n, k, pi in zip(sinks["node"], sinks["k"], sinks["pi"], strict=True)}
        self._backlog_since = {}
        self.last_issues = ()

    def _trace(self, event, **fields):
        if self.trace_enabled:
            self._trace_rows.append({"event": event, **fields})

    def _arrival_is_eligible(self, state, arrival):
        """Whether a dated lot may reduce demand by its receipt deadline.

        Conditional queue forecasts are eligible only through their dedicated
        switch. This keeps uncertain WIP and inferred pipeline ETAs excluded
        under the conservative default.
        """
        observed = arrival.source == "observed" and arrival.quantity.source == "observed"
        if observed or self.include_estimated_arrivals:
            return True
        return (
            self.include_queue_forecast_arrivals
            and state.queue_forecast is not None
            and arrival.source_kind == "queue"
            and arrival.source == "estimated"
            and arrival.quantity.source == "estimated"
        )

    def _downstream_penalty(self, output_name):
        """Return max reachable sink penalty through the declared 1:1 BOM.

        Sink ``pi`` has units USD/(sink unit * weekly cost period). The BOM
        currently has no conversion coefficients, so propagation is allowed
        only when adjacent commodity units match exactly.
        """
        units = self.config["static"].get("units", {})
        candidates = []
        for osat_node in self.config["layout"]["osats"]:
            profile = self.profiles[osat_node].get("osat", {}).get("packages", {})
            for raw, package in profile.items():
                if raw != output_name:
                    continue
                if units.get(raw) != units.get(package):
                    raise ValueError(
                        f"VOLL unit mismatch in OSAT BOM {raw}->{package}: {units.get(raw)!r} != {units.get(package)!r}"
                    )
                product = self.commodities[package]
                candidates.extend(pi for (sink, commodity), pi in self.penalties.items() if commodity == product)
        return max(candidates, default=None)

    def _shortage_cost(self, pair, reason, reader=None):
        """Marginal damage in USD per native input unit per weekly period.

        The optional model propagates published sink ``pi`` through explicit
        1:1 BOM links. For grid fuel, one GWh of unavailable fuel removes at
        most one GWh of segment generation (the simulator's segment stock is
        also GWh); its damage is grid VOLL [USD/GWh] plus the lost Fab output
        value [USD/output/week] times ``R/e`` [output/GWh]. Fuel shares set the
        segment's generation cap and are not a fuel conversion factor. The
        Fab term is a conservative max across Fabs connected to this grid.
        Missing energy observations disable the estimate. A unit mismatch is
        a configuration error and raises ValueError so an enabled VOLL model
        cannot silently return a dimensionally invalid cost.
        """
        if not self.shortage_cost_model:
            return None
        direct = self.penalties.get(pair)
        if direct is not None:
            return direct
        node, commodity = pair
        profile = self.profiles[node]
        if reason == "grid_fuel" and "grid" in profile:
            grid = profile["grid"]
            name = self.config["static"]["commodities"]["id"][commodity]
            unit = self.config["static"].get("units", {}).get(name)
            if unit not in {"GWh", "GWh fuel"}:
                raise ValueError(f"VOLL unit mismatch for grid fuel {name}: expected GWh, got {unit!r}")
            if float(grid.get("shares", {}).get(name, 0.0)) <= 0.0:
                return None
            voll = float(grid.get("voll", 0.0))
            if reader is None:
                return None
            terms = []
            for row, fab_node in enumerate(self.config["layout"]["fabs"]):
                fab = self.profiles[fab_node].get("fab", {})
                if fab.get("grid") != self.config["static"]["nodes"]["id"][node]:
                    continue
                e = float(fab.get("e", 0.0))  # GWh per output unit
                penalty = self._downstream_penalty(fab.get("product"))
                restoration = reader.number("graph_now.fab.R", row)
                if e <= 0 or penalty is None or restoration is None:
                    continue
                terms.append(penalty * restoration / e)
            return voll + max(terms, default=0.0)
        if reason == "production":
            input_name = self.config["static"]["commodities"]["id"][commodity]
            if "osat" in profile:
                package = profile["osat"].get("packages", {}).get(input_name)
                units = self.config["static"].get("units", {})
                if package is None:
                    return None
                if units.get(input_name) != units.get(package):
                    raise ValueError(
                        f"VOLL unit mismatch in OSAT BOM {input_name}->{package}: "
                        f"{units.get(input_name)!r} != {units.get(package)!r}"
                    )
                package_id = self.commodities[package]
                return max(
                    (pi for (sink, item), pi in self.penalties.items() if item == package_id),
                    default=None,
                )
            for fab_node in self.config["layout"]["fabs"]:
                fab = self.profiles[fab_node].get("fab", {})
                if fab_node != node or fab.get("input") != input_name:
                    continue
                input_unit = self.config["static"].get("units", {}).get(input_name)
                output_unit = self.config["static"].get("units", {}).get(fab.get("product"))
                if input_unit != output_unit:
                    raise ValueError(
                        f"VOLL unit mismatch in Fab BOM {input_name}->{fab.get('product')}: "
                        f"{input_unit!r} != {output_unit!r}"
                    )
                return self._downstream_penalty(fab.get("product"))
        return None

    def _coverage(self, state, pair, cutoff_week):
        stock = state.available_stock.get(pair)
        if stock is None or stock.value is None:
            return None
        coverage = stock.value
        for arrival in state.arrivals:
            if (arrival.destination_node, arrival.commodity_id) != pair:
                continue
            if arrival.quantity.value is None or arrival.arrival_week is None or arrival.arrival_week > cutoff_week:
                continue
            if not self._arrival_is_eligible(state, arrival):
                continue
            coverage += arrival.quantity.value
        return coverage

    def _minimum_route_transit_weeks(self, network, reader, source, destination, commodity):
        """Observed no-queue ETA for a specific source/destination commodity route.

        Queue work is modeled by the allocator's route completion estimate. This
        helper only backs production dates out by current observed edge transit;
        it never substitutes static ``tau0`` when a current edge time is hidden.
        """
        if network is None:
            return 0
        options = []
        for slot in network.slots_to.get((destination, commodity), ()):
            route = network.routes[slot]
            if route.source_node != source:
                continue
            transit = []
            for edge in route.edges:
                weeks = reader.number("graph_now.tau", edge)
                if weeks is None or weeks != int(weeks):
                    break
                transit.append(int(weeks))
            else:
                options.append(sum(transit))
        return min(options) if options else None

    def _fuel_horizon(self, state, reader, network, count):
        """Cover real fuel replenishment lead, using an explicit persistence model.

        This extends only the grid projection from today's observed load and
        known Fab input. Published sink demand is never extrapolated.
        """
        if network is None or not self.fuel_replenishment_enabled:
            return count
        horizon = count
        for node in self.config["layout"]["grids"]:
            for name in self.profiles[node]["grid"]["shares"]:
                if name not in self.commodities:
                    continue
                commodity = self.commodities[name]
                alternatives = []
                for slot in network.slots_to.get((node, commodity), ()):
                    route = network.routes[slot]
                    transit = self._minimum_route_transit_weeks(
                        network, reader, route.source_node, node, commodity
                    )
                    if transit is None:
                        continue
                    upstream = []
                    for predecessor in network.slots_to.get((route.source_node, commodity), ()):
                        source = network.routes[predecessor].source_node
                        if source == node:
                            continue
                        lead = self._minimum_route_transit_weeks(
                            network, reader, source, route.source_node, commodity
                        )
                        if lead is not None:
                            upstream.append(lead)
                    alternatives.append(transit + min(upstream) + 1 if upstream else transit)
                if alternatives:
                    horizon = max(horizon, min(alternatives) + 1)
        return min(horizon, state.horizon - state.week + 1)

    def _trace_chain_horizons(self, state, reader, network, forecast_width, issues):
        if network is None:
            return
        for node in self.config["layout"]["fabs"]:
            fab = self.profiles[node]["fab"]
            raw = self.commodities[fab["product"]]
            leads = []
            for osat in self.config["layout"]["osats"]:
                package = self.profiles[osat]["osat"]["packages"].get(fab["product"])
                first = self._minimum_route_transit_weeks(network, reader, node, osat, raw)
                if package is None or first is None:
                    continue
                product = self.commodities[package]
                for sink, item in self.config["layout"]["demands"]:
                    if item != product:
                        continue
                    last = self._minimum_route_transit_weeks(network, reader, osat, sink, product)
                    if last is not None:
                        leads.append(int(fab["tau"]) + first + int(self.profiles[osat]["osat"].get("tau", 0))
                                     + last + 2)
            minimum = min(leads) if leads else None
            truncated = minimum is not None and minimum >= forecast_width
            if truncated:
                issues.append(f"fab:{node}:forecast_window_shorter_than_chain:{forecast_width}:{minimum}")
            self._trace("production_horizon", pair=(node, self.commodities[fab["input"]]),
                        forecast_weeks=forecast_width, configured_horizon=self.production_horizon,
                        minimum_input_receipt_to_sink_weeks=minimum, forecast_window_too_short=truncated,
                        demand_extrapolated=False)

    def _supply_calendar(self, state, pair, issues, *, dispatch=False):
        """Mutable dated lots; receipt can feed production, dispatch is next week.

        The simulator dispatches from t-1 inventory before regular arrivals
        and WIP mature. A producer's arrival is consequently available for
        shipment at t+1, while a sink/production requirement may consume it at t.
        """
        stock = state.available_stock.get(pair)
        if stock is None or stock.value is None:
            issues.append(f"production:{pair}:unknown_time_phased_stock")
            self._trace("need_not_created", pair=pair, reason="unknown_stock", stage="calendar")
            return None
        lots = [[state.week, stock.value, "stock", "stock", state.week - 1]]
        for arrival in state.arrivals:
            if (arrival.destination_node, arrival.commodity_id) != pair:
                continue
            eligible = (
                arrival.arrival_week is not None
                and arrival.quantity.value is not None
                and self._arrival_is_eligible(state, arrival)
            )
            self._trace(
                "arrival_coverage", pair=pair, source_id=getattr(arrival, "source_id", None),
                source_kind=getattr(arrival, "source_kind", None), arrival_week=arrival.arrival_week,
                quantity=arrival.quantity.value, eligible=eligible, dispatch=dispatch,
            )
            if eligible:
                lots.append([
                    arrival.arrival_week + int(dispatch), arrival.quantity.value,
                    getattr(arrival, "arrival_id", "arrival"), getattr(arrival, "source_kind", "arrival"),
                    arrival.arrival_week,
                ])
        lots.sort(key=lambda item: (item[0], item[2]))
        return lots

    def _consume_supply(self, calendar, deadline, quantity, *, pair=None, stage="coverage"):
        remaining = quantity
        for lot in calendar:
            if lot[0] > deadline:
                break
            covered = min(lot[1], remaining)
            lot[1] -= covered
            remaining -= covered
            if covered > 0:
                self._trace(
                    "calendar_consumption", pair=pair, stage=stage, deadline=deadline,
                    source_id=lot[2], source_kind=lot[3], available_week=lot[0],
                    arrival_week=lot[4], consumed=covered, lot_remaining=lot[1],
                )
            if remaining <= 0:
                break
        return remaining

    def _net_events(self, state, pair, events, issues):
        """Net dated requirements against stock and eligible arrivals once."""
        calendar = self._supply_calendar(state, pair, issues)
        if calendar is None:
            return None
        remaining = []
        for due, quantity in sorted(events):
            missing = self._consume_supply(calendar, max(state.week, due), quantity, pair=pair)
            if missing > 0:
                remaining.append((due, missing))
        return remaining

    def _requirements(self, state, reader, issues, network=None):
        requirements = defaultdict(list)
        values, _seen = reader.field("demand_forecast.qty")
        if values.ndim != 2 or values.shape[0] != len(self.config["layout"]["demands"]):
            raise ValueError("demand_forecast.qty: shape does not match layout")
        for row, pair in enumerate(self.config["layout"]["demands"]):
            pair = tuple(pair)
            for h in range(min(values.shape[1], state.horizon - state.week + 1)):
                qty = reader.number("demand_forecast.qty", (row, h))
                if qty is None:
                    issues.append(f"demand:{pair}:{state.week + h}:unknown")
                elif qty > 0:
                    reason = "current_demand" if h == 0 else "forecast_demand"
                    requirements[pair].append(
                        (
                            state.week + h,
                            qty,
                            3.0,
                            reason,
                            self.penalties[pair],
                            ("published demand forecast, not realized demand",),
                        )
                    )
        count = min(self.production_horizon, state.horizon - state.week + 1)
        self._trace_chain_horizons(state, reader, network, min(count, values.shape[1]), issues)
        grid_energy = defaultdict(float)
        # Translate only published downstream package demand into upstream
        # production inputs. No capacity becomes a target by itself.
        if self.production_enabled:
            # First remove each sink's own stock/arrivals and backlog, in date
            # order. Gross forecast is not an upstream production requirement.
            package_events = defaultdict(list)
            for (sink, product), rows in tuple(requirements.items()):
                events = [(due, qty) for due, qty, *_ in rows if due < state.week + count]
                backlog = state.backlog.get((sink, product))
                if backlog is not None and backlog.value is not None and backlog.value > 0:
                    events.append((self._backlog_since.get((sink, product), state.week), backlog.value))
                events = self._net_events(state, (sink, product), events, issues)
                if events is None:
                    continue
                package_events[product].extend((sink, due, qty) for due, qty in events)

            # Package inventories belong to their physical OSAT. A shared
            # dispatch calendar can cover only sinks with a declared route.
            raw_events = defaultdict(list)
            osat_weekly_capacity = defaultdict(float)
            osat_rows = {node: row for row, node in enumerate(self.config["layout"]["osats"])}
            package_calendars = {}
            for product, events in sorted(package_events.items()):
                package = self.config["static"]["commodities"]["id"][product]
                for sink, need_date, quantity in sorted(events, key=lambda item: (item[1], item[0])):
                    choices = []
                    for node in self.config["layout"]["osats"]:
                        raw_names = [raw for raw, output in self.profiles[node]["osat"]["packages"].items()
                                     if output == package]
                        if not raw_names:
                            continue
                        transit = self._minimum_route_transit_weeks(network, reader, node, sink, product)
                        if transit is not None:
                            choices.append((transit, node, raw_names[0]))
                    choices.sort()
                    if not choices:
                        issues.append(f"osat:{product}:no_known_route_to:{sink}")
                        self._trace("need_not_created", pair=(sink, product), reason="no_route",
                                    stage="osat_projection", quantity=quantity, due_week=need_date)
                        continue
                    remaining = quantity
                    for transit, node, _raw in choices:
                        pair = node, product
                        if pair not in package_calendars:
                            package_calendars[pair] = self._supply_calendar(state, pair, issues, dispatch=True)
                        calendar = package_calendars[pair]
                        if calendar is not None:
                            remaining = self._consume_supply(
                                calendar, max(state.week, need_date - transit), remaining,
                                pair=pair, stage="osat_output",
                            )
                        if remaining <= 0:
                            break
                    for transit, node, raw in choices:
                        if remaining <= 0:
                            break
                        if package_calendars[(node, product)] is None:
                            continue
                        cap = reader.number("graph_now.osat.thr_eff", osat_rows[node])
                        if cap is None:
                            issues.append(f"osat:{node}:unknown_throughput")
                            continue
                        tau = int(self.profiles[node]["osat"].get("tau", 0))
                        # Output matures after dispatch; it must exist at the
                        # end of the preceding week. Keep past deadlines past.
                        production_week = need_date - transit - 1 - tau
                        actionable_week = max(state.week, production_week)
                        capacity_left = max(0.0, cap - osat_weekly_capacity[node, actionable_week])
                        target = min(capacity_left, remaining)
                        if target <= 0:
                            continue
                        osat_weekly_capacity[node, actionable_week] += target
                        remaining -= target
                        pair = node, self.commodities[raw]
                        inputs = self.safety_buffer_policy.apply(target)
                        raw_events[pair].append((production_week, inputs))
                        requirements[pair].append((
                            production_week, inputs, 2.0, "production", self._shortage_cost(pair, "production"),
                            (
                                f"BOM: {raw} input maps to {package}; due before OSAT production",
                                f"OSAT lead {tau}, sink transit {transit}, and one pre-dispatch-stock week",
                                "past receipt deadlines remain overdue; shared capacity uses earliest actionable week",
                            ),
                        ))
                        self._trace("production_projection", pair=pair, downstream_pair=(sink, product),
                                    downstream_due_week=need_date, due_week=production_week,
                                    actionable_week=actionable_week, quantity=inputs, output_quantity=target,
                                    production_lead_weeks=tau, outbound_transit_weeks=transit,
                                    transfer_lead_weeks=1)
                    if remaining > 0:
                        issues.append(f"osat:{product}:unplanned_input:{sink}:{need_date}:{remaining:g}")
                        self._trace("need_not_created", pair=(sink, product), reason="resource_capacity",
                                    stage="osat_projection", quantity=remaining, due_week=need_date)

            fab_weekly_capacity = defaultdict(float)
            raw_to_ship = defaultdict(list)
            for (osat_node, raw_id), events in raw_events.items():
                net_raw = self._net_events(state, (osat_node, raw_id), events, issues)
                if net_raw is None:
                    continue
                raw_name = self.config["static"]["commodities"]["id"][raw_id]
                raw_to_ship[raw_name].extend((osat_node, due, qty) for due, qty in net_raw)

            fab_rows = {node: row for row, node in enumerate(self.config["layout"]["fabs"])}
            for output, demand_events in raw_to_ship.items():
                output_id = self.commodities[output]
                fab_nodes = [
                    fab_node
                    for fab_node in self.config["layout"]["fabs"]
                    if self.profiles[fab_node]["fab"]["product"] == output
                ]
                calendars, capacities, options_by_osat = {}, {}, {}
                residual_events = []
                # Existing output is netted first, using one ledger per Fab.
                # Stock at an unrelated/unreachable Fab cannot cover this OSAT.
                for osat_node, due, quantity in sorted(demand_events, key=lambda event: (event[1], event[0])):
                    if osat_node not in options_by_osat:
                        possible = []
                        for fab_node in fab_nodes:
                            route_eta = self._minimum_route_transit_weeks(
                                network, reader, fab_node, osat_node, output_id
                            )
                            if route_eta is not None:
                                possible.append((fab_node, route_eta))
                        options_by_osat[osat_node] = sorted(possible, key=lambda option: (option[1], option[0]))
                    options = options_by_osat[osat_node]
                    if not options:
                        issues.append(f"fab:{output}:unknown_route_eta_to:{osat_node}")
                        continue
                    remaining = quantity
                    for fab_node, route_eta in options:
                        if fab_node not in calendars:
                            calendars[fab_node] = self._supply_calendar(
                                state, (fab_node, output_id), issues, dispatch=True
                            )
                        if calendars[fab_node] is not None:
                            ship_date = max(state.week, due - route_eta)
                            remaining = self._consume_supply(
                                calendars[fab_node], ship_date, remaining,
                                pair=(fab_node, output_id), stage="fab_output",
                            )
                        if remaining <= 0:
                            break
                    if remaining > 0:
                        residual_events.append((osat_node, due, remaining))

                # Each residual can use every reachable Fab's remaining
                # capacity, with a source-specific transit and production date.
                for osat_node, due, quantity in residual_events:
                    remaining = quantity
                    for fab_node, route_eta in options_by_osat[osat_node]:
                        if calendars[fab_node] is None:
                            continue
                        fab = self.profiles[fab_node]["fab"]
                        if fab_node not in capacities:
                            capacities[fab_node] = reader.number("graph_now.fab.cap_eff", fab_rows[fab_node])
                            if capacities[fab_node] is None:
                                issues.append(f"fab:{fab_node}:unknown_capacity")
                        capacity = capacities[fab_node]
                        if capacity is None:
                            continue
                        tau = int(fab.get("tau", 0))
                        ship_date = due - route_eta
                        production_week = ship_date - tau - 1
                        actionable_week = max(state.week, production_week)
                        capacity_left = max(0.0, capacity - fab_weekly_capacity[fab_node, actionable_week])
                        target = min(capacity_left, remaining)
                        if target <= 0:
                            continue
                        fab_weekly_capacity[fab_node, actionable_week] += target
                        remaining -= target
                        input_name = fab["input"]
                        pair = fab_node, self.commodities[input_name]
                        nominal_inputs = _nominal_fab_inputs(target)
                        inputs = self.safety_buffer_policy.apply(nominal_inputs)
                        requirements[pair].append(
                            (
                                production_week,
                                inputs,
                                2.0,
                                "production",
                                self._shortage_cost(pair, "production"),
                                (
                                    f"BOM: {input_name} per {output} uses the nominal one-to-one input ratio",
                                    f"Fab lead time {tau} weeks and observed route ETA {route_eta} "
                                    f"weeks plus one pre-dispatch-stock week; starts by {production_week}",
                                ),
                            )
                        )
                        self._trace(
                            "production_projection", pair=pair, downstream_pair=(osat_node, output_id),
                            downstream_due_week=due, due_week=production_week, actionable_week=actionable_week,
                            quantity=inputs, output_quantity=target, production_lead_weeks=tau,
                            outbound_transit_weeks=route_eta, transfer_lead_weeks=1,
                        )
                        restoration = (
                            reader.number("graph_now.fab.R", fab_rows[fab_node])
                            if float(fab.get("e", 0.0)) > 0
                            else 1.0
                        )
                        if restoration is None:
                            issues.append(f"fab:{fab_node}:unknown_energy_restoration")
                        energy = float(fab.get("e", 0.0)) * target / restoration if restoration else 0.0
                        grid_name = fab.get("grid")
                        if grid_name:
                            grid_energy[fab_node, actionable_week] += energy
                        if remaining <= 0:
                            break
                    if remaining > 0:
                        issues.append(f"fab:{output}:unplanned_input:{osat_node}:{due}:{remaining:g}")
        fuel_count = self._fuel_horizon(state, reader, network, count)
        automatic_energy = self._automatic_fab_energy(state, reader, issues, fuel_count)
        for row, node in enumerate(self.config["layout"]["grids"]):
            generation = reader.number("graph_now.grid.G_bar", row)
            if generation is None:
                issues.append(f"grid:{node}:unknown_generation")
                continue
            grid_profile = self.profiles[node]["grid"]
            base_load = reader.number("graph_now.grid.y_bar", row)
            if base_load is None:
                issues.append(f"grid:{node}:unknown_base_load")
                continue
            for name, share in grid_profile["shares"].items():
                if name not in self.commodities:
                    continue
                pair = node, self.commodities[name]
                for h in range(fuel_count):
                    due_week = state.week + h
                    planned_generation = min(
                        generation,
                        base_load
                        + sum(
                            max(
                                grid_energy.get((fab_node, due_week), 0.0),
                                automatic_energy.get((fab_node, due_week), 0.0),
                            )
                            for fab_node in self.config["layout"]["fabs"]
                            if self.profiles[fab_node]["fab"].get("grid") == self.config["static"]["nodes"]["id"][node]
                        ),
                    )
                    requirements[pair].append(
                        (
                            due_week,
                            float(share) * planned_generation,
                            3.0,
                            "grid_fuel",
                            self._shortage_cost(pair, "grid_fuel", reader),
                            (
                                "fuel requirement covers observed base load plus grounded automatic/planned Fab energy",
                                "generation is capped by observed deliverable G_bar, "
                                "not treated as a production target",
                                "Fab energy is assigned to its scheduled production week",
                                "Fab energy uses e * output / observed R; automatic starts consume known input once",
                                "future grid load and generation persist from current observations; no future marks assumed",
                            ),
                        )
                    )
        return requirements

    def _automatic_fab_energy(self, state, reader, issues, count):
        """Forecast automatic Fab draw from known input; do not reuse stock each week."""
        result = defaultdict(float)
        for row, node in enumerate(self.config["layout"]["fabs"]):
            fab = self.profiles[node]["fab"]
            coefficient = float(fab.get("e", 0.0))
            if not coefficient or not fab.get("grid"):
                continue
            restoration = reader.number("graph_now.fab.R", row)
            capacity = reader.number("graph_now.fab.cap_eff", row)
            pair = node, self.commodities[fab["input"]]
            stock = state.available_stock.get(pair)
            if restoration is None or capacity is None or stock is None or stock.value is None:
                issues.append(f"fab:{node}:unknown_automatic_energy")
                continue
            if restoration == 0:
                continue
            arrivals = defaultdict(float)
            for arrival in state.arrivals:
                if (arrival.destination_node, arrival.commodity_id) != pair:
                    continue
                if arrival.arrival_week is None or arrival.quantity.value is None:
                    continue
                if not self._arrival_is_eligible(state, arrival):
                    continue
                if state.week <= arrival.arrival_week < state.week + count:
                    arrivals[arrival.arrival_week] += arrival.quantity.value
            balance = stock.value
            for week in range(state.week, state.week + count):
                balance += arrivals[week]
                started = min(balance, capacity)
                balance -= started
                result[node, week] += coefficient * started / restoration
        return result

    def _fuel_replenishment(self, state, needs, network, observation, issues):
        """Project grid requests through one dated ledger per reachable feeder.

        Alternative feeders share demand; ordinary arrivals become dispatchable
        the following week. Neither unknown nor late cargo provides timely cover.
        """
        grids = set(self.config["layout"]["grids"])
        reader = ObservationReader(observation)
        fuel_names = {name for node in grids for name in self.profiles[node]["grid"]["shares"]}
        fuel_ids = {self.commodities[name] for name in fuel_names if name in self.commodities}
        positions = {}
        feeders = {}
        for need in needs:
            if need.destination_node not in grids or need.commodity_id not in fuel_ids:
                continue
            key = need.destination_node, need.commodity_id
            if key in feeders:
                continue
            choices = {}
            for slot in network.slots_to.get(key, ()):
                route = network.routes[slot]
                pair = route.source_node, need.commodity_id
                # External supply nodes require no predecessor action. Only
                # stock nodes with an actual incoming slot need replenishment.
                imports = network.slots_to.get(pair, ())
                if not any(network.routes[s].source_node != need.destination_node for s in imports):
                    continue
                stock = state.available_stock.get(pair)
                if stock is None or stock.value is None:
                    issues.append(f"fuel_replenishment:{pair}:unknown_inventory")
                    continue
                transit = self._minimum_route_transit_weeks(
                    network, reader, pair[0], need.destination_node, need.commodity_id
                )
                # Hidden live transit is not replaced with nominal tau0 for
                # order timing. Keep an uncertain feeder as a last-resort
                # candidate; its replenishment order is due immediately.
                rank = (
                    float("inf") if transit is None else transit,
                    route.nominal_freight_per_unit,
                    route.slot_id,
                )
                if pair not in choices or rank < choices[pair][0]:
                    choices[pair] = rank, route
                positions.setdefault(pair, float(stock.value))
            feeders[key] = sorted(choices.items(), key=lambda item: item[1][0])

        calendars = {pair: self._supply_calendar(state, pair, issues, dispatch=True) for pair in positions}
        requests = {}
        for need in sorted(needs, key=lambda item: (item.due_week, -item.priority, item.need_id)):
            choices = feeders.get((need.destination_node, need.commodity_id), ())
            if not choices:
                continue
            remaining = need.quantity
            # Terminal inventory is credited by its usable dispatch date.
            # Late/unknown inbound cargo cannot cancel an earlier grid need.
            for pair, (_rank, _route) in choices:
                transit = self._minimum_route_transit_weeks(
                    network, reader, pair[0], need.destination_node, need.commodity_id
                )
                ship_by = state.week if transit is None else max(state.week, need.due_week - transit)
                remaining = self._consume_supply(calendars[pair], ship_by, remaining,
                                                 pair=pair, stage="fuel_feeder")
                if remaining <= 0:
                    break
            if remaining <= 0:
                continue
            pair, (_rank, route) = choices[0]
            transit = self._minimum_route_transit_weeks(
                network, reader, pair[0], need.destination_node, need.commodity_id
            )
            due = state.week if transit is None else need.due_week - transit - 1
            if transit is None:
                issues.append(f"fuel_replenishment:{pair}:unknown_downstream_transit")
            key = pair, due, need.priority
            if key not in requests:
                requests[key] = [0.0, [], []]
            requests[key][0] += remaining
            requests[key][1].append(need.need_id)
            if need.shortage_cost_per_unit_usd is not None:
                requests[key][2].append(need.shortage_cost_per_unit_usd)
            self._trace("fuel_projection", pair=pair, parent_need_id=need.need_id,
                        downstream_due_week=need.due_week, due_week=due,
                        outbound_transit_weeks=transit, transfer_lead_weeks=1, quantity=remaining)
        return tuple(
            DeliveryNeed(
                f"fuel_replenishment:{pair[0]}:{pair[1]}:{due}:{priority:g}",
                *pair,
                quantity,
                due,
                priority,
                "fuel_replenishment",
                max(costs) if costs else None,
                assumptions=(
                    "net grid fuel demand propagated through existing action slots",
                    "dated feeder stock and eligible inbound imports shared once across consumer requests",
                    "unknown or late inbound never counts as timely coverage",
                    "deadline uses observed downstream transit plus one pre-dispatch-stock week; "
                    "hidden transit orders immediately",
                    "deterministic feeder choice; current feasibility and shared capacities belong to allocator",
                    "parents: " + ",".join(parents),
                ),
            )
            for (pair, due, priority), (quantity, parents, costs) in sorted(requests.items())
        )

    def plan(self, state, observation, network):
        self._trace_rows = []
        reader = ObservationReader(observation)
        issues = []
        ids = set()
        for arrival in state.arrivals:
            if arrival.arrival_id in ids:
                raise ValueError("duplicate arrival_id would double count an arrival")
            ids.add(arrival.arrival_id)
        # Update backlog age before propagating it through the production BOM.
        for pair, backlog in state.backlog.items():
            if backlog.value is not None and backlog.value > 0:
                self._backlog_since.setdefault(pair, state.week)
            elif backlog.value == 0:
                self._backlog_since.pop(pair, None)
        requirements = self._requirements(state, reader, issues, network)
        for pair, backlog in state.backlog.items():
            if backlog.value is None:
                issues.append(f"backlog:{pair}:unknown")
                continue
            if backlog.value <= 0:
                self._backlog_since.pop(pair, None)
                continue
            due = self._backlog_since.setdefault(pair, state.week)
            requirements[pair].append(
                (
                    due,
                    backlog.value,
                    4.0,
                    "backlog",
                    self.penalties.get(pair),
                    ("due week is earliest observation of continuous backlog, not original order date",),
                )
            )
        result = []
        need_id_counts = defaultdict(int)
        for pair, requests in sorted(requirements.items()):
            stock = state.available_stock.get(pair)
            if stock is None:
                raise ValueError(f"need destination {pair} has no stock slot")
            assumptions = (
                () if stock.value is not None else ("hidden stock: zero lower-bound coverage, not observed zero",)
            )
            if stock.value is None:
                issues.append(f"stock:{pair}:unknown_coverage")
                # Do not silently turn unknown stock into a numeric zero and
                # manufacture a replenishment order from missing information.
                self._trace("need_not_created", pair=pair, reason="unknown_stock", stage="final_netting")
                continue
            calendar = self._supply_calendar(state, pair, issues)
            last_week = state.week
            for due, qty, priority, reason, cost, basis in sorted(
                requests, key=lambda item: (item[0], -item[2], item[3])
            ):
                coverage_week = max(state.week, due)
                last_week = coverage_week
                missing = self._consume_supply(calendar, coverage_week, qty, pair=pair, stage="final_netting")
                need_id = None
                if missing > 0:
                    base_id = f"{reason}:{pair[0]}:{pair[1]}:{due}"
                    occurrence = need_id_counts[base_id]
                    need_id_counts[base_id] += 1
                    need_id = base_id if occurrence == 0 else f"{base_id}:{occurrence + 1}"
                    result.append(
                        DeliveryNeed(
                            need_id,
                            *pair,
                            missing,
                            due,
                            priority,
                            reason,
                            cost,
                            assumptions=assumptions + basis,
                        )
                    )
                self._trace(
                    "requirement", pair=pair, week=state.week, initial_stock=stock.value,
                    backlog=getattr(state.backlog.get(pair), "value", None), due_week=due,
                    requirement=qty, covered=qty - missing, missing=missing, reason=reason,
                    need_id=need_id, no_need_reason="covered_by_dated_supply" if need_id is None else None,
                )
            if self.safety_stock and "grid" in self.profiles[pair[0]]:
                name = self.config["static"]["commodities"]["id"][pair[1]]
                target = float(self.profiles[pair[0]]["grid"].get("ibar", {}).get(name, 0.0))
                due = max(state.week, max(request[0] for request in requests))
                balance = sum(lot[1] for lot in calendar if lot[0] <= last_week)
                missing = max(0.0, target - balance)
                if missing > 0:
                    result.append(
                        DeliveryNeed(
                            f"safety_stock:{pair[0]}:{pair[1]}:{due}",
                            *pair,
                            missing,
                            due,
                            1.0,
                            "safety_stock",
                            None,
                            assumptions=assumptions + ("static grid ibar reserve target",),
                        )
                    )
        if self.fuel_replenishment_enabled:
            result.extend(self._fuel_replenishment(state, result, network, observation, issues))
        annotated = []
        for need in result:
            route_rows = []
            for slot in network.slots_to.get((need.destination_node, need.commodity_id), ()):
                route = network.routes[slot]
                transit = self._minimum_route_transit_weeks(
                    network, reader, route.source_node, need.destination_node, need.commodity_id
                )
                # The pair minimum must not label a slower parallel slot.
                times = [reader.number("graph_now.tau", edge) for edge in route.edges]
                transit = int(sum(times)) if all(t is not None and int(t) == t for t in times) else None
                route_rows.append({
                    "slot_id": slot, "source_node": route.source_node,
                    "physical_min_transit_weeks": transit,
                    "no_wait_arrival_week": None if transit is None else state.week + transit,
                    "latest_feasible_dispatch_week": None if transit is None else need.due_week - transit,
                })
            times = [row["physical_min_transit_weeks"] for row in route_rows
                     if row["physical_min_transit_weeks"] is not None]
            physical = min(times) if times else None
            latest = None if physical is None else need.due_week - physical
            status = ("no_route" if not route_rows else "unknown" if latest is None
                      else "created_too_late" if latest < state.week else "physical_bound_only")
            if status == "created_too_late":
                issues.append(f"need:{need.need_id}:created_too_late:{latest}")
            annotated.append(replace(need, physical_min_transit_weeks=physical,
                                     latest_feasible_dispatch_week=latest, timing_status=status))
            self._trace("need_timing", need_id=need.need_id, pair=(need.destination_node, need.commodity_id),
                        quantity=need.quantity, due_week=need.due_week, timing_status=status,
                        latest_feasible_dispatch_week=latest, route_candidates=route_rows)
        for node in self.config["layout"]["fabs"]:
            pair = node, self.commodities[self.profiles[node]["fab"]["input"]]
            if pair not in requirements:
                self._trace("need_not_created", pair=pair, reason="no_net_downstream_production_target",
                            stage="fab_input", week=state.week,
                            initial_stock=getattr(state.available_stock.get(pair), "value", None))
        self.last_issues = tuple(dict.fromkeys(issues))
        self.last_trace = tuple(self._trace_rows)
        return tuple(sorted(annotated, key=need_order_key))

    @staticmethod
    def export_examples(needs):
        """JSON-ready handoff rows; keep rank and assumptions beside quantities."""
        ordered = sorted(needs, key=need_order_key)
        return tuple(
            {
                "priority_rank": rank,
                "need_id": need.need_id,
                "destination_node": need.destination_node,
                "commodity_id": need.commodity_id,
                "quantity": need.quantity,
                "due_week": need.due_week,
                "priority": need.priority,
                "reason": need.reason,
                "quantity_source": need.quantity_source,
                "shortage_cost_per_unit_usd": need.shortage_cost_per_unit_usd,
                "confidence": need.confidence,
                "assumptions": list(need.assumptions),
                "physical_min_transit_weeks": need.physical_min_transit_weeks,
                "latest_feasible_dispatch_week": need.latest_feasible_dispatch_week,
                "timing_status": need.timing_status,
            }
            for rank, need in enumerate(ordered, 1)
        )
