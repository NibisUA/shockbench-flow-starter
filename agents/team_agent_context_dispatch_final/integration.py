"""Connect teammate modules and validate their boundaries without changing policy."""

import math
from dataclasses import dataclass
from operator import index

import numpy as np
from contracts import AllocationResult, Allocator, NeedPlanner, Quantity, StateBuilder, need_order_key


def _integer(value, label, low, high):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label}: expected an integer, not bool")
    try:
        number = index(value)
    except TypeError as exc:
        raise ValueError(f"{label}: expected an integer") from exc
    if not low <= number <= high:
        raise ValueError(f"{label}: {number} outside [{low}, {high}]")
    return number


def _number(value, label, *, nonnegative=True):
    if isinstance(value, (bool, np.bool_, str, bytes)) or not np.isscalar(value) or np.iscomplexobj(value):
        raise ValueError(f"{label}: expected a finite real number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label}: expected a finite real number") from exc
    if not math.isfinite(number) or (nonnegative and number < 0):
        raise ValueError(f"{label}: expected a finite {'nonnegative ' if nonnegative else ''}number")
    return number


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: expected nonempty text")


def _confidence(value, label):
    if value is not None and _number(value, label) > 1:
        raise ValueError(f"{label}: confidence must be in [0, 1] or None")


def _quantity(value, label):
    if not isinstance(value, Quantity):
        raise TypeError(f"{label}: expected contracts.Quantity")
    if value.source not in ("observed", "estimated", "unknown"):
        raise ValueError(f"{label}: invalid source")
    if (value.value is None) != (value.source == "unknown"):
        raise ValueError(f"{label}: unknown requires None; observed/estimated require a value")
    if value.value is not None:
        _number(value.value, label)
    _confidence(value.confidence, label)
    if value.source == "unknown" and value.confidence is not None:
        raise ValueError(f"{label}: unknown quantity cannot have forecast confidence")


class ActionValidator:
    """Strict interface gate: no silent clipping or fabricated capacity/stock.

    Allocator owns physical feasibility and the weekly event order. This gate
    checks shapes, numbers, modes and visible prohibitions; hidden constraints
    are not replaced by zeros. Exceptions expose bugs instead of hiding them.
    """

    def __init__(self, config):
        self.specs = config["spaces"]["action"]
        self.release_modes = frozenset(config["release_modes"].values())

    def _array(self, name, values):
        arr = np.asarray(values)
        if arr.shape != tuple(self.specs[name]["shape"]):
            raise ValueError(f"{name}: shape {arr.shape} does not match config")
        if arr.dtype.kind not in "iuf" or not np.all(np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError(f"{name}: expected finite nonnegative real numbers")
        if name == "release_mode":
            if arr.dtype.kind not in "iu" or not np.all(np.isin(arr, tuple(self.release_modes))):
                raise ValueError("release_mode: expected integer modes from config['release_modes']")
        with np.errstate(over="ignore", invalid="ignore"):
            arr = np.array(arr, dtype=self.specs[name]["dtype"], copy=True)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name}: values overflow the action dtype")
        return arr

    @staticmethod
    def _mask(name, mask_name, array, observation):
        mask = np.asarray(observation[mask_name])
        if mask.shape != array.shape or not np.all(np.isin(mask, (0, 1))):
            raise ValueError(f"{mask_name}: expected a binary mask matching {name}")
        seen = np.asarray(observation[f"{mask_name}.observed"])
        if seen.shape != (1,) or not np.all(np.isin(seen, (0, 1))):
            raise ValueError(f"{mask_name}.observed: expected one visibility flag")
        if seen[0] and np.any(array[mask == 0] != 0):
            raise ValueError(f"{name}: positive quantity on an observed prohibited slot")

    def validate(self, action, observation):
        if not isinstance(action, dict) or "flows" not in action or action.keys() - self.specs.keys():
            raise ValueError("action: expected flows and only supported action fields")
        result = {name: self._array(name, values) for name, values in action.items()}
        self._mask("flows", "action_mask", result["flows"], observation)
        if "override_qty" in result:
            self._mask("override_qty", "override_mask", result["override_qty"], observation)
        return result


@dataclass(slots=True)
class DecisionPipeline:
    state_builder: StateBuilder
    need_planner: NeedPlanner
    allocator: Allocator

    def run(self, observation, network, config):
        state = self.state_builder.build(observation, network)
        week = _integer(state.week, "state.week", 1, config["T"])
        horizon = _integer(state.horizon, "state.horizon", 1, config["T"])
        if horizon != config["T"] or week != _integer(observation["week"][0], "week", 1, config["T"]):
            raise ValueError("state: week/horizon do not match observation/config")
        stock_keys = frozenset(map(tuple, config["layout"]["stock_slots"]))
        backlog_keys = frozenset(map(tuple, config["layout"]["demands"]))
        for name, allowed in (("available_stock", stock_keys), ("backlog", backlog_keys)):
            if set(getattr(state, name)) != allowed:
                raise ValueError(f"state.{name}: include every layout pair; use unknown quantities for hidden data")
            for key, qty in getattr(state, name).items():
                if not isinstance(key, tuple) or len(key) != 2:
                    raise ValueError(f"state.{name}: expected (node, commodity) keys")
                _integer(key[0], f"{name}.node", 0, len(network.node_names) - 1)
                _integer(key[1], f"{name}.commodity", 0, len(network.commodity_names) - 1)
                if key not in allowed:
                    raise ValueError(f"state.{name}: {key} is not a config layout pair")
                _quantity(qty, f"state.{name}[{key}]")
        # Preserve the same snapshot, raw visibility masks and network for both modules.
        needs = tuple(self.need_planner.plan(state, observation, network))
        ids = {}
        for need in needs:
            _text(need.need_id, "need_id")
            if need.need_id in ids:
                raise ValueError(f"duplicate need_id: {need.need_id}")
            ids[need.need_id] = need
            _integer(need.destination_node, "need.destination_node", 0, len(network.node_names) - 1)
            _integer(need.commodity_id, "need.commodity_id", 0, len(network.commodity_names) - 1)
            _integer(need.due_week, "need.due_week", 1, config["T"])
            _number(need.quantity, "need.quantity")
            _number(need.priority, "need.priority", nonnegative=False)
            _text(need.reason, "need.reason")
            _confidence(need.confidence, "need.confidence")
            if need.shortage_cost_per_unit_usd is not None:
                _number(need.shortage_cost_per_unit_usd, "need.shortage_cost_per_unit_usd")
        result = self.allocator.allocate(state, tuple(sorted(needs, key=need_order_key)), observation, network)
        if not isinstance(result, AllocationResult):
            raise TypeError("allocator: expected contracts.AllocationResult")
        seen_unmet = set()
        for unmet in result.unmet_needs:
            if unmet.need_id not in ids or unmet.need_id in seen_unmet:
                raise ValueError("unmet_needs: unknown or duplicate need_id")
            seen_unmet.add(unmet.need_id)
            remaining = _number(unmet.remaining_quantity, "unmet.remaining_quantity")
            if remaining > ids[unmet.need_id].quantity:
                raise ValueError("unmet_needs: remaining quantity exceeds the request")
            _text(unmet.reason, "unmet.reason")
        resource_sizes = {
            "stock": len(stock_keys),
            "edge": len(network.edge_names),
            "chokepoint_pool": len(network.chokepoints),
        }
        seen_resources = set()
        for usage in result.resource_usage:
            if usage.kind not in resource_sizes:
                raise ValueError("resource_usage: unknown resource kind")
            _integer(usage.resource_index, "resource_index", 0, resource_sizes[usage.kind] - 1)
            key = usage.kind, usage.resource_index, usage.pool
            if key in seen_resources:
                raise ValueError("resource_usage: report each shared resource only once")
            seen_resources.add(key)
            if (usage.kind == "chokepoint_pool" and usage.pool not in ("tb", "ct")) or (
                usage.kind != "chokepoint_pool" and usage.pool is not None
            ):
                raise ValueError("resource_usage: pool is required only for chokepoint_pool")
            _text(usage.unit, "resource.unit")
            used = _number(usage.used, "resource.used")
            _quantity(Quantity(usage.limit, usage.limit_source), "resource.limit")
            if usage.limit is not None and used > usage.limit + 1e-9 * max(1.0, usage.limit):
                raise ValueError("resource_usage: planned usage exceeds the reported limit")
            if usage.kind == "stock":
                commodity = config["layout"]["stock_slots"][usage.resource_index][1]
                expected_unit = config["static"]["units"][network.commodity_names[commodity]]
                if usage.unit != expected_unit:
                    raise ValueError("resource_usage: stock units do not match config")
        for reason in result.reasons:
            _text(reason.code, "reason.code")
            _text(reason.message, "reason.message")
            if reason.need_id is not None and reason.need_id not in ids:
                raise ValueError("reason: unknown need_id")
            if reason.slot_id is not None:
                _integer(reason.slot_id, "reason.slot_id", 0, len(network.routes) - 1)
        return result


def validate_planner_options(options):
    if options is None:
        return {}
    if not isinstance(options, dict):
        raise ValueError("planner_options must be an object")
    boolean_fields = {
        "include_estimated_arrivals", "safety_stock", "production_enabled",
        "shortage_cost_model", "fuel_replenishment_enabled",
    }
    if options.keys() - (boolean_fields | {"production_horizon"}):
        raise ValueError("planner_options contains unsupported fields")
    for name in boolean_fields & options.keys():
        if not isinstance(options[name], bool):
            raise ValueError(f"planner_options.{name} must be a boolean")
    if "production_horizon" in options:
        value = options["production_horizon"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("planner_options.production_horizon must be a positive integer")
    return dict(options)


def build_pipeline(
    config, network, *, enabled=False, queue_eta_enabled=False,
    queue_forecast_enabled=False, planner_options=None,
):
    """V4 is explicit opt-in until paired evaluation supports promotion."""
    for name, value in (
        ("enabled", enabled), ("queue_eta_enabled", queue_eta_enabled),
        ("queue_forecast_enabled", queue_forecast_enabled),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean")
    options = validate_planner_options(planner_options)
    if not enabled:
        return None
    from allocation import Allocator
    from needs import NeedPlanner
    from state import StateBuilder

    forecaster = None
    if queue_forecast_enabled:
        from queue_forecast import QueueForecaster

        forecaster = QueueForecaster(config)
    return DecisionPipeline(
        StateBuilder(config, queue_forecaster=forecaster),
        NeedPlanner(config, **options),
        Allocator(config, network, queue_eta_enabled=queue_eta_enabled),
    )
