"""A dependency-free catalogue of this policy's actual parameter consumers.

Defaults come from the selected submission's source, not a stale experiment
preset. Constructor checks still define semantic constraints; the catalogue
provides types, known bounds and component ownership for controlled searches.
"""

import ast
import math
from pathlib import Path


BOUNDS = {
    "fraction": (0, 1),
    "air_fraction": (0, 1),
    "raw_utility_strength": (0, 2),
    "fuel_release_warning_strength": (0, 4),
    "fuel_mpc_horizon": (0, 24),
    "fuel_lookahead_horizon": (0, 6),
    "batch_end_buffer": (0, 26),
    "production_horizon": (1, None),
}


def catalogue(folder):
    tree = ast.parse((Path(folder) / "agent.py").read_text(encoding="utf-8-sig"))
    properties = {}

    def literal(node):
        if node is None:
            return None
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError):
            if isinstance(node, ast.Call) and len(node.args) >= 2:
                return literal(node.args[1])
            return None

    for statement in ast.walk(tree):
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        owner = next((target.attr for target in targets if isinstance(target, ast.Attribute)), "Agent")
        for node in ast.walk(statement.value):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "PARAMS"
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                continue
            key = node.args[0].value
            default = literal(node.args[1]) if len(node.args) > 1 else None
            kind = (
                "boolean"
                if isinstance(default, bool)
                else "integer"
                if isinstance(default, int)
                else (
                    "number"
                    if isinstance(default, float)
                    else "object"
                    if isinstance(default, dict)
                    else "nullable_object"
                )
            )
            entry = {"type": kind, "default": default, "component": owner, "minimum": None, "maximum": None}
            if kind in {"integer", "number"}:
                entry["minimum"], entry["maximum"] = BOUNDS.get(key, (0, None))
            properties.setdefault(key, entry)
    properties.update(
        fraction={"type": "number_or_array", "default": 1.0, "component": "Agent", "minimum": 0, "maximum": 1},
        closure_power={"type": "number", "default": 1.0, "component": "Agent", "minimum": 0, "maximum": None},
    )
    planner = ast.parse((Path(folder) / "needs.py").read_text())
    for node in ast.walk(planner):
        if not isinstance(node, ast.FunctionDef) or node.name != "__init__":
            continue
        for arg, value in zip(node.args.kwonlyargs, node.args.kw_defaults):
            if arg.arg in {"safety_buffer_policy", "trace_enabled"}:
                continue
            default = literal(value)
            properties[f"planner_options.{arg.arg}"] = {
                "type": "boolean" if isinstance(default, bool) else "integer",
                "default": default,
                "component": "NeedPlanner",
                "minimum": None if isinstance(default, bool) else 1,
                "maximum": None,
            }
    return {"format": "shockbench.parameter-catalogue.v1", "properties": dict(sorted(properties.items()))}


def validate(params, schema):
    properties = schema["properties"]
    for key, value in params.items():
        if key not in properties:
            raise ValueError(f"unknown parameter: {key}")
        if key == "planner_options":
            if not isinstance(value, dict):
                raise ValueError("planner_options must be an object")
            nested = {
                k.removeprefix("planner_options."): v for k, v in properties.items() if k.startswith("planner_options.")
            }
            validate(value, {"properties": nested})
            continue
        spec = properties[key]
        kind = spec["type"]
        if kind == "boolean":
            good = isinstance(value, bool)
        elif kind in {"integer", "number", "number_or_array"}:
            values = value if kind == "number_or_array" and isinstance(value, list) else [value]
            good = bool(values)
            for item in values:
                good &= not isinstance(item, bool) and isinstance(item, (int, float)) and math.isfinite(item)
                if not good:
                    break
                good &= kind != "integer" or isinstance(item, int)
                good &= spec["minimum"] is None or item >= spec["minimum"]
                good &= spec["maximum"] is None or item <= spec["maximum"]
        else:
            good = isinstance(value, dict) or (value is None and kind == "nullable_object")
            if isinstance(value, dict):
                good &= all(
                    not isinstance(v, bool) and isinstance(v, (float, int)) and math.isfinite(v) and v >= 0
                    for v in value.values()
                )
        if not good:
            raise ValueError(f"invalid {key}: expected {kind} within declared bounds")
    return params
