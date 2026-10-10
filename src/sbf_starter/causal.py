"""Offline, observation-grounded loss traces; simulator truth never enters Agent.act.

Records are evidence, not a claim that a correlated shipment caused a shortage.
Native quantities always retain their commodity, node and unit. Ambiguous
attribution is explicitly left unresolved instead of charging the same loss to
several competing explanations.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np


LOSS_CATEGORIES = (
    "need_not_created",
    "created_too_late",
    "no_source_stock",
    "no_route",
    "resource_capacity",
    "priority_displaced",
    "late_arrival",
    "action_clipped",
    "arrival_not_consumed",
    "overstock_or_waste",
)


def observed(obs, key, row):
    values, mask = obs.get(key), obs.get(key + ".observed")
    if values is None or mask is None or not mask[row]:
        return None
    value = float(values[row])
    return value if np.isfinite(value) else None


def action_digest(action):
    digest = hashlib.sha256()
    for key, value in sorted(action.items()):
        array = np.asarray(value)
        digest.update(key.encode() + str(array.shape).encode() + array.dtype.str.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def warning_reliability(rows, horizon=3, threshold=0.5):
    """Backtest raw signal bins against *subsequent* observed restrictions.

    Scores are not probabilities. Missing outcomes and right-censored windows
    are excluded, never counted as negative events. Evaluate on a separate root
    from model fitting. This measures association, not the benefit of rerouting.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    bins = defaultdict(lambda: {"count": 0, "events": 0, "false_positive": 0, "false_negative": 0})
    by_week = {row["week"]: row for row in rows}
    for row in rows:
        for node, sample in row.get("warnings", {}).items():
            signal = sample["score"]
            if signal is None or sample["open"] != 1.0:
                continue  # onset risk conditional on currently unrestricted
            future = [
                by_week.get(row["week"] + offset, {}).get("warnings", {}).get(node) for offset in range(1, horizon + 1)
            ]
            if any(item is None or item["open"] is None for item in future):
                continue
            event = any(item["open"] < 1.0 for item in future)
            key = f"[{np.floor(signal * 2) / 2:g},{np.floor(signal * 2) / 2 + 0.5:g})"
            bucket = bins[key]
            bucket["count"] += 1
            bucket["events"] += int(event)
            bucket["false_positive"] += int(signal >= threshold and not event)
            bucket["false_negative"] += int(signal < threshold and event)
    return {
        "horizon": horizon,
        "raw_score_threshold": threshold,
        "scores_are_probabilities": False,
        "bins": {
            key: dict(value, event_frequency=value["events"] / value["count"]) for key, value in sorted(bins.items())
        },
    }


def replay(
    agent_path, task="tiny", entropy=12345, episode=0, weeks=0, replay_seed=0, planner_path=None, trace_file=None
):
    """Trace a frozen submission with a separate, non-executing diagnostic planner.

    ZIP SHA salts policy_seed exactly as the scoring package does. The shadow
    planner is diagnostic only: absence of its Need is not proof of a policy
    defect. Complete simulator StepRecords provide realised arrivals/consumption
    evidence and cost accounting after act has returned.
    """
    import tempfile

    import gymnasium as gym
    from shockbench_flow_agent.scoring import _policy_seed
    from shockbench_flow_agent.submission import build_submission, extract_submission
    from shockbench_flow_gym import agent_config_from_reset

    from sbf_starter import env_id
    from sbf_starter.agents import load, resolve

    if isinstance(weeks, bool) or weeks < 0:
        raise ValueError("weeks must be nonnegative")
    with tempfile.TemporaryDirectory(prefix="sbf-causal-") as temporary:
        path = resolve(agent_path)
        archive = path if path.suffix == ".zip" else Path(build_submission(path, Path(temporary) / "agent.zip"))
        extract_submission(archive, Path(temporary) / "submission")
        sha = hashlib.sha256(archive.read_bytes()).hexdigest()
        seed = _policy_seed(entropy, episode, sha)
        env = gym.make(env_id(task), entropy=entropy, policy_seed=seed)
        stream = None
        try:
            obs, info = env.reset(seed=replay_seed, options={"episode": episode})
            config = agent_config_from_reset(env, obs, info)
            agent_class = load(Path(temporary) / "submission")
            started = time.process_time()
            agent = agent_class(config)
            init_cpu = time.process_time() - started
            # Capture the policy before loading any other submission modules.
            act = agent.act
            diagnostic_class = load(planner_path or "team_agent_hybrid_v16")
            globals_ = diagnostic_class.__init__.__globals__
            network = globals_["StaticNetwork"](config)
            pipeline = globals_["build_pipeline"](
                config,
                network,
                enabled=True,
                planner_options={
                    "production_horizon": 8,
                    "fuel_replenishment_enabled": True,
                    "shortage_cost_model": True,
                    "safety_stock": False,
                },
            )
            planner = pipeline.need_planner
            planner.trace_enabled = True
            stock_rows = {tuple(pair): row for row, pair in enumerate(config["layout"]["stock_slots"])}
            warning_rows = {
                int(unit): row
                for row, (kind, unit) in enumerate(config["layout"]["warning_units"])
                if kind == "chokepoint"
            }
            rows, costs, cents, max_cpu = [], defaultdict(float), 0, 0.0
            quantity_totals = defaultdict(lambda: defaultdict(float))
            if trace_file:
                stream = Path(trace_file).open("x", encoding="utf-8")
            done = False
            while not done and (weeks == 0 or len(rows) < weeks):
                week = int(obs["week"][0])
                state = pipeline.state_builder.build(obs, network)
                needs = planner.plan(state, obs, network)
                started = time.process_time()
                action = act(obs)
                cpu = time.process_time() - started + (init_cpu if not rows else 0.0)
                max_cpu = max(cpu, max_cpu)
                pair_needs = defaultdict(list)
                for need in needs:
                    pair_needs[need.destination_node, need.commodity_id].append(need)
                row = {
                    "week": week,
                    "policy_cpu_s": cpu,
                    "action_sha256": action_digest(action),
                    "needs": [asdict(need) for need in needs],
                    "planner_trace": list(planner.last_trace),
                    "planner_issues": list(planner.last_issues),
                    "sinks": [],
                    "warnings": {},
                    "hybrid_trace": list(getattr(getattr(agent, "hybrid", None), "last_trace", ())),
                }
                for pos, node in enumerate(config["layout"]["chokepoints"]):
                    wr = warning_rows.get(node)
                    row["warnings"][network.node_names[node]] = {
                        "score": None if wr is None else observed(obs, "warning.score", wr),
                        "open": observed(obs, "graph_now.open", pos),
                    }
                for demand_row, (node, commodity) in enumerate(config["layout"]["demands"]):
                    pair = node, commodity
                    routes = [network.routes[slot] for slot in network.slots_to.get(pair, ())]
                    candidates = []
                    for route in routes:
                        transit = [observed(obs, "graph_now.tau", edge) for edge in route.edges]
                        lower = None if any(value is None for value in transit) else sum(transit)
                        source = stock_rows.get((route.source_node, commodity))
                        candidates.append(
                            {
                                "slot": route.slot_id,
                                "source": route.source_node,
                                "source_stock": None if source is None else observed(obs, "stock.qty", source),
                                "requested": float(action["flows"][route.slot_id]),
                                "physical_arrival_lower_bound": None if lower is None else week + lower,
                                "permission": float(obs["action_mask"][route.slot_id]),
                            }
                        )
                    row["sinks"].append(
                        {
                            "node": network.node_names[node],
                            "commodity": network.commodity_names[commodity],
                            "unit": config["static"]["units"][network.commodity_names[commodity]],
                            "stock": observed(obs, "stock.qty", stock_rows[pair]),
                            "backlog_before": observed(obs, "backlog.qty", demand_row),
                            "needs": [need.need_id for need in pair_needs[pair]],
                            "candidates": candidates,
                        }
                    )
                next_obs, _, terminated, truncated, info = env.step(action)
                record = env.unwrapped.core.trajectory.records[-1]
                row["realised"] = record.as_json()
                row["cost_usd"] = record.cost_cents / 100
                for name, value in record.costs.as_dict().items():
                    costs[name] += value
                for demand_row, sink in enumerate(row["sinks"]):
                    loss = float(record.lost[demand_row] + record.backlog[demand_row])
                    sink.update(
                        demand=float(record.demand[demand_row]),
                        served=float(record.served[demand_row]),
                        lost=float(record.lost[demand_row]),
                        backlog_after=float(record.backlog[demand_row]),
                    )
                    evidence = []
                    if loss > 0:
                        candidates = sink["candidates"]
                        if not sink["needs"]:
                            evidence.append("need_not_created")
                        if not candidates:
                            evidence.append("no_route")
                        if candidates and all(c["source_stock"] == 0 for c in candidates):
                            evidence.append("no_source_stock")
                        if any(
                            record.requested.get(c["slot"], 0) > record.executed.get(c["slot"], 0) + 1e-9
                            for c in candidates
                        ):
                            evidence.append("action_clipped")
                    sink["evidence_categories"] = evidence
                    sink["causal_attribution"] = "unresolved" if loss else "no_shortage"
                    key = sink["node"], sink["commodity"], sink["unit"]
                    for name in ("demand", "served", "lost", "backlog_after"):
                        quantity_totals[key][name if name != "backlog_after" else "backlog_unit_weeks"] += sink[name]
                cents -= int(info["reward_cents"])
                rows.append(row)
                if stream:
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                obs, done = next_obs, terminated or truncated
            return {
                "kind": "causal_replay_not_rss",
                "task": task,
                "entropy": entropy,
                "episode": episode,
                "submission_sha256": sha,
                "policy_seed": seed,
                "replay_seed": replay_seed,
                "complete_episode": bool(done),
                "weeks": len(rows),
                "cost_usd": cents / 100,
                "salvage_usd": env.unwrapped.core.trajectory.salvage,
                "cost_components_usd": dict(costs),
                "max_policy_cpu_s": max_cpu,
                "action_hashes": [row["action_sha256"] for row in rows],
                "quantity_totals": [
                    dict(node=key[0], commodity=key[1], unit=key[2], **values)
                    for key, values in sorted(quantity_totals.items())
                ],
                "warning_reliability": warning_reliability(rows),
                "attribution_limit": "Evidence categories overlap; unproven causal losses remain unresolved.",
            }
        finally:
            if stream:
                stream.close()
            env.close()
