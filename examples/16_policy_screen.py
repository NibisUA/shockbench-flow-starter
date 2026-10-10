"""Replay frozen candidates on training scenarios; raw costs are screening only.

This saves each episode immediately. It does not tune on the validation root or
claim RSS without benchmark references. Use example 14 for final paired gates.
"""

import json
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import fire
import gymnasium as gym
import numpy as np
from shockbench_flow_agent.scoring import _policy_seed
from shockbench_flow_agent.submission import extract_submission
from shockbench_flow_gym import agent_config_from_reset

from sbf_starter import env_id
from sbf_starter.agents import load
from sbf_starter.causal import action_digest
from sbf_starter.experiment import freeze, write_json


def play(archive, sha, task, entropy, episode):
    seed = _policy_seed(entropy, episode, sha)
    with tempfile.TemporaryDirectory(prefix="sbf-screen-") as temporary:
        submission = extract_submission(archive, Path(temporary) / "agent")
        agent_class = load(submission.root)
        env = gym.make(env_id(task), entropy=entropy, policy_seed=seed)
        try:
            obs, info = env.reset(seed=0, options={"episode": episode})
            config = agent_config_from_reset(env, obs, info)
            start = time.process_time()
            agent = agent_class(config)
            init_cpu = time.process_time() - start
            costs, quantities, hashes, cpus = defaultdict(float), defaultdict(lambda: defaultdict(float)), [], []
            cents, switches, changes, done = 0, 0, 0, False
            while not done:
                start = time.process_time()
                action = agent.act(obs)
                cpus.append(time.process_time() - start + (init_cpu if not hashes else 0))
                hashes.append(action_digest(action))
                hybrid = getattr(agent, "hybrid", None)
                trace = getattr(hybrid, "last_trace", ())
                changes += sum(item.get("stage") == "changed_action" for item in trace)
                gate = getattr(agent, "risk_gate", None)
                switches += int(bool(getattr(gate, "last_reasons", ())))
                obs, _, terminated, truncated, info = env.step(action)
                record = env.unwrapped.core.trajectory.records[-1]
                for key, value in record.costs.as_dict().items():
                    costs[key] += value
                for index, (_, commodity) in enumerate(config["layout"]["demands"]):
                    name = config["static"]["commodities"]["id"][commodity]
                    quantities[name]["lost"] += float(record.lost[index])
                    quantities[name]["backlog_unit_weeks"] += float(record.backlog[index])
                    quantities[name]["served"] += float(record.served[index])
                for slot, value in record.requested.items():
                    commodity = config["static"]["action_slots"]["k"][slot]
                    name = config["static"]["commodities"]["id"][commodity]
                    quantities[name]["clipped"] += max(0, value - record.executed.get(slot, 0))
                    quantities[name]["executed"] += record.executed.get(slot, 0)
                cents -= int(info["reward_cents"])
                done = terminated or truncated
            return {
                "episode": episode,
                "submission_sha256": sha,
                "policy_seed": seed,
                "cost_cents": cents,
                "cost_components_usd": dict(costs),
                "salvage_usd": env.unwrapped.core.trajectory.salvage,
                "quantities_by_commodity": dict(quantities),
                "weeks": len(hashes),
                "action_hashes": hashes,
                "max_cpu_s": max(cpus),
                "gate_switch_weeks": switches,
                "hybrid_changed_slots": changes,
                "cpu_budget_enforced": False,
            }
        finally:
            env.close()


def main(
    agents=(
        "team_agent_receiver_overflow_v15",
        "team_agent_hybrid_risk_gate_v18",
        "team_agent_hybrid_shadow_price_v18",
    ),
    task="tiny",
    episodes=4,
    entropy=12345,
    out=None,
    n_jobs=1,
):
    if entropy != 12345:
        raise ValueError("exploratory screening is restricted to the training root 12345")
    if isinstance(agents, str):
        agents = [agents]
    folder = Path(out or f"outputs/16_policy_screen/{datetime.now():%Y-%m-%d_%H-%M-%S_%f}")
    folder.mkdir(parents=True, exist_ok=True)
    if not isinstance(n_jobs, int) or isinstance(n_jobs, bool) or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer")
    settings = {"task": task, "entropy": entropy, "episodes": episodes, "agents": list(agents)}
    settings_path = folder / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("screen settings changed; use a new output directory")
    write_json(settings_path, settings)
    report = {
        "kind": "training_raw_cost_screen_not_rss",
        "task": task,
        "entropy": entropy,
        "episodes": episodes,
        "agents": {},
    }
    for number, source in enumerate(agents):
        frozen = freeze(source, folder / "frozen" / f"{number}.zip")
        rows, missing = [], []
        for episode in range(episodes):
            checkpoint = folder / "rollouts" / str(number) / f"{episode}.json"
            if checkpoint.exists():
                row = json.loads(checkpoint.read_text())
                if row["submission_sha256"] != frozen["sha256"]:
                    raise ValueError("checkpoint submission differs")
            else:
                missing.append((episode, checkpoint))
                continue
            rows.append(row)
        # These are unmetered screening replays. Final operational checks and
        # CPU-budgeted score rollouts remain sequential in example 14.
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(play, frozen["zip"], frozen["sha256"], task, entropy, episode): checkpoint
                       for episode, checkpoint in missing}
            for future in as_completed(futures):
                row = future.result()
                write_json(futures[future], row)
                rows.append(row)
                print(f"{Path(source).name} episode {row['episode']}: ${row['cost_cents'] / 100:,.2f}; "
                      f"CPU {row['max_cpu_s']:.3f}s (unmetered screen)", flush=True)
        rows.sort(key=lambda row: row["episode"])
        report["agents"][source] = dict(
            frozen=frozen, rows=rows, mean_cost_usd=float(np.mean([r["cost_cents"] for r in rows])) / 100
        )
        write_json(folder / "report.json", report)
    return None


if __name__ == "__main__":
    fire.Fire(main)
