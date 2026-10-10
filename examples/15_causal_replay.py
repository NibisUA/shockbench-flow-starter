"""Trace frozen actions, needs and realised costs without changing the policy.

uv run python examples/15_causal_replay.py --task=full --weeks=0
"""

import json
from datetime import datetime
from pathlib import Path

import fire

from sbf_starter.causal import replay


def main(
    task="tiny",
    agent="team_agent_hybrid_v16",
    entropy=12345,
    episode=0,
    weeks=0,
    replay_seed=0,
    planner="team_agent_hybrid_v16",
    out=None,
):
    folder = Path(out or f"outputs/15_causal_replay/{datetime.now():%Y-%m-%d_%H-%M-%S_%f}")
    folder.mkdir(parents=True, exist_ok=False)
    report = replay(agent, task, entropy, episode, weeks, replay_seed, planner, folder / "trace.jsonl")
    (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {folder / 'report.json'}; complete={report['complete_episode']}; cost=${report['cost_usd']:,.2f}")


if __name__ == "__main__":
    fire.Fire(main)
