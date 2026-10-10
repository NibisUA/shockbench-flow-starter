"""Freeze and resume a paired experiment, prepare references, or check audit gates.

    uv run python examples/14_audit_experiment.py --task=tiny --episodes=1 --quick
    uv run python examples/14_audit_experiment.py --task=full --episodes=64 --entropy=12345 \
        --agent=team_agent_hybrid_v16 --against=team_agent_receiver_overflow_v15 \
        --out=outputs/14_audit_experiment/search --reference_jobs=3

Repeat the same command/output directory to resume. --prepare_only warms cached
references without running agents; omit that flag to continue policy scoring.
Use --phase=validation --entropy=67890 for independent Full and Small evidence.
Root 0 is reserved for --phase=confirmation. Defaults run on Tiny.
"""

import json
from pathlib import Path

import fire

from sbf_starter.experiment import acceptance_gates, create_ablation, run_experiment, write_json


def main(
    agent="heuristic",
    against="template",
    task="tiny",
    episodes=4,
    entropy=12345,
    phase="search",
    quick=False,
    cpu_budget=True,
    reference_jobs=1,
    reference_chunk=4,
    prepare_only=False,
    bootstrap_seed=0,
    out=None,
    cache_dir="outputs/cache",
    operation="score",
    key=None,
    value=None,
    group=None,
    reports=(),
    operational=None,
    causal=None,
):
    """Score (default), create one-factor ablation, or validate acceptance gates.

    gate reports: tuple/list of result.json files; operational: JSON list of
    packed checks; causal: measured metric JSON. Missing evidence fails closed.
    ablation: --agent=source --out=new_folder --key=dotted.key --value=VALUE
    --group=component. Evidence goes beside, never inside, the submission.
    """
    if operation == "ablation":
        if not out:
            raise ValueError("ablation requires a new output candidate directory")
        result = create_ablation(agent, out, key, value, group)
    elif operation == "gates":
        if isinstance(reports, str):
            reports = [reports]
        result = acceptance_gates(
            [json.loads(Path(path).read_text()) for path in reports],
            json.loads(Path(operational).read_text()) if operational else None,
            json.loads(Path(causal).read_text()) if causal else None,
        )
        if out:
            write_json(Path(out) / "gates.json", result)
    elif operation == "score":
        result = run_experiment(
            agent,
            against,
            task,
            episodes,
            entropy,
            phase,
            quick,
            cpu_budget,
            reference_jobs,
            reference_chunk,
            prepare_only,
            bootstrap_seed,
            out,
            cache_dir,
        )
    else:
        raise ValueError("operation must be score, ablation or gates")
    if operation != "score":
        print(json.dumps(result, indent=2))
    return None


if __name__ == "__main__":
    fire.Fire(main)
