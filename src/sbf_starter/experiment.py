"""Frozen, resumable paired experiments and conservative audit acceptance gates.

Policy rollouts are always sequential. Reference calculations may use workers;
they finish before the policy CPU meter starts. Scoring/bootstrapping delegates
to the locked benchmark so exclusions and harm weights have one implementation.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
import zipfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from sbf_starter import ROOT, check_task, cpu_budget_s, scoring
from sbf_starter.agents import resolve


SCHEMA_VERSION = 1


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Atomically replace a checkpoint; an interrupted write leaves the previous one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def freeze(source: str | Path, target: Path) -> dict:
    """Copy exact ZIP bytes, or make the server's deterministic ZIP; never overwrite."""
    from shockbench_flow_agent.submission import build_submission, check_zip

    source = resolve(source).resolve()
    if source.is_file() and source.suffix != ".zip":
        raise ValueError("freeze a complete submission folder or zip, not an individual agent.py")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".freeze-", dir=target.parent) as temporary:
        staged = Path(temporary) / "submission.zip"
        if source.is_dir():
            build_submission(source, staged)
        else:
            shutil.copyfile(source, staged)
        checked = check_zip(staged)
        if target.exists():
            if sha256(target) != checked.sha256:
                raise FileExistsError(f"frozen submission differs: {target}; choose a new experiment directory")
        else:
            # Exclusive creation prevents a concurrent freezer from replacing evidence.
            with target.open("xb") as stream:
                stream.write(staged.read_bytes())
            target.chmod(0o444)
    with zipfile.ZipFile(target) as archive:
        params = (
            json.loads(archive.read("params.json").decode("utf-8-sig")) if "params.json" in archive.namelist() else {}
        )
    return {
        "source": str(source),
        "zip": str(target.resolve()),
        "sha256": checked.sha256,
        "files": dict(checked.files),
        "params": params,
        "params_sha256": json_fingerprint(params),
    }


def environment_manifest(root: Path = ROOT) -> dict:
    from shockbench_flow.evaluation.cache import package_sha256

    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    status = git("status", "--porcelain=v1", "--untracked-files=normal")
    lock = root / "uv.lock"
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())),
        "benchmark_source_sha256": package_sha256(),
        "uv_lock_sha256": sha256(lock) if lock.exists() else None,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "git_status": status,
    }


def runtime_identity(environment: dict) -> dict:
    """Dirty unrelated working files may change; the executed frozen code must not."""
    keys = ("python", "implementation", "packages", "benchmark_source_sha256", "uv_lock_sha256")
    return {key: environment[key] for key in keys}


def episode_ids(episodes: int | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(episodes, int) and not isinstance(episodes, bool):
        episodes = list(range(episodes))
    if not isinstance(episodes, (tuple, list)) or not episodes:
        raise ValueError("episodes must be a positive count or a nonempty list of explicit IDs")
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in episodes):
        raise ValueError("episode IDs must be nonnegative integers")
    if len(set(episodes)) != len(episodes):
        raise ValueError("episode IDs must be unique")
    return list(episodes)


@contextlib.contextmanager
def exclusive_run(path: Path):
    """OS lock released automatically on crashes; no stale PID lock to delete."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another experiment owns {path}; CPU scoring must run sequentially") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def prepare_references(settings: dict, folder: Path):
    """Bound reference work lost to a crash to one chunk, then reuse official cache."""
    from shockbench_flow_agent import EpisodeSet

    ids = settings["episode_ids"]
    common = {
        "task": settings["task"],
        "entropy": settings["entropy"],
        "quick": settings["quick"],
        "cache_dir": settings["cache_dir"],
        "n_jobs": settings["reference_jobs"],
        "verbose": True,
    }
    chunk_size = settings["reference_chunk"]
    for offset in range(0, len(ids), chunk_size):
        subset = EpisodeSet.build(episodes=ids[offset : offset + chunk_size], **common)
        for row in subset.references:
            write_json(folder / "references" / f"{row['episode']}.json", row)
        print(f"Reference checkpoint: {min(offset + chunk_size, len(ids))}/{len(ids)}", flush=True)
    return EpisodeSet.build(episodes=ids, **{**common, "verbose": False})


def comparison_from_rows(es, candidate: list[dict], baseline: list[dict], budget, seed: int = 0):
    """Rebuild exactly the benchmark's paired report without replaying saved episodes.

    0.1.2 exposes play() and rss(), but no public score-from-rows method. Keep
    this small adapter to its scoring internals; the runtime/lock fingerprint
    prevents resuming across package updates, and tests compare its output.
    """
    import numpy as np
    from shockbench_flow_agent.scoring import LEVEL, N_BOOT, Comparison, _boot_stats, _interval

    expected = list(es.episodes)
    for rows in (candidate, baseline):
        if [r["episode"] for r in rows] != expected:
            raise ValueError("paired episode IDs/order do not match the reference set")
        for reference, row in zip(es.references, rows, strict=True):
            if row["omega_hash"] != reference["omega_hash"]:
                raise ValueError("rollout scenario hash does not match the reference")
    costs = [[r["J_policy_cents"] for r in rows] for rows in (candidate, baseline)]
    pooled = all(es.rss(cost)["pooled"] for cost in costs)
    ba, bb = _boot_stats(es.references, costs, pooled, N_BOOT, seed)
    a = es._score("candidate", candidate, es._budget(budget), LEVEL, ba)
    b = es._score("baseline", baseline, es._budget(budget), LEVEL, bb)
    delta = ba - bb
    finite = delta[np.isfinite(delta)]
    diff = a.rss - b.rss if a.rss is not None and b.rss is not None else None
    return Comparison(a, b, diff, _interval(delta, LEVEL), LEVEL, float((finite > 0).mean()) if finite.size else None)


def full_score_dict(score) -> dict:
    doc = scoring.as_dict(score)
    doc.update(
        rss_by_stratum=score.rss_by_stratum,
        excluded=list(score.excluded),
        invalid_entries=score.invalid_entries,
        cpu_budget_s=score.cpu_budget_s,
        interval_level=score.level,
        per_episode=list(score.rows),
    )
    doc["by_harm"] = {}
    for harm in sorted({row["stratum"] for row in score.rows}, key=str):
        rows = [row for row in score.rows if row["stratum"] == harm]
        doc["by_harm"][str(harm)] = {
            "episodes": len(rows),
            "rss": score.rss_by_stratum.get(harm),
            "mean_cost_usd": sum(row["J_policy_cents"] for row in rows) / (100 * len(rows)),
            "fallback_weeks": sum(row["fallback_weeks"] for row in rows),
            "cpu_weeks": sum(row["cpu_weeks"] for row in rows),
            "invalid_entries": sum(row["invalid_entries"] for row in rows),
        }
    return doc


def run_experiment(
    agent: str = "heuristic",
    against: str = "template",
    task: str = "tiny",
    episodes: int | list[int] = 4,
    entropy: int = 12345,
    phase: str = "search",
    quick: bool = False,
    cpu_budget: bool | float = True,
    reference_jobs: int = 1,
    reference_chunk: int = 4,
    prepare_only: bool = False,
    bootstrap_seed: int = 0,
    out: str | None = None,
    cache_dir: str = "outputs/cache",
) -> dict:
    from shockbench_flow_agent.scoring import _policy_seed

    check_task(task)
    ids = episode_ids(episodes)
    if phase not in {"search", "validation", "confirmation", "smoke"}:
        raise ValueError("phase must be search, validation, confirmation or smoke")
    if entropy == 0 and phase not in {"confirmation", "smoke"}:
        raise ValueError("root 0 is reserved for final confirmation, never search or validation")
    if phase == "confirmation" and entropy != 0:
        raise ValueError("confirmation uses root 0; label other held-out roots validation")
    if reference_chunk < 1 or reference_jobs == 0:
        raise ValueError("reference_chunk must be positive and reference_jobs nonzero")
    if not isinstance(cpu_budget, (int, float)) or (cpu_budget is not False and cpu_budget <= 0):
        raise ValueError("cpu_budget must be False, True, or positive seconds")
    folder = Path(out or f"outputs/14_audit_experiment/{datetime.now():%Y-%m-%d_%H-%M-%S_%f}_{task}").resolve()
    folder.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir).resolve()
    settings = {
        "task": task,
        "episode_ids": ids,
        "entropy": entropy,
        "phase": phase,
        "quick": bool(quick),
        "cpu_budget": cpu_budget,
        "cpu_budget_s": cpu_budget_s(task) if cpu_budget is True else (float(cpu_budget) if cpu_budget else None),
        "policy_jobs": 1,
        "reference_jobs": reference_jobs,
        "reference_chunk": reference_chunk,
        "cache_dir": str(cache),
        "bootstrap_seed": bootstrap_seed,
        "replay_seed": 0,
        "replay_seed_scope": "diagnostic replay only; scored policy seeds are salted by submission SHA-256",
        "regime": "standard",
    }
    # One machine-wide advisory lock prevents separate run directories from
    # accidentally competing while their in-process CPU meters are active.
    with exclusive_run(Path(tempfile.gettempdir()) / "sbf-audit-policy.lock"):
        snapshots = {
            "candidate": freeze(agent, folder / "frozen" / "candidate.zip"),
            "baseline": freeze(against, folder / "frozen" / "baseline.zip"),
        }
        environment = environment_manifest()
        manifest_path = folder / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["settings"] != settings or runtime_identity(manifest["environment"]) != runtime_identity(
                environment
            ):
                raise ValueError(
                    "resume settings or Python/lock/package fingerprints changed; choose a new run directory"
                )
            if {k: v["sha256"] for k, v in snapshots.items()} != {
                k: v["sha256"] for k, v in manifest["submissions"].items()
            }:
                raise ValueError("resume submission hashes changed")
        else:
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "settings": settings,
                "environment": environment,
                "submissions": snapshots,
                "policy_seeds": {
                    label: {str(n): _policy_seed(entropy, n, item["sha256"]) for n in ids}
                    for label, item in snapshots.items()
                },
            }
            if (ROOT / "uv.lock").exists():
                shutil.copyfile(ROOT / "uv.lock", folder / "frozen" / "uv.lock")
                (folder / "frozen" / "uv.lock").chmod(0o444)
            write_json(manifest_path, manifest)
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "preparing",
            "manifest_sha256": sha256(manifest_path),
            "settings": settings,
            "submission_sha256": {k: v["sha256"] for k, v in snapshots.items()},
            "completed_rollouts": 0,
            "comparison": None,
        }
        started = time.perf_counter()
        result_path = folder / "result.json"
        write_json(result_path, report)
        print(f"Frozen experiment: {folder}", flush=True)
        try:
            es = prepare_references(settings, folder)
            if prepare_only:
                report["status"] = "references_ready"
                return report
            report["status"] = "scoring"
            write_json(result_path, report)
            rows = {"candidate": [], "baseline": []}
            for index, episode in enumerate(es.episodes):
                singleton = replace(es, episodes=(episode,), references=(es.references[index],), verbose=False)
                for label in rows:
                    frozen = Path(snapshots[label]["zip"])
                    if sha256(frozen) != snapshots[label]["sha256"]:
                        raise RuntimeError("frozen submission was modified during the experiment")
                    checkpoint = folder / "rollouts" / label / f"{episode}.json"
                    if checkpoint.exists():
                        saved = json.loads(checkpoint.read_text())
                        if saved["manifest_sha256"] != report["manifest_sha256"]:
                            raise ValueError("rollout checkpoint belongs to a different manifest")
                        row = saved["row"]
                    else:
                        row = singleton.play(str(frozen), cpu_budget=cpu_budget, n_jobs=1)[0]
                        write_json(checkpoint, {"manifest_sha256": report["manifest_sha256"], "row": row})
                    if row["episode"] != episode or row["omega_hash"] != es.references[index]["omega_hash"]:
                        raise ValueError("rollout checkpoint scenario differs from reference")
                    rows[label].append(row)
                    report["completed_rollouts"] += 1
                    write_json(result_path, report)
                    print(f"Episode {episode} {label}: ${row['J_policy_cents'] / 100:,.2f}", flush=True)
            comparison = comparison_from_rows(es, rows["candidate"], rows["baseline"], cpu_budget, bootstrap_seed)
            report["comparison"] = {
                "a": full_score_dict(comparison.a),
                "b": full_score_dict(comparison.b),
                "difference": comparison.diff,
                "interval": comparison.interval,
                "interval_level": comparison.level,
                "p_a_better": comparison.p_a_better,
            }
            report["status"] = "completed"
            text = f"{comparison}\n\n{comparison.a}\n\n{comparison.b}\n"
            (folder / "report.txt").write_text(text, encoding="utf-8")
            print(text, flush=True)
            return report
        except BaseException as exc:
            report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
            report["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            report["elapsed_seconds_this_attempt"] = round(time.perf_counter() - started, 3)
            write_json(result_path, report)


def create_ablation(source: str, out: str, key: str, value: object, group: str) -> dict:
    """Create a separate candidate changing exactly one existing dotted params key."""
    from shockbench_flow_agent.submission import extract_submission

    target = Path(out)
    if target.exists():
        raise FileExistsError("candidate directory already exists; ablations never overwrite agents")
    source_path = resolve(source)
    if not group or not key:
        raise ValueError("provide the parameter group and exactly one parameter key")
    with tempfile.TemporaryDirectory(prefix="sbf-ablation-") as temporary:
        temp = Path(temporary)
        frozen = freeze(source_path, temp / "source.zip")
        submission = extract_submission(temp / "source.zip", temp / "source")
        params = json.loads(json.dumps(frozen["params"]))
        cursor = params
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                raise ValueError(f"unknown parameter path: {key}")
            cursor = cursor[part]
        leaf = parts[-1]
        if leaf not in cursor:
            raise ValueError(f"unknown parameter: {key}")
        before = cursor[leaf]
        if type(value) is not type(before) and not (type(before) is float and type(value) in (float, int)):
            raise ValueError(f"{key} expects {type(before).__name__}, got {type(value).__name__}")
        if before == value:
            raise ValueError("ablation must change the parameter value")
        cursor[leaf] = value
        # Validate a shipped schema when the selected policy provides one.
        schema = submission.root / "parameter_catalogue.json"
        if schema.exists():
            from sbf_starter.parameters import validate

            validate(params, json.loads(schema.read_text()))
        shutil.copytree(submission.root, target)
        write_json(target / "params.json", params)
    change = {
        "source_sha256": frozen["sha256"],
        "source_params_sha256": frozen["params_sha256"],
        "params_sha256": json_fingerprint(params),
        "group": group,
        "changed_keys": [key],
        "changes": {key: {"before": before, "after": value}},
        "candidate": str(target.resolve()),
    }
    # Evidence lives outside the submission so it does not change its policy seed.
    write_json(target.with_name(target.name + ".ablation.json"), change)
    return change


def acceptance_gates(
    reports: list[dict], operational: list[dict] | None = None, causal: dict | None = None, target_rss: float = 0.85
) -> dict:
    """Missing/quick/underpowered evidence can never authorize an RSS success claim."""
    gates = {}
    selected = {}
    for phase, task, root in (("search", "full", 12345), ("validation", "full", 67890), ("validation", "small", 67890)):
        name = f"{task}_{phase}"
        matches = [
            r
            for r in reports
            if r.get("status") == "completed"
            and r.get("settings", {}).get("phase") == phase
            and r["settings"].get("task") == task
            and r["settings"].get("entropy") == root
        ]
        eligible = []
        for report in matches:
            settings, comparison = report["settings"], report.get("comparison") or {}
            a, b = comparison.get("a", {}), comparison.get("b", {})
            ids = settings.get("episode_ids", [])
            enough = len(ids) >= 64 and len(set(ids)) == len(ids)
            eligible.append(
                (
                    report,
                    enough
                    and not settings.get("quick", True)
                    and a.get("pooled", False)
                    and b.get("pooled", False)
                    and not a.get("quick", True)
                    and not b.get("quick", True)
                    and a.get("episodes") == len(ids)
                    and b.get("episodes") == len(ids),
                )
            )
        good = [r for r, ok in eligible if ok]
        gates[name] = {"passed": bool(good), "reason": "full pooled paired RSS on at least 64 unique episodes"}
        if good:
            selected[name] = good[-1]
    identity = [r["submission_sha256"] for r in selected.values()]
    gates["same_frozen_agents"] = {
        "passed": len(identity) == 3 and all(item == identity[0] for item in identity),
        "reason": "search, independent validation and Small must use the same candidate/baseline ZIPs",
    }
    search, validation = selected.get("full_search"), selected.get("full_validation")
    gates["same_full_budget"] = {
        "passed": bool(
            search
            and validation
            and search["settings"]["cpu_budget_s"] is not None
            and search["settings"]["cpu_budget_s"] == validation["settings"]["cpu_budget_s"]
        ),
        "reason": "same metered CPU budget on Full search and validation",
    }
    if validation:
        comp = validation["comparison"]
        interval = comp.get("interval")
        gates["full_rss"] = {
            "passed": comp["a"].get("rss") is not None and comp["a"]["rss"] >= target_rss,
            "observed": comp["a"].get("rss"),
            "required": target_rss,
        }
        gates["paired_improvement"] = {
            "passed": comp.get("difference") is not None
            and comp["difference"] > 0
            and interval is not None
            and interval[0] > 0,
            "difference": comp.get("difference"),
            "interval": interval,
            "reason": "positive 90% paired interval; widen the episode set if inconclusive",
        }
    else:
        gates["full_rss"] = gates["paired_improvement"] = {
            "passed": False,
            "reason": "independent Full evidence missing",
        }
    small = selected.get("small_validation")
    interval = small["comparison"].get("interval") if small else None
    gates["small_nonregression"] = {
        "passed": bool(small and interval is not None and interval[0] >= 0),
        "interval": interval,
        "reason": "Small paired interval excludes a regression",
    }
    candidate_hash = identity[0]["candidate"] if identity else None
    checks = operational or []
    gates["operational"] = {
        "passed": candidate_hash is not None
        and all(
            any(
                c.get("candidate_sha256") == candidate_hash
                and c.get("task") == task
                and c.get("passed") is True
                and c.get("fallback_weeks") == 0
                and c.get("invalid_entries") == 0
                and c.get("max_cpu_s", float("inf")) < cpu_budget_s(task)
                for c in checks
            )
            for task in ("small", "full")
        ),
        "reason": "packed candidate passes Small/Full checks, no fallbacks/invalid entries, CPU below task budget",
    }
    causal = causal or {}
    improved = False
    if isinstance(causal.get("before"), (float, int)) and isinstance(causal.get("after"), (float, int)):
        improved = (
            causal["after"] < causal["before"]
            if causal.get("lower_is_better", True)
            else causal["after"] > causal["before"]
        )
    gates["causal"] = {
        "passed": candidate_hash is not None
        and causal.get("candidate_sha256") == candidate_hash
        and bool(causal.get("metric"))
        and bool(causal.get("unit"))
        and improved
        and causal.get("cost_delta_usd", float("inf")) <= 0,
        "reason": "matching measured deficit metric improves with no total-cost increase",
    }
    return {"accepted": all(g["passed"] for g in gates.values()), "gates": gates}
