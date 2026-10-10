import json
from pathlib import Path

import numpy as np
import pytest
from shockbench_flow_agent import EpisodeSet

from sbf_starter.causal import warning_reliability
from sbf_starter.experiment import acceptance_gates, comparison_from_rows, episode_ids, freeze
from sbf_starter.parameters import catalogue, validate


def test_frozen_submission_is_deterministic_and_never_overwritten(tmp_path):
    source = tmp_path / "agent"
    source.mkdir()
    (source / "agent.py").write_text("class Agent:\n    def act(self, observation):\n        return {}\n")
    target = tmp_path / "frozen.zip"
    first = freeze(source, target)
    assert freeze(source, target)["sha256"] == first["sha256"]
    (source / "agent.py").write_text("class Agent:\n    def act(self, observation):\n        return {'flows': []}\n")
    with pytest.raises(FileExistsError):
        freeze(source, target)
    assert freeze(target, tmp_path / "copy.zip")["sha256"] == first["sha256"]


@pytest.mark.parametrize("value", [False, 0, -2, [], [1, 1], [True], [1.5]])
def test_episode_ids_reject_invalid_or_pseudoreplicated_samples(value):
    with pytest.raises(ValueError):
        episode_ids(value)


def test_no_validation_evidence_cannot_pass_requested_rss_gate():
    result = acceptance_gates([])
    assert not result["accepted"]
    assert not result["gates"]["full_rss"]["passed"]


def test_warning_backtest_excludes_censored_and_hidden_future():
    rows = [
        {"week": n, "warnings": {"strait": {"score": 0.8, "open": opening}}}
        for n, opening in enumerate([1.0, 1.0, 0.0, None], start=1)
    ]
    result = warning_reliability(rows, horizon=2)
    assert result["bins"]["[0.5,1)"]["count"] == 1
    assert result["bins"]["[0.5,1)"]["events"] == 1
    assert not result["scores_are_probabilities"]


def test_catalogue_tracks_real_defaults_and_rejects_bool_as_integer():
    source = Path("agents/team_agent_clipping_repair_v19")
    schema = catalogue(source)
    params = json.loads((source / "params.json").read_text())
    validate(params, schema)
    assert schema["properties"]["fuel_mpc_horizon"]["maximum"] == 24
    assert schema["properties"]["planner_options.production_horizon"]["default"] == 8
    with pytest.raises(ValueError):
        validate(dict(params, fuel_mpc_horizon=True), schema)
    with pytest.raises(ValueError):
        validate(dict(params, fuel_mpc_horizon=25), schema)
    with pytest.raises(ValueError):
        validate(dict(params, nonexistent_switch=True), schema)


def test_checkpoint_comparison_rejects_scenario_mismatch():
    # A minimal reference container suffices: mismatch must be caught before
    # bootstrapping or expensive scoring is attempted.
    from types import SimpleNamespace

    es = SimpleNamespace(episodes=(3,), references=({"omega_hash": "expected"},))
    rows = [{"episode": 3, "omega_hash": "wrong", "J_policy_cents": 100}]
    with pytest.raises(ValueError, match="scenario"):
        comparison_from_rows(es, rows, rows, True)


def test_score_reconstruction_matches_official_pairing(tmp_path):
    # No environment rollouts or LP solves: genuine benchmark RSS/bootstrap
    # functions over a small fixed, stratified set of integer-cent costs.
    from shockbench_flow_agent.scoring import LEVEL, N_BOOT, _boot_stats

    refs = tuple(
        {
            "episode": n,
            "omega_hash": str(n),
            "stratum": n % 4 + 1,
            "J_naive_cents": 10000,
            "J_oracle_cents": 1000,
            "J_clairvoyant_cents": 1000,
            "excluded": None,
        }
        for n in range(8)
    )
    es = EpisodeSet("tiny", "standard", 12345, tuple(range(8)), refs, 1000, 2000, tmp_path, False, False)
    a = [
        dict(
            episode=n,
            omega_hash=str(n),
            J_policy_cents=2000 + n * 100,
            weeks=26,
            fallback_weeks=0,
            cpu_weeks=0,
            invalid_entries=0,
        )
        for n in range(8)
    ]
    b = [dict(row, J_policy_cents=row["J_policy_cents"] + 500) for row in a]
    comp = comparison_from_rows(es, a, b, True)
    boot = _boot_stats(refs, [[r["J_policy_cents"] for r in a], [r["J_policy_cents"] for r in b]], True, N_BOOT, 0)
    official_a = es._score("candidate", a, es._budget(True), LEVEL, boot[0])
    assert comp.a.rss == official_a.rss
    assert comp.diff == pytest.approx(500 / 9000)
    np.testing.assert_allclose(comp.a.boot, official_a.boot)
