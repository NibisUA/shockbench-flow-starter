"""Conservation and isolated flags for the incremental v16 policy."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from shockbench_flow_gym import agent_config_from_reset

from agents.team_agent_hybrid_v16.allocation import Allocator
from agents.team_agent_hybrid_v16.contracts import Quantity
from agents.team_agent_hybrid_v16.hybrid import DispatchLedger, HybridController
from sbf_starter import env_id
from sbf_starter.agents import load
from tests.conftest import ROOT
from tests.test_current_network import make_current_config, observation


def need(identifier="need", *, destination=4, quantity=5.0, due=8, penalty=100.0):
    return SimpleNamespace(
        need_id=identifier,
        destination_node=destination,
        commodity_id=0,
        quantity=quantity,
        due_week=due,
        priority=1.0,
        reason="forecast_demand",
        shortage_cost_per_unit_usd=penalty,
        confidence=None,
    )


@pytest.fixture
def fixture():
    config = make_current_config()
    config["static"]["edges"].update(alt_of=[None] * 10, mode=["sea"] * 10)
    config["static"]["lanes"]["alt_of"] = [None] * 3
    config["static"]["instance"]["params"] = {
        "fleet_share": {"tb": 1.0, "ct": 1.0},
        "fleet_measure": {"tb": 100.0, "ct": 100.0},
    }
    config["layout"]["stock_slots"] = [[0, 0], [3, 0], [0, 1]]
    config["layout"]["release_pairs"] = []
    allocator = Allocator(config)
    state = SimpleNamespace(
        week=1,
        horizon=12,
        available_stock={
            (0, 0): Quantity(10.0, "observed"),
            (3, 0): Quantity(7.0, "observed"),
            (0, 1): Quantity(5.0, "observed"),
        },
        queues=(),
        pipeline=(),
        arrivals=(),
        issues=(),
    )
    obs = observation(config)
    original = allocator.delivery.options

    def empty_queues(*args, **kwargs):
        return original(*args, queue_work_at_arrival={(1, "tb"): 0, (2, "tb"): 0, (1, "ct"): 0, (2, "ct"): 0}, **kwargs)

    allocator.delivery.options = empty_queues
    return SimpleNamespace(config=config, allocator=allocator, state=state, obs=obs)


def controller(fixture, needs, *, priority=False, residual=False):
    pipeline = SimpleNamespace(
        allocator=fixture.allocator,
        state_builder=SimpleNamespace(build=lambda *_args: fixture.state),
        need_planner=SimpleNamespace(plan=lambda *_args: needs),
    )
    return HybridController(
        fixture.config, fixture.allocator.network, pipeline, priority_enabled=priority, residual_enabled=residual
    )


def assert_conserved(ledger):
    report = ledger.report()
    for item in (*report["stock"], *report["entry"]):
        assert 0 <= item["used"] <= item["limit"] + 1e-9
    for item in report["fleet"]:
        assert item["used"] + item["reserved_release"] <= item["limit"] + 1e-9


def test_shared_source_and_entry_priority_displaces_low_value_request(fixture):
    hybrid = controller(
        fixture,
        [need("low", destination=3, quantity=8, penalty=10), need("critical", quantity=8, penalty=1000)],
        priority=True,
    )
    result = hybrid.apply({"flows": np.array([8.0, 8.0, 0, 0, 0])}, fixture.obs)
    np.testing.assert_allclose(result["flows"], [2, 8, 0, 0, 0])
    assert_conserved(hybrid.last_ledger)
    assert any(item["stage"] == "changed_action" and item["slot_id"] == 0 for item in hybrid.last_trace)


def test_residual_uses_same_stock_and_subtracts_heuristic_arrivals_once(fixture):
    hybrid = controller(fixture, [need("already", destination=3, quantity=8), need("new", quantity=5)], residual=True)
    result = hybrid.apply({"flows": np.array([8.0, 0, 0, 0, 0])}, fixture.obs)
    np.testing.assert_allclose(result["flows"], [8, 2, 0, 0, 0])
    assert sum(item["quantity"] for item in hybrid.last_trace if item["stage"] == "residual_assignment") == 2
    assert not any(
        item.get("need_id") == "already" and item["stage"] == "residual_assignment" for item in hybrid.last_trace
    )
    assert_conserved(hybrid.last_ledger)


def test_late_heuristic_cargo_does_not_cover_early_need(fixture):
    hybrid = controller(fixture, [need(quantity=5, due=2)], residual=True)
    result = hybrid.apply({"flows": np.array([0, 5.0, 0, 0, 0])}, fixture.obs)
    np.testing.assert_allclose(result["flows"], [0, 5, 0, 0, 5])
    assert any(
        item["stage"] == "residual_prefilter" and item["reason"] == "created_too_late" for item in hybrid.last_trace
    )
    assert_conserved(hybrid.last_ledger)


@pytest.mark.parametrize("penalty,reason", [(None, "shortage_value_unknown"), (0.0, "nonpositive_marginal_value")])
def test_unknown_and_zero_shortage_values_are_distinct(fixture, penalty, reason):
    hybrid = controller(fixture, [need(penalty=penalty)], residual=True)
    result = hybrid.apply({"flows": np.zeros(5)}, fixture.obs)
    assert not result["flows"].any()
    assert any(item["reason"] == reason for item in hybrid.last_trace if item["stage"] == "residual_prefilter")


def test_unknown_queue_eta_does_not_become_an_on_time_arrival(fixture):
    # Restore the actual production evaluator, which makes no empty-queue claim.
    from agents.team_agent_hybrid_v16.delivery import DeliveryEvaluator

    fixture.allocator.delivery = DeliveryEvaluator(fixture.config, fixture.allocator.network)
    hybrid = controller(fixture, [need(destination=3)], residual=True)
    result = hybrid.apply({"flows": np.zeros(5)}, fixture.obs)
    assert not result["flows"].any()
    assert any(item.get("reason") == "eta_unknown" for item in hybrid.last_trace)


def test_hidden_stock_preserves_entire_heuristic_action(fixture):
    fixture.state.available_stock[0, 0] = Quantity(None, "unknown")
    hybrid = controller(fixture, [need()], priority=True, residual=True)
    action = {"flows": np.array([2.0, 3.0, 0, 0, 0])}
    assert hybrid.apply(action, fixture.obs) is action
    assert hybrid.last_ledger is None
    assert hybrid.last_trace[0]["reason"] == "source_stock_unknown"


def test_release_override_reserves_fleet_before_residual_dispatch(fixture):
    cfg = fixture.config
    cfg["static"]["edges"]["alt_of"][6] = {"edge": 0}
    cfg["static"]["edges"]["tau0"][6] = 5  # Four fleet weeks per new unit.
    cfg["static"]["edges"]["alt_of"][3] = {"edge": 2}  # Three per override unit.
    cfg["static"]["instance"]["params"]["fleet_measure"]["tb"] = 12.0
    cfg["static"]["override_slots"] = {"chokepoint": [2], "k": [0], "lane": [1], "out_edge": [3]}
    cfg["layout"]["release_pairs"] = [[2, 0]]
    allocator = Allocator(cfg)
    obs = observation(cfg)
    fixture.state.queues = (
        SimpleNamespace(
            chokepoint_node=2, commodity_id=0, lane_id=0, next_edge_id=2, quantity=Quantity(2.0, "observed")
        ),
    )
    action = {"flows": np.array([0, 0, 0, 0, 9.0]), "override_qty": np.array([2.0]), "release_mode": np.array([1])}
    ledger = DispatchLedger(allocator, fixture.state, obs, allocator._snapshot(obs), action)
    assert ledger.release_bound["tb"] == 6.0
    assert ledger.reserve(4, 9.0, "heuristic") == 1.5
    assert ledger.reserve(4, 9.0, "residual") == 0.0
    assert_conserved(ledger)


def test_projection_matches_edge_then_stock_order(fixture):
    requests = np.array([20.0, 20.0, 0, 0, 10.0])
    ledger = DispatchLedger(
        fixture.allocator, fixture.state, fixture.obs, fixture.allocator._snapshot(fixture.obs), {"flows": requests}
    )
    # Entry clips 40 -> 10 before shared stock clips the combined 10 + 9.
    np.testing.assert_allclose(ledger.project(requests), [50 / 19, 50 / 19, 0, 0, 90 / 19])


def test_hybrid_flags_off_preserve_baseline_actions():
    env = gym.make(env_id("tiny"), entropy=715827)
    try:
        obs, info = env.reset(seed=0, options={"episode": 0})
        cfg = agent_config_from_reset(env, obs, info)
        baseline = load(ROOT / "agents" / "team_agent_receiver_overflow_v15")(cfg)
        candidate = load(ROOT / "agents" / "team_agent_hybrid_v16")(cfg)
        assert candidate.hybrid is None
        for _week in range(4):
            expected, actual = baseline.act(obs), candidate.act(obs)
            assert expected.keys() == actual.keys()
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key])
            obs, _reward, terminated, truncated, _info = env.step(actual)
            if terminated or truncated:
                break
    finally:
        env.close()
