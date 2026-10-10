from types import SimpleNamespace

import numpy as np

from agents.team_agent_clipping_repair_v19.hybrid import HybridController
from tests import test_hybrid_v16 as helpers
from tests.test_hybrid_v16 import assert_conserved, need


fixture = helpers.fixture


def repair(fixture, needs):
    pipeline = SimpleNamespace(
        allocator=fixture.allocator,
        state_builder=SimpleNamespace(build=lambda *_: fixture.state),
        need_planner=SimpleNamespace(plan=lambda *_: needs),
    )
    return HybridController(
        fixture.config, fixture.allocator.network, pipeline, priority_enabled=True, clipping_repair_enabled=True
    )


def test_unknown_value_does_not_trigger_arbitrary_reordering(fixture):
    obj = repair(fixture, [need(penalty=None)])
    action = {"flows": np.array([8.0, 8.0, 0.0, 0.0, 0.0])}
    assert obj.apply(action, fixture.obs) is action
    assert obj.last_trace[0]["reason"] == "no_certified_profitable_recovery"


def test_nonbinding_action_preserved_exactly(fixture):
    obj = repair(fixture, [need(penalty=1000)])
    action = {"flows": np.array([1.0, 1.0, 0.0, 0.0, 0.0])}
    assert obj.apply(action, fixture.obs) is action


def test_recover_edge_clipping_on_available_alternative_without_double_spend(fixture):
    fixture.state.available_stock[0, 0] = fixture.state.available_stock[0, 0].__class__(30.0, "observed")
    obj = repair(fixture, [need(quantity=20, penalty=1000)])
    # Slots 1 and 4 reach the same sink from the same source; entry 0
    # binds at ten units, while the alternative has nine spare units.
    action = {"flows": np.array([0.0, 20.0, 0.0, 0.0, 0.0])}
    actual = obj.apply(action, fixture.obs)
    np.testing.assert_allclose(actual["flows"], [0, 10, 0, 0, 9])
    assert_conserved(obj.last_ledger)


def test_physically_late_alternative_cannot_claim_shortage_prevention(fixture):
    fixture.state.available_stock[0, 0] = fixture.state.available_stock[0, 0].__class__(30.0, "observed")
    obj = repair(fixture, [need(quantity=20, due=1, penalty=1000)])
    action = {"flows": np.array([0.0, 20.0, 0.0, 0.0, 0.0])}
    assert obj.apply(action, fixture.obs) is action
