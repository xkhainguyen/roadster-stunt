import numpy as np
import pytest

from roadster import params
from roadster.trajopt import Plan, plan_kickflip, reduced_wrench


@pytest.fixture(scope="module")
def plan():
    return plan_kickflip()


def test_plan_meets_end_state(plan):
    y, z, vy, vz, phi, om = plan.X[-1]
    assert phi == pytest.approx(2 * np.pi, abs=0.01)
    assert z == pytest.approx(params.Z_TOUCH, abs=0.01)
    assert abs(vz) <= 0.6 + 1e-6
    assert abs(om) < 1e-3 and abs(y) < 1e-3 and abs(vy) < 1e-3


def test_plan_clears_obstacles(plan):
    x = params.VX0 * plan.t
    over = (x >= params.ROW_START) & (x <= params.ROW_START + params.ROW_LEN)
    assert over.any()
    assert np.all(plan.X[over, 1] >= params.ROW_HEIGHT + params.CLEARANCE_RADIUS - 1e-6)


def test_plan_budget(plan):
    dt = np.diff(plan.t)
    impulse = float((plan.U.sum(axis=1) * dt).sum())
    assert impulse < 0.5 * params.IMPULSE_BUDGET


def test_plan_roundtrip(plan, tmp_path):
    p = tmp_path / "plan.npz"
    plan.save(p)
    q = Plan.load(p)
    np.testing.assert_array_equal(plan.t, q.t)
    np.testing.assert_array_equal(plan.X, q.X)
    np.testing.assert_array_equal(plan.U, q.U)
    assert len(q.iterates) > 0
    assert len(q.iterates_T) == len(q.iterates)


def test_torque_sign_matches_allocation():
    from roadster.model import load
    from roadster.thrusters import allocation_matrix
    m, _ = load()
    A = allocation_matrix(m)
    # unit force on the left-side lift pair (fl + rl) vs the reduced model
    f = np.zeros(m.nu)
    f[[m.actuator("lift_fl").id, m.actuator("lift_rl").id]] = 0.5
    Fz, tx = reduced_wrench(np.array([1.0, 0, 0, 0]))
    assert (A @ f)[2] == pytest.approx(Fz)
    assert (A @ f)[3] == pytest.approx(tx)
    f = np.zeros(m.nu)
    f[[m.actuator("roof_fl").id, m.actuator("roof_rl").id]] = 0.5
    Fz, tx = reduced_wrench(np.array([0, 0, 1.0, 0]))
    assert (A @ f)[2] == pytest.approx(Fz)
    assert (A @ f)[3] == pytest.approx(tx)
