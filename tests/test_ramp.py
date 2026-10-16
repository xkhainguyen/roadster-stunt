import mujoco
import numpy as np
import pytest

from roadster import params
from roadster.flight import Log, touchdown_metrics
from roadster.model import load, ramp_point
from roadster.ramp import MPC_EVERY, SCENARIOS, get_plan, lip_state, simulate
from roadster.ramp_plan import R_UPRIGHT, attitude, body_rate


@pytest.fixture(scope="module")
def plan():
    return get_plan()


@pytest.fixture(scope="module")
def runs():
    return {(s, cl): simulate(s, closed_loop=cl) for s in SCENARIOS for cl in (True, False)}


def _ok(m):
    return (m["touchdown_vz"] <= 1.5 and m["roll_err_deg"] <= 10 and m["pitch_err_deg"] <= 10
            and not m["hit_obstacle"] and m["final_upright"] > 0.9)


def _touchdown_nose(log):
    """Body x axis and velocity at the first touchdown step."""
    i = np.flatnonzero(log.phase == "rollout")[0]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, log.qpos[i, 3:7])
    return R.reshape(3, 3)[:, 0], log.qvel[i, :3]


def _heading_err_deg(nose):
    """Angle between the nose, projected on the ground, and -x (away from the ramp)."""
    h = nose[:2] / np.linalg.norm(nose[:2])
    return float(np.degrees(np.arccos(np.clip(-h[0], -1, 1))))


def test_ramp_facets_follow_the_arc():
    m, d = load("ramp")
    ids = [i for i in range(m.ngeom) if m.geom(i).name.startswith("ramp_")]
    assert len(ids) == round(params.RAMP_LIP_DEG / params.RAMP_SEG_DEG)
    center = np.array([params.RAMP_X0, params.RAMP_R])
    for i in ids:
        R = d.geom_xmat[i].reshape(3, 3)
        top = d.geom_xpos[i] + m.geom_size[i][2] * R[:, 2]  # top face center, toward the arc center
        r = np.linalg.norm(top[[0, 2]] - center)
        assert params.RAMP_R * np.cos(np.radians(params.RAMP_SEG_DEG / 2)) - 1e-3 <= r <= params.RAMP_R + 1e-3  # XML rounds to 0.1 mm
        assert np.dot(R[[0, 2], 2], center - top[[0, 2]]) > 0


def test_lip_overhangs():
    x_lip, z_lip = ramp_point(np.radians(params.RAMP_LIP_DEG))
    assert x_lip < params.RAMP_X0 + params.RAMP_R  # past vertical
    assert z_lip > 10.0


def test_attitude_path_holds_heading():
    beta0 = np.radians(-121.0)
    c, sn = np.cos(beta0), np.sin(beta0)
    np.testing.assert_allclose(attitude(0.0, beta0), [[c, 0, sn], [0, 1, 0], [-sn, 0, c]], atol=1e-12)
    np.testing.assert_allclose(attitude(1.0, beta0), R_UPRIGHT, atol=1e-12)
    for p in np.linspace(0, 1, 21):
        R = attitude(p, beta0)
        nose, up = R[:, 0], R[:, 2]
        assert abs(nose[1]) < 1e-12 and nose[0] < 0  # nose stays in the travel plane, away from the ramp
    # belly-up to belly-down by rolling about the long axis: halfway through, the roof faces sideways
    assert attitude(0.0, beta0)[2, 2] < 0
    assert abs(attitude(0.5, beta0)[1, 2]) == pytest.approx(1.0)


def test_body_rate_matches_attitude_derivative():
    beta0, p, dp, h = -2.1, 0.37, 1.3, 1e-6
    R0, R1 = attitude(p, beta0), attitude(p + dp * h, beta0)
    W = R0.T @ (R1 - R0) / h
    w, _ = body_rate(p, dp, 0.0, beta0)
    np.testing.assert_allclose([W[2, 1], W[0, 2], W[1, 0]], w, atol=1e-4)
    w0, dw = body_rate(p, dp, 0.7, beta0)
    w1, _ = body_rate(p + dp * h, dp + 0.7 * h, 0.7, beta0)
    np.testing.assert_allclose((np.array(w1) - np.array(w0)) / h, dw, atol=1e-4)


def test_leaves_lip_upside_down():
    x = lip_state()
    assert x[1] > 10.0 and x[3] > 0  # high and still climbing
    assert np.degrees(x[4]) < -100  # past vertical: the roof faces the ground


def test_plan_flips_then_lands(plan):
    assert plan.t_flip <= 1.0 + 1e-6
    k = int(np.searchsorted(plan.t, plan.t_flip))
    np.testing.assert_allclose(plan.X[k:, 6], 1.0, atol=1e-6)
    x, y, z, vx, vy, vz = plan.X[-1, :6]
    assert z == pytest.approx(params.Z_TOUCH, abs=0.01)
    assert -0.6 - 1e-6 <= vz <= 1e-6
    assert x <= params.RAMP_X0 - params.RAMP_LAND_GAP + 1e-6
    assert vx < -1.0  # still moving away from the ramp: lands nose-first
    assert abs(y) < 1e-3 and abs(vy) < 1e-3
    assert np.all(plan.X[:, 2] >= params.Z_TOUCH - 1e-6)


def test_plan_starts_at_lip(plan):
    x = lip_state()
    np.testing.assert_allclose(plan.X[0], [x[0], 0, x[1], x[2], 0, x[3], 0, 0], atol=1e-6)
    assert plan.beta0 == pytest.approx(x[4])


def test_plan_budget(plan):
    impulse = float((plan.U[:, :12].sum(axis=1) * np.diff(plan.t)).sum())
    assert impulse < 0.5 * params.IMPULSE_BUDGET


def test_nominal_lands(runs):
    m = touchdown_metrics(runs[("nominal", True)])
    assert _ok(m), m


def test_lands_nose_first_away_from_ramp(runs):
    for s in SCENARIOS:
        nose, v = _touchdown_nose(runs[(s, True)])
        assert _heading_err_deg(nose) <= 10, s
        assert v[0] < -1.0 and np.dot(nose[:2], v[:2]) > 0, s


def test_gust_closed_vs_open(runs):
    mc = touchdown_metrics(runs[("gust", True)])
    mo = touchdown_metrics(runs[("gust", False)])
    assert _ok(mc), mc
    assert (max(mo["roll_err_deg"], mo["pitch_err_deg"]) > 30 or mo["touchdown_vz"] > 3), mo


def test_slow_ramp_lands(runs):
    m = touchdown_metrics(runs[("slow_ramp", True)])
    assert _ok(m), m


def test_open_loop_lands_hard(runs):
    for s in SCENARIOS:
        assert touchdown_metrics(runs[(s, False)])["touchdown_vz"] > 3, s


def test_chassis_never_touches_ramp(runs):
    for key, log in runs.items():
        assert not log.obstacle_contact.any(), key


def test_tank_never_negative(runs):
    for log in runs.values():
        assert np.all(log.tank_remaining >= 0)


def test_tinympc_solves_every_landing_step(runs):
    for s in SCENARIOS:
        log = runs[(s, True)]
        n_ticks = int(np.sum(log.phase[::int(round(params.DT_CTRL / params.DT_SIM))] == "landing"))
        assert len(log.mpc_solver) == pytest.approx(n_ticks / MPC_EVERY, abs=1), s
        assert len(log.mpc_solver) > 30
        assert set(log.mpc_solver) == {"tinympc"}, s


def test_log_roundtrip_keeps_scene(runs, tmp_path):
    log = runs[("nominal", True)]
    p = tmp_path / "log.npz"
    log.save(p)
    q = Log.load(p)
    assert q.scene == "ramp"
    np.testing.assert_array_equal(log.qpos, q.qpos)
    assert list(log.phase) == list(q.phase)


def test_logged_mpc_paths_start_at_the_car(runs):
    log = runs[("nominal", True)]
    i = np.searchsorted(log.t, log.mpc_t)
    np.testing.assert_allclose(log.mpc_paths[:, 0, 1], log.qpos[i, 2], atol=0.05)
