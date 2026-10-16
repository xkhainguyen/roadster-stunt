"""Breakable car for the no-feedback crash: parts are welded to the chassis and the welds release on a hard impact."""
import numpy as np
import pytest

from roadster import params as P
from roadster.flight import BREAK_SPEED, simulate, touchdown_metrics
from roadster.model import load


def test_breakable_model_keeps_total_mass():
    m, _ = load("kickflip:breakable")
    pieces = [i for i in range(m.nbody) if m.body(i).name.startswith("piece_")]
    assert len(pieces) == 7
    total = m.body("car").subtreemass[0] + sum(m.body(i).subtreemass[0] for i in pieces)
    assert total == pytest.approx(P.MASS, rel=1e-6)


@pytest.fixture(scope="module")
def crash():
    return simulate("gust_model_error", closed_loop=False, breakable=True)


def test_hard_impact_breaks_the_car(crash):
    assert np.isfinite(crash.broken_at)
    m, d = load("kickflip:breakable")
    i = np.searchsorted(crash.t, crash.broken_at + 1.0)
    d.qpos[:] = crash.qpos[min(i, len(crash.t) - 1)]
    import mujoco
    mujoco.mj_forward(m, d)
    car = d.body("car").xpos
    far = [np.linalg.norm(d.body(f"piece_{n}").xpos - car) for n in ("hood", "cabin", "rear")]
    assert max(far) > 1.0


def test_flight_matches_unbreakable_run_until_impact(crash):
    """Welds are slightly compliant and panel inertia is approximate, so allow a small drift; the impact speed
    shown in the video comes from the unbreakable run and must agree."""
    plain = simulate("gust_model_error", closed_loop=False)
    k = min(np.searchsorted(crash.t, crash.broken_at), len(plain.t)) - 1
    assert np.max(np.abs(crash.qpos[:k, :3] - plain.qpos[:k, :3])) < 0.25
    assert touchdown_metrics(crash)["touchdown_vz"] == pytest.approx(touchdown_metrics(plain)["touchdown_vz"], abs=0.2)


def test_soft_landing_stays_in_one_piece():
    log = simulate("nominal", closed_loop=True, breakable=True)
    assert not np.isfinite(log.broken_at)
    assert touchdown_metrics(log)["touchdown_vz"] < BREAK_SPEED


def test_loose_parts_stay_above_the_floor(crash):
    """After the break the parts must collide with the ground, not fall through it."""
    import mujoco
    m, d = load("kickflip:breakable")
    for i in np.flatnonzero(crash.t > crash.broken_at)[::200]:
        d.qpos[:] = crash.qpos[i]
        mujoco.mj_kinematics(m, d)
        for n in ("hood", "cabin", "rear", "wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"):
            assert d.body(f"piece_{n}").xpos[2] > -0.05, (n, crash.t[i])


def test_chassis_rests_on_its_visible_slab(crash):
    import mujoco
    m, d = load("kickflip:breakable")
    d.qpos[:] = crash.qpos[-1]
    mujoco.mj_kinematics(m, d)
    R = d.body("car").xmat.reshape(3, 3)
    corners = [d.body("car").xpos + R @ np.array([sx * 2.15, sy * 0.98, -0.08 + sz * 0.20])
               for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    assert min(c[2] for c in corners) < 0.05  # the slab touches the ground, it does not float


def test_chassis_never_sinks_into_the_floor(crash):
    import mujoco
    m, d = load("kickflip:breakable")
    for i in np.flatnonzero(crash.t > crash.broken_at)[::100]:
        d.qpos[:] = crash.qpos[i]
        mujoco.mj_kinematics(m, d)
        R = d.body("car").xmat.reshape(3, 3)
        lowest = min((d.body("car").xpos + R @ np.array([sx * 2.15, sy * 0.98, -0.08 + sz * 0.20]))[2]
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1))
        assert lowest > -0.1, (crash.t[i], lowest)
