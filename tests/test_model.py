import mujoco
import numpy as np
import pytest

from roadster import params
from roadster.model import load


def _rp_deg(d):
    q = d.body("car").xquat
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    R = R.reshape(3, 3)
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    pitch = np.degrees(np.arcsin(-np.clip(R[2, 0], -1, 1)))
    return roll, pitch


def _settle(m, d, seconds):
    zs = []
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
        zs.append(d.body("car").xipos[2])
    return np.array(zs)


def test_rests_on_wheels():
    m, d = load()
    zs = _settle(m, d, 2.0)
    tail = zs[-int(0.5 / m.opt.timestep):]
    assert tail.max() - tail.min() < 0.01
    roll, pitch = _rp_deg(d)
    assert abs(roll) < 1 and abs(pitch) < 1
    assert abs(tail[-1] - params.RIDE_Z) < 0.02


def test_hover_thrust_lifts():
    m, d = load()
    _settle(m, d, 1.0)
    for n in ("fl", "fr", "rl", "rr"):
        d.actuator(f"lift_{n}").ctrl = params.F_LIFT_MAX
    z0 = d.body("car").xipos[2]
    T = 0.5
    # skip the first 0.15 s while the suspension unloads
    for _ in range(int(0.15 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    z1, v1 = d.body("car").xipos[2], d.qvel[2]
    for _ in range(int((T - 0.15) / m.opt.timestep)):
        mujoco.mj_step(m, d)
    v2 = d.qvel[2]
    a = (v2 - v1) / (T - 0.15)
    expected = 4 * params.F_LIFT_MAX / params.MASS - 9.81
    assert d.body("car").xipos[2] > z0 + 0.5
    assert a == pytest.approx(expected, rel=0.10)


def test_roll_couple_pure_torque():
    m, d = load()
    m.opt.gravity[:] = 0
    d.qpos[2] += 5.0  # well clear of the ground
    mujoco.mj_forward(m, d)
    F = 5000.0
    for n in ("fl", "rl"):
        d.actuator(f"lift_{n}").ctrl = F
    for n in ("fr", "rr"):
        d.actuator(f"roof_{n}").ctrl = F
    for _ in range(int(0.1 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    mujoco.mj_subtreeVel(m, d)
    lin_acc = np.linalg.norm(d.subtree_linvel[m.body("car").id]) / 0.1
    assert lin_acc < 0.1
    # body-frame roll rate (free joint angular velocity is in the body frame)
    assert d.qvel[3] > 0


def test_plumes_are_visual_only():
    m, d = load()
    names = [m.geom(i).name for i in range(m.ngeom)]
    plumes = [n for n in names if n.startswith("plume_")]
    assert len(plumes) == m.nu
    for n in plumes:
        g = m.geom(n)
        assert g.contype[0] == 0 and g.conaffinity[0] == 0
    # plumes live on the car body; total car mass must equal MASS (no plume mass)
    assert m.body("car").subtreemass[0] == pytest.approx(params.MASS, rel=1e-3)


def test_touch_height_matches_wheels():
    """In free fall the suspension relaxes; Z_TOUCH is the chassis height where the tires meet the ground."""
    m, d = load()
    m.opt.gravity[:] = 0
    d.qpos[2] = params.Z_TOUCH + 0.005
    mujoco.mj_forward(m, d)
    for _ in range(200):
        mujoco.mj_step(m, d)
        d.qpos[2] = params.Z_TOUCH + 0.005
        d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    tire_bottom = min(d.geom(f"tire_{n}").xpos[2] for n in ("fl", "fr", "rl", "rr")) - params.WHEEL_RADIUS
    assert tire_bottom == pytest.approx(0.005, abs=0.01)
