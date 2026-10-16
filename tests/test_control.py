import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from roadster import params
from roadster.control import AttitudeLQR, LandingMPC, attitude_error


def test_attitude_error_wraps():
    R = Rot.from_euler("x", 359, degrees=True).as_matrix()
    Rd = Rot.from_euler("x", 1, degrees=True).as_matrix()
    e = attitude_error(R, Rd)
    assert np.degrees(np.linalg.norm(e)) == pytest.approx(2.0, abs=0.05)


def test_lqr_converges():
    J = np.diag(params.INERTIA)
    ctrl = AttitudeLQR(params.INERTIA)
    R = Rot.from_euler("xy", [20, 20], degrees=True).as_matrix()
    w = np.zeros(3)
    dt = 0.001
    tau_max = 0
    for _ in range(1000):
        tau = ctrl.torque(R, w, np.eye(3), np.zeros(3), np.zeros(3))
        tau_max = max(tau_max, np.abs(tau).max())
        wdot = np.linalg.solve(J, tau - np.cross(w, J @ w))
        w = w + dt * wdot
        R = R @ Rot.from_rotvec(w * dt).as_matrix()
    ang = np.degrees(np.linalg.norm(Rot.from_matrix(R).as_rotvec()))
    assert ang < 1.0
    # roof+lift couple on one axis can make about 2 * 0.85 * (2*F_LIFT + 2*F_ROOF)
    assert tau_max <= 0.85 * 2 * (2 * params.F_LIFT_MAX + 2 * params.F_ROOF_MAX)


def _drop(mpc):
    g = 9.81
    x = np.array([0.0, params.RIDE_Z + 5.0, 0.0, -8.0])
    xref = np.array([0.0, params.RIDE_Z, 0.0, -0.5])
    dt = 0.02
    zmin, forces = np.inf, []
    while x[1] > params.RIDE_Z + 0.02:
        F, _ = mpc.force(x, xref)
        forces.append(F)
        for _ in range(20):
            a = np.array([F[0], F[1] - params.MASS * g]) / params.MASS
            x = x + 0.001 * np.array([x[2], x[3], a[0], a[1]])
            zmin = min(zmin, x[1])
            if x[1] <= params.RIDE_Z + 0.02:
                break
        if len(forces) > 500:
            break
    return x, zmin, np.array(forces)


def test_mpc_lands_softly():
    x, zmin, F = _drop(LandingMPC())
    assert len(F) <= 500, "never touched down"
    assert abs(x[3]) <= 1.5
    assert zmin >= params.RIDE_Z - 0.01


def test_mpc_respects_bounds():
    _, _, F = _drop(LandingMPC())
    assert np.all(F[:, 1] >= -1e-6) and np.all(F[:, 1] <= 4 * params.F_LIFT_MAX + 1e-6)
    assert np.all(np.abs(F[:, 0]) <= 2 * params.F_SIDE_MAX + 1e-6)


def test_both_solvers_failing_returns_bounded_tinympc_answer():
    mpc = LandingMPC(max_iter=1, osqp_max_iter=1)
    F, path = mpc.force(np.array([0.3, 5.0, 1.0, -8.0]), np.array([0.0, params.Z_TOUCH, 0.0, -0.5]))
    assert mpc.solver_used == "tinympc-maxiter"
    assert np.all(np.isfinite(F)) and 0 <= F[1] <= 4 * params.F_LIFT_MAX


def test_mpc_path_survives_next_solve():
    """The returned prediction must be a copy, not a view the next solve overwrites."""
    mpc = LandingMPC()
    ref = np.tile(np.array([0, params.Z_TOUCH, 0, -0.5])[:, None], (1, mpc.N))
    _, X1 = mpc.force(np.array([0.0, 10.0, 0.0, -5.0]), ref)
    first = X1.copy()
    mpc.force(np.array([0.0, 3.0, 0.0, -2.0]), ref)
    np.testing.assert_array_equal(X1, first)
    assert X1[0, 1] == pytest.approx(10.0)


def test_thrust_vector_attitude_tilts_toward_the_force():
    from roadster.flight import landing_attitude
    from scipy.spatial.transform import Rotation as Rot
    level = landing_attitude(np.array([0.0, 20000.0]))
    np.testing.assert_allclose(level, np.eye(3), atol=1e-9)
    R = landing_attitude(np.array([-5000.0, 20000.0]))  # need force toward -y
    assert (R @ np.array([0, 0, 1.0]))[1] < 0  # body z (lift direction) leans toward -y
    big = landing_attitude(np.array([-50000.0, 10000.0]))
    tilt = np.degrees(np.arccos((big @ np.array([0, 0, 1.0]))[2]))
    assert tilt == pytest.approx(25.0, abs=0.1)  # capped


def test_landing_tilt_fades_out_near_touchdown():
    from roadster.flight import landing_attitude
    high = landing_attitude(np.array([-50000.0, 10000.0]), height=2.0)
    low = landing_attitude(np.array([-50000.0, 10000.0]), height=0.25)
    tilt = lambda R: np.degrees(np.arccos((R @ np.array([0, 0, 1.0]))[2]))
    assert tilt(high) == pytest.approx(25.0, abs=0.1)
    assert tilt(low) == pytest.approx(25.0 * 0.25, abs=0.1)
