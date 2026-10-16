import numpy as np
import pytest

from roadster import params
from roadster.model import load
from roadster.thrusters import AirTank, Valves, allocate, allocation_matrix


@pytest.fixture(scope="module")
def A_fmax():
    m, _ = load()
    return allocation_matrix(m), m.actuator_ctrlrange[:, 1].copy()


def test_allocate_pure_roll(A_fmax):
    A, fmax = A_fmax
    w = np.array([0, 0, 0, 4000.0, 0, 0])
    f = allocate(w, A, fmax)
    assert np.all(f >= -1e-9) and np.all(f <= fmax + 1e-9)
    np.testing.assert_allclose(A @ f, w, atol=0.01 * 4000)


def test_allocate_infeasible_is_clipped(A_fmax):
    A, fmax = A_fmax
    f = allocate(np.array([0, 0, 1e6, 0, 0, 0]), A, fmax)
    assert not np.any(np.isnan(f))
    assert np.all(f >= -1e-9) and np.all(f <= fmax + 1e-9)
    m, _ = load()
    lifts = [m.actuator(f"lift_{n}").id for n in ("fl", "fr", "rl", "rr")]
    np.testing.assert_allclose(f[lifts], params.F_LIFT_MAX, rtol=1e-3)


def test_sigma_delta_average():
    fmax = np.array([1000.0, 2000.0])
    v = Valves(fmax)
    outs = np.array([v.step(0.3 * fmax) for _ in range(100)])
    assert np.all(np.isin(outs[:, 0], [0, 1000.0])) and np.all(np.isin(outs[:, 1], [0, 2000.0]))
    np.testing.assert_allclose(outs.mean(axis=0), 0.3 * fmax, rtol=0.02)


def test_tank_runs_dry():
    tank = AirTank(budget=1000.0)
    dt = 0.001
    f = np.full(4, 15000.0)
    total = 0.0
    out = None
    for _ in range(100):
        out = tank.draw(f, dt)
        total += out.sum() * dt
    assert np.all(out == 0)
    assert tank.empty_at is not None
    assert total == pytest.approx(1000.0, abs=60.0 * dt * 1000)
    assert tank.remaining == 0
