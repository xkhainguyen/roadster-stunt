import numpy as np
import pytest

from roadster import params
from roadster.flight import Log, simulate, touchdown_metrics


@pytest.fixture(scope="module")
def runs():
    return {
        ("nominal", True): simulate("nominal", closed_loop=True),
        ("gust", True): simulate("gust", closed_loop=True),
        ("gust", False): simulate("gust", closed_loop=False),
        ("fast_launch", True): simulate("fast_launch", closed_loop=True),
        ("model_error", True): simulate("model_error", closed_loop=True),
        ("model_error", False): simulate("model_error", closed_loop=False),
        ("gust_model_error", True): simulate("gust_model_error", closed_loop=True),
        ("gust_model_error", False): simulate("gust_model_error", closed_loop=False),
    }


def _ok(m):
    return (m["touchdown_vz"] <= 1.5 and m["roll_err_deg"] <= 10 and m["pitch_err_deg"] <= 10
            and not m["hit_obstacle"])


def test_nominal_lands(runs):
    m = touchdown_metrics(runs[("nominal", True)])
    assert _ok(m), m


def test_gust_closed_vs_open(runs):
    mc = touchdown_metrics(runs[("gust", True)])
    mo = touchdown_metrics(runs[("gust", False)])
    assert _ok(mc), mc
    assert (max(mo["roll_err_deg"], mo["pitch_err_deg"]) > 30 or mo["touchdown_vz"] > 3), mo


def test_fast_launch_lands(runs):
    m = touchdown_metrics(runs[("fast_launch", True)])
    assert _ok(m), m


def test_tank_never_negative(runs):
    for log in runs.values():
        assert np.all(log.tank_remaining >= 0)


def test_log_roundtrip(runs, tmp_path):
    log = runs[("nominal", True)]
    p = tmp_path / "log.npz"
    log.save(p)
    q = Log.load(p)
    np.testing.assert_array_equal(log.qpos, q.qpos)
    np.testing.assert_array_equal(log.thrust, q.thrust)
    assert list(log.phase) == list(q.phase)


def test_landing_is_solved_by_tinympc(runs):
    for key in [("nominal", True), ("gust", True), ("fast_launch", True), ("model_error", True),
                ("gust_model_error", True)]:
        log = runs[key]
        assert len(log.mpc_solver) > 10
        assert set(log.mpc_solver) == {"tinympc"}, key


def test_log_records_solve_times(runs):
    log = runs[("nominal", True)]
    assert len(log.mpc_solve_ms) == len(log.mpc_t) > 10
    assert np.all(log.mpc_solve_ms > 0)


def _fails(m):
    return max(m["roll_err_deg"], m["pitch_err_deg"]) > 30 or m["touchdown_vz"] > 3


@pytest.mark.parametrize("scenario", ["model_error", "gust_model_error"])
def test_feedback_absorbs_model_error(runs, scenario):
    """The real car differs from the controller's model (mass, inertia, CoM, thrust); feedback still lands it."""
    mc = touchdown_metrics(runs[(scenario, True)])
    mo = touchdown_metrics(runs[(scenario, False)])
    assert _ok(mc), mc
    assert _fails(mo), mo


def test_model_error_changes_the_simulated_car_only():
    from roadster.flight import SCENARIOS
    err = SCENARIOS["model_error"]["model_error"]
    assert err["mass"] != 1.0 and err["thrust"] != 1.0 and err["com_y"] != 0.0
