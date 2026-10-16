"""Numbers shown or spoken in the video, computed from the plan and logs so they cannot go stale."""
import numpy as np

from roadster import params as P


def air_fraction(plan) -> float:
    return float((plan.U.sum(axis=1) * np.diff(plan.t)).sum()) / P.IMPULSE_BUDGET


def burn_end(plan) -> float:
    """Time the launch burn stops: first knot with thrust below 1% of peak (IPOPT never hits exact zero)."""
    total = plan.U.sum(axis=1)
    return float(plan.t[int(np.argmax(total < 0.01 * total.max()))])


def solve_ms(logs) -> tuple[float, float]:
    allt = np.concatenate([log.mpc_solve_ms for log in logs])
    return float(np.median(allt)), float(allt.max())


def semi_touchdown(log) -> tuple[float, float]:
    """Touchdown speed [m/s] and sideways offset from the trailer centerline [cm] for the semi stunt."""
    from roadster.flight import touchdown_metrics
    m = touchdown_metrics(log)
    return m["touchdown_vz"], 100 * abs(m["touchdown_y"])


def apex_and_airtime(log) -> tuple[float, float]:
    tp = log.t - log.launch_t
    flight = np.isin(log.phase, ["flight", "landing"])
    apex = float(np.max(np.where(flight, log.qpos[:, 2], -np.inf)))
    land = int(np.flatnonzero(log.phase == "rollout")[0])
    return apex, float(tp[land])
