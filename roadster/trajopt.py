"""Offline kickflip plan: direct multiple shooting on a planar roll model (CasADi + IPOPT).

State  X = [y, z, vy, vz, phi, omega]   (x decoupled: x = VX0 * t)
Input  U = [lift_L, lift_R, roof_L, roof_R]  (front + rear thruster on each side, newtons)
"""
from dataclasses import dataclass, field
from pathlib import Path

import casadi as ca
import numpy as np

from roadster import params as P

U_MAX = np.array([2 * P.F_LIFT_MAX, 2 * P.F_LIFT_MAX, 2 * P.F_ROOF_MAX, 2 * P.F_ROOF_MAX])


def reduced_wrench(u):
    """Body-z force and roll torque from the four side-summed thruster groups."""
    Fz = u[0] + u[1] - u[2] - u[3]
    tx = P.THRUSTER_Y * (u[0] - u[1]) + P.THRUSTER_Y * (u[3] - u[2])
    return Fz, tx


def _f(x, u):
    Fz, tx = reduced_wrench(u)
    s, c = ca.sin(x[4]), ca.cos(x[4])
    return ca.vertcat(x[2], x[3], -s * Fz / P.MASS, c * Fz / P.MASS - P.G, x[5], tx / P.INERTIA[0])


def _rk4(x, u, h):
    k1 = _f(x, u)
    k2 = _f(x + h / 2 * k1, u)
    k3 = _f(x + h / 2 * k2, u)
    k4 = _f(x + h * k3, u)
    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def _smooth_abs(v, eps=1e-4):
    """|v| that is differentiable at 0 and exact (to ~eps) away from it; zero at v=0."""
    return ca.sqrt(v ** 2 + eps) - np.sqrt(eps)


def _bump(x, a, b, k=8.0):
    return 1 / (1 + ca.exp(-k * (x - a))) * 1 / (1 + ca.exp(-k * (b - x)))


@dataclass
class Plan:
    t: np.ndarray
    X: np.ndarray
    U: np.ndarray
    iterates: list = field(default_factory=list)
    iterates_T: list = field(default_factory=list)

    @property
    def T(self) -> float:
        return float(self.t[-1])

    def save(self, path):
        np.savez(path, t=self.t, X=self.X, U=self.U, iterates=np.array(self.iterates),
                 iterates_T=np.array(self.iterates_T))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["t"], z["X"], z["U"], list(z["iterates"]), list(z["iterates_T"]))

    def at(self, t: float):
        """State by linear interpolation, input by zero-order hold."""
        t = float(np.clip(t, self.t[0], self.t[-1]))
        x = np.array([np.interp(t, self.t, self.X[:, i]) for i in range(self.X.shape[1])])
        k = min(int(np.searchsorted(self.t, t, side="right")) - 1, len(self.U) - 1)
        return x, self.U[max(k, 0)].copy()


def _plan_roll(N, z_end, path_constraints, apex_guess, verbose=False, y0=0.0, t_min=1.8) -> Plan:
    """Shared solver: from the road, a full roll, ending upright at height z_end with a soft descent.
    path_constraints(opti, X, h, T) adds the stunt's clearance constraints."""
    opti = ca.Opti()
    X = opti.variable(6, N + 1)
    Ukn = opti.variable(4, N)  # kN for scaling
    T = opti.variable()
    h = T / N

    for k in range(N):
        opti.subject_to(X[:, k + 1] == _rk4(X[:, k], 1e3 * Ukn[:, k], h))

    z0 = P.RIDE_Z
    opti.subject_to(X[:, 0] == np.array([y0, z0, 0, 0, 0, 0]))
    opti.subject_to(X[0, N] == 0)
    opti.subject_to(X[1, N] == z_end)
    opti.subject_to(X[2, N] == 0)
    opti.subject_to(opti.bounded(-0.6, X[3, N], 0.0))
    opti.subject_to(X[4, N] == 2 * np.pi)
    opti.subject_to(X[5, N] == 0)
    opti.subject_to(opti.bounded(t_min, T, 3.5))
    opti.subject_to(opti.bounded(0, Ukn, U_MAX[:, None] / 1e3))
    path_constraints(opti, X, h, T)

    cost = ca.sum1(ca.sum2(Ukn)) * h + 1e-3 * ca.sumsqr(Ukn[:, 1:] - Ukn[:, :-1])
    opti.minimize(cost)

    # warm start: ballistic arc with a linear roll ramp
    T0 = 2.6
    tt = np.linspace(0, T0, N + 1)
    opti.set_initial(T, T0)
    opti.set_initial(X[1, :], z0 + (z_end - z0) * tt / T0 + 4 * apex_guess * tt / T0 * (1 - tt / T0))
    opti.set_initial(X[4, :], 2 * np.pi * tt / T0)
    opti.set_initial(X[0, :], y0 * (1 - tt / T0))
    opti.set_initial(Ukn, np.tile(np.array([10, 10, 1, 1])[:, None], (1, N)))

    iterates, iterates_T = [], []

    def _record(i):
        iterates.append(np.array(opti.debug.value(X)).T.copy())
        iterates_T.append(float(opti.debug.value(T)))

    opti.callback(_record)
    opti.solver("ipopt", {"print_time": False}, {"print_level": 5 if verbose else 0, "max_iter": 3000,
                                                 "tol": 1e-8, "acceptable_tol": 1e-6})
    sol = opti.solve()
    Tn = float(sol.value(T))
    return Plan(t=np.linspace(0, Tn, N + 1), X=np.array(sol.value(X)).T, U=1e3 * np.array(sol.value(Ukn)).T,
                iterates=iterates, iterates_T=iterates_T)


def plan_kickflip(N: int = 80, verbose: bool = False) -> Plan:
    def over_the_row(opti, X, h, T):
        a, b = P.ROW_START - 1.5, P.ROW_START + P.ROW_LEN + 1.5
        h_req = P.ROW_HEIGHT + P.CLEARANCE_RADIUS - P.RIDE_Z
        for k in range(N + 1):
            opti.subject_to(X[1, k] >= P.RIDE_Z + h_req * _bump(P.VX0 * k * h, a, b))
    return _plan_roll(N, P.Z_TOUCH, over_the_row, apex_guess=7.5, verbose=verbose)


def plan_semi(N: int = 80, verbose: bool = False) -> Plan:
    """My own stunt: from the next lane, a kickflip sideways onto the flatbed of a Semi driving alongside.
    The sideways move comes from the roll tilting the lift jets; y is measured from the trailer centerline."""
    z_deck = P.DECK_H + P.Z_TOUCH

    def over_the_trailer(opti, X, h, T):
        # height the car needs above the deck depends on its roll: the lowest tire corner sits
        # Z_TOUCH*|cos| + half_width*|sin| below the center (touchdown height when upright)
        reach = P.CLEARANCE_RADIUS
        a, b = -(P.DECK_W / 2 + reach), P.DECK_W / 2 + reach
        # knot 0 is pinned to the road by the start condition, and the smooth step is never exactly zero, so
        # constraining it would demand ~1e-5 m of height the car cannot have; skip it
        for k in range(1, N + 1):
            phi = X[4, k]
            need = P.Z_TOUCH * _smooth_abs(ca.cos(phi)) + 1.05 * _smooth_abs(ca.sin(phi))  # lowest tire corner
            opti.subject_to(X[1, k] >= P.RIDE_Z + (P.DECK_H + need - P.RIDE_Z) * _bump(X[0, k], a, b))
        # rocket-style approach: line up over the centerline first, then come straight down
        for k in range(int(np.ceil(0.8 * N)), N + 1):
            opti.subject_to(opti.bounded(-0.1, X[0, k], 0.1))
            opti.subject_to(opti.bounded(-0.3, X[2, k], 0.3))
    # at least SEMI_MIN_FLIGHT in the air: the cheapest plan is a short hop that leaves feedback no time to correct
    return _plan_roll(N, z_deck, over_the_trailer, apex_guess=5.0, verbose=verbose, y0=P.SEMI_LANE_Y,
                      t_min=P.SEMI_MIN_FLIGHT)


if __name__ == "__main__":
    plan = plan_kickflip(verbose=False)
    out = Path(__file__).resolve().parents[1] / "data"
    out.mkdir(exist_ok=True)
    plan.save(out / "kickflip_plan.npz")
    imp = float((plan.U.sum(axis=1) * np.diff(plan.t)).sum())
    print(f"T={plan.T:.3f}s apex={plan.X[:, 1].max():.2f}m impulse={imp / 1e3:.1f}kNs iterates={len(plan.iterates)}")
