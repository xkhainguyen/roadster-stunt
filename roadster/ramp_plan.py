"""Offline ramp plan: from the lip state, roll belly-up to belly-down about the long axis while the nose comes
down to level and keeps pointing away from the ramp, then fall upright and brake (CasADi + IPOPT).

The attitude follows a fixed path R(p) = Ry(beta0 + dbeta p) Rx(pi p), p from 0 to 1, with dbeta = -pi - beta0:
the nose stays in the vertical x-z plane on the far side of the ramp and drops from beta0 to level, while the
car rolls 180 deg about its long axis. R(1) = Rz(pi): upright, nose along -x. The optimizer picks the timing
p(t) and all 12 thruster forces, so the torque that path needs and the translation it causes are planned
together in 6-DoF.

State  X = [x, y, z, vx, vy, vz, p, dp]
Input  U = [12 thruster forces (newtons, actuator order), ddp]
"""
from dataclasses import dataclass

import casadi as ca
import numpy as np

from roadster import params as P
from roadster.model import load
from roadster.thrusters import allocation_matrix
from roadster.trajopt import Plan

PLAN_MARGIN = 0.8  # the plan uses at most this fraction of each thruster so the feedback has room
R_UPRIGHT = np.diag([-1.0, -1.0, 1.0])  # Rz(pi): upright, nose along -x


def _Ry(a, lib=np):
    c, s = lib.cos(a), lib.sin(a)
    return [[c, 0, s], [0, 1, 0], [-s, 0, c]]


def _Rx(a, lib=np):
    c, s = lib.cos(a), lib.sin(a)
    return [[1, 0, 0], [0, c, -s], [0, s, c]]


def attitude(p: float, beta0: float) -> np.ndarray:
    return np.array(_Ry(beta0 + (-np.pi - beta0) * p)) @ np.array(_Rx(np.pi * p))


def body_rate(p, dp, ddp, beta0, lib=np):
    """Body angular velocity and acceleration along the path: w = dp * a(p), dw = ddp * a(p) + dp^2 * a'(p)."""
    db = -np.pi - beta0
    c, s = lib.cos(np.pi * p), lib.sin(np.pi * p)
    a = [np.pi, db * c, -db * s]
    da = [0.0, -db * np.pi * s, -db * np.pi * c]
    w = [dp * ai for ai in a]
    dw = [ddp * ai + dp ** 2 * dai for ai, dai in zip(a, da)]
    return w, dw


@dataclass
class RampPlan(Plan):
    beta0: float = 0.0

    @property
    def t_flip(self) -> float:
        """When the car is upright and the landing MPC can take over."""
        return float(self.t[np.argmax(self.X[:, 6] >= 1.0 - 1e-6)])

    def save(self, path):
        np.savez(path, t=self.t, X=self.X, U=self.U, iterates=np.array(self.iterates),
                 iterates_T=np.array(self.iterates_T), beta0=self.beta0)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["t"], z["X"], z["U"], list(z["iterates"]), list(z["iterates_T"]), float(z["beta0"]))


def plan_ramp(x_lip, N1: int = 40, N2: int = 50, verbose: bool = False) -> RampPlan:
    """x_lip is the planar lip state [x, z, vx, vz, beta, omega] from roadster.ramp.lip_state."""
    m, _ = load("ramp")
    A = allocation_matrix(m)
    fmax = m.actuator_ctrlrange[:, 1].copy()
    J = np.diag(P.INERTIA)
    beta0 = float(x_lip[4])
    N = N1 + N2

    def mat(rows):
        return ca.vertcat(*[ca.horzcat(*r) for r in rows])

    def f(x, u):
        R = ca.mtimes(mat(_Ry(beta0 + (-np.pi - beta0) * x[6], ca)), mat(_Rx(np.pi * x[6], ca)))
        F_world = ca.mtimes(R, ca.mtimes(A[:3], u[:12]))
        acc = F_world / P.MASS - ca.vertcat(0, 0, P.G)
        return ca.vertcat(x[3:6], acc, x[7], u[12])

    def rk4(x, u, h):
        k1 = f(x, u)
        k2 = f(x + h / 2 * k1, u)
        k3 = f(x + h / 2 * k2, u)
        k4 = f(x + h * k3, u)
        return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    opti = ca.Opti()
    X = opti.variable(8, N + 1)
    Fkn = opti.variable(12, N)  # kN for scaling
    V = opti.variable(1, N)     # ddp
    T1, T2 = opti.variable(), opti.variable()
    h1, h2 = T1 / N1, T2 / N2
    for k in range(N):
        u = ca.vertcat(1e3 * Fkn[:, k], V[k])
        opti.subject_to(X[:, k + 1] == rk4(X[:, k], u, h1 if k < N1 else h2))
        # thrusters must make the torque the path needs (gyroscopic term included)
        w, dw = body_rate(X[6, k], X[7, k], V[k], beta0, ca)
        w, dw = ca.vertcat(*w), ca.vertcat(*dw)
        tau = ca.mtimes(J, dw) + ca.cross(w, ca.mtimes(J, w))
        opti.subject_to(ca.mtimes(A[3:], 1e3 * Fkn[:, k]) == tau)

    x0, z0, vx0, vz0 = (float(v) for v in x_lip[:4])
    opti.subject_to(X[:, 0] == np.array([x0, 0, z0, vx0, 0, vz0, 0, 0]))
    opti.subject_to(X[6, N1] == 1)
    opti.subject_to(X[7, N1] == 0)
    opti.subject_to(V[N1:] == 0)
    opti.subject_to(X[0, N] <= P.RAMP_X0 - P.RAMP_LAND_GAP)
    opti.subject_to(X[1, N] == 0)
    opti.subject_to(X[2, N] == P.Z_TOUCH)
    opti.subject_to(X[4, N] == 0)
    opti.subject_to(opti.bounded(-0.6, X[5, N], 0.0))
    opti.subject_to(opti.bounded(0.6, T1, 1.0))
    opti.subject_to(opti.bounded(0.5, T2, 3.0))
    opti.subject_to(opti.bounded(0, Fkn, PLAN_MARGIN * fmax[:, None] / 1e3))
    opti.subject_to(X[2, :] >= P.Z_TOUCH)

    h = ca.horzcat(ca.repmat(h1, 1, N1), ca.repmat(h2, 1, N2))
    cost = (ca.sum1(ca.sum2(Fkn * ca.repmat(h, 12, 1))) + 1e-3 * ca.sumsqr(Fkn[:, 1:] - Fkn[:, :-1])
            + 1e-3 * ca.sumsqr(V))
    opti.minimize(cost)

    # warm start: ballistic from the lip, smooth p ramp during the flip
    T10, T20 = 0.9, 1.6
    tt = np.concatenate([np.linspace(0, T10, N1 + 1), T10 + np.linspace(0, T20, N2 + 1)[1:]])
    s = np.clip(tt / T10, 0, 1)
    opti.set_initial(T1, T10)
    opti.set_initial(T2, T20)
    opti.set_initial(X[0, :], x0 + vx0 * tt)
    opti.set_initial(X[2, :], np.maximum(z0 + vz0 * tt - 0.5 * P.G * tt ** 2, P.Z_TOUCH))
    opti.set_initial(X[3, :], vx0)
    opti.set_initial(X[6, :], 3 * s ** 2 - 2 * s ** 3)
    opti.set_initial(X[7, :], np.where(tt < T10, 6 * s * (1 - s) / T10, 0.0))
    opti.set_initial(Fkn, 1.0)

    iterates, iterates_T = [], []

    def _record(i):
        iterates.append(np.array(opti.debug.value(X)).T.copy())
        iterates_T.append(float(opti.debug.value(T1) + opti.debug.value(T2)))

    opti.callback(_record)
    opti.solver("ipopt", {"print_time": False}, {"print_level": 5 if verbose else 0, "max_iter": 3000,
                                                 "tol": 1e-8, "acceptable_tol": 1e-6})
    sol = opti.solve()
    t1, t2 = float(sol.value(T1)), float(sol.value(T2))
    t = np.concatenate([np.linspace(0, t1, N1 + 1), t1 + np.linspace(0, t2, N2 + 1)[1:]])
    U = np.hstack([1e3 * np.array(sol.value(Fkn)).T, np.array(sol.value(V)).reshape(-1, 1)])
    return RampPlan(t=t, X=np.array(sol.value(X)).T, U=U, iterates=iterates, iterates_T=iterates_T, beta0=beta0)
