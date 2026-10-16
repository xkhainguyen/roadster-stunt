"""Online controllers: LQR attitude tracking and a TinyMPC landing tracker."""
import numpy as np
import osqp
import scipy.sparse as sp
import tinympc
from scipy.linalg import solve_continuous_are

from roadster import params as P


def vee(S: np.ndarray) -> np.ndarray:
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def attitude_error(R: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    """SO(3) error in the body frame; no angle wrapping issues."""
    return 0.5 * vee(R_des.T @ R - R.T @ R_des)


class AttitudeLQR:
    """Per-axis LQR on the attitude error, computed for a unit-inertia double integrator and
    scaled by each axis' inertia; gyroscopic term cancelled."""

    def __init__(self, inertia: np.ndarray, wn=(8.0, 6.0, 4.0), r: float = 1.0):
        self.J = np.diag(inertia)
        A = np.array([[0.0, 1.0], [0.0, 0.0]])
        B = np.array([[0.0], [1.0]])
        self.K = []
        for w in wn:
            Q = np.diag([w ** 4, 0.0])
            Pm = solve_continuous_are(A, B, Q, np.array([[r]]))
            self.K.append((B.T @ Pm / r).ravel())
        self.K = np.array(self.K)  # rows: [k_angle, k_rate] per axis

    def torque(self, R, w, R_des, w_des, tau_ff) -> np.ndarray:
        e_R = attitude_error(R, R_des)
        e_w = w - R.T @ R_des @ w_des
        alpha = -(self.K[:, 0] * e_R + self.K[:, 1] * e_w)
        return self.J @ alpha + np.cross(w, self.J @ w) + tau_ff


class LandingMPC:
    """Translational tracking MPC, state [y, z, vy, vz], input world [Fy, Fz] in kN internally.
    Solved with TinyMPC. If TinyMPC hits its iteration cap, OSQP solves the same dynamics, costs and
    bounds (terminal cost Q rather than TinyMPC's infinite-horizon cost); if OSQP also fails, the
    clipped TinyMPC iterate is used and solver_used reads "tinympc-maxiter"."""

    KN = 1e3

    def __init__(self, dt: float = 0.02, horizon: int = 40, q=(20.0, 800.0, 10.0, 300.0), r=(0.05, 0.02),
                 rho: float = 1.0, max_iter: int = 500, osqp_max_iter: int = 20000, z_touch: float = P.Z_TOUCH,
                 fy_max: float = 2 * P.F_SIDE_MAX):
        self.dt, self.N = dt, horizon
        m, g = P.MASS, P.G
        self.A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1.0]])
        self.B = self.KN / m * np.array([[0.5 * dt ** 2, 0], [0, 0.5 * dt ** 2], [dt, 0], [0, dt]])
        self.f = np.array([0, -0.5 * g * dt ** 2, 0, -g * dt])
        self.Q, self.R = np.diag(np.asarray(q, dtype=float)), np.diag(np.asarray(r, dtype=float))
        self.u_min = np.array([-fy_max, 0.0]) / self.KN
        self.u_max = np.array([fy_max, 4 * P.F_LIFT_MAX]) / self.KN
        self.x_min = np.array([-1e3, z_touch - 0.3, -1e3, -1e3])  # slack below touchdown keeps ADMM off an active floor
        self.x_max = np.array([1e3, 1e3, 1e3, 1e3])
        self.u_hover = np.array([0.0, m * g / self.KN])
        self.solver_used = "tinympc"
        self.osqp_max_iter = osqp_max_iter

        self.mpc = tinympc.TinyMPC()
        self.mpc.setup(self.A, self.B, self.Q, self.R, self.N, rho=rho, fdyn=self.f,
                       x_min=self.x_min, x_max=self.x_max, u_min=self.u_min, u_max=self.u_max,
                       max_iter=max_iter, abs_pri_tol=1e-3, abs_dua_tol=1e-3)
        self.mpc.set_u_ref(self.u_hover)

    def _ref(self, x_ref):
        x_ref = np.asarray(x_ref, dtype=float)
        return np.tile(x_ref[:, None], (1, self.N)) if x_ref.ndim == 1 else x_ref[:, :self.N]

    def force(self, x: np.ndarray, x_ref: np.ndarray):
        X_ref = self._ref(x_ref)
        self.mpc.set_x0(np.asarray(x, dtype=float))
        self.mpc.set_x_ref(X_ref)
        sol = self.mpc.solve()
        # copy: TinyMPC returns views into its own memory, overwritten by the next solve
        u0, Xs = np.array(sol["controls"], copy=True), np.array(sol["states_all"], copy=True)
        ok = self.mpc._solver.solution.solved and np.all(np.isfinite(u0))
        if ok:
            self.solver_used = "tinympc"
        else:
            fallback = self._osqp(np.asarray(x, dtype=float), X_ref)
            if fallback is not None:
                u0, Xs = fallback
                self.solver_used = "osqp"
            else:
                u0 = np.nan_to_num(u0, nan=0.0)
                self.solver_used = "tinympc-maxiter"
        u0 = np.clip(u0, self.u_min, self.u_max)
        return u0 * self.KN, Xs

    def _osqp(self, x0, X_ref):
        """Sparse QP over [x_0..x_N, u_0..u_{N-1}]; returns None unless OSQP reports a solution."""
        nx, nu, N = 4, 2, self.N
        nX, nU = nx * (N + 1), nu * N
        Qs = [self.Q] * N + [self.Q]
        H = sp.block_diag(Qs + [self.R] * N, format="csc")
        Xr = np.concatenate([X_ref[:, min(k, N - 1)] for k in range(N + 1)])
        qv = -np.concatenate([Q @ Xr[k * nx:(k + 1) * nx] for k, Q in enumerate(Qs)]
                             + [np.tile(self.R @ self.u_hover, N)])
        Ax = sp.kron(sp.eye(N + 1), -sp.eye(nx)) + sp.kron(sp.eye(N + 1, k=-1), sp.csc_matrix(self.A))
        Bu = sp.kron(sp.vstack([sp.csc_matrix((1, N)), sp.eye(N)]), sp.csc_matrix(self.B))
        Aeq = sp.hstack([Ax, Bu])
        beq = np.concatenate([-x0, np.tile(-self.f, N)])
        Aineq = sp.eye(nX + nU)
        lo = np.concatenate([np.tile(self.x_min, N + 1), np.tile(self.u_min, N)])
        hi = np.concatenate([np.tile(self.x_max, N + 1), np.tile(self.u_max, N)])
        Acon = sp.vstack([Aeq, Aineq], format="csc")
        prob = osqp.OSQP()
        prob.setup(H, qv, Acon, np.concatenate([beq, lo]), np.concatenate([beq, hi]), verbose=False,
                   eps_abs=1e-5, eps_rel=1e-5, max_iter=self.osqp_max_iter)
        res = prob.solve()
        if res.info.status not in ("solved", "solved inaccurate") or res.x is None or not np.all(np.isfinite(res.x)):
            return None
        z = res.x
        Xs = z[:nX].reshape(N + 1, nx)
        return z[nX:nX + nu], Xs
