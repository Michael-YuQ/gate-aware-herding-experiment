# CBF/QP herding baseline: safety-filtered push controller using Control Barrier Functions
# Uses OSQP with fixed dense sparsity structure (all entries nonzero) for fast update().

import numpy as np
import scipy.sparse as sp
import osqp
from ..sim.collision import point_segment_distance


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


class CBFQPController:
    """
    Safety-filtered herding controller using Control Barrier Functions and QP (OSQP).

    Constraint rows (in order):
      - n herding CBF constraints (robots within R_herd of target)
      - n*(n-1)//2 inter-robot collision CBF constraints
      - n * n_walls wall-avoidance CBF constraints
      - 2n box constraints (velocity bounds, diagonal identity block)

    The CBF rows use a fully dense structure (eps-nonzero placeholder) so that
    OSQP's update_data_mat can replace values without changing the sparsity pattern.
    The box rows use a diagonal identity (only nonzero on diagonal).
    """

    _EPS = 1e-15  # placeholder for structural nonzero that is logically zero

    def __init__(self, world, v_robot,
                 d_push=0.5, R_herd=2.0, alpha_herd=1.0,
                 alpha_col=2.0, alpha_wall=2.0,
                 R_body=0.05, d_wall_safe=0.15):
        self.world = world
        self.v_robot = float(v_robot)
        self.d_push = float(d_push)
        self.R_herd = float(R_herd)
        self.alpha_herd = float(alpha_herd)
        self.alpha_col = float(alpha_col)
        self.alpha_wall = float(alpha_wall)
        self.R_body = float(R_body)
        self.d_wall_safe = float(d_wall_safe)
        self._prob = None
        self._n_robots = None

    def _n_walls(self):
        return len(self.world.wall_segments)

    def _setup(self, n):
        """Initialize OSQP with fixed structure for n robots."""
        dim = 2 * n
        nw = self._n_walls()
        n_herd = n
        n_col = n * (n - 1) // 2
        n_wall = n * nw
        n_cbf = n_herd + n_col + n_wall
        n_box = dim
        n_rows = n_cbf + n_box

        P = sp.eye(dim, format='csc') * 2.0
        q = np.zeros(dim)

        # CBF rows: dense (dim columns each) with eps placeholder
        # Box rows: identity (only diagonal)
        # Build A as block matrix
        A_cbf = np.full((n_cbf, dim), self._EPS)
        A_box = np.eye(dim)
        A_dense = np.vstack([A_cbf, A_box])
        A_sp = sp.csc_matrix(A_dense)

        l = np.full(n_rows, -1e9)
        u = np.full(n_rows, 1e9)
        u[n_cbf:] = self.v_robot
        l[n_cbf:] = -self.v_robot

        prob = osqp.OSQP()
        prob.setup(P, q, A_sp, l, u,
                   warm_starting=True,
                   verbose=False,
                   eps_abs=1e-4, eps_rel=1e-4,
                   max_iter=2000,
                   adaptive_rho=True,
                   polish=False)

        self._prob = prob
        self._n_robots = n
        self._dim = dim
        self._n_cbf = n_cbf
        self._n_box = n_box
        self._n_rows = n_rows
        self._row_offsets = (0, n_herd, n_herd + n_col, n_cbf, n_rows)
        # OSQP 1.x requires explicit data indices when updating sparse matrix values.
        self._Ax_idx = np.arange(A_sp.nnz, dtype=np.int32)

    def _build_qp_data(self, n, robot_positions, target_pos, u_des_flat):
        """
        Build the per-step QP matrices and vectors for the current robot/target state.
        """
        dim = self._dim
        n_cbf = self._n_cbf
        r0, r_col, r_wall, r_box, _ = self._row_offsets

        A_cbf = np.full((n_cbf, dim), self._EPS)
        l = np.full(self._n_rows, -1e9)
        u_vec = np.full(self._n_rows, 1e9)
        u_vec[r_box:] = self.v_robot
        l[r_box:] = -self.v_robot

        for i in range(n):
            ri = robot_positions[i]
            diff_si = target_pos - ri
            h_herd = self.R_herd ** 2 - float(np.dot(diff_si, diff_si))
            row = r0 + i
            A_cbf[row, 2 * i: 2 * i + 2] = 2.0 * diff_si
            l[row] = -self.alpha_herd * h_herd

        crow = r_col
        for i in range(n):
            for j in range(i + 1, n):
                ri = robot_positions[i]
                rj = robot_positions[j]
                diff_ij = ri - rj
                dist_sq = float(np.dot(diff_ij, diff_ij))
                h_col = dist_sq - (2.0 * self.R_body) ** 2
                A_cbf[crow, 2 * i: 2 * i + 2] = 2.0 * diff_ij
                A_cbf[crow, 2 * j: 2 * j + 2] = -2.0 * diff_ij
                l[crow] = -self.alpha_col * h_col
                crow += 1

        wrow = r_wall
        for i in range(n):
            ri = robot_positions[i]
            for seg_p1, seg_p2 in self.world.wall_segments:
                d = point_segment_distance(ri, seg_p1, seg_p2)
                h_wall = d ** 2 - self.d_wall_safe ** 2
                seg = np.asarray(seg_p2, dtype=float) - np.asarray(seg_p1, dtype=float)
                seg_len = np.linalg.norm(seg)
                if seg_len >= 1e-12:
                    seg_unit = seg / seg_len
                    t_proj = float(np.dot(ri - np.asarray(seg_p1, dtype=float), seg_unit))
                    t_proj = max(0.0, min(seg_len, t_proj))
                    closest = np.asarray(seg_p1, dtype=float) + t_proj * seg_unit
                    diff_wall = ri - closest
                    dw = np.linalg.norm(diff_wall)
                    if dw >= 1e-12:
                        A_cbf[wrow, 2 * i: 2 * i + 2] = 2.0 * diff_wall
                l[wrow] = -self.alpha_wall * h_wall
                wrow += 1

        # Build full A (cbf dense + box identity) and extract CSC data in correct order
        A_box = np.eye(dim)
        A_full = np.vstack([A_cbf, A_box])
        A_sp_new = sp.csc_matrix(A_full)
        q = -2.0 * u_des_flat
        return q, A_sp_new, l, u_vec

    def compute(self, robot_positions, target_pos):
        """
        Compute safe velocity commands for each robot.
        Returns list of np.ndarray (dx, dy) per robot.
        """
        target_pos = np.asarray(target_pos, dtype=float)
        robot_positions = [np.asarray(r, dtype=float) for r in robot_positions]
        n = len(robot_positions)

        if self._prob is None or self._n_robots != n:
            self._setup(n)

        entrance_center, _ = self.world.nearest_entrance(target_pos)
        if entrance_center is None:
            entrance_center = np.zeros(2)

        direction = _normalize(target_pos - entrance_center)
        push_pos = target_pos + self.d_push * direction

        u_des = []
        for i in range(n):
            diff = push_pos - robot_positions[i]
            u_des.append(self.v_robot * _normalize(diff) if np.linalg.norm(diff) >= 1e-6 else np.zeros(2))
        u_des_flat = np.concatenate(u_des)

        q, A_sp, l, u_vec = self._build_qp_data(n, robot_positions, target_pos, u_des_flat)

        try:
            prob = osqp.OSQP()
            P = sp.eye(self._dim, format='csc') * 2.0
            prob.setup(
                P, q, A_sp, l, u_vec,
                warm_starting=False,
                verbose=False,
                eps_abs=1e-4, eps_rel=1e-4,
                max_iter=2000,
                adaptive_rho=True,
                polish=False,
            )
            res = prob.solve()
            if res.info.status in ('solved', 'solved_inaccurate') and res.x is not None:
                u_flat = res.x.copy()
            else:
                u_flat = u_des_flat.copy()
        except Exception:
            u_flat = u_des_flat.copy()

        for i in range(n):
            speed = np.linalg.norm(u_flat[2 * i: 2 * i + 2])
            if speed > self.v_robot:
                u_flat[2 * i: 2 * i + 2] *= self.v_robot / speed

        return [u_flat[2 * i: 2 * i + 2].copy() for i in range(n)]
