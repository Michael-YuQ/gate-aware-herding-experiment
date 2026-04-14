# Target agent models: repulsion-sum and myopic avoidance behaviors

import numpy as np
from .collision import crosses_segment, reflect_velocity


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


class BaseTarget:
    """Base class for target agents in the 2D herding sim."""

    def __init__(self, pos, v_target, dt=0.02, sigma_noise=0.05):
        self.pos = np.asarray(pos, dtype=float).copy()
        self.v_target = float(v_target)
        self.dt = float(dt)
        self.sigma_noise = float(sigma_noise)
        self.vel = np.zeros(2)

    def _compute_desired_velocity(self, robot_positions, rng):
        raise NotImplementedError

    def _apply_wall_collision(self, pos, next_pos, world):
        """
        Check if step pos->next_pos crosses any wall segment.
        Reflect velocity if so. Allow free passage through entrance segments.
        Returns adjusted next_pos and updated self.vel.
        """
        for seg_p1, seg_p2 in world.wall_segments:
            if crosses_segment(pos, next_pos, seg_p1, seg_p2):
                normal = world.wall_normal(seg_p1, seg_p2)
                self.vel = reflect_velocity(self.vel, normal)
                next_pos = pos + self.vel * self.dt
                break
        return next_pos

    def step(self, robot_positions, world, rng):
        """Compute and apply one time step."""
        self.vel = self._compute_desired_velocity(robot_positions, rng)
        speed = np.linalg.norm(self.vel)
        if speed > self.v_target:
            self.vel = self.vel / speed * self.v_target
        next_pos = self.pos + self.vel * self.dt
        next_pos = self._apply_wall_collision(self.pos, next_pos, world)
        self.pos = next_pos


class RepulsionSumTarget(BaseTarget):
    """
    velocity = v_target * normalize(sum_i (s - r_i) / ||s - r_i||^2 + noise)
    """

    def _compute_desired_velocity(self, robot_positions, rng):
        s = self.pos
        repulsion = np.zeros(2)
        for r in robot_positions:
            diff = s - np.asarray(r, dtype=float)
            dist_sq = float(np.dot(diff, diff))
            if dist_sq < 1e-6:
                dist_sq = 1e-6
            repulsion += diff / dist_sq
        noise = rng.normal(0.0, self.sigma_noise, size=2)
        repulsion += noise
        return self.v_target * _normalize(repulsion)


class MyopicAvoidanceTarget(BaseTarget):
    """
    velocity = v_target * normalize(s - r_near + noise), where r_near is nearest robot.
    """

    def _compute_desired_velocity(self, robot_positions, rng):
        s = self.pos
        if len(robot_positions) == 0:
            noise = rng.normal(0.0, self.sigma_noise, size=2)
            return self.v_target * _normalize(noise)
        dists = [np.linalg.norm(s - np.asarray(r, dtype=float)) for r in robot_positions]
        nearest = np.asarray(robot_positions[int(np.argmin(dists))], dtype=float)
        diff = s - nearest
        noise = rng.normal(0.0, self.sigma_noise, size=2)
        return self.v_target * _normalize(diff + noise)
