# Robot dynamics: holonomic point-mass and unicycle (differential-drive) models

import numpy as np


class HolonomicRobot:
    """Holonomic point-mass robot with bounded speed and elastic inter-robot separation."""

    def __init__(self, pos, v_robot, R_body=0.05, dt=0.02):
        self.pos = np.asarray(pos, dtype=float).copy()
        self.v_robot = float(v_robot)
        self.R_body = float(R_body)
        self.dt = float(dt)
        self.vel = np.zeros(2)

    def step(self, cmd):
        """Apply velocity command (dx, dy), clipped to v_robot."""
        cmd = np.asarray(cmd, dtype=float)
        speed = np.linalg.norm(cmd)
        if speed > self.v_robot:
            cmd = cmd / speed * self.v_robot
        self.vel = cmd
        self.pos = self.pos + cmd * self.dt


def apply_elastic_separation(robots):
    """
    Pairwise elastic separation: if two robots overlap (distance < 2*R_body),
    push them apart along the line connecting their centers.
    Operates in-place on the robot list.
    """
    n = len(robots)
    for i in range(n):
        for j in range(i + 1, n):
            diff = robots[i].pos - robots[j].pos
            dist = np.linalg.norm(diff)
            min_dist = 2.0 * robots[i].R_body
            if dist < min_dist and dist > 1e-12:
                correction = 0.5 * (min_dist - dist) * diff / dist
                robots[i].pos = robots[i].pos + correction
                robots[j].pos = robots[j].pos - correction
            elif dist < 1e-12:
                # Exactly coincident: push in random direction
                angle = np.random.uniform(0, 2 * np.pi)
                correction = 0.5 * min_dist * np.array([np.cos(angle), np.sin(angle)])
                robots[i].pos = robots[i].pos + correction
                robots[j].pos = robots[j].pos - correction
