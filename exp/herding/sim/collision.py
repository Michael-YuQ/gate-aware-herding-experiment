# Collision detection: ray-segment intersection, wall reflection and clipping utilities

import numpy as np


def ray_segment_intersection(origin, direction, p1, p2):
    """Return distance t along ray origin+t*direction to segment p1->p2, or None."""
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    d = direction
    v = p2 - p1
    denom = d[0] * v[1] - d[1] * v[0]
    if abs(denom) < 1e-12:
        return None
    w = p1 - origin
    t = (w[0] * v[1] - w[1] * v[0]) / denom
    s = (w[0] * d[1] - w[1] * d[0]) / denom
    if t >= 0.0 and 0.0 <= s <= 1.0:
        return t
    return None


def point_segment_distance(point, p1, p2):
    """Minimum distance from point to line segment p1->p2."""
    point = np.asarray(point, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    seg = p2 - p1
    seg_len_sq = float(np.dot(seg, seg))
    if seg_len_sq < 1e-12:
        return float(np.linalg.norm(point - p1))
    t = float(np.dot(point - p1, seg)) / seg_len_sq
    t = max(0.0, min(1.0, t))
    closest = p1 + t * seg
    return float(np.linalg.norm(point - closest))


def reflect_velocity(vel, wall_normal):
    """Reflect velocity off a wall with given outward normal (normalized)."""
    vel = np.asarray(vel, dtype=float)
    n = np.asarray(wall_normal, dtype=float)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        return vel.copy()
    n = n / n_norm
    return vel - 2.0 * np.dot(vel, n) * n


def crosses_segment(a, b, p1, p2):
    """Return True if line segment a->b crosses segment p1->p2."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    def _cross2d(u, v):
        return u[0] * v[1] - u[1] * v[0]

    d = b - a
    v = p2 - p1

    denom = _cross2d(d, v)
    if abs(denom) < 1e-12:
        return False
    w = p1 - a
    t = _cross2d(w, v) / denom
    s = _cross2d(w, d) / denom
    return 0.0 <= t <= 1.0 and 0.0 <= s <= 1.0
