# World class: enclosure geometry, entrance segments, and wall segments for the 2D herding sim

import numpy as np


class World:
    """
    Square enclosure of side length L centered at origin.
    entrance_config: '1' | '2_opposite' | '4'
    w: entrance width (absolute, not fraction of L)
    """

    def __init__(self, L, entrance_config, w):
        self.L = float(L)
        self.entrance_config = entrance_config
        self.w = float(w)
        self.wall_segments = []
        self.entrance_segments = []
        self._build()

    def _build(self):
        L = self.L
        h = L / 2.0
        w = self.w

        # Determine which walls have entrances
        # Walls: 0=bottom (y=-h), 1=right (x=+h), 2=top (y=+h), 3=left (x=-h)
        if self.entrance_config == '1':
            entrance_walls = {0}
        elif self.entrance_config == '2_opposite':
            entrance_walls = {0, 2}
        elif self.entrance_config == '4':
            entrance_walls = {0, 1, 2, 3}
        else:
            raise ValueError(f"Unknown entrance_config: {self.entrance_config}")

        # Build each wall
        self.wall_segments = []
        self.entrance_segments = []
        wall_defs = self._wall_defs(h)
        for idx, (p_start, p_end, midpoint, axis) in enumerate(wall_defs):
            if idx in entrance_walls:
                segs, ent = self._split_wall(p_start, p_end, midpoint, axis, w)
                self.wall_segments.extend(segs)
                self.entrance_segments.extend(ent)
            else:
                self.wall_segments.append((np.array(p_start), np.array(p_end)))

    def _wall_defs(self, h):
        """Returns list of (start, end, midpoint, axis) for each wall."""
        return [
            ((-h, -h), (h, -h), (0.0, -h), 'x'),   # bottom
            ((h, -h),  (h,  h), (h,  0.0), 'y'),   # right
            ((h,  h),  (-h, h), (0.0,  h), 'x'),   # top
            ((-h, h),  (-h, -h), (-h, 0.0), 'y'),  # left
        ]

    def _split_wall(self, p_start, p_end, midpoint, axis, w):
        """Split a wall at its midpoint to create a gap of width w."""
        p_start = np.array(p_start, dtype=float)
        p_end = np.array(p_end, dtype=float)
        midpoint = np.array(midpoint, dtype=float)
        half_w = w / 2.0

        if axis == 'x':
            gap_l = np.array([midpoint[0] - half_w, midpoint[1]])
            gap_r = np.array([midpoint[0] + half_w, midpoint[1]])
            segs = []
            if p_start[0] < gap_l[0]:
                segs.append((p_start.copy(), gap_l.copy()))
            if gap_r[0] < p_end[0]:
                segs.append((gap_r.copy(), p_end.copy()))
            ent = [(gap_l.copy(), gap_r.copy())]
        else:  # axis == 'y'
            gap_l = np.array([midpoint[0], midpoint[1] - half_w])
            gap_r = np.array([midpoint[0], midpoint[1] + half_w])
            segs = []
            if p_start[1] < gap_l[1]:
                segs.append((p_start.copy(), gap_l.copy()))
            if gap_r[1] < p_end[1]:
                segs.append((gap_r.copy(), p_end.copy()))
            ent = [(gap_l.copy(), gap_r.copy())]
        return segs, ent

    def entrance_centers(self):
        """Return list of entrance midpoints as np.ndarray."""
        return [0.5 * (p1 + p2) for p1, p2 in self.entrance_segments]

    def nearest_entrance(self, pos):
        """Return (center, index) of nearest entrance to pos."""
        pos = np.asarray(pos, dtype=float)
        centers = self.entrance_centers()
        if not centers:
            return None, -1
        dists = [np.linalg.norm(c - pos) for c in centers]
        idx = int(np.argmin(dists))
        return centers[idx], idx

    def is_inside(self, pos):
        """Return True if pos is strictly inside the square enclosure."""
        h = self.L / 2.0
        return -h <= pos[0] <= h and -h <= pos[1] <= h

    def wall_normal(self, seg_p1, seg_p2):
        """Return inward-pointing unit normal for a wall segment."""
        d = np.asarray(seg_p2, dtype=float) - np.asarray(seg_p1, dtype=float)
        n = np.array([-d[1], d[0]])
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            return np.array([0.0, 0.0])
        n = n / norm
        # Make sure it points inward (toward origin)
        center = 0.5 * (np.asarray(seg_p1, dtype=float) + np.asarray(seg_p2, dtype=float))
        if np.dot(n, -center) < 0:
            n = -n
        return n
