"""Coordinate frame and polyline utilities.

Canvas frame: 1024 x 1024, origin top-left, y down. Make Me a Hanzi data is
y-up with the box spanning y in [-124, 900], so screen_y = 900 - y.
"""

from __future__ import annotations

import numpy as np

CANVAS = 1024
MMAH_Y_FLIP = 900


def mmah_to_canvas(pts) -> np.ndarray:
    """Map MMAH (x, y-up) points to the canvas frame. Works on any (..., 2) array."""
    a = np.asarray(pts, dtype=np.float64)
    out = a.copy()
    out[..., 1] = MMAH_Y_FLIP - a[..., 1]
    return out


def segment_lengths(pts: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)


def polyline_length(pts) -> float:
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    return float(segment_lengths(pts).sum())


def _resample_at(pts: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Linearly interpolate all columns of pts at the given arc-length positions."""
    cum = np.concatenate([[0.0], np.cumsum(segment_lengths(pts))])
    return np.stack([np.interp(targets, cum, pts[:, k]) for k in range(pts.shape[1])], axis=1)


def resample_count(pts, n: int) -> np.ndarray:
    """n points evenly spaced along the polyline (extra columns interpolated linearly)."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) == 1:
        return np.repeat(pts, n, axis=0)
    total = polyline_length(pts)
    if total == 0:
        return np.repeat(pts[:1], n, axis=0)
    return _resample_at(pts, np.linspace(0.0, total, n))


def resample_spacing(pts, spacing: float) -> np.ndarray:
    """Points at (approximately) fixed arc-length spacing, endpoints included."""
    pts = np.asarray(pts, dtype=np.float64)
    total = polyline_length(pts)
    n = max(2, int(np.ceil(total / spacing)) + 1)
    return resample_count(pts, n)


def catmull_rom(points, spacing: float = 8.0, alpha: float = 0.5) -> np.ndarray:
    """Centripetal Catmull-Rom through points (N, 2+k), sampled every ~spacing units.

    Columns beyond x, y (e.g. pressure) are interpolated linearly per segment.
    The curve passes through every input point.
    """
    p = np.asarray(points, dtype=np.float64)
    # Drop consecutive duplicates; they make the parametrization degenerate.
    if len(p) > 1:
        keep = np.concatenate([[True], np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1) > 1e-9])
        p = p[keep]
    if len(p) < 3:
        return resample_spacing(p, spacing) if len(p) == 2 else p.copy()

    xy = p[:, :2]
    ext = np.vstack([2 * xy[0] - xy[1], xy, 2 * xy[-1] - xy[-2]])
    out = [p[:1]]
    for i in range(len(p) - 1):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0), 1e-6) ** alpha
        t2 = t1 + max(np.linalg.norm(p2 - p1), 1e-6) ** alpha
        t3 = t2 + max(np.linalg.norm(p3 - p2), 1e-6) ** alpha
        n = max(1, int(np.ceil(np.linalg.norm(p2 - p1) / spacing)))
        u = np.linspace(0.0, 1.0, n + 1)[1:]
        t = (t1 + (t2 - t1) * u)[:, None]
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        c = (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2
        extra = p[i, 2:] + (p[i + 1, 2:] - p[i, 2:]) * u[:, None]
        out.append(np.hstack([c, extra]))
    return np.vstack(out)
