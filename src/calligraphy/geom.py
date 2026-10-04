"""Vectorized polygon queries (numpy): inside test, distance to edge, ray casting."""

from __future__ import annotations

import numpy as np


def _edges(poly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(poly, dtype=np.float64)
    return a, np.roll(a, -1, axis=0)


def points_in_polygon(pts, poly) -> np.ndarray:
    """Even-odd inside test for (K, 2) points."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    a, b = _edges(poly)
    x, y = pts[:, :1], pts[:, 1:2]
    crosses = (a[:, 1] > y) != (b[:, 1] > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_int = a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])
    return (crosses & (x < x_int)).sum(axis=1) % 2 == 1


def distance_to_boundary(pts, poly) -> np.ndarray:
    """Distance from each point to the nearest polygon edge."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    a, b = _edges(poly)
    ab = b - a
    denom = np.maximum((ab**2).sum(axis=1), 1e-12)
    ap = pts[:, None, :] - a[None]
    t = np.clip((ap * ab[None]).sum(axis=2) / denom, 0, 1)
    closest = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(pts[:, None, :] - closest, axis=2).min(axis=1)


def inscribed_radius(pts, poly) -> np.ndarray:
    """Distance to the edge for points inside the polygon, 0 outside."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return np.where(points_in_polygon(pts, poly), distance_to_boundary(pts, poly), 0.0)


def ray_hit_distance(origins, dirs, poly) -> np.ndarray:
    """Distance along each unit direction to the nearest polygon edge (inf if none)."""
    o = np.asarray(origins, dtype=np.float64).reshape(-1, 2)
    d = np.asarray(dirs, dtype=np.float64).reshape(-1, 2)
    a, b = _edges(poly)
    e = b - a  # (N, 2)
    # Solve o + t d = a + s e  for t >= 0, s in [0, 1].
    cross = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]  # (K, N)
    ao = a[None] - o[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ao[..., 0] * e[None, :, 1] - ao[..., 1] * e[None, :, 0]) / cross
        s = (ao[..., 0] * d[:, None, 1] - ao[..., 1] * d[:, None, 0]) / cross
    hit = (np.abs(cross) > 1e-12) & (t > 1e-9) & (s >= 0) & (s <= 1)
    return np.where(hit, t, np.inf).min(axis=1)
