"""SFT converter: dataset character -> phase-1 create_stroke calls.

The dataset has no pressure, so it is recovered from the outline: the brush
radius at a centerline point is its inscribed distance to the stroke's edge.
See docs/IMPLEMENTATION.md section 3.2.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from calligraphy.data.types import CharacterEntry
from calligraphy.frame import CANVAS, resample_count, resample_spacing
from calligraphy.geom import inscribed_radius, points_in_polygon, ray_hit_distance


@dataclass(frozen=True)
class ConvertConfig:
    spacing: float = 8.0  # fixed arc-length spacing (matches the renderer)
    fixed_count: int | None = None  # resample to a fixed count instead (naive preset)
    recenter: bool = True  # move samples to the outline chord midpoint (local medial axis)
    recenter_iters: int = 2
    extend_ends: bool = True  # extend to the outline along the end tangents; trim outside ends
    max_extension: float = 100.0
    radius_scale: float = 1.0
    max_points: int | None = 16  # RDP downsample over (x, y, r); None keeps every sample
    rdp_eps: float = 2.0  # starting RDP tolerance (canvas units); raised until <= max_points
    quantize: bool = True  # integer coordinates, p to 2 decimals


DEFAULT = ConvertConfig()
NAIVE = ConvertConfig(spacing=0, fixed_count=40, recenter=False, extend_ends=False, max_points=None, quantize=False)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def _trim_outside_ends(median: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Cut the median back to its first and last points inside the outline."""
    dense = resample_spacing(median, 2.0)
    inside = points_in_polygon(dense, poly)
    if not inside.any():
        return median
    idx = np.flatnonzero(inside)
    return dense[idx[0] : idx[-1] + 1] if idx[-1] > idx[0] else dense[idx[0] : idx[0] + 1]


def _recenter(samples: np.ndarray, poly: np.ndarray, chord_tol: float = 1.25) -> np.ndarray:
    """Move interior samples to the midpoint of the outline chord along the local normal.

    A move is accepted only where the half-chord is close to the inscribed radius
    at the new point, i.e. the chord really crosses the stroke. At corners the
    normal runs diagonally through the bend and the chord is much longer, so
    those samples stay put (moving them cuts the corner).
    """
    if len(samples) < 3:
        return samples
    out = samples.copy()
    tangent = _unit(samples[2:] - samples[:-2])
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    mid = samples[1:-1]
    d_pos = ray_hit_distance(mid, normal, poly)
    d_neg = ray_hit_distance(mid, -normal, poly)
    finite = np.isfinite(d_pos) & np.isfinite(d_neg)
    shift = np.where(finite, (d_pos - d_neg) / 2, 0.0)
    moved = mid + normal * shift[:, None]
    half_chord = np.where(finite, (d_pos + d_neg) / 2, np.inf)
    ok = finite & points_in_polygon(mid, poly) & (half_chord <= chord_tol * inscribed_radius(moved, poly) + 1.0)
    out[1:-1] = np.where(ok[:, None], moved, mid)
    return out


def _end_direction(samples: np.ndarray, end: int, baseline: float = 24.0) -> np.ndarray:
    """Outward tangent at an end, measured over ~baseline units (ignores tiny end hooks)."""
    pts = samples if end == -1 else samples[::-1]
    dist = np.linalg.norm(pts - pts[-1], axis=1)
    far = np.flatnonzero(dist >= baseline)
    ref = pts[far[-1]] if len(far) else pts[0]
    return _unit(pts[-1] - ref)


def _extend(samples: np.ndarray, poly: np.ndarray, spacing: float, max_ext: float) -> np.ndarray:
    """Extend both ends along the end tangent up to the outline edge."""
    if len(samples) < 2:
        return samples
    pieces = [samples]
    for end, direction in ((0, _end_direction(samples, 0)), (-1, _end_direction(samples, -1))):
        d = _unit(direction)
        origin = samples[end]
        h = float(ray_hit_distance(origin[None], d[None], poly)[0])
        if not np.isfinite(h) or h < 1.0:
            continue
        h = min(h, max_ext)
        steps = np.arange(spacing, h, spacing).tolist() + [h - 0.5]
        ext = origin + np.asarray(steps)[:, None] * d
        if end == 0:
            pieces.insert(0, ext[::-1])
        else:
            pieces.append(ext)
    return np.vstack(pieces)


def _rdp_keep(pts: np.ndarray, eps: float) -> np.ndarray:
    """Ramer-Douglas-Peucker on N-D points; returns a boolean keep mask."""
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        ab = b - a
        seg = pts[i + 1 : j] - a
        denom = float(ab @ ab)
        t = np.clip(seg @ ab / denom, 0, 1) if denom > 0 else np.zeros(len(seg))
        dist = np.linalg.norm(seg - t[:, None] * ab, axis=1)
        m = int(np.argmax(dist))
        if dist[m] > eps:
            keep[i + 1 + m] = True
            stack += [(i, i + 1 + m), (i + 1 + m, j)]
    return keep


def downsample(samples: np.ndarray, max_points: int, eps: float = 2.0) -> np.ndarray:
    """RDP points (over x, y, r) within tolerance eps, raised until <= max_points; endpoints kept."""
    while True:
        keep = _rdp_keep(samples, eps)
        if keep.sum() <= max_points:
            return samples[keep]
        eps *= 1.3


def stroke_samples(median: np.ndarray, poly: np.ndarray, cfg: ConvertConfig) -> np.ndarray:
    """Centerline samples with radii, (K, 3) x, y, r in canvas units (before normalisation)."""
    m = np.asarray(median, dtype=np.float64)
    if cfg.extend_ends:
        m = _trim_outside_ends(m, poly)
    if len(m) == 1:
        s = m.copy()
    elif cfg.fixed_count:
        s = resample_count(m, cfg.fixed_count)
    else:
        s = resample_spacing(m, cfg.spacing)
    if cfg.recenter:
        for _ in range(cfg.recenter_iters):
            s = _recenter(s, poly)
            if not cfg.fixed_count:
                s = resample_spacing(s, cfg.spacing)
    if cfg.extend_ends:
        s = _extend(s, poly, cfg.spacing or 8.0, cfg.max_extension)
    r = inscribed_radius(s, poly) * cfg.radius_scale
    return np.column_stack([s, r])


def convert(entry: CharacterEntry, cfg: ConvertConfig = DEFAULT, action_frame: int = CANVAS) -> list[dict]:
    """Phase-1 strokes [{"points": [[x, y, p], ...], "width": w}, ...] in action_frame units."""
    per_stroke = [stroke_samples(m, o, cfg) for m, o in zip(entry.medians, entry.outlines)]
    r_max = max((float(s[:, 2].max()) for s in per_stroke if len(s)), default=0.0)
    r_max = max(r_max, 1.0)
    unit = CANVAS / action_frame
    out = []
    for s in per_stroke:
        if cfg.max_points:
            s = downsample(s, cfg.max_points, cfg.rdp_eps)
        xy = s[:, :2] / unit
        p = np.clip(s[:, 2] / r_max, 0, 1)
        width = 2 * r_max / unit
        if cfg.quantize:
            xy = np.round(xy)
            p = np.round(p, 2)
            width = round(width, 1)
            pts = [[int(x), int(y), float(q)] for (x, y), q in zip(xy, p)]
        else:
            pts = [[float(x), float(y), float(q)] for (x, y), q in zip(xy, p)]
        out.append({"points": pts, "width": width})
    return out


def variant(**kw) -> ConvertConfig:
    return replace(DEFAULT, **kw)
