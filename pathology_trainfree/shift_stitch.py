"""True 2-D shift-and-stitch geometry for UNI token margins.

Patch embeddings remain 16x16 with Transformer context.  The 16 phases only
provide a nominal 4-pixel sampling pitch; they do not create independent 4x4
semantic receptive fields.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt


def phase_grid(offsets=(0, 8)):
    """Return a deterministic Cartesian set of (dy, dx) phases."""
    values = tuple(int(v) for v in offsets)
    if not values or any(v < 0 for v in values) or len(set(values)) != len(values):
        raise ValueError("offsets must be distinct non-negative integers")
    return [(dy, dx) for dy in values for dx in values]


def starts(length: int, size: int, stride: int, offset: int):
    n = max(1, math.ceil((length + offset - size) / stride) + 1)
    return [-offset + stride * i for i in range(n)]


def windows_xy(image, size: int, stride: int, phase, mode="reflect"):
    """Same historical scalar-offset rule, independently applied to Y and X."""
    dy, dx = map(int, phase)
    h, w = image.shape[:2]
    for y in starts(h, size, stride, dy):
        for x in starts(w, size, stride, dx):
            top, left = max(0, -y), max(0, -x)
            bottom, right = max(0, y + size - h), max(0, x + size - w)
            crop = image[max(0, y):min(h, y + size), max(0, x):min(w, x + size)]
            if ((top or bottom) and crop.shape[0] < 2) or ((left or right) and crop.shape[1] < 2):
                raise ValueError("Reflection padding requires at least two source pixels per padded axis")
            pad = ((top, bottom), (left, right)) + (((0, 0),) if image.ndim == 3 else ())
            crop = np.pad(crop, pad, mode=mode)
            if crop.shape[:2] != (size, size):
                raise RuntimeError("Unexpected window shape")
            yield y, x, crop


def reflect_coordinate(value, length):
    """Continuous reflect-101 mapping of a pixel-index coordinate to [0,L-1]."""
    if length < 2:
        return np.zeros_like(np.asarray(value, dtype=np.float64))
    span = float(length - 1)
    phase = np.mod(np.asarray(value, dtype=np.float64), 2.0 * span)
    return np.where(phase <= span, phase, 2.0 * span - phase)


def token_centers(positions, grid=28, patch=16):
    """Token center coordinates in native pixel-index coordinates (x,y).

    The half-pixel term converts geometric coordinates to the index convention
    used by reflection and image masks. Window positions are the actual padded
    crop origins, including negative origins.
    """
    local = (np.arange(grid, dtype=np.float64) + 0.5) * patch - 0.5
    all_x, all_y = [], []
    for y, x in np.asarray(positions, dtype=np.int64):
        all_x.append(x + local)
        all_y.append(y + local)
    return np.asarray(all_x), np.asarray(all_y)


def dense_stitch(margin_grids, positions_by_phase, phases, image_shape,
                 grid=28, patch=16, output_size=(256, 256)):
    """Scatter unique physical token-margin samples, then interpolate to output.

    Duplicate physical centers from overlapping windows are averaged once via
    count normalization. Reflection maps padded token centers back to native
    image coordinates; the resulting edge grid may have sub-4px intervals.
    """
    h, w = map(int, image_shape)
    samples = defaultdict(lambda: [0.0, 0])
    raw_centers = 0
    for phase in phases:
        maps = margin_grids[phase]
        positions = positions_by_phase[phase]
        if maps.shape != (len(positions), grid, grid):
            raise ValueError(f"Margin/position mismatch for phase {phase}: {maps.shape}")
        xs, ys = token_centers(positions, grid, patch)
        for win in range(len(positions)):
            xx = reflect_coordinate(xs[win], w)
            yy = reflect_coordinate(ys[win], h)
            gx, gy = np.meshgrid(xx, yy)
            raw_centers += grid * grid
            for yv, xv, mv in zip(gy.ravel(), gx.ravel(), maps[win].astype(np.float32).ravel()):
                # Coordinates are rounded only to collapse floating-point copies
                # of a theoretically identical reflected physical point.
                key = (round(float(yv), 7), round(float(xv), 7))
                samples[key][0] += float(mv)
                samples[key][1] += 1
    if not samples:
        raise ValueError("No token margins supplied to stitcher")
    ys = np.asarray(sorted({k[0] for k in samples}), np.float64)
    xs = np.asarray(sorted({k[1] for k in samples}), np.float64)
    total = np.zeros((len(ys), len(xs)), np.float64)
    count = np.zeros((len(ys), len(xs)), np.int32)
    yi, xi = {v: i for i, v in enumerate(ys)}, {v: i for i, v in enumerate(xs)}
    for (y, x), (value, n) in samples.items():
        total[yi[y], xi[x]] = value / n
        count[yi[y], xi[x]] = n
    missing_grid = count == 0
    missing_grid_count = int(missing_grid.sum())
    if missing_grid_count:
        # A missing dense location is filled by the nearest observed margin.
        # This is a fixed geometric rule and never consults a label.
        nearest = distance_transform_edt(missing_grid, return_distances=False, return_indices=True)
        total[missing_grid] = total[tuple(axis[missing_grid] for axis in nearest)]
    interp = RegularGridInterpolator((ys, xs), total, method="linear", bounds_error=False,
                                     fill_value=np.nan)
    oy, ox = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij")
    native = interp(np.stack([oy.ravel(), ox.ravel()], axis=-1)).reshape(h, w).astype(np.float32)
    missing = ~np.isfinite(native)
    missing_count = int(missing.sum())
    if missing_count:
        # Fixed, GT-free nearest observed-value extension for uncovered borders.
        nearest = distance_transform_edt(missing, return_distances=False, return_indices=True)
        native[missing] = native[tuple(axis[missing] for axis in nearest)]
    try:
        import torch
        import torch.nn.functional as F
        out = F.interpolate(torch.from_numpy(native)[None, None], size=tuple(output_size),
                            mode="bilinear", align_corners=False)[0, 0].numpy()
    except ImportError:
        from PIL import Image
        out = np.asarray(Image.fromarray(native).resize((output_size[1], output_size[0]), Image.Resampling.BILINEAR))
    intervals_x = np.diff(xs)
    intervals_y = np.diff(ys)
    diagnostics = {
        "raw_token_centers": raw_centers,
        "unique_reflected_positions": len(samples),
        "dense_grid_shape": [int(len(ys)), int(len(xs))],
        "dense_coverage": float(np.count_nonzero(count) / count.size),
        "missing_dense_grid_cells": missing_grid_count,
        "missing_dense_grid_fraction": missing_grid_count / count.size,
        "overlap_observations": int(count.sum() - len(samples)),
        "mean_observations_per_unique_position": float(count[count > 0].mean()),
        "missing_native_pixels_before_extension": missing_count,
        "missing_native_pixel_fraction": missing_count / (h * w),
        "x_pitch_median": float(np.median(intervals_x)) if len(intervals_x) else None,
        "y_pitch_median": float(np.median(intervals_y)) if len(intervals_y) else None,
        "x_pitch_min": float(intervals_x.min()) if len(intervals_x) else None,
        "y_pitch_min": float(intervals_y.min()) if len(intervals_y) else None,
    }
    return native, out.astype(np.float32), diagnostics, (ys, xs, total.astype(np.float32), count)
