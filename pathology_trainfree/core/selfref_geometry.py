"""Native token geometry for cached UNI windows."""

import numpy as np
import torch
import torch.nn.functional as F

from .geometry import average_windows, reconstruct

WINDOW = 448
STRIDE = 224
PATCH = 16
GRID = WINDOW // PATCH
LOCAL_Y, LOCAL_X = np.indices((GRID, GRID)).reshape(2, -1) * PATCH


def _reflect(values, length):
    if length <= 1:
        return np.zeros_like(values)
    span = length - 1
    folded = np.mod(values, 2 * span)
    return np.minimum(folded, 2 * span - folded)


def token_geometry(positions, shape):
    y = (positions[:, 0, None] + LOCAL_Y).ravel()
    x = (positions[:, 1, None] + LOCAL_X).ravel()
    h, w = shape
    y0, y1 = np.clip(y, 0, h), np.clip(y + PATCH, 0, h)
    x0, x1 = np.clip(x, 0, w), np.clip(x + PATCH, 0, w)
    valid = (y1 - y0) * (x1 - x0)
    centers = np.stack((_reflect(x + 7.5, w), _reflect(y + 7.5, h)), axis=1).astype(np.float32)
    return centers, valid, (y0, y1, x0, x1)


def phase_candidates(positions, shape, min_valid_pixels):
    centers, valid, bounds = token_geometry(positions, shape)
    edge = np.minimum.reduce((LOCAL_Y + 8, WINDOW - LOCAL_Y - 8,
                              LOCAL_X + 8, WINDOW - LOCAL_X - 8))
    rows = np.flatnonzero(valid >= min_valid_pixels)
    order = np.lexsort((rows // GRID**2, -edge[rows % GRID**2],
                        centers[rows, 0], centers[rows, 1]))
    ordered = rows[order]
    first = np.ones(len(ordered), dtype=bool)
    first[1:] = np.any(centers[ordered[1:]] != centers[ordered[:-1]], axis=1)
    return centers, valid, bounds, rows, np.sort(ordered[first])


def support_candidates(features, positions, gt, min_valid_pixels, purity_fg, purity_bg):
    integral = np.pad(gt.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    axis = np.interp(np.arange(GRID) * PATCH + 7.5, np.arange(WINDOW), np.hanning(WINDOW)).astype(np.float32)
    hann = np.outer(axis, axis).ravel()
    vectors, labels = [], []
    for values, pos in zip(features, positions):
        centers, valid, bounds, rows, chosen = phase_candidates(pos, gt.shape, min_valid_pixels)
        if not len(chosen):
            continue
        y0, y1, x0, x1 = bounds
        count = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
        occupancy = np.divide(count[chosen], valid[chosen], dtype=np.float32)
        pure = (occupancy >= purity_fg) | (occupancy <= purity_bg)
        if not pure.any():
            continue
        _, inverse = np.unique(centers[rows], axis=0, return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        ordered = rows[order]
        groups = inverse[order]
        keep_groups = inverse[np.searchsorted(rows, chosen[pure])]
        # Filtering whole unused groups preserves the reference reduction order.
        take = np.isin(groups, keep_groups)
        ordered, groups = ordered[take], groups[take]
        starts = np.r_[0, np.flatnonzero(np.diff(groups)) + 1]
        weights = hann[ordered % GRID**2]
        means = np.add.reduceat(values[ordered] * weights[:, None], starts, axis=0)
        means /= np.add.reduceat(weights, starts)[:, None]
        vectors.append(means[np.searchsorted(groups[starts], keep_groups)])
        labels.append(occupancy[pure] >= purity_fg)
    if not vectors:
        raise ValueError("Support has no pure candidates at the configured thresholds")
    return np.concatenate(vectors), np.concatenate(labels)


def reconstruct_native(margin, positions, shape, mode="bilinear"):
    phases = []
    start = 0
    for pos in positions:
        end = start + len(pos) * GRID**2
        grids = margin[start:end].reshape(-1, GRID, GRID)
        if mode == "bilinear-window":
            values = reconstruct(grids, pos, shape, WINDOW)
        else:
            maps = torch.from_numpy(grids.astype(np.float16).astype(np.float32))[:, None]
            kwargs = {"align_corners": False} if mode == "bilinear" else {}
            expanded = F.interpolate(maps, size=(WINDOW, WINDOW), mode=mode, **kwargs)[:, 0].numpy()
            values = average_windows(expanded, pos, shape, WINDOW)
        phases.append(values)
        start = end
    if start != len(margin):
        raise ValueError("Margin length differs from cached window geometry")
    return np.mean(phases, axis=0, dtype=np.float32)
