"""Native evaluation_v6 object pairing and Euclidean Hausdorff metrics."""

from __future__ import annotations
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull

def _objects(labels):
    ids = np.unique(labels)
    return ids[ids != 0]

def _overlap(pred, gt, npred, ngt):
    return np.bincount((pred.astype(np.int64) * (ngt + 1) + gt).ravel(),
                       minlength=(npred + 1) * (ngt + 1)).reshape(npred + 1, ngt + 1)

def _mode_positive(row):
    if len(row) <= 1 or not np.any(row[1:]):
        return 0
    return int(np.argmax(row[1:]) + 1)

def _mode_with_background_if_intersecting(row):
    # The published evaluation_v6 Python translation checks for any positive
    # intersection, then applies mode to the entire label vector (including 0).
    # Return -1 only when the object does not intersect any foreground object.
    return int(np.argmax(row)) if np.any(row[1:]) else -1

def official_object_metrics(pred_binary, gt_instances, gt_distance_maps, distance_transform):
    from functools import lru_cache

    pred_binary = np.asarray(pred_binary, dtype=bool)
    gt_raw = np.asarray(gt_instances)
    if pred_binary.shape != gt_raw.shape or gt_raw.ndim != 2:
        raise ValueError("Prediction and instance GT shape mismatch")
    pred, npred = ndimage.label(pred_binary, structure=np.ones((3, 3), bool))
    gt_ids = _objects(gt_raw)
    ngt = len(gt_ids)
    # Preserve disconnected ground-truth instances with different native IDs.
    gt = np.searchsorted(gt_ids, gt_raw)
    gt = np.where(gt_raw == 0, 0, gt + 1).astype(np.int32)
    overlap = _overlap(pred, gt, npred, ngt)
    pa = overlap[1:, :].sum(axis=1)
    ga = overlap[:, 1:].sum(axis=0)
    if npred == 0 and ngt == 0:
        return dict(official_object_f1=1.0, official_object_dice=1.0,
                    official_object_hd=0.0, missed_glands=0)
    if npred == 0 or ngt == 0:
        return dict(official_object_f1=0.0, official_object_dice=0.0,
                    official_object_hd=float("inf"), missed_glands=ngt)
    best_g_f1 = np.array([_mode_positive(overlap[i]) for i in range(1, npred + 1)])
    tp = sum(overlap[i+1, j] / ga[j-1] > .5 for i, j in enumerate(best_g_f1) if j)
    # Challenge F1 counts each predicted component separately; it does not
    # apply the historical Hungarian one-to-one matching or min-area filter.
    f1 = 2 * tp / (npred + ngt)

    best_g = np.array([_mode_with_background_if_intersecting(overlap[i]) for i in range(1, npred + 1)])
    best_p = np.array([_mode_with_background_if_intersecting(overlap[:, j]) for j in range(1, ngt + 1)])
    pred_areas = overlap.sum(axis=1)
    gt_areas = overlap.sum(axis=0)

    def dice(i, j):
        return 2 * overlap[i, j] / (pred_areas[i] + gt_areas[j])

    d1 = sum(pa[i-1] / pa.sum() * dice(i, int(j)) for i, j in enumerate(best_g, 1) if j >= 0)
    d2 = sum(ga[j-1] / ga.sum() * dice(int(i), j) for j, i in enumerate(best_p, 1) if i >= 0)
    out = dict(official_object_f1=float(f1), official_object_dice=float((d1+d2)/2),
               missed_glands=int((overlap[1:, 1:].max(axis=0) < .5 * ga).sum()))

    pred_slices = ndimage.find_objects(pred)
    gt_slices = ndimage.find_objects(gt)

    @lru_cache(maxsize=None)
    def points(which, index):
        # Repeated evaluations of one Query may share its immutable GT points.
        if which == "gt" and gt_distance_maps is not None and hasattr(gt_distance_maps, "points"):
            return gt_distance_maps.points(index)
        labels = pred if which == "pred" else gt
        sl = pred_slices[index-1] if which == "pred" and index > 0 else (
             gt_slices[index-1] if which == "gt" and index > 0 else None)
        if sl is None:
            return np.argwhere(labels == index)
        local = np.argwhere(labels[sl] == index)
        return local + np.array([sl[0].start, sl[1].start])

    @lru_cache(maxsize=None)
    def hull(j):
        if gt_distance_maps is not None and hasattr(gt_distance_maps,"hull"):
            return gt_distance_maps.hull(j)
        p=points("gt",j)
        return p[ConvexHull(p).vertices] if len(p)>=3 and np.linalg.matrix_rank(p-p[0])==2 else p

    @lru_cache(maxsize=None)
    def hd(i, j):
        # Exact Euclidean point-set HD via distance transforms on every pixel.
        # A caller evaluating many Supports against one Query may reuse the GT
        # transforms; predicted transforms are specific to this pair.
        gd = gt_distance_maps[j] if gt_distance_maps is not None else distance_transform(gt != j)
        pp=points("pred",i)
        forward=float(gd[pp[:,0],pp[:,1]].max())
        gp=points("gt",j)
        if len(pp)==1:
            reverse=float(np.sqrt(((gp-pp[0])**2).sum(axis=1)).max())
        else:
            lo=np.minimum(pp.min(axis=0),gp.min(axis=0))
            hi=np.maximum(pp.max(axis=0),gp.max(axis=0))+1
            crop=pred[lo[0]:hi[0],lo[1]:hi[1]]!=i
            local=distance_transform(crop)
            reverse=float(local[gp[:,0]-lo[0],gp[:,1]-lo[1]].max())
        return max(forward,reverse)

    def nearest_gt(i):
        pp=points("pred",i)
        lo,hi=pp.min(axis=0),pp.max(axis=0)
        lower=[]
        for j in range(1,ngt+1):
            gd=gt_distance_maps[j] if gt_distance_maps is not None else distance_transform(gt != j)
            forward=float(gd[pp[:,0],pp[:,1]].max())
            hp=hull(j)
            delta=np.maximum(np.maximum(lo-hp,hp-hi),0)
            bbox=float(np.sqrt((delta.astype(np.float64)**2).sum(axis=1)).max())
            lower.append((max(forward,bbox),j))
        best=float("inf")
        for bound,j in sorted(lower):
            if bound>=best:break
            best=min(best,hd(i,j))
        return best

    def nearest_pred(j):
        # Exact branch-and-bound, symmetric to nearest_gt. Both terms are
        # lower bounds on the same point-set Hausdorff distance used by hd.
        gd = gt_distance_maps[j] if gt_distance_maps is not None else distance_transform(gt != j)
        hp = hull(j)
        candidates = []
        for i in range(1, npred + 1):
            pp = points("pred", i)
            forward = float(gd[pp[:, 0], pp[:, 1]].max())
            sl = pred_slices[i - 1]
            lo = np.array([sl[0].start, sl[1].start])
            hi = np.array([sl[0].stop - 1, sl[1].stop - 1])
            delta = np.maximum(np.maximum(lo - hp, hp - hi), 0)
            bbox = float(np.sqrt((delta.astype(np.float64) ** 2).sum(axis=1)).max())
            candidates.append((max(forward, bbox), i))
        best = float("inf")
        for lower, i in sorted(candidates):
            if lower >= best:
                break
            best = min(best, hd(i, j))
        return best

    hd1 = 0.0
    for i, j in enumerate(best_g, 1):
        distance = hd(i, int(j)) if j >= 0 else nearest_gt(i)
        hd1 += pa[i-1] / pa.sum() * distance
    hd2 = 0.0
    for j, i in enumerate(best_p, 1):
        distance = hd(int(i), j) if i >= 0 else nearest_pred(j)
        hd2 += ga[j-1] / ga.sum() * distance
    out["official_object_hd"] = float((hd1 + hd2)/2)
    return out
