"""Per-pair Native metrics and Support-level one-shot summaries."""

from collections import defaultdict
from functools import lru_cache
import math

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

from ..selfref_data import load_mask
from .evaluation_v6 import official_object_metrics


class QueryGround:
    def __init__(self, instances, transform):
        self.instances = instances
        self.transform = transform
        self.ids = np.unique(instances)
        self.ids = self.ids[self.ids > 0]
        self.slices = ndimage.find_objects(instances.astype(np.int32))
        self.point_cache = {}
        self.hull_cache = {}
        self.distance_cache = {}

    def points(self, index):
        if index not in self.point_cache:
            if index == 0:
                self.point_cache[index] = np.argwhere(self.instances == 0)
            else:
                ident = self.ids[index - 1]
                sl = self.slices[ident - 1]
                self.point_cache[index] = (np.argwhere(self.instances[sl] == ident) +
                                           np.array([sl[0].start, sl[1].start]))
        return self.point_cache[index]

    def hull(self, index):
        if index not in self.hull_cache:
            points = self.points(index)
            if len(points) >= 3:
                try:
                    points = points[ConvexHull(points).vertices]
                except QhullError:
                    pass
            self.hull_cache[index] = points
        return self.hull_cache[index]

    def __getitem__(self, index):
        if index not in self.distance_cache:
            ident = self.ids[index - 1] if index else 0
            self.distance_cache[index] = self.transform(self.instances != ident)
        return self.distance_cache[index]


@lru_cache(maxsize=1)
def _ground(dataset, mask, mask_sha, instance, instance_sha, height, width):
    row = dict(mask=mask, mask_sha256=mask_sha, instance=instance, instance_sha256=instance_sha,
               height=height, width=width)
    truth = load_mask(row, "mask") > 0
    instances = load_mask(row, "instance")
    if dataset == "CRAG":
        from .selfref_edt import exact_edt
        transform = exact_edt
    else:
        transform = ndimage.distance_transform_edt
    return truth, instances, QueryGround(instances, transform)


def _dice_iou(pred, truth):
    overlap = np.count_nonzero(pred & truth)
    combined = np.count_nonzero(pred) + np.count_nonzero(truth)
    return (2 * overlap / combined, overlap / (combined - overlap)) if combined else (1., 1.)


def evaluate_mask(dataset, row, pred):
    truth, instances, ground = _ground(dataset, row["mask"], row["mask_sha256"], row["instance"],
                                       row["instance_sha256"], row["height"], row["width"])
    dice, fg_iou = _dice_iou(pred, truth)
    _, bg_iou = _dice_iou(~pred, ~truth)
    objects = official_object_metrics(pred, instances, ground, ground.transform)
    return dict(dice=dice, miou=(fg_iou + bg_iou) / 2, object_f1=objects["official_object_f1"],
                object_dice=objects["official_object_dice"], object_hd=objects["official_object_hd"],
                missed_glands=objects["missed_glands"])


def summarize_pairs(rows, failure_threshold):
    by_run = defaultdict(list)
    for row in rows:
        by_run[(row["dataset"], row["split"], row["branch"])].append(row)
    summaries = []
    for (dataset, split, branch), group in by_run.items():
        by_support = defaultdict(list)
        for row in group:
            by_support[row["support_id"]].append(row["dice"])
        means = np.sort([np.mean(scores) for scores in by_support.values()])
        hd = np.array([row["object_hd"] for row in group])
        finite = hd[np.isfinite(hd)]
        fp_high = [row["dice"] for row in group if row["fp_high"]]
        missed = sum(row["missed_glands"] for row in group)
        summary = dict(dataset=dataset, split=split, branch=branch, pairs=len(group),
                       supports=len(means), queries=len({r["query_id"] for r in group}))
        summary.update({key: float(np.mean([r[key] for r in group]))
                        for key in ("dice", "miou", "object_f1", "object_dice")})
        summary.update(object_hd=float(finite.mean()) if len(finite) else None, nonfinite_hd=len(hd) - len(finite),
                       support_mean=float(means.mean()), support_sd=float(means.std(ddof=1)) if len(means) > 1 else None,
                       support_worst10=float(means[:math.ceil(.1 * len(means))].mean()),
                       support_failure=float((means < failure_threshold).mean()), failure_threshold=failure_threshold,
                       missed_glands_total=missed, missed_glands_per_pair=missed / len(group),
                       fp_high_pairs=len(fp_high), fp_high_dice=float(np.mean(fp_high)) if fp_high else None)
        summaries.append(summary)
    return summaries


def summarize_supports(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["dataset"], row["split"], row["branch"], row["support_id"]].append(row)
    return [dict(dataset=dataset, split=split, branch=branch, support_id=support_id,
                 queries=len(group), mean_dice=float(np.mean([row["dice"] for row in group])),
                 mean_miou=float(np.mean([row["miou"] for row in group])))
            for (dataset, split, branch, support_id), group in groups.items()]


def compare_sam(rows, baseline, cleanup, refined):
    pairs = defaultdict(dict)
    for row in rows:
        pairs[row["dataset"], row["support_id"], row["query_id"]][row["branch"]] = row
    datasets = defaultdict(list)
    for (dataset, _, _), pair in pairs.items():
        datasets[dataset].append((pair[baseline], pair[cleanup], pair[refined]))

    output = []
    for dataset, entries in datasets.items():
        dice_deltas = [sam["dice"] - clean["dice"] for _, clean, sam in entries]
        output.append(dict(dataset=dataset, baseline=baseline, control=cleanup, refined=refined, pairs=len(entries),
                           sam_attempted=sum(sam["sam_attempted"] for _, _, sam in entries),
                           sam_accepted=sum(sam["sam_accepted"] for _, _, sam in entries),
                           dice_improved_vs_cleanup=sum(delta > 1e-12 for delta in dice_deltas),
                           dice_worse_vs_cleanup=sum(delta < -1e-12 for delta in dice_deltas),
                           mean_dice_delta_vs_baseline=float(np.mean([sam["dice"] - base["dice"]
                                                                    for base, _, sam in entries]))))
        for metric in ("dice", "miou", "object_f1", "object_dice"):
            output[-1][f"mean_{metric}_delta_vs_cleanup"] = float(np.mean(
                [sam[metric] - clean[metric] for _, clean, sam in entries]))
    return output
