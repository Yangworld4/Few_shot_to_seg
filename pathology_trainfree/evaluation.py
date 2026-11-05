import csv
import json
from pathlib import Path
import numpy as np
from .data import load_mask, resize_mask


def object_metrics(pred, gt, min_area=50, match_overlap=.5):
    """Optional binary connected-component protocol used in DUAL at 256 px."""
    from scipy import ndimage
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree

    def components(mask):
        labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
        areas = np.bincount(labels.ravel())
        kept = np.flatnonzero(areas >= min_area)
        kept = kept[kept != 0]
        lut = np.zeros(n + 1, np.int32)
        lut[kept] = np.arange(1, len(kept) + 1)
        return lut[labels], areas[kept]

    pl, pa = components(pred)
    gl, ga = components(gt)
    npred, ngt = len(pa), len(ga)
    if not npred and not ngt:
        return 1., 0.
    if not npred or not ngt:
        return 0., float(np.hypot(*pred.shape))
    overlap = np.bincount((pl.ravel().astype(np.int64) * (ngt + 1) + gl.ravel()).astype(np.int64),
                          minlength=(npred + 1) * (ngt + 1)).reshape(npred + 1, ngt + 1)[1:, 1:]
    ri, ci = linear_sum_assignment(-overlap / np.maximum(1, ga[None, :]))
    matches = int(np.sum(overlap[ri, ci] / ga[ci] > match_overlap))
    bestp, bestg = np.argmax(overlap, axis=1), np.argmax(overlap, axis=0)

    def boundaries(labels):
        out = []
        for i, slices in enumerate(ndimage.find_objects(labels), 1):
            local = labels[slices] == i
            edge = local & ~ndimage.binary_erosion(local)
            out.append(np.argwhere(edge) + np.array([slices[0].start, slices[1].start]))
        return out

    pbound, gbound = boundaries(pl), boundaries(gl)
    ptree, gtree = [cKDTree(x) for x in pbound], [cKDTree(x) for x in gbound]

    def hd(i, j):
        return max(float(gtree[j].query(pbound[i], workers=1)[0].max()),
                   float(ptree[i].query(gbound[j], workers=1)[0].max()))

    hp = [hd(i, int(bestp[i])) if overlap[i, bestp[i]] else min(hd(i, j) for j in range(ngt)) for i in range(npred)]
    hg = [hd(int(bestg[j]), j) if overlap[bestg[j], j] else min(hd(i, j) for i in range(npred)) for j in range(ngt)]
    return 2 * matches / (npred + ngt), float((np.dot(pa / pa.sum(), hp) + np.dot(ga / ga.sum(), hg)) / 2)


def metrics(pred, truth, class_ids):
    if pred.shape != truth.shape:
        raise ValueError("Prediction and GT shape mismatch")
    ids = [int(x) for x in class_ids]
    result = {}
    for cls in ids:
        p, g = pred == cls, truth == cls
        tp, fp, fn = int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum())
        # Match DUAL's zero-denominator convention for binary micro scores.
        result[cls] = {"tp": tp, "fp": fp, "fn": fn,
                       "dice": 2 * tp / max(1, 2 * tp + fp + fn),
                       "iou": tp / max(1, tp + fp + fn),
                       "precision": tp / max(1, tp + fp),
                       "recall": tp / max(1, tp + fn)}
    return result


def evaluate_predictions(predictions, entries, cfg, out_dir=None):
    ids = [int(x) for x in cfg["memory"]["class_ids"]]
    rows = []
    totals = {cls: {key: 0 for key in ("tp", "fp", "fn")} for cls in ids}
    by_id = {entry["id"]: entry for entry in entries}
    for image_id, pred in predictions.items():
        entry = by_id[image_id]
        if entry["mask"] is None:
            raise ValueError(f"No GT mask for {image_id}")
        truth = resize_mask(load_mask(entry["mask"], cfg["dataset"]), pred.shape)
        values = metrics(pred, truth, ids)
        row = {"image_id": image_id}
        for cls in ids:
            for key, value in values[cls].items():
                row[f"class_{cls}_{key}"] = value
                if key in totals[cls]:
                    totals[cls][key] += value
        if ids == [0, 1] and cfg["evaluation"]["object_metrics"]:
            row["object_f1"], row["object_hd"] = object_metrics(
                pred != 0, truth != 0, int(cfg["evaluation"]["object_min_area"]),
                float(cfg["evaluation"]["object_match_overlap"]))
        rows.append(row)
    overall = {"images": len(rows), "per_class": {}}
    for cls in ids:
        tp, fp, fn = (totals[cls][k] for k in ("tp", "fp", "fn"))
        overall["per_class"][str(cls)] = {**totals[cls], "dice": 2 * tp / max(1, 2 * tp + fp + fn),
                                         "iou": tp / max(1, tp + fp + fn),
                                         "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn)}
    overall["miou"] = float(np.mean([overall["per_class"][str(cls)]["iou"] for cls in ids]))
    if ids == [0, 1]:
        overall.update(overall["per_class"]["1"])
        overall["mean_image_dice"] = float(np.mean([r["class_1_dice"] for r in rows]))
        overall["mean_image_iou"] = float(np.mean([r["class_1_iou"] for r in rows]))
        if cfg["evaluation"]["object_metrics"]:
            overall["object_f1"] = float(np.mean([r["object_f1"] for r in rows]))
            overall["object_hd"] = float(np.mean([r["object_hd"] for r in rows]))
            shape = next(iter(predictions.values())).shape
            overall["object_hd_unit"] = f"{shape[0]}px" if shape[0] == shape[1] else f"{shape[0]}x{shape[1]}px"
    if out_dir:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "per_image.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        (path / "metrics.json").write_text(json.dumps(overall, indent=2))
    return overall
