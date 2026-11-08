"""Native N3-bil diagnostics from validated seed-1337 token-margin caches."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathology_trainfree.data import load_image, load_mask, read_manifest, resize_mask
from pathology_trainfree.geometry import reconstruct, resize_map
from pathology_trainfree.shift_stitch import windows_xy
from dense_feature_matching import DC, edge, guided_filter

SOURCE = ROOT / "outputs/uni_shift_stitch_seed1337/phase_margins"
LEGACY = ROOT / "outputs/dense_feature_matching_seed1337/cases"
DATA_ROOT = ROOT / "datasets"
DEFAULT_OUT = ROOT / "outputs/native_diagnostics_seed1337"
PHASES = {1: [(0, 0)], 2: [(0, 0), (8, 8)],
          4: [(y, x) for y in (0, 8) for x in (0, 8)],
          16: [(y, x) for y in (0, 4, 8, 12) for x in (0, 4, 8, 12)]}
FIELDS = ("dice", "fg_iou", "object_f1", "object_dice", "object_hd", "boundary_iou", "bf1_2px", "bf1_4px")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def components(mask, area):
    labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
    sizes = np.bincount(labels.ravel())
    kept = np.flatnonzero(sizes >= area)
    kept = kept[kept != 0]
    lut = np.zeros(n + 1, np.int32)
    lut[kept] = np.arange(1, len(kept) + 1)
    return lut[labels], sizes[kept]


def object_details(pred, gt, area):
    pl, pa = components(pred, area)
    gl, ga = components(gt, area)
    pn, gn = len(pa), len(ga)
    ov = np.bincount((pl.ravel().astype(np.int64) * (gn + 1) + gl.ravel()),
                     minlength=(pn + 1) * (gn + 1)).reshape(pn + 1, gn + 1)[1:, 1:]
    matched_p, matched_g = np.zeros(pn, bool), np.zeros(gn, bool)
    if pn and gn:
        ri, ci = linear_sum_assignment(-ov / ga[None, :])
        ok = ov[ri, ci] / ga[ci] > .5
        matched_p[ri[ok]], matched_g[ci[ok]] = True, True
    tp = int(matched_g.sum())
    f1 = 2 * tp / (pn + gn) if pn + gn else 1.
    # Matched-region Dice is an added diagnostic; unmatched objects contribute zero.
    obj_dice = float(sum(2 * ov[i, j] / (pa[i] + ga[j]) for i, j in zip(ri[ok], ci[ok])) / max(1, pn + gn - tp)) if pn and gn else float(not (pn or gn))
    if pn and gn:
        pb = [np.argwhere(edge(pl == i)) for i in range(1, pn + 1)]
        gb = [np.argwhere(edge(gl == j)) for j in range(1, gn + 1)]
        pt, gt = [cKDTree(x) for x in pb], [cKDTree(x) for x in gb]
        def hd(i, j):
            return max(float(gt[j].query(pb[i])[0].max()), float(pt[i].query(gb[j])[0].max()))
        bp, bg = ov.argmax(1), ov.argmax(0)
        hp = [hd(i, int(bp[i])) if ov[i, bp[i]] else min(hd(i, j) for j in range(gn)) for i in range(pn)]
        hg = [hd(int(bg[j]), j) if ov[bg[j], j] else min(hd(i, j) for i in range(pn)) for j in range(gn)]
        obj_hd = float((np.dot(pa / pa.sum(), hp) + np.dot(ga / ga.sum(), hg)) / 2)
    else:
        obj_hd = 0. if not (pn or gn) else float(np.hypot(*pred.shape))
    links = (ov >= max(10, round(area * .2))) & (ov / np.maximum(1, np.minimum(pa[:, None], ga[None, :])) >= .1) if pn and gn else np.zeros((pn, gn), bool)
    miss = ~matched_g
    false = ~matched_p
    details = dict(gt_objects=gn, pred_objects=pn, matched_objects=tp,
                   unmatched_gt_no_overlap=int(np.sum(miss & ((ov.sum(0) == 0) if pn else np.ones(gn, bool)))),
                   unmatched_gt_partial=int(np.sum(miss & ((ov.sum(0) > 0) if pn else np.zeros(gn, bool)))),
                   unmatched_pred_no_overlap=int(np.sum(false & ((ov.sum(1) == 0) if gn else np.ones(pn, bool)))),
                   unmatched_pred_partial=int(np.sum(false & ((ov.sum(1) > 0) if gn else np.zeros(pn, bool)))),
                   merge_count=int(np.sum(links.sum(1) >= 2)))
    return dict(object_f1=f1, object_dice=obj_dice, object_hd=obj_hd), details


def evaluate(margin, gt, area):
    pred = margin > 0
    g = gt.astype(bool)
    tp = int((pred & g).sum()); fp = int((pred & ~g).sum()); fn = int((~pred & g).sum())
    values = dict(dice=2 * tp / max(1, 2 * tp + fp + fn), fg_iou=tp / max(1, tp + fp + fn), tp=tp, fp=fp, fn=fn)
    obj, details = object_details(pred, g, area)
    values.update(obj)
    pb, gb = edge(pred), edge(g)
    dg = ndimage.distance_transform_edt(~gb) if gb.any() else np.full(g.shape, np.inf)
    dp = ndimage.distance_transform_edt(~pb) if pb.any() else np.full(g.shape, np.inf)
    for radius in (2, 4):
        precision = int((pb & (dg <= radius)).sum()) / max(1, int(pb.sum()))
        recall = int((gb & (dp <= radius)).sum()) / max(1, int(gb.sum()))
        values[f"bf1_{radius}px"] = 2 * precision * recall / max(1e-30, precision + recall) if pb.any() or gb.any() else 1.
    def inner(x):
        return x & (ndimage.distance_transform_edt(np.pad(x, 1))[1:-1, 1:-1] <= 2)
    bi, gi = inner(pred), inner(g)
    values["boundary_iou"] = int((bi & gi).sum()) / max(1, int((bi | gi).sum()))
    return values, details


def get_entries(ds):
    manifest = DATA_ROOT / ds / "manifests/paper_seed_2026/test.csv"
    entries = read_manifest(manifest, require_mask=True)
    if len(entries) != (80 if ds == "GlaS" else 40):
        raise ValueError(f"Unexpected {ds} test count: {len(entries)}")
    if ds == "GlaS":
        ids = [e["id"] for e in entries]
        if sum("testA" in x for x in ids) != 60 or sum("testB" in x for x in ids) != 20:
            raise ValueError("GLaS TestA/TestB are not 60+20")
    return entries


def area_for(shape, output):
    # The historical 50 px at 256x256 is scaled by actual pixel-area ratio.
    return max(1, int(round(50 * shape[0] * shape[1] / (256 * 256)))) if output == "native" else 50


def phase_grids(ds, iid, phase):
    suffix = f"dy{phase[0]}_dx{phase[1]}"
    with np.load(SOURCE / ds / f"{iid}.npz", allow_pickle=False) as z:
        return z[f"margin_{suffix}"].copy(), z[f"positions_{suffix}"].copy()


def baseline(ds, entries, out):
    rows = []
    for index, e in enumerate(entries, 1):
        iid = e["id"]
        image = load_image(e["image"]); gt = load_mask(e["mask"], DC)
        shape = image.shape[:2]
        if gt.shape != shape:
            raise ValueError(f"Native GT/image size mismatch: {ds}/{iid}")
        phase_maps = []
        for phase in PHASES[2]:
            grid, pos = phase_grids(ds, iid, phase)
            expected = np.array([(y, x) for y, x, _ in windows_xy(image, 448, 224, phase)])
            if not np.array_equal(pos, expected) or grid.shape != (len(pos), 28, 28):
                raise ValueError(f"Phase geometry/cache mismatch: {ds}/{iid}/{phase}")
            phase_maps.append(reconstruct(grid, pos, shape, 448, "float16"))
        native = np.mean(np.stack(phase_maps), axis=0, dtype=np.float32)
        oldpath = LEGACY / ds / f"{iid}_B0.npz"
        with np.load(oldpath, allow_pickle=False) as z:
            old = z["margin"].astype(np.float32)
        down = resize_map(native, (256, 256))
        if not np.array_equal(down > 0, old > 0):
            raise ValueError(f"Native reconstruction fails validated legacy parity: {ds}/{iid}")
        path = out / "baseline" / ds / f"{iid}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            np.savez_compressed(path, margin=native, mask=(native > 0).astype(np.uint8), image_id=np.array(iid),
                                image_shape=np.array(shape), gt_path=np.array(str(e["mask"])),
                                phases=np.array(PHASES[2]), window=np.array(448), stride=np.array(224), threshold=np.array(0.))
        for method, margin, truth in (("B0-native", native, gt), ("B0-native256", down, resize_mask(gt, (256, 256))),
                                      ("B0-legacy256", old, resize_mask(gt, (256, 256)))):
            v, _ = evaluate(margin, truth, area_for(truth.shape, "native" if method == "B0-native" else "256"))
            rows.append(dict(dataset=ds, image_id=iid, method=method, height=truth.shape[0], width=truth.shape[1], **v))
        if index % 10 == 0: print(f"baseline {ds} {index}/{len(entries)}", flush=True)
    write_csv(out / "tables" / f"baseline_{ds}_per_image.csv", rows)
    return rows


def oracle_grids(gt, phase):
    # Window coordinates are identical to UNI; padded pixels have zero weight.
    dummy = np.zeros(gt.shape, np.uint8)
    pos = np.asarray([(y, x) for y, x, _ in windows_xy(dummy, 448, 224, phase)])
    h, w = gt.shape
    iy = pos[:, 0, None, None] + np.arange(28)[None, :, None] * 16
    ix = pos[:, 1, None, None] + np.arange(28)[None, None, :] * 16
    y0 = np.clip(iy, 0, h); y1 = np.clip(iy + 16, 0, h)
    x0 = np.clip(ix, 0, w); x1 = np.clip(ix + 16, 0, w)
    integral = np.pad(gt.astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    area = (y1-y0) * (x1-x0)
    # A token with no source pixel carries no observation; neutral margin at fusion.
    occ = np.divide(sums, area, out=np.full_like(sums, .5), where=area > 0).astype(np.float32)
    return occ, pos, np.stack(np.broadcast_arrays(y0, y1, x0, x1), axis=-1).reshape(-1, 4)


def bilinear_oracle(gt, phases):
    maps = []
    observations = []
    for phase in phases:
        occ, pos, boxes = oracle_grids(gt, phase)
        maps.append(reconstruct(2 * occ - 1, pos, gt.shape, 448, "float16"))
        observations.append((occ.ravel(), boxes))
    return np.mean(np.stack(maps), axis=0, dtype=np.float32), observations


def oracle(ds, entries, out):
    rows = []
    for index, e in enumerate(entries, 1):
        gt = load_mask(e["mask"], DC); iid = e["id"]
        for name, hard in (("D0a_soft_oracle", False), ("D0c_hard_oracle", True)):
            maps = []
            for phase in PHASES[2]:
                occ, pos, _ = oracle_grids(gt, phase)
                margin = np.where(occ > .5, 1., -1.) if hard else 2 * occ - 1
                maps.append(reconstruct(margin, pos, gt.shape, 448, "float16"))
            native = np.mean(np.stack(maps), axis=0, dtype=np.float32)
            p = out / "oracle" / ds / f"{iid}_{name}.npz"
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists(): np.savez_compressed(p, margin=native, mask=(native > 0).astype(np.uint8))
            values, _ = evaluate(native, gt, area_for(gt.shape, "native"))
            rows.append(dict(dataset=ds, image_id=iid, method=name, **values))
        if index % 10 == 0: print(f"oracle {ds} {index}/{len(entries)}", flush=True)
    write_csv(out / "tables" / f"oracle_{ds}_per_image.csv", rows)
    return rows


def error_analysis(ds, entries, out):
    bands = [(0, 4), (4, 8), (8, 16), (16, 32), (32, np.inf)]
    b_rows, o_rows = [], []
    figcases = []
    for index, e in enumerate(entries, 1):
        iid = e["id"]
        with np.load(out / "baseline" / ds / f"{iid}.npz", allow_pickle=False) as z: margin = z["margin"].copy()
        gt = load_mask(e["mask"], DC).astype(bool); pred = margin > 0
        boundary = edge(gt)
        dist = ndimage.distance_transform_edt(~boundary) if boundary.any() else np.full(gt.shape, np.inf)
        fp, fn = pred & ~gt, ~pred & gt
        fv, nv = 0, 0
        for lo, hi in bands:
            sel = (dist >= lo) & (dist < hi) if np.isfinite(hi) else dist >= lo
            f, n, pixels = int((fp & sel).sum()), int((fn & sel).sum()), int(sel.sum())
            fv += f; nv += n
            b_rows.append(dict(dataset=ds, image_id=iid, band=f"{lo}-{hi if np.isfinite(hi) else 'inf'}", pixels=pixels,
                               fp=f, fn=n, errors=f+n, error_rate=(f+n)/max(1,pixels),
                               error_share=(f+n)/max(1,int(fp.sum()+fn.sum())),
                               fp_share=f/max(1,int(fp.sum())), fn_share=n/max(1,int(fn.sum()))))
        if fv != int(fp.sum()) or nv != int(fn.sum()): raise ValueError(f"Boundary bins fail conservation: {ds}/{iid}")
        _, obj = object_details(pred, gt, area_for(gt.shape, "native"))
        o_rows.append(dict(dataset=ds, image_id=iid, **obj))
        figcases.append((iid, int(obj["merge_count"]), int(fn.sum()) / max(1,int(gt.sum()))))
        if index % 10 == 0: print(f"errors {ds} {index}/{len(entries)}", flush=True)
    write_csv(out / "error_analysis" / f"d1_boundary_error_{ds}_per_image.csv", b_rows)
    write_csv(out / "error_analysis" / f"d1_object_errors_{ds}_per_image.csv", o_rows)
    return b_rows, o_rows


def operator_boxes(observations):
    boxes = np.concatenate([b for _, b in observations])
    occ = np.concatenate([o for o, _ in observations])
    areas = (boxes[:, 1] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 2])
    boxes, occ = boxes[areas > 0], occ[areas > 0]
    unique, inverse, counts = np.unique(boxes, axis=0, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=occ, minlength=len(unique))
    target = (sums / counts).astype(np.float32)
    return unique.astype(np.int32), target, counts.astype(np.float32)


def forward_boxes(image, boxes):
    integral = np.pad(image.astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    y0, y1, x0, x1 = boxes.T
    area = (y1-y0)*(x1-x0)
    return ((integral[y1,x1]-integral[y0,x1]-integral[y1,x0]+integral[y0,x0])/area).astype(np.float32)


def adjoint_boxes(values, boxes, shape):
    y0, y1, x0, x1 = boxes.T
    weight = values / ((y1-y0)*(x1-x0))
    diff = np.zeros((shape[0]+1, shape[1]+1), np.float32)
    np.add.at(diff, (y0,x0), weight); np.add.at(diff, (y1,x1), weight)
    np.add.at(diff, (y1,x0), -weight); np.add.at(diff, (y0,x1), -weight)
    return diff.cumsum(0).cumsum(1)[:-1,:-1]


def pcbp(init, observations, iterations=8):
    boxes, target, counts = operator_boxes(observations)
    coverage = adjoint_boxes(counts, boxes, init.shape)
    estimate = np.clip(init.astype(np.float32), 0, 1)
    residuals = []
    def loss(x):
        delta = forward_boxes(x, boxes) - target
        return float(np.dot(counts.astype(np.float64), (delta.astype(np.float64)**2)) / counts.sum()), delta
    previous, delta = loss(estimate); residuals.append(previous)
    for _ in range(iterations):
        gradient = adjoint_boxes(counts * delta, boxes, init.shape)
        direction = gradient / np.maximum(coverage, 1e-8)
        step = 1.
        for _attempt in range(15):
            candidate = np.clip(estimate - step * direction, 0, 1)
            current, new_delta = loss(candidate)
            if current <= previous + 1e-12: break
            step *= .5
        else:
            candidate, current, new_delta = estimate, previous, delta
        estimate, previous, delta = candidate, current, new_delta
        residuals.append(current)
    return estimate, residuals, len(boxes)


def multiphase(ds, entries, out, phase_counts):
    rows, residual_rows = [], []
    for index, e in enumerate(entries, 1):
        iid = e["id"]
        image = load_image(e["image"]); gt = load_mask(e["mask"], DC)
        guide = cv2.cvtColor(image.astype(np.float32)/255, cv2.COLOR_RGB2GRAY)
        for count in phase_counts:
            t0 = time.perf_counter()
            margin, observations = bilinear_oracle(gt, PHASES[count])
            bilinear_seconds = time.perf_counter()-t0
            values, _ = evaluate(margin, gt, area_for(gt.shape, "native"))
            boxes, target, weights = operator_boxes(observations)
            initial_residual = float(np.dot(weights.astype(np.float64), (forward_boxes((margin+1)/2, boxes)-target).astype(np.float64)**2)/weights.sum())
            rows.append(dict(dataset=ds, image_id=iid, phase_count=count, method="bilinear_oracle",
                             observation_residual=initial_residual, reconstruction_seconds=bilinear_seconds,
                             peak_gpu_mib=0, **values))
            t0 = time.perf_counter()
            estimate, residuals, unique_boxes = pcbp((margin+1)/2, observations)
            pcbp_seconds = time.perf_counter()-t0
            values, _ = evaluate(2*estimate-1, gt, area_for(gt.shape, "native"))
            rows.append(dict(dataset=ds, image_id=iid, phase_count=count, method="pcbp_no_prior",
                             observation_residual=residuals[-1], reconstruction_seconds=pcbp_seconds,
                             peak_gpu_mib=0, **values))
            for iteration, residual in enumerate(residuals):
                residual_rows.append(dict(dataset=ds, image_id=iid, phase_count=count, iteration=iteration,
                                          observation_residual=residual, unique_boxes=unique_boxes))
            t0 = time.perf_counter()
            filtered = np.clip(guided_filter(guide, estimate, r=4, eps=.001), 0, 1)
            jbu_seconds = time.perf_counter()-t0
            values, _ = evaluate(2*filtered-1, gt, area_for(gt.shape, "native"))
            jbu_residual = float(np.dot(weights.astype(np.float64), (forward_boxes(filtered, boxes)-target).astype(np.float64)**2)/weights.sum())
            rows.append(dict(dataset=ds, image_id=iid, phase_count=count, method="pcbp_jbu_rgb",
                             observation_residual=jbu_residual, reconstruction_seconds=pcbp_seconds+jbu_seconds,
                             peak_gpu_mib=0, **values))
        write_csv(out / "multiphase" / f"{ds}_per_image.csv", rows)
        write_csv(out / "multiphase" / f"{ds}_residuals.csv", residual_rows)
        if index % 5 == 0: print(f"multiphase {ds} {index}/{len(entries)}", flush=True)
    return rows, residual_rows


def summarize(out, datasets, phase_status):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    display={"GlaS":"GLaS-Test","CRAG":"CRAG-Test"}

    table = out / "tables"
    baseline = pd.concat([pd.read_csv(table / f"baseline_{ds}_per_image.csv") for ds in datasets], ignore_index=True)
    oracle_df = pd.concat([pd.read_csv(table / f"oracle_{ds}_per_image.csv") for ds in datasets], ignore_index=True)
    for ds in datasets:
        expected=len(get_entries(ds))
        for method in ("B0-native","B0-native256","B0-legacy256"):
            sub=baseline[(baseline.dataset==ds)&(baseline.method==method)]
            if len(sub)!=expected or sub.image_id.nunique()!=expected: raise ValueError(f"Incomplete baseline {ds}/{method}")
        for method in ("D0a_soft_oracle","D0c_hard_oracle"):
            sub=oracle_df[(oracle_df.dataset==ds)&(oracle_df.method==method)]
            if len(sub)!=expected or sub.image_id.nunique()!=expected: raise ValueError(f"Incomplete Oracle {ds}/{method}")
    inventory=[]
    for ds in datasets:
        for e in get_entries(ds):
            source=SOURCE/ds/f"{e['id']}.npz"
            saved=out/"baseline"/ds/f"{e['id']}.npz"
            with np.load(saved,allow_pickle=False) as z:
                shape=z["image_shape"].tolist()
                if z["margin"].shape!=tuple(shape) or z["margin"].dtype!=np.float32 or z["mask"].shape!=tuple(shape) or str(z["image_id"])!=e["id"]:
                    raise ValueError(f"Baseline cache alignment mismatch: {ds}/{e['id']}")
            inventory.append(dict(dataset=ds,image_id=e["id"],height=shape[0],width=shape[1],
                                  image_path=str(e["image"]),gt_path=str(e["mask"]),
                                  phase_grid_path=str(source),phase_grid_sha256=sha(source),
                                  native_cache_path=str(saved)))
    write_csv(out/"configs/cache_inventory.csv",inventory)
    all_metrics = pd.concat([baseline, oracle_df], ignore_index=True)
    summary = all_metrics.groupby(["dataset", "method"], sort=False)[list(FIELDS)].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a,b in summary.columns]
    summary = summary.reset_index()
    write_csv(table / "method_summary.csv", summary.to_dict("records"))
    write_csv(table / "baseline_native_vs_256.csv", summary[summary.method.str.startswith("B0-")].to_dict("records"))
    paired = []
    paired_images = []
    rng = np.random.default_rng(2026)
    for ds in datasets:
        sub = all_metrics[all_metrics.dataset == ds]
        base = sub[sub.method == "B0-native"].set_index("image_id")
        for method in ("D0a_soft_oracle", "D0c_hard_oracle"):
            target = sub[sub.method == method].set_index("image_id").loc[base.index]
            for metric in FIELDS:
                delta = (target[metric]-base[metric]).to_numpy()
                means = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(1)
                paired.append(dict(dataset=ds, method=method, metric=metric, delta_mean=float(delta.mean()),
                                   ci95_low=float(np.quantile(means,.025)), ci95_high=float(np.quantile(means,.975))))
                paired_images.extend(dict(dataset=ds, image_id=iid, method=method, metric=metric, delta=float(value))
                                     for iid,value in zip(base.index,delta))
        soft=sub[sub.method=="D0a_soft_oracle"].set_index("image_id").loc[base.index]
        hard=sub[sub.method=="D0c_hard_oracle"].set_index("image_id").loc[base.index]
        for metric in FIELDS:
            delta=(soft[metric]-hard[metric]).to_numpy()
            means=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
            paired.append(dict(dataset=ds,method="D0a_minus_D0c",metric=metric,delta_mean=float(delta.mean()),
                               ci95_low=float(np.quantile(means,.025)),ci95_high=float(np.quantile(means,.975))))
    write_csv(table / "oracle_paired_deltas.csv", paired)
    write_csv(table / "oracle_paired_per_image.csv", paired_images)
    boundary = pd.concat([pd.read_csv(out / "error_analysis" / f"d1_boundary_error_{ds}_per_image.csv") for ds in datasets], ignore_index=True)
    objects = pd.concat([pd.read_csv(out / "error_analysis" / f"d1_object_errors_{ds}_per_image.csv") for ds in datasets], ignore_index=True)
    bsum = []
    for (ds, band), g in boundary.groupby(["dataset","band"], sort=False):
        dataset = boundary[boundary.dataset == ds]
        bsum.append(dict(dataset=ds, band=band, images=g.image_id.nunique(), pixels=int(g.pixels.sum()),
                         fp=int(g.fp.sum()), fn=int(g.fn.sum()), error_share_micro=float(g.errors.sum()/max(1,dataset.errors.sum())),
                         fp_share_micro=float(g.fp.sum()/max(1,dataset.fp.sum())), fn_share_micro=float(g.fn.sum()/max(1,dataset.fn.sum())),
                         error_rate_micro=float(g.errors.sum()/max(1,g.pixels.sum())),
                         error_share_macro=float(g.error_share.mean()),fp_share_macro=float(g.fp_share.mean()),
                         fn_share_macro=float(g.fn_share.mean()),error_rate_macro=float(g.error_rate.mean())))
    write_csv(out / "error_analysis/d1_boundary_error_summary.csv", bsum)
    cumulative=[]
    for ds in datasets:
        part=boundary[boundary.dataset==ds]
        for name,bands in (("0-4",["0-4"]),("0-8",["0-4","4-8"]),("0-16",["0-4","4-8","8-16"]),
                           (">16",["16-32","32-inf"]),(" >32",["32-inf"])):
            selected=part[part.band.isin(bands)]
            cumulative.append(dict(dataset=ds,range=name.strip(),error_share_micro=float(selected.errors.sum()/max(1,part.errors.sum())),
                                   fp_share_micro=float(selected.fp.sum()/max(1,part.fp.sum())),fn_share_micro=float(selected.fn.sum()/max(1,part.fn.sum())),
                                   error_share_macro=float(selected.groupby("image_id").error_share.sum().mean())))
    write_csv(out/"error_analysis/d1_boundary_cumulative_summary.csv",cumulative)
    osum = objects.groupby("dataset", sort=False).agg({c:["sum","mean"] for c in objects.columns if c not in ("dataset","image_id")})
    osum.columns = [f"{a}_{b}" for a,b in osum.columns]
    write_csv(out / "error_analysis/d1_object_errors_summary.csv", osum.reset_index().to_dict("records"))
    figdir = out / "figures"; figdir.mkdir(exist_ok=True)
    order = ["0-4", "4-8", "8-16", "16-32", "32-inf"]
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for ax, ds in zip(axes,datasets):
        part = pd.DataFrame(bsum); part = part[part.dataset==ds].set_index("band").loc[order]
        ax.plot(order, part.fp_share_micro, "o-", label="FP share")
        ax.plot(order, part.fn_share_micro, "s-", label="FN share")
        ax.plot(order, part.error_rate_micro, "^-", label="Band error rate")
        ax.set_title(ds); ax.tick_params(axis="x",rotation=35); ax.set_ylim(bottom=0); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figdir/"d1_fp_fn_distance.png",dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for ax,ds in zip(axes,datasets):
        part=pd.DataFrame(bsum);part=part[part.dataset==ds].set_index("band").loc[order]
        ax.bar(order,part.error_share_micro);ax.set_title(ds);ax.tick_params(axis="x",rotation=35);ax.set_ylabel("Micro error share")
    fig.tight_layout();fig.savefig(figdir/"d1_error_share.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    cols=["unmatched_gt_no_overlap","unmatched_gt_partial","unmatched_pred_no_overlap","unmatched_pred_partial","merge_count"]
    xs=np.arange(len(cols));width=.35
    for i,ds in enumerate(datasets):
        part=objects[objects.dataset==ds]
        ax.bar(xs+(i-.5)*width,[part[c].sum() for c in cols],width,label=ds)
    ax.set_xticks(xs);ax.set_xticklabels(["GT no overlap","GT partial","Pred no overlap","Pred partial","Merge"],rotation=25,ha="right")
    ax.legend();fig.tight_layout();fig.savefig(figdir/"d1_object_errors.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(len(datasets),2,figsize=(10,4*len(datasets)))
    for row,ds in enumerate(datasets):
        part=pd.DataFrame(paired_images)
        part=part[(part.dataset==ds)&(part.method=="D0a_soft_oracle")]
        for col,metric in enumerate(("boundary_iou","object_hd")):
            vals=part[part.metric==metric].delta
            axes[row,col].hist(vals,bins=20)
            axes[row,col].axvline(0,color="black",lw=1)
            axes[row,col].set_title(f"{ds}: soft Oracle − B0 {metric}")
    fig.tight_layout();fig.savefig(figdir/"oracle_paired_boundary_object_hd.png",dpi=160);plt.close(fig)
    for ds in datasets:
        sub=baseline[(baseline.dataset==ds)&(baseline.method=="B0-native")]
        picks={"lowest_boundary_iou":sub.sort_values(["boundary_iou","image_id"]).iloc[0].image_id,
               "max_merge":objects[objects.dataset==ds].sort_values(["merge_count","image_id"],ascending=[False,True]).iloc[0].image_id,
               "max_fn":sub.sort_values(["fn","image_id"],ascending=[False,True]).iloc[0].image_id}
        by_id={e["id"]:e for e in get_entries(ds)}
        for tag,iid in picks.items():
            e=by_id[iid];image=load_image(e["image"]);gt=load_mask(e["mask"],DC).astype(bool)
            with np.load(out/"baseline"/ds/f"{iid}.npz") as z: margin=z["margin"]
            pred=margin>0;fp=pred&~gt;fn=~pred&gt
            overlay=image.copy();overlay[fp]=[255,0,0];overlay[fn]=[0,80,255]
            boundary_gt=edge(gt);dist=ndimage.distance_transform_edt(~boundary_gt) if boundary_gt.any() else np.full(gt.shape,np.inf)
            distance_band=np.digitize(dist,[4,8,16,32]).astype(np.uint8)
            fig,axes=plt.subplots(1,5,figsize=(15,3))
            for ax,arr,title,cmap in zip(axes,[image,gt,pred,overlay,distance_band],
                                          ["RGB","GT","Prediction","FP red / FN blue","GT distance bands"],
                                          [None,"gray","gray",None,"viridis"]):
                ax.imshow(arr,cmap=cmap);ax.set_title(title,fontsize=9);ax.axis("off")
            cb=fig.colorbar(axes[-1].images[0],ax=axes[-1],ticks=[0,1,2,3,4],fraction=.045)
            cb.ax.set_yticklabels(["0–4","4–8","8–16","16–32",">32"])
            fig.suptitle(f"{ds} {iid}: {tag}");fig.tight_layout();fig.savefig(figdir/f"d1_{ds}_{tag}_{iid}.png",dpi=130);plt.close(fig)
            fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(margin,cmap="coolwarm",vmin=-.2,vmax=.2);ax.set_title(f"{ds} {iid} native margin");ax.axis("off");fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(figdir/f"d1_{ds}_{tag}_{iid}_margin.png",dpi=130);plt.close(fig)
        iid=picks["lowest_boundary_iou"];e=by_id[iid];gt=load_mask(e["mask"],DC).astype(bool)
        with np.load(out/"baseline"/ds/f"{iid}.npz") as z: baseline_margin=z["margin"]
        with np.load(out/"oracle"/ds/f"{iid}_D0a_soft_oracle.npz") as z: soft_margin=z["margin"]
        with np.load(out/"oracle"/ds/f"{iid}_D0c_hard_oracle.npz") as z: hard_margin=z["margin"]
        fig,axes=plt.subplots(1,4,figsize=(12,3))
        for ax,arr,title in zip(axes,[gt,baseline_margin>0,soft_margin>0,hard_margin>0],
                                ["Native GT","B0 native","D0-a soft","D0-c hard"]):
            ax.imshow(arr,cmap="gray");ax.set_title(title);ax.axis("off")
        fig.suptitle(f"{ds} {iid}: lowest B0 Boundary IoU");fig.tight_layout();fig.savefig(figdir/f"oracle_{ds}_{iid}.png",dpi=130);plt.close(fig)
    multi_paths=[out/"multiphase"/f"{ds}_per_image.csv" for ds in datasets]
    multi=pd.concat([pd.read_csv(p) for p in multi_paths],ignore_index=True) if all(p.exists() for p in multi_paths) else None
    if multi is not None:
        for ds in datasets:
            expected=len(get_entries(ds))
            for count in (1,2,4,16):
                for method in ("bilinear_oracle","pcbp_no_prior","pcbp_jbu_rgb"):
                    sub=multi[(multi.dataset==ds)&(multi.phase_count==count)&(multi.method==method)]
                    if len(sub)!=expected or sub.image_id.nunique()!=expected:
                        raise ValueError(f"Incomplete multiphase {ds}/{count}/{method}")
        ms=multi.groupby(["dataset","phase_count","method"],sort=False)[list(FIELDS)+["observation_residual","reconstruction_seconds"]].agg(["mean","std"])
        ms.columns=[f"{a}_{b}" for a,b in ms.columns]
        write_csv(table/"multiphase_summary.csv",ms.reset_index().to_dict("records"))
        multi_paired=[]
        for (ds,count,method),g in multi.groupby(["dataset","phase_count","method"],sort=False):
            base=baseline[(baseline.dataset==ds)&(baseline.method=="B0-native")].set_index("image_id")
            for _,row in g.iterrows():
                for metric in FIELDS:
                    multi_paired.append(dict(dataset=ds,image_id=row.image_id,phase_count=count,method=method,metric=metric,
                                             delta_vs_b0=float(row[metric]-base.loc[row.image_id,metric])))
        write_csv(table/"multiphase_paired_vs_b0_per_image.csv",multi_paired)
        matched=[]
        for (ds,count),g in multi.groupby(["dataset","phase_count"]):
            bil=g[g.method=="bilinear_oracle"].set_index("image_id")
            for method in ("pcbp_no_prior","pcbp_jbu_rgb"):
                comp=g[g.method==method].set_index("image_id").loc[bil.index]
                for metric in FIELDS:
                    delta=(comp[metric]-bil[metric]).to_numpy()
                    means=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
                    matched.append(dict(dataset=ds,phase_count=count,method=method,metric=metric,
                                        delta_mean=float(delta.mean()),ci95_low=float(np.quantile(means,.025)),
                                        ci95_high=float(np.quantile(means,.975))))
        write_csv(table/"multiphase_paired_vs_matched_bilinear.csv",matched)
        residuals=pd.concat([pd.read_csv(out/"multiphase"/f"{ds}_residuals.csv") for ds in datasets],ignore_index=True)
        if not residuals.groupby(["dataset","image_id","phase_count"]).apply(
                lambda g: len(g)==9 and np.all(np.diff(g.sort_values("iteration").observation_residual.to_numpy())<=1e-9)).all():
            raise ValueError("PCBP residual did not decrease monotonically for all image/phase cases")
        fig,axes=plt.subplots(1,len(datasets),figsize=(5*len(datasets),4))
        if len(datasets)==1: axes=[axes]
        for ax,ds in zip(axes,datasets):
            part=residuals[residuals.dataset==ds]
            for count,g in part.groupby("phase_count"):
                curve=g.groupby("iteration").observation_residual.mean()
                ax.plot(curve.index,curve.values,marker="o",label=f"{count} phases")
            ax.set_title(ds);ax.set_xlabel("PCBP iteration");ax.set_ylabel("Mean observation MSE");ax.legend()
        fig.tight_layout();fig.savefig(figdir/"multiphase_residual_trajectory.png",dpi=160);plt.close(fig)
    lines=["# Native N3-bil diagnostics — seed 1337","",
           "## A. Experiment configuration","",
           "GLaS TestA+TestB is one logical 80-image Test set (60+20); CRAG official Test has 40 images. Test IDs are preserved. Historical 10% labeled support is 7 GLaS and 16 CRAG training images; the source study uses SSL seed 1337 and the fixed historical N3 memory. See `configs/source_metadata.json` for memory and checkpoint hashes. No test GT entered memory or retrieval.","",
           "Frozen UNI ViT-L/16; 448 window, 224 stride, 28×28 token grid; cosine class-wise Top-50; FP16 token margin before bilinear 448 reconstruction; mean overlap fusion, then native mean of diagonal offsets (0,0) and (8,8); threshold >0. Original N3 scalar offset shifts both axes. Native GT is read directly, and 256 GT uses nearest-neighbor. Object components are 8-connected, Hungarian match by GT coverage >0.5. Native minimum area is round(50×H×W/256²) pixels; no trustworthy µm/px metadata was found. Object Dice is a supplemental matched-region diagnostic. Boundary IoU retains the historical 2px inner-foreground-band definition; BF1 uses 2px and 4px tolerances.","",
           "Oracle occupancy averages valid original-image pixels in each token box. Fully padded boxes receive neutral occupancy 0.5 for bilinear fusion and are omitted from PCBP observations.","",
           "## B. Native vs 256 baseline","",
           "| Dataset | Method | n | Dice | Boundary IoU | Object F1 | Object HD |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for ds in datasets:
        for method in ("B0-native","B0-native256","B0-legacy256"):
            r=summary[(summary.dataset==ds)&(summary.method==method)].iloc[0]
            lines.append(f"| {display[ds]} | {method} | {len(get_entries(ds))} | {r.dice_mean:.4f} | {r.boundary_iou_mean:.4f} | {r.object_f1_mean:.4f} | {r.object_hd_mean:.2f} |")
    lines += ["","Native→256 uses the same native margin. Legacy256 is the saved original pipeline and was checked for exact mask parity on every image. The 2px boundary band becomes physically narrower at native size, and Object-HD is measured in output pixels; their native/256 values are not directly comparable without scale normalization. Differences among the three cannot all be attributed to downsampling.","",
              "## C. D0-a and D0-c token Oracle","",
              "| Dataset | Method | Dice mean±SD | FG IoU | Boundary IoU | Object F1 | Object HD | BF1 2px | BF1 4px |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    oracle_notes=[]
    for ds in datasets:
        for method in ("B0-native","D0a_soft_oracle","D0c_hard_oracle"):
            r=summary[(summary.dataset==ds)&(summary.method==method)].iloc[0]
            lines.append(f"| {display[ds]} | {method} | {r.dice_mean:.4f}±{r.dice_std:.4f} | {r.fg_iou_mean:.4f} | {r.boundary_iou_mean:.4f} | {r.object_f1_mean:.4f} | {r.object_hd_mean:.2f} | {r['bf1_2px_mean']:.4f} | {r['bf1_4px_mean']:.4f} |")
        b=summary[(summary.dataset==ds)&(summary.method=="B0-native")].iloc[0]
        a=summary[(summary.dataset==ds)&(summary.method=="D0a_soft_oracle")].iloc[0]
        c=summary[(summary.dataset==ds)&(summary.method=="D0c_hard_oracle")].iloc[0]
        oracle_notes.append(f"{display[ds]}: Δ(soft−B0) Dice={a.dice_mean-b.dice_mean:+.4f}, Boundary IoU={a.boundary_iou_mean-b.boundary_iou_mean:+.4f}; Δ(soft−hard) Dice={a.dice_mean-c.dice_mean:+.4f}; residual Oracle gap 1−soft Dice={1-a.dice_mean:.4f}, 1−soft Boundary IoU={1-a.boundary_iou_mean:.4f}. Paired image bootstrap intervals and Object-HD deltas are in `tables/oracle_paired_deltas.csv`.")
    lines += [""] + oracle_notes
    lines += ["","The soft−B0 gap is the benefit of GT occupancy versus real retrieval margins, not a pure token-classification error. Soft−hard measures continuous occupancy under this Oracle. The remaining 1−soft gap is not necessarily recoverable by upsampling.","",
              "## D. Native error decomposition","",
              "Bands are disjoint [0,4), [4,8), [8,16), [16,32), [32,∞) pixels from the GT boundary. Micro pools pixel counts; macro averages per-image shares/rates. All band FP and FN totals equal full-image FP and FN.",""]
    for ds in datasets:
        part=pd.DataFrame(bsum);part=part[part.dataset==ds].set_index("band").loc[order]
        within4=part.iloc[0].error_share_micro
        within8=part.iloc[:2].fp.add(part.iloc[:2].fn).sum()/max(1,part.fp.add(part.fn).sum())
        within16=part.iloc[:3].fp.add(part.iloc[:3].fn).sum()/max(1,part.fp.add(part.fn).sum())
        beyond16=1-within16
        beyond32=part.iloc[-1].error_share_micro
        op=objects[objects.dataset==ds]
        lines.append(f"{display[ds]}: micro error shares 0–4px {within4:.3%}, 0–8px {within8:.3%}, 0–16px {within16:.3%}, >16px {beyond16:.3%}, >32px {beyond32:.3%}; unmatched GT (no overlap/partial) {op.unmatched_gt_no_overlap.sum()}/{op.unmatched_gt_partial.sum()}; unmatched predictions {op.unmatched_pred_no_overlap.sum()}/{op.unmatched_pred_partial.sum()}; merges {op.merge_count.sum()}. Detailed micro and macro rates, object counts, and deterministic cases are in `error_analysis/` and `figures/`.")
    lines += ["","Merge links require ≥max(10, round(native area threshold×0.2)) overlap pixels and ≥10% of the smaller object. Narrow bridges are not classified because a stable criterion was unavailable. Images may share WSI origin; a reliable WSI identifier was unavailable, so image bootstrap intervals are not independent-case intervals.","",
              "## E. Multi-phase Oracle","",phase_status.replace("GlaS:","GLaS-Test:").replace("CRAG:","CRAG-Test:")]
    if multi is not None:
        lines += ["","| Dataset | Phases | Method | Dice | Boundary IoU | BF1 2px | Object F1 | Object HD | Residual | Time s/image |",
                  "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
        for (ds,count,method),g in multi.groupby(["dataset","phase_count","method"],sort=False):
            m=g.mean(numeric_only=True)
            lines.append(f"| {display[ds]} | {count} | {method} | {m.dice:.4f} | {m.boundary_iou:.4f} | {m['bf1_2px']:.4f} | {m.object_f1:.4f} | {m.object_hd:.2f} | {m.observation_residual:.6g} | {m.reconstruction_seconds:.3f} |")
        lines += ["","Every PCBP comparison is paired to the Bilinear Oracle using the same phase set. Observation residual trajectories are saved per image and plotted in `figures/multiphase_residual_trajectory.png`; matched metric deltas with image bootstrap intervals are in `tables/multiphase_paired_vs_matched_bilinear.csv`. Lower residual is interpreted separately from segmentation quality."]
    lines += ["","## F. Research decision","" ]
    if multi is not None:
        soft=summary[summary.method=="D0a_soft_oracle"].set_index("dataset")
        base=summary[summary.method=="B0-native"].set_index("dataset")
        retrieval=float((soft.dice_mean-base.dice_mean).mean())
        matched_frame=pd.DataFrame(matched)
        improve=matched_frame[(matched_frame.metric=="boundary_iou")&(matched_frame.phase_count==16)]
        no_prior=float(improve[improve.method=="pcbp_no_prior"].delta_mean.mean())
        jbu=float(improve[improve.method=="pcbp_jbu_rgb"].delta_mean.mean())
        reconstruction=max(no_prior,jbu)
        direction="C: Both" if retrieval>.02 and reconstruction>.01 else ("B: Upsampling / Reconstruction First" if reconstruction>.01 else "A: Prototype / Retrieval First")
        lines.append(f"{direction}. Mean soft-Oracle Dice gain over B0={retrieval:+.4f}; 16-phase matched Boundary-IoU gain averaged across datasets: no-prior PCBP={no_prior:+.4f}, PCBP-JBU-RGB={jbu:+.4f}. The no-prior result is heterogeneous across datasets; RGB-guided gain also includes image-prior information. This is a diagnostic priority, not a deployable performance claim. A lower observation residual without segmentation gain is not evidence of successful reconstruction.")
        lines += ["", "**Q1.** The large paired B0→soft-Oracle Dice gain identifies token-observation quality as the main Dice bottleneck in this fixed setup. The Oracle Boundary-IoU gap and D1 boundary-local errors show a separate spatial reconstruction limit; the >32px error share also indicates nonlocal semantic failures.",""]
        for ds in datasets:
            g=matched_frame[(matched_frame.dataset==ds)&(matched_frame.phase_count==16)&(matched_frame.method=="pcbp_no_prior")&(matched_frame.metric=="boundary_iou")]
            if len(g):
                r=g.iloc[0]
                j=matched_frame[(matched_frame.dataset==ds)&(matched_frame.phase_count==16)&(matched_frame.method=="pcbp_jbu_rgb")&(matched_frame.metric=="boundary_iou")].iloc[0]
                lines.append(f"**Q2, {display[ds]}.** 16-phase no-prior PCBP−matched Bilinear Oracle Boundary IoU={r.delta_mean:+.4f} (image bootstrap 95% CI {r.ci95_low:+.4f} to {r.ci95_high:+.4f}); PCBP-JBU-RGB={j.delta_mean:+.4f} (CI {j.ci95_low:+.4f} to {j.ci95_high:+.4f}). The no-prior comparison tests spatial information in ideal GT occupancy; JBU also uses an RGB-derived image prior. Neither establishes a gain for retrieved UNI margins.")
        lines += ["","Next: use a 2×2 paired factorial study with the same support, images, and fixed threshold: current versus improved token retrieval (D2 occupancy calibration, then P1 soft memory/P2 debiasing), crossed with current bilinear versus the predeclared 16-phase PCBP-JBU reconstruction. Report interaction as well as main effects; confirm any Oracle advantage with real UNI margins before proposing a deployable method."]
    else:
        lines.append("Research priority is based on B0, Oracle, and D1 only; multi-phase reconstruction was conditionally skipped.")
    (out/"diagnostic_report.md").write_text("\n".join(lines)+"\n")


def main():
    global SOURCE, LEGACY, DATA_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("GlaS", "CRAG", "all"), default="all")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--output-size", choices=("native", "256"), default="native")
    ap.add_argument("--save-margin", choices=("true", "false"), default="true")
    ap.add_argument("--stage", choices=("baseline", "oracle", "error-analysis", "multiphase", "report", "all"), default="all")
    ap.add_argument("--phase-count", type=int, choices=(1, 2, 4, 16), nargs="+")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dataset-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--phase-margin-dir", type=Path, default=SOURCE)
    ap.add_argument("--legacy-case-dir", type=Path, default=LEGACY)
    args = ap.parse_args()
    if args.seed != 1337 or args.output_size != "native" or args.save_margin != "true":
        raise ValueError("This fixed diagnostic requires seed 1337, native output and saved margins")
    SOURCE = args.phase_margin_dir.resolve(); LEGACY = args.legacy_case_dir.resolve(); DATA_ROOT = args.dataset_root.resolve()
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    datasets = ("GlaS", "CRAG") if args.dataset == "all" else (args.dataset,)
    meta = json.loads((SOURCE.parent / "run_metadata.json").read_text())
    if meta.get("status") != "COMPLETE": raise ValueError("Source phase-margin study is incomplete")
    (out / "configs").mkdir(exist_ok=True)
    (out / "configs" / "source_metadata.json").write_text(json.dumps(meta, indent=2))
    for ds in datasets:
        source_cfg = LEGACY.parent / "configs" / f"{ds}.json"
        (out / "configs" / f"{ds}_fixed_source.json").write_bytes(source_cfg.read_bytes())
    (out / "configs" / "protocol.json").write_text(json.dumps(dict(seed=1337, source=str(SOURCE), phases=PHASES,
        gt_occupancy="valid source pixels only; padded pixels excluded; fully padded tokens neutral 0.5 and omitted from PCBP", object_area="round(50*H*W/65536)",
        object_match="Hungarian maximum GT coverage > 0.5", merge_link="intersection>=max(10,round(area*.2)) and >=0.1*min object area",
        threshold=0, boundary_iou="historical inner foreground band radius 2", boundary_f1=[2,4]), indent=2))
    all_baseline, all_oracle, all_boundary, all_objects = [], [], [], []
    for ds in datasets:
        entries = get_entries(ds)
        if args.stage in ("baseline", "all"):
            all_baseline += baseline(ds, entries, out)
            write_csv(out / "tables" / "baseline_native_vs_256.csv", all_baseline)
        if args.stage in ("oracle", "all"):
            all_oracle += oracle(ds, entries, out)
            write_csv(out / "tables" / "oracle_per_image.csv", all_oracle)
        if args.stage in ("error-analysis", "all"):
            b, o = error_analysis(ds, entries, out); all_boundary += b; all_objects += o
            write_csv(out / "error_analysis" / "d1_boundary_error_per_image.csv", all_boundary)
            write_csv(out / "error_analysis" / "d1_object_errors_per_image.csv", all_objects)
    phase_status = "Not requested in this run."
    if args.stage in ("multiphase", "all"):
        import pandas as pd
        go = False
        reasons = []
        for ds in datasets:
            oracle_table = pd.read_csv(out / "tables" / f"oracle_{ds}_per_image.csv")
            soft = oracle_table[oracle_table.method == "D0a_soft_oracle"]
            b = pd.read_csv(out / "error_analysis" / f"d1_boundary_error_{ds}_per_image.csv")
            near = b[b.band.isin(("0-4","4-8","8-16"))].errors.sum() / max(1,b.errors.sum())
            gap_b = 1-soft.boundary_iou.mean(); gap_d=1-soft.dice.mean()
            selected = gap_b >= .05 or gap_d >= .02
            go |= selected
            reasons.append(f"{ds}: Oracle Boundary-IoU gap={gap_b:.4f}, Dice gap={gap_d:.4f}, B0 errors within 16px={near:.3%}; {'execute' if selected else 'skip'}")
        phase_status = "; ".join(reasons)
        if go:
            counts = args.phase_count or [1,2,4,16]
            for ds in datasets:
                multiphase(ds, get_entries(ds), out, counts)
            phase_status += ". PCBP uses exact valid-pixel 16×16 box means, their weighted adjoint, box-coverage preconditioning, projection to [0,1], eight fixed iterations, and monotone backtracking. RGB-derived grayscale guided filtering uses historical radius 4 and epsilon 0.001. No PSF or test-tuned prior. Peak GPU allocation is zero (CPU reconstruction)."
        else:
            phase_status += ". Skipped: both Oracle gaps are below the preregistered thresholds."
        (out / "multiphase").mkdir(exist_ok=True)
        (out / "multiphase/status.txt").write_text(phase_status + "\n")
    if args.stage in ("all", "report") and args.dataset == "all":
        if args.stage == "report":
            p = out / "multiphase/status.txt"
            phase_status = p.read_text() if p.exists() else "Multi-phase status unavailable."
            phase_status = phase_status.replace("RGB guided filtering", "RGB-derived grayscale guided filtering")
        summarize(out, datasets, phase_status)
    print("Completed requested stages", flush=True)


if __name__ == "__main__": main()
