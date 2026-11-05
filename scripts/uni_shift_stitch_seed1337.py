"""Fixed UNI shift-and-stitch study, seed 1337. Resumable at image granularity."""
from __future__ import annotations

import argparse, csv, hashlib, json, math, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/uni_shift_stitch_seed1337"
OLD = ROOT / "outputs/dense_feature_matching_seed1337"
PHASES_2 = [(0, 0), (8, 8)]
PHASES_4 = [(dy, dx) for dy in (0, 8) for dx in (0, 8)]
PHASES_16 = [(dy, dx) for dy in (0, 4, 8, 12) for dx in (0, 4, 8, 12)]
BOOT_SEED, BOOT_N = 2026, 10000


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r)) or ["status"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)


def dump_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str)); tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", choices=("GlaS", "CRAG"), default=("GlaS", "CRAG"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "configs").mkdir(exist_ok=True)
    (OUT / "execution.log").touch(exist_ok=True)
    def log(msg):
        line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
        print(line, flush=True)
        with (OUT / "execution.log").open("a") as f: f.write(line + "\n")

    try:
        import cv2
        from pathology_trainfree.data import load_image, load_mask, read_manifest, resize_mask
        from pathology_trainfree.encoder import FeatureEncoder, sha256
        from pathology_trainfree.geometry import reconstruct, resize_map
        from pathology_trainfree.memory import ReferenceMemory
        from pathology_trainfree.retrieval import SimilarityRetriever
        from pathology_trainfree.shift_stitch import windows_xy, dense_stitch, token_centers, phase_grid
        from dense_feature_matching import Study, evaluate, DC, WEIGHTS

        torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        assert PHASES_2 == [(0, 0), (8, 8)]
        assert len(set(PHASES_4)) == 4 and len(set(PHASES_16)) == 16
        assert set(PHASES_2).issubset(PHASES_4) and set(PHASES_4).issubset(PHASES_16)
        source_code = Path(__file__).read_bytes()
        protocol = {"seed": 1337, "window": 448, "stride": 224, "patch": 16, "grid": 28,
                    "s0": PHASES_2, "s1": PHASES_4, "s2_s3": PHASES_16, "k": 50,
                    "margin_round": "fp16 before bilinear window reconstruction",
                    "threshold": 0, "output": [256, 256], "bootstrap_seed": BOOT_SEED,
                    "bootstrap_replicates": BOOT_N, "dense_pitch": 4,
                    "dense_semantics": "16px patch and context; 4px nominal observation pitch is not 4x4 independent semantic resolution"}
        dump_json(OUT / "configs/protocol.json", protocol)
        protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        weight_path = WEIGHTS
        if not weight_path.exists() or sha256(weight_path) != weight_path.name:
            raise FileNotFoundError(f"Expected frozen UNI checkpoint missing or SHA mismatch: {weight_path}")

        # Existing validated class-wise historical N3 Memory; loaded read-only.
        base = Study()
        meta = {"status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(),
                "protocol": protocol, "protocol_sha256": protocol_hash,
                "uni_checkpoint": str(weight_path), "uni_sha256": sha256(weight_path),
                "source_script_sha256": hashlib.sha256(source_code).hexdigest(),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
                "datasets": args.datasets, "support_memory_sha256": {d: sha(base.data[d]["source"]) for d in args.datasets}}
        dump_json(OUT / "run_metadata.json", meta)

        # For every phase the cache key binds image bytes, checkpoint, preprocessing,
        # geometry, precision and physical-coordinate algorithm version.
        cfg = base.cfg.copy(); cfg["encoder"] = cfg.get("encoder", {}).copy()
        cfg["encoder"].update(name="uni", weights=str(weight_path), device=args.device,
                               batch_size=2, precision="float32", token_precision="float16", normalize=True)
        encoder = FeatureEncoder(cfg["encoder"])
        retrievers = {d: base.data[d]["retriever"] for d in args.datasets}
        all_phase_rows, phase_count_rows, cache_rows, timing_rows, predictions = [], [], [], [], {}
        phase_consistency_rows, margin_delta_rows = [], []
        GT_PATHS = {}

        def phase_cache(ds, entry, image, phase):
            dy, dx = phase; image_digest = hashlib.sha256(image.tobytes()).hexdigest()
            key_obj = {"image_sha256": image_digest, "weights_sha256": encoder.weights_sha256,
                       "preprocess": encoder.input_normalization, "window": 448, "stride": 224,
                       "padding": "reflect", "offset_y": dy, "offset_x": dx,
                       "precision": encoder.precision, "token_precision": encoder.token_precision,
                       "grid": [28, 28], "feature_dim": 1024, "coordinates": "window-origin+patch-center-reflect101-v1"}
            key = hashlib.sha256(json.dumps(key_obj, sort_keys=True).encode()).hexdigest()
            p = OUT / "feature_cache" / ds / f"{entry['id']}_dy{dy}_dx{dx}.npz"
            if p.exists() and not args.overwrite:
                t0 = time.perf_counter()
                with np.load(p, allow_pickle=False) as z:
                    if str(z["cache_key"].item()) != key: raise ValueError(f"Feature cache key mismatch: {p}")
                    tokens, positions = z["tokens"].copy(), z["positions"].copy()
                cache_seconds = time.perf_counter() - t0; extraction_seconds = 0.0; cache_hit = True
            else:
                cache_seconds = 0.0
                # Keep the historical 0/8 diagonal features byte-for-byte when possible.
                legacy = None
                if dy == dx and dy in (0, 8):
                    lp = ROOT / f"outputs/support_size_study/cache/query_features/{ds}/feature_cache/{entry['id']}_offset{dy}.npz"
                    if lp.exists():
                        legacy_t0 = time.perf_counter()
                        with np.load(lp, allow_pickle=False) as z:
                            legacy_key = hashlib.sha256(image.tobytes() + str((encoder.weights_sha256, 448, 224, dy, "float32", "float16")).encode()).hexdigest()
                            if str(z["cache_key"].item()) == legacy_key:
                                legacy = (z["tokens"].copy(), z["positions"].copy())
                        if legacy is not None: cache_seconds = time.perf_counter() - legacy_t0
                expected_views = list(windows_xy(image, 448, 224, phase, "reflect"))
                positions = np.asarray([(y, x) for y, x, _ in expected_views], dtype=np.int64)
                if legacy is not None:
                    tokens, old_pos = legacy
                    if np.array_equal(old_pos, positions):
                        extraction_seconds = 0.0; cache_hit = True
                    else: legacy = None
                if legacy is None:
                    t0 = time.perf_counter()
                    tokens = encoder.encode([crop for _, _, crop in expected_views])
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    extraction_seconds = time.perf_counter() - t0; cache_seconds = 0.0; cache_hit = False
                if tokens.shape != (len(positions), 784, 1024) or tokens.dtype != np.float16:
                    raise ValueError(f"Unexpected feature tensor for {entry['id']} phase {phase}: {tokens.shape} {tokens.dtype}")
                p.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(p, tokens=tokens, positions=positions, cache_key=np.array(key),
                                    key_json=np.array(json.dumps(key_obj, sort_keys=True)))
            if not np.array_equal(positions, np.asarray([(y, x) for y, x, _ in windows_xy(image, 448, 224, phase, "reflect")])):
                raise ValueError("Cached window coordinate mismatch")
            if tokens.shape != (len(positions), 784, 1024): raise ValueError("Cached feature shape mismatch")
            cache_rows.append({"dataset": ds, "image_id": entry["id"], "dy": dy, "dx": dx,
                               "path": str(p), "cache_key": key, "cache_hit": cache_hit,
                               "windows": len(positions), "tokens": int(tokens.shape[0] * tokens.shape[1]),
                               "sha256": sha(p)})
            return tokens, positions, extraction_seconds, cache_seconds

        for ds in args.datasets:
            st = base.data[ds]
            for index, entry in enumerate(st["queries"]):
                image = load_image(entry["image"])
                GT_PATHS[(ds, entry["id"])] = entry["mask"]
                phase_margins, phase_positions = {}, {}
                inference_times = {"feature_extraction_seconds": 0., "cache_read_seconds": 0.,
                                   "knn_seconds": 0., "reconstruction_seconds": 0.}
                phase_rows = []
                for phase in PHASES_16:
                    dy, dx = phase
                    tokens, positions, fe, cr = phase_cache(ds, entry, image, phase)
                    inference_times["feature_extraction_seconds"] += fe
                    inference_times["cache_read_seconds"] += cr
                    t0 = time.perf_counter()
                    scores = retrievers[ds].scores(tokens).numpy()
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    inference_times["knn_seconds"] += time.perf_counter() - t0
                    raw_margin = (scores[:, 1] - scores[:, 0]).reshape(-1, 28, 28)
                    # Original N3 margin precision: fp16 before bilinear reconstruction.
                    raw_margin = raw_margin.astype(np.float16)
                    phase_margins[phase], phase_positions[phase] = raw_margin, positions
                    xs, ys = token_centers(positions, 28, 16)
                    phfile = OUT / "phase_margins" / ds / f"{entry['id']}.npz"
                    phase_rows.append((phase, raw_margin, positions, xs, ys))
                    phase_count_rows.append({"dataset": ds, "image_id": entry["id"], "dy": dy, "dx": dx,
                                             "window_tokens": int(raw_margin.size), "windows": len(positions),
                                             "unique_raw_centers": int(len(np.unique(np.stack([xs.ravel(), ys.ravel()], axis=1), axis=0)))})
                # Persist all 16 exact margin grids and raw + reflected coordinates.
                phfile.parent.mkdir(parents=True, exist_ok=True)
                payload = {}
                for phase, margin, pos, xs, ys in phase_rows:
                    suffix = f"dy{phase[0]}_dx{phase[1]}"
                    payload[f"margin_{suffix}"] = margin
                    payload[f"positions_{suffix}"] = pos
                    payload[f"center_x_{suffix}"] = xs.astype(np.float32)
                    payload[f"center_y_{suffix}"] = ys.astype(np.float32)
                np.savez_compressed(phfile, **payload)

                outputs = {}; phase_pixel_maps = {}; phase_native_maps = {}
                # S0 is the saved original baseline; do not re-run it.
                with np.load(OLD / "cases" / ds / f"{entry['id']}_B0.npz", allow_pickle=False) as z:
                    outputs["S0"] = z["margin"].astype(np.float32)

                t0 = time.perf_counter()
                for phase in PHASES_16:
                    native_phase = reconstruct(phase_margins[phase].astype(np.float32),
                        phase_positions[phase], image.shape[:2], 448, "float16")
                    phase_native_maps[phase] = native_phase
                    phase_pixel_maps[phase] = resize_map(native_phase, (256, 256)).astype(np.float32)
                # Preserve original order: window fusion per phase -> phase mean at native scale -> 256 resize.
                outputs["S1"] = resize_map(np.mean(np.stack([phase_native_maps[p] for p in PHASES_4]), axis=0, dtype=np.float32), (256, 256)).astype(np.float32)
                outputs["S2"] = resize_map(np.mean(np.stack([phase_native_maps[p] for p in PHASES_16]), axis=0, dtype=np.float32), (256, 256)).astype(np.float32)
                inference_times["reconstruction_seconds"] += time.perf_counter() - t0

                t0 = time.perf_counter()
                native_s3, dense256, dense_diag, dense_grid = dense_stitch(phase_margins, phase_positions,
                    PHASES_16, image.shape[:2], output_size=(256, 256))
                outputs["S3"] = dense256
                inference_times["reconstruction_seconds"] += time.perf_counter() - t0
                densepath = OUT / "dense_grids" / ds / f"{entry['id']}.npz"; densepath.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(densepath, y=dense_grid[0], x=dense_grid[1], margin=dense_grid[2], count=dense_grid[3], native_margin=native_s3)
                for p1, p2 in zip(PHASES_16[:-1], PHASES_16[1:]):
                    a, b = phase_pixel_maps[p1], phase_pixel_maps[p2]
                    phase_consistency_rows.append({"dataset": ds, "image_id": entry["id"],
                        "phase_a": str(p1), "phase_b": str(p2), "paired_token_count": int(a.size),
                        "margin_abs_difference_mean": float(np.abs(a-b).mean()),
                        "margin_pearson": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.std() and b.std() else None})
                margin_delta_rows.append({"dataset": ds, "image_id": entry["id"],
                    "s2_s3_mae": float(np.abs(outputs["S2"]-outputs["S3"]).mean()),
                    "s2_s3_rmse": float(np.sqrt(np.square(outputs["S2"]-outputs["S3"]).mean())),
                    "s2_s3_max_abs": float(np.abs(outputs["S2"]-outputs["S3"]).max()),
                    "s2_s3_sign_flip_ratio": float(((outputs["S2"] > 0) != (outputs["S3"] > 0)).mean())})
                for method, margin in outputs.items():
                    if margin.shape != (256, 256) or not np.isfinite(margin).all(): raise ValueError(f"Bad {method} margin {entry['id']}")
                    pred_path = OUT / "predictions" / ds / f"{entry['id']}_{method}.npz"; pred_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(pred_path, margin=margin, mask=(margin > 0).astype(np.uint8))
                    predictions[(ds, entry["id"], method)] = margin
                # Use the best available feature timing; encoder execution once per missing phase.
                timing_rows.append({"dataset": ds, "image_id": entry["id"], **inference_times,
                                    "total_query_seconds_including_new_extraction": sum(inference_times.values()),
                                    "cache_based_query_seconds": inference_times["cache_read_seconds"] + inference_times["knn_seconds"] + inference_times["reconstruction_seconds"],
                                    "new_phase_uni_forwards": int(sum(1 for r in cache_rows if r["dataset"] == ds and r["image_id"] == entry["id"] and not r["cache_hit"])),
                                    "phase_forward_count_total": 16, "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
                                    **{f"dense_{k}": v for k, v in dense_diag.items()}})
                log(f"INFER {ds} {index+1}/{len(st['queries'])} {entry['id']}")

        # Separate evaluation stage. GT is first read here, after every prediction is saved.
        metric_rows, boundary_rows, object_rows, flip_rows = [], [], [], []
        from scipy import ndimage
        from dense_feature_matching import edge
        for ds in args.datasets:
            for entry in base.data[ds]["queries"]:
                gt = resize_mask(load_mask(entry["mask"], DC), (256, 256)).astype(bool)
                per_method = {}
                for method in ("S0", "S1", "S2", "S3"):
                    margin = predictions[(ds, entry["id"], method)]; row = {"dataset": ds, "image_id": entry["id"], "method": method, **evaluate(margin, gt)}
                    metric_rows.append(row); per_method[method] = margin > 0
                    boundary_rows.append({"dataset": ds, "image_id": entry["id"], "method": method,
                        **{k: row[k] for k in row if k.endswith("_pixels") or k.endswith("_errors") or k.endswith("_fg_to_bg") or k.endswith("_bg_to_fg") or k.startswith("boundary_")}})
                    object_rows.append({"dataset": ds, "image_id": entry["id"], "method": method,
                        **{k: row[k] for k in row if "object" in k or "gland" in k or "merge" in k or "split" in k}})
                gt_edge = edge(gt); dist = ndimage.distance_transform_edt(~gt_edge) if gt_edge.any() else np.full(gt.shape, np.inf)
                strata = {"boundary_0_2": dist <= 2, "boundary_2_5": (dist > 2) & (dist <= 5), "interior_gt_5": dist > 5}
                for a, b in (("S1", "S0"), ("S2", "S0"), ("S3", "S0"), ("S3", "S2")):
                    pa, pb = per_method[a], per_method[b]; changed = pa != pb
                    row = {"dataset": ds, "image_id": entry["id"], "comparison": f"{a}-{b}",
                           "corrected_pixels": int(((pb != gt) & (pa == gt)).sum()),
                           "damaged_pixels": int(((pb == gt) & (pa != gt)).sum()),
                           "fg_to_bg": int((pb & ~pa).sum()), "bg_to_fg": int((~pb & pa).sum()),
                           "both_wrong": int(((pa != gt) & (pb != gt)).sum()),
                           "margin_sign_change_ratio": float(changed.mean()), "pixels": int(changed.size)}
                    for label, sel in strata.items():
                        corrected = (pb != gt) & (pa == gt) & sel; damaged = (pb == gt) & (pa != gt) & sel
                        row[label + "_total_pixels"] = int(sel.sum()); row[label + "_corrected"] = int(corrected.sum()); row[label + "_damaged"] = int(damaged.sum()); row[label + "_net"] = int(corrected.sum() - damaged.sum())
                    flip_rows.append(row)
        write_csv(OUT / "per_image_metrics.csv", metric_rows)
        write_csv(OUT / "boundary_analysis.csv", boundary_rows)
        write_csv(OUT / "object_analysis.csv", object_rows)
        write_csv(OUT / "decision_flip_analysis.csv", flip_rows)
        write_csv(OUT / "sampling_coverage.csv", phase_count_rows)
        write_csv(OUT / "feature_cache_manifest.csv", cache_rows)
        write_csv(OUT / "runtime_memory.csv", timing_rows)
        write_csv(OUT / "margin_consistency.csv", phase_consistency_rows + margin_delta_rows)

        # Paired image bootstrap within each dataset; vectors are sampled by image.
        metric_names = ["dice", "fg_iou", "object_f1", "object_precision", "object_recall", "boundary_f1", "boundary_iou", "small_gland_recall", "merge_count", "split_count", "predicted_fg_ratio"]
        summary, effects = [], []
        rng = np.random.default_rng(BOOT_SEED)
        lookup = {(r["dataset"], r["image_id"], r["method"]): r for r in metric_rows}
        for ds in args.datasets:
            ids = [e["id"] for e in base.data[ds]["queries"]]
            for method in ("S0", "S1", "S2", "S3"):
                vals = [lookup[(ds, iid, method)] for iid in ids]
                summary.append({"dataset": ds, "method": method, "images": len(ids), **{m: float(np.nanmean([v[m] if v[m] is not None else np.nan for v in vals])) for m in metric_names}})
            for a, b in (("S1", "S0"), ("S2", "S1"), ("S3", "S2"), ("S2", "S0"), ("S3", "S0")):
                for metric in metric_names:
                    paired_ids = [iid for iid in ids if lookup[(ds, iid, a)][metric] is not None and lookup[(ds, iid, b)][metric] is not None]
                    deltas = np.asarray([lookup[(ds, iid, a)][metric] - lookup[(ds, iid, b)][metric] for iid in paired_ids], np.float64)
                    if len(deltas) == 0:
                        effects.append({"dataset": ds, "comparison": f"{a}-{b}", "metric": metric, "n_images": 0, "paired_delta_mean": None, "ci95_low": None, "ci95_high": None, "improved_images": 0, "worsened_images": 0, "tied_images": 0, "bootstrap_seed": BOOT_SEED, "bootstrap_n": BOOT_N}); continue
                    sample_ix = rng.integers(0, len(paired_ids), size=(BOOT_N, len(paired_ids)))
                    boot = deltas[sample_ix].mean(axis=1)
                    effects.append({"dataset": ds, "comparison": f"{a}-{b}", "metric": metric,
                                    "paired_delta_mean": float(deltas.mean()), "ci95_low": float(np.quantile(boot, .025)),
                                    "ci95_high": float(np.quantile(boot, .975)), "improved_images": int((deltas > 0).sum()),
                                    "worsened_images": int((deltas < 0).sum()), "tied_images": int((deltas == 0).sum()),
                                    "n_images": len(paired_ids), "bootstrap_seed": BOOT_SEED, "bootstrap_n": BOOT_N})
        write_csv(OUT / "summary.csv", summary); write_csv(OUT / "paired_effects.csv", effects)
        # Compare against frozen, previously recorded baseline values and masks.
        parity = []
        for ds in args.datasets:
            oldcfg = json.loads((OLD / f"configs/{ds}.json").read_text())
            hist = {r["image_id"]: r for r in csv.DictReader((OLD / "baseline_parity_per_image.csv").open()) if r["dataset"] == ds}
            for e in base.data[ds]["queries"]:
                pred = (predictions[(ds, e["id"], "S0")] > 0).astype(np.uint8)
                saved = np.load(OLD / "cases" / ds / f"{e['id']}_B0.npz")["mask"]
                parity.append({"dataset": ds, "image_id": e["id"], "s0_dice_historical": hist[e["id"]]["dice"],
                               "mask_disagreement_pixels": int((pred != saved).sum()), "historical_memory_sha256": oldcfg["memory_sha256"]})
        write_csv(OUT / "baseline_parity_per_image.csv", parity)
        if any(r["mask_disagreement_pixels"] for r in parity): raise RuntimeError("S0 mask parity failed vs validated historical output")
        baseline_lines = ["# S0 baseline parity", "", "S0 uses the original N3 cache-based retrieval and margin reconstruction, with historical support memory read-only.", ""]
        for ds in args.datasets:
            rr = [r for r in parity if r["dataset"] == ds]
            hist = [float(r["s0_dice_historical"]) for r in rr]
            baseline_lines.append(f"{ds}: {len(rr)} images; saved masks agree pixelwise={sum(r['mask_disagreement_pixels'] for r in rr)==0}; disagreement pixels={sum(r['mask_disagreement_pixels'] for r in rr)}; historical macro Dice={np.mean(hist):.12f}.")
        (OUT / "baseline_parity.md").write_text("\n".join(baseline_lines) + "\n")
        failure_rows = []
        for ds in args.datasets:
            for iid in [e["id"] for e in base.data[ds]["queries"]]:
                delta = lookup[(ds, iid, "S3")]["dice"] - lookup[(ds, iid, "S0")]["dice"]
                failure_rows.append({"dataset": ds, "image_id": iid, "s3_minus_s0_dice": delta,
                    "s0_dice": lookup[(ds, iid, "S0")]["dice"], "s3_dice": lookup[(ds, iid, "S3")]["dice"],
                    "case_role": "improvement" if delta > 0 else "degradation" if delta < 0 else "tie"})
        write_csv(OUT / "failure_cases.csv", failure_rows)
        report = ["# True UNI Shift-and-Stitch — Seed 1337 全量结果", "",
                  "## 方法与假设", "", "S0 沿用历史 `(0,0)/(8,8)` 原始Margin重建；S1使用`{0,8}²`四个真实二维相位；S2/S3共享16个真实UNI相位Margin，区别仅在Margin重建。UNI patch仍为16×16且保留Transformer上下文，4px仅是名义采样间距，不等于4×4独立语义分辨率。", "",
                  "## Offset语义审计", "", "旧实现的标量offset同时改动Y、X窗口起点，`windows(..., offset)`首窗为`(-offset,-offset)`；历史`{0,8}`是两个对角相位。S1/S2显式建立笛卡尔二维偏移。Reflection使用原始窗口起点和patch中心映射，S2/S3相位边际与坐标保存于`phase_margins/`。", "",
                  "## 结果", "", "宏平均和图像配对bootstrap见`summary.csv`与`paired_effects.csv`；Boundary、对象与决策翻转见对应CSV。", "",
                  "## S0回归", "", f"全量保存预测逐像素比对：{len(parity)}张，差异像素总数 {sum(r['mask_disagreement_pixels'] for r in parity)}。历史协议与Memory SHA见`baseline_parity_per_image.csv`。", "",
                  "## 计算成本", "", "缓存推理与真实UNI前向分开记录于`runtime_memory.csv`。历史0/8缓存读入复用；其余相位使用冻结UNI真实前向。", "",
                  "## 研究结论与限制", "", "请根据实测配对差值和时间结果判断S1/S2/S3；不得将本固定Seed官方测试的探索性结果解释为跨Seed鲁棒性。是否比Feature Bilinear Upsampling有效、16相位是否值得成本、S3是否优于S2及是否需要Test-time优化，见下列自动汇总与对应效应表。", ""]
        for ds in args.datasets:
            sub = [r for r in summary if r["dataset"] == ds]
            report += [f"### {ds}", "", "| Method | Dice | Object-F1 | Boundary-F1 | Small gland recall |", "|---|---:|---:|---:|---:|"]
            for r in sub: report.append(f"| {r['method']} | {r['dice']:.6f} | {r['object_f1']:.6f} | {r['boundary_f1']:.6f} | {r['small_gland_recall']:.6f} |")
            report.append("")
        (OUT / "report.md").write_text("\n".join(report))
        meta.update(status="COMPLETE", finished_utc=datetime.now(timezone.utc).isoformat(),
                    feature_cache_entries=len(cache_rows), images=sum(len(base.data[d]["queries"]) for d in args.datasets),
                    prediction_count=len(predictions), all_s0_pixelwise=True)
        dump_json(OUT / "run_metadata.json", meta)
        log("COMPLETE")
    except Exception:
        meta = json.loads((OUT / "run_metadata.json").read_text()) if (OUT / "run_metadata.json").exists() else {}
        meta.update(status="FAILED", traceback=traceback.format_exc(), failed_utc=datetime.now(timezone.utc).isoformat())
        dump_json(OUT / "run_metadata.json", meta); log(traceback.format_exc()); raise


if __name__ == "__main__": main()
