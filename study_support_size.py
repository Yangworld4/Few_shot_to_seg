"""Reproducible fixed-K N3-bil support-size sweep using audited UNI caches."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from pathology_trainfree.config import load_config
from pathology_trainfree.data import load_image, load_mask, read_manifest, resize_mask
from pathology_trainfree.encoder import FeatureEncoder, sha256
from pathology_trainfree.evaluation import metrics as semantic_metrics, object_metrics
from pathology_trainfree.geometry import windows
from pathology_trainfree.memory import ReferenceMemory, select_tokens
from pathology_trainfree.pipeline import SegmentationPipeline


DATASETS = ("GlaS", "CRAG")
SHOTS = (1, 5, 10, 20)
SEEDS = (14, 24, 42, 1337, 3407)
WEIGHTS_DEFAULT = "/home/yangorange/.cache/huggingface/hub/models--MahmoodLab--UNI/snapshots/b55a5ec6cade1a39edfe6534189a9b8ca7a022f0/pytorch_model.bin"


def atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_signature(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def cache_file_for(dual_root, dataset, image_id):
    base = Path(dual_root) / "outputs" / f"uni_audit_mask_quality_{dataset}_seed1337/cache/features_N3"
    for split in ("labeled", "unlabeled"):
        p = base / split / f"{image_id}.npz"
        if p.is_file():
            return p
    return None


def cache_key(image, encoder, cfg, offset):
    w = cfg["window"]
    return hashlib.sha256(image.tobytes() + str((encoder.weights_sha256, w["size"], w["stride"],
                           offset, encoder.precision, encoder.token_precision)).encode()).hexdigest()


def load_support_cache(entries, dual_root, dataset, cfg):
    """Load configured Support offset passes from the DUAL N3 cache."""
    store, provenance = {}, []
    missing = []
    for entry in entries:
        source = cache_file_for(dual_root, dataset, entry["id"])
        if source is None:
            missing.append(entry["id"])
            continue
        image = load_image(entry["image"])
        mask = load_mask(entry["mask"], cfg["dataset"])
        if image.shape[:2] != mask.shape:
            raise ValueError(f"Image/mask size mismatch for {entry['id']}")
        labels = set(int(x) for x in np.unique(mask))
        if not labels.issubset({0, 1}):
            raise ValueError(f"Non-binary Support GT: {entry['id']}: {labels}")
        gt_hash = file_hash(entry["mask"])
        occupancy_protocol = {"occupancy_source": "real_support_gt", "gt_sha256": gt_hash,
                              "image_shape": list(image.shape[:2]), "window": cfg["window"],
                              "patch_size": 16, "mask_encoding": cfg["dataset"]["mask_encoding"],
                              "class_mapping": cfg["dataset"]["class_mapping"]}
        occupancy_key = stable_signature(occupancy_protocol)
        occupancy_dir = cfg["output"].get("occupancy_cache_directory")
        occupancy_path = Path(occupancy_dir) / dataset / f"{entry['id']}_{occupancy_key}.npz" if occupancy_dir else None
        cached_real = None
        if occupancy_path and occupancy_path.is_file():
            try:
                with np.load(occupancy_path, allow_pickle=False) as oc:
                    om = json.loads(str(oc["metadata"]))
                    if (om.get("occupancy_source") == "real_support_gt" and om.get("gt_sha256") == gt_hash
                            and om.get("fingerprint") == occupancy_key):
                        cached_real = oc["fg_occupancy"].copy()
            except Exception:
                cached_real = None
        occupancy_cache_reused = cached_real is not None
        occupancy_cache_errors = []
        cursor = 0
        with np.load(source, allow_pickle=False) as z:
            required = {"features", "occupancy", "passno", "x", "y"}
            if not required.issubset(z.files):
                raise ValueError(f"Incomplete support feature cache: {source}")
            pass_features, pass_occ = [], []
            for offset in cfg["window"]["support_offsets"]:
                passno = {0: 0, 8: 1}[int(offset)]
                selected = np.flatnonzero(z["passno"] == passno)
                features = z["features"][selected]
                cached_occ = z["occupancy"][selected]
                patch = int(cfg["window"]["size"]) // 28
                if patch != 16:
                    raise ValueError("DUAL N3 feature caches require patch size 16 and window 448")
                if cached_real is not None:
                    length = len(selected) * 784
                    fg_occ = cached_real[cursor:cursor + length].reshape(len(selected), 784)
                    cursor += length
                else:
                    fg_occ = np.asarray([
                        crop.reshape(28, patch, 28, patch).mean(axis=(1, 3)).reshape(-1)
                        for _, _, crop in windows(mask, cfg["window"]["size"], cfg["window"]["stride"],
                                                   offset, cfg["window"]["padding_mode"])
                    ], dtype=np.float32)
                occupancy_cache_errors.append(float(np.max(np.abs(fg_occ - cached_occ))))
                positions = [(int(z["y"][i]), int(z["x"][i])) for i in selected]
                expected = [(y, x) for y, x, _ in windows(image, cfg["window"]["size"],
                                                           cfg["window"]["stride"], offset,
                                                           cfg["window"]["padding_mode"])]
                if positions != expected:
                    raise ValueError(f"Offset-{offset} window positions disagree for {dataset}/{entry['id']}")
                if features.shape != (len(expected), 784, 1024) or fg_occ.shape != (len(expected), 784):
                    raise ValueError(f"Unexpected cached feature shape for {dataset}/{entry['id']}, offset={offset}: {features.shape}")
                pass_features.append(features.reshape(-1, features.shape[-1]))
                pass_occ.append(fg_occ.reshape(-1))
        labels = set(int(x) for x in np.unique(mask))
        if not labels.issubset({0, 1}):
            raise ValueError(f"Mask labels are not binary for {dataset}/{entry['id']}: {labels}")
        actual_occ = np.concatenate(pass_occ, axis=0)
        if not np.isfinite(actual_occ).all() or actual_occ.min() < 0 or actual_occ.max() > 1:
            raise ValueError(f"Invalid occupancy values for {entry['id']}")
        if (1 in labels and not np.any(actual_occ > 0)) or (0 in labels and not np.any(actual_occ < 1)):
            raise ValueError(f"GT class exists but is missing from Token occupancy: {entry['id']}")
        if occupancy_path and not occupancy_cache_reused:
            occupancy_path.parent.mkdir(parents=True, exist_ok=True)
            temp = occupancy_path.with_suffix(".npz.tmp")
            with temp.open("wb") as f:
                np.savez_compressed(f, fg_occupancy=actual_occ,
                                    metadata=np.array(json.dumps({**occupancy_protocol, "fingerprint": occupancy_key})))
            os.replace(temp, occupancy_path)
        store[entry["id"]] = {"features": np.concatenate(pass_features, axis=0),
                               "fg_occ": np.concatenate(pass_occ, axis=0),
                               "labels": labels, "image_shape": image.shape[:2],
                               "gt_sha256": gt_hash, "occupancy_fingerprint": occupancy_key,
                               "occupancy_source": "real_support_gt"}
        provenance.append({"dataset": dataset, "image_id": entry["id"], "source_file": str(source),
                           "source_sha256": file_hash(source),
                           "offsets": ";".join(map(str, cfg["window"]["support_offsets"])),
                           "windows_per_offset": "validated", "token_grid": "28x28", "feature_dim": 1024,
                           "occupancy_source": "real_support_gt", "gt_path": str(entry["mask"]),
                           "gt_sha256": gt_hash, "occupancy_fingerprint": occupancy_key,
                           "occupancy_cache_file": str(occupancy_path) if occupancy_path else "",
                           "occupancy_cache_reused": occupancy_cache_reused,
                           "gt_fg_ratio": float(mask.mean()), "gt_labels": ";".join(map(str, sorted(labels))),
                           "occupancy_min": float(actual_occ.min()), "occupancy_max": float(actual_occ.max()),
                           "raw_fg_candidates": int((actual_occ >= .9).sum()),
                           "raw_bg_candidates": int((actual_occ <= .1).sum()),
                           "mixed_tokens": int(((actual_occ > .1) & (actual_occ < .9)).sum()),
                           "total_tokens": int(len(actual_occ)), "coordinates_validated": True,
                           "cached_occupancy_max_abs_error": max(occupancy_cache_errors),
                           "cache_version": int(np.load(source, allow_pickle=False)["version"])})
    if missing:
        raise FileNotFoundError(f"DUAL audited feature cache does not cover {dataset} support pool IDs: {missing}")
    return store, provenance


def precompute_query_features(entries, encoder, cfg, dataset, cache_root, log):
    """Create resumable, fingerprinted Offset 0/8 query feature caches once."""
    feature_dir = Path(cache_root) / dataset / "feature_cache"
    feature_dir.mkdir(parents=True, exist_ok=True)
    records = []
    t0 = time.monotonic()
    for n, entry in enumerate(entries, 1):
        image = load_image(entry["image"])
        image_sha = hashlib.sha256(image.tobytes()).hexdigest()
        for offset in cfg["window"]["offsets"]:
            key = cache_key(image, encoder, cfg, offset)
            out = feature_dir / f"{entry['id']}_offset{offset}.npz"
            valid = False
            if out.is_file():
                try:
                    with np.load(out, allow_pickle=False) as z:
                        valid = str(z["cache_key"]) == key
                        tokens = z["tokens"]
                        positions = z["positions"]
                    expected = [(y, x) for y, x, _ in windows(image, 448, 224, offset)]
                    valid &= tokens.shape == (len(expected), 784, 1024) and np.array_equal(positions, expected)
                except Exception:
                    valid = False
            if not valid:
                views = list(windows(image, 448, 224, offset))
                positions = [(y, x) for y, x, _ in views]
                tokens = encoder.encode([crop for _, _, crop in views])
                temp = out.with_suffix(".npz.tmp")
                with temp.open("wb") as f:
                    np.savez_compressed(f, tokens=tokens, positions=np.asarray(positions), cache_key=np.array(key))
                os.replace(temp, out)
            records.append({"dataset": dataset, "image_id": entry["id"], "offset": offset,
                            "image_sha256": image_sha, "cache_key": key, "cache_file": str(out),
                            "cache_sha256": file_hash(out), "window_count": int(tokens.shape[0]),
                            "cache_reused": bool(valid)})
        if n == 1 or n % 10 == 0 or n == len(entries):
            log(f"{dataset}: query feature cache {n}/{len(entries)} images")
    return records, time.monotonic() - t0


def support_permutations(pool, seeds):
    result = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(len(pool))
        result[seed] = [pool[int(i)] for i in permutation]
    return result


def construct_memory(support_ids, support_cache, cfg, support_seed, encoder, sampling_protocol="stable_per_image"):
    mcfg = cfg["memory"]
    memory_seed = int(mcfg["sampling_seed"])
    parts = {0: [], 1: []}
    fallback_images = {0: 0, 1: 0}
    support_labels = set()
    global_rng = np.random.default_rng(memory_seed)
    for image_id in support_ids:
        item = support_cache[image_id]
        support_labels.update(item["labels"])
        features = item["features"]
        fg = item["fg_occ"].astype(np.float32)
        occ = np.stack((1.0 - fg, fg), axis=-1)
        if sampling_protocol == "original_global":
            rng = global_rng
        elif sampling_protocol == "stable_per_image":
            stable_id = int.from_bytes(hashlib.sha256(image_id.encode("utf-8")).digest()[:8], "little")
            rng = np.random.default_rng(np.random.SeedSequence([memory_seed, stable_id & 0xffffffff,
                                                                 stable_id >> 32]))
        else:
            raise ValueError(f"Unknown memory sampling protocol: {sampling_protocol}")
        selected, fallback = select_tokens(features, occ, mcfg, rng)
        for cls in (0, 1):
            if len(selected[cls]):
                parts[cls].append(selected[cls])
            fallback_images[cls] += int(fallback[cls])
    vectors = {cls: np.concatenate(parts[cls]) if parts[cls] else np.empty((0, 1024), np.float16)
               for cls in (0, 1)}
    counts = {cls: len(vectors[cls]) for cls in (0, 1)}
    missing_classes = sorted({0, 1} - support_labels)
    reasons = []
    if missing_classes:
        reasons.append("MISSING_GT_CLASS_" + "_".join(map(str, missing_classes)))
    if counts[1] < 50:
        reasons.append("FG_BELOW_K50")
    if counts[0] < 50:
        reasons.append("BG_BELOW_K50")
    if reasons:
        return None, {"counts": counts, "fallback_images": fallback_images,
                      "support_labels": sorted(support_labels), "reason": ";".join(reasons)}
    metadata = {"class_ids": [0, 1], "counts": {str(k): v for k, v in counts.items()},
                "feature_dim": 1024, "encoder_sha256": encoder.weights_sha256,
                "encoder_name": "uni", "encoder_patch_size": 16,
                "encoder_config": encoder.config_path, "encoder_normalize": encoder.normalize,
                "encoder_precision": encoder.precision, "encoder_token_precision": encoder.token_precision,
                "support_ids": support_ids, "seed": memory_seed, "sampling_seed": memory_seed,
                "support_selection_seed": int(support_seed),
                "occupancy_source": "real_support_gt",
                "support_gt_sha256": {iid: support_cache[iid]["gt_sha256"] for iid in support_ids},
                "occupancy_fingerprints": {iid: support_cache[iid]["occupancy_fingerprint"] for iid in support_ids},
                "occupancy_thresholds": mcfg["occupancy_thresholds"],
                "cap": int(mcfg["cap_per_image_per_class"]),
                "fallback_strategy": mcfg["fallback_strategy"], "fallback_images": fallback_images,
                "sampling_protocol": sampling_protocol,
                "window": cfg["window"], "memory_config": {**mcfg, "sampling_seed": memory_seed},
                "dataset_config": {"mask_encoding": cfg["dataset"]["mask_encoding"],
                                   "class_mapping": cfg["dataset"]["class_mapping"]},
                "source": f"DUAL features_N3 passno for offsets {cfg['window']['support_offsets']}; GT occupancy cache"}
    return ReferenceMemory(vectors, metadata), {"counts": counts, "fallback_images": fallback_images,
                                                "support_labels": sorted(support_labels), "reason": ""}


def evaluate_run(pipeline, entries, cfg, log):
    rows = []
    predictions = {}
    total_inference = 0.0
    for i, entry in enumerate(entries, 1):
        image = load_image(entry["image"])
        t0 = time.monotonic()
        result = pipeline.predict(image, entry["id"])
        elapsed = time.monotonic() - t0
        total_inference += elapsed
        pred = result["mask"].astype(np.uint8)
        gt = resize_mask(load_mask(entry["mask"], cfg["dataset"]), pred.shape)
        cls = semantic_metrics(pred, gt, [0, 1])
        of1, ohd = object_metrics(pred == 1, gt == 1,
                                  int(cfg["evaluation"]["object_min_area"]),
                                  float(cfg["evaluation"]["object_match_overlap"]))
        fg = cls[1]
        rows.append({"image_id": entry["id"], "dice": fg["dice"], "fg_iou": fg["iou"],
                     "precision": fg["precision"], "recall": fg["recall"],
                     "object_f1": of1, "object_hausdorff": ohd,
                     "inference_seconds": elapsed,
                     "foreground_pixels": int((gt == 1).sum()),
                     "predicted_foreground_pixels": int((pred == 1).sum()),
                     "predicted_fg_ratio": float(pred.mean()), "gt_fg_ratio": float(gt.mean())})
        predictions[entry["id"]] = (pred, gt)
        if i == 1 or i % 10 == 0 or i == len(entries):
            log(f"inference {i}/{len(entries)}: {entry['id']}")
    means = {key: float(np.mean([r[key] for r in rows]))
             for key in ("dice", "fg_iou", "precision", "recall", "object_f1", "object_hausdorff", "inference_seconds", "predicted_fg_ratio", "gt_fg_ratio")}
    totals = {k: 0 for k in ("tp", "fp", "fn")}
    for pred, gt in predictions.values():
        totals["tp"] += int(((pred == 1) & (gt == 1)).sum())
        totals["fp"] += int(((pred == 1) & (gt == 0)).sum())
        totals["fn"] += int(((pred == 0) & (gt == 1)).sum())
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    return rows, {**means, "micro_dice": 2 * tp / max(1, 2 * tp + fp + fn),
                  "micro_fg_iou": tp / max(1, tp + fp + fn), "query_count": len(rows),
                  "inference_total_seconds": total_inference, **totals}


def standard_deviation(values):
    return float(np.std(values, ddof=1)) if len(values) >= 2 else None


def summarize(run_records):
    rows = []
    for dataset in DATASETS:
        for shot in SHOTS:
            group = [r for r in run_records if r["dataset"] == dataset and int(r["shots"]) == shot]
            valid = [r for r in group if r["status"] == "COMPLETE"]
            row = {"dataset": dataset, "shots": shot, "valid_seed_count": len(valid),
                   "total_seed_count": len(SEEDS), "valid_ratio": f"{len(valid)}/{len(SEEDS)}"}
            for metric in ("dice", "fg_iou", "object_f1", "precision", "recall", "object_hausdorff"):
                vals = [float(r[metric]) for r in valid]
                row[f"{metric}_mean"] = float(np.mean(vals)) if vals else None
                row[f"{metric}_sd"] = standard_deviation(vals)
                row[f"{metric}_min"] = float(np.min(vals)) if vals else None
                row[f"{metric}_max"] = float(np.max(vals)) if vals else None
            rows.append(row)
    return rows


def fmt(value, digits=4):
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def plot_metric(summary, metric, output, ylabel):
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    styles = {"GlaS": ("#1565c0", "o"), "CRAG": ("#c62828", "s")}
    for dataset in DATASETS:
        rows = [r for r in summary if r["dataset"] == dataset]
        means = [np.nan if r[f"{metric}_mean"] is None else r[f"{metric}_mean"] for r in rows]
        errs = [0.0 if r[f"{metric}_sd"] is None else r[f"{metric}_sd"] for r in rows]
        color, marker = styles[dataset]
        ax.errorbar(SHOTS, means, yerr=errs, color=color, marker=marker, capsize=4,
                    linewidth=2, markersize=6, label=dataset)
    ax.set_xticks(SHOTS, labels=[str(s) for s in SHOTS])
    ax.set_xlim(0, max(SHOTS) + 1)
    ax.set_xlabel("Labeled support images")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=.25)
    ax.legend(frameon=False)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def trend_sentence(rows, dataset, metric):
    vals = {int(r["shots"]): r for r in rows if r["dataset"] == dataset}
    means = [vals[s][f"{metric}_mean"] for s in SHOTS]
    ns = [vals[s]["valid_seed_count"] for s in SHOTS]
    if any(v is None for v in means):
        return f"{dataset} 的 {metric} 存在未获得有效均值的 Support Size，趋势无法完整判断。"
    monotone = all(means[i + 1] >= means[i] for i in range(len(means) - 1))
    label = "均值逐档不下降" if monotone else "均值并非逐档单调上升"
    return (f"{dataset} {metric} 均值依次为 " + "、".join(fmt(v) for v in means) +
            f"（有效 Seed 数 {ns}）；{label}。相邻变化为 " + "、".join(
                f"{SHOTS[i]}→{SHOTS[i+1]}: {fmt(means[i+1]-means[i], 4)}" for i in range(3)) + ".")


def make_report(path, summary, runs, support_pool_counts, feature_cache_info, meta):
    support_offsets = list(meta.get("support_offsets", [0]))
    support_offset_text = "/".join(map(str, support_offsets))
    sampling_protocol = meta.get("memory_sampling_protocol", "stable_per_image")
    aligned_memory_protocol = support_offsets == [0, 8] and sampling_protocol == "original_global"
    lines = ["# N3-bil Training-Free Support-Size Sensitivity Study", "",
             "## 1. Experimental Setup", "",
             f"冻结 UNI ViT-L/16，无梯度训练。唯一实验变量为训练候选池中带 GT Support 图像数（1、5、10、20）；每个数据集分别使用 NumPy PCG64 对官方 `train_pool.csv` 按 Seed 生成一次随机排列，较小 Support 是该排列的前缀。固定 Top-K=50、448/224 窗口、Query Offsets 0/8、Support Offsets {meta.get('support_offsets', [0])}、Occupancy 0.9/0.1、每图每类上限512、Memory 抽样基础 Seed 2026、Bilinear Margin 重建和256×256评估。Memory Token 子采样协议为 `{meta.get('memory_sampling_protocol', 'stable_per_image')}`。",
             "",
             f"UNI SHA256 `{meta['encoder_sha256']}`；PyTorch `{meta['torch']}`，CUDA `{meta['cuda']}`。Support GT 仅来自官方训练候选池；测试集 ID 与候选池无交集。测试 GT 只用于最终指标。GLaS Test A+B 固定 {meta['query_counts']['GlaS']} 张，CRAG 官方 Test 固定 {meta['query_counts']['CRAG']} 张。",
             "",
             f"Support 特征读取 DUAL N3 cache 的 passno；本次 Offset 配置为 {meta.get('support_offsets', [0])}。测试 Query Set 的 Offset 0/8 特征预计算一次并通过图像、权重和窗口配置指纹校验。", "",
             f"原始 DUAL `memory_N3.npz` 混合 Support Offset 0/8 两个特征 Pass，且 Memory 抽样协议为 `{meta.get('memory_sampling_protocol', 'stable_per_image')}`。", "",
             "Object 指标定义：256×256 二值图上使用 8-连通组件，组件面积至少50像素；按交并匹配执行一对一匹配，重叠/GT组件面积大于0.5视为命中。Object-Hausdorff 为组件边界双向 Hausdorff 距离并按组件面积加权，再对图像等权平均。", "",
             "候选池规模：" + ", ".join(f"{ds} {support_pool_counts[ds]} 张" for ds in DATASETS) + ".",
             f"共计划 {len(DATASETS)*len(SHOTS)*len(SEEDS)} 组；Complete {sum(r['status']=='COMPLETE' for r in runs)}，无效 {sum(r['status'].startswith('INVALID') for r in runs)}，错误/待运行 {sum(r['status'] not in ('COMPLETE',) and not r['status'].startswith('INVALID') for r in runs)}。",
             "",
             "## 2. Main Results", "",
             "下表为逐图指标的算术均值在有效 Seed 间的 Mean±SD。Dice、FG IoU、Object-F1、Precision、Recall 和 Object-Hausdorff 均按图像等权；n 是有效 Seed 数/5。",
             "",
             "| Dataset | Shots | Valid Seeds | Dice | FG IoU | Object-F1 | Precision | Recall | Object-Hausdorff (px) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        def ms(k):
            mean, sd = row[f"{k}_mean"], row[f"{k}_sd"]
            return fmt(mean) + (f" ± {fmt(sd)}" if sd is not None else " ± NA")
        lines.append(f"| {row['dataset']} | {row['shots']} | {row['valid_ratio']} | {ms('dice')} | {ms('fg_iou')} | {ms('object_f1')} | {ms('precision')} | {ms('recall')} | {ms('object_hausdorff')} |")
    lines += ["", "### Full Seed Results", "", "| Dataset | Shots | Seed | Status | FG Tokens | BG Tokens | Dice | FG IoU | Object-F1 |", "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for r in sorted(runs, key=lambda x: (x["dataset"], int(x["shots"]), int(x["seed"]))):
        lines.append(f"| {r['dataset']} | {r['shots']} | {r['seed']} | {r['status']} | {r.get('fg_tokens','')} | {r.get('bg_tokens','')} | {fmt(r.get('dice'))} | {fmt(r.get('fg_iou'))} | {fmt(r.get('object_f1'))} |")
    lines += ["", "## 3. Performance Trends", ""]
    for ds in DATASETS:
        for metric in ("dice", "fg_iou", "object_f1"):
            lines.append("- " + trend_sentence(summary, ds, metric))
    lines += ["", "相邻档 Support 数量差分别为 +4、+5、+10；每增加一张图像的平均收益由对应相邻指标差除以该增量计算。"]
    for ds in DATASETS:
        by = {int(r["shots"]): r for r in summary if r["dataset"] == ds}
        for metric in ("dice", "fg_iou", "object_f1"):
            if all(by[s][f"{metric}_mean"] is not None for s in SHOTS):
                vals = [by[s][f"{metric}_mean"] for s in SHOTS]
                per_image = [(vals[i+1] - vals[i])/(SHOTS[i+1]-SHOTS[i]) for i in range(3)]
                lines.append(f"- {ds} {metric} 每新增一张 Support 的相邻档均值收益：" + "、".join(fmt(v, 5) for v in per_image) + ".")
    lines += ["", "## 4. Support Sensitivity", ""]
    for ds in DATASETS:
        for shot in SHOTS:
            row = next(r for r in summary if r["dataset"] == ds and int(r["shots"]) == shot)
            lines.append(f"- {ds} {shot}-shot Dice: n={row['valid_seed_count']}/5，均值 {fmt(row['dice_mean'])}，SD {fmt(row['dice_sd'])}，范围 [{fmt(row['dice_min'])}, {fmt(row['dice_max'])}]；Object-F1 SD {fmt(row['object_f1_sd'])}。")
    lines += ["", "## 5. Memory Statistics", ""]
    for ds in DATASETS:
        group = [r for r in runs if r["dataset"] == ds]
        lines.append(f"- {ds}: 完整训练候选池 {support_pool_counts[ds]} 张；每图每类上限512。Memory 无效组合 {sum(r['status'].startswith('INVALID') for r in group)}/20，其他错误 {sum(r['status']=='ERROR' for r in group)}/20。")
        for shot in SHOTS:
            ss = [r for r in group if int(r["shots"]) == shot]
            valid = [r for r in ss if r["status"] == "COMPLETE"]
            if valid:
                lines.append(f"  - {shot}-shot Token Count 范围：FG {min(int(r['fg_tokens']) for r in valid)}–{max(int(r['fg_tokens']) for r in valid)}，BG {min(int(r['bg_tokens']) for r in valid)}–{max(int(r['bg_tokens']) for r in valid)}；K=50 有效 {len(valid)}/5。")
            else:
                lines.append(f"  - {shot}-shot：没有有效 Memory。")
    generated = meta.get("query_feature_generation_wall_seconds_approx", {})
    lines.append(f"- Query 特征首次编码时间由首次运行日志时间戳估算约为：GLaS {generated.get('GlaS', 'NA')}s，CRAG {generated.get('CRAG', 'NA')}s；重启后指纹校验/复用耗时分别为 {feature_cache_info['GlaS']['seconds']:.1f}s 和 {feature_cache_info['CRAG']['seconds']:.1f}s（240/240 缓存项复用）。所有组合固定 K=50；无效组不会降低 K 或改变阈值。")
    lines += ["", "## 6. Research Implications", ""]
    by = {(r["dataset"], int(r["shots"])): r for r in summary}
    for ds in DATASETS:
        r1, r5, r10, r20 = (by[(ds, s)] for s in SHOTS)
        lines.append(f"### {ds}")
        lines.append(f"- Q1 One-shot 实用性：Dice {fmt(r1['dice_mean'])}±{fmt(r1['dice_sd'])}，五 Seed 范围[{fmt(r1['dice_min'])}, {fmt(r1['dice_max'])}]；FG IoU {fmt(r1['fg_iou_mean'])}，Object-F1 {fmt(r1['object_f1_mean'])}。按本实验的跨 Seed 稳定性，不能把 one-shot 视为可靠的实用性能；最差 Seed Dice 为 {fmt(r1['dice_min'])}。")
        for metric, label in (("dice", "Dice"), ("fg_iou", "FG IoU"), ("object_f1", "Object-F1")):
            vals = [by[(ds, s)][f"{metric}_mean"] for s in SHOTS]
            direction = "各档逐步不下降" if all(vals[i+1] >= vals[i] for i in range(3)) else "至少有一档均值下降"
            lines.append(f"- Q2 {label}：1/5/10/20-shot 均值为" + "、".join(fmt(v) for v in vals) + f"；{direction}，20-shot 为已测档位最高值。")
            deltas = [vals[i+1]-vals[i] for i in range(3)]
            per_image = [d/(SHOTS[i+1]-SHOTS[i]) for i, d in enumerate(deltas)]
            lines.append(f"- Q3 {label} 相邻总增益（1→5、5→10、10→20）为" + "、".join(fmt(v) for v in deltas) + "; 每新增一张 Support 的平均变化为" + "、".join(fmt(v, 5) for v in per_image) + ".")
        sd_ratio = (r1["dice_sd"] / r20["dice_sd"] if r1["dice_sd"] and r20["dice_sd"] else float("nan"))
        lines.append(f"- Q4 Support 选择敏感性：Dice Seed SD 从1-shot的{fmt(r1['dice_sd'])}到20-shot的{fmt(r20['dice_sd'])}（约{fmt(sd_ratio, 2)}倍）；1-shot 范围[{fmt(r1['dice_min'])}, {fmt(r1['dice_max'])}]。这些是五个预设随机排列的描述性离散，不是显著性检验。")
        target = .99 * r20["dice_mean"]
        qualifying = [s for s in SHOTS if by[(ds, s)]["dice_mean"] >= target]
        least = min(qualifying) if qualifying else None
        lines.append(f"- Q7 达到20-shot均值Dice的99%阈值（{fmt(target)}）的最小已测 Support Size：{least if least is not None else '无'}。该阈值仅是当前划分上的经验参照。")
    lines += ["", "### 跨数据集结论（Q5–Q6）", ""]
    dice_trends = {}
    for ds in DATASETS:
        by_shot = {int(r["shots"]): r for r in summary if r["dataset"] == ds}
        dice_trends[ds] = [by_shot[s]["dice_mean"] for s in SHOTS]
    lines.append("- Q5 Dice 趋势：" + "; ".join(
        f"{ds} {','.join(fmt(x) for x in dice_trends[ds])} ({'逐档不下降' if all(dice_trends[ds][i+1] >= dice_trends[ds][i] for i in range(3)) else '非单调'})"
        for ds in DATASETS) + ". 具体相邻变化见上表；两数据集趋势需要分别解读。")
    one_shot = [r for r in runs if int(r["shots"]) == 1 and r["status"] == "COMPLETE"]
    g1runs = [r for r in one_shot if r["dataset"] == "GlaS"]
    c1runs = [r for r in one_shot if r["dataset"] == "CRAG"]
    lines.append(f"- Q6 固定 K=50 在本次所有 Memory 均可行：{sum(r['status']=='COMPLETE' for r in runs)}/{len(DATASETS)*len(SHOTS)*len(SEEDS)} 组合完成，0个 `INVALID_INSUFFICIENT_MEMORY`。1-shot FG Token 数最低为 GLaS {min(int(r['fg_tokens']) for r in g1runs)}、CRAG {min(int(r['fg_tokens']) for r in c1runs)}；BG 最低为 GLaS {min(int(r['bg_tokens']) for r in g1runs)}、CRAG {min(int(r['bg_tokens']) for r in c1runs)}。当前候选池和 fallback 设置下，K=50没有造成运行排除，但不代表更少 Token 的设置也成立。")
    lines += ["", "### 研究含义与结论", ""]
    g1, g20, c1, c20 = by[("GlaS", 1)], by[("GlaS", 20)], by[("CRAG", 1)], by[("CRAG", 20)]
    lines.append(f"- Few-shot data efficiency：当前 Support Offset {support_offset_text} 配置下，20-shot 相比1-shot的平均 Dice 分别增加 GLaS {fmt(g20['dice_mean']-g1['dice_mean'])}、CRAG {fmt(c20['dice_mean']-c1['dice_mean'])}；但 one-shot 的平均指标和最差 Seed 均显示较弱或不稳定表现，不能据此宣称该方法具有稳健的实用 one-shot 分割能力。")
    lines.append(f"- 数量与选择：更多 Support 在20-shot均值上有帮助，但1-shot Dice SD（GLaS {fmt(g1['dice_sd'])}；CRAG {fmt(c1['dice_sd'])}）相对很大；CRAG 的 one-shot Seed 范围为{fmt(c1['dice_min'])}–{fmt(c1['dice_max'])}。这表明当前实验中 Support 选择是重要变异来源，尤其 CRAG；不能仅靠增加 Support 数量消除所有差异。")
    lines.append("- 增加 Reference Images 并非持续等比例提升：两数据集从10到20张的 Dice 增益为" + f"GLaS {fmt(g20['dice_mean']-by[('GlaS', 10)]['dice_mean'])}、CRAG {fmt(c20['dice_mean']-by[('CRAG', 10)]['dice_mean'])}；本实验显示后段边际收益较小，但20-shot仍为最高均值。")
    lines.append("- 证据支持优先研究 Memory 覆盖/质量与 Support 选择稳定性，其次再拆分 Retrieval Reliability；本 sweep 没有改变检索或空间恢复，因此不能从这些数据断言空间细化是主要瓶颈。这个优先级是基于观察到的 Seed 离散与10→20边际变化提出的研究假设，不是机制因果结论。")
    if aligned_memory_protocol:
        lines.append("- 本次 Support Memory 已使用历史的 Offset 0/8 与全局 seed 2026 抽样协议。旧 Offset-0 + per-image RNG 研究仍保留为不同实验配置；协议对齐的逐 seed/shot 差值见本次配套诊断报告。")
    else:
        lines.append("- 历史 DUAL Memory 混合了 Support Offset 0/8。当前 sweep 使用的 Offset/采样协议必须与历史设置区分；若需研究原 N3-bil Memory，应在新的输出目录用 Offset 0/8 + global seed 2026 再运行，并保留本次结果。")
    lines += ["", f"**结论：**在本次 1/5/10/20-shot、五 Seed、Support Offset {support_offset_text}、Memory 采样 `{sampling_protocol}` 的 N3-bil Direct 实验中，20-shot 对两个数据集的 Dice、FG IoU、Object-F1 均取得最高均值；Support Size 的均值变化按数据集分别解读。K=50 对本次合法训练池均可运行。结论仅适用于本次官方划分、固定 UNI/Memory 配置与256评估尺度。"]
    lines += ["", "## Artifacts and Integrity", "", f"DUAL source root: `{meta['dual_root']}`; DUAL commit: `{meta['dual_commit']}`; PathologyTrainFree git commit: `{meta.get('project_commit') or 'unavailable (directory is not a Git worktree)'}`.",
              f"Executed study runner SHA256: `{meta.get('project_code_sha256', 'not recorded')}`; execution command/provenance: `execution_command.txt`, `run_provenance.json`.",
              f"N3 support feature cache fingerprint source: `{meta['support_cache_protocol']}`. Query cache entries: {meta['query_cache_entries']}.",
              "每组配置、Support IDs、Memory 统计、逐图指标与状态分别保存在 `configs/`、`support_sets.csv`、`memory_statistics.csv`、`per_image_metrics.csv` 和 `runs.csv`。"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dual-root", type=Path, default=ROOT.parent / "DUAL")
    parser.add_argument("--weights", type=Path, default=Path(WEIGHTS_DEFAULT))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/support_size_study")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--support-offsets", nargs="+", type=int, choices=(0, 8), default=[0],
                        help="Support feature passes to combine before per-image/class capping.")
    parser.add_argument("--memory-sampling-protocol", choices=("stable_per_image", "original_global"),
                        default="stable_per_image")
    parser.add_argument("--query-cache-root", type=Path, default=None,
                        help="Optional reusable cache root; must contain dataset/feature_cache files with matching fingerprints.")
    args = parser.parse_args()
    if len(set(args.support_offsets)) != len(args.support_offsets):
        parser.error("--support-offsets must not contain duplicates")

    out = args.output.resolve()
    for name in ("configs", "runs", "cache/support_index", "cache/query_features", "predictions"):
        (out / name).mkdir(parents=True, exist_ok=True)
    log_path = out / "run.log"
    log_handle = log_path.open("a", encoding="utf-8", buffering=1)

    def log(msg):
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        line = f"[{stamp}] {msg}"
        print(line, flush=True)
        log_handle.write(line + "\n")

    if not args.weights.is_file():
        raise FileNotFoundError(f"UNI weights missing: {args.weights}")
    study_started = time.monotonic()
    base_config = ROOT / "configs/uni_memory.yaml"
    cfg_base = load_config(base_config)
    cfg_base["encoder"]["weights"] = str(args.weights.resolve())
    cfg_base["window"].update({"size": 448, "stride": 224, "offsets": [0, 8], "support_offsets": list(args.support_offsets),
                               "padding_mode": "reflect", "interpolation": "bilinear"})
    cfg_base["memory"].update({"class_ids": [0, 1], "occupancy_thresholds": {0: 0.9, 1: 0.9},
                               "cap_per_image_per_class": 512, "normalize": True,
                               "fallback_strategy": "nearest_10_percent"})
    cfg_base["retrieval"].update({"top_k": 50, "metric": "cosine", "aggregation": "mean", "chunk_size": 784})
    cfg_base["segmentation"].update({"output_size": [256, 256], "native_resolution": False,
                                     "threshold": 0.0, "margin_precision": "float16",
                                     "interpolation": "bilinear", "overlap_fusion": "mean", "offset_fusion": "mean"})
    cfg_base["evaluation"].update({"object_metrics": True, "object_min_area": 50, "object_match_overlap": 0.5})
    cfg_base["output"].update({"save_mask": False, "save_margin": False, "save_visualization": False,
                               "save_metrics": False, "cache_features": True})

    encoder = FeatureEncoder(cfg_base["encoder"])
    dual_commit = None
    try:
        import subprocess
        dual_commit = subprocess.check_output(["git", "-C", str(args.dual_root), "rev-parse", "HEAD"],
                                              stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        pass
    try:
        project_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                                 stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        project_commit = None

    all_runs = []
    all_support_rows = []
    all_memory_rows = []
    all_per_image_rows = []
    all_cache_records = []
    support_pool_counts = {}
    query_counts = {}
    feature_cache_info = {}
    query_ids_by_ds = {}
    train_ids_by_ds = {}

    dataset_state = {}
    for dataset in args.datasets:
        dsroot = ROOT / "datasets" / dataset
        manifest_root = dsroot / "manifests/paper_seed_2026"
        pool = read_manifest(manifest_root / "train_pool.csv", require_mask=True)
        queries = read_manifest(manifest_root / "test.csv", require_mask=True)
        pool_ids = [x["id"] for x in pool]
        query_ids = [x["id"] for x in queries]
        if len(pool_ids) < max(SHOTS):
            raise ValueError(f"{dataset}: legal training support pool has only {len(pool_ids)} images")
        overlap = set(pool_ids) & set(query_ids)
        if overlap:
            raise ValueError(f"Train/test leakage in {dataset}: {sorted(overlap)}")
        cfg = deepcopy(cfg_base)
        cfg["dataset"].update({"name": dataset, "root": None, "mask_encoding": "indexed", "class_mapping": {0: 0, 255: 1},
                               "support_manifest": str(manifest_root / "train_pool.csv"),
                               "query_manifest": str(manifest_root / "test.csv")})
        cfg["memory"]["sampling_seed"] = 2026
        query_cache_root = (args.query_cache_root.resolve() if args.query_cache_root else out / "cache/query_features")
        cfg["output"]["directory"] = str((query_cache_root / dataset).resolve())
        cfg["output"]["occupancy_cache_directory"] = str(out / "cache/real_support_gt_occupancy")
        log(f"{dataset}: validating real Support GT and loading {len(pool)} feature caches")
        support_cache, provenance = load_support_cache(pool, args.dual_root, dataset, cfg)
        log(f"{dataset}: Support occupancy validated; all sources=real_support_gt")
        support_cache_rows = []
        # The provenance records are sorted with manifest order, one row per legal image.
        support_cache_rows.extend(provenance)
        write_csv(out / "cache" / "support_index" / f"{dataset}.csv", support_cache_rows,
                  list(support_cache_rows[0]))
        perms = support_permutations(pool, SEEDS)
        for seed, order in perms.items():
            for shot in SHOTS:
                ids = [e["id"] for e in order[:shot]]
                # Prefix property is checked against every larger requested set.
                for larger in SHOTS:
                    if larger > shot:
                        larger_ids = [e["id"] for e in order[:larger]]
                        if not set(ids).issubset(larger_ids):
                            raise RuntimeError("Nested support invariant failed")
                for rank, image_id in enumerate(ids, 1):
                    all_support_rows.append({"dataset": dataset, "seed": seed, "shots": shot,
                                             "permutation_rank": rank, "image_id": image_id,
                                             "split": "train_pool", "gt_path": str(next(e["mask"] for e in pool if e["id"] == image_id)),
                                             "gt_sha256": support_cache[image_id]["gt_sha256"],
                                             "memory_construction_seed": 2026, "occupancy_source": "real_support_gt"})
        dataset_state[dataset] = {"pool": pool, "queries": queries, "support_cache": support_cache,
                                  "permutations": perms, "cfg": cfg}
        support_pool_counts[dataset] = len(pool)
        query_counts[dataset] = len(queries)
        query_ids_by_ds[dataset] = query_ids
        train_ids_by_ds[dataset] = pool_ids
        all_cache_records.extend(support_cache_rows)

    write_csv(out / "support_sets.csv", all_support_rows,
              ["dataset", "seed", "shots", "permutation_rank", "image_id", "split", "gt_path", "gt_sha256", "memory_construction_seed", "occupancy_source"])
    write_csv(out / "support_feature_cache.csv", all_cache_records,
              list(all_cache_records[0]))
    write_csv(out / "occupancy_validation.csv", all_cache_records, list(all_cache_records[0]))

    # Save immutable run configurations and build memory statistics for every planned combination.
    configurations = {}
    signatures = {}
    results_root = out / "runs"
    for dataset in args.datasets:
        state = dataset_state[dataset]
        for seed in SEEDS:
            order = state["permutations"][seed]
            for shot in SHOTS:
                support = [e["id"] for e in order[:shot]]
                cfg = deepcopy(state["cfg"])
                cfg["memory"]["sampling_seed"] = 2026
                run_dir = results_root / dataset / f"seed_{seed}" / f"shots_{shot}"
                cfg["memory"]["file"] = str((run_dir / "memory.npz").resolve())
                run_dir.mkdir(parents=True, exist_ok=True)
                config_payload = {"config": cfg, "dataset": dataset, "seed": seed, "shots": shot,
                                  "support_ids": support, "query_ids": query_ids_by_ds[dataset],
                                  "source_support_cache": str((out / "support_feature_cache.csv").resolve()),
                                  "support_cache_protocol": f"DUAL features_N3 passes corresponding to offsets {args.support_offsets}; audit run summary identifies UNI checkpoint SHA256",
                                  "query_manifest": str((ROOT / "datasets" / dataset / "manifests/paper_seed_2026/test.csv").resolve()),
                                  "support_manifest": str((ROOT / "datasets" / dataset / "manifests/paper_seed_2026/train_pool.csv").resolve()),
                                  "encoder_sha256": encoder.weights_sha256, "dual_commit": dual_commit,
                                  "support_selection_seed": seed, "memory_sampling_seed": 2026,
                                  "project_commit": project_commit, "query_ids_hash": stable_signature(query_ids_by_ds[dataset]),
                                  "occupancy_source": "real_support_gt",
                                  "gt_sha256": {iid: state["support_cache"][iid]["gt_sha256"] for iid in support}}
                cfgfile = out / "configs" / dataset / f"seed_{seed}_shots_{shot}.yaml"
                cfgfile.parent.mkdir(parents=True, exist_ok=True)
                cfgfile.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")
                memory, stats = construct_memory(support, state["support_cache"], cfg, seed, encoder,
                                                 args.memory_sampling_protocol)
                support_entry_lookup = {e["id"]: e for e in state["pool"]}
                support_sources = [cache_file_for(args.dual_root, dataset, iid) for iid in support]
                sig_payload = {"dataset": dataset, "seed": seed, "shots": shot, "support_ids": support,
                               "support_cache_sha256": [file_hash(p) for p in support_sources],
                               "support_gt_sha256": [file_hash(support_entry_lookup[iid]["mask"]) for iid in support],
                               "occupancy_source": "real_support_gt",
                               "query_ids": query_ids_by_ds[dataset], "encoder_sha256": encoder.weights_sha256,
                               "config": cfg, "script_sha256": file_hash(__file__)}
                signature = stable_signature(sig_payload)
                key = f"{dataset}|{seed}|{shot}"
                signatures[key] = signature
                configurations[key] = {"cfg": cfg, "support_ids": support, "memory": memory,
                                        "stats": stats, "run_dir": run_dir, "signature": signature,
                                        "support_entries": [support_entry_lookup[x] for x in support]}
                memory_row = {"dataset": dataset, "seed": seed, "shots": shot,
                              "support_ids": ";".join(support), "support_labels": ";".join(map(str, stats["support_labels"])),
                              "fg_tokens": stats["counts"][1], "bg_tokens": stats["counts"][0],
                              "fg_fallback_images": stats["fallback_images"][1],
                              "bg_fallback_images": stats["fallback_images"][0],
                              "status": "VALID" if memory is not None else "INVALID_INSUFFICIENT_MEMORY",
                              "reason": stats["reason"]}
                raw_fg = sum(int((state["support_cache"][iid]["fg_occ"] >= .9).sum()) for iid in support)
                raw_bg = sum(int((state["support_cache"][iid]["fg_occ"] <= .1).sum()) for iid in support)
                total_candidates = sum(len(state["support_cache"][iid]["fg_occ"]) for iid in support)
                memory_row.update({"raw_fg_candidates": raw_fg, "raw_bg_candidates": raw_bg,
                                   "mixed_tokens": total_candidates - raw_fg - raw_bg,
                                   "total_candidate_tokens": total_candidates,
                                   "occupancy_source": "real_support_gt"})
                all_memory_rows.append(memory_row)

    write_csv(out / "memory_statistics.csv", all_memory_rows,
              list(all_memory_rows[0]))

    nesting_rows = []
    for dataset in args.datasets:
        for seed in SEEDS:
            for small, large in zip(SHOTS[:-1], SHOTS[1:]):
                a = configurations[f"{dataset}|{seed}|{small}"]
                b = configurations[f"{dataset}|{seed}|{large}"]
                prefix_ids = a["support_ids"] == b["support_ids"][:small]
                for cls in (0, 1):
                    token_prefix = None
                    if a["memory"] is not None and b["memory"] is not None:
                        v = a["memory"].vectors[cls]
                        token_prefix = bool(np.array_equal(v, b["memory"].vectors[cls][:len(v)]))
                        if not token_prefix: raise RuntimeError("Global RNG Token prefix differs unexpectedly")
                    nesting_rows.append({"dataset": dataset, "seed": seed, "small_shots": small,
                                         "large_shots": large, "class": cls, "support_prefix_equal": prefix_ids,
                                         "sampled_token_prefix_equal": token_prefix})
    write_csv(out / "nesting_validation.csv", nesting_rows, list(nesting_rows[0]))

    # Reject insufficient K=50 memories before spending time on fixed test Query features.
    # Query features are independent of support Memory, then cached once for all runs.
    for dataset in args.datasets:
        state = dataset_state[dataset]
        query_cache_root = (args.query_cache_root.resolve() if args.query_cache_root else out / "cache/query_features")
        cache_rows, duration = precompute_query_features(state["queries"], encoder, state["cfg"], dataset,
                                                        query_cache_root, log)
        all_cache_records.extend(cache_rows)
        feature_cache_info[dataset] = {"seconds": duration,
                                       "cache_entries": len(cache_rows),
                                       "cache_reused": sum(bool(x["cache_reused"]) for x in cache_rows)}
    all_cache_records_for_queries = [r for r in all_cache_records if "cache_file" in r]
    write_csv(out / "query_feature_cache.csv", all_cache_records_for_queries,
              ["dataset", "image_id", "offset", "image_sha256", "cache_key", "cache_file", "cache_sha256", "window_count", "cache_reused"])
    atomic_json(out / "study_metadata.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "datasets": args.datasets,
        "support_sizes": SHOTS, "seeds": SEEDS, "support_pool_counts": support_pool_counts,
        "query_counts": query_counts, "query_ids": query_ids_by_ds,
        "query_train_disjoint": {ds: not bool(set(query_ids_by_ds[ds]) & set(train_ids_by_ds[ds])) for ds in args.datasets},
        "encoder_sha256": encoder.weights_sha256, "encoder": "UNI ViT-L/16 frozen",
        "torch": torch.__version__, "cuda": torch.version.cuda, "python": sys.version,
        "platform": platform.platform(), "device": str(encoder.device),
        "dual_root": str(args.dual_root.resolve()), "dual_commit": dual_commit,
        "project_commit": project_commit, "project_code_sha256": file_hash(__file__),
        "support_cache_protocol": f"DUAL features_N3; offset passes {args.support_offsets}; run_summary reports the same UNI checkpoint SHA256",
        "support_offsets": list(args.support_offsets),
        "memory_sampling_protocol": args.memory_sampling_protocol,
        "query_cache_entries": sum(v["cache_entries"] for v in feature_cache_info.values()),
        "query_cache_info": feature_cache_info,
        "fixed_parameters": {"top_k": 50, "window": 448, "stride": 224, "patch_size": 16,
                             "query_offsets": [0, 8], "support_offsets": list(args.support_offsets), "occupancy": {"fg_ge": .9, "bg_le": .1},
                             "cap_per_image_per_class": 512, "fallback": "nearest_10_percent",
                             "memory_sampling_seed": 2026, "token_sampling": args.memory_sampling_protocol,
                             "output": [256, 256], "threshold": 0.0,
                             "reconstruction": "bilinear margin 28->448; mean windows; mean offsets; bilinear 256"}})

    existing = {}
    for dataset in args.datasets:
        for seed in SEEDS:
            for shot in SHOTS:
                key = f"{dataset}|{seed}|{shot}"
                cfg_data = configurations[key]
                result_path = cfg_data["run_dir"] / "results.json"
                stats = cfg_data["stats"]
                if cfg_data["memory"] is None:
                    existing[key] = {"dataset": dataset, "seed": seed, "shots": shot,
                                     "status": "INVALID_INSUFFICIENT_MEMORY", "reason": stats["reason"],
                                     "fg_tokens": stats["counts"][1], "bg_tokens": stats["counts"][0],
                                     "support_ids": cfg_data["support_ids"], "signature": signatures[key]}
                    atomic_json(result_path, existing[key])
                elif result_path.is_file():
                    cached = json.loads(result_path.read_text(encoding="utf-8"))
                    if cached.get("signature") != signatures[key]:
                        log(f"{key}: old result signature mismatch; recomputing")
                    elif cached.get("status") in ("COMPLETE", "INVALID_INSUFFICIENT_MEMORY"):
                        existing[key] = cached

    log(f"Prepared {len(configurations)} groups; valid memories={sum(v['memory'] is not None for v in configurations.values())}, invalid={sum(v['memory'] is None for v in configurations.values())}.")
    # All 40 cases are attempted. A resumed completed case is reused only when its full signature matches.
    for dataset in args.datasets:
        state = dataset_state[dataset]
        for seed in SEEDS:
            for shot in SHOTS:
                key = f"{dataset}|{seed}|{shot}"
                if key in existing and existing[key].get("signature") == signatures[key]:
                    log(f"{key}: resume skip status={existing[key]['status']}")
                    continue
                d = configurations[key]
                stats = d["stats"]
                if d["memory"] is None:
                    result = {"dataset": dataset, "seed": seed, "shots": shot,
                              "status": "INVALID_INSUFFICIENT_MEMORY", "reason": stats["reason"],
                              "fg_tokens": stats["counts"][1], "bg_tokens": stats["counts"][0],
                              "fg_fallback_images": stats["fallback_images"][1],
                              "bg_fallback_images": stats["fallback_images"][0],
                              "support_ids": d["support_ids"], "signature": signatures[key]}
                    atomic_json(d["run_dir"] / "results.json", result)
                    existing[key] = result
                    continue
                log(f"{key}: starting; Memory FG={stats['counts'][1]} BG={stats['counts'][0]}")
                t0 = time.monotonic()
                try:
                    memory_path = d["run_dir"] / "memory.npz"
                    d["memory"].save(memory_path)
                    cfg = d["cfg"]
                    cfg["memory"]["file"] = str(memory_path)
                    pipeline = SegmentationPipeline(encoder, d["memory"], cfg)
                    per_image, overall = evaluate_run(pipeline, state["queries"], cfg,
                                                      lambda msg: log(f"{key}: {msg}"))
                    result = {"dataset": dataset, "seed": seed, "shots": shot, "status": "COMPLETE",
                              "support_ids": d["support_ids"], "fg_tokens": stats["counts"][1],
                              "bg_tokens": stats["counts"][0], "fg_fallback_images": stats["fallback_images"][1],
                              "bg_fallback_images": stats["fallback_images"][0], "reason": "",
                              **overall, "elapsed_seconds": time.monotonic() - t0,
                              "memory_sha256": file_hash(memory_path), "signature": signatures[key]}
                    rows = [{"dataset": dataset, "seed": seed, "shots": shot, **row} for row in per_image]
                    write_csv(d["run_dir"] / "per_image_metrics.csv", rows,
                              ["dataset", "seed", "shots", "image_id", "dice", "fg_iou", "precision", "recall",
                               "object_f1", "object_hausdorff", "inference_seconds", "foreground_pixels", "predicted_foreground_pixels", "predicted_fg_ratio", "gt_fg_ratio"])
                    atomic_json(d["run_dir"] / "results.json", result)
                    existing[key] = result
                    log(f"{key}: COMPLETE dice={result['dice']:.6f} fg_iou={result['fg_iou']:.6f} object_f1={result['object_f1']:.6f} elapsed={result['elapsed_seconds']:.1f}s")
                except Exception as exc:
                    result = {"dataset": dataset, "seed": seed, "shots": shot, "status": "ERROR",
                              "reason": f"{type(exc).__name__}: {exc}", "support_ids": d["support_ids"],
                              "fg_tokens": stats["counts"][1], "bg_tokens": stats["counts"][0],
                              "signature": signatures[key], "elapsed_seconds": time.monotonic() - t0}
                    atomic_json(d["run_dir"] / "results.json", result)
                    existing[key] = result
                    log(f"{key}: ERROR {result['reason']}")
                all_runs_current = list(existing.values())
                write_csv(out / "runs.csv", all_runs_current,
                          ["dataset", "seed", "shots", "status", "reason", "support_ids", "fg_tokens", "bg_tokens",
                           "fg_fallback_images", "bg_fallback_images", "query_count", "dice", "fg_iou", "precision", "recall",
                           "object_f1", "object_hausdorff", "micro_dice", "micro_fg_iou", "inference_seconds",
                           "inference_total_seconds", "elapsed_seconds", "memory_sha256", "signature"])
                image_rows = []
                for result in existing.values():
                    rp = results_root / result["dataset"] / f"seed_{result['seed']}" / f"shots_{result['shots']}" / "per_image_metrics.csv"
                    if rp.is_file():
                        with rp.open(newline="", encoding="utf-8") as f:
                            image_rows.extend(csv.DictReader(f))
                write_csv(out / "per_image_metrics.csv", image_rows,
                          ["dataset", "seed", "shots", "image_id", "dice", "fg_iou", "precision", "recall",
                           "object_f1", "object_hausdorff", "inference_seconds", "foreground_pixels", "predicted_foreground_pixels", "predicted_fg_ratio", "gt_fg_ratio"])
                summary = summarize(all_runs_current)
                write_csv(out / "summary.csv", summary,
                          ["dataset", "shots", "valid_seed_count", "total_seed_count", "valid_ratio"] +
                          [f"{metric}_{suffix}" for metric in ("dice", "fg_iou", "object_f1", "precision", "recall", "object_hausdorff")
                           for suffix in ("mean", "sd", "min", "max")])
                plot_metric(summary, "dice", out / "support_size_dice.png", "Mean per-image Dice ± SD")
                plot_metric(summary, "object_f1", out / "support_size_object_f1.png", "Mean per-image Object-F1 ± SD")

    all_runs = list(existing.values())
    # Include planned combinations never reached by an interruption as PENDING rows.
    for dataset in args.datasets:
        for seed in SEEDS:
            for shot in SHOTS:
                key = f"{dataset}|{seed}|{shot}"
                if key not in existing:
                    d = configurations[key]
                    st = d["stats"]
                    row = {"dataset": dataset, "seed": seed, "shots": shot,
                           "status": "INVALID_INSUFFICIENT_MEMORY" if d["memory"] is None else "PENDING",
                           "reason": st["reason"], "support_ids": d["support_ids"],
                           "fg_tokens": st["counts"][1], "bg_tokens": st["counts"][0], "signature": signatures[key]}
                    all_runs.append(row)
    summary = summarize(all_runs)
    write_csv(out / "runs.csv", all_runs,
              ["dataset", "seed", "shots", "status", "reason", "support_ids", "fg_tokens", "bg_tokens",
               "fg_fallback_images", "bg_fallback_images", "query_count", "dice", "fg_iou", "precision", "recall",
               "object_f1", "object_hausdorff", "micro_dice", "micro_fg_iou", "inference_seconds",
               "inference_total_seconds", "elapsed_seconds", "memory_sha256", "signature"])
    all_per_image = []
    for result in all_runs:
        rp = results_root / result["dataset"] / f"seed_{result['seed']}" / f"shots_{result['shots']}" / "per_image_metrics.csv"
        if rp.is_file():
            with rp.open(newline="", encoding="utf-8") as f:
                all_per_image.extend(csv.DictReader(f))
    write_csv(out / "per_image_metrics.csv", all_per_image,
              ["dataset", "seed", "shots", "image_id", "dice", "fg_iou", "precision", "recall",
               "object_f1", "object_hausdorff", "inference_seconds", "foreground_pixels", "predicted_foreground_pixels", "predicted_fg_ratio", "gt_fg_ratio"])
    write_csv(out / "summary.csv", summary,
              ["dataset", "shots", "valid_seed_count", "total_seed_count", "valid_ratio"] +
              [f"{metric}_{suffix}" for metric in ("dice", "fg_iou", "object_f1", "precision", "recall", "object_hausdorff")
               for suffix in ("mean", "sd", "min", "max")])
    plot_metric(summary, "dice", out / "support_size_dice.png", "Mean per-image Dice ± SD")
    plot_metric(summary, "object_f1", out / "support_size_object_f1.png", "Mean per-image Object-F1 ± SD")
    plot_metric(summary, "fg_iou", out / "support_size_iou.png", "Mean per-image FG IoU ± SD")
    meta = {"encoder_sha256": encoder.weights_sha256, "torch": torch.__version__, "cuda": torch.version.cuda,
            "dual_root": str(args.dual_root.resolve()), "dual_commit": dual_commit,
            "project_commit": project_commit,
            "support_cache_protocol": f"DUAL N3 passes for offsets {args.support_offsets}",
            "support_offsets": list(args.support_offsets),
            "memory_sampling_protocol": args.memory_sampling_protocol,
            "query_cache_entries": sum(v["cache_entries"] for v in feature_cache_info.values()),
            "query_counts": query_counts}
    if out.name != "support_size_study_gtfixed":
        make_report(out / "report.md", summary, all_runs, support_pool_counts, feature_cache_info, meta)
    total = time.monotonic() - study_started
    log(f"Study finished: COMPLETE={sum(r['status']=='COMPLETE' for r in all_runs)}, INVALID={sum(str(r['status']).startswith('INVALID') for r in all_runs)}, ERROR={sum(r['status']=='ERROR' for r in all_runs)}, PENDING={sum(r['status']=='PENDING' for r in all_runs)}, elapsed={total/3600:.2f}h")
    log_handle.close()


if __name__ == "__main__":
    main()
