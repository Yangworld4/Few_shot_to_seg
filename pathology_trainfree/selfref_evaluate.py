"""Run the one-shot protocol and summarize sensitivity to Support choice."""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
from time import monotonic

import numpy as np
from PIL import Image
import torch
import yaml

from .core.selfref import CachedQuery, build_support, predict_margins, prepare_support
from .core.selfref_geometry import reconstruct_native
from .selfref_config import ROOT
from .selfref_data import WEIGHTS_SHA, hash_file, load_features, load_image_features, load_mask, selected_pairs, shape
from .utils.selfref_metrics import evaluate_mask, summarize_pairs, summarize_supports

FIELDS = ("dataset", "split", "support_id", "query_id", "branch", "dice", "miou", "object_f1",
          "object_dice", "object_hd", "missed_glands", "fp_high", "fg_bank_size", "bg_bank_size")
STRIPE = 8


def _csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _evaluate_stripe(dataset, row, positions, bundle, cfg, destination):
    torch.set_num_threads(1)
    output = []
    qid = row["image_id"]
    for sid, sizes, fp_high, margins, prepared, stats in bundle:
        previous = {}
        predictions = {}
        for branch, margin in margins.items():
            if branch in prepared:
                previous[id(margin)] = prepared[branch]
            elif id(margin) not in previous:
                native = reconstruct_native(margin, positions, shape(row), cfg["reconstruction"])
                previous[id(margin)] = (native, native > cfg["threshold"])
            predictions[branch] = previous[id(margin)]
        predictions.update(prepared)
        scored = {}
        for branch, (native, mask) in predictions.items():
            if id(mask) not in scored:
                scored[id(mask)] = evaluate_mask(dataset, row, mask)
            metrics = scored[id(mask)]
            if cfg["save_margins"] or cfg["save_masks"]:
                folder = destination / "predictions" / dataset / sid / qid
                folder.mkdir(parents=True, exist_ok=True)
                if cfg["save_margins"] and native is not None:
                    np.save(folder / f"{branch}_margin.npy", native, allow_pickle=False)
                if cfg["save_masks"]:
                    Image.fromarray(mask.astype(np.uint8) * 255).save(folder / f"{branch}_mask.png")
            output.append(dict(dataset=dataset, split=cfg["split"], support_id=sid, query_id=qid,
                               branch=branch, fp_high=fp_high, bg_bank_size=sizes[0], fg_bank_size=sizes[1],
                               **metrics, **stats.get(branch, {})))
    return output


def _build_banks(dataset, groups, cfg, features):
    supports = {support["image_id"]: support for _, group in groups for support in group}

    def prepare(row):
        matrices, positions = features(dataset, row)
        return prepare_support(matrices, positions, load_mask(row, "mask") > 0, cfg)

    banks = {}
    prefetch = 1 if cfg["feature_source"] == "images" else max(1, min(4, cfg["num_workers"]))
    with ThreadPoolExecutor(prefetch) as pool:
        rows = list(supports.items())
        for start in range(0, len(rows), prefetch):
            batch = rows[start:start + prefetch]
            prepared = pool.map(prepare, (row for _, row in batch))
            for (sid, _), (values, indices) in zip(batch, prepared):
                banks[sid] = build_support(values, indices, cfg)
            if len(banks) % 32 == 0:
                print(f"{dataset}: built {len(banks)}/{len(supports)} Support banks", flush=True)
    if torch.device(cfg["device"]).type == "cuda" and cfg["feature_format"] == "f_raw":
        free, _ = torch.cuda.mem_get_info()
        required = sum(b.features.numel() * b.features.element_size() for b in banks.values())
        if required < .35 * free:
            for bank in banks.values():
                bank.features = bank.features.to(cfg["device"])
                bank.indices = tuple(ids.to(cfg["device"]) for ids in bank.indices)
    return banks


def _run_dataset(dataset, groups, fp_high, cfg, features, destination, stream, pool, timings, refinement):
    started = monotonic()
    banks = _build_banks(dataset, groups, cfg, features)
    timings["bank_seconds"] = monotonic() - started
    print(f"{dataset}: {len(banks)} Support banks ready in {timings['bank_seconds']:.1f}s", flush=True)
    writer = csv.DictWriter(stream, fieldnames=FIELDS + refinement.fields if refinement else FIELDS)
    pending = set()
    results = []

    def collect(rows):
        writer.writerows(rows)
        stream.flush()
        results.extend(rows)

    for idx, (row, supports) in enumerate(groups, 1):
        if torch.device(cfg["device"]).type == "cuda":
            torch.cuda.empty_cache()
        values, positions = features(dataset, row, query=True)
        if refinement:
            refinement.start_query(row)
        query = CachedQuery(values, positions, shape(row), cfg)
        for start in range(0, len(supports), STRIPE):
            bundle = []
            for support in supports[start:start + STRIPE]:
                sid, qid = support["image_id"], row["image_id"]
                bank = banks[sid]
                margins = predict_margins(query, bank, cfg)
                prepared, stats = refinement.prepare(margins, positions, row) if refinement else ({}, {})
                bundle.append((sid, bank.sizes, fp_high.get((sid, qid), False), margins, prepared, stats))
            args = (dataset, row, positions, bundle, cfg, destination)
            if pool is None:
                collect(_evaluate_stripe(*args))
            else:
                pending.add(pool.submit(_evaluate_stripe, *args))
                if len(pending) >= 2 * cfg["num_workers"]:
                    finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for job in finished:
                        collect(job.result())
        del query
        if idx % 10 == 0 or idx == len(groups):
            print(f"{dataset}: scored {idx}/{len(groups)} Query images", flush=True)
    for job in pending:
        collect(job.result())
    timings["wall_seconds"] = monotonic() - started
    return results


def _resolved_config(cfg, checkpoint_sha256):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args])

    metadata = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                    torch_version=str(torch.__version__), checkpoint_sha256=checkpoint_sha256)
    try:
        metadata.update(git_commit=git("rev-parse", "HEAD").decode().strip(),
                        git_diff_sha256=hashlib.sha256(git("diff", "--no-ext-diff", "HEAD", "--")).hexdigest())
    except (OSError, subprocess.CalledProcessError):
        metadata.update(git_commit=None, git_diff_sha256=None)
    sources = [ROOT / "scripts/evaluate.py", ROOT / "scripts/evaluate_selfref.py",
               *ROOT.glob("pathology_trainfree/**/selfref*.py"),
               *ROOT.glob("pathology_trainfree/**/sam*.py"),
               ROOT / "pathology_trainfree/utils/evaluation_v6.py", ROOT / "pathology_trainfree/core/geometry.py",
               ROOT / "pathology_trainfree/models/encoder.py", ROOT / "pathology_trainfree/cli.py"]
    metadata["source_sha256"] = {str(path.relative_to(ROOT)): hash_file(path) for path in sources}
    metadata["manifest_sha256"] = {}
    for dataset in cfg["dataset"]:
        for kind in ("splits", "fp_high"):
            path = Path(cfg["manifest_dir"]) / f"{dataset.lower()}_{kind}.csv"
            metadata["manifest_sha256"][str(path)] = hash_file(path)
    return {**cfg, "provenance": metadata}


def run_selfref(cfg):
    from_images = cfg["feature_source"] == "images"
    if not from_images and not Path(cfg["feature_cache"]).is_dir():
        raise FileNotFoundError(f"Set one_shot.feature_cache to an existing directory: {cfg['feature_cache']}")
    device = torch.device(cfg["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device if device.index is not None else torch.cuda.current_device())
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.set_num_threads(1)
    inputs = {dataset: selected_pairs(cfg, dataset) for dataset in cfg["dataset"]}
    destination = Path(cfg["output_dir"]) / cfg["tag"]
    refinement = None
    branches = list(cfg["branches"])
    fields = FIELDS
    if cfg["sam"]["enabled"]:
        from .plugins.sam import SamRefinement

        refinement = SamRefinement(cfg)
        branches.extend(refinement.branches)
        fields += refinement.fields
    if from_images:
        from .models.encoder import UNIEncoder

        encoder = UNIEncoder(cfg["encoder_weights"], cfg["layer"], cfg["device"])
        checkpoint_sha256 = encoder.weights_sha256
        features = partial(load_image_features, encoder, cfg)
    else:
        if cfg["normalized_cache"] is None and cfg["feature_format"] == "f_raw":
            cfg["normalized_cache"] = str(Path(cfg["feature_cache"]).with_name("normalized_features"))
        checkpoint_sha256 = WEIGHTS_SHA
        features = partial(load_features, cfg)
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(f"Output already exists: {destination}. Choose a new one_shot.tag.") from None
    resolved = _resolved_config(cfg, checkpoint_sha256)
    provenance = resolved["provenance"]
    provenance.update(status="running", timings={dataset: {} for dataset in inputs})
    config_path = destination / "resolved_config.yaml"
    started = monotonic()
    config_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    try:
        executor = (ProcessPoolExecutor(cfg["num_workers"], mp_context=mp.get_context("spawn"))
                    if cfg["num_workers"] else nullcontext())
        partial_path = destination / "per_pair.csv.partial"
        rows = []
        with executor as pool, partial_path.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=fields).writeheader()
            for dataset, (groups, fp_high) in inputs.items():
                rows.extend(_run_dataset(dataset, groups, fp_high, cfg, features, destination, stream, pool,
                                         provenance["timings"][dataset], refinement))
        expected = sum(len(supports) for groups, _ in inputs.values() for _, supports in groups) * len(branches)
        keys = {(r["dataset"], r["support_id"], r["query_id"], r["branch"]) for r in rows}
        if len(rows) != expected or len(keys) != expected:
            raise RuntimeError(f"Expected {expected} unique output rows, got {len(rows)} ({len(keys)} unique)")
        branch_order = {name: index for index, name in enumerate(branches)}
        rows.sort(key=lambda r: (r["dataset"], r["support_id"], r["query_id"], branch_order[r["branch"]]))
        _csv(destination / "per_pair.csv", rows, fields)
        partial_path.unlink()
        per_support = summarize_supports(rows)
        _csv(destination / "per_support.csv", per_support, list(per_support[0]))
        summaries = summarize_pairs(rows, cfg["failure_threshold"])
        if refinement:
            comparison = refinement.comparison(rows)
            _csv(destination / "sam_comparison.csv", comparison, list(comparison[0]))
        for name in [None, *branches]:
            active = [r for r in summaries if name is None or r["branch"] == name]
            folder = destination / name if name else destination
            folder.mkdir(exist_ok=True)
            _csv(folder / "summary.csv", active, list(summaries[0]))
            (folder / "summary.json").write_text(json.dumps(active, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        provenance["status"] = "complete"
        scores = ", ".join(f"{r['dataset']} {r['branch']} Dice={r['dice']:.6f}" for r in summaries)
        print(f"{scores} | {len(rows)} rows | {destination}", flush=True)
    except BaseException as exc:
        provenance.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        provenance.update(finished_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=monotonic() - started)
        config_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
