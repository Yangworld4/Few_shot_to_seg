"""One-shot manifests, original-image features, and historical feature caches."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .core.geometry import starts, windows
from .core.selfref_geometry import GRID, STRIDE, WINDOW
from .utils import sha256 as hash_file

WEIGHTS_SHA = "56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40"


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def selected_pairs(cfg, dataset):
    manifest = Path(cfg["manifest_dir"]) / f"{dataset.lower()}_splits.csv"
    rows = read_rows(manifest)
    by_id = {}
    for row in rows:
        iid = row["image_id"]
        if iid != Path(iid).name or iid in ("", ".", "..") or iid in by_id:
            raise ValueError(f"Invalid or duplicate image ID: {iid}")
        for field in ("image", "mask", "instance"):
            path = Path(row[field])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Expected relative dataset path: {path}")
            row[field] = str(Path(cfg["data_root"]) / path)
        by_id[iid] = row
    support = [r["image_id"] for r in rows if r["split"] == "Train"]
    test = [r["image_id"] for r in rows if r["split"] != "Train"]
    expected = {"GlaS": (85, 80), "CRAG": (173, 40)}
    if (len(support), len(test)) != expected[dataset]:
        raise ValueError(f"Unexpected Train/Test count in {manifest}")
    query = support if cfg["split"] == "train-oof" else test
    for field, available in (("support_ids", support), ("query_ids", query)):
        requested = cfg[field] or available
        if not set(requested) <= set(available) or len(set(requested)) != len(requested):
            raise ValueError(f"Invalid or duplicate {dataset} {field}: {requested}")
    support_filter, query_filter = set(cfg["support_ids"] or support), set(cfg["query_ids"] or query)
    if cfg["split"] == "train-oof":
        groups = [(qid, [support[(i + 1) % len(support)]]) for i, qid in enumerate(support)]
    else:
        groups = [(qid, support) for qid in test]
    groups = [(by_id[qid], [by_id[sid] for sid in ids if sid in support_filter])
              for qid, ids in groups if qid in query_filter and support_filter.intersection(ids)]
    if not groups:
        raise ValueError(f"No {dataset} pairs after applying ID filters")
    fp_rows = read_rows(Path(cfg["manifest_dir"]) / f"{dataset.lower()}_fp_high.csv")
    fp_high = {(r["support_id"], r["query_id"]): r["FP_high"].lower() == "true" for r in fp_rows}
    if cfg["split"] == "test" and any((support_row["image_id"], query_row["image_id"]) not in fp_high
                                     for query_row, group in groups for support_row in group):
        raise ValueError(f"Incomplete {dataset} FP_high manifest")
    return groups, fp_high


def shape(row):
    return int(row["height"]), int(row["width"])


def load_mask(row, field):
    path = row[field]
    if hash_file(path) != row[field + "_sha256"]:
        raise ValueError(f"Mask checksum differs from manifest: {path}")
    with Image.open(path) as image:
        values = np.array(image)
    if values.shape != shape(row):
        raise ValueError(f"Unexpected mask shape: {path}")
    return values


def _image_pixel_sha(row):
    with Image.open(row["image"]) as image:
        return hashlib.sha256(np.asarray(image.convert("RGB")).tobytes()).hexdigest()


def _check_phase(values, positions, row, phase, path):
    h, w = shape(row)
    expected = [(y, x) for y in starts(h, WINDOW, STRIDE, phase)
                for x in starts(w, WINDOW, STRIDE, phase)]
    if (values.dtype != np.float16 or values.shape != (len(expected), GRID**2, 1024)
            or not np.array_equal(positions, expected)):
        raise ValueError(f"Unexpected feature shape, dtype, or positions: {path}")
    return values.reshape(-1, 1024), positions


def _normalize_phases(values, device):
    matrix = torch.as_tensor(np.concatenate(values), device=device)
    normalized = F.normalize(matrix, dim=1).half().cpu().numpy()
    return np.split(normalized, np.cumsum([len(v) for v in values])[:-1])


def read_original_image(row):
    path = row["image"]
    if hash_file(path) != row["image_sha256"]:
        raise ValueError(f"Image checksum differs from manifest: {path}")
    with Image.open(path) as source:
        image = np.asarray(source.convert("RGB"))
    if image.shape[:2] != shape(row):
        raise ValueError(f"Unexpected image shape: {path}")
    return image


def load_image_features(encoder, cfg, _dataset, row, query=False):
    image = read_original_image(row)

    values, positions = [], []
    for phase in cfg["phases"]:
        crops = list(windows(image, WINDOW, STRIDE, phase))
        pos = np.asarray([(y, x) for y, x, _ in crops])
        tokens = encoder.encode(crop for _, _, crop in crops)
        matrix, pos = _check_phase(tokens, pos, row, phase, row["image"])
        values.append(matrix.astype(np.float32))
        positions.append(pos)
    if query:
        if row["split"] == "Train":
            values = _normalize_phases(values, cfg["device"])
        else:
            values = [_normalize_phases([value], cfg["device"])[0] for value in values]
    return values, positions


def _raw_phases(root, dataset, row, phases, key, normalized):
    role = "support" if row["split"] == "Train" else "query"
    features, positions = [], []
    image_sha = None
    for phase in phases:
        path = root / dataset / role / row["image_id"] / f"offset_{phase}.npz"
        with np.load(path, allow_pickle=False) as cache:
            metadata = json.loads(str(cache["metadata"]))
            if "image_sha256" in metadata and image_sha is None:
                image_sha = _image_pixel_sha(row)
            if metadata.get("weights_sha256") != WEIGHTS_SHA or metadata.get("image_sha256") not in (
                    None, row["image_sha256"], image_sha):
                raise ValueError(f"Feature provenance differs from manifest: {path}")
            values, pos = _check_phase(cache[key], cache["positions"], row, phase, path)
        features.append(values if normalized else values.astype(np.float32))
        positions.append(pos)
    return features, positions


def _n3_phases(cfg, dataset, row):
    root = Path(cfg["feature_cache"])
    iid = row["image_id"]
    features, positions = [], []
    if row["split"] == "Train":
        dual = Path(cfg["n3_dual_root"]) / "outputs" / f"uni_audit_mask_quality_{dataset}_seed1337/cache/features_N3"
        candidates = [dual / folder / f"{iid}.npz" for folder in ("labeled", "unlabeled", "validation")]
        candidates.append(root / f"oneshot_full_enum/support_feature_cache/{dataset}/{iid}.npz")
        path = next((file for file in candidates if file.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"No historical N3 Support features for {dataset}/{iid}: {candidates}")
        with np.load(path, allow_pickle=False) as cache:
            if "metadata" in cache and json.loads(str(cache["metadata"])).get("weights_sha256") != WEIGHTS_SHA:
                raise ValueError(f"Unexpected N3 checkpoint: {path}")
            values = cache["features"]
            for phase in cfg["phases"]:
                keep = cache["passno"] == (phase // 8)
                pos = np.stack((cache["y"][keep], cache["x"][keep]), axis=1)
                matrix, pos = _check_phase(values[keep], pos, row, phase, path)
                features.append(matrix)
                positions.append(pos)
    else:
        image_sha = _image_pixel_sha(row)
        for phase in cfg["phases"]:
            path = root / f"uni_shift_stitch_seed1337/feature_cache/{dataset}/{iid}_dy{phase}_dx{phase}.npz"
            with np.load(path, allow_pickle=False) as cache:
                metadata = json.loads(str(cache["key_json"]))
                if metadata["weights_sha256"] != WEIGHTS_SHA or metadata["image_sha256"] != image_sha:
                    raise ValueError(f"Unexpected N3 provenance: {path}")
                matrix, pos = _check_phase(cache["tokens"], cache["positions"], row, phase, path)
            features.append(matrix)
            positions.append(pos)
    return features, positions


def load_features(cfg, dataset, row, query=False):
    if cfg["feature_format"] == "n3":
        return _n3_phases(cfg, dataset, row)
    raw = Path(cfg["feature_cache"])
    key = f"f{cfg['layer']}"
    if not query:
        return _raw_phases(raw, dataset, row, cfg["phases"], key, False)
    normalized = Path(cfg["normalized_cache"]) if cfg["normalized_cache"] else raw.with_name("normalized_features")
    def load_group(phases):
        try:
            return _raw_phases(normalized, dataset, row, phases, key, True)
        except (FileNotFoundError, KeyError):
            values, pos = _raw_phases(raw, dataset, row, phases, key, False)
            return _normalize_phases(values, cfg["device"]), pos

    # Accepted OOF computes both phases together if either normalized phase is absent.
    if row["split"] == "Train":
        return load_group(cfg["phases"])
    pairs = [load_group([phase]) for phase in cfg["phases"]]
    return [features[0] for features, _ in pairs], [positions[0] for _, positions in pairs]
