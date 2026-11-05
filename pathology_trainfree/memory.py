import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .geometry import windows
from .data import load_image, load_mask


class ReferenceMemory:
    def __init__(self, vectors, metadata):
        self.vectors = {int(k): np.asarray(v) for k, v in vectors.items()}
        self.metadata = metadata
        if not self.vectors or any(v.ndim != 2 or len(v) == 0 for v in self.vectors.values()):
            raise ValueError("Memory contains an empty class")
        dims = {v.shape[1] for v in self.vectors.values()}
        if len(dims) != 1:
            raise ValueError("Memory feature dimensions differ")

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, metadata=np.array(json.dumps(self.metadata)),
                            **{f"class_{k}": v for k, v in self.vectors.items()})

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as file:
            vectors = {int(k[6:]): file[k] for k in file.files if k.startswith("class_")}
            metadata = json.loads(str(file["metadata"]))
        return cls(vectors, metadata)

    def validate_for(self, cfg, encoder):
        if sorted(self.vectors) != sorted(int(x) for x in cfg["memory"]["class_ids"]):
            raise ValueError("Memory class IDs differ from configuration")
        if self.metadata.get("encoder_sha256") != encoder.weights_sha256:
            raise ValueError("Memory encoder fingerprint differs from loaded encoder")
        for name in ("normalize", "precision", "token_precision"):
            key = f"encoder_{name}"
            if key in self.metadata and self.metadata[key] != getattr(encoder, name):
                raise ValueError(f"Loaded Memory was built with different encoder.{name}")
        if next(iter(self.vectors.values())).shape[1] != encoder.feature_dim:
            raise ValueError("Memory feature dimension differs from encoder")
        saved_memory = self.metadata.get("memory_config")
        if saved_memory:
            for field in ("class_ids", "occupancy_thresholds", "cap_per_image_per_class", "sampling_seed",
                          "normalize", "fallback_strategy"):
                if json.dumps(saved_memory[field], sort_keys=True) != json.dumps(cfg["memory"][field], sort_keys=True):
                    raise ValueError(f"Loaded Memory was built with different memory.{field}; rebuild it or use matching config")
        saved_window = self.metadata.get("window")
        if saved_window:
            for field in ("size", "stride", "support_offsets", "padding_mode"):
                if saved_window[field] != cfg["window"][field]:
                    raise ValueError(f"Loaded Memory was built with different window.{field}; rebuild it or use matching config")
        saved_dataset = self.metadata.get("dataset_config")
        if saved_dataset:
            for field in ("mask_encoding", "class_mapping"):
                if json.dumps(saved_dataset[field], sort_keys=True) != json.dumps(cfg["dataset"][field], sort_keys=True):
                    raise ValueError(f"Loaded Memory was built with different dataset.{field}")
        k = int(cfg["retrieval"]["top_k"])
        if any(len(v) < k for v in self.vectors.values()):
            raise ValueError(f"Top-K={k} exceeds a class memory size")


def select_tokens(features, occupancy, cfg, rng):
    """Select up to cap tokens per class from one support image."""
    features = np.asarray(features).reshape(-1, features.shape[-1])
    occupancy = np.asarray(occupancy).reshape(-1, len(cfg["class_ids"]))
    cap = int(cfg["cap_per_image_per_class"])
    ids = [int(i) for i in cfg["class_ids"]]
    thresholds = {int(k): float(v) for k, v in cfg["occupancy_thresholds"].items()}
    selected, fallback = {}, {}
    # The verified binary N3 cache samples FG first, then BG, from one RNG.
    selection_order = [1, 0] if ids == [0, 1] else ids
    for cls in selection_order:
        j = ids.index(cls)
        row = occupancy[:, j]
        candidates = np.flatnonzero(row >= thresholds[cls])
        used_fallback = False
        if not len(candidates) and cfg["fallback_strategy"] == "nearest_10_percent":
            size = max(1, min(cap, math.ceil(len(row) * 0.1)))
            candidates = np.argsort(-row, kind="stable")[:size]
            used_fallback = True
        if len(candidates) > cap:
            candidates = np.sort(rng.choice(candidates, cap, replace=False))
        selected[cls] = features[candidates]
        fallback[cls] = used_fallback
    return selected, fallback


@torch.inference_mode()
def build_memory(entries, encoder, cfg):
    w, m = cfg["window"], cfg["memory"]
    ids = [int(x) for x in m["class_ids"]]
    rng = np.random.default_rng(int(m["sampling_seed"]))
    parts = {i: [] for i in ids}
    fallback_counts = {i: 0 for i in ids}
    grid = w["size"] // encoder.patch_size
    for entry in entries:
        image = load_image(entry["image"])
        mask = load_mask(entry["mask"], cfg["dataset"])
        if image.shape[:2] != mask.shape:
            raise ValueError(f"Image/mask size mismatch for {entry['id']}")
        image_windows = [view for offset in w["support_offsets"] for view in windows(image, w["size"], w["stride"], offset)]
        mask_windows = [view for offset in w["support_offsets"] for view in windows(mask, w["size"], w["stride"], offset)]
        tokens = encoder.encode([crop for _, _, crop in image_windows])
        occupancy = []
        for _, _, crop in mask_windows:
            one_hot = torch.stack([torch.from_numpy((crop == cls).astype(np.float32)) for cls in ids])[None]
            pooled = F.avg_pool2d(one_hot, encoder.patch_size, encoder.patch_size)
            occupancy.append(pooled[0].permute(1, 2, 0).reshape(grid * grid, len(ids)).numpy())
        occupancy = np.stack(occupancy)
        chosen, fallback = select_tokens(tokens, occupancy, m, rng)
        for cls in ids:
            if len(chosen[cls]):
                parts[cls].append(chosen[cls])
            fallback_counts[cls] += int(fallback[cls])
    vectors = {}
    for cls in ids:
        if not parts[cls]:
            raise ValueError(f"Empty memory for class {cls}")
        vec = np.concatenate(parts[cls])
        # UNI already returns L2-normalized tokens, rounded to fp16 like N3.
        # Preserve those bytes in the memory file; retrieval renormalizes in fp32.
        if m["normalize"] and not np.allclose(np.linalg.norm(vec.astype(np.float32), axis=1), 1, atol=1e-3):
            vec = F.normalize(torch.from_numpy(vec.astype(np.float32)), dim=1).numpy()
        vectors[cls] = vec
    metadata = {"class_ids": ids, "counts": {str(k): len(v) for k, v in vectors.items()},
                "feature_dim": encoder.feature_dim, "encoder_sha256": encoder.weights_sha256,
                "encoder_name": getattr(encoder, "name", "uni"), "encoder_patch_size": encoder.patch_size,
                "encoder_config": getattr(encoder, "config_path", None),
                "encoder_normalize": getattr(encoder, "normalize", True),
                "encoder_precision": getattr(encoder, "precision", "float32"),
                "encoder_token_precision": getattr(encoder, "token_precision", "float16"),
                "support_ids": [e["id"] for e in entries], "seed": m["sampling_seed"],
                "occupancy_thresholds": m["occupancy_thresholds"], "cap": m["cap_per_image_per_class"],
                "fallback_strategy": m["fallback_strategy"], "fallback_counts": fallback_counts,
                "window": w, "memory_config": m,
                "dataset_config": {"mask_encoding": cfg["dataset"]["mask_encoding"],
                                   "class_mapping": cfg["dataset"]["class_mapping"]}}
    return ReferenceMemory(vectors, metadata)
