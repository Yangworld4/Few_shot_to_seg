import copy
from pathlib import Path
import yaml


def load_config(path, overrides=None):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a YAML mapping")
    cfg = copy.deepcopy(cfg)
    for key, value in (overrides or {}).items():
        section, field = key.split(".", 1)
        if section not in cfg or field not in cfg[section]:
            raise ValueError(f"Unknown configuration key: {key}")
        if key == "memory.occupancy_thresholds" and "memory.class_ids" not in (overrides or {}) and isinstance(value, dict):
            cfg[section][field].update(value)
        else:
            cfg[section][field] = value
    validate(cfg)
    return cfg


def validate(c):
    for section in ("encoder", "window", "memory", "retrieval", "segmentation", "dataset", "evaluation", "output"):
        if section not in c:
            raise ValueError(f"Missing configuration section: {section}")
    patches = {"uni": 16, "uni2_h": 14, "kaiko_vitb8": 8}
    if c["encoder"]["name"] not in patches:
        raise ValueError("Unknown encoder.name")
    patch = patches[c["encoder"]["name"]]
    if not isinstance(c["encoder"]["batch_size"], int) or c["encoder"]["batch_size"] < 1:
        raise ValueError("encoder.batch_size must be a positive integer")
    w = c["window"]
    if not isinstance(w["size"], int) or w["size"] < patch or w["size"] % patch:
        raise ValueError(f"Window size must be a positive multiple of encoder patch size {patch}")
    if not isinstance(w["stride"], int) or not 0 < w["stride"] <= w["size"]:
        raise ValueError("Stride must be in (0, window size]")
    if not w["offsets"] or any(not isinstance(x, int) or x < 0 or x >= w["size"] for x in w["offsets"]):
        raise ValueError("Offsets must be nonempty integers in [0, window size)")
    if not w["support_offsets"] or any(not isinstance(x, int) or x < 0 or x >= w["size"] for x in w["support_offsets"]):
        raise ValueError("Support offsets must be nonempty integers in [0, window size)")
    if w["padding_mode"] != "reflect" or w["interpolation"] != "bilinear":
        raise ValueError("Only reflect padding and bilinear window reconstruction are implemented")
    m = c["memory"]
    if not isinstance(m["class_ids"], list) or any(not isinstance(x, int) for x in m["class_ids"]):
        raise ValueError("memory.class_ids must be a list of integer labels")
    ids = [int(x) for x in m["class_ids"]]
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError("Memory requires at least two unique class IDs")
    thresholds = {int(k): float(v) for k, v in m["occupancy_thresholds"].items()}
    if set(thresholds) != set(ids) or any(not 0 < v <= 1 for v in thresholds.values()):
        raise ValueError("Each class needs a purity threshold in (0,1]")
    if not isinstance(m["cap_per_image_per_class"], int) or m["cap_per_image_per_class"] < 1:
        raise ValueError("Memory cap must be positive")
    if not isinstance(m["sampling_seed"], int):
        raise ValueError("Memory sampling seed must be an integer")
    if m["fallback_strategy"] not in ("nearest_10_percent", "none"):
        raise ValueError("Unsupported fallback strategy")
    r = c["retrieval"]
    if (not isinstance(r["top_k"], int) or r["top_k"] < 1 or
            not isinstance(r["chunk_size"], int) or r["chunk_size"] < 1 or
            r["metric"] != "cosine" or r["aggregation"] != "mean"):
        raise ValueError("Retrieval requires positive top_k/chunk_size, cosine and mean")
    s = c["segmentation"]
    if s["interpolation"] != "bilinear" or s["overlap_fusion"] != "mean" or s["offset_fusion"] != "mean":
        raise ValueError("Only bilinear interpolation and mean fusion are implemented")
    if s["margin_precision"] not in ("float16", "float32"):
        raise ValueError("margin_precision must be float16 or float32")
    if s["output_size"] != "native" and (not isinstance(s["output_size"], list) or len(s["output_size"]) != 2 or
                                        any(not isinstance(v, int) or v < 1 for v in s["output_size"])):
        raise ValueError("output_size must be native or [height,width]")
    if c["dataset"]["mask_encoding"] not in ("indexed", "rgb"):
        raise ValueError("mask_encoding must be indexed or rgb")
    if c["encoder"]["precision"] not in ("float32", "float16"):
        raise ValueError("precision must be float32 or float16")
    if c["encoder"]["token_precision"] not in ("float32", "float16"):
        raise ValueError("token_precision must be float32 or float16")
    e = c["evaluation"]
    if int(e["object_min_area"]) < 1 or not 0 < float(e["object_match_overlap"]) < 1:
        raise ValueError("Invalid object metric area or overlap threshold")


def resolved_path(value, config_path):
    if value is None:
        return None
    p = Path(value).expanduser()
    return p if p.is_absolute() else (Path(config_path).resolve().parent / p).resolve()
