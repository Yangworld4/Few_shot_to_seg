"""Configuration for cache-only Native one-shot evaluation."""

import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/eval_query_refinement.yaml"
BRANCHES = ("support_only", "query_refined")
SAM_DEFAULTS = dict(enabled=False, model_type="vit_l", checkpoint=None, base_branch="auto",
                    iou_threshold=.5, opening_radius=1, negative_points=3, negative_radius=64,
                    negative_spacing=12, max_hole_area=256, min_area=32)
DEFAULTS = dict(
    dataset=["GlaS", "CRAG"], split="test", support_ids=None, query_ids=None,
    data_root=str(ROOT / "datasets"), manifest_dir=str(ROOT / "configs/selfref_manifests"),
    feature_cache=str(ROOT / "outputs/uni_layernorm_depth_diag/raw_features"),
    feature_source="cache", encoder_weights=None, feature_format="f_raw", normalized_cache=None,
    n3_dual_root=str(ROOT.parent / "DUAL"),
    layer=19, phases=[0, 8], min_valid_pixels=128, purity_fg=.9, purity_bg=.1,
    bank_select="kcenter", budget_mode="ratio", rho=.125, budget_cap=2048, random_seed=2026,
    support_topk=5, refine_query=True, seed_ratio=.3, selfref_topk=50, exclude_same_position=True,
    threshold=0., reconstruction="bilinear", failure_threshold=.8, branches=None,
    output_dir=str(ROOT / "outputs/selfref"), tag="query_refinement", save_margins=False, save_masks=False,
    cache_dir=None, device="cuda", num_workers=8,
)
CHOICES = {
    "dataset": ("GlaS", "CRAG"), "split": ("test", "train-oof"),
    "feature_source": ("images", "cache"), "feature_format": ("f_raw", "n3"),
    "bank_select": ("kcenter", "random", "all"),
    "budget_mode": ("ratio", "fixed"), "branches": BRANCHES,
    "reconstruction": ("bilinear", "bilinear-window", "nearest"),
}
PATHS = ("data_root", "manifest_dir", "encoder_weights", "feature_cache", "normalized_cache",
         "n3_dual_root", "output_dir", "cache_dir")
HELP = {
    "feature_source": "Encode original images or read an existing cache.",
    "encoder_weights": "Path to the local UNI checkpoint used with original images.",
    "feature_cache": "Directory of cached UNI features.",
    "bank_select": "Select Support tokens by k-center, random sampling, or use all candidates.",
    "budget_mode": "Use a shared ratio budget or a fixed token cap per class.",
    "rho": "Fraction of Support candidates used to calculate the shared ratio budget.",
    "budget_cap": "Maximum tokens per class when using a fixed budget.",
    "support_topk": "Number of retrieved Support tokens per class.",
    "refine_query": "Refine the prediction using confident tokens from the Query image.",
    "branches": "Choose which predictions to save and score.",
    "seed_ratio": "Fraction of confident tokens selected as seeds on each side.",
    "selfref_topk": "Number of retrieved Query seeds per class used for refinement.",
    "tag": "Run directory name under --output-dir.",
}


def _layer(value):
    return int(str(value).lower().removeprefix("f"))


def _boolean(value):
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise argparse.ArgumentTypeError("Use true or false")


def branch_name(name):
    """Read branch names from current and archived runs."""
    return {"e0": "support_only", "e1a": "query_refined"}.get(name, name)


def _legacy_refinement(value):
    if value in ("none", "e1a"):
        return value != "none"
    raise argparse.ArgumentTypeError("Use --refine-query true or false")


def _validate(cfg):
    for key, choices in CHOICES.items():
        values = cfg[key]
        if key in ("dataset", "branches"):
            values = values if isinstance(values, list) else [values]
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{key} must contain unique values")
            cfg[key] = values
        else:
            values = [values]
        if any(value not in choices for value in values):
            raise ValueError(f"{key} must use {', '.join(choices)}")
    for key in ("budget_cap", "support_topk", "selfref_topk", "min_valid_pixels"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("rho", "seed_ratio"):
        if not 0 < cfg[key] <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    if not 0 <= cfg["purity_bg"] < cfg["purity_fg"] <= 1:
        raise ValueError("Purity must satisfy 0 <= BG < FG <= 1")
    if not 0 <= cfg["failure_threshold"] <= 1 or not math.isfinite(cfg["threshold"]):
        raise ValueError("Invalid failure or mask threshold")
    if cfg["min_valid_pixels"] > 256 or cfg["random_seed"] < 0 or cfg["num_workers"] < 0:
        raise ValueError("Invalid minimum valid pixels, random seed, or worker count")
    if not cfg["phases"] or len(set(cfg["phases"])) != len(cfg["phases"]) or any(
            not isinstance(phase, int) or not 0 <= phase < 448 for phase in cfg["phases"]):
        raise ValueError("Phases must be unique integer window offsets in [0, 448)")
    if cfg["feature_format"] == "n3" and (not set(cfg["phases"]) <= {0, 8} or cfg["split"] != "test"):
        raise ValueError("Historical N3 caches support only Test and phases 0 and 8")
    if cfg["feature_source"] == "images" and cfg["feature_format"] != "f_raw":
        raise ValueError("Original-image extraction uses raw UNI block features. Historical N3 requires a cache.")
    for key in ("refine_query", "exclude_same_position", "save_margins", "save_masks"):
        if not isinstance(cfg[key], bool):
            raise ValueError(f"{key} must be a Boolean")
    tag = Path(cfg["tag"])
    if not cfg["tag"] or tag.name != cfg["tag"] or tag.name in (".", ".."):
        raise ValueError("tag must be a single directory name")


def _sam_config(settings, override, path, branches):
    if not isinstance(settings, dict) or settings.keys() - SAM_DEFAULTS.keys():
        raise ValueError("sam must contain only supported settings")
    sam = {**SAM_DEFAULTS, **settings}
    if override is not None:
        sam["enabled"] = override
    if not isinstance(sam["enabled"], bool):
        raise ValueError("sam.enabled must be true or false")
    if sam["model_type"] not in ("vit_b", "vit_l", "vit_h"):
        raise ValueError("sam.model_type must be vit_b, vit_l, or vit_h")
    tau = sam["iou_threshold"]
    if not isinstance(tau, (int, float)) or not 0 <= tau <= 1:
        raise ValueError("sam.iou_threshold must be between 0 and 1")
    for name in ("opening_radius", "negative_points", "negative_radius", "negative_spacing", "max_hole_area", "min_area"):
        if type(sam[name]) is not int or sam[name] < 0:
            raise ValueError(f"sam.{name} must be a nonnegative integer")
    if sam["base_branch"] == "auto":
        sam["base_branch"] = "query_refined" if "query_refined" in branches else "support_only"
    if sam["enabled"] and sam["base_branch"] not in branches:
        raise ValueError("sam.base_branch must be auto or an evaluated prediction branch")
    if sam["checkpoint"] is not None:
        checkpoint = Path(sam["checkpoint"]).expanduser()
        sam["checkpoint"] = str((path.resolve().parent / checkpoint).resolve())
    return sam


def add_selfref_arguments(parser):
    for key, default in DEFAULTS.items():
        options = {"help": HELP.get(key)}
        if key in CHOICES:
            options["choices"] = CHOICES[key]
        if key in ("dataset", "branches", "support_ids", "query_ids", "phases"):
            options["nargs"] = "+"
        if key == "branches":
            options["type"] = branch_name
        elif key in PATHS:
            options["type"] = Path
        elif key == "layer":
            options["type"] = _layer
        elif isinstance(default, bool):
            options.update(type=_boolean, nargs="?", const=True)
        elif isinstance(default, (int, float)):
            options["type"] = type(default)
        elif key == "phases":
            options["type"] = int
        flags = ["--" + key.replace("_", "-")]
        if key == "encoder_weights":
            flags.extend(("--uni-weights", "--weights"))
        parser.add_argument(*flags, **options)
    parser.add_argument("--selfref", dest="refine_query", type=_legacy_refinement, help=argparse.SUPPRESS)
    parser.add_argument("--no-exclude-same-position", action="store_false", dest="exclude_same_position", default=None)
    parser.add_argument("--sam", action=argparse.BooleanOptionalAction,
                        help="Enable the optional SAM refinement plugin. --no-sam overrides an enabled YAML setting.")


def resolve_selfref_config(args):
    from .config import read_config

    cli = {key: getattr(args, key) for key in DEFAULTS}
    config = args.config
    document = read_config(config) if config else {}
    supplied = document.get("one_shot", {key: value for key, value in document.items() if key != "sam"})
    if not isinstance(supplied, dict) or set(supplied) - (DEFAULTS.keys() | {"selfref"}):
        raise ValueError(f"Expected known one-shot configuration keys in {config}")
    supplied = supplied.copy()
    if "encoder" in document:
        supplied.setdefault("encoder_weights", document["encoder"].get("weights"))
    legacy_refinement = supplied.pop("selfref", None)
    if legacy_refinement is not None and "refine_query" not in supplied:
        try:
            supplied["refine_query"] = _legacy_refinement(legacy_refinement)
        except argparse.ArgumentTypeError as error:
            raise ValueError(str(error)) from None
    cfg = {**DEFAULTS, **supplied, **{k: v for k, v in cli.items() if v is not None}}
    branches = cfg["branches"]
    if branches is None or (cli["refine_query"] is not None and cli["branches"] is None):
        branches = BRANCHES if cfg["refine_query"] else BRANCHES[:1]
    if isinstance(branches, str):
        branches = [branches]
    if not isinstance(branches, (list, tuple)) or not all(isinstance(name, str) for name in branches):
        raise ValueError("branches must contain prediction names")
    cfg["branches"] = [branch_name(name) for name in branches]
    if not cfg["refine_query"] and "query_refined" in cfg["branches"]:
        raise ValueError("query_refined requires refine_query: true")
    for key in PATHS:
        value = cfg[key]
        if value is not None:
            base = config.resolve().parent if key in supplied and cli[key] is None else Path.cwd()
            cfg[key] = str((base / Path(value).expanduser()).resolve())
    cfg["config"] = str(config.resolve()) if config else None
    cfg["layer"] = _layer(cfg["layer"])
    if not 1 <= cfg["layer"] <= 24:
        raise ValueError("UNI layer must be in [1, 24]")
    _validate(cfg)
    cfg["sam"] = _sam_config(document.get("sam", {}), args.sam, config, cfg["branches"])
    return cfg


def load_selfref_config(argv=None, *, default_config=None):
    parser = argparse.ArgumentParser(description="Evaluate one-shot Support sensitivity on original GLaS and CRAG images.")
    parser.add_argument("--config", type=Path, default=default_config, help="One-shot evaluation YAML preset.")
    add_selfref_arguments(parser)
    return resolve_selfref_config(parser.parse_args(argv))
