import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
import yaml
from .config import load_config, resolved_path
from .data import read_manifest, load_image
from .encoder import FeatureEncoder, sha256
from .memory import ReferenceMemory, build_memory
from .pipeline import SegmentationPipeline, save_prediction
from .evaluation import evaluate_predictions


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m pathology_trainfree")
    p.add_argument("action", choices=("build-memory", "infer", "evaluate", "run"))
    p.add_argument("--config", required=True)
    p.add_argument("--support-manifest")
    p.add_argument("--query-manifest")
    p.add_argument("--memory-file")
    p.add_argument("--output-directory")
    p.add_argument("--weights")
    p.add_argument("--top-k", type=int)
    p.add_argument("--window-size", type=int)
    p.add_argument("--stride", type=int)
    p.add_argument("--offsets")
    p.add_argument("--memory-cap", type=int)
    p.add_argument("--fg-threshold", type=float)
    p.add_argument("--bg-threshold", type=float)
    p.add_argument("--output-size")
    p.add_argument("--save-mask", choices=("true", "false"))
    p.add_argument("--save-margin", choices=("true", "false"))
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=YAML_VALUE")
    return p.parse_args(argv)


def overrides(args):
    result = {}
    aliases = {"weights": "encoder.weights", "support_manifest": "dataset.support_manifest",
               "query_manifest": "dataset.query_manifest", "memory_file": "memory.file",
               "output_directory": "output.directory", "top_k": "retrieval.top_k",
               "window_size": "window.size", "stride": "window.stride",
               "memory_cap": "memory.cap_per_image_per_class"}
    for arg, key in aliases.items():
        value = getattr(args, arg)
        if value is not None:
            result[key] = value
    if args.offsets is not None:
        result["window.offsets"] = [int(v) for v in args.offsets.split(",")]
    if args.output_size:
        if args.output_size == "native":
            size = "native"
        else:
            parts = [int(v) for v in args.output_size.split(",")]
            size = parts * 2 if len(parts) == 1 else parts
        result["segmentation.output_size"] = size
        result["segmentation.native_resolution"] = args.output_size == "native"
    for arg, key in (("save_mask", "output.save_mask"), ("save_margin", "output.save_margin")):
        if getattr(args, arg) is not None:
            result[key] = getattr(args, arg) == "true"
    for arg, cls in (("fg_threshold", 1), ("bg_threshold", 0)):
        if getattr(args, arg) is not None:
            # class purity is FG occupancy >= t, BG occupancy <= t.
            current = result.get("memory.occupancy_thresholds")
            if current is None:
                current = {}
            current[cls] = getattr(args, arg) if cls == 1 else 1 - getattr(args, arg)
            result["memory.occupancy_thresholds"] = current
    for setting in args.set:
        key, sep, value = setting.partition("=")
        if not sep:
            raise ValueError("--set requires SECTION.KEY=YAML_VALUE")
        result[key] = yaml.safe_load(value)
    return result


def prepare_config(args):
    cfg = load_config(args.config, overrides(args))
    for section, field in (("encoder", "weights"), ("encoder", "config"),
                           ("dataset", "root"), ("dataset", "support_manifest"),
                           ("dataset", "query_manifest"), ("memory", "file"), ("output", "directory")):
        value = cfg[section][field]
        if value is not None and not Path(value).is_absolute():
            # Explicit command-line paths are relative to CWD; YAML paths to config.
            cli_keys = {"encoder.weights": args.weights, "dataset.support_manifest": args.support_manifest,
                        "dataset.query_manifest": args.query_manifest, "memory.file": args.memory_file,
                        "output.directory": args.output_directory}
            key = f"{section}.{field}"
            cfg[section][field] = str(Path(value).resolve() if cli_keys.get(key) else resolved_path(value, args.config))
    return cfg


def write_run_info(out, action, info):
    payload = json.dumps(info, indent=2)
    (out / "run_info.json").write_text(payload)
    (out / f"run_info_{action}.json").write_text(payload)
    with (out / "run_history.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"action": action, **info}) + "\n")


def main(argv=None):
    args = parse_args(argv)
    cfg = prepare_config(args)
    out = Path(cfg["output"]["directory"])
    out.mkdir(parents=True, exist_ok=True)
    resolved_yaml = yaml.safe_dump(cfg, sort_keys=False)
    (out / "resolved_config.yaml").write_text(resolved_yaml)
    (out / f"resolved_config_{args.action}.yaml").write_text(resolved_yaml)
    process_argv = [sys.executable, *sys.argv] if argv is None else ["python", "-m", "pathology_trainfree", *argv]
    command = " ".join(shlex.quote(v) for v in process_argv)
    try:
        git_commit = subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_commit = None
    run_info = {"command": command, "git_commit": git_commit, "torch": torch.__version__,
                "cuda": torch.version.cuda, "seed": cfg["memory"]["sampling_seed"]}
    encoder = None
    memory_path = Path(cfg["memory"]["file"])
    if args.action in ("build-memory", "run") and (args.action == "build-memory" or not memory_path.exists()):
        encoder = FeatureEncoder(cfg["encoder"])
        support = read_manifest(cfg["dataset"]["support_manifest"], cfg["dataset"]["root"], require_mask=True)
        memory = build_memory(support, encoder, cfg)
        memory.save(memory_path)
        print(json.dumps({"memory": str(memory_path), "counts": memory.metadata["counts"]}))
    if args.action == "build-memory":
        write_run_info(out, args.action, {**run_info, "encoder_sha256": encoder.weights_sha256,
                                          "memory_sha256": sha256(memory_path),
                                          "support_ids": memory.metadata["support_ids"]})
        return
    if args.action in ("infer", "run"):
        if encoder is None:
            encoder = FeatureEncoder(cfg["encoder"])
        memory = ReferenceMemory.load(memory_path)
        pipeline = SegmentationPipeline(encoder, memory, cfg)
        entries = read_manifest(cfg["dataset"]["query_manifest"], cfg["dataset"]["root"])
        support_ids = set(memory.metadata.get("support_ids", []))
        if args.action == "run" and support_ids.intersection(e["id"] for e in entries):
            raise ValueError("Query IDs overlap support IDs in end-to-end run")
        predictions = {}
        for entry in entries:
            image = load_image(entry["image"])
            result = pipeline.predict(image, entry["id"])
            save_prediction(result, image, entry["id"], cfg)
            predictions[entry["id"]] = result["mask"]
            print(f"inferred {entry['id']} {result['mask'].shape}", flush=True)
        if args.action == "run" and all(e["mask"] for e in entries):
            result = evaluate_predictions(predictions, entries, cfg, out if cfg["output"]["save_metrics"] else None)
            print(json.dumps(result))
        run_info.update({"encoder_sha256": encoder.weights_sha256, "memory": str(memory_path),
                         "memory_sha256": sha256(memory_path),
                         "support_ids": memory.metadata.get("support_ids"),
                         "query_ids": [e["id"] for e in entries]})
    else:
        entries = read_manifest(cfg["dataset"]["query_manifest"], cfg["dataset"]["root"], require_mask=True)
        predictions = {}
        for entry in entries:
            path = out / f"{entry['id']}_mask.png"
            npy = out / f"{entry['id']}_mask.npy"
            if path.exists():
                from PIL import Image
                predictions[entry["id"]] = np.asarray(Image.open(path))
            elif npy.exists():
                predictions[entry["id"]] = np.load(npy)
            else:
                raise FileNotFoundError(f"Missing prediction for {entry['id']}")
        print(json.dumps(evaluate_predictions(predictions, entries, cfg, out if cfg["output"]["save_metrics"] else None)))
        run_info["query_ids"] = [e["id"] for e in entries]
    write_run_info(out, args.action, run_info)


if __name__ == "__main__":
    main()
