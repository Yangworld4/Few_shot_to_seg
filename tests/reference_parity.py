"""Run explicit numerical parity against local DUAL artifacts and Direct function.

This validation script is optional; the installed package has no DUAL dependency.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathology_trainfree.config import load_config
from pathology_trainfree.data import load_image, read_manifest
from pathology_trainfree.encoder import FeatureEncoder
from pathology_trainfree.geometry import reconstruct, resize_map, windows
from pathology_trainfree.memory import ReferenceMemory
from pathology_trainfree.pipeline import SegmentationPipeline
from pathology_trainfree.retrieval import SimilarityRetriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dual", type=Path, required=True)
    parser.add_argument("--dataset", choices=("GlaS", "CRAG"), required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--built-memory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(ROOT / "configs/uni_memory.yaml")
    cfg["encoder"]["weights"] = args.weights
    encoder = FeatureEncoder(cfg["encoder"])
    cache = args.dual / f"outputs/uni_audit_mask_quality_{args.dataset}_seed1337/cache"
    source = np.load(cache / "memory_N3.npz")
    source_memory = ReferenceMemory({0: source["bg"], 1: source["fg"]},
                                    {"encoder_sha256": encoder.weights_sha256, "support_ids": source["source_ids"].tolist()})
    built = ReferenceMemory.load(args.built_memory)
    report = {"dataset": args.dataset, "source_ids_match": built.metadata["support_ids"] == source["source_ids"].tolist(),
              "memory": {}}
    for cls, old_key in ((0, "bg"), (1, "fg")):
        diff = np.abs(built.vectors[cls].astype(np.float32) - source[old_key].astype(np.float32))
        report["memory"][str(cls)] = {"count": len(source[old_key]), "new_count": len(built.vectors[cls]),
                                       "max_abs": float(diff.max()), "mean_abs": float(diff.mean())}
    # Verify the sampling path independently from newly computed encoder values.
    rng = np.random.default_rng(2026)
    exact = True
    for index, image_id in enumerate(source["source_ids"]):
        cached = np.load(cache / "features_N3/labeled" / f"{image_id}.npz")
        feature = cached["features"].reshape(-1, 1024)
        occupancy = cached["occupancy"].reshape(-1)
        for key, candidate in (("fg", occupancy >= .9), ("bg", occupancy <= .1)):
            chosen = np.sort(rng.choice(np.flatnonzero(candidate), 512, replace=False))
            exact &= np.array_equal(feature[chosen], source[key][index * 512:(index + 1) * 512])
    report["cached_memory_selection_exact"] = bool(exact)
    # Token score parity on a cached validation image.
    validation = next((cache / "features_N3/validation").glob("*.npz"))
    cached = np.load(validation)
    score = SimilarityRetriever(source_memory, cfg, "cuda").scores(cached["features"]).numpy()
    token = (score[:, 1] - score[:, 0]).reshape(-1, 28, 28).astype(np.float16)
    old_token = np.load(cache / "token_margins_N3/validation" / validation.name)["margin"]
    report["cached_token_margin_max_abs"] = float(np.max(np.abs(token.astype(np.float32) - old_token.astype(np.float32))))
    report["cached_token_margin_exact"] = bool(np.array_equal(token, old_token))
    # Reconstruct the same cached token maps; compare native and 256 outputs.
    image_path = args.dual / "datasets" / args.dataset / "images/train" / f"{validation.stem}.png"
    image = load_image(image_path)
    offset_maps = []
    for passno in (0, 1):
        ix = np.flatnonzero(cached["passno"] == passno)
        offset_maps.append(reconstruct(old_token[ix], [(int(cached["y"][i]), int(cached["x"][i])) for i in ix],
                                       image.shape[:2], 448))
    native = (offset_maps[0] + offset_maps[1]) / 2
    old_native = np.load(cache / "margin_maps_N3_bil/validation" / f"{validation.stem}.npy")
    output = resize_map(native, (256, 256))
    old_output = resize_map(old_native, (256, 256))
    report["cached_native_fp16_exact"] = bool(np.array_equal(native.astype(np.float16), old_native))
    report["cached_native_max_abs"] = float(np.max(np.abs(native - old_native.astype(np.float32))))
    report["cached_256_max_abs"] = float(np.max(np.abs(output - old_output)))
    report["cached_256_mask_differences"] = int(np.count_nonzero((output > 0) != (old_output > 0)))
    # Compare the actual Direct function and new pipeline with identical weights/memory/image.
    spec = importlib.util.spec_from_file_location("dual_n3_direct", args.dual / "analysis/n3_bil_closed_experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    test = read_manifest(args.dual / "datasets" / args.dataset / "manifests/paper_seed_2026/test.csv")[0]
    test_image = load_image(test["image"])
    old_model, _, _ = module.load_uni(args.weights, str(Path(args.weights).with_name("config.json")))
    fg = F.normalize(torch.from_numpy(source["fg"].astype(np.float32)).cuda(), dim=1)
    bg = F.normalize(torch.from_numpy(source["bg"].astype(np.float32)).cuda(), dim=1)
    old_margin = module.infer(test_image, old_model, fg, bg)
    new = SegmentationPipeline(encoder, source_memory, cfg).predict(test_image)
    report["direct_image_id"] = test["id"]
    report["direct_native_max_abs"] = float(np.max(np.abs(old_margin - new["native_scores"])))
    report["direct_256_max_abs"] = float(np.max(np.abs(module.resize_margin(old_margin) - new["scores"])))
    report["direct_mask_differences"] = int(np.count_nonzero((module.resize_margin(old_margin) > 0) != new["mask"]))
    # Compare independent query token scores for the first window.
    first = next(windows(test_image, 448, 224, 0))[2]
    new_token = encoder.encode([first])[0]
    with torch.inference_mode():
        inp = torch.from_numpy(first.copy()).permute(2, 0, 1)[None].cuda().float() / 255
        means = torch.tensor((.485, .456, .406), device="cuda")[None, :, None, None]
        stds = torch.tensor((.229, .224, .225), device="cuda")[None, :, None, None]
        old_token_feature = F.normalize(old_model.dense_patch_tokens((inp-means)/stds)[0].half().float(), dim=1)
        old_fg = (old_token_feature @ fg.T).topk(50, dim=1).values.mean(1)
        old_bg = (old_token_feature @ bg.T).topk(50, dim=1).values.mean(1)
        new_scores = SimilarityRetriever(source_memory, cfg, "cuda").scores(new_token).numpy()
        report["direct_query_score_fg_max_abs"] = float(np.max(np.abs(old_fg.cpu().numpy() - new_scores[:, 1])))
        report["direct_query_score_bg_max_abs"] = float(np.max(np.abs(old_bg.cpu().numpy() - new_scores[:, 0])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
