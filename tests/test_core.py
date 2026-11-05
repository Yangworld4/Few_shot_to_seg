import csv
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from pathology_trainfree.config import load_config
from pathology_trainfree.geometry import starts, windows, reconstruct, resize_map
from pathology_trainfree.memory import ReferenceMemory, select_tokens, build_memory
from pathology_trainfree.retrieval import SimilarityRetriever, classify
from pathology_trainfree.pipeline import SegmentationPipeline
from pathology_trainfree.evaluation import evaluate_predictions
from pathology_trainfree.data import read_manifest, load_mask
from pathology_trainfree.cli import parse_args, overrides


CONFIG = Path(__file__).resolve().parents[1] / "configs/uni_memory.yaml"


def test_starts_and_offsets():
    assert starts(512, 448, 224, 0) == [0, 224]
    assert starts(512, 448, 224, 8) == [-8, 216]
    assert starts(100, 448, 224, 8) == [-8]


def test_reflect_and_recovery():
    image = np.arange(6 * 9).reshape(6, 9).astype(np.float32)
    views = list(windows(image, 8, 4, 2))
    assert views[0][2][2:8, 2:8].shape == (6, 6)
    assert views[0][2][2, 2] == image[0, 0]
    assert views[0][2][1, 2] == image[1, 0]
    maps = [np.full((2, 2), 3, np.float32) for _ in views]
    assert np.array_equal(reconstruct(maps, [(y, x) for y, x, _ in views], image.shape, 8), np.full(image.shape, 3))


def test_bilinear_and_overlap():
    grid = np.arange(28 * 28, dtype=np.float32).reshape(28, 28) / 1000
    expected = F.interpolate(torch.from_numpy(grid)[None, None], size=(448, 448), mode="bilinear", align_corners=False)[0, 0].numpy()
    assert np.allclose(resize_map(grid, (448, 448)), expected)
    actual = reconstruct([np.ones((2, 2)), np.full((2, 2), 3)], [(0, 0), (0, 2)], (4, 6), 4)
    assert np.array_equal(actual, np.array([[1, 1, 2, 2, 3, 3]] * 4))


def test_selection_cap_seed_and_fallback():
    cfg = {"class_ids": [0, 1], "occupancy_thresholds": {0: .9, 1: .9},
           "cap_per_image_per_class": 5, "fallback_strategy": "nearest_10_percent"}
    features = np.arange(100, dtype=np.float32)[:, None]
    occupancy = np.stack([np.linspace(1, 0, 100), np.linspace(0, 1, 100)], axis=-1)
    a, _ = select_tokens(features, occupancy, cfg, np.random.default_rng(2026))
    b, _ = select_tokens(features, occupancy, cfg, np.random.default_rng(2026))
    assert all(len(a[c]) == 5 and np.array_equal(a[c], b[c]) for c in (0, 1))
    cfg["occupancy_thresholds"] = {0: 1, 1: 1}
    occupancy[:] = .5
    _, fallback = select_tokens(features, occupancy, cfg, np.random.default_rng(0))
    assert fallback == {0: True, 1: True}


def test_topk_and_classification():
    memory = ReferenceMemory({0: np.array([[1., 0], [0., 1.]]),
                              1: np.array([[-1., 0], [0., -1.]])}, {})
    cfg = load_config(CONFIG, {"retrieval.top_k": 1})
    retriever = SimilarityRetriever(memory, cfg, "cpu")
    scores = retriever.scores(np.array([[1., 0], [-1., 0]]))
    assert torch.allclose(scores, torch.tensor([[1., 0], [0, 1.]]))
    mask, margin = classify(scores, [0, 1])
    assert mask.tolist() == [0, 1] and margin.tolist() == [-1, 1]
    multi = torch.tensor([[.2, .8, .3]])
    assert classify(multi, [0, 2, 5])[0].tolist() == [2]
    with pytest.raises(ValueError, match="Top-K"):
        SimilarityRetriever(memory, load_config(CONFIG, {"retrieval.top_k": 3}), "cpu")
    with pytest.raises(ValueError, match="empty"):
        ReferenceMemory({0: np.zeros((0, 2)), 1: np.ones((1, 2))}, {})


def test_invalid_config():
    with pytest.raises(ValueError, match="multiple"):
        load_config(CONFIG, {"window.size": 447})
    with pytest.raises(ValueError, match="purity"):
        load_config(CONFIG, {"memory.occupancy_thresholds": {0: 0}})
    args = parse_args(["infer", "--config", str(CONFIG), "--output-size", "256", "--top-k", "20"])
    assert overrides(args)["segmentation.output_size"] == [256, 256]
    assert overrides(args)["retrieval.top_k"] == 20
    multiclass = load_config(CONFIG, {"memory.class_ids": [1, 2, 3],
                                      "memory.occupancy_thresholds": {1: .9, 2: .8, 3: .95},
                                      "dataset.class_mapping": {1: 1, 2: 2, 3: 3}})
    assert set(multiclass["memory"]["occupancy_thresholds"]) == {1, 2, 3}
    assert set(multiclass["dataset"]["class_mapping"]) == {1, 2, 3}


def test_rgb_mapping_and_image_only_manifest(tmp_path):
    rgb = np.zeros((3, 4, 3), np.uint8)
    rgb[:, 2:] = (255, 0, 0)
    Image.fromarray(rgb).save(tmp_path / "rgb.png")
    assert set(np.unique(load_mask(tmp_path / "rgb.png", {"mask_encoding": "rgb",
                                                         "class_mapping": {"0,0,0": 0, "255,0,0": 2}}))) == {0, 2}
    with pytest.raises(ValueError, match="unmapped"):
        load_mask(tmp_path / "rgb.png", {"mask_encoding": "rgb", "class_mapping": {"0,0,0": 0}})
    with (tmp_path / "query.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image"])
        writer.writerow(["rgb.png"])
    rows = read_manifest(tmp_path / "query.csv")
    assert rows[0]["mask"] is None and rows[0]["id"] == "rgb"


class FakeEncoder:
    patch_size = 16
    feature_dim = 2
    weights_sha256 = "fake"
    device = "cpu"
    precision = "float32"
    token_precision = "float16"
    normalize = True

    def encode(self, crops):
        # Deterministic dense RGB summaries, no model weights needed.
        output = []
        for crop in crops:
            gray = crop[..., 0].astype(np.float32)
            patches = gray.reshape(crop.shape[0] // 16, 16, crop.shape[1] // 16, 16).mean((1, 3)).reshape(-1)
            vectors = np.stack([patches / 255, 1 - patches / 255], axis=1)
            output.append(vectors.astype(np.float16))
        return np.stack(output)


def test_synthetic_end_to_end(tmp_path):
    cfg = load_config(CONFIG, {"window.size": 32, "window.stride": 16, "window.offsets": [0, 1],
                               "window.support_offsets": [0], "memory.cap_per_image_per_class": 20,
                               "retrieval.top_k": 1, "segmentation.output_size": [32, 48],
                               "dataset.class_mapping": {0: 0, 1: 1}})
    cfg["output"]["directory"] = str(tmp_path)
    image = np.zeros((32, 48, 3), np.uint8)
    image[:, :24] = 255
    mask = np.zeros((32, 48), np.uint8)
    mask[:, :24] = 1
    Image.fromarray(image).save(tmp_path / "image.png")
    Image.fromarray(mask).save(tmp_path / "mask.png")
    with (tmp_path / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "image", "mask"])
        writer.writeheader(); writer.writerow({"image_id": "a", "image": "image.png", "mask": "mask.png"})
    entries = read_manifest(tmp_path / "manifest.csv", require_mask=True)
    encoder = FakeEncoder()
    memory = build_memory(entries, encoder, cfg)
    memory.save(tmp_path / "memory.npz")
    loaded = ReferenceMemory.load(tmp_path / "memory.npz")
    result = SegmentationPipeline(encoder, loaded, cfg).predict(image)
    assert result["mask"].shape == (32, 48)
    assert result["native_scores"].shape == (32, 48)
    assert evaluate_predictions({"a": result["mask"]}, entries, cfg)["images"] == 1
    cfg["output"]["cache_features"] = True
    cached_pipeline = SegmentationPipeline(encoder, loaded, cfg)
    assert np.array_equal(cached_pipeline.predict(image, "a")["mask"], cached_pipeline.predict(image, "a")["mask"])
    assert len(list((tmp_path / "feature_cache").glob("*.npz"))) == 2
    altered = image.copy(); altered[0, 0] = 17
    with pytest.raises(ValueError, match="Cached query features differ"):
        cached_pipeline.predict(altered, "a")
    cfg["output"]["cache_features"] = False
    mismatched = load_config(CONFIG, {"window.size": 32, "window.stride": 16,
                                      "window.support_offsets": [0], "memory.cap_per_image_per_class": 21,
                                      "retrieval.top_k": 1, "dataset.class_mapping": {0: 0, 1: 1}})
    with pytest.raises(ValueError, match="memory.cap_per_image_per_class"):
        SegmentationPipeline(encoder, loaded, mismatched)
    cfg["segmentation"]["output_size"] = [256, 256]
    assert SegmentationPipeline(encoder, loaded, cfg).predict(image)["mask"].shape == (256, 256)


def test_multiclass_pipeline(tmp_path):
    cfg = load_config(CONFIG, {"window.size": 32, "window.stride": 16, "window.offsets": [0],
                               "window.support_offsets": [0], "memory.class_ids": [0, 1, 2],
                               "memory.occupancy_thresholds": {0: .9, 1: .9, 2: .9},
                               "retrieval.top_k": 1, "segmentation.output_size": [32, 48],
                               "dataset.class_mapping": {0: 0, 1: 1, 2: 2}})
    image = np.zeros((32, 48, 3), np.uint8)
    image[:, 16:32] = 128
    image[:, 32:] = 255
    mask = np.zeros((32, 48), np.uint8)
    mask[:, 16:32] = 1
    mask[:, 32:] = 2
    Image.fromarray(image).save(tmp_path / "image.png")
    Image.fromarray(mask).save(tmp_path / "mask.png")
    entry = {"id": "multi", "image": tmp_path / "image.png", "mask": tmp_path / "mask.png"}
    encoder = FakeEncoder()
    memory = build_memory([entry], encoder, cfg)
    prediction = SegmentationPipeline(encoder, memory, cfg).predict(image)["mask"]
    assert prediction.shape == mask.shape
    assert set(np.unique(prediction)).issubset({0, 1, 2})
    assert 2 in prediction
    assert evaluate_predictions({"multi": prediction}, [entry], cfg)["images"] == 1
