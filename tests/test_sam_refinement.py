import csv
from types import SimpleNamespace

import numpy as np

from pathology_trainfree.core.sam_refinement import cleanup_mask, refine_instances, split_instances
from pathology_trainfree.selfref_config import DEFAULT_CONFIG, SAM_DEFAULTS, load_selfref_config


def test_opening_splits_thin_bridges_without_discarding_source_pixels():
    base = np.zeros((11, 19), dtype=bool)
    base[2:9, 2:7] = True
    base[2:9, 12:17] = True
    base[5, 7:12] = True
    labels, count = split_instances(base, radius=1)
    assert count == 2
    np.testing.assert_array_equal(labels > 0, base)
    assert labels[5, 4] != labels[5, 14]


def test_sam_prompt_and_consistency_gate_keep_or_replace_the_candidate():
    base = np.zeros((17, 17), dtype=bool)
    base[5:12, 5:12] = True
    margin = np.where(base, 1.0, -0.1)
    margin[8, 3] = -2
    margin[8, 2] = -1.9
    margin[8, 13] = -1.8
    proposed = base.copy()
    proposed[5, 5:7] = False
    proposed[5, 7] = False

    class Predictor:
        def predict(self, *, point_coords, point_labels, box, multimask_output):
            np.testing.assert_array_equal(box, [5, 5, 11, 11])
            np.testing.assert_array_equal(point_coords, [[8, 8], [3, 8], [13, 8]])
            np.testing.assert_array_equal(point_labels, [1, 0, 0])
            assert multimask_output
            return np.stack([np.ones_like(base), proposed]), np.array([0.99, 0.1]), None

    cfg = {**SAM_DEFAULTS, "negative_points": 2, "negative_radius": 6, "negative_spacing": 3}
    accepted, stats = refine_instances(Predictor(), margin, base, 0.0, cfg)
    np.testing.assert_array_equal(accepted, proposed)
    assert stats == {"sam_attempted": 1, "sam_accepted": 1}

    rejected, stats = refine_instances(Predictor(), margin, base, 0.0, {**cfg, "iou_threshold": 1.0})
    np.testing.assert_array_equal(rejected, base)
    assert stats == {"sam_attempted": 1, "sam_accepted": 0}


def test_cleanup_removes_small_objects_and_preserves_large_holes():
    mask = np.zeros((15, 15), dtype=bool)
    mask[2:12, 2:12] = True
    mask[4, 4] = False
    mask[7:9, 7:9] = False
    mask[13, 13] = True
    cleaned = cleanup_mask(mask, max_hole_area=1, min_area=2)
    assert cleaned[4, 4] and not cleaned[13, 13]
    assert not cleaned[7:9, 7:9].any()


def test_sam_yaml_defaults_and_cli_override():
    cfg = load_selfref_config([], default_config=DEFAULT_CONFIG)
    assert not cfg["sam"]["enabled"]
    cfg = load_selfref_config(["--sam", "--refine-query", "false"], default_config=DEFAULT_CONFIG)
    assert cfg["sam"]["enabled"]
    assert cfg["sam"]["base_branch"] == "support_only"


def test_no_sam_overrides_enabled_yaml(tmp_path):
    config = tmp_path / "sam.yaml"
    config.write_text("one_shot: {}\nsam:\n  enabled: true\n")
    cfg = load_selfref_config(["--config", str(config)])
    assert cfg["sam"]["enabled"]
    cfg = load_selfref_config(["--config", str(config), "--no-sam"])
    assert not cfg["sam"]["enabled"]


def test_evaluator_runs_sam_only_when_requested(tmp_path, monkeypatch):
    from pathology_trainfree import selfref_evaluate as evaluator
    from pathology_trainfree.plugins import sam

    query = {"image_id": "query", "height": 448, "width": 448}
    groups = [(query, [{"image_id": "support"}])]
    vectors = np.ones((784, 2), dtype=np.float32)
    margin = np.full((28, 28), -1.0, dtype=np.float32)
    margin[10:18, 10:18] = 1
    calls = []

    class Predictor:
        def set_image(self, image):
            calls.append("encode")
            assert image.shape == (448, 448, 3)

        def predict(self, **prompt):
            calls.append("predict")
            mask = np.zeros((448, 448), dtype=bool)
            mask[160:288, 160:288] = True
            return mask[None], [1.0], None

    def load_predictor(*args):
        calls.append("load")
        return Predictor()

    monkeypatch.setattr(sam, "load_sam_predictor", load_predictor)
    monkeypatch.setattr(sam, "read_original_image", lambda row: np.zeros((448, 448, 3), dtype=np.uint8))
    monkeypatch.setattr(evaluator, "selected_pairs", lambda cfg, dataset: (groups, {}))
    monkeypatch.setattr(evaluator, "load_features", lambda *args, **kwargs: ([vectors], [np.array([[0, 0]])]))
    monkeypatch.setattr(evaluator, "_build_banks", lambda *args: {"support": SimpleNamespace(sizes=(1, 1))})
    monkeypatch.setattr(evaluator, "predict_margins", lambda *args: {"support_only": margin.ravel()})
    metrics = dict(dice=1.0, miou=1.0, object_f1=1.0, object_dice=1.0, object_hd=0.0, missed_glands=0)
    monkeypatch.setattr(evaluator, "evaluate_mask", lambda *args: metrics)

    args = ["--dataset", "GlaS", "--feature-source", "cache", "--feature-cache", str(tmp_path),
            "--output-dir", str(tmp_path), "--device", "cpu", "--num-workers", "0", "--refine-query", "false"]
    evaluator.run_selfref(load_selfref_config(args + ["--tag", "default"], default_config=DEFAULT_CONFIG))
    assert not calls
    assert not (tmp_path / "default/sam_comparison.csv").exists()
    with (tmp_path / "default/per_pair.csv").open(newline="") as stream:
        baseline = list(csv.DictReader(stream))
    assert [row["branch"] for row in baseline] == ["support_only"]
    assert "sam_attempted" not in baseline[0]

    evaluator.run_selfref(load_selfref_config(args + ["--tag", "sam", "--sam"], default_config=DEFAULT_CONFIG))
    assert calls == ["load", "encode", "predict"]
    with (tmp_path / "sam/per_pair.csv").open(newline="") as stream:
        enabled = list(csv.DictReader(stream))
    assert [row["branch"] for row in enabled] == ["support_only", "support_only_cleanup", "support_only_sam"]
    assert enabled[-1]["sam_accepted"] == "1"
    assert (tmp_path / "sam/sam_comparison.csv").is_file()
