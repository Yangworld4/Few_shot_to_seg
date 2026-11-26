from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage
import torch
import torch.nn.functional as F

from pathology_trainfree.core.geometry import reconstruct, resize_map, starts, windows
from pathology_trainfree.core.selfref import (
    CachedQuery, SupportBank, class_budgets, kcenter_order, predict_margins, select_seeds,
)
from pathology_trainfree.selfref_config import load_selfref_config
from pathology_trainfree.selfref_data import hash_file, load_image_features, selected_pairs
from pathology_trainfree.utils.selfref_metrics import QueryGround, summarize_supports
from scripts.evaluate import main as evaluate_main
from scripts.compare_selfref import key as comparison_key


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
    result = reconstruct(maps, [(y, x) for y, x, _ in views], image.shape, 8)
    np.testing.assert_array_equal(result, np.full(image.shape, 3))


def test_bilinear_and_overlap():
    grid = np.arange(28 * 28, dtype=np.float32).reshape(28, 28) / 1000
    expected = F.interpolate(torch.from_numpy(grid)[None, None], size=(448, 448),
                             mode="bilinear", align_corners=False)[0, 0].numpy()
    np.testing.assert_allclose(resize_map(grid, (448, 448)), expected)
    actual = reconstruct([np.ones((2, 2)), np.full((2, 2), 3)], [(0, 0), (0, 2)], (4, 6), 4)
    np.testing.assert_array_equal(actual, np.array([[1, 1, 2, 2, 3, 3]] * 4))


def test_budget_is_shared_then_clipped_per_class():
    assert class_budgets((2, 21), "ratio", .25, 2048) == (2, 3)
    assert class_budgets((2, 21), "fixed", .25, 4) == (2, 4)


def test_kcenter_is_deterministic_and_budget_is_a_prefix():
    features = torch.nn.functional.normalize(torch.randn(16, 8, generator=torch.Generator().manual_seed(7)))
    small, _ = class_budgets((16, 16), "ratio", .25, 2048)
    large, _ = class_budgets((16, 16), "ratio", .5, 2048)
    prefix = kcenter_order(features, small)
    np.testing.assert_array_equal(prefix, kcenter_order(features, small))
    np.testing.assert_array_equal(prefix, kcenter_order(features, large)[:small])


def test_configuration_precedence(tmp_path):
    preset = tmp_path / "method.yaml"
    preset.write_text("rho: 0.25\nsupport_topk: 7\nexclude_same_position: false\n")
    settings = load_selfref_config(["--config", str(preset), "--rho", "0.5", "--exclude-same-position", "true"])
    assert settings["rho"] == .5
    assert settings["support_topk"] == 7
    assert settings["selfref_topk"] == 50
    assert settings["exclude_same_position"] is True


def test_evaluate_dispatches_ablation_flags_to_the_one_shot_workflow(tmp_path, monkeypatch):
    preset = tmp_path / "method.yaml"
    preset.write_text("selfref: none\nbranches: [e0]\nbank_select: kcenter\n")
    runs = []
    monkeypatch.setattr("pathology_trainfree.selfref_evaluate.run_selfref", runs.append)
    args = ["--config", str(preset), "--feature-cache", str(tmp_path)]
    evaluate_main(args + ["--bank-select", "random", "--selfref", "e1a", "--seed-ratio", "0.4"])
    evaluate_main(args + ["--refine-query", "false"])
    with pytest.raises(SystemExit, match="query_refined requires"):
        evaluate_main(args + ["--branches", "query_refined"])

    random, kcenter = runs
    assert random["bank_select"] == "random"
    assert random["branches"] == ["support_only", "query_refined"]
    assert random["seed_ratio"] == 0.4
    assert random["feature_cache"] == str(tmp_path)
    assert kcenter["bank_select"] == "kcenter"
    assert kcenter["branches"] == ["support_only"]


def test_one_shot_preset_selects_original_images_and_inherits_shared_parameters(monkeypatch):
    configs = Path(__file__).resolve().parents[1] / "configs"
    runs = []
    monkeypatch.setattr("pathology_trainfree.selfref_evaluate.run_selfref", runs.append)
    args = ["--config", str(configs / "eval_query_refinement.yaml")]
    evaluate_main(args)
    evaluate_main(args + ["--bank-select", "random", "--rho", "0.25"])

    kcenter, random = runs
    assert kcenter["bank_select"] == "kcenter"
    assert random["bank_select"] == "random"
    assert kcenter["layer"] == random["layer"] == 19
    assert kcenter["feature_source"] == "images"
    assert kcenter["encoder_weights"].endswith("/checkpoints/uni/pytorch_model.bin")
    assert kcenter["feature_cache"] == random["feature_cache"]
    assert kcenter["rho"] == 0.125
    assert random["rho"] == 0.25
    assert kcenter["branches"] == random["branches"] == ["support_only", "query_refined"]


def test_disabling_refinement_only_returns_the_support_margin():
    cfg = load_selfref_config(["--device", "cpu", "--refine-query", "false"])
    vectors = np.array([[1., 0.], [0., 1.]], dtype=np.float16)
    query = CachedQuery([vectors], [], (448, 448), cfg)
    bank = SupportBank(torch.from_numpy(vectors), (torch.tensor([0]), torch.tensor([1])))
    margins = predict_margins(query, bank, cfg)
    assert list(margins) == ["support_only"]
    np.testing.assert_array_equal(margins["support_only"], np.array([-1., 1.]))


def test_comparison_reads_archived_branch_labels():
    pair = {"dataset": "GlaS", "support_id": "train_1", "query_id": "test_a_testA_1"}
    assert comparison_key({**pair, "branch": "E0"}) == comparison_key({**pair, "branch": "support_only"})
    assert comparison_key({**pair, "branch": "E1a"}) == comparison_key({**pair, "branch": "query_refined"})


def test_seeds_use_both_strict_signs_and_stable_ties():
    bg, fg = select_seeds(np.array([-2., -2., -1., 0., 1., 2., 2.]), .3)
    np.testing.assert_array_equal(bg, [0])
    np.testing.assert_array_equal(fg, [5])


def test_oof_uses_the_frozen_manifest_order_and_wraps():
    cfg = load_selfref_config(["--dataset", "GlaS", "--split", "train-oof"])
    groups, _ = selected_pairs(cfg, "GlaS")
    assert len(groups) == 85
    assert groups[0][0]["image_id"] == "train_1"
    assert groups[0][1][0]["image_id"] == "train_10"
    assert groups[-1][1][0]["image_id"] == "train_1"


def test_original_image_features_preserve_windows_and_normalize_query(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (493, 451), (123, 12, 1)).save(path)
    row = dict(image=path, image_sha256=hash_file(path), height=451, width=493, split="Test")
    cfg = load_selfref_config(["--device", "cpu"])

    class Encoder:
        def encode(self, crops):
            crops = list(crops)
            assert all(crop.shape == (448, 448, 3) and crop[0, 0, 0] == 123 for crop in crops)
            values = np.zeros((len(crops), 784, 1024), dtype=np.float16)
            values[..., 0] = 2
            return values

    support, positions = load_image_features(Encoder(), cfg, "GlaS", row)
    query, query_positions = load_image_features(Encoder(), cfg, "GlaS", row, query=True)
    assert [position[0].tolist() for position in positions] == [[0, 0], [-8, -8]]
    for raw, norm, support_pos, query_pos in zip(support, query, positions, query_positions):
        np.testing.assert_array_equal(support_pos, query_pos)
        assert raw.shape == norm.shape == (4 * 784, 1024)
        assert raw.dtype == np.float32 and norm.dtype == np.float16
        assert np.all(raw[:, 0] == 2) and np.all(norm[:, 0] == 1)


def test_support_sensitivity_summary_groups_by_support():
    common = dict(dataset="GlaS", split="test", branch="support_only")
    rows = [{**common, "support_id": sid, "dice": dice, "miou": 0.5}
            for sid, dice in [("train_1", 0.5), ("train_1", 1.0), ("train_2", 0.25)]]
    support = summarize_supports(rows)
    assert [(r["support_id"], r["queries"], r["mean_dice"]) for r in support] == [
        ("train_1", 2, 0.75), ("train_2", 1, 0.25),
    ]


def test_object_hd_ground_preserves_background_and_native_instance_ids():
    ground = QueryGround(np.array([[0, 0, 5], [0, 2, 5]]), ndimage.distance_transform_edt)
    np.testing.assert_array_equal(ground.points(0), [[0, 0], [0, 1], [1, 0]])
    np.testing.assert_array_equal(ground.points(1), [[1, 1]])
    np.testing.assert_array_equal(ground.points(2), [[0, 2], [1, 2]])
