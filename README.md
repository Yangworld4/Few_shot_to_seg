# PathologyTrainFree

Frozen UNI / UNI2-h / Kaiko ViT-B/8 feature extraction and class-wise reference retrieval for pathology semantic segmentation. The default YAML encodes the validated N3-bil Direct protocol; the package itself accepts generic CSV manifests and class mappings. There is no training step and no DUAL import at runtime.

## Research result correction (2026-11-04)

The earlier random Support-Size studies used zero placeholder occupancy from DUAL's unlabeled feature caches. Their low random-support scores are invalid as labeled-support experiments. The study cache loader now recomputes occupancy from the actual Support GT. Original historical-Memory parity remains valid. See [the diagnosis and paired occupancy-only controls](outputs/reference_coverage_diagnosis/report.md); earlier output directories are preserved as evidence.

The completed fixed-GT study is available in [support_size_study_gtfixed/report.md](outputs/support_size_study_gtfixed/report.md), with all 40 runs and exact corrected-20-shot regression checks. Reproduce with the command in that directory’s `execution_command.txt`.

## Setup

Use a Python environment with a CUDA-enabled PyTorch installation, then install this project:

```bash
pip install -e .
```

UNI weights are supplied as a local `pytorch_model.bin` through `--weights` or `encoder.weights`. The file must be the original `vit_large_patch16_224` UNI checkpoint. `encoder.config` is recorded but construction validates the architecture and tensor shape directly. For this machine, the validation environment was `/data/software/miniconda3/envs/cg-seg/bin/python` with PyTorch 2.0.1+cu118 and an RTX 4090.

## Commands

Run from this project directory. Set `UNI_WEIGHTS` to your local UNI checkpoint path.

```bash
# 1. Build labeled-only memory.
python -m pathology_trainfree build-memory --config configs/glas_example.yaml --weights "$UNI_WEIGHTS"

# 2. Direct inference; images-only query manifests also work.
python -m pathology_trainfree infer --config configs/glas_example.yaml --weights "$UNI_WEIGHTS"

# 3. Evaluate saved masks against the query manifest's GT.
python -m pathology_trainfree evaluate --config configs/glas_example.yaml

# 4. Build if needed, infer and evaluate in one command.
python -m pathology_trainfree run --config configs/crag_example.yaml --weights "$UNI_WEIGHTS"
```

Use `--memory-file`, `--support-manifest`, `--query-manifest`, and `--output-directory` to replace example paths. `run` reuses an existing Memory file only when its encoder, class, sampling, support-window, and label-mapping settings agree with the current config; use `build-memory` or a new path after changing them. `build-memory` always rebuilds. The CLI saves action-specific `resolved_config_*.yaml` and `run_info_*.json` files plus `run_history.jsonl`, Memory statistics, label PNGs, optional float32 score arrays, optional overlays, `per_image.csv`, and `metrics.json`.

Example result-preserving or exploratory overrides:

```bash
python -m pathology_trainfree infer --config configs/glas_example.yaml --weights "$UNI_WEIGHTS" \
  --top-k 20 --window-size 448 --stride 224 --offsets 0,8 --output-size native \
  --save-margin true --set retrieval.chunk_size=512

python -m pathology_trainfree build-memory --config configs/glas_example.yaml --weights "$UNI_WEIGHTS" \
  --memory-cap 256 --fg-threshold 0.95 --bg-threshold 0.05 \
  --set 'window.support_offsets=[0]'
```

Every YAML leaf can be overridden with `--set section.key=YAML_VALUE`. CLI aliases cover common changes. Top-K must not exceed any class Memory size; invalid window, empty class, missing mapping, and unrecognized labels cause explicit errors.

`encoder.precision` controls UNI computation (`float32` by default); `encoder.token_precision` controls stored token precision (`float16` by default to match N3). `output.cache_features: true` caches query tokens with an image/encoder/window fingerprint. The binary GLaS/CRAG examples enable the optional 256-pixel connected-component Object-F1 and Object-Hausdorff protocol through `evaluation.object_metrics`; the generic default leaves it disabled.

## Default protocol

| Parameter | Value |
|---|---|
| UNI | Frozen ViT-L/16, RGB ImageNet mean/std, 1024-d tokens |
| Window | 448, stride 224, reflect padding, 28×28 token grid |
| Query offsets | 0, 8 |
| Support offsets | 0, 8 in the verified Memory cache |
| Memory | FG occupancy ≥0.9, BG occupancy ≤0.1, 512 per image/class, seed 2026 |
| Retrieval | Independent cosine Top-50 per class, mean scores |
| Reconstruction | fp16 token margin, bilinear to 448, mean window and offset fusion |
| Output | Bilinear native margin to 256×256, threshold >0 |

The task's parameter table states support offset 0, but DUAL's actual `memory_N3.npz` was made from **both 0 and 8**. This was verified by exact reconstruction of every stored FG/BG vector from the corresponding `features_N3/labeled/*.npz` caches using one seed-2026 RNG, 512 samples per image/class, FG then BG. The default follows those verified artifacts. Set `window.support_offsets` to `[0]` for the requested single-offset variant; it will yield a different Memory.

In binary mode the class IDs are `[0,1]`. The pipeline computes continuous `score_1 - score_0` and thresholds only after spatial reconstruction. For more classes, it reconstructs every class score and takes the highest class score. Window and output interpolation use bilinear with `align_corners=False`.

## Dataset contract

A CSV row has `image` and optionally `mask`, `image_id`, and `instance_mask`. `image_path` and `mask_path` aliases are accepted. Relative paths resolve from the manifest directory, or from `dataset.root` if set. Support and evaluation rows need masks; inference rows need only images. Image IDs default to filename stems and must be unique.

`dataset.mask_encoding: indexed` maps exact grayscale or palette indices through `dataset.class_mapping`. The binary examples map 0→0 and 255→1. RGB masks require `mask_encoding: rgb` and explicit quoted color keys such as `"255,0,0": 1`; unknown labels raise errors. Multi-class examples set `memory.class_ids`, per-class purity thresholds, and all mapping values. The current output is semantic classes, not instances.

## Validation and parity

Run unit and synthetic integration tests:

```bash
python -m pytest -q tests
```

Where DUAL source and caches are available, run the optional reference check:

```bash
python tests/reference_parity.py --dual ../DUAL --dataset GlaS --weights "$UNI_WEIGHTS" \
  --built-memory outputs/glas10/memory.npz --output outputs/glas10/parity.json
```

This checks Memory counts and values, exact cached sampling indices, token margins, native and 256 margins, final masks, and one same-input Direct function comparison. See [VALIDATION.md](VALIDATION.md) for actual results and remaining numerical tolerance.

## Modules

`encoder.py` owns UNI construction and frozen dense tokens; `geometry.py` owns windows and reconstruction; `data.py` owns manifests and exact label mapping; `memory.py` owns selection and persistence; `retrieval.py` owns chunked cosine Top-K; `pipeline.py` owns inference and output; `evaluation.py` owns semantic and optional binary object metrics; `cli.py` owns commands and provenance.

## Fixed-support multi-foundation-model diagnosis

`encoder.name` accepts `uni`, `uni2_h`, and `kaiko_vitb8`. All share the same N3-bil segmentation pipeline; model-specific preprocessing and spatial-prefix removal live in `encoder.py`. Local pretrained weights are strictly loaded. See `configs/uni2_h_memory.yaml` and `configs/kaiko_vitb8_memory.yaml` for standalone adapter configurations.

The closed 20-shot × 5-seed experiment reuses the corrected Support-Size Study manifests, rebuilds real-GT occupancy at each encoder's patch size, validates UNI regression, and saves per-image masks/margins and all diagnostics under `outputs/multi_fm_complementarity/`. From this directory:

```bash
python scripts/smoke_encoders.py
python scripts/check_encoder_batching.py
python -m pytest -q tests
python run_complementarity.py
python analyze_complementarity.py
python visualize_complementarity.py
python verify_complementarity.py
python report_complementarity.py
```

Use `run_complementarity.py --models uni2_h kaiko_vitb8` to resume the new encoders once all ten UNI regression checks exist. Cache identities include preprocessing, checkpoint, patch geometry, image identity, offsets and algorithm version. Support GT occupancy is always recomputed from current labels. No fallback to mixed tokens or reduction of Top-K is allowed in this experiment. Oracle results use test GT and are exclusively retrospective analysis.

## Self-reference seed diagnosis

The separate [RGB-only A/B/C seed diagnosis](pathology_trainfree/self_reference/README.md) runs E0 pixel reliability, E1 native Token mapping and E2 fixed-Top50 self-reference with spatial exclusion. Its configuration is `configs/self_reference_e0.yaml`; all empty/insufficient memories remain in the report. It does not change the labeled-support pipeline.

