# Validation record

Environment: RTX 4090; `/data/software/miniconda3/envs/cg-seg/bin/python`; PyTorch 2.0.1+cu118, CUDA 11.8, timm 0.9.16, NumPy 1.26.4. UNI SHA256: `56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40`. DUAL source commit: `4e7782b9c1df2a1bc17703a1cc1dbb1e3f443e88`; its N3 Direct script and audit cache files are untracked working-tree artifacts. The requested `analysis/native_magnification_audit.py` is absent from the available DUAL tree, so the audited `features_N3`, `token_margins_N3`, and `margin_maps_N3_bil` caches were used for cross checks.

## Tests

`python -m pytest -q tests`: **9 passed**. The tests cover window starts and offsets, reflection padding, coordinate restoration, bilinear interpolation, overlap averaging, occupancy selection, cap and seed, cosine Top-K, binary thresholding, multi-class argmax and pipeline, native/256 output shapes, query feature caching and fingerprint invalidation, RGB label mapping, image-only manifests, empty Memory, invalid Top-K and configuration, and synthetic Support→Memory→Query→Mask→Metrics.

A separate actual UNI end-to-end smoke run used one GLaS support image and one query with `run --output-size native --save-margin true`; see `outputs/e2e_smoke` for its Memory, native mask, native and output score arrays, metrics, and resolved configuration.

An image-only manifest was tested with `infer --top-k 20 --output-size native --save-margin true` on a 522×775 GLaS query. Three independent runs saved the same binary Mask. Between two saved native Margin arrays, maximum absolute difference was `5.73e-5`, mean difference `1.64e-7`, and thresholded masks were identical. This is the observed CUDA floating-point repeatability tolerance for that check.

## DUAL numerical parity

The optional `tests/reference_parity.py` was executed for both datasets using the same weights, support IDs, seed, query image, and DUAL Memory. Detailed machine-readable records are `outputs/glas10/parity.json` and `outputs/crag10/parity.json`.

| Check | GLaS | CRAG |
|---|---:|---:|
| Rebuilt FG/BG counts | 3584 / 3584 | 8192 / 8192 |
| Cached Memory sampling indices/vectors | Exact | Exact |
| Fresh Memory max absolute value difference vs DUAL | 1.22e-4 | 2.44e-4 |
| Cached query token margins | Exact | Exact |
| Reconstructed native margin, rounded fp16 | Exact | Exact |
| Cached 256 mask differences | 0 | 0 |
| Direct function vs new pipeline, native max absolute difference | 5.73e-5 | 5.73e-5 |
| Direct function vs new pipeline, 256 mask differences | 0 | 0 |

The fresh Memory difference is from floating-point UNI feature extraction: support window coordinates, GT occupancies, selected indices, vector counts, and fp16 storage all match. Using DUAL's cached features makes selection and token retrieval exactly equal. The `reference_parity.py` Direct comparison uses one official test image per dataset; full-set mask comparison is below. `align_corners=False`, fp16 token margin rounding, window mean, offset mean, and 256 bilinear resize follow the actual Direct script. DUAL also contains `N3_nn` caches, but the verified Direct path upsamples margin with bilinear interpolation; this project does not substitute token repeat.

## Full test regression

Both runs used only `labeled_10.csv` for Memory and official `test.csv` for inference/evaluation. GT was resized by OpenCV nearest neighbor to 256, matching DUAL's Direct evaluation. No test GT was used for Memory or parameter choice.

| Dataset | Images | New mean-image Dice | Historical mean-image Dice | New micro Dice | Historical micro Dice | Pixel differences vs historical masks |
|---|---:|---:|---:|---:|---:|---:|
| GLaS 10% | 80 | 0.922127099 | 0.922126761 | 0.925135361 | 0.925135000 | 2 |
| CRAG 10% | 40 | 0.911317009 | 0.911317009 | 0.917144811 | 0.917144811 | 0 |

GLaS has 2 changed pixels in its full test set after independently extracting fp16 UNI features; CRAG masks match exactly. GLaS object-F1 is 0.808487782 and object-Hausdorff is 39.309716 pixels; CRAG values are 0.791011248 and 33.079697 pixels. Full per-image metrics and saved masks are under `outputs/glas10` and `outputs/crag10`.

## Reproduction commands

Set `UNI_WEIGHTS` to the local checkpoint, then run from this project directory:

```bash
python -m pathology_trainfree build-memory --config configs/glas_example.yaml --weights "$UNI_WEIGHTS"
python -m pathology_trainfree run --config configs/glas_example.yaml --weights "$UNI_WEIGHTS"
python -m pathology_trainfree build-memory --config configs/crag_example.yaml --weights "$UNI_WEIGHTS"
python -m pathology_trainfree run --config configs/crag_example.yaml --weights "$UNI_WEIGHTS"
python tests/reference_parity.py --dual ../DUAL --dataset GlaS --weights "$UNI_WEIGHTS" --built-memory outputs/glas10/memory.npz --output outputs/glas10/parity.json
python tests/reference_parity.py --dual ../DUAL --dataset CRAG --weights "$UNI_WEIGHTS" --built-memory outputs/crag10/memory.npz --output outputs/crag10/parity.json
```

The target project is not currently a Git repository, so its `run_info.json` records a null target commit. The exact resolved configuration, runtime versions, support/query IDs, command, and UNI fingerprint are retained in output files.
