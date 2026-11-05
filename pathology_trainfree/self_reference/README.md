# Self-reference seed diagnosis

The generator accepts only RGB and the predeclared YAML parameters. No masks,
training supports, external segmenters or GT-derived occupancy enter generation.
A/B/C are fixed exploratory heuristics; none is asserted to be a reliable gland
segmenter without the reported diagnostics.

From the project root:

```bash
pip install -e '.[self-reference]'
python scripts/diagnose_self_reference_seeds.py --config configs/self_reference_e0.yaml --stage E0 --resume
python scripts/diagnose_self_reference_seeds.py --config configs/self_reference_e0.yaml --stage E1 --resume
python scripts/diagnose_self_reference_seeds.py --config configs/self_reference_e0.yaml --stage E2 --gpu cuda:0 --resume
python scripts/summarize_self_reference_seeds.py
python scripts/self_reference_extra_summaries.py
python scripts/self_reference_physical_diversity.py
python scripts/audit_self_reference_gates.py
python scripts/report_self_reference_seeds.py
python scripts/validate_self_reference_artifacts.py
```

`--stage all` executes the ordered stages. The entry point also accepts
`--dataset GlaS CRAG`, `--seed-generator A B C`, `--cache-root`, `--output-root`,
`--gpu`, `--resume`, and `--overwrite`. Dataset manifests and algorithm parameters
are in YAML. A changed protocol needs a separate output directory or explicit
`--overwrite`; downstream stage fingerprints prevent mixing old masks/memories
with a new protocol. The default summaries target the task's default output root.

E0 generates every requested image before reading GT pixels for evaluation. E1
saves seed-only candidate indices and memories before loading a separately named
GT oracle. E2 computes predictions before GT evaluation. Cache fingerprints must
match RGB, original UNI checkpoint, precision and window geometry. No encoder
forward is allowed in E2's cache-only adapter.

A missing class produces EMPTY_MEMORY; 1–49 vectors produces
INSUFFICIENT_MEMORY. S1 rejects an entire image/mode if any query token has fewer
than 50 eligible references in either class. It never lowers K or falls back to
S0. Native reflected coordinates exclude duplicated overlapping-window positions
and neighbours within the fixed 32-pixel radius. S0/S1 comparisons additionally
report matched valid-image cohorts.

Per-image checkpoints support recovery. All empty and invalid conditions remain
in the CSV denominators. Macro precision omits empty predictions as N/A and
reports its valid denominator; coverage and nonempty rates retain those images.
Object recovery uses GT connected components at 256 resolution and a predeclared
coverage criterion, so it must be interpreted together with specificity and
Object-F1. It is not equivalent to correct instance segmentation.

Tests:

```bash
python -m pytest tests/test_self_reference.py -q
```
