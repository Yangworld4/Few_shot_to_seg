# Query 自参考细化验证记录

当前入口为 `python scripts/evaluate.py --config configs/eval_query_refinement.yaml`，默认从原图提取特征。新结果使用 `support_only` 和 `query_refined`，下文保留历史结果文件中的实验标签和路径。

## 仓库结构与实现位置

主包的 `models/` 包含 UNI 编码器，`core/` 管理参考库、检索与几何，`utils/` 管理评估指标。YAML 配置位于 `configs/`，测试位于 `tests/`。

| 环节 | 现有实现与移植依据 |
| --- | --- |
| 特征 | `models/encoder.py` 从原图提取 UNI 原始 block 特征，`selfref_data.py` 也可读取历史缓存。 |
| Support 库 | 归档 `analysis/coreset_memory_bank/banks.py::prepare/make_bank` 与参考 `tokens.py::pure_tokens`、`bank.py::kcenter_order`。 |
| 跨图打分 | 归档 `analysis/coreset_memory_bank/experiment.py::SupportScorer` 和参考 `scoring.py::support_margins`。 |
| Seed 与自参考 | 归档 `analysis/self_reference_stage1/src/seed_construction.py::select` 与 `reference_matching.py::set_query_cache/match`。 |
| 重建 | 当前 `core/geometry.py::reconstruct` 与归档 `token_alignment.py::reconstruction`、参考 `reconstruct.py::reconstruct_windows_batched`。 |
| 评估 | 归档 `experiment.py::evaluate_query_batch/dice_iou/missed_glands` 和参考 `evaluate.py`、`_objects.py` 的 evaluation_v6。 |
| OOF | 归档 `experiment.py::run_oof` 与参考 `experiments/seed_quality/study.py::oof_pairs`，按清单相邻循环。 |
| 方法入口 | `configs/eval_query_refinement.yaml`。历史 C1-2048 的入口是归档 `analysis/coreset_radius/run.py`，配置是参考仓库 `configs/support_bank_c1_2048.yaml`。 |

`scripts/evaluate.py` 对 Support × Query 配对运行原图尺寸评估，并输出每个 Support 的平均成绩。

Support 失败定义为该 Support 的平均 Dice 严格小于 **0.8**。位置是参考仓库 `experiments/support_rule/support_test.py::summary`，其中 `support_failure=(s<.8).mean()`。

## 新增和修改

| 文件 | 作用 |
| --- | --- |
| `pathology_trainfree/core/selfref_geometry.py` | Native 反射中心、Hann 加权、有效像素纯度和多相位重建。 |
| `pathology_trainfree/core/selfref.py` | 共用预算、选点、k-center 缓存、Top-k 和自参考。 |
| `pathology_trainfree/selfref_data.py` | 缓存、正式清单、FP_high 和相邻循环 OOF。 |
| `pathology_trainfree/selfref_config.py` | 命令行、YAML、代码默认值的优先级及参数检查。 |
| `pathology_trainfree/selfref_evaluate.py` | 组织配对、并行 Native 评估、输出和运行溯源。 |
| `pathology_trainfree/utils/evaluation_v6.py` | Native 对象配对、对象指标及整腺体漏检。 |
| `pathology_trainfree/utils/selfref_edt.py` | CRAG 精确整数网格欧氏距离变换。 |
| `pathology_trainfree/utils/selfref_metrics.py` | 语义指标和 Support 汇总。 |
| `scripts/evaluate_selfref.py` | 缓存评估入口。 |
| `scripts/compare_selfref.py` | 按冻结逐对键值核对 Dice，并输出逐对差异和汇总。 |
| `configs/eval_query_refinement.yaml`、`configs/default.yaml` | 通用自参考评估入口及实验参数。 |
| `configs/selfref_manifests/{glas,crag}_{splits,fp_high}.csv` | 固定评估顺序、GT 路径与校验摘要、历史 FP_high 标签。 |
| `pyproject.toml` | 在可选依赖 `selfref` 中声明 Native CRAG 的 Numba 依赖。 |
| `tests/test_selfref_protocol.py` | 预算、k-center 前缀、参数优先级、无自参考、seed 和 OOF 顺序测试。 |
| `README.md` | 使用命令、调参、缓存目录和输出约定。 |

## 协议细节与偏离

- Support 按相位去重，按 Hann 窗中心权重对重叠窗口聚合。GT 纯度仅用 Native 有效像素。k-center 在 FP16 存储向量转回 FP32 并再次归一化后选点，对相同距离取候选原顺序的首项。跨图评分保留参考实现的 FP16 相似度与 FP32 均值。
- E1a 对正负两侧分别向上取整保留 seed 数。自参考相似度使用 FP32，最多取 Top-k，排除相同反射坐标，包括不同相位中重合的坐标。任一类没有可用参照时，该位置返回 E0。默认重建先将 margin 舍入至 FP16，再按 FP32 做窗口双线性和相位融合。
- 默认恢复官方清单顺序。OOF 用词典序清单的下一张 Train 图作 Support，末项循环。FP_high 是冻结的 Test 配对分组，OOF 无历史标签，其 FP_high 分组计数为零。
- N3-bil 采用历史 UNI 最终特征、反射窗口纯度、种子 2026 的 PCG64 按 FG 然后 BG 抽样、每类最多 512 和 Top-50。使用该预设时，`--feature-cache` 指向历史 outputs 根目录，DUAL 缓存仅作为特征输入，不加载 DUAL Python 代码。
- 本次验证采用只读参考仓库和此前清理后仍位于系统回收站的历史缓存。没有恢复、改写或删除归档中的实验，也没有运行 UNI 前向。复现时把 `--feature-cache` 指向可持续访问的同格式缓存。
- 当前入口通过 `one_shot.refine_query` 控制细化。设为 `false` 时只输出基于 Support 检索的结果。
- 历史固定预算实验可用 `--budget-mode fixed --budget-cap 2048`。历史最终层实验使用 `--feature-format n3 --layer 24 --bank-select random --budget-mode fixed --budget-cap 512 --support-topk 50 --refine-query false --reconstruction bilinear-window`，并通过 `--feature-cache` 指定历史缓存目录。

## 测试

在 `/data/software/miniconda3/envs/cg-seg/bin/python` 中运行 `python -m pytest -p no:cacheprovider -q tests`，23 项全部通过。[测试日志](../outputs/selfref/verification/tests.txt)

- GlaS 和 CRAG 各选一组 Support × Query，E0 与 E1a 的 Dice 最大差为 5.55×10⁻¹⁷。CPU 单配对也验证了可选的 Native PNG、margin NPY，以及 `--selfref none` 下两个分支文件逐字节相同。
- 固定 OOF 清单的 85 个 GlaS 和 173 个 CRAG 映射与归档清单一一对应。N3-bil 的 `train_1` 参考库在两个数据集都与历史保存的 FG/BG 向量逐元素相同。
- 首轮指标核对发现背景标签 0 的坐标被映射到了最后一个前景实例，影响 Object HD。修正后，回归样本 `GlaS/train_36/test_a_testA_41/E0` 得到 319.1164845907915 原生像素，与参考值相同。全量报告使用修正后的运行。 [回归样本](../outputs/selfref/hd_background_verification/per_pair.csv)

## 门槛核对方式

逐对核对键为 `dataset, support_id, query_id, branch`。S1 Test 核对 13,720 个配对的两条分支，并直接比较 `summary.json` 与冻结 `T3_test.csv` 的均值和最差 10%。C1-2048 和 N3-bil 抽取各数据集官方清单的前五个 Support，分别跨全部 80 和 40 个 Query。Train OOF 核对完整的循环清单，E1a 使用 1e-4 的逐对门槛。

```bash
python scripts/compare_selfref.py --gate kcenter --run outputs/selfref/s1_test_verified \
  --reference ../oneshot_gland/results/support_rule
python scripts/compare_selfref.py --gate train-oof --run outputs/selfref/s1_train_oof \
  --reference ../oneshot_gland/results/support_rule
```

保留旧入口的测试使用一个历史 N3-bil Support（`train_1`）对每个数据集的所有掩码。分别运行 Git HEAD 的原评估命令和当前 `scripts/evaluate.py`，比较标准输出、`per_image.csv` 和 `metrics.json` 的字节内容。 [旧入口核对日志](../outputs/selfref/verification/legacy.txt)

## S1 + E1a 完整 Test 结果

| 数据集 | 分支 | 配对数 | 平均 Dice | 最差 10% Support Dice | 逐对 Dice 最大绝对差 |
| --- | --- | ---: | ---: | ---: | ---: |
| GlaS | E0 | 6,800 | 0.901950 | 0.864108 | 1.11×10⁻¹⁶ |
| GlaS | E1a | 6,800 | 0.910392 | 0.898551 | 1.11×10⁻¹⁶ |
| CRAG | E0 | 6,920 | 0.856337 | 0.772702 | 1.11×10⁻¹⁶ |
| CRAG | E1a | 6,920 | 0.878674 | 0.826376 | 1.11×10⁻¹⁶ |

四组平均 Dice 与最差 10% 的最大绝对差为 2.22×10⁻¹⁶。额外核对的 Native Object HD 最大绝对差为 2.27×10⁻¹³ 像素，整腺体漏检数逐对一致。 [汇总](../outputs/selfref/s1_test_verified/summary.csv) · [完整逐对结果](../outputs/selfref/s1_test_verified/per_pair.csv) · [逐对 Dice 核对](../outputs/selfref/s1_test_verified/parity.csv) · [Native 指标核对](../outputs/selfref/s1_test_verified/native_metrics_parity.json)

完整 Test 使用单张 RTX 4090 和 24 个评估进程，墙钟耗时 **1,175.0 秒（19 分 35 秒）**，其中 GlaS 173.9 秒、CRAG 998.5 秒。支持库重建分别耗时 37.5 和 244.4 秒，本次使用同一工作区保存的部分 k-center 顺序缓存。 [运行配置和计时](../outputs/selfref/s1_test_verified/resolved_config.yaml)

## 复现门槛状态

| 门槛 | 状态 | 核对结果 |
| --- | --- | --- |
| S1 + E1a 完整 Test | 通过 | 两条分支共 27,440 条记录，逐对 Dice 最大差 1.11×10⁻¹⁶，汇总最大差 2.22×10⁻¹⁶。 [逐对核对](../outputs/selfref/s1_test_verified/parity.csv) |
| C1-2048 + E1a 抽样 Test | 通过 | 600 个配对、1,200 条分支记录，逐对 Dice 最大差 1.11×10⁻¹⁶。 [逐对核对](../outputs/selfref/c1_sample_parity/parity.csv) |
| N3-bil 抽样 Test | 通过 | 600 个配对，逐对 Dice 完全相同。 [逐对核对](../outputs/selfref/n3_sample_parity/parity.csv) |
| 旧 evaluate 默认行为 | 通过 | 每套数据用 train_1 的一套掩码评估全部 Query，GlaS 80 张、CRAG 40 张，标准输出、逐图 CSV 和 JSON 汇总逐字节一致。 [核对日志](../outputs/selfref/verification/legacy.txt) |
| S1 + E1a Train OOF | 输入缺失，待完成 | 已落盘的 46 个 GlaS 配对两条分支的最大 Dice 差都是 1.11×10⁻¹⁶。完整门槛需要 85 个 GlaS 和 173 个 CRAG 配对。 [部分逐对核对](../outputs/selfref/s1_train_oof/parity.partial.csv) · [中断记录](../outputs/selfref/s1_train_oof/parity.partial.json) |

C1-2048 抽样在 GlaS 用时 31.6 秒，在 CRAG 用时 67.2 秒。N3-bil 抽样分别用时 20.9 和 52.9 秒。Train OOF 运行 47.7 秒时，原始和归一化 F19 缓存目录从此前可读的位置消失，首个报错文件是 GlaS `support/train_78/offset_0.npz`。剩余参考缓存仅含 Support 候选和 OT 压缩统计，无法重建完整 Query 特征。该项目前只记录已完成配对的核对结果。

## Git 工作区

本次实现留在工作区。Git 统计以当前 HEAD 为基准，包含本轮开始前已存在的重构和研究代码清理。常规 `git diff --stat` 只统计已跟踪文件。 [完整状态](../outputs/selfref/verification/git_status.txt) · [差异统计](../outputs/selfref/verification/git_diff_stat.txt)

<details>
<summary>git status --short --branch --untracked-files=all</summary>

```text
## main...origin/main
 M .gitignore
 M README.md
 D VALIDATION.md
 D analyze_complementarity.py
 D configs/self_reference_e0.yaml
 M pathology_trainfree/cache.py
 M pathology_trainfree/cli.py
 D pathology_trainfree/diagnostics.py
 M pathology_trainfree/encoder.py
 M pathology_trainfree/evaluation.py
 M pathology_trainfree/geometry.py
 M pathology_trainfree/memory.py
 M pathology_trainfree/pipeline.py
 M pathology_trainfree/retrieval.py
 D pathology_trainfree/self_reference/README.md
 D pathology_trainfree/self_reference/__init__.py
 D pathology_trainfree/self_reference/diagnostics.py
 D pathology_trainfree/self_reference/features.py
 D pathology_trainfree/self_reference/retrieval.py
 D pathology_trainfree/self_reference/seed_generator.py
 D pathology_trainfree/self_reference/token_mapping.py
 D pathology_trainfree/shift_stitch.py
 D plot_complementarity.py
 M pyproject.toml
 D report_complementarity.py
 D run_complementarity.py
 D scripts/audit_self_reference_gates.py
 D scripts/check_encoder_batching.py
 D scripts/check_near_duplicate_candidates.py
 D scripts/check_uni_shift_stitch_geometry.py
 D scripts/dense_feature_matching.py
 D scripts/diagnose_self_reference_seeds.py
 D scripts/diagnostic_memory_statistics.py
 D scripts/finalize_uni_shift_stitch.py
 D scripts/index_raw_interference_evidence.py
 D scripts/native_diagnostics_seed1337.py
 D scripts/real_occupancy_seed1337.py
 D scripts/reference_coverage_diagnosis.py
 D scripts/reference_source_stain.py
 D scripts/repair_memory_serialization.py
 D scripts/report_dense_feature_matching.py
 D scripts/report_reference_coverage.py
 D scripts/report_reference_source_stain.py
 D scripts/report_self_reference_seeds.py
 D scripts/report_support_size_gtfixed.py
 D scripts/report_uni_shift_stitch.py
 D scripts/save_interference_evidence.py
 D scripts/self_reference_extra_summaries.py
 D scripts/self_reference_physical_diversity.py
 D scripts/smoke_encoders.py
 D scripts/summarize_reference_coverage.py
 D scripts/summarize_self_reference_seeds.py
 D scripts/uni_shift_stitch_seed1337.py
 D scripts/validate_self_reference_artifacts.py
 D scripts/verify_dense_feature_matching.py
 D scripts/verify_reference_coverage.py
 D scripts/verify_reference_source_stain.py
 D study_support_size.py
 D tests/reference_parity.py
 M tests/test_core.py
 D tests/test_diagnostics.py
 D tests/test_self_reference.py
 D tests/test_support_cache_occupancy.py
 D verify_complementarity.py
 D visualize_complementarity.py
?? configs/c1_2048_e1a.yaml
?? configs/n3_bil.yaml
?? configs/s1_e1a.yaml
?? configs/selfref_manifests/crag_fp_high.csv
?? configs/selfref_manifests/crag_splits.csv
?? configs/selfref_manifests/glas_fp_high.csv
?? configs/selfref_manifests/glas_splits.csv
?? docs/protocol.md
?? docs/s1_e1a_implementation_report.md
?? pathology_trainfree/core/__init__.py
?? pathology_trainfree/core/geometry.py
?? pathology_trainfree/core/memory.py
?? pathology_trainfree/core/pipeline.py
?? pathology_trainfree/core/retrieval.py
?? pathology_trainfree/core/selfref.py
?? pathology_trainfree/core/selfref_geometry.py
?? pathology_trainfree/models/__init__.py
?? pathology_trainfree/models/encoder.py
?? pathology_trainfree/selfref_config.py
?? pathology_trainfree/selfref_data.py
?? pathology_trainfree/selfref_evaluate.py
?? pathology_trainfree/utils/__init__.py
?? pathology_trainfree/utils/cache.py
?? pathology_trainfree/utils/evaluation.py
?? pathology_trainfree/utils/evaluation_v6.py
?? pathology_trainfree/utils/selfref_edt.py
?? pathology_trainfree/utils/selfref_metrics.py
?? requirements.txt
?? scripts/__init__.py
?? scripts/build_memory.py
?? scripts/compare_selfref.py
?? scripts/evaluate.py
?? scripts/evaluate_selfref.py
?? scripts/infer.py
?? scripts/run_pipeline.py
?? tests/test_selfref_protocol.py
```

</details>

<details>
<summary>git diff --stat</summary>

```text
 .gitignore                                         |    4 +
 README.md                                          |  163 +--
 VALIDATION.md                                      |   54 -
 analyze_complementarity.py                         |  127 ---
 configs/self_reference_e0.yaml                     |   81 --
 pathology_trainfree/cache.py                       |   97 +-
 pathology_trainfree/cli.py                         |   28 +-
 pathology_trainfree/diagnostics.py                 |   76 --
 pathology_trainfree/encoder.py                     |  129 +--
 pathology_trainfree/evaluation.py                  |  119 +--
 pathology_trainfree/geometry.py                    |   47 +-
 pathology_trainfree/memory.py                      |  143 +--
 pathology_trainfree/pipeline.py                    |   95 +-
 pathology_trainfree/retrieval.py                   |   37 +-
 pathology_trainfree/self_reference/README.md       |   54 -
 pathology_trainfree/self_reference/__init__.py     |    3 -
 pathology_trainfree/self_reference/diagnostics.py  |   55 -
 pathology_trainfree/self_reference/features.py     |   23 -
 pathology_trainfree/self_reference/retrieval.py    |   56 -
 .../self_reference/seed_generator.py               |  133 ---
 .../self_reference/token_mapping.py                |   42 -
 pathology_trainfree/shift_stitch.py                |  153 ---
 plot_complementarity.py                            |   38 -
 pyproject.toml                                     |    4 +-
 report_complementarity.py                          |  108 --
 run_complementarity.py                             |  228 ----
 scripts/audit_self_reference_gates.py              |   51 -
 scripts/check_encoder_batching.py                  |   24 -
 scripts/check_near_duplicate_candidates.py         |   16 -
 scripts/check_uni_shift_stitch_geometry.py         |   56 -
 scripts/dense_feature_matching.py                  |  216 ----
 scripts/diagnose_self_reference_seeds.py           |  201 ----
 scripts/diagnostic_memory_statistics.py            |   31 -
 scripts/finalize_uni_shift_stitch.py               |  323 ------
 scripts/index_raw_interference_evidence.py         |   22 -
 scripts/native_diagnostics_seed1337.py             |  697 ------------
 scripts/real_occupancy_seed1337.py                 | 1107 --------------------
 scripts/reference_coverage_diagnosis.py            |  211 ----
 scripts/reference_source_stain.py                  |  278 -----
 scripts/repair_memory_serialization.py             |   19 -
 scripts/report_dense_feature_matching.py           |  163 ---
 scripts/report_reference_coverage.py               |   44 -
 scripts/report_reference_source_stain.py           |  318 ------
 scripts/report_self_reference_seeds.py             |   67 --
 scripts/report_support_size_gtfixed.py             |  144 ---
 scripts/report_uni_shift_stitch.py                 |  129 ---
 scripts/save_interference_evidence.py              |   32 -
 scripts/self_reference_extra_summaries.py          |   29 -
 scripts/self_reference_physical_diversity.py       |   13 -
 scripts/smoke_encoders.py                          |   19 -
 scripts/summarize_reference_coverage.py            |   76 --
 scripts/summarize_self_reference_seeds.py          |  114 --
 scripts/uni_shift_stitch_seed1337.py               |  357 -------
 scripts/validate_self_reference_artifacts.py       |   58 -
 scripts/verify_dense_feature_matching.py           |   43 -
 scripts/verify_reference_coverage.py               |   36 -
 scripts/verify_reference_source_stain.py           |   38 -
 study_support_size.py                              |  917 ----------------
 tests/reference_parity.py                          |  116 --
 tests/test_core.py                                 |   13 +
 tests/test_diagnostics.py                          |   60 --
 tests/test_self_reference.py                       |  131 ---
 tests/test_support_cache_occupancy.py              |   42 -
 verify_complementarity.py                          |  105 --
 visualize_complementarity.py                       |   71 --
 65 files changed, 150 insertions(+), 8334 deletions(-)
```

</details>
