# PathologyTrainFree

在 GlaS 和 CRAG 原图上做 one-shot 腺体分割，评估换一张标注 Support 图对结果的影响。每个配对只用一张 Support，预测掩码与 Query 原图同尺寸。

## 运行

从项目根目录运行。激活 `cg-seg` 环境，确认 [`configs/default.yaml`](configs/default.yaml) 中的 `encoder.weights` 和 `one_shot.data_root` 指向本地权重与数据目录。

```bash
conda activate cg-seg
python scripts/evaluate.py --config configs/eval_query_refinement.yaml --dataset GlaS --tag glas
```

CRAG 使用同一份配置。

```bash
python scripts/evaluate.py --config configs/eval_query_refinement.yaml --dataset CRAG --tag crag
```

省略 `--dataset` 会评估两个数据集。默认每张训练图单独作为 Support，与每张测试图配对。先试一对图可加 `--support-ids train_1 --query-ids test_a_testA_1 --tag glas_smoke`，并指定 `--dataset GlaS`。

## 调参和结果

- `one_shot.refine_query: true` 输出 Support 检索和 Query 细化两组结果。设为 `false` 时只输出 Support 检索结果。
- `one_shot.bank_select` 控制 Support token 的选点方式，可在 `kcenter` 和 `random` 间切换。开启 `one_shot.save_masks` 可保存原分辨率掩码。
- `per_support.csv` 按 Support 图汇总测试集成绩。`per_pair.csv` 记录各对结果，`summary.csv` 汇总 Support 均值、标准差、最差 10% 的平均 Dice 和失败比例。结果保存在 `outputs/selfref/<tag>/`。
- 默认从原图提取冻结 UNI 第 19 层特征。复用已有特征时，把 `one_shot.feature_source` 设为 `cache` 并设置 `one_shot.feature_cache`。

## SAM 对照实验

`default.yaml` 的 `sam.enabled` 默认是 `false`。需要验证 SAM 时，可以在命令后加 `--sam`，或在 YAML 中设置 `sam.enabled: true`。命令行开关优先于 YAML。

```bash
python scripts/evaluate.py --config configs/eval_query_refinement.yaml --dataset CRAG --tag crag_sam --sam
```

启用时，SAM 使用候选腺体的外接框、距边缘最远的正点和附近低 margin 背景点。`sam.iou_threshold` 控制采纳阈值。默认比较 `query_refined`、`query_refined_cleanup` 和 `query_refined_sam`，后两组采用相同的清理设置。成对差值和候选采用数量见 `sam_comparison.csv`。用 `--no-sam` 可覆盖启用了 SAM 的 YAML 配置。

在新环境可用 `pip install -r requirements.txt` 安装依赖。历史实验结果见 [自参考评估验证记录](docs/query_refinement_validation.md)。
