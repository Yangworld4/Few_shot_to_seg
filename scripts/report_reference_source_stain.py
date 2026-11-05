"""Analysis of actual saved outputs; query-cluster paired bootstrap, no token-level inference."""
from pathlib import Path
import sys, json, csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reference_source_stain import OUT, SEEDS, table, dump, load_image, load_mask, resize_mask, DC
METRICS = ['dice', 'fg_iou', 'object_f1', 'auroc']

def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(2026)
    boots = x[rng.integers(0, len(x), (10000, len(x)))].mean(1)
    return (float(x.mean()), *np.quantile(boots, [0.025, 0.975]).tolist())

def fmt(x):
    (m, l, h) = ci(x)
    return f'{m:.5f} [{l:.5f}, {h:.5f}]'

def report():
    augment_strict_unseeded_objects()
    meta = json.loads((OUT / 'run_metadata.json').read_text())
    e = pd.read_csv(OUT / 'external_results_per_pair.csv')
    s = pd.read_csv(OUT / 'self_oracle_results_per_image.csv')
    gap = pd.read_csv(OUT / 'stain_gap_metrics.csv')
    fig = OUT / 'figures'
    fig.mkdir(exist_ok=True)
    ex = e[e.condition != 'RAW'].copy()
    sx = s[s['mode'] == 'SpatialExclusion'].copy()
    no = s[s['mode'] == 'NoExclusion'].copy()
    effects = []
    sources = []
    paired = []
    common = {}
    for metric in METRICS:
        ep = ex.pivot(index=['query_id', 'seed'], columns='condition', values=metric)
        sp = sx.pivot(index='query_id', columns='condition', values=metric)
        npiv = no.pivot(index='query_id', columns='condition', values=metric)
        validq = ep.dropna().reset_index().groupby('query_id').size()
        ids = sorted(set(validq[validq == 5].index) & set(sp.dropna().index))
        common[metric] = ids
        ee = ep.loc[ep.index.get_level_values('query_id').isin(ids)]
        ss = sp.loc[ids]
        ed = (ee.AA + ee.BB - ee.AB - ee.BA) / 2
        sd = (ss.AA + ss.BB - ss.AB - ss.BA) / 2
        for (source, piv) in [('External', ep), ('Self', sp)]:
            for (contrast, vals) in [('Matched-Mismatched', (piv.AA + piv.BB - piv.AB - piv.BA) / 2), ('AA-BA', piv.AA - piv.BA), ('BB-AB', piv.BB - piv.AB)]:
                if source == 'External':
                    for seed in SEEDS:
                        v = vals.xs(seed, level='seed')
                        (m, l, h) = ci(v)
                        effects.append({'metric': metric, 'source': source, 'contrast': contrast, 'support_seed': seed, 'n_query': int(v.notna().sum()), 'effect': m, 'ci_low': l, 'ci_high': h})
                    vals = vals.groupby('query_id').agg(lambda v: v.mean() if v.notna().sum() == 5 else np.nan)
                (m, l, h) = ci(vals)
                effects.append({'metric': metric, 'source': source, 'contrast': contrast, 'support_seed': 'ALL', 'n_query': int(vals.notna().sum()), 'effect': m, 'ci_low': l, 'ci_high': h})
        em = ((ee.AA + ee.BB) / 2).groupby('query_id').mean()
        sm = (ss.AA + ss.BB) / 2
        ed = ed.groupby('query_id').mean()
        sourcevals = {'delta_E': ed, 'delta_S': sd, 'delta_source': sm - em, 'delta_interaction': ed - sd, 'source_queryA': ss.AA - ee.AA.groupby('query_id').mean(), 'source_queryB': ss.BB - ee.BB.groupby('query_id').mean()}
        for (name, v) in sourcevals.items():
            (m, l, h) = ci(v)
            sources.append({'metric': metric, 'effect_name': name, 'n_query': len(ids), 'effect': m, 'ci_low': l, 'ci_high': h})
        for q in ids:
            paired.append({'query_id': q, 'metric': metric, **{k: float(v.loc[q]) for (k, v) in sourcevals.items()}})
    table(OUT / 'paired_effects.csv', effects)
    table(OUT / 'source_effects.csv', sources)
    table(OUT / 'paired_query_effects.csv', paired)
    flips = []
    for df in [ex, sx, no]:
        for ((qid, seed), group) in df.groupby(['query_id', 'seed']):
            by = {r.condition: r for r in group.itertuples()}
            for (a, b) in [('AA', 'BA'), ('BB', 'AB')]:
                if by[a].status != 'VALID' or by[b].status != 'VALID':
                    continue
                (r1, r2) = (by[a], by[b])
                with np.load(OUT / 'cases' / f'{r1.case}.npz') as z:
                    x = z['token_scores']
                with np.load(OUT / 'cases' / f'{r2.case}.npz') as z:
                    y = z['token_scores']
                with np.load(OUT / 'cache/geometry' / f'{qid}.npz') as z:
                    oc = z['occupancy']
                xm = x[:, 1] - x[:, 0]
                ym = y[:, 1] - y[:, 0]
                for (cls, sel) in [(1, oc >= 0.9), (0, oc <= 0.1)]:
                    correct = (xm > 0) == cls
                    harmed = correct & ((ym > 0) != cls) & sel
                    flips.append({'query_id': qid, 'seed': seed, 'source': r1.source, 'mode': r1.mode, 'query_style': r1.query_style, 'gt_class': cls, 'tokens': int(sel.sum()), 'correct_matched': int((correct & sel).sum()), 'harmed': int(harmed.sum()), 'error_flip_ratio_all_class': float(harmed.sum() / max(1, sel.sum())), 'error_flip_ratio_correct_matched': float(harmed.sum() / max(1, (correct & sel).sum())), 'delta_correct_cos_mismatch_minus_match': float((y[sel, cls] - x[sel, cls]).mean()), 'delta_wrong_cos_mismatch_minus_match': float((y[sel, 1 - cls] - x[sel, 1 - cls]).mean()), 'delta_margin_mismatch_minus_match': float((ym[sel] - xm[sel]).mean())})
    table(OUT / 'retrieval_paired_changes.csv', flips)
    gg = gap[(gap.source == 'External') & (gap.condition != 'RAW')]
    joined = ex.merge(gg, on=['query_id', 'reference_id', 'seed', 'condition', 'mode', 'source'])
    ar = []
    for ((q, se), g) in joined.groupby(['query_id', 'seed']):
        g = g.set_index('condition')
        for (ma, mi) in [('AA', 'BA'), ('BB', 'AB')]:
            if not all((g.loc[c, 'status'] == 'VALID' for c in [ma, mi])):
                continue
            ar.append({'query_id': q, 'seed': se, 'fixed_query_style': g.loc[ma, 'query_style'], 'gap_increase': g.loc[mi, 'lab_distance'] - g.loc[ma, 'lab_distance'], 'he_gap_increase': g.loc[mi, 'he_distance'] - g.loc[ma, 'he_distance'], 'dice_degradation': g.loc[ma, 'dice'] - g.loc[mi, 'dice'], 'auroc_degradation': g.loc[ma, 'auroc'] - g.loc[mi, 'auroc']})
    ad = pd.DataFrame(ar)
    table(OUT / 'paired_stain_gap_changes.csv', ar)
    (f, axs) = plt.subplots(1, 2, figsize=(10, 4))
    for (ax, m) in zip(axs, ['dice', 'auroc']):
        for (style, color) in [('A', '#305c9a'), ('B', '#c75b45')]:
            v = ad[ad.fixed_query_style == style]
            ax.scatter(v.gap_increase, v[m + '_degradation'], s=12, alpha=0.4, c=color, label='Query ' + style)
        ax.axhline(0, c='gray', lw=0.7)
        ax.axvline(0, c='gray', lw=0.7)
        ax.set(xlabel='Within-pair increase in measured Lab gap', ylabel='Matched - mismatched ' + m)
        ax.legend()
    f.suptitle('A  Stain Distance vs Retrieval Degradation')
    f.tight_layout()
    f.savefig(fig / 'figure_A.png', dpi=220)
    f.savefig(fig / 'figure_A.pdf')
    plt.close(f)
    (f, axs) = plt.subplots(1, 3, figsize=(12, 4))
    summary = []
    for (ax, m) in zip(axs, ['dice', 'object_f1', 'auroc']):
        for (i, (source, df)) in enumerate([('External', ex), ('Self Oracle', sx)]):
            p = df.pivot(index=['query_id', 'seed'], columns='condition', values=m)
            p = p.dropna()
            p = p.groupby('query_id').mean()
            for (j, (label, v)) in enumerate([('Matched', (p.AA + p.BB) / 2), ('Mismatched', (p.AB + p.BA) / 2)]):
                (mean, l, h) = ci(v)
                x = i * 3 + j
                ax.errorbar(x, mean, yerr=[[mean - l], [h - mean]], fmt='o', capsize=4)
                summary.append({'source': source, 'metric': m, 'relation': label, 'mean': mean, 'ci_low': l, 'ci_high': h, 'n_query': len(v)})
        ax.set_xticks([0, 1, 3, 4], ['E match', 'E mismatch', 'S match', 'S mismatch'], rotation=25)
        ax.set_title(m)
    f.suptitle('B  Matched vs Mismatched Retrieval (query bootstrap 95% CI)')
    f.tight_layout()
    f.savefig(fig / 'figure_B.png', dpi=220)
    f.savefig(fig / 'figure_B.pdf')
    plt.close(f)
    table(OUT / 'condition_summary.csv', summary)
    (f, axs) = plt.subplots(1, 3, figsize=(13, 4))
    sourceframe = pd.DataFrame(sources)
    for (ax, m) in zip(axs, ['dice', 'object_f1', 'auroc']):
        sub = [r for r in summary if r['metric'] == m]
        mat = np.array([r['mean'] for r in sub]).reshape(2, 2)
        ax.imshow(mat, cmap='viridis', vmin=0, vmax=1)
        for (k, r) in enumerate(sub):
            ax.text(k % 2, k // 2, f"{r['mean']:.4f}\n[{r['ci_low']:.4f},{r['ci_high']:.4f}]", ha='center', va='center', color='black', fontsize=9)
        ax.set_xticks([0, 1], ['Matched', 'Mismatched'])
        ax.set_yticks([0, 1], ['External', 'Self Oracle'])
        ax.set_title(m)
    f.suptitle('C  Reference Source × Stain Style')
    f.tight_layout()
    f.savefig(fig / 'figure_C.png', dpi=220)
    f.savefig(fig / 'figure_C.pdf')
    plt.close(f)
    representative = ad.sort_values('dice_degradation', ascending=False).iloc[0]
    qid = representative.query_id
    seed = int(representative.seed)
    qs = representative.fixed_query_style
    ma = 'AA' if qs == 'A' else 'BB'
    mi = 'BA' if qs == 'A' else 'AB'
    cfg = yaml_load(OLD_PATH())
    entries = {}
    from pathology_trainfree.data import read_manifest
    for path in [cfg['dataset']['support_manifest'], cfg['dataset']['query_manifest']]:
        entries.update({r['id']: r for r in read_manifest(path, require_mask=True)})
    er = ex[(ex.query_id == qid) & (ex.seed == seed)]
    sr = sx[sx.query_id == qid]
    rid = er.iloc[0].reference_id
    panels = [('Raw Query', load_image(entries[qid]['image']), None), ('External Reference', load_image(entries[rid]['image']), None), ('Query A', load_image(OUT / 'cache/rgb/A' / f'{qid}.png'), None), ('Query B', load_image(OUT / 'cache/rgb/B' / f'{qid}.png'), None), ('GT', resize_mask(load_mask(entries[qid]['mask'], DC), (256, 256)), 'gray')]
    margins = []
    for (label, df, c) in [('External matched', er, ma), ('External mismatched', er, mi), ('Self matched', sr, ma), ('Self mismatched', sr, mi)]:
        r = df[df.condition == c].iloc[0]
        if r.status == 'VALID':
            with np.load(OUT / 'cases' / f'{r.case}.npz') as z:
                mm = z['margin']
                panels.append((label, mm > 0, 'gray'))
                margins.append((label + ' margin', mm, 'coolwarm'))
    panels += [('Reference A', load_image(OUT / 'cache/rgb/A' / f'{rid}.png'), None), ('Reference B', load_image(OUT / 'cache/rgb/B' / f'{rid}.png'), None)]
    panels += margins
    (f, axs) = plt.subplots(3, 5, figsize=(15, 9))
    for ax in axs.ravel():
        ax.axis('off')
    margin_limit=max(float(np.max(np.abs(item[1]))) for item in margins)
    for (ax, (label, img, cmap)) in zip(axs.ravel(), panels):
        ax.imshow(img, cmap=cmap, vmin=-margin_limit if cmap == 'coolwarm' else None, vmax=margin_limit if cmap == 'coolwarm' else None)
        ax.set_title(label, fontsize=9)
    f.suptitle(f'D  Maximum external Dice loss: {qid}, support {rid}; margin scale +/-{margin_limit:.3f}')
    f.tight_layout()
    f.savefig(fig / 'figure_D.png', dpi=180)
    f.savefig(fig / 'figure_D.pdf')
    plt.close(f)
    dump(OUT / 'representative_selection.json', representative.to_dict())
    (f, axs) = plt.subplots(1, 3, figsize=(12, 4))
    leak = []
    for (ax, m) in zip(axs, ['dice', 'auroc', 'unseeded_dice']):
        for (i, (label, df)) in enumerate([('No exclusion', no), ('Spatial exclusion', sx)]):
            vv = df.groupby('query_id')[m].mean() if m in df else pd.Series(dtype=float)
            (v, l, h) = ci(vv)
            ax.errorbar(i, v, yerr=[[v - l], [h - v]], fmt='o', capsize=4)
            leak.append({'metric': m, 'mode': label, 'mean': v, 'ci_low': l, 'ci_high': h, 'n_query': int(vv.notna().sum())})
        ax.set_xticks([0, 1], ['No exclusion', 'Spatial exclusion'])
        ax.set_title(m)
    f.suptitle('E  Self-Matching Leakage (four stain conditions averaged per query)')
    f.tight_layout()
    f.savefig(fig / 'figure_E.png', dpi=220)
    f.savefig(fig / 'figure_E.pdf')
    plt.close(f)
    table(OUT / 'self_matching_summary.csv', leak)
    leakage_paired=[]
    unseeded_summary=[]
    for metric in ['dice','auroc','unseeded_dice','unseeded_auroc','unseeded_fg_recall','unseeded_bg_specificity','unseeded_object_coverage','strictly_unseeded_object_coverage']:
        if metric not in sx or metric not in no:continue
        pp=sx.pivot(index='query_id',columns='condition',values=metric).dropna().mean(axis=1)
        nn=no.pivot(index='query_id',columns='condition',values=metric).dropna().mean(axis=1)
        delta=(nn-pp).dropna();mean,lo,hi=ci(delta)
        leakage_paired.append({'metric':metric,'contrast':'NoExclusion-SpatialExclusion','effect':mean,'ci_low':lo,'ci_high':hi,'n_query':len(delta)})
        if 'unseeded' in metric:
            for mode,vals in [('NoExclusion',nn),('SpatialExclusion',pp)]:
                mean,lo,hi=ci(vals);unseeded_summary.append({'metric':metric,'mode':mode,'mean':mean,'ci_low':lo,'ci_high':hi,'n_query':len(vals)})
    table(OUT/'self_matching_paired_effects.csv',leakage_paired)
    table(OUT/'unseeded_summary.csv',unseeded_summary)

    mechanism = []
    for ((source, mode, cls), fg) in pd.DataFrame(flips).groupby(['source', 'mode', 'gt_class']):
        for field in ['delta_correct_cos_mismatch_minus_match', 'delta_wrong_cos_mismatch_minus_match', 'delta_margin_mismatch_minus_match', 'error_flip_ratio_all_class', 'error_flip_ratio_correct_matched']:
            v = fg.groupby('query_id')[field].mean()
            (m, l, h) = ci(v)
            mechanism.append({'source': source, 'mode': mode, 'gt_class': cls, 'metric': field, 'effect': m, 'ci_low': l, 'ci_high': h, 'n_query': len(v)})
    table(OUT / 'retrieval_mechanism_effects.csv', mechanism)

    def effect(m, n):
        return sourceframe[(sourceframe.metric == m) & (sourceframe.effect_name == n)].iloc[0]
    de = effect('dice', 'delta_E')
    ds = effect('dice', 'delta_source')
    da = effect('auroc', 'delta_E')
    if len(common['dice']) < 30:
        outcome = 'Outcome D：有效配对或空间未播种评价受限，证据不足。'
    elif de.ci_low > 0 and da.ci_low > 0 and (ds.ci_low > 0):
        outcome = 'Outcome A：支持本受控设置下的 Stain Gap Motivation。'
    elif ds.ci_low > 0:
        outcome = 'Outcome B：支持更广义的 Target-specific Reference Motivation；染色不能解释全部来源差异。'
    elif ds.ci_high <= 0:
        outcome = 'Outcome C：未发现同图 Oracle 的明显正向优势。'
    else:
        outcome = 'Outcome D：当前区间不能稳定区分参考来源优势，证据不足。'
    qchecks = pd.read_csv(OUT / 'stain_quality_checks.csv')
    targ = json.loads((OUT / 'stain_templates.json').read_text())
    reg = json.loads((OUT / 'raw_regression.json').read_text())
    ff = pd.DataFrame(flips)
    range_summary=stain_range_summary()
    measured_gap = gap.groupby(['source', 'mode', 'condition'])[['lab_distance', 'he_distance', 'vector_angle_deg']].mean()
    text = ['# Reference Source × Stain Style — Self-Reference Motivation Validation', '\n## 1. Research Question & Motivation', '检验受控染色失配对冻结 UNI/N3-bil 检索的影响，以及同图 GT Oracle 的额外来源效应。假设未用于挑选图像、模板或强度。', '\n## 2. Experimental Protocol', f"CRAG 153 train / 40 official test，五组原始保存 Support：{meta['supports']}。External 1000 条（含200 Raw）；Self 160 SpatialExclusion +160 NoExclusion。冻结UNI FP32编码、FP16 tokens/余弦点积与Top50均值，1024维；448/224、offset 0/8；每类≤512；global PCG64 seed2026 FG先BG；occupancy FG≥.9/BG≤.1。复用原始bilinear margin重建（FP16 margin→448，窗口/offset均值→256），margin>0。", 'Stage 2 是 **Oracle Mechanism Diagnosis，不是无标注、可部署方法**。Query GT只参与Stage 2 memory和评价；不参与模板、强度、邻域、K或阈值选择。', '\n## 3. Stain Transformation Validity', f"Macenko 参数与完整153张训练RGB统计在 stain_templates.json。对标准化的 H/E 向量和log p99浓度描述做PCA，取训练PC1 25%/75%分位附近的真实图像；A={targ['A']['id']}，B={targ['B']['id']}。无GT/性能筛选。训练模板间距离：{gap_values(targ['A'], targ['B'])}。", f"RGB变换不改尺寸、坐标或GT；新增截断通道最大比例 {qchecks.get('new_clipped_channel_fraction', pd.Series([0])).max():.6f}，灰度x梯度相关性最小 {qchecks.gray_gradient_correlation.min():.4f}。逐图参数、hash及转换后统计均保存；视觉案例见Figure D。颜色可能改变结构可见性，不能以几何不变宣称病理外观完全无损。", '转换后训练观察范围核对：', markdown_table(range_summary,index=True), '各分量范围不代表联合分布包络，超界可能来自组织组成或归一化后的方差压缩；完整超界图像保留，不做筛选。', '实际配对染色距离（条件均值；逐对数据保留）：', markdown_table(measured_gap,index=True), '距离定义：组织像素（所有OD通道>.15）的33个1%–99%分位数，Lab三通道/HE两通道绝对差均值；HE方向对应向量夹角均值。均为未标准化的各自单位。浓度、Lab受组织组成和OD筛选影响，不等于纯技术性染色距离。同模板不保证分布相同。', '\n## 4. External Reference Stain Sensitivity', markdown_table(pd.DataFrame(summary)[lambda x: x.source == 'External'], index=False), '配对效应（ALL先每Query平均五Support，CI按Query重采样）：', markdown_table(pd.DataFrame(effects)[lambda x: x.support_seed == 'ALL'], index=False), '固定Query style的AA−BA、BB−AB与各Support的CI见 paired_effects.csv，任何方向均保留。', '\n## 5. Reference Source × Stain Style', markdown_table(pd.DataFrame(summary),index=False), markdown_table(pd.DataFrame(sources), index=False), 'delta_source包含形态覆盖、上下文及同图特征分布差异；delta_interaction描述失配敏感性差别，不能单独证明染色因果机制。所有Source比较取同Query四Self及五Support全部有效的共同子集。', '\n## 6. FG/BG Retrieval Mechanism', '本实验External失配下，纯FG正确类相似度约下降0.00596，错误类也下降约0.00479；纯BG正确类下降约0.00585，错误类下降约0.00506。正确类相似度的下降并未伴随错误类相似度上升。类竞争margin有小幅不利变化，但重建AUROC配对差区间跨0；故证据支持相似度会改变，不支持本次干预导致稳定的整体排序或分割退化。', '检索统计在纯FG/BG token上计算，Top50 class ranking是两类Top50均值竞争的正确率，token AUROC只描述token排序；主auroc为重建256图上的margin AUROC。统计推断单位始终是Query。', markdown_table(pd.DataFrame(mechanism), index=False), markdown_table(ff.groupby(['source', 'mode', 'gt_class'])[['delta_correct_cos_mismatch_minus_match', 'delta_wrong_cos_mismatch_minus_match', 'delta_margin_mismatch_minus_match', 'error_flip_ratio_all_class', 'error_flip_ratio_correct_matched']].mean()), '以上变化方向为 mismatched−matched；FG类margin下降为不利，BG类margin上升为不利。FG→BG/BG→FG错误翻转只计matched正确、mismatched错误；分母分别保存全部该类token和matched正确token。正类/错类cos变化、margin变化与AUROC必须合看：margin平移可能改变阈值分割而不破坏排序，Dice下降本身不能证明UNI表示失效。', '\n## 7. Self-Matching Control', '固定真实footprint边缘间隔16原图像素，重合、重叠和邻域内reference全部排除。反射padding映射回原图，offset/多窗口重复按实际空间判断。合成检查见self_matching_checks.csv。每token每类有效邻居数保存在cases/*.npz；不足50则整条件无效，不降K。', markdown_table(pd.DataFrame(leak), index=False), markdown_table(pd.DataFrame(leakage_paired),index=False), markdown_table(pd.DataFrame(unseeded_summary),index=False), 'Unseeded排除所有已选token真实区域及16px邻域，256像素只有整片区域未播种才计入。object coverage：可评价未播种部分≥50输出像素的GT连通腺体中，预测FG覆盖>50%的比例。另存strictly_unseeded_objects和strictly_unseeded_object_coverage，要求整个腺体没有任何已播种输出像素；若没有合法腺体则留空，不填0。详细召回、specificity、AUROC、Dice和有效像素数在self_oracle_results_per_image.csv。Full GT候选选择仍是Oracle，空间排除不能消除共享窗口上下文；密集512×2采样可能使未播种区域极少。', '\n## 8. Statistical Analysis', '固定10000次seed2026 Query cluster bootstrap；五External Support和全部条件随Query整体重采样。报告各Support单独Query CI、固定Query style对比及共同有效Source效应。没有token独立性假设。CI为描述性、未多重校正，40张且无患者ID，不能排除患者层级相关。', f'Raw FP32逐图与Memory回归：{reg}。历史0.808421±0.077639是五Support均值±样本SD，不是CI。历史FP32检索与任务FP16要求不同；主实验均固定FP16，历史数值偏差单独报告。', '\n## 9. Failure Cases', f'Figure D按External固定Query style配对Dice损失最大选例：{representative.to_dict()}。此选择仅用于解释真实失败，不用于效应估计；全部40张参与。', '\n## 10. Implications for Self-Reference Gland Segmentation', outcome, '结果只能为自动self-reference设计提供机制证据；未测试SLIC、形态学或无标注seed精度，不能推断真实无监督性能。', '\n## 11. Limitations', '两个训练RGB风格代表有限的受控技术干预；Macenko重建、浓度非负截断和组织组成变化共同影响外观。外部仅5张Support；同图GT Oracle可访问全图标签并有共享上下文。空间排除和未播种区域不等于独立切片或真实稀疏seed；来源效应不能全归因于染色。未运行GLaS复核。', f"有效条件：External {int((ex.status == 'VALID').sum())}/{len(ex)}，Self-excluded {int((sx.status == 'VALID').sum())}/{len(sx)}；共同有效Queries {len(common['dice'])}/40。", f"严格整颗未播种腺体仅{sx.loc[sx.strictly_unseeded_object_coverage.notna(), 'query_id'].nunique()}张Query可评价；该子分析样本不足，不能据其退化CI宣称等效或全腺体泛化。", '\n## 12. Final Conclusion', outcome, 'H1只得到相似度变化的部分支持，未得到稳定的整体检索/分割损害证据；H2中共享目标外观的设想不能据本次结果解释显著性能收益；H3在严格GT Oracle和本空间排除规则下有正向证据，但仍混合形态覆盖与共享上下文。', f'External Dice染色配对效应 {de.effect:.5f} [{de.ci_low:.5f},{de.ci_high:.5f}]；染色匹配后Self来源效应 {ds.effect:.5f} [{ds.ci_low:.5f},{ds.ci_high:.5f}]。不能根据该Oracle实验声称无标注self-reference已获验证。', '\n### English Introduction / Motivation', f'In a controlled CRAG study with frozen UNI features, five fixed external references, and all 40 official test images, stain matching changed Dice by {de.effect:.4f} (query-cluster bootstrap 95% CI {de.ci_low:.4f} to {de.ci_high:.4f}). With matched stain styles, a same-image ground-truth oracle reference under spatial exclusion changed Dice relative to external references by {ds.effect:.4f} (95% CI {ds.ci_low:.4f} to {ds.ci_high:.4f}). These observations quantify the sensitivity of non-parametric gland retrieval to controlled appearance changes and reference origin. The source contrast also includes morphology coverage and shared image context; it cannot be attributed exclusively to staining. Because the same-image reference uses ground-truth labels, these findings motivate further investigation of target-specific reference construction without establishing the performance of an unsupervised self-reference method.']
    (OUT / 'report.md').write_text('\n\n'.join(text))
    dump(OUT / 'analysis_completion.json', {'status': 'COMPLETE', 'outcome': outcome, 'common_queries': common, 'figures': list((str(p) for p in fig.glob('*')))})

def OLD_PATH():
    return ROOT / 'outputs/support_size_study_gtfixed/configs/CRAG/seed_14_shots_1.yaml'

def yaml_load(p):
    import yaml
    return yaml.safe_load(p.read_text())['config']

def gap_values(a, b):
    from reference_source_stain import gap
    return gap(a, b)


def markdown_table(frame,index=False):
 if index: frame=frame.reset_index()
 def cell(x):
  if isinstance(x,float):return f"{x:.6f}"
  return str(x).replace("|","/").replace("\n"," ")
 rows=[[cell(x) for x in frame.columns]]+[["---"]*len(frame.columns)]+[[cell(x) for x in row] for row in frame.itertuples(index=False,name=None)]
 return "\n".join("| "+" | ".join(row)+" |" for row in rows)


def augment_strict_unseeded_objects():
    # Additional evaluation only: glands with no seeded output pixels at all.
    from scipy import ndimage
    import cv2
    from pathology_trainfree.data import read_manifest
    cfg=yaml_load(OLD_PATH())
    entries={r['id']:r for r in read_manifest(cfg['dataset']['query_manifest'],require_mask=True)}
    path=OUT/'self_oracle_results_per_image.csv'
    rows=list(csv.DictReader(path.open()))
    for qid in sorted({r['query_id'] for r in rows}):
        mask=load_mask(entries[qid]['mask'],DC);seeded=np.zeros(mask.shape,bool)
        with np.load(OUT/'memories'/f'{qid}_A.npz') as z:coords=np.concatenate([z['footprints_0'],z['footprints_1']])
        for y,x,y1,x1 in coords:seeded[max(0,y-16):min(len(mask),y1+16),max(0,x-16):min(mask.shape[1],x1+16)]=True
        un=cv2.resize((~seeded).astype(np.float32),(256,256),interpolation=cv2.INTER_AREA)>.999999
        gt=resize_mask(mask,(256,256));labels,n=ndimage.label(gt,structure=np.ones((3,3)))
        glands=[labels==j for j in range(1,n+1) if (labels==j).sum()>=50 and un[labels==j].all()]
        folder=OUT/'unseeded_masks';folder.mkdir(exist_ok=True);np.save(folder/f'{qid}.npy',un)
        for r in rows:
            if r['query_id']!=qid:continue
            r['strictly_unseeded_objects']=len(glands)
            r['strictly_unseeded_object_coverage']=''
            if r['status']=='VALID' and glands:
                with np.load(OUT/'cases'/f"{r['case']}.npz") as z:pred=z['margin']>0
                r['strictly_unseeded_object_coverage']=float(np.mean([pred[g].mean()>.5 for g in glands]))
    table(path,rows)


def stain_range_summary():
    t=json.loads((OUT/'stain_templates.json').read_text());post=json.loads((OUT/'post_transform_statistics.json').read_text());rows=[]
    for key in ['lab_mean','lab_std','concentration_p99']:
        train=np.array([s[key] for s in t['train_statistics']]);lo=train.min(0);hi=train.max(0)
        for id,styles in post.items():
            for style,stats in styles.items():
                val=np.array(stats[key]);rows.append({'image_id':id,'style':style,'statistic':key,'within_observed_train_componentwise_range':bool(((val>=lo)&(val<=hi)).all()),'train_min':lo.tolist(),'train_max':hi.tolist(),'value':val.tolist()})
    table(OUT/'stain_range_checks.csv',rows)
    return pd.DataFrame(rows).groupby(['style','statistic']).within_observed_train_componentwise_range.agg(['sum','count'])

if __name__ == '__main__':
    report()
