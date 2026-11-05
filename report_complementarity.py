"""Chinese research report built from actual completed-run and diagnostic CSVs."""
import json
import numpy as np
import pandas as pd
from run_complementarity import OUT,MODELS,SEEDS,write_csv


def table(df,columns=None,digits=4):
    if columns:df=df[columns]
    def fmt(x):
        if isinstance(x,(float,np.floating)):return 'NA' if pd.isna(x) else f'{x:.{digits}f}'
        return str(x)
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))


def stats(values,percent=False):
    v=np.asarray(values,float)*(100 if percent else 1)
    return f'{v.mean():.4f} ± {v.std(ddof=1):.4f} [{v.min():.4f}, {v.max():.4f}]'


def main():
    d=OUT/'metrics';runs=pd.read_csv(d/'individual_models.csv');fusion=pd.concat([pd.read_csv(d/'majority_voting.csv'),pd.read_csv(d/'oracle_upper_bound.csv')])
    pairs=pd.read_csv(d/'pairwise_seed.csv');coverage=pd.read_csv(d/'correction_seed.csv');triple=pd.read_csv(d/'triple_seed.csv')
    boundary=pd.read_csv(d/'boundary_seed.csv');bc=pd.read_csv(d/'boundary_correction_seed.csv');bt=pd.read_csv(d/'boundary_triple_seed.csv')
    margins=pd.read_csv(d/'margin_diagnostics.csv');paired=pd.read_csv(d/'paired_dice.csv')
    allruns=pd.concat([runs,fusion]);parts=[]
    parts.append('# UNI / UNI2-h / Kaiko N3-bil 独立分割与互补性诊断\n\n30/30 个模型运行已完成；GLaS Official Test A+B 80张、CRAG Official Test40张；20-shot × Seeds [14,24,42,1337,3407]。主指标为每图指标均值，再在5 Seed上计算样本标准差（ddof=1）；不把重复Seed的测试图当作独立样本。数值为0–1，Object-HD单位为256网格像素。')
    parts.append('## A. Implementation\n\n新增统一 FeatureEncoder 适配 `uni | uni2_h | kaiko_vitb8`；保留 `encode` 的[B,N,C]接口，并提供 `extract_features` 的[B,H,W,C]接口。两个新模型使用与DUAL相同的timm架构/kwargs，直接从已缓存官方权重严格加载，避免运行时网络依赖；绝无随机权重替代。DUAL源码未修改，源码与权重SHA256见 `configs/source_audit.json` 和各模型Smoke JSON。\n\nUNI2-h真实前缀为1 CLS+8 Register，只保留1024个空间Token；Kaiko只去掉1 CLS。两者均使用timm动态位置编码插值处理448输入，没有把窗口缩到224。模型eval、requires_grad=False、torch.inference_mode；计算fp32、存储Token和重建前Margin经fp16，沿用基线。\n\n分割共用原有SegmentationPipeline、SimilarityRetriever、select_tokens与geometry模块：448窗口/224步长、Support和Query均Offset[0,8]（原图8像素）、FG≥.9/BG≤.1、每图每类最多512、全局PCG64 seed2026按Support顺序先FG后BG、Top50独立余弦均值、Bilinear align_corners=False、窗口均值/Offset均值、Native Margin缩到256后阈值>0。不相加跨模型原始Margin；无训练、校准、调参或分割后处理。\n\n新FeatureStore缓存指纹包含Encoder名称、权重Hash、预处理、Patch Size、层、图像、坐标、Offset、算法版本；Occupancy每次从当前Support真实GT重新计算并存GT Hash。历史UNI只迁移已校验特征，忽略原unlabeled占位Occupancy。禁用低纯度Fallback；所有实际Run为零Fallback，K没有降低。已验证历史UNI20-shot无Fallback，故该限制不改变UNI结果。\n\nSupport直接冻结修正后Study20-shot IDs及顺序；训练池为GLaS68、CRAG153。数据泄漏检查包括ID、路径、RGB文件Hash不相交；测试GT只进入预测之后的评价与诊断。可从逐图NPZ和模型缓存恢复。')
    specs=[]
    for m in MODELS:
        x=json.loads((OUT/f'configs/{m}_smoke.json').read_text())
        specs.append({'FM':m,'architecture':x['architecture'],'patch':x['patch_size'],'dim':x['feature_dim'],'grid':str(x['shape'][1:3]),'params_M':x['parameters']/1e6,'prefix':x['prefix_tokens'],'mean/std':'.5/.5' if m=='kaiko_vitb8' else 'ImageNet','encoder_peak_GiB_B1':x['peak_cuda_bytes']/2**30})
    parts.append('## B. Model Specifications\n\n'+table(pd.DataFrame(specs))+'\n\n均输入448×448原生空间RGB窗口并L2 Normalize。UNI/UNI2-h mean=[0.485,0.456,0.406]，std=[0.229,0.224,0.225]；Kaiko mean/std均为[0.5,0.5,0.5]。UNI2-h的缓存架构名实际为vit_giant_patch14_224，显式覆盖配置后参数数为681.394M；不能仅凭架构字符串或名称推断大小。UNI编码batch1；UNI2-h/Kaiko编码batch4，批量数值与范数验证见configs/*_batch_smoke.json。Smoke表峰值为单窗口编码的allocated CUDA内存，包含模型；不含桌面/驱动及reserved缓存。硬件RTX4090，环境见run_metadata.json。')
    costs=[]
    for (ds,m),g in runs.groupby(['dataset','model'],sort=False):
        cache=json.loads((OUT/f'configs/{m}_{ds}_cache.json').read_text())
        costs.append({'dataset':ds,'model':m,'support_cache_s_once':cache['support_seconds'],'query_cache_s_once':cache['query_seconds'],
                      'cached_infer_s_per_run':g.inference_total_seconds.mean(),'FG_tokens_range':f'{g.fg_tokens.min()}–{g.fg_tokens.max()}',
                      'BG_tokens_range':f'{g.bg_tokens.min()}–{g.bg_tokens.max()}','bank_MiB':g.memory_bytes.mean()/2**20,'retrieval_peak_GiB':g.peak_cuda_bytes.max()/2**30})
    parts.append(table(pd.DataFrame(costs))+'\n\n时间包含缓存I/O与校验；编码缓存每模型/数据集只算一次，5 Seed检索复用。UNI时间含历史特征迁移而非重新编码，不能作为三个编码器速度的直接排名；单Run时间为缓存后的检索与空间重建，评价另计。完整逐Seed时间/Memory/显存见individual_models.csv。')
    parts.append('## C. Independent Segmentation Results\n\n以下均为5 Seed Mean±SD [Min,Max]，数值不乘100。')
    parts.append('![三模型与Voting/Oracle对比](visualizations/independent_voting_oracle.png)')
    headline=[]
    for m in MODELS:
        r={'FM':m}
        for ds,label in [('GlaS','GLaS'),('CRAG','CRAG')]:
            g=runs[(runs.dataset==ds)&(runs.model==m)]
            r[label+' Dice']=stats(g.dice);r[label+' Object-F1']=stats(g.object_f1)
        headline.append(r)
    parts.append(table(pd.DataFrame(headline),['FM','GLaS Dice','CRAG Dice','GLaS Object-F1','CRAG Object-F1']))
    summary=[]
    for (ds,m),g in runs.groupby(['dataset','model'],sort=False):
        summary.append({'dataset':ds,'FM':m,**{k:stats(g[k]) for k in ['dice','fg_iou','object_f1','object_hausdorff','precision','recall','predicted_fg_ratio']}})
    parts.append(table(pd.DataFrame(summary)))
    parts.append('### 逐Seed数据\n\n'+table(runs,['dataset','model','seed','dice','fg_iou','precision','recall','object_f1','object_hausdorff','predicted_fg_ratio']))
    parts.append('### 同Seed相对UNI的配对Dice差值\n\n'+table(paired[paired.model!='uni'])+'\n\n逐图完整指标见individual_per_image.csv。Object-F1/HD沿用原框架256网格的二值连通域协议：8连通、评价时面积≥50、GT覆盖>0.5匹配；这不是原生实例标注的官方实例分割分数，也没有改变输出Mask。')
    parts.append('## D. Pairwise Complementarity\n\n每个图像/Seed先计算严格对齐的错误集合；每Seed累加分子分母，再报告Seed均值和SD。Error IoU分母为错误并集；a-only、b-only、shared、FP/FN比例分母为所有评价像素。空分母记NA，CSV为空，不填0。')
    pp=[]
    for keys,g in pairs.groupby(['dataset','model_a','model_b'],sort=False):
        pp.append(dict(zip(['dataset','A','B'],keys))|{k:stats(g[k]) for k in ['error_iou','a_only_error_ratio','b_only_error_ratio','shared_error_ratio','shared_fp_ratio','shared_fn_ratio']})
    parts.append(table(pd.DataFrame(pp)))
    parts.append('![方向性纠错覆盖率](visualizations/directional_correction.png)')
    cc=[]
    for keys,g in coverage.groupby(['dataset','model_a','model_b','tissue'],sort=False):
        cc.append(dict(zip(['dataset','A','B','GT_region'],keys))|{'Correction':stats(g.correction),'Damage':stats(g.damage)})
    parts.append('### 方向性纠错及新增错误代价\n\nCorrection(A→B)是在A错误中B正确的比例；Damage(A→B)是在A正确中B错误的比例，两个分母不同，不能直接相减。\n\n'+table(pd.DataFrame(cc)))
    parts.append('原始像素计数和归一化比例均保存在pairwise_errors.csv、correction_coverage.csv；逐Seed池化计数在pairwise_seed.csv、correction_seed.csv。每个条件保留dataset/seed/image_id，未跨图错误集合直接求交。')
    parts.append('## E. Triple Complementarity\n\n'+table(triple,['dataset','seed','all_correct','all_wrong','only_uni_correct','only_uni2_h_correct','only_kaiko_vitb8_correct','exactly_two_correct','exactly_one_correct','triple_shared_fp','triple_shared_fn','all_wrong_over_uni_error','oracle_correctable_uni_ratio']))
    ts=[]
    for ds,g in triple.groupby('dataset',sort=False):ts.append({'dataset':ds,**{k:stats(g[k]) for k in ['all_wrong_ratio','all_wrong_over_uni_error','oracle_correctable_uni_ratio','disagreement_ratio','disagreement_uni_accuracy','disagreement_uni2_h_accuracy','disagreement_kaiko_vitb8_accuracy','disagreement_majority_accuracy']}})
    parts.append(table(pd.DataFrame(ts))+'\n\n二值三模型有分歧时至少一个正确。上表仅说明存在可利用的正确预测；无法证明在无GT时能识别正确模型。所有共同错误与独有正确像素计数见triple_errors.csv。')
    parts.append('## F. Fusion Diagnostic\n\nMajority为三张二值Mask票数≥2，完全无需训练；Oracle在任一模型正确时选择GT，三者都错时保留共同错误标签。Oracle显式使用测试GT，只是潜在选择上限，不能部署，也不能声称为Training-Free性能。')
    ff=[]
    for (ds,m),g in allruns.groupby(['dataset','model'],sort=False):ff.append({'dataset':ds,'method':m,**{k:stats(g[k]) for k in ['dice','fg_iou','object_f1','object_hausdorff']}})
    parts.append(table(pd.DataFrame(ff)))
    parts.append('### 融合逐Seed与事后最佳单模型比较\n\n'+table(fusion,['dataset','seed','model','dice','fg_iou','object_f1','object_hausdorff','best_single_posthoc','delta_dice_vs_best_single','delta_dice_vs_uni'])+'\n\n最佳单模型仅用于事后比较，未用测试GT选择推理模型、权重或阈值。Oracle可纠正UNI错误比例见E节；它比单独Oracle Dice更直接地反映可选择空间。')
    parts.append('## G. Spatial vs Semantic Difference\n\n在统一256 GT上定义两侧相邻标签不同的像素为界面边界；欧氏距离≤4为Near Boundary，(4,8]为Transition，>8为Interior，空GT界面视为Interior。4和8像素带在查看结果前已固定，距离单位是256网格像素，不代表跨数据集相同物理微米尺度。以下为每区域各自分母的错误率及方向纠错率。')
    br=[]
    for keys,g in boundary.groupby(['dataset','region','model'],sort=False):br.append(dict(zip(['dataset','region','FM'],keys))|{'Error rate':stats(g.error_rate)})
    parts.append(table(pd.DataFrame(br)))
    cr=[]
    for keys,g in bc[bc.tissue=='overall'].groupby(['dataset','region','model_a','model_b'],sort=False):cr.append(dict(zip(['dataset','region','A','B'],keys))|{'Correction':stats(g.correction),'Damage':stats(g.damage)})
    parts.append(table(pd.DataFrame(cr)))
    tissue_rows=[]
    for keys,g in bc[(bc.region=='interior')&(bc.model_a=='uni')&(bc.tissue!='overall')].groupby(['dataset','model_b','tissue'],sort=False):
        tissue_rows.append(dict(zip(['dataset','correcting_FM','GT_interior_region'],keys))|{'Correction of UNI':stats(g.correction),'Damage to UNI':stats(g.damage),'UNI error pixels/seed mean':g.source_errors.mean()})
    parts.append('### 远离边界的前景/背景内部纠错\n\nGT前景指腺体语义区域，GT背景指非腺体区域；不细分无标注的组织亚型。\n\n'+table(pd.DataFrame(tissue_rows)))
    bs=[]
    for keys,g in bt.groupby(['dataset','region'],sort=False):bs.append(dict(zip(['dataset','region'],keys))|{'Triple shared error rate':stats(g.all_wrong_ratio),'shared pixels/seed mean':g.all_wrong.mean()})
    parts.append(table(pd.DataFrame(bs)))
    au=margins.groupby(['dataset','model','seed'],as_index=False).auroc.mean();ar=[]
    for keys,g in au.groupby(['dataset','model'],sort=False):ar.append(dict(zip(['dataset','FM'],keys))|{'Mean image AUROC':stats(g.auroc)})
    parts.append('### 模型自身Margin的符号与AUROC\n\n'+table(pd.DataFrame(ar))+'\n\nMargin从各自Token网格重建到统一空间后分析。不同模型余弦Margin数值不被视为同一尺度；AUROC仅为各自排序质量，不用于阈值/融合调参。完整Margin范围、符号错误率及AUROC在margin_diagnostics.csv。')
    parts.append('较小Patch Size、预训练语义、模型容量和训练数据同时变化，本实验是完整模型系统比较。Interior仍有纠错只能说明互补不局限于边界，无法隔离语义因果机制；边界纠错也不能单独证明Patch Size因果优势。诊断基于二值GT区域，没有组织亚型注释，不能可靠将内部纠错命名为特定病理组织成分。')
    parts.append('## 可视化与完整性\n\n案例选择在模型结果完成前冻结，固定seed14，每类按预定错误计数或image_id排序、类别间不重复；all_good要求三模型逐图Dice均≥0.90。[全部案例图索引](visualizations/index.md)、`visualizations/selection_policy.json`和selection_manifest.csv记录规则与所选样本；若某类别不足3张则如实披露，绝不替换Seed或制造案例。每张12-Panel含同一RGB/GT、三个Mask、Voting、两组UNI成对纠错/错误图、三重错误及三个独立色标Margin。\n\n'+table(pd.read_csv(OUT/'visualizations/category_counts.csv'))+'\n\nV1–V8实际验证与局限见validation_checks.md、acceptance_checks.json。UNI回归600个图像条件、七项指标与历史值相同，FG/BG Memory逐值完全一致；历史Study没有保存Mask/Margin，不将指标相同伪称为历史预测数组逐像素比较。')
    parts.append('## H. Research Conclusion\n\n')
    conclusion=OUT/'research_conclusion.md'
    if not conclusion.exists():raise RuntimeError('Actual-data research conclusion must be written before final report')
    parts.append(conclusion.read_text())
    parts.append('## 复现与文件索引\n\n在PathologyTrainFree目录、cg-seg环境下依次运行：\n\n```bash\npython scripts/smoke_encoders.py\npython scripts/check_encoder_batching.py\npython -m pytest -q tests\npython run_complementarity.py\npython analyze_complementarity.py\npython visualize_complementarity.py\npython verify_complementarity.py\npython report_complementarity.py\n```\n\n首次运行先完整UNI回归，再新模型。已完成图像在指纹一致时复用，Support/Query编码缓存分模型保存；Memory每次按固定种子重建并校验。所有逐图256 float32 Margin和Mask在predictions；Memory向量及逐图纯Token统计在memory；冻结Support、GT/编码器/配置指纹在configs；30个Run原始数据、配对差值与全部诊断CSV在metrics；执行日志execution_uni.log（UNI阶段）、execution.log（新模型阶段）、diagnostics.log、visualizations.log；源码来源source_audit.md。')
    (OUT/'final_report.md').write_text('\n\n'.join(parts),encoding='utf-8')
    meta=json.loads((OUT/'run_metadata.json').read_text());meta['status']='COMPLETE'
    from pathology_trainfree.encoder import sha256
    from run_complementarity import ROOT
    meta['final_code_sha256']={str(p.relative_to(ROOT)):sha256(p) for p in list((ROOT/'pathology_trainfree').glob('*.py'))+list(ROOT.glob('*complementarity.py'))}
    (OUT/'run_metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
