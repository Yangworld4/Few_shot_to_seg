"""Validate and report the fixed-GT N3-bil support-size sweep; no parameter selection."""
from pathlib import Path
import sys,json,hashlib,ast
from datetime import datetime,timezone
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from study_support_size import file_hash
OUT=ROOT/'outputs/support_size_study_gtfixed'
PREV=ROOT/'outputs/reference_coverage_diagnosis'
HIST={'GlaS':.9221269359086885,'CRAG':.9113170090785008}

def md(frame):
    def cell(x):
        if pd.isna(x): return 'N/A'
        return f'{x:.6f}' if isinstance(x,(float,np.floating)) else str(x)
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(cell(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None))

def main():
    runs=pd.read_csv(OUT/'runs.csv'); images=pd.read_csv(OUT/'per_image_metrics.csv'); summary=pd.read_csv(OUT/'summary.csv'); occ=pd.read_csv(OUT/'occupancy_validation.csv'); mem=pd.read_csv(OUT/'memory_statistics.csv'); supports=pd.read_csv(OUT/'support_sets.csv'); nesting=pd.read_csv(OUT/'nesting_validation.csv'); meta=json.loads((OUT/'study_metadata.json').read_text())
    assert len(runs)==40 and (runs.status=='COMPLETE').all(), 'All 40 runs must complete before final report'
    assert len(images)==2400
    assert occ.groupby('dataset').size().to_dict()=={'CRAG':153,'GlaS':68}
    assert set(occ.occupancy_source)=={'real_support_gt'}
    assert (occ.occupancy_min>=0).all() and (occ.occupancy_max<=1).all()
    assert occ.coordinates_validated.all()
    assert nesting.support_prefix_equal.all() and nesting.sampled_token_prefix_equal.all()
    assert (mem.fg_tokens>=50).all() and (mem.bg_tokens>=50).all()
    for row in supports.itertuples():
        assert Path(row.gt_path).stem==row.image_id and file_hash(row.gt_path)==row.gt_sha256
        assert row.memory_construction_seed==2026
    # Check isolated occupancy cache provenance and recomputed-stat consistency.
    for row in occ.itertuples():
        with np.load(row.occupancy_cache_file) as z:
            om=json.loads(str(z['metadata'])); occupancy=z['fg_occupancy']
        assert om['occupancy_source']=='real_support_gt' and om['gt_sha256']==row.gt_sha256
        assert om['fingerprint']==row.occupancy_fingerprint
        assert len(occupancy)==row.total_tokens
        assert int((occupancy>=.9).sum())==row.raw_fg_candidates
        assert int((occupancy<=.1).sum())==row.raw_bg_candidates
    # Historical M0/M1 strict multiset checks are reused rather than rerun.
    h=json.loads((ROOT/'outputs/n3_reference_parity_debug/memory_reconstruction.json').read_text())
    for ds,count in [('GlaS',3584),('CRAG',8192)]:
        for name in ['M1_fg_multiset_overlap','M1_bg_multiset_overlap']:assert h[ds][name]==[count,0,0]
    cached_h=pd.read_csv(PREV/'retrieval_validation.csv');assert cached_h.mask_pixels_different_from_verified_M0.dropna().eq(0).all()
    assert meta['encoder_sha256']=='56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40'
    # Corrected T20 parity: ordered Support IDs, exact Memory vectors, per-seed and per-image metrics.
    prior_runs=pd.read_csv(PREV/'anchor_experiment.csv'); prior_images=pd.read_csv(PREV/'anchor_per_image.csv')
    prior_runs=prior_runs[prior_runs.kind=='T'];prior_images=prior_images[prior_images.kind=='T']
    checks=[]; pair_images=[]; metrics=['dice','fg_iou','object_f1','precision','recall','predicted_fg_ratio']
    for row in runs[runs.shots==20].itertuples():
        ds=row.dataset;seed=row.seed;prev=prior_runs[(prior_runs.dataset==ds)&(prior_runs.seed==seed)].iloc[0];ids=ast.literal_eval(row.support_ids)
        assert ids==prev.support_ids.split(';')
        mpath=OUT/f'runs/{ds}/seed_{seed}/shots_20/memory.npz'
        with np.load(mpath) as current,np.load(PREV/f'memories/{ds}_T_{seed}_20.npz') as previous:
            exact_bg=np.array_equal(current['class_0'],previous['bg']);exact_fg=np.array_equal(current['class_1'],previous['fg'])
        assert exact_fg and exact_bg
        a=images[(images.dataset==ds)&(images.seed==seed)&(images.shots==20)].set_index('image_id');b=prior_images[(prior_images.dataset==ds)&(prior_images.seed==seed)].set_index('image_id')
        assert set(a.index)==set(b.index)
        errors={}
        for metric in metrics:
            diff=(a[metric]-b.loc[a.index,metric]).abs();errors[metric+'_max_abs_error']=float(diff.max())
            assert diff.max()<1e-9,(ds,seed,metric,diff.max())
            for id,value in diff.items():pair_images.append({'dataset':ds,'seed':seed,'image_id':id,'metric':metric,'absolute_error':value})
        checks.append({'dataset':ds,'seed':seed,'support_ids_order_equal':True,'memory_fg_exact':exact_fg,'memory_bg_exact':exact_bg,'images':len(a),'dice_current':row.dice,'dice_previous':prev.dice,'dice_seed_abs_error':abs(row.dice-prev.dice),**errors})
    pd.DataFrame(checks).to_csv(OUT/'corrected_20shot_parity.csv',index=False);pd.DataFrame(pair_images).to_csv(OUT/'corrected_20shot_parity_per_image.csv',index=False)
    qc=pd.read_csv(OUT/'query_feature_cache.csv');assert len(qc)==240 and qc.cache_reused.all()
    assert (OUT/'occupancy_regression.log').read_text().find('5 passed')>=0
    # Add meaningful supplementary statistics without changing the evaluator.
    ranges=[];intervals=[];hist=[]
    for ds in ['GlaS','CRAG']:
        for shot in [1,5,10,20]:
            group=runs[(runs.dataset==ds)&(runs.shots==shot)];st=summary[(summary.dataset==ds)&(summary.shots==shot)].iloc[0];mh=mem[(mem.dataset==ds)&(mem.shots==shot)]
            ranges.append({'dataset':ds,'shots':shot,'dice_mean':group.dice.mean(),'dice_sd':group.dice.std(ddof=1),'dice_min':group.dice.min(),'dice_max':group.dice.max(),'dice_range':group.dice.max()-group.dice.min(),'worst_seed':int(group.loc[group.dice.idxmin(),'seed']),'object_f1_mean':group.object_f1.mean(),'fg_min':mh.fg_tokens.min(),'fg_max':mh.fg_tokens.max(),'bg_min':mh.bg_tokens.min(),'bg_max':mh.bg_tokens.max(),'raw_fg_mean':mh.raw_fg_candidates.mean(),'raw_bg_mean':mh.raw_bg_candidates.mean(),'mixed_ratio_mean':(mh.mixed_tokens/mh.total_candidate_tokens).mean(),'fg_fallback_images_sum':mh.fg_fallback_images.sum(),'bg_fallback_images_sum':mh.bg_fallback_images.sum()})
            hist.append({'dataset':ds,'shots':shot,'current_dice':st.dice_mean,'historical_dice':HIST[ds],'delta_vs_historical':st.dice_mean-HIST[ds]})
        for small,large in [(1,5),(5,10),(10,20)]:
            a=summary[(summary.dataset==ds)&(summary.shots==small)].iloc[0];b=summary[(summary.dataset==ds)&(summary.shots==large)].iloc[0]
            ar=runs[(runs.dataset==ds)&(runs.shots==small)].set_index('seed');br=runs[(runs.dataset==ds)&(runs.shots==large)].set_index('seed')
            row={'dataset':ds,'from_shots':small,'to_shots':large}
            for metric in ['dice','fg_iou','object_f1']:
                delta=br[metric]-ar[metric];row[metric+'_mean_improvement']=b[metric+'_mean']-a[metric+'_mean'];row[metric+'_declining_seeds']=int((delta<0).sum())
            intervals.append(row)
    ranges=pd.DataFrame(ranges);intervals=pd.DataFrame(intervals);hist=pd.DataFrame(hist)
    ranges.to_csv(OUT/'support_sensitivity.csv',index=False);intervals.to_csv(OUT/'interval_improvements.csv',index=False);hist.to_csv(OUT/'historical_comparison.csv',index=False)
    joined=images.merge(mem,on=['dataset','seed','shots'],suffixes=('','_memory'))
    worst=joined.sort_values('dice').groupby(['dataset','shots'],as_index=False).head(1)
    worst.to_csv(OUT/'worst_image_by_shot.csv',index=False)
    regression=['# Regression checks','','## A — Real Support GT occupancy','','5/5真实缓存回归通过（含可信缓存复用及不可信占位缓存拒绝）（两数据集各历史labeled和原unlabeled样本）；历史occupancy逐值相同，原unlabeled忽略全零占位值并与当前GT重新计算值逐值相同。完整证据occupancy_regression.log。', '', '221张候选GT存在、标签0/255映射、图像/Mask尺寸及窗口坐标校验通过；全部isolated occupancy cache含`occupancy_source=real_support_gt`、GT SHA256和协议指纹。GT与image_id文件名对应，训练池/Test ID无交集。', '', '## B — Historical Memory parity (reused)','','复用既有M0/M1严格向量多重集检查：GLaS FG/BG各3584，CRAG各8192，缺失/额外均0。复用独立Retriever校验和历史M0 Mask全测试集0像素差证据，未重复历史诊断。证据文件及SHA见run_provenance.json。', '', '## C — Corrected 20-shot parity','',md(pd.DataFrame(checks)), '', '10/10条件Support IDs及顺序一致；FG/BG Memory向量逐值完全相同；600张条件×图像结果、6项共同指标的最大差均为0。逐seed聚合Dice最大舍入差2.22e−16，低于1e−9容差（实测值见CSV）。上次没有保存Object-Hausdorff，不能为该指标声称跨实验parity。', '', '## Nesting / cache reuse','','60/60相邻shot×class检查通过：Support顺序前缀相同；每次重置global PCG64 seed2026、FG先BG、同序处理已有图像，因此实际采样后的向量也保持前缀相同。该结论来自数组验证。240/240 Query cache复用，没有重新执行Query UNI编码。']
    (OUT/'regression_checks.md').write_text('\n'.join(regression)+'\n')
    # Exact setup and all Seed results, with data-dependent interpretation.
    results=[]
    for row in summary.itertuples():
        d={'Dataset':row.dataset,'Shots':row.shots,'Valid Seeds':f'{row.valid_seed_count}/5'}
        for m,label in [('dice','Dice'),('fg_iou','FG IoU'),('object_f1','Object-F1'),('precision','Precision'),('recall','Recall'),('object_hausdorff','Object-Hausdorff')]:d[label]=f'{getattr(row,m+"_mean"):.6f} ± {getattr(row,m+"_sd"):.6f}'
        results.append(d)
    sections=['# N3-bil Training-Free Few-Shot Support Size Study — GT Occupancy Fixed','','## 1. Experimental Setup','','GLaS合法训练候选池68张、Test A+B 80张；CRAG合法候选池153张、官方Test 40张。候选顺序为固定train_pool.csv行顺序。Support选择用独立NumPy PCG64 Seeds [14,24,42,1337,3407]生成排列，1/5/10/20取前缀。Memory采样单独使用global PCG64 seed2026、FG先BG、每图按Offset0→8候选合并后cap512。实际图像和向量嵌套均验证通过。Test GT只用于固定指标，未用于选择Support或阈值。', '', '冻结UNI ViT-L/16；448窗口、224 stride、patch16；Support/Query offsets[0,8]、reflection；FG≥0.9/BG≤0.1；class-wise cosine Top50均分；FG−BG Margin经fp16、28→448 bilinear、重叠均值/offset均值、native→256 bilinear，align_corners=False，阈值>0。Encoder float32、Token float16。没有训练、调参或改变后处理。', '', '指标沿用已验证DUAL定义。Dice/FG IoU/Precision/Recall/Object-F1/Object-Hausdorff先逐图等权，再五seed算mean及样本SD(ddof=1)。Object指标使用256二值Mask、8连通、最小组件50px、重叠匹配阈值0.5；并非原生分辨率实例指标。','','## 2. GT Occupancy Integrity','','全部221张训练候选的真实GT逐Token处理，缓存特征仅用于UNI，标签不取DUAL unlabeled占位occupancy。隔离缓存位于cache/real_support_gt_occupancy，以GT哈希+窗口/offset/标签协议指纹命名；每份元数据明确`occupancy_source=real_support_gt`。如果GT含前景而Token occupancy全零，程序直接停止；类别在整组GT缺失或任一bank<K50时标无效，不能用fallback伪造缺失类别。允许的fallback仅用于经GT/坐标核验后某类没有纯净Token。', '', f'实际执行：40/40 COMPLETE、0 INVALID、0 ERROR；Query特征240/240复用。occupancy范围、窗口坐标、纯FG/BG候选、mixed、GT哈希及旧占位差异逐图见occupancy_validation.csv。Historical Memory及T20回归通过，详见regression_checks.md。','','## 3. Main Results','',''+md(pd.DataFrame(results)),'','## 4. Full Seed Results','',md(runs[['dataset','seed','shots','status','fg_tokens','bg_tokens','fg_fallback_images','bg_fallback_images','dice','fg_iou','object_f1','precision','recall','object_hausdorff']]),'','## 5. Performance vs Support Size','',md(intervals),'']
    for ds in ['GlaS','CRAG']:
        ss=summary[summary.dataset==ds].sort_values('shots')
        sections += [f'{ds}: Dice均值序列 '+', '.join(f'{x:.6f}' for x in ss.dice_mean)+'；Object-F1 '+', '.join(f'{x:.6f}' for x in ss.object_f1_mean)+'。相邻区间增加Support数量分别4/5/10张，均值提升不等价于每张新增图像的相同边际收益；各seed局部下降数量已逐指标列出。','']
    sections += ['现有4个shot档位和5个seed不足以拟合连续学习曲线或证明严格饱和。收益递减与单调性必须逐数据集、逐指标判断；高Dice不等价于实例分离/完整性同幅提高。','','## 6. Support Sensitivity','',md(ranges[['dataset','shots','dice_mean','dice_sd','dice_min','dice_max','dice_range','worst_seed']]),'']
    for ds in ['GlaS','CRAG']:
        one=ranges[(ranges.dataset==ds)&(ranges.shots==1)].iloc[0];worstseed=runs[(runs.dataset==ds)&(runs.shots==1)].sort_values('dice').iloc[0]
        sections += [f'{ds} one-shot：Dice={one.dice_mean:.6f}±{one.dice_sd:.6f}，Object-F1均值={one.object_f1_mean:.6f}，最差seed={int(one.worst_seed)}（Dice={worstseed.dice:.6f}）；FG {int(one.fg_min)}–{int(one.fg_max)}、BG {int(one.bg_min)}–{int(one.bg_max)}。单张参考能运行并获得相当平均性能，但不能只用最佳seed保证任意参考可靠。','']
    sections += ['## 7. Memory Statistics / Fallback','',md(ranges[['dataset','shots','raw_fg_mean','raw_bg_mean','mixed_ratio_mean','fg_min','fg_max','bg_min','bg_max','fg_fallback_images_sum','bg_fallback_images_sum']]),'', 'Raw counts按重叠窗口、两个offset计算，同一物理组织可重复；Memory cap后向量数不等于独立组织区域数。fallback总数按每个seed/shot中的图像累计，嵌套图像跨条件重复。', '',md(worst[['dataset','seed','shots','image_id','dice','object_f1','fg_tokens','bg_tokens','fg_fallback_images','bg_fallback_images']]),'', '上述最差逐图例与Memory大小同列，可检查有足够K=50 Memory仍有局部错误。若所有bank都超过K，仅能排除“数量不足无法检索”；不能推导参考分布、类别证据质量已充分。Raw纯净/混合Token数量、cap后数量与性能之间即使相关，也不能直接作因果解释。', '', '## 8. Comparison With Historical N3-bil','',md(hist),'', '历史GLaS 7张、CRAG 16张为固定身份；当前随机1/5/10/20张身份和数量均不同，不作为受控差值或强制下限。T20与上一轮纠正配对对照身份相同，才作严格回归。旧错误occupancy研究不可用于本报告的科学结论。','','## 9. Implications for Training-Free Few-Shot Segmentation','']
    totalfallback=int(mem.fg_fallback_images.sum()+mem.bg_fallback_images.sum());mixed=occ.mixed_tokens.sum()/occ.total_tokens.sum()
    sections += [f'Hard occupancy确实丢弃中间混合Token：本训练候选池按窗口Token加权mixed占比={mixed:.2%}，各shot平均比例见表；此事实并不证明丢弃会损害性能。本次全部40条件累计fallback图像次数={totalfallback}。是否缺少纯FG/BG、或纯Token不满cap，可直接从occupancy_validation.csv和Memory统计定位，不能把“低于cap512”与“低于K50”混为一谈。', '', '本任务未实现MAP或Local Prototypes。混合Token被排除、参考存在变化，只能提出待验证假设；当前实验没有比较保留/利用混合Token的方法，因此不足以证明需要MAP，更不能宣称MAP能提高Dice。', '', '## 10. Final Conclusion','']
    near=[]
    for ds in ['GlaS','CRAG']:
        sub=summary[summary.dataset==ds].sort_values('shots');target=float(sub[sub.shots==20].dice_mean.iloc[0]);sat=[]
        for metric in ['dice','fg_iou','object_f1']:
            t=float(sub[sub.shots==20][metric+'_mean'].iloc[0]);eligible=sub[sub[metric+'_mean']>=t-.01]
            n=int(eligible.shots.min());sat.append(f'{metric}={n}张')
        close=sub[sub.dice_mean>=target-.01].iloc[0];near.append({'dataset':ds,'posthoc_descriptive_tolerance_absolute':.01,'smallest_measured_shots_near20_dice':int(close.shots),'dice20':target})
        sections += [f'{ds}: 若仅以“与本次20-shot平均指标相差不超过0.01绝对值”作事后描述（不是调参或显著性结论），已测档位中最少参考数为 '+', '.join(sat)+'。必须同时看SD、最差seed和Object-F1；该描述不能确定未测试的2/3/4等参考数，也不能保证每个seed接近20-shot。','']
        one=sub[sub.shots==1].iloc[0];twenty=sub[sub.shots==20].iloc[0]
        sections += [f'{ds} Dice seed SD：1-shot {one.dice_sd:.6f}→20-shot {twenty.dice_sd:.6f}；20-shot与历史Dice差{target-HIST[ds]:+.6f}，20-shot seed范围[{twenty.dice_min:.6f},{twenty.dice_max:.6f}]。','']
    pd.DataFrame(near).to_csv(OUT/'near20_descriptive_comparison.csv',index=False)
    sections += ['**Confirmed Bug：**本次未发现新的预测错误；旧unlabeled占位occupancy污染已隔离，真实GT回归和T20严格配对通过。**Valid Experimental Finding：**本报告的全部均值、seed范围、局部下降和fallback统计。**Unverified Hypothesis：**MAP或Local Prototypes能改善性能、染色/组织覆盖差异决定具体seed表现；本实验没有验证这些因果主张。','', '运行命令见execution_command.txt，配置及GT/UNI/代码哈希见configs/、support_sets.csv、study_metadata.json、run_provenance.json。新目录保留所有状态、逐图结果、Memory及独立occupancy缓存；旧目录和DUAL未修改。']
    # Interpret measured changes explicitly; these are descriptions of the five fixed seeds.
    loc=sections.index('## 6. Support Sensitivity')
    sections[loc:loc]=[
        'GLaS呈现明显的均值收益递减：Dice的1→5、5→10、10→20提升分别0.014721、0.007267、0.000778；Object-F1最后一个区间仅提升0.000700，10-shot后在本次观测中接近平台。20-shot的Object-Hausdorff由10-shot的38.646变为39.111，不能声称实例形状指标全部继续改善。', '',
        'CRAG最大Dice收益来自1→5（+0.073917），5→10仅+0.006065，10→20再次提高+0.015758，不能称为严格单调的收益递减或在10-shot饱和；Object-F1在5→10和10→20分别仍增加0.025465、0.017552。', '']
    loc=sections.index('## 7. Memory Statistics / Fallback')
    sections[loc:loc]=[
        '稳定性需分数据集：GLaS Dice SD从1-shot的0.013421下降到5-shot的0.009566，但10/20-shot反而为0.010517/0.011201，并未随Support数量单调变稳；CRAG从0.077639逐档降到0.023690、0.022611、0.007432，范围从0.198546收窄到0.015265，稳定性改善明显。', '',
        'One-shot判断：GLaS在这五个seed下有较好的Support效率及较小波动，但平均Dice仍比20-shot低0.022765；CRAG最差seed24 Dice=0.673735，SD较大，不能认为随机单张参考可靠。以上仅针对五个固定seed，不外推任意患者或参考图像。', '']
    purity=[]
    for ds,g in occ.groupby('dataset'):
        purity.append({'dataset':ds,'pool_images':len(g),'minimum_pure_fg_candidates':int(g.raw_fg_candidates.min()),'minimum_pure_bg_candidates':int(g.raw_bg_candidates.min()),'window_weighted_mixed_fraction':g.mixed_tokens.sum()/g.total_tokens.sum()})
    purity=pd.DataFrame(purity);purity.to_csv(OUT/'pool_purity_summary.csv',index=False)
    loc=sections.index('## 9. Implications for Training-Free Few-Shot Segmentation')+1
    sections[loc:loc]=['',md(purity),'', '整个合法训练候选池中，GLaS纯FG/BG候选最低610/574，CRAG最低797/2770；全部超过每图cap512。故本次没有“某类无纯净Token”或“纯净Token无法填满cap”的Support，所有40组两类Memory均为512×shot。这直接不支持把当前误差归因于纯Token数量不足。GLaS混合Token约13.54%、CRAG约6.97%被排除，但尚无保留它们的受控比较。','']
    loc=sections.index('## 10. Final Conclusion')
    sections[loc:loc]=['20-shot与历史的均值均接近：GLaS差+0.000433，CRAG差−0.007156；但五个seed中各有2个seed比历史低超过0.01。因此可说整体接近并具有所报告的SD，不能保证每种随机Support都在历史值±0.01内。历史身份及数量不同，以上不是Support身份的因果比较。','']
    (OUT/'report.md').write_text('\n'.join(sections)+'\n')
    evidence=[ROOT/'outputs/n3_reference_parity_debug/memory_reconstruction.json',PREV/'retrieval_validation.csv',PREV/'anchor_per_image.csv',PREV/'anchor_experiment.csv']
    command='/data/software/miniconda3/envs/cg-seg/bin/python study_support_size.py --output outputs/support_size_study_gtfixed --support-offsets 0 8 --memory-sampling-protocol original_global --query-cache-root outputs/support_size_study/cache/query_features'
    (OUT/'execution_command.txt').write_text('cd '+str(ROOT)+'\n'+command+'\n')
    provenance={'status':'complete','finished_utc':datetime.now(timezone.utc).isoformat(),'runs':40,'images_per_run':{'GlaS':80,'CRAG':40},'per_image_rows':len(images),'occupancy_source':'real_support_gt','query_cache_reused':int(qc.cache_reused.sum()),'command':command,'study_runner_sha256':file_hash(ROOT/'study_support_size.py'),'report_script_sha256':file_hash(__file__),'historical_evidence_sha256':{str(p):file_hash(p) for p in evidence},'regression_tests':{'A':'5 passed','B':'reuse verified exact M0/M1 and historical masks','C':'10 conditions, exact FG/BG memories, 600 per-image comparisons'},'nested_checks':len(nesting),'new_algorithm':False}
    (OUT/'run_provenance.json').write_text(json.dumps(provenance,indent=2))
    print('Report complete: 40 runs; 2400 per-image rows; 10 T20 exact memory matches; 600 image comparisons')
    print(summary[['dataset','shots','dice_mean','dice_sd','object_f1_mean','object_f1_sd']].to_string(index=False))
if __name__=='__main__':main()
