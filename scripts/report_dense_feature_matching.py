"""Paired image analysis and real-output figures for the locked E0-E3 experiment."""
from pathlib import Path
import sys,json,csv,time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
sys.path.insert(0,str(Path(__file__).resolve().parent))
from dense_feature_matching import OUT,ROOT,PROTO,FP,Study,table,dump,load_image,load_mask,resize_mask,DC,regions,edge,auc,windows
METRICS=['dice','fg_iou','object_f1','object_precision','object_recall','boundary_f1','boundary_iou','small_gland_recall','medium_gland_recall','large_gland_recall','merge_count','split_count','auroc','predicted_fg_ratio']

def ci(x):
 x=np.asarray(x,float);x=x[np.isfinite(x)]
 if not len(x):return np.nan,np.nan,np.nan
 r=np.random.default_rng(2026);b=x[r.integers(0,len(x),(10000,len(x)))].mean(1)
 return float(x.mean()),*np.quantile(b,[.025,.975]).tolist()
def md(df,index=False):
 if index:df=df.reset_index()
 def cell(x):return f'{x:.6f}' if isinstance(x,float) else str(x).replace('|','/').replace('\n',' ')
 return '\n'.join('| '+' | '.join(row)+' |' for row in [[cell(x) for x in df.columns],['---']*len(df.columns)]+[[cell(x) for x in row] for row in df.itertuples(index=False,name=None)])
def margin(ds,id,m):
 with np.load(OUT/'cases'/ds/f'{id}_{m}.npz') as z:return z['margin']
def savefig(f,name):
 f.tight_layout();f.savefig(OUT/'figures'/f'{name}.png',dpi=200);f.savefig(OUT/'figures'/f'{name}.pdf');plt.close(f)

def token_diagnostics(study):
 path=OUT/'token_retrieval_diagnostics.csv'
 if path.exists():return pd.read_csv(path)
 rows=[]
 for ds,state in study.data.items():
  for e in state['queries']:
   image,passes=study.cached(ds,e);mask=load_mask(e['mask'],DC);tm=[];oc=[]
   for (tokens,pos),offset in zip(passes,[0,8]):
    scores=state['retriever'].scores(tokens).numpy();tm.append(scores[:,1]-scores[:,0]);views=list(windows(mask,448,224,offset));assert np.array_equal(pos,[(y,x) for y,x,_ in views]);oc.append(np.array([v.reshape(28,16,28,16).mean((1,3)).ravel() for _,_,v in views]).ravel())
   tm=np.concatenate(tm);oc=np.concatenate(oc);fg=oc>=.9;bg=oc<=.1;pure=fg|bg
   rows.append({'dataset':ds,'image_id':e['id'],'tokens':len(tm),'pure_fg_tokens':int(fg.sum()),'pure_bg_tokens':int(bg.sum()),'mixed_tokens':int((~pure).sum()),'pure_fg_accuracy':float((tm[fg]>0).mean()) if fg.any() else None,'pure_bg_accuracy':float((tm[bg]<=0).mean()) if bg.any() else None,'pure_token_auroc':auc(oc[pure]>=.9,tm[pure]),'fg_margin_mean':float(tm[fg].mean()) if fg.any() else None,'bg_margin_mean':float(tm[bg].mean()) if bg.any() else None})
   p=OUT/'token_diagnostics'/ds/f'{e["id"]}.npz';p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,token_margin=tm,fg_occupancy=oc)
 table(path,rows);return pd.DataFrame(rows)

def report():
 (OUT/'figures').mkdir(parents=True,exist_ok=True);df=pd.read_csv(OUT/'per_image_metrics.csv');oracle=pd.read_csv(OUT/'oracle_reconstruction_metrics.csv');assert len(df)==480 and len(oracle)==120
 study=Study();summary=[];paired=[];perdelta=[];runtime=[];flips=[];margins=[];objects=[];boundary=[];failures=[]
 for ds,state in study.data.items():
  expected={e['id'] for e in state['queries']}
  for method in ['B0','B1','B2','B3']:
   sub=df[(df.dataset==ds)&(df.method==method)];assert set(sub.image_id)==expected
   for metric in METRICS:
    mean,lo,hi=ci(sub[metric]);summary.append({'dataset':ds,'method':method,'metric':metric,'mean':mean,'ci_low':lo,'ci_high':hi,'n_image':int(sub[metric].notna().sum())})
  for method,baseline in [('B1','B0'),('B2','B0'),('B3','B0'),('B2','B1'),('B3','B1')]:
   a=df[(df.dataset==ds)&(df.method==method)].set_index('image_id');b=df[(df.dataset==ds)&(df.method==baseline)].set_index('image_id')
   for metric in METRICS:
    delta=a[metric]-b[metric];mean,lo,hi=ci(delta);paired.append({'dataset':ds,'contrast':method+'-'+baseline,'metric':metric,'delta':mean,'ci_low':lo,'ci_high':hi,'n_image':int(delta.notna().sum())})
    for id,v in delta.items():perdelta.append({'dataset':ds,'image_id':id,'contrast':method+'-'+baseline,'metric':metric,'delta':v})
  base=df[(df.dataset==ds)&(df.method=='B0')].set_index('image_id')
  for method in ['B0','B1','B2','B3']:
   sub=df[(df.dataset==ds)&(df.method==method)].set_index('image_id')
   for id,row in sub.iterrows():
    runtime.append({'dataset':ds,'image_id':id,'method':method,'grid':{'B0':28,'B1':56,'B2':112,'B3':28}[method],'query_positions_per_window':{'B0':784,'B1':3136,'B2':12544,'B3':784}[method],**{k:row.get(k,0.) for k in ['cache_read_seconds','feature_interpolation_seconds','knn_seconds','reconstruction_seconds','guided_filter_seconds','incremental_refinement_seconds','total_query_seconds','peak_gpu_mib']},'relative_query_time':row.total_query_seconds/base.loc[id,'total_query_seconds'],'delta_dice':row.dice-base.loc[id,'dice'],'delta_object_f1':row.object_f1-base.loc[id,'object_f1']})
  for e in state['queries']:
   id=e['id'];gt=resize_mask(load_mask(e['mask'],DC),(256,256)).astype(bool);b0=margin(ds,id,'B0');p0=b0>0
   for method in ['B0','B1','B2','B3']:
    row=df[(df.dataset==ds)&(df.image_id==id)&(df.method==method)].iloc[0]
    objects.append({k:row[k] for k in ['dataset','image_id','method','object_f1','object_precision','object_recall','object_tp','pred_objects','gt_objects','small_glands','medium_glands','large_glands','small_gland_recall','medium_gland_recall','large_gland_recall','merge_count','split_count']});boundary.append({k:row[k] for k in ['dataset','image_id','method','boundary_f1','boundary_iou','boundary_precision','boundary_recall','boundary_fp','boundary_fn']})
    mm=margin(ds,id,method);pred=mm>0;delta=mm-b0
    mr={'dataset':ds,'image_id':id,'method':method,'delta_margin_mean':float(delta.mean()),'delta_margin_std':float(delta.std()),'sign_change_ratio':float((pred!=p0).mean()),'auroc':row.auroc,'predicted_fg_ratio':row.predicted_fg_ratio}
    for q in [1,5,25,50,75,95,99]:mr[f'delta_margin_q{q}']=float(np.percentile(delta,q))
    for cls,sel in [('fg',gt),('bg',~gt)]:
     mr[cls+'_margin_mean']=float(mm[sel].mean()) if sel.any() else None
     for q in [5,50,95]:mr[cls+f'_margin_q{q}']=float(np.percentile(mm[sel],q)) if sel.any() else None
    margins.append(mr)
    if method=='B0':continue
    for region,sel in regions(gt).items():
     flips.append({'dataset':ds,'image_id':id,'method':method,'region':region,'pixels':int(sel.sum()),'corrected_pixels':int(((p0!=gt)&(pred==gt)&sel).sum()),'damaged_pixels':int(((p0==gt)&(pred!=gt)&sel).sum()),'pred_fg_to_bg':int((p0&~pred&sel).sum()),'pred_bg_to_fg':int((~p0&pred&sel).sum()),'agree_both_wrong':int(((p0==pred)&(pred!=gt)&sel).sum()),'fg_corrected':int((gt&~p0&pred&sel).sum()),'bg_corrected':int((~gt&p0&~pred&sel).sum()),'fg_damaged':int((gt&p0&~pred&sel).sum()),'bg_damaged':int((~gt&~p0&pred&sel).sum())})
  for method in ['B1','B2','B3']:
   sub=df[(df.dataset==ds)&(df.method==method)].set_index('image_id');delta=sub.dice-base.dice
   for label,id in [('improvement',delta.idxmax()),('degradation',delta.idxmin())]:failures.append({'dataset':ds,'method':method,'case_type':label,'image_id':id,'delta_dice':delta.loc[id],'selection':'maximum/minimum per-image Dice delta; illustration only'})
 for name,rows in [('summary',summary),('paired_deltas',paired),('paired_deltas_per_image',perdelta),('runtime_memory',runtime),('decision_flip_analysis',flips),('boundary_analysis',boundary),('object_error_analysis',objects),('margin_analysis',margins),('failure_cases',failures)]:table(OUT/(name+'.csv'),rows)
 token=token_diagnostics(study)
 if not study.audit:
  for ds,state in study.data.items():
   for e in state['queries']:study.cached(ds,e)
 table(OUT/'feature_cache_audit.csv',study.audit)
 summ=pd.DataFrame(summary);pair=pd.DataFrame(paired);rt=pd.DataFrame(runtime);flip=pd.DataFrame(flips);failure=pd.DataFrame(failures)
 runtime_summary=rt.groupby(['dataset','method'])[['feature_interpolation_seconds','knn_seconds','reconstruction_seconds','guided_filter_seconds','total_query_seconds','relative_query_time','peak_gpu_mib']].agg(['mean','median','max']);runtime_summary.to_csv(OUT/'runtime_summary.csv')
 # Figure 1: image bootstrap intervals, no pixel-level hypothesis tests.
 f,axs=plt.subplots(2,3,figsize=(12,7))
 for row,ds in enumerate(study.data):
  for ax,metric in zip(axs[row],['dice','object_f1','boundary_f1']):
   sub=summ[(summ.dataset==ds)&(summ.metric==metric)].set_index('method').loc[['B0','B1','B2','B3']];x=np.arange(4);ax.errorbar(x,sub['mean'],yerr=np.array([sub['mean']-sub.ci_low,sub.ci_high-sub['mean']]),fmt='o',capsize=4);ax.set_xticks(x,['B0','B1','B2','B3']);ax.set_title(ds+' '+metric);ax.grid(alpha=.2)
 f.suptitle('1  Frozen UNI: method comparison (image bootstrap 95% CI)');savefig(f,'figure_1')
 f,axs=plt.subplots(2,2,figsize=(10,7))
 for row,ds in enumerate(study.data):
  means=df[(df.dataset==ds)&df.method.isin(['B0','B1','B2'])].groupby('method').mean(numeric_only=True).loc[['B0','B1','B2']]
  for metric in ['dice','object_f1','boundary_f1']:axs[row,0].plot([28,56,112],means[metric],'o-',label=metric)
  axs[row,0].set(title=ds+' performance',xlabel='Feature grid side');axs[row,0].legend(fontsize=8)
  rr=rt[(rt.dataset==ds)&rt.method.isin(['B0','B1','B2'])].groupby('method').mean(numeric_only=True).loc[['B0','B1','B2']];axs[row,1].plot([784,3136,12544],rr.total_query_seconds,'o-');axs[row,1].set(xscale='log',xlabel='KNN query positions/window',ylabel='Cached-feature query time (s)',title=ds+' measured cost')
 f.suptitle('2  Dense matching accuracy and cost');savefig(f,'figure_2')
 f,axs=plt.subplots(1,2,figsize=(12,4))
 for ax,ds in zip(axs,study.data):
  sub=flip[(flip.dataset==ds)&(flip.region!='all')];names=['boundary_0_2','boundary_2_5','interior_gt_5'];x=np.arange(3)
  for i,m in enumerate(['B1','B2','B3']):
   g=sub[sub.method==m].groupby('region')[['corrected_pixels','damaged_pixels']].sum().reindex(names);ax.bar(x+(i-1)*.24,g.corrected_pixels,width=.22,label=m+' corrected');ax.bar(x+(i-1)*.24,-g.damaged_pixels,width=.22,alpha=.45,label=m+' damaged')
  ax.axhline(0,c='black',lw=.7);ax.set_xticks(x,['0-2 px','2-5 px','>5 px']);ax.set_title(ds);ax.set_ylabel('Actual pixel counts (+ corrected / - damaged)');ax.legend(fontsize=7,ncol=2)
 f.suptitle('3  Spatial distribution of decision flips');savefig(f,'figure_3')
 # Representative cases selected after evaluation, never a performance subset.
 selected=failure[failure.method=='B1'];f,axs=plt.subplots(len(selected),4,figsize=(12,3*len(selected)))
 for row,sel in enumerate(selected.itertuples()):
  e=next(e for e in study.data[sel.dataset]['queries'] if e['id']==sel.image_id);rgb=cv2.resize(load_image(e['image']),(256,256));gt=resize_mask(load_mask(e['mask'],DC),(256,256)).astype(bool)
  for ax,m in zip(axs[row],['B0','B1','B2','B3']):
   overlay=rgb.copy();overlay[edge(gt)]=[0,255,0];overlay[edge(margin(sel.dataset,sel.image_id,m)>0)]=[255,0,0];ax.imshow(overlay);ax.set_title(f'{sel.dataset} {sel.image_id} {m}',fontsize=9);ax.axis('off')
 f.suptitle('4  Boundaries: GT green, prediction red; B1 best/worst Dice examples');savefig(f,'figure_4')
 for sel in failure.itertuples():
  e=next(e for e in study.data[sel.dataset]['queries'] if e['id']==sel.image_id);rgb=cv2.resize(load_image(e['image']),(256,256));gt=resize_mask(load_mask(e['mask'],DC),(256,256)).astype(bool);maps=[margin(sel.dataset,sel.image_id,m) for m in ['B0','B1','B2','B3']];p0=maps[0]>0;bound=max(float(np.max(abs(m))) for m in maps)
  panels=[('RGB',rgb,None),('GT',gt,'gray')]+[(m+' margin',a,'coolwarm') for m,a in zip(['B0','B1','B2','B3'],maps)]+[(m+' mask',a>0,'gray') for m,a in zip(['B0','B1','B2','B3'],maps)]
  for m,a in zip(['B1','B2','B3'],maps[1:]):
   p=a>0;overlay=(rgb.astype(float)*.5).astype(np.uint8);overlay[(p0!=gt)&(p==gt)]=[0,255,0];overlay[(p0==gt)&(p!=gt)]=[255,0,0];panels.append((m+' corrected green / damaged red',overlay,None))
  f,axs=plt.subplots(3,5,figsize=(14,8))
  for ax in axs.ravel():ax.axis('off')
  for ax,(title,a,cmap) in zip(axs.ravel(),panels):ax.imshow(a,cmap=cmap,vmin=-bound if cmap=='coolwarm' else None,vmax=bound if cmap=='coolwarm' else None);ax.set_title(title,fontsize=8)
  f.suptitle(f'5  {sel.dataset} {sel.image_id} {sel.method} {sel.case_type}; shared margin scale +/-{bound:.3f}');savefig(f,f'figure_5_{sel.dataset}_{sel.case_type}' if sel.method=='B1' else f'figure_5_{sel.dataset}_{sel.method}_{sel.case_type}')
 # Plain-language Chinese research report, all claims linked to actual paired values.
 key=['dice','object_f1','boundary_f1','boundary_iou','small_gland_recall'];lines=['# E0–E3：Frozen UNI Dense Matching 与空间重建诊断','本报告仅使用原始labeled10 Seed1337，GLaS 7张Support/80 Test，CRAG 16张Support/40 Test。全部条件共享历史Memory、冻结UNI、Top50、阈值0和相同测试图像。没有训练、梯度优化或测试超参数搜索。','## 协议与回归',(OUT/'baseline_parity.md').read_text(),'448 RGB → 缓存F28（1024通道）→ B0 KNN28 / B1 bilinear F56+L2+KNN56 / B2 bilinear F112+L2+KNN112 → FP16 margin舍入→bilinear448→原窗口与offset均值→bilinear256→阈值0。B3复用B0连续Margin256，经256灰度Guide、r=4、eps=.001局部线性Guided Filter后再阈值。align_corners=False，反射padding。','计时起点为已存在F28缓存：包括缓存读取、特征插值、检索、重建；不含历史UNI提取、GT评价和结果文件写入。B3总时间=B0测量时间+独立测量的读取RGB/连续Margin与Guided Filter增量。显存为同一环境中PyTorch峰值allocated，包含两数据集常驻Memory，不含已离线的UNI模型；不是端到端部署延迟或整机显存。','## 全量主指标',md(summ[summ.metric.isin(key)]),'## 配对比较与统计',md(pair[pair.metric.isin(key)]),'每数据集10000次seed2026图像配对Bootstrap，CI未做多重比较校正；小腺体指标只在具有该面积分层GT对象的同图配对上计算，n_image明确报告。不能把像素/Token当独立统计样本；单Support Seed不能推断跨Seed稳定性。','## Q1：原始空间错误在哪里？']
 choices=[]
 for ds in study.data:
  base=df[(df.dataset==ds)&(df.method=='B0')];err=base.all_errors.sum();band=base.boundary_0_2_errors.sum();near=base.boundary_2_5_errors.sum();inter=base.interior_gt_5_errors.sum();orows=oracle[oracle.dataset==ds];tt=token[token.dataset==ds]
  lines += [f'### {ds}',f'B0累计错误像素{int(err)}；边界0–2px {int(band)}（{band/max(1,err):.1%}），2–5px {int(near)}（{near/max(1,err):.1%}），内部>5px {int(inter)}（{inter/max(1,err):.1%}）。这些是错误构成比例，不是显著性检验；各区域面积和FG/BG错误数在e0_error_diagnostics.csv。',f'B0 FG→BG {int(base.fn.sum())}，BG→FG {int(base.fp.sum())}；Merge计数总和{int(base.merge_count.sum())}，Split计数总和{int(base.split_count.sum())}。',md(tt[['pure_fg_accuracy','pure_bg_accuracy','pure_token_auroc']].agg(['mean','min','max']),index=True),f'GT occupancy重建诊断平均Dice={orows.dice.mean():.6f}、Boundary-F1={orows.boundary_f1.mean():.6f}。**这是粗网格类别占比的空间重建诊断，不是模型真实性能上界，也不是UNI语义Oracle。** 纯Token仍有分类错误且部分像素位于内部，不能把所有B0错误都归因于双线性空间恢复。']
 lines+=['## Q2–Q3：B1优于B0吗？B2进一步改善吗？']
 for ds in study.data:
  lines.append('### '+ds)
  for contrast in ['B1-B0','B2-B0','B2-B1','B3-B0']:
   sub=pair[(pair.dataset==ds)&(pair.contrast==contrast)];d=sub[sub.metric=='dice'].iloc[0];o=sub[sub.metric=='object_f1'].iloc[0];b=sub[sub.metric=='boundary_f1'].iloc[0]
   interpretation='Dice均值增加且图像CI高于0' if d.ci_low>0 else 'Dice均值下降且图像CI低于0' if d.ci_high<0 else 'Dice配对区间跨0，不能确认稳定增益'
   lines.append(f'{contrast}: {interpretation}；ΔDice={d.delta:.6f} [{d.ci_low:.6f},{d.ci_high:.6f}]；ΔObject-F1={o.delta:.6f}；ΔBoundary-F1={b.delta:.6f}。')
   if d.delta>0 and o.delta<0:lines.append('像素Dice上升而Object-F1下降，不能称为实例分割改善。对应结构变化如下（方法−对照）：'+', '.join(f"{metric}={sub[sub.metric==metric].iloc[0].delta:+.6f}" for metric in ['merge_count','split_count','object_precision','object_recall','small_gland_recall'])+'。Merge/Split增加提示连通对象的合并/碎裂；Precision/Recall下降提示新增错误对象或GT对象匹配损失。这些是语义连通对象诊断，不是独立实例标签的病理判读。')
  sub=flip[(flip.dataset==ds)&(flip.region=='all')];lines.append(md(sub.groupby('method')[['corrected_pixels','damaged_pixels','pred_fg_to_bg','pred_bg_to_fg','agree_both_wrong']].sum(),index=True))
 lines+=['## Q4：B3的成本和边界收益',md(rt.groupby(['dataset','method'])[['total_query_seconds','relative_query_time','peak_gpu_mib']].mean(),index=True),'runtime_memory.csv保存逐图插值、KNN、重建、Guided Filter时间；runtime_summary.csv另存均值、中位数和最大值。B1/B2的KNN位置数精确为B0的4/16倍，但总耗时比由真实测量决定。B3复用B0连续Margin，绝不滤波二值Mask。','## Q5：收益是否集中在边界？',md(flip[flip.region!='all'].groupby(['dataset','method','region'])[['pixels','corrected_pixels','damaged_pixels']].sum(),index=True),'同时查看纠正与破坏，不能只报纠正像素。Boundary-F1/IoU、Boundary FP/FN、Merge/Split和Small Recall均有逐图保存。对象定义为256语义Mask中面积≥50的8连通分量，沿用历史Hungarian GT覆盖>0.5规则；这不等于拥有独立instance标签的所有接触腺体。Merge/Split使用交集≥10像素且交集/min对象面积≥0.1的关联图，计度数>1节点。小/中/大阈值固定50–255、256–1023、≥1024。边界F1欧氏容差2px；Boundary IoU为FG内距离背景≤2px带的IoU（包括图像外部背景）。','## Margin与检索机制','所有分数比较在相同256输出物理网格上进行；不同密度Token索引没有被当成同一位置。margin_analysis.csv记录差值分布、符号变化、FG/BG分位数和AUROC。插值和TopK是非交换操作：即使B1/B2更平滑或Margin变大，也不能据此自动推断语义或实例改善。','## Q6：是否继续增加复杂度？']

 for ds in study.data:
  for method in ['B1','B2','B3']:
   sf=flip[(flip.dataset==ds)&(flip.method==method)].groupby('region')[['pixels','corrected_pixels','damaged_pixels']].sum()
   vals=[]
   for region in ['boundary_0_2','boundary_2_5','interior_gt_5']:
    rr=sf.loc[region];net=int(rr.corrected_pixels-rr.damaged_pixels);rate=1000*net/max(1,rr.pixels);vals.append(f'{region}: 净纠正{net}，每千区域像素净纠正{rate:.3f}')
   # Insert before the margin section to keep Q5 findings with the corresponding analysis.
   ix=lines.index('## Margin与检索机制');lines.insert(ix,ds+' '+method+'：'+'；'.join(vals)+'。净纠正为实际纠正−破坏像素，需同时结合区域面积与Boundary-F1判断是否集中在边界。')

 for ds in study.data:
  means=df[df.dataset==ds].groupby('method').mean(numeric_only=True);b=means.loc['B0'];v=means.loc['B3'];ix=lines.index('## Q5：收益是否集中在边界？')
  lines.insert(ix,f'{ds} B3结构诊断：平均预测对象数从{b.pred_objects:.3f}降至{v.pred_objects:.3f}，Merge从{b.merge_count:.3f}变为{v.merge_count:.3f}，Split从{b.split_count:.3f}变为{v.split_count:.3f}，Object Recall从{b.object_recall:.4f}变为{v.object_recall:.4f}，Small Gland Recall从{b.small_gland_recall:.4f}变为{v.small_gland_recall:.4f}，Boundary FN从{b.boundary_fn:.3f}变为{v.boundary_fn:.3f}。对象减少、合并及漏边界/小腺体匹配损失共同表明该固定细化并未实现低成本边界改善；不能用平滑外观代替结构收益。')
 # Conservative joint-dataset progression rule: gains must survive paired image intervals and avoid object degradation.
 eligible=[]
 for method in ['B1','B2','B3']:
  vals=pair[pair.contrast==method+'-B0'];ok=all(vals[(vals.dataset==ds)&(vals.metric=='dice')].iloc[0].ci_low>0 and vals[(vals.dataset==ds)&(vals.metric=='object_f1')].iloc[0].delta>=0 and vals[(vals.dataset==ds)&(vals.metric=='boundary_f1')].iloc[0].delta>=0 for ds in study.data)
  if ok:eligible.append(method)
 if 'B3' in eligible:choice='Guided Margin Refinement';why='固定Guided Filter在两个数据集均显示正向配对Dice证据，且没有平均Object-F1/Boundary-F1退化；应优先验证低成本连续Margin细化。'
 elif eligible:choice='Dense Feature Matching';why='特征先插值的某个固定密度在两个数据集均显示正向配对Dice证据，且没有平均对象/边界退化；应在独立数据上验证成本效益。'
 else:choice='当前证据不足，不继续增加复杂度';why='本轮没有一种新增方案同时满足两数据集Dice配对区间高于0且对象/边界均值不下降的保守推进标准；不据测试结果追加新的网络或梯度优化。'
 lines += [f'选择：**{choice}**。{why}','最有价值的下一项实验：在独立验证数据上，预先锁定本轮固定方法、相同Memory/阈值与计时范围，复核像素、边界、对象及运行成本的配对收益。该建议由本轮Test观察产生，属于探索性研究；不是跨患者、跨数据集或跨Support Seed泛化结论。','## 成功、失败案例与限制',md(failure),'Figure 4按B1逐图Dice最大/最小差选择；Figure 5另完整展示B1/B2/B3各自最大/最小差案例，均只用于说明，全部IDs保存在failure_cases.csv，全部固定Test图像参与统计。GT仅进入事后评价及独立ORACLE，不进入B1/B2/B3推理。模型、阈值、滤波半径、尺度和Memory没有按测试表现调整。缓存式计时不支持端到端部署速度的断言；二值连通对象、面积分层、边界带定义和单Seed限制都影响解释。','## 复现','运行 `OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /data/software/miniconda3/envs/cg-seg/bin/python scripts/dense_feature_matching.py`；仅重建分析 `... scripts/dense_feature_matching.py --report-only`。按图/方法原子保存并断点恢复；协议指纹不符即拒绝。输入依赖和实际Memory SHA256、Support IDs、Test IDs及cache指纹分别在configs和feature_cache_audit.csv。']
 (OUT/'report.md').write_text('\n\n'.join(lines));dump(OUT/'analysis_completion.json',{'status':'COMPLETE','cases':len(df),'oracle_cases':len(oracle),'next_direction':choice,'figures':[p.name for p in (OUT/'figures').glob('*.png')]})
if __name__=='__main__':report()
