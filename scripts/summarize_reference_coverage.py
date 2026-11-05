"""Aggregate completed per-query diagnosis artifacts; no model or threshold selection."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parent))
from reference_coverage_diagnosis import OUT,ROOT,setup,table,auc,PROTO,FP,CODE,NS,SEEDS

def main():
 assert (OUT/'inference_complete.json').exists()
 allmeta=[]; frames={k:[] for k in ['metrics','interference','coverage','attribution','contribution']}; dsdata={}
 for ds in ['GlaS','CRAG']:
  meta=pd.read_csv(OUT/f'{ds}_support_sets.csv');allmeta.append(meta); images=[];gts=[];conds=None
  for marker in sorted((OUT/'queries'/ds).glob('*/complete.json')):
   folder=marker.parent
   for kind in frames:
    f=folder/f'{kind}.csv'
    if f.exists():
     df=pd.read_csv(f);df['dataset']=ds;df['image_id']=folder.name;frames[kind].append(df)
   with np.load(folder/'margins.npz') as z:images.append(z['margins']);gts.append(z['gt']);conds=z['conditions'].tolist()
  dsdata[ds]=(np.stack(images),np.stack(gts),conds)
 meta=pd.concat(allmeta,ignore_index=True);meta.to_csv(OUT/'support_sets.csv',index=False)
 frames={k:pd.concat(v,ignore_index=True) for k,v in frames.items()}
 def decorate(df):
  return df.merge(meta,on=['dataset','condition'],how='left')
 per=frames['metrics']; decorate(per).to_csv(OUT/'anchor_per_image.csv',index=False)
 keys=['dice','fg_iou','object_f1','precision','recall','predicted_fg_ratio','gt_fg_ratio','margin_mean','margin_q5','margin_q50','margin_q95','corrected_pixels','harmed_pixels']
 runs=per.groupby(['dataset','condition'])[keys].mean().reset_index(); runs=decorate(runs);runs.to_csv(OUT/'anchor_experiment.csv',index=False)
 summary=runs.groupby(['dataset','kind','additional'])[keys].agg(['mean','std']).reset_index();summary.columns=['_'.join(c).strip('_') for c in summary.columns];summary.to_csv(OUT/'anchor_summary.csv',index=False)
 inter=frames['interference'];decorate(inter).to_csv(OUT/'memory_interference.csv',index=False)
 flips=inter.copy()
 for num,den in [('harmed','correct_before'),('corrected','tokens'),('positive_to_negative','tokens'),('negative_to_positive','tokens')]:flips[num+'_ratio']=flips[num]/flips[den].replace(0,np.nan)
 decorate(flips).to_csv(OUT/'margin_flip_analysis.csv',index=False)
 cover=frames['coverage'];decorate(cover).to_csv(OUT/'reference_coverage.csv',index=False)
 attrs=frames['attribution'];attrs=attrs.groupby(['dataset','condition','class','stratum','source_image_id','historical_source'],as_index=False).neighbor_count.sum();attrs['neighbor_fraction']=attrs.neighbor_count/attrs.groupby(['dataset','condition','class','stratum']).neighbor_count.transform('sum');decorate(attrs).to_csv(OUT/'topk_source_attribution.csv',index=False)
 contrib=frames['contribution'].groupby(['dataset','condition','class','source_image_id'],as_index=False)[['top1_count','sole_source_top50_tokens']].sum();decorate(contrib).to_csv(OUT/'reference_contribution.csv',index=False)
 disc=[]; distdir=OUT/'margin_distribution_plots';distdir.mkdir(exist_ok=True)
 for ds,(margins,gts,conds) in dsdata.items():
  gt=gts.ravel().astype(bool)
  for ix,co in enumerate(conds):
   x=margins[:,ix].ravel();fg=x[gt];bg=x[~gt]; fgh,bins=np.histogram(fg,bins=200,range=(-1,1),density=False); bgh,_=np.histogram(bg,bins=bins,density=False);overlap=float(np.minimum(fgh/fgh.sum(),bgh/bgh.sum()).sum())
   sub=per[(per.dataset==ds)&(per.condition==co)]
   disc.append({'dataset':ds,'condition':co,'macro_per_image_auroc':sub.auroc.mean(),'pooled_pixel_auroc':auc(gt,x),'auroc_valid_images':sub.auroc.notna().sum(),'auroc_na_images':sub.auroc.isna().sum(),'fixed_zero_dice':sub.dice.mean(),'predicted_fg_ratio':sub.predicted_fg_ratio.mean(),'pooled_fg_mean':fg.mean(),'pooled_fg_median':np.median(fg),'pooled_bg_mean':bg.mean(),'pooled_bg_median':np.median(bg),'histogram_overlap_200_bins_minus1_1':overlap})
   fig,ax=plt.subplots(figsize=(5,3)); centers=(bins[:-1]+bins[1:])/2;ax.plot(centers,fgh/fgh.sum(),label='GT FG');ax.plot(centers,bgh/bgh.sum(),label='GT BG');ax.axvline(0,color='black',ls='--');ax.set_xlim(-.5,.5);ax.set_title(f'{ds} {co}: Dice={sub.dice.mean():.3f}, AUC={sub.auroc.mean():.3f}');ax.legend();ax.set_xlabel('N3-bil margin');fig.tight_layout();fig.savefig(distdir/f'{ds}_{co}.png',dpi=120);plt.close(fig)
  fig,ax=plt.subplots(figsize=(6,4));r=runs[runs.dataset==ds];h=float(r[r.kind=='H'].dice.iloc[0]);ax.axhline(h,label='Historical anchor',color='black')
  for kind,label in [('A','Anchor + random'),('R','Random only (same total images)')]:
   ss=r[r.kind==kind].groupby('additional').dice.agg(['mean','std']);ax.errorbar(ss.index,ss['mean'],yerr=ss['std'],marker='o',capsize=4,label=label)
  ax.set_xlabel('Additional support images (random-only matched total)');ax.set_ylabel('Mean per-image Dice');ax.legend();fig.tight_layout();fig.savefig(OUT/f'{ds}_anchor_vs_random.png',dpi=160);plt.close(fig)
 decorate(pd.DataFrame(disc)).to_csv(OUT/'margin_discriminability.csv',index=False)
 # Paired control: only occupancy source changes, all Support IDs/order/seed unchanged.
 old=pd.read_csv(ROOT/'outputs/support_size_study_original_memory/runs.csv'); print('old columns',old.columns.tolist(),flush=True)
 paired=runs[runs.kind=='T'].copy();paired.to_csv(OUT/'corrected_20shot.csv',index=False)
 vis=OUT/'diagnostic_visualizations';vis.mkdir(exist_ok=True)
 for ds in ['GlaS','CRAG']:
  pool,queries,hids,h=setup(ds);emap={e['id']:e for e in queries};p=per[per.dataset==ds];base=p[p.condition=='H'].set_index('image_id').dice;aa=p[p.condition.str.startswith('A_')].copy();aa['delta_dice']=aa.dice-aa.image_id.map(base)
  for direction,ascending in [('improvement',False),('degradation',True)]:
   chosen=aa.sort_values('delta_dice',ascending=ascending).drop_duplicates('image_id').head(3)
   table(vis/f'{ds}_{direction}_selection.csv',chosen.to_dict('records'))
   for num,row in enumerate(chosen.itertuples()):
    folder=OUT/'queries'/ds/row.image_id
    with np.load(folder/'margins.npz') as z: cs=z['conditions'].tolist();before=z['margins'][0];after=z['margins'][cs.index(row.condition)];gt=z['gt']
    im=np.asarray(Image.open(emap[row.image_id]['image']).convert('RGB').resize((256,256)));fig,axes=plt.subplots(1,6,figsize=(15,3));panels=[(im,'Image'),(gt,'GT'),(before>0,'Historical'),(after>0,row.condition),(after-before,'Margin change'),(((before>0)==gt).astype(int)-((after>0)==gt).astype(int),'Harmed +1 / corrected -1')]
    for ax,(v,title) in zip(axes,panels):ax.imshow(v,cmap='coolwarm' if 'change' in title or 'Harmed' in title else 'gray');ax.set_title(title);ax.axis('off')
    fig.suptitle(f'{ds} {row.image_id}: Dice delta={row.delta_dice:+.4f}');fig.tight_layout();fig.savefig(vis/f'{ds}_{direction}_{num+1}.png',dpi=140);plt.close(fig)
 # Source concentration and purity-sensitive coverage aggregates.
 agg=[]
 for (ds,co,cl),g in cover.groupby(['dataset','condition','gt_class']):
  agg.append({'dataset':ds,'condition':co,'gt_class':cl,'tokens':g.tokens.sum(),'correct_max_cos':np.average(g.correct_max_cos_mean,weights=g.tokens),'wrong_max_cos':np.average(g.wrong_max_cos_mean,weights=g.tokens),'top1_class_accuracy':g.correct_top1_class_wins.sum()/g.tokens.sum(),'top50_class_accuracy':g.correct_top50_class_wins.sum()/g.tokens.sum()})
 decorate(pd.DataFrame(agg)).to_csv(OUT/'reference_coverage_summary.csv',index=False)
 print('aggregation complete',len(runs),len(per),flush=True)
if __name__=='__main__':main()
