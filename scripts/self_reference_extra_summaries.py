from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parents[1]/'outputs/self_reference_seed_diagnosis'
s=pd.read_csv(out/'seed_summary.csv');t=pd.read_csv(out/'token_reliability_summary.csv');r=pd.read_csv(out/'self_reference_per_image.csv');e=pd.read_csv(out/'seed_expansion_metrics.csv')
fig,axes=plt.subplots(2,2,figsize=(10,7))
for j,ds in enumerate(['GlaS','CRAG']):
 for k,cls in enumerate(['fg','bg']):
  dsrows=s[s.dataset==ds].set_index('generator');counts=', '.join(f"{gg} {int(dsrows.loc[gg,cls+'_precision_valid_images'])}/{int(dsrows.loc[gg,'images'])}" for gg in ['A','B','C'])
  ax=axes[j,k];ax.set_title(f'{ds} {cls.upper()} seeds\nNonempty: {counts}',fontsize=10);ax.set_xlabel('Macro coverage');ax.set_ylabel('Macro precision (nonempty only)');ax.set_xlim(0,.4);ax.set_ylim(0,1.05);ax.grid(alpha=.3)
  for g,color in zip(['A','B','C'],['#1976d2','#e97820','#258a40']):
   row=s[(s.dataset==ds)&(s.generator==g)].iloc[0];p=row['macro_'+cls+'_precision'];c=row['macro_'+cls+'_coverage']
   if pd.notna(p):ax.scatter(c,p,color=color,label=g);ax.annotate(g,(c,p),xytext=(4,4),textcoords='offset points')
   else:ax.text(.02,.12 if g=='B' else .04,f'{g}: precision N/A (empty)',transform=ax.transAxes,color=color)
fig.tight_layout();fig.savefig(out/'seed_precision_coverage.png',dpi=150);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4))
for x,ds,color in [(0,'GlaS','#1976d2'),(1,'CRAG','#e97820')]:
 row=t[t.dataset==ds].set_index('generator').loc[['A','B','C']];xs=[j+(x-.5)*.3 for j in range(3)];ax.bar(xs,row.valid_rate*100,width=.3,label=ds,color=color)
ax.set_xticks([0,1,2]);ax.set_xticklabels(['A','B','C']);ax.set_ylabel('Valid two-class Top-50 memory (%)');ax.legend();ax.set_ylim(0,100);fig.tight_layout();fig.savefig(out/'memory_feasibility.png',dpi=150);plt.close(fig)
paired=[]
for (ds,g),sub in r.groupby(['dataset','generator']):
 ids=set(sub[(sub['mode']=='S0')&(sub.status=='VALID')].image_id)&set(sub[(sub['mode']=='S1')&(sub.status=='VALID')].image_id)
 for mode in ['S0','S1']:
  ss=e[(e.dataset==ds)&(e.generator==g)&(e['mode']==mode)&(e.region=='ignore')&e.image_id.isin(ids)]
  paired.append({'dataset':ds,'generator':g,'mode':mode,'paired_images':len(ids),'pooled_ignore_fg_recall':ss.true_fg_predicted.sum()/ss.gt_fg_pixels.sum() if len(ss) and ss.gt_fg_pixels.sum() else None,'pooled_ignore_bg_specificity':ss.true_bg_predicted.sum()/ss.gt_bg_pixels.sum() if len(ss) and ss.gt_bg_pixels.sum() else None,'macro_ignore_auc':ss.margin_auroc.mean() if len(ss) else None,'macro_ignore_auc_valid_images':int(ss.margin_auroc.notna().sum())})
pd.DataFrame(paired).to_csv(out/'paired_ignore_expansion.csv',index=False)
print('Additional fixed-result summaries complete')
