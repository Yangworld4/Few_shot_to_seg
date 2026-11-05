from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from reference_coverage_diagnosis import *
import pandas as pd
r=pd.read_csv(OUT/'near_duplicate_candidates.csv'); out=OUT/'diagnostic_visualizations';out.mkdir(exist_ok=True);rows=[]
for ds in ['GlaS','CRAG']:
 pool,queries,ids,h=setup(ds); es={e['id']:e for e in pool+queries};chosen=r[r.dataset==ds].sort_values('dhash_distance').head(6)
 fig,axes=plt.subplots(6,2,figsize=(8,15))
 for i,row in enumerate(chosen.itertuples()):
  a=load_image(es[row.support_id]['image']);b=load_image(es[row.query_id]['image'])
  aa=np.array(Image.fromarray(a).convert('L').resize((64,64)),np.float32).ravel();bb=np.array(Image.fromarray(b).convert('L').resize((64,64)),np.float32).ravel();corr=float(np.corrcoef(aa,bb)[0,1]);rows.append({**row._asdict(),'grayscale_64x64_pearson':corr,'interpretation':'candidate only; not patient/slide identity'})
  axes[i,0].imshow(a);axes[i,0].set_title(f'{row.support_id} historical={row.historical}');axes[i,1].imshow(b);axes[i,1].set_title(f'{row.query_id} dHash={row.dhash_distance} corr={corr:.3f}')
  for ax in axes[i]:ax.axis('off')
 fig.tight_layout();fig.savefig(out/f'{ds}_near_duplicate_candidates.png',dpi=100);plt.close(fig)
table(OUT/'near_duplicate_review.csv',rows)
