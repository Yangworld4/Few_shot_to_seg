from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from reference_coverage_diagnosis import *
import pandas as pd
rows=[]
for ds in ['GlaS','CRAG']:
 meta=pd.read_csv(OUT/f'{ds}_support_sets.csv');_,_,_,h=setup(ds);hc={c:np.mean(h[c].astype(np.float32),axis=0) for c in [0,1]}
 for row in meta.to_dict('records'):
  co=row['condition'];p=OUT/f'memories/{ds}_{co}.npz'
  try:
   with np.load(p) as z:b={0:z['bg'],1:z['fg']};src={0:z['bg_source'],1:z['fg_source']}
  except Exception:
   # Only an interrupted first anchor serialization may be incomplete; raw H never rebuilt.
   if co!='H':raise
   _,_,hids,_=setup(ds);b=h
   from collections import defaultdict,deque
   sys.path.insert(0,str(ROOT/'outputs/n3_reference_parity_debug'))
   from memory_reconstruction import sample_image
   rng=np.random.default_rng(2026);lookup={c:defaultdict(deque) for c in [0,1]}
   for id in hids:
    sel,_,_=sample_image(ds,id,[0,1],512,rng)
    for c in [0,1]:
     for v in sel[c]:lookup[c][v.tobytes()].append(id)
   src={c:np.array([lookup[c][v.tobytes()].popleft() for v in b[c]]) for c in [0,1]}
   np.savez_compressed(p,bg=b[0],fg=b[1],bg_source=src[0],fg_source=src[1],metadata=json.dumps(row))
  for c in [0,1]:
   assert len(src[c])==len(b[c]);x=b[c].astype(np.float32);norm=np.linalg.norm(x,axis=1);centroid=x.mean(0);rng=np.random.default_rng(2026);u=rng.integers(0,len(x),10000);v=rng.integers(0,len(x),10000);cs=(x[u]*x[v]).sum(1)/(norm[u]*norm[v]);ref=hc[c];other=b[1-c].astype(np.float32).mean(0)
   rows.append({**row,'class':c,'norm_mean':norm.mean(),'norm_min':norm.min(),'norm_max':norm.max(),'within_cos_mean_sampled_pairs':cs.mean(),'within_cos_std_sampled_pairs':cs.std(),'dimension_variance_mean':x.var(0).mean(),'centroid_cos_to_historical':float(centroid@ref/(np.linalg.norm(centroid)*np.linalg.norm(ref))),'fg_bg_centroid_cos':float(centroid@other/(np.linalg.norm(centroid)*np.linalg.norm(other)))})
table(OUT/'memory_feature_distribution.csv',rows)
print('all saved memory files validated; classes=',len(rows))
