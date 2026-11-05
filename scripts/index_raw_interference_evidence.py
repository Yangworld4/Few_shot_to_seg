from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from reference_coverage_diagnosis import *
import pandas as pd
meta=pd.read_csv(OUT/'support_sets.csv');rows=[];flips=[]
for ds in ['GlaS','CRAG']:
 for id in json.loads((OUT/f'{ds}_source_subset.json').read_text()):
  folder=OUT/'queries'/ds/id
  with np.load(folder/'token_scores.npz') as z:
   sc=z['scores'];occ=z['occupancy'];conds=z['conditions'].tolist();bm=sc[0,1]-sc[0,0]
  for ix,co in enumerate(conds):
   if not co.startswith('A_'):continue
   am=sc[ix,1]-sc[ix,0];fg=occ>=.9;bg=occ<=.1
   flips.append({'dataset':ds,'condition':co,'image_id':id,'pure_fg_tokens':int(fg.sum()),'original_correct_fg_tokens':int((fg&(bm>0)).sum()),'strict_fg_positive_to_negative':int((fg&(bm>0)&(am<0)).sum()),'pure_bg_tokens':int(bg.sum()),'original_correct_bg_tokens':int((bg&(bm<=0)).sum()),'strict_bg_negative_to_positive':int((bg&(bm<0)&(am>0)).sum()),'zero_after_tokens':int((am==0).sum())})
   p=folder/'flip_topk_indices'/f'{co}.npz'
   with np.load(p) as z:
    n=len(z['token_indices']);assert all(z[k].shape==(n,50) for k in z.files if k.endswith('indices') and k!='token_indices')
   rows.append({'dataset':ds,'condition':co,'image_id':id,'changed_pure_tokens':n,'raw_neighbor_file':str(p)})
pd.DataFrame(rows).merge(meta,on=['dataset','condition']).to_csv(OUT/'raw_topk_evidence_index.csv',index=False)
pd.DataFrame(flips).merge(meta,on=['dataset','condition']).to_csv(OUT/'strict_margin_flip_analysis_subset.csv',index=False)
print('raw neighbor files validated and indexed',len(rows))
