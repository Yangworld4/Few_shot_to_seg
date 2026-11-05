"""Verify diagnostic retrieval against the independent project retriever, and original anchor masks."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
from reference_coverage_diagnosis import *
from pathology_trainfree.memory import ReferenceMemory
from pathology_trainfree.retrieval import SimilarityRetriever
from pathology_trainfree.geometry import reconstruct,resize_map
from pathology_trainfree.config import load_config
rows=[]
for ds in ['GlaS','CRAG']:
 pool,qs,hids,h=setup(ds);subset=json.loads((OUT/f'{ds}_source_subset.json').read_text());id=subset[0]
 folder=OUT/'queries'/ds/id
 with np.load(folder/'token_scores.npz') as z:scores=z['scores'];conditions=z['conditions'].tolist()
 with np.load(ROOT/f'outputs/support_size_study/cache/query_features/{ds}/feature_cache/{id}_offset0.npz') as z:q=z['tokens'].reshape(-1,1024)[:784]
 cfg=load_config(ROOT/'configs/uni_memory.yaml');cfg['retrieval'].update({'top_k':50,'chunk_size':784})
 for co in ['H','A_14_20','R_14_20','T_14_20']:
  try:
   with np.load(OUT/f'memories/{ds}_{co}.npz') as z:banks={0:z['bg'],1:z['fg']}
  except Exception:
   if co!='H':raise
   banks=h
   np.savez_compressed(OUT/f'memories/{ds}_H.npz',bg=h[0],fg=h[1])
  retriever=SimilarityRetriever(ReferenceMemory(banks,{}),cfg,'cuda');actual=retriever.scores(q).numpy();expected=scores[conditions.index(co),:, :784].T
  err=float(np.max(np.abs(actual-expected)));assert err<1e-6
  rows.append({'dataset':ds,'condition':co,'image_id':id,'max_score_abs_error_vs_independent_retriever':err})
 # Compare every H prediction to prior verified baseline masks (no repeated old experiment).
 mismatch=0
 for e in qs:
  with np.load(OUT/f'queries/{ds}/{e["id"]}/margins.npz') as z:pred=z['margins'][0]>0
  old=np.load(ROOT/f'outputs/n3_reference_parity_debug/ablations/{ds}/M0/{e["id"]}.npy')>0
  mismatch+=int((pred!=old).sum())
 assert mismatch<=1
 rows.append({'dataset':ds,'condition':'H','image_id':'ALL','mask_pixels_different_from_verified_M0':mismatch})
table(OUT/'retrieval_validation.csv',rows)
print(json.dumps(rows,indent=2))
