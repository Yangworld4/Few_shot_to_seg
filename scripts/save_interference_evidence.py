"""Preserve exact Top50 indices for changed pure-GT tokens in the preset source subset."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from reference_coverage_diagnosis import *
rows=[];torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
for ds in ['GlaS','CRAG']:
 for id in json.loads((OUT/f'{ds}_source_subset.json').read_text()):
  folder=OUT/'queries'/ds/id
  with np.load(folder/'token_scores.npz') as z:sc=z['scores'];occ=z['occupancy'];conds=z['conditions'].tolist()
  tokens=[]
  for offset in [0,8]:
   with np.load(ROOT/f'outputs/support_size_study/cache/query_features/{ds}/feature_cache/{id}_offset{offset}.npz') as z:tokens.append(z['tokens'].reshape(-1,1024))
  tokens=np.concatenate(tokens);pure=(occ>=.9)|(occ<=.1);bp=(sc[0,1]-sc[0,0])>0
  for ix,co in enumerate(conds):
   if not co.startswith('A_'):continue
   ap=(sc[ix,1]-sc[ix,0])>0;selected=np.flatnonzero(pure&(bp!=ap));out=folder/'flip_topk_indices'/f'{co}.npz';out.parent.mkdir(exist_ok=True)
   if out.exists():continue
   allidx={};allvalues={}
   for name in ['H',co]:
    with np.load(OUT/f'memories/{ds}_{name}.npz') as z:bank={0:z['bg'],1:z['fg']}
    for c in [0,1]:
     b=F.normalize(torch.tensor(bank[c].astype(np.float32),device='cuda'),dim=1);inds=[];values=[]
     for subset in np.array_split(selected,max(1,int(np.ceil(len(selected)/784)))):
      if not len(subset):continue
      q=F.normalize(torch.tensor(tokens[subset].astype(np.float32),device='cuda'),dim=1)
      v,j=(q@b.T).topk(50,dim=1);inds.append(j.cpu().numpy().astype(np.int32));values.append(v.cpu().numpy())
     allidx[f'{name}_class{c}_indices']=np.concatenate(inds) if inds else np.empty((0,50),np.int32);allvalues[f'{name}_class{c}_cosines']=np.concatenate(values) if values else np.empty((0,50),np.float32)
   np.savez_compressed(out,token_indices=selected,occupancy=occ[selected],before_scores=sc[0,:,selected],after_scores=sc[ix,:,selected],**allidx,**allvalues)
   rows.append({'dataset':ds,'image_id':id,'condition':co,'changed_pure_tokens':len(selected),'raw_neighbor_file':str(out),'config_fingerprint':FP,'code_sha256':CODE})
  print(ds,id,'raw flip indices complete',flush=True)
table(OUT/'raw_topk_evidence_index.csv',rows)
