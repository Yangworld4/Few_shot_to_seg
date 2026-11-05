"""Final non-optimizing artifact audit: seed partition, strict candidates and source identity."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import yaml
import torch,skimage,sklearn,scipy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pathology_trainfree.data import load_image
from pathology_trainfree.self_reference.seed_generator import validate
from pathology_trainfree.self_reference.token_mapping import load_query_cache
OUT=ROOT/'outputs/self_reference_seed_diagnosis';cfg=yaml.safe_load((OUT/'configs/resolved.yaml').read_text());rows=[]
protocols=[json.loads((OUT/f'configs/protocol_{stage}.json').read_text()) for stage in ['E0','E1','E2']]
assert len({p['config_sha256'] for p in protocols})==1
for folder in sorted((OUT/'images').glob('*/*')):
 if not (folder/'seed_complete.json').exists():continue
 native={}
 with np.load(folder/'intermediate_maps.npz') as z:lumen=z['lumen_candidate']
 for g in ['A','B','C']:
  with np.load(folder/f'seeds_{g}.npz') as z:seed={k:z[k] for k in z.files}
  validate(seed);assert not (seed['fg']&lumen).any();native[g]=seed
  with np.load(folder/f'token_mapping_{g}.npz') as z:
   fg,bg,ig=z['fg'],z['bg'],z['ignore'];assert np.allclose(fg+bg+ig,1)
   expected={1:np.flatnonzero((fg>=.9)&(bg==0)),0:np.flatnonzero((bg>=.9)&(fg==0))};selected={1:z['fg_memory_indices'],0:z['bg_memory_indices']};centers=z['centers']
   assert np.array_equal(expected[1],z['fg_candidate_indices']) and np.array_equal(expected[0],z['bg_candidate_indices'])
   for c in [0,1]:assert np.isin(selected[c],expected[c]).all() and len(selected[c])<=512
  with np.load(folder/f'memory_{g}.npz') as z:
   metadata=json.loads(str(z['metadata']));assert metadata['gt_used'] is False and metadata['occupancy_source']=='generated_tristate_seed'
   for c,name in [(0,'bg'),(1,'fg')]:assert z[name].shape==(len(selected[c]),1024);np.testing.assert_array_equal(z[name+'_centers'],centers[selected[c]])
  rows.append({'dataset':folder.parent.name,'image_id':folder.name,'generator':g,'tristate_valid':True,'lumen_excluded_from_fg':True,'candidate_indices_exact_seed_rule':True,'memory_indices_seed_only':True,'ignore_not_added_to_memory':True,'gt_used_for_memory':False})
 assert np.all(native['C']['fg']<=native['B']['fg']) and np.all(native['C']['bg']<=native['B']['bg'])
assert len(rows)==360
pd.DataFrame(rows).to_csv(OUT/'artifact_validation.csv',index=False)
# Compare actual vectors for a quantity-valid case from each dataset; no GT is read.
f=pd.read_csv(OUT/'memory_feasibility.csv');checks=[]
for ds in ['GlaS','CRAG']:
 valid=f[(f.dataset==ds)&(f.status=='VALID')].sort_values(['image_id','generator'])
 if not len(valid):continue
 row=valid.iloc[0];folder=OUT/'images'/ds/row.image_id
 # Manifest adapter opens file paths only. GT pixel data is not used in this audit.
 from pathology_trainfree.data import read_manifest
 entry=next(e for e in read_manifest(ROOT/cfg['datasets'][ds]['manifest']) if e['id']==row.image_id)
 tokens,_,_,_=load_query_cache(load_image(entry['image']),row.image_id,ROOT/cfg['cache_root'],ds,cfg)
 with np.load(folder/f'token_mapping_{row.generator}.npz') as m,np.load(folder/f'memory_{row.generator}.npz') as z:
  for c in ['fg','bg']:np.testing.assert_array_equal(z[c],tokens.reshape(-1,1024)[m[c+'_memory_indices']])
 checks.append({'dataset':ds,'image_id':row.image_id,'generator':row.generator,'actual_vectors_equal_cached_seed_selected_vectors':True})
pd.DataFrame(checks).to_csv(OUT/'memory_vector_source_checks.csv',index=False)
selfrows=pd.read_csv(OUT/'self_reference_per_image.csv');assert len(selfrows)==720
for row in selfrows.itertuples():
 path=OUT/'images'/row.dataset/row.image_id/f'retrieval_{row.generator}_{row.mode}.npz'
 if row.status=='VALID':
  assert path.exists()
  with np.load(path) as z:assert z['mask'].shape==(256,256);np.testing.assert_array_equal(z['mask'],z['margin']>0)
 else:assert not path.exists(), 'Invalid cases must not contain substituted predictions'
env={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'numpy':np.__version__,'scipy':scipy.__version__,'skimage':skimage.__version__,'sklearn':sklearn.__version__,'source_fingerprint_configs_equal_across_stages':True}
(OUT/'environment.json').write_text(json.dumps(env,indent=2))
receipt=json.loads((OUT/'acceptance_checks.json').read_text());receipt.update({'all_360_seed_and_memory_conditions_audited':True,'all_720_inference_statuses_audited':True,'C_never_upgrades_ignore_verified':True,'invalid_cases_have_no_substitute_prediction':True,'source_vector_identity_checks':checks});(OUT/'acceptance_checks.json').write_text(json.dumps(receipt,indent=2))
print('PASS: 360 strict seed/memory audits; 720 valid/invalid prediction checks; original cached vectors verified for',len(checks),'datasets')
