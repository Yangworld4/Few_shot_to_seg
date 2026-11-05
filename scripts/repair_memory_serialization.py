from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from reference_coverage_diagnosis import *
rows=[]
for ds in ['GlaS','CRAG']:
 _,_,_,conds=prepare(ds)
 for co in conds:
  p=OUT/f'memories/{ds}_{co["condition"]}.npz';valid=False;error=''
  try:
   with np.load(p) as z:valid=all(np.array_equal(z['bg' if c==0 else 'fg'],co['bank'][c]) and np.array_equal(z['bg_source' if c==0 else 'fg_source'],co['src'][c]) for c in [0,1])
  except Exception as e:error=repr(e)
  if not valid:
   temp=p.with_suffix('.npz.tmp')
   with temp.open('wb') as f:np.savez_compressed(f,bg=co['bank'][0],fg=co['bank'][1],bg_source=co['src'][0],fg_source=co['src'][1],metadata=json.dumps(co['meta']))
   temp.replace(p)
  rows.append({'dataset':ds,'condition':co['condition'],'serialization_repaired':not valid,'prior_error':error,'vectors_equal_original_in_memory_build':True})
 print(ds,'memory serialization validated',flush=True)
table(OUT/'memory_serialization_validation.csv',rows)
