from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'outputs/self_reference_seed_diagnosis';rows=[]
for path in sorted((out/'images').glob('*/*/memory_*.npz')):
 with np.load(path) as z:
  row={'dataset':path.parent.parent.name,'image_id':path.parent.name,'generator':path.stem.split('_')[-1]}
  for cls in ['fg','bg']:
   xy=z[cls+'_centers'];n=len(xy);unique=len(np.unique(xy,axis=0))
   row[cls+'_memory_tokens']=n;row[cls+'_unique_physical_centers']=unique;row[cls+'_repeated_center_tokens']=n-unique
  rows.append(row)
pd.DataFrame(rows).to_csv(out/'physical_reference_diversity.csv',index=False)
print('physical reference diversity conditions',len(rows))
