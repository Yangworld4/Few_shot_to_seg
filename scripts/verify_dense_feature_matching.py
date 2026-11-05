"""Final acceptance checks for locked E0-E3 experiments and GT isolation."""
from pathlib import Path
import sys,json,inspect,hashlib
import numpy as np,pandas as pd,torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from dense_feature_matching import Study,OUT,ROOT,PROTO,FP,table,load_mask,resize_mask,DC,object_stats
from pathology_trainfree.evaluation import object_metrics
from pathology_trainfree.pipeline import SegmentationPipeline

def main():
 study=Study();df=pd.read_csv(OUT/'per_image_metrics.csv');oracle=pd.read_csv(OUT/'oracle_reconstruction_metrics.csv');rows=[]
 def check(name,ok,detail=''):
  rows.append({'check':name,'passed':bool(ok),'detail':str(detail)})
 check('480 non-oracle cases',len(df)==480,len(df));check('120 isolated oracle cases',len(oracle)==120,len(oracle));check('no duplicate paired key',not df.duplicated(['dataset','image_id','method']).any())
 for ds,st in study.data.items():
  cfg=json.loads((OUT/f'configs/{ds}.json').read_text());check(ds+' original labeled10 support count',len(cfg['support_ids_from_manifest'])==PROTO['datasets'][ds]['support']);check(ds+' identical historical bank bytes',cfg['memory_sha256']==hashlib.sha256(st['source'].read_bytes()).hexdigest())
  original_vector_hash={str(c):hashlib.sha256(st['memory'].vectors[c].tobytes()).hexdigest() for c in [0,1]}
  for c,label in [(0,'bg'),(1,'fg')]:check(ds+' bank '+label,original_vector_hash[str(c)]==cfg[label+'_vectors_sha256'])
  for m in ['B0','B1','B2','B3']:check(ds+' '+m+' full IDs',set(df[(df.dataset==ds)&(df.method==m)].image_id)=={e['id'] for e in st['queries']})
  # Independent public pipeline comparison, using exact cached features and original memory.
  first=st['queries'][0];image=__import__('pathology_trainfree.data',fromlist=['load_image']).load_image(first['image']);pipeline=SegmentationPipeline(study.encoder,st['memory'],st['cfg']);pred=pipeline.predict(image,first['id'])
  with np.load(study.path(ds,first['id'],'B0').with_suffix('.npz')) as z:check(ds+' original pipeline first margin exact',np.array_equal(z['margin'],pred['scores']))
  for e in st['queries']:
   gt=resize_mask(load_mask(e['mask'],DC),(256,256));basepred=None
   for m in ['B0','B1','B2','B3','ORACLE']:
    p=study.path(ds,e['id'],m);r=json.loads(p.read_text())
    with np.load(p.with_suffix('.npz')) as z:
     mm=z['margin'];mask=z['mask'];check(ds+' '+e['id']+' '+m+' finite and threshold',mm.shape==gt.shape==(256,256) and np.isfinite(mm).all() and np.array_equal(mask,mm>0))
    check(ds+' '+e['id']+' '+m+' protocol',r['protocol_fingerprint']==FP)
    if m=='B0':
     # Object F1 extension must preserve historical matching exactly.
     f1,_=object_metrics(mask.astype(bool),gt.astype(bool));check(ds+' '+e['id']+' object parity',abs(r['object_f1']-f1)<1e-12)
 # GT reads are absent from dense predict; evaluation occurs after the predict call.
 check('no GT in dense predict',all(x not in inspect.getsource(Study.predict) for x in ['load_mask','occupancy','gt','mask']))
 run_source=inspect.getsource(Study.run);check('oracle separate explicit branch',"method=='ORACLE'" in run_source);check('B3 fixed continuous margin input',"['margin']" in run_source and 'guided_filter(guide,base)' in run_source)
 impl=pd.read_csv(OUT/'implementation_checks.csv');check('all implementation tests',impl.passed.all(),len(impl));par=pd.read_csv(OUT/'baseline_parity_per_image.csv');check('all prior verified baseline masks exact',(par.validated_replay_disagreement_pixels==0).all())
 for fn in ['baseline_parity.md','e0_error_diagnostics.csv','oracle_reconstruction_metrics.csv','per_image_metrics.csv','summary.csv','paired_deltas.csv','decision_flip_analysis.csv','boundary_analysis.csv','object_error_analysis.csv','margin_analysis.csv','runtime_memory.csv','failure_cases.csv','report.md','execution.log']:
  check('artifact '+fn,(OUT/fn).is_file())
 for i in range(1,5):check('figure '+str(i),(OUT/f'figures/figure_{i}.png').is_file())
 for ds in study.data:
  for case in ['improvement','degradation']:check(ds+' figure5 '+case,(OUT/f'figures/figure_5_{ds}_{case}.png').is_file())
 table(OUT/'acceptance_checks.csv',rows);result={'passed':sum(r['passed'] for r in rows),'total':len(rows),'all_passed':all(r['passed'] for r in rows)};(OUT/'acceptance_checks.json').write_text(json.dumps(result,indent=2));print(result);assert result['all_passed']
if __name__=='__main__':main()
