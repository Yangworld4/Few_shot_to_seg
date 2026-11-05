"""Post-execution acceptance audit of real outputs and immutable paired geometry."""
import sys,json
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from reference_source_stain import OUT,Experiment,SEEDS,table,signature,PROTO,geometry,allowed

def main():
 exp=Experiment();e=pd.read_csv(OUT/'external_results_per_pair.csv');s=pd.read_csv(OUT/'self_oracle_results_per_image.csv');checks=[]
 def check(name,passed,detail=''):
  checks.append({'check':name,'passed':bool(passed),'detail':str(detail)})
 check('External complete 5x40x5',len(e)==1000,len(e));check('Self complete 40x4x2',len(s)==320,len(s))
 check('All official queries external',set(e.query_id)=={q['id'] for q in exp.queries});check('All official queries self',set(s.query_id)=={q['id'] for q in exp.queries})
 check('External unique paired keys',not e.duplicated(['query_id','seed','condition']).any());check('Self unique paired keys',not s.duplicated(['query_id','condition','mode']).any())
 check('Raw historical exact',json.loads((OUT/'raw_regression.json').read_text())['fp32_exact_parity'])
 for id in list(exp.supports.values())+[q['id'] for q in exp.queries]:
  ix=exp.indices(id);co,oc,pos,length=exp.geom(id)
  for st in ['A','B']:
   with np.load(OUT/'memories'/f'{id}_{st}.npz') as z:
    check(f'{id} {st} spatial sampling',all(np.array_equal(z[f'indices_{c}'],ix[c]) and np.array_equal(z[f'footprints_{c}'],co[ix[c]]) for c in [0,1]))
    check(f'{id} {st} pure occupancy',np.all(oc[ix[1]]>=.9) and np.all(oc[ix[0]]<=.1))
  if id in exp.supports.values():
   for sd in [se for se in SEEDS if exp.supports[se]==id]:check(f'{id} support identity',set(e[e.seed==sd].reference_id)=={id})
 for r in s.itertuples():
  if r.status!='VALID':continue
  with np.load(OUT/'cases'/f'{r.case}.npz') as z:n=z['eligible_counts']
  check(r.case+' fixed K',np.all(n>=50) and r.actual_k==50)
  if r.mode=='SpatialExclusion':
   co=exp.geom(r.query_id)[0];ix=exp.indices(r.query_id)
   for c in [0,1]:
    cc=np.concatenate([allowed(co[k:k+784],co[ix[c]]).sum(1) for k in range(0,len(co),784)])
    check(r.case+f' eligible class{c}',np.array_equal(cc,n[:,c]))
 for fn in ['stain_templates.json','stain_quality_checks.csv','support_query_pairs.csv','external_results_per_pair.csv','self_oracle_results_per_image.csv','memory_diagnostics.csv','retrieval_diagnostics.csv','stain_gap_metrics.csv','paired_effects.csv','source_effects.csv','self_matching_checks.csv','invalid_cases.csv','regression_checks.md','run_metadata.json','report.md','execution.log']:
  check('Artifact '+fn,(OUT/fn).is_file())
 for l in 'ABCDE':check('Figure '+l,(OUT/f'figures/figure_{l}.png').is_file())
 table(OUT/'acceptance_checks.csv',checks);result={'passed':sum(r['passed'] for r in checks),'total':len(checks),'all_passed':all(r['passed'] for r in checks)};(OUT/'acceptance_checks.json').write_text(json.dumps(result,indent=2));print(result)
 assert result['all_passed']
if __name__=='__main__':main()
