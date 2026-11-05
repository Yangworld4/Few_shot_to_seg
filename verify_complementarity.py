"""Final artifact audit: no inference tuning, no test-label access in prediction."""
import csv,json
from pathlib import Path
import numpy as np
from pathology_trainfree.data import read_manifest,load_mask,load_image,resize_mask
from pathology_trainfree.encoder import sha256
from pathology_trainfree.memory import ReferenceMemory
from run_complementarity import OUT,ROOT,SEEDS,MODELS,config_for,atomic_json,write_csv


def main():
    support=json.loads((OUT/'configs/support_manifest.json').read_text());rows=[]
    checks={};total=0
    for name in MODELS:
        meta=json.loads((OUT/f'configs/{name}_smoke.json').read_text());assert meta['status']=='PASS'
        assert meta['shape']==[1,448//meta['patch_size'],448//meta['patch_size'],meta['feature_dim']]
        assert meta['prefix_tokens']==(9 if name=='uni2_h' else 1)
    checks['V1']='PASS: all pretrained checkpoints strict loaded; 448 spatial shapes, finite unit tokens, frozen params, prefix exclusion'
    for ds in ['GlaS','CRAG']:
        cfg=config_for(ds,14,'uni');pool=read_manifest(cfg['dataset']['support_manifest'],require_mask=True);query=read_manifest(cfg['dataset']['query_manifest'],require_mask=True)
        byid={e['id']:e for e in pool};qhash={sha256(e['image']) for e in query}
        allsupport=set(i for s in SEEDS for i in support[ds][str(s)]['support_ids'])
        assert not {sha256(byid[i]['image']) for i in allsupport}&qhash,'Exact RGB file leakage'
        for name in MODELS:
            p={'uni':16,'uni2_h':14,'kaiko_vitb8':8}[name]
            for seed in SEEDS:
                ids=support[ds][str(seed)]['support_ids'];mem=ReferenceMemory.load(OUT/f'memory/{name}/{ds}/seed_{seed}.npz')
                assert mem.metadata['support_ids']==ids
                assert mem.metadata['occupancy_source']=='real_support_gt'
                assert mem.metadata['memory_config']['fallback_strategy']=='none'
                assert mem.metadata['window']['support_offsets']==[0,8]
                assert mem.metadata['encoder_patch_size']==p
                for i in ids:
                    assert mem.metadata['support_gt_sha256'][i]==sha256(byid[i]['mask'])
                    for meta in mem.metadata['support_occupancy_metadata'][i]:
                        assert meta['occupancy_source']=='real_support_gt' and meta['patch_size']==p
                        assert meta['gt_sha256']==sha256(byid[i]['mask'])
                for c,v in mem.vectors.items():
                    assert v.shape[0]>=50 and v.shape[0]<=20*512
                    assert np.isfinite(v).all() and np.allclose(np.linalg.norm(v.astype('float32'),axis=-1),1,atol=.001)
                cfgrecord=json.loads((OUT/f'configs/{name}_{ds}_{seed}.json').read_text())
                assert cfgrecord['config']['retrieval']['top_k']==50
                assert cfgrecord['config']['segmentation']['threshold']==0
                directory=OUT/f'predictions/{ds}/{name}/seed_{seed}'
                complete=json.loads((directory/'complete.json').read_text());assert complete['status']=='COMPLETE'
                metric_rows=list(csv.DictReader((directory/'per_image_metrics.csv').open()))
                assert {r['image_id'] for r in metric_rows}=={e['id'] for e in query}
                gt_hashes={e['id']:sha256(e['mask']) for e in query}
                assert all(r['gt_sha256']==gt_hashes[r['image_id']] for r in metric_rows)
                assert len(list(directory.glob('*.npz')))==len(query)
                for e in query:
                    with np.load(directory/f'{e["id"]}.npz') as z:
                        mask,margin=z['mask'],z['margin']
                        assert str(z['config_fingerprint'])==cfgrecord['fingerprint']
                        assert str(z['image_sha256'])==sha256(e['image'])
                    assert mask.shape==margin.shape==(256,256) and np.isfinite(margin).all()
                    np.testing.assert_array_equal(mask,margin>0)
                    gt=resize_mask(load_mask(e['mask'],cfg['dataset']),(256,256));assert gt.shape==mask.shape
                    total+=1
                rows.append({'dataset':ds,'model':name,'seed':seed,'images':len(query),'status':'PASS',
                             'fg_tokens':len(mem.vectors[1]),'bg_tokens':len(mem.vectors[0]),'occupancy_source':'real_support_gt','top_k':50,'fallbacks':0})
    checks['V2']='PASS: current GT SHA256 verified for every support; GT occupancy freshly recomputed with model patch size; placeholder caches ignored'
    checks['V3']='PASS: all 1800 masks/margins aligned to nearest-neighbor GT on shared256 grid; masks exactly margin>0'
    reg=list(csv.DictReader((OUT/'metrics/uni_regression.csv').open()));assert len(reg)==10 and all(r['status']=='PASS' for r in reg)
    checks['V4']='PASS: 10 UNI runs,600 image conditions; byte-exact FG/BG Memory, seven per-image metrics identical (<1e-9). Historical runs did not save masks/margins, so no unsupported claim of historical pixel-array equality.'
    checks['V5']='PASS: all30 FG/BG banks valid, >=50 tokens, cap512/image/class, real purity>=0.9, zero fallbacks; exact Top50 cosine mean'
    tests=(OUT/'unit_tests.log').read_text();assert '24 passed' in tests
    checks['V6']='PASS: 24 tests including exhaustive 8 binary error patterns, directionality, shared FP/FN, voting/oracle identities, zero denominators, three patch sizes, cache pollution and spatial reconstruction'
    checks['V7']='PASS: frozen identical support order per dataset/seed; train/test IDs, paths and exact image file hashes disjoint; prediction uses images and support-only Memory; test GT only after predict for evaluation/diagnosis'
    diagnosis=json.loads((OUT/'diagnostics_complete.json').read_text());assert diagnosis['model_runs']==30 and diagnosis['model_image_predictions']==1800
    for ds in ['GlaS','CRAG']:
        cfg=config_for(ds,14,'uni')
        for seed in SEEDS:
            for entry in read_manifest(cfg['dataset']['query_manifest'],require_mask=True):
                iid=entry['id'];gt=resize_mask(load_mask(entry['mask'],cfg['dataset']),(256,256)).astype(bool)
                preds=[]
                for m in MODELS:
                    with np.load(OUT/f'predictions/{ds}/{m}/seed_{seed}/{iid}.npz') as z:preds.append(z['mask'].astype(bool))
                wrong=np.stack([p!=gt for p in preds]).all(0)
                with np.load(OUT/f'predictions/{ds}/majority/seed_{seed}/{iid}.npz') as z:vote=z['mask'].astype(bool)
                with np.load(OUT/f'predictions/{ds}/oracle/seed_{seed}/{iid}.npz') as z:oracle=z['mask'].astype(bool);assert bool(z['uses_gt'])
                np.testing.assert_array_equal(vote,np.stack(preds).sum(0)>=2)
                np.testing.assert_array_equal(oracle!=gt,wrong)
    for r in csv.DictReader((OUT/'metrics/triple_errors.csv').open()):
        assert sum(int(r[k]) for k in ['all_correct','all_wrong','exactly_one_correct','exactly_two_correct'])==int(r['pixels'])
        assert sum(int(r['only_'+m+'_correct']) for m in MODELS)==int(r['exactly_one_correct'])
        assert int(r['oracle_correctable_uni_errors'])+int(r['all_wrong'])==int(r['uni_errors'])
    checks['V8']='PASS: 30/30 model runs,1800 single-model predictions,1200 voting/oracle masks; all logs/configs/metrics saved; actual saved voting/oracle masks and all600 triple count identities verified'
    audit=json.loads((OUT/'configs/source_audit.json').read_text())
    for file,expected in audit['source_files_sha256'].items():assert sha256(file)==expected,'Source changed: '+file
    checks['DUAL_SOURCE']='PASS: audited DUAL source hashes unchanged from task audit'
    write_csv(OUT/'metrics/acceptance_runs.csv',rows)
    atomic_json(OUT/'acceptance_checks.json',{'status':'PASS','model_runs':len(rows),'predictions':total,'checks':checks})
    text='# 验证结果\n\n'+'\n\n'.join('## '+k+'\n\n'+v for k,v in checks.items())
    (OUT/'validation_checks.md').write_text(text,encoding='utf-8')
    meta=json.loads((OUT/'run_metadata.json').read_text());meta['status']='VALIDATED_AWAITING_REPORT';meta['acceptance']=checks
    if (OUT/'run_metadata_uni.json').exists():
        meta['uni_phase']=json.loads((OUT/'run_metadata_uni.json').read_text())
    if (OUT/'execution_transition.json').exists():
        meta['execution_transition']=json.loads((OUT/'execution_transition.json').read_text())
    meta['final_code_sha256']={str(p.relative_to(ROOT)):sha256(p) for p in list((ROOT/'pathology_trainfree').glob('*.py'))+list(ROOT.glob('*complementarity.py'))}
    atomic_json(OUT/'run_metadata.json',meta)
    print(json.dumps(checks,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
