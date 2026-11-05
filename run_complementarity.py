"""Fixed-support three-encoder N3-bil experiment; resumable per image."""
import argparse,csv,gc,json,os,sys,time,traceback
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import yaml
from pathology_trainfree.cache import FeatureStore,signature,save_npz,ALGORITHM_VERSION
from pathology_trainfree.config import validate
from pathology_trainfree.data import read_manifest,load_image,load_mask,resize_mask
from pathology_trainfree.encoder import FeatureEncoder,sha256
from pathology_trainfree.evaluation import metrics,object_metrics
from pathology_trainfree.memory import ReferenceMemory,select_tokens
from pathology_trainfree.pipeline import SegmentationPipeline
from study_support_size import load_support_cache,atomic_json
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/multi_fm_complementarity'
HISTORY=ROOT/'outputs/support_size_study_gtfixed'
SEEDS=[14,24,42,1337,3407]
MODELS=['uni','uni2_h','kaiko_vitb8']
METRICS=['dice','fg_iou','precision','recall','object_f1','object_hausdorff','predicted_fg_ratio']


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.with_suffix('.tmp').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    os.replace(path.with_suffix('.tmp'),path)


def log(s):print(datetime.now().isoformat(timespec='seconds'),s,flush=True)


def score_mask(pred,gt,cfg):
    fg=metrics(pred,gt,[0,1])[1]
    f1,hd=object_metrics(pred,gt,cfg['evaluation']['object_min_area'],cfg['evaluation']['object_match_overlap'])
    return {'dice':fg['dice'],'fg_iou':fg['iou'],'precision':fg['precision'],'recall':fg['recall'],
            'object_f1':f1,'object_hausdorff':hd,'predicted_fg_ratio':float(pred.mean()),
            'tp':fg['tp'],'fp':fg['fp'],'fn':fg['fn']}


def freeze_protocol():
    manifest={}
    for ds in ['GlaS','CRAG']:
        manifest[ds]={}
        for seed in SEEDS:
            source=HISTORY/f'configs/{ds}/seed_{seed}_shots_20.yaml'
            c=yaml.safe_load(source.read_text());pool=read_manifest(c['support_manifest'],require_mask=True)
            queries=read_manifest(c['query_manifest'],require_mask=True)
            ids=c['support_ids'];byid={x['id']:x for x in pool}
            assert len(ids)==len(set(ids))==20 and all(i in byid for i in ids)
            assert not set(ids)&{x['id'] for x in queries}
            assert not {byid[i]['image'] for i in ids}&{x['image'] for x in queries}
            assert len(queries)==(80 if ds=='GlaS' else 40)
            gt_hash={i:sha256(byid[i]['mask']) for i in ids}
            assert gt_hash==c['gt_sha256']
            manifest[ds][str(seed)]={'support_ids':ids,'support_gt_sha256':gt_hash,
                                    'query_ids':[x['id'] for x in queries],
                                    'source':str(source),'source_sha256':sha256(source)}
            write_csv(OUT/f'configs/{ds}/support_seed_{seed}.csv',[
                {'image_id':i,'image':str(byid[i]['image']),'mask':str(byid[i]['mask']),'gt_sha256':gt_hash[i]} for i in ids])
    path=OUT/'configs/support_manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise RuntimeError('Frozen Support manifest changed')
    atomic_json(path,manifest)
    return manifest


def config_for(ds,seed,name):
    cfg=deepcopy(yaml.safe_load((HISTORY/f'configs/{ds}/seed_{seed}_shots_20.yaml').read_text())['config'])
    meta=json.loads((OUT/f'configs/{name}_smoke.json').read_text())
    cfg['encoder'].update(name=name,weights=meta['checkpoint_path'],config=None)
    if name != 'uni':
        cfg['encoder']['batch_size']=4
        cfg['memory']['file']=str(OUT/f'memory/{name}/{ds}/seed_{seed}.npz')
        cfg['output']['directory']=str(OUT/f'predictions/{ds}/{name}/seed_{seed}')
    cfg['memory']['fallback_strategy']='none'
    cfg['output'].update(cache_features=False)
    cfg['retrieval']['chunk_size']=784
    validate(cfg)
    return cfg


def make_memory(ids,cache,cfg,encoder,seed):
    rng=np.random.default_rng(2026);parts={0:[],1:[]};stats=[]
    for iid in ids:
        item=cache[iid];fg=item['fg_occ'];occ=np.stack([1-fg,fg],-1)
        selected,fallback=select_tokens(item['features'],occ,cfg['memory'],rng)
        row={'image_id':iid,'candidate_tokens':len(fg),'gt_sha256':item['gt_sha256'],
             'occupancy_source':'real_support_gt','encoder_id':encoder.name,'patch_size':encoder.patch_size}
        for cls,label in [(1,'fg'),(0,'bg')]:
            parts[cls].append(selected[cls]);row[label+'_pure_tokens']=int((occ[:,cls]>=.9).sum())
            row[label+'_selected_tokens']=len(selected[cls]);row[label+'_fallback']=fallback[cls]
        stats.append(row)
    vec={c:np.concatenate(parts[c]) for c in [0,1]}
    if any(len(x)<50 for x in vec.values()):raise RuntimeError('INVALID_RUN: class Memory below K=50')
    meta={'encoder_name':encoder.name,'encoder_sha256':encoder.weights_sha256,'encoder_patch_size':encoder.patch_size,
          'encoder_metadata':encoder.metadata,'feature_dim':encoder.feature_dim,'occupancy_source':'real_support_gt',
          'support_ids':ids,'support_gt_sha256':{i:cache[i]['gt_sha256'] for i in ids},
          'support_occupancy_metadata':{i:cache[i]['occupancy_metadata'] for i in ids},
          'encoder_normalize':encoder.normalize,'encoder_precision':encoder.precision,
          'encoder_token_precision':encoder.token_precision,'memory_config':cfg['memory'],'window':cfg['window'],
          'sampling_protocol':'original_global PCG64 seed2026, image order, FG then BG, sorted sampled indices',
          'support_selection_seed':seed,'counts':{str(c):len(vec[c]) for c in [0,1]},'fallback_counts':{'0':0,'1':0},
          'algorithm_version':ALGORITHM_VERSION}
    return ReferenceMemory(vec,meta),stats


def run_model(name,manifest,regression):
    encoder=FeatureEncoder(config_for('GlaS',14,name)['encoder'])
    for ds in ['GlaS','CRAG']:
        cfg=config_for(ds,14,name)
        pool=read_manifest(cfg['dataset']['support_manifest'],require_mask=True);byid={e['id']:e for e in pool}
        entries=read_manifest(cfg['dataset']['query_manifest'],require_mask=True)
        ids=list(dict.fromkeys(i for seed in SEEDS for i in manifest[ds][str(seed)]['support_ids']))
        support_store=FeatureStore(OUT/f'cache/{name}/{ds}/support',encoder,cfg)
        legacy=ROOT/f'outputs/support_size_study/cache/query_features/{ds}/feature_cache'
        query_store=FeatureStore(OUT/f'cache/{name}/{ds}/query',encoder,cfg,legacy_root=legacy if name=='uni' else None)
        support_cache={};precompute_start=time.monotonic()
        for n,iid in enumerate(ids,1):
            entry=byid[iid];image=load_image(entry['image']);mask=load_mask(entry['mask'],cfg['dataset'])
            assert image.shape[:2]==mask.shape
            legacy_item=None
            if name=='uni':
                legacy_store,_=load_support_cache([entry],ROOT.parent/'DUAL',ds,cfg);legacy_item=legacy_store[iid]
            features=[];occupancy=[];metadata=[];cursor=0
            for offset in cfg['window']['support_offsets']:
                supplied=None
                if legacy_item is not None:
                    from pathology_trainfree.geometry import windows
                    count=len(list(windows(image,448,224,offset)))
                    supplied=legacy_item['features'][cursor:cursor+count*784].reshape(count,784,1024);cursor+=count*784
                tokens,positions=support_store.get(image,iid,offset,supplied=supplied,supplied_source='audited DUAL features_N3' if supplied is not None else None)
                occ,meta=support_store.occupancy(mask,iid,entry['mask'],offset,positions)
                features.append(tokens.reshape(-1,encoder.feature_dim));occupancy.append(occ.ravel());metadata.append(meta)
            support_cache[iid]={'features':np.concatenate(features),'fg_occ':np.concatenate(occupancy),
                                'gt_sha256':sha256(entry['mask']),'occupancy_metadata':metadata}
            if n==1 or n%10==0 or n==len(ids):log(f'{name} {ds}: support {n}/{len(ids)}')
        support_seconds=time.monotonic()-precompute_start
        t=time.monotonic()
        for n,entry in enumerate(entries,1):
            image=load_image(entry['image'])
            for offset in cfg['window']['offsets']:query_store.get(image,entry['id'],offset)
            if n==1 or n%10==0 or n==len(entries):log(f'{name} {ds}: query cache {n}/{len(entries)}')
        query_seconds=time.monotonic()-t
        atomic_json(OUT/f'configs/{name}_{ds}_cache.json',{'support_seconds':support_seconds,'query_seconds':query_seconds,
                    'support_records':support_store.records,'query_records':query_store.records})
        for seed in SEEDS:
            cfg=config_for(ds,seed,name);support_ids=manifest[ds][str(seed)]['support_ids']
            mem,stats=make_memory(support_ids,support_cache,cfg,encoder,seed)
            mempath=OUT/f'memory/{name}/{ds}/seed_{seed}.npz';mem.save(mempath)
            write_csv(mempath.with_suffix('.csv'),stats)
            olddir=HISTORY/f'runs/{ds}/seed_{seed}/shots_20'
            if name=='uni':
                old=ReferenceMemory.load(olddir/'memory.npz')
                assert old.metadata['support_ids']==support_ids
                assert all(np.array_equal(old.vectors[c],mem.vectors[c]) for c in [0,1]),'UNI Memory regression failed'
            config_hash=signature({'config':cfg,'encoder':encoder.metadata,'support':manifest[ds][str(seed)],
                                   'algorithm_version':ALGORITHM_VERSION,'memory_content_fingerprint':signature({'metadata':mem.metadata,'vectors':{c:__import__('hashlib').sha256(v.tobytes()).hexdigest() for c,v in mem.vectors.items()}})})
            run_dir=OUT/f'predictions/{ds}/{name}/seed_{seed}';run_dir.mkdir(parents=True,exist_ok=True)
            atomic_json(OUT/f'configs/{name}_{ds}_{seed}.json',{'config':cfg,'fingerprint':config_hash,
                        'encoder':encoder.metadata,'support_ids':support_ids})
            pipeline=SegmentationPipeline(encoder,mem,cfg,feature_store=query_store)
            rows=[];torch.cuda.reset_peak_memory_stats();start=time.monotonic()
            oldrows={r['image_id']:r for r in csv.DictReader((olddir/'per_image_metrics.csv').open())} if name=='uni' else {}
            for n,entry in enumerate(entries,1):
                target=run_dir/f'{entry["id"]}.npz';result=None
                if target.exists():
                    with np.load(target) as z:
                        if str(z['config_fingerprint'])==config_hash:
                            result={'mask':z['mask'],'scores':z['margin']};elapsed=float(z['inference_seconds'])
                if result is None:
                    t=time.monotonic();result=pipeline.predict(load_image(entry['image']),entry['id']);elapsed=time.monotonic()-t
                    save_npz(target,mask=result['mask'],margin=result['scores'].astype(np.float32),
                             config_fingerprint=np.array(config_hash),inference_seconds=np.array(elapsed),
                             image_sha256=np.array(sha256(entry['image'])))
                from PIL import Image
                mask=result['mask'];assert mask.shape==(256,256)
                Image.fromarray(mask*255).save(run_dir/f'{entry["id"]}.png')
                gt=resize_mask(load_mask(entry['mask'],cfg['dataset']),(256,256)).astype(bool)
                row={'dataset':ds,'model':name,'seed':seed,'image_id':entry['id'],
                     **score_mask(mask.astype(bool),gt,cfg),'inference_seconds':elapsed,
                     'config_fingerprint':config_hash,'gt_sha256':sha256(entry['mask'])}
                rows.append(row)
                if name=='uni':
                    diffs={k:abs(row[k]-float(oldrows[entry['id']][k])) for k in METRICS}
                    assert max(diffs.values())<1e-9, f'UNI regression failed {ds} {seed} {entry["id"]}: {diffs}'
                if n==1 or n%20==0 or n==len(entries):log(f'{name} {ds} seed{seed}: predictions {n}/{len(entries)}')
            summary={'dataset':ds,'model':name,'seed':seed,'status':'COMPLETE','query_count':len(rows),
                     **{k:float(np.mean([r[k] for r in rows])) for k in METRICS},'config_fingerprint':config_hash,
                     'inference_total_seconds':sum(r['inference_seconds'] for r in rows),'run_seconds':time.monotonic()-start,
                     'support_precompute_seconds':support_seconds,'query_precompute_seconds':query_seconds,
                     'fg_tokens':len(mem.vectors[1]),'bg_tokens':len(mem.vectors[0]),'memory_bytes':sum(v.nbytes for v in mem.vectors.values()),
                     'memory_disk_bytes':mempath.stat().st_size,'peak_cuda_bytes':torch.cuda.max_memory_allocated()}
            write_csv(run_dir/'per_image_metrics.csv',rows);atomic_json(run_dir/'complete.json',summary)
            if name=='uni':
                regression.append({'dataset':ds,'seed':seed,'images':len(rows),'memory_exact':True,'metrics_max_abs_error':0.,'status':'PASS'})
                write_csv(OUT/'metrics/uni_regression.csv',regression)
            log(f'COMPLETE {name} {ds} seed{seed}: Dice={summary["dice"]:.6f}')
            del pipeline,mem;gc.collect();torch.cuda.empty_cache()
        del support_cache;gc.collect()
    del encoder;gc.collect();torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',default=MODELS);args=p.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True);manifest=freeze_protocol()
    atomic_json(OUT/'run_metadata.json',{'started_utc':datetime.now(timezone.utc).isoformat(),'command':sys.argv,
                'algorithm_version':ALGORITHM_VERSION,'seeds':SEEDS,'support_manifest_sha256':sha256(OUT/'configs/support_manifest.json'),
                'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda,
                'code_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in list((ROOT/'pathology_trainfree').glob('*.py'))+[Path(__file__)]},
                'boundary_definition':'GT interface pixels on both sides, Euclidean distance; <=4, (4,8], >8',
                'status':'RUNNING'})
    regression=[]
    for name in args.models:
        if name!='uni':
            reg=OUT/'metrics/uni_regression.csv'
            assert reg.exists() and len(list(csv.DictReader(reg.open())))==10,'UNI regression must pass before new models'
        try:run_model(name,manifest,regression)
        except Exception as exc:
            atomic_json(OUT/f'{name}_BLOCKED.json',{'status':'BLOCKED','reason':str(exc),'traceback':traceback.format_exc()})
            raise
    meta=json.loads((OUT/'run_metadata.json').read_text());meta['status']='SEGMENTATION_COMPLETE';atomic_json(OUT/'run_metadata.json',meta)

if __name__=='__main__':main()
