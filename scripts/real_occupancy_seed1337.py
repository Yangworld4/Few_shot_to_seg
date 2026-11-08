"""Fixed support-LOO occupancy, soft-memory and real-PCBP study (seed 1337)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from scipy import ndimage
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from scipy.stats import rankdata, pearsonr, spearmanr
from sklearn.isotonic import IsotonicRegression

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pathology_trainfree.config import load_config
from pathology_trainfree.data import read_manifest,load_image,load_mask
from pathology_trainfree.geometry import reconstruct
from pathology_trainfree.memory import ReferenceMemory,select_tokens
from pathology_trainfree.retrieval import SimilarityRetriever
from pathology_trainfree.shift_stitch import windows_xy
from study_support_size import load_support_cache
from native_diagnostics_seed1337 import (DC,PHASES,DEFAULT_OUT as NATIVE_OUT,
    evaluate,object_details,components,area_for,oracle_grids,pcbp,forward_boxes,operator_boxes,write_csv,sha)
from dense_feature_matching import guided_filter

OUT_DEFAULT=ROOT/'outputs/real_occupancy_diagnostics_seed1337'
TAUS=(.01,.02,.05)
BIN_NAMES=('pure_bg','bg_mixed','strong_mixed','fg_mixed','pure_fg')
MIX_BOUNDS=(0,.1,.3,.7,.9,1.0000001)


def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,default=str))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()


def setup(ds,args):
    manifest=args.dataset_root/ds/'manifests/paper_seed_2026'
    support=read_manifest(manifest/'labeled_10.csv',require_mask=True)
    test=read_manifest(manifest/'test.csv',require_mask=True)
    if (len(support),len(test))!=((7,80) if ds=='GlaS' else (16,40)):
        raise ValueError(f'Unexpected fixed split for {ds}')
    if set(e['id'] for e in support)&set(e['id'] for e in test):raise ValueError('Support/Test ID overlap')
    cfg=load_config(ROOT/'configs/uni_memory.yaml')
    cfg['window'].update(size=448,stride=224,offsets=[0,8],support_offsets=[0,8],padding_mode='reflect')
    cfg['dataset']=dict(DC,name=ds)
    cfg['retrieval'].update(top_k=50,chunk_size=784)
    cfg['output']['occupancy_cache_directory']=str(args.output_dir/'d2_occupancy/support_gt_cache')
    store,provenance=load_support_cache(support,args.dual_root,ds,cfg)
    source=args.dual_root/'outputs/n3_bil_distillation_glas_crag_10'/ds/'memory_N3.npz'
    with np.load(source,allow_pickle=False) as z:
        ids=z['source_ids'].tolist(); historical={0:z['bg'].copy(),1:z['fg'].copy()}
    if set(ids)!={e['id'] for e in support} or any(v.shape!=(len(ids)*512,1024) for v in historical.values()):
        raise ValueError('Historical memory source/shape mismatch')
    # Verify every image block before using it for leakage-safe LOO.
    rng=np.random.default_rng(2026)
    for i,id in enumerate(ids):
        if id not in store:raise ValueError(f'Missing Support cache {id}')
        feat=store[id]['features'];fg=store[id]['fg_occ']
        chosen,_=select_tokens(feat,np.stack([1-fg,fg],axis=1),cfg['memory'],rng)
        for cls in (0,1):
            if not np.array_equal(chosen[cls],historical[cls][i*512:(i+1)*512]):
                raise ValueError(f'Historical Memory block/source mismatch {ds}/{id}/class{cls}')
    return dict(ds=ds,support=support,test=test,cfg=cfg,store=store,provenance=provenance,
                source=source,source_sha256=sha(source),source_ids=ids,historical=historical)


def valid_occ(mask,phase):
    occ,pos,boxes=oracle_grids(mask,phase)
    return occ.ravel(),pos,boxes


def support_arrays(st):
    result={}
    for e in st['support']:
        iid=e['id'];gt=load_mask(e['mask'],DC)
        passes=[valid_occ(gt,p) for p in PHASES[2]]
        occ=np.concatenate([p[0] for p in passes]).astype(np.float32)
        feat=st['store'][iid]['features']
        if len(occ)!=len(feat):raise ValueError(f'Support token alignment {iid}')
        result[iid]=(feat,occ)
    return result


def sample_soft(arrays,ids,kind,budget_per_image=1024):
    parts=[];labels=[];sources=[];bin_rows=[]
    for image_index,iid in enumerate(ids):
        feat,occ=arrays[iid]
        rng=np.random.default_rng(np.random.SeedSequence([1337,image_index,0 if kind=='random' else 1]))
        if len(occ)<budget_per_image:raise ValueError(f'Insufficient support tokens {iid}')
        binid=np.minimum((occ*5).astype(np.int32),4)
        chosen=[]
        if kind=='stratified':
            for b in range(5):
                candidates=np.flatnonzero(binid==b)
                take=min(len(candidates),budget_per_image//5+(b<budget_per_image%5))
                if take:chosen.extend(rng.choice(candidates,take,replace=False).tolist())
            remaining=budget_per_image-len(chosen)
            if remaining:
                pool=np.setdiff1d(np.arange(len(occ)),np.asarray(chosen,dtype=np.int64),assume_unique=True)
                chosen.extend(rng.choice(pool,remaining,replace=False).tolist())
        else:
            chosen=rng.choice(len(occ),budget_per_image,replace=False).tolist()
        chosen=np.sort(np.asarray(chosen,dtype=np.int64))
        parts.append(feat[chosen]);labels.append(occ[chosen]);sources.extend([iid]*len(chosen))
        for b in range(5):bin_rows.append(dict(image_id=iid,bin=b,kind=kind,candidates=int((binid==b).sum()),selected=int((binid[chosen]==b).sum())))
    return dict(features=np.concatenate(parts),occupancy=np.concatenate(labels),source=np.asarray(sources),bin_rows=bin_rows)


class SoftRetriever:
    def __init__(self,memory,device='cuda',k=50,chunk=784):
        self.features=F.normalize(torch.as_tensor(memory['features'].astype(np.float32),device=device),dim=1)
        self.labels=torch.as_tensor(memory['occupancy'],dtype=torch.float32,device=device)
        self.device=device;self.k=k;self.chunk=chunk
        if len(self.labels)<k:raise ValueError('Memory smaller than K')

    @torch.inference_mode()
    def predict_many(self,features,taus=TAUS):
        q=F.normalize(torch.as_tensor(features,dtype=torch.float32,device=self.device).reshape(-1,1024),dim=1)
        result={t:[] for t in taus}
        for batch in q.split(self.chunk):
            scores,indices=(batch@self.features.T).topk(self.k,dim=1)
            occ=self.labels[indices]
            for t in taus:
                weights=torch.softmax(scores/t,dim=1)
                result[t].append((weights*occ).sum(1).cpu().numpy())
        return {t:np.concatenate(v).astype(np.float32) for t,v in result.items()}


def historical_loo(st,iid):
    ids=st['source_ids'];i=ids.index(iid)
    bank={c:np.concatenate((v[:i*512],v[(i+1)*512:])) for c,v in st['historical'].items()}
    retr=SimilarityRetriever(ReferenceMemory(bank,{}),st['cfg'],'cuda')
    return (retr.scores(st['store'][iid]['features']).numpy()[:,1]-
            retr.scores(st['store'][iid]['features']).numpy()[:,0]).astype(np.float32)


def loo_stage(st,arrays,args):
    out=args.output_dir/'d2_occupancy/support_loo'/st['ds'];out.mkdir(parents=True,exist_ok=True)
    soft=sample_soft(arrays,st['source_ids'],'stratified')
    dump(args.output_dir/'configs'/f"{st['ds']}_support.json",dict(ids=st['source_ids'],historical_memory_sha256=st['source_sha256'],
        checkpoint_sha256='56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40',
        support_provenance=st['provenance'],soft_budget_per_image=1024))
    for index,iid in enumerate(st['source_ids'],1):
        path=out/f'{iid}.npz'
        key=fingerprint(dict(dataset=st['ds'],source_id=iid,source_memory_sha256=st['source_sha256'],
                             support_ids=st['source_ids'],soft_seed=1337,feature_sha256=st['provenance'][next(j for j,p in enumerate(st['provenance']) if p['image_id']==iid)]['source_sha256']))
        if path.exists():
            with np.load(path,allow_pickle=False) as z:
                if str(z['fingerprint'])!=key:raise ValueError(f'LOO cache mismatch {path}')
            print('reuse loo',st['ds'],iid,flush=True);continue
        feat,gt=arrays[iid]
        i=st['source_ids'].index(iid)
        bank={c:np.concatenate((v[:i*512],v[(i+1)*512:])) for c,v in st['historical'].items()}
        retr=SimilarityRetriever(ReferenceMemory(bank,{}),st['cfg'],'cuda')
        scores=retr.scores(feat).numpy();margin=(scores[:,1]-scores[:,0]).astype(np.float32)
        keep=soft['source']!=iid
        sret=SoftRetriever(dict(features=soft['features'][keep],occupancy=soft['occupancy'][keep]))
        pred=sret.predict_many(feat)
        np.savez_compressed(path,gt=gt.astype(np.float16),margin=margin.astype(np.float16),
            **{f'soft_{str(t).replace(".","p")}':v.astype(np.float16) for t,v in pred.items()},
            fingerprint=np.array(key),source_id=np.array(iid))
        print('loo',st['ds'],index,len(st['source_ids']),iid,flush=True)
    return soft


def load_loo(st,args):
    result={}
    for iid in st['source_ids']:
        with np.load(args.output_dir/'d2_occupancy/support_loo'/st['ds']/f'{iid}.npz',allow_pickle=False) as z:
            result[iid]={k:z[k].astype(np.float32) for k in z.files if k in ('gt','margin','soft_0p01','soft_0p02','soft_0p05')}
    return result


def fit_sigmoid(x,y,logit_input=False):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    x=np.clip(x,1e-5,1-1e-5) if logit_input else x
    if logit_input:x=logit(x)
    def loss_grad(p):
        pred=expit(p[0]*x+p[1]);err=pred-y
        local=2*err*pred*(1-pred)
        return float(np.mean(err**2)),np.array([np.mean(local*x),np.mean(local)],dtype=np.float64)
    result=minimize(loss_grad,[1.,0.],jac=True,method='L-BFGS-B',bounds=[(0,None),(None,None)])
    if not result.success:raise RuntimeError(f'Sigmoid calibration failed: {result.message}')
    return ('sigmoid',float(result.x[0]),float(result.x[1]),bool(logit_input))


def fit_iso(x,y):
    model=IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip').fit(x,y)
    return ('isotonic',model.X_thresholds_.tolist(),model.y_thresholds_.tolist())


def apply_cal(model,x):
    if model[0]=='sigmoid':
        a,b,logit_input=model[1:]
        return expit(a*(logit(np.clip(x,1e-5,1-1e-5)) if logit_input else x)+b).astype(np.float32)
    return np.interp(x,model[1],model[2]).astype(np.float32)


def balanced_training(loo,ids,key):
    xx=[];yy=[]
    for iid in ids:
        r=loo[iid];n=len(r['gt']);idx=np.linspace(0,n-1,min(3000,n),dtype=np.int64)
        xx.append(r[key][idx]);yy.append(r['gt'][idx])
    return np.concatenate(xx),np.concatenate(yy)


def occ_metrics(pred,truth):
    pred=np.clip(np.asarray(pred,np.float32),0,1);truth=np.asarray(truth,np.float32)
    err=pred-truth;label=truth>.5
    bins=np.minimum((pred*10).astype(int),9);ece=0.
    for b in range(10):
        sel=bins==b
        if sel.any():ece+=sel.mean()*abs(float(pred[sel].mean()-truth[sel].mean()))
    n1=int(label.sum());n0=len(label)-n1
    auc=float((rankdata(pred)[label].sum()-n1*(n1+1)/2)/(n1*n0)) if n1 and n0 else np.nan
    values=dict(mae=float(np.abs(err).mean()),rmse=float(np.sqrt(np.square(err).mean())),ece=float(ece),
        pearson=float(pearsonr(pred,truth).statistic) if np.std(pred)>0 and np.std(truth)>0 else np.nan,
        spearman=float(spearmanr(pred,truth).statistic) if np.std(pred)>0 and np.std(truth)>0 else np.nan,
        fg_accuracy=float(((pred>.5)==label).mean()),fg_auroc=auc,tokens=len(pred))
    for i,name in enumerate(BIN_NAMES):
        sel=(truth>=MIX_BOUNDS[i])&(truth<MIX_BOUNDS[i+1])
        values[f'{name}_n']=int(sel.sum());values[f'{name}_mae']=float(np.abs(err[sel]).mean()) if sel.any() else np.nan
        values[f'{name}_rmse']=float(np.sqrt(np.square(err[sel]).mean())) if sel.any() else np.nan
    sel=(truth>.1)&(truth<.9)
    values['mixed_n']=int(sel.sum());values['mixed_mae']=float(np.abs(err[sel]).mean()) if sel.any() else np.nan
    values['mixed_rmse']=float(np.sqrt(np.square(err[sel]).mean())) if sel.any() else np.nan
    values['mean_pred']=float(pred.mean());values['mean_gt']=float(truth.mean())
    return values


def calibrate_stage(st,args):
    loo=load_loo(st,args);ids=st['source_ids'];rows=[]
    # Outer image LOO: every reported calibrated prediction excludes its own image.
    outer={}
    for iid in ids:
        train=[j for j in ids if j!=iid]
        x,y=balanced_training(loo,train,'margin')
        k=minimize_scalar(lambda value:float(np.mean((expit(value*x)-y)**2)),bounds=(.1,100),method='bounded').x
        outer[(iid,'M0')]=expit(k*loo[iid]['margin'])
        for kind,model in (('M1_sigmoid',fit_sigmoid(x,y)),('M1_isotonic',fit_iso(x,y))):
            outer[(iid,kind)]=apply_cal(model,loo[iid]['margin'])
        # Select temperature using only other-image OOF predictions.
        t=min(TAUS,key=lambda t:np.mean([np.abs(loo[j][f'soft_{str(t).replace(".","p")}']-loo[j]['gt']).mean() for j in train]))
        key=f'soft_{str(t).replace(".","p")}'
        outer[(iid,'M2_raw')]=loo[iid][key]
        x,y=balanced_training(loo,train,key)
        outer[(iid,'M2_sigmoid')]=apply_cal(fit_sigmoid(x,y,True),loo[iid][key])
        outer[(iid,'M2_isotonic')]=apply_cal(fit_iso(x,y),loo[iid][key])
        for method in ('M0','M1_sigmoid','M1_isotonic','M2_raw','M2_sigmoid','M2_isotonic'):
            rows.append(dict(dataset=st['ds'],image_id=iid,method=method,**occ_metrics(outer[(iid,method)],loo[iid]['gt'])))
        outer_path=args.output_dir/'d2_occupancy/calibration/outer_predictions'/st['ds']/f'{iid}.npz'
        outer_path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(outer_path,gt=loo[iid]['gt'].astype(np.float32),
            **{method:outer[(iid,method)].astype(np.float32) for method in
               ('M0','M1_sigmoid','M1_isotonic','M2_raw','M2_sigmoid','M2_isotonic')})
    write_csv(args.output_dir/'d2_occupancy/tables'/f"{st['ds']}_support_loo_metrics.csv",rows)
    aggregate={m:np.nanmean([r['mae'] for r in rows if r['method']==m]) for m in ('M1_sigmoid','M1_isotonic','M2_sigmoid','M2_isotonic')}
    m1=min(('M1_sigmoid','M1_isotonic'),key=aggregate.get)
    m2=min(('M2_sigmoid','M2_isotonic'),key=aggregate.get)
    chosen_tau=min(TAUS,key=lambda t:np.mean([np.abs(loo[j][f'soft_{str(t).replace(".","p")}']-loo[j]['gt']).mean() for j in ids]))
    x,y=balanced_training(loo,ids,'margin')
    k=minimize_scalar(lambda value:float(np.mean((expit(value*x)-y)**2)),bounds=(.1,100),method='bounded').x
    m1_model=fit_sigmoid(x,y) if m1=='M1_sigmoid' else fit_iso(x,y)
    softkey=f'soft_{str(chosen_tau).replace(".","p")}'
    x,y=balanced_training(loo,ids,softkey)
    m2_model=fit_sigmoid(x,y,True) if m2=='M2_sigmoid' else fit_iso(x,y)
    config=dict(dataset=st['ds'],k=50,temperature=chosen_tau,raw_margin_kappa=float(k),
                m1_method=m1,m1_model=m1_model,m2_method=m2,m2_model=m2_model,
                selection='support-image outer LOO macro MAE; final fit on all support OOF; no test GT',
                outer_macro_mae=aggregate,support_ids=ids)
    dump(args.output_dir/'d2_occupancy/calibration'/f"{st['ds']}.json",config)
    print('calibration',st['ds'],m1,m2,chosen_tau,aggregate,flush=True)
    return config


def memory_stage(st,arrays,args):
    result={};rows=[]
    for kind in ('random','stratified'):
        mem=sample_soft(arrays,st['source_ids'],kind)
        expected=1024*len(st['source_ids'])
        if len(mem['occupancy'])!=expected:raise ValueError('Soft memory budget mismatch')
        path=args.output_dir/'p1_soft_memory/memories'/f"{st['ds']}_{kind}.npz"
        path.parent.mkdir(parents=True,exist_ok=True)
        config=dict(dataset=st['ds'],kind=kind,seed=1337,source_ids=st['source_ids'],
                    budget=expected,feature_dim=1024,occupancy='valid original pixels; neutral fully padded tokens',
                    baseline_memory_sha256=st['source_sha256'],support_feature_sha256={p['image_id']:p['source_sha256'] for p in st['provenance']})
        key=fingerprint(config)
        if path.exists():
            with np.load(path,allow_pickle=False) as z:
                if str(z['fingerprint'])!=key:raise ValueError(f'Soft Memory cache mismatch {path}')
        else:
            np.savez_compressed(path,features=mem['features'],occupancy=mem['occupancy'].astype(np.float16),
                source=mem['source'],fingerprint=np.array(key),metadata=np.array(json.dumps(config)))
        result[kind]=dict(**mem,fingerprint=key,path=path)
        for b in range(5):
            candidates=sum(r['candidates'] for r in mem['bin_rows'] if r['bin']==b)
            selected=sum(r['selected'] for r in mem['bin_rows'] if r['bin']==b)
            rows.append(dict(dataset=st['ds'],memory=kind,bin=b,candidates=candidates,selected=selected,
                fraction=selected/expected,budget=expected,feature_bytes=mem['features'].nbytes,
                occupancy_bytes=mem['occupancy'].nbytes,source_bytes=mem['source'].nbytes))
        print('memory',st['ds'],kind,expected,flush=True)
    write_csv(args.output_dir/'p1_soft_memory/tables'/f"{st['ds']}_memory_bins.csv",rows)
    return result


def load_query_features(st,e,phase,args,image=None):
    image=load_image(e['image']) if image is None else image
    dy,dx=phase
    path=args.phase_cache_dir/st['ds']/f"{e['id']}_dy{dy}_dx{dx}.npz"
    with np.load(path,allow_pickle=False) as z:
        tokens=z['tokens'].copy();positions=z['positions'].copy();key=json.loads(str(z['key_json']))
    expected=np.array([(y,x) for y,x,_ in windows_xy(image,448,224,phase,'reflect')])
    if key['image_sha256']!=hashlib.sha256(image.tobytes()).hexdigest() or key['offset_y']!=dy or key['offset_x']!=dx:
        raise ValueError(f'Query feature fingerprint mismatch {path}')
    if key['weights_sha256']!='56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40' or key['window']!=448 or key['stride']!=224:
        raise ValueError(f'Query UNI/config mismatch {path}')
    if tokens.shape!=(len(expected),784,1024) or not np.array_equal(positions,expected):
        raise ValueError(f'Query feature geometry mismatch {path}')
    return tokens,positions,key,path


def query_soft_stage(st,memories,cal,args):
    retrievers={kind:SoftRetriever(memories[kind],k=50) for kind in ('random','stratified')}
    phase_list={'random':PHASES[2],'stratified':PHASES[16]}
    rows=[]
    for index,e in enumerate(st['test'],1):
        image=load_image(e['image'])
        image_sha=hashlib.sha256(image.tobytes()).hexdigest()
        for kind,phases in phase_list.items():
            retr=retrievers[kind]
            for phase in phases:
                dy,dx=phase
                out=args.output_dir/'p1_soft_memory/token_predictions'/st['ds']/kind/f"{e['id']}_dy{dy}_dx{dx}.npz"
                out.parent.mkdir(parents=True,exist_ok=True)
                cachekey=fingerprint(dict(dataset=st['ds'],image_id=e['id'],phase=phase,
                    memory=memories[kind]['fingerprint'],temperature=cal['temperature'],k=50))
                if out.exists():
                    with np.load(out,allow_pickle=False) as z:
                        if str(z['fingerprint'])!=cachekey:raise ValueError(f'Token prediction cache mismatch {out}')
                        seconds=float(z['retrieval_seconds'])
                        stored_feature_key=str(z['source_feature_key'])
                    feature_path=args.phase_cache_dir/st['ds']/f"{e['id']}_dy{dy}_dx{dx}.npz"
                    with np.load(feature_path,allow_pickle=False) as f:
                        feature_meta=json.loads(str(f['key_json']))
                    if feature_meta['image_sha256']!=image_sha or fingerprint(feature_meta)!=stored_feature_key:
                        raise ValueError(f'Cached Token prediction source image/feature changed: {out}')
                    hit=True
                else:
                    tokens,positions,key,feature_path=load_query_features(st,e,phase,args,image)
                    start=time.perf_counter()
                    pred=retr.predict_many(tokens,(cal['temperature'],))[cal['temperature']].reshape(len(positions),28,28)
                    seconds=time.perf_counter()-start
                    np.savez_compressed(out,occupancy=pred.astype(np.float16),positions=positions,
                        phase=np.array(phase),image_id=np.array(e['id']),window=np.array(448),stride=np.array(224),
                        source_feature_path=np.array(str(feature_path)),source_feature_key=np.array(fingerprint(key)),
                        valid_shape=np.array(image.shape[:2]),retrieval_seconds=np.array(seconds),
                        fingerprint=np.array(cachekey))
                    hit=False
                rows.append(dict(dataset=st['ds'],image_id=e['id'],memory=kind,dy=dy,dx=dx,
                    output=str(out),cache_hit=hit,retrieval_seconds=seconds))
        if index%10==0:print('query soft',st['ds'],index,len(st['test']),flush=True)
        write_csv(args.output_dir/'p1_soft_memory/tables'/f"{st['ds']}_query_cache.csv",rows)
    return rows


def token_prediction(st,e,kind,phase,args):
    dy,dx=phase
    path=args.output_dir/'p1_soft_memory/token_predictions'/st['ds']/kind/f"{e['id']}_dy{dy}_dx{dx}.npz"
    with np.load(path,allow_pickle=False) as z:
        occ=z['occupancy'].astype(np.float32);pos=z['positions'].copy()
        if str(z['image_id'])!=e['id'] or tuple(z['phase'])!=phase:
            raise ValueError('Token prediction ID/phase mismatch')
    return occ,pos


def p0_token_prediction(st,e,phase,cal,args):
    suffix=f'dy{phase[0]}_dx{phase[1]}'
    with np.load(args.phase_margin_dir/st['ds']/f"{e['id']}.npz",allow_pickle=False) as z:
        margin=z[f'margin_{suffix}'].astype(np.float32);pos=z[f'positions_{suffix}'].copy()
    return apply_cal(cal['m1_model'],margin),pos


def bilinear_from_tokens(st,e,method,phases,cal,args):
    shape=load_image(e['image']).shape[:2]
    maps=[];observations=[]
    for phase in phases:
        if method=='P0-cal':occ,pos=p0_token_prediction(st,e,phase,cal,args)
        else:
            kind='random' if method=='P1-a' else 'stratified'
            occ,pos=token_prediction(st,e,kind,phase,args)
            if method=='P1-c':occ=apply_cal(cal['m2_model'],occ)
        if occ.shape!=(len(pos),28,28):raise ValueError('Token occupancy grid mismatch')
        # Geometry-only call; no Test GT enters the observation or reconstruction.
        _,expected,boxes=oracle_grids(np.zeros(shape,np.uint8),phase)
        if not np.array_equal(pos,expected):raise ValueError('Token/PCBP coordinate mismatch')
        maps.append(reconstruct(2*occ-1,pos,shape,448,'float16'))
        observations.append((occ.ravel(),boxes))
    margin=np.mean(np.stack(maps),axis=0,dtype=np.float32)
    return margin,observations


def p1_bilinear_stage(st,cal,args):
    # Predictions are written before loading any Test labels.
    methods=('P0','P0-cal','P1-a','P1-b','P1-c')
    import pandas as pd
    old_metric_path=args.output_dir/'p1_soft_memory/tables'/f"{st['ds']}_per_image.csv"
    old_occ_path=args.output_dir/'p1_soft_memory/tables'/f"{st['ds']}_occupancy_per_image.csv"
    old_metrics={(r.image_id,r.method):r._asdict() for r in pd.read_csv(old_metric_path).itertuples(index=False)} if old_metric_path.exists() else {}
    old_occ={(r.image_id,r.method):r._asdict() for r in pd.read_csv(old_occ_path).itertuples(index=False)} if old_occ_path.exists() else {}
    timings={}
    for index,e in enumerate(st['test'],1):
        iid=e['id']
        for method in methods:
            path=args.output_dir/'p1_soft_memory/predictions'/st['ds']/f'{iid}_{method}.npz'
            path.parent.mkdir(parents=True,exist_ok=True)
            method_model=cal['m1_model'] if method=='P0-cal' else (cal['m2_model'] if method=='P1-c' else None)
            cachekey=fingerprint(dict(dataset=st['ds'],image_id=iid,method=method,
                                      calibration=method_model,temperature=cal['temperature'],
                                      baseline_memory_sha256=st['source_sha256'],geometry='native_bilinear_v1'))
            if path.exists() and method in ('P0','P1-a','P1-b') and (iid,method) in old_metrics:
                continue
            if path.exists():
                with np.load(path,allow_pickle=False) as z:
                    if 'fingerprint' in z.files and str(z['fingerprint'])==cachekey:continue
            start=time.perf_counter()
            if method=='P0':
                with np.load(args.native_baseline_dir/'baseline'/st['ds']/f'{iid}.npz',allow_pickle=False) as z:
                    margin=z['margin'].copy()
                # Native P0 is the validated original baseline, not an occupancy remapping.
                occupancy=np.full_like(margin,np.nan,dtype=np.float32)
            else:
                margin,_=bilinear_from_tokens(st,e,method,PHASES[2],cal,args)
                occupancy=np.clip((margin+1)/2,0,1)
            seconds=time.perf_counter()-start
            np.savez_compressed(path,margin=margin.astype(np.float32),occupancy=occupancy,
                mask=(margin>0).astype(np.uint8),image_id=np.array(iid),method=np.array(method),
                reconstruction_seconds=np.array(seconds),fingerprint=np.array(cachekey))
            timings[(iid,method)]=seconds
        if index%10==0:print('p1 bilinear',st['ds'],index,len(st['test']),flush=True)
    rows=[];occupancy_rows=[]
    for e in st['test']:
        iid=e['id'];gt=load_mask(e['mask'],DC)
        for method in methods:
            if method in ('P0','P1-a','P1-b') and (iid,method) in old_metrics:
                rows.append(old_metrics[(iid,method)]);continue
            with np.load(args.output_dir/'p1_soft_memory/predictions'/st['ds']/f'{iid}_{method}.npz',allow_pickle=False) as z:
                margin=z['margin'].copy();seconds=float(z['reconstruction_seconds'])
            if margin.shape!=gt.shape:raise ValueError(f'Native size mismatch {iid}/{method}')
            vals,_=evaluate(margin,gt,area_for(gt.shape,'native'))
            rows.append(dict(dataset=st['ds'],image_id=iid,method=method,reconstruction_seconds=seconds,**vals))
        for method in ('P0-cal','P1-a','P1-b','P1-c'):
            if method in ('P1-a','P1-b') and (iid,method) in old_occ:
                occupancy_rows.append(old_occ[(iid,method)]);continue
            pred=[];truth=[]
            for phase in PHASES[2]:
                if method=='P0-cal':occ,_=p0_token_prediction(st,e,phase,cal,args)
                else:
                    occ,_=token_prediction(st,e,'random' if method=='P1-a' else 'stratified',phase,args)
                    if method=='P1-c':occ=apply_cal(cal['m2_model'],occ)
                go,_,_=valid_occ(gt,phase)
                pred.append(occ.ravel());truth.append(go)
            occupancy_rows.append(dict(dataset=st['ds'],image_id=iid,method=method,
                **occ_metrics(np.concatenate(pred),np.concatenate(truth))))
    write_csv(args.output_dir/'p1_soft_memory/tables'/f"{st['ds']}_per_image.csv",rows)
    write_csv(args.output_dir/'p1_soft_memory/tables'/f"{st['ds']}_occupancy_per_image.csv",occupancy_rows)
    return rows


def real_observation(st,e,source,phase,cal,args,shape):
    dy,dx=phase
    path=args.output_dir/'u1_real_pcbp/token_observations'/st['ds']/source/f"{e['id']}_dy{dy}_dx{dx}.npz"
    key=fingerprint(dict(dataset=st['ds'],image_id=e['id'],phase=phase,source=source,
                         calibration=cal['m1_model'] if source=='P0-cal' else cal['m2_model'],
                         memory_sha256=st['source_sha256'] if source=='P0-cal' else st['soft_memory_sha256'],
                         geometry='valid-box-v1'))
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            if str(z['fingerprint'])!=key:raise ValueError(f'Real observation cache mismatch {path}')
            return z['occupancy'].astype(np.float32),z['positions'].copy(),z['boxes'].copy()
    if source=='P0-cal':occ,pos=p0_token_prediction(st,e,phase,cal,args)
    else:
        occ,pos=token_prediction(st,e,'stratified',phase,args)
        occ=apply_cal(cal['m2_model'],occ)
    _,expected,boxes=oracle_grids(np.zeros(shape,np.uint8),phase)
    if not np.array_equal(pos,expected) or occ.shape!=(len(pos),28,28):
        raise ValueError(f'Real observation geometry mismatch {st["ds"]}/{e["id"]}/{phase}')
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,occupancy=occ.astype(np.float32),positions=pos,boxes=boxes.astype(np.int32),
                        fingerprint=np.array(key),image_id=np.array(e['id']),phase=np.array(phase),
                        shape=np.array(shape),window=np.array(448),stride=np.array(224))
    return occ.astype(np.float32),pos,boxes


def bridge_details(pred,gt,area):
    pl,pa=components(pred,area);gl,ga=components(gt,area)
    if not len(pa) or not len(ga):return dict(narrow_bridge_merge=0,merge_count=0)
    overlap=np.bincount((pl.ravel().astype(np.int64)*(len(ga)+1)+gl.ravel()),
                        minlength=(len(pa)+1)*(len(ga)+1)).reshape(len(pa)+1,len(ga)+1)[1:,1:]
    links=(overlap>=max(10,round(area*.2)))&(overlap/np.maximum(1,np.minimum(pa[:,None],ga[None,:]))>=.1)
    merged=np.flatnonzero(links.sum(1)>=2)
    narrow=0
    slices=ndimage.find_objects(pl)
    for i in merged:
        sl=slices[i]
        component=pl[sl]==i+1
        eroded=ndimage.binary_erosion(component,structure=np.ones((3,3)),iterations=2)
        pieces,n=ndimage.label(eroded,structure=np.ones((3,3)))
        if n<2:continue
        linked=np.flatnonzero(links[i])+1
        hit=0
        for j in range(1,n+1):
            if (pieces==j).sum()<max(5,round(area*.05)):continue
            present=np.unique(gl[sl][pieces==j])
            if np.intersect1d(present,linked).size:hit+=1
        if hit>=2:narrow+=1
    return dict(narrow_bridge_merge=narrow,merge_count=len(merged))


def u1_stage(st,cal,args):
    st['soft_memory_sha256']=sha(args.output_dir/'p1_soft_memory/memories'/f"{st['ds']}_stratified.npz")
    import pandas as pd
    metric_path=args.output_dir/'u1_real_pcbp/tables'/f"{st['ds']}_per_image.csv"
    residual_path=args.output_dir/'u1_real_pcbp/residuals'/f"{st['ds']}.csv"
    all_rows=pd.read_csv(metric_path).to_dict('records') if metric_path.exists() else []
    all_res=pd.read_csv(residual_path).to_dict('records') if residual_path.exists() else []
    complete={iid for iid in set(r['image_id'] for r in all_rows)
              if sum(r['image_id']==iid for r in all_rows)==12 and sum(r['image_id']==iid for r in all_res)==36}
    all_rows=[r for r in all_rows if r['image_id'] in complete]
    all_res=[r for r in all_res if r['image_id'] in complete]
    for index,e in enumerate(st['test'],1):
        iid=e['id'];image=load_image(e['image']);shape=image.shape[:2]
        if iid in complete:
            if index%5==0:print('u1 reuse',st['ds'],index,len(st['test']),flush=True)
            continue
        guide=cv2.cvtColor(image.astype(np.float32)/255,cv2.COLOR_RGB2GRAY)
        # The same saved Token observations feed all three reconstruction methods.
        case_rows=[];case_res=[]
        for source in ('P0-cal','P1-c'):
            phase_map={}
            for phase in PHASES[16]:
                occ,pos,boxes=real_observation(st,e,source,phase,cal,args,shape)
                phase_map[phase]=(occ,pos,boxes)
            for count in (2,16):
                start=time.perf_counter();maps=[];obs=[]
                for phase in PHASES[count]:
                    occ,pos,boxes=phase_map[phase]
                    maps.append(reconstruct(2*occ-1,pos,shape,448,'float16'))
                    obs.append((occ.ravel(),boxes))
                bil=np.mean(np.stack(maps),axis=0,dtype=np.float32)
                bil_seconds=time.perf_counter()-start
                boxes,target,weights=operator_boxes(obs)
                def residual(x):
                    delta=forward_boxes(x,boxes)-target
                    return float(np.dot(weights.astype(np.float64),delta.astype(np.float64)**2)/weights.sum())
                bilinear_occupancy=np.clip((bil.astype(np.float64)+1)/2,0,1)
                init=bilinear_occupancy.astype(np.float32)
                initial=residual(init)
                start=time.perf_counter()
                estimate,trajectory,_=pcbp(init,obs,iterations=8)
                pc_seconds=time.perf_counter()-start
                start=time.perf_counter()
                jbu=np.clip(guided_filter(guide,estimate,r=4,eps=.001),0,1)
                jbu_seconds=time.perf_counter()-start
                versions=(('bilinear',bilinear_occupancy,initial,bil_seconds),('pcbp_no_prior',estimate,trajectory[-1],pc_seconds),
                          ('pcbp_jbu_rgb',jbu,residual(jbu),pc_seconds+jbu_seconds))
                for method,occupancy,obs_res,seconds in versions:
                    path=args.output_dir/'u1_real_pcbp/predictions'/st['ds']/f'{iid}_{source}_phase{count}_{method}.npz'
                    path.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(path,occupancy=occupancy.astype(np.float16),mask=(occupancy>.5).astype(np.uint8),
                        image_id=np.array(iid),source=np.array(source),phase_count=np.array(count),method=np.array(method),
                        observation_residual=np.array(obs_res),reconstruction_seconds=np.array(seconds))
                    # Read GT only after all three predictions for this condition are saved.
                gt=load_mask(e['mask'],DC)
                for method,occupancy,obs_res,seconds in versions:
                    margin=bil if method=='bilinear' else 2*occupancy-1
                    values,details=evaluate(margin,gt,area_for(gt.shape,'native'))
                    bridges=bridge_details(occupancy>.5,gt.astype(bool),area_for(gt.shape,'native'))
                    case_rows.append(dict(dataset=st['ds'],image_id=iid,source=source,phase_count=count,method=method,
                        observation_residual=obs_res,initial_residual=initial,reconstruction_seconds=seconds,
                        **values,**details,narrow_bridge_merge=bridges['narrow_bridge_merge']))
                for iteration,res in enumerate(trajectory):
                    case_res.append(dict(dataset=st['ds'],image_id=iid,source=source,phase_count=count,
                                         iteration=iteration,observation_residual=res))
                # The two-phase Bilinear path must reproduce the P1 prediction with identical tokens.
                if count==2:
                    with np.load(args.output_dir/'p1_soft_memory/predictions'/st['ds']/f'{iid}_{source}.npz',allow_pickle=False) as z:
                        saved=z['mask']
                    if not np.array_equal(saved,bilinear_occupancy>.5):raise ValueError(f'P1/U1 bilinear parity failed {iid}/{source}')
        all_rows.extend(case_rows);all_res.extend(case_res)
        write_csv(args.output_dir/'u1_real_pcbp/tables'/f"{st['ds']}_per_image.csv",all_rows)
        write_csv(args.output_dir/'u1_real_pcbp/residuals'/f"{st['ds']}.csv",all_res)
        if index%5==0:print('u1',st['ds'],index,len(st['test']),flush=True)
    return all_rows


def paired_effects(values,metric,seed=2026):
    x=np.asarray(values,dtype=np.float64)
    rng=np.random.default_rng(seed)
    draws=x[rng.integers(0,len(x),(10000,len(x)))].mean(1)
    return dict(metric=metric,mean=float(x.mean()),sd=float(x.std(ddof=1)),median=float(np.median(x)),
                ci95_low=float(np.quantile(draws,.025)),ci95_high=float(np.quantile(draws,.975)),
                improved=int((x<0).sum()) if metric in ('object_hd','merge_count','narrow_bridge_merge') else int((x>0).sum()),
                declined=int((x>0).sum()) if metric in ('object_hd','merge_count','narrow_bridge_merge') else int((x<0).sum()),n=len(x))


def factorial_stage(args,datasets):
    import pandas as pd
    fields=('dice','fg_iou','boundary_iou','bf1_2px','bf1_4px','object_f1','object_dice','object_hd')
    rows=[];summary=[]
    for ds in datasets:
        u=pd.read_csv(args.output_dir/'u1_real_pcbp/tables'/f'{ds}_per_image.csv')
        expected=80 if ds=='GlaS' else 40
        for count in (2,16):
            groups={}
            for source in ('P0-cal','P1-c'):
                for method in ('bilinear','pcbp_jbu_rgb'):
                    sub=u[(u.source==source)&(u.phase_count==count)&(u.method==method)].set_index('image_id')
                    if len(sub)!=expected or not sub.index.is_unique:raise ValueError(f'Incomplete factorial cell {ds}/{count}/{source}/{method}')
                    groups[(source,method)]=sub
            ids=groups[('P0-cal','bilinear')].index
            if any(set(g.index)!=set(ids) for g in groups.values()):raise ValueError('Factorial image IDs differ')
            a=groups[('P0-cal','bilinear')].loc[ids]
            b=groups[('P0-cal','pcbp_jbu_rgb')].loc[ids]
            c=groups[('P1-c','bilinear')].loc[ids]
            d=groups[('P1-c','pcbp_jbu_rgb')].loc[ids]
            for metric in fields:
                effects={'retrieval':c[metric]-a[metric],
                         'reconstruction':b[metric]-a[metric],
                         'interaction':d[metric]-c[metric]-b[metric]+a[metric],
                         'combined':d[metric]-a[metric]}
                for effect,delta in effects.items():
                    rows.extend(dict(dataset=ds,image_id=iid,phase_count=count,effect=effect,metric=metric,delta=float(value))
                                for iid,value in zip(ids,delta))
                    summary.append(dict(dataset=ds,phase_count=count,effect=effect,**paired_effects(delta,metric)))
    write_csv(args.output_dir/'factorial/per_image.csv',rows)
    write_csv(args.output_dir/'factorial/summary.csv',summary)
    return pd.DataFrame(rows),pd.DataFrame(summary)


def finalize_observation_cache(args,datasets):
    for ds in datasets:
        expected=(80 if ds=='GlaS' else 40)*16*2
        paths=list((args.output_dir/'u1_real_pcbp/token_observations'/ds).glob('*/*.npz'))
        if len(paths)!=expected:raise ValueError(f'Incomplete real Token observation cache {ds}: {len(paths)}/{expected}')
        for path in paths:
            with np.load(path,allow_pickle=False) as z:
                data={k:z[k].copy() for k in z.files}
            occ=data['occupancy'];pos=data['positions'];boxes=data['boxes']
            if occ.shape!=(len(pos),28,28) or boxes.shape!=(occ.size,4):
                raise ValueError(f'Observation shape mismatch {path}')
            if 'window_id' not in data:
                data.update(window_id=np.arange(len(pos),dtype=np.int32),
                            token_y=np.arange(28,dtype=np.int16),token_x=np.arange(28,dtype=np.int16),
                            valid_area=((boxes[:,1]-boxes[:,0])*(boxes[:,3]-boxes[:,2])).reshape(occ.shape).astype(np.int32))
                temp=path.with_suffix('.npz.tmp')
                with temp.open('wb') as f:np.savez_compressed(f,**data)
                temp.replace(path)
            if not np.isfinite(occ).all() or occ.min()<0 or occ.max()>1:
                raise ValueError(f'Invalid predicted occupancy {path}')


def report_stage(args,datasets):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir=args.output_dir/'figures';figdir.mkdir(parents=True,exist_ok=True)
    finalize_observation_cache(args,datasets)
    display={'GlaS':'GLaS-Test','CRAG':'CRAG-Test'}
    d2=pd.concat([pd.read_csv(args.output_dir/'d2_occupancy/tables'/f'{ds}_support_loo_metrics.csv') for ds in datasets],ignore_index=True)
    p1=pd.concat([pd.read_csv(args.output_dir/'p1_soft_memory/tables'/f'{ds}_per_image.csv') for ds in datasets],ignore_index=True)
    p1occ=pd.concat([pd.read_csv(args.output_dir/'p1_soft_memory/tables'/f'{ds}_occupancy_per_image.csv') for ds in datasets],ignore_index=True)
    u1=pd.concat([pd.read_csv(args.output_dir/'u1_real_pcbp/tables'/f'{ds}_per_image.csv') for ds in datasets],ignore_index=True)
    for ds in datasets:
        n=80 if ds=='GlaS' else 40
        if len(p1[p1.dataset==ds])!=5*n or len(u1[u1.dataset==ds])!=12*n:
            raise ValueError(f'Incomplete test outputs {ds}')
        if len(d2[d2.dataset==ds])!=6*(7 if ds=='GlaS' else 16):raise ValueError('Incomplete LOO')
        residual=pd.read_csv(args.output_dir/'u1_real_pcbp/residuals'/f'{ds}.csv')
        if len(residual)!=n*2*2*9:raise ValueError(f'Incomplete PCBP trajectory {ds}')
        monotone=residual.groupby(['image_id','source','phase_count']).apply(
            lambda g:np.all(np.diff(g.sort_values('iteration').observation_residual.to_numpy())<=1e-9))
        if not monotone.all():raise ValueError(f'Non-monotone PCBP trajectory {ds}')
    d2numeric=d2.select_dtypes(include=[np.number]).columns.tolist()
    d2sum=d2.groupby(['dataset','method'],sort=False)[d2numeric].mean().reset_index()
    d2sd=d2.groupby(['dataset','method'],sort=False)[d2numeric].std().reset_index()
    d2sd=d2sd.rename(columns={c:c+'_sd' for c in d2sd.columns if c not in ('dataset','method')})
    d2sum=d2sum.merge(d2sd,on=['dataset','method'],validate='one_to_one')
    p1sum=p1.groupby(['dataset','method'],sort=False).agg({k:['mean','std'] for k in ('dice','fg_iou','boundary_iou','bf1_2px','bf1_4px','object_f1','object_dice','object_hd')})
    p1sum.columns=[f'{a}_{b}' for a,b in p1sum.columns]
    p1sum=p1sum.reset_index()
    u1sum=u1.groupby(['dataset','source','phase_count','method'],sort=False).agg({k:['mean','std'] for k in ('dice','boundary_iou','object_f1','object_hd','observation_residual','reconstruction_seconds',
        'merge_count','narrow_bridge_merge','unmatched_gt_no_overlap','unmatched_gt_partial',
        'unmatched_pred_no_overlap','unmatched_pred_partial','gt_objects','pred_objects')})
    u1sum.columns=[f'{a}_{b}' for a,b in u1sum.columns]
    u1sum=u1sum.reset_index()
    write_csv(args.output_dir/'tables/d2_support_loo_summary.csv',d2sum.to_dict('records'))
    write_csv(args.output_dir/'tables/p1_native_summary.csv',p1sum.to_dict('records'))
    write_csv(args.output_dir/'tables/u1_native_summary.csv',u1sum.to_dict('records'))
    cache_log=pd.concat([pd.read_csv(args.output_dir/'p1_soft_memory/tables'/f'{ds}_query_cache.csv') for ds in datasets],ignore_index=True)
    if len(cache_log)!=sum((80 if ds=='GlaS' else 40)*18 for ds in datasets):raise ValueError('Incomplete query Token occupancy cache')
    p1time=p1.copy()
    p1time['retrieval_seconds']=np.nan
    for method,kind in (('P1-a','random'),('P1-b','stratified'),('P1-c','stratified')):
        sub=cache_log[(cache_log.memory==kind)&(cache_log.dy==cache_log.dx)&(cache_log.dy.isin((0,8)))]
        lookup=sub.groupby(['dataset','image_id']).retrieval_seconds.sum()
        mask=p1time.method==method
        p1time.loc[mask,'retrieval_seconds']=[lookup.loc[(ds,iid)] for ds,iid in zip(p1time.loc[mask,'dataset'],p1time.loc[mask,'image_id'])]
    p1time['standalone_seconds_cached_features']=p1time.retrieval_seconds+p1time.reconstruction_seconds
    write_csv(args.output_dir/'tables/p1_per_image_with_timing.csv',p1time.to_dict('records'))
    memory_rows=[]
    for ds in datasets:
        count=1024*(7 if ds=='GlaS' else 16)
        memory_rows.append(dict(dataset=ds,memory='historical_fg_bg',tokens=count,
            feature_mib=count*1024*4/2**20,archive_mib=np.nan))
        for kind in ('random','stratified'):
            path=args.output_dir/'p1_soft_memory/memories'/f'{ds}_{kind}.npz'
            with np.load(path,allow_pickle=False) as z:
                if z['features'].shape!=(count,1024) or len(z['occupancy'])!=count or len(z['source'])!=count:
                    raise ValueError(f'Unexpected Memory size {path}')
                feature_mib=z['features'].nbytes/2**20
            memory_rows.append(dict(dataset=ds,memory=kind,tokens=count,
                feature_mib=feature_mib,archive_mib=path.stat().st_size/2**20))
    memory_summary=pd.DataFrame(memory_rows)
    write_csv(args.output_dir/'tables/memory_size_summary.csv',memory_rows)
    p1runtime=p1time.groupby(['dataset','method'],sort=False).agg(
        retrieval_core_seconds_mean=('retrieval_seconds','mean'),
        reconstruction_seconds_mean=('reconstruction_seconds','mean'),
        cached_feature_segment_seconds_mean=('standalone_seconds_cached_features','mean')).reset_index()
    write_csv(args.output_dir/'tables/p1_runtime_summary.csv',p1runtime.to_dict('records'))
    u1_observation_files=sum(1 for _ in (args.output_dir/'u1_real_pcbp/token_observations').rglob('*.npz'))
    if u1_observation_files!=sum((80 if ds=='GlaS' else 40)*2*16 for ds in datasets):
        raise ValueError('Incomplete U1 Token observation cache')
    run_config=dict(seed=1337,datasets={ds:dict(support=7 if ds=='GlaS' else 16,test=80 if ds=='GlaS' else 40) for ds in datasets},
        device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        torch=torch.__version__,cuda=torch.version.cuda,window=448,stride=224,grid=[28,28],feature_dim=1024,k=50,
        phases={'2':PHASES[2],'16':PHASES[16]},pcbp_iterations=8,jbu_radius=4,jbu_epsilon=.001,
        query_occupancy_cache_files=len(cache_log),u1_token_observation_files=u1_observation_files,
        retrieval_seconds_sum=float(cache_log.retrieval_seconds.sum()),
        source_query_feature_cache=str(args.phase_cache_dir),
        uni_checkpoint_sha256='56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40',
        uni_checkpoint=json.loads((args.native_baseline_dir/'configs/source_metadata.json').read_text()).get('uni_checkpoint'),
        test_ids={ds:[e['id'] for e in read_manifest(args.dataset_root/ds/'manifests/paper_seed_2026/test.csv')] for ds in datasets},
        support_memory_sha256={ds:sha(args.dual_root/'outputs/n3_bil_distillation_glas_crag_10'/ds/'memory_N3.npz') for ds in datasets},
        support_memory_paths={ds:str(args.dual_root/'outputs/n3_bil_distillation_glas_crag_10'/ds/'memory_N3.npz') for ds in datasets},
        calibration={ds:json.loads((args.output_dir/'d2_occupancy/calibration'/f'{ds}.json').read_text()) for ds in datasets},
        output_size='native',threshold={'P0':'margin>0','occupancy':'occupancy>0.5'})
    dump(args.output_dir/'configs/run_config.json',run_config)
    effects,effsum=factorial_stage(args,datasets)
    # Per-image paired comparisons for P1 and U1.
    matched=[];matched_images=[]
    for ds in datasets:
        sub=p1[p1.dataset==ds]
        base=sub[sub.method=='P0-cal'].set_index('image_id')
        for method in ('P0','P1-a','P1-b','P1-c'):
            target=sub[sub.method==method].set_index('image_id').loc[base.index]
            for metric in ('dice','boundary_iou','object_f1','object_hd'):
                delta=target[metric]-base[metric]
                matched.append(dict(dataset=ds,comparison=f'{method}-P0-cal',**paired_effects(delta,metric)))
                matched_images.extend(dict(dataset=ds,image_id=iid,comparison=f'{method}-P0-cal',metric=metric,delta=float(value))
                                      for iid,value in delta.items())
        sub=u1[u1.dataset==ds]
        for source in ('P0-cal','P1-c'):
            for count in (2,16):
                part=sub[(sub.source==source)&(sub.phase_count==count)]
                base=part[part.method=='bilinear'].set_index('image_id')
                for method in ('pcbp_no_prior','pcbp_jbu_rgb'):
                    target=part[part.method==method].set_index('image_id').loc[base.index]
                    for metric in ('dice','boundary_iou','object_f1','object_hd','merge_count','narrow_bridge_merge'):
                        delta=target[metric]-base[metric]
                        matched.append(dict(dataset=ds,comparison=f'{source}/phase{count}/{method}-bilinear',
                                            **paired_effects(delta,metric)))
                        matched_images.extend(dict(dataset=ds,image_id=iid,
                            comparison=f'{source}/phase{count}/{method}-bilinear',metric=metric,delta=float(value))
                            for iid,value in delta.items())
    write_csv(args.output_dir/'tables/paired_comparisons.csv',matched)
    write_csv(args.output_dir/'tables/paired_per_image.csv',matched_images)
    # Honest outer-LOO token predictions are used for reliability figures and pooled micro metrics.
    token_rows=[];reliability=[];bin_rows=[]
    for ds in datasets:
        ids=json.loads((args.output_dir/'d2_occupancy/calibration'/f'{ds}.json').read_text())['support_ids']
        pool={m:[] for m in ('M0','M1_sigmoid','M1_isotonic','M2_raw','M2_sigmoid','M2_isotonic')};truth=[]
        for iid in ids:
            with np.load(args.output_dir/'d2_occupancy/calibration/outer_predictions'/ds/f'{iid}.npz',allow_pickle=False) as z:
                truth.append(z['gt'].astype(np.float32))
                for method in pool:pool[method].append(z[method].astype(np.float32))
        gt=np.concatenate(truth)
        for method,parts in pool.items():
            pred=np.concatenate(parts);token_rows.append(dict(dataset=ds,method=method,**occ_metrics(pred,gt)))
            for b in range(10):
                sel=(pred>=b/10)&(pred<(b+1)/10 if b<9 else pred<=1)
                reliability.append(dict(dataset=ds,method=method,bin=b,n=int(sel.sum()),
                                        mean_pred=float(pred[sel].mean()) if sel.any() else np.nan,
                                        mean_gt=float(gt[sel].mean()) if sel.any() else np.nan))
            for i,name in enumerate(BIN_NAMES):
                sel=(gt>=MIX_BOUNDS[i])&(gt<MIX_BOUNDS[i+1])
                bin_rows.append(dict(dataset=ds,method=method,bin=name,n=int(sel.sum()),
                                     mae=float(np.abs(pred[sel]-gt[sel]).mean()) if sel.any() else np.nan))
    write_csv(args.output_dir/'tables/d2_token_micro.csv',token_rows)
    write_csv(args.output_dir/'tables/occupancy_reliability_bins.csv',reliability)
    write_csv(args.output_dir/'tables/occupancy_mae_by_gt_bin.csv',bin_rows)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,ds in zip(axes,datasets):
        ids=json.loads((args.output_dir/'d2_occupancy/calibration'/f'{ds}.json').read_text())['support_ids']
        gtparts=[];pparts=[]
        for iid in ids:
            with np.load(args.output_dir/'d2_occupancy/calibration/outer_predictions'/ds/f'{iid}.npz') as z:
                gtparts.append(z['gt'].astype(np.float32));pparts.append(z['M2_raw'].astype(np.float32))
        gt=np.concatenate(gtparts);pred=np.concatenate(pparts)
        take=np.linspace(0,len(gt)-1,min(20000,len(gt)),dtype=np.int64)
        ax.hexbin(gt[take],pred[take],gridsize=40,mincnt=1,cmap='viridis')
        ax.plot([0,1],[0,1],'r--');ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_title(display[ds]);ax.set_xlabel('GT occupancy');ax.set_ylabel('Retrieved occupancy')
    fig.tight_layout();fig.savefig(figdir/'predicted_vs_gt_occupancy.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,ds in zip(axes,datasets):
        ids=json.loads((args.output_dir/'d2_occupancy/calibration'/f'{ds}.json').read_text())['support_ids']
        errs=[]
        for iid in ids:
            with np.load(args.output_dir/'d2_occupancy/calibration/outer_predictions'/ds/f'{iid}.npz') as z:
                gt=z['gt'].astype(np.float32);pred=z['M2_raw'].astype(np.float32)
                sel=(gt>.1)&(gt<.9);errs.append(pred[sel]-gt[sel])
        ax.hist(np.concatenate(errs),bins=50,density=True);ax.axvline(0,color='k');ax.set_title(display[ds]);ax.set_xlabel('Mixed Token prediction − GT')
    fig.tight_layout();fig.savefig(figdir/'mixed_token_error_distribution.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,ds in zip(axes,datasets):
        cal=json.loads((args.output_dir/'d2_occupancy/calibration'/f'{ds}.json').read_text())
        margin=np.linspace(-.5,.5,300);raw=np.linspace(0,1,300)
        ax.plot(margin,expit(cal['raw_margin_kappa']*margin),label='M0 fixed sigmoid')
        ax.plot(margin,apply_cal(cal['m1_model'],margin),label=cal['m1_method'])
        ax.set_title(display[ds]);ax.set_xlabel('Historical margin');ax.set_ylabel('Predicted occupancy');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(figdir/'margin_occupancy_calibration.png',dpi=160);plt.close(fig)
    for ds in datasets:
        sub=d2[(d2.dataset==ds)&(d2.method=='M2_raw')]
        pick=sub.assign(mixed_fraction=sub.mixed_n/sub.tokens).sort_values(['mixed_fraction','image_id'],ascending=[False,True]).iloc[0].image_id
        with np.load(args.output_dir/'d2_occupancy/calibration/outer_predictions'/ds/f'{pick}.npz') as z:
            gt=z['gt'][:784].reshape(28,28);pred=z['M2_raw'][:784].reshape(28,28)
        fig,axes=plt.subplots(1,3,figsize=(9,3))
        for ax,arr,title in zip(axes,(gt,pred,np.abs(pred-gt)),('GT Token occupancy','LOO predicted occupancy','Absolute error')):
            im=ax.imshow(arr,vmin=0,vmax=1,cmap='viridis');ax.set_title(title);ax.axis('off')
        fig.colorbar(im,ax=axes.ravel().tolist(),fraction=.02)
        fig.suptitle(f'{display[ds]} {pick}: highest Support mixed fraction')
        fig.savefig(figdir/f'{ds}_token_occupancy_case.png',dpi=160);plt.close(fig)
    # Figures required by the fixed protocol.
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    rel=pd.DataFrame(reliability)
    for ax,ds in zip(axes,datasets):
        for method in ('M0','M1_isotonic','M2_raw','M2_sigmoid'):
            g=rel[(rel.dataset==ds)&(rel.method==method)&(rel.n>0)]
            ax.plot(g.mean_pred,g.mean_gt,marker='o',label=method)
        ax.plot([0,1],[0,1],'k--');ax.set_title(display[ds]);ax.set_xlabel('Predicted occupancy');ax.set_ylabel('Mean GT occupancy');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(figdir/'occupancy_reliability.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    bins=pd.DataFrame(bin_rows)
    for ax,ds in zip(axes,datasets):
        g=bins[(bins.dataset==ds)&(bins.method.isin(('M1_isotonic','M2_raw','M2_sigmoid')))]
        for method,part in g.groupby('method'):
            ax.plot(part['bin'],part.mae,marker='o',label=method)
        ax.set_title(display[ds]);ax.tick_params(axis='x',rotation=35);ax.set_ylabel('Token MAE');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(figdir/'occupancy_mae_by_gt_bin.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,ds in zip(axes,datasets):
        g=d2sum[d2sum.dataset==ds].set_index('method')
        ax.bar(g.index,g.mae);ax.tick_params(axis='x',rotation=45);ax.set_title(display[ds]);ax.set_ylabel('Outer image LOO MAE')
    fig.tight_layout();fig.savefig(figdir/'soft_memory_vs_margin.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,ds in zip(axes,datasets):
        g=p1sum[p1sum.dataset==ds]
        ax.bar(g.method,g.dice_mean);ax.set_ylim(0,1);ax.set_title(display[ds]);ax.set_ylabel('Native Dice');ax.tick_params(axis='x',rotation=35)
    fig.tight_layout();fig.savefig(figdir/'native_segmentation_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,ds in zip(axes,datasets):
        g=u1sum[(u1sum.dataset==ds)&(u1sum.source=='P1-c')]
        for method,part in g.groupby('method'):
            ax.plot(part.phase_count,part.boundary_iou_mean,marker='o',label=method)
        ax.set_xticks([2,16]);ax.set_title(display[ds]);ax.set_ylabel('Native Boundary IoU');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(figdir/'real_pcbp_phase_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,ds in zip(axes,datasets):
        g=u1[(u1.dataset==ds)&(u1.source=='P1-c')&(u1.phase_count==16)]
        b=g[g.method=='bilinear'].set_index('image_id')
        for method in ('pcbp_no_prior','pcbp_jbu_rgb'):
            x=g[g.method==method].set_index('image_id').loc[b.index]
            ax.scatter(x.observation_residual-b.observation_residual,x.boundary_iou-b.boundary_iou,s=12,label=method,alpha=.6)
        ax.axhline(0,color='k',lw=.7);ax.axvline(0,color='k',lw=.7);ax.set_title(display[ds]);ax.set_xlabel('Δ observation residual');ax.set_ylabel('Δ Boundary IoU');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(figdir/'pcbp_residual_vs_segmentation.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,8))
    for row,ds in enumerate(datasets):
        g=u1sum[(u1sum.dataset==ds)&(u1sum.phase_count==16)&(u1sum.source=='P1-c')]
        axes[row,0].bar(g.method,g.object_f1_mean);axes[row,0].set_ylim(0,1);axes[row,0].set_title(display[ds]+' Object F1')
        axes[row,1].bar(g.method,g.object_hd_mean);axes[row,1].set_title(display[ds]+' Object HD')
        for ax in axes[row]:ax.tick_params(axis='x',rotation=30)
    fig.tight_layout();fig.savefig(figdir/'object_f1_hd_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,ds in zip(axes,datasets):
        g=effsum[(effsum.dataset==ds)&(effsum.metric=='boundary_iou')]
        for effect,part in g.groupby('effect'):
            ax.plot(part.phase_count,part['mean'],marker='o',label=effect)
        ax.axhline(0,color='k',lw=.7);ax.set_xticks([2,16]);ax.set_title(display[ds]);ax.set_ylabel('Paired Boundary IoU effect');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(figdir/'factorial_interaction.png',dpi=160);plt.close(fig)
    # Native representative cases selected by deterministic best/worst paired Boundary IoU.
    for ds in datasets:
        g=u1[(u1.dataset==ds)&(u1.source=='P1-c')&(u1.phase_count==16)]
        b=g[g.method=='bilinear'].set_index('image_id');j=g[g.method=='pcbp_jbu_rgb'].set_index('image_id').loc[b.index]
        delta=(j.boundary_iou-b.boundary_iou).sort_values()
        for tag,iid in (('worst',delta.index[0]),('best',delta.index[-1])):
            e=next(e for e in read_manifest(args.dataset_root/ds/'manifests/paper_seed_2026/test.csv',require_mask=True) if e['id']==iid)
            gt=load_mask(e['mask'],DC)
            with np.load(args.output_dir/'u1_real_pcbp/predictions'/ds/f'{iid}_P1-c_phase16_bilinear.npz') as z:bil=z['mask']
            with np.load(args.output_dir/'u1_real_pcbp/predictions'/ds/f'{iid}_P1-c_phase16_pcbp_jbu_rgb.npz') as z:pc=z['mask']
            rgb=load_image(e['image']);overlay=rgb.copy();overlay[(pc>0)&(gt==0)]=[255,0,0];overlay[(pc==0)&(gt>0)]=[0,80,255]
            fig,axes=plt.subplots(1,5,figsize=(15,3))
            for ax,arr,title in zip(axes,(rgb,gt,bil,pc,overlay),('RGB','GT','Bilinear','PCBP-JBU','PCBP FP red / FN blue')):
                ax.imshow(arr,cmap='gray' if arr.ndim==2 else None);ax.set_title(title);ax.axis('off')
            fig.suptitle(f'{display[ds]} {iid}: {tag} Δ Boundary IoU={delta.loc[iid]:+.3f}')
            fig.tight_layout();fig.savefig(figdir/f'{ds}_{tag}_case.png',dpi=130);plt.close(fig)
        worst=p1[(p1.dataset==ds)&(p1.method=='P0')].sort_values(['dice','image_id']).iloc[0].image_id
        e=next(e for e in read_manifest(args.dataset_root/ds/'manifests/paper_seed_2026/test.csv',require_mask=True) if e['id']==worst)
        with np.load(args.output_dir/'p1_soft_memory/predictions'/ds/f'{worst}_P0.npz') as z:p0=z['mask']
        with np.load(args.output_dir/'p1_soft_memory/predictions'/ds/f'{worst}_P0-cal.npz') as z:pcal=z['mask']
        with np.load(args.output_dir/'p1_soft_memory/predictions'/ds/f'{worst}_P1-c.npz') as z:psoft=z['mask']
        fig,axes=plt.subplots(1,5,figsize=(15,3))
        for ax,arr,title in zip(axes,(load_image(e['image']),load_mask(e['mask'],DC),p0,pcal,psoft),
                                ('RGB','GT','P0 Margin','P0-cal','P1-c')):
            ax.imshow(arr,cmap='gray' if arr.ndim==2 else None);ax.set_title(title);ax.axis('off')
        fig.suptitle(f'{display[ds]} {worst}: lowest P0 Dice')
        fig.tight_layout();fig.savefig(figdir/f'{ds}_worst_margin_case.png',dpi=130);plt.close(fig)
    # Full narrative from measured values; no Test label enters method selection.
    lines=['# D2 + P1 + U1 真实 Occupancy 诊断','',
           '## A. Experimental Configuration','',
           f"GLaS-Test 为合并的 TestA+TestB 80 张，CRAG-Test 为官方 40 张。历史 seed1337 Support 分别为 7/16 张，固定 UNI ViT-L/16、448 window、224 stride、28×28 Token、K50。运行设备 {run_config['device']}。历史 Memory 与 1920 份 16相位 Query Feature 均复用，不重新提取 UNI；本轮保存 {len(cache_log)} 份真实检索 Query Token 占比缓存及 {u1_observation_files} 份 U1 Window/Phase Token 观测缓存，检索核心计时累计 {run_config['retrieval_seconds_sum']:.1f}s（不含读取/评估）。Soft Memory 每张 Support 采样 1024 Token，保持历史总预算。Temperature 在 Support 图像级 LOO 选择；校准质量用外层留图预测报告，Test GT 不参与拟合或选择。原图指标定义沿用上一轮 `native_diagnostics_seed1337.py`。完整 Checkpoint/Memory 哈希、Support ID、测试 ID 和缓存配置见 `configs/`。",
           '',f"UNI Checkpoint：`{run_config['uni_checkpoint']}`；SHA256：`{run_config['uni_checkpoint_sha256']}`。两相位为 (0,0)/(8,8)，16 相位为 {{0,4,8,12}}²；PCBP 8 次迭代、覆盖归一化及单调回溯，RGB JBU 半径 4、ε=0.001。评估在原图分辨率进行，P0 用 Margin>0，其余方法用 Occupancy>0.5；Mask、连通域、对象面积阈值、Boundary IoU 与 Object-HD 均复用既有实现。",
           '', '| Dataset | Memory | Token 数 | Feature MiB | NPZ MiB |',
           '|---|---|---:|---:|---:|']
    for r in memory_summary.itertuples():
        archive='—' if np.isnan(r.archive_mib) else f'{r.archive_mib:.1f}'
        lines.append(f'| {display[r.dataset]} | {r.memory} | {r.tokens} | {r.feature_mib:.1f} | {archive} |')
    lines += ['', 'Historical Memory 的 NPZ 大小未重计；Feature MiB 为保存时特征数组大小（历史 float32，Soft float16；检索时统一转为 float32），Soft NPZ MiB 为压缩后的整个存档。计时复用已有 UNI Feature，因而不含特征提取与磁盘加载。',
              '', '| Dataset | 方法 | 检索核心秒/图 | 重建秒/图 | 已缓存特征下小计秒/图 |',
              '|---|---|---:|---:|---:|']
    for r in p1runtime.itertuples():
        ret='已缓存旧 Margin' if np.isnan(r.retrieval_core_seconds_mean) else f'{r.retrieval_core_seconds_mean:.3f}'
        total='—' if np.isnan(r.cached_feature_segment_seconds_mean) else f'{r.cached_feature_segment_seconds_mean:.3f}'
        lines.append(f'| {display[r.dataset]} | {r.method} | {ret} | {r.reconstruction_seconds_mean:.3f} | {total} |')
    lines += ['', '## B. D2：Can UNI Predict Occupancy?','',
           '| Dataset | 方法 | LOO Macro MAE | RMSE | ECE | Mixed MAE | Pearson | Spearman | FG Accuracy | FG AUROC |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ds in datasets:
        for method in ('M0','M1_sigmoid','M1_isotonic','M2_raw','M2_sigmoid','M2_isotonic'):
            r=d2sum[(d2sum.dataset==ds)&(d2sum.method==method)].iloc[0]
            lines.append(f'| {display[ds]} | {method} | {r.mae:.4f}±{r.mae_sd:.4f} | {r.rmse:.4f} | {r.ece:.4f} | {r.mixed_mae:.4f} | {r.pearson:.4f} | {r.spearman:.4f} | {r.fg_accuracy:.4f} | {r.fg_auroc:.4f} |')
    lines += ['']
    for ds in datasets:
        g=d2sum[d2sum.dataset==ds].set_index('method')
        c=json.loads((args.output_dir/'d2_occupancy/calibration'/f'{ds}.json').read_text())
        lines.append(f"{display[ds]}: M0 MAE={g.loc['M0','mae']:.4f}，M2 raw={g.loc['M2_raw','mae']:.4f}；Support 留图选择 {c['m1_method']} / {c['m2_method']}，τ={c['temperature']}。M2 校准后 Mixed MAE={g.loc[c['m2_method'],'mixed_mae']:.4f}，整体 MAE={g.loc[c['m2_method'],'mae']:.4f}；其 logit 斜率={c['m2_model'][1]:.3f}，低于 1 表示预测向中间占比收缩。应同时看纯 Token 与混合 Token，不能只凭分类准确率判断可用性。")
    lines += ['', '主结果按 Support 图像等权。`tables/d2_token_micro.csv` 为合并 Token 的 Micro 结果；不同 GT 占比区间见逐图 CSV 与图。最终校准器及 Temperature 见 `d2_occupancy/calibration/`。',
              '', '## C. P1：Does Soft Memory Improve Segmentation?','',
              '| Dataset | 方法 | Dice mean±SD | Boundary IoU | Object F1 | Object HD | Occupancy MAE | ECE |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for ds in datasets:
        for method in ('P0','P0-cal','P1-a','P1-b','P1-c'):
            r=p1sum[(p1sum.dataset==ds)&(p1sum.method==method)].iloc[0]
            o=p1occ[(p1occ.dataset==ds)&(p1occ.method==method)]
            occ_fields=f'{o.mae.mean():.4f} | {o.ece.mean():.4f}' if len(o) else '— | —'
            lines.append(f'| {display[ds]} | {method} | {r.dice_mean:.4f}±{r.dice_std:.4f} | {r.boundary_iou_mean:.4f} | {r.object_f1_mean:.4f} | {r.object_hd_mean:.2f} | {occ_fields} |')
    lines += ['']
    for ds in datasets:
        g=p1sum[p1sum.dataset==ds].set_index('method')
        o=p1occ[p1occ.dataset==ds].groupby('method').mean(numeric_only=True)
        lines.append(f"{display[ds]}: 混合 Token 进入随机 Memory 后，P1-a−P0-cal Dice={g.loc['P1-a','dice_mean']-g.loc['P0-cal','dice_mean']:+.4f}（Occupancy MAE {o.loc['P0-cal','mae']:.4f}→{o.loc['P1-a','mae']:.4f}）；分层抽样 P1-b−P1-a Dice={g.loc['P1-b','dice_mean']-g.loc['P1-a','dice_mean']:+.4f}；校准 P1-c−P1-b Dice={g.loc['P1-c','dice_mean']-g.loc['P1-b','dice_mean']:+.4f}。P1-c−P0-cal Dice={g.loc['P1-c','dice_mean']-g.loc['P0-cal','dice_mean']:+.4f}；P1-b/P1-c Mixed MAE={o.loc['P1-b','mixed_mae']:.4f}/{o.loc['P1-c','mixed_mae']:.4f}。")
    lines += ['', 'P1-a 与 P1-b 区分随机/分层采样，P1-b 与 P1-c 区分校准，P0-cal 与 P1-c 比较在相同占比输出形式下的 Memory 效果。分层采用 [0,.2), [.2,.4), [.4,.6), [.6,.8), [.8,1] 五箱，先等额抽样；不足的配额从所有未采样候选中随机回填，始终保持每图 1024 的固定预算。逐图 Occupancy 与分割指标、Memory 各箱候选/采样数量见 `p1_soft_memory/`。',
              '', '## D. U1：Does PCBP Work with Real Occupancy?','',
              '| Dataset | 来源 | Phases | 重建 | Dice | Boundary IoU | Object F1 | Object HD | Residual | 重建秒/图 |',
              '|---|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for ds in datasets:
        for source in ('P0-cal','P1-c'):
            for count in (2,16):
                for method in ('bilinear','pcbp_no_prior','pcbp_jbu_rgb'):
                    r=u1sum[(u1sum.dataset==ds)&(u1sum.source==source)&(u1sum.phase_count==count)&(u1sum.method==method)].iloc[0]
                    lines.append(f'| {display[ds]} | {source} | {count} | {method} | {r.dice_mean:.4f} | {r.boundary_iou_mean:.4f} | {r.object_f1_mean:.4f} | {r.object_hd_mean:.2f} | {r.observation_residual_mean:.6g} | {r.reconstruction_seconds_mean:.3f} |')
    lines += ['']
    for ds in datasets:
        for source in ('P0-cal','P1-c'):
            base=u1sum[(u1sum.dataset==ds)&(u1sum.source==source)&(u1sum.phase_count==16)&(u1sum.method=='bilinear')].iloc[0]
            pc=u1sum[(u1sum.dataset==ds)&(u1sum.source==source)&(u1sum.phase_count==16)&(u1sum.method=='pcbp_jbu_rgb')].iloc[0]
            lines.append(f"{display[ds]} {source} 16相位：JBU 相对 Bilinear 的残差 {pc.observation_residual_mean-base.observation_residual_mean:+.6f}，Dice {pc.dice_mean-base.dice_mean:+.4f}，Boundary IoU {pc.boundary_iou_mean-base.boundary_iou_mean:+.4f}，Object-F1 {pc.object_f1_mean-base.object_f1_mean:+.4f}，Merge/图 {pc.merge_count_mean-base.merge_count_mean:+.3f}。")
    lines += ['','16相位对象结构诊断（每图均值；漏匹配和误检依次为无交集/部分交集）：','',
              '| Dataset | 来源 | 重建 | GT/Pred 对象 | GT 漏匹配 | 预测误检 | Merge | 窄桥 Merge |',
              '|---|---|---|---:|---:|---:|---:|---:|']
    for ds in datasets:
        for source in ('P0-cal','P1-c'):
            for method in ('bilinear','pcbp_jbu_rgb'):
                r=u1sum[(u1sum.dataset==ds)&(u1sum.source==source)&(u1sum.phase_count==16)&(u1sum.method==method)].iloc[0]
                lines.append(f"| {display[ds]} | {source} | {method} | {r.gt_objects_mean:.2f}/{r.pred_objects_mean:.2f} | {r.unmatched_gt_no_overlap_mean:.2f}/{r.unmatched_gt_partial_mean:.2f} | {r.unmatched_pred_no_overlap_mean:.2f}/{r.unmatched_pred_partial_mean:.2f} | {r.merge_count_mean:.2f} | {r.narrow_bridge_merge_mean:.2f} |")
    lines += ['', '每组 Bilinear、PCBP 和 PCBP-JBU 共用同一批已保存的 Window/Phase Token 占比。PCBP 固定 8 次迭代、覆盖归一化、单调回溯；JBU 为 RGB 灰度引导 r=4、ε=0.001。残差下降不自动表示分割改善。漏匹配、误检、Merge、窄桥启发式与对象数详见 U1 逐图表。窄桥启发式：对合并连通域做 2px 侵蚀，若分裂为分别覆盖不同 GT 腺体的显著碎片则计数；它不是官方实例指标。',
              '', '## E. 2×2 Factorial Analysis','',
              '| Dataset | Phases | 指标 | Prototype C−A (95% CI) | Reconstruction B−A (95% CI) | Interaction D−C−B+A (95% CI) |',
              '|---|---:|---|---:|---:|---:|']
    for ds in datasets:
        for count in (2,16):
            for metric in ('dice','boundary_iou','object_f1','object_hd'):
                g=effsum[(effsum.dataset==ds)&(effsum.phase_count==count)&(effsum.metric==metric)].set_index('effect')
                def fmt(effect):
                    r=g.loc[effect]
                    return f"{r['mean']:+.4f} [{r['ci95_low']:+.4f},{r['ci95_high']:+.4f}]"
                lines.append(f"| {display[ds]} | {count} | {metric} | {fmt('retrieval')} | {fmt('reconstruction')} | {fmt('interaction')} |")
    lines += ['', 'Object-HD 为越低越好；所有原始配对差值、95% 图像 Bootstrap CI、改善/下降图像数及中位数见 `factorial/summary.csv`。CRAG 缺少可靠 WSI ID，区间不解释为独立病例区间。',
              '', '## F. Comparison with Previous Oracle Diagnostics','',
              '上一轮 GT Soft Oracle 的 Native Dice 为 GLaS-Test 0.9901、CRAG-Test 0.9943。它是理想占比观测的诊断上限参考；本轮所有 Test 预测和 PCBP 输入均来自真实 UNI 检索。Oracle 结果未用于本轮 Temperature、校准器或重建参数选择。',
              '', '| Dataset | 16相位重建 | GT Oracle Boundary IoU | 真实 P1-c Boundary IoU |',
              '|---|---|---:|---:|']
    previous=pd.read_csv(args.native_baseline_dir/'tables/multiphase_summary.csv')
    for ds in datasets:
        for oracle_method,real_method in (('bilinear_oracle','bilinear'),('pcbp_no_prior','pcbp_no_prior'),('pcbp_jbu_rgb','pcbp_jbu_rgb')):
            old=previous[(previous.dataset==ds)&(previous.phase_count==16)&(previous.method==oracle_method)].iloc[0]
            real=u1sum[(u1sum.dataset==ds)&(u1sum.source=='P1-c')&(u1sum.phase_count==16)&(u1sum.method==real_method)].iloc[0]
            lines.append(f'| {display[ds]} | {real_method} | {old.boundary_iou_mean:.4f} | {real.boundary_iou_mean:.4f} |')
    lines += ['', '## G. Research Decision','']
    for ds in datasets:
        r=p1sum[(p1sum.dataset==ds)&(p1sum.method=='P1-c')].iloc[0]
        b=p1sum[(p1sum.dataset==ds)&(p1sum.method=='P0-cal')].iloc[0]
        g=effsum[(effsum.dataset==ds)&(effsum.phase_count==16)&(effsum.metric=='boundary_iou')].set_index('effect')
        lines.append(f"{display[ds]}: P1-c−P0-cal Dice={r.dice_mean-b.dice_mean:+.4f}; 16 相位 Reconstruction Boundary IoU={g.loc['reconstruction','mean']:+.4f} (CI {g.loc['reconstruction','ci95_low']:+.4f} 至 {g.loc['reconstruction','ci95_high']:+.4f}); Interaction={g.loc['interaction','mean']:+.4f}。")
    p1_gain=np.mean([p1sum[(p1sum.dataset==ds)&(p1sum.method=='P1-c')].iloc[0].dice_mean-
                     p1sum[(p1sum.dataset==ds)&(p1sum.method=='P0-cal')].iloc[0].dice_mean for ds in datasets])
    reconstruction_positive=all(effsum[(effsum.dataset==ds)&(effsum.phase_count==16)&
                                   (effsum.metric=='boundary_iou')&(effsum.effect=='reconstruction')].iloc[0].ci95_low>0 for ds in datasets)
    if p1_gain>0 and reconstruction_positive:
        decision='Direction B：Prototype + PCBP。P1 与固定 PCBP-JBU 在真实观测上均有配对收益；下一步才考虑完整方法设计与 U2 引导。'
    elif p1_gain>0:
        decision='Direction A：Prototype First。P1 有收益，但固定 PCBP-JBU 的跨数据集边界收益不稳定；先研究 Memory 可靠性，暂缓复杂重建。'
    else:
        decision='Direction C：Occupancy 表示存在根本问题。P1-c 相对 P0-cal 的平均 Dice 未改善；优先检查 Token 语义、混合区域占比与检索偏差，停止扩展 PCBP。'
    lines += ['',decision,'',
              'H1：冻结 UNI 特征包含一定前景/背景排序信息，但占比回归尤其在 Mixed Token 上不够可靠；Support LOO 与 Test 诊断的误差及 P1 分割退化共同支持这一判断。',
              'H2：在真实检索占比下，PCBP 的观测残差下降必须与 Dice、Boundary IoU 和对象结构一起判断；本轮同观测、同相位配对结果见上表，不可将 Oracle 16相位收益直接外推。',
              '这项方向判断基于已固定的 Support 选择规则与全量 Test 配对结果；没有用 Test 指标改动 Temperature、校准器、PCBP 或阈值。']
    lines += ['', '## H. Figures and Case Studies', '',
              '总体图：' + '、'.join(f'[{name}](figures/{name})' for name in (
                  'occupancy_reliability.png','occupancy_mae_by_gt_bin.png','soft_memory_vs_margin.png',
                  'native_segmentation_comparison.png','real_pcbp_phase_comparison.png',
                  'pcbp_residual_vs_segmentation.png','object_f1_hd_comparison.png','factorial_interaction.png')) + '。',
              '', '补充占比诊断：' + '、'.join(f'[{name}](figures/{name})' for name in (
                  'predicted_vs_gt_occupancy.png','mixed_token_error_distribution.png',
                  'margin_occupancy_calibration.png')) + '。',
              '', '代表性图像按固定规则选择：16 相位 P0-cal 的 PCBP-JBU 相对 Bilinear Boundary IoU 最大/最小变化、以及 P0 Dice 最低图像；同时保存真实 Token 占比案例。', '']
    for ds in datasets:
        lines.append(f"{display[ds]}：" + '、'.join(f'[{tag}](figures/{ds}_{tag}.png)' for tag in (
            'best_case','worst_case','worst_margin_case','token_occupancy_case')) + '。')
    (args.output_dir/'research_report.md').write_text('\n'.join(lines)+'\n')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',choices=('d2','p1','u1','audit','report','all'),default='all')
    ap.add_argument('--dataset',choices=('GlaS','CRAG','all'),default='all')
    ap.add_argument('--dataset-root',type=Path,default=ROOT/'datasets')
    ap.add_argument('--dual-root',type=Path,default=ROOT.parent/'DUAL')
    ap.add_argument('--phase-cache-dir',type=Path,default=ROOT/'outputs/uni_shift_stitch_seed1337/feature_cache')
    ap.add_argument('--phase-margin-dir',type=Path,default=ROOT/'outputs/uni_shift_stitch_seed1337/phase_margins')
    ap.add_argument('--native-baseline-dir',type=Path,default=NATIVE_OUT)
    ap.add_argument('--output-dir',type=Path,default=OUT_DEFAULT)
    ap.add_argument('--output-size',choices=('native',),default='native')
    ap.add_argument('--seed',type=int,default=1337)
    args=ap.parse_args()
    if args.seed!=1337:raise ValueError('This study uses the fixed historical seed 1337')
    for field in ('dataset_root','dual_root','phase_cache_dir','phase_margin_dir','native_baseline_dir','output_dir'):
        setattr(args,field,getattr(args,field).resolve())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    datasets=('GlaS','CRAG') if args.dataset=='all' else (args.dataset,)
    for ds in datasets:
        st=setup(ds,args)
        arrays=support_arrays(st)
        if args.stage in ('d2','all'):
            loo_stage(st,arrays,args)
            calibrate_stage(st,args)
        if args.stage in ('p1','all'):
            cal=json.loads((args.output_dir/'d2_occupancy/calibration'/f'{ds}.json').read_text())
            memories=memory_stage(st,arrays,args)
            query_soft_stage(st,memories,cal,args)
            p1_bilinear_stage(st,cal,args)
        if args.stage=='audit':
            cal=json.loads((args.output_dir/'d2_occupancy/calibration'/f'{ds}.json').read_text())
            memories=memory_stage(st,arrays,args)
            query_soft_stage(st,memories,cal,args)
        if args.stage in ('u1','all'):
            cal=json.loads((args.output_dir/'d2_occupancy/calibration'/f'{ds}.json').read_text())
            u1_stage(st,cal,args)
    if args.stage in ('report','all') and args.dataset=='all':
        report_stage(args,datasets)


if __name__=='__main__':main()
