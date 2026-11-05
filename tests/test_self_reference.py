import ast,inspect
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from pathology_trainfree.self_reference.seed_generator import validate,generate,stability_filter
from pathology_trainfree.self_reference.token_mapping import map_seeds,load_query_cache
from pathology_trainfree.self_reference.retrieval import candidates,memory_status,SpatiallyExcludedRetriever,IneligibleMemory
from pathology_trainfree.self_reference.diagnostics import seed_metrics
from pathology_trainfree.memory import ReferenceMemory
from pathology_trainfree.geometry import windows
from pathology_trainfree.config import load_config
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def cfg():return yaml.safe_load((ROOT/'configs/self_reference_e0.yaml').read_text())

def seed_from(fg,bg):return {'fg':fg,'bg':bg,'ignore':~(fg|bg),'epithelial':fg.copy(),'interior':np.zeros_like(fg)}

def test_tristate_and_empty_precision():
    zero=np.zeros((32,48),bool);s=seed_from(zero,zero);validate(s)
    row=seed_metrics(s,np.ones_like(zero),np.ones_like(zero));assert row['fg_precision'] is None and row['bg_precision'] is None and row['fg_coverage']==0 and row['ignore_ratio']==1
    s['bg'][:]=True;s['fg'][:]=True
    with pytest.raises(ValueError):validate(s)


def test_spatial_occupancy_offsets_reflection_and_ignore(cfg):
    fg=np.zeros((97,113),bool);bg=fg.copy();fg[16:48,16:48]=True;bg[64:80,64:80]=True;s=seed_from(fg,bg)
    m=map_seeds(s,cfg['window'])
    expected=[];pos=[]
    for o in [0,8]:
        for y,x,crop in windows(fg.astype(float),448,224,o):expected.append(crop.reshape(28,16,28,16).mean((1,3)).ravel());pos.append((y,x))
    np.testing.assert_array_equal(m['fg'],np.concatenate(expected));np.testing.assert_array_equal(m['positions'],pos)
    np.testing.assert_allclose(m['fg']+m['bg']+m['ignore'],1)
    assert ((m['centers']>=0)&(m['centers']<=np.array([96,112]))).all()
    ix=candidates(m,cfg);assert (m['fg'][ix[1]]>=.9).all() and (m['bg'][ix[1]]==0).all()
    empty=map_seeds(seed_from(np.zeros_like(fg),np.zeros_like(fg)),cfg['window']);assert all(len(x)==0 for x in candidates(empty,cfg).values())


def test_stability_never_upgrades_original_ignore():
    fg=np.zeros((8,9),bool);bg=fg.copy();fg[1:3,1:3]=True;bg[6,6]=True;original=seed_from(fg,bg)
    view=seed_from(np.ones_like(fg),np.zeros_like(fg));c=stability_filter(original,[view]);assert np.array_equal(c['fg'],fg) and not c['bg'].any() and c['ignore'][4,4]


def test_synthetic_generation_deterministic_no_gt(cfg):
    cfg=deepcopy(cfg);cfg['superpixels'].update({'maximum':24,'minimum':8,'kmeans_n_init':2})
    rgb=np.full((80,112,3),255,np.uint8);rgb[12:68,20:92]=[215,155,185];rgb[26:54,38:74]=[250,242,246]
    a,m=generate(rgb,cfg);b,_=generate(rgb,cfg)
    for g in ['A','B','C']:
        validate(a[g]);assert a[g]['fg'].shape==rgb.shape[:2]
        for k in ['fg','bg','ignore']:np.testing.assert_array_equal(a[g][k],b[g][k])
    assert np.all(a['C']['fg']<=a['B']['fg']) and np.all(a['C']['bg']<=a['B']['bg'])


def test_blank_image_empty_seeds(cfg):
    cfg=deepcopy(cfg);cfg['superpixels'].update({'maximum':8,'minimum':8,'kmeans_n_init':1});cfg['stability']['views']=[]
    seeds,_=generate(np.full((48,64,3),255,np.uint8),cfg)
    for s in seeds.values():assert not s['fg'].any() and not s['bg'].any() and s['ignore'].all()


def test_spatial_exclusion_duplicate_coordinates_fixed_k():
    bank=np.zeros((51,1024),np.float32);bank[:,0]=1;memory=ReferenceMemory({0:bank,1:bank},{})
    cfg=load_config(ROOT/'configs/uni_memory.yaml');cfg['retrieval']['top_k']=50
    query=np.array([[0,0],[100,100]],np.float32);ref={c:np.zeros((51,2),np.float32) for c in [0,1]}
    retriever=SpatiallyExcludedRetriever(memory,cfg,'cpu',query,ref,32);v=retriever.feasibility()
    assert v[0]['minimum_eligible']==0 and v[1]['insufficient_query_tokens']==1
    with pytest.raises(IneligibleMemory):retriever.scores(np.tile(bank[:1],(2,1)))
    assert retriever.k==50
    assert memory_status({0:0,1:100},50)=='EMPTY_MEMORY'
    assert memory_status({0:49,1:100},50)=='INSUFFICIENT_MEMORY'


def test_spatial_exclusion_matches_analytic_cosines():
    bank=np.zeros((52,1024),np.float32);bank[:,0]=1;bank[:2,0]=0;bank[:2,1]=1
    cfg=load_config(ROOT/'configs/uni_memory.yaml');cfg['retrieval']['top_k']=50
    xy=np.ones((52,2),np.float32)*100;xy[:2]=0
    r=SpatiallyExcludedRetriever(ReferenceMemory({0:bank,1:bank},{}),cfg,'cpu',np.zeros((1,2)),{0:xy,1:xy},32)
    q=np.zeros((1,1024),np.float32);q[0,0]=1
    np.testing.assert_allclose(r.scores(q).numpy(),[[1,1]])


def test_seed_generator_no_gt_call_path():
    import pathology_trainfree.self_reference.seed_generator as module
    source=inspect.getsource(module);tree=ast.parse(source)
    assert list(inspect.signature(module.generate).parameters)==['rgb','cfg']
    forbidden={'load_mask','read_manifest','read_csv','torch','FeatureEncoder','ReferenceMemory'}
    assert not any(isinstance(n,ast.Name) and n.id in forbidden for n in ast.walk(tree))


def test_cached_feature_fingerprint_rejects_mismatch(tmp_path,cfg):
    rgb=np.full((448,448,3),200,np.uint8);p=tmp_path/'example/feature_cache';p.mkdir(parents=True)
    np.savez(p/'id_offset0.npz',tokens=np.zeros((1,784,1024),np.float16),positions=[[0,0]],cache_key='incorrect')
    with pytest.raises(ValueError,match='fingerprint'):load_query_cache(rgb,'id',tmp_path,'example',cfg)


def test_overlapping_windows_share_physical_centers(cfg):
    shape=(512,700);blank=np.zeros(shape,bool);m=map_seeds(seed_from(blank,blank),cfg['window'])
    window0=int(np.flatnonzero((m['positions']==[0,0]).all(1))[0]);window1=int(np.flatnonzero((m['positions']==[0,224]).all(1))[0])
    np.testing.assert_array_equal(m['centers'][window0*784+14],m['centers'][window1*784])
    assert np.unique(m['centers'],axis=0).shape[0]<len(m['centers'])


def test_stage_integration_empty_memory_and_resume(tmp_path,cfg):
    import sys,hashlib,json
    from PIL import Image
    sys.path.insert(0,str(ROOT/'scripts'))
    from diagnose_self_reference_seeds import stage0,stage1,stage2
    import pandas as pd
    cfg=deepcopy(cfg);cfg['superpixels'].update({'maximum':8,'minimum':8,'kmeans_n_init':1});cfg['stability']['views']=[]
    image=np.full((80,112,3),255,np.uint8);gt=np.zeros((80,112),np.uint8);gt[16:64,24:88]=255
    ip=tmp_path/'toy.png';gp=tmp_path/'toy_gt.png';Image.fromarray(image).save(ip);Image.fromarray(gt).save(gp)
    entries={'Toy':[{'id':'toy','image':ip,'mask':gp}]};out=tmp_path/'out';out.mkdir();root=tmp_path/'cache';folder=root/'Toy/feature_cache';folder.mkdir(parents=True)
    for offset in [0,8]:
        views=list(windows(image,448,224,offset));key=hashlib.sha256(image.tobytes()+str((cfg['encoder']['weights_sha256'],448,224,offset,'float32','float16')).encode()).hexdigest()
        np.savez_compressed(folder/f'toy_offset{offset}.npz',tokens=np.zeros((len(views),784,1024),np.float16),positions=np.array([(y,x) for y,x,_ in views]),cache_key=key)
    logs=[];stage0(entries,cfg,out,['A','B','C'],False,logs.append);stage1(entries,cfg,out,['A','B','C'],False,root,logs.append);stage2(entries,cfg,out,['A','B','C'],False,root,'cpu',logs.append)
    rows=pd.read_csv(out/'self_reference_per_image.csv');assert len(rows)==6 and rows.status.eq('EMPTY_MEMORY').all() and rows.top_k.eq(50).all()
    assert pd.read_csv(out/'seed_metrics_per_image.csv').fg_precision.isna().all()
    stage1(entries,cfg,out,['A','B','C'],True,root,logs.append);stage2(entries,cfg,out,['A','B','C'],True,root,'cpu',logs.append)
    assert any(x.startswith('E1 resume') for x in logs) and any(x.startswith('E2 resume') for x in logs)
    with np.load(out/'images/Toy/toy/memory_A.npz') as z:assert not json.loads(str(z['metadata']))['gt_used']
    bad=deepcopy(cfg);bad['memory']['purity']=.95
    with pytest.raises(ValueError,match='Seed stage fingerprint'):stage1(entries,bad,out,['A','B','C'],True,root,logs.append)
    with pytest.raises(ValueError,match='Token stage fingerprint'):stage2(entries,bad,out,['A','B','C'],True,root,'cpu',logs.append)


def test_feature_preprocessing_mismatch_rejected(tmp_path,cfg):
    bad=deepcopy(cfg);bad['encoder']['normalize']=False
    with pytest.raises(ValueError,match='preprocessing'):load_query_cache(np.zeros((32,48,3),np.uint8),'id',tmp_path,'Toy',bad)
