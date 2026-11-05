"""Regression for dummy occupancy in DUAL unlabeled feature caches (real cached data)."""
import numpy as np
import pytest
from study_support_size import load_support_cache,cache_file_for,ROOT
from pathology_trainfree.config import load_config
from pathology_trainfree.data import read_manifest,load_mask
from pathology_trainfree.geometry import windows

@pytest.mark.parametrize('dataset,image_id',[('GlaS','train_1'),('GlaS','train_42'),('CRAG','train_1'),('CRAG','train_17')])
def test_real_support_gt_replaces_dummy_occupancy(dataset,image_id):
 cfg=load_config(ROOT/'configs/uni_memory.yaml');cfg['window']['support_offsets']=[0,8];cfg['dataset'].update({'mask_encoding':'indexed','class_mapping':{0:0,255:1}})
 entries=read_manifest(ROOT/f'datasets/{dataset}/manifests/paper_seed_2026/train_pool.csv',require_mask=True);entry=next(e for e in entries if e['id']==image_id)
 store,provenance=load_support_cache([entry],ROOT.parent/'DUAL',dataset,cfg)
 gt=load_mask(entry['mask'],cfg['dataset']);expected=np.concatenate([crop.reshape(28,16,28,16).mean((1,3)).ravel() for offset in [0,8] for _,_,crop in windows(gt,448,224,offset)])
 np.testing.assert_array_equal(store[image_id]['fg_occ'],expected)
 with np.load(cache_file_for(ROOT.parent/'DUAL',dataset,image_id)) as z:
  if image_id=='train_1':assert not z['occupancy'].any() and expected.max()==1
  else:np.testing.assert_array_equal(z['occupancy'].ravel(),expected)
 assert provenance[0]['occupancy_source']=='real_support_gt'

def test_verified_occupancy_cache_reuse_and_untrusted_cache_rejection(tmp_path, monkeypatch):
 import json
 import study_support_size as study
 cfg=load_config(ROOT/'configs/uni_memory.yaml');cfg['window']['support_offsets']=[0,8];cfg['dataset'].update({'mask_encoding':'indexed','class_mapping':{0:0,255:1}})
 cfg['output']['occupancy_cache_directory']=str(tmp_path)
 entry=next(e for e in read_manifest(ROOT/'datasets/GlaS/manifests/paper_seed_2026/train_pool.csv',require_mask=True) if e['id']=='train_1')
 first,prov=load_support_cache([entry],ROOT.parent/'DUAL','GlaS',cfg)
 calls=[];original=study.windows
 def tracked_windows(image,*args,**kwargs):
  if image.ndim==2:calls.append(True)
  yield from original(image,*args,**kwargs)
 monkeypatch.setattr(study,'windows',tracked_windows)
 second,prov2=load_support_cache([entry],ROOT.parent/'DUAL','GlaS',cfg)
 assert prov2[0]['occupancy_cache_reused'] and not calls
 np.testing.assert_array_equal(first['train_1']['fg_occ'],second['train_1']['fg_occ'])
 path=prov[0]['occupancy_cache_file']
 with np.load(path) as z:metadata=json.loads(str(z['metadata']));dummy=np.zeros_like(z['fg_occupancy'])
 metadata['occupancy_source']='untrusted_placeholder'
 np.savez_compressed(path,fg_occupancy=dummy,metadata=json.dumps(metadata))
 repaired,prov3=load_support_cache([entry],ROOT.parent/'DUAL','GlaS',cfg)
 assert not prov3[0]['occupancy_cache_reused'] and len(calls)==2
 np.testing.assert_array_equal(repaired['train_1']['fg_occ'],first['train_1']['fg_occ'])
