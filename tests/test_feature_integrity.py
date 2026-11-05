"""Spatial/GT/cache tests use synthetic data, without claiming model accuracy."""
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from pathology_trainfree.cache import FeatureStore
from pathology_trainfree.config import load_config
from pathology_trainfree.geometry import windows,reconstruct,resize_map


class StubEncoder:
    def __init__(self,name,patch):
        self.name=name;self.patch_size=patch;self.feature_dim=2
        self.metadata={'encoder_name':name,'patch_size':patch,'preprocessing':{'mean':[.5]*3},'checkpoint_sha256':'test'}
        self.weights_sha256='test';self.precision='float32';self.token_precision='float16';self.calls=0
    def encode(self,crops):
        self.calls+=1
        g=448//self.patch_size
        return np.tile(np.array([1.,0.],dtype=np.float16),(len(crops),g*g,1))


@pytest.mark.parametrize('name,patch',[('uni',16),('uni2_h',14),('kaiko_vitb8',8)])
def test_real_occupancy_geometry_and_pollution_rejection(tmp_path,name,patch):
    cfg=load_config(Path(__file__).resolve().parents[1]/'configs/uni_memory.yaml')
    encoder=StubEncoder(name,patch);store=FeatureStore(tmp_path/'cache',encoder,cfg)
    image=np.zeros((473,571,3),np.uint8)
    mask=np.zeros(image.shape[:2],np.uint8);mask[17:389,31:433]=1
    gtfile=tmp_path/'gt.bin';gtfile.write_bytes(mask.tobytes())
    for offset in [0,8]:
        _,positions=store.get(image,'sample',offset)
        occ,meta=store.occupancy(mask,'sample',gtfile,offset,positions)
        expected=np.stack([torch.nn.functional.avg_pool2d(torch.from_numpy(crop.astype(np.float32))[None,None],patch,patch).numpy().ravel() for _,_,crop in windows(mask,448,224,offset)])
        np.testing.assert_array_equal(occ,expected)
        assert meta['patch_size']==patch and meta['occupancy_source']=='real_support_gt'
        path=tmp_path/f'cache/occupancy/sample_offset{offset}.npz'
        np.savez_compressed(path,fg_occupancy=np.zeros_like(occ),metadata=json.dumps({'occupancy_source':'placeholder'}))
        repaired,_=store.occupancy(mask,'sample',gtfile,offset,positions)
        np.testing.assert_array_equal(repaired,expected)
        g=448//patch;views=list(windows(image,448,224,offset));maps=np.full((len(views),g,g),.125)
        native=reconstruct(maps,positions,mask.shape,448)
        np.testing.assert_array_equal(resize_map(native,(256,256)),np.full((256,256),.125))
    assert not np.array_equal(store.records[0]['fingerprint'],store.records[1]['fingerprint'])


def test_cache_fingerprint_changes_with_preprocessing_and_pixels(tmp_path):
    cfg=load_config(Path(__file__).resolve().parents[1]/'configs/uni_memory.yaml')
    encoder=StubEncoder('uni',16);store=FeatureStore(tmp_path,encoder,cfg)
    image=np.zeros((448,448,3),np.uint8)
    store.get(image,'same',0);store.get(image,'same',0)
    assert encoder.calls==1 and store.records[-1]['cache_reused']
    encoder.metadata['preprocessing']['mean']=[0.,0.,0.]
    store.get(image,'same',0)
    assert encoder.calls==2
    image[0,0]=255;store.get(image,'same',0)
    assert encoder.calls==3
