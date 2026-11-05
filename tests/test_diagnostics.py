import numpy as np
import pytest
from pathology_trainfree.diagnostics import error_diagnostics,boundary_regions,margin_auroc
from pathology_trainfree.config import load_config
from pathlib import Path


def test_all_eight_error_patterns():
    # columns enumerate 000,001,...111 errors against alternating GT.
    gt=np.array([[0,1,0,1,0,1,0,1]],dtype=bool)
    errors=np.array([[0,0,0,0,1,1,1,1],[0,0,1,1,0,0,1,1],[0,1,0,1,0,1,0,1]],bool).reshape(3,1,8)
    p={k:np.logical_xor(gt,e) for k,e in zip(['uni','uni2_h','kaiko_vitb8'],errors)}
    d=error_diagnostics(p,gt);a=d['pairwise'][0];t=d['triple']
    assert (a['shared_error'],a['a_only_error'],a['b_only_error'],a['error_union'])==(2,2,2,6)
    assert a['error_iou']==pytest.approx(1/3)
    assert d['coverage'][0]['correction']==.5 and d['coverage'][0]['damage']==.5
    assert t['all_correct']==t['all_wrong']==1
    assert t['exactly_one_correct']==t['exactly_two_correct']==3
    assert t['only_uni_correct']==t['only_uni2_h_correct']==t['only_kaiko_vitb8_correct']==1
    assert t['triple_shared_fn']==1 and t['triple_shared_fp']==0
    assert t['all_wrong_over_uni_error']==.25 and t['oracle_correctable_uni_ratio']==.75
    assert t['disagreement']==6 and t['disagreement_majority_accuracy']==.5
    np.testing.assert_array_equal(d['majority']!=gt,errors.sum(0)>=2)
    np.testing.assert_array_equal(d['oracle']!=gt,errors.all(0))


def test_direction_and_empty_denominators():
    gt=np.zeros((1,4),bool)
    p={'a':np.array([[1,1,0,0]]),'b':np.array([[1,0,0,0]]),'c':gt}
    d=error_diagnostics(p,gt)
    ab=next(x for x in d['coverage'] if x['model_a']=='a' and x['model_b']=='b' and x['tissue']=='overall')
    ba=next(x for x in d['coverage'] if x['model_a']=='b' and x['model_b']=='a' and x['tissue']=='overall')
    assert ab['correction']==.5 and ba['correction']==0
    assert ab['damage']==0 and ba['damage']==pytest.approx(1/3)
    assert all(x['correction'] is None and x['damage'] is None for x in d['coverage'] if x['tissue']=='gt_fg')
    perfect=error_diagnostics({'a':gt,'b':gt,'c':gt},gt)
    assert perfect['pairwise'][0]['error_iou'] is None
    assert perfect['triple']['oracle_correctable_uni_ratio'] is None


def test_boundary_and_auroc():
    gt=np.zeros((32,32),bool);gt[:,16:]=1
    r=boundary_regions(gt)
    assert np.stack(list(r.values())).sum(0).min()==np.stack(list(r.values())).sum(0).max()==1
    assert r['near_boundary'][:,11:21].all() and r['interior'][:,:7].all()
    assert boundary_regions(np.zeros((2,2),bool))['interior'].all()
    assert margin_auroc(np.array([0,1,2,3]),np.array([0,0,1,1]))==1
    assert margin_auroc(np.zeros(4),np.array([0,0,1,1]))==.5
    assert margin_auroc(np.zeros(4),np.zeros(4)) is None


@pytest.mark.parametrize('name,patch',[('uni',16),('uni2_h',14),('kaiko_vitb8',8)])
def test_encoder_specific_window_validation(name,patch):
    from pathology_trainfree.config import validate
    cfg=load_config(Path(__file__).resolve().parents[1]/'configs/uni_memory.yaml')
    cfg['encoder']['name']=name;validate(cfg)
    cfg['window']['size']=patch*3;cfg['window']['stride']=patch;validate(cfg)
    cfg['window']['stride']=patch
    cfg['window']['size']=patch*3+1
    with pytest.raises(ValueError,match='multiple'):validate(cfg)
