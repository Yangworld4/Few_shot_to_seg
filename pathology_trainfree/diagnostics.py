"""Post-hoc binary error diagnosis. GT is never used by the predictor."""
from itertools import combinations, permutations
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.stats import rankdata


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def boundary_regions(gt):
    gt=np.asarray(gt,bool)
    edge=np.zeros_like(gt)
    v=gt[1:]!=gt[:-1]; edge[1:]|=v;edge[:-1]|=v
    v=gt[:,1:]!=gt[:,:-1];edge[:,1:]|=v;edge[:,:-1]|=v
    d=distance_transform_edt(~edge) if edge.any() else np.full(gt.shape,np.inf)
    return {'near_boundary':d<=4,'transition':(d>4)&(d<=8),'interior':d>8}


def error_diagnostics(predictions, gt, region=None):
    gt=np.asarray(gt,bool)
    if len(predictions)!=3 or any(np.asarray(x).shape!=gt.shape for x in predictions.values()):
        raise ValueError('Require three predictions aligned with the same GT')
    if any(not np.isin(x,[0,1]).all() for x in predictions.values()):
        raise ValueError('Predictions must be binary')
    region=np.ones_like(gt) if region is None else np.asarray(region,bool)
    names=list(predictions)
    preds={k:np.asarray(v,bool) for k,v in predictions.items()}
    err={k:v!=gt for k,v in preds.items()}
    n=int(region.sum())
    pairwise=[];coverage=[]
    def count(x): return int((x&region).sum())
    for a,b in combinations(names,2):
        ea,eb=err[a],err[b]
        row={'model_a':a,'model_b':b,'pixels':n,'error_union':count(ea|eb)}
        for key,value in {'a_only_error':ea&~eb,'b_only_error':eb&~ea,'shared_error':ea&eb,
                          'shared_fp':ea&eb&~gt,'shared_fn':ea&eb&gt}.items():
            row[key]=count(value);row[key+'_ratio']=ratio(row[key],n)
        row['error_iou']=ratio(row['shared_error'],row['error_union'])
        pairwise.append(row)
    for a,b in permutations(names,2):
        for label,condition in [('overall',np.ones_like(gt)),('gt_fg',gt),('gt_bg',~gt)]:
            source_error=count(err[a]&condition);source_correct=count(~err[a]&condition)
            corrected=count(err[a]&~err[b]&condition);damaged=count(~err[a]&err[b]&condition)
            coverage.append({'model_a':a,'model_b':b,'tissue':label,'pixels':count(condition),
                             'source_errors':source_error,'corrected':corrected,'correction':ratio(corrected,source_error),
                             'source_correct':source_correct,'damaged':damaged,'damage':ratio(damaged,source_correct)})
    errors=np.stack([err[k] for k in names]);wrong=errors.sum(0);allwrong=wrong==3
    disagree=~(np.stack(list(preds.values())).all(0)|~np.stack(list(preds.values())).any(0))
    vote=np.stack(list(preds.values())).sum(0)>=2
    oracle=np.where(allwrong,~gt,gt)
    triple={'pixels':n,'uni_errors':count(err[names[0]]),'disagreement':count(disagree)}
    masks={'all_correct':wrong==0,'all_wrong':allwrong,'exactly_two_correct':wrong==1,
           'exactly_one_correct':wrong==2,'triple_shared_fp':allwrong&~gt,'triple_shared_fn':allwrong&gt}
    for k in names:
        masks['only_'+k+'_correct']=(wrong==2)&~err[k]
    for key,value in masks.items():
        triple[key]=count(value);triple[key+'_ratio']=ratio(triple[key],n)
    triple['all_wrong_over_uni_error']=ratio(triple['all_wrong'],triple['uni_errors'])
    triple['oracle_correctable_uni_errors']=triple['uni_errors']-triple['all_wrong']
    triple['oracle_correctable_uni_ratio']=ratio(triple['oracle_correctable_uni_errors'],triple['uni_errors'])
    triple['disagreement_ratio']=ratio(triple['disagreement'],n)
    for k,pred in {**preds,'majority':vote}.items():
        triple['disagreement_'+k+'_correct']=count(disagree&(pred==gt))
        triple['disagreement_'+k+'_accuracy']=ratio(triple['disagreement_'+k+'_correct'],triple['disagreement'])
    rates=[{'model':k,'pixels':n,'errors':count(err[k]),'error_rate':ratio(count(err[k]),n)} for k in names]
    return {'pairwise':pairwise,'coverage':coverage,'triple':triple,'rates':rates,'majority':vote,'oracle':oracle}


def margin_auroc(margin,gt):
    y=np.asarray(gt,bool).ravel();s=np.asarray(margin).ravel()
    if not np.isfinite(s).all(): raise ValueError('Non-finite margins')
    npos=int(y.sum());nneg=len(y)-npos
    if not npos or not nneg:return None
    return float((rankdata(s)[y].sum()-npos*(npos+1)/2)/(npos*nneg))
