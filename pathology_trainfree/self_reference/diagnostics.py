"""GT-assisted metrics kept outside RGB-only seed construction and memory selection."""
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import rankdata
from ..data import resize_mask


def divide(a,b):return float(a/b) if b else None


def seed_metrics(seed,gt,tissue):
    fg,bg=seed['fg'],seed['bg'];truth=gt.astype(bool);nfg=int(fg.sum());nbg=int(bg.sum())
    tpfg=int((fg&truth).sum());tpbg=int((bg&~truth).sum())
    return {'fg_seed_pixels':nfg,'bg_seed_pixels':nbg,'gt_fg_pixels':int(truth.sum()),'gt_bg_pixels':int((~truth).sum()),
            'fg_correct_pixels':tpfg,'bg_correct_pixels':tpbg,'pixels':gt.size,'tissue_pixels':int(tissue.sum()),
            'ignore_pixels':int(seed['ignore'].sum()),'tissue_ignore_pixels':int((seed['ignore']&tissue).sum()),
            'fg_precision':divide(tpfg,nfg),'bg_precision':divide(tpbg,nbg),
            'fg_coverage':divide(tpfg,int(truth.sum())),'bg_coverage':divide(tpbg,int((~truth).sum())),
            'ignore_ratio':float(seed['ignore'].mean()),'tissue_ignore_ratio':divide(int((seed['ignore']&tissue).sum()),int(tissue.sum())),
            'fg_nonempty':nfg>0,'bg_nonempty':nbg>0}


def object_coverage(gt,fg,cfg,scale='native',pred=None):
    if scale=='256':gt=resize_mask(gt,(256,256));fg=resize_mask(fg.astype(np.uint8),(256,256)).astype(bool)
    factor=gt.size/(256*256);area_min=max(1,int(round(cfg['evaluation']['object_min_area_256']*factor)))
    bins=np.asarray(cfg['evaluation']['object_size_bins_256'])*factor
    labels,n=ndi.label(gt.astype(bool),structure=np.ones((3,3)));counts=np.bincount(labels.ravel());seed_counts=np.bincount(labels.ravel(),weights=fg.ravel(),minlength=n+1)
    if pred is not None:pred_counts=np.bincount(labels.ravel(),weights=pred.ravel(),minlength=n+1)
    rows=[]
    for i in range(1,n+1):
        if counts[i]<area_min:continue
        area=int(counts[i]);s=int(seed_counts[i]);row={'scale':scale,'object_id':i,'area':area,'area_min':area_min,'size_group':['small','medium','large'][np.searchsorted(bins,area,side='right')],
              'fg_seed_pixels':s,'has_fg_seed':s>0,'seed_coverage':s/area}
        if pred is not None:
            coverage=float(pred_counts[i]/area);row.update({'pred_fg_pixels':int(pred_counts[i]),'pred_gt_object_coverage':coverage,'recovered':coverage>cfg['evaluation']['recovery_overlap'],'originally_unseeded':s==0})
        rows.append(row)
    return rows


def auc(gt,margin):
    g=np.asarray(gt,bool).ravel();m=np.asarray(margin).ravel();a=int(g.sum());b=len(g)-a
    return float((rankdata(m)[g].sum()-a*(a+1)/2)/(a*b)) if a and b else None


def expansion_metrics(seed,gt,margin,cfg):
    seed={k:resize_mask(v.astype(np.uint8),margin.shape).astype(bool) for k,v in seed.items()};pred=margin>0;truth=gt.astype(bool);rows=[]
    for name,region in [('known_fg',seed['fg']),('known_bg',seed['bg']),('ignore',seed['ignore'])]:
        fg=region&truth;bg=region&~truth
        rows.append({'region':name,'pixels':int(region.sum()),'gt_fg_pixels':int(fg.sum()),'gt_bg_pixels':int(bg.sum()),'true_fg_predicted':int((pred&fg).sum()),'true_bg_predicted':int((~pred&bg).sum()),
                     'fg_recall':divide(int((pred&fg).sum()),int(fg.sum())),'bg_specificity':divide(int((~pred&bg).sum()),int(bg.sum())),
                     'margin_auroc':auc(truth[region],margin[region]) if region.any() else None})
    objects=object_coverage(gt,seed['fg'],cfg,'256',pred)
    unseeded=[r for r in objects if r['originally_unseeded']];recovered=sum(r['recovered'] for r in unseeded)
    for row in rows:row.update({'unseeded_gt_objects':len(unseeded),'recovered_unseeded_objects':recovered,'unseeded_object_recovery_rate':divide(recovered,len(unseeded))})
    return rows,objects
