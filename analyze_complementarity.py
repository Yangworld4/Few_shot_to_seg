"""Post-hoc diagnostics of aligned predictions, with pooled count denominators."""
import csv,json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from pathology_trainfree.data import read_manifest,load_mask,load_image,resize_mask
from pathology_trainfree.diagnostics import error_diagnostics,boundary_regions,margin_auroc,ratio
from pathology_trainfree.cache import save_npz
from run_complementarity import ROOT,OUT,SEEDS,MODELS,METRICS,config_for,score_mask,write_csv,atomic_json,log


def pool_pair(rows,keys):
    df=pd.DataFrame(rows);out=[]
    counts=['pixels','error_union','a_only_error','b_only_error','shared_error','shared_fp','shared_fn']
    for group,g in df.groupby(keys,sort=False,dropna=False):
        group=group if isinstance(group,tuple) else (group,)
        r=dict(zip(keys,group));r.update({k:int(g[k].sum()) for k in counts})
        r['error_iou']=ratio(r['shared_error'],r['error_union'])
        for k in counts[2:]:r[k+'_ratio']=ratio(r[k],r['pixels'])
        out.append(r)
    return out


def pool_coverage(rows,keys):
    out=[]
    for group,g in pd.DataFrame(rows).groupby(keys,sort=False,dropna=False):
        group=group if isinstance(group,tuple) else (group,)
        r=dict(zip(keys,group));r.update({k:int(g[k].sum()) for k in ['pixels','source_errors','corrected','source_correct','damaged']})
        r.update(correction=ratio(r['corrected'],r['source_errors']),damage=ratio(r['damaged'],r['source_correct']))
        out.append(r)
    return out


def pool_triple(rows,keys):
    out=[]
    for group,g in pd.DataFrame(rows).groupby(keys,sort=False,dropna=False):
        group=group if isinstance(group,tuple) else (group,);r=dict(zip(keys,group))
        countkeys=[k for k in g.columns if k not in keys+['image_id','region'] and not any(s in k for s in ['ratio','accuracy','over_uni'])]
        for k in countkeys:r[k]=int(g[k].sum())
        for k in ['all_correct','all_wrong','exactly_two_correct','exactly_one_correct','triple_shared_fp','triple_shared_fn']+['only_'+m+'_correct' for m in MODELS]+['disagreement']:
            r[k+'_ratio']=ratio(r[k],r['pixels'])
        r['all_wrong_over_uni_error']=ratio(r['all_wrong'],r['uni_errors'])
        r['oracle_correctable_uni_ratio']=ratio(r['oracle_correctable_uni_errors'],r['uni_errors'])
        for k in MODELS+['majority']:r['disagreement_'+k+'_accuracy']=ratio(r['disagreement_'+k+'_correct'],r['disagreement'])
        out.append(r)
    return out


def mean_sd_table(df,keys,values):
    out=[]
    for group,g in df.groupby(keys,sort=False):
        group=group if isinstance(group,tuple) else (group,);r=dict(zip(keys,group))
        r['n_seeds']=len(g)
        for k in values:
            x=pd.to_numeric(g[k],errors='coerce');r.update({k+'_mean':float(x.mean()),k+'_sd':float(x.std(ddof=1)),k+'_min':float(x.min()),k+'_max':float(x.max())})
        out.append(r)
    return out


def main():
    runs=[];individual=[];pairs=[];coverage=[];triples=[];boundary_pairs=[];boundary_cov=[];boundary_triples=[];rates=[];margins=[];fusion=[];fusion_images=[]
    for ds in ['GlaS','CRAG']:
        cfg=config_for(ds,14,'uni');entries=read_manifest(cfg['dataset']['query_manifest'],require_mask=True)
        for seed in SEEDS:
            for m in MODELS:
                directory=OUT/f'predictions/{ds}/{m}/seed_{seed}'
                summary=json.loads((directory/'complete.json').read_text());assert summary['status']=='COMPLETE'
                runs.append(summary);individual.extend(list(csv.DictReader((directory/'per_image_metrics.csv').open())))
            for n,entry in enumerate(entries,1):
                preds={};scores={};image_id=entry['id'];gt=resize_mask(load_mask(entry['mask'],cfg['dataset']),(256,256)).astype(bool)
                common={'dataset':ds,'seed':seed,'image_id':image_id}
                for m in MODELS:
                    with np.load(OUT/f'predictions/{ds}/{m}/seed_{seed}/{image_id}.npz') as z:
                        preds[m]=z['mask'].astype(bool);scores[m]=z['margin']
                    margins.append({**common,'model':m,'auroc':margin_auroc(scores[m],gt),
                                    'sign_error_rate':float((preds[m]!=gt).mean()),'margin_min':float(scores[m].min()),
                                    'margin_max':float(scores[m].max()),'margin_mean':float(scores[m].mean()),
                                    'scope':'model_own_scale_no_cross_model_magnitude_comparison'})
                d=error_diagnostics(preds,gt)
                pairs.extend({**common,**r} for r in d['pairwise']);coverage.extend({**common,**r} for r in d['coverage'])
                triples.append({**common,**d['triple']})
                for region,selection in boundary_regions(gt).items():
                    b=error_diagnostics(preds,gt,selection);base={**common,'region':region}
                    boundary_pairs.extend({**base,**r} for r in b['pairwise'])
                    boundary_cov.extend({**base,**r} for r in b['coverage'])
                    boundary_triples.append({**base,**b['triple']});rates.extend({**base,**r} for r in b['rates'])
                for method in ['majority','oracle']:
                    pred=d[method]
                    dest=OUT/f'predictions/{ds}/{method}/seed_{seed}/{image_id}'
                    save_npz(dest.with_suffix('.npz'),mask=pred.astype(np.uint8),uses_gt=np.array(method=='oracle'))
                    Image.fromarray(pred.astype(np.uint8)*255).save(dest.with_suffix('.png'))
                    fusion_images.append({**common,'model':method,**score_mask(pred,gt,cfg)})
            for method in ['majority','oracle']:
                rows=[r for r in fusion_images if r['dataset']==ds and r['seed']==seed and r['model']==method]
                summary={'dataset':ds,'seed':seed,'model':method,**{k:float(np.mean([r[k] for r in rows])) for k in METRICS},'uses_gt':method=='oracle'}
                same=[r for r in runs if r['dataset']==ds and r['seed']==seed]
                summary['delta_dice_vs_best_single']=summary['dice']-max(r['dice'] for r in same)
                summary['delta_dice_vs_uni']=summary['dice']-next(r['dice'] for r in same if r['model']=='uni')
                summary['best_single_posthoc']=max(same,key=lambda r:r['dice'])['model'];fusion.append(summary)
            log(f'diagnostics {ds} seed{seed} done')
    dest=OUT/'metrics';dest.mkdir(exist_ok=True)
    write_csv(dest/'individual_models.csv',runs);write_csv(dest/'individual_per_image.csv',individual)
    write_csv(dest/'individual_summary.csv',mean_sd_table(pd.DataFrame(runs),['dataset','model'],METRICS))
    for name,rows in [('pairwise_errors',pairs),('correction_coverage',coverage),('triple_errors',triples),('margin_diagnostics',margins),('fusion_per_image',fusion_images),('boundary_complementarity',rates),('boundary_correction_per_image',boundary_cov),('boundary_triple_per_image',boundary_triples),('boundary_pairwise_per_image',boundary_pairs)]:write_csv(dest/f'{name}.csv',rows)
    for name,rows,keys,fun in [
        ('pairwise_seed',pairs,['dataset','seed','model_a','model_b'],pool_pair),
        ('correction_seed',coverage,['dataset','seed','model_a','model_b','tissue'],pool_coverage),
        ('triple_seed',triples,['dataset','seed'],pool_triple),
        ('boundary_pairwise_seed',boundary_pairs,['dataset','seed','region','model_a','model_b'],pool_pair),
        ('boundary_correction_seed',boundary_cov,['dataset','seed','region','model_a','model_b','tissue'],pool_coverage),
        ('boundary_triple_seed',boundary_triples,['dataset','seed','region'],pool_triple)]:
        write_csv(dest/f'{name}.csv',fun(rows,keys))
    rdf=pd.DataFrame(rates);rr=[]
    for (ds,seed,region,model),g in rdf.groupby(['dataset','seed','region','model']):
        n=int(g['pixels'].sum());e=int(g['errors'].sum());rr.append({'dataset':ds,'seed':seed,'region':region,'model':model,'pixels':n,'errors':e,'error_rate':ratio(e,n)})
    write_csv(dest/'boundary_seed.csv',rr)
    for method,file in [('majority','majority_voting'),('oracle','oracle_upper_bound')]:write_csv(dest/f'{file}.csv',[r for r in fusion if r['model']==method])
    write_csv(dest/'fusion_summary.csv',mean_sd_table(pd.DataFrame(fusion),['dataset','model'],METRICS+['delta_dice_vs_best_single','delta_dice_vs_uni']))
    paired=[]
    for r in runs:
        base=next(x for x in runs if x['dataset']==r['dataset'] and x['seed']==r['seed'] and x['model']=='uni')
        paired.append({'dataset':r['dataset'],'seed':r['seed'],'model':r['model'],'delta_dice_vs_uni':r['dice']-base['dice']})
    write_csv(dest/'paired_dice.csv',paired)
    atomic_json(OUT/'diagnostics_complete.json',{'status':'COMPLETE','model_runs':len(runs),'model_image_predictions':len(individual),'aligned_image_conditions':len(triples),'fusion_masks':len(fusion_images),'zero_denominators':'None in JSON / blank CSV; exclude from macro means and retain denominators'})

if __name__=='__main__':main()
