"""Aggregate E0/E1/E2 with failures retained; create full-frame diagnostic panels."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import yaml
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from diagnose_self_reference_seeds import csv,get_seed
from pathology_trainfree.data import read_manifest,load_image,load_mask,resize_mask
from pathology_trainfree.geometry import reconstruct
from pathology_trainfree.self_reference.diagnostics import auc
OUT=ROOT/'outputs/self_reference_seed_diagnosis'

def main():
    cfg=yaml.safe_load((OUT/'configs/resolved.yaml').read_text());s=pd.read_csv(OUT/'seed_metrics_per_image.csv');t=pd.read_csv(OUT/'token_reliability.csv');f=pd.read_csv(OUT/'memory_feasibility.csv');r=pd.read_csv(OUT/'self_reference_per_image.csv');obj=pd.read_csv(OUT/'object_seed_coverage.csv')
    token=[]
    for (ds,g),sub in t.groupby(['dataset','generator']):
        feasible=f[(f.dataset==ds)&(f.generator==g)];row={'dataset':ds,'generator':g,'images':len(sub),'valid_memories':int((feasible.status=='VALID').sum()),'valid_rate':float((feasible.status=='VALID').mean()),'empty_memories':int((feasible.status=='EMPTY_MEMORY').sum()),'insufficient_memories':int((feasible.status=='INSUFFICIENT_MEMORY').sum()),'macro_token_coverage':sub.token_coverage.mean(),'pooled_token_coverage':(sub.reliable_fg_tokens.sum()+sub.reliable_bg_tokens.sum())/sub.total_tokens.sum()}
        for cls in ['fg','bg']:
            count=sub[f'reliable_{cls}_tokens'];row[cls+'_tokens_total']=int(count.sum());row[cls+'_tokens_mean']=count.mean();row[cls+'_tokens_median']=count.median();row[cls+'_zero_images']=int((count==0).sum());row[cls+'_one_to49_images']=int(((count>0)&(count<50)).sum());row[cls+'_pooled_gt_mean_purity']=sub[cls+'_purity_sum'].sum()/count.sum() if count.sum() else None;row[cls+'_pooled_gt_strict_purity']=sub[cls+'_strict_correct_tokens'].sum()/count.sum() if count.sum() else None;row[cls+'_macro_gt_mean_purity']=sub[cls+'_gt_mean_purity'].mean()
        token.append(row)
    csv(OUT/'token_reliability_summary.csv',token)
    ob=[]
    for key,sub in obj.groupby(['dataset','generator','scale','size_group']):
        ob.append(dict(zip(['dataset','generator','scale','size_group'],key))|{'gt_objects':len(sub),'seeded_objects':int(sub.has_fg_seed.sum()),'seeded_fraction':sub.has_fg_seed.mean(),'mean_object_seed_coverage':sub.seed_coverage.mean(),'object_area_weighted_seed_coverage':sub.fg_seed_pixels.sum()/sub.area.sum()})
    csv(OUT/'object_seed_coverage_summary.csv',ob)
    failures=s.merge(f,on=['dataset','image_id','generator']).merge(t,on=['dataset','image_id','generator'],suffixes=('','_token'))
    failures['wrong_fg_seed_pixels']=failures.fg_seed_pixels-failures.fg_correct_pixels;failures['wrong_bg_seed_pixels']=failures.bg_seed_pixels-failures.bg_correct_pixels;failures['unseeded_gt_objects_256']=failures.gt_objects_256-failures.seeded_objects_256
    csv(OUT/'failure_cases.csv',failures.to_dict('records'))
    paired=[]
    for (ds,g),sub in r.groupby(['dataset','generator']):
        a=sub[(sub['mode']=='S0')&(sub.status=='VALID')].set_index('image_id');b=sub[(sub['mode']=='S1')&(sub.status=='VALID')].set_index('image_id');ids=sorted(set(a.index)&set(b.index))
        row={'dataset':ds,'generator':g,'paired_images':len(ids),'all_images':sub.image_id.nunique(),'paired_image_ids':';'.join(ids)}
        for metric in ['dice','fg_iou','object_f1','margin_auroc','predicted_fg_ratio']:
            row['S0_'+metric]=a.loc[ids,metric].mean() if ids else None;row['S1_'+metric]=b.loc[ids,metric].mean() if ids else None;row['S1_minus_S0_'+metric]=(b.loc[ids,metric]-a.loc[ids,metric]).mean() if ids else None
        paired.append(row)
    csv(OUT/'self_matching_paired_comparison.csv',paired)
    ex=pd.read_csv(OUT/'seed_expansion_metrics.csv');expanded=[]
    if len(ex):
        for (ds,g,mode,region),sub in ex.groupby(['dataset','generator','mode','region']):
            expanded.append({'dataset':ds,'generator':g,'mode':mode,'region':region,'valid_images':len(sub),'pooled_fg_recall':sub.true_fg_predicted.sum()/sub.gt_fg_pixels.sum() if sub.gt_fg_pixels.sum() else None,'pooled_bg_specificity':sub.true_bg_predicted.sum()/sub.gt_bg_pixels.sum() if sub.gt_bg_pixels.sum() else None,'macro_auroc':sub.margin_auroc.mean(),'macro_auroc_valid_images':int(sub.margin_auroc.notna().sum()),'unseeded_objects':int(sub.unseeded_gt_objects.sum()),'recovered_unseeded_objects':int(sub.recovered_unseeded_objects.sum()),'pooled_unseeded_object_recovery':sub.recovered_unseeded_objects.sum()/sub.unseeded_gt_objects.sum() if sub.unseeded_gt_objects.sum() else None})
    csv(OUT/'seed_expansion_summary.csv',expanded,['dataset','generator','mode','region','valid_images','pooled_fg_recall','pooled_bg_specificity','macro_auroc','macro_auroc_valid_images','unseeded_objects','recovered_unseeded_objects','pooled_unseeded_object_recovery'])
    rules=[]
    for path in (OUT/'images').glob('*/*/morphology_rules.csv'):
        sub=pd.read_csv(path)
        if not len(sub):continue
        for rule,rr in sub.groupby('rule'):rules.append({'dataset':path.parent.parent.name,'image_id':path.parent.name,'rule':rule,'input_pixels':int(rr.input_pixels.sum()),'rejected_pixels':int(rr.rejected_pixels.sum()),'candidate_components':int((rr.input_pixels>0).sum())})
    csv(OUT/'morphology_rule_rejections.csv',rules)
    # Display selections are post-hoc diagnostics only; every image remains in all statistical tables.
    selected=[]
    for ds in s.dataset.unique():
        sub=failures[failures.dataset==ds].copy();valid=sub[sub.status=='VALID']
        options=[]
        if len(valid):options.append(('quantity_valid_seed_example',valid.sort_values('fg_precision',ascending=False).iloc[0]))
        for name,metric,ascending in [('high_precision_seed_example','fg_precision',False),('fg_seed_error','fg_precision',True),('bg_seed_error','bg_precision',True)]:
            keep=sub[sub[metric].notna()]
            if len(keep):options.append((name,keep.sort_values(metric,ascending=ascending).iloc[0]))
        missing=sub[sub.status!='VALID']
        if len(missing):options.append(('empty_or_insufficient_memory',missing.sort_values(['generator','fg_seed_pixels'],ascending=[False,False]).iloc[0]))
        valid_r=r[(r.dataset==ds)&(r.status=='VALID')&(r['mode']=='S0')]
        if len(valid_r):
            for name,ascending in [('retrieval_best_dice',False),('retrieval_worst_dice',True)]:
                rr=valid_r.sort_values('dice',ascending=ascending).iloc[0];options.append((name,sub[(sub.image_id==rr.image_id)&(sub.generator==rr.generator)].iloc[0]))
        entries=read_manifest(ROOT/cfg['datasets'][ds]['manifest'],require_mask=True);lookup={e['id']:e for e in entries}
        for name,row in options:
            id=row.image_id;g=row.generator;folder=OUT/'images'/ds/id;seed=get_seed(folder,g);image=load_image(lookup[id]['image']);gt=load_mask(lookup[id]['mask'],{'mask_encoding':'indexed','class_mapping':cfg['evaluation']['gt_encoding']})
            with np.load(folder/'intermediate_maps.npz') as z:maps={k:z[k] for k in ['slic','clusters','proposal','closed','accepted_inner','lumen_candidate','tissue']}
            case=OUT/'diagnostic_visualizations'/f'{ds}_{name}_{id}_{g}';case.mkdir(parents=True,exist_ok=True)
            slic=image.copy();slic[find_boundaries(maps['slic'])]=[255,255,0]
            tri=np.full(image.shape,150,np.uint8);tri[seed['fg']]=[230,70,40];tri[seed['bg']]=[30,110,230]
            errors=image.copy();errors[seed['fg']&(gt==0)]=[255,0,0];errors[seed['bg']&(gt==1)]=[0,80,255]
            morphology=np.zeros_like(image);morphology[...,0]=maps['accepted_inner']*255;morphology[...,1]=maps['closed']*255;morphology[...,2]=maps['proposal']*255
            with np.load(folder/f'token_mapping_{g}.npz') as z:
                fg=np.zeros(len(z['fg']),np.float32);bg=np.zeros_like(fg);fg[z['fg_candidate_indices']]=1;bg[z['bg_candidate_indices']]=1
                pos=z['positions'];offsets=z['offset_ids'];nwin=len(pos);fg=fg.reshape(nwin,28,28);bg=bg.reshape(nwin,28,28)
                fgmap=np.mean([reconstruct(fg[offsets==o],pos[offsets==o],gt.shape,448,'float32') for o in [0,8]],axis=0);bgmap=np.mean([reconstruct(bg[offsets==o],pos[offsets==o],gt.shape,448,'float32') for o in [0,8]],axis=0)
            token_rgb=np.stack([fgmap,np.zeros_like(fgmap),bgmap],axis=2)
            panels=[(image,'RGB'),(gt,'GT (diagnostic only)'),(slic,'SLIC'),(maps['clusters'],'K-means appearance'),(morphology,'Morphology: inner R / closed G / proposal B'),(seed['epithelial'],'Epithelial branch'+(' (A: undefined)' if g=='A' else '')),(seed['interior'],'Interior branch'+(' (A: undefined)' if g=='A' else '')),(seed['bg'],'Positive BG candidate'),(tri,'Final FG red / BG blue / Ignore gray'),(errors,'Wrong FG red / wrong BG blue'),(token_rgb,'Reliable tokens: FG red / BG blue')]
            available=[]
            for mode in ['S0','S1']:
                path=folder/f'retrieval_{g}_{mode}.npz'
                if path.exists():
                    with np.load(path) as z:margin=z['margin'];pred=z['mask']
                    panels +=[(margin,f'{mode} Margin (256)'),(pred,f'{mode} mask (256)')];available.append(mode)
                    np.save(case/f'{mode}_margin.npy',margin);Image.fromarray(pred.astype(np.uint8)*255).save(case/f'{mode}_mask.png')
            if not available:panels +=[(np.zeros((256,256)),f'E2 unavailable: {row.status}')]
            fig,axes=plt.subplots(3,5,figsize=(21,12));axes=axes.ravel()
            for ax,(data,title) in zip(axes,panels):
                if 'Margin' in title:im=ax.imshow(data,cmap='coolwarm',vmin=-.25,vmax=.25);fig.colorbar(im,ax=ax,fraction=.035)
                else:ax.imshow(data,cmap='tab10' if 'K-means' in title else 'gray')
                ax.set_title(title,fontsize=9);ax.axis('off')
            for ax in axes[len(panels):]:ax.axis('off')
            precision_fg='N/A' if pd.isna(row.fg_precision) else format(row.fg_precision,'.3f')
            precision_bg='N/A' if pd.isna(row.bg_precision) else format(row.bg_precision,'.3f')
            fig.suptitle(f'{ds} {id} / {g}: full RGB {gt.shape[0]}x{gt.shape[1]}; all views full-frame; prediction panels 256x256.\nFG P={precision_fg}, BG P={precision_bg}, FG coverage={row.fg_coverage:.4g}; pixels FG={int(row.fg_seed_pixels)}, BG={int(row.bg_seed_pixels)}; status={row.status}',fontsize=12)
            fig.tight_layout();fig.savefig(case/'diagnostic_panel.png',dpi=100);plt.close(fig)
            Image.fromarray(image).save(case/'original_rgb.png');Image.fromarray((gt*255).astype(np.uint8)).save(case/'gt_diagnostic.png');Image.fromarray(tri).save(case/'tristate.png');Image.fromarray(errors).save(case/'seed_error_overlay.png');Image.fromarray((maps['lumen_candidate']*255).astype(np.uint8)).save(case/'lumen_candidate.png')
            selected.append({'dataset':ds,'image_id':id,'generator':g,'selection':name,'directory':str(case),'E2_modes':';'.join(available)})
    csv(OUT/'visualization_selection.csv',selected)
    # Native-vs-256 luminance/GT overlays on predetermined first/middle/last queries.
    for ds in s.dataset.unique():
        entries=read_manifest(ROOT/cfg['datasets'][ds]['manifest'],require_mask=True);fig,axes=plt.subplots(3,3,figsize=(12,10))
        for row,index in enumerate([0,len(entries)//2,len(entries)-1]):
            entry=entries[index];rgb=load_image(entry['image']);gt=load_mask(entry['mask'],{'mask_encoding':'indexed','class_mapping':cfg['evaluation']['gt_encoding']})
            with np.load(OUT/'images'/ds/entry['id']/'intermediate_maps.npz') as z:lum=z['lumen_candidate']
            for ax,data,title in zip(axes[row],[rgb,gt,lum],[entry['id']+' RGB','Binary GT','RGB-derived lumen candidate']):ax.imshow(data,cmap='gray');ax.set_title(title);ax.axis('off')
        fig.tight_layout();fig.savefig(OUT/'diagnostic_visualizations'/f'{ds}_lumen_definition_audit.png',dpi=100);plt.close(fig)
    print('Summary and diagnostic panels complete',len(selected),flush=True)
if __name__=='__main__':main()
