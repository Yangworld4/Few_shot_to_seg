"""Deterministic seed-14 error-category panels; all masks share one image/GT."""
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from pathology_trainfree.data import read_manifest,load_image,load_mask,resize_mask
from pathology_trainfree.diagnostics import error_diagnostics
from run_complementarity import OUT,MODELS,config_for,write_csv,atomic_json,log


def main():
    policy=json.loads((OUT/'configs/visualization_selection.json').read_text());seed=policy['seed'];selected=[];counts=[]
    image_metrics=pd.read_csv(OUT/'metrics/individual_per_image.csv')
    for ds in ['GlaS','CRAG']:
        cfg=config_for(ds,seed,'uni');entries=read_manifest(cfg['dataset']['query_manifest'],require_mask=True)
        byid={e['id']:e for e in entries};items={}
        for e in entries:
            iid=e['id'];gt=resize_mask(load_mask(e['mask'],cfg['dataset']),(256,256)).astype(bool)
            preds={};margins={}
            for m in MODELS:
                with np.load(OUT/f'predictions/{ds}/{m}/seed_{seed}/{iid}.npz') as z:preds[m]=z['mask'].astype(bool);margins[m]=z['margin']
            d=error_diagnostics(preds,gt);err={m:preds[m]!=gt for m in MODELS}
            scores=image_metrics[(image_metrics.dataset==ds)&(image_metrics.seed==seed)&(image_metrics.image_id==iid)]
            assert len(scores)==3 and set(scores.model)==set(MODELS)
            items[iid]={'gt':gt,'preds':preds,'margins':margins,'diagnostics':d,
                'all_good':bool((scores.dice>=.9).all()),
                'uni_corrected':int((err['uni']&(~err['uni2_h']|~err['kaiko_vitb8'])).sum()),
                'others_corrected_by_uni':int((~err['uni']&(err['uni2_h']|err['kaiko_vitb8'])).sum()),
                'shared_failure':int((err['uni']&err['uni2_h']&err['kaiko_vitb8']).sum())}
        used=set()
        for category in policy['categories']:
            eligible=[iid for iid,item in items.items() if iid not in used and item[category]>0]
            order=sorted(eligible) if category=='all_good' else sorted(eligible,key=lambda iid:(-items[iid][category],iid))
            chosen=order[:policy['cases_per_category']];used.update(chosen)
            counts.append({'dataset':ds,'category':category,'eligible_remaining':len(eligible),'saved':len(chosen),'seed':seed})
            for iid in chosen:
                item=items[iid];gt=item['gt'];preds=item['preds'];d=item['diagnostics'];margins=item['margins']
                fig,axes=plt.subplots(3,4,figsize=(16,13));fig.subplots_adjust(left=.025,right=.94,top=.90,bottom=.025,hspace=.22,wspace=.22);axes=axes.ravel()
                rgb=np.asarray(Image.fromarray(load_image(byid[iid]['image'])).resize((256,256),Image.Resampling.BILINEAR))
                panels=[('Original (same RGB, resized)',rgb),('GT (nearest)',gt),('UNI',preds['uni']),('UNI2-h',preds['uni2_h']),('Kaiko ViT-B/8',preds['kaiko_vitb8']),('Majority voting',d['majority'])]
                for ax,(title,arr) in zip(axes,panels):ax.imshow(arr,cmap='gray',vmin=0,vmax=1) if arr.ndim==2 else ax.imshow(arr);ax.set_title(title,fontsize=10);ax.axis('off')
                cmap=ListedColormap(['#f4f4f4','#2878b5','#ee982c','#c82423']);norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4)
                for ax,other in zip(axes[6:8],MODELS[1:]):
                    ea=preds['uni']!=gt;eb=preds[other]!=gt
                    code=np.zeros(gt.shape,int);code[ea&~eb]=1;code[~ea&eb]=2;code[ea&eb]=3
                    ax.imshow(code,cmap=cmap,norm=norm);ax.set_title(f'UNI vs {other}\nBlue: corrects UNI | Orange: adds errors\nRed: shared error',fontsize=8.5);ax.axis('off')
                allwrong=np.stack([p!=gt for p in preds.values()]).all(0)
                axes[8].imshow(allwrong,cmap='Reds',vmin=0,vmax=1);axes[8].set_title('Triple shared error');axes[8].axis('off')
                for ax,m in zip(axes[9:],MODELS):
                    limit=max(float(np.abs(margins[m]).max()),1e-8)
                    im=ax.imshow(margins[m],cmap='coolwarm',vmin=-limit,vmax=limit)
                    ax.set_title(f'{m}: own margin scale');ax.axis('off');fig.colorbar(im,ax=ax,fraction=.045,pad=.02)
                fig.suptitle(f'{ds} | seed {seed} | {iid} | {category}\nPost-hoc deterministic selection; 256 x 256 common grid',fontsize=15)
                path=OUT/f'visualizations/{ds}/{category}_{iid}.png';path.parent.mkdir(parents=True,exist_ok=True)
                fig.savefig(path,dpi=120);plt.close(fig)
                selected.append({'dataset':ds,'seed':seed,'image_id':iid,'category':category,'selection_score':item[category],'path':str(path)})
            log(f'panels {ds} {category}: {len(chosen)}')
    write_csv(OUT/'visualizations/selection_manifest.csv',selected);write_csv(OUT/'visualizations/category_counts.csv',counts)
    atomic_json(OUT/'visualizations/selection_policy.json',policy)
    index=['# 可视化案例索引','', '固定Seed14；规则见selection_policy.json。每张图均显示同一原图/GT与对齐预测。','']
    for r in selected:
        relative=__import__('pathlib').Path(r['path']).relative_to(OUT/'visualizations')
        label=str(r['dataset'])+' / '+str(r['category'])+' / '+str(r['image_id'])
        index.append('- ['+label+']('+str(relative)+')')
    (OUT/'visualizations/index.md').write_text('\n'.join(index),encoding='utf-8')

if __name__=='__main__':main()
