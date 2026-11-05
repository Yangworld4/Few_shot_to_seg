"""E0/E1/E2 stage-separated RGB-only self-reference seed diagnosis."""
from pathlib import Path
import sys,argparse,json,hashlib,time,os
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pathology_trainfree.data import read_manifest,load_image,load_mask,resize_mask
from pathology_trainfree.geometry import reconstruct
from pathology_trainfree.config import load_config
from pathology_trainfree.memory import ReferenceMemory
from pathology_trainfree.pipeline import SegmentationPipeline
from pathology_trainfree.evaluation import metrics,object_metrics
from pathology_trainfree.self_reference.seed_generator import generate,validate
from pathology_trainfree.self_reference.token_mapping import map_seeds,load_query_cache
from pathology_trainfree.self_reference.retrieval import candidates,memory_status,SpatiallyExcludedRetriever
from pathology_trainfree.self_reference.diagnostics import seed_metrics,object_coverage,auc,expansion_metrics

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def signature(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def js(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,default=str));tmp.replace(path)
def csv(path,rows,fields=None):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.csv.tmp');pd.DataFrame(rows,columns=fields).to_csv(tmp,index=False);tmp.replace(path)
def save_npz(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    tmp.replace(path)
def get_seed(folder,g):
    with np.load(folder/f'seeds_{g}.npz') as z:return {k:z[k] for k in ['fg','bg','ignore','epithelial','interior']}
def seed_fp(image,cfg):return signature({'rgb_sha256':hashlib.sha256(image.tobytes()).hexdigest(),'config':cfg})

def stage0(datasets,cfg,out,generators,resume,log):
    # Generation pass for ALL images completes before evaluation loads ANY GT.
    for ds,entries in datasets.items():
        for i,e in enumerate(entries):
            image=load_image(e['image']);folder=out/'images'/ds/e['id'];marker=folder/'seed_complete.json';fp=seed_fp(image,cfg)
            if resume and marker.exists():
                if json.loads(marker.read_text())['fingerprint']!=fp:raise ValueError('Seed resume fingerprint differs')
                if set(generators).issubset(json.loads(marker.read_text())['generators']):continue
            seeds,maps=generate(image,cfg);folder.mkdir(parents=True,exist_ok=True)
            for g in generators:save_npz(folder/f'seeds_{g}.npz',**seeds[g])
            arrays={k:v.astype(np.float16) if v.dtype.kind=='f' else v for k,v in maps.items() if isinstance(v,np.ndarray)}
            save_npz(folder/'intermediate_maps.npz',**arrays);csv(folder/'morphology_rules.csv',maps['rules'],['rule','input_pixels','rejected_pixels','component','pass','enclosure','contrast'])
            js(marker,{'fingerprint':fp,'image_sha256':hashlib.sha256(image.tobytes()).hexdigest(),'generators':generators,'gt_read':False,'shape':image.shape[:2]})
            log(f'E0 generation {ds} {i+1}/{len(entries)} {e["id"]}: '+', '.join(f'{g} FG={seeds[g]["fg"].sum()} BG={seeds[g]["bg"].sum()}' for g in generators))
    rows=[];objects=[];lumen=[]
    for ds,entries in datasets.items():
        for e in entries:
            folder=out/'images'/ds/e['id'];gt=load_mask(e['mask'],{'mask_encoding':'indexed','class_mapping':cfg['evaluation']['gt_encoding']})
            with np.load(folder/'intermediate_maps.npz') as z:tissue=z['tissue'];lum=z['lumen_candidate']
            lumen.append({'dataset':ds,'image_id':e['id'],'candidate_pixels':int(lum.sum()),'candidate_gt_fg_pixels':int((lum&(gt==1)).sum()),'candidate_gt_fg_fraction':float(gt[lum].mean()) if lum.any() else None,'definition':'RGB-derived lumen candidate; GT used after generation only'})
            for g in generators:
                seed=get_seed(folder,g);validate(seed);row={'dataset':ds,'image_id':e['id'],'generator':g,**seed_metrics(seed,gt,tissue)}
                for scale in ['native','256']:
                    obs=object_coverage(gt,seed['fg'],cfg,scale);objects.extend([{'dataset':ds,'image_id':e['id'],'generator':g,**o} for o in obs]);row[f'gt_objects_{scale}']=len(obs);row[f'seeded_objects_{scale}']=sum(o['has_fg_seed'] for o in obs)
                rows.append(row)
    csv(out/'seed_metrics_per_image.csv',rows);csv(out/'object_seed_coverage.csv',objects,['dataset','image_id','generator','scale','object_id','area','area_min','size_group','fg_seed_pixels','has_fg_seed','seed_coverage']);csv(out/'lumen_gt_diagnostic.csv',lumen)
    summary=[];df=pd.DataFrame(rows)
    for (ds,g),sub in df.groupby(['dataset','generator']):
        row={'dataset':ds,'generator':g,'images':len(sub)}
        for key in ['fg_precision','bg_precision','fg_coverage','bg_coverage','ignore_ratio','tissue_ignore_ratio','fg_nonempty','bg_nonempty']:row['macro_'+key]=float(sub[key].mean()) if sub[key].notna().any() else None
        row.update({'pooled_fg_precision':sub.fg_correct_pixels.sum()/sub.fg_seed_pixels.sum() if sub.fg_seed_pixels.sum() else None,'pooled_bg_precision':sub.bg_correct_pixels.sum()/sub.bg_seed_pixels.sum() if sub.bg_seed_pixels.sum() else None,
                    'pooled_fg_coverage':sub.fg_correct_pixels.sum()/sub.gt_fg_pixels.sum(),'pooled_bg_coverage':sub.bg_correct_pixels.sum()/sub.gt_bg_pixels.sum(),
                    'pooled_ignore_ratio':sub.ignore_pixels.sum()/sub.pixels.sum(),'pooled_tissue_ignore_ratio':sub.tissue_ignore_pixels.sum()/sub.tissue_pixels.sum() if sub.tissue_pixels.sum() else None,
                    'fg_precision_valid_images':int(sub.fg_precision.notna().sum()),'bg_precision_valid_images':int(sub.bg_precision.notna().sum())})
        for scale in ['native','256']:
            row['pooled_object_seed_coverage_'+scale]=sub['seeded_objects_'+scale].sum()/sub['gt_objects_'+scale].sum() if sub['gt_objects_'+scale].sum() else None
        summary.append(row)
    csv(out/'seed_summary.csv',summary);js(out/'E0_complete.json',{'images':len(rows),'gt_used_for_generation':False});log('E0 complete; all images and empty seeds included')


def stage1(datasets,cfg,out,generators,resume,cache_root,log):
    if not (out/'E0_complete.json').exists():raise RuntimeError('Complete E0 first')
    rows=[];feas=[];cache_records=[]
    for ds,entries in datasets.items():
        for i,e in enumerate(entries):
            image=load_image(e['image']);folder=out/'images'/ds/e['id']
            seed_record=json.loads((folder/'seed_complete.json').read_text())
            if seed_record['fingerprint']!=seed_fp(image,cfg) or not set(generators).issubset(seed_record['generators']):
                raise ValueError('Seed stage fingerprint differs; rerun E0 before E1')
            stamp={'fingerprint':seed_fp(image,cfg),'gt_sha256':sha(e['mask']),'generators':generators}
            marker=folder/'E1_complete.json'
            if resume and marker.exists():
                previous=json.loads(marker.read_text())
                if all(previous.get(k)==v for k,v in stamp.items()):
                    rows.extend(previous['rows']);feas.extend(previous['feasibility']);cache_records.extend(previous['cache_records']);log(f'E1 resume {ds}/{e["id"]}');continue
            token_rows=[];feas_rows=[]
            tokens,positions,offset_ids,records=load_query_cache(image,e['id'],cache_root,ds,cfg);cache_records.extend([{'dataset':ds,'image_id':e['id'],**r} for r in records])
            selected_maps={}
            # Decide memory candidates entirely from generated Seeds, before loading GT.
            for g in generators:
                seed=get_seed(folder,g);m=map_seeds(seed,cfg['window']);assert np.array_equal(m['positions'],positions) and np.array_equal(m['offset_ids'],offset_ids)
                indices=candidates(m,cfg);rng=np.random.default_rng(cfg['memory']['sampling_seed']);chosen={}
                for c in [1,0]:
                    ix=indices[c];chosen[c]=np.sort(rng.choice(ix,cfg['memory']['cap_per_class'],replace=False)) if len(ix)>cfg['memory']['cap_per_class'] else ix
                selected_maps[g]=(m,indices,chosen)
                save_npz(folder/f'token_mapping_{g}.npz',**m,fg_candidate_indices=indices[1],bg_candidate_indices=indices[0],fg_memory_indices=chosen[1],bg_memory_indices=chosen[0])
                save_npz(folder/f'memory_{g}.npz',fg=tokens.reshape(-1,1024)[chosen[1]],bg=tokens.reshape(-1,1024)[chosen[0]],fg_centers=m['centers'][chosen[1]],bg_centers=m['centers'][chosen[0]],
                         metadata=np.array(json.dumps({'occupancy_source':'generated_tristate_seed','generator':g,'query_id':e['id'],'image_sha256':hashlib.sha256(image.tobytes()).hexdigest(),'encoder_sha256':cfg['encoder']['weights_sha256'],'sampling_seed':cfg['memory']['sampling_seed'],'gt_used':False})))
            gt=load_mask(e['mask'],{'mask_encoding':'indexed','class_mapping':cfg['evaluation']['gt_encoding']})
            # Oracle is an independent diagnostic path, never passed into candidate selection or memory saving.
            oracle_seed={'fg':gt==1,'bg':gt==0,'ignore':np.zeros(gt.shape,bool),'epithelial':np.zeros(gt.shape,bool),'interior':np.zeros(gt.shape,bool)}
            oracle=map_seeds(oracle_seed,cfg['window'])['fg'];save_npz(folder/'oracle_gt_occupancy_DIAGNOSTIC_ONLY.npz',fg_occupancy=oracle)
            for g,(m,indices,chosen) in selected_maps.items():
                counts={c:len(chosen[c]) for c in [0,1]};status=memory_status(counts,cfg['retrieval']['top_k']);row={'dataset':ds,'image_id':e['id'],'generator':g,'total_tokens':len(oracle),'reliable_fg_tokens':len(indices[1]),'reliable_bg_tokens':len(indices[0]),'ignore_tokens':len(oracle)-len(indices[1])-len(indices[0]),'token_coverage':(len(indices[1])+len(indices[0]))/len(oracle),
                     'fg_branch_epithelial_tokens':int(((m['epithelial']>=cfg['memory']['purity'])&(m['bg']==0)).sum()),'fg_branch_interior_tokens':int(((m['interior']>=cfg['memory']['purity'])&(m['bg']==0)).sum()),'fg_combined_only_tokens':int(((m['fg']>=cfg['memory']['purity'])&(m['bg']==0)&(m['epithelial']<cfg['memory']['purity'])&(m['interior']<cfg['memory']['purity'])).sum())}
                for c,name in [(1,'fg'),(0,'bg')]:
                    values=oracle[indices[c]] if c==1 else 1-oracle[indices[c]]
                    row[name+'_gt_mean_purity']=float(values.mean()) if len(values) else None;row[name+'_gt_strict_purity_rate']=float((values>=.9).mean()) if len(values) else None;row[name+'_gt_contamination']=float((1-values).mean()) if len(values) else None
                    row[name+'_purity_sum']=float(values.sum());row[name+'_strict_correct_tokens']=int((values>=.9).sum())
                rows.append(row);token_rows.append(row);feas.append({'dataset':ds,'image_id':e['id'],'generator':g,'fg_candidates':len(indices[1]),'bg_candidates':len(indices[0]),'fg_memory':counts[1],'bg_memory':counts[0],'status':status,'top_k':50,'gt_used_for_memory':False})
            js(marker,{**stamp,'rows':token_rows,'feasibility':feas[-len(generators):],
                       'cache_records':[{'dataset':ds,'image_id':e['id'],**r} for r in records]})
            log(f'E1 {ds} {i+1}/{len(entries)} {e["id"]}: '+', '.join(f'{g} {memory_status({c:len(ch[c]) for c in [0,1]},50)}' for g,(_,_,ch) in selected_maps.items()))
    csv(out/'token_reliability.csv',rows);csv(out/'memory_feasibility.csv',feas);csv(out/'query_cache_validation.csv',cache_records);js(out/'E1_complete.json',{'conditions':len(rows),'all_cache_entries_validated':True});log('E1 complete')


class FeatureStore:
    def __init__(self,tokens,offset_ids):self.tokens=tokens;self.offset_ids=offset_ids
    def get(self,image,image_id,offset):return self.tokens[self.offset_ids==offset],None


def stage2(datasets,cfg,out,generators,resume,cache_root,device,log):
    if not (out/'E1_complete.json').exists():raise RuntimeError('Complete E1 first')
    pcfg=load_config(ROOT/'configs/uni_memory.yaml');pcfg['retrieval'].update({'top_k':50,'chunk_size':cfg['retrieval']['chunk_size']});pcfg['segmentation'].update({'output_size':[256,256],'margin_precision':'float16','threshold':0.0,'native_resolution':False})
    enc=SimpleNamespace(device=torch.device(device),weights_sha256=cfg['encoder']['weights_sha256'],patch_size=16,feature_dim=1024,normalize=True,precision='float32',token_precision='float16')
    def never_encode(*args,**kwargs):raise RuntimeError('E0/E1/E2 must reuse validated cached UNI features')
    enc.encode=never_encode;rows=[];expansions=[];recovery=[]
    for ds,entries in datasets.items():
        for i,e in enumerate(entries):
            image=load_image(e['image']);folder=out/'images'/ds/e['id']
            marker=folder/'E2_complete.json';stamp={'fingerprint':seed_fp(image,cfg),'gt_sha256':sha(e['mask']),'generators':generators}
            token_record=json.loads((folder/'E1_complete.json').read_text())
            if any(token_record.get(k)!=v for k,v in stamp.items()):
                raise ValueError('Token stage fingerprint differs; rerun E1 before E2')
            if resume and marker.exists():
                previous=json.loads(marker.read_text())
                if all(previous.get(k)==v for k,v in stamp.items()):
                    rows.extend(previous['rows']);expansions.extend(previous['expansions']);recovery.extend(previous['objects']);log(f'E2 resume {ds}/{e["id"]}');continue
            begin_exp=len(expansions);begin_obj=len(recovery)
            tokens,positions,offsets,_=load_query_cache(image,e['id'],cache_root,ds,cfg);store=FeatureStore(tokens,offsets)
            prepared=[]
            for g in generators:
                with np.load(folder/f'memory_{g}.npz') as z:banks={0:z['bg'],1:z['fg']};centers={0:z['bg_centers'],1:z['fg_centers']}
                status=memory_status({c:len(banks[c]) for c in [0,1]},50)
                memory=ReferenceMemory(banks,{'encoder_sha256':enc.weights_sha256}) if status=='VALID' else None
                with np.load(folder/f'token_mapping_{g}.npz') as z:query_centers=z['centers']
                for mode in ['S0','S1']:
                    row={'dataset':ds,'image_id':e['id'],'generator':g,'mode':mode,'fg_memory':len(banks[1]),'bg_memory':len(banks[0]),'status':status,'top_k':50,'exclusion_radius':cfg['retrieval']['exclusion_radius'] if mode=='S1' else 0}
                    if status!='VALID':prepared.append((row,None,g,mode));continue
                    pipe=SegmentationPipeline(enc,memory,pcfg,feature_store=store)
                    if mode=='S1':
                        pipe.retriever=SpatiallyExcludedRetriever(memory,pcfg,device,query_centers,centers,cfg['retrieval']['exclusion_radius']);validity=pipe.retriever.feasibility()
                        row.update({f'{"fg" if c==1 else "bg"}_{k}':v for c,stats in validity.items() for k,v in stats.items()})
                        if any(stats['insufficient_query_tokens'] for stats in validity.values()):row['status']='INSUFFICIENT_MEMORY_AFTER_EXCLUSION';prepared.append((row,None,g,mode));continue
                    result=pipe.predict(image,e['id']);save_npz(folder/f'retrieval_{g}_{mode}.npz',margin=result['scores'],native_margin=result['native_scores'],mask=result['mask']);row['status']='VALID';prepared.append((row,result,g,mode))
            # GT only after inference, all decisions already made.
            gt=resize_mask(load_mask(e['mask'],{'mask_encoding':'indexed','class_mapping':cfg['evaluation']['gt_encoding']}),(256,256))
            for row,result,g,mode in prepared:
                if result is not None:
                    pred=result['mask'];v=metrics(pred,gt,[0,1])[1];of1,_=object_metrics(pred,gt,cfg['evaluation']['object_min_area_256'],cfg['evaluation']['object_match_overlap'])
                    row.update({'dice':v['dice'],'fg_iou':v['iou'],'object_f1':of1,'margin_auroc':auc(gt,result['scores']),'predicted_fg_ratio':float(pred.mean())})
                    seed=get_seed(folder,g);ex,obs=expansion_metrics(seed,gt,result['scores'],cfg);expansions.extend([{'dataset':ds,'image_id':e['id'],'generator':g,'mode':mode,**x} for x in ex]);recovery.extend([{'dataset':ds,'image_id':e['id'],'generator':g,'mode':mode,**x} for x in obs])
                rows.append(row)
            js(marker,{**stamp,'rows':[r for r,_,_,_ in prepared],'expansions':expansions[begin_exp:],'objects':recovery[begin_obj:]})
            csv(folder/'E2_results.csv',[r for r,_,_,_ in prepared]);log(f'E2 {ds} {i+1}/{len(entries)} {e["id"]} completed including invalid modes')
    fields=['dataset','image_id','generator','mode','fg_memory','bg_memory','status','top_k','exclusion_radius','fg_minimum_eligible','fg_insufficient_query_tokens','bg_minimum_eligible','bg_insufficient_query_tokens','dice','fg_iou','object_f1','margin_auroc','predicted_fg_ratio']
    csv(out/'self_reference_per_image.csv',rows,fields);csv(out/'seed_expansion_metrics.csv',expansions,['dataset','image_id','generator','mode','region','pixels','gt_fg_pixels','gt_bg_pixels','true_fg_predicted','true_bg_predicted','fg_recall','bg_specificity','margin_auroc','unseeded_gt_objects','recovered_unseeded_objects','unseeded_object_recovery_rate']);csv(out/'unseeded_object_recovery.csv',recovery,['dataset','image_id','generator','mode','scale','object_id','area','area_min','size_group','fg_seed_pixels','has_fg_seed','seed_coverage','pred_fg_pixels','pred_gt_object_coverage','recovered','originally_unseeded'])
    summary=[];df=pd.DataFrame(rows)
    for (ds,g,mode),sub in df.groupby(['dataset','generator','mode']):
        valid=sub[sub.status=='VALID'];row={'dataset':ds,'generator':g,'mode':mode,'all_images':len(sub),'valid_images':len(valid),'valid_rate':len(valid)/len(sub)}
        for status in ['EMPTY_MEMORY','INSUFFICIENT_MEMORY','INSUFFICIENT_MEMORY_AFTER_EXCLUSION']:row[status]=int((sub.status==status).sum())
        for key in ['dice','fg_iou','object_f1','margin_auroc','predicted_fg_ratio']:row['valid_only_'+key]=float(valid[key].mean()) if key in valid and len(valid) else None
        summary.append(row)
    csv(out/'self_reference_summary.csv',summary);js(out/'E2_complete.json',{'image_generator_modes':len(rows)});log('E2 complete')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default=str(ROOT/'configs/self_reference_e0.yaml'));parser.add_argument('--dataset',nargs='+',default=None);parser.add_argument('--stage',choices=['all','E0','E1','E2'],default='all');parser.add_argument('--seed-generator',nargs='+',choices=['A','B','C'],default=['A','B','C']);parser.add_argument('--cache-root');parser.add_argument('--output-root');parser.add_argument('--gpu',default='cuda:0');flags=parser.add_mutually_exclusive_group();flags.add_argument('--resume',action='store_true');flags.add_argument('--overwrite',action='store_true');args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text());out=Path(args.output_root or ROOT/cfg['output_root']).resolve();cache=Path(args.cache_root or ROOT/cfg['cache_root']).resolve();names=args.dataset or list(cfg['datasets'])
    out.mkdir(parents=True,exist_ok=True);frozen=out/'configs/resolved.yaml';frozen.parent.mkdir(exist_ok=True)
    if frozen.exists() and yaml.safe_load(frozen.read_text())!=cfg and not args.overwrite:raise ValueError('Configuration changed; use independent output root or explicit --overwrite')
    if (out/'E0_complete.json').exists() and not (args.resume or args.overwrite):raise ValueError('Existing experiment; explicitly use --resume or --overwrite')
    frozen.write_text(yaml.safe_dump(cfg,sort_keys=False));torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    def log(msg):
        line=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())+' '+msg;print(line,flush=True)
        with (out/'execution.log').open('a') as f:f.write(line+'\n')
    datasets={ds:read_manifest(ROOT/cfg['datasets'][ds]['manifest'],require_mask=True) for ds in names}
    # Dataset adapter only checks paths at this point; no GT pixel is read until all seed masks exist.
    source_files=[Path(__file__),*sorted((ROOT/'pathology_trainfree/self_reference').glob('*.py'))]
    js(out/'configs/preregistered_protocol.json',{'config':cfg,'config_sha256':signature(cfg),'command':' '.join(sys.argv),'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in source_files},'seed_gt_dependency':False,'parameters_selected_before_evaluation':True,'datasets':{ds:[e['id'] for e in es] for ds,es in datasets.items()}})
    js(out/f'configs/protocol_{args.stage}.json',json.loads((out/'configs/preregistered_protocol.json').read_text()))
    if args.stage in ['all','E0']:stage0(datasets,cfg,out,args.seed_generator,args.resume,log)
    if args.stage in ['all','E1']:stage1(datasets,cfg,out,args.seed_generator,args.resume,cache,log)
    if args.stage in ['all','E2']:stage2(datasets,cfg,out,args.seed_generator,args.resume,cache,args.gpu,log)
    log('Requested stages complete')
if __name__=='__main__':main()
