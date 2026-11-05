"""Controlled CRAG stain/source diagnosis. Never trains, tunes against GT, or edits legacy outputs."""
from pathlib import Path
import sys, json, csv, hashlib, argparse, time, traceback, platform
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from scipy.stats import rankdata
from scipy import ndimage
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pathology_trainfree.data import load_image,load_mask,read_manifest,resize_mask
from pathology_trainfree.geometry import windows,reconstruct,resize_map
from pathology_trainfree.encoder import FeatureEncoder,sha256
from pathology_trainfree.cache import FeatureStore,save_npz,signature
from pathology_trainfree.memory import select_tokens,ReferenceMemory
from pathology_trainfree.retrieval import SimilarityRetriever
from pathology_trainfree.evaluation import metrics,object_metrics
from study_support_size import cache_file_for
import yaml
OUT=ROOT/'outputs/reference_source_stain_motivation'; OLD=ROOT/'outputs/support_size_study_gtfixed'
SEEDS=[14,24,42,1337,3407]; DC={'mask_encoding':'indexed','class_mapping':{0:0,255:1}}
# All choices fixed before any new segmentation measurements.
PROTO={'version':1,'seed':2026,'window':448,'stride':224,'offsets':[0,8],'patch':16,'k':50,'capacity':512,
 'occupancy':[.1,.9],'exclusion_gap_pixels':16,'exclusion':'axis-aligned footprint distance: disjoint beyond 16 px in either axis; reflected padding mapped to real pixels',
 'stain':'Macenko PCA OD>0.15 all channels; 1/99 angle percentiles; least-squares concentrations clipped nonnegative; p99 scaling',
 'templates':'nearest actual train RGB to 25th/75th PC1 percentile of robust-standardized stain vector/concentration descriptor; PC sign fixed by largest loading',
 'bootstrap':10000,'bootstrap_unit':'query, retain all five support pairs','similarity':'fp32 normalize -> fp16 dot/top50/mean -> fp32 scores',
 'smoke_support_seeds':[14,24],'smoke_query_manifest_indices':[0,1,2], 'unseeded':'exclude union of sampled real footprints dilated by fixed 16px; require entire resized pixel area unseeded',
 'invalid':'any token with <50 eligible neighbors invalidates full condition; no K reduction or imputation','template_quality_gate':'newly clipped RGB channels <1%; finite stain stats; exact geometry'}

def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,default=str));t.replace(p)
def table(p,rows,fields=None):
 fields=fields or list(dict.fromkeys(k for r in rows for k in r)) or ['status','reason']
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def log(s):
 print(s,flush=True)
 with (OUT/'execution.log').open('a') as f:f.write(time.strftime('%Y-%m-%dT%H:%M:%S')+' '+s+'\n')
def auc(g,x):
 g=np.asarray(g).ravel().astype(bool);x=np.asarray(x).ravel();n=g.sum();b=len(g)-n
 return float((rankdata(x)[g].sum()-n*(n+1)/2)/(n*b)) if n and b else None

def stain_fit(im):
 rgb=im.reshape(-1,3)[::max(1,im.size//3//50000)].astype(float)
 od=-np.log((rgb+1)/256);t=np.all(od>.15,axis=1)
 if t.sum()<100:raise ValueError('Insufficient OD tissue pixels')
 _,v=np.linalg.eigh(np.cov(od[t].T));v=v[:,-2:];proj=od[t]@v;angles=np.arctan2(proj[:,1],proj[:,0]);lo,hi=np.percentile(angles,[1,99])
 vec=np.stack([v@np.array([np.cos(a),np.sin(a)]) for a in [lo,hi]],axis=1)
 vec*=np.sign(vec.sum(0));vec=vec[:,np.argsort(-vec[0])];vec/=np.linalg.norm(vec,axis=0)
 c=np.maximum(0,np.linalg.lstsq(vec,od.T,rcond=None)[0].T);p99=np.percentile(c[t],99,axis=0)
 lab=cv2.cvtColor((rgb.reshape(-1,1,3)/255).astype(np.float32),cv2.COLOR_RGB2LAB).reshape(-1,3)
 return {'vectors':vec.tolist(),'concentration_p99':p99.tolist(),'concentration_quantiles':np.quantile(c[t],np.linspace(.01,.99,33),axis=0).tolist(),
  'lab_quantiles':np.quantile(lab[t],np.linspace(.01,.99,33),axis=0).tolist(),'lab_mean':lab[t].mean(0).tolist(),'lab_std':lab[t].std(0).tolist(),'tissue_fraction':float(t.mean())}

def transform(im,source,target):
 od=-np.log((im.reshape(-1,3).astype(float)+1)/256)
 c=np.maximum(0,np.linalg.lstsq(np.array(source['vectors']),od.T,rcond=None)[0].T)
 scale=np.array(target['concentration_p99'])/np.maximum(np.array(source['concentration_p99']),1e-6)
 rgb=256*np.exp(-(c*scale)@np.array(target['vectors']).T)-1
 clipped=float(np.mean((rgb<0)|(rgb>255)))
 return np.clip(np.round(rgb),0,255).astype(np.uint8).reshape(im.shape),{'concentration_scale':scale.tolist(),'new_clipped_channel_fraction':clipped}

def gap(a,b):
 va,vb=np.array(a['vectors']),np.array(b['vectors'])
 return {'lab_distance':float(np.mean(np.abs(np.array(a['lab_quantiles'])-b['lab_quantiles']))),
 'he_distance':float(np.mean(np.abs(np.array(a['concentration_quantiles'])-b['concentration_quantiles']))),
 'vector_angle_deg':float(np.degrees(np.arccos(np.clip(np.sum(va*vb,axis=0),-1,1))).mean())}

def prepare_templates(pool):
 path=OUT/'stain_templates.json'
 if path.exists():return json.loads(path.read_text())
 stats=[]
 for i,e in enumerate(pool):
  st=stain_fit(load_image(e['image']));stats.append({'id':e['id'],'rgb_sha256':sha256(e['image']),**st})
  if i%30==0:log(f'Train RGB stain statistics {i+1}/{len(pool)}')
 x=np.array([np.r_[np.array(s['vectors']).ravel(),np.log(s['concentration_p99'])] for s in stats]);med=np.median(x,axis=0);scale=np.maximum(np.subtract(*np.percentile(x,[75,25],axis=0)),1e-6)
 z=(x-med)/scale;_,_,v=np.linalg.svd(z-z.mean(0),full_matrices=False);pc=v[0];pc*=np.sign(pc[np.argmax(np.abs(pc))]);scores=z@pc
 ids=[int(np.argmin(abs(scores-np.quantile(scores,q)))) for q in [.25,.75]]
 result={'protocol':PROTO,'A':stats[ids[0]],'B':stats[ids[1]],'train_statistics':stats,'selection_pc':pc.tolist(),'selection_scores':scores.tolist(),'standardization_median':med.tolist(),'standardization_iqr':scale.tolist()}
 result['fingerprint']=signature(result);dump(path,result);return result

def geometry(mask):
 coords=[];occs=[];positions=[];lengths=[];h,w=mask.shape
 # Map reflected padded pixels back into original coordinates, including offset 8.
 yy=np.broadcast_to(np.arange(h)[:,None],(h,w));xx=np.broadcast_to(np.arange(w)[None,:],(h,w))
 for o in [0,8]:
  mv=list(windows(mask,448,224,o));yv=list(windows(yy,448,224,o));xv=list(windows(xx,448,224,o));pos=[]
  for (y,x,m),(_,_,yc),(_,_,xc) in zip(mv,yv,xv):
   occs.append(m.reshape(28,16,28,16).mean((1,3)).ravel());pos.append((y,x))
   yb=yc.reshape(28,16,28,16);xb=xc.reshape(28,16,28,16)
   coords.append(np.stack([yb.min((1,3)),xb.min((1,3)),yb.max((1,3))+1,xb.max((1,3))+1],-1).reshape(-1,4))
  positions.append(np.array(pos));lengths.append(len(pos)*784)
 return np.concatenate(coords),np.concatenate(occs),positions,lengths

def allowed(q,r,gap_px=16):
 q=q[:,None,:];r=r[None,:,:]
 return (q[:,:,2]+gap_px<r[:,:,0])|(r[:,:,2]+gap_px<q[:,:,0])|(q[:,:,3]+gap_px<r[:,:,1])|(r[:,:,3]+gap_px<q[:,:,1])

def checks():
 q=np.array([[0,0,16,16]]);refs=np.array([[0,0,16,16],[0,0,16,16],[8,8,24,24],[0,33,16,49],[0,16,16,32],[0,32,16,48]])
 want=[False,False,False,True,False,False];names=['self','duplicate_overlap_window','offset8_overlap','beyond_gap','touching','exact_gap_boundary']
 rows=[{'check':n,'eligible':bool(v),'expected':w,'passed':bool(v)==w} for n,v,w in zip(names,allowed(q,refs)[0],want)]
 # Non-overlapping cells with the same reflected real region must also be excluded.
 coord,_,_,_=geometry(np.zeros((512,512),np.uint8));assert len(coord)==6272
 rows.append({'check':'reflected_footprints_in_bounds','passed':bool((coord>=0).all() and (coord[:,[0,2]]<=512).all() and (coord[:,[1,3]]<=512).all())})
 table(OUT/'self_matching_checks.csv',rows);assert all(r['passed'] for r in rows)

class Experiment:
 def __init__(self):
  self.cfg=yaml.safe_load((OLD/'configs/CRAG/seed_14_shots_1.yaml').read_text())['config'];self.cfg['memory']['fallback_strategy']='none'
  self.pool=read_manifest(self.cfg['dataset']['support_manifest'],require_mask=True);self.queries=read_manifest(self.cfg['dataset']['query_manifest'],require_mask=True)
  assert len(self.pool)==153 and len(self.queries)==40
  rows=list(csv.DictReader((OLD/'support_sets.csv').open()));self.supports={s:next(r['image_id'] for r in rows if r['dataset']=='CRAG' and int(r['seed'])==s and int(r['shots'])==1) for s in SEEDS}
  assert not set(self.supports.values())&{e['id'] for e in self.queries}
  self.entries={e['id']:e for e in self.pool+self.queries};self.encoder=None;self.templates=None;self.stores={};self.banks={};self.quality=[];self.stain={};self.geometries={}
 def geom(self,id):
  if id not in self.geometries:
   e=self.entries[id];mask=load_mask(e['mask'],DC);g=geometry(mask)
   self.geometries[id]=g;save_npz(OUT/'cache/geometry'/f'{id}.npz',coordinates=g[0],occupancy=g[1],gt_sha256=np.array(sha256(e['mask'])))
  return self.geometries[id]
 def indices(self,id):
  co,oc,_,_=self.geom(id)
  ix,_=select_tokens(np.arange(len(oc))[:,None],np.stack([1-oc,oc],1),self.cfg['memory'],np.random.default_rng(2026))
  return {c:ix[c].ravel() for c in [0,1]}
 def image(self,id,style):
  raw=load_image(self.entries[id]['image'])
  if style=='RAW':return raw
  path=OUT/'cache/rgb'/style/f'{id}.png'
  if not path.exists():
   st=stain_fit(raw);im,pars=transform(raw,st,self.templates[style]);assert im.shape==raw.shape
   source_hash=hashlib.sha256(raw.tobytes()).hexdigest();pars.update({'image_id':id,'style':style,'rgb_source_hash':source_hash,'target_fingerprint':self.templates['fingerprint'],'geometry_equal':True,'rgb_hash':hashlib.sha256(im.tobytes()).hexdigest()})
   pars['transformation_fingerprint']=signature(pars);path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(im).save(path);dump(path.with_suffix('.json'),pars)
  return load_image(path)
 def features(self,id,style):
  im=self.image(id,style);g=self.geom(id)
  if style=='RAW':
   if id in {e['id'] for e in self.pool}:
    with np.load(cache_file_for(ROOT.parent/'DUAL','CRAG',id)) as z:t=z['features'].reshape(-1,1024).copy();pos=[np.stack([z['y'][z['passno']==j],z['x'][z['passno']==j]],1) for j in [0,1]]
   else:
    tt=[];pos=[]
    for o in [0,8]:
     with np.load(ROOT/f'outputs/support_size_study/cache/query_features/CRAG/feature_cache/{id}_offset{o}.npz') as z:
      from study_support_size import cache_key
      assert str(z['cache_key'])==cache_key(im,self.encoder,self.cfg,o)
      tt.append(z['tokens'].reshape(-1,1024).copy());pos.append(z['positions'])
    t=np.concatenate(tt)
   assert all(np.array_equal(a,b) for a,b in zip(pos,g[2]));return t
  store=self.stores[style];tt=[]
  for o,pos in zip([0,8],g[2]):
   tok,p=store.get(im,id,o);assert np.array_equal(p,pos);tt.append(tok.reshape(-1,1024))
  return np.concatenate(tt)
 def bank(self,id,style):
  key=(id,style)
  if key not in self.banks:
   t=self.features(id,style);ix=self.indices(id);co=self.geom(id)[0];b={c:t[ix[c]] for c in [0,1]};self.banks[key]=(b,{c:co[ix[c]] for c in [0,1]})
   save_npz(OUT/'memories'/f'{id}_{style}.npz',**{f'class_{c}':b[c] for c in [0,1]},**{f'indices_{c}':ix[c] for c in [0,1]},**{f'footprints_{c}':co[ix[c]] for c in [0,1]})
  return self.banks[key]
 def setup_encoder(self):
  self.encoder=FeatureEncoder(self.cfg['encoder']);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
  for st in ['A','B']:self.stores[st]=FeatureStore(OUT/'cache/features'/self.templates['fingerprint']/st,self.encoder,self.cfg)
 @torch.inference_mode()
 def infer(self,t,b,qco,rco,exclude,fp32=False):
  banks={c:F.normalize(torch.as_tensor(b[c].astype(np.float32),device='cuda'),dim=1) for c in [0,1]};scores=[];counts=[]
  for start in range(0,len(t),784):
   q=F.normalize(torch.as_tensor(t[start:start+784].astype(np.float32),device='cuda'),dim=1);ss=[];nn=[]
   for c in [0,1]:
    a=allowed(qco[start:start+784],rco[c]) if exclude else np.ones((len(q),len(b[c])),bool);n=a.sum(1);nn.append(n)
    if np.any(n<50):ss.append(np.full(len(q),np.nan,np.float32));continue
    sim=q@banks[c].T if fp32 else q.half()@banks[c].half().T
    sim=sim.masked_fill(~torch.as_tensor(a,device='cuda'),-torch.inf);ss.append(sim.topk(50,dim=1).values.mean(1).float().cpu().numpy())
   scores.append(np.stack(ss,1));counts.append(np.stack(nn,1))
  return np.concatenate(scores),np.concatenate(counts)
 def evaluate(self,t,id,rid,rs,qs,seed,source,exclude,fp32=False):
  cond=rs+qs if rs!='RAW' else 'RAW';mode='SpatialExclusion' if exclude else ('NoExclusion' if source=='Self' else 'External')
  name=f'{source}_{seed}_{rid}_{id}_{cond}_{mode}'+('_fp32' if fp32 else '')
  path=OUT/'cases'/f'{name}.json'
  if path.exists():return json.loads(path.read_text())
  g=self.geom(id);b,rc=self.bank(rid,rs);scores,count=self.infer(t,b,g[0],rc,exclude,fp32);valid=bool((count>=50).all())
  row={'query_id':id,'reference_id':rid,'seed':seed,'source':source,'condition':cond,'reference_style':rs,'query_style':qs,'mode':mode,'fp32_regression':fp32,'status':'VALID' if valid else 'INSUFFICIENT_SPATIAL_MEMORY' if exclude else 'INSUFFICIENT_MEMORY','fg_memory':len(b[1]),'bg_memory':len(b[0]),'query_tokens':len(t),'fg_min_eligible':int(count[:,1].min()),'bg_min_eligible':int(count[:,0].min()),'invalid_token_ratio':float(np.mean(np.any(count<50,axis=1))),'actual_k':50 if valid else None,'case':name}
  if valid:
   tm=scores[:,1]-scores[:,0];maps=[];cursor=0
   for pos,n in zip(g[2],g[3]):maps.append(reconstruct(tm[cursor:cursor+n].reshape(-1,28,28),pos,load_image(self.entries[id]['image']).shape[:2],448,'float16'));cursor+=n
   margin=resize_map(np.mean(maps,axis=0,dtype=np.float32),(256,256));gt=resize_mask(load_mask(self.entries[id]['mask'],DC),(256,256));pred=margin>0
   m=metrics(pred,gt,[0,1])[1];of,hd=object_metrics(pred,gt.astype(bool));row.update({'dice':m['dice'],'fg_iou':m['iou'],'precision':m['precision'],'recall':m['recall'],'object_f1':of,'object_hausdorff':hd,'predicted_fg_ratio':float(pred.mean()),'auroc':auc(gt,margin)})
   oc=g[1];pure=(oc>=.9)|(oc<=.1);row['token_auroc']=auc(oc[pure]>=.9,tm[pure])
   for c,sel in [(1,oc>=.9),(0,oc<=.1)]:
    row.update({f'class{c}_tokens':int(sel.sum()),f'class{c}_correct_cos':float(scores[sel,c].mean()) if sel.any() else None,f'class{c}_wrong_cos':float(scores[sel,1-c].mean()) if sel.any() else None,f'class{c}_margin':float(tm[sel].mean()) if sel.any() else None,f'class{c}_ranking_accuracy':float(((tm[sel]>0)==c).mean()) if sel.any() else None})
   if source=='Self':
    mask=load_mask(self.entries[id]['mask'],DC);seeded=np.zeros(mask.shape,bool)
    for co in np.concatenate(list(rc.values())):
     y,x,y1,x1=co;seeded[max(0,y-16):min(len(mask),y1+16),max(0,x-16):min(mask.shape[1],x1+16)]=True
    un=cv2.resize((~seeded).astype(np.float32),(256,256),interpolation=cv2.INTER_AREA)>.999999
    ug=gt[un];up=pred[un];um=margin[un];row['unseeded_fraction']=float(un.mean());row['unseeded_pixels']=int(un.sum());row['unseeded_fg_pixels']=int(ug.sum());row['unseeded_bg_pixels']=int((ug==0).sum())
    if len(ug):
     row['unseeded_dice']=metrics(up,ug,[0,1])[1]['dice'];row['unseeded_auroc']=auc(ug,um);row['unseeded_fg_recall']=float(up[ug==1].mean()) if np.any(ug==1) else None;row['unseeded_bg_specificity']=float((~up[ug==0]).mean()) if np.any(ug==0) else None
     lab,n=ndimage.label(gt,structure=np.ones((3,3)));eligible=[]
     for j in range(1,n+1):
      region=(lab==j)&un
      if region.sum()>=50:eligible.append(float(pred[region].mean()))
     row['unseeded_object_coverage']=float(np.mean(np.array(eligible)>.5)) if eligible else None;row['unseeded_objects']=len(eligible)
    row['unseeded_status']='VALID' if len(ug) and np.any(ug==1) and np.any(ug==0) else 'INSUFFICIENT_UNSEEDED_REGION'
   save_npz(OUT/'cases'/f'{name}.npz',margin=margin,token_scores=scores,eligible_counts=count)
  dump(path,row);return row
 def raw_regression(self):
  historical=list(csv.DictReader((OLD/'per_image_metrics.csv').open()));checks=[];rows=[]
  for q in self.queries:
   t=self.features(q['id'],'RAW')
   for s,rid in self.supports.items():
    rr=self.evaluate(t,q['id'],rid,'RAW','RAW',s,'External',False,True);rf=self.evaluate(t,q['id'],rid,'RAW','RAW',s,'External',False)
    h=next(r for r in historical if r['dataset']=='CRAG' and int(r['seed'])==s and int(r['shots'])==1 and r['image_id']==q['id'])
    mem=ReferenceMemory.load(OLD/f'runs/CRAG/seed_{s}/shots_1/memory.npz');b,_=self.bank(rid,'RAW');exact=all(np.array_equal(b[c],mem.vectors[c]) for c in [0,1])
    checks.append({'query_id':q['id'],'seed':s,'memory_exact':exact,'fp32_dice_error':rr['dice']-float(h['dice']),'fp16_dice_error':rf['dice']-float(h['dice'])});rows.append(rf)
   log('Raw regression '+q['id'])
  table(OUT/'raw_regression.csv',checks);means=[np.mean([r['dice'] for r in rows if r['seed']==s]) for s in SEEDS]
  exact=all(r['memory_exact'] and abs(r['fp32_dice_error'])<1e-12 for r in checks)
  result={'fp32_exact_parity':exact,'fp16_mean':float(np.mean(means)),'fp16_support_std_ddof1':float(np.std(means,ddof=1)),'fp16_max_image_error':max(abs(r['fp16_dice_error']) for r in checks),'fp32_max_image_error':max(abs(r['fp32_dice_error']) for r in checks)}
  dump(OUT/'raw_regression.json',result);(OUT/'regression_checks.md').write_text('# Raw regression\n\n'+json.dumps(result,indent=2)+'\n\nHistorical target used FP32 similarity. Main conditions use requested FP16 similarity. Identical raw features, indices and memory verified against historical saved files.\n')
  assert exact,'Raw FP32 regression failed; prohibit stain interpretation'
  assert abs(result['fp16_mean']-.8084210813678725)<.001,'Material precision discrepancy; stop'
 def quality_checks(self,ids):
  rows=[];stats={}
  for id in ids:
   raw=load_image(self.entries[id]['image']);stats[id]={}
   for st in ['RAW','A','B']:
    im=self.image(id,st);s=stain_fit(im);stats[id][st]=s;pars={} if st=='RAW' else json.loads((OUT/'cache/rgb'/st/f'{id}.json').read_text())
    gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY);base=cv2.cvtColor(raw,cv2.COLOR_RGB2GRAY)
    row={'image_id':id,'style':st,'shape_equal':im.shape==raw.shape,'zero_channel_ratio':float((im==0).mean()),'max_channel_ratio':float((im==255).mean()),'gray_gradient_correlation':float(np.corrcoef(cv2.Sobel(gray,cv2.CV_32F,1,0).ravel(),cv2.Sobel(base,cv2.CV_32F,1,0).ravel())[0,1]),**pars};rows.append(row)
  self.stain=stats;table(OUT/'stain_quality_checks.csv',rows);dump(OUT/'post_transform_statistics.json',stats)
  assert all(r['shape_equal'] and r.get('new_clipped_channel_fraction',0)<.01 for r in rows),'Stain quality gate failed; no automatic tuning'
 def run_queries(self,queries,seeds):
  for q in queries:
   id=q['id'];log('Stain features and inference '+id)
   for qs in ['A','B']:
    t=self.features(id,qs)
    for rs in ['A','B']:
     for s in seeds:self.evaluate(t,id,self.supports[s],rs,qs,s,'External',False)
     for exclusion in [True,False]:self.evaluate(t,id,id,rs,qs,2026,'Self',exclusion)
   # Self banks kept only for current query to bound host RAM.
   for st in ['A','B']:self.banks.pop((id,st),None)
 def export(self):
  rows=[json.loads(p.read_text()) for p in sorted((OUT/'cases').glob('*.json'))];main=[r for r in rows if not r['fp32_regression']];external=[r for r in main if r['source']=='External'];selfrows=[r for r in main if r['source']=='Self']
  table(OUT/'external_results_per_pair.csv',external);table(OUT/'self_oracle_results_per_image.csv',selfrows)
  table(OUT/'memory_diagnostics.csv',[{k:r[k] for k in ['query_id','reference_id','seed','condition','mode','status','fg_memory','bg_memory','fg_min_eligible','bg_min_eligible','invalid_token_ratio','actual_k']} for r in main]);table(OUT/'invalid_cases.csv',[r for r in main if r['status']!='VALID'] or [],['query_id','reference_id','condition','mode','status','invalid_token_ratio'])
  ret=[];gaps=[]
  for r in main:
   ret.append({k:v for k,v in r.items() if k.startswith('class') or k in ['query_id','reference_id','seed','condition','mode','source','auroc','token_auroc','status']})
   if self.stain:
    gaps.append({k:r[k] for k in ['query_id','reference_id','seed','condition','mode','source']}|gap(self.stain[r['reference_id']][r['reference_style']],self.stain[r['query_id']][r['query_style']]))
  table(OUT/'retrieval_diagnostics.csv',ret);table(OUT/'stain_gap_metrics.csv',gaps)
  table(OUT/'support_query_pairs.csv',[{'seed':s,'reference_id':rid,'query_id':q['id']} for s,rid in self.supports.items() for q in self.queries])
  return main

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['run','report'],default='run');a=ap.parse_args();OUT.mkdir(exist_ok=True,parents=True)
 if a.phase=='report':
  from report_reference_source_stain import report
  report();return
 dump(OUT/'configs/preregistered_protocol.json',PROTO);checks();exp=Experiment()
 meta={'status':'RUNNING','protocol':PROTO,'code_sha256':sha256(__file__),'python':sys.version,'torch':torch.__version__,'supports':exp.supports,'train_count':153,'query_count':40,'started':time.strftime('%Y-%m-%dT%H:%M:%S'),'command':sys.argv,'gpu':torch.cuda.get_device_name(0)};dump(OUT/'run_metadata.json',meta)
 try:
  exp.templates=prepare_templates(exp.pool);exp.setup_encoder();meta['encoder']=exp.encoder.metadata;dump(OUT/'run_metadata.json',meta)
  exp.raw_regression();log('Raw parity passed')
  ids=list(dict.fromkeys(list(exp.supports.values())+[q['id'] for q in exp.queries]));exp.quality_checks(ids)
  exp.run_queries(exp.queries[:3],SEEDS[:2]);smoke=exp.export();expected=24+24
  stain_smoke=[r for r in smoke if r['condition']!='RAW'];assert len(stain_smoke)==expected
  dump(OUT/'smoke_checks.json',{'status':'PASS','conditions':expected,'valid':sum(r['status']=='VALID' for r in stain_smoke),'no_dice_based_adjustment':True})
  log('Smoke complete. Fixed full protocol unchanged.');exp.run_queries(exp.queries,SEEDS);rows=exp.export()
  assert len([r for r in rows if r['source']=='External'])==1000
  assert len([r for r in rows if r['source']=='Self'])==320
  meta['status']='COMPLETE';meta['finished']=time.strftime('%Y-%m-%dT%H:%M:%S');dump(OUT/'run_metadata.json',meta)
  from report_reference_source_stain import report
  report()
 except Exception:
  meta['status']='BLOCKED';meta['error']=traceback.format_exc();dump(OUT/'run_metadata.json',meta);log(meta['error']);exp.export();raise
if __name__=='__main__':main()
