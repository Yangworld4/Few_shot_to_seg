"""E0-E3: historical N3-bil, feature-first interpolation and fixed guided margins."""
from pathlib import Path
import sys,json,csv,time,hashlib,traceback,argparse
from datetime import datetime,timezone
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pathology_trainfree.data import load_image,load_mask,read_manifest,resize_mask
from pathology_trainfree.geometry import windows,reconstruct,resize_map
from pathology_trainfree.memory import ReferenceMemory
from pathology_trainfree.retrieval import SimilarityRetriever
from pathology_trainfree.pipeline import SegmentationPipeline
from pathology_trainfree.encoder import sha256
from pathology_trainfree.config import load_config
OUT=ROOT/'outputs/dense_feature_matching_seed1337';DUAL=ROOT.parent/'DUAL'
DC={'mask_encoding':'indexed','class_mapping':{0:0,255:1}}
WEIGHTS=Path('/home/yangorange/.cache/huggingface/hub/models--MahmoodLab--UNI/blobs/56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40')
PROTO={'version':1,'ssl_seed':1337,'datasets':{'GlaS':{'support':7,'test':80},'CRAG':{'support':16,'test':40}},'window':448,'stride':224,'offsets':[0,8],'grid':28,'dim':1024,'k':50,'chunk':784,'feature_interpolation':'FP32 bilinear align_corners=False, spatial only, L2 normalize after interpolation','retrieval':'historical FP16 stored tokens -> FP32 normalize/cosine/top50/mean','margin':'FP16 round before bilinear448 -> mean windows -> mean offsets -> bilinear256 -> >0','guided':{'radius':4,'epsilon':.001,'guide':'RGB resized256 INTER_LINEAR -> cv2 grayscale float32/255','border':'REFLECT_101'},'boundary_tolerance':2,'distance_strata':[2,5],'object_min_area':50,'size_strata':{'small':[50,256],'medium':[256,1024],'large':[1024,None]},'object_match':'historical Hungarian maximum GT coverage; match intersection/GT_area>0.5','merge_split':'binary 8-connected components area>=50; link intersection>=10px and intersection/min(area_pred,area_gt)>=0.1; each node degree>1 counts one merge/split','boundary_iou':'inner foreground band distance to background<=2 including image exterior','bootstrap':{'seed':2026,'resamples':10000,'unit':'paired test image within each dataset'},'timing':'cache-read-inclusive inference from cached F28; excludes historical UNI encoding, GT evaluation and disk result writing; CUDA synchronize around stages','smoke':'first GLaS test image in manifest order, fixed B1 and B2 sanity only; no tuning'}
FP=hashlib.sha256(json.dumps(PROTO,sort_keys=True).encode()).hexdigest()

def dump(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,default=str));tmp.replace(p)
def table(p,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r)) or ['status','reason'];p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def log(s):
 print(s,flush=True)
 with (OUT/'execution.log').open('a') as f:f.write(datetime.now(timezone.utc).isoformat()+' '+s+'\n')
def auc(g,x):
 g=np.asarray(g).ravel().astype(bool);x=np.asarray(x).ravel();n=g.sum();b=len(g)-n
 return float((rankdata(x)[g].sum()-n*(n+1)/2)/(n*b)) if n and b else None

def guided_filter(guide,p,r=4,eps=.001):
 """Standard scalar locally linear guided filter, all operations on continuous margins."""
 guide=np.asarray(guide,np.float32);p=np.asarray(p,np.float32)
 if guide.shape!=p.shape or guide.ndim!=2:raise ValueError('Guide and continuous margin must be 2D with identical geometry')
 def mean(x):return cv2.boxFilter(x,-1,(2*r+1,2*r+1),normalize=True,borderType=cv2.BORDER_REFLECT_101)
 mi,mp=mean(guide),mean(p);cov=mean(guide*p)-mi*mp;var=np.maximum(0,mean(guide*guide)-mi*mi)
 a=cov/(var+eps);b=mp-a*mi
 return mean(a)*guide+mean(b)

@torch.inference_mode()
def dense_features(tokens,factor):
 t=torch.as_tensor(tokens,dtype=torch.float32,device='cuda').reshape(28,28,1024).permute(2,0,1)[None]
 t=F.interpolate(t,size=(28*factor,28*factor),mode='bilinear',align_corners=False)
 return F.normalize(t,dim=1)[0].permute(1,2,0).reshape(-1,1024).contiguous()

def edge(mask):return mask & ~ndimage.binary_erosion(mask,structure=np.ones((3,3)),border_value=0)
def regions(gt):
 b=edge(gt);dist=ndimage.distance_transform_edt(~b) if b.any() else np.full(gt.shape,np.inf)
 return {'all':np.ones(gt.shape,bool),'boundary_0_2':dist<=2,'boundary_2_5':(dist>2)&(dist<=5),'interior_gt_5':dist>5}
def components(mask):
 lab,n=ndimage.label(mask,structure=np.ones((3,3)));areas=np.bincount(lab.ravel());keep=np.flatnonzero(areas>=50);keep=keep[keep>0];lut=np.zeros(n+1,np.int32);lut[keep]=np.arange(1,len(keep)+1)
 return lut[lab],areas[keep]
def object_stats(pred,gt):
 pl,pa=components(pred);gl,ga=components(gt);npred,ngt=len(pa),len(ga)
 overlap=np.bincount((pl.ravel()*(ngt+1)+gl.ravel()).astype(np.int64),minlength=(npred+1)*(ngt+1)).reshape(npred+1,ngt+1)[1:,1:]
 matched=np.zeros(ngt,bool)
 if npred and ngt:
  ri,ci=linear_sum_assignment(-overlap/ga[None,:]);ok=overlap[ri,ci]/ga[ci]>.5;matched[ci[ok]]=True
 tp=int(matched.sum());op=tp/npred if npred else float(not ngt);orec=tp/ngt if ngt else float(not npred)
 out={'object_f1':2*tp/(npred+ngt) if npred+ngt else 1.,'object_precision':op,'object_recall':orec,'object_tp':tp,'pred_objects':npred,'gt_objects':ngt}
 links=(overlap>=10)&(overlap/np.maximum(1,np.minimum(pa[:,None],ga[None,:]))>=.1) if npred and ngt else np.zeros((npred,ngt),bool)
 out.update(merge_count=int((links.sum(1)>1).sum()),split_count=int((links.sum(0)>1).sum()))
 for name,lo,hi in [('small',50,256),('medium',256,1024),('large',1024,np.inf)]:
  sel=(ga>=lo)&(ga<hi);out[name+'_glands']=int(sel.sum());out[name+'_gland_recall']=float(matched[sel].mean()) if sel.any() else None
 return out

def evaluate(margin,gt):
 pred=margin>0;g=gt.astype(bool);tp=int((pred&g).sum());fp=int((pred&~g).sum());fn=int((~pred&g).sum());den=2*tp+fp+fn
 out={'dice':2*tp/max(1,den),'fg_iou':tp/max(1,tp+fp+fn),'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'tp':tp,'fp':fp,'fn':fn,'predicted_fg_ratio':float(pred.mean()),'auroc':auc(gt,margin)}
 out.update(object_stats(pred,g));pb,gb=edge(pred),edge(g);dg=ndimage.distance_transform_edt(~gb) if gb.any() else np.full(g.shape,np.inf);dp=ndimage.distance_transform_edt(~pb) if pb.any() else np.full(g.shape,np.inf)
 bp=int((pb&(dg<=2)).sum());br=int((gb&(dp<=2)).sum());pr=bp/max(1,pb.sum());re=br/max(1,gb.sum())
 out.update(boundary_f1=float(2*pr*re/max(1e-30,pr+re)) if pb.any() or gb.any() else 1.,boundary_precision=float(pr),boundary_recall=float(re),boundary_fp=int(pb.sum()-bp),boundary_fn=int(gb.sum()-br))
 def inner(x):return x & (ndimage.distance_transform_edt(np.pad(x,1))[1:-1,1:-1]<=2)
 pi,gi=inner(pred),inner(g);out['boundary_iou']=float((pi&gi).sum()/max(1,(pi|gi).sum()))
 for c,sel in [('fg',g),('bg',~g)]:
  vals=margin[sel];out[c+'_margin_mean']=float(vals.mean()) if len(vals) else None
  for q in [5,50,95]:out[c+f'_margin_q{q}']=float(np.percentile(vals,q)) if len(vals) else None
 for name,sel in regions(g).items():
  out[name+'_pixels']=int(sel.sum());out[name+'_errors']=int(((pred!=g)&sel).sum());out[name+'_fg_to_bg']=int((g&~pred&sel).sum());out[name+'_bg_to_fg']=int((~g&pred&sel).sum())
 return out

class Study:
 def __init__(self):
  self.cfg=load_config(ROOT/'configs/uni_memory.yaml');self.cfg['retrieval'].update(top_k=50,chunk_size=784);self.cfg['memory']['class_ids']=[0,1];self.data={};self.audit=[]
  wh=sha256(WEIGHTS);assert wh==WEIGHTS.name
  self.encoder=SimpleNamespace(weights_sha256=wh,precision='float32',token_precision='float16',patch_size=16,feature_dim=1024,normalize=True,name='uni',device=torch.device('cuda'))
  self.cfg['window'].update(size=448,stride=224,offsets=[0,8],support_offsets=[0,8],padding_mode='reflect')
  self.cfg['segmentation'].update(margin_precision='float16',output_size=[256,256],native_resolution=False,threshold=0.)
  for ds in ['GlaS','CRAG']:
   md=ROOT/f'datasets/{ds}/manifests/paper_seed_2026';support=read_manifest(md/'labeled_10.csv',require_mask=True);queries=read_manifest(md/'test.csv',require_mask=True)
   assert len(support)==PROTO['datasets'][ds]['support'] and len(queries)==PROTO['datasets'][ds]['test']
   source=DUAL/f'outputs/n3_bil_distillation_glas_crag_10/{ds}/memory_N3.npz'
   with np.load(source,allow_pickle=False) as z:
    ids=z['source_ids'].tolist();b={0:z['bg'].copy(),1:z['fg'].copy()}
   assert set(ids)=={e['id'] for e in support} and not set(ids)&{e['id'] for e in queries}
   assert all(v.shape==(len(support)*512,1024) and v.dtype==np.float16 for v in b.values())
   memory=ReferenceMemory(b,{'encoder_sha256':wh,'source':str(source),'support_ids':ids})
   cfg=json.loads(json.dumps(self.cfg));cfg['dataset']=dict(DC,name=ds);cfg['output']['directory']=str(ROOT/f'outputs/support_size_study/cache/query_features/{ds}');cfg['output']['cache_features']=True
   retriever=SimilarityRetriever(memory,cfg,'cuda');historical={r['image_id']:r for r in csv.DictReader((source.parent/'direct_test_per_image.csv').open())}
   self.data[ds]={'support':support,'queries':queries,'memory':memory,'retriever':retriever,'cfg':cfg,'old':historical,'source':source}
   dump(OUT/f'configs/{ds}.json',{'protocol':PROTO,'support_ids_from_manifest':[e['id'] for e in support],'memory_source_order':ids,'memory_sha256':sha256(source),'fg_vectors_sha256':hashlib.sha256(b[1].tobytes()).hexdigest(),'bg_vectors_sha256':hashlib.sha256(b[0].tobytes()).hexdigest(),'feature_checkpoint_sha256':wh,'queries':[e['id'] for e in queries]})
 def cached(self,ds,e):
  image=load_image(e['image']);cache=ROOT/f'outputs/support_size_study/cache/query_features/{ds}/feature_cache';passes=[]
  for o in [0,8]:
   p=cache/f'{e["id"]}_offset{o}.npz'
   key=hashlib.sha256(image.tobytes()+str((self.encoder.weights_sha256,448,224,o,'float32','float16')).encode()).hexdigest()
   with np.load(p,allow_pickle=False) as z:t=z['tokens'].copy();pos=z['positions'].copy();assert str(z['cache_key'])==key
   want=[(y,x) for y,x,_ in windows(image,448,224,o)];assert np.array_equal(pos,want) and t.shape==(len(want),784,1024) and t.dtype==np.float16 and np.isfinite(t).all()
   assert np.allclose(np.linalg.norm(t.astype(np.float32),axis=-1),1,atol=.001)
   passes.append((t,pos));self.audit.append({'dataset':ds,'image_id':e['id'],'offset':o,'cache_file':str(p),'fingerprint':key,'positions_valid':True,'dtype':str(t.dtype),'windows':len(pos)})
  return image,passes
 @torch.inference_mode()
 def predict(self,ds,passes,shape,factor):
  retriever=self.data[ds]['retriever'];timing={'feature_interpolation_seconds':0.,'knn_seconds':0.,'reconstruction_seconds':0.};offsetmaps=[]
  for tokens,pos in passes:
   grids=[]
   if factor==1:
    torch.cuda.synchronize();st=time.perf_counter();scores=retriever.scores(tokens).numpy();torch.cuda.synchronize();timing['knn_seconds']+=time.perf_counter()-st;grids=(scores[:,1]-scores[:,0]).reshape(-1,28,28)
   else:
    for tok in tokens:
     torch.cuda.synchronize();st=time.perf_counter();dense=dense_features(tok,factor);torch.cuda.synchronize();timing['feature_interpolation_seconds']+=time.perf_counter()-st
     st=time.perf_counter();scores=retriever.scores(dense).numpy();torch.cuda.synchronize();timing['knn_seconds']+=time.perf_counter()-st;grids.append((scores[:,1]-scores[:,0]).reshape(28*factor,28*factor));del dense
   st=time.perf_counter();offsetmaps.append(reconstruct(grids,pos,shape,448,'float16'));timing['reconstruction_seconds']+=time.perf_counter()-st
  st=time.perf_counter();margin=resize_map(np.mean(offsetmaps,axis=0,dtype=np.float32),(256,256));timing['reconstruction_seconds']+=time.perf_counter()-st
  return margin,timing
 def path(self,ds,id,method):return OUT/'cases'/ds/f'{id}_{method}.json'
 def run(self,ds,e,method):
  path=self.path(ds,e['id'],method)
  if path.exists():
   r=json.loads(path.read_text());assert r['protocol_fingerprint']==FP;return r
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();st=time.perf_counter()
  if method=='B3':
   base=np.load(self.path(ds,e['id'],'B0').with_suffix('.npz'))['margin'];image=load_image(e['image']);guide=cv2.cvtColor(cv2.resize(image,(256,256),interpolation=cv2.INTER_LINEAR).astype(np.float32)/255,cv2.COLOR_RGB2GRAY)
   gf=time.perf_counter();margin=guided_filter(guide,base);filtertime=time.perf_counter()-gf;increment=time.perf_counter()-st
   b0=json.loads(self.path(ds,e['id'],'B0').read_text());timing={k:b0[k] for k in ['feature_interpolation_seconds','knn_seconds','reconstruction_seconds','cache_read_seconds']};timing.update(guided_filter_seconds=filtertime,incremental_refinement_seconds=increment,total_query_seconds=b0['total_query_seconds']+increment,peak_gpu_mib=b0['peak_gpu_mib'])
  elif method=='ORACLE':
   mask=load_mask(e['mask'],DC);offsetmaps=[];rt=time.perf_counter()
   for o in [0,8]:
    views=list(windows(mask,448,224,o));grids=np.array([2*crop.reshape(28,16,28,16).mean((1,3))-1 for _,_,crop in views],np.float32)
    offsetmaps.append(reconstruct(grids,[(y,x) for y,x,_ in views],mask.shape,448,'float16'))
   margin=resize_map(np.mean(offsetmaps,axis=0,dtype=np.float32),(256,256));timing={'total_query_seconds':time.perf_counter()-st,'reconstruction_seconds':time.perf_counter()-rt,'peak_gpu_mib':0.}
  else:
   image,passes=self.cached(ds,e);read=time.perf_counter()-st;margin,timing=self.predict(ds,passes,image.shape[:2],{'B0':1,'B1':2,'B2':4}[method]);torch.cuda.synchronize();timing.update(cache_read_seconds=read,total_query_seconds=time.perf_counter()-st,peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2)
  # Only after all inference has finished may the evaluation branch read GT.
  gt=resize_mask(load_mask(e['mask'],DC),(256,256));assert margin.shape==gt.shape and np.isfinite(margin).all()
  row={'dataset':ds,'image_id':e['id'],'method':method,'protocol_fingerprint':FP,**evaluate(margin,gt),**timing}
  path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
  with tmp.open('wb') as f:np.savez_compressed(f,margin=margin.astype(np.float32),mask=(margin>0).astype(np.uint8))
  tmp.replace(path.with_suffix('.npz'));dump(path,row);return row
 def baseline(self):
  rows=[]
  for ds,state in self.data.items():
   for i,e in enumerate(state['queries']):
    r=self.run(ds,e,'B0');old=state['old'][e['id']];saved=np.load(state['source'].parent/'direct_predictions'/f'{e["id"]}.npy');pred=np.load(self.path(ds,e['id'],'B0').with_suffix('.npz'))['mask']
    rows.append({'dataset':ds,'image_id':e['id'],'dice':r['dice'],'historical_dice':float(old['dice']),'dice_delta':r['dice']-float(old['dice']),'mask_disagreement_pixels':int((pred!=saved).sum()),'validated_replay_disagreement_pixels':int((pred!=np.load(ROOT/'outputs/n3_reference_parity_debug/reproduced_predictions'/ds/f'{e["id"]}.npy')).sum())})
    if i%10==0 or i==len(state['queries'])-1:log(f'B0 {ds} {i+1}/{len(state["queries"])}')
  table(OUT/'baseline_parity_per_image.csv',rows);table(OUT/'feature_cache_audit.csv',self.audit)
  lines=['# Baseline parity','Historical FP16 normalized tokens; FP32 cosine/top50/mean; FP16 margin before original bilinear reconstruction. No threshold tuning.']
  for ds in self.data:
   rr=[r for r in rows if r['dataset']==ds];lines.append(f'{ds}: current Dice={np.mean([r["dice"] for r in rr]):.12f}; historical={np.mean([r["historical_dice"] for r in rr]):.12f}; max per-image error={max(abs(r["dice_delta"]) for r in rr):.12g}; total mask disagreement={sum(r["mask_disagreement_pixels"] for r in rr)}')
  (OUT/'baseline_parity.md').write_text('\n\n'.join(lines));assert all(r['validated_replay_disagreement_pixels']==0 for r in rows),'Baseline differs from independently verified prior pipeline replay';assert all(r['mask_disagreement_pixels']==0 or (r['dataset']=='GlaS' and r['image_id']=='test_a_testA_23' and r['mask_disagreement_pixels']==1 and abs(r['dice_delta'])<1.5e-5) for r in rows),'Unexpected historical mismatch';
  with (OUT/'baseline_parity.md').open('a') as f:f.write('\n\nAll 120 masks exactly match the previously independently verified PathologyTrainFree replay. Historical Direct GLaS test_a_testA_23 differs at only (113,75), current margin 4.9628625e-6: the same already documented near-zero floating-point difference. No threshold, memory or reconstruction adjustment. CRAG historical masks are exact.\n')
 def export(self):
  rows=[json.loads(p.read_text()) for p in sorted((OUT/'cases').glob('*/*.json'))];table(OUT/'per_image_metrics.csv',[r for r in rows if r['method']!='ORACLE']);table(OUT/'oracle_reconstruction_metrics.csv',[r for r in rows if r['method']=='ORACLE']);table(OUT/'e0_error_diagnostics.csv',[r for r in rows if r['method']=='B0']);return rows

def unit_checks():
 torch.manual_seed(2026);rows=[]
 def check(n,v):rows.append({'check':n,'passed':bool(v)});assert v,n
 x=torch.randn(784,1024);d=dense_features(x,2);check('feature56 shape',d.shape==(3136,1024));check('dense L2',torch.allclose(torch.linalg.norm(d,dim=1),torch.ones(3136,device='cuda'),atol=2e-6));check('feature112 shape',dense_features(x,4).shape==(12544,1024))
 # Channels remain separate: constant orthogonal channel signals cannot move into zero channels.
 x=torch.zeros(784,1024);x[:,0]=1;x[:,7]=2;d=dense_features(x,2);check('spatial-only no channel mixing',torch.all(d[:,1:7]==0) and torch.allclose(d[:,7]/d[:,0],torch.full((3136,),2.,device='cuda')))
 mem=ReferenceMemory({c:np.random.default_rng(2026+c).normal(size=(75,1024)).astype(np.float16) for c in [0,1]},{});cfg={'memory':{'class_ids':[0,1]},'retrieval':{'top_k':50,'chunk_size':11}};retr=SimilarityRetriever(mem,cfg,'cuda');q=np.random.default_rng(1).normal(size=(37,1024)).astype(np.float32);got=retr.scores(q).numpy();qt=F.normalize(torch.tensor(q,device='cuda'),dim=1);direct=torch.stack([(qt@retr.banks[c].T).topk(50,dim=1).values.mean(1) for c in [0,1]],1).cpu().numpy();check('chunked vs direct cosine top50',np.allclose(got,direct,atol=2e-7));check('fixed K50',retr.k==50)
 guide=np.random.default_rng(2).random((17,19)).astype(np.float32);p=np.full_like(guide,.17);check('guided constant margin preserved',np.allclose(guided_filter(guide,p),p,atol=2e-6))
 # Independent box mean using explicit reflected padding and patch means.
 def slowmean(a):
  a=np.pad(a,4,mode='reflect');return np.array([[a[y:y+9,x:x+9].mean() for x in range(19)] for y in range(17)])
 p=np.random.default_rng(3).normal(size=(17,19)).astype(np.float32);mi,mp=slowmean(guide),slowmean(p);a=(slowmean(guide*p)-mi*mp)/(np.maximum(0,slowmean(guide*guide)-mi*mi)+.001);b=mp-a*mi;ref=slowmean(a)*guide+slowmean(b)
 check('guided filter independent local linear reference',np.allclose(guided_filter(guide,p),ref,atol=3e-6));check('continuous margin input retained',np.unique(guided_filter(guide,p)).size>2)
 synthetic=np.arange(512*512,dtype=np.int32).reshape(512,512);v=list(windows(synthetic,448,224,8));check('offset8 starts and reflection',v[0][0:2]==(-8,-8) and v[0][2][0,0]==synthetic[8,8]);check('fusion reconstruct constant',np.allclose(reconstruct(np.ones((len(v),28,28)),[(y,x) for y,x,_ in v],(512,512),448),1))
 table(OUT/'implementation_checks.csv',rows)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--report-only',action='store_true');a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 if a.report_only:
  from report_dense_feature_matching import report
  report();return
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 dump(OUT/'configs/preregistered_protocol.json',PROTO);meta={'status':'RUNNING','protocol':PROTO,'protocol_fingerprint':FP,'started_utc':datetime.now(timezone.utc).isoformat(),'command':sys.argv,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'code_sha256':sha256(__file__)};dump(OUT/'run_metadata.json',meta)
 study=None
 try:
  unit_checks();study=Study();study.baseline();log('B0 parity passed on both complete test sets')
  for ds,state in study.data.items():
   for i,e in enumerate(state['queries']):study.run(ds,e,'ORACLE')
   log('E0 Oracle '+ds+' complete')
  first=study.data['GlaS']['queries'][0];study.run('GlaS',first,'B1');study.run('GlaS',first,'B2');dump(OUT/'smoke_checks.json',{'status':'PASS','query':first['id'],'methods':['B1','B2'],'parameters_unchanged':True});log('Dense smoke passed; no parameter changes')
  for method in ['B1','B2','B3']:
   for ds,state in study.data.items():
    for i,e in enumerate(state['queries']):
     r=study.run(ds,e,method);log(f'{method} {ds} {i+1}/{len(state["queries"])} {e["id"]} time={r["total_query_seconds"]:.3f}s')
   study.export()
  rows=study.export();assert len(rows)==600
  from report_dense_feature_matching import report
  report();meta.update(status='COMPLETE',finished_utc=datetime.now(timezone.utc).isoformat(),cases=len(rows));dump(OUT/'run_metadata.json',meta);log('COMPLETE all E0-E3 results and report')
 except Exception:
  meta.update(status='BLOCKED',error=traceback.format_exc());dump(OUT/'run_metadata.json',meta);log(meta['error'])
  if study is not None:study.export()
  raise
if __name__=='__main__':main()
