"""Fixed N3-bil reference coverage diagnosis; cached features only, resumable per query."""
from pathlib import Path
import sys, json, csv, hashlib, time, argparse
from collections import defaultdict, Counter, deque
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pathology_trainfree.data import read_manifest, load_image, load_mask, resize_mask
from pathology_trainfree.geometry import windows, reconstruct, resize_map
from pathology_trainfree.evaluation import metrics, object_metrics
from pathology_trainfree.memory import select_tokens
from study_support_size import cache_file_for
OUT=ROOT/'outputs/reference_coverage_diagnosis'; DUAL=ROOT.parent/'DUAL'
SEEDS=[14,24,42,1337,3407]; NS=[1,5,10,20]
DC={'mask_encoding':'indexed','class_mapping':{0:0,255:1}}
MC={'class_ids':[0,1],'occupancy_thresholds':{0:.9,1:.9},'cap_per_image_per_class':512,'fallback_strategy':'nearest_10_percent'}
PROTO={'occupancy_source':'real training GT recomputed; ignore dummy unlabeled occupancy','weights_sha256':'56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40','window':448,'stride':224,'offsets':[0,8],'support_offsets':[0,8],'k':50,'seed':2026,'occupancy':[.1,.9],'cap':512,'margin_precision':'float16','threshold':0,'output':[256,256],'rng':'original_global','source_subset':'first, quartiles, last query IDs in manifest order'}
FP=hashlib.sha256(json.dumps(PROTO,sort_keys=True).encode()).hexdigest()
CODE=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(x,indent=2,default=str)); q.replace(p)
def table(p,rows):
 if not rows:return
 fields=list(dict.fromkeys(k for r in rows for k in r)); p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
def auc(gt,m):
 g=gt.ravel().astype(bool); x=m.ravel(); n=int(g.sum()); b=len(g)-n
 if not n or not b:return None
 return float((rankdata(x)[g].sum()-n*(n+1)/2)/(n*b))
def stats(x,prefix=''):
 if not len(x):return {prefix+'mean':None}
 return {prefix+'mean':float(np.mean(x)),prefix+'std':float(np.std(x)),**{prefix+f'q{q}':float(np.quantile(x,q/100)) for q in [1,5,25,50,75,95,99]}}
def setup(ds):
 d=ROOT/f'datasets/{ds}/manifests/paper_seed_2026'
 pool=read_manifest(d/'train_pool.csv',require_mask=True); queries=read_manifest(d/'test.csv',require_mask=True)
 with np.load(DUAL/f'outputs/n3_bil_distillation_glas_crag_10/{ds}/memory_N3.npz') as z:
  ids=z['source_ids'].tolist(); h={0:z['bg'].copy(),1:z['fg'].copy()}
 return pool,queries,ids,h

def integrity():
 rows=[]; near=[]; exact=[]
 for ds in ['GlaS','CRAG']:
  pool,queries,hids,h=setup(ds); pids={e['id'] for e in pool}; imgs={}; hashes=defaultdict(list)
  assert set(hids)<=pids
  for split,entries in [('train',pool),('test',queries)]:
   for e in entries:
    im=load_image(e['image']); mask=load_mask(e['mask'],DC); assert im.shape[:2]==mask.shape
    digest=hashlib.sha256(im.tobytes()).hexdigest(); hashes[digest].append((split,e['id']))
    thumb=np.asarray(Image.fromarray(im).convert('L').resize((9,8)),dtype=np.float32); dh=(thumb[:,1:]>thumb[:,:-1]).ravel()
    imgs[(split,e['id'])]=dh
    occerr=None
    if split=='train':
     with np.load(cache_file_for(DUAL,ds,e['id'])) as z:
      vals=[]
      for o in [0,8]:
       for y,x,crop in windows(mask,448,224,o): vals.append(crop.reshape(28,16,28,16).mean((1,3)).ravel())
      occerr=float(np.max(np.abs(np.array(vals)-z['occupancy'])))
      if e["id"] in hids: assert occerr<.001
    rgb=im.reshape(-1,3).mean(0)
    rows.append({'dataset':ds,'split':split,'image_id':e['id'],'historical':e['id'] in hids,'in_training_pool':e['id'] in pids,'image':str(e['image']),'mask':str(e['mask']),'sha256_rgb':digest,'height':im.shape[0],'width':im.shape[1],'gt_fg_ratio':float(mask.mean()),'mask_values':';'.join(map(str,np.unique(np.asarray(Image.open(e['mask']))))),'occupancy_max_abs_error':occerr,'cache_occupancy_valid':occerr is None or occerr<.001,'mean_r':rgb[0],'mean_g':rgb[1],'mean_b':rgb[2],'patient_id':'unavailable','slide_id':'unavailable','config_fingerprint':FP,'code_sha256':CODE})
   print('integrity',ds,split,len(entries),flush=True)
  for k,v in hashes.items():
   if len(v)>1:exact.append({'dataset':ds,'sha':k,'members':v})
  for e in pool:
   for q in queries:
    dist=int(np.count_nonzero(imgs[('train',e['id'])]!=imgs[('test',q['id'])]))
    near.append({'dataset':ds,'support_id':e['id'],'historical':e['id'] in hids,'query_id':q['id'],'dhash_distance':dist})
 table(OUT/'support_integrity.csv',rows); table(OUT/'near_duplicate_candidates.csv',sorted(near,key=lambda r:r['dhash_distance'])[:100])
 dump(OUT/'exact_duplicates.json',exact)
 cross=[e for e in exact if {x[0] for x in e['members']}=={'train','test'}]
 report='# Data integrity\n\n所有历史 Support 均在训练候选池；图像/Mask 尺寸及二值编码经逐图检查。训练候选池的缓存 occupancy 与反射填充后16×16 GT均值逐Token核对；不一致将隔离缓存标签，后续只复用特征并从真实训练GT重算occupancy。\n\n跨训练/测试 RGB 内容完全相同组数：'+str(len(cross))+'. 患者ID、切片ID、扫描元数据未提供，不能排除患者/切片级重叠。dHash仅作近重复候选，不认定患者或切片身份；RGB均值仅描述染色颜色，不能替代扫描或染色来源元数据。\n'
 (OUT/'support_integrity_report.md').write_text(report); (OUT/'protocol_and_integrity.md').write_text(report+'\n固定协议：\n```json\n'+json.dumps(PROTO,indent=2)+'\n```\n')
 if cross:raise RuntimeError('Exact cross-split duplicates; isolate before experiments')

def prepare(ds):
 pool,queries,hids,h=setup(ds); entries={e['id']:e for e in pool}; store={}
 # One load of each Support feature cache; original_global consumes FG then BG.
 for e in pool:
  with np.load(cache_file_for(DUAL,ds,e['id'])) as z:
   mask=load_mask(e['mask'],DC)
   true_occ=np.array([v.reshape(28,16,28,16).mean((1,3)).ravel() for o in [0,8] for _,_,v in windows(mask,448,224,o)],np.float32).ravel()
   store[e['id']]=(z['features'].reshape(-1,1024),true_occ)
 def build(ids):
  rng=np.random.default_rng(2026); b={0:[],1:[]}; src={0:[],1:[]}
  for id in ids:
   feat,occ=store[id]; selected,_=select_tokens(feat,np.stack([1-occ,occ],1),MC,rng)
   for c in [0,1]:b[c].append(selected[c]); src[c].extend([id]*len(selected[c]))
  return {c:np.concatenate(b[c]) for c in [0,1]}, {c:np.array(src[c]) for c in [0,1]}
 rebuilt,rs=build(hids); hs={}
 for c in [0,1]:
  lookup=defaultdict(deque)
  for v,id in zip(rebuilt[c],rs[c]):lookup[v.tobytes()].append(id)
  hs[c]=np.array([lookup[v.tobytes()].popleft() for v in h[c]])
 groups=[{'kind':'H','seed':0,'ids':hids,'bank':h,'src':hs}]; conditions=[{'condition':'H','kind':'H','seed':0,'additional':0,'ids':hids,'bank':h,'src':hs}]
 for seed in SEEDS:
  extra=[e['id'] for e in pool if e['id'] not in hids]; extra=list(np.random.default_rng(seed).permutation(extra))[:20]
  # Independent random-only stream, legal full pool, historical references permitted naturally.
  rand=list(np.random.default_rng(np.random.SeedSequence([seed,9187])).permutation([e['id'] for e in pool]))[:len(hids)+20]
  with (ROOT/'outputs/support_size_study_original_memory/support_sets.csv').open() as f:
   paired=[r['image_id'] for r in csv.DictReader(f) if r['dataset']==ds and int(r['seed'])==seed and int(r['shots'])==20]
  assert len(paired)==20
  for kind,ids in [('A',extra),('R',rand),('T',paired)]:
   b,s=build(ids); group={'kind':kind,'seed':seed,'ids':ids,'bank':b,'src':s}; groups.append(group)
   for n in ([20] if kind=='T' else NS):
    chosen=ids[:n if kind in ['A','T'] else len(hids)+n]; ends={c:int(np.isin(s[c],chosen).sum()) for c in [0,1]}
    bb={c:b[c][:ends[c]] for c in [0,1]}; ss={c:s[c][:ends[c]] for c in [0,1]}
    if kind=='A':bb={c:np.concatenate([h[c],bb[c]]) for c in [0,1]}; ss={c:np.concatenate([hs[c],ss[c]]) for c in [0,1]}
    conditions.append({'condition':f'{kind}_{seed}_{n}','kind':kind,'seed':seed,'additional':n,'ids':hids+chosen if kind=='A' else chosen,'bank':bb,'src':ss,'ends':ends})
 del store
 for co in conditions:
  co['meta']={'dataset':ds,'condition':co['condition'],'kind':co['kind'],'seed':co['seed'],'additional':co['additional'],'support_ids':';'.join(co['ids']),'support_count':len(co['ids']),'fg_size':len(co['bank'][1]),'bg_size':len(co['bank'][0]),'query_set':';'.join(e['id'] for e in queries),'config_fingerprint':FP,'code_sha256':CODE}
 table(OUT/f'{ds}_support_sets.csv',[c['meta'] for c in conditions])
 for co in conditions:
  path=OUT/'memories'/f'{ds}_{co["condition"]}.npz'; path.parent.mkdir(exist_ok=True)
  valid=False
  if path.exists():
   try:
    with np.load(path) as saved:
     valid=all(np.array_equal(saved['bg' if c==0 else 'fg'],co['bank'][c]) and np.array_equal(saved['bg_source' if c==0 else 'fg_source'],co['src'][c]) for c in [0,1])
   except Exception:valid=False
  if not valid:
   temp=path.with_suffix('.npz.tmp')
   with temp.open('wb') as f:np.savez_compressed(f,fg=co['bank'][1],bg=co['bank'][0],fg_source=co['src'][1],bg_source=co['src'][0],metadata=json.dumps(co['meta']))
   temp.replace(path)
 return queries,hids,groups,conditions

@torch.inference_mode()
def run_dataset(ds):
 queries,hids,groups,conditions=prepare(ds); subset={queries[int(i)]['id'] for i in np.linspace(0,len(queries)-1,5,dtype=int)}
 dump(OUT/f'{ds}_source_subset.json',sorted(subset)); condmap={c['condition']:i for i,c in enumerate(conditions)}
 for g in groups:g['gpu']={c:F.normalize(torch.tensor(g['bank'][c].astype(np.float32),device='cuda'),dim=1) for c in [0,1]}
 for qi,e in enumerate(queries):
  folder=OUT/'queries'/ds/e['id']; marker=folder/'complete.json'
  if marker.exists():
   assert json.loads(marker.read_text())['config_fingerprint']==FP
   continue
  folder.mkdir(parents=True,exist_ok=True); image=load_image(e['image']); gtfull=load_mask(e['mask'],DC); gt=resize_mask(gtfull,(256,256)); cached=[]; occ=[]
  for o in [0,8]:
   with np.load(ROOT/f'outputs/support_size_study/cache/query_features/{ds}/feature_cache/{e["id"]}_offset{o}.npz') as z:tok=z['tokens']; pos=z['positions']
   views=list(windows(gtfull,448,224,o)); assert np.array_equal(pos,[(y,x) for y,x,_ in views])
   occ.append(np.array([v.reshape(28,16,28,16).mean((1,3)).ravel() for _,_,v in views]).ravel()); cached.append((tok,pos))
  tokens=np.concatenate([x[0].reshape(-1,1024) for x in cached]); occupancy=np.concatenate(occ); n=len(tokens); purefg=occupancy>=.9; purebg=occupancy<=.1; boundary=~(purefg|purebg)
  scores=np.empty((len(conditions),2,n),np.float32); nearest=np.empty_like(scores); attrs=[]; contributions=[]
  # Only representative queries save token scores and neighbour source attribution; inference covers all.
  for start in range(0,n,784):
   end=min(start+784,n); q=F.normalize(torch.tensor(tokens[start:end].astype(np.float32),device='cuda'),dim=1); hv={}; hi={}
   for c in [0,1]:
    v,idx=(q@groups[0]['gpu'][c].T).topk(50,dim=1); hv[c]=v; hi[c]=idx; scores[0,c,start:end]=v.mean(1).cpu(); nearest[0,c,start:end]=v[:,0].cpu()
   selected={0:(hv,hi)}
   for g in groups[1:]:
    cos={c:q@g['gpu'][c].T for c in [0,1]}
    for extra in ([20] if g['kind']=='T' else NS):
     ix=condmap[f'{g["kind"]}_{g["seed"]}_{extra}']; co=conditions[ix]; vv={}; ii={}
     for c in [0,1]:
      v,idx=cos[c][:,:co['ends'][c]].topk(50,dim=1)
      if g['kind']=='A':
       comb=torch.cat([hv[c],v],1); inds=torch.cat([hi[c],idx+len(groups[0]['bank'][c])],1); v,sel=comb.topk(50,dim=1); idx=inds.gather(1,sel)
      vv[c]=v; ii[c]=idx; scores[ix,c,start:end]=v.mean(1).cpu(); nearest[ix,c,start:end]=v[:,0].cpu()
     if e['id'] in subset:selected[ix]=(vv,ii)
   if e['id'] in subset:
    for ix,(vv,ii) in selected.items():
     co=conditions[ix]; mar=scores[ix,1,start:end]-scores[ix,0,start:end]; before=scores[0,1,start:end]-scores[0,0,start:end]
     strata={'all':np.ones(end-start,bool),'pure_fg':purefg[start:end],'pure_bg':purebg[start:end],'boundary':boundary[start:end], 'fg_harmed':purefg[start:end]&(before>0)&(mar<0),'fg_corrected':purefg[start:end]&(before<=0)&(mar>0)}
     for c in [0,1]:
      source=co['src'][c][ii[c].cpu().numpy()]
      for stratum,sel in strata.items():
       if not sel.any():continue
       counts=Counter(source[sel].ravel()); den=int(sel.sum())*50
       for id,count in counts.items():attrs.append({'condition':co['condition'],'class':c,'stratum':stratum,'source_image_id':id,'historical_source':id in hids,'neighbor_count':count,'denominator':den})
      # Unique contribution = sole source in Top50 for a query token; plus top1 frequency.
      top=Counter(source[:,0]); sole=Counter(source[np.all(source==source[:,:1],axis=1),0])
      for id in np.unique(source):contributions.append({'condition':co['condition'],'class':c,'source_image_id':id,'top1_count':top[id],'sole_source_top50_tokens':sole[id],'query_tokens':end-start})
  margins=[]; rows=[]; inter=[]; coverage=[]
  for ix,co in enumerate(conditions):
   tm=scores[ix,1]-scores[ix,0]; bm=scores[0,1]-scores[0,0]; offsetmaps=[]; cursor=0
   for tok,pos in cached:
    size=tok.shape[0]*784; offsetmaps.append(reconstruct(tm[cursor:cursor+size].reshape(-1,28,28),pos,image.shape[:2],448,'float16')); cursor+=size
   margin=resize_map(np.mean(offsetmaps,axis=0,dtype=np.float32),(256,256)); margins.append(margin); pred=margin>0
   if ix==0:basepred=pred.copy()
   m=metrics(pred,gt,[0,1])[1]; of1,_=object_metrics(pred,gt.astype(bool),50,.5)
   fgvals=margin[gt==1]; bgvals=margin[gt==0]
   row={'condition':co['condition'],'image_id':e['id'],'dice':m['dice'],'fg_iou':m['iou'],'object_f1':of1,'precision':m['precision'],'recall':m['recall'],'predicted_fg_ratio':float(pred.mean()),'gt_fg_ratio':float(gt.mean()),'auroc':auc(gt,margin),'fg_margin_mean':float(fgvals.mean()) if len(fgvals) else None,'fg_margin_median':float(np.median(fgvals)) if len(fgvals) else None,'bg_margin_mean':float(bgvals.mean()) if len(bgvals) else None,'bg_margin_median':float(np.median(bgvals)) if len(bgvals) else None,'fg_nonpositive_ratio':float(np.mean(fgvals<=0)) if len(fgvals) else None,'bg_positive_ratio':float(np.mean(bgvals>0)) if len(bgvals) else None,'corrected_pixels':int(((basepred!=gt)&(pred==gt)).sum()),'harmed_pixels':int(((basepred==gt)&(pred!=gt)).sum()),'pixel_fg_to_bg_error':int(((gt==1)&basepred&~pred).sum()),'pixel_bg_to_fg_error':int(((gt==0)&~basepred&pred).sum()),**stats(margin.ravel(),'margin_')}; rows.append(row)
   for label,sel in [('pure_fg',purefg),('pure_bg',purebg),('boundary',boundary)]:
    if not sel.any():continue
    deltafg=scores[ix,1,sel]-scores[0,1,sel]; deltabg=scores[ix,0,sel]-scores[0,0,sel]; valid=label!='boundary'; actual=label=='pure_fg'; bp=bm[sel]>0; ap=tm[sel]>0
    inter.append({'condition':co['condition'],'image_id':e['id'],'stratum':label,'tokens':int(sel.sum()),'delta_fg_mean':float(deltafg.mean()),'delta_bg_mean':float(deltabg.mean()),'delta_margin_mean':float((deltafg-deltabg).mean()),'delta_fg_min':float(deltafg.min()),'delta_bg_min':float(deltabg.min()),'fg_monotonic_violations':int((deltafg < -1e-6).sum()) if co['kind']=='A' else None,'bg_monotonic_violations':int((deltabg < -1e-6).sum()) if co['kind']=='A' else None,'positive_to_negative':int((bp&~ap).sum()),'negative_to_positive':int((~bp&ap).sum()),'correct_before':int((bp==actual).sum()) if valid else None,'corrected':int(((bp!=actual)&(ap==actual)).sum()) if valid else None,'harmed':int(((bp==actual)&(ap!=actual)).sum()) if valid else None})
    if valid:
     correct=1 if actual else 0; wrong=1-correct; cs=nearest[ix,correct,sel]; ws=nearest[ix,wrong,sel]
     coverage.append({'condition':co['condition'],'image_id':e['id'],'gt_class':correct,'tokens':int(sel.sum()),'correct_max_cos_mean':float(cs.mean()),'wrong_max_cos_mean':float(ws.mean()),'correct_top1_class_wins':int((cs>ws).sum()),'correct_top50_class_wins':int(((tm[sel]>0)==actual).sum()),'diagnostic':'GT-assisted only'})
  for name,data in [('metrics',rows),('interference',inter),('coverage',coverage),('attribution',attrs),('contribution',contributions)]:table(folder/f'{name}.csv',data)
  np.savez_compressed(folder/'margins.npz',margins=np.array(margins),gt=gt,conditions=np.array([c['condition'] for c in conditions]))
  if e['id'] in subset:np.savez_compressed(folder/'token_scores.npz',scores=scores,nearest=nearest,occupancy=occupancy,conditions=np.array([c['condition'] for c in conditions]))
  dump(marker,{'config_fingerprint':FP,'code_sha256':CODE,'tokens':n,'source_analysis':e['id'] in subset}); print(ds,qi+1,len(queries),e['id'],'complete',flush=True)
 del groups; torch.cuda.empty_cache()

def main():
 OUT.mkdir(parents=True,exist_ok=True); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
 dump(OUT/'run_metadata.json',{'protocol':PROTO,'config_fingerprint':FP,'code_sha256':CODE,'pytorch':torch.__version__,'cuda':torch.version.cuda,'command':' '.join(sys.argv),'started':time.strftime('%Y-%m-%d %H:%M:%S')})
 if not (OUT/'support_integrity.csv').exists():integrity()
 for group in json.loads((OUT/'exact_duplicates.json').read_text()):
  if {x[0] for x in group['members']}=={'train','test'}:raise RuntimeError('Unresolved cross-split exact duplicates')
 for ds in ['GlaS','CRAG']:run_dataset(ds)
 dump(OUT/'inference_complete.json',{'status':'complete','config_fingerprint':FP})
if __name__=='__main__':main()
