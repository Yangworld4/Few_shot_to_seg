"""Image-only audit of fixed pixel gates; does not change generated seeds or use GT."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml
from scipy import ndimage as ndi
from skimage.morphology import disk
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pathology_trainfree.self_reference.features import extract
from pathology_trainfree.self_reference.seed_generator import quantile
from pathology_trainfree.data import read_manifest,load_image
OUT=ROOT/'outputs/self_reference_seed_diagnosis';cfg=yaml.safe_load((OUT/'configs/resolved.yaml').read_text());mc=cfg['morphology'];rows=[]
for ds,dc in cfg['datasets'].items():
 for entry in read_manifest(ROOT/dc['manifest']):
  folder=OUT/'images'/ds/entry['id'];m=extract(load_image(entry['image']),cfg);tissue=m['tissue'];h,e,g,texture=[m[k] for k in ['h','e','gray','texture']]
  with np.load(folder/'intermediate_maps.npz') as z:labels=z['slic'];clusters=z['clusters'];proposal=z['proposal'];accepted=z['accepted_inner'];adj=z['adjacency'];lumen=z['lumen_candidate']
  n=int(labels.max())+1;areas=np.bincount(labels.ravel(),minlength=n)
  means={k:np.bincount(labels.ravel(),weights=m[k].ravel(),minlength=n)/np.maximum(areas,1) for k in ['h','e','gray','texture']}
  conditions=[('positive_eosin',means['e']>=quantile(e,tissue,mc['bg_e_quantile'])),('low_h_A',means['h']<=quantile(h,tissue,mc['bg_h_quantile_a'])),('low_texture_A',means['texture']<=quantile(texture,tissue,mc['bg_texture_quantile_a'])),('e_over_h',means['e']>=mc['bg_e_h_ratio_min']*means['h'])]
  keep=tissue.copy();region=np.ones(n,bool)
  def record(branch,name,before,after):rows.append({'dataset':ds,'image_id':entry['id'],'branch':branch,'rule':name,'input_pixels':int(before.sum()),'accepted_pixels':int(after.sum()),'rejected_pixels':int(before.sum()-after.sum()),'gt_used':False})
  for name,rule in conditions:
   before=keep.copy();region&=rule;keep&=rule[labels];record('stroma_A',name,before,keep)
  # Reconstruct the fixed epithelial/interior gates from RGB, not from GT.
  hcut=quantile(h,tissue,mc['dense_h_quantile']);gcut=quantile(g,tissue,mc['dense_gray_quantile']);tcut=quantile(texture,tissue,mc['dense_texture_quantile'])
  ok=[]
  for j in range(int(clusters.max())+1):
   loc=clusters==j;ok.append(bool(loc.any() and h[loc].mean()>=hcut and g[loc].mean()<=gcut))
  dense=tissue.copy()
  for name,rule in [('cluster_stain_gray',np.array(ok)[clusters]),('high_h',h>=hcut),('wall_texture',texture>=tcut)]:
   before=dense.copy();dense&=rule;record('epithelial_dense',name,before,dense)
  epi=ndi.binary_dilation(accepted,structure=disk(mc['epithelial_radius']))&~ndi.binary_erosion(accepted,structure=disk(mc['rim_radius']))
  for name,rule in [('dense_appearance',dense),('tissue',tissue),('exclude_lumen',~lumen)]:
   before=epi.copy();epi&=rule;record('epithelial_seed',name,before,epi)
  before=epi.copy();epi=ndi.binary_erosion(epi,structure=disk(mc['branch_erosion']));record('epithelial_seed','erosion',before,epi)
  inner=ndi.binary_erosion(accepted,structure=disk(mc['interior_erosion']))
  for name,rule in [('tissue',tissue),('exclude_lumen',~lumen),('interior_texture',texture<=quantile(texture,tissue,mc['interior_texture_quantile'])),('interior_stain',h<quantile(h,tissue,mc['interior_h_quantile']))]:
   before=inner.copy();inner&=rule;record('interior_seed',name,before,inner)
  before=inner.copy();inner=ndi.binary_erosion(inner,structure=disk(mc['branch_erosion']));record('interior_seed','erosion',before,inner)
  before=inner.copy();inner&=~epi;record('interior_seed','disjoint_epithelial',before,inner)
  a_region=region.copy()
  for name,rule in [('low_h_B',means['h']<=quantile(h,tissue,mc['bg_h_quantile_b'])),('low_texture_B',means['texture']<=quantile(texture,tissue,mc['bg_texture_quantile_b'])),('bright_region_B',means['gray']>=quantile(g,tissue,mc['bg_gray_quantile']) )]:
   before=keep.copy();region&=rule;keep&=rule[labels];record('stroma_B',name,before,keep)
  near=np.zeros(n,float);degree=np.zeros(n,float)
  if len(adj):np.add.at(near,adj[:,0],region[adj[:,1]]);np.add.at(degree,adj[:,0],1)
  rule=(degree>0)&(near/np.maximum(1,degree)>=mc['bg_neighbor_fraction_min']);before=keep.copy();keep&=rule[labels];record('stroma_B','consistent_adjacent_regions',before,keep)
  before=keep.copy();keep&=~ndi.binary_dilation(proposal|accepted,structure=disk(mc['bg_exclusion_radius']));record('stroma_B','avoid_proposals',before,keep)
  before=keep.copy();keep=ndi.binary_erosion(keep,structure=disk(mc['erosion_a']));record('stroma_B','erosion',before,keep)
 pd.DataFrame(rows).to_csv(OUT/'positive_stroma_gate_audit.csv',index=False)
 print(ds,'image-only stroma gate audit complete',flush=True)
