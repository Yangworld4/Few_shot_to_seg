"""Validate larger encoder batches without changing the segmentation protocol."""
import sys,json,time,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from run_complementarity import OUT,config_for
from pathology_trainfree.encoder import FeatureEncoder
from pathology_trainfree.data import read_manifest,load_image
from pathology_trainfree.geometry import windows

torch.set_num_threads(4)
for name in ['uni2_h','kaiko_vitb8']:
 cfg=config_for('GlaS',14,name);cfg['encoder']['batch_size']=4
 e=FeatureEncoder(cfg['encoder']);entry=read_manifest(cfg['dataset']['support_manifest'])[0]
 crops=[crop for _,_,crop in windows(load_image(entry['image']),448,224,0)][:4]
 while len(crops)<4:crops.append(crops[0])
 torch.cuda.reset_peak_memory_stats();t=time.monotonic();batch=e.encode(crops);seconds=time.monotonic()-t
 e.batch_size=1;single=e.encode(crops[:1])
 delta=float(np.abs(batch[0].astype('float32')-single[0].astype('float32')).max())
 assert np.isfinite(batch).all() and np.allclose(np.linalg.norm(batch.astype('float32'),axis=-1),1,atol=.001)
 assert delta<.002
 result={'model':name,'batch_size':4,'status':'PASS','shape':list(batch.shape),'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'seconds_four_windows':seconds,'max_abs_fp16_token_difference_vs_batch1':delta,'note':'compute batching only; real pretrained weights, no GT used'}
 (OUT/f'configs/{name}_batch_smoke.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
 del e;gc.collect();torch.cuda.empty_cache()
