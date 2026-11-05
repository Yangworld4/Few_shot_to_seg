import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from pathology_trainfree.encoder import FeatureEncoder
from pathology_trainfree.config import load_config
root=Path(__file__).resolve().parents[1]
paths={'uni':next(Path('/home/yangorange/.cache/huggingface/hub/models--MahmoodLab--UNI/snapshots').glob('*/pytorch_model.bin')),'uni2_h':next(Path('/home/yangorange/.cache/huggingface/hub/models--MahmoodLab--UNI2-h/snapshots').glob('*/pytorch_model.bin')),'kaiko_vitb8':Path('/home/yangorange/.cache/torch/hub/checkpoints/vitb8.pth')}
torch.set_num_threads(4)
for name,weights in paths.items():
 cfg=load_config(root/'configs/uni_memory.yaml');cfg['encoder'].update(name=name,weights=str(weights));t=time.monotonic()
 encoder=FeatureEncoder(cfg['encoder']);torch.cuda.reset_peak_memory_stats()
 arr=encoder.extract_features([np.random.default_rng(0).integers(0,256,(448,448,3),dtype=np.uint8)])
 norms=np.linalg.norm(arr.astype('float32'),axis=-1)
 assert np.isfinite(arr).all() and np.allclose(norms,1,atol=.001)
 assert not encoder.model.training and all(not p.requires_grad and p.grad is None for p in encoder.model.parameters())
 meta={**encoder.metadata,'shape':list(arr.shape),'norm_min':float(norms.min()),'norm_max':float(norms.max()),'seconds_including_load':time.monotonic()-t,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'status':'PASS'}
 (root/f'outputs/multi_fm_complementarity/configs/{name}_smoke.json').write_text(json.dumps(meta,indent=2))
 print(name,meta,flush=True);del encoder;torch.cuda.empty_cache()
