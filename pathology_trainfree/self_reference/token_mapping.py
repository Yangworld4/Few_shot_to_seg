"""Native seed occupancy and reflected physical token coordinates; no GT dependency."""
from pathlib import Path
import hashlib
import numpy as np
from ..geometry import windows


def reflect_coordinate(x,length):
    if length<2:raise ValueError('Reflection needs at least two pixels')
    period=2*(length-1);v=np.mod(x,period)
    return np.where(v<=length-1,v,period-v)


def map_seeds(seed,window):
    shape=seed['fg'].shape;size=window['size'];patch=window['patch_size'];grid=size//patch
    out={k:[] for k in ['fg','bg','ignore','epithelial','interior']};positions=[];centers=[];offset_ids=[]
    for offset in window['offsets']:
        views={k:list(windows(seed[k].astype(np.float32),size,window['stride'],offset,window['padding_mode'])) for k in out}
        for j,(y,x,_) in enumerate(views['fg']):
            positions.append((y,x));offset_ids.append(offset)
            yy,xx=np.meshgrid(y+np.arange(grid)*patch+(patch-1)/2,x+np.arange(grid)*patch+(patch-1)/2,indexing='ij')
            centers.append(np.column_stack([reflect_coordinate(yy.ravel(),shape[0]),reflect_coordinate(xx.ravel(),shape[1])]))
            for k in out:out[k].append(views[k][j][2].reshape(grid,patch,grid,patch).mean((1,3)).ravel())
    out={k:np.concatenate(v) for k,v in out.items()}
    if not np.allclose(out['fg']+out['bg']+out['ignore'],1):raise ValueError('Invalid token tri-state occupancy')
    return {**out,'positions':np.asarray(positions),'centers':np.concatenate(centers).astype(np.float32),'offset_ids':np.asarray(offset_ids)}


def load_query_cache(image,image_id,root,dataset,cfg):
    expected={'precision':'float32','token_precision':'float16','feature_dim':1024,'normalize':True}
    if any(cfg['encoder'][k]!=v for k,v in expected.items()):raise ValueError('UNI preprocessing/precision configuration differs from validated cache protocol')
    if cfg['window']['size']!=448 or cfg['window']['stride']!=224 or cfg['window']['patch_size']!=16:raise ValueError('Validated N3 caches require 448/224/16')
    features=[];positions=[];offset_ids=[];records=[]
    for offset in cfg['window']['offsets']:
        key=hashlib.sha256(image.tobytes()+str((cfg['encoder']['weights_sha256'],448,224,offset,'float32','float16')).encode()).hexdigest()
        path=Path(root)/dataset/'feature_cache'/f'{image_id}_offset{offset}.npz'
        with np.load(path,allow_pickle=False) as z:tokens=z['tokens'];pos=z['positions'];saved=str(z['cache_key'])
        expected=np.asarray([(y,x) for y,x,_ in windows(image,448,224,offset)])
        if saved!=key or not np.array_equal(pos,expected) or tokens.shape!=(len(pos),784,1024):
            raise ValueError(f'UNI cache fingerprint / geometry mismatch: {path}')
        features.append(tokens);positions.extend(pos);offset_ids.extend([offset]*len(pos));records.append({'offset':offset,'cache_key':key,'file':str(path)})
    return np.concatenate(features),np.asarray(positions),np.asarray(offset_ids),records
