"""Encoder-isolated, fingerprinted native-scale features and real-GT occupancy."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from .geometry import windows
from .encoder import sha256

ALGORITHM_VERSION = 'n3bil-multifm-v1'


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def save_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        writer = np.savez if "tokens" in values else np.savez_compressed
        writer(f, **values)
    os.replace(temp, path)


class FeatureStore:
    def __init__(self, root, encoder, cfg, legacy_root=None):
        self.root, self.encoder, self.cfg = Path(root), encoder, cfg
        self.legacy_root = Path(legacy_root) if legacy_root else None
        self.records = []

    def get(self, image, image_id, offset, supplied=None, supplied_source=None):
        e, w = self.encoder, self.cfg['window']
        views = list(windows(image, w['size'], w['stride'], offset, w['padding_mode']))
        positions = np.asarray([(y,x) for y,x,_ in views])
        meta = {'algorithm_version': ALGORITHM_VERSION, 'encoder': e.metadata,
                'window': w['size'], 'stride': w['stride'], 'offset_pixels': offset,
                'padding_mode': w['padding_mode'], 'image_id': image_id,
                'image_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                'positions': positions.tolist()}
        key = signature(meta)
        path = self.root / f'{image_id}_offset{offset}.npz'
        reused, migrated = False, False
        expected = (len(views), (w['size']//e.patch_size)**2, e.feature_dim)
        tokens = None
        if path.is_file():
            try:
                with np.load(path, allow_pickle=False) as z:
                    if str(z['fingerprint']) == key and json.loads(str(z['metadata'])) == meta:
                        tokens = z['tokens']; reused = True
            except (ValueError, KeyError, OSError):
                pass
        if tokens is None and supplied is not None:
            tokens = supplied; migrated = True; meta['migration_source'] = supplied_source
        if tokens is None and self.legacy_root and e.name == 'uni':
            legacy = self.legacy_root / f'{image_id}_offset{offset}.npz'
            if legacy.is_file():
                oldkey = hashlib.sha256(image.tobytes() + str((e.weights_sha256,w['size'],w['stride'],offset,e.precision,e.token_precision)).encode()).hexdigest()
                with np.load(legacy, allow_pickle=False) as z:
                    if str(z['cache_key']) == oldkey and np.array_equal(z['positions'],positions):
                        tokens = z['tokens']; migrated = True
                # Migration provenance is separate from identity to keep cache reuse stable.
        if tokens is None:
            tokens = e.encode([crop for _,_,crop in views])
        if tokens.shape != expected or not np.isfinite(tokens).all():
            raise ValueError(f'Invalid feature cache geometry or values: {path}: {tokens.shape} != {expected}')
        norms = np.linalg.norm(tokens.astype(np.float32), axis=-1)
        if not np.allclose(norms, 1, atol=.001):
            raise ValueError(f'Non-unit cached features: {path}')
        # Do not embed migration_source into the canonical identity.
        meta.pop('migration_source', None)
        if not reused:
            save_npz(path,tokens=tokens,positions=positions,metadata=np.array(json.dumps(meta)),fingerprint=np.array(key))
        self.records.append({'image_id':image_id,'offset':offset,'fingerprint':key,
                             'cache_file':str(path),'cache_reused':reused,'legacy_migrated':migrated,
                             'source':supplied_source,'window_count':len(views)})
        return tokens, positions

    def occupancy(self, mask, image_id, gt_path, offset, positions):
        e,w = self.encoder,self.cfg['window']
        views = list(windows(mask,w['size'],w['stride'],offset,w['padding_mode']))
        if not np.array_equal(positions,[(y,x) for y,x,_ in views]):
            raise ValueError('GT and RGB window positions differ')
        p,g = e.patch_size,w['size']//e.patch_size
        # Recompute from current real GT every time, never trust unlabeled occupancy.
        occ=np.stack([crop.reshape(g,p,g,p).mean((1,3)).reshape(-1) for _,_,crop in views]).astype(np.float32)
        meta={'algorithm_version':ALGORITHM_VERSION,'image_id':image_id,'gt_sha256':sha256(gt_path),
              'encoder_id':e.name,'checkpoint_sha256':e.weights_sha256,'patch_size':p,
              'window':w['size'],'stride':w['stride'],'offset_pixels':offset,
              'positions':positions.tolist(),'occupancy_source':'real_support_gt',
              'mask_encoding':self.cfg['dataset']['mask_encoding'],
              'class_mapping':self.cfg['dataset']['class_mapping']}
        path=self.root/'occupancy'/f'{image_id}_offset{offset}.npz'
        save_npz(path,fg_occupancy=occ,metadata=np.array(json.dumps(meta)),fingerprint=np.array(signature(meta)))
        return occ,meta
