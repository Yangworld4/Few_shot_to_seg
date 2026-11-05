"""Fixed optical-density deconvolution; image-only robust scaling."""
import numpy as np
from scipy import ndimage as ndi
from skimage.color import rgb2lab


def extract(rgb, cfg):
    f=cfg['features'];x=rgb.astype(np.float32)/255
    gray=x@np.array([.2126,.7152,.0722],np.float32)
    saturation=x.max(2)-x.min(2)
    od=-np.log(np.maximum(x,f['od_epsilon']))
    tissue=(od.sum(2)>f['tissue_od_min'])&(gray>f['tissue_gray_min'])&((gray<f['tissue_gray_max'])|(saturation>f['tissue_saturation_min']))
    # Reproducible HED convention: normalized log RGB times inverse mixing matrix, clipped nonnegative.
    stains=np.maximum((np.log(np.maximum(x,f['od_epsilon']))/np.log(f['od_epsilon']))@np.linalg.inv(np.asarray(f['stain_matrix'])),0).astype(np.float32)
    he=stains[...,:2].copy()
    for j in [0,1]:
        values=he[...,j][tissue]
        lo,hi=np.percentile(values,f['stain_quantiles']) if len(values) else (0,1)
        he[...,j]=np.clip((he[...,j]-lo)/max(float(hi-lo),1e-6),0,1)
    size=f['texture_size'];mu=ndi.uniform_filter(gray,size=size,mode='reflect')
    texture=np.sqrt(np.maximum(ndi.uniform_filter(gray*gray,size=size,mode='reflect')-mu*mu,0))
    return {'lab':rgb2lab(x).astype(np.float32),'h':he[...,0],'e':he[...,1],
            'gray':gray,'texture':texture,'saturation':saturation,'tissue':tissue,'raw_stains':stains}
