"""Image-only fixed heuristics. No GT, manifests, memories or evaluator are imported here."""
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import slic
from skimage.morphology import disk
from sklearn.cluster import KMeans
from sklearn import __version__ as sklearn_version
from threadpoolctl import threadpool_limits
from .features import extract


def quantile(x,tissue,q):
    values=x[tissue]
    return float(np.percentile(values,q)) if len(values) else 0.0


def region_mining(rgb,maps,cfg):
    sc=cfg['superpixels'];h,w=rgb.shape[:2]
    n=int(np.clip(round(h*w/sc['target_area']),sc['minimum'],sc['maximum']))
    labels=slic(rgb,n_segments=n,compactness=sc['compactness'],sigma=sc['sigma'],
                max_num_iter=sc['max_num_iter'],start_label=0,channel_axis=-1,enforce_connectivity=True)
    n=int(labels.max())+1;flat=labels.ravel();area=np.bincount(flat,minlength=n)
    yy,xx=np.indices((h,w),dtype=np.float32)
    fields=[maps['lab'][...,j] for j in range(3)]+[maps[k] for k in ['h','e','gray','texture']]+[np.ones((h,w)),yy/h,xx/w]
    means=np.array([np.bincount(flat,weights=x.ravel(),minlength=n)/np.maximum(area,1) for x in fields]).T
    means[:,7]=np.log1p(area)
    standardized=(means-means.mean(0))/np.maximum(means.std(0),1e-6)
    standardized*=np.array(sc['feature_weights'])
    k=min(sc['clusters'],n)
    with threadpool_limits(limits=1):
        km=KMeans(n_clusters=k,random_state=cfg['seed'],n_init=sc['kmeans_n_init'],algorithm='lloyd' if tuple(map(int,sklearn_version.split('.')[:2])) >= (1,1) else 'full').fit(standardized)
    # Display labels sorted by mean brightness; no semantic meaning assigned to five clusters.
    brightness=np.array([means[km.labels_==j,0].mean() for j in range(k)])
    lut=np.empty(k,int);lut[np.argsort(brightness,kind='stable')]=np.arange(k)
    clusters=lut[km.labels_];cmap=clusters[labels]
    adjacency=set()
    for a,b in [(labels[:-1,:],labels[1:,:]),(labels[:,:-1],labels[:,1:])]:
        pairs=np.column_stack((a[a!=b],b[a!=b]))
        for x,y in np.unique(pairs,axis=0):adjacency.add((int(x),int(y)));adjacency.add((int(y),int(x)))
    return labels,cmap,means,area,np.array(sorted(adjacency),np.int32).reshape(-1,2),clusters


def validate(seed):
    fg,bg,ignore=(np.asarray(seed[k],bool) for k in ['fg','bg','ignore'])
    if np.any(fg&bg) or not np.all(fg.astype(int)+bg.astype(int)+ignore.astype(int)==1):
        raise ValueError('Tri-state masks must be mutually exclusive and exhaustive')


def base_generate(rgb,cfg):
    maps=extract(rgb,cfg);tissue=maps['tissue'];mc=cfg['morphology'];shape=tissue.shape
    labels,clusters,means,areas,adj,region_clusters=region_mining(rgb,maps,cfg)
    h,e,g,tex=(maps[k] for k in ['h','e','gray','texture'])
    hcut=quantile(h,tissue,mc['dense_h_quantile']);gcut=quantile(g,tissue,mc['dense_gray_quantile']);tcut=quantile(tex,tissue,mc['dense_texture_quantile'])
    # Kmeans appearance clusters are selected by their measured stain and gray means, never labels 0..4.
    cluster_ok=[]
    for j in range(int(clusters.max())+1):
        p=clusters==j
        cluster_ok.append(bool(p.any() and h[p].mean()>=hcut and g[p].mean()<=gcut))
    epithelial_like=tissue&np.array(cluster_ok)[clusters]&(h>=hcut)&(tex>=tcut)
    closed=ndi.binary_closing(epithelial_like,structure=disk(mc['closing_radius']))
    filled=ndi.binary_fill_holes(closed);holes=filled&~closed
    lumen=holes&(g>=mc['lumen_gray_min'])&(maps['saturation']<=mc['lumen_saturation_max'])
    components,n=ndi.label(filled,structure=np.ones((3,3)));proposal=np.zeros(shape,bool);rules=[]
    for i,sl in enumerate(ndi.find_objects(components),1):
        local=components[sl]==i;area=int(local.sum());ratio=float((holes[sl]&local).sum()/max(area,1))
        valid=(mc['proposal_min_area']<=area<=np.prod(shape)*mc['proposal_max_image_fraction'] and ratio>=mc['proposal_hole_fraction_min']
               and sl[0].start>0 and sl[1].start>0 and sl[0].stop<shape[0] and sl[1].stop<shape[1])
        if valid:proposal[sl]|=local
    fg_a=ndi.binary_erosion(proposal,structure=disk(mc['erosion_a']))&tissue&~lumen
    # Positive eosin-rich / H-poor low-texture evidence, not complement-of-FG.
    stroma_a=(means[:,4]>=quantile(e,tissue,mc['bg_e_quantile']))&(means[:,3]<=quantile(h,tissue,mc['bg_h_quantile_a']))&(means[:,6]<=quantile(tex,tissue,mc['bg_texture_quantile_a']))&(means[:,4]>=mc['bg_e_h_ratio_min']*means[:,3])
    bg_a=ndi.binary_erosion(stroma_a[labels]&tissue&~ndi.binary_dilation(proposal,structure=disk(mc['bg_exclusion_radius'])),structure=disk(mc['erosion_a']))
    accepted=np.zeros(shape,bool);holelabels,nh=ndi.label(holes,structure=np.ones((3,3)))
    for j,sl in enumerate(ndi.find_objects(holelabels),1):
        # Use an explicit bounding crop for rim measurements, avoiding full-image morphology per component.
        r=mc['epithelial_radius'];pad=(slice(max(0,sl[0].start-r),min(shape[0],sl[0].stop+r)),slice(max(0,sl[1].start-r),min(shape[1],sl[1].stop+r)))
        obj=holelabels[pad]==j;area=int(obj.sum());height=sl[0].stop-sl[0].start;width=sl[1].stop-sl[1].start
        rim=ndi.binary_dilation(obj,structure=disk(mc['rim_radius']))&~obj
        enclosure=float(epithelial_like[pad][rim].mean()) if rim.any() else 0
        contrast=float(h[pad][rim].mean()-h[pad][obj].mean()) if rim.any() else 0
        neighbors=np.unique(labels[pad][rim&epithelial_like[pad]])
        checks=[('area',mc['hole_min_area']<=area<=np.prod(shape)*mc['hole_max_image_fraction']),
                ('border',sl[0].start>r and sl[1].start>r and sl[0].stop<shape[0]-r and sl[1].stop<shape[1]-r),
                ('aspect',max(height,width)/max(1,min(height,width))<=mc['hole_aspect_max']),
                ('enclosure',enclosure>=mc['enclosure_min']),
                ('stain_contrast',contrast>=mc['wall_interior_h_difference_min']),('adjacent_regions',len(neighbors)>=2)]
        passed=True
        for name,ok in checks:
            rules.append({'rule':name,'input_pixels':area if passed else 0,'rejected_pixels':area if passed and not ok else 0,'component':j,'pass':bool(ok),'enclosure':enclosure,'contrast':contrast})
            passed=passed and ok
        if passed:accepted[pad]|=obj
    epithelial=ndi.binary_dilation(accepted,structure=disk(mc['epithelial_radius']))&~ndi.binary_erosion(accepted,structure=disk(mc['rim_radius']))&epithelial_like&tissue&~lumen
    interior=ndi.binary_erosion(accepted,structure=disk(mc['interior_erosion']))&tissue&~lumen&(tex<=quantile(tex,tissue,mc['interior_texture_quantile']))&(h<quantile(h,tissue,mc['interior_h_quantile']))
    epithelial=ndi.binary_erosion(epithelial,structure=disk(mc['branch_erosion']))
    interior=ndi.binary_erosion(interior,structure=disk(mc['branch_erosion']))&~epithelial
    fg_b=epithelial|interior
    stroma_b=stroma_a&(means[:,3]<=quantile(h,tissue,mc['bg_h_quantile_b']))&(means[:,6]<=quantile(tex,tissue,mc['bg_texture_quantile_b']))&(means[:,5]>=quantile(g,tissue,mc['bg_gray_quantile']))
    near=np.zeros(len(means),np.float64);degree=np.zeros(len(means),np.float64)
    if len(adj):
        np.add.at(near,adj[:,0],stroma_b[adj[:,1]]);np.add.at(degree,adj[:,0],1)
    stroma_b&=(degree>0)&(near/np.maximum(1,degree)>=mc['bg_neighbor_fraction_min'])
    bg_b=ndi.binary_erosion(stroma_b[labels]&tissue&~ndi.binary_dilation(proposal|accepted,structure=disk(mc['bg_exclusion_radius'])),structure=disk(mc['erosion_a']))
    def make(fg,bg,epi,inner):
        conflict=fg&bg;fg=fg&~conflict;bg=bg&~conflict
        seed={'fg':fg,'bg':bg,'ignore':~(fg|bg),'epithelial':epi&fg,'interior':inner&fg}
        validate(seed);return seed
    return {'A':make(fg_a,bg_a,np.zeros(shape,bool),np.zeros(shape,bool)), 'B':make(fg_b,bg_b,epithelial,interior)}, {
        **maps,'slic':labels,'clusters':clusters,'region_features':means,'region_area':areas,'region_clusters':region_clusters,
        'adjacency':adj,'dense_candidate':epithelial_like,'closed':closed,'filled':filled,'proposal':proposal,
        'accepted_inner':accepted,'lumen_candidate':lumen,'rules':rules}


def stability_filter(original,views):
    fg=original['fg'].copy();bg=original['bg'].copy()
    for view in views:fg&=view['fg'];bg&=view['bg']
    out={'fg':fg,'bg':bg,'ignore':~(fg|bg),'epithelial':original['epithelial']&fg,'interior':original['interior']&fg}
    validate(out);return out


def generate(rgb,cfg):
    """Only RGB and predeclared configuration can enter seed construction."""
    seeds,maps=base_generate(rgb,cfg);views=[]
    for name in cfg['stability']['views']:
        if name=='hflip':transformed=rgb[:,::-1].copy();inverse=lambda a:a[:,::-1]
        elif name=='vflip':transformed=rgb[::-1].copy();inverse=lambda a:a[::-1]
        elif name=='rot90':transformed=np.rot90(rgb).copy();inverse=lambda a:np.rot90(a,-1)
        elif name=='brightness':
            gains=np.asarray(cfg['stability']['channel_gains'])*cfg['stability']['brightness_gain']
            transformed=np.clip(rgb*gains,0,255).astype(np.uint8);inverse=lambda a:a
        else:raise ValueError(f'Unknown predeclared view {name}')
        result,_=base_generate(transformed,cfg);views.append({k:inverse(v) for k,v in result['B'].items()})
    seeds['C']=stability_filter(seeds['B'],views)
    return seeds,maps
