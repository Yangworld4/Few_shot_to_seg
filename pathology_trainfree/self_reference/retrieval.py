"""Existing cosine TopK with an explicit reference eligibility constraint."""
import numpy as np
import torch
import torch.nn.functional as F
from ..retrieval import SimilarityRetriever


class IneligibleMemory(ValueError):
    pass


def candidates(mapping, cfg):
    purity=cfg['memory']['purity']
    return {1:np.flatnonzero((mapping['fg']>=purity)&(mapping['bg']==0)),
            0:np.flatnonzero((mapping['bg']>=purity)&(mapping['fg']==0))}


def memory_status(counts,k):
    if min(counts.values())==0:return 'EMPTY_MEMORY'
    if min(counts.values())<k:return 'INSUFFICIENT_MEMORY'
    return 'VALID'


class SpatiallyExcludedRetriever(SimilarityRetriever):
    def __init__(self,memory,cfg,device,query_centers,reference_centers,radius):
        super().__init__(memory,cfg,device)
        self.query_centers=torch.as_tensor(query_centers,dtype=torch.float32,device=device)
        self.reference_centers={c:torch.as_tensor(reference_centers[c],dtype=torch.float32,device=device) for c in self.ids}
        self.radius=radius;self.cursor=0

    @torch.inference_mode()
    def feasibility(self):
        result={}
        for c in self.ids:
            minimum=10**9;bad=0
            for q in self.query_centers.split(self.chunk_size):
                count=(torch.cdist(q,self.reference_centers[c])>self.radius).sum(1)
                minimum=min(minimum,int(count.min()));bad+=int((count<self.k).sum())
            result[c]={'minimum_eligible':minimum,'insufficient_query_tokens':bad}
        return result

    @torch.inference_mode()
    def scores(self,tokens):
        q=F.normalize(torch.as_tensor(tokens,dtype=torch.float32,device=self.device).reshape(-1,1024),dim=1)
        centers=self.query_centers[self.cursor:self.cursor+len(q)];self.cursor+=len(q)
        if len(centers)!=len(q):raise ValueError('Query spatial cursor out of bounds')
        rows=[]
        for x,xy in zip(q.split(self.chunk_size),centers.split(self.chunk_size)):
            channels=[]
            for c in self.ids:
                eligible=torch.cdist(xy,self.reference_centers[c])>self.radius
                if (eligible.sum(1)<self.k).any():raise IneligibleMemory('Fewer than fixed K=50 spatially eligible neighbours')
                similarity=(x@self.banks[c].T).masked_fill(~eligible,-torch.inf)
                channels.append(similarity.topk(self.k,dim=1).values.mean(1))
            rows.append(torch.stack(channels,1))
        return torch.cat(rows).cpu()
