"""Exportable scientific summary figures from completed experimental CSVs."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_complementarity import OUT,MODELS


def main():
    d=OUT/'metrics';dest=OUT/'visualizations';dest.mkdir(exist_ok=True)
    single=pd.read_csv(d/'individual_models.csv');vote=pd.read_csv(d/'majority_voting.csv');oracle=pd.read_csv(d/'oracle_upper_bound.csv')
    allrows=pd.concat([single,vote,oracle]);order=MODELS+['majority','oracle'];colors=['#277da8','#f8961e','#43aa8b','#8f5caa','#a0a0a0']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,ds in zip(axes,['GlaS','CRAG']):
        groups=[allrows[(allrows.dataset==ds)&(allrows.model==m)].dice.to_numpy()*100 for m in order]
        ax.bar(np.arange(5),[x.mean() for x in groups],yerr=[x.std(ddof=1) for x in groups],capsize=4,color=colors)
        for j,x in enumerate(groups):ax.scatter(np.full(len(x),j)+np.linspace(-.07,.07,len(x)),x,s=16,color='black',zorder=4)
        ax.set_xticks(np.arange(5),['UNI','UNI2-h','Kaiko','Voting','Oracle*']);ax.set_ylim(0,100)
        ax.set_title(ds);ax.set_ylabel('Mean image Dice (%)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('20-shot / 5 fixed seeds; error bars: sample SD\n* Oracle uses GT and is not deployable')
    fig.savefig(dest/'independent_voting_oracle.png',dpi=180);fig.savefig(dest/'independent_voting_oracle.pdf');plt.close(fig)
    c=pd.read_csv(d/'correction_seed.csv');c=c[c.tissue=='overall']
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
    for ax,ds in zip(axes,['GlaS','CRAG']):
        values=np.full((3,3),np.nan)
        for i,a in enumerate(MODELS):
            for j,b in enumerate(MODELS):
                if a!=b:values[i,j]=c[(c.dataset==ds)&(c.model_a==a)&(c.model_b==b)].correction.mean()*100
        im=ax.imshow(values,cmap='Blues',vmin=0,vmax=100)
        for (i,j),x in np.ndenumerate(values):
            if np.isfinite(x):ax.text(j,i,f'{x:.1f}%',ha='center',va='center',color='white' if x>60 else 'black')
        ax.set_xticks(range(3),['UNI','UNI2-h','Kaiko']);ax.set_yticks(range(3),['UNI','UNI2-h','Kaiko'])
        ax.set_xlabel('Correcting model B');ax.set_ylabel('Source error model A');ax.set_title(ds);fig.colorbar(im,ax=ax,fraction=.045,label='Correction(A→B), %')
    fig.suptitle('Directional correction of source-model errors\nSeed-level pooled fractions, averaged over five seeds')
    fig.savefig(dest/'directional_correction.png',dpi=180);fig.savefig(dest/'directional_correction.pdf');plt.close(fig)

if __name__=='__main__':main()
