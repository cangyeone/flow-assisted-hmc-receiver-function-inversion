from pathlib import Path
import numpy as np

def plot_result(result,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=result.data;pred=result.predictive;c=len(d.rf)
    fig,axes=plt.subplots(c,1,figsize=(7,1.8*c),sharex=True,squeeze=False)
    q=np.quantile(pred,[.025,.975],axis=0);mean=pred.mean(0)
    for i,ax in enumerate(axes[:,0]):
        ax.fill_between(d.time,q[0,i],q[1,i],color='#176B8E',alpha=.2,label='95% structural predictive interval')
        ax.plot(d.time,d.rf[i],color='#22323D',lw=1,label='Observed RF')
        ax.plot(d.time,mean[i],color='#C67735',lw=1,label='Physical posterior mean RF')
        b,g=divmod(i,len(d.gaussians));ax.set_ylabel('RF');ax.set_title(f'p={d.p[b]:.4f} s/km; Gaussian={d.gaussians[g]:g}',loc='left',fontsize=9)
        ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend(fontsize=7);axes[-1,0].set_xlabel('Time after P arrival (s)')
    fig.suptitle(result.summary['status'].replace('_',' '));fig.tight_layout()
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=200);plt.close(fig)
    return path
