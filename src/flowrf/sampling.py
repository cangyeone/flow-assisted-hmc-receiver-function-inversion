"""Our CPU Metropolis-within-Gibbs and a fixed HMC/MWG composition.

The composition uses frozen HMC step size/identity metric and symmetric
coordinate updates. It does not approximate or replace the physical target.
"""
import numpy as np
import torch
from ._utils import backend

def _coordinate_move(target,x,lp,lo,hi,scale,rng):
    accepted=torch.zeros_like(x)
    for j in range(x.shape[1]):
        y=x.clone();y[:,j]+=scale[j]*torch.randn(len(x),dtype=x.dtype,generator=rng)
        valid=(y[:,j]>lo[j])&(y[:,j]<hi[j])
        safe=torch.maximum(torch.minimum(y,hi-1e-12*(hi-lo)),lo+1e-12*(hi-lo))
        lpy=target(safe)
        take=valid&torch.isfinite(lpy)&(torch.rand(len(x),dtype=x.dtype,generator=rng).log()<torch.minimum(lpy-lp,torch.zeros_like(lp)))
        x=torch.where(take[:,None],y,x);lp=torch.where(take,lpy,lp);accepted[:,j]=take
    return x,lp,accepted

def mwg_sample(target,initial,lower,upper,*,steps=3000,burn_in=1000,seed=728,device='cpu',progress=None,scale=.01):
    if device!='cpu':raise ValueError('CPU only')
    if not 0<=burn_in<steps:raise ValueError('Invalid burn_in')
    x=torch.tensor(initial,dtype=torch.float64);lo=torch.tensor(lower);hi=torch.tensor(upper)
    if not bool(torch.isfinite(x).all() & (x>lo).all() & (x<hi).all()):raise ValueError('Initial states must lie strictly within bounds')
    if not np.isfinite(scale) or scale<=0:raise ValueError('scale must be positive')
    proposal=(hi-lo)*scale;rng=torch.Generator().manual_seed(seed)
    samples=[];density=[];accepted=torch.zeros_like(x)
    with torch.no_grad():
        lp=target(x)
        for i in range(steps):
            x,lp,take=_coordinate_move(target,x,lp,lo,hi,proposal,rng)
            if i>=burn_in:accepted+=take
            samples.append(x.clone());density.append(lp.clone())
            if progress and ((i+1)%100==0 or i+1==steps):progress(i+1,steps,lp)
    chain=torch.stack(samples)
    return dict(chain=chain,posterior=chain[burn_in:],log_prob=torch.stack(density),acceptance=(accepted/(steps-burn_in)).mean(-1))

def mixed_sample(target,initial,lower,upper,*,steps=3000,burn_in=1000,seed=728,device='cpu',leapfrog_steps=8,progress=None,step_size=.01):
    if device!='cpu':raise ValueError('CPU only')
    if not 0<=burn_in<steps:raise ValueError('Invalid burn_in')
    hmc=backend()[1];x=torch.tensor(initial,dtype=torch.float64);lo=torch.tensor(lower);hi=torch.tensor(upper)
    rng=torch.Generator().manual_seed(seed);samples=[];density=[];accepted=[];div=torch.zeros(len(x),dtype=torch.int64)
    for i in range(steps):
        # Two corrected HMC transitions, then one complete coordinate sweep.
        r=hmc(target,x,lower,upper,steps=2,burn_in=0,step_size=step_size,leapfrog_steps=leapfrog_steps,adapt_mass=False,seed=seed+i+1,device='cpu')
        x=r['chain'][-1];lp=r['log_prob'][-1]
        with torch.no_grad():x,lp,take=_coordinate_move(target,x,lp,lo,hi,(hi-lo)*.005,rng)
        samples.append(x.clone());density.append(lp.clone());accepted.append(r['acceptance'])
        if i>=burn_in:div+=r['production_divergences']
        if progress and ((i+1)%100==0 or i+1==steps):progress(i+1,steps,lp)
    chain=torch.stack(samples)
    return dict(chain=chain,posterior=chain[burn_in:],log_prob=torch.stack(density),acceptance=torch.stack(accepted)[burn_in:].mean(0),production_divergences=div)
