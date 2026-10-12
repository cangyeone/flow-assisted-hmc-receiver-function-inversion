from dataclasses import dataclass
from pathlib import Path
import time
import platform
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import qmc
from .data import RFData
from .forward import forward
from .parameterizations import BasinModel
from .neural.api import NeuralModels
from ._utils import backend,write_json

class PhysicalTarget:
    def __init__(self,data,model):
        self.data,self.model=data,model
        self.y=torch.tensor(data.rf[:,data.fit_mask],dtype=torch.float64)
        self.chol=torch.linalg.cholesky(torch.tensor(data.covariance,dtype=torch.float64))
    def predict(self,x):
        d=self.data
        return forward(self.model.layers(x),d.p,gaussians=d.gaussians,dt=d.dt,n_samples=d.rf.shape[-1],t0=d.t0,nfft=d.nfft,tau=self.model.shift(x),p_sd=d.p_sd,attenuation=self.model.attenuation).flatten(1,2)
    def __call__(self,x):
        residual=self.predict(x)[:,:,self.data.fit_mask]-self.y
        if self.chol.ndim==3:
            white=torch.linalg.solve_triangular(self.chol[None],residual[...,None],upper=False).squeeze(-1)
        else:
            white=torch.linalg.solve_triangular(self.chol,residual.flatten(1).T,upper=False).T
        return -.5*white.square().flatten(1).sum(-1)+self.model.log_prior(x)

@dataclass
class InversionResult:
    samples: np.ndarray
    chain: np.ndarray
    log_prob: np.ndarray
    predictive: np.ndarray
    summary: dict
    data: RFData
    def save(self,folder):
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'posterior.npz',samples=self.samples,chain=self.chain,log_prob=self.log_prob,predictive=self.predictive)
        self.data.save(folder/'input.npz');write_json(folder/'summary.json',self.summary)
        return folder
    def plot(self,path):
        from .plot import plot_result
        return plot_result(self,path)

class RFInverter:
    """Flow or Sobol proposals -> multi-start MAP -> corrected CPU sampling.

    The flow only supplies starts. Every optimized/sampled density uses the
    physical forward model and declared prior. Diagnostics are local checks,
    not a guarantee of exploring all disconnected modes.
    """
    def __init__(self,model=None,*,model_set='global-v2',threads=1):
        self.model=model or BasinModel();self.model_set=model_set
        if int(threads)!=threads or threads<1:raise ValueError('threads must be positive integer')
        self.threads=int(threads)
    def invert(self,data,*,initializer='sobol',sampler='hmc',starts=16,chains=4,steps=3000,burn_in=1000,flow_draws=256,seed=728,maxiter=180,leapfrog_steps=8,progress=None):
        if not isinstance(data,RFData):data=RFData.load(data)
        if initializer not in ('sobol','flow'):raise ValueError('initializer must be sobol or flow')
        if sampler not in ('hmc','dram','mwg','hmc_mwg'):raise ValueError('Unknown sampler')
        if initializer=='flow' and starts<2:raise ValueError('Flow initialization requires at least two starts to retain independent prior starts')
        for name,v,minimum in [('starts',starts,1),('chains',chains,2),('steps',steps,10),('burn_in',burn_in,0),('maxiter',maxiter,1),('flow_draws',flow_draws,1),('leapfrog_steps',leapfrog_steps,1)]:
            if int(v)!=v or v<minimum:raise ValueError(name+' has invalid value')
        if steps-burn_in<8:raise ValueError('At least 8 production draws required for diagnostics')
        torch.set_num_threads(self.threads)
        target=PhysicalTarget(data,self.model);lo=self.model.lower;hi=self.model.upper;span=hi-lo;dim=len(lo)
        rng=np.random.default_rng(seed);begin=time.perf_counter()
        sobol=qmc.Sobol(dim,scramble=True,seed=seed).random_base2(int(np.ceil(np.log2(starts))))[:starts]
        candidates=lo+span*(.001+.998*sobol)
        proposal_seconds=0.;proposal_count=0
        if initializer=='flow':
            t=time.perf_counter();s=NeuralModels(self.model_set).propose(data,flow_draws,seed=seed)
            projected=np.stack([self.model.project(m[:int(k)+1],tau) for m,k,tau in zip(s['layers'],s['count'],s['tau'])])
            # Retain independent prior starts to protect against learned-prior mismatch.
            scored=[]
            for i,x in enumerate(projected):
                try:
                    with torch.no_grad():value=float(target(torch.tensor(x,dtype=torch.float64)[None])[0])
                    if np.isfinite(value):scored.append((value,i))
                except (FloatingPointError,RuntimeError,ValueError):continue
            scored.sort(reverse=True);n=min(len(scored),max(1,starts//2))
            if n==0:raise RuntimeError('No finite physically evaluated flow candidates')
            candidates[:n]=projected[[i for _,i in scored[:n]]]
            proposal_count=len(projected);proposal_seconds=time.perf_counter()-t
        modes=[]
        for x0 in candidates:
            def fn(z):
                zt=torch.tensor(z,dtype=torch.float64,requires_grad=True)
                loss=-target((torch.tensor(lo)+torch.tensor(span)*zt)[None])[0]
                g=torch.autograd.grad(loss,zt)[0]
                if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(g).all()):return 1e100,np.zeros(dim)
                return float(loss.detach()),g.detach().numpy()
            out=minimize(fn,(x0-lo)/span,jac=True,method='L-BFGS-B',bounds=[(.0001,.9999)]*dim,options={'maxiter':maxiter,'ftol':1e-9,'gtol':1e-5})
            modes.append(dict(initial=x0,x=lo+span*out.x,negative_log_density=float(out.fun),success=bool(out.success),iterations=out.nit,evaluations=out.nfev))
        modes.sort(key=lambda m:m['negative_log_density'])
        selected=[modes[0]['x']]
        for mode in modes[1:]:
            if mode['negative_log_density']-modes[0]['negative_log_density']<=5 and all(np.linalg.norm((mode['x']-v)/span)>.03 for v in selected):selected.append(mode['x'])
        initial=np.stack([selected[i%len(selected)] for i in range(chains)])+rng.normal(size=(chains,dim))*.0005*span
        initial=np.clip(initial,lo+1e-5*span,hi-1e-5*span)
        initialization_seconds=time.perf_counter()-begin;t=time.perf_counter()
        _,hmc,dram,diagnose=backend()
        common=dict(steps=steps,burn_in=burn_in,seed=seed,device='cpu',progress=progress)
        if sampler=='hmc':r=hmc(target,initial,lo,hi,leapfrog_steps=leapfrog_steps,target_accept=.9,**common)
        elif sampler=='dram':r=dram(target,initial,lo,hi,proposal_scale=.015,**common)
        else:
            from .sampling import mwg_sample,mixed_sample
            r=mwg_sample(target,initial,lo,hi,**common) if sampler=='mwg' else mixed_sample(target,initial,lo,hi,leapfrog_steps=leapfrog_steps,**common)
        sampling_seconds=time.perf_counter()-t;post=r['posterior'].detach().numpy();diag=diagnose(post)
        divergence=np.asarray(r.get('production_divergences',np.zeros(chains)),int)
        passed=bool(diag['converged'] and not divergence.any())
        flat=post.reshape(-1,dim);take=np.linspace(0,len(flat)-1,min(64,len(flat)),dtype=int)
        prediction=[]
        with torch.no_grad():
            for chunk in np.array_split(flat[take],max(1,int(np.ceil(len(take)/4)))):prediction.append(target.predict(torch.tensor(chunk,dtype=torch.float64)).numpy())
        pred=np.concatenate(prediction);mean=pred.mean(0)
        summary=dict(version='0.1.1',device='cpu',threads=self.threads,python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,seed=seed,initializer=initializer,sampler=sampler,names=self.model.names,lower=lo,upper=hi,steps=steps,burn_in=burn_in,chains=chains,starts=starts,model_type=type(self.model).__name__,model_set=self.model_set if initializer=='flow' else None,
          flow_candidate_count=proposal_count,proposal_seconds=proposal_seconds,initialization_seconds=initialization_seconds,sampling_seconds=sampling_seconds,diagnostics=diag,production_divergences=divergence,
          status='local_diagnostics_passed' if passed else 'provisional_unconverged',acceptance=r.get('acceptance',r.get('acceptance_first',0)+r.get('acceptance_delayed',0)),initial=initial,candidates=modes,distinct_near_optimal_candidates=len(selected),mean=flat.mean(0),q025=np.quantile(flat,.025,axis=0),q975=np.quantile(flat,.975,axis=0),rf_rmse=float(np.sqrt(np.mean((mean[:,data.fit_mask]-data.rf[:,data.fit_mask])**2))),
          covariance_assumption='full cross-channel' if data.covariance.ndim==2 else 'independent channels with within-channel covariance',
          limitations='Conditional on the chosen parameterization/prior and covariance. Local diagnostics do not prove global mode coverage. Predictive arrays describe structural uncertainty, not new observation noise.')
        return InversionResult(post,r['chain'].numpy(),r['log_prob'].numpy(),pred,summary,data)
