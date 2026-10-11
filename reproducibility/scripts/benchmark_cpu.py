"""New CPU equal-protocol accuracy/coverage checks; not historical paper results."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from flowrf import forward,RFData,RFInverter,BasinModel
from flowrf._utils import write_json

def profile(x,z):
    hs,vs,H,k=x
    return np.where(z<hs,vs,np.where(z<H,6.5/k,4.5))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='reproducibility/outputs/new-cpu-benchmark');ap.add_argument('--repeats',type=int,default=4);ap.add_argument('--steps',type=int,default=3000);ap.add_argument('--burn-in',type=int,default=1000);ap.add_argument('--starts',type=int,default=16);ap.add_argument('--maxiter',type=int,default=180);ap.add_argument('--flow-draws',type=int,default=256);ap.add_argument('--methods',nargs='+',choices=['hmc','dram','mwg','hmc_mwg'],default=['hmc','dram','mwg']);a=ap.parse_args()
    torch.set_num_threads(1);root=Path(a.output);root.mkdir(parents=True,exist_ok=True);rows=[]
    truths={'basin':np.array([4.,1.8,40.,1.78]),'thin_sediment':np.array([.25,2.8,35.,1.78])};model=BasinModel();p=[.044,.059,.075];bands=[1,2.5,4];depth=np.arange(0,65,.2)
    write_json(root/'protocol.json',dict(repeats=a.repeats,truths=truths,steps=a.steps,burn_in=a.burn_in,starts=a.starts,maxiter=a.maxiter,flow_draws=a.flow_draws,methods=a.methods,seed_base=72000,noise='independent Gaussian sigma .03, conditioned to zero at normalization sample',scope='New CPU smoke/benchmark; two fixed structures with noise repeats do not establish population coverage or universal transfer; no original author program included'))
    for case,truth in truths.items():
        clean=forward(model.layers(torch.tensor(truth[None],dtype=torch.float64)),p,gaussians=bands).detach().numpy()[0]
        for repeat in range(a.repeats):
            seed=72000+repeat;noise=np.random.default_rng(seed).normal(0,.03,clean.shape);noise[...,10]=0
            data=RFData(clean+noise,p,gaussians=tuple(bands),sigma=.03)
            for method in a.methods:
                for init in ['sobol','flow']:
                    out=root/case/f'repeat-{repeat}'/(init+'_'+method)
                    r=RFInverter().invert(data,initializer=init,sampler=method,steps=a.steps,burn_in=a.burn_in,starts=a.starts,maxiter=a.maxiter,flow_draws=a.flow_draws,seed=seed)
                    s=r.samples.reshape(-1,4);q=np.quantile(s,[.025,.975],axis=0)
                    row=dict(case=case,repeat=repeat,initializer=init,sampler=method,parameter_rmse_normalized=float(np.sqrt(np.mean(((s.mean(0)-truth)/(model.upper-model.lower))**2))),velocity_profile_rmse=float(np.sqrt(np.mean((profile(s.mean(0),depth)-profile(truth,depth))**2))),parameter_covered=(q[0]<=truth)&(truth<=q[1]),interval_width=q[1]-q[0],rf_rmse=r.summary['rf_rmse'],status=r.summary['status'],initialization_seconds=r.summary['initialization_seconds'],sampling_seconds=r.summary['sampling_seconds'])
                    r.summary['accuracy_check']=row;r.save(out);rows.append(row);write_json(root/'metrics.json',rows);print(json.dumps({k:v for k,v in row.items() if isinstance(v,(str,int,float))}),flush=True)
    write_json(root/'interpretation.json',dict(resolved=sum(r['status']=='local_diagnostics_passed' for r in rows),unresolved=sum(r['status']!='local_diagnostics_passed' for r in rows),warning='Coverage from unresolved chains is provisional. Report every failure. Compare total time and diagnostics; do not infer structural superiority from RF fit alone.'))
if __name__=='__main__':main()
