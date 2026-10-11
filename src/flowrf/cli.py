import argparse
import json
from pathlib import Path
import numpy as np
import torch
from . import RFData,forward,RFInverter,NeuralModels,VelocityControls
from ._utils import plain,backend

def demo(folder,steps=30):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    layers=np.array([[0.,3.6,2.,2.5],[2.,6.5,6.5/1.78,2.8],[35.,8.,4.5,3.3]])
    p=np.array([.044,.059,.075]);bands=(1.,2.5,4.)
    clean=forward(layers,p,gaussians=bands).detach().numpy()[0]
    rng=np.random.default_rng(728);observed=clean+rng.normal(0,.03,clean.shape);observed[...,10]=1.
    d=RFData(observed,p,bands,sigma=.03);d.save(folder/'synthetic.npz')
    result=RFInverter().invert(d,initializer='sobol',starts=2,steps=steps,burn_in=steps//2,maxiter=12,leapfrog_steps=3)
    result.summary['demo_only']='Short-chain installation check, not a converged scientific benchmark.'
    result.save(folder)
    try:result.plot(folder/'rf-fit.png')
    except ImportError:pass
    print(json.dumps(plain(dict(output=folder,status=result.summary['status'],rf_rmse=result.summary['rf_rmse'])),indent=2))

def main(argv=None):
    ap=argparse.ArgumentParser(description='CPU flow-assisted Bayesian receiver-function inversion')
    sub=ap.add_subparsers(dest='command',required=True)
    sub.add_parser('install-grtm',help='Install the external CPU RF preview, with pinned SHA-256')
    sub.add_parser('doctor',help='Check published dependency and bundled model integrity')
    sub.add_parser('models',help='List released model sets')
    p=sub.add_parser('demo',help='Run a small synthetic installation check');p.add_argument('--output',default='outputs/demo');p.add_argument('--steps',type=int,default=30)
    p=sub.add_parser('forward');p.add_argument('model',help='JSON with layers and p');p.add_argument('--output',required=True)
    p=sub.add_parser('propose');p.add_argument('data');p.add_argument('--output',required=True);p.add_argument('--draws',type=int,default=256);p.add_argument('--count',type=int);p.add_argument('--seed',type=int,default=728)
    p=sub.add_parser('invert');p.add_argument('data');p.add_argument('--output',required=True);p.add_argument('--initializer',choices=['sobol','flow'],default='sobol');p.add_argument('--sampler',choices=['hmc','dram','mwg','hmc_mwg'],default='hmc');p.add_argument('--starts',type=int,default=16);p.add_argument('--chains',type=int,default=4);p.add_argument('--steps',type=int,default=3000);p.add_argument('--burn-in',type=int,default=1000);p.add_argument('--threads',type=int,default=1);p.add_argument('--controls',type=int);p.add_argument('--seed',type=int,default=728)
    a=ap.parse_args(argv)
    if a.command=='install-grtm':
        from .install import install_grtm
        install_grtm()
    elif a.command=='doctor':
        import grtm
        backend();n=NeuralModels();n.load('forward');n.load('inverse')
        print(json.dumps(dict(flowrf='0.1.0',grtm=grtm.__version__,torch=torch.__version__,device='cpu',models='SHA-256 and strict state-dict checks passed'),indent=2))
    elif a.command=='models':
        from .neural.api import WEIGHTS
        print((WEIGHTS/'manifest.json').read_text())
    elif a.command=='demo':demo(a.output,a.steps)
    elif a.command=='forward':
        Path(a.output).parent.mkdir(parents=True,exist_ok=True)
        doc=json.loads(Path(a.model).read_text());layers=doc.pop('layers');p=doc.pop('p')
        y=forward(layers,p,**doc).detach().numpy();np.savez_compressed(a.output,rf=y)
    elif a.command=='propose':
        Path(a.output).parent.mkdir(parents=True,exist_ok=True)
        r=NeuralModels().propose(RFData.load(a.data),a.draws,count=a.count,seed=a.seed)
        np.savez_compressed(a.output,**{k:v for k,v in r.items() if isinstance(v,np.ndarray)})
    else:
        model=VelocityControls(a.controls) if a.controls else None
        r=RFInverter(model,threads=a.threads).invert(a.data,initializer=a.initializer,sampler=a.sampler,starts=a.starts,chains=a.chains,steps=a.steps,burn_in=a.burn_in,seed=a.seed,progress=lambda i,n,lp:print(f'{i}/{n}',flush=True))
        r.save(a.output)
        try:r.plot(Path(a.output)/'rf-fit.png')
        except ImportError:pass
        print(json.dumps(plain(dict(output=a.output,status=r.summary['status'],rf_rmse=r.summary['rf_rmse'])),indent=2))
