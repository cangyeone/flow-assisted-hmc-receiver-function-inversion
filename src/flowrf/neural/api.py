from pathlib import Path
import json
import numpy as np
import torch
from .models import ForwardTransformer,PosteriorTransformerFlow
from .data import Noise
from .._utils import sha256

WEIGHTS=Path(__file__).resolve().parents[1]/'weights'

class NeuralModels:
    """Released CPU emulators/candidate flows; no exact-posterior claim."""
    def __init__(self,model_set='global-v2'):
        self.folder=WEIGHTS/model_set
        manifest=json.loads((WEIGHTS/'manifest.json').read_text())
        self.record=manifest['model_sets'][model_set]
        self._models={}
    def load(self,arm):
        if arm not in ('forward','inverse'):raise ValueError('arm must be forward or inverse')
        if arm not in self._models:
            path=self.folder/f'{arm}.pt'
            if sha256(path)!=self.record[arm]['sha256']:raise ValueError('Checkpoint integrity mismatch')
            saved=torch.load(path,map_location='cpu',weights_only=True)
            c=saved['metadata']['configuration']
            m=ForwardTransformer(c['width'],c['depth'],c['std_min'],c['std_max']) if arm=='forward' else PosteriorTransformerFlow(c['max_layers'],c['min_layers'],c['width'],c['depth'],c['blocks'])
            m.load_state_dict(saved['state_dict']);m.eval()
            self._models[arm]=m
        return self._models[arm]
    def propose(self,data,draws=256,*,count=None,seed=728):
        data.require_neural_grid()
        if int(draws)!=draws or draws<1:raise ValueError('draws must be positive integer')
        tensors=[torch.as_tensor(v,dtype=torch.float32)[None] for v in [data.rf,data.p,data.p_sd,data.sigma]]
        model=self.load('inverse');pieces=[]
        with torch.random.fork_rng(devices=[]),torch.no_grad():
            torch.manual_seed(seed)
            for first in range(0,draws,32):pieces.append(model.sample(*tensors,draws=min(32,draws-first),count=count))
        result={k:torch.cat([p[k] for p in pieces]).numpy() for k in ['u','count','layers','tau','coordinate_log_jacobian']}
        result['count_probabilities']=pieces[0]['count_probabilities'][0].numpy()
        result['semantics']='learned simulator-prior candidates; requires physical refinement and convergence checks'
        return result
    def generate(self,layers,p,*,p_sd=0.,sigma=.012,tau=0.,draws=128,seed=728):
        if int(draws)!=draws or draws<0:raise ValueError('draws must be a nonnegative integer')
        x=torch.as_tensor(layers,dtype=torch.float32)
        if x.ndim==2:x=x[None]
        if x.ndim!=3 or x.shape[-1] not in (4,6) or x.shape[1]<2:raise ValueError('Expected finite layers plus halfspace, 4 or 6 columns')
        if x.shape[-1]==4:x=torch.cat((x,torch.full_like(x[...,:2],torch.inf)),-1)
        valid=torch.isfinite(x[...,:4]).all() & (x[:,0,0]==0).all() & (x[:,1:,0]>x[:,:-1,0]).all() & (x[...,2:4]>0).all() & (x[...,1]>np.sqrt(4/3)*x[...,2]).all() & (~torch.isnan(x[...,4:])).all() & (x[...,4:]>0).all()
        if not bool(valid):raise ValueError('Invalid layer structure')
        b=len(x)
        p=torch.tensor(np.broadcast_to(np.asarray(p,float),(b,3)).copy(),dtype=torch.float32)
        ps=torch.tensor(np.broadcast_to(np.asarray(p_sd,float),(b,3)).copy(),dtype=torch.float32)
        sg=torch.tensor(np.broadcast_to(np.asarray(sigma,float),(b,3)).copy(),dtype=torch.float32)
        t=torch.tensor(np.broadcast_to(np.asarray(tau,float),(b,)).copy(),dtype=torch.float32)
        if not bool(torch.isfinite(p).all() & (p>0).all() & torch.isfinite(ps).all() & (ps>=0).all() & torch.isfinite(sg).all() & (sg>0).all() & torch.isfinite(t).all()):raise ValueError('Invalid condition metadata')
        with torch.random.fork_rng(devices=[]),torch.no_grad():
            torch.manual_seed(seed)
            mean,std=self.load('forward')(x,torch.full((b,),x.shape[1]),p,ps,sg,t)
            noise=Noise('cpu')
            common=torch.randn(b,draws,3,1,151);independent=torch.randn(b,draws,3,3,151)
            z=(np.sqrt(.8)*common+np.sqrt(.2)*independent).reshape(b,draws,9,151)@noise.factor.T
            marginal=noise.marginal.clamp_min(1e-12)
            samples=mean[:,None]+z*(std/marginal)[:,None]
        return dict(mean=mean.numpy(),std=std.numpy(),samples=samples.numpy(),time=np.arange(151)*.2-2,
                    semantics='approximate neural RF; bounded heteroscedastic std is not emulator-error calibration')
