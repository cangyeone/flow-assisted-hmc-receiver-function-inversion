import numpy as np
import torch
from grtm import hmc_sample,dram_sample,rank_diagnostics
from flowrf.sampling import mwg_sample,mixed_sample

def target(x):return -.5*x.square().sum(-1)

def test_hmc_recovers_gaussian():
    r=hmc_sample(target,np.zeros((4,2)),[-6,-6],[6,6],steps=1500,burn_in=500,seed=543,device='cpu')
    s=r['posterior'].numpy().reshape(-1,2)
    np.testing.assert_allclose(s.mean(0),0,atol=.12);np.testing.assert_allclose(s.var(0),1,atol=.18)
    assert not r['production_divergences'].any()

def test_dram_recovers_gaussian():
    r=dram_sample(target,np.zeros((4,2)),[-6,-6],[6,6],steps=5000,burn_in=1000,seed=544,device='cpu')
    s=r['posterior'].numpy().reshape(-1,2)
    np.testing.assert_allclose(s.mean(0),0,atol=.16);np.testing.assert_allclose(s.var(0),1,atol=.22)

def test_mwg_recovers_gaussian_and_keeps_rejections():
    r=mwg_sample(target,np.zeros((4,2)),[-6,-6],[6,6],steps=5000,burn_in=1000,seed=545,scale=.1)
    s=r['posterior'].numpy().reshape(-1,2)
    np.testing.assert_allclose(s.mean(0),0,atol=.16);np.testing.assert_allclose(s.var(0),1,atol=.22)
    assert (r['chain'][1:]==r['chain'][:-1]).all(-1).any()

def test_fixed_mixture_kernel_and_constant_chain_diagnostics():
    r=mixed_sample(target,np.zeros((4,1)),[-6],[6],steps=700,burn_in=100,seed=546,leapfrog_steps=3,step_size=.15)
    s=r['posterior'].numpy().reshape(-1)
    assert abs(s.mean())<.2 and abs(s.var()-1)<.25
    assert not rank_diagnostics(np.zeros((100,4,2)))['converged']
