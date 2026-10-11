import ast
from pathlib import Path
import numpy as np
import pytest
import torch
from flowrf import RFData,NeuralModels,RFInverter,forward
from flowrf.neural.models import PosteriorTransformerFlow
from flowrf.neural.structure import decode,encode

def example():
    layers=[[0,3.6,2,2.5],[2,6.5,6.5/1.78,2.8],[35,8,4.5,3.3]]
    y=forward(layers,[.044,.059,.075],gaussians=[1,2.5,4]).detach().numpy()[0]
    return layers,RFData(y,[.044,.059,.075],gaussians=(1,2.5,4))

def test_neural_models_cpu_shape_seed_and_integrity():
    layers,d=example();n=NeuralModels()
    a=n.propose(d,draws=3,count=2,seed=123);b=n.propose(d,draws=3,count=2,seed=123)
    np.testing.assert_array_equal(a['layers'],b['layers']);assert (a['count']==2).all()
    y=n.generate(layers,d.p,draws=4)
    assert y['mean'].shape==(1,9,151) and y['samples'].shape==(1,4,9,151)
    assert y['std'].min()>=0 and y['std'].max()<=.030001
    np.testing.assert_allclose(y['samples'][...,10],1.,atol=1e-6)
    assert all(p.device.type=='cpu' for p in n.load('inverse').parameters())
    with pytest.raises(ValueError,match='support'):n.propose(d,draws=1,count=17)

def test_coordinate_roundtrip_and_exact_flow_inverse():
    u=torch.randn(2,4,5,dtype=torch.float64)*.2;k=torch.tensor([2,3])
    layers,t,j=decode(u,k);v=encode(layers,k,t,3)
    from flowrf.neural.structure import active_mask
    mask=active_mask(k,3,dtype=torch.float64)
    torch.testing.assert_close(u*mask,v,rtol=1e-8,atol=1e-9)
    net=PosteriorTransformerFlow(max_layers=3,width=16,depth=1,blocks=2).double();context=torch.randn(2,16,dtype=torch.float64)
    a=u*mask
    for b in net.flow:
        x,ld=b(a,mask,context);z,ild=b(x,mask,context,inverse=True)
        torch.testing.assert_close(z,a);torch.testing.assert_close(ld,-ild)
        a=x
    assert torch.isfinite(j).all()

def test_validation_roundtrip_and_no_missing_band_fabrication(tmp_path):
    _,d=example();p=tmp_path/'data.npz';d.save(p);r=RFData.load(p)
    np.testing.assert_array_equal(r.rf,d.rf);np.testing.assert_array_equal(r.covariance,d.covariance)
    y=d.rf.copy();y[:,10]=0
    with pytest.raises(ValueError,match='Normalize'):RFData(y,d.p,d.gaussians)
    single=RFData(d.rf[1::3],d.p)
    with pytest.raises(ValueError,match='do not invent'):NeuralModels().propose(single,1)
    with pytest.raises(ValueError,match='production draws'):RFInverter().invert(d,steps=10,burn_in=5,starts=1)
    with pytest.raises(ValueError,match='two starts'):RFInverter().invert(d,initializer='flow',starts=1)

def test_end_to_end_result_and_plot(tmp_path):
    _,d=example()
    r=RFInverter().invert(d,starts=1,chains=2,steps=18,burn_in=8,maxiter=2,leapfrog_steps=2)
    r.save(tmp_path);assert r.samples.shape==(10,2,4)
    assert r.summary['status']=='provisional_unconverged'
    assert np.isfinite(r.predictive).all()
    r.plot(tmp_path/'rf.png');assert (tmp_path/'rf.png').stat().st_size>1000

def test_no_accelerator_calls_or_vendor_code_in_public_package():
    root=Path(__file__).resolve().parents[1]
    for path in (root/'src').rglob('*.py'):
        text=path.read_text();ast.parse(text)
        assert 'torch.cuda' not in text and 'cuda_graph' not in text and 'rf_cuda' not in text
    assert not list((root/'src').rglob('*.so'))
