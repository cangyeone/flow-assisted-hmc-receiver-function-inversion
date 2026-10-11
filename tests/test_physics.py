import numpy as np
import torch
from flowrf import forward,RFData,BasinModel,VelocityControls
from flowrf.inversion import PhysicalTarget

def test_physical_gradient_and_release_backend():
    from grtm import JointRFForward
    import grtm
    assert grtm.__version__=='0.6.0.dev2'
    model=BasinModel();x=torch.tensor([[2.,2.,35.,1.78]],dtype=torch.float64,requires_grad=True)
    w=torch.linspace(.01,1.,151,dtype=torch.float64)
    f=lambda v:(forward(model.layers(v),[.06],nfft=512).flatten()*w).sum()
    g=torch.autograd.grad(f(x),x)[0].numpy()[0]
    for j in range(4):
        xp=x.detach().clone();xm=x.detach().clone();h=1e-5;xp[0,j]+=h;xm[0,j]-=h
        fd=float((f(xp)-f(xm))/(2*h))
        np.testing.assert_allclose(g[j],fd,rtol=3e-5,atol=2e-6)
    assert forward(model.layers(x),[.06]).device.type=='cpu'
    import pytest
    with pytest.raises(ValueError,match='CPU'):JointRFForward([.06],device='mps')

def test_covariance_likelihood_joint_and_block_equivalent():
    layers=[[0,3.6,2,2.5],[2,6.5,6.5/1.78,2.8],[35,8,4.5,3.3]]
    y=forward(layers,[.045,.065],nfft=512).detach().numpy()[0]
    d=RFData(y,[.045,.065],nfft=512)
    c=d.covariance;m=c.shape[-1];joint=np.zeros((2*m,2*m));joint[:m,:m]=c[0];joint[m:,m:]=c[1]
    e=RFData(y,[.045,.065],covariance=joint,nfft=512)
    x=torch.tensor([[2.1,2.05,35.1,1.8]],dtype=torch.float64)
    torch.testing.assert_close(PhysicalTarget(d,BasinModel())(x),PhysicalTarget(e,BasinModel())(x))

def test_attenuation_and_controls_finite():
    model=VelocityControls(3);x=torch.tensor((model.lower+model.upper)[None]/2,dtype=torch.float64,requires_grad=True)
    layers=model.layers(x);layers=torch.cat((layers,torch.full_like(layers[...,:2],100.)),-1)
    rf=forward(layers,[.06],tau=model.shift(x),attenuation='constant_q')
    assert torch.isfinite(torch.autograd.grad(rf.square().sum(),x)[0]).all()
    assert np.allclose(rf.detach().numpy()[...,10],1.)
