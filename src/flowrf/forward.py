import numpy as np
import torch
from ._utils import backend

def forward(layers,p,*,gaussians=(2.5,),dt=.2,n_samples=151,t0=-2.,nfft=1024,tau=0.,p_sd=0.,attenuation=None):
    """Differentiable CPU physical RF, [batch, bin, band, time].

    Layer rows: top_km,Vp_km_s,Vs_km_s,rho_g_cm3[,Qp,Qs]. Last row
    is the halfspace. All internal tensors are float64/complex128 on CPU.
    With p_sd>0, use three quadrature rays and average P-normalized RFs.
    Positive tau delays the predicted arrivals; data are never shifted.
    """
    x=torch.as_tensor(layers,dtype=torch.float64,device='cpu')
    if x.ndim==2:x=x[None]
    if x.ndim!=3 or x.shape[-1] not in (4,6):raise ValueError('Expected [batch,layer,4 or 6]')
    p=np.atleast_1d(np.asarray(p,float));sd=np.broadcast_to(np.asarray(p_sd,float),p.shape)
    if not np.isfinite(sd).all() or np.any(sd<0):raise ValueError('p_sd must be finite nonnegative')
    rays=(p[:,None]+sd[:,None]*np.sqrt(1.5)*np.array([-1.,0,1])).ravel()
    att=attenuation or ('constant_q' if x.shape[-1]==6 else 'elastic')
    F=backend()[0](rays,gaussians=gaussians,periods=[0.],dt=dt,n_samples=n_samples,t0=t0,nfft=nfft,device='cpu',attenuation=att)
    spectra=F._evaluate(x,check=True)
    r,z=spectra['radial_spectrum'],spectra['vertical_spectrum']
    power=z.abs().square();den=torch.maximum(power,F.water_level*power.amax(-1,keepdim=True))
    norm=torch.fft.irfft(power[:,:,None]/den[:,:,None]*F.filters,n=F.nfft)[...,:1]
    shift=torch.as_tensor(tau,dtype=x.dtype,device='cpu').reshape(-1,1,1,1)
    phase=F.shift*torch.exp(-1j*F.omega*shift)
    y=torch.fft.irfft(r[:,:,None]*z[:,:,None].conj()/den[:,:,None]*F.filters/norm*phase,n=F.nfft)[...,:n_samples]
    a=-t0/dt
    if abs(a-round(a))>1e-6 or not 0<=round(a)<n_samples:raise ValueError('Grid must contain t=0')
    ref=y[...,int(round(a)):int(round(a))+1]
    if not bool(torch.isfinite(y).all()) or bool((ref.abs()<1e-10).any()):raise FloatingPointError('Invalid or near-zero P normalization')
    return (y/ref).reshape(len(y),len(p),3,len(gaussians),n_samples).mean(2)
