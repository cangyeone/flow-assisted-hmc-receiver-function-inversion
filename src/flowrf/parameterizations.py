import numpy as np
import torch

class BasinModel:
    """Two finite layers plus mantle; physical box prior, sediment aware."""
    lower=np.array([.2,1.5,22.,1.6]);upper=np.array([8.,3.2,65.,2.])
    names=('sediment_thickness_km','sediment_Vs_km_s','Moho_depth_km','crust_VpVs')
    attenuation='elastic'
    def layers(self,x):
        hs,vs,H,k=x.unbind(-1);zero=hs*0;one=zero+1
        return torch.stack((torch.stack((zero,1.8*vs,vs,one*2.5),-1),torch.stack((hs,one*6.5,6.5/k,one*2.8),-1),torch.stack((H,one*8,one*4.5,one*3.3),-1)),-2)
    def shift(self,x):return x[:,0]*0
    def log_prior(self,x):return x[:,0]*0
    def project(self,layers,tau=0.):
        m=np.asarray(layers);h=m[1,0];H=m[-1,0]
        w=np.diff(m[1:,0]);vs=np.average(m[1:-1,2],weights=w)
        return np.clip([h,m[0,2],H,6.5/vs],self.lower+1e-5*(self.upper-self.lower),self.upper-1e-5*(self.upper-self.lower))

class VelocityControls:
    """Sediment and n piecewise constant crustal velocity controls.

    Fixed equal crustal thickness fractions; Gaussian smoothness, Vp/Vs
    and P-shift priors. Changing n changes the scientific model/prior.
    """
    attenuation='elastic'
    def __init__(self,n=4,smoothness=1.):
        if int(n)!=n or not 1<=n<=32:raise ValueError('n must be in [1,32]')
        if not np.isfinite(smoothness) or smoothness<0:raise ValueError('smoothness must be nonnegative finite')
        self.n=int(n);self.smoothness=float(smoothness)
        self.lower=np.array([.05,.7,15.]+[2.7]*self.n+[1.55,-.6])
        self.upper=np.array([10.,3.5,75.]+[4.3]*self.n+[2.05,.6])
        self.names=('sediment_thickness_km','sediment_Vs_km_s','Moho_depth_km')+tuple(f'crust_Vs_{i+1}_km_s' for i in range(self.n))+('crust_VpVs','P_time_shift_s')
    def layers(self,x):
        hs,vs,H=x[:,:3].unbind(-1);v=x[:,3:3+self.n];k=x[:,-2];zero=hs*0;one=zero+1
        top=hs[:,None]+(H-hs)[:,None]*torch.arange(self.n,dtype=x.dtype)[None]/self.n
        crust=torch.stack((top,v*k[:,None],v,torch.full_like(v,2.8)),-1)
        sed=torch.stack((zero,1.8*vs,vs,one*2.5),-1)[:,None]
        mantle=torch.stack((H,one*8,one*4.5,one*3.3),-1)[:,None]
        return torch.cat((sed,crust,mantle),1)
    def shift(self,x):return x[:,-1]
    def log_prior(self,x):
        v=x[:,3:3+self.n]
        penalty=(x[:,-1]/.25).square()+((x[:,-2]-1.78)/.12).square()+.25*((v-3.6)/.5).square().sum(-1)
        penalty=penalty+self.smoothness*((v[:,1:]-v[:,:-1])/.8).square().sum(-1)
        return -.5*penalty
    def project(self,layers,tau=0.):
        m=np.asarray(layers);hs=m[1,0];H=m[-1,0]
        z=hs+(H-hs)*(np.arange(self.n)+.5)/self.n
        j=np.searchsorted(m[:,0],z,side='right')-1
        v=m[j,2];k=np.mean(m[j,1]/m[j,2])
        x=np.r_[hs,m[0,2],H,v,k,tau]
        return np.clip(x,self.lower+1e-5*(self.upper-self.lower),self.upper-1e-5*(self.upper-self.lower))
