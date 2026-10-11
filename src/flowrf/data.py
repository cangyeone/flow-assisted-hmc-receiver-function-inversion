from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class RFData:
    """P-normalized RFs, in bin-major / Gaussian-band-major order.

    rf: [bin, band, time] or [bin*band, time]. Covariance is for the
    selected samples: [channel, sample, sample], or a full joint matrix.
    Missing covariance uses independent Gaussian errors with declared sigma.
    """
    rf: np.ndarray
    p: np.ndarray
    gaussians: tuple = (2.5,)
    dt: float = .2
    t0: float = -2.
    sigma: object = .03
    p_sd: object = 0.
    fit_mask: object = None
    covariance: object = None
    nfft: int = 1024

    def __post_init__(self):
        self.p=np.atleast_1d(np.asarray(self.p,float))
        self.gaussians=tuple(np.atleast_1d(self.gaussians).astype(float))
        if self.p.ndim!=1 or not len(self.p) or not np.isfinite(self.p).all() or np.any(self.p<=0):raise ValueError('p must be a positive finite vector in s/km')
        if not self.gaussians or not np.isfinite(self.gaussians).all() or min(self.gaussians)<=0:raise ValueError('gaussians must be positive finite widths')
        self.rf=np.asarray(self.rf,float)
        c=len(self.p)*len(self.gaussians)
        if self.rf.ndim==3:
            if self.rf.shape[:2]!=(len(self.p),len(self.gaussians)):raise ValueError('RF bin/band dimensions disagree with metadata')
            self.rf=self.rf.reshape(c,-1)
        if self.rf.ndim!=2 or self.rf.shape[0]!=c or self.rf.shape[1]<2 or not np.isfinite(self.rf).all():raise ValueError('RF must be finite [bin*band,time]')
        if not np.isfinite(self.dt) or self.dt<=0 or not np.isfinite(self.t0):raise ValueError('Invalid time sampling')
        if int(self.nfft)!=self.nfft or self.nfft<2*self.rf.shape[-1] or self.nfft&(self.nfft-1):raise ValueError('nfft must be a power of two >= twice the trace length')
        i=-self.t0/self.dt
        if not 0<=round(i)<self.rf.shape[-1] or abs(i-round(i))>1e-6:raise ValueError('Grid must contain a sample at P arrival t=0')
        self.anchor=int(round(i))
        if not np.allclose(self.rf[:,self.anchor],1.,atol=1e-4,rtol=0):raise ValueError('Normalize every RF by its t=0 amplitude before inversion')
        self.sigma=np.broadcast_to(np.asarray(self.sigma,float),(len(self.p),)).copy()
        self.p_sd=np.broadcast_to(np.asarray(self.p_sd,float),(len(self.p),)).copy()
        if not np.isfinite(self.sigma).all() or np.any(self.sigma<=0):raise ValueError('sigma must be positive finite per-bin amplitudes')
        if not np.isfinite(self.p_sd).all() or np.any(self.p_sd<0) or np.any(self.p-np.sqrt(1.5)*self.p_sd<=0):raise ValueError('Invalid slowness spread')
        t=self.time
        self.fit_mask=((t>=.8-1e-8)&(t<=26+1e-8)) if self.fit_mask is None else np.asarray(self.fit_mask,bool).copy()
        if self.fit_mask.shape!=(len(t),) or not self.fit_mask.any():raise ValueError('fit_mask must select at least one time sample')
        if self.fit_mask[self.anchor]:raise ValueError('Exclude the deterministically normalized t=0 sample from the likelihood')
        m=int(self.fit_mask.sum())
        if self.covariance is None:
            self.covariance=np.stack([np.eye(m)*s*s for s in np.repeat(self.sigma,len(self.gaussians))])
        else:
            self.covariance=np.asarray(self.covariance,float)
            if self.covariance.shape not in [(c,m,m),(c*m,c*m)]:raise ValueError('Covariance shape must match selected RF channels and samples')
            if not np.isfinite(self.covariance).all() or not np.allclose(self.covariance,self.covariance.swapaxes(-1,-2)):raise ValueError('Covariance must be finite and symmetric')
            try:np.linalg.cholesky(self.covariance)
            except np.linalg.LinAlgError as e:raise ValueError('Covariance must be positive definite') from e

    @property
    def time(self):return self.t0+np.arange(self.rf.shape[-1])*self.dt

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            keys=set(z.files)
            t=z['time'] if 'time' in keys else np.arange(z['rf'].shape[-1])*.2-2
            return cls(z['rf'],z['p'],tuple(z['gaussians']) if 'gaussians' in keys else (2.5,),
                       float(t[1]-t[0]),float(t[0]),z['sigma'] if 'sigma' in keys else .03,
                       z['p_sd'] if 'p_sd' in keys else 0.,z['fit_mask'] if 'fit_mask' in keys else None,
                       z['covariance'] if 'covariance' in keys else z['cov'] if 'cov' in keys else None,
                       int(z['nfft']) if 'nfft' in keys else 512)

    def save(self,path):
        np.savez_compressed(path,rf=self.rf,p=self.p,gaussians=self.gaussians,time=self.time,
                            sigma=self.sigma,p_sd=self.p_sd,fit_mask=self.fit_mask,covariance=self.covariance,nfft=self.nfft)

    def require_neural_grid(self):
        if self.rf.shape!=(9,151) or len(self.p)!=3 or not np.allclose(self.gaussians,[1,2.5,4]) or not np.allclose(self.time,np.arange(151)*.2-2):
            raise ValueError('Released neural models require 3 slowness bins x widths [1,2.5,4], 151 samples at -2:.2:28 s. Use Sobol initialization for other inputs; do not invent missing bands.')
