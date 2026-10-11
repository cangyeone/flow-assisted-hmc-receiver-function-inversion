import hashlib
import json
from pathlib import Path
import numpy as np
import torch

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def plain(v):
    if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    if isinstance(v,(np.ndarray,torch.Tensor)):return plain(v.tolist())
    if isinstance(v,(np.integer,np.floating)):return plain(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    if isinstance(v,Path):return str(v)
    return v

def write_json(path,value):
    Path(path).write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n')

def backend():
    try:
        from grtm import JointRFForward
        from grtm.rf_hmc import hmc_sample,rank_diagnostics
        from grtm.rf_bayes import dram_sample
    except ImportError as e:
        raise ImportError('CPU differentiable RF requires the GRTM community RF preview. Run: flowrf install-grtm. See docs/installation.md.') from e
    return JointRFForward,hmc_sample,dram_sample,rank_diagnostics
