"""Import result arrays, never a third-party program, archive or executable."""
import argparse
from pathlib import Path
import numpy as np
from flowrf._utils import sha256,write_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('samples',help='NPZ with samples [draw,chain,parameter]');ap.add_argument('--output',required=True);ap.add_argument('--citation',required=True);ap.add_argument('--parameter-names',nargs='+',required=True);a=ap.parse_args()
    with np.load(a.samples,allow_pickle=False) as z:x=z['samples']
    if x.ndim!=3 or x.shape[-1]!=len(a.parameter_names) or not np.isfinite(x).all():raise ValueError('Declared parameter order and finite [draw,chain,parameter] samples required')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'samples.npz',samples=x)
    write_json(out/'provenance.json',dict(citation=a.citation,source_array_sha256=sha256(a.samples),parameter_names=a.parameter_names,scope='Externally obtained result arrays only. Check matching target, parameterization, units, prior, likelihood, data, warmup and timing before comparisons. No third-party code imported.'))
if __name__=='__main__':main()
