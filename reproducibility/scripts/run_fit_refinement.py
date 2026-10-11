# Our CPU research program; ported imports use the public GRTM RF preview.
"""Freeze a refinement protocol before fitting the non-development stations."""
import argparse, hashlib, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/fit-refinement-v3'

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--stations', nargs='+')
    a = p.parse_args()
    OUT.mkdir(exist_ok=True, parents=True)
    script = ROOT / 'scripts/fit_refinement.py'
    manifest = json.loads((ROOT / 'data/processed/manifest.json').read_text())
    stations = a.stations or [m['station'] for m in manifest if m['status'] == 'ready']
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), seed=728, starts=16, maxiter=300, development_stations=['U23A', 'X26A', 'W19A', 'Z19A', 'Z20A'], model_candidates=['legacy', 'timing', 'layer3', 'layer5', 'flex5'], likelihood_candidates=['full fixed covariance', 'diagonal composite with power 0.2'], complexity_selection='inner validation RMSE; lower dimension within 2% of minimum', inner_split='every fifth training event within its original slowness bin', score_window_s=[0.8, 26.0], outer_split='unchanged from original manifest', inner_initialization='Sobol plus fixed reference model [1.5,2.4,40,1.78]; no parameters learned from inner validation', note='Five stations with inspected exploratory output are excluded from primary post-refinement validation statistics; all 42 are shown. Model/likelihood selection is not uncertainty calibration.', script_sha256=hashlib.sha256(script.read_bytes()).hexdigest())
    frozen = OUT / 'protocol.json'
    if frozen.exists():
        prior = json.loads(frozen.read_text())
        if prior['script_sha256'] != protocol['script_sha256']:
            raise RuntimeError('Script changed after protocol freeze; choose a new version/output directory')
    else:
        frozen.write_text(json.dumps(protocol, indent=2))

    def run(station):
        target = OUT / f'{station}.json'
        if target.exists():
            old = json.loads(target.read_text())
            if old['script_sha256'] != protocol['script_sha256']:
                raise RuntimeError(f'Stale station result: {station}')
            return dict(station=station, returncode=0, reused=True)
        with (OUT / f'{station}.log').open('w') as f:
            r = subprocess.run([sys.executable, str(script), station, '--starts', '16', '--maxiter', '300'], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        print(station, r.returncode, flush=True)
        return dict(station=station, returncode=r.returncode)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(run, stations))
    (OUT / 'cohort-status.json').write_text(json.dumps(results, indent=2))
    if any((r['returncode'] != 0 for r in results)):
        raise RuntimeError('One or more station fits failed; inspect station logs')
if __name__ == '__main__':
    main()
