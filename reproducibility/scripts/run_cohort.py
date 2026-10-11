# Our CPU research program; ported imports use the public GRTM RF preview.
"""Bounded station-level CPU jobs; one torch thread per station process."""
import argparse, json, subprocess, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--partition', type=int, default=0)
p.add_argument('--partitions', type=int, default=2)
p.add_argument('--workers', type=int, default=6)
p.add_argument('--stations', nargs='+')
a = p.parse_args()
manifest = json.loads((root / 'data/processed/manifest.json').read_text())
stations = a.stations or [x['station'] for x in manifest if x['status'] == 'ready'][a.partition::a.partitions]

def run(sta):
    with (root / 'results' / f'{sta}.log').open('w') as f:
        r = subprocess.run([sys.executable, str(root / 'scripts/cpu_hmc.py'), sta, '--steps', '4000', '--burn', '1500', '--starts', '32'], cwd=root, stdout=f, stderr=subprocess.STDOUT)
    print(sta, r.returncode, flush=True)
    return {'station': sta, 'returncode': r.returncode}
with ThreadPoolExecutor(max_workers=a.workers) as pool:
    results = list(pool.map(run, stations))
(root / 'results' / ('cohort-reassigned.json' if a.stations else f'cohort-{a.partition}.json')).write_text(json.dumps(results, indent=2))
