"""Fetch only declared observational archives; never extract or execute code."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--station',action='append');a=ap.parse_args()
    rows=json.loads((ROOT/'data/processed/manifest.json').read_text())
    out=ROOT/'data/source';out.mkdir(parents=True,exist_ok=True)
    for r in rows:
        if r['status']!='ready' or a.station and r['station'] not in a.station:continue
        p=out/('ears-'+r['station']+'.zip')
        if not p.exists():
            with urllib.request.urlopen(r['zip_url'],timeout=120) as response:blob=response.read()
            if hashlib.sha256(blob).hexdigest()!=r['sha256']:raise ValueError('Archive changed: '+r['station'])
            p.write_bytes(blob)
        if hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('SHA mismatch: '+r['station'])
        print(r['station'],'verified')
if __name__=='__main__':main()
