# Our CPU research program; ported imports use the public GRTM RF preview.
"""Download a geographically selected EARS cohort; never execute archive code."""
import csv, io, json, math, hashlib, re, zipfile
from pathlib import Path
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor
import requests
import numpy as np
from obspy import read
from scipy.signal import resample_poly
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/source'
OUT = ROOT / 'data/processed'
OUT.mkdir(exist_ok=True, parents=True)
INDEX = 'https://data.unavco.org/archive/seismology/products/ears/summary/TA/X26A/downloaded_content/station_content/stationList_netCode_TA_gaussian_2.5.html'
SOURCE.mkdir(exist_ok=True, parents=True)
for filename, url in [('ears-ta-stations.html', INDEX), ('ears-ta-stations.csv', urljoin(INDEX, 'stationList_netCode_TA_gaussian_2.5_content/stationList_netCode_TA_gaussian_2.5.csv'))]:
    if not (SOURCE / filename).exists():
        req = requests.get(url, timeout=60)
        req.raise_for_status()
        (SOURCE / filename).write_text(req.text)
html = (SOURCE / 'ears-ta-stations.html').read_text()
links = {x: urljoin(INDEX, h) for h, x in re.findall('<a href="([^"]+)">([A-Z0-9]+)</a>', html)}
rows = list(csv.reader((SOURCE / 'ears-ta-stations.csv').open()))[2:]
eligible = [r for r in rows if 33 <= float(r[2]) < 39 and -110 <= float(r[3]) < -103 and (int(r[12]) >= 25)]
cells = {}
for r in eligible:
    cells.setdefault((math.floor(float(r[2])), math.floor(float(r[3]))), []).append(r)
chosen = [min(v, key=lambda r: ((float(r[2]) - k[0] - 0.5) ** 2 + (float(r[3]) - k[1] - 0.5) ** 2, r[1])) for k, v in sorted(cells.items())]
protocol = {'selection': '33<=lat<39, -110<=lon<-103; archive count>=25; closest station to each 1-degree cell centre, station-code tie break', 'eligible_count': len(eligible), 'selected_count': len(chosen), 'selected_stations': [r[1] for r in chosen], 'waveforms': 'EARS radial iterative RF, gaussian=2.5, archive match>=80', 'processing': 'resample to 5 Hz; arrival time b-a; normalize by interpolated t=0 radial amplitude (>0.05); p in [0.04,0.085]; 3 equal-count p bins', 'split': 'sort event filename within each p bin; every fourth event held out; at least 6 training and 2 test events/bin', 'likelihood': 'fit 0.8 to 26 s inclusive; empirical training-mean covariance shrunk 25% toward diagonal, plus fixed 0.03-amplitude Gaussian-correlated discrepancy (0.5 s scale) and 0.005 diagonal floor', 'created_utc': '2026-10-09', 'index_url': INDEX}
(OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2))

def process(r):
    sta = r[1]
    meta = {'station': sta, 'latitude': float(r[2]), 'longitude': float(r[3]), 'archive_count': int(r[12]), 'page_url': links[sta]}
    try:
        page = SOURCE / f'ears-{sta}.html'
        if not page.exists():
            req = requests.get(links[sta], timeout=45)
            req.raise_for_status()
            page.write_text(req.text)
        href = next((h for h in re.findall('href="([^"]+)"', page.read_text()) if 'receiverFunction_' in h and '.zip' in h))
        url = urljoin(links[sta], href)
        path = SOURCE / f'ears-{sta}.zip'
        if not path.exists():
            req = requests.get(url, timeout=90)
            req.raise_for_status()
            path.write_bytes(req.content)
        blob = path.read_bytes()
        meta.update(zip_url=url, sha256=hashlib.sha256(blob).hexdigest())
        traces = []
        rejected = []
        time = np.arange(-2, 28.01, 0.2)
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            for name in sorted(z.namelist()):
                if not name.endswith('.itr'):
                    continue
                tr = read(io.BytesIO(z.read(name)), format='SAC')[0]
                s = tr.stats.sac
                p = float(s.user2)
                a = float(s.a)
                b = float(s.b)
                dt = float(tr.stats.delta)
                y = resample_poly(tr.data.astype(float), 1, round(0.2 / dt))
                t = b - a + np.arange(len(y)) * 0.2
                amp = np.interp(0, t, y)
                if not (0.04 <= p <= 0.085 and amp > 0.05 and np.isfinite(y).all() and (abs(float(s.user1) - 2.5) < 1e-05)):
                    rejected.append(name)
                    continue
                yi = np.interp(time, t, y) / amp
                if np.max(np.abs(yi)) > 8:
                    rejected.append(name)
                    continue
                traces.append((p, name, yi, float(s.baz)))
        traces.sort(key=lambda a: (a[0], a[1]))
        bins = np.array_split(np.arange(len(traces)), 3)
        ys = []
        tests = []
        ps = []
        ptest = []
        counts = []
        testcounts = []
        covs = []
        events = []
        mask = (time >= 0.799) & (time <= 26.001)
        tt = time[mask]
        corr = np.exp(-0.5 * ((tt[:, None] - tt[None, :]) / 0.5) ** 2)
        for idx in bins:
            group = sorted([traces[i] for i in idx], key=lambda a: a[1])
            test = [a for i, a in enumerate(group) if i % 4 == 0]
            train = [a for i, a in enumerate(group) if i % 4 != 0]
            if len(train) < 6 or len(test) < 2:
                raise ValueError('insufficient events in a slowness bin')
            ar = np.stack([a[2] for a in train])
            te = np.stack([a[2] for a in test])
            c = np.cov(ar[:, mask], rowvar=False) / len(train)
            c = 0.75 * c + 0.25 * np.diag(np.diag(c)) + 0.03 ** 2 * corr + 0.005 ** 2 * np.eye(len(tt))
            ys.append(ar.mean(0))
            tests.append(te.mean(0))
            ps.append(np.mean([a[0] for a in train]))
            ptest.append(np.mean([a[0] for a in test]))
            covs.append(c)
            counts.append(len(train))
            testcounts.append(len(test))
            events.extend(({'member': a[1], 'p': a[0], 'baz': a[3], 'split': 'test' if i % 4 == 0 else 'train'} for i, a in enumerate(group)))
        np.savez_compressed(OUT / f'{sta}.npz', time=time, fit_mask=mask, rf=ys, test_rf=tests, p=ps, test_p=ptest, cov=covs, counts=counts, test_counts=testcounts)
        meta.update(status='ready', accepted=len(traces), rejected=rejected, train_counts=counts, test_counts=testcounts, events=events)
    except Exception as e:
        meta.update(status='excluded', reason=str(e))
    print(sta, meta['status'], meta.get('accepted', meta.get('reason')), flush=True)
    return meta
if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(process, chosen))
    (OUT / 'manifest.json').write_text(json.dumps(result, indent=2))
    print('Ready', sum((x['status'] == 'ready' for x in result)), 'of', len(result))
