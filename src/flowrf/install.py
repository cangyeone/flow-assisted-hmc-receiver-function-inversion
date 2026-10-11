"""Install a pinned, hash-checked external GRTM community wheel."""
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

VERSION='0.6.0.dev2'
BASE='https://github.com/cangyeone/seismicx-grtm-community/releases/download/v'+VERSION

def install_grtm():
    tags={('Darwin','arm64',(3,12)):'cp312-cp312-macosx_11_0_arm64',('Linux','x86_64',(3,13)):'cp313-cp313-manylinux_2_28_x86_64'}
    key=(platform.system(),platform.machine(),sys.version_info[:2])
    if key not in tags:
        raise RuntimeError('GRTM differentiable RF preview currently supports macOS arm64/Python 3.12 and Linux x86_64/Python 3.13. Neural-only tools work on other platforms. See docs/installation.md.')
    name='seismicx_grtm-'+VERSION+'-'+tags[key]+'.whl'
    def get(url):
        req=urllib.request.Request(url,headers={'User-Agent':'FlowRF/0.1.0'})
        with urllib.request.urlopen(req,timeout=120) as r:return r.read()
    # Pin hashes in our release, rather than trusting a second mutable download.
    hashes=json.loads((Path(__file__).parent/'weights'/'grtm-release.json').read_text())
    blob=get(BASE+'/'+name)
    if hashlib.sha256(blob).hexdigest()!=hashes['sha256'][name]:raise RuntimeError('GRTM wheel SHA-256 mismatch')
    with tempfile.TemporaryDirectory(prefix='flowrf-grtm-') as tmp:
        path=Path(tmp)/name;path.write_bytes(blob)
        subprocess.run([sys.executable,'-m','pip','install',str(path)],check=True)
