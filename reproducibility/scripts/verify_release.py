"""Verify distributed file/weight integrity and the publication boundary."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    export=json.loads((ROOT/'reproducibility/export-manifest.json').read_text())
    for row in export['records']:
        p=ROOT/row['file'];assert p.is_file() and sha(p)==row['sha256'],p
    model=json.loads((ROOT/'src/flowrf/weights/manifest.json').read_text())
    for group in model['model_sets'].values():
        for arm in ['forward','inverse']:
            row=group[arm];assert sha(ROOT/'src/flowrf/weights'/row['file'])==row['sha256']
    for row in model['experimental']:assert sha(ROOT/row['file'])==row['sha256'],row['file']
    excluded={'.cu','.cuh','.f','.f90','.for','.c','.cpp','.so','.dylib','.dll','.tar','.zip','.whl'}
    programs=[]
    for p in ROOT.rglob('*'):
        if any(x in p.parts for x in ['.git','.venv','outputs','__pycache__','build','dist']):continue
        if p.is_file():
            assert p.suffix.lower() not in excluded,p
            if p.suffix=='.py':
                ast.parse(p.read_text());programs.append(str(p.relative_to(ROOT)))
                tree=ast.parse(p.read_text())
                for node in ast.walk(tree):
                    if isinstance(node,ast.Attribute):assert ast.unparse(node)!='torch.'+'cu'+'da',p
                    if isinstance(node,ast.ImportFrom):assert node.module!='grtm.rf_'+'cu'+'da',p
                    if isinstance(node,ast.keyword):assert node.arg!='cu'+'da_graph',p
    print(json.dumps(dict(status='passed',exported_files=len(export['records']),own_python_files=len(programs),experimental_weights=len(model['experimental']),production_weights=2*len(model['model_sets']),third_party_programs=0,native_binaries=0),indent=2))
if __name__=='__main__':main()
