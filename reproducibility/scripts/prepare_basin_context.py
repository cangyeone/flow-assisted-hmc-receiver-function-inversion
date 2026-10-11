# Our CPU research program; ported imports use the public GRTM RF preview.
"""Reproduce regional USGS basin geometry and point memberships (NAD83)."""
import hashlib, io, json, urllib.request, zipfile
from pathlib import Path
import shapefile
from matplotlib.path import Path as MPath
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/basin-context'
URL = 'https://pubs.usgs.gov/of/2012/1111/data/Sedimentary_Basins.zip'

def main():
    OUT.mkdir(exist_ok=True, parents=True)
    archive = OUT / 'Sedimentary_Basins.zip'
    if not archive.exists():
        archive.write_bytes(urllib.request.urlopen(urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0'}), timeout=30).read())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert digest == '161ab957eb86bf1323e9c35c52762b512da0d233ea9c609710f7d8d61de69579', 'USGS archive changed; review before replacing geometry'
    with zipfile.ZipFile(archive) as z:
        name = 'Sedimentary_Basins_of_the_United_States'
        reader = shapefile.Reader(shp=io.BytesIO(z.read(name + '.shp')), shx=io.BytesIO(z.read(name + '.shx')), dbf=io.BytesIO(z.read(name + '.dbf')))
        features = [dict(type='Feature', geometry=s.shape.__geo_interface__, properties=s.record.as_dict()) for s in reader.iterShapeRecords() if s.shape.bbox[0] < -103 and s.shape.bbox[2] > -110 and (s.shape.bbox[1] < 39) and (s.shape.bbox[3] > 33)]
        projection = z.read(name + '.prj').decode()
    (OUT / 'usgs-regional-basins.geojson').write_text(json.dumps(dict(type='FeatureCollection', features=features)))
    rows = []
    for m in json.loads((ROOT / 'data/processed/manifest.json').read_text()):
        p = [m['longitude'], m['latitude']]
        names = []
        for f in features:
            g = f['geometry']
            polys = [g['coordinates']] if g['type'] == 'Polygon' else g['coordinates']
            if any((MPath(poly[0]).contains_point(p) and (not any((MPath(hole).contains_point(p) for hole in poly[1:]))) for poly in polys)):
                names.append(f['properties']['Name'])
        rows.append(dict(station=m['station'], latitude=m['latitude'], longitude=m['longitude'], basins=names))
    (OUT / 'station-context.json').write_text(json.dumps(rows, indent=2))
    (OUT / 'source.json').write_text(json.dumps(dict(source=URL, documentation='https://pubs.usgs.gov/of/2012/1111/Sedimentary_Basins_of_the_United_States.html', sha256=digest, projection=projection, scope='Reconnaissance sedimentary-basin polygons at regional scale; NAD83 geographical degrees; membership is not a local sediment-thickness measurement'), indent=2))
    print(len(features), 'regional polygons;', sum((bool(r['basins']) for r in rows)), 'member stations')
if __name__ == '__main__':
    main()
