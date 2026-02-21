import csv
from pathlib import Path
p=Path(r"A:\omni read\temps\analysis_report.csv")
rows=[]
with p.open('r',encoding='utf-8') as f:
    r=csv.DictReader(f)
    for it in r:
        it['nonzero_ratio']=float(it['nonzero_ratio'])
        it['bbox_frac']=float(it['bbox_frac'])
        it['shape_h']=int(it['shape_h'])
        it['shape_w']=int(it['shape_w'])
        rows.append(it)
bad=[]
edge=[]
for it in rows:
    bbox=it['bbox']
    h=it['shape_h']; w=it['shape_w']
    bx1=by1=bx2=by2=None
    if bbox:
        s=bbox.strip('()')
        parts=[int(x.strip()) for x in s.split(',')]
        bx1,by1,bx2,by2=parts
    if it['nonzero_ratio']>0.18 or it['bbox_frac']>0.20:
        bad.append(it['mask'])
    if bx1 is not None and (bx1==0 or by1==0 or bx2==w or by2==h):
        edge.append(it['mask'])
print(f"Total pairs: {len(rows)}")
print(f"Likely bad (>0.18 ratio or bbox_frac>0.20): {len(bad)}")
for m in bad:
    print(' -', m)
print('\nMasks touching edges:')
for m in edge:
    print(' -', m)
