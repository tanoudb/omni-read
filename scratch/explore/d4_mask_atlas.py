# -*- coding: utf-8 -*-
"""D4 : cartographie des sources de masque. Mesure seule, aucune ecriture."""
import sys, pickle, json
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np, cv2
from pipeline import TranslationPipeline as TP
from core.renderer import TextRenderer as TR
from core.bubble_shape import ink_mask_from_regions

CACHE = Path('scratch/bareme/cache')

def poly_mask(regions, h, w):
    m = np.zeros((h, w), np.uint8)
    for r in regions or []:
        pts = r.get('bbox') if isinstance(r, dict) else None
        if not pts or len(pts) < 3: continue
        a = np.array([[int(p[0]), int(p[1])] for p in pts], np.int32)
        a[:,0] = np.clip(a[:,0], 0, w-1); a[:,1] = np.clip(a[:,1], 0, h-1)
        cv2.fillPoly(m, [a], 255)
    return m

def nz(m): return int(np.count_nonzero(m))

def cov(a, b):
    """part de b couverte par a"""
    nb = nz(b)
    return (nz(cv2.bitwise_and(a, b)) / nb) if nb else float('nan')

def rebuild_chir(crop, regions, mask_binary, class_name):
    """Refait pas a pas la chaine de pipeline._build_masks_for_detection."""
    h, w = crop.shape[:2]
    steps = {}
    steps['polygons'] = poly_mask(regions, h, w)
    ocr = TP._ocr_mask_from_regions(regions, h, w, crop_bgr=crop)
    if ocr is None: return None, steps
    steps['ocr_mask'] = ocr.copy()
    m = ocr
    steps['interior_applied'] = False
    if str(class_name).lower() != 'out_text':
        try: interior = TR._bubble_mask_from_image(crop)
        except Exception: interior = None
        if interior is not None and interior.shape[:2] == (h, w):
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
            interior = cv2.erode(interior, k, 1)
            steps['interior'] = interior
            bounded = cv2.bitwise_and(m, interior)
            if nz(bounded) >= 0.5*nz(m):
                m = bounded; steps['interior_applied'] = True
    steps['after_interior'] = m.copy()
    steps['bubble_applied'] = False
    bub = mask_binary
    if bub is not None:
        if bub.ndim == 3: bub = bub[:,:,0]
        if bub.shape[:2] != (h, w): bub = cv2.resize(bub, (w,h), interpolation=cv2.INTER_NEAREST)
        bub = (bub > 0).astype(np.uint8)*255
        stroke = max(3, int(round(min(h,w)*0.025)))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*stroke+1, 2*stroke+1))
        er = cv2.erode(bub, ker, 1)
        if nz(er) >= 0.35*nz(bub): bub = er
        steps['bubble_eroded'] = bub
        inter = cv2.bitwise_and(m, bub)
        if int(np.sum(inter)) > 0.30*int(np.sum(m)):
            m = inter; steps['bubble_applied'] = True
    steps['after_bubble'] = m.copy()
    chir = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    steps['chirurgical'] = chir
    return chir, steps

def xprofile(m, nb=8):
    w = m.shape[1]; out=[]
    for i in range(nb):
        a, b = i*w//nb, (i+1)*w//nb
        out.append(nz(m[:, a:b]))
    return out

def main():
    focus = sys.argv[1] if len(sys.argv)>1 else None
    rows = []
    for d in sorted(CACHE.iterdir()):
        pk = d/'dets.pkl'; pg = d/'page.png'
        if not pk.exists() or not pg.exists(): continue
        data = pickle.load(open(pk,'rb'))
        page = cv2.imread(str(pg))
        for i, it in enumerate(data['items']):
            key = f"{d.name}#{i}"
            if focus and focus not in key: continue
            x1,y1,x2,y2 = it['bbox']
            crop = page[max(0,y1):y2, max(0,x1):x2]
            if crop.size == 0: continue
            h,w = crop.shape[:2]
            regions = it.get('text_regions') or []
            cn = it.get('class_name','')
            chir, st = rebuild_chir(crop, regions, it.get('mask_binary'), cn)
            if chir is None: continue
            ink = ink_mask_from_regions(crop, regions)
            cached = it.get('chirurgical_mask')
            blk = TR._block_mask_from_regions(w, h, regions, 0, 0)
            row = dict(key=key, cls=cn, h=h, w=w, area=h*w,
                pol=nz(st['polygons']), ocr=nz(st['ocr_mask']), ink=nz(ink),
                chir=nz(chir), cached=nz(cached) if cached is not None else -1,
                mb=nz(it['mask_binary']) if it.get('mask_binary') is not None else -1,
                blk=nz(blk) if blk is not None else -1,
                int_app=st['interior_applied'], bub_app=st['bubble_applied'],
                cov_ocr_ink=round(cov(st['ocr_mask'], ink),3),
                cov_chir_ink=round(cov(chir, ink),3),
                cov_pol_ink=round(cov(st['polygons'], ink),3),
                cov_blk_ink=round(cov(blk, ink),3) if blk is not None else None,
                cov_mb_ink=round(cov(it['mask_binary'], ink),3) if it.get('mask_binary') is not None else None,
                cov_chir_ocr=round(cov(chir, st['ocr_mask']),3),
                agree_cached=round(cov(cached, chir),3) if cached is not None else None,
            )
            rows.append(row)
            if focus:
                print(json.dumps(row, indent=1, default=str))
                print('x-profil (8 tranches) sur', w, 'px de large')
                for name in ('polygons','ocr_mask','after_interior','after_bubble'):
                    print(f'  {name:16s}', xprofile(st[name]))
                print(f'  {"ink(otsu)":16s}', xprofile(ink))
                print(f'  {"cached_chir":16s}', xprofile(cached))
                print(f'  {"mask_binary":16s}', xprofile(it['mask_binary']))
                if 'interior' in st: print(f'  {"interior(7er)":16s}', xprofile(st['interior']))
                if 'bubble_eroded' in st: print(f'  {"bubble_eroded":16s}', xprofile(st['bubble_eroded']))
                if blk is not None: print(f'  {"block_mask":16s}', xprofile(blk))
    if not focus:
        import csv
        out = Path('scratch/explore/d4_mask_atlas.csv')
        with open(out,'w',newline='',encoding='utf8') as f:
            wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wcsv.writeheader(); wcsv.writerows(rows)
        print('n =', len(rows), '->', out)
main()
