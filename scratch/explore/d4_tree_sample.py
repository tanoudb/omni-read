# -*- coding: utf-8 -*-
"""D4 : recensement des branches de inpaint_region + attribution des pertes de masque.
Rejoue la logique de inpaint_region SANS LaMa (aucun GPU) jusqu'au point de decision."""
import sys, pickle, csv
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np, cv2
from pipeline import TranslationPipeline as TP
from core.renderer import TextRenderer as TR
from core.bubble_shape import ink_mask_from_regions
from config import config

CACHE = Path('scratch/bareme/cache')
CROP_MARGIN, MIN_H = TR.CROP_MARGIN, TR.INPAINT_MIN_HEIGHT
nz = lambda m: int(np.count_nonzero(m))

def branch(page, it):
    x1,y1,x2,y2 = it['bbox']; cn = str(it.get('class_name','')).lower()
    regions = it.get('text_regions') or []
    chir = it.get('chirurgical_mask'); bub = it.get('mask_binary')
    H,W = page.shape[:2]; bh = y2-y1
    r = dict(cls=cn)
    if bh < MIN_H: return dict(r, branch='skip_petit')
    if not regions and chir is None: return dict(r, branch='skip_sans_masque')
    m = max(CROP_MARGIN*2, 2*bh) if cn in ('out_text','system') else max(CROP_MARGIN, bh)
    cx1,cy1 = max(0,x1-m), max(0,y1-m); cx2,cy2 = min(W,x2+m), min(H,y2+m)
    crop = page[cy1:cy2, cx1:cx2].copy(); ch,cw = crop.shape[:2]
    ox,oy = x1-cx1, y1-cy1
    blk = TR._block_mask_from_regions(cw, ch, regions, ox, oy) if regions else None
    use_block = cn in ('out_text','system')
    if use_block and blk is not None and nz(blk)>0:
        local, src = blk, 'block_mask'
    elif chir is not None and chir.size>0:
        dh,dw = max(1,y2-y1), max(1,x2-x1)
        cm = chir if chir.shape[:2]==(dh,dw) else cv2.resize(chir,(dw,dh),interpolation=cv2.INTER_NEAREST)
        local = np.zeros((ch,cw),np.uint8); local[oy:oy+dh, ox:ox+dw] = (cm>0).astype(np.uint8)*255
        k = config.rendering.out_text_mask_dilate_kernel if cn in ('out_text','system') else config.rendering.inpaint_mask_dilate_kernel
        if k and k>1:
            local = cv2.dilate(local, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)),1)
        src = 'chirurgical'
    elif regions:
        local = TR._build_local_mask_from_regions(TR, cw, ch, regions)
        sh = np.zeros_like(local)
        for reg in regions:
            pts = reg.get('bbox')
            if not pts: continue
            a = np.array([[max(0,min(int(p[0])+ox,cw-1)), max(0,min(int(p[1])+oy,ch-1))] for p in pts], np.int32)
            if a.shape[0]>=3: cv2.fillPoly(sh,[a],255)
        local = cv2.dilate(sh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)),1)
        src = 'polygones'
    else: return dict(r, branch='skip_sans_masque')
    if np.sum(local)==0: return dict(r, branch='skip_masque_vide', src=src)
    # local_bubble
    lb = None
    if bub is not None and bub.size>0:
        dh,dw = max(1,y2-y1), max(1,x2-x1)
        bm = bub if bub.shape[:2]==(dh,dw) else cv2.resize(bub,(dw,dh),interpolation=cv2.INTER_NEAREST)
        lb = np.zeros((ch,cw),np.uint8); lb[oy:oy+dh, ox:ox+dw] = (bm>0).astype(np.uint8)*255
    # fill_limit
    fl_state='n/a_out_text'
    if cn != 'out_text':
        dh,dw = max(1,y2-y1), max(1,x2-x1)
        interior = TR._bubble_mask_from_image(page[max(0,y1):y2, max(0,x1):x2])
        if interior is None or interior.shape[:2]!=(dh,dw): fl_state='pas_de_forme'
        else:
            interior = cv2.erode(interior, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)),1)
            lim = np.full((ch,cw),255,np.uint8); lim[oy:oy+dh, ox:ox+dw] = interior
            fl_state = 'actif' if nz(cv2.bitwise_and(local,lim)) >= 0.9*nz(local) else 'rejete_90pct'
    flat = TR._flat_fill_color(crop, local, local_bubble_mask=lb, class_name=cn)
    if flat is not None: return dict(r, branch='flat_fill', src=src, fill_limit=fl_state,
                                    mask_frac=round(nz(local)/(ch*cw),4))
    smooth = TR._smooth_fill(crop, local)
    if smooth is not None: return dict(r, branch='smooth_fill', src=src, fill_limit=fl_state,
                                       mask_frac=round(nz(local)/(ch*cw),4))
    return dict(r, branch='lama', src=src, fill_limit=fl_state, mask_frac=round(nz(local)/(ch*cw),4),
                blk_dispo=blk is not None)

rows=[]
for d in sorted(CACHE.iterdir()):
    pk, pg = d/'dets.pkl', d/'page.png'
    if not (pk.exists() and pg.exists()): continue
    data = pickle.load(open(pk,'rb')); page = cv2.imread(str(pg))
    for i,it in enumerate(data['items']):
        if i%4: continue
        try: rr = branch(page, it)
        except Exception as e: rr = dict(cls=it.get('class_name',''), branch='ERREUR:'+type(e).__name__+str(e)[:60])
        rr["key"] = f"{d.name}#{i}"; rows.append(rr)
    print("  ", d.name, len(rows), flush=True)
keys = ['key','cls','branch','src','fill_limit','mask_frac','blk_dispo']
with open('scratch/explore/d4_tree_sample.csv','w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
from collections import Counter
print('n=',len(rows))
print('BRANCHE :', Counter(r['branch'] for r in rows).most_common())
print('SOURCE  :', Counter(r.get('src') for r in rows).most_common())
print('FILL_LIM:', Counter(r.get('fill_limit') for r in rows).most_common())
print()
for cl in ('bulle','out_text','system'):
    sub=[r for r in rows if r['cls']==cl]
    print(f'{cl:9s} n={len(sub):4d}', Counter(r['branch'] for r in sub).most_common(), Counter(r.get('src') for r in sub).most_common())
