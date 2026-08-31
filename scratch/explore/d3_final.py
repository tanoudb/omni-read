"""D3 — comparaison en conditions de PRODUCTION.

inpaint_region n'utilise pas le masque brut : il le DILATE (kernel 7 pour
`bulle`, 11 pour `out_text`/`System`). On compare donc les masques reellement
envoyes a l'effacement :

  CHd   = dilate(chirurgical_mask, k)         -> production actuelle
  OCRd  = dilate(_ocr_mask_from_regions, k)   -> "polygones + encre" sans garde-fous
  FULL  = dilate(fillPoly(polygones), 11)     -> branche de repli REELLE si on
                                                 supprime chirurgical_mask

Metriques INDEPENDANTES du `_bubble_mask_from_image` qui produit le rognage
(sinon la mesure est circulaire) :
  - glyphes residuels : composantes d'encre non effacees, taille de glyphe
  - structure detruite : px effaces, hors encre et hors halo, tombant sur un
    bord Canny (= trait de ballon ou dessin)
"""
import sys, pickle, os, json
sys.path.insert(0, '.')
import numpy as np
import cv2
from pipeline import TranslationPipeline
from core.renderer import TextRenderer
sys.path.insert(0, 'scratch/explore')
from d3_audit import poly_union, full_poly_mask, ink_reference, nz

CACHE = 'scratch/bareme/cache'
OUT = 'scratch/explore/d3_final.json'


def dilate(m, k):
    if m is None or k <= 1:
        return m
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def glyph_residuals(ink, mask, min_area=40, min_side=6):
    """Composantes d'encre NON effacees de taille de glyphe."""
    left = ((ink > 0) & (mask == 0)).astype(np.uint8)
    if left.sum() == 0:
        return 0, 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(left, 8)
    cnt, area = 0, 0
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a >= min_area and w >= min_side and h >= min_side:
            cnt += 1
            area += int(a)
    return cnt, area


def run():
    rows = []
    for page in sorted(os.listdir(CACHE)):
        pk = os.path.join(CACHE, page, 'dets.pkl')
        im = os.path.join(CACHE, page, 'page.png')
        if not (os.path.exists(pk) and os.path.exists(im)):
            continue
        d = pickle.load(open(pk, 'rb'))
        img = cv2.imread(im)
        for idx, it in enumerate(d['items']):
            x1, y1, x2, y2 = it['bbox']
            h, w = max(1, y2 - y1), max(1, x2 - x1)
            crop = img[max(0, y1):y2, max(0, x1):x2]
            if crop.shape[0] != h or crop.shape[1] != w:
                continue
            regions = it.get('text_regions')
            pu = poly_union(regions, h, w)
            cls = str(it.get('class_name')).lower()
            row = {'page': page, 'idx': idx, 'cls': cls, 'w': w, 'h': h,
                   'text': (it.get('text') or '')[:50]}
            if nz(pu) == 0:
                row['skip'] = 'no_poly'
                rows.append(row)
                continue
            search = cv2.dilate(pu, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
            ink = ink_reference(crop, search)
            halo = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            edges = cv2.Canny(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 60, 160)
            k = 11 if cls in ('out_text', 'system') else 7

            ocr = TranslationPipeline._ocr_mask_from_regions(regions, h, w, crop_bgr=crop)
            ch = it.get('chirurgical_mask')
            if ch is not None and ch.shape[:2] != (h, w):
                ch = cv2.resize(ch, (w, h), interpolation=cv2.INTER_NEAREST)
            full = full_poly_mask(regions, h, w)

            variants = {'CHd': dilate(ch, k), 'OCRd': dilate(ocr, k), 'FULL': full}
            inkn = nz(ink)
            row['ink_nz'] = inkn
            row['area'] = h * w
            # le masque au BLOC est-il disponible ? (il prime pour out_text/System)
            row['block_ok'] = bool(TextRenderer._block_mask_from_regions(w, h, regions, 0, 0) is not None)
            for name, m in variants.items():
                if m is None:
                    row[name] = None
                    continue
                mb = m > 0
                gc, ga = glyph_residuals(ink, m)
                spill = mb & (halo == 0)
                row[name] = {
                    'cover': round(100.0 * int((mb & (ink > 0)).sum()) / max(1, inkn), 1),
                    'area_pct': round(100.0 * int(mb.sum()) / (h * w), 1),
                    'res_comp': gc, 'res_area': ga,
                    'res_pct': round(100.0 * ga / max(1, inkn), 1),
                    'spill_pct': round(100.0 * int(spill.sum()) / (h * w), 1),
                    'struct_px': int((spill & (edges > 0)).sum()),
                }
            rows.append(row)
        print('%-46s ok' % page, flush=True)
    json.dump(rows, open(OUT, 'w'), indent=0)
    print('ecrit', OUT, len(rows))


if __name__ == '__main__':
    run()
