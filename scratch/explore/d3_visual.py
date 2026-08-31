"""D3 — planches de controle visuel : ce que le garde-fou retire.
Colonne 1 crop, 2 masque chirurgical, 3 masque polygones(encre),
4 differentiel : ROUGE = encre source que le garde-fou laisse en place
                 VERT  = structure (bord Canny hors encre) qu'il protege
"""
import sys, pickle, os
sys.path.insert(0, '.')
import numpy as np
import cv2
from pipeline import TranslationPipeline
sys.path.insert(0, 'scratch/explore')
from d3_audit import poly_union, ink_reference, nz

CACHE = 'scratch/bareme/cache'
OUTDIR = 'scratch/explore/d3_vis'
CASES = [
    ('30-years-have-passed-since-the-prologue__p01', 14),
    ('the_cleaner__p01', 20),
    ('i-married-the-dragon-i-killed__p01', 11),
    ('hellogin__p02', 27),
    ('the-frontier-count-s-10th-class-outcas__p01', 27),
    ('path-of-vengeance__p01', 20),
]

os.makedirs(OUTDIR, exist_ok=True)
for page, idx in CASES:
    d = pickle.load(open(os.path.join(CACHE, page, 'dets.pkl'), 'rb'))
    img = cv2.imread(os.path.join(CACHE, page, 'page.png'))
    it = d['items'][idx]
    x1, y1, x2, y2 = it['bbox']
    h, w = y2 - y1, x2 - x1
    crop = img[y1:y2, x1:x2]
    pu = poly_union(it['text_regions'], h, w)
    search = cv2.dilate(pu, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    ink = ink_reference(crop, search)
    halo = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    edges = cv2.Canny(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 60, 160)
    ocr = TranslationPipeline._ocr_mask_from_regions(it['text_regions'], h, w, crop_bgr=crop)
    ch = it['chirurgical_mask']
    if ch.shape[:2] != (h, w):
        ch = cv2.resize(ch, (w, h), interpolation=cv2.INTER_NEAREST)
    k = 11 if str(it['class_name']).lower() in ('out_text', 'system') else 7
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    chd, ocrd = cv2.dilate(ch, kd), cv2.dilate(ocr, kd)

    def over(m, col):
        o = crop.copy().astype(np.float32)
        a = (m > 0).astype(np.float32)[:, :, None] * 0.55
        o = o * (1 - a) + np.array(col, np.float32) * a
        return o.astype(np.uint8)

    rem = (ocrd > 0) & (chd == 0)
    diff = crop.copy()
    diff[rem & (ink > 0)] = (0, 0, 255)
    diff[rem & (halo == 0) & (edges > 0)] = (0, 255, 0)
    sep = np.full((h, 6, 3), 255, np.uint8)
    sheet = np.hstack([crop, sep, over(chd, (255, 0, 0)), sep,
                       over(ocrd, (0, 200, 255)), sep, diff])
    p = os.path.join(OUTDIR, '%s_%02d.png' % (page[:34], idx))
    cv2.imwrite(p, sheet)
    print('%-46s #%-3d %-8s -> %s  (encre laissee=%d px, structure protegee=%d px)'
          % (page[:46], idx, it['class_name'], p,
             int((rem & (ink > 0)).sum()),
             int((rem & (halo == 0) & (edges > 0)).sum())))
