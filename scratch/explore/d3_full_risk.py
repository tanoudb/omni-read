"""D3 — risque de la branche de repli REELLE (FULL = polygones pleins + 11 px),
c'est-a-dire ce qui se passe si on supprime simplement `chirurgical_mask`.
Meme classification que d3_who_wins : ENCRE / STRUCTURE / FOND.
Cherche aussi la bulle canonique de E5 (« IT MIGHT JUST BE A NORMAL RUN »).
"""
import sys, pickle, os
sys.path.insert(0, '.')
import numpy as np
import cv2
from pipeline import TranslationPipeline
sys.path.insert(0, 'scratch/explore')
from d3_audit import poly_union, full_poly_mask, ink_reference, nz

CACHE = 'scratch/bareme/cache'
NEEDLE = 'NORMAL RUN'


def main():
    tot = {'ink': 0, 'struct': 0, 'bg': 0}
    per = []
    found = []
    for page in sorted(os.listdir(CACHE)):
        d = pickle.load(open(os.path.join(CACHE, page, 'dets.pkl'), 'rb'))
        img = cv2.imread(os.path.join(CACHE, page, 'page.png'))
        for idx, it in enumerate(d['items']):
            x1, y1, x2, y2 = it['bbox']
            h, w = max(1, y2 - y1), max(1, x2 - x1)
            crop = img[max(0, y1):y2, max(0, x1):x2]
            if crop.shape[0] != h or crop.shape[1] != w:
                continue
            regions = it.get('text_regions')
            pu = poly_union(regions, h, w)
            if nz(pu) == 0:
                continue
            cls = str(it.get('class_name')).lower()
            txt = (it.get('text') or '')
            search = cv2.dilate(pu, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
            ink = ink_reference(crop, search)
            halo = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            edges = cv2.Canny(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 60, 160)
            ch = it.get('chirurgical_mask')
            if ch is None:
                continue
            if ch.shape[:2] != (h, w):
                ch = cv2.resize(ch, (w, h), interpolation=cv2.INTER_NEAREST)
            kk = 11 if cls in ('out_text', 'system') else 7
            chd = cv2.dilate(ch, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
            full = full_poly_mask(regions, h, w)
            extra = (full > 0) & (chd == 0)
            n = int(extra.sum())
            if n:
                a = int((extra & (ink > 0)).sum())
                b = int((extra & (halo == 0) & (edges > 0)).sum())
                c = max(0, n - a - b)
                tot['ink'] += a; tot['struct'] += b; tot['bg'] += c
                per.append((b, n, a, c, page, idx, cls, txt[:34]))
            if NEEDLE in txt.upper():
                ocr = TranslationPipeline._ocr_mask_from_regions(regions, h, w, crop_bgr=crop)
                ocrd = cv2.dilate(ocr, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
                def cls_of(m):
                    mm = m > 0
                    return (int((mm & (ink > 0)).sum()),
                            int((mm & (halo == 0) & (edges > 0)).sum()),
                            round(100.0 * int((mm & (ink > 0)).sum()) / max(1, nz(ink)), 1))
                found.append((page, idx, cls, txt[:50], h, w, nz(ink),
                              cls_of(chd), cls_of(ocrd), cls_of(full)))
        print('.', end='', flush=True)
    print()
    T = sum(tot.values())
    print('=== FULL (polygones pleins+11) EN PLUS de dilate(chirurgical) : %d px ===' % T)
    for k, v in tot.items():
        print('  %-8s %8d px  (%.1f %%)' % (k, v, 100.0 * v / max(1, T)))
    per.sort(reverse=True)
    print('\n=== 12 pires zones par STRUCTURE detruite en plus ===')
    print('   %8s %8s %8s %8s  zone' % ('struct', 'total', 'encre', 'fond'))
    for b, n, a, c, page, idx, cls, txt in per[:12]:
        print('   %8d %8d %8d %8d  %-40s #%-3d %-8s %s' % (b, n, a, c, page[:40], idx, cls, txt))
    if found:
        print('\n=== Bulle canonique E5 ===')
        for page, idx, cls, txt, h, w, inkn, a, b, c in found:
            print(' %s #%d (%s) %dx%d encre=%d | %s' % (page, idx, cls, w, h, inkn, txt))
            print('   CHd  encre_couverte=%d struct=%d couv=%.1f%%' % a)
            print('   OCRd encre_couverte=%d struct=%d couv=%.1f%%' % b)
            print('   FULL encre_couverte=%d struct=%d couv=%.1f%%' % c)
    else:
        print('\n(bulle E5 « %s » absente du corpus de 16 planches)' % NEEDLE)


if __name__ == '__main__':
    main()
