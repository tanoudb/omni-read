"""D3 — que RETIRENT exactement les garde-fous, et sensibilite a la croissance
de `_extend_fill_mask` (portee 5 px dans la bbox).

Pour chaque zone ou chirurgical != ocr, on classe les pixels retires :
  - ENCRE          : texte source qui restera VISIBLE (defaut eliminatoire)
  - STRUCTURE      : hors encre, sur un bord Canny  -> ce que le garde-fou protege
  - FOND           : hors encre, hors bord          -> retrait sans effet visuel
"""
import sys, pickle, os, json
sys.path.insert(0, '.')
import numpy as np
import cv2
from pipeline import TranslationPipeline
sys.path.insert(0, 'scratch/explore')
from d3_audit import poly_union, ink_reference, nz

CACHE = 'scratch/bareme/cache'


def main():
    tot = {'ink': 0, 'struct': 0, 'bg': 0}
    per = []
    sens = []
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
            search = cv2.dilate(pu, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
            ink = ink_reference(crop, search)
            halo = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            edges = cv2.Canny(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 60, 160)
            ocr = TranslationPipeline._ocr_mask_from_regions(regions, h, w, crop_bgr=crop)
            ch = it.get('chirurgical_mask')
            if ocr is None or ch is None:
                continue
            if ch.shape[:2] != (h, w):
                ch = cv2.resize(ch, (w, h), interpolation=cv2.INTER_NEAREST)
            removed = (ocr > 0) & (ch == 0)
            n = int(removed.sum())
            if n:
                a = int((removed & (ink > 0)).sum())
                b = int((removed & (halo == 0) & (edges > 0)).sum())
                c = n - a - b
                tot['ink'] += a; tot['struct'] += b; tot['bg'] += max(0, c)
                per.append((n, a, b, max(0, c), page, idx, cls, (it.get('text') or '')[:34]))
            # sensibilite : +5 px (portee de _extend_fill_mask dans la bbox)
            k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            kk = 11 if cls in ('out_text', 'system') else 7
            kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))
            chg = cv2.dilate(cv2.dilate(ch, kd), k5) > 0
            ocg = cv2.dilate(cv2.dilate(ocr, kd), k5) > 0
            s_ch = int((chg & (halo == 0) & (edges > 0)).sum())
            s_oc = int((ocg & (halo == 0) & (edges > 0)).sum())
            sens.append((s_oc - s_ch, cls, page, idx))
        print('.', end='', flush=True)
    print()
    T = sum(tot.values())
    print('=== Ce que les garde-fous retirent, TOUS zones confondues (%d px) ===' % T)
    for k, v in tot.items():
        print('  %-8s %8d px  (%.1f %%)' % (k, v, 100.0 * v / max(1, T)))
    print('\n=== Les 15 zones ou ils retirent le plus ===')
    per.sort(reverse=True)
    print('   %8s %8s %8s %8s  zone' % ('retire', 'ENCRE', 'struct', 'fond'))
    for n, a, b, c, page, idx, cls, txt in per[:15]:
        print('   %8d %8d %8d %8d  %-40s #%-3d %-8s %s' % (n, a, b, c, page[:40], idx, cls, txt))
    d = np.array([s[0] for s in sens], float)
    db = np.array([s[0] for s in sens if s[1] == 'bulle'], float)
    print('\n=== Sensibilite +5 px (proxy _extend_fill_mask) : struct OCR - struct CH ===')
    print('  toutes : med %.0f p90 %.0f p99 %.0f max %.0f  (>200px : %d zones)'
          % (np.median(d), np.percentile(d, 90), np.percentile(d, 99), d.max(), (d > 200).sum()))
    print('  bulle  : med %.0f p90 %.0f p99 %.0f max %.0f  (>200px : %d zones)'
          % (np.median(db), np.percentile(db, 90), np.percentile(db, 99), db.max(), (db > 200).sum()))
    worst = sorted(sens, reverse=True)[:8]
    for v, cls, page, idx in worst:
        print('   +%-6d %-8s %-42s #%d' % (v, cls, page[:42], idx))


if __name__ == '__main__':
    main()
