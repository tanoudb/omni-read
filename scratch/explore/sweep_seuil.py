# -*- coding: utf-8 -*-
"""Balayage du seuil du GARDE-FOU 1 de `pipeline.py::_build_masks_for_detection`.

Le garde-fou borne le masque d'encre a l'« interieur du ballon » deduit de
l'image (`TextRenderer._bubble_mask_from_image`, erode de 7), et n'accepte ce
bornage que si `|borne| >= SEUIL * |encre|`. Le seuil vaut 0,50 en production.

Son propre commentaire enonce le bon principe — « mieux vaut un contour
legerement entame qu'un texte d'origine qui survit sous la traduction » — mais
0,50 autorise a PERDRE LA MOITIE de l'encre. Le cas 30-years p01 #14 passe a
0,5081, soit 0,8 point au-dessus du seuil, et 49,2 % de son texte reste visible.

Ce script mesure, pour plusieurs seuils et sur TOUT le corpus :
  - couverture de l'encre reelle par le masque d'effacement retenu
  - structure (trait de ballon / decor) que le bornage protege encore
Aucun GPU, aucune instanciation de TextRenderer (on n'appelle que des
@staticmethod). Aucun fichier de production modifie.
"""
import sys, os, glob, pickle, json
sys.path.insert(0, '.')
import numpy as np
import cv2

from pipeline import TranslationPipeline
from core.renderer import TextRenderer

sys.path.insert(0, 'scratch/explore')
from d2_ink import ink_mask  # estimateur d'encre valide visuellement par D2

SEUILS = [0.50, 0.70, 0.85, 0.90, 0.95, 0.99, 2.00]  # 2.00 = ne borne JAMAIS


def interior_borne(crop, h, w, class_name):
    """Reproduit exactement le garde-fou 1 de pipeline.py (l.709-725)."""
    if str(class_name or '').lower() == 'out_text':
        return None
    try:
        interior = TextRenderer._bubble_mask_from_image(crop)
    except Exception:
        return None
    if interior is None or interior.shape[:2] != (h, w):
        return None
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.erode(interior, k, iterations=1)


def structure_map(crop, ink):
    """Bords francs qui ne sont PAS de l'encre : trait de ballon, decor.

    C'est ce que le bornage est cense proteger. On l'estime par Canny, prive de
    l'encre dilatee — sinon les glyphes eux-memes compteraient comme structure.
    """
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(g, 60, 160)
    if ink is not None:
        edges = cv2.bitwise_and(
            edges, cv2.bitwise_not(
                cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))))
    return edges


def main():
    lignes = []
    for cd in sorted(glob.glob('scratch/bareme/cache/*')):
        f = os.path.join(cd, 'dets.pkl')
        if not os.path.exists(f):
            continue
        blob = pickle.load(open(f, 'rb'))
        img = cv2.imread(os.path.join(cd, 'page.png'))
        if img is None:
            continue
        H, W = img.shape[:2]
        for i, it in enumerate(blob['items']):
            x1, y1, x2, y2 = it['bbox']
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img[y1:y2, x1:x2]
            h, w = y2 - y1, x2 - x1
            regs = it.get('text_regions')
            if not regs:
                continue
            ocr = TranslationPipeline._ocr_mask_from_regions(regs, h, w, crop, dilate=3)
            if ocr is None or int(np.count_nonzero(ocr)) == 0:
                continue
            ink, _ = ink_mask(crop, regs, h, w)
            if ink is None or int(np.count_nonzero(ink)) < 30:
                continue
            n_ink = int(np.count_nonzero(ink))

            interior = interior_borne(crop, h, w, it.get('class_name'))
            if interior is None:
                bounded = None
                ratio = None
            else:
                bounded = cv2.bitwise_and(ocr, interior)
                ratio = float(np.count_nonzero(bounded)) / float(np.count_nonzero(ocr))

            struct = structure_map(crop, ink)
            rec = {
                'serie': blob['series'], 'page': blob.get('page', 'p01'), 'idx': i,
                'classe': it.get('class_name'), 'ratio': ratio, 'n_ink': n_ink,
                'texte': (it.get('text') or '')[:40],
            }
            for s in SEUILS:
                if bounded is not None and ratio is not None and ratio >= s:
                    m = bounded
                else:
                    m = ocr
                # couverture de l'encre par le masque d'effacement
                rec['cov_%.2f' % s] = float(np.count_nonzero(cv2.bitwise_and(m, ink))) / n_ink
                # structure touchee par le masque (ce qu'on risque d'abimer)
                rec['str_%.2f' % s] = int(np.count_nonzero(cv2.bitwise_and(m, struct)))
            lignes.append(rec)
        del img

    json.dump(lignes, open('scratch/explore/sweep_seuil.json', 'w'), indent=1)

    n = len(lignes)
    print('zones mesurees : %d' % n)
    print()
    print('%-8s %10s %10s %10s %12s %12s' % (
        'seuil', 'cov p05', 'cov p25', 'cov p50', 'zones<70%', 'structure'))
    print('-' * 68)
    for s in SEUILS:
        cov = sorted(r['cov_%.2f' % s] for r in lignes)
        st = sum(r['str_%.2f' % s] for r in lignes)
        q = lambda p: cov[min(len(cov) - 1, int(p * len(cov)))]
        bas = sum(1 for c in cov if c < 0.70)
        lbl = 'jamais' if s > 1 else '%.2f' % s
        print('%-8s %10.3f %10.3f %10.3f %12d %12d' % (lbl, q(.05), q(.25), q(.50), bas, st))

    print()
    print('Zones que le bornage AMPUTE (ratio dans [0.50, 1.00)) :')
    amp = [r for r in lignes if r['ratio'] is not None and 0.50 <= r['ratio'] < 1.0]
    amp.sort(key=lambda r: r['ratio'])
    print('  total : %d' % len(amp))
    for r in amp[:15]:
        print('   %-30s %s #%-3d %-9s ratio=%.3f  cov 0.50=%.2f -> jamais=%.2f  %r' % (
            r['serie'][:30], r['page'], r['idx'], r['classe'], r['ratio'],
            r['cov_0.50'], r['cov_2.00'], r['texte']))


if __name__ == '__main__':
    main()
