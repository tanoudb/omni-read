"""D3 — audit chiffre : chirurgical_mask vs masques polygones, 632 bulles.

Trois masques compares, tous en coordonnees locales bbox :
  CH   = chirurgical_mask (cache)          -> ce que le rendu utilise aujourd'hui
  OCR  = _ocr_mask_from_regions(...)       -> "polygones + encre", SANS les 2 garde-fous
  FULL = fillPoly(polygones) dilate 11     -> branche de repli REELLE de inpaint_region
                                              quand chirurgical_mask est None

References image (independantes du seuillage Otsu par ligne du pipeline) :
  INK    = encre source, modele de fond morphologique + ecart robuste
  EDGES  = structure du dessin (Canny)
  BAND   = bande du trait de ballon (frontiere de l'interieur deduit de l'image)
"""
import sys, pickle, os, json, math
sys.path.insert(0, '.')
import numpy as np
import cv2
from pipeline import TranslationPipeline
from core.renderer import TextRenderer

CACHE = 'scratch/bareme/cache'
OUT = 'scratch/explore/d3_audit.json'


def nz(m):
    return 0 if m is None else int(np.count_nonzero(m))


def poly_union(regions, h, w):
    m = np.zeros((h, w), np.uint8)
    for r in regions or []:
        pts = r.get('bbox') if isinstance(r, dict) else None
        if not pts:
            continue
        a = np.array(pts, np.int32)
        if a.ndim != 2 or a.shape[0] < 3:
            continue
        a[:, 0] = np.clip(a[:, 0], 0, w - 1)
        a[:, 1] = np.clip(a[:, 1], 0, h - 1)
        cv2.fillPoly(m, [a], 255)
    return m


def full_poly_mask(regions, h, w):
    """Reproduit EXACTEMENT la 3e branche de inpaint_region (repli polygones)."""
    m = poly_union(regions, h, w)
    if nz(m) == 0:
        return None
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    return cv2.dilate(m, k, iterations=1)


def ink_reference(crop_bgr, search, ks=31, thr=28):
    """Encre source, estimee sans le seuillage Otsu-par-ligne du pipeline.

    Fond = flou MEDIAN de rayon 15 px (le texte occupe moins de la moitie de
    la fenetre, il disparait ; le degrade de fond, lui, survit). L'encre est
    ce qui s'en ecarte de plus de 28 niveaux, des DEUX cotes (texte sombre sur
    clair comme clair sur sombre). Restreint a `search`.
    Valide sur 30-years p01 #14 : encre jusqu'a x=575 (polygones jusqu'a 573,
    chirurgical s'arrete a 429).
    """
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    k = ks if min(g.shape[:2]) > ks else (max(3, (min(g.shape[:2]) // 2) * 2 - 1))
    bg = cv2.medianBlur(g, k)
    diff = cv2.absdiff(g, bg)
    ink = ((diff >= thr) & (search > 0)).astype(np.uint8) * 255
    return cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))


def stroke_band(crop_bgr, h, w):
    """Bande du trait de ballon : frontiere de l'interieur deduit de l'image."""
    try:
        interior = TextRenderer._bubble_mask_from_image(crop_bgr)
    except Exception:
        interior = None
    if interior is None or interior.shape[:2] != (h, w):
        return None, None
    s = max(3, int(round(min(h, w) * 0.025)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * s + 1, 2 * s + 1))
    band = cv2.subtract(cv2.dilate(interior, k), cv2.erode(interior, k))
    return interior, band


def measure(mask, ink, halo, edges, band, area):
    if mask is None:
        return None
    m = mask > 0
    inkn = nz(ink)
    r = {
        'nz': int(m.sum()),
        'cover_pct': round(100.0 * int((m & (ink > 0)).sum()) / max(1, inkn), 1),
        'area_pct': round(100.0 * int(m.sum()) / max(1, area), 1),
    }
    # residu = encre source non couverte
    r['residu_pct'] = round(100.0 - r['cover_pct'], 1)
    # bavure = efface hors encre et hors halo d'antialias
    spill = m & (halo == 0)
    r['spill_px'] = int(spill.sum())
    r['spill_pct_area'] = round(100.0 * int(spill.sum()) / max(1, area), 1)
    # structure detruite = bavure sur un bord du dessin
    r['struct_px'] = int((spill & (edges > 0)).sum())
    # morsure de trait
    r['band_px'] = int((m & (band > 0)).sum()) if band is not None else -1
    return r



def guard_telemetry(it, crop, ocr, h, w):
    """Rejoue les DEUX garde-fous de _build_masks_for_detection et rapporte,
    pour chacun, le ratio compare a son seuil (0.50 pour l'interieur,
    0.30 pour le ballon)."""
    out = {}
    cur = ocr
    if ocr is None or nz(ocr) == 0:
        return out
    if str(it.get('class_name')).lower() != 'out_text':
        try:
            interior = TextRenderer._bubble_mask_from_image(crop)
        except Exception:
            interior = None
        if interior is not None and interior.shape[:2] == (h, w):
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            interior = cv2.erode(interior, k, iterations=1)
            bounded = cv2.bitwise_and(cur, interior)
            out['int_ratio'] = round(nz(bounded) / max(1, nz(cur)), 4)
            out['int_applied'] = bool(nz(bounded) >= 0.5 * nz(cur))
            if out['int_applied']:
                cur = bounded
    bub = it.get('mask_binary')
    if bub is not None:
        if getattr(bub, 'ndim', 0) == 3:
            bub = bub[:, :, 0]
        if bub.shape[:2] != (h, w):
            bub = cv2.resize(bub, (w, h), interpolation=cv2.INTER_NEAREST)
        bub = (bub > 0).astype(np.uint8) * 255
        st = max(3, int(round(min(h, w) * 0.025)))
        kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * st + 1, 2 * st + 1))
        er = cv2.erode(bub, kk, iterations=1)
        if nz(er) >= 0.35 * nz(bub):
            bub = er
        inter = cv2.bitwise_and(cur, bub)
        out['bub_ratio'] = round(int(np.sum(inter)) / max(1, int(np.sum(cur))), 4)
        out['bub_applied'] = bool(int(np.sum(inter)) > 0.30 * int(np.sum(cur)))
    return out


def run():
    rows = []
    pages = sorted(os.listdir(CACHE))
    for page in pages:
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
            if crop.size == 0 or crop.shape[0] != h or crop.shape[1] != w:
                continue
            regions = it.get('text_regions')
            pu = poly_union(regions, h, w)
            row = {'page': page, 'idx': idx, 'cls': str(it.get('class_name')),
                   'w': w, 'h': h, 'area': h * w, 'text': (it.get('text') or '')[:60],
                   'n_regions': len(regions or []), 'poly_nz': nz(pu)}
            if nz(pu) == 0:
                row['skip'] = 'no_poly'
                rows.append(row)
                continue
            search = cv2.dilate(pu, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
            ink = ink_reference(crop, search)
            halo = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
            edges = cv2.Canny(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 60, 160)
            interior, band = stroke_band(crop, h, w)
            row['ink_nz'] = nz(ink)

            ocr = TranslationPipeline._ocr_mask_from_regions(regions, h, w, crop_bgr=crop)
            full = full_poly_mask(regions, h, w)
            ch = it.get('chirurgical_mask')
            if ch is not None and ch.shape[:2] != (h, w):
                ch = cv2.resize(ch, (w, h), interpolation=cv2.INTER_NEAREST)

            row.update(guard_telemetry(it, crop, ocr, h, w))
            row['CH'] = measure(ch, ink, halo, edges, band, h * w)
            row['OCR'] = measure(ocr, ink, halo, edges, band, h * w)
            row['FULL'] = measure(full, ink, halo, edges, band, h * w)
            if ch is not None and ocr is not None:
                a, b = ch > 0, ocr > 0
                row['ch_eq_ocr'] = bool(np.array_equal(a, b))
                row['ch_minus_ocr'] = int((a & ~b).sum())
                row['ocr_minus_ch'] = int((b & ~a).sum())
                row['ocr_minus_ch_ink'] = int((b & ~a & (ink > 0)).sum())
            rows.append(row)
        print('%-46s %d zones' % (page, len(d['items'])), flush=True)
    json.dump(rows, open(OUT, 'w'), indent=0)
    print('ecrit', OUT, len(rows), 'lignes')


if __name__ == '__main__':
    run()
