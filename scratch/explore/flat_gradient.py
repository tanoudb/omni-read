# -*- coding: utf-8 -*-
"""Le fond de #14 n'est PAS uni : l'aplat blanc unique deverse le blanc de la
fenetre sur l'arche beige. On veut un test qui REFUSE l'aplat quand le fond
varie SPATIALEMENT (gauche != droite), sans casser les vrais fonds unis.

Idee mesuree ici : la couronne echantillonnee autour du masque doit etre
homogene NON SEULEMENT en ecart global (deja teste), mais aussi entre ses
REGIONS gauche/droite/haut/bas. Si deux moities de la couronne ont des couleurs
medianes eloignees, le fond a un gradient -> pas d'aplat.
"""
import sys, pickle
sys.path.insert(0, '.')
import numpy as np, cv2
from pipeline import TranslationPipeline
from core.detector import Detection


def couronne(crop, mask):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    ring = (cv2.dilate(mask, k, iterations=1) > 0) & (mask == 0)
    return ring


def gradient_spatial(crop, ring):
    """Ecart maximal de couleur mediane entre moities gauche/droite et haut/bas
    de la couronne. Grand => fond non uniforme spatialement."""
    ys, xs = np.nonzero(ring)
    if xs.size < 128:
        return None
    cx, cy = xs.mean(), ys.mean()
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    def med(sel):
        s = lab[ys[sel], xs[sel]]
        return np.median(s, axis=0) if s.shape[0] >= 32 else None
    g, d = med(xs < cx), med(xs >= cx)
    h, b = med(ys < cy), med(ys >= cy)
    ecarts = []
    if g is not None and d is not None:
        ecarts.append(float(np.linalg.norm(g - d)))
    if h is not None and b is not None:
        ecarts.append(float(np.linalg.norm(h - b)))
    return max(ecarts) if ecarts else None


def main():
    import glob, os
    # cas ciblé + quelques temoins de vrais fonds unis (dialogue en bulle)
    cibles = [
        ("30-years-have-passed-since-the-prologue", "p01", 14, "#14 fond NON uni (blanc+beige)"),
    ]
    # temoins : prendre des bulles au hasard, fond cense uni
    temoins = []
    for cd in sorted(glob.glob('scratch/bareme/cache/*'))[:6]:
        b = pickle.load(open(os.path.join(cd, 'dets.pkl'), 'rb'))
        for i, it in enumerate(b['items'][:12]):
            if str(it['class_name']).lower() == 'bulle':
                temoins.append((b['series'], b.get('page','p01'), i, "temoin bulle"))
                if len([t for t in temoins]) >= 20:
                    break
        if len(temoins) >= 20:
            break

    print("%-42s %10s %10s" % ("zone", "grad_LR/HB", "verdict"))
    print("-" * 66)
    for serie, page, idx, lbl in cibles + temoins:
        slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in serie).strip("-").lower()
        cd = "scratch/bareme/cache/%s__%s" % (slug, page)
        if not os.path.exists(cd):
            continue
        blob = pickle.load(open(cd + "/dets.pkl", "rb"))
        img = cv2.imread(cd + "/page.png")
        it = blob['items'][idx]
        x1, y1, x2, y2 = it['bbox']
        d = Detection(it['class_name'], [float(v) for v in it['bbox']], it['score'])
        d.text_original = d.text_translated = it['text']; d.ocr_confidence = it['ocr_confidence']
        d.text_regions = it['text_regions']; d.mask_regions = it.get('mask_regions'); d.mask_binary = it.get('mask_binary')
        p = TranslationPipeline.__new__(TranslationPipeline); p.segmenter=None; p.logger=None; p.debug=False
        p._build_masks_for_detection(img, d)
        # reconstruire le crop + masque local comme inpaint_region
        h, w = y2-y1, x2-x1
        m = max(30, h)
        cx1, cy1 = max(0, x1-m), max(0, y1-m)
        cx2, cy2 = min(img.shape[1], x2+m), min(img.shape[0], y2+m)
        crop = img[cy1:cy2, cx1:cx2]
        ch = d.chirurgical_mask
        lm = np.zeros(crop.shape[:2], np.uint8)
        oy, ox = y1-cy1, x1-cx1
        lm[oy:oy+ch.shape[0], ox:ox+ch.shape[1]] = (ch>0).astype(np.uint8)*255
        lm = cv2.dilate(lm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
        ring = couronne(crop, lm)
        grad = gradient_spatial(crop, ring)
        verdict = "APLAT" if (grad is not None and grad < 12) else "-> LaMa"
        print("%-42s %10s %10s  %s" % (lbl[:42], "%.1f"%grad if grad else "?", verdict, ""))


if __name__ == '__main__':
    main()
