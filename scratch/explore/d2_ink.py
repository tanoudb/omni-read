"""D2 - estimation robuste de l'ENCRE reelle + couverture des deux masques.

Definition de l'encre (justifiee dans le rapport) :
  1. enveloppe E = union des polygones de LIGNE OCR, dilatee de 2 px
     (les jambages debordent un peu du rectangle OCR).
  2. fond local B = median blur du crop en LAB, noyau impair ~ hauteur de
     ligne mediane (borne 9..41). Un median de rayon > epaisseur de trait
     efface les glyphes et ne garde que le fond -> marche pour du noir sur
     blanc comme du blanc sur sombre, et sur fond degrade/colore.
  3. residu R = distance euclidienne LAB entre le pixel et son fond local.
  4. seuil t = max(8.0, Otsu(R sur E)) : R est unilateral (fond ~ 0), Otsu y
     separe fond et encre sans supposer ni le sens du contraste ni que
     l'encre soit minoritaire (elle ne l'est pas sur du gros lettrage).
  5. encre = (R > t) dans E, nettoyee des composantes < 4 px.
"""
import sys, os, glob, pickle, json
sys.path.insert(0, '.')
import numpy as np
import cv2


def polygons_and_lines(regions, h, w):
    per_line = []
    env = np.zeros((h, w), np.uint8)
    for r in regions or []:
        pts = r.get('bbox') if isinstance(r, dict) else None
        if not pts:
            continue
        arr = np.array(pts, dtype=np.int32)
        if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
            continue
        arr[:, 0] = np.clip(arr[:, 0], 0, max(0, w - 1))
        arr[:, 1] = np.clip(arr[:, 1], 0, max(0, h - 1))
        line = np.zeros((h, w), np.uint8)
        cv2.fillPoly(line, [arr], 255)
        cv2.fillPoly(env, [arr], 255)
        per_line.append(line)
    return env, per_line


def ink_mask(crop_bgr, regions, h, w, dbg=None):
    env, per_line = polygons_and_lines(regions, h, w)
    if int(np.count_nonzero(env)) == 0:
        return None, None
    heights = [float(np.count_nonzero(l.any(axis=1))) for l in per_line if l.any()]
    line_h = float(np.median(heights)) if heights else float(h)
    k = int(round(line_h))
    k = max(9, min(151, k | 1))  # doit depasser l'EPAISSEUR DE TRAIT

    env_d = cv2.dilate(env, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = np.dstack([cv2.medianBlur(lab[..., c].astype(np.uint8), k) for c in range(3)]).astype(np.float32)
    res = np.sqrt(((lab - bg) ** 2).sum(axis=2))

    # Seuil sur le RESIDU par Otsu, restreint a l'enveloppe.
    # Une regle robuste du type median+3*MAD suppose que l'encre est
    # MINORITAIRE dans l'enveloppe. C'est faux sur du gros lettrage
    # (« HUP! », « EVE... ») ou l'encre occupe 40 % de l'enveloppe : la
    # mediane du residu tombe alors DANS l'encre, le seuil explose
    # (mesure : thr=273 sur « HUP! ») et le masque d'encre devient VIDE.
    # Le residu est unilateral (fond ~ 0, encre elevee) : Otsu y separe les
    # deux modes sans hypothese sur leurs proportions ni sur le sens du
    # contraste.
    vals = np.clip(res[env_d > 0], 0, 255).astype(np.uint8)
    try:
        otsu, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    except Exception:
        otsu = 8.0
    thr = max(8.0, float(otsu))
    sigma = float(np.median(np.abs(vals.astype(np.float32) - np.median(vals))))
    ink = ((res > thr) & (env_d > 0)).astype(np.uint8)

    n, lab_cc, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        keep[i] = stats[i, cv2.CC_STAT_AREA] >= 4
    ink = (keep[lab_cc]).astype(np.uint8) * 255
    # Variante de SENSIBILITE : sans les composantes qui touchent le bord de
    # la bbox. Sur une bulle a mot unique (« HUP! »), le rectangle du polygone
    # OCR englobe le TRAIT du ballon ; celui-ci passe alors pour de l'encre et
    # penalise les deux masques a tort. Une lettre ne touche pas le bord de sa
    # bbox de detection, le trait du ballon si.
    n2, l2, st2, _ = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), 8)
    keep2 = np.zeros(n2, bool)
    for i in range(1, n2):
        x, y, ww, hh, a = st2[i]
        keep2[i] = not (x <= 0 or y <= 0 or x + ww >= w or y + hh >= h)
    ink_nb = (keep2[l2]).astype(np.uint8) * 255
    if dbg is not None:
        dbg['ink_nb'] = ink_nb
    if dbg is not None:
        dbg.update(line_h=line_h, k=k, thr=thr, sigma=sigma, res=res, env=env, env_d=env_d, bg=bg)
    return ink, env_d


def core_of(ink):
    """Corps des glyphes, sans la frange d'anticrenelage de 1 px.

    Une couverture mesuree sur l'encre BRUTE plafonne vers 90 % meme quand un
    masque couvre visuellement toute la lettre : ma detection d'encre est un
    poil plus large que celle du pipeline et la difference est un liisere de
    1 px. Erodee, la frange disparait et il ne reste que ce qui, s'il n'est pas
    efface, se LIT encore."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.erode(ink, k, iterations=1)


def cov(ink, mask):
    tot = int(np.count_nonzero(ink))
    if tot == 0:
        return None
    if mask is None:
        return 0.0
    m = mask
    if m.shape[:2] != ink.shape[:2]:
        m = cv2.resize(m, (ink.shape[1], ink.shape[0]), interpolation=cv2.INTER_NEAREST)
    return float(np.count_nonzero((m > 0) & (ink > 0))) / tot
