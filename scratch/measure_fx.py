# -*- coding: utf-8 -*-
"""Mesure A BLANC des trois quantites typographiques du texte d'origine.

N'ECRIT RIEN dans le rendu.

METHODE : profil RADIAL. On calcule la distance de chaque pixel au glyphe le
plus proche, puis la couleur mediane de chaque anneau a distance 1, 2, 3... px.
Le profil obtenu se lit directement :

    distance :  0        1  2  3        4  5  6  7  8       9+
    couleur  : [encre] [ contour  ] [ ---- lueur ---- ] [ fond ]
                        plateau net    rampe monotone     stable

  - GRAISSE  : epaisseur du trait du glyphe (transformee de distance sur l'encre)
  - CONTOUR  : longueur du PLATEAU initial — anneau de couleur quasi constante,
               distincte du fond. Bord franc, donc mesurable sans tolerance
               arbitraire.
  - LUEUR    : distance a laquelle l'ecart au fond repasse sous le bruit, APRES
               le plateau. Pas de garde-fou anti-decor ici : on ne modifie rien,
               on veut l'etendue reelle.

Lire tout le profil PUIS decider evite le piege de la premiere version, qui
reutilisait le critere d'arret de l'effacement — calibre pour etre prudent,
donc structurellement incapable de mesurer une etendue.

Usage:
    python scratch/measure_fx.py <run_dir> <image_source.jpg>
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"A:\omni read")))

import cv2
import numpy as np

MAX_RADIUS = 24


def _ink_mask(crop, regions):
    """Masque du coeur des glyphes, borne aux polygones de ligne OCR."""
    h, w = crop.shape[:2]
    poly = np.zeros((h, w), np.uint8)
    for r in regions:
        pts = r.get("bbox")
        if not pts or len(pts) < 3:
            continue
        arr = np.array([[int(p[0]), int(p[1])] for p in pts], np.int32)
        arr[:, 0] = np.clip(arr[:, 0], 0, w - 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, h - 1)
        cv2.fillPoly(poly, [arr], 255)
    if int(np.count_nonzero(poly)) < 64:
        return None, None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    inside = gray[poly > 0]
    thr, _ = cv2.threshold(inside, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink_dark = float(np.mean(inside < thr)) <= 0.5
    ink = (((gray < thr) if ink_dark else (gray >= thr)) & (poly > 0)).astype(np.uint8) * 255
    return (ink if int(np.count_nonzero(ink)) >= 64 else None), ink_dark


def _radial_profile(crop, ink):
    """[(distance, couleur mediane BGR, nb pixels)] pour d = 1..MAX_RADIUS."""
    dist = cv2.distanceTransform((ink == 0).astype(np.uint8), cv2.DIST_L2, 5)
    prof = []
    for d in range(1, MAX_RADIUS + 1):
        band = (dist >= d - 0.5) & (dist < d + 0.5)
        n = int(np.count_nonzero(band))
        if n < 30:
            prof.append((d, None, n))
            continue
        prof.append((d, np.median(crop[band].reshape(-1, 3), axis=0), n))
    return prof, dist



def _outline_by_normals(crop, ink, max_len=12, n_samples=400):
    """Largeur du contour, mesuree LE LONG DE LA NORMALE au bord du glyphe.

    Les anneaux concentriques ne peuvent pas voir un contour : l'anti-crenelage
    fait varier la couleur a chaque pixel de distance, donc aucun "plateau" ne
    survit a une comparaison anneau par anneau. Mesure precedente : 1 px sur les
    34 bulles, ecart-type nul, avec deux logiques opposees — le signe que la
    geometrie de lecture etait en cause, pas le seuil.

    Ici on tire des profils 1D perpendiculaires au bord, on cherche sur chacun
    le plateau de couleur entre le glyphe et le fond, et on prend la MEDIANE des
    largeurs trouvees. Un profil bruite ou ambigu ne fait que deplacer la
    mediane a la marge, la ou il cassait tout dans la version par anneaux.
    """
    edges = cv2.Canny((ink > 0).astype(np.uint8) * 255, 50, 150)
    ys, xs = np.nonzero(edges)
    if len(xs) < 12:
        return 0, None

    # Normale = gradient de la carte de distance a l'encre.
    dist = cv2.distanceTransform((ink == 0).astype(np.uint8), cv2.DIST_L2, 5)
    gx = cv2.Sobel(dist, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(dist, cv2.CV_32F, 0, 1, ksize=3)

    h, w = ink.shape[:2]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)

    idx = np.linspace(0, len(xs) - 1, min(n_samples, len(xs))).astype(int)
    widths, colors = [], []
    for i in idx:
        x0, y0 = int(xs[i]), int(ys[i])
        vx, vy = float(gx[y0, x0]), float(gy[y0, x0])
        n = (vx * vx + vy * vy) ** 0.5
        if n < 1e-3:
            continue
        vx, vy = vx / n, vy / n

        prof = []
        for t in range(1, max_len + 1):
            xx, yy = int(round(x0 + vx * t)), int(round(y0 + vy * t))
            if not (0 <= xx < w and 0 <= yy < h):
                break
            prof.append(lab[yy, xx])
        if len(prof) < 5:
            continue
        prof = np.array(prof)

        # Fond = fin du profil ; contour = prefixe qui en reste loin ET reste
        # proche de lui-meme. On saute le premier pixel, toujours anti-crenele.
        bg = np.median(prof[-3:], axis=0)
        far = [float(np.linalg.norm(pv - bg)) for pv in prof]
        if far[1] < 12.0:
            continue
        ref = prof[1]
        wdt = 0
        for t in range(1, len(prof)):
            if far[t] < 12.0 or float(np.linalg.norm(prof[t] - ref)) > 14.0:
                break
            wdt = t
        if wdt:
            widths.append(wdt)
            colors.append(crop[
                min(h - 1, max(0, int(round(y0 + vy * max(1, wdt // 2))))),
                min(w - 1, max(0, int(round(x0 + vx * max(1, wdt // 2))))),
            ])

    if len(widths) < 8:
        return 0, None
    med = int(round(float(np.median(widths))))
    col = np.median(np.array(colors), axis=0)
    return med, col


def measure_one(img, det):
    x1, y1, x2, y2 = det["bbox"]
    crop = img[max(0, y1):y2, max(0, x1):x2]
    regions = det.get("text_regions") or []
    if crop.size == 0 or min(crop.shape[:2]) < 16 or not regions:
        return None

    ink, ink_dark = _ink_mask(crop, regions)
    if ink is None:
        return None

    # 1. GRAISSE du trait.
    dt_ink = cv2.distanceTransform((ink > 0).astype(np.uint8), cv2.DIST_L2, 5)
    weight = 2.0 * float(np.percentile(dt_ink[ink > 0], 80))

    prof, dist = _radial_profile(crop, ink)
    valid = [(d, c) for d, c, n in prof if c is not None]
    if len(valid) < 6:
        return None

    # Fond : couleur des anneaux les plus lointains.
    bg = np.median(np.array([c for d, c in valid[-5:]]), axis=0)
    noise = float(np.median([
        np.linalg.norm(valid[i][1] - valid[i + 1][1]) for i in range(len(valid) - 5, len(valid) - 1)
    ]))
    noise = max(noise, 3.0)

    dev = [(d, float(np.linalg.norm(c - bg))) for d, c in valid]

    # ── Les trois mesures sont HERMETIQUES : aucune ne lit le resultat d'une
    #    autre. Le couplage precedent (la lueur demarrait a `d > outline`)
    #    fabriquait un faux signal : les zeros que j'avais pris pour la preuve
    #    que la lueur distinguait le FX du fond venaient en realite du saut du
    #    premier anneau impose par `outline = 1`. Changer le contour les a fait
    #    disparaitre — la lueur n'avait jamais ete validee.

    # 2. CONTOUR — independant, mesure le long de la NORMALE au bord.
    outline, outline_col = _outline_by_normals(crop, ink)

    # 3. LUEUR — independante, demarre elle aussi a d=1.
    #    Ce qui distingue une lueur d'un fond texture n'est pas la distance mais
    #    la MONOTONIE : une lueur decroit regulierement vers le fond, une texture
    #    fluctue. On mesure donc la longueur du prefixe ou l'ecart au fond
    #    DECROIT de facon continue, en tolerant le bruit.
    glow = 0
    for i in range(1, len(dev)):
        if dev[i][1] < 2.0 * noise:
            break
        if dev[i][1] > dev[i - 1][1] + noise:   # ca remonte : ce n'est plus une rampe
            break
        glow = dev[i][0]

    def rgb(v):
        return (int(v[2]), int(v[1]), int(v[0])) if v is not None else None

    outline_c = outline_col
    glow_c = next((c for d, c in valid if d == max(2, glow)), None) if glow else None

    return {
        "index": det["index"], "class": det["class"],
        "weight_px": round(weight, 2),
        "outline_px": int(outline),
        "glow_px": int(glow),
        "noise": round(noise, 2),
        "ink_rgb": rgb(np.median(crop[ink > 0].reshape(-1, 3), axis=0)),
        "outline_rgb": rgb(outline_c),
        "glow_rgb": rgb(glow_c),
        "bg_rgb": rgb(bg),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("source", type=Path)
    a = ap.parse_args()

    img = cv2.imread(str(a.source))
    if img is None:
        raise SystemExit(f"Image illisible: {a.source}")
    meta = json.loads((a.run_dir / "bubbles_meta.json").read_text(encoding="utf-8"))
    if not any(d.get("text_regions") for d in meta):
        raise SystemExit("Aucun polygone OCR dans le meta — relancer render_iterate.py")

    rows = [m for m in (measure_one(img, d) for d in meta) if m]

    print(f"{'#':>3s} {'classe':9s} {'graisse':>8s} {'contour':>8s} {'lueur':>7s}  "
          f"{'encre':>15s} {'contour':>15s} {'lueur':>15s}")
    for m in rows:
        print(f"{m['index']:3d} {m['class']:9s} {m['weight_px']:8.2f} "
              f"{m['outline_px']:8d} {m['glow_px']:7d}  "
              f"{str(m['ink_rgb']):>15s} {str(m['outline_rgb']):>15s} {str(m['glow_rgb']):>15s}")

    print("\n--- stabilite (mediane / moyenne / ecart-type / n non nul) ---")
    for key in ("weight_px", "outline_px", "glow_px"):
        vals = [m[key] for m in rows]
        nz = [v for v in vals if v]
        print(f"  {key:12s} {statistics.median(vals):7.2f} {statistics.mean(vals):7.2f} "
              f"{statistics.pstdev(vals):7.2f}   {len(nz)}/{len(vals)}")

    out = a.run_dir / "fx_measures.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
