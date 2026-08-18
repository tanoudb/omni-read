# -*- coding: utf-8 -*-
"""PoC : forme du ballon par CROISSANCE DEPUIS L'ENCRE.

Un seul cas, pas de generalisation : la bulle ronde SOMBRE de
i-married-the-dragon, sur laquelle `_bubble_mask_from_image` fuit.

Principe : la teinte de reference ne marche pas (ballon noir sur decor noir),
mais le TRAIT du ballon est une arete franche quelle que soit la polarite. On
part donc de l'encre du texte et on irradie vers l'exterieur, bloque par les
aretes, plafonne par la bbox de detection.

Garde-fous :
  - la croissance ne sort JAMAIS de la bbox (plafond dur, pas d'inondation) ;
  - si elle atteint le bord de la bbox sur plusieurs cotes, c'est une fuite :
    on revient au dernier etat coherent et on le referme par son enveloppe
    convexe. Pas l'enveloppe de l'ENCRE — ce serait l'encombrement du texte
    anglais, et on retuerait le budget dynamique.

Usage: python scratch/poc_ink_growth.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"A:\omni read")))

import cv2
import numpy as np

RUN = "scratch/render_out/DRAG_DBG"
SRC_ORIG = "scratch/slices/drag_ov.png"      # texte ENCORE present
SRC_ERASED = "scratch/render_out/DRAG_DBG/page_erased.png"


def ink_from_regions(crop, regions):
    """Encre REELLE des glyphes, pas les polygones remplis.

    Premier jet errone : je remplissais les polygones de ligne OCR, ce qui
    donnait 40 504 px pour une bulle de 118 335 — un tiers de la boite. La
    croissance partait donc deja au contact des bords et fuyait a l'etape 3.
    Et je cherchais cette encre sur l'image EFFACEE, ou le texte n'existe plus.
    """
    poly = np.zeros(crop.shape[:2], np.uint8)
    for r in regions:
        pts = r.get("bbox")
        if pts and len(pts) >= 3:
            cv2.fillPoly(poly, [np.array(pts, np.int32)], 255)
    if int(np.count_nonzero(poly)) < 64:
        return poly
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dedans = gray[poly > 0]
    thr, _ = cv2.threshold(dedans, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sombre = float(np.mean(dedans < thr)) <= 0.5
    encre = ((gray < thr) if sombre else (gray >= thr)) & (poly > 0)
    return (encre.astype(np.uint8) * 255)


def grow_from_ink(crop, ink, max_steps=400):
    """Retourne (masque, diagnostic)."""
    h, w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 40, 40)          # lisse le grain, garde les aretes

    # Arete = gradient fort. Insensible a la POLARITE : un trait clair sur fond
    # sombre et un trait sombre sur fond clair donnent tous deux un gradient.
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    seuil = float(np.percentile(mag, 88))
    murs = (mag >= seuil)
    # Le texte lui-meme est plein d'aretes : on les neutralise, sinon la
    # croissance ne sort jamais de l'encre.
    murs[cv2.dilate(ink, np.ones((9, 9), np.uint8)) > 0] = False

    libre = (~murs)
    grown = ink > 0
    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dernier_sain = grown.copy()
    fuite_a = None

    for step in range(max_steps):
        nouveau = (cv2.dilate(grown.astype(np.uint8), k) > 0) & libre
        if nouveau.sum() == grown.sum():
            break
        grown = nouveau
        # Bords atteints ?
        cotes = sum([
            grown[0, :].any(), grown[-1, :].any(),
            grown[:, 0].any(), grown[:, -1].any(),
        ])
        if cotes >= 2:
            fuite_a = step
            break
        dernier_sain = grown.copy()

    if fuite_a is not None:
        cnts, _ = cv2.findContours(dernier_sain.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        ferme = np.zeros((h, w), np.uint8)
        if cnts:
            hull = cv2.convexHull(np.vstack(cnts))
            cv2.drawContours(ferme, [hull], -1, 255, -1)
        return ferme, f"FUITE a l'etape {fuite_a} -> enveloppe convexe de la zone conquise"

    ferme = np.zeros((h, w), np.uint8)
    cnts, _ = cv2.findContours(grown.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(ferme, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return ferme, "cloture atteinte sans fuite"


def bandes(mask, n=9):
    rows = np.nonzero(mask.any(axis=1))[0]
    if rows.size == 0:
        return []
    out = []
    for r in np.linspace(rows.min(), rows.max(), n).astype(int):
        cols = np.nonzero(mask[r])[0]
        if cols.size:
            out.append((r, int(cols.min()), int(cols.max()),
                        int(cols.max() - cols.min() + 1),
                        (cols.min() + cols.max()) / 2.0))
    return out


def main():
    m = json.load(open(f"{RUN}/bubbles_meta.json", encoding="utf-8"))[0]
    x1, y1, x2, y2 = m["bbox"]
    crop_orig = cv2.imread(SRC_ORIG)[y1:y2, x1:x2]
    crop = cv2.imread(SRC_ERASED)[y1:y2, x1:x2]
    # L'encre se lit sur l'ORIGINAL, la croissance se fait sur l'EFFACE
    # (ou le trait du ballon subsiste mais plus le texte).
    ink = ink_from_regions(crop_orig, m.get("text_regions") or [])
    print(f"bbox {x2-x1}x{y2-y1}   encre {int(np.count_nonzero(ink))} px")

    mask, diag = grow_from_ink(crop, ink)
    print("croissance :", diag)
    print(f"remplissage : {100*np.count_nonzero(mask)/mask.size:.1f}%  "
          f"(un cercle inscrit ferait ~78.5%)")

    print("\n  y      x_min  x_max  largeur  centre")
    for r, a, b, larg, c in bandes(mask):
        print(f"  {r:4d}   {a:5d}  {b:5d}  {larg:6d}   {c:6.1f}")

    cs = [c for *_, c in bandes(mask)]
    if cs:
        print(f"\n  derive du centre : {max(cs)-min(cs):.1f} px "
              f"({100*(max(cs)-min(cs))/mask.shape[1]:.1f}% de la largeur)")

    ov = crop.copy()
    ov[mask > 0] = (0, 0, 255)
    cv2.imwrite("scratch/diag/poc_growth.png",
                cv2.addWeighted(crop, .45, ov, .55, 0))
    print("-> scratch/diag/poc_growth.png")


if __name__ == "__main__":
    main()
