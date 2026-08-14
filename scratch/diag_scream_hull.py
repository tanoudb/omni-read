# -*- coding: utf-8 -*-
"""Mesure le ratio aire/convex-hull (indicateur de 'jaggedness') des deux
bulles de cri, compare a une bulle ovale normale de POV_V2, pour calibrer
un correctif qui module le pad d'erosion de `_inset()` selon la forme.

Lecture seule — n'importe/n'appelle que des methodes statiques deja
publiques de TextRenderer.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(r"A:\omni read")))

import cv2
import numpy as np
from core import TextRenderer


def hull_ratio(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1.0, 1.0
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    perim = cv2.arcLength(c, True)
    hr = float(area / hull_area) if hull_area > 0 else 1.0
    # Deficit isoperimetrique : 1.0 pour un cercle parfait, grimpe vite avec
    # les dentelures (chaque pointe ajoute beaucoup de perimetre pour peu
    # d'aire) mais reste proche de 1 pour une simple queue de bulle ovale.
    isoperim = float((perim * perim) / (4 * np.pi * area)) if area > 0 else 1.0
    return hr, isoperim


CASES = [
    ("bulle0_IF_YOU_LEAVE (dentelee)", r"A:\omni read\scratch\render_out\scream_repro\page_erased.png", (317, 409, 571, 839)),
    ("bulle1_TOO_SLOW_KAZUKI (dentelee)", r"A:\omni read\scratch\render_out\scream_repro\page_erased.png", (403, 307, 581, 609)),
]

# Toutes les bulles ("bulle", pas "out_text") de POV_V2 pour calibrer un seuil
POV = r"A:\omni read\scratch\render_out\POV_V2\page_erased.png"
import json as _json
meta = _json.load(open(r"A:\omni read\scratch\render_out\POV_V2\bubbles_meta.json", encoding="utf-8"))
for m in meta:
    if m["class"] != "bulle":
        continue
    x1, y1, x2, y2 = m["bbox"]
    CASES.append((f"POV#{m['index']:02d}_{m['ocr_text'][:20]!r}", POV, (x1, y1, x2, y2)))

for name, img_path, (x1, y1, x2, y2) in CASES:
    img = cv2.imread(img_path)
    if img is None:
        print(f"{name}: image introuvable {img_path}")
        continue
    crop = img[y1:y2, x1:x2]
    box_w, box_h = x2 - x1, y2 - y1
    derived = TextRenderer._bubble_mask_from_image(crop)
    if derived is None:
        print(f"{name}: _bubble_mask_from_image -> None")
        continue
    hr, iso = hull_ratio(derived)
    nnz = int(np.count_nonzero(derived))
    print(f"{name}: bbox_wh=({box_w},{box_h}) mask_coverage={nnz/(box_w*box_h):.3f} hull_ratio={hr:.3f} isoperim={iso:.3f}")
