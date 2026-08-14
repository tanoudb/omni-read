# -*- coding: utf-8 -*-
"""Simule le correctif propose (pad d'erosion reduit sur les bulles
dentelees, detectees via deficit isoperimetrique) et mesure l'effet sur
les largeurs de bande + la taille de police qui en resulterait, SANS
toucher core/renderer.py.
"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(r"A:\omni read")))

import cv2
import numpy as np
from core import TextRenderer

ERASED = cv2.imread(str(Path(r"A:\omni read\scratch\render_out\scream_repro\page_erased.png")))
BUBBLES = {
    "bulle0_IF_YOU_LEAVE": (317, 409, 571, 839),
    "bulle1_TOO_SLOW_KAZUKI": (403, 307, 581, 609),
}


def row_span(mask, y0, y1):
    h = mask.shape[0]
    y0c = max(0, min(h, int(round(y0)))); y1c = max(y0c + 1, min(h, int(round(y1))))
    band = mask[y0c:y1c, :]
    if band.size == 0:
        return 0.0
    return float(np.count_nonzero(band > 0)) / float(band.shape[0])


def widths_profile(mask, n_bands=12):
    h = mask.shape[0]; band = max(1, h // n_bands)
    return [round(row_span(mask, y, y + band), 1) for y in range(0, h, band)]


def isoperim(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c); perim = cv2.arcLength(c, True)
    return (perim * perim) / (4 * math.pi * area) if area > 0 else None


for name, (x1, y1, x2, y2) in BUBBLES.items():
    crop = ERASED[y1:y2, x1:x2]
    box_w, box_h = x2 - x1, y2 - y1
    derived = TextRenderer._bubble_mask_from_image(crop)
    nnz_pre = int(np.count_nonzero(derived))
    iso = isoperim(derived)
    pad_orig = max(3, int(round(min(box_w, box_h) * 0.06)))
    pad_new = max(2, int(round(pad_orig * 0.4))) if (iso is not None and iso >= 1.75) else pad_orig

    def apply_pad(pad):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
        eroded = cv2.erode(derived, kernel, iterations=1)
        if int(np.count_nonzero(eroded)) < 0.25 * nnz_pre:
            return derived
        return eroded

    post_orig = apply_pad(pad_orig)
    post_new = apply_pad(pad_new)

    print(f"\n=== {name} bbox_wh=({box_w},{box_h}) isoperim={iso:.3f} ===")
    print(f"  pad ORIG={pad_orig} -> coverage={np.count_nonzero(post_orig)/(box_w*box_h):.3f} "
          f"widths={widths_profile(post_orig)}")
    print(f"  pad NEW ={pad_new} -> coverage={np.count_nonzero(post_new)/(box_w*box_h):.3f} "
          f"widths={widths_profile(post_new)}")

    typ_orig = float(np.median([w for w in widths_profile(post_orig, 20)[5:15] if w > 0]))
    typ_new = float(np.median([w for w in widths_profile(post_new, 20)[5:15] if w > 0]))
    print(f"  typical_w (bande centrale) ORIG~{typ_orig:.0f}px -> NEW~{typ_new:.0f}px "
          f"(+{(typ_new/typ_orig-1)*100:.0f}%)")
