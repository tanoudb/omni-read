# -*- coding: utf-8 -*-
"""Isole l'effet de l'erosion _inset dans _bubble_shape_mask sur les deux
bulles de cri. Compare le masque DERIVE DE L'IMAGE avant/apres _inset, et
mesure les largeurs de bande (_mask_row_span-like) aux deux etapes.

Ne modifie AUCUN fichier core/. Utilise uniquement les methodes deja
publiques/statiques de TextRenderer, appelees en lecture seule sur les
crops APRES effacement de scratch/render_out/scream_diag (recalcules ici
directement depuis scream_slice.png + les bbox connues des 2 detections).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(r"A:\omni read")))

import cv2
import numpy as np
from core import TextRenderer

# bbox mesurees par diag_scream.py sur scream_slice.png (apres detection)
BUBBLES = {
    "bulle0_IF_YOU_LEAVE": (317, 409, 571, 839),
    "bulle1_TOO_SLOW_KAZUKI": (403, 307, 581, 609),
}

# On travaille sur l'image APRES effacement (texte source efface), la meme
# que celle vue par insert_text -> _bubble_shape_mask -> _bubble_mask_from_image.
ERASED = cv2.imread(str(Path(r"A:\omni read\scratch\render_out\scream_diag\..\scream_repro\page_erased.png")))
if ERASED is None:
    raise SystemExit("page_erased.png introuvable — lance d'abord render_iterate.py sur scream_slice.png")

r = TextRenderer()


def row_span_manual(mask: np.ndarray, y0: float, y1: float) -> float:
    h = mask.shape[0]
    y0c = max(0, min(h, int(round(y0))))
    y1c = max(y0c + 1, min(h, int(round(y1))))
    band = mask[y0c:y1c, :]
    if band.size == 0:
        return 0.0
    return float(np.count_nonzero(band > 0)) / float(band.shape[0])


def widths_profile(mask: np.ndarray, n_bands: int = 12) -> list:
    h = mask.shape[0]
    band = max(1, h // n_bands)
    out = []
    for y in range(0, h, band):
        out.append(round(row_span_manual(mask, y, y + band), 1))
    return out


for name, (x1, y1, x2, y2) in BUBBLES.items():
    crop = ERASED[y1:y2, x1:x2]
    box_w, box_h = x2 - x1, y2 - y1
    print(f"\n=== {name}  bbox_wh=({box_w},{box_h}) ===")

    derived = TextRenderer._bubble_mask_from_image(crop)
    if derived is None:
        print("  _bubble_mask_from_image -> None (fallback ellipse/None en aval)")
        continue
    nnz_pre = int(np.count_nonzero(derived))
    print(f"  PRE-inset  : nnz={nnz_pre} coverage={nnz_pre/(box_w*box_h):.4f}")
    print(f"  PRE-inset  widths (12 bandes verticales): {widths_profile(derived)}")

    # Reproduit EXACTEMENT la fermeture _inset() de core/renderer.py::_bubble_shape_mask
    pad = max(3, int(round(min(box_w, box_h) * 0.06)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
    eroded = cv2.erode(derived, kernel, iterations=1)
    kept = int(np.count_nonzero(eroded)) >= 0.25 * nnz_pre
    post = eroded if kept else derived
    nnz_post = int(np.count_nonzero(post))
    print(f"  pad={pad} kernel={2*pad+1}x{2*pad+1} kept_branch={'eroded' if kept else 'original (25% floor hit)'}")
    print(f"  POST-inset : nnz={nnz_post} coverage={nnz_post/(box_w*box_h):.4f}  (delta={nnz_post-nnz_pre}, {(nnz_post-nnz_pre)/max(1,nnz_pre)*100:.1f}%)")
    print(f"  POST-inset widths (12 bandes verticales): {widths_profile(post)}")

    cv2.imwrite(str(Path(rf"A:\omni read\scratch\render_out\scream_diag\{name}_mask_pre.png")), derived)
    cv2.imwrite(str(Path(rf"A:\omni read\scratch\render_out\scream_diag\{name}_mask_post.png")), post)
