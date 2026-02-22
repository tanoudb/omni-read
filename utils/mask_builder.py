"""
═══════════════════════════════════════════════════════════════════════════════
MASK BUILDER — Module centralisé de construction de masques

Découple l'OCR de l'inpainting :
- L'OCR produit des text_regions (polygones précis autour du texte)
- Ce module transforme ces polygones en masque d'inpainting PROPRE
- Le masque est toujours une zone unie, sans trous, avec marge uniforme

Stratégie : enveloppe convexe → fermeture morphologique → dilatation
═══════════════════════════════════════════════════════════════════════════════
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple


def build_inpainting_mask(
    crop_h: int,
    crop_w: int,
    text_regions: Optional[List[Dict]],
    dilate_px: int = 10,
    close_kernel: int = 15,
    use_convex_hull: bool = True,
) -> np.ndarray:
    """
    Construit un masque d'inpainting PROPRE à partir des régions OCR.

    Peu importe la précision de l'OCR, le masque résultant est toujours
    une zone unie qui couvre tout le texte + sa marge.

    Args:
        crop_h, crop_w: dimensions du crop
        text_regions: list de dict avec clé 'bbox' (liste de points [x,y])
        dilate_px: marge en pixels autour du texte (défaut 10)
        close_kernel: taille du kernel morphologique CLOSE (défaut 15)
        use_convex_hull: si True, utilise l'enveloppe convexe (défaut True)

    Returns:
        masque binaire uint8 (255 = zone à inpainter)
    """
    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    if not text_regions:
        return mask

    # 1. Collecter tous les points de toutes les régions
    all_points = []
    per_region_polys = []

    for region in text_regions:
        pts = region.get('bbox')
        if not pts:
            continue
        arr = np.array(pts, dtype=np.int32)
        if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
            continue
        arr[:, 0] = np.clip(arr[:, 0], 0, max(0, crop_w - 1))
        arr[:, 1] = np.clip(arr[:, 1], 0, max(0, crop_h - 1))
        per_region_polys.append(arr)
        all_points.extend(arr.tolist())

    if not all_points or len(all_points) < 3:
        return mask

    if use_convex_hull:
        # Enveloppe convexe de tous les points → zone unie sans trous
        pts_array = np.array(all_points, dtype=np.int32)
        hull = cv2.convexHull(pts_array)
        if hull is not None and len(hull) >= 3:
            cv2.fillConvexPoly(mask, hull, 255)
    else:
        # Remplir chaque polygone individuellement
        for poly in per_region_polys:
            cv2.fillPoly(mask, [poly], 255)

    if np.sum(mask) == 0:
        return mask

    # 2. Fermeture morphologique → bouche les trous entre lettres
    k_close = max(3, close_kernel | 1)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # 3. Dilatation uniforme → marge autour du texte
    if dilate_px > 0:
        k_dilate = max(3, (dilate_px * 2 + 1) | 1)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_dilate, k_dilate))
        mask = cv2.dilate(mask, kernel_dilate, iterations=1)

    return mask


def build_inpainting_mask_bbox_fallback(
    crop_h: int,
    crop_w: int,
    det_x1: int,
    det_y1: int,
    det_x2: int,
    det_y2: int,
    crop_x1: int,
    crop_y1: int,
    shrink_ratio: float = 0.05,
) -> np.ndarray:
    """
    Masque fallback basé sur la bbox de détection (quand pas de text_regions).
    Utilisé uniquement pour les classes 'system'.
    """
    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)

    lx1 = max(0, det_x1 - crop_x1)
    ly1 = max(0, det_y1 - crop_y1)
    lx2 = min(crop_w - 1, det_x2 - crop_x1)
    ly2 = min(crop_h - 1, det_y2 - crop_y1)

    w = lx2 - lx1
    h = ly2 - ly1
    if w <= 0 or h <= 0:
        return mask

    sx = max(1, int(w * shrink_ratio))
    sy = max(1, int(h * shrink_ratio))

    cv2.rectangle(mask, (lx1 + sx, ly1 + sy), (lx2 - sx, ly2 - sy), 255, -1)
    return mask


def regions_to_crop_coords(
    text_regions: Optional[List[Dict]],
    det_x1: int,
    det_y1: int,
    crop_x1: int,
    crop_y1: int,
    crop_w: int,
    crop_h: int,
) -> List[Dict]:
    """
    Convertit les régions detection-local → crop-local (une seule fois).

    Les text_regions sont en coordonnées relatives à la bbox de détection.
    Le crop LaMa a une marge supplémentaire autour de la bbox.
    Cette fonction fait le décalage.
    """
    if not text_regions:
        return []

    dx = det_x1 - crop_x1
    dy = det_y1 - crop_y1

    result = []
    for region in text_regions:
        pts = region.get('bbox')
        if not pts:
            continue
        local_pts = []
        for pt in pts:
            lx = int(pt[0]) + dx
            ly = int(pt[1]) + dy
            lx = max(0, min(lx, crop_w - 1))
            ly = max(0, min(ly, crop_h - 1))
            local_pts.append([lx, ly])
        result.append({**region, 'bbox': local_pts})

    return result


def rescale_regions(
    text_regions: Optional[List[Dict]],
    upscale_factor: float,
) -> List[Dict]:
    """Remap les régions OCR quand le crop a été upscalé pour l'OCR."""
    if not text_regions or upscale_factor <= 1.0:
        return text_regions or []

    result = []
    for region in text_regions:
        if not isinstance(region, dict):
            result.append(region)
            continue
        pts = region.get('bbox')
        if pts:
            pts = [[p[0] / upscale_factor, p[1] / upscale_factor] for p in pts]
        result.append({**region, 'bbox': pts})

    return result


def build_ocr_polygon_mask(
    crop_h: int,
    crop_w: int,
    text_regions: Optional[List[Dict]],
) -> np.ndarray:
    """
    Masque OCR brut (polygones individuels, sans enveloppe convexe).
    Utilisé uniquement pour le debug et l'extraction de couleur de texte.
    """
    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    if not text_regions:
        return mask

    for region in text_regions:
        pts = region.get('bbox')
        if not pts:
            continue
        arr = np.array(pts, dtype=np.int32)
        if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
            continue
        arr[:, 0] = np.clip(arr[:, 0], 0, max(0, crop_w - 1))
        arr[:, 1] = np.clip(arr[:, 1], 0, max(0, crop_h - 1))
        cv2.fillPoly(mask, [arr], 255)

    return mask
