# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════════
FORME DU BALLON PAR CROISSANCE DEPUIS L'ENCRE
═══════════════════════════════════════════════════════════════════════════════

Déduit la surface intérieure d'un ballon en partant de l'encre du texte et en
irradiant vers l'extérieur jusqu'à heurter le trait du ballon.

POURQUOI PAS LA TEINTE
`TextRenderer._bubble_mask_from_image` cherche « la plus grande zone homogène »
autour de la teinte médiane du centre. C'est fiable quand le ballon tranche sur
le décor, mais la teinte de référence n'a pas de sens absolu : sur un ballon
sombre posé sur un décor lui-même sombre, la zone homogène peut fuir à travers
le contour et produire une coulée en biais.

La croissance depuis l'encre ne dépend d'aucune teinte de référence. Elle
s'appuie sur le GRADIENT, qui marque une frontière quelle que soit sa polarité :
un trait clair sur fond sombre et un trait sombre sur fond clair produisent tous
deux une arête.

GARDE-FOUS
1. La croissance ne sort JAMAIS du crop (plafond dur : pas d'inondation de la
   planche sur un texte libre, qui n'a aucun contour pour l'arrêter).
2. Un texte LIBRE se reconnaît au REMPLISSAGE FINAL, pas au contact avec les
   bords. Mesuré sur path-of-vengeance :

       ballons légitimes (12 cas)  : 60 à 78 % de la bbox
       textes libres    (6 cas)    : 86 à 98 %

   L'écart est franc et sans recouvrement, d'où le seuil à 82 %.

   Le contact avec un bord, lui, ne sépare RIEN : la bbox étant dessinée autour
   du ballon, un ballon légitime touche ses bords par construction. Mesuré :
   pov#23 (ballon ordinaire) recouvre 48 % d'un bord, pov#06 (bulle de cri)
   51 % — un critère de bord classait le premier en fuite et amputait son
   masque de 78 % à 48 % de remplissage.
3. Au-delà du seuil, la fonction rend None avec le mode `texte_libre` : il n'y a
   pas de ballon à décrire, et l'appelant doit router vers le régime out_text,
   qui met en page sur les polygones de ligne et n'a besoin d'aucune forme.
4. Un remplissage anormalement bas fait aussi renoncer la fonction, pour que
   l'appelant garde la main sur son propre repli.

Aucun seuil n'est exprimé en valeur de pixel : le seuil d'arête est un centile
du gradient du crop lui-même, donc il se recalibre sur chaque image.
"""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


# Part du gradient considérée comme « mur ». Un centile plutôt qu'une valeur
# absolue : le contraste varie considérablement d'une planche à l'autre.
WALL_PERCENTILE = 88.0

# Rayon autour de l'encre où les murs sont neutralisés. Le texte est lui-même
# plein d'arêtes ; sans ça la croissance resterait prisonnière des glyphes.
INK_CLEARANCE = 9

# Bornes de plausibilité du résultat. Le plafond sépare un ballon d'un texte
# libre — voir la mesure dans l'en-tête du module.
MIN_FILL = 0.12
MAX_FILL = 0.82


def _walls(crop_erased: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """Carte booléenne des frontières infranchissables."""
    gray = cv2.cvtColor(crop_erased, cv2.COLOR_BGR2GRAY)
    # Lisse le grain d'impression sans émousser les traits.
    gray = cv2.bilateralFilter(gray, 7, 40, 40)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    walls = mag >= float(np.percentile(mag, WALL_PERCENTILE))

    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * INK_CLEARANCE + 1, 2 * INK_CLEARANCE + 1)
    )
    walls[cv2.dilate(ink, k) > 0] = False
    return walls


def _touched_borders(mask: np.ndarray) -> int:
    return (
        int(mask[0, :].any())
        + int(mask[-1, :].any())
        + int(mask[:, 0].any())
        + int(mask[:, -1].any())
    )


def _close_contour(mask: np.ndarray, hull: bool = False) -> np.ndarray:
    """Plus grande composante, refermée — enveloppe convexe si demandé."""
    h, w = mask.shape[:2]
    out = np.zeros((h, w), np.uint8)
    cnts, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not cnts:
        return out
    if hull:
        cv2.drawContours(out, [cv2.convexHull(np.vstack(cnts))], -1, 255, -1)
    else:
        cv2.drawContours(out, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return out


def grow_from_ink(
    crop_erased: np.ndarray,
    ink: np.ndarray,
    max_steps: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], Dict]:
    """Surface du ballon, ou None si le résultat n'est pas plausible.

    `crop_erased` : le crop APRÈS effacement — le trait du ballon y subsiste,
    le texte non. Le faire sur l'image d'origine ferait prendre les glyphes pour
    des murs et la croissance ne sortirait jamais du bloc de texte.
    `ink` : masque de l'encre du texte SOURCE, lu sur l'image d'ORIGINE.

    Le dictionnaire de diagnostic porte toujours `mode`, `steps`, `fill`.
    """
    diag: Dict = {"mode": "?", "steps": 0, "fill": 0.0, "leak_step": None}

    if crop_erased is None or crop_erased.size == 0:
        diag["mode"] = "crop_vide"
        return None, diag
    h, w = crop_erased.shape[:2]
    if min(h, w) < 24:
        diag["mode"] = "crop_trop_petit"
        return None, diag

    ink2 = ink if ink.ndim == 2 else ink[:, :, 0]
    if int(np.count_nonzero(ink2)) < 32:
        diag["mode"] = "encre_insuffisante"
        return None, diag

    free = ~_walls(crop_erased, ink2)
    grown = ink2 > 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    steps = int(max_steps or (max(h, w) + 8))

    # On laisse la croissance CONVERGER. Elle s'arrête d'elle-même sur le trait
    # du ballon — vérifié y compris sur une bulle de cri dentelée, qui converge
    # à 60 % de remplissage sans jamais s'échapper. C'est le remplissage final,
    # et lui seul, qui dira ensuite s'il y avait un ballon.
    for step in range(steps):
        suivant = (cv2.dilate(grown.astype(np.uint8), kernel) > 0) & free
        if int(suivant.sum()) == int(grown.sum()):
            diag["mode"] = "cloture"
            diag["steps"] = step
            break
        grown = suivant
    else:
        diag["mode"] = "steps_epuises"
        diag["steps"] = steps

    diag["border_cov"] = round(float(max(
        grown[0, :].mean(), grown[-1, :].mean(),
        grown[:, 0].mean(), grown[:, -1].mean(),
    )), 3)

    mask = _close_contour(grown, hull=False)

    fill = float(np.count_nonzero(mask)) / float(h * w)
    diag["fill"] = round(fill, 4)

    if fill > MAX_FILL:
        # Rien n'a arrêté la croissance : il n'y a pas de contour fermé, donc
        # pas de ballon. Ce n'est PAS un échec — c'est l'information que ce
        # texte relève du régime out_text.
        diag["mode"] = "texte_libre"
        return None, diag
    if fill < MIN_FILL:
        diag["mode"] = diag["mode"] + "_trop_petit"
        return None, diag
    return mask, diag


def ink_mask_from_regions(crop_orig: np.ndarray, regions) -> np.ndarray:
    """Encre RÉELLE des glyphes, bornée aux polygones de ligne OCR.

    Remplir les polygones donnerait un tiers de la boîte au lieu de l'encre —
    la croissance partirait alors déjà au contact des bords.
    """
    h, w = crop_orig.shape[:2]
    poly = np.zeros((h, w), np.uint8)
    for r in (regions or []):
        pts = r.get("bbox") if isinstance(r, dict) else None
        if not pts or len(pts) < 3:
            continue
        arr = np.array([[int(p[0]), int(p[1])] for p in pts], np.int32)
        arr[:, 0] = np.clip(arr[:, 0], 0, w - 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, h - 1)
        cv2.fillPoly(poly, [arr], 255)
    if int(np.count_nonzero(poly)) < 64:
        return poly

    gray = cv2.cvtColor(crop_orig, cv2.COLOR_BGR2GRAY)
    dedans = gray[poly > 0]
    thr, _ = cv2.threshold(dedans, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # L'encre est le côté MINORITAIRE : ni « toujours sombre » ni « toujours
    # clair », le lettrage blanc sur ballon noir est courant.
    sombre = float(np.mean(dedans < thr)) <= 0.5
    encre = ((gray < thr) if sombre else (gray >= thr)) & (poly > 0)
    return encre.astype(np.uint8) * 255


def has_closed_bubble(img_orig, img_erased, bbox, regions) -> Optional[bool]:
    """Y a-t-il un contour FERMÉ autour de ce texte ?

    Trois réponses : True (ballon), False (texte libre), None (indécidable —
    pas assez d'encre, crop dégénéré, erreur). L'appelant ne doit changer de
    comportement que sur False : c'est le seul cas MESURÉ, les deux autres
    laissent la main aux heuristiques existantes.

    Les DEUX images sont nécessaires. L'encre se lit sur l'ORIGINE, où le texte
    source existe encore ; les murs se lisent sur l'EFFACÉE, où le trait du
    ballon subsiste sans les glyphes. Mesuré sur path-of-vengeance, tout lire
    sur l'origine ne donne que 30/36 : le dégagement autour de l'encre perce le
    contour là où le lettrage le frôle, et six vrais ballons passent pour du
    texte libre.
    """
    try:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        h, w = img_orig.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 24 or y2 - y1 < 24:
            return None
        c_o = img_orig[y1:y2, x1:x2]
        c_e = img_erased[y1:y2, x1:x2]
        if c_o.shape[:2] != c_e.shape[:2]:
            return None
        mask, diag = grow_from_ink(c_e, ink_mask_from_regions(c_o, regions))
    except Exception:
        return None

    if diag.get("mode") == "texte_libre":
        return False
    if mask is not None:
        return True
    return None
