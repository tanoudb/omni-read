"""
═══════════════════════════════════════════════════════════════════════════════
COLOR RESOLVER — Résolution intelligente des couleurs texte/outline

Règles :
1. "Contraste Pro" : supprime l'outline si contrast_ratio > 12.0
2. Interdit outline blanc sur fond blanc (anti-aliasing = effet sale)
3. Écrans "System" → style holographique (blanc + outline cyan électrique)
4. Luma WCAG 2.1 pour un contraste réel, pas approximatif
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

# Type alias
RGB = Tuple[int, int, int]

# ── Constantes ────────────────────────────────────────────────────────────

# Seuil au-dessus duquel l'outline est inutile (contraste déjà excellent)
CONTRAST_PRO_THRESHOLD = 12.0

# Seuil minimum acceptable (en dessous, on force un outline contrasté)
CONTRAST_MIN_THRESHOLD = 4.5

# Style holographique pour les écrans System
SYSTEM_TEXT_COLOR: RGB = (255, 255, 255)          # Blanc pur
SYSTEM_OUTLINE_COLOR: RGB = (0, 180, 255)         # Bleu électrique / Cyan
SYSTEM_OUTLINE_WIDTH: int = 3

# Couleurs de fallback
BLACK: RGB = (0, 0, 0)
WHITE: RGB = (255, 255, 255)


# ── Fonctions WCAG ───────────────────────────────────────────────────────

def _srgb_to_linear(c: int) -> float:
    """Convertit un canal sRGB [0-255] en luminance linéaire [0-1]."""
    s = c / 255.0
    return s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4


def relative_luminance(color: RGB) -> float:
    """Luminance relative WCAG 2.1 (range 0.0 - 1.0)."""
    r, g, b = color
    return (
        0.2126 * _srgb_to_linear(r)
        + 0.7152 * _srgb_to_linear(g)
        + 0.0722 * _srgb_to_linear(b)
    )


def contrast_ratio(c1: RGB, c2: RGB) -> float:
    """Ratio de contraste WCAG (range 1.0 - 21.0)."""
    l1 = relative_luminance(c1)
    l2 = relative_luminance(c2)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def simple_luma(color: RGB) -> float:
    """Luma rapide (ITU-R BT.709)."""
    return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]


# ── Détection fond ───────────────────────────────────────────────────────

def detect_background_rgb(
    img_bgr: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
) -> RGB:
    """
    Détecte la couleur de fond dominante dans la bbox.
    Échantillonne la zone centrale (50%) pour éviter les bords du texte.
    """
    h, w = img_bgr.shape[:2]
    # Clamp
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return WHITE

    # Zone centrale (50% intérieur)
    cx1 = x1 + bw // 4
    cy1 = y1 + bh // 4
    cx2 = x2 - bw // 4
    cy2 = y2 - bh // 4

    crop = img_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return WHITE

    # Médiane BGR → RGB
    bg_bgr = np.median(crop.reshape(-1, 3), axis=0)
    return (int(bg_bgr[2]), int(bg_bgr[1]), int(bg_bgr[0]))


# ── Résolveur principal ──────────────────────────────────────────────────

def resolve_colors(
    img_bgr: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    class_name: str = "",
    text_color_override: Optional[RGB] = None,
) -> Tuple[RGB, Optional[RGB], int]:
    """
    Résout text_color, outline_color et outline_width pour une détection.

    Returns:
        (text_color, outline_color, outline_width)
        outline_color = None signifie PAS d'outline.
    """
    cls = (class_name or "").lower().strip()

    # ── CAS SPÉCIAL : Écrans System → Style holographique ──────────
    if cls in ("system", "system_card", "sys"):
        return SYSTEM_TEXT_COLOR, SYSTEM_OUTLINE_COLOR, SYSTEM_OUTLINE_WIDTH

    # ── Détection du fond ──────────────────────────────────────────
    bg = detect_background_rgb(img_bgr, x1, y1, x2, y2)

    # ── Choix de la couleur texte ──────────────────────────────────
    if text_color_override is not None:
        text_color = text_color_override
    else:
        # Noir sur fond clair, blanc sur fond sombre
        text_color = BLACK if simple_luma(bg) > 128 else WHITE

    # ── Calcul du contraste réel ───────────────────────────────────
    cr = contrast_ratio(text_color, bg)

    # ── Règle "Contraste Pro" ──────────────────────────────────────
    # Si le contraste texte/fond est déjà excellent → PAS d'outline
    if cr >= CONTRAST_PRO_THRESHOLD:
        return text_color, None, 0

    # ── Contraste suffisant mais pas parfait → outline léger ───────
    if cr >= CONTRAST_MIN_THRESHOLD:
        # Outline discret dans la couleur du fond (fondu)
        outline = _blended_outline(text_color, bg)

        # RÈGLE CRITIQUE : Interdit outline blanc sur fond blanc
        if _is_white_on_white(outline, bg):
            outline = BLACK if simple_luma(text_color) > 128 else None
            if outline is None:
                return text_color, None, 0

        return text_color, outline, 2

    # ── Contraste faible → outline fort contrasté ──────────────────
    outline = BLACK if simple_luma(text_color) > 128 else WHITE

    # RÈGLE CRITIQUE : Jamais outline blanc sur fond blanc
    if _is_white_on_white(outline, bg):
        outline = BLACK

    # Vérifier que l'outline aide vraiment
    cr_outline = contrast_ratio(outline, bg)
    if cr_outline < 3.0:
        # Fallback : inverser tout
        text_color = WHITE if simple_luma(bg) > 128 else BLACK
        outline = BLACK if text_color == WHITE else WHITE

    return text_color, outline, 2


# ── Helpers ───────────────────────────────────────────────────────────────

def _blended_outline(text_color: RGB, bg: RGB) -> RGB:
    """Crée un outline semi-transparent entre texte et fond."""
    return tuple(
        int(t * 0.3 + b * 0.7) for t, b in zip(text_color, bg)
    )  # type: ignore


def _is_white_on_white(color: RGB, bg: RGB) -> bool:
    """
    Détecte si une couleur est "blanche" sur un fond "blanc".
    Seuil : les deux ont une luma > 200.
    """
    return simple_luma(color) > 200 and simple_luma(bg) > 200


def _is_near_bg(color: RGB, bg: RGB, threshold: float = 30.0) -> bool:
    """Vérifie si une couleur est trop proche du fond."""
    return abs(simple_luma(color) - simple_luma(bg)) < threshold


# ── Intégration avec renderer.py ──────────────────────────────────────────

def apply_to_detection(
    img_bgr: np.ndarray,
    detection,  # core.Detection
) -> Tuple[RGB, Optional[RGB], int]:
    """
    Wrapper pour intégrer directement avec le pipeline existant.
    Extrait bbox et class_name depuis un objet Detection.
    """
    x1, y1, x2, y2 = detection.x1, detection.y1, detection.x2, detection.y2
    class_name = getattr(detection, 'class_name', '') or ''

    # Récupérer une couleur overridée si elle existe
    color_override = None
    if hasattr(detection, 'text_color_rgb') and detection.text_color_rgb:
        color_override = detection.text_color_rgb

    return resolve_colors(img_bgr, x1, y1, x2, y2, class_name, color_override)
