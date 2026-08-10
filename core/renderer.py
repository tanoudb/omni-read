"""
═══════════════════════════════════════════════════════════════════════════════
RENDERER v7 - ColorResolver intégré + Fix Contraste Pro + System Holo

V7 CHANGELOG :
- ColorResolver intégré : contraste WCAG réel, suppression outline si >12
- FIX : plus jamais d'outline blanc sur fond blanc (anti-aliasing sale)
- System holographique : blanc + cyan (0,180,255) automatique
- get_text_colors() retourne maintenant 3 valeurs (text, outline, width)
- outline_color=None → rendu SANS outline (propre)
- Suppression de l'ancien bloc bulle hacky avec try/except imbriqués

Base V6 conservée :
- LaMa sur crop local + masque OCR
- Skip inpainting si pas de text_regions
- Font sizing dynamique + style inference
═══════════════════════════════════════════════════════════════════════════════
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Optional, List, Dict
from pathlib import Path
import math
import re
import logging
import torch


import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from utils import ImageUtils
from utils.mask_builder import (
    build_inpainting_mask,
    build_inpainting_mask_bbox_fallback,
    regions_to_crop_coords,
)
from utils.gemini_prompt import FONT_MAP

# ── ColorResolver (inline pour éviter import circulaire) ──────────────────

logger = logging.getLogger(__name__)

def _srgb_to_linear(c: int) -> float:
    s = c / 255.0
    return s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4

def _relative_luminance(color: Tuple[int, int, int]) -> float:
    r, g, b = color
    return 0.2126 * _srgb_to_linear(r) + 0.7152 * _srgb_to_linear(g) + 0.0722 * _srgb_to_linear(b)

def _contrast_ratio(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    l1 = _relative_luminance(c1)
    l2 = _relative_luminance(c2)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)

def _simple_luma(color: Tuple[int, int, int]) -> float:
    return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]

# Constantes ColorResolver
_CONTRAST_PRO_THRESHOLD = 12.0
_CONTRAST_MIN_THRESHOLD = 4.5
_SYSTEM_TEXT: Tuple[int, int, int] = (255, 255, 255)
_SYSTEM_OUTLINE: Tuple[int, int, int] = (0, 180, 255)
_SYSTEM_WIDTH = 3
_BLACK: Tuple[int, int, int] = (0, 0, 0)
_WHITE: Tuple[int, int, int] = (255, 255, 255)


def _resolve_colors(
    img_bgr: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    class_name: str = "",
    text_color_override: Optional[Tuple[int, int, int]] = None,
) -> Tuple[Tuple[int, int, int], Optional[Tuple[int, int, int]], int]:
    """
    Résout (text_color, outline_color_or_None, outline_width).
    outline_color=None → pas d'outline.
    """
    cls = (class_name or "").lower().strip()

    # NB: le style "System" se distingue déjà par sa police dédiée (font_key
    # "SYSTEM" -> FONT_MAP), pas besoin d'un gimmick couleur cyan holographique
    # en plus — jugé peu lisible/moche. On laisse tomber dans la même logique
    # de contraste auto que le reste (noir/blanc selon le fond réel).

    # ── Détection fond ──
    h, w = img_bgr.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    bw, bh = x2c - x1c, y2c - y1c

    if bw > 0 and bh > 0:
        cx1 = x1c + bw // 4
        cy1 = y1c + bh // 4
        cx2 = x2c - bw // 4
        cy2 = y2c - bh // 4
        crop = img_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            crop = img_bgr[y1c:y2c, x1c:x2c]
        if crop.size > 0:
            bg_bgr = np.median(crop.reshape(-1, 3), axis=0)
            bg: Tuple[int, int, int] = (int(bg_bgr[2]), int(bg_bgr[1]), int(bg_bgr[0]))
        else:
            bg = _WHITE
    else:
        bg = _WHITE

    # ── Couleur texte ──
    if text_color_override is not None:
        text_color = text_color_override
    else:
        text_color = _BLACK if _simple_luma(bg) > 128 else _WHITE

    # ── Contraste réel WCAG ──
    cr = _contrast_ratio(text_color, bg)

    # Cas standard "texte noir sur bulle blanche" : jamais d'outline, même si
    # le bruit d'échantillonnage du fond fait légèrement chuter le ratio sous
    # le seuil "Contraste Pro" — un contour sur ce combo classique donne un
    # effet "texte à intérieur blanc" indésirable.
    if _simple_luma(bg) > 200 and _simple_luma(text_color) < 60:
        return text_color, None, 0

    # Cas texte coloré (override / couleur originale préservée) sur fond
    # clair : la couleur unie suffit, pas besoin d'outline disgracieux.
    # Exception : out_text sur artwork a besoin d'outline pour la lisibilité.
    if text_color_override is not None and _simple_luma(bg) > 150 and cls != "out_text":
        cr_override = _contrast_ratio(text_color, bg)
        if cr_override >= 3.0:
            return text_color, None, 0

    # Contraste Pro : si ratio > 12 → PAS d'outline
    if cr >= _CONTRAST_PRO_THRESHOLD:
        return text_color, None, 0

    # Contraste suffisant (4.5-12) → outline discret
    if cr >= _CONTRAST_MIN_THRESHOLD:
        outline = tuple(int(t * 0.3 + b * 0.7) for t, b in zip(text_color, bg))
        # Interdit blanc-sur-blanc
        if _simple_luma(outline) > 200 and _simple_luma(bg) > 200:
            outline = _BLACK if _simple_luma(text_color) > 128 else None
            if outline is None:
                return text_color, None, 0
        return text_color, outline, 2

    # Contraste faible (<4.5) → outline fort
    outline = _BLACK if _simple_luma(text_color) > 128 else _WHITE
    # Interdit blanc-sur-blanc
    if _simple_luma(outline) > 200 and _simple_luma(bg) > 200:
        outline = _BLACK

    # Vérifier que l'outline aide
    cr_outline = _contrast_ratio(outline, bg)
    if cr_outline < 3.0:
        text_color = _WHITE if _simple_luma(bg) > 128 else _BLACK
        outline = _BLACK if text_color == _WHITE else _WHITE

    return text_color, outline, 2


# ── Charger LaMa ──
try:
    from simple_lama_inpainting import SimpleLama
    LAMA_AVAILABLE = True
except ImportError:
    LAMA_AVAILABLE = False
    print("⚠️  simple-lama-inpainting non installé → pip install simple-lama-inpainting")

try:
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class TextRenderer:
    """Rendu texte avec LaMa inpainting local + ColorResolver V2"""

    SHRINK_RATIO = 0.22
    CROP_MARGIN = 30
    INPAINT_MIN_HEIGHT = 20

    @staticmethod
    def _is_white_background(crop_bgr: np.ndarray) -> bool:
        if crop_bgr is None or crop_bgr.size == 0:
            return False
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))
        return mean_val >= 242.0 and std_val <= 14.0

    def __init__(self):
        self.cfg = config.rendering
        self.fonts = self._load_fonts()
        self.lama = None
        self.anime_inpainter = None
        self.anime_inpainter_ready = False

        self._init_anime_inpainter()

        if LAMA_AVAILABLE:
            try:
                print("⏳ Chargement LaMa inpainting...")
                self.lama = SimpleLama()
                print("✅ LaMa chargé !")
            except Exception as e:
                print(f"⚠️  Erreur LaMa: {e}. Fallback cv2.inpaint.")

    def _init_anime_inpainter(self):
        if not HF_AVAILABLE:
            return
        try:
            model_dir = Path(self.cfg.inpainting_model_path)
            model_dir.mkdir(parents=True, exist_ok=True)

            if not any(model_dir.iterdir()):
                snapshot_download(
                    repo_id=self.cfg.inpainting_model_id,
                    local_dir=str(model_dir),
                    local_dir_use_symlinks=False,
                )

            from lama_cleaner.model_manager import ModelManager
            from lama_cleaner.schema import Config as LamaConfig

            self.anime_inpainter = ModelManager(
                name="lama",
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            self._anime_config = LamaConfig(
                hd_strategy="Original",
                ldm_steps=20,
                hd_strategy_crop_margin=64,
                hd_strategy_crop_trigger_size=1024,
                hd_strategy_resize_limit=2048,
            )
            self.anime_inpainter_ready = True
            print("✅ AnimeMangaInpainting prêt (lama-cleaner)")
        except Exception as exc:
            self.anime_inpainter = None
            self.anime_inpainter_ready = False
            print(f"⚠️  AnimeMangaInpainting indisponible: {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # FONTS
    # ─────────────────────────────────────────────────────────────────────────

    def _load_fonts(self) -> List[str]:
        fonts = []
        primary = getattr(self.cfg, 'primary_font_path', '') or ''
        if primary and Path(primary).exists():
            try:
                ImageFont.truetype(primary, 24)
                fonts.append(primary)
            except Exception:
                pass

        for font_path in self.cfg.font_paths:
            if Path(font_path).exists():
                try:
                    ImageFont.truetype(font_path, 24)
                    fonts.append(font_path)
                except Exception:
                    continue
        return fonts

    def get_font(self, size: int) -> Optional[ImageFont.FreeTypeFont]:
        for font_path in self.fonts:
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue
        try:
            return ImageFont.load_default()
        except Exception:
            return None

    def _load_font_from_path(self, path: Optional[str], size: int) -> Optional[ImageFont.FreeTypeFont]:
        """Charge un chemin de police donné, avec repli sur la police générique."""
        if path:
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
        return self.get_font(size)

    def _resolve_font_path(
        self, font_key: Optional[str], style: str = "dialogue", font_hint: str = "regular",
    ) -> Optional[str]:
        """
        Détermine QUEL fichier de police utiliser — appelé une fois AVANT le
        fitting (pas après) pour que la mesure/découpe du texte et le rendu
        final utilisent exactement les mêmes métriques. Un swap de police
        après coup, à la même taille, peut faire chevaucher les lignes si les
        métriques diffèrent (police plus large/haute que celle mesurée).

        Priorité : font_key choisi par le LLM (fichier canonique FONT_MAP,
        cohérent sur toute la série) > heuristique par mots-clés de dossier.
        """
        if font_key:
            p = FONT_MAP.get(font_key)
            if p and Path(p).exists():
                return p

        if not self.fonts:
            return None

        style = (style or "dialogue").lower()
        keyword_map = {
            "scream": ["cris", "sfx", "expressive", "trash", "creepy"],
            "whisper": ["lower", "light", "thin"],
            "narration": ["special", "spéciales", "serif"],
            "dialogue": ["bulles", "classiques"],
            "system_card": ["system", "système", "argone"],
        }
        hint_map = {
            "bold": ["cris", "trash", "expressive", "creepy"],
            "thin": ["lower", "light", "thin", "system"],
            "regular": ["bulles", "classiques", "system"],
        }

        preferred = keyword_map.get(style, keyword_map["dialogue"]) + hint_map.get(font_hint, [])
        ordered_paths = sorted(
            self.fonts,
            key=lambda p: 0 if any(k in str(p).lower() for k in preferred) else 1,
        )

        # Parmi les polices préférées, favoriser Bold/Regular pour la
        # cohérence visuelle et la lisibilité (pas d'Italic aléatoire).
        top_picks = [p for p in ordered_paths if any(k in str(p).lower() for k in preferred)]
        if top_picks:
            for pick in top_picks:
                low = str(pick).lower()
                if "bold" in low and "italic" not in low:
                    return pick
            for pick in top_picks:
                low = str(pick).lower()
                if "regular" in low:
                    return pick
            return top_picks[0]
        return ordered_paths[0] if ordered_paths else None

    # ─────────────────────────────────────────────────────────────────────────
    # INNER ZONE
    # ─────────────────────────────────────────────────────────────────────────

    def _get_inner_zone(
        self, x1: int, y1: int, x2: int, y2: int, img_shape: Tuple[int, ...],
        bubble_mask: Optional[np.ndarray] = None,
    ) -> Tuple[int, int, int, int]:
        h_img, w_img = img_shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))

        box_w, box_h = x2 - x1, y2 - y1
        sx = max(5, int(box_w * self.SHRINK_RATIO))
        sy = max(5, int(box_h * self.SHRINK_RATIO))
        inner_w = max(1, box_w - 2 * sx)
        inner_h = max(1, box_h - 2 * sy)

        # Par défaut : centré sur le centre géométrique de la bbox.
        cx, cy = x1 + box_w / 2.0, y1 + box_h / 2.0

        # Si un masque de bulle précis est dispo, recentrer sur son centre de
        # masse plutôt que sur le centre de la bbox : une bulle avec une queue
        # (pointeur vers le personnage) a une bbox asymétrique, donc un
        # centrage bbox tire visuellement le texte du côté de la queue.
        if bubble_mask is not None and isinstance(bubble_mask, np.ndarray) and bubble_mask.size > 0:
            m = bubble_mask
            if m.ndim == 3:
                m = m[:, :, 0]
            if m.shape[:2] != (box_h, box_w) and box_w > 0 and box_h > 0:
                try:
                    m = cv2.resize(m, (box_w, box_h), interpolation=cv2.INTER_NEAREST)
                except Exception:
                    m = None
            if m is not None:
                ys, xs = np.nonzero(m > 0)
                if xs.size > 200:
                    cx = x1 + float(np.mean(xs))
                    cy = y1 + float(np.mean(ys))

        ix1 = cx - inner_w / 2.0
        iy1 = cy - inner_h / 2.0
        ix2 = ix1 + inner_w
        iy2 = iy1 + inner_h

        # Clamp pour rester dans la bbox d'origine
        if ix1 < x1:
            ix2 += (x1 - ix1); ix1 = x1
        if iy1 < y1:
            iy2 += (y1 - iy1); iy1 = y1
        if ix2 > x2:
            ix1 -= (ix2 - x2); ix2 = x2
        if iy2 > y2:
            iy1 -= (iy2 - y2); iy2 = y2
        ix1, iy1 = max(x1, ix1), max(y1, iy1)
        ix2, iy2 = min(x2, ix2), min(y2, iy2)

        return (int(ix1), int(iy1), int(ix2), int(iy2))

    # ─────────────────────────────────────────────────────────────────────────
    # INPAINTING LOCAL
    # ─────────────────────────────────────────────────────────────────────────

    def inpaint_region(
        self,
        img: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        text_regions: Optional[List[Dict]] = None,
        class_name: str = "",
        chirurgical_mask: Optional[np.ndarray] = None,
        bubble_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Inpainting local :
        1. Crop (bbox + marge)
        2. Masque OCR en coords locales
        3. LaMa inpaint sur le crop
        4. Remettre le crop
        """
        h_img, w_img = img.shape[:2]
        bubble_h = y2 - y1

        # Skip si trop petit
        if bubble_h < self.INPAINT_MIN_HEIGHT:
            return img

        # Skip si pas de régions texte et pas de masque chirurgical
        if not text_regions and chirurgical_mask is None:
            return img

        m = self.CROP_MARGIN
        crop_x1 = max(0, x1 - m)
        crop_y1 = max(0, y1 - m)
        crop_x2 = min(w_img, x2 + m)
        crop_y2 = min(h_img, y2 + m)

        crop = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        crop_h, crop_w = crop.shape[:2]
        if crop_h <= 0 or crop_w <= 0:
            return img

        # Construire le masque local
        if chirurgical_mask is not None and isinstance(chirurgical_mask, np.ndarray) and chirurgical_mask.size > 0:
            # chirurgical_mask est construit local à la bbox de détection (det_h x det_w),
            # PAS aux coordonnées globales de l'image — il faut le replacer dans le
            # repère du crop (qui inclut la marge CROP_MARGIN) avant de l'utiliser.
            det_h, det_w = max(1, y2 - y1), max(1, x2 - x1)
            cm = chirurgical_mask
            if cm.shape[:2] != (det_h, det_w):
                cm = cv2.resize(cm, (det_w, det_h), interpolation=cv2.INTER_NEAREST)
            offset_x = x1 - crop_x1
            offset_y = y1 - crop_y1
            local_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
            dst_x1, dst_y1 = max(0, offset_x), max(0, offset_y)
            dst_x2 = min(crop_w, offset_x + det_w)
            dst_y2 = min(crop_h, offset_y + det_h)
            src_x1, src_y1 = dst_x1 - offset_x, dst_y1 - offset_y
            src_x2, src_y2 = src_x1 + (dst_x2 - dst_x1), src_y1 + (dst_y2 - dst_y1)
            if dst_x2 > dst_x1 and dst_y2 > dst_y1:
                local_mask[dst_y1:dst_y2, dst_x1:dst_x2] = (
                    cm[src_y1:src_y2, src_x1:src_x2] > 0
                ).astype(np.uint8) * 255
        elif text_regions:
            local_mask = self._build_local_mask_from_regions(crop_w, crop_h, text_regions)
            # Offset vers coords crop
            offset_x = x1 - crop_x1
            offset_y = y1 - crop_y1
            local_mask_shifted = np.zeros_like(local_mask)
            for region in text_regions:
                bbox_pts = region.get('bbox') if isinstance(region, dict) else None
                if not bbox_pts:
                    continue
                pts = []
                for pt in bbox_pts:
                    lx = int(pt[0]) + offset_x
                    ly = int(pt[1]) + offset_y
                    lx = max(0, min(lx, crop_w - 1))
                    ly = max(0, min(ly, crop_h - 1))
                    pts.append([lx, ly])
                arr = np.array(pts, dtype=np.int32)
                if arr.shape[0] >= 3:
                    cv2.fillPoly(local_mask_shifted, [arr], 255)
            local_mask = local_mask_shifted

            # Dilater le masque OCR — les polices stylisées/cursives (script,
            # italique) ont des jambages/fioritures qui dépassent souvent le
            # polygone OCR serré, laissant un résidu coloré visible après
            # effacement (ex: "CONGRATULATIONS," en police script). Marge
            # élargie par rapport à l'ancien (7,7), toujours raisonnable vu
            # la marge de crop existante (CROP_MARGIN).
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            local_mask = cv2.dilate(local_mask, kernel, iterations=1)
        else:
            return img

        if np.sum(local_mask) == 0:
            return img

        # Fond blanc pur → simple fill
        if self._is_white_background(crop):
            crop[local_mask > 0] = [255, 255, 255]
            img[crop_y1:crop_y2, crop_x1:crop_x2] = crop
            return img

        # LaMa inpaint
        if self.lama is not None:
            try:
                result = self._inpaint_lama(crop, local_mask)
                img[crop_y1:crop_y2, crop_x1:crop_x2] = self._blend_masked(crop, result, local_mask)
                return img
            except Exception:
                pass

        # Anime inpainter fallback
        if self.anime_inpainter_ready and self.anime_inpainter is not None:
            try:
                result = self._inpaint_anime(crop, local_mask)
                img[crop_y1:crop_y2, crop_x1:crop_x2] = self._blend_masked(crop, result, local_mask)
                return img
            except Exception:
                pass

        # cv2 fallback
        try:
            result = cv2.inpaint(crop, local_mask, 7, cv2.INPAINT_TELEA)
            img[crop_y1:crop_y2, crop_x1:crop_x2] = self._blend_masked(crop, result, local_mask)
        except Exception:
            pass

        return img

    @staticmethod
    def _blend_masked(crop: np.ndarray, result: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Ne remplace QUE les pixels réellement masqués (le texte) par le
        résultat de l'inpainting — le reste du crop reste strictement
        identique à l'original.

        Les modèles d'inpainting (LaMa en particulier) redimensionnent
        parfois l'image en interne pour respecter une taille multiple de 8,
        puis la redimensionnent en sens inverse pour revenir à la taille
        d'origine — cet aller-retour peut décaler l'image ENTIÈRE de
        quelques pixels, y compris les zones non masquées. Sans ce filtre,
        une ligne droite (bordure de case, trait noir) qui traverse le crop
        sans être masquée pouvait ressortir légèrement décalée pile à la
        frontière du crop, visible comme une "marche" sur la ligne.
        """
        if result.shape[:2] != crop.shape[:2]:
            return crop
        m = mask
        if m.ndim == 3:
            m = m[:, :, 0]
        mask_bool = m > 0
        blended = crop.copy()
        blended[mask_bool] = result[mask_bool]
        return blended

    def _inpaint_lama(self, crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h_orig, w_orig = crop.shape[:2]
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask).convert('L')
        result_pil = self.lama(crop_pil, mask_pil)
        result = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
        if result.shape[:2] != (h_orig, w_orig):
            result = cv2.resize(result, (w_orig, h_orig), interpolation=cv2.INTER_LANCZOS4)
        return result

    def _inpaint_anime(self, crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = self.anime_inpainter(image_rgb, mask, self._anime_config)
        if isinstance(result, np.ndarray):
            return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        return crop

    # ─────────────────────────────────────────────────────────────────────────
    # MASQUES
    # ─────────────────────────────────────────────────────────────────────────

    def _build_local_mask_from_regions(
        self, width: int, height: int, regions: Optional[List[Dict]],
    ) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        if not regions:
            return mask
        for region in regions:
            raw = region.get('bbox') if isinstance(region, dict) else None
            if not raw:
                continue
            pts = np.array(raw, dtype=np.int32)
            if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
                continue
            pts[:, 0] = np.clip(pts[:, 0], 0, max(0, width - 1))
            pts[:, 1] = np.clip(pts[:, 1], 0, max(0, height - 1))
            cv2.fillPoly(mask, [pts], 255)
        return mask

    def _compute_anchor_box_from_regions(
        self, x1: int, y1: int, x2: int, y2: int,
        regions: Optional[List[Dict]] = None,
    ) -> Optional[Tuple[int, int, int, int]]:
        if not regions:
            return None
        pts = []
        for region in regions:
            raw = region.get('bbox') if isinstance(region, dict) else None
            if not raw:
                continue
            arr = np.array(raw, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
                continue
            for p in arr:
                pts.append((int(x1 + p[0]), int(y1 + p[1])))

        if len(pts) < 3:
            return None

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax1 = max(x1, min(xs))
        ay1 = max(y1, min(ys))
        ax2 = min(x2, max(xs))
        ay2 = min(y2, max(ys))

        if ax2 - ax1 < 8 or ay2 - ay1 < 8:
            return None

        pad = 2
        return (max(x1, ax1 - pad), max(y1, ay1 - pad), min(x2, ax2 + pad), min(y2, ay2 + pad))

    # ─────────────────────────────────────────────────────────────────────────
    # COULEURS — ColorResolver V2 intégré
    # ─────────────────────────────────────────────────────────────────────────

    def get_text_colors(
        self,
        img: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        class_name: str = "",
        text_color_override: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[Tuple[int, int, int], Optional[Tuple[int, int, int]], int]:
        """
        V7: Retourne (text_color, outline_color_or_None, outline_width).
        Utilise le ColorResolver WCAG avec contraste pro.
        """
        return _resolve_colors(img, x1, y1, x2, y2, class_name, text_color_override)

    # ─────────────────────────────────────────────────────────────────────────
    # EXTRACTION COULEUR ORIGINALE
    # ─────────────────────────────────────────────────────────────────────────

    def extract_original_text_color(
        self,
        img: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        mask_regions: Optional[List[Dict]] = None,
    ) -> Optional[Tuple[int, int, int]]:
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        local_mask = self._build_local_mask_from_regions(crop.shape[1], crop.shape[0], mask_regions)
        if np.sum(local_mask) == 0:
            return None

        pixels = crop[local_mask > 0]
        if pixels.size == 0:
            return None

        # Référence de fond : pixels du crop EN DEHORS du masque de texte
        # (le remplissage de la bulle/boîte), pour savoir quel cluster de
        # k-means est vraiment "le texte" plutôt que de trancher au hasard.
        outside_pixels = crop[local_mask == 0]
        if outside_pixels.size > 0:
            bg_ref = outside_pixels.reshape(-1, 3).astype(np.float32).mean(axis=0)
        else:
            bg_ref = None

        sample = pixels.astype(np.float32)
        if sample.shape[0] > 4000:
            idx = np.random.choice(sample.shape[0], 4000, replace=False)
            sample = sample[idx]

        try:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
            _, labels, centers = cv2.kmeans(sample, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
            labels = labels.flatten()

            counts = [np.sum(labels == i) for i in range(2)]

            if bg_ref is not None:
                # Le texte est le cluster le plus DIFFÉRENT du fond réel de la
                # bulle (fiable même pour du texte noir/blanc peu saturé, où
                # l'ancien départage par saturation était quasi aléatoire).
                dists = [float(np.linalg.norm(centers[i] - bg_ref)) for i in range(2)]
                pick = int(np.argmax(dists))
            else:
                # Pas de référence de fond dispo : retombe sur l'heuristique
                # saturation + minorité de pixels (le texte est en général
                # une minorité de pixels dans un masque de région élargi).
                sat_scores = []
                for i in range(2):
                    b, g, r = centers[i]
                    mx = max(float(r), float(g), float(b))
                    mn = min(float(r), float(g), float(b))
                    sat = (mx - mn) / max(1.0, mx)
                    sat_scores.append(sat)
                minority_bonus = [1.0 - (c / max(1, sum(counts))) for c in counts]
                pick = int(np.argmax(
                    np.array(sat_scores) + 0.15 * np.array(minority_bonus)
                ))
            bgr = centers[pick]
        except Exception:
            bgr = np.median(sample, axis=0)

        return (int(bgr[2]), int(bgr[1]), int(bgr[0]))

    def detect_font_hint(
        self,
        img: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        mask_regions: Optional[List[Dict]] = None,
    ) -> str:
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return "regular"

        mask = self._build_local_mask_from_regions(crop.shape[1], crop.shape[0], mask_regions)
        if np.sum(mask) == 0:
            return "regular"

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 180)

        mask_pixels = np.sum(mask > 0)
        if mask_pixels <= 0:
            return "regular"

        edge_density = float(np.sum((edges > 0) & (mask > 0))) / float(mask_pixels)

        if edge_density > 0.32:
            return "bold"
        if edge_density < 0.15:
            return "thin"
        return "regular"

    # ─────────────────────────────────────────────────────────────────────────
    # STYLE INFERENCE
    # ─────────────────────────────────────────────────────────────────────────

    def infer_text_style(
        self, text: str, box_w: int, box_h: int, class_name: str = "",
    ) -> str:
        if not self.cfg.auto_style_typesetting:
            return "dialogue"

        if str(class_name).lower() == "system":
            return "system_card"

        clean = (text or "").strip()
        if not clean:
            return "dialogue"

        upper_ratio = sum(1 for c in clean if c.isupper()) / max(1, sum(1 for c in clean if c.isalpha()))
        # Scream uniquement si majorité uppercase ET exclamation forte
        # (double !! ou ! en fin + ratio élevé). Un simple "C'EST VRAI !"
        # ne doit pas déclencher la police de cri.
        has_strong_exclamation = '!!' in clean or (clean.endswith('!') and upper_ratio > 0.70)
        if has_strong_exclamation and upper_ratio > 0.60:
            return "scream"
        if clean.startswith("(") or clean.startswith("["):
            return "narration"
        if box_h < 70 and len(clean) > 12:
            return "whisper"
        return "dialogue"

    @staticmethod
    def _format_system_card_text(text: str) -> str:
        if not text:
            return text
        if "\n" in text:
            return text

        compact = re.sub(r"\s+", " ", text).strip()

        # Retour à la ligne après les deux-points (style V3)
        if ":" in compact:
            compact = re.sub(r":\s+(?!\n)", ":\n", compact)
            return compact.strip()

        if "," in compact:
            left, right = compact.split(",", 1)
            if 3 <= len(left.strip()) <= 42 and len(right.strip()) >= 8:
                return f"{left.strip()}\n{right.strip()}"

        if ". " in compact:
            left, right = compact.split(". ", 1)
            if 3 <= len(left.strip()) <= 42 and len(right.strip()) >= 8:
                return f"{left.strip()}\n{right.strip()}"

        return compact

    # ─────────────────────────────────────────────────────────────────────────
    # SIZING
    # ─────────────────────────────────────────────────────────────────────────

    def wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
        raw = (text or "").strip()
        if raw and len(raw) < 30:
            try:
                one_line_w = font.getbbox(raw)[2] - font.getbbox(raw)[0]
            except Exception:
                one_line_w = len(raw) * (font.size // 2)
            if one_line_w <= max_width:
                return [raw]

        words = text.split()
        lines: List[str] = []
        current: List[str] = []
        for word in words:
            test = ' '.join(current + [word])
            try:
                w = font.getbbox(test)[2] - font.getbbox(test)[0]
            except Exception:
                w = len(test) * (font.size // 2)
            if w <= max_width:
                current.append(word)
            else:
                if current:
                    lines.append(' '.join(current))
                current = [word]
        if current:
            lines.append(' '.join(current))
        return lines if lines else [""]

    def calculate_optimal_font_size(self, text: str, bbox_width: int, bbox_height: int) -> int:
        if not self.cfg.enable_dynamic_sizing:
            return max(self.cfg.min_font_size, min(bbox_height // 3, self.cfg.max_font_size))
        nb_chars = len(text)
        if nb_chars == 0:
            return self.cfg.min_font_size
        r = 1.0 - self.SHRINK_RATIO
        area = int(bbox_width * r) * int(bbox_height * r) * self.cfg.target_fill_ratio
        fs = int(math.sqrt(area / (nb_chars * 0.6)))
        if nb_chars < 20:
            fs = int(fs * 1.20)
        return max(self.cfg.min_font_size, min(fs, self.cfg.max_font_size))

    def refine_font_size(self, text: str, font_size: int, bbox_width: int, bbox_height: int) -> int:
        r = 1.0 - self.SHRINK_RATIO
        uw = int(bbox_width * r) - 2 * self.cfg.padding_horizontal
        uh = int(bbox_height * r) - 2 * self.cfg.padding_vertical
        if uw <= 0 or uh <= 0:
            return font_size
        for _ in range(self.cfg.max_iterations):
            font = self.get_font(font_size)
            if not font:
                break
            lines = self.wrap_text(text, font, int(uw * self.cfg.word_wrap_ratio))
            try:
                lh = font.getbbox("Tg")[3] - font.getbbox("Tg")[1]
            except Exception:
                lh = font_size
            sp = int(lh * self.cfg.line_spacing_ratio)
            th = len(lines) * lh + (len(lines) - 1) * sp
            fr = th / uh if uh > 0 else 1
            if abs(fr - self.cfg.target_fill_ratio) < 0.1:
                break
            font_size += self.cfg.font_size_step if fr < self.cfg.target_fill_ratio else -self.cfg.font_size_step
            font_size = max(self.cfg.min_font_size, min(font_size, self.cfg.max_font_size))
        return font_size

    @staticmethod
    def _mask_row_span(mask: np.ndarray, y0: float, y1: float) -> float:
        """
        Largeur RÉELLE (max_x - min_x, en pixels du masque) de la bulle sur
        la bande de lignes [y0, y1) — pas une estimation, une mesure directe
        du masque de segmentation à l'endroit précis où une ligne de texte
        sera dessinée. Retourne 0 si la bande ne contient aucun pixel opaque
        (hors de la bulle, ou bande vide).
        """
        h = mask.shape[0]
        y0c = max(0, min(h, int(round(y0))))
        y1c = max(y0c + 1, min(h, int(round(y1))))
        band = mask[y0c:y1c, :]
        xs = np.nonzero(band > 0)[1]
        if xs.size == 0:
            return 0.0
        return float(xs.max() - xs.min())

    def _wrap_text_by_mask(
        self, text: str, font: ImageFont.FreeTypeFont,
        inner_w: int, inner_h: int, line_h: int, spacing: int,
        bubble_mask: np.ndarray, mask_y_offset: float,
    ) -> List[str]:
        """
        Wrap "intelligent" : la largeur disponible pour CHAQUE ligne est
        calculée par une mesure directe du masque réel de la bulle à la
        position verticale où cette ligne sera effectivement dessinée
        (le bloc de texte est centré verticalement dans la bulle) — pas une
        approximation par bandes fixes, un calcul sur la géométrie réelle.
        """
        # Passage 1 (grossier, rectangle) : juste pour savoir où le bloc de
        # texte sera centré verticalement avant de connaître son vrai wrap.
        rough_w = max(10, int(inner_w * self.cfg.word_wrap_ratio))
        n_est = max(1, len(self.wrap_text(text, font, rough_w)))
        total_h_est = n_est * line_h + max(0, n_est - 1) * spacing
        ys_local = max(0, (inner_h - total_h_est) // 2)

        def _width_for_line(idx: int) -> int:
            y0 = mask_y_offset + ys_local + idx * (line_h + spacing)
            y1 = y0 + line_h
            w = self._mask_row_span(bubble_mask, y0, y1)
            if w <= 0:
                return rough_w
            return max(10, int(w * self.cfg.word_wrap_ratio))

        words = (text or "").split()
        if not words:
            return [""]

        lines: List[str] = []
        current: List[str] = []
        for word in words:
            test = ' '.join(current + [word])
            try:
                w = font.getbbox(test)[2] - font.getbbox(test)[0]
            except Exception:
                w = len(test) * (font.size // 2)
            if w <= _width_for_line(len(lines)):
                current.append(word)
            else:
                if current:
                    lines.append(' '.join(current))
                current = [word]
        if current:
            lines.append(' '.join(current))
        return lines if lines else [""]

    def _fit_font_hard(
        self, text: str, font_size: int, inner_w: int, inner_h: int,
        bubble_mask: Optional[np.ndarray] = None, class_name: str = "",
        font_path: Optional[str] = None, mask_y_offset: float = 0,
    ) -> Tuple[Optional[ImageFont.FreeTypeFont], int, List[str], int, int]:
        """
        Ajuste strictement la taille pour éviter tout débordement.
        Mesure/découpe avec `font_path` (la police RÉELLEMENT utilisée au
        rendu) — mesurer avec une police puis dessiner avec une autre à la
        même taille peut faire chevaucher les lignes si les métriques (largeur
        de glyphe, hauteur de ligne) diffèrent entre les deux polices.
        """
        fs = max(self.cfg.min_font_size, min(font_size, self.cfg.max_font_size))

        # Wrap "forme de bulle" (calcul direct sur le masque réel) seulement
        # pour les vraies bulles rondes — une boîte System/narration est déjà
        # rectangulaire, pas besoin.
        use_mask_wrap = (
            str(class_name).lower().strip() == "bulle"
            and isinstance(bubble_mask, np.ndarray)
            and bubble_mask.size > 0
            and bubble_mask.shape[0] >= 20
        )

        while fs >= self.cfg.min_font_size:
            font = self._load_font_from_path(font_path, fs)
            if not font:
                return None, fs, [], 0, 0

            try:
                line_h = font.getbbox("Tg")[3] - font.getbbox("Tg")[1]
            except Exception:
                line_h = fs
            spacing = int(line_h * self.cfg.line_spacing_ratio)

            if use_mask_wrap:
                lines = self._wrap_text_by_mask(
                    text, font, inner_w, inner_h, line_h, spacing,
                    bubble_mask, mask_y_offset,
                )
            else:
                wrap_w = max(10, int(inner_w * self.cfg.word_wrap_ratio))
                lines = self.wrap_text(text, font, wrap_w)

            total_h = len(lines) * line_h + max(0, len(lines) - 1) * spacing

            line_widths = []
            for line in lines:
                try:
                    line_widths.append(font.getbbox(line)[2] - font.getbbox(line)[0])
                except Exception:
                    line_widths.append(len(line) * max(1, fs // 2))

            max_line_w = max(line_widths) if line_widths else 0

            if total_h <= inner_h and max_line_w <= inner_w:
                return font, fs, lines, line_h, spacing

            fs -= self.cfg.font_size_step

        font = self._load_font_from_path(font_path, self.cfg.min_font_size)
        if not font:
            return None, self.cfg.min_font_size, [], 0, 0
        logger.warning(
            "⚠️ Texte forcé à min_font_size=%d — débordement possible : %.50s",
            self.cfg.min_font_size, text,
        )
        lines = self.wrap_text(text, font, max(10, int(inner_w * self.cfg.word_wrap_ratio)))
        try:
            line_h = font.getbbox("Tg")[3] - font.getbbox("Tg")[1]
        except Exception:
            line_h = self.cfg.min_font_size
        spacing = int(line_h * self.cfg.line_spacing_ratio)
        return font, self.cfg.min_font_size, lines, line_h, spacing

    # ─────────────────────────────────────────────────────────────────────────
    # RENDU PRINCIPAL
    # ─────────────────────────────────────────────────────────────────────────

    def render_text(
        self,
        img: np.ndarray,
        text: str,
        x1: int, y1: int, x2: int, y2: int,
        text_regions: Optional[List[Dict]] = None,
        mask_regions: Optional[List[Dict]] = None,
        text_color_rgb: Optional[Tuple[int, int, int]] = None,
        text_style: str = "dialogue",
        font_hint: str = "regular",
        class_name: str = "",
        chirurgical_mask: Optional[np.ndarray] = None,
        bubble_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        effective_regions = mask_regions if mask_regions else text_regions

        if text_color_rgb is None and self.cfg.preserve_original_text_color:
            text_color_rgb = self.extract_original_text_color(img, x1, y1, x2, y2, effective_regions)

        img = self.inpaint_region(
            img, x1, y1, x2, y2,
            text_regions=effective_regions,
            class_name=class_name,
            chirurgical_mask=chirurgical_mask,
            bubble_mask=bubble_mask,
        )
        img = self.insert_text(
            img, text, x1, y1, x2, y2,
            text_regions=effective_regions,
            text_color_rgb=text_color_rgb,
            text_style=text_style,
            font_hint=font_hint,
            class_name=class_name,
        )
        return img

    def render_text_with_timing(
        self,
        img: np.ndarray,
        text: str,
        x1: int, y1: int, x2: int, y2: int,
        text_regions: Optional[List[Dict]] = None,
        mask_regions: Optional[List[Dict]] = None,
        text_color_rgb: Optional[Tuple[int, int, int]] = None,
        text_style: str = "dialogue",
        font_hint: str = "regular",
        class_name: str = "",
        chirurgical_mask: Optional[np.ndarray] = None,
        bubble_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, float]:
        import time

        effective_regions = mask_regions if mask_regions else text_regions

        if text_color_rgb is None and self.cfg.preserve_original_text_color:
            text_color_rgb = self.extract_original_text_color(img, x1, y1, x2, y2, effective_regions)

        t0 = time.perf_counter()
        img = self.inpaint_region(
            img, x1, y1, x2, y2,
            text_regions=effective_regions,
            class_name=class_name,
            chirurgical_mask=chirurgical_mask,
            bubble_mask=bubble_mask,
        )
        inpaint_seconds = max(0.0, time.perf_counter() - t0)

        t1 = time.perf_counter()
        img = self.insert_text(
            img, text, x1, y1, x2, y2,
            text_regions=effective_regions,
            text_color_rgb=text_color_rgb,
            text_style=text_style,
            font_hint=font_hint,
            class_name=class_name,
        )
        render_text_seconds = max(0.0, time.perf_counter() - t1)
        return img, inpaint_seconds, render_text_seconds

    # ─────────────────────────────────────────────────────────────────────────
    # INSERT TEXT — V7 avec ColorResolver intégré
    # ─────────────────────────────────────────────────────────────────────────

    def insert_text(
        self,
        img: np.ndarray,
        text: str,
        x1: int, y1: int, x2: int, y2: int,
        text_regions: Optional[List[Dict]] = None,
        text_color_rgb: Optional[Tuple[int, int, int]] = None,
        text_style: str = "dialogue",
        font_hint: str = "regular",
        class_name: str = "",
        bubble_mask: Optional[np.ndarray] = None,
        font_key: Optional[str] = None,
    ) -> np.ndarray:
        if not text:
            return img

        # ── Locked mode (System OCR regions) ──
        use_locked_mode = bool(getattr(self.cfg, 'lock_text_to_ocr_regions', False))
        system_only = bool(getattr(self.cfg, 'lock_text_system_only', True))
        is_system = str(class_name).lower() == 'system'
        if system_only and not is_system:
            use_locked_mode = False

        if use_locked_mode:
            anchor_box = self._compute_anchor_box_from_regions(x1, y1, x2, y2, text_regions)
            if anchor_box is not None:
                ix1, iy1, ix2, iy2 = anchor_box
            else:
                ix1, iy1, ix2, iy2 = self._get_inner_zone(x1, y1, x2, y2, img.shape, bubble_mask=bubble_mask)
        else:
            ix1, iy1, ix2, iy2 = self._get_inner_zone(x1, y1, x2, y2, img.shape, bubble_mask=bubble_mask)

        tw, th = ix2 - ix1, iy2 - iy1
        if tw <= 0 or th <= 0:
            return img

        # Normalise bubble_mask à la taille de la bbox (repère dans lequel
        # mask_y_offset sera calculé plus bas) — au cas où le masque stocké
        # n'a pas exactement cette résolution.
        mask_for_wrap: Optional[np.ndarray] = None
        if isinstance(bubble_mask, np.ndarray) and bubble_mask.size > 0:
            box_h_full, box_w_full = max(1, y2 - y1), max(1, x2 - x1)
            m = bubble_mask[:, :, 0] if bubble_mask.ndim == 3 else bubble_mask
            if m.shape[:2] != (box_h_full, box_w_full):
                try:
                    m = cv2.resize(m, (box_w_full, box_h_full), interpolation=cv2.INTER_NEAREST)
                except Exception:
                    m = None
            mask_for_wrap = m

        # ── V7: ColorResolver — 3 valeurs, outline peut être None ──
        text_color, outline_color, outline_width = self.get_text_colors(
            img, x1, y1, x2, y2,
            class_name=class_name,
            text_color_override=text_color_rgb,
        )

        # ── Style inference ──
        bw, bh = x2 - x1, y2 - y1
        if text_style == "dialogue":
            text_style = self.infer_text_style(text, bw, bh, class_name=class_name)

        if text_style == "system_card":
            text = self._format_system_card_text(text)

        # ── Uppercase auto pour les fonts comics ──
        # Les polices BD/comics (CCWildWords, AnimeAce, etc.) sont conçues
        # pour du ALL CAPS. Le texte mixte (minuscules) donne un rendu amateur.
        # Exception : whisper (chuchotement) et system_card (interface).
        if text_style not in ("whisper", "system_card"):
            text = text.upper()

        # ── Font sizing ──
        fs = self.calculate_optimal_font_size(text, bw, bh)
        if text_style == "scream":
            fs = min(self.cfg.max_font_size, int(fs * 1.15))
        elif text_style == "whisper":
            fs = max(self.cfg.min_font_size, int(fs * 0.90))
        elif text_style == "system_card":
            fs = max(self.cfg.min_font_size, int(fs * 0.92))

        if self.cfg.enable_dynamic_sizing:
            fs = self.refine_font_size(text, fs, bw, bh)

        inner_w = max(10, tw - 2 * self.cfg.padding_horizontal)
        inner_h = max(10, th - 2 * self.cfg.padding_vertical)
        resolved_font_path = self._resolve_font_path(font_key, text_style, font_hint)
        # Offset de l'inner zone par rapport au haut de la bbox — bubble_mask
        # est indexé dans le repère de la bbox (0 = y1), pas de l'inner zone.
        mask_y_offset = iy1 - y1
        font, fs, lines, lh, sp = self._fit_font_hard(
            text, fs, inner_w, inner_h, bubble_mask=mask_for_wrap, class_name=class_name,
            font_path=resolved_font_path, mask_y_offset=mask_y_offset,
        )

        if not font:
            return img

        # ── Dessin PIL ──
        img_pil = ImageUtils.cv2_to_pil(img)
        draw = ImageDraw.Draw(img_pil)

        total_h = len(lines) * lh + (len(lines) - 1) * sp

        # Y start
        if use_locked_mode or text_style == "system_card":
            ys = iy1 + self.cfg.padding_vertical
        elif self.cfg.vertical_align == 'center':
            ys = iy1 + self.cfg.padding_vertical + (inner_h - total_h) // 2
        elif self.cfg.vertical_align == 'top':
            ys = iy1 + self.cfg.padding_vertical
        else:
            ys = iy2 - total_h - self.cfg.padding_vertical

        for i, line in enumerate(lines):
            try:
                lw = font.getbbox(line)[2] - font.getbbox(line)[0]
            except Exception:
                lw = len(line) * (fs // 2)

            # X position
            if use_locked_mode or text_style == "system_card":
                xp = ix1 + self.cfg.padding_horizontal
            elif self.cfg.horizontal_align == 'center':
                xp = ix1 + self.cfg.padding_horizontal + (inner_w - lw) // 2
            elif self.cfg.horizontal_align == 'left':
                xp = ix1 + self.cfg.padding_horizontal
            else:
                xp = ix2 - lw - self.cfg.padding_horizontal

            yp = ys + i * (lh + sp)

            # Clamp
            xp = max(ix1 + self.cfg.padding_horizontal, min(xp, ix2 - lw - self.cfg.padding_horizontal))
            yp = max(iy1 + self.cfg.padding_vertical, min(yp, iy2 - lh - self.cfg.padding_vertical))

            # ── V7: Rendu avec ou sans outline ──
            if outline_color is not None and outline_width > 0:
                draw.text(
                    (xp, yp), line, font=font,
                    fill=text_color,
                    stroke_width=outline_width,
                    stroke_fill=outline_color,
                )
            else:
                # Pas d'outline → texte propre sans anti-aliasing sale
                draw.text((xp, yp), line, font=font, fill=text_color)

        return ImageUtils.pil_to_cv2(img_pil)