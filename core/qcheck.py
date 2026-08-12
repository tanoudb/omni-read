"""
\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
QCHECK ENGINE \u2014 Post-production auto-repair
\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

Module de contr\u00f4le qualit\u00e9 automatique lanc\u00e9 apr\u00e8s chaque rendu de planche.

D\u00e9tecte et r\u00e9pare :
  1. Overflow   \u2014 texte d\u00e9bordant de la bbox (trop grand pour la bulle)
  2. Ghost      \u2014 fant\u00f4me r\u00e9siduel d'inpainting (texte anglais encore visible)
  3. Contrast   \u2014 couleur de texte trop proche du fond (illisible)
  4. Empty      \u2014 aucun pixel de texte d\u00e9tect\u00e9 apr\u00e8s rendu (rendu silencieux)

Z\u00e9ro appel API : toutes les r\u00e9parations sont locales (cv2, PIL).
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import logging
import json

logger = logging.getLogger(__name__)


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# Helpers
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

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


def _crop_bgr(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return img[y1:y2, x1:x2]


def _median_bg_color(crop: np.ndarray) -> Tuple[int, int, int]:
    """Couleur de fond m\u00e9diane (centre de la zone, format RGB)."""
    if crop is None or crop.size == 0:
        return (255, 255, 255)
    h, w = crop.shape[:2]
    cx1, cy1 = w // 4, h // 4
    cx2, cy2 = w - w // 4, h - h // 4
    center = crop[cy1:cy2, cx1:cx2]
    if center.size == 0:
        center = crop
    bgr = np.median(center.reshape(-1, 3), axis=0)
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))  # BGR->RGB


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# QCheckIssue \u2014 r\u00e9sultat d'une v\u00e9rification
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

class QCheckIssue:
    def __init__(
        self,
        issue_type: str,       # 'overflow' | 'ghost' | 'contrast' | 'empty'
        severity: str,         # 'minor' | 'major' | 'critical'
        det_index: int,
        bbox: Tuple[int, int, int, int],
        class_name: str,
        detail: str,
        repaired: bool = False,
        repair_action: str = "",
    ):
        self.issue_type = issue_type
        self.severity = severity
        self.det_index = det_index
        self.bbox = bbox
        self.class_name = class_name
        self.detail = detail
        self.repaired = repaired
        self.repair_action = repair_action

    def to_dict(self) -> Dict:
        return {
            'type': self.issue_type,
            'severity': self.severity,
            'det_index': self.det_index,
            'bbox': list(self.bbox),
            'class': self.class_name,
            'detail': self.detail,
            'repaired': self.repaired,
            'repair_action': self.repair_action,
        }


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# QCheckEngine
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

class QCheckEngine:
    """
    Moteur de contr\u00f4le qualit\u00e9 post-render.

    Usage:
        engine = QCheckEngine()
        img_fixed, issues = engine.check_and_repair(
            img_before=original_img,      # image AVANT inpainting + rendu
            img_after=translated_img,     # image APRES rendu
            detections=valid_detections,  # liste Detection avec text_translated
            renderer=renderer,            # TextRenderer pour re-render
        )
        engine.save_report(issues, output_dir / "qcheck_report.json", page_name)
    """

    # Seuil de contraste WCAG minimum acceptable
    MIN_CONTRAST_RATIO = 3.0

    def __init__(self, font_paths: Optional[List[str]] = None):
        self.font_paths = font_paths or []
        self._fallback_font = self._load_fallback_font()

    def _load_fallback_font(self) -> Optional[Any]:
        """Police de secours PIL pour le re-render d'urgence."""
        for p in self.font_paths:
            try:
                return ImageFont.truetype(p, 16)
            except Exception:
                continue
        for sys_font in [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
        ]:
            if Path(sys_font).exists():
                try:
                    return ImageFont.truetype(sys_font, 16)
                except Exception:
                    continue
        try:
            return ImageFont.load_default()
        except Exception:
            return None

    # \u2500\u2500 D\u00e9tections \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _detect_overflow(self, img_after, det, det_index):
        """
        D\u00e9tecte si du texte rendu d\u00e9borde de la bbox de la d\u00e9tection.
        Pixels sombres dans la marge ext\u00e9rieure = overflow probable.
        """
        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
        h, w = img_after.shape[:2]

        if x2 - x1 < 4 or y2 - y1 < 4:
            return None

        margin = 10
        outer_x1, outer_y1 = max(0, x1 - margin), max(0, y1 - margin)
        outer_x2, outer_y2 = min(w, x2 + margin), min(h, y2 + margin)

        outer_region = img_after[outer_y1:outer_y2, outer_x1:outer_x2].copy()
        # Effacer la zone int\u00e9rieure pour ne regarder que l'ext\u00e9rieur
        iy1r = y1 - outer_y1
        iy2r = y2 - outer_y1
        ix1r = x1 - outer_x1
        ix2r = x2 - outer_x1
        outer_region[iy1r:iy2r, ix1r:ix2r] = 255

        gray_outer = cv2.cvtColor(outer_region, cv2.COLOR_BGR2GRAY)
        dark_outer = float(np.mean(gray_outer < 80))

        if dark_outer > 0.03:
            return QCheckIssue(
                issue_type='overflow',
                severity='major' if dark_outer > 0.08 else 'minor',
                det_index=det_index,
                bbox=(x1, y1, x2, y2),
                class_name=getattr(det, 'class_name', ''),
                detail=f"D\u00e9bordement texte (densit\u00e9 ext={dark_outer:.1%})",
            )
        return None

    def _detect_ghost(self, img_before, img_after, det, det_index):
        """
        D\u00e9tecte un fant\u00f4me r\u00e9siduel : la zone n'a presque pas chang\u00e9
        mais contient encore des pixels sombres (texte anglais visible).
        """
        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
        before = _crop_bgr(img_before, x1, y1, x2, y2)
        after = _crop_bgr(img_after, x1, y1, x2, y2)

        if before.size == 0 or after.size == 0 or before.shape != after.shape:
            return None

        diff = cv2.absdiff(before, after)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        changed_fraction = float(np.mean(diff_gray > 20))

        if changed_fraction < 0.05:
            after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
            dark_fraction = float(np.mean(after_gray < 80))
            if dark_fraction > 0.05:
                return QCheckIssue(
                    issue_type='ghost',
                    severity='major',
                    det_index=det_index,
                    bbox=(x1, y1, x2, y2),
                    class_name=getattr(det, 'class_name', ''),
                    detail=(
                        f"Fant\u00f4me r\u00e9siduel probable: "
                        f"changement={changed_fraction:.1%}, pixels sombres={dark_fraction:.1%}"
                    ),
                )
        return None

    def _detect_contrast(self, img_after, det, det_index):
        """
        D\u00e9tecte un contraste texte/fond insuffisant (texte illisible).
        """
        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
        crop = _crop_bgr(img_after, x1, y1, x2, y2)
        if crop.size == 0:
            return None

        bg_rgb = _median_bg_color(crop)
        text_color_rgb = getattr(det, 'text_color_rgb', None)
        if text_color_rgb is None:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            luma = float(np.mean(gray))
            text_color_rgb = (0, 0, 0) if luma > 128 else (255, 255, 255)

        cr = _contrast_ratio(text_color_rgb, bg_rgb)
        if cr < self.MIN_CONTRAST_RATIO:
            return QCheckIssue(
                issue_type='contrast',
                severity='major' if cr < 2.0 else 'minor',
                det_index=det_index,
                bbox=(x1, y1, x2, y2),
                class_name=getattr(det, 'class_name', ''),
                detail=f"Contraste insuffisant: ratio={cr:.2f} (min={self.MIN_CONTRAST_RATIO})",
            )
        return None

    def _detect_empty_render(self, img_before, img_after, det, det_index):
        """
        D\u00e9tecte qu'aucun texte n'a \u00e9t\u00e9 rendu : diff avant/apr\u00e8s quasi nulle.
        """
        if not getattr(det, 'text_translated', None):
            return None

        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
        before = _crop_bgr(img_before, x1, y1, x2, y2)
        after = _crop_bgr(img_after, x1, y1, x2, y2)

        if before.size == 0 or after.size == 0 or before.shape != after.shape:
            return None

        diff = cv2.absdiff(before, after)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        changed_fraction = float(np.mean(diff_gray > 15))

        if changed_fraction < 0.01:
            return QCheckIssue(
                issue_type='empty',
                severity='critical',
                det_index=det_index,
                bbox=(x1, y1, x2, y2),
                class_name=getattr(det, 'class_name', ''),
                detail=f"Rendu silencieux (changed={changed_fraction:.1%}) \u2014 aucun pixel modifi\u00e9",
            )
        return None

    # \u2500\u2500 R\u00e9parations \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _repair_ghost(self, img, det):
        """2e passe d'inpainting cv2-TELEA sur la zone du fant\u00f4me."""
        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
        crop = _crop_bgr(img, x1, y1, x2, y2)
        if crop.size == 0:
            return img, "repair_skipped_empty_crop"

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=1)

        if int(np.sum(mask)) == 0:
            return img, "repair_skipped_no_mask"

        try:
            inpainted = cv2.inpaint(crop, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            result = img.copy()
            cy2 = min(img.shape[0], y1 + crop.shape[0])
            cx2 = min(img.shape[1], x1 + crop.shape[1])
            result[y1:cy2, x1:cx2] = inpainted[:cy2-y1, :cx2-x1]
            return result, "ghost_inpainted_telea"
        except Exception as e:
            logger.warning(f"QCheck ghost repair: {e}")
            return img, f"repair_failed: {e}"

    def _repair_overflow(self, img, det, renderer):
        """Re-render avec police 30% plus petite."""
        if renderer is None:
            return img, "repair_skipped_no_renderer"
        text = getattr(det, 'text_translated', None)
        if not text:
            return img, "repair_skipped_no_text"

        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
        original_slh = getattr(det, 'source_line_height', None)
        if original_slh is not None:
            det.source_line_height = float(original_slh) * 0.70

        try:
            result = renderer.insert_text(
                img, text, x1, y1, x2, y2,
                text_regions=getattr(det, 'text_regions', None),
                text_color_rgb=getattr(det, 'text_color_rgb', None),
                text_style=getattr(det, 'text_style', 'dialogue'),
                font_hint=getattr(det, 'font_hint', 'regular'),
                class_name=getattr(det, 'class_name', ''),
                bubble_mask=getattr(det, 'mask_binary', None),
                font_key=getattr(det, 'font_key', None),
                source_line_height=getattr(det, 'source_line_height', None),
                sibling_boxes=[],
            )
            return result, "overflow_rerendered_font_x0.70"
        except Exception as e:
            logger.warning(f"QCheck overflow repair: {e}")
            return img, f"overflow_repair_failed: {e}"
        finally:
            if original_slh is not None:
                det.source_line_height = original_slh

    def _repair_contrast(self, img, det, renderer):
        """Re-render avec couleur de contraste forc\u00e9e (noir/blanc selon fond)."""
        if renderer is None:
            return img, "repair_skipped_no_renderer"
        text = getattr(det, 'text_translated', None)
        if not text:
            return img, "repair_skipped_no_text"

        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
        crop = _crop_bgr(img, x1, y1, x2, y2)
        bg_rgb = _median_bg_color(crop)
        luma_bg = 0.2126 * bg_rgb[0] + 0.7152 * bg_rgb[1] + 0.0722 * bg_rgb[2]
        forced_color = (0, 0, 0) if luma_bg > 128 else (255, 255, 255)

        original_color = getattr(det, 'text_color_rgb', None)
        det.text_color_rgb = forced_color
        try:
            result = renderer.insert_text(
                img, text, x1, y1, x2, y2,
                text_regions=getattr(det, 'text_regions', None),
                text_color_rgb=forced_color,
                text_style=getattr(det, 'text_style', 'dialogue'),
                font_hint=getattr(det, 'font_hint', 'regular'),
                class_name=getattr(det, 'class_name', ''),
                bubble_mask=getattr(det, 'mask_binary', None),
                font_key=getattr(det, 'font_key', None),
                source_line_height=getattr(det, 'source_line_height', None),
                sibling_boxes=[],
            )
            return result, f"contrast_forced_{forced_color}"
        except Exception as e:
            logger.warning(f"QCheck contrast repair: {e}")
            return img, f"contrast_repair_failed: {e}"
        finally:
            det.text_color_rgb = original_color

    def _repair_empty(self, img, det, renderer):
        """Fallback rendu PIL direct si le renderer principal a \u00e9chou\u00e9."""
        text = getattr(det, 'text_translated', None)
        if not text:
            return img, "repair_skipped_no_text"

        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2

        # Tenter d'abord via renderer
        if renderer is not None:
            try:
                result = renderer.insert_text(
                    img, text, x1, y1, x2, y2,
                    text_regions=getattr(det, 'text_regions', None),
                    text_color_rgb=None,
                    text_style='dialogue',
                    font_hint='regular',
                    class_name=getattr(det, 'class_name', ''),
                    bubble_mask=None,
                    font_key=None,
                    source_line_height=None,
                    sibling_boxes=[],
                )
                return result, "empty_rerendered_renderer"
            except Exception:
                pass

        # Fallback PIL pur
        try:
            bw, bh = max(1, x2 - x1), max(1, y2 - y1)
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)

            crop = _crop_bgr(img, x1, y1, x2, y2)
            bg_rgb = _median_bg_color(crop)
            luma = 0.2126 * bg_rgb[0] + 0.7152 * bg_rgb[1] + 0.0722 * bg_rgb[2]
            fill_color = (0, 0, 0) if luma > 128 else (255, 255, 255)

            font_size = max(10, min(bh // 3, bw // max(1, len(text) // 2 + 1)))
            font = self._fallback_font
            if self.font_paths:
                try:
                    font = ImageFont.truetype(self.font_paths[0], font_size)
                except Exception:
                    pass

            draw.text((x1 + 4, y1 + 4), text, font=font, fill=fill_color)
            result_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            return result_bgr, "empty_pil_fallback"
        except Exception as e:
            logger.warning(f"QCheck empty repair PIL: {e}")
            return img, f"empty_repair_failed: {e}"

    # \u2500\u2500 Interface principale \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def check_and_repair(
        self,
        img_before: np.ndarray,
        img_after: np.ndarray,
        detections: list,
        renderer=None,
    ):
        """
        V\u00e9rifie et r\u00e9pare automatiquement les probl\u00e8mes de rendu.

        Args:
            img_before : image ORIGINALE avant tout traitement
            img_after  : image apr\u00e8s inpainting + rendu du texte traduit
            detections : liste Detection avec text_translated rempli
            renderer   : instance TextRenderer pour le re-render

        Returns:
            (img_fixed, issues) \u2014 image corrig\u00e9e + liste QCheckIssue
        """
        img_result = img_after.copy()
        issues: List[QCheckIssue] = []

        for idx, det in enumerate(detections):
            if not getattr(det, 'text_translated', None):
                continue

            det_issues: List[QCheckIssue] = []

            ghost = self._detect_ghost(img_before, img_result, det, idx)
            if ghost:
                det_issues.append(ghost)

            overflow = self._detect_overflow(img_result, det, idx)
            if overflow:
                det_issues.append(overflow)

            contrast = self._detect_contrast(img_result, det, idx)
            if contrast:
                det_issues.append(contrast)

            empty = self._detect_empty_render(img_before, img_result, det, idx)
            if empty:
                det_issues.append(empty)

            # R\u00e9paration dans l'ordre de priorit\u00e9
            for issue in det_issues:
                repaired_img, action = img_result, "no_repair"

                if issue.issue_type == 'ghost':
                    # Pas de reparation auto ici : ce detecteur post-rendu
                    # compare avant/apres et cherche des pixels sombres dans
                    # la bbox -- un texte traduit court/fonce peut matcher ce
                    # profil (peu de changement, pixels sombres presents)
                    # sans etre un fantome du tout. Une reparation aveugle
                    # relancerait un inpainting TELEA PAR-DESSUS un texte
                    # correctement rendu. Le pipeline principal a deja son
                    # propre detecteur de fantome, avant le rendu du texte
                    # (donc jamais confondu avec lui) -- celui-ci reste en
                    # signalement seul, pour verification manuelle ciblee.
                    action = "flagged_not_repaired"
                elif issue.issue_type == 'overflow':
                    repaired_img, action = self._repair_overflow(img_result, det, renderer)
                elif issue.issue_type == 'contrast':
                    repaired_img, action = self._repair_contrast(img_result, det, renderer)
                elif issue.issue_type == 'empty':
                    repaired_img, action = self._repair_empty(img_result, det, renderer)

                if repaired_img is not img_result:
                    img_result = repaired_img
                    issue.repaired = True
                issue.repair_action = action
                issues.append(issue)

        return img_result, issues

    # \u2500\u2500 Rapport \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    @staticmethod
    def save_report(issues: List[QCheckIssue], output_path: Path, page_name: str = "") -> None:
        """Sauvegarde le rapport QCheck en JSON (append si le fichier existe)."""
        if not issues:
            return
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            'page': page_name,
            'total_issues': len(issues),
            'repaired': sum(1 for i in issues if i.repaired),
            'not_repaired': sum(1 for i in issues if not i.repaired),
            'by_type': {},
            'issues': [i.to_dict() for i in issues],
        }
        for issue in issues:
            t = issue.issue_type
            report['by_type'].setdefault(t, {'total': 0, 'repaired': 0})
            report['by_type'][t]['total'] += 1
            if issue.repaired:
                report['by_type'][t]['repaired'] += 1

        try:
            existing = []
            if output_path.exists():
                with open(output_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    existing = data if isinstance(data, list) else [data]
            existing.append(report)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"QCheck rapport non sauvegard\u00e9: {e}")

    @staticmethod
    def format_summary(issues: List[QCheckIssue]) -> str:
        """Resume humain lisible pour les logs."""
        if not issues:
            return "[QCheck] OK - aucun probleme detecte"

        total = len(issues)
        repaired = sum(1 for i in issues if i.repaired)
        types: Dict[str, int] = {}
        for i in issues:
            types[i.issue_type] = types.get(i.issue_type, 0) + 1

        parts = [f"{k}={v}" for k, v in types.items()]
        return (
            f"[QCheck] {total} probleme(s) [{', '.join(parts)}] "
            f"-- {repaired}/{total} auto-repare(s)"
        )
