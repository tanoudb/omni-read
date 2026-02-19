"""
═══════════════════════════════════════════════════════════════════════════════
RENDERER v6 - LaMa sur crop local + masque OCR + SKIP pour petites bulles
═══════════════════════════════════════════════════════════════════════════════

FIX PERF : LaMa travaille sur un CROP local autour de la bulle (avec marge),
pas sur l'image entière de 690x10000. ~50x plus rapide.

FIX QUALITÉ : Skip inpainting si pas de text_regions OCR (évite d'effacer 
des bulles vides ou des zones mal détectées).

✅ NOUVEAU: Skip inpainting pour bulles < 150px de haut
LaMa crée des artefacts sur micro-texte, on laisse juste le rendu texte.

Pipeline :
1. Extraire un crop local (bbox + marge de 30px)
2. Créer masque OCR en coords locales
3. LaMa inpaint sur le crop (sauf si trop petit)
4. Remettre le crop dans l'image
5. Dessiner le texte traduit
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Optional, List, Dict
from pathlib import Path
import math
import re
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from utils import ImageUtils

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
    """Rendu texte avec LaMa inpainting local"""
    
    SHRINK_RATIO = 0.18
    CROP_MARGIN = 30  # Marge autour de la bbox pour le crop LaMa
    INPAINT_MIN_HEIGHT = 20  # ✅ RÉDUIT pour inpainter aussi les petites bulles

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

            self.anime_inpainter = ModelManager(name="lama", device="cuda" if torch.cuda.is_available() else "cpu")
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
    
    def _load_fonts(self) -> List[str]:
        fonts = []
        for font_path in self.cfg.font_paths:
            if Path(font_path).exists():
                try:
                    ImageFont.truetype(font_path, 24)
                    fonts.append(font_path)
                except:
                    continue
        return fonts
    
    def get_font(self, size: int) -> Optional[ImageFont.FreeTypeFont]:
        for font_path in self.fonts:
            try:
                return ImageFont.truetype(font_path, size)
            except:
                continue
        try:
            return ImageFont.load_default()
        except:
            return None

    def get_font_for_style(self, size: int, style: str = "dialogue", font_hint: str = "regular") -> Optional[ImageFont.FreeTypeFont]:
        if not self.fonts:
            return self.get_font(size)

        style = (style or "dialogue").lower()
        keyword_map = {
            "scream": ["cris", "sfx", "expressive", "trash", "creepy"],
            "whisper": ["lower", "light", "thin"],
            "narration": ["special", "spéciales", "serif"],
            "dialogue": ["bulles", "classiques"],
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

        for font_path in ordered_paths:
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue

        return self.get_font(size)
    
    def _get_inner_zone(self, x1: int, y1: int, x2: int, y2: int, 
                         img_shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
        h_img, w_img = img_shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))
        
        box_w, box_h = x2 - x1, y2 - y1
        sx = max(5, int(box_w * self.SHRINK_RATIO))
        sy = max(5, int(box_h * self.SHRINK_RATIO))
        
        return (x1 + sx, y1 + sy, x2 - sx, y2 - sy)
    
    # ─────────────────────────────────────────────────────────────────────────
    # INPAINTING LOCAL (LaMa sur crop, pas image entière)
    # ─────────────────────────────────────────────────────────────────────────
    
    def inpaint_region(self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                        text_regions: Optional[List[Dict]] = None,
                        class_name: str = "",
                        chirurgical_mask: Optional[np.ndarray] = None,
                        bubble_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Inpainting sur un crop local autour de la bulle.
        
        IMPORTANT : si pas de text_regions OCR, on skip l'inpainting.
        Ça évite d'effacer des bulles vides ou des faux positifs.
        
        ✅ NOUVEAU: Skip inpainting si bulle trop petite (< 150px haut)
        LaMa crée des artefacts sur micro-texte
        """
        h_img, w_img = img.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))

        cls = (class_name or "").strip().lower()
        if cls == "out_text" and not bool(getattr(self.cfg, 'inpaint_out_text', True)):
            print(f"[OUT_TEXT] skip inpaint bbox=({x1},{y1},{x2},{y2})")
            return img
        
        if x2 - x1 < 10 or y2 - y1 < 10:
            return img
        
        # ✅ NOUVEAU: Skip inpainting pour bulles trop petites
        bubble_height = y2 - y1
        if bubble_height < self.INPAINT_MIN_HEIGHT:
            print(f"[SKIP LAMA] petite zone ({bubble_height}px < {self.INPAINT_MIN_HEIGHT}px) bbox=({x1},{y1},{x2},{y2})")
            return img  # Pas d'inpainting, juste du rendu texte
        
        # Pas de régions OCR → pas d'inpainting (on sait pas quoi effacer)
        if not text_regions or len(text_regions) == 0:
            print(f"[SKIP LAMA] pas de mask_regions bbox=({x1},{y1},{x2},{y2})")
            return img
        
        # ── Crop local avec marge ──
        m = self.CROP_MARGIN
        crop_x1 = max(0, x1 - m)
        crop_y1 = max(0, y1 - m)
        crop_x2 = min(w_img, x2 + m)
        crop_y2 = min(h_img, y2 + m)
        
        crop = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        crop_h, crop_w = crop.shape[:2]

        if self._is_white_background(crop):
            print(f"[SKIP LAMA] fond blanc bbox=({x1},{y1},{x2},{y2})")
            return img
        
        # Build OCR mask in crop coords. Prefer provided chirurgical_mask when present.
        mask = np.zeros((crop_h, crop_w), dtype=np.uint8)

        def _place_local_mask(src_mask: np.ndarray, dst_shape, dx, dy):
            if src_mask is None or not isinstance(src_mask, np.ndarray):
                return np.zeros(dst_shape, dtype=np.uint8)
            h_src, w_src = src_mask.shape[:2]
            h_dst, w_dst = dst_shape
            out = np.zeros(dst_shape, dtype=np.uint8)
            sx = max(0, dx)
            sy = max(0, dy)
            ex = min(w_dst, dx + w_src)
            ey = min(h_dst, dy + h_src)
            src_x0 = max(0, -dx)
            src_y0 = max(0, -dy)
            src_x1 = src_x0 + (ex - sx)
            src_y1 = src_y0 + (ey - sy)
            if ex > sx and ey > sy:
                out[sy:ey, sx:ex] = src_mask[src_y0:src_y1, src_x0:src_x1]
            return out

        # Text regions are relative to detection (x1,y1). We need to map them into crop coords (crop_x1, crop_y1)
        if text_regions:
            for region in text_regions:
                bbox_points = region.get('bbox')
                if not bbox_points:
                    continue
                local_points = []
                for pt in bbox_points:
                    lx = int(pt[0]) + (x1 - crop_x1)
                    ly = int(pt[1]) + (y1 - crop_y1)
                    lx = max(0, min(lx, crop_w - 1))
                    ly = max(0, min(ly, crop_h - 1))
                    local_points.append([lx, ly])
                pts = np.array(local_points, dtype=np.int32)
                if pts.shape[0] >= 3:
                    cv2.fillPoly(mask, [pts], 255)

        # If a chirurgical_mask (from pipeline) is provided use it preferentially.
        if isinstance(chirurgical_mask, np.ndarray) and chirurgical_mask.size > 0:
            # chirurgical_mask expected in detection-local coords (h_det, w_det)
            dx = x1 - crop_x1
            dy = y1 - crop_y1
            placed = _place_local_mask(chirurgical_mask, (crop_h, crop_w), dx, dy)
            if np.sum(placed) > 0:
                mask = placed
        
        # Dilater pour couvrir anti-alias
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        if np.sum(mask) == 0:
            return img
        
        # Decide mode based on class_name and background complexity
        # OUT_TEXT: single pass on OCR strict (dilated 12px), blend sigma=1.0
        # SYSTEM: mask = full bbox; skip LaMa if white background (variance <15 and median>220)
        # BUBBLE (default): pass1 = chirurgical_mask (OCR∩SAM), pass2 optional background outside bubble

        def _run_lama_once(mask_use: np.ndarray, sigma_blur: float = 1.5) -> Optional[np.ndarray]:
            try:
                if self.anime_inpainter_ready and self.anime_inpainter is not None:
                    out = self._inpaint_anime(crop, mask_use)
                elif self.lama is not None:
                    out = self._inpaint_lama(crop, mask_use)
                else:
                    out = cv2.inpaint(crop, mask_use, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
                alpha = cv2.GaussianBlur(mask_use, (0, 0), sigmaX=sigma_blur, sigmaY=sigma_blur).astype(np.float32) / 255.0
                alpha = np.clip(alpha, 0.0, 1.0)
                alpha = np.expand_dims(alpha, axis=2)
                blended = (crop.astype(np.float32) * (1.0 - alpha) + out.astype(np.float32) * alpha).astype(np.uint8)
                return blended
            except Exception:
                return None

        # Prepare masks for decisions
        # Build ocr_mask_dilated (in crop coords) if needed
        ocr_mask = mask.copy()
        if np.sum(ocr_mask) == 0 and text_regions:
            # fallback: rebuild from regions
            ocr_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
            for region in text_regions:
                bbox_points = region.get('bbox')
                if not bbox_points:
                    continue
                local_points = []
                for pt in bbox_points:
                    lx = int(pt[0]) + (x1 - crop_x1)
                    ly = int(pt[1]) + (y1 - crop_y1)
                    lx = max(0, min(lx, crop_w - 1))
                    ly = max(0, min(ly, crop_h - 1))
                    local_points.append([lx, ly])
                pts = np.array(local_points, dtype=np.int32)
                if pts.shape[0] >= 3:
                    cv2.fillPoly(ocr_mask, [pts], 255)

        kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        ocr_mask_dilated = cv2.dilate(ocr_mask, kernel_big, iterations=1)

        # Map bubble_mask (detection-local) into crop coords if provided
        bubble_in_crop = None
        if isinstance(bubble_mask, np.ndarray) and bubble_mask.size > 0:
            dx = x1 - crop_x1
            dy = y1 - crop_y1
            bubble_in_crop = _place_local_mask(bubble_mask, (crop_h, crop_w), dx, dy)

        # CASES
        if cls == 'out_text':
            # For out_text: use strict OCR mask (no intersection), dilate 12px, single LaMa pass, sigma=1.0
            kernel_ot = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
            ot_mask = cv2.dilate(ocr_mask, kernel_ot, iterations=1)
            if np.sum(ot_mask) == 0:
                return img
            out_crop = _run_lama_once(ot_mask, sigma_blur=1.0)
            if out_crop is None:
                return img
            img[crop_y1:crop_y2, crop_x1:crop_x2] = out_crop
            return img

        if cls == 'system':
            # system: mask = full bbox
            sys_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
            cv2.rectangle(sys_mask, (x1 - crop_x1, y1 - crop_y1), (x2 - crop_x1 - 1, y2 - crop_y1 - 1), 255, -1)
            # Check white background heuristics
            mask_pixels = crop[sys_mask > 0]
            if mask_pixels.size > 0:
                var = float(np.var(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)[sys_mask > 0]))
                med = float(np.median(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)[sys_mask > 0]))
                if var < 15.0 and med > 220.0:
                    # fill white rectangle in original image coordinates
                    img[crop_y1:crop_y2, crop_x1:crop_x2][sys_mask > 0] = 255
                    return img
            out_crop = _run_lama_once(sys_mask, sigma_blur=1.5)
            if out_crop is None:
                return img
            img[crop_y1:crop_y2, crop_x1:crop_x2] = out_crop
            return img

        # Default / bubble: Pass 1 - chirurgical mask preferred
        pass1_mask = mask.copy()
        if np.sum(pass1_mask) == 0 and isinstance(chirurgical_mask, np.ndarray) and chirurgical_mask.size > 0:
            dx = x1 - crop_x1
            dy = y1 - crop_y1
            pass1_mask = _place_local_mask(chirurgical_mask, (crop_h, crop_w), dx, dy)

        if np.sum(pass1_mask) == 0:
            # fallback to ocr_mask_dilated
            pass1_mask = ocr_mask_dilated.copy()

        # Run pass1
        out_crop = _run_lama_once(pass1_mask, sigma_blur=1.5)
        if out_crop is None:
            # fallback single-pass as before
            out_crop = _run_lama_once(pass1_mask, sigma_blur=3.0)
            if out_crop is None:
                return img

        # Decide on pass2: only if background complex (variance > 30) OR out_text handled earlier
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        var_full = float(np.var(gray_crop))

        do_pass2 = False
        if var_full > 30.0:
            do_pass2 = True

        if do_pass2 and bubble_in_crop is not None:
            # mask = ocr_mask_dilated - bubble_mask (only text outside bubble)
            bg_mask = cv2.bitwise_and(ocr_mask_dilated, cv2.bitwise_not(bubble_in_crop))
            if np.sum(bg_mask) > 0:
                out_crop2 = _run_lama_once(bg_mask, sigma_blur=1.5)
                if out_crop2 is not None:
                    # Merge out_crop and out_crop2: prefer pass2 pixels where mask>0
                    alpha2 = (cv2.GaussianBlur(bg_mask, (0, 0), sigmaX=1.5).astype(np.float32) / 255.0)
                    alpha2 = np.clip(alpha2, 0.0, 1.0)
                    alpha2 = np.expand_dims(alpha2, axis=2)
                    merged = (out_crop.astype(np.float32) * (1.0 - alpha2) + out_crop2.astype(np.float32) * alpha2).astype(np.uint8)
                    img[crop_y1:crop_y2, crop_x1:crop_x2] = merged
                    return img

        # If no pass2 or it failed, place pass1 result
        img[crop_y1:crop_y2, crop_x1:crop_x2] = out_crop
        return img
    
    def _inpaint_lama(self, crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """LaMa sur un crop local (rapide !)"""
        h_orig, w_orig = crop.shape[:2]
        
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask).convert('L')
        
        result_pil = self.lama(crop_pil, mask_pil)
        result = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
        
        # Sécurité taille
        if result.shape[:2] != (h_orig, w_orig):
            result = cv2.resize(result, (w_orig, h_orig), interpolation=cv2.INTER_LANCZOS4)
        
        return result

    def _inpaint_anime(self, crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = self.anime_inpainter(image_rgb, mask, self._anime_config)
        if isinstance(result, np.ndarray):
            return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        return crop

    def _build_local_mask_from_regions(self, width: int, height: int, regions: Optional[List[Dict]]) -> np.ndarray:
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
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
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
        ax1 = max(x1, ax1 - pad)
        ay1 = max(y1, ay1 - pad)
        ax2 = min(x2, ax2 + pad)
        ay2 = min(y2, ay2 + pad)
        return (ax1, ay1, ax2, ay2)

    def extract_original_text_color(
        self,
        img: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
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

        sample = pixels.astype(np.float32)
        if sample.shape[0] > 4000:
            idx = np.random.choice(sample.shape[0], 4000, replace=False)
            sample = sample[idx]

        try:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
            _, labels, centers = cv2.kmeans(sample, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
            labels = labels.flatten()

            counts = [np.sum(labels == i) for i in range(2)]
            sat_scores = []
            for i in range(2):
                b, g, r = centers[i]
                mx = max(float(r), float(g), float(b))
                mn = min(float(r), float(g), float(b))
                sat = (mx - mn) / max(1.0, mx)
                sat_scores.append(sat)

            # Favorise cluster texte (souvent plus saturé/contrasté)
            pick = int(np.argmax(np.array(sat_scores) + 0.15 * (np.array(counts) / max(1, sum(counts)))))
            bgr = centers[pick]
        except Exception:
            bgr = np.median(sample, axis=0)

        return (int(bgr[2]), int(bgr[1]), int(bgr[0]))

    def detect_font_hint(
        self,
        img: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
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

        # Heuristique simple "font detector"
        if edge_density > 0.32:
            return "bold"
        if edge_density < 0.15:
            return "thin"
        return "regular"

    def infer_text_style(self, text: str, box_w: int, box_h: int, class_name: str = "") -> str:
        if not self.cfg.auto_style_typesetting:
            return "dialogue"

        if str(class_name).lower() == "system":
            return "system_card"

        clean = (text or "").strip()
        if not clean:
            return "dialogue"

        upper_ratio = sum(1 for c in clean if c.isupper()) / max(1, sum(1 for c in clean if c.isalpha()))
        if "!" in clean and upper_ratio > 0.45:
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

        # Cas courant: titre + description séparés par virgule ou point
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
    # COULEURS
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_text_colors(self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        box_w, box_h = x2 - x1, y2 - y1
        cx1, cy1 = x1 + box_w // 4, y1 + box_h // 4
        cx2, cy2 = x2 - box_w // 4, y2 - box_h // 4
        
        crop = img[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            crop = img[y1:y2, x1:x2]
        
        bg_color = tuple(int(x) for x in np.median(crop.reshape(-1, 3), axis=0))
        luminosity = sum(bg_color) / 3
        
        if luminosity > self.cfg.luminosity_threshold:
            tc, oc = (0, 0, 0), (255, 255, 255)
        else:
            tc, oc = (255, 255, 255), (0, 0, 0)
        
        return (tc[2], tc[1], tc[0]), (oc[2], oc[1], oc[0])
    
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
        lines, current = [], []
        for word in words:
            test = ' '.join(current + [word])
            try:
                w = font.getbbox(test)[2] - font.getbbox(test)[0]
            except:
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
            except:
                lh = font_size
            sp = int(lh * self.cfg.line_spacing_ratio)
            th = len(lines) * lh + (len(lines) - 1) * sp
            fr = th / uh if uh > 0 else 1
            if abs(fr - self.cfg.target_fill_ratio) < 0.1:
                break
            font_size += self.cfg.font_size_step if fr < self.cfg.target_fill_ratio else -self.cfg.font_size_step
            font_size = max(self.cfg.min_font_size, min(font_size, self.cfg.max_font_size))
        return font_size

    def _fit_font_hard(self, text: str, font_size: int, inner_w: int, inner_h: int) -> Tuple[Optional[ImageFont.FreeTypeFont], int, List[str], int, int]:
        """Ajuste strictement la taille pour éviter tout débordement."""
        fs = max(self.cfg.min_font_size, min(font_size, self.cfg.max_font_size))

        while fs >= self.cfg.min_font_size:
            font = self.get_font(fs)
            if not font:
                return None, fs, [], 0, 0

            wrap_w = max(10, int(inner_w * self.cfg.word_wrap_ratio))
            lines = self.wrap_text(text, font, wrap_w)

            try:
                line_h = font.getbbox("Tg")[3] - font.getbbox("Tg")[1]
            except Exception:
                line_h = fs

            spacing = int(line_h * self.cfg.line_spacing_ratio)
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

        font = self.get_font(self.cfg.min_font_size)
        if not font:
            return None, self.cfg.min_font_size, [], 0, 0
        lines = self.wrap_text(text, font, max(10, int(inner_w * self.cfg.word_wrap_ratio)))
        try:
            line_h = font.getbbox("Tg")[3] - font.getbbox("Tg")[1]
        except Exception:
            line_h = self.cfg.min_font_size
        spacing = int(line_h * self.cfg.line_spacing_ratio)
        return font, self.cfg.min_font_size, lines, line_h, spacing
    
    # ─────────────────────────────────────────────────────────────────────────
    # RENDU
    # ─────────────────────────────────────────────────────────────────────────
    
    def render_text(self, img: np.ndarray, text: str, 
                     x1: int, y1: int, x2: int, y2: int,
                     text_regions: Optional[List[Dict]] = None,
                     mask_regions: Optional[List[Dict]] = None,
                     text_color_rgb: Optional[Tuple[int, int, int]] = None,
                     text_style: str = "dialogue",
                     font_hint: str = "regular",
                     class_name: str = "",
                     chirurgical_mask: Optional[np.ndarray] = None,
                     bubble_mask: Optional[np.ndarray] = None) -> np.ndarray:
        effective_regions = mask_regions if mask_regions else text_regions

        if text_color_rgb is None and self.cfg.preserve_original_text_color:
            text_color_rgb = self.extract_original_text_color(img, x1, y1, x2, y2, effective_regions)

        img = self.inpaint_region(img, x1, y1, x2, y2, text_regions=effective_regions, class_name=class_name,
                      chirurgical_mask=chirurgical_mask, bubble_mask=bubble_mask)
        img = self.insert_text(
            img,
            text,
            x1,
            y1,
            x2,
            y2,
            text_regions=effective_regions,
            text_color_rgb=text_color_rgb,
            text_style=text_style,
            font_hint=font_hint,
            class_name=class_name,
        )
        return img

    def render_text_with_timing(self, img: np.ndarray, text: str,
                                x1: int, y1: int, x2: int, y2: int,
                                text_regions: Optional[List[Dict]] = None,
                                mask_regions: Optional[List[Dict]] = None,
                                text_color_rgb: Optional[Tuple[int, int, int]] = None,
                                text_style: str = "dialogue",
                                font_hint: str = "regular",
                                class_name: str = "",
                                chirurgical_mask: Optional[np.ndarray] = None,
                                bubble_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, float]:
        import time

        effective_regions = mask_regions if mask_regions else text_regions

        if text_color_rgb is None and self.cfg.preserve_original_text_color:
            text_color_rgb = self.extract_original_text_color(img, x1, y1, x2, y2, effective_regions)

        t0 = time.perf_counter()
        img = self.inpaint_region(img, x1, y1, x2, y2, text_regions=effective_regions, class_name=class_name,
                      chirurgical_mask=chirurgical_mask, bubble_mask=bubble_mask)
        inpaint_seconds = max(0.0, time.perf_counter() - t0)

        t1 = time.perf_counter()
        img = self.insert_text(
            img,
            text,
            x1,
            y1,
            x2,
            y2,
            text_regions=effective_regions,
            text_color_rgb=text_color_rgb,
            text_style=text_style,
            font_hint=font_hint,
            class_name=class_name,
        )
        render_text_seconds = max(0.0, time.perf_counter() - t1)
        return img, inpaint_seconds, render_text_seconds
    
    def insert_text(
        self,
        img: np.ndarray,
        text: str,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        text_regions: Optional[List[Dict]] = None,
        text_color_rgb: Optional[Tuple[int, int, int]] = None,
        text_style: str = "dialogue",
        font_hint: str = "regular",
        class_name: str = "",
    ) -> np.ndarray:
        if not text:
            return img

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
                ix1, iy1, ix2, iy2 = self._get_inner_zone(x1, y1, x2, y2, img.shape)
        else:
            ix1, iy1, ix2, iy2 = self._get_inner_zone(x1, y1, x2, y2, img.shape)

        tw, th = ix2 - ix1, iy2 - iy1
        if tw <= 0 or th <= 0:
            return img
        
        if text_color_rgb is None:
            text_color, outline_color = self.get_text_colors(img, x1, y1, x2, y2)
        else:
            text_color = text_color_rgb
            avg_luma = (text_color[0] + text_color[1] + text_color[2]) / 3
            outline_color = (0, 0, 0) if avg_luma > 140 else (255, 255, 255)
        
        bw, bh = x2 - x1, y2 - y1
        if text_style == "dialogue":
            text_style = self.infer_text_style(text, bw, bh)

        if text_style == "system_card":
            text = self._format_system_card_text(text)

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
        font, fs, lines, lh, sp = self._fit_font_hard(text, fs, inner_w, inner_h)
        if font is not None:
            style_font = self.get_font_for_style(fs, text_style, font_hint=font_hint)
            if style_font is not None:
                font = style_font
        if not font:
            return img

        preview = (text or "").replace("\n", " ").strip()
        if len(preview) > 60:
            preview = preview[:60] + "..."
        print(
            f"[RENDER_FS] bbox=({x1},{y1},{x2},{y2}) inner=({inner_w}x{inner_h}) "
            f"fs={fs} lines={len(lines)} lh={lh} sp={sp} style={text_style} text='{preview}'"
        )
        
        img_pil = ImageUtils.cv2_to_pil(img)
        draw = ImageDraw.Draw(img_pil)
        
        total_h = len(lines) * lh + (len(lines) - 1) * sp
        
        if use_locked_mode:
            ys = iy1 + self.cfg.padding_vertical
        elif text_style == "system_card":
            ys = iy1 + self.cfg.padding_vertical
        elif self.cfg.vertical_align == 'center':
            bbox_height = iy2 - iy1
            ys = iy1 + (bbox_height - total_h) // 2
        elif self.cfg.vertical_align == 'top':
            ys = iy1 + self.cfg.padding_vertical
        else:
            ys = iy2 - total_h - self.cfg.padding_vertical
        
        for i, line in enumerate(lines):
            try:
                lw = font.getbbox(line)[2] - font.getbbox(line)[0]
            except:
                lw = len(line) * (fs // 2)
            
            if use_locked_mode:
                xp = ix1 + self.cfg.padding_horizontal
            elif text_style == "system_card":
                xp = ix1 + self.cfg.padding_horizontal
            elif self.cfg.horizontal_align == 'center':
                xp = ix1 + (tw - lw) // 2
            elif self.cfg.horizontal_align == 'left':
                xp = ix1 + self.cfg.padding_horizontal
            else:
                xp = ix2 - lw - self.cfg.padding_horizontal
            
            yp = ys + i * (lh + sp)

            # Clamp final pour ne jamais dessiner hors zone
            xp = max(ix1 + self.cfg.padding_horizontal, min(xp, ix2 - lw - self.cfg.padding_horizontal))
            yp = max(iy1 + self.cfg.padding_vertical, min(yp, iy2 - lh - self.cfg.padding_vertical))
            
            if self.cfg.enable_outline:
                for ox in range(-self.cfg.outline_width, self.cfg.outline_width + 1):
                    for oy in range(-self.cfg.outline_width, self.cfg.outline_width + 1):
                        if ox != 0 or oy != 0:
                            draw.text((xp + ox, yp + oy), line, font=font, fill=outline_color)
            
            draw.text((xp, yp), line, font=font, fill=text_color)
        
        return ImageUtils.pil_to_cv2(img_pil)