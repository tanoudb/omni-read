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
    # La couleur « d'origine » vient d'un k-means à 2 classes sur les pixels du
    # masque OCR. Quand ce masque est imprécis, le cluster retenu peut être
    # celui du FOND : on écrivait alors du texte quasi invisible, et aucune des
    # branches suivantes ne le rattrapait (elles ajoutent un contour, elles ne
    # remplacent jamais la couleur). On rejette donc tout override qui ne
    # ressort pas du fond.
    if text_color_override is not None and _contrast_ratio(tuple(text_color_override), bg) < 2.0:
        text_color_override = None

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
    logger.warning("simple-lama-inpainting non installe -> pip install simple-lama-inpainting")

try:
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class TextRenderer:
    """Rendu texte avec LaMa inpainting local + ColorResolver V2"""

    # Repli seulement : la marge intérieure réelle est décidée par
    # `_shrink_ratio_for()` selon la classe et la disponibilité du masque.
    SHRINK_RATIO = 0.18
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
                logger.info("Chargement LaMa inpainting...")
                self.lama = SimpleLama()
                logger.info("LaMa charge")
            except Exception as e:
                logger.warning("Erreur LaMa: %s. Fallback cv2.inpaint.", e)

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
            logger.info("AnimeMangaInpainting pret (lama-cleaner)")
        except Exception as exc:
            self.anime_inpainter = None
            self.anime_inpainter_ready = False
            logger.warning("AnimeMangaInpainting indisponible: %s", exc)

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

        # Fond uni autour des lettres → remplissage à plat.
        #
        # C'est le cas de loin le plus fréquent : du texte dans une bulle ou une
        # boîte de narration. LaMa, lui, reconstruit une TEXTURE — sur un bloc
        # de plusieurs lignes il laissait un fantôme gris parfaitement lisible
        # du texte d'origine sous la traduction. Un aplat de la couleur du fond
        # local ne peut pas produire d'artefact, et c'est exactement le geste
        # attendu ici. LaMa reste pour le texte posé sur du dessin.
        flat = self._flat_fill_color(crop, local_mask)
        if flat is not None:
            crop[self._extend_fill_mask(crop, local_mask, flat) > 0] = flat
            img[crop_y1:crop_y2, crop_x1:crop_x2] = crop
            return img

        # LaMa inpaint
        if self.lama is not None:
            try:
                result = self._inpaint_lama(crop, local_mask)
                if self._erasure_failed(crop, result, local_mask) and self._background_is_diffusable(crop, local_mask):
                    img[crop_y1:crop_y2, crop_x1:crop_x2] = self._diffuse_fill(crop, local_mask)
                else:
                    img[crop_y1:crop_y2, crop_x1:crop_x2] = self._blend_masked(crop, result, local_mask)
                return img
            except Exception:
                pass

        # Anime inpainter fallback
        if self.anime_inpainter_ready and self.anime_inpainter is not None:
            try:
                result = self._inpaint_anime(crop, local_mask)
                if self._erasure_failed(crop, result, local_mask) and self._background_is_diffusable(crop, local_mask):
                    img[crop_y1:crop_y2, crop_x1:crop_x2] = self._diffuse_fill(crop, local_mask)
                else:
                    img[crop_y1:crop_y2, crop_x1:crop_x2] = self._blend_masked(crop, result, local_mask)
                return img
            except Exception:
                pass

        # cv2 fallback
        try:
            img[crop_y1:crop_y2, crop_x1:crop_x2] = self._diffuse_fill(crop, local_mask)
        except Exception:
            pass

        return img

    @staticmethod
    def _erasure_failed(crop: np.ndarray, result: np.ndarray, mask: np.ndarray) -> bool:
        """
        Détecte un inpainting qui n'a pas vraiment effacé le texte.

        Mesuré sur une carte System (fond en dégradé holographique) : LaMa
        rendait un résultat où le contour des lettres restait lisible — un
        premier essai de détection par écart de pixel moyen (`|après-avant|`)
        s'est révélé aveugle à ça : LaMa avait bien DÉCALÉ la couleur (52
        niveaux de moyenne), mais en préservant la FORME des lettres, donc
        toujours visibles, juste plus pâles. L'écart moyen ne capture pas
        « est-ce que la silhouette du texte a disparu ».
        Ce qui la capture : l'énergie de bord (Laplacien) DANS le masque,
        avant vs après. Mesuré : avant=108, après LaMa=32 (ratio 0.30, texte
        encore net à l'œil), après diffusion pure=6.6 (ratio 0.06, propre).
        """
        if result.shape[:2] != crop.shape[:2] or not mask.any():
            return False
        m = mask > 0
        gray_before = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_after = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.float32)

        edge_before = float(np.mean(np.abs(cv2.Laplacian(gray_before, cv2.CV_32F, ksize=3))[m]))
        # Peu ou pas de contraste net dans le masque au départ (texte déjà
        # discret, halo plutôt que trait) : rien à trancher, on fait confiance
        # au modèle plutôt que de risquer un faux positif.
        if edge_before < 20.0:
            return False

        edge_after = float(np.mean(np.abs(cv2.Laplacian(gray_after, cv2.CV_32F, ksize=3))[m]))
        return (edge_after / edge_before) > 0.18

    @staticmethod
    def _background_is_diffusable(crop: np.ndarray, mask: np.ndarray, margin: int = 25) -> bool:
        """
        Le repli en diffusion pure (`_diffuse_fill`) n'a de sens QUE sur un
        fond lisse (dégradé, halo) : il ne peut reproduire aucune texture,
        juste étaler la couleur des pixels voisins. Sur un fond avec du vrai
        motif (ruban, trame, bordure décorative), il produit de grosses
        taches de couleur — mesuré sur une carte System à bordure dorée avec
        motif : plus visible et plus faux que le fantôme que LaMa avait
        laissé. Autant garder alors le résultat de LaMa, imparfait mais
        crédible, plutôt que d'y substituer un artefact pire.

        Test : le résidu haute fréquence (écart à un flou large) dans la
        couronne juste hors masque. Mesuré : ~11 sur fond en dégradé (sûr
        pour la diffusion), ~17 sur fond à motif réel (pas sûr).
        """
        ring = (cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin, margin))) > 0) & (mask == 0)
        if not ring.any():
            return True
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=5)
        residual = float(np.mean(np.abs(gray - blurred)[ring]))
        return residual < 13.0

    @staticmethod
    def _diffuse_fill(crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Reconstruction par pure diffusion depuis les pixels voisins non
        masqués — aucune hallucination de structure possible, contrairement à
        LaMa. Le bon choix quand la zone à effacer est une variation douce
        (dégradé, halo) : Navier-Stokes converge vers un aplat lisse là où un
        modèle génératif peut « recopier » son entrée si le contexte ne lui
        donne pas prise pour halluciner autre chose.

        Retourne le CROP DÉJÀ RECOMPOSÉ (pas juste le résultat brut à blender
        avec le masque d'origine) : on diffuse ET on recompose sur un masque
        ÉLARGI de quelques pixels. Une première version élargissait seulement
        le calcul de diffusion en gardant `_blend_masked` sur le masque
        étroit d'origine — insuffisant, puisque le vrai résidu visible était
        la frange antialiasée juste à l'EXTÉRIEUR du masque étroit : cette
        frange n'était jamais recomposée (elle restait la valeur d'origine),
        donc son contour restait lisible quelle que soit la qualité de la
        diffusion à l'intérieur du masque étroit lui-même. Il faut élargir la
        zone qu'on RÉÉCRIT, pas seulement celle qu'on regarde pour diffuser.
        """
        extra = max(6, int(round(min(mask.shape[:2]) * 0.02)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * extra + 1, 2 * extra + 1))
        wide_mask = cv2.dilate(mask, kernel)
        result = cv2.inpaint(crop, wide_mask, 20, cv2.INPAINT_NS)
        return TextRenderer._blend_masked(crop, result, wide_mask)

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

    @staticmethod
    def _extend_fill_mask(
        crop: np.ndarray, mask: np.ndarray, flat: np.ndarray, reach: int = 21,
    ) -> np.ndarray:
        """
        Étend le masque, pour un remplissage à plat, à tout ce qui touche le
        texte et s'écarte du fond.

        Deux résidus que le masque strict laissait passer :
        - le halo d'anticrénelage autour des lettres, plus large que l'encre ;
        - la partie d'un glyphe SORTIE de la bbox de détection. Sur un cas réel,
          le W de « NEW » était coupé par la boîte YOLO — l'OCR lisait « NEV »
          et la moitié droite du W restait à l'écran, bien noire.

        Sans risque ici : on écrit la couleur du fond local, donc élargir ne
        peut rien abîmer tant qu'on reste sur des pixels connexes au texte.
        """
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach, reach))
            near = cv2.dilate(mask, kernel, iterations=1) > 0
            # Seuil très bas, et c'est voulu : mesuré sur une boîte de
            # narration, le fond est parfaitement uni (écart-type local 0.000)
            # tandis que le halo d'anticrénelage traîne entre 247 et 254. À un
            # seuil de 8 ces pixels restaient en place et redessinaient le
            # contour des lettres d'origine. `_flat_fill_color` n'a de toute
            # façon accepté ce chemin que parce que le fond est uni : il n'y a
            # pas de texture légitime à préserver ici.
            deviates = np.abs(
                crop.astype(np.int16) - flat.astype(np.int16)
            ).max(axis=2) > 3

            candidate = ((mask > 0) | (near & deviates)).astype(np.uint8)

            # Uniquement les composantes qui touchent le texte détecté : un
            # élément de dessin voisin ne doit pas être repeint au passage.
            n_labels, labels = cv2.connectedComponents(candidate, 8)
            if n_labels <= 1:
                return mask
            touching = np.unique(labels[mask > 0])
            keep = np.isin(labels, touching[touching > 0])
            return keep.astype(np.uint8) * 255
        except Exception:
            return mask

    @staticmethod
    def _flat_fill_color(
        crop: np.ndarray, mask: np.ndarray, max_std: float = 12.0,
    ) -> Optional[np.ndarray]:
        """
        Couleur de remplissage si le fond AUTOUR des lettres est uni, sinon None.

        On échantillonne une couronne juste autour du masque — pas le crop
        entier. L'ancien test portait sur tout le crop (bbox + 30 px de marge),
        qui contient les lettres elles-mêmes et souvent un bout de dessin : il
        ne se déclenchait donc quasiment jamais, même au milieu d'une bulle
        parfaitement blanche.
        """
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            ring = (cv2.dilate(mask, kernel, iterations=1) > 0) & (mask == 0)
            samples = crop[ring]
            if samples.shape[0] < 64:
                return None
            samples = samples.reshape(-1, 3)
            reference = np.median(samples.astype(np.float32), axis=0)

            # Critère robuste plutôt qu'un écart-type : la couronne d'un texte
            # multiligne attrape forcément quelques lettres de la ligne voisine,
            # et ces pixels noirs faisaient exploser l'écart-type — le fond
            # d'une bulle blanche était alors jugé « non uni » et repartait
            # vers LaMa, qui y laissait un fantôme.
            deviation = np.abs(samples.astype(np.float32) - reference).max(axis=1)
            if float(np.mean(deviation <= max_std)) < 0.85:
                return None

            # Mode et non médiane : sur une bulle blanche bruitée par le JPEG,
            # la médiane sort à 254 alors que le fond dominant est à 255. On
            # repeignait donc le texte en 254 sur un fond 255 — un écart d'un
            # seul niveau, mais réparti exactement sur la forme des lettres,
            # donc visible comme un fantôme du texte d'origine.
            inliers = samples[deviation <= max_std]
            if inliers.shape[0] < 32:
                inliers = samples
            mode = np.array([
                int(np.bincount(inliers[:, c], minlength=256).argmax())
                for c in range(3)
            ], dtype=np.uint8)
            return mode.astype(crop.dtype)
        except Exception:
            return None

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
    # MÉTRIQUES DE POLICE
    #
    # `font.getbbox("Tg")` mesure l'ENCRE de "Tg" (hauteur de capitale +
    # jambage), pas la hauteur de ligne typographique. L'utiliser comme
    # interligne serrait les lignes au point que les accents des capitales
    # françaises (É, À, Ê) mordaient sur la ligne du dessus ; et comme
    # `draw.text()` ancre sur l'ASCENDANTE, le bloc était en plus dessiné plus
    # bas que le centre calculé.
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _font_metrics(font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
        """(hauteur de ligne, ascendante) en pixels."""
        try:
            ascent, descent = font.getmetrics()
            return max(1, int(ascent + descent)), int(ascent)
        except Exception:
            size = int(getattr(font, 'size', 16) or 16)
            return max(1, int(size * 1.2)), size

    @staticmethod
    def _line_extents(font: ImageFont.FreeTypeFont, text: str) -> Tuple[int, int]:
        """
        (décalage gauche, largeur d'encre).

        `draw.text((x, y), texte)` dessine l'encre de `x + décalage` à
        `x + décalage + largeur`. Centrer sur la seule largeur, comme avant,
        décale chaque ligne du left side bearing de sa première lettre.
        """
        try:
            box = font.getbbox(text)
            return int(box[0]), max(0, int(box[2] - box[0]))
        except Exception:
            size = int(getattr(font, 'size', 16) or 16)
            return 0, len(text) * max(1, size // 2)

    # ─────────────────────────────────────────────────────────────────────────
    # SIZING
    # ─────────────────────────────────────────────────────────────────────────

    def wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
        raw = (text or "").strip()
        if not raw:
            return [""]

        if len(raw) < 30 and self._line_extents(font, raw)[1] <= max_width:
            return [raw]

        lines: List[str] = []
        # Les retours à la ligne explicites (cartes System) sont respectés.
        for paragraph in raw.split("\n"):
            words = paragraph.split()
            if not words:
                lines.append("")
                continue
            current: List[str] = []
            for word in words:
                test = ' '.join(current + [word])
                if self._line_extents(font, test)[1] <= max_width:
                    current.append(word)
                else:
                    if current:
                        lines.append(' '.join(current))
                    current = [word]
            if current:
                lines.append(' '.join(current))
        return lines if lines else [""]

    def calculate_optimal_font_size(
        self, text: str, inner_w: int, inner_h: int,
        source_line_height: Optional[float] = None,
    ) -> int:
        """
        Estimation de départ, calculée sur la zone RÉELLEMENT utilisable.

        L'ancienne version raisonnait sur la bbox complète (facteur 0.78) alors
        que la zone utile vaut ~0.56 de la bbox : l'estimation partait toujours
        beaucoup trop haut, la boucle d'ajustement devait redescendre par pas
        de 2 et finissait souvent au plancher, où plus rien ne garantissait que
        le texte tenait.

        `source_line_height` (hauteur des lignes du texte ORIGINAL, mesurée sur
        les polygones OCR) borne le résultat : sans elle, on remplit la bulle à
        `target_fill_ratio` quoi qu'il arrive, ce qui transforme un « ENFIN… »
        discret en titre pleine bulle.
        """
        clean = re.sub(r"\s+", " ", text or "").strip()
        n_chars = len(clean)
        if n_chars == 0:
            return self.cfg.min_font_size

        usable = max(1, int(inner_w)) * max(1, int(inner_h)) * float(self.cfg.target_fill_ratio)
        # ~0.62 em² par caractère : largeur moyenne ~0.5 em, hauteur de ligne ~1.25 em.
        fs = int(math.sqrt(usable / (n_chars * 0.62)))

        if source_line_height and source_line_height > 4:
            # La hauteur d'une ligne OCR couvre approximativement la capitale
            # + le jambage, soit ~0.75 em. On VISE la taille d'origine : c'est
            # le geste du letterer. Le texte français est plus long que
            # l'anglais, donc l'ajustement final peut encore réduire — mais
            # jamais grossir au-delà du corps de la planche.
            em_source = float(source_line_height) / 0.75
            fs = min(fs, int(em_source * 1.05))

        return max(self.cfg.min_font_size, min(fs, self.cfg.max_font_size))

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
    ) -> Tuple[List[str], List[int]]:
        """
        Wrap « forme de bulle » : la largeur disponible pour CHAQUE ligne est
        mesurée directement sur le masque de la bulle, à la hauteur où cette
        ligne sera effectivement dessinée.

        Retourne aussi la largeur autorisée de chaque ligne : c'est elle qui
        fait foi pour décider si le texte tient, et non la largeur du rectangle
        `inner_w`. Ce rectangle est volontairement étroit (il doit rester
        inscrit dans l'ovale à l'aveugle) : le comparer aux lignes issues du
        masque déclarait « ne tient pas » à toutes les tailles, et le texte
        tombait systématiquement au plancher.
        """
        rough_w = max(10, int(inner_w * self.cfg.word_wrap_ratio))
        words = re.sub(r"\s+", " ", text or "").strip().split()
        if not words:
            return [""], [rough_w]

        def _wrap_with(n_lines_assumed: int) -> Tuple[List[str], List[int]]:
            total_h = n_lines_assumed * line_h + max(0, n_lines_assumed - 1) * spacing
            ys_local = max(0, (inner_h - total_h) // 2)

            def _width_for_line(idx: int) -> int:
                y0 = mask_y_offset + ys_local + idx * (line_h + spacing)
                w = self._mask_row_span(bubble_mask, y0, y0 + line_h)
                if w <= 0:
                    return rough_w
                return max(10, int(w * self.cfg.word_wrap_ratio))

            lines: List[str] = []
            allowed: List[int] = []
            current: List[str] = []
            current_w = _width_for_line(0)
            for word in words:
                test = ' '.join(current + [word])
                if self._line_extents(font, test)[1] <= current_w:
                    current.append(word)
                else:
                    if current:
                        lines.append(' '.join(current))
                        allowed.append(current_w)
                    current = [word]
                    current_w = _width_for_line(len(lines))
            if current:
                lines.append(' '.join(current))
                allowed.append(current_w)
            return lines, allowed

        # Point fixe : la hauteur du bloc décide d'où commence la première
        # ligne, donc de la bande de masque mesurée pour chaque ligne — mais
        # cette hauteur dépend du découpage qu'on est en train de calculer.
        # Une seule passe sur une estimation rectangulaire mesurait les lignes
        # du bas à des bandes trop hautes (donc trop larges) : elles sortaient
        # de l'ovale par la gauche et la droite.
        n = max(1, len(self.wrap_text(text, font, rough_w)))
        lines, allowed = _wrap_with(n)
        for _ in range(3):
            if len(lines) == n:
                break
            n = len(lines)
            lines, allowed = _wrap_with(n)

        return (lines, allowed) if lines else ([""], [rough_w])

    def _layout_at_size(
        self, text: str, font_size: int, inner_w: int, inner_h: int,
        font_path: Optional[str], use_mask_wrap: bool,
        bubble_mask: Optional[np.ndarray], mask_y_offset: float,
    ) -> Optional[Dict]:
        """Découpe + mesure du bloc de texte à une taille donnée."""
        font = self._load_font_from_path(font_path, font_size)
        if font is None:
            return None

        line_h, ascent = self._font_metrics(font)
        spacing = int(line_h * self.cfg.line_spacing_ratio)

        if use_mask_wrap and bubble_mask is not None:
            lines, allowed = self._wrap_text_by_mask(
                text, font, inner_w, inner_h, line_h, spacing, bubble_mask, mask_y_offset,
            )
            # Chaque ligne est jugée sur la largeur de la bulle À SA hauteur.
            fits_width = all(
                self._line_extents(font, ln)[1] <= aw
                for ln, aw in zip(lines, allowed)
            )
        else:
            lines = self.wrap_text(text, font, max(10, int(inner_w * self.cfg.word_wrap_ratio)))
            fits_width = all(
                self._line_extents(font, ln)[1] <= inner_w for ln in lines
            )

        total_h = len(lines) * line_h + max(0, len(lines) - 1) * spacing
        max_line_w = max((self._line_extents(font, ln)[1] for ln in lines), default=0)

        return {
            'font': font,
            'size': font_size,
            'lines': lines,
            'line_h': line_h,
            'ascent': ascent,
            'spacing': spacing,
            'total_h': total_h,
            'max_line_w': max_line_w,
            'fits': total_h <= inner_h and fits_width,
        }

    def _fit_font_hard(
        self, text: str, font_size: int, inner_w: int, inner_h: int,
        bubble_mask: Optional[np.ndarray] = None, shape_wrap: bool = False,
        font_path: Optional[str] = None, mask_y_offset: float = 0,
        max_font_size: Optional[int] = None,
    ) -> Optional[Dict]:
        """
        Cherche la PLUS GRANDE taille qui tient dans la zone, par dichotomie.

        L'ancienne version ne faisait que décroître depuis l'estimation : quand
        celle-ci était trop basse le texte restait trop petit, et quand elle
        était trop haute il fallait jusqu'à 35 itérations de mesure. La
        dichotomie en fait ~7 et corrige dans les deux sens.

        Mesure et découpe avec `font_path` — la police RÉELLEMENT utilisée au
        rendu. Mesurer avec une police puis dessiner avec une autre à la même
        taille fait déborder ou chevaucher les lignes.
        """
        use_mask_wrap = (
            shape_wrap
            and isinstance(bubble_mask, np.ndarray)
            and bubble_mask.size > 0
            and bubble_mask.shape[0] >= 20
        )

        lo = int(self.cfg.min_font_size)
        hi = int(min(self.cfg.max_font_size, max_font_size or self.cfg.max_font_size))
        hi = max(lo, hi)

        best: Optional[Dict] = None
        while lo <= hi:
            mid = (lo + hi) // 2
            layout = self._layout_at_size(
                text, mid, inner_w, inner_h, font_path, use_mask_wrap, bubble_mask, mask_y_offset,
            )
            if layout is None:
                return None
            if layout['fits']:
                best = layout
                lo = mid + 1
            else:
                hi = mid - 1

        if best is not None:
            return best

        # Rien ne tient, même au plancher : on rend quand même, mais on le dit.
        fallback = self._layout_at_size(
            text, int(self.cfg.min_font_size), inner_w, inner_h,
            font_path, use_mask_wrap, bubble_mask, mask_y_offset,
        )
        if fallback is not None:
            logger.warning(
                "⚠️ Texte au plancher min_font_size=%d, débordement probable : %.60s",
                self.cfg.min_font_size, text,
            )
        return fallback

    # ─────────────────────────────────────────────────────────────────────────
    # ZONE UTILE
    # ─────────────────────────────────────────────────────────────────────────

    def _shrink_ratio_for(self, is_round: bool, has_mask_wrap: bool) -> float:
        """
        Marge intérieure, en fraction de la bbox, de CHAQUE côté.

        Un `SHRINK_RATIO` fixe de 0.22 ne laissait que 56 % de la largeur, puis
        16 px de padding, puis 90 % de wrap : sur une bulle de 215 px il restait
        88 px utiles, moins que le mot « ROYAUMES » au plancher de taille. La
        marge n'a besoin d'être large que quand on inscrit un rectangle dans un
        ovale à l'aveugle ; dès qu'on mesure le masque de la bulle ligne par
        ligne, la forme est déjà respectée.
        """
        if is_round:
            return 0.08 if has_mask_wrap else 0.18
        return 0.06

    def _get_inner_zone(
        self, x1: int, y1: int, x2: int, y2: int, img_shape: Tuple[int, ...],
        bubble_mask: Optional[np.ndarray] = None, shrink: Optional[float] = None,
    ) -> Tuple[int, int, int, int]:
        h_img, w_img = img_shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))

        ratio = self.SHRINK_RATIO if shrink is None else float(shrink)
        box_w, box_h = x2 - x1, y2 - y1
        sx = max(3, int(box_w * ratio))
        sy = max(3, int(box_h * ratio))
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
    # ORIENTATION DU TEXTE SOURCE
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _source_text_angle(regions: Optional[List[Dict]]) -> float:
        """
        Angle (degrés, sens horaire positif) des lignes de texte d'origine,
        déduit des quadrilatères OCR.

        Sans ça, un texte incliné sur un objet du décor (pancarte, feuille de
        papier posée de travers) était réécrit strictement à l'horizontale, ce
        qui casse la perspective du dessin.
        """
        angles: List[float] = []
        for region in regions or []:
            pts = region.get('bbox') if isinstance(region, dict) else None
            if not pts or len(pts) < 4:
                continue
            try:
                arr = np.array(pts, dtype=np.float32)
            except Exception:
                continue
            if arr.ndim != 2 or arr.shape[0] < 4:
                continue
            # Bord supérieur du quadrilatère OCR : les points sont ordonnés
            # (haut-gauche, haut-droite, bas-droite, bas-gauche).
            dx = float(arr[1][0] - arr[0][0])
            dy = float(arr[1][1] - arr[0][1])
            if abs(dx) < 1e-3:
                continue
            angle = math.degrees(math.atan2(dy, dx))
            if abs(angle) <= 45.0:
                angles.append(angle)

        if not angles:
            return 0.0
        angles.sort()
        return float(angles[len(angles) // 2])

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
        font_key: Optional[str] = None,
        source_line_height: Optional[float] = None,
        stroke_color_rgb: Optional[Tuple[int, int, int]] = None,
        stroke_width: Optional[int] = None,
        bg_color_rgb: Optional[Tuple[int, int, int]] = None,
        angle_override: Optional[float] = None,
    ) -> np.ndarray:
        """Efface puis réécrit. `text_regions` = polygones OCR (le texte),
        `mask_regions` = segmentation de la bulle."""
        erase_regions = text_regions or mask_regions

        if text_color_rgb is None and self.cfg.preserve_original_text_color:
            text_color_rgb = self.extract_original_text_color(img, x1, y1, x2, y2, erase_regions)

        img = self.inpaint_region(
            img, x1, y1, x2, y2,
            text_regions=erase_regions,
            class_name=class_name,
            chirurgical_mask=chirurgical_mask,
            bubble_mask=bubble_mask,
        )
        return self.insert_text(
            img, text, x1, y1, x2, y2,
            text_regions=erase_regions,
            text_color_rgb=text_color_rgb,
            text_style=text_style,
            font_hint=font_hint,
            class_name=class_name,
            bubble_mask=bubble_mask,
            font_key=font_key,
            source_line_height=source_line_height,
            stroke_color_rgb=stroke_color_rgb,
            stroke_width=stroke_width,
            bg_color_rgb=bg_color_rgb,
            angle_override=angle_override,
        )

    @staticmethod
    def _bubble_mask_from_image(crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Déduit l'intérieur du ballon depuis l'image elle-même : plus grande
        zone connexe de teinte homogène, contour rebouché.

        `insert_text` reçoit l'image DÉJÀ détourée, donc l'intérieur du ballon
        est uniforme à ce stade — ce qui rend la mesure fiable. L'érosion avant
        l'étiquetage sert à colmater la couronne de hachures des bulles de cri,
        qui est poreuse et laisserait fuir la zone vers l'extérieur.
        """
        h, w = crop_bgr.shape[:2]
        if h < 24 or w < 24:
            return None
        try:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            ref = float(np.median(gray[h // 3:2 * h // 3, w // 3:2 * w // 3]))
            similar = (np.abs(gray.astype(np.int16) - ref) < 40).astype(np.uint8) * 255

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            eroded = cv2.erode(similar, kernel, iterations=1)
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(eroded, 8)
            if n_labels < 2:
                return None

            biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = cv2.dilate((labels == biggest).astype(np.uint8) * 255, kernel, iterations=1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            filled = np.zeros_like(mask)
            cv2.drawContours(filled, contours, -1, 255, -1)
        except Exception:
            return None

        fill = float(np.count_nonzero(filled)) / float(w * h)
        if not (0.20 <= fill <= 0.98):
            return None
        return filled

    @staticmethod
    def _is_non_rectangular(mask: np.ndarray, threshold: float = 0.90) -> bool:
        """
        Vrai si la forme s'écarte franchement de son rectangle englobant.

        Repères mesurés sur la planche : ~0.77 pour une bulle ovale ou une
        bulle de cri, ~1.0 pour un cartouche. Le wrap ligne-par-ligne n'a
        d'intérêt que dans le premier cas.
        """
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return False
        area = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
        if area <= 0:
            return False
        return (float(np.count_nonzero(mask)) / float(area)) < threshold

    @staticmethod
    def _container_box(
        img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Cherche le CONTENANT uni (bulle, boîte de narration, cartouche) dans
        lequel la détection est posée, et renvoie ses limites.

        La bbox de YOLO serre le texte SOURCE. Le français étant plus long que
        l'anglais, l'y enfermer force à réduire la police jusqu'à ce que tout
        rentre : sur une boîte de narration mesurée, le texte tombait à 11 px
        dans une boîte capable d'en accueillir 22, alors que la place existait
        juste à côté. Un letterer utilise la boîte, pas l'empreinte du texte
        qu'il remplace.

        Renvoie None si aucun contenant franc n'est trouvé (texte posé à même
        le dessin) : on reste alors sur la bbox.
        """
        h_img, w_img = img.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        if bw < 20 or bh < 20:
            return None

        px = int(bw * 0.6)
        py = int(bh * 0.9)
        cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
        cx2, cy2 = min(w_img, x2 + px), min(h_img, y2 + py)
        crop = img[cy1:cy2, cx1:cx2]
        if crop.size == 0 or crop.shape[0] < 24 or crop.shape[1] < 24:
            return None

        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            # Référence prise dans la bbox : c'est là qu'est le fond du contenant.
            inner = gray[y1 - cy1:y2 - cy1, x1 - cx1:x2 - cx1]
            if inner.size == 0:
                return None
            ref = float(np.median(inner))
            similar = (np.abs(gray.astype(np.int16) - ref) < 32).astype(np.uint8) * 255

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            eroded = cv2.erode(similar, kernel, iterations=1)
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(eroded, 8)
            if n_labels < 2:
                return None
            biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = cv2.dilate((labels == biggest).astype(np.uint8) * 255, kernel, iterations=1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            filled = np.zeros_like(mask)
            cv2.drawContours(filled, contours, -1, 255, -1)
        except Exception:
            return None

        ys, xs = np.nonzero(filled)
        if xs.size == 0:
            return None
        bx1, by1 = cx1 + int(xs.min()), cy1 + int(ys.min())
        bx2, by2 = cx1 + int(xs.max()), cy1 + int(ys.max())

        # Le contenant doit englober la détection, sinon ce n'en est pas un.
        if bx1 > x1 + 4 or by1 > y1 + 4 or bx2 < x2 - 4 or by2 < y2 - 4:
            return None
        # ...et rester crédible : au-delà, on a probablement attrapé le fond
        # de la case entière.
        if (bx2 - bx1) > bw * 4 or (by2 - by1) > bh * 5:
            return None
        if (bx2 - bx1) <= bw and (by2 - by1) <= bh:
            return None

        return (bx1, by1, bx2, by2)

    def _bubble_shape_mask(
        self, raw_mask: Optional[np.ndarray], crop_bgr: np.ndarray,
        box_w: int, box_h: int, is_bubble: bool,
    ) -> Optional[np.ndarray]:
        """
        Masque de la FORME de la bulle, au repère de la bbox.

        Le masque du segmenter n'est pas celui du ballon : avec le backend
        `hybrid` (le défaut quand le checkpoint SAM2 est absent), il est
        construit à partir des régions OCR puis raffiné — c'est donc un masque
        des LETTRES d'origine. Mesuré sur une vraie bulle : 20 % de la bbox,
        et rien du tout au-dessus ni en dessous du texte anglais. Le wrap
        « forme de bulle » donnait alors un bloc en sablier dont les premières
        et dernières lignes sortaient de l'ovale.

        Ordre de préférence : masque du segmenter s'il ressemble vraiment à un
        ballon → forme déduite de l'image → ellipse inscrite.
        """
        if box_w < 8 or box_h < 8:
            return None

        if isinstance(raw_mask, np.ndarray) and raw_mask.size > 0:
            m = raw_mask[:, :, 0] if raw_mask.ndim == 3 else raw_mask
            if m.shape[:2] != (box_h, box_w):
                try:
                    m = cv2.resize(m, (box_w, box_h), interpolation=cv2.INTER_NEAREST)
                except Exception:
                    m = None
            # Un ballon remplit largement sa bbox ; un masque de lettres non.
            if m is not None and float(np.count_nonzero(m)) / float(box_w * box_h) >= 0.55:
                return (m > 0).astype(np.uint8) * 255

        # Tentée quelle que soit l'étiquette : c'est une MESURE, elle n'a pas
        # besoin que YOLO ait vu juste. L'appelant décidera ensuite, d'après la
        # forme obtenue, s'il y a lieu d'épouser le contour.
        derived = self._bubble_mask_from_image(crop_bgr)
        if derived is not None:
            return derived

        # Dernier recours seulement : l'ellipse inscrite est une SUPPOSITION,
        # on ne la fait donc que si l'étiquette annonce une bulle.
        if not is_bubble:
            return None

        ellipse = np.zeros((box_h, box_w), dtype=np.uint8)
        cv2.ellipse(
            ellipse,
            (box_w // 2, box_h // 2),
            (max(1, box_w // 2 - 1), max(1, box_h // 2 - 1)),
            0, 0, 360, 255, -1,
        )
        return ellipse

    # ─────────────────────────────────────────────────────────────────────────
    # INSERT TEXT
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
        source_line_height: Optional[float] = None,
        stroke_color_rgb: Optional[Tuple[int, int, int]] = None,
        stroke_width: Optional[int] = None,
        bg_color_rgb: Optional[Tuple[int, int, int]] = None,
        angle_override: Optional[float] = None,
    ) -> np.ndarray:
        if not text or not str(text).strip():
            return img

        labelled_bubble = str(class_name).lower().strip() == "bulle"

        # Repère du texte SOURCE : `text_regions` y est exprimé, donc tout ce
        # qui s'y rapporte (ancrage System, angle, recentrage du bloc tourné)
        # doit continuer à l'utiliser.
        ox1, oy1, ox2, oy2 = x1, y1, x2, y2

        # ── Mise en page décidée sur la FORME, pas sur l'étiquette ──
        #
        # L'étiquette de YOLO n'est pas stable : selon le cadrage des fenêtres
        # glissantes, la même planche donne « 32 bulle + 3 out_text + 1 System »
        # en tranches et « 36 bulle » en pleine hauteur. Se fier au libellé
        # revenait à composer une boîte de narration comme un ovale, ou à
        # perdre le wrap de bulle sur un ballon mal étiqueté.
        #
        # Mesuré sur la planche : une boîte de narration a un contenant uni qui
        # l'englobe, une vraie bulle n'en a pas (ses abords sont du dessin ou
        # des hachures). C'est ce test-là qui tranche.
        container = self._container_box(img, x1, y1, x2, y2)
        if container is not None:
            x1, y1, x2, y2 = container

        box_h_full, box_w_full = max(1, y2 - y1), max(1, x2 - x1)

        if container is not None:
            # Un contenant trouvé EST la décision de forme : par construction
            # (_container_box exige un contour qui englobe la détection sur
            # ses 4 côtés), c'est une boîte. Inutile — et dangereux — de
            # revérifier avec `_bubble_shape_mask` : sur un cartouche System
            # aux bords ornementés (fioritures, volutes), la zone de couleur
            # uniforme trouvée par cette fonction est amputée par les
            # ornements et son taux de remplissage tombe sous le seuil de
            # rectangularité — la carte était alors jugée « non rectangulaire »
            # à tort. Conséquence concrète : le wrap bascule sur
            # `_wrap_text_by_mask`, qui aplatit tous les `\n` en espaces — or
            # c'est justement ces retours à la ligne que Gemini insère après
            # chaque « : » (JOB:\nPRIEST) pour séparer libellé et valeur.
            # Les écraser rendait les fiches System illisibles.
            mask_for_wrap = None
            has_mask_wrap = False
        else:
            mask_for_wrap = self._bubble_shape_mask(
                bubble_mask,
                img[max(0, y1):y2, max(0, x1):x2],
                box_w_full, box_h_full,
                is_bubble=labelled_bubble,
            )
            # Le wrap « forme » ne sert que si la forme n'est PAS un rectangle :
            # sur un cartouche il ne ferait qu'ajouter du bruit.
            has_mask_wrap = mask_for_wrap is not None and self._is_non_rectangular(mask_for_wrap)
        is_bubble = has_mask_wrap

        # ── Zone utile ──
        use_locked_mode = bool(getattr(self.cfg, 'lock_text_to_ocr_regions', False))
        if bool(getattr(self.cfg, 'lock_text_system_only', True)) and str(class_name).lower() != 'system':
            use_locked_mode = False

        anchor_box = None
        if use_locked_mode:
            anchor_box = self._compute_anchor_box_from_regions(ox1, oy1, ox2, oy2, text_regions)

        if anchor_box is not None:
            ix1, iy1, ix2, iy2 = anchor_box
        else:
            use_locked_mode = False
            ix1, iy1, ix2, iy2 = self._get_inner_zone(
                x1, y1, x2, y2, img.shape,
                bubble_mask=mask_for_wrap,
                shrink=self._shrink_ratio_for(is_bubble, has_mask_wrap),
            )

        tw, th = ix2 - ix1, iy2 - iy1
        if tw <= 0 or th <= 0:
            return img

        # ── Couleurs ──
        # Échantillonnées sur la bbox d'origine : elle est centrée sur le texte
        # effacé, donc représentative du fond sur lequel on va écrire. Le
        # contenant, lui, peut mordre sur une bordure ou un dégradé de bord.
        text_color, outline_color, outline_width_auto = self.get_text_colors(
            img, ox1, oy1, ox2, oy2,
            class_name=class_name,
            text_color_override=text_color_rgb,
        )
        
        if stroke_color_rgb is not None:
            outline_color = stroke_color_rgb
        if stroke_width is not None:
            outline_width_auto = stroke_width
            
        if bg_color_rgb is not None:
            # Remplir le fond d'une boîte opaque si bg_color_rgb est fourni
            cv2.rectangle(img, (ox1, oy1), (ox2, oy2), bg_color_rgb[::-1], -1)

        # ── Style ──
        bw, bh = x2 - x1, y2 - y1
        if text_style == "dialogue":
            text_style = self.infer_text_style(text, bw, bh, class_name=class_name)
        if text_style == "system_card":
            text = self._format_system_card_text(text)

        # Les polices BD sont dessinées pour du ALL CAPS ; le texte mixte y a
        # l'air amateur. Exceptions : chuchotement et interface système.
        if text_style not in ("whisper", "system_card"):
            text = text.upper()

        inner_w = max(10, tw - 2 * self.cfg.padding_horizontal)
        inner_h = max(10, th - 2 * self.cfg.padding_vertical)
        resolved_font_path = self._resolve_font_path(font_key, text_style, font_hint)

        # ── Orientation du texte source ──
        # Pas de filtre sur la classe : YOLO étiquette régulièrement en "bulle"
        # un texte posé sur un objet du décor. C'est l'angle mesuré qui
        # tranche — les quadrilatères OCR d'un vrai ballon sont à ~0°.
        angle = 0.0
        if angle_override is not None:
            angle = angle_override
        elif self.cfg.follow_source_text_angle:
            angle = self._source_text_angle(text_regions)
            if abs(angle) < float(self.cfg.min_text_angle_deg):
                angle = 0.0

        if angle != 0.0:
            # Sur un texte incliné, la largeur utile est la LONGUEUR de la ligne
            # d'origine, pas la largeur de la bbox englobante (qui est plus
            # large et ferait rentrer un texte qui, une fois tourné, dépasse).
            diag_w = self._rotated_line_width(text_regions, angle)
            if diag_w > 20:
                inner_w = max(20, int(diag_w))
                inner_h = max(10, int(max(inner_h, th)))

        # ── Taille + découpe ──
        fs = self.calculate_optimal_font_size(text, inner_w, inner_h, source_line_height)
        if text_style == "scream":
            fs = min(self.cfg.max_font_size, int(fs * 1.12))
        elif text_style == "whisper":
            fs = max(self.cfg.min_font_size, int(fs * 0.92))

        size_cap = None
        if source_line_height and source_line_height > 4:
            size_cap = int((float(source_line_height) / 0.75) * 1.05)

        layout = self._fit_font_hard(
            text, fs, inner_w, inner_h,
            bubble_mask=mask_for_wrap if angle == 0.0 else None,
            shape_wrap=has_mask_wrap and angle == 0.0,
            font_path=resolved_font_path,
            # Origine verticale RÉELLE du bloc dans le repère de la bbox : le
            # padding manquait, les bandes de masque étaient mesurées 14 px
            # trop haut.
            mask_y_offset=(iy1 + self.cfg.padding_vertical) - y1,
            max_font_size=size_cap,
        )
        if layout is None:
            return img

        if angle != 0.0:
            return self._draw_rotated(
                img, layout, angle, text_regions, ox1, oy1, ox2, oy2,
                text_color, outline_color, outline_width_auto,
            )

        return self._draw_block(
            img, layout,
            ix1, iy1, ix2, iy2, inner_w, inner_h,
            text_color, outline_color, outline_width_auto,
            top_aligned=(use_locked_mode or text_style == "system_card"),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # DESSIN
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_block(
        self, img: np.ndarray, layout: Dict,
        ix1: int, iy1: int, ix2: int, iy2: int, inner_w: int, inner_h: int,
        text_color, outline_color, outline_width: int,
        top_aligned: bool = False,
    ) -> np.ndarray:
        """
        Dessine le bloc de texte, à l'endroit, sur un CROP de l'image.

        Deux raisons de ne pas convertir l'image entière en PIL comme avant :
        c'était 4 copies d'une planche de ~90 Mo par bulle (des dizaines de Go
        de recopie par planche), et le résultat était réécrit intégralement à
        chaque appel.
        """
        font = layout['font']
        lines = layout['lines']
        line_h, spacing = layout['line_h'], layout['spacing']
        total_h = layout['total_h']

        left = ix1 + self.cfg.padding_horizontal
        top = iy1 + self.cfg.padding_vertical

        if top_aligned:
            ys = top
        elif self.cfg.vertical_align == 'top':
            ys = top
        elif self.cfg.vertical_align == 'bottom':
            ys = iy2 - self.cfg.padding_vertical - total_h
        else:
            # Centrage du BLOC. L'ancienne version recalait ensuite chaque ligne
            # individuellement dans la zone : dès que le bloc était trop haut,
            # toutes les lignes du bas se retrouvaient à la même ordonnée,
            # dessinées les unes sur les autres.
            ys = top + (inner_h - total_h) // 2

        # Zone à convertir en PIL : elle doit contenir TOUT le texte. Avec le
        # wrap sur masque, une ligne peut légitimement être plus large que
        # `inner_w` (la bulle est plus large que le rectangle inscrit) et
        # déborder des deux côtés — la découper ici la tronquerait au rendu.
        overflow_x = max(0, (layout['max_line_w'] - inner_w + 1) // 2)
        pad = max(4, outline_width * 2 + layout['size'] // 2)
        rx1 = max(0, min(left, ix1) - overflow_x - pad)
        ry1 = max(0, min(ys, iy1) - pad)
        rx2 = min(img.shape[1], max(ix2, left + inner_w) + overflow_x + pad)
        ry2 = min(img.shape[0], max(iy2, ys + total_h) + pad)
        if rx2 <= rx1 or ry2 <= ry1:
            return img

        crop = img[ry1:ry2, rx1:rx2]
        crop_pil = ImageUtils.cv2_to_pil(crop)
        draw = ImageDraw.Draw(crop_pil)

        for i, line in enumerate(lines):
            if not line:
                continue
            offset_x, ink_w = self._line_extents(font, line)

            if self.cfg.horizontal_align == 'left' or top_aligned:
                xp = left
            elif self.cfg.horizontal_align == 'right':
                xp = left + inner_w - ink_w
            else:
                xp = left + (inner_w - ink_w) // 2
            # `draw.text` place l'ORIGINE, pas le bord gauche de l'encre.
            xp -= offset_x

            yp = ys + i * (line_h + spacing)

            if outline_color is not None and outline_width > 0:
                draw.text(
                    (xp - rx1, yp - ry1), line, font=font,
                    fill=text_color, stroke_width=outline_width, stroke_fill=outline_color,
                )
            else:
                draw.text((xp - rx1, yp - ry1), line, font=font, fill=text_color)

        img[ry1:ry2, rx1:rx2] = ImageUtils.pil_to_cv2(crop_pil)
        return img

    @staticmethod
    def _rotated_line_width(regions: Optional[List[Dict]], angle: float) -> float:
        """Longueur de la plus longue ligne OCR, mesurée le long de son axe."""
        best = 0.0
        for region in regions or []:
            pts = region.get('bbox') if isinstance(region, dict) else None
            if not pts or len(pts) < 4:
                continue
            try:
                arr = np.array(pts, dtype=np.float32)
            except Exception:
                continue
            width = float(np.hypot(arr[1][0] - arr[0][0], arr[1][1] - arr[0][1]))
            best = max(best, width)
        return best

    def _draw_rotated(
        self, img: np.ndarray, layout: Dict, angle: float,
        regions: Optional[List[Dict]], x1: int, y1: int, x2: int, y2: int,
        text_color, outline_color, outline_width: int,
    ) -> np.ndarray:
        """
        Dessine le bloc sur un calque transparent, le tourne de `angle`, puis le
        compose sur l'image au centre du texte d'origine.
        """
        font = layout['font']
        lines = layout['lines']
        line_h, spacing, total_h = layout['line_h'], layout['spacing'], layout['total_h']

        widths = [self._line_extents(font, ln) for ln in lines]
        block_w = max((w for _, w in widths), default=0)
        if block_w <= 0 or total_h <= 0:
            return img

        pad = max(4, outline_width * 2 + layout['size'] // 2)
        layer = Image.new('RGBA', (block_w + 2 * pad, total_h + 2 * pad), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        for i, line in enumerate(lines):
            if not line:
                continue
            offset_x, ink_w = widths[i]
            xp = pad + (block_w - ink_w) // 2 - offset_x
            yp = pad + i * (line_h + spacing)
            if outline_color is not None and outline_width > 0:
                draw.text(
                    (xp, yp), line, font=font, fill=tuple(text_color) + (255,),
                    stroke_width=outline_width, stroke_fill=tuple(outline_color) + (255,),
                )
            else:
                draw.text((xp, yp), line, font=font, fill=tuple(text_color) + (255,))

        # PIL tourne dans le sens antihoraire pour un angle positif.
        rotated = layer.rotate(-angle, resample=Image.BICUBIC, expand=True)

        # Centre du texte d'origine (à défaut, centre de la bbox).
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        pts: List[Tuple[float, float]] = []
        for region in regions or []:
            raw = region.get('bbox') if isinstance(region, dict) else None
            for p in raw or []:
                pts.append((x1 + float(p[0]), y1 + float(p[1])))
        if len(pts) >= 3:
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)

        rw, rh = rotated.size
        px1 = int(round(cx - rw / 2.0))
        py1 = int(round(cy - rh / 2.0))

        # Découpe du calque à ce qui rentre dans l'image.
        sx1 = max(0, -px1)
        sy1 = max(0, -py1)
        dx1 = max(0, px1)
        dy1 = max(0, py1)
        dx2 = min(img.shape[1], px1 + rw)
        dy2 = min(img.shape[0], py1 + rh)
        if dx2 <= dx1 or dy2 <= dy1:
            return img

        rotated = rotated.crop((sx1, sy1, sx1 + (dx2 - dx1), sy1 + (dy2 - dy1)))

        base = ImageUtils.cv2_to_pil(img[dy1:dy2, dx1:dx2]).convert('RGBA')
        base.alpha_composite(rotated)
        img[dy1:dy2, dx1:dx2] = ImageUtils.pil_to_cv2(base.convert('RGB'))
        return img

