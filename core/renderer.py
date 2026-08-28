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

from core.bubble_shape import grow_from_ink, ink_mask_from_regions
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path
import math
import os
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

if HF_AVAILABLE:
    # `diffusers` (dépendance de lama_cleaner pour AnimeMangaInpainting) importe
    # encore `huggingface_hub.cached_download`, retiré des versions récentes de
    # huggingface_hub (remplacé par `hf_hub_download`, signature quasi
    # identique) — sans ce shim, l'import de lama_cleaner.model_manager
    # échouait systématiquement et le second inpainter (utile sur les fonds
    # d'artwork complexes que LaMa générique reconstruit mal) ne se chargeait
    # jamais.
    try:
        import huggingface_hub as _hf_hub_compat
        if not hasattr(_hf_hub_compat, 'cached_download'):
            _hf_hub_compat.cached_download = _hf_hub_compat.hf_hub_download
    except Exception:
        pass


_HYPHEN_STATE: Dict[str, object] = {"loaded": False, "dic": None}


def _hyphenator(lang: str = "fr_FR"):
    """Dictionnaire de césure `pyphen`, chargé une seule fois.

    On coupe selon les RÈGLES DE LA LANGUE, pas selon la place disponible : une
    coupure géométrique donnerait « RAVITA-ILLEMENT ». Pyphen s'appuie sur les
    motifs de césure Hunspell/LibreOffice et rend « ra-vi-taille-ment ».
    Changer `lang` suffit pour l'allemand ou l'espagnol.
    """
    if not _HYPHEN_STATE["loaded"]:
        _HYPHEN_STATE["loaded"] = True
        try:
            import pyphen
            _HYPHEN_STATE["dic"] = pyphen.Pyphen(lang=lang)
        except Exception as exc:
            _HYPHEN_STATE["dic"] = None
            print(f"⚠️ pyphen indisponible ({exc}) — césure des mots longs DÉSACTIVÉE.")
    return _HYPHEN_STATE["dic"]



class TextRenderer:
    """Rendu texte avec LaMa inpainting local + ColorResolver V2"""

    # Repli seulement : la marge intérieure réelle est décidée par
    # `_shrink_ratio_for()` selon la classe et la disponibilité du masque.
    SHRINK_RATIO = 0.18
    # Marge de débordement autorisée aux cartouches (cf. _draw_exact_lines).
    CAPTION_WIDTH_ALLOWANCE = 1.22
    CAPTION_SIZE_ALLOWANCE = 1.30

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

        # QCheck : rempli par `_apply_erasure_by_group` quand un groupe reste
        # imparfaitement effacé (LaMa a échoué ET la diffusion n'était pas sûre
        # sur ce fond) — l'appelant (pipeline.py) le consulte pour savoir si LA
        # détection en cours doit être signalée dans le rapport post-rendu.
        self.qcheck_flags: List[Dict[str, Any]] = []

        # Hook de diagnostic (additif, lecture seule) : rempli par `insert_text`
        # à chaque appel avec les mesures du dernier layout calculé — voir
        # scratch/measure_layout.py. Ne change aucun comportement de rendu.
        self.last_layout_debug: Optional[Dict[str, Any]] = None
        # Coût de chaque route de rendu au dernier `insert_text` (cf. _route_cost).
        self._last_route_costs: Dict[str, float] = {}

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

    # Polices de lettrage retenues par style, dans l'ordre de préférence
    # (fragments de nom de fichier, insensibles à la casse). Ce sont les
    # polices de scanlation usuelles, présentes dans `assets/fonts`.
    PREFERRED_FONTS = {
        ("dialogue", "regular"): ["CCWILDWORDS-REGULAR", "ANIME_ACE_2.0", "CCAskForMercy-Regular"],
        ("dialogue", "bold"): ["CCWILDWORDS-BOLD", "ANIMEACE2_BLD", "CCWILDWORDS-REGULAR"],
        ("dialogue", "thin"): ["CCWILDWORDS-REGULAR", "ANIME_ACE_2.0"],
        ("scream", "regular"): ["CCWILDWORDS-BOLD", "ANIMEACE2_BLD", "KOMIKA_AXIS"],
        ("scream", "bold"): ["CCWILDWORDS-BOLD", "ANIMEACE2_BLD", "KOMIKA_AXIS"],
        ("scream", "thin"): ["CCWILDWORDS-BOLD", "ANIMEACE2_BLD"],
        ("whisper", "regular"): ["CCWILDWORDS-ITALIC", "ANIMEACE2_ITAL", "CCWILDWORDS-REGULAR"],
        ("whisper", "bold"): ["CCWILDWORDS-ITALIC", "CCWILDWORDS-REGULAR"],
        ("whisper", "thin"): ["CCWILDWORDS-ITALIC", "CCWILDWORDS-REGULAR"],
        ("narration", "regular"): ["CCWILDWORDS-REGULAR", "ANIME_ACE_2.0"],
        ("narration", "bold"): ["CCWILDWORDS-BOLD", "ANIMEACE2_BLD"],
        ("narration", "thin"): ["CCWILDWORDS-REGULAR", "ANIME_ACE_2.0"],
    }

    @staticmethod
    def _find_font_by_fragment(fragment: str) -> Optional[str]:
        """Chemin de la première police dont le nom contient `fragment`."""
        try:
            import glob
            frag = fragment.lower()
            for path in glob.glob("assets/fonts/**/*.*", recursive=True):
                if not path.lower().endswith((".ttf", ".otf")):
                    continue
                if frag in Path(path).name.lower():
                    return path
        except Exception:
            pass
        return None

    def _preferred_font(self, style: str, font_hint: str) -> Optional[str]:
        """Première police de `PREFERRED_FONTS` réellement disponible, ou None.

        `system_card` est volontairement absent : le rendu HUD de ces cartes
        assume sa police techno, zéro barré compris.
        """
        # Court-circuit d'essai : `WEBTOON_FONT_FORCE` impose une police par
        # fragment de nom, pour comparer des candidats sur les mêmes crops sans
        # toucher au code. Sert au A/B de `scratch/ab_fonts.py`.
        import os as _os
        forced = _os.environ.get("WEBTOON_FONT_FORCE", "").strip()
        if forced:
            hit = self._find_font_by_fragment(forced)
            if hit:
                return hit

        names = self.PREFERRED_FONTS.get((style, (font_hint or "regular").lower()))
        if not names:
            names = self.PREFERRED_FONTS.get((style, "regular"))
        if not names:
            return None
        available = {str(p).lower(): p for p in self.fonts}
        for wanted in names:
            low = wanted.lower()
            for key, path in available.items():
                if low in key:
                    return path
        return None

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

        # Choix EXPLICITE avant l'heuristique par dossier. Celle-ci retenait
        # « le premier fichier du dossier dont le nom contient bold », c'est-à-
        # dire `buddychampionbold.ttf` pour presque tous les styles : une police
        # techno à zéro barré (« LEVEL: 10 » sortait « LEVEL: 1Ø », « TOKYO »
        # sortait « TOKYØ ») et à dessin carré, qui ne ressemble à aucun
        # lettrage de manhwa. Les polices de lettrage standard sont pourtant
        # présentes dans `assets/fonts` — on les nomme.
        picked = self._preferred_font(style, font_hint)
        if picked:
            return picked

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

        # Les cartons out_text sont souvent posés sur des décors chargés
        # (éclairs, flammes, trames de vitesse) où le texte couvre une
        # bonne partie du panneau : avec la marge standard, LaMa n'a que
        # peu de pixels de vrai fond pour apprendre la texture à reproduire
        # et retombe sur un aplat plus terne que l'effet d'origine. Une
        # marge plus large lui donne davantage de contexte à imiter.
        # La marge de crop est le CONTEXTE que voit le modèle d'inpainting. Une
        # marge fixe de 60 px sur un cartouche de 423x284 laissait ~40 % du crop
        # masqué : LaMa n'avait presque que du trou à regarder et rendait une
        # bouillie violette de la taille du bloc de texte. Mesuré sur le même
        # cartouche avec une marge de 2x la hauteur de boîte : la part masquée
        # du crop tombe de 39 % à 6 %, et la reconstruction redevient plausible.
        # On l'indexe donc sur la taille de la zone, pas sur une constante.
        # `System` traité comme `out_text` : ce sont aussi des cartouches de
        # texte d'impact posés sur du décor, souvent avec halo. Mesuré sur le
        # panneau ouvragé « THE RED DRAGON, SOVEREIGN VALDROVA. » : avec un
        # masque au glyphe, LaMa laissait des fantômes de lettres bien
        # lisibles (essayé à 5, 9 et 15 px de dilatation, tous mauvais) alors
        # que le masque au BLOC rendait le panneau propre.
        if str(class_name).lower() in ("out_text", "system"):
            m = max(self.CROP_MARGIN * 2, 2 * bubble_h)
        else:
            m = max(self.CROP_MARGIN, bubble_h)
        crop_x1 = max(0, x1 - m)
        crop_y1 = max(0, y1 - m)
        crop_x2 = min(w_img, x2 + m)
        crop_y2 = min(h_img, y2 + m)

        crop = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        crop_h, crop_w = crop.shape[:2]
        if crop_h <= 0 or crop_w <= 0:
            return img

        # Construire le masque local
        #
        # Les cartouches out_text sont du texte d'IMPACT : corps plein, gros
        # contour, et surtout une LUEUR externe qui fait partie de l'effet et
        # s'étale bien au-delà du glyphe. Un masque au glyphe, même parfait,
        # laisse cette lueur en place — on voyait donc encore la silhouette
        # rectangulaire du bloc de texte après effacement, en violet/bleu.
        # Mesuré sur path-of-vengeance ch1, six cartouches : masque au glyphe
        # dilaté = halo résiduel systématique ; masque au BLOC (polygones de
        # ligne OCR dilatés de 0,30 × hauteur de ligne) = aucune trace, sur
        # fond noir comme sur les rayures rouges ou les éclairs. Le surcoût est
        # faible (7 % → 13 % du crop masqué) parce que la marge de crop donne à
        # LaMa largement de quoi reconstruire.
        # Le masque au bloc est calculé pour TOUTES les classes : même quand il
        # ne sert pas d'emblée, il sert de repli si un fantôme subsiste après la
        # première passe (cf. la deuxième passe plus bas). Le texte à lueur
        # n'est pas l'apanage des classes `out_text`/`System` — « JUST KILL ME
        # ALREADY!! » est classé `bulle`.
        block_mask = None
        if text_regions:
            block_mask = self._block_mask_from_regions(
                crop_w, crop_h, text_regions, x1 - crop_x1, y1 - crop_y1,
            )

        use_block_first = str(class_name).lower() in ("out_text", "system")
        if use_block_first and block_mask is not None and int(np.count_nonzero(block_mask)) > 0:
            local_mask = block_mask
        elif chirurgical_mask is not None and isinstance(chirurgical_mask, np.ndarray) and chirurgical_mask.size > 0:
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

            # `chirurgical_mask` ne subit qu'une fermeture 3x3 (bouche les trous
            # entre lettres) sans dilatation vers l'extérieur. Sur un texte
            # d'impact (contour épais, souvent "out_text") posé sur un fond
            # texturé/à fort contraste, les derniers pixels du trait restent
            # hors masque et forment un contour fantôme bien visible — un fond
            # uni les aurait masqués, pas un fond chargé. Mesuré sur
            # path-of-vengeance ch1 : systématique sur les cartouches de titre
            # stylisés, quasi absent sur le dialogue en bulle (trait fin, fond
            # uni). `out_text_mask_dilate_kernel` existait déjà dans la config
            # pour ce cas mais n'était jamais lu.
            dilate_k = (
                self.cfg.out_text_mask_dilate_kernel
                if str(class_name).lower() in ("out_text", "system")
                else self.cfg.inpaint_mask_dilate_kernel
            )
            if dilate_k and dilate_k > 1:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
                local_mask = cv2.dilate(local_mask, kernel, iterations=1)
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
        local_bubble = None
        if bubble_mask is not None and bubble_mask.size > 0:
            det_h, det_w = max(1, y2 - y1), max(1, x2 - x1)
            bm = bubble_mask
            if bm.shape[:2] != (det_h, det_w):
                bm = cv2.resize(bm, (det_w, det_h), interpolation=cv2.INTER_NEAREST)
            offset_x = x1 - crop_x1
            offset_y = y1 - crop_y1
            local_bubble = np.zeros((crop_h, crop_w), dtype=np.uint8)
            dst_x1, dst_y1 = max(0, offset_x), max(0, offset_y)
            dst_x2 = min(crop_w, offset_x + det_w)
            dst_y2 = min(crop_h, offset_y + det_h)
            src_x1, src_y1 = dst_x1 - offset_x, dst_y1 - offset_y
            src_x2, src_y2 = src_x1 + (dst_x2 - dst_x1), src_y1 + (dst_y2 - dst_y1)
            if dst_x2 > dst_x1 and dst_y2 > dst_y1:
                local_bubble[dst_y1:dst_y2, dst_x1:dst_x2] = (
                    bm[src_y1:src_y2, src_x1:src_x2] > 0
                ).astype(np.uint8) * 255

        # Limite au TRAIT DE CONTOUR près : `_extend_fill_mask` absorbe tout ce
        # qui, dans un rayon de 21 px autour des lettres, s'écarte du fond et
        # leur est connexe. Une ligne large passe à moins de 21 px du trait du
        # ballon : le trait est sombre, donc « s'écarte du fond », et se
        # retrouve repeint en blanc. Mesuré sur « IT MIGHT JUST BE A NORMAL
        # RUN, BUT... » : 574 px absorbés, dont 48 sur le trait, des deux côtés
        # à la même hauteur — un décrochement bien visible du contour.
        #
        # `local_bubble` ne peut pas servir de borne (c'est un masque de
        # LETTRES quand le segmenter tourne en mode `hybrid`) ; l'intérieur
        # déduit de l'image, si, à condition de l'éroder de l'épaisseur du trait.
        # La forme se déduit sur la BBOX, pas sur le crop élargi : avec une
        # marge de la taille de la bulle, le fond blanc de la page fusionne
        # avec l'intérieur blanc du ballon et « la plus grande zone homogène »
        # devient la page entière (71 % du crop mesuré) — inexploitable.
        fill_limit = None
        if str(class_name).lower() != "out_text":
            det_h, det_w = max(1, y2 - y1), max(1, x2 - x1)
            interior = self._bubble_mask_from_image(img[max(0, y1):y2, max(0, x1):x2])
            if interior is not None and interior.shape[:2] == (det_h, det_w):
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                interior = cv2.erode(interior, k, iterations=1)
                # Hors bbox on laisse faire : c'est là que vit le morceau de
                # glyphe coupé par la boîte de détection, qu'on veut toujours
                # pouvoir rattraper. La borne ne sert qu'à protéger le trait du
                # ballon, qui est dans la bbox.
                limit = np.full((crop_h, crop_w), 255, dtype=np.uint8)
                ox, oy = x1 - crop_x1, y1 - crop_y1
                dx1, dy1 = max(0, ox), max(0, oy)
                dx2, dy2 = min(crop_w, ox + det_w), min(crop_h, oy + det_h)
                if dx2 > dx1 and dy2 > dy1:
                    limit[dy1:dy2, dx1:dx2] = interior[
                        dy1 - oy:dy2 - oy, dx1 - ox:dx2 - ox
                    ]
                    # Seulement si le texte tient vraiment dedans : sinon la
                    # forme déduite est fausse et borner ferait survivre le
                    # texte d'origine sous la traduction.
                    if int(np.count_nonzero(cv2.bitwise_and(local_mask, limit))) >= (
                        0.9 * int(np.count_nonzero(local_mask))
                    ):
                        fill_limit = limit

        flat = self._flat_fill_color(crop, local_mask, local_bubble_mask=local_bubble, class_name=class_name)
        if flat is not None:
            inside_box = np.zeros((crop_h, crop_w), dtype=np.uint8)
            bx1, by1 = max(0, x1 - crop_x1), max(0, y1 - crop_y1)
            bx2, by2 = min(crop_w, x2 - crop_x1), min(crop_h, y2 - crop_y1)
            if bx2 > bx1 and by2 > by1:
                inside_box[by1:by2, bx1:bx2] = 255
            extended = self._extend_fill_mask(
                crop, local_mask, flat, inside_box=inside_box,
            )
            if fill_limit is not None:
                extended = cv2.bitwise_or(
                    cv2.bitwise_and(extended, fill_limit), local_mask,
                )
            crop[extended > 0] = flat
            img[crop_y1:crop_y2, crop_x1:crop_x2] = crop
            return img

        # Fond LISSE mais non uni (dégradé de bulle, halo) : modèle lisse.
        #
        # `_flat_fill_color` exige un fond quasi UNI ; sur une bulle éclairée en
        # dégradé il refuse, et LaMa prend la main — mesuré sur « A UNIQUE
        # CONSTITUTION? » (the-frontier-count ch1), il y laissait une cicatrice
        # grise parfaitement visible à l'emplacement de la deuxième ligne,
        # alors que la première ligne, elle, sortait propre.
        #
        # Le test n'est pas une classification du fond (aucun critère mesuré
        # jusqu'ici ne sépare proprement « dégradé » de « texturé » : la
        # couronne autour du texte contient le trait noir de la bulle, qui
        # domine toutes les statistiques). Il est AUTO-VALIDÉ : on construit le
        # modèle lisse, puis on regarde s'il explique les pixels NON masqués du
        # voisinage. S'il les explique, il explique aussi ceux qu'on remplace.
        smooth = self._smooth_fill(crop, local_mask)
        if smooth is not None:
            img[crop_y1:crop_y2, crop_x1:crop_x2] = smooth
            return img

        # LaMa inpaint
        if self.lama is not None:
            try:
                result = self._inpaint_lama(crop, local_mask)
                erased = self._apply_erasure_by_group(crop, result, local_mask, class_name)

                # Deuxième passe au BLOC si un fantôme subsiste.
                #
                # Sur du texte à LUEUR, le masque d'encre ne retient que le
                # cœur clair des lettres ; le halo, lui, reste — et il en
                # redessine la forme. Mesuré sur « JUST KILL ME ALREADY!! »
                # (lettrage blanc à lueur magenta sur fond noir) : le blanc
                # partait, la lueur rose restait parfaitement lisible, et la
                # traduction se posait par-dessus, illisible.
                #
                # Le routage par CLASSE ne suffit pas — cette case-là est
                # classée `bulle`, pas `out_text`. On mesure donc le résultat :
                # s'il reste un fantôme, on recommence avec le masque au bloc,
                # qui emporte le glyphe ET son halo d'un coup.
                if block_mask is not None and (
                    self._erasure_failed(crop, erased, local_mask)
                    or self._ghost_remains(erased, local_mask)
                ):
                    # La lueur s'étend bien au-delà du bloc calculé sur les
                    # polygones. On l'attrape en suivant sa RAMPE plutôt qu'en
                    # dilatant à l'aveugle : la croissance s'arrête sur les
                    # arêtes du décor au lieu de les manger.
                    wider = self._halo_grow(
                        crop, cv2.bitwise_or(local_mask, block_mask), max_radius=30,
                    )
                    # Le masque au BLOC est un rectangle autour des polygones,
                    # dilate de 0,30 x hauteur de ligne. Sur une bulle dont le
                    # texte occupe presque toute la surface il atteint le TRAIT
                    # avant meme la croissance — mesure sur « JUST KILL ME
                    # ALREADY!! » : 9,5 % du feston blanc dans le masque, que
                    # LaMa devait ensuite halluciner, d'ou le contour mordu et
                    # les trainees en bas de bulle. On le borne donc a
                    # l'interieur du ballon, deduit de la passe 1.
                    interior_limit = self._bubble_interior_limit(
                        erased, text_regions, crop,
                        max(0, x1 - crop_x1), max(0, y1 - crop_y1),
                        min(crop_w, x2 - crop_x1), min(crop_h, y2 - crop_y1),
                    )
                    if interior_limit is not None:
                        wider = cv2.bitwise_or(
                            cv2.bitwise_and(wider, interior_limit), local_mask,
                        )
                    retry = self._inpaint_lama(crop, wider)
                    erased2 = crop.copy()
                    erased2[wider > 0] = retry[wider > 0]
                    # On garde la passe qui laisse le MOINS de structure.
                    if self._ghost_score(erased2, wider) < self._ghost_score(erased, local_mask):
                        erased = erased2

                img[crop_y1:crop_y2, crop_x1:crop_x2] = erased
                return img
            except Exception:
                pass

        # Anime inpainter fallback
        if self.anime_inpainter_ready and self.anime_inpainter is not None:
            try:
                result = self._inpaint_anime(crop, local_mask)
                img[crop_y1:crop_y2, crop_x1:crop_x2] = self._apply_erasure_by_group(crop, result, local_mask, class_name)
                return img
            except Exception:
                pass

        # cv2 fallback
        try:
            img[crop_y1:crop_y2, crop_x1:crop_x2] = self._diffuse_fill(crop, local_mask)
        except Exception:
            pass

        return img


    # Marge autour de la bbox pour deduire l'interieur du ballon. Les festons
    # et la queue DEPASSENT de la boite de detection : borner sur la bbox seule
    # laissait encore 4,9 % du trait dans le masque (mesure Dragon #0). A 20 %
    # il n'en reste rien, et le masque ne perd que 2,6 points de couverture.
    INTERIOR_PAD_FRAC = 0.20
    INTERIOR_ERODE = 9

    def _bubble_interior_limit(
        self, crop_erased: np.ndarray, regions, crop_orig: np.ndarray,
        bx1: int, by1: int, bx2: int, by2: int,
    ) -> Optional[np.ndarray]:
        """Borne « ne franchis pas le trait du ballon », au repere du crop.

        Renvoie None quand il n'y a pas de ballon a proteger (texte libre) ou
        que la forme n'est pas deduisible — l'appelant garde alors son masque.

        `crop_erased` doit etre la sortie de la PREMIERE passe : le texte y a
        disparu mais le trait du ballon est intact, ce qui est exactement
        l'image dont `grow_from_ink` a besoin pour trouver ses murs. Sur
        l'image d'ORIGINE la croissance s'echappe par les endroits ou le
        lettrage frole le contour (mesure : 30/36 seulement).
        """
        try:
            ch, cw = crop_erased.shape[:2]
            px = int((bx2 - bx1) * self.INTERIOR_PAD_FRAC)
            py = int((by2 - by1) * self.INTERIOR_PAD_FRAC)
            ax1, ay1 = max(0, bx1 - px), max(0, by1 - py)
            ax2, ay2 = min(cw, bx2 + px), min(ch, by2 + py)
            if ax2 - ax1 < 24 or ay2 - ay1 < 24:
                return None

            ink = np.zeros((ay2 - ay1, ax2 - ax1), np.uint8)
            sub = ink_mask_from_regions(crop_orig[by1:by2, bx1:bx2], regions)
            ink[by1 - ay1:by2 - ay1, bx1 - ax1:bx2 - ax1] = sub

            interior, _diag = grow_from_ink(crop_erased[ay1:ay2, ax1:ax2], ink)
            if interior is None:
                return None
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.INTERIOR_ERODE, self.INTERIOR_ERODE),
            )
            interior = cv2.erode(interior, k, iterations=1)
            limit = np.full((ch, cw), 255, np.uint8)
            limit[ay1:ay2, ax1:ax2] = interior
            return limit
        except Exception:
            return None

    def _apply_erasure_by_group(
        self, crop: np.ndarray, result: np.ndarray, mask: np.ndarray, class_name: str = ""
    ) -> np.ndarray:
        """
        Décide LaMa vs diffusion PAR GROUPE de lettres, pas pour tout le
        masque à la fois.

        Un même masque peut couvrir des lignes posées sur des fonds très
        différents — mesuré sur une carte "System" dont le filigrane de scan
        ("CRAWLED BY MANHWACLAN.COM") est OCR dans la MÊME détection que la
        narration en dessous : le filigrane est sur un ciel étoilé texturé
        (pas diffusable, à raison), la narration sur l'intérieur uni de la
        carte (diffusable). Juger le masque entier d'un bloc faisait hériter
        la narration du refus de diffuser décidé pour le filigrane, et
        gardait le résultat LaMa — où le filigrane restait pourtant
        parfaitement lisible en rouge/blanc.

        Regroupe les lettres proches (dilatation généreuse avant
        `connectedComponents`, pour ne pas séparer les lignes d'un même
        paragraphe) puis applique `_erasure_failed`/`_background_is_diffusable`
        indépendamment à chaque groupe.
        """
        out = crop.copy()
        m = mask if mask.ndim == 2 else mask[:, :, 0]

        group_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        grouped = cv2.dilate(m, group_kernel, iterations=1)
        n_labels, labels = cv2.connectedComponents(grouped, 8)

        if n_labels <= 2:
            # Un seul groupe (ou aucun) : pas de gain à fragmenter, chemin
            # d'origine.
            failed = self._erasure_failed(crop, result, m)
            if (
                failed
                and self._diffusion_is_safe(crop, m)
                and self._background_is_diffusable(crop, m, class_name=class_name)
            ):
                return self._diffuse_fill(crop, m)
            if failed:
                # Effacement raté ET fond jugé pas sûr à diffuser : on garde le
                # résultat LaMa (imparfait mais le moins pire des deux), mais on
                # le signale — c'est exactement le genre de résidu qu'un
                # relecteur humain repère et qu'aucun signal ici ne peut
                # corriger automatiquement sans risquer pire.
                self.qcheck_flags.append({'type': 'ghost_residual'})
            return self._blend_masked(crop, result, m)

        for label in range(1, n_labels):
            group_mask = ((labels == label) & (m > 0)).astype(np.uint8) * 255
            if not group_mask.any():
                continue
            group_failed = self._erasure_failed(crop, result, group_mask)
            if (
                group_failed
                and self._diffusion_is_safe(crop, group_mask)
                and self._background_is_diffusable(crop, group_mask, class_name=class_name)
            ):
                diffused = self._diffuse_fill(crop, group_mask)
                # `_diffuse_fill` recompose sur un masque ÉLARGI en interne (la
                # frange antialiasée juste hors de l'encre stricte). Reblender
                # ici avec `group_mask` (étroit) jette ce travail : la frange
                # reste alors la valeur d'ORIGINE, non celle, propre, que
                # `diffused` contient déjà à cet endroit — un fantôme fin mais
                # bien visible du contour des lettres survit. Même élargissement
                # que `_diffuse_fill` pour transférer aussi cette frange.
                extra = max(6, int(round(min(group_mask.shape[:2]) * 0.02)))
                wide_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * extra + 1, 2 * extra + 1))
                wide_group_mask = cv2.dilate(group_mask, wide_kernel)
                out = self._blend_masked(out, diffused, wide_group_mask)
            else:
                if group_failed:
                    self.qcheck_flags.append({'type': 'ghost_residual'})
                out = self._blend_masked(out, result, group_mask)

        return out

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
        if (edge_after / edge_before) > 0.18:
            return True

        # Second signal, orthogonal au premier : la COULEUR de remplissage,
        # pas sa structure. Mesuré sur une bulle grise unie : LaMa a rempli le
        # masque en blanc quasi pur (243) alors que le fond réel est gris
        # clair (231) — à peine 12 niveaux d'écart, donc peu de relief
        # interne (le ratio de bord ci-dessus ne voyait rien d'anormal), mais
        # assez pour dessiner un halo fantôme visible à l'endroit exact des
        # lettres d'origine. Comparaison à la couleur MÉDIANE de la couronne
        # juste hors masque, pas à `crop` (qui contient encore les lettres).
        ring = (cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))) > 0) & (mask == 0)
        if not ring.any():
            return False
        ring_color = np.median(crop[ring].reshape(-1, 3), axis=0)
        result_color = np.median(result[m].reshape(-1, 3), axis=0)
        if float(np.max(np.abs(ring_color - result_color))) > 15.0:
            return True

        # Troisième signal : sur un fond très texturé (motif floral décoratif),
        # les deux signaux ci-dessus peuvent rater un fantôme pourtant
        # parfaitement lisible. Mesuré sur une carte System à fond fleuri :
        # LaMa avait remplacé le texte par un contour clair de MÊME NATURE
        # statistique que le motif environnant (fines lignes claires sur fond
        # clair) — bord/avant ratio 0.09 (sous le seuil), écart de couleur
        # médiane 6 (sous le seuil), et pourtant "THE HOLY SCRIPT STATES"
        # restait entièrement lisible à l'œil. Le point commun des deux ratés :
        # tous deux comparaient le résultat à l'ÉTAT AVANT (texte, donc très
        # contrasté) ou à la couleur du fond — jamais à la TEXTURE naturelle
        # du fond. Un carton correctement effacé doit rester plus LISSE que le
        # motif décoratif qui l'entoure, pas aussi structuré que lui : ratio
        # mesuré 0.64 pour le fantôme contre 0.05 pour un remplissage propre.
        edge_ring = float(np.mean(np.abs(cv2.Laplacian(gray_before, cv2.CV_32F, ksize=3))[ring]))
        if edge_ring >= 20.0 and (edge_after / edge_ring) > 0.35:
            return True

        return False

    @staticmethod
    def _outline_by_normals(crop, ink, max_len=12, n_samples=400):
        """Largeur du contour, mesuree LE LONG DE LA NORMALE au bord du glyphe.
    
        Les anneaux concentriques ne peuvent pas voir un contour : l'anti-crenelage
        fait varier la couleur a chaque pixel de distance, donc aucun "plateau" ne
        survit a une comparaison anneau par anneau. Mesure precedente : 1 px sur les
        34 bulles, ecart-type nul, avec deux logiques opposees — le signe que la
        geometrie de lecture etait en cause, pas le seuil.
    
        Ici on tire des profils 1D perpendiculaires au bord, on cherche sur chacun
        le plateau de couleur entre le glyphe et le fond, et on prend la MEDIANE des
        largeurs trouvees. Un profil bruite ou ambigu ne fait que deplacer la
        mediane a la marge, la ou il cassait tout dans la version par anneaux.
        """
        edges = cv2.Canny((ink > 0).astype(np.uint8) * 255, 50, 150)
        ys, xs = np.nonzero(edges)
        if len(xs) < 12:
            return 0, None
    
        # Normale = gradient de la carte de distance a l'encre.
        dist = cv2.distanceTransform((ink == 0).astype(np.uint8), cv2.DIST_L2, 5)
        gx = cv2.Sobel(dist, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(dist, cv2.CV_32F, 0, 1, ksize=3)
    
        h, w = ink.shape[:2]
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    
        idx = np.linspace(0, len(xs) - 1, min(n_samples, len(xs))).astype(int)
        widths, colors = [], []
        for i in idx:
            x0, y0 = int(xs[i]), int(ys[i])
            vx, vy = float(gx[y0, x0]), float(gy[y0, x0])
            n = (vx * vx + vy * vy) ** 0.5
            if n < 1e-3:
                continue
            vx, vy = vx / n, vy / n
    
            prof = []
            for t in range(1, max_len + 1):
                xx, yy = int(round(x0 + vx * t)), int(round(y0 + vy * t))
                if not (0 <= xx < w and 0 <= yy < h):
                    break
                prof.append(lab[yy, xx])
            if len(prof) < 5:
                continue
            prof = np.array(prof)
    
            # Fond = fin du profil ; contour = prefixe qui en reste loin ET reste
            # proche de lui-meme. On saute le premier pixel, toujours anti-crenele.
            bg = np.median(prof[-3:], axis=0)
            far = [float(np.linalg.norm(pv - bg)) for pv in prof]
            if far[1] < 12.0:
                continue
            ref = prof[1]
            wdt = 0
            for t in range(1, len(prof)):
                if far[t] < 12.0 or float(np.linalg.norm(prof[t] - ref)) > 14.0:
                    break
                wdt = t
            if wdt:
                widths.append(wdt)
                colors.append(crop[
                    min(h - 1, max(0, int(round(y0 + vy * max(1, wdt // 2))))),
                    min(w - 1, max(0, int(round(x0 + vx * max(1, wdt // 2))))),
                ])
    
        if len(widths) < 8:
            return 0, None
        med = int(round(float(np.median(widths))))
        col = np.median(np.array(colors), axis=0)
        return med, col

    @staticmethod
    def _halo_grow(crop: np.ndarray, ink_mask: np.ndarray, max_radius: int = 30) -> np.ndarray:
        """Étend le masque le long de la RAMPE du halo, en s'arrêtant aux arêtes.

        Une dilatation aveugle de N pixels emporte tout ce qui se trouve autour
        du texte, décor compris : sur « JUST KILL ME ALREADY!! », les 25 px
        nécessaires pour couvrir la lueur magenta mordaient aussi les pointes
        violettes du rideau.

        Or les deux se distinguent physiquement :
        - un HALO est une rampe monotone — son écart au fond DÉCROÎT à mesure
          qu'on s'éloigne du glyphe, sans discontinuité ;
        - un élément de DÉCOR commence par une arête — un saut de gradient.

        On fait donc croître le masque pixel par pixel depuis l'encre, en
        n'acceptant un voisin que s'il se rapproche du fond (rampe) et qu'il ne
        porte pas d'arête. La croissance s'arrête d'elle-même sur le décor.
        """
        try:
            m = ink_mask if ink_mask.ndim == 2 else ink_mask[:, :, 0]
            grown = m > 0
            if not grown.any():
                return (grown.astype(np.uint8)) * 255

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

            # Niveau du fond, mesuré LOIN du texte.
            k_far = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * max_radius + 1, 2 * max_radius + 1),
            )
            far = cv2.dilate(m, k_far) == 0
            bg = float(np.median(gray[far])) if far.any() else float(np.median(gray))

            deviation = np.abs(gray - bg)

            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            edge = cv2.magnitude(gx, gy)
            # Seuil d'arête calibré sur le décor lui-même, pas sur une constante.
            edge_thr = float(np.percentile(edge[far], 90)) if far.any() else float(edge.mean() * 2)
            edge_thr = max(edge_thr, 8.0)

            # En deçà de cette tolérance, on est revenu au fond : plus rien à prendre.
            tol = max(4.0, 0.05 * float(deviation[grown].mean()))

            k3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            for _ in range(int(max_radius)):
                g8 = grown.astype(np.uint8)
                cand = (cv2.dilate(g8, k3) > 0) & (~grown)
                if not cand.any():
                    break
                # Écart au fond du voisin DÉJÀ retenu : la rampe doit descendre.
                neigh = cv2.dilate(np.where(grown, deviation, 0.0).astype(np.float32), k3)
                ok = cand & (deviation < neigh) & (deviation > tol) & (edge < edge_thr)
                if not ok.any():
                    break
                grown |= ok

            return (grown.astype(np.uint8)) * 255
        except Exception:
            return ink_mask

    @staticmethod
    def _ghost_remains(erased: np.ndarray, mask: np.ndarray, ratio: float = 1.6) -> bool:
        """Reste-t-il une STRUCTURE là où le texte était ?

        `_erasure_failed` compare l'avant et l'après : il conclut « effacé »
        dès que la zone a beaucoup changé. Or sur du texte à lueur, elle change
        énormément — le cœur blanc disparaît — tout en laissant un halo qui
        redessine les lettres. Mesuré sur « JUST KILL ME ALREADY!! » : blanc
        parti, lueur magenta parfaitement lisible, et « effacement réussi »
        selon ce critère.

        On regarde donc le RÉSULTAT seul : si la zone effacée porte encore
        beaucoup plus d'énergie de contours que le fond juste autour, c'est
        qu'une forme y subsiste. Un fond réellement texturé (trames de vitesse,
        rideau) fait monter les deux côtés et ne déclenche pas.
        """
        try:
            m = mask if mask.ndim == 2 else mask[:, :, 0]
            inside = m > 0
            if int(np.count_nonzero(inside)) < 200:
                return False
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            ring = (cv2.dilate(m, k) > 0) & (~inside)
            if int(np.count_nonzero(ring)) < 200:
                return False

            gray = cv2.cvtColor(erased, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            energy = cv2.magnitude(gx, gy)

            e_in = float(np.mean(energy[inside]))
            e_out = float(np.mean(energy[ring]))
            return (e_in / max(e_out, 1.0)) > ratio
        except Exception:
            return False

    @staticmethod
    def _ghost_score(erased: np.ndarray, mask: np.ndarray) -> float:
        """Énergie de contours dans la zone effacée, rapportée à son pourtour.

        Sert à ACCEPTER ou non une deuxième passe : on garde le résultat qui
        laisse le moins de structure, plutôt que de le juger sur
        `_erasure_failed` — lequel répond « effacé » dès que la zone a changé,
        ce qui est toujours vrai et ne dit rien du fantôme restant.
        """
        try:
            m = mask if mask.ndim == 2 else mask[:, :, 0]
            inside = m > 0
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            ring = (cv2.dilate(m, k) > 0) & (~inside)
            if int(np.count_nonzero(inside)) < 200 or int(np.count_nonzero(ring)) < 200:
                return 0.0
            gray = cv2.cvtColor(erased, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            energy = cv2.magnitude(gx, gy)
            return float(np.mean(energy[inside])) / max(float(np.mean(energy[ring])), 1.0)
        except Exception:
            return 0.0

    @staticmethod
    def _smooth_fill(
        crop: np.ndarray, mask: np.ndarray, max_residual: float = 6.0,
    ) -> Optional[np.ndarray]:
        """Remplace le texte par un modèle de fond LISSE, si ce modèle est
        vérifié sur les pixels voisins non masqués. Sinon None.

        Le modèle : amorce par diffusion à court rayon, puis flou large. Il ne
        peut représenter qu'une variation douce — donc il ne peut pas inventer
        de structure, ni recopier la forme des lettres.

        La validation porte sur les pixels NON masqués d'une bande autour du
        texte : si le modèle les reproduit à `max_residual` près, c'est que le
        fond y est effectivement lisse, et il n'y a aucune raison qu'il cesse de
        l'être sous les lettres. Un fond à trames de vitesse ou à motif échoue
        ce test et repart vers LaMa.
        """
        try:
            m = mask if mask.ndim == 2 else mask[:, :, 0]
            if int(np.count_nonzero(m)) == 0:
                return None

            dist = cv2.distanceTransform((m > 0).astype(np.uint8), cv2.DIST_L2, 5)
            thickness = float(dist.max()) * 2.0
            if thickness <= 0:
                return None
            # Le flou n'a qu'à enjamber l'épaisseur du trait ; plus large, il
            # va chercher des pixels hors de la bulle et le modèle cesse
            # d'expliquer son propre voisinage. Même raison pour la bande de
            # validation : elle doit rester COLLÉE au texte.
            sigma = max(3.0, thickness * 0.6)

            seed = cv2.inpaint(crop, (m > 0).astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
            model = cv2.GaussianBlur(seed, (0, 0), sigma)

            r = max(3, int(round(thickness * 0.8)))
            band_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            band = (cv2.dilate(m, band_k) > 0) & (m == 0)
            if int(np.count_nonzero(band)) < 200:
                return None

            residual = np.abs(
                crop.astype(np.float32) - model.astype(np.float32)
            ).max(axis=2)[band]
            # Médiane ET p75 : la bande peut contenir quelques pixels du trait
            # de la bulle, qu'on ne veut pas laisser décider seuls, mais un vrai
            # motif fait monter les deux.
            if float(np.median(residual)) > max_residual:
                return None
            # Mesuré : dégradé de bulle lisse → médiane 3, p75 4 ; bulle à
            # trames de vitesse → médiane 12, p75 18, donc refusée et confiée à
            # LaMa, qui la reconstruit très bien.
            if float(np.percentile(residual, 75)) > max_residual * 2.0:
                return None

            out = crop.copy()
            out[m > 0] = model[m > 0]
            return out
        except Exception:
            return None

    @staticmethod
    def _diffusion_is_safe(crop: np.ndarray, mask: np.ndarray, max_ratio: float = 0.06) -> bool:
        """Le repli en diffusion Navier-Stokes est-il autorisé ?

        Ce repli a été ajouté quand l'inpainting ne voyait que 30 px de
        contexte autour de la zone : LaMa y laissait des fantômes, et étaler
        les couleurs voisines était le moindre mal. Depuis que la marge de crop
        est indexée sur la taille de la zone, LaMa reconstruit correctement, et
        la diffusion est devenue le pire des deux : elle ne peut reproduire
        aucune structure et fusionne les lignes d'un bloc de dialogue en larges
        bandes grises. Mesuré sur « YOU LITTLE-! YOU'RE WAY TOO CASUAL ABOUT
        THIS! » (bulle à trames de vitesse) : trois bandes grises franches en
        diffusion, trames reconstruites sans défaut par LaMa au même endroit.

        Le critère de couleur de `_background_is_diffusable` ne sépare pas ces
        cas, et un critère de taille non plus (l'épaisseur du trait suit la
        taille de police). On désactive donc le repli par défaut, en le
        laissant réactivable si une série y perdait quelque chose.
        """
        try:
            from config import config as _cfg
            if not bool(getattr(_cfg.rendering, 'diffusion_fallback_enabled', False)):
                return False
            area = float(mask.shape[0] * mask.shape[1])
            if area <= 0:
                return False
            return (float(np.count_nonzero(mask)) / area) <= max_ratio
        except Exception:
            return False

    @staticmethod
    def _background_is_diffusable(crop: np.ndarray, mask: np.ndarray, margin: int = 25, class_name: str = "") -> bool:
        """
        Le repli en diffusion pure (`_diffuse_fill`) n'a de sens QUE sur un
        fond lisse (dégradé, halo) : il ne peut reproduire aucune texture,
        juste étaler la couleur des pixels voisins. Sur un fond avec du vrai
        motif (ruban, trame, bordure décorative), il produit de grosses
        taches de couleur — mesuré sur une carte System à bordure dorée avec
        motif : plus visible et plus faux que le fantôme que LaMa avait
        laissé. Autant garder alors le résultat de LaMa, imparfait mais
        crédible, plutôt que d'y substituer un artefact pire.

        Premier essai : résidu haute fréquence en NIVEAUX DE GRIS (écart à un
        flou large) dans la couronne juste hors masque. Rejeté : un fond en
        dégradé sombre (bulle noire, carte System avec halo) accumule du bruit
        de compression JPEG proportionnellement plus grand dans les tons
        foncés, et ce bruit — pourtant sans aucun rapport avec une vraie
        texture — faisait grimper le résidu au-delà même du cas de motif réel
        qu'on cherchait à détecter. Mesuré : ~28 sur un fond noir uni (qu'on
        voulait accepter) contre ~17 sur le ruban à motif (qu'on voulait
        rejeter) — l'ordre était inversé, aucun seuil ne pouvait séparer les
        deux correctement.

        Ce qui sépare net : l'écart-type de la COULEUR (BGR, pas la
        luminosité) dans cette même couronne. Un dégradé ou un halo, même
        sombre et bruité, reste dans une gamme de TEINTES étroite — seule la
        luminosité varie. Un vrai motif (ruban doré + éléments numériques
        teal/marine) mélange des teintes franchement différentes. Mesuré :
        9.9 à 44.3 sur quatre fonds à dégradé/halo (bulle grise, bulle noire,
        carte teal, panneau noir+lueur) contre 78.4 sur le ruban à motif —
        écart net, pas de zone ambiguë entre les deux groupes.
        """
        ring = (cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin, margin))) > 0) & (mask == 0)
        if not ring.any():
            return True
        color_std = float(crop[ring].reshape(-1, 3).std(axis=0).max())
        max_std = 20.0 if str(class_name).lower() == "out_text" else 55.0
        return color_std < max_std

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
        inside_box: Optional[np.ndarray] = None,
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

            # La longue portée n'est justifiée que HORS de la boîte de
            # détection : c'est là que vit le morceau de glyphe coupé par la
            # boîte, qu'on veut rattraper. À l'INTÉRIEUR, l'anticrénelage tient
            # dans 2 ou 3 px, tandis que 21 px suffisent à atteindre le trait
            # du ballon dès qu'une ligne de texte est large — le trait, sombre,
            # « s'écarte du fond », est connexe aux lettres, et se retrouve
            # repeint en blanc. Mesuré sur « IT MIGHT JUST BE A NORMAL RUN,
            # BUT... » : 574 px absorbés, dont 48 sur le contour, à gauche et à
            # droite à la même hauteur — un décrochement bien visible.
            if inside_box is not None and inside_box.shape[:2] == mask.shape[:2]:
                short = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                near_short = cv2.dilate(mask, short, iterations=1) > 0
                inside = inside_box > 0
                near = (near_short & inside) | (near & ~inside)
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
        crop: np.ndarray, mask: np.ndarray, max_std: float = 12.0, local_bubble_mask: Optional[np.ndarray] = None, class_name: str = ""
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
            if local_bubble_mask is not None:
                ring = ring & (local_bubble_mask > 0)
            samples = crop[ring]
            if samples.shape[0] < 64:
                return None
            samples = samples.reshape(-1, 3)
            reference = np.median(samples.astype(np.float32), axis=0)

            # Critère robuste : on accepte plus de bruit sur du vrai blanc/noir (bulles),
            # mais on est très strict sur les couleurs pour ne pas aplatir des ciels/décors.
            is_white_or_black = np.all(reference >= 235) or np.all(reference <= 30)

            # Les out_text sont typiquement posés sur des décors. Un aplat ferait une
            # barre — y COMPRIS quand le carve-out blanc/noir ci-dessous serait
            # satisfait : un dégradé texturé (trame de vitesse, halo) qui s'estompe
            # vers le blanc ressort blanc en MÉDIANE locale sans être un fond uni.
            # Mesuré sur "EIGHT YEARS AGO" (trame de vitesse s'estompant vers le bas) :
            # la couronne autour de "AGO" ressortait blanche en médiane, le carve-out
            # is_white_or_black passait outre l'exclusion out_text, et l'aplat
            # dessinait un rectangle flou par-dessus la trame — pas de "fond uni"
            # légitime à distinguer ici, donc pas de carve-out pour cette classe.
            if str(class_name).lower() == "out_text":
                return None
                
            # L'ancien seuil de 3.0 pour les fonds colorés refusait les gris
            # clairs (RGB~230, écart ~4-6) et les beiges. Tout partait à LaMa
            # qui laissait des fantômes. 6.0 accepte les fonds quasi-unis
            # courants (narration beige, cases grises) sans risquer d'aplatir
            # un vrai décor (qui a bien plus de 6 niveaux d'écart).
            effective_std = max_std if is_white_or_black else 6.0

            deviation = np.abs(samples.astype(np.float32) - reference).max(axis=1)
            if float(np.mean(deviation <= effective_std)) < 0.85:
                return None

            # Mode et non médiane
            inliers = samples[deviation <= effective_std]
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

    @staticmethod
    def _block_mask_from_regions(
        crop_w: int, crop_h: int, regions: Optional[List[Dict]],
        offset_x: int, offset_y: int, grow_ratio: float = 0.30,
    ) -> Optional[np.ndarray]:
        """Masque au BLOC : polygones de ligne OCR dilatés de `grow_ratio` fois
        la hauteur de ligne médiane, en coordonnées du crop.

        Destiné au texte d'impact, dont la lueur externe et le contour épais
        débordent largement du glyphe : viser le glyphe y laisse un halo qui
        redessine le rectangle du bloc de texte. On efface donc la bande de
        ligne entière et on laisse le modèle d'inpainting reconstruire — ce
        qu'il fait bien tant qu'on lui laisse assez de contexte autour.
        """
        if not regions:
            return None
        mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        heights: List[float] = []
        for region in regions:
            raw = region.get('bbox') if isinstance(region, dict) else None
            if not raw or len(raw) < 3:
                continue
            try:
                pts = np.array(
                    [[int(p[0]) + offset_x, int(p[1]) + offset_y] for p in raw],
                    dtype=np.int32,
                )
            except Exception:
                continue
            pts[:, 0] = np.clip(pts[:, 0], 0, crop_w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, crop_h - 1)
            cv2.fillPoly(mask, [pts], 255)
            heights.append(float(pts[:, 1].max() - pts[:, 1].min()))
        if int(np.count_nonzero(mask)) == 0:
            return None
        ref_h = float(np.median(heights)) if heights else 30.0
        grow = int(max(2.0, round(grow_ratio * ref_h)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
        return cv2.dilate(mask, kernel, iterations=1)

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
        ink_mask: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[int, int, int]]:
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # Le masque d'ENCRE d'abord, les polygones seulement en repli.
        #
        # Un polygone de ligne OCR est un rectangle : le remplir prend les
        # lettres ET le fond entre les lettres ET, sur du texte d'impact, tout
        # le contour blanc. La couleur qui en sort est une moyenne délavée.
        # Mesuré sur les cartouches rouges de the-frontier-count ch1 : le
        # polygone donnait (220, 183, 179), un rose pâle qui, faute de
        # contraste, se faisait ensuite recouvrir par son propre contour au
        # rendu — le texte sortait blanc ou noir au lieu de rouge. Le masque
        # d'encre, lui, donne (150, 60, 55), le vrai cramoisi de la planche.
        local_mask = None
        from_ink = False
        if isinstance(ink_mask, np.ndarray) and ink_mask.size > 0:
            m = ink_mask[:, :, 0] if ink_mask.ndim == 3 else ink_mask
            if m.shape[:2] != crop.shape[:2]:
                try:
                    m = cv2.resize(m, (crop.shape[1], crop.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
                except Exception:
                    m = None
            if m is not None and int(np.count_nonzero(m)) > 32:
                # Éroder un peu pour ne garder que le CŒUR du trait, sans la
                # frange anticrénelée qui tire la couleur vers le fond.
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                core = cv2.erode((m > 0).astype(np.uint8) * 255, k, iterations=1)
                local_mask = core if int(np.count_nonzero(core)) > 32 else m
                from_ink = True

        if from_ink and local_mask is not None:
            # Médiane directe, PAS le k-means du chemin polygone.
            #
            # Ce k-means retient le groupe le plus ÉLOIGNÉ du fond. Avec le
            # masque d'encre, « le fond » est tout ce qui n'est pas encre —
            # c'est-à-dire le noir de la case. Sur un cartouche rouge à contour
            # blanc, le groupe le plus éloigné du noir est le CONTOUR BLANC,
            # pas le rouge : mesuré (224, 187, 182) là où la médiane du cœur du
            # trait donne (150, 60, 55), le vrai cramoisi. L'érosion ayant déjà
            # ôté le contour, la médiane est ici la bonne mesure — et elle ne
            # peut pas se tromper de groupe.
            core_px = crop[local_mask > 0]
            if core_px.size >= 3 * 32:
                med = np.median(core_px.reshape(-1, 3), axis=0)
                return (int(med[2]), int(med[1]), int(med[0]))

        if local_mask is None:
            local_mask = self._build_local_mask_from_regions(
                crop.shape[1], crop.shape[0], mask_regions,
            )
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


    # Un fragment plus court que ça en début ou fin de mot est illisible et
    # laid : mieux vaut renoncer à couper.
    HYPHEN_MIN_PIECE = 3
    HYPHEN_MIN_WORD = 8

    def _hyphenate_to_fit(
        self, word: str, font: ImageFont.FreeTypeFont, max_width: int,
    ) -> Optional[Tuple[str, str]]:
        """(début + trait d'union, reste) si le mot peut être coupé, sinon None.

        Un mot est INSÉCABLE au sens de la mise en page : il doit tenir sur une
        ligne. Quand il n'y tient pas, tout le bloc rétrécit pour lui. Mesuré
        sur « RAVITAILLEMENT... » dans une bulle de 156 px : il plafonnait la
        bulle entière à 16 px, alors que le reste du texte tenait bien plus
        gros. La césure lève ce plafond.
        """
        core = word.strip()
        if len(core) < self.HYPHEN_MIN_WORD:
            return None
        dic = _hyphenator()
        if dic is None:
            return None

        # La ponctuation collée (« ... ») ne se coupe pas : on l'isole.
        head_letters = "".join(c for c in core if c.isalpha())
        if len(head_letters) < self.HYPHEN_MIN_WORD:
            return None
        try:
            positions = [i for i in dic.positions(head_letters.lower())]
        except Exception:
            return None
        if not positions:
            return None

        best = None
        for pos in positions:
            if pos < self.HYPHEN_MIN_PIECE or len(head_letters) - pos < self.HYPHEN_MIN_PIECE:
                continue
            piece = head_letters[:pos] + "-"
            if self._line_extents(font, piece)[1] <= max_width:
                best = pos
            else:
                break
        if best is None:
            return None

        cut = core.find(head_letters[best - 1]) if False else best
        return (core[:cut] + "-", core[cut:])


    def _hyphenate_word(
        self, word: str, font: ImageFont.FreeTypeFont,
        first_width: int, next_width: int, max_pieces: int = 4,
    ) -> Optional[List[str]]:
        """Découpe un mot en morceaux qui tiennent TOUS dans leur ligne.

        Deux défauts d'une première version, tous deux mesurés sur
        « RAVITAILLEMENT... » dans une bulle de 157 px de large :

        - une seule coupure était tentée. À 28 px, la première césure possible
          donne « RAVI- » et laisse « TAILLEMENT... » large de 198 px — soit
          une ligne qui déborde, donc une mise en page rejetée, donc le moteur
          qui redescend en taille. Il faut recouper le RESTE tant qu'il ne
          tient pas.
        - la coupure n'était tentée que dans la place restante sur la ligne
          courante. Quand cette place est trop petite, il faut repartir sur une
          ligne NEUVE et couper sur toute sa largeur, au lieu d'abandonner.
        """
        pieces: List[str] = []
        rest = word
        width = first_width
        for _ in range(max_pieces):
            if self._line_extents(font, rest)[1] <= width:
                pieces.append(rest)
                return pieces if len(pieces) > 1 else None
            cut = self._hyphenate_to_fit(rest, font, width)
            if not cut:
                return None
            head, rest = cut
            pieces.append(head)
            width = next_width
        return None

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
                    # Le mot ne rentre pas : on tente la césure AVANT de le
                    # renvoyer seul sur la ligne suivante, où il imposerait sa
                    # largeur à tout le bloc.
                    placed = False
                    if self._line_extents(font, word)[1] > max_width:
                        room = max_width
                        if current:
                            room -= self._line_extents(font, ' '.join(current) + ' ')[1]
                        cut = self._hyphenate_to_fit(word, font, room) if room > 20 else None
                        if cut:
                            head, tail = cut
                            lines.append(' '.join(current + [head]) if current else head)
                            current = [tail]
                            placed = True
                    if not placed:
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

        # Sécurité absolue : la police ne doit jamais être plus haute que la bulle
        fs = min(fs, int(inner_h * 0.85))

        return max(self.cfg.min_font_size, min(fs, self.cfg.max_font_size))

    @staticmethod
    def _mask_row_center(mask: np.ndarray, y0: float, y1: float) -> Optional[float]:
        """
        Centre-X (coordonnées LOCALES au masque) de la zone solide sur la
        bande [y0, y1), ou None si la bande est vide.

        `_mask_row_span` donne la LARGEUR utilisable (par comptage, robuste
        aux bulles à deux pointes) mais pas sa POSITION. Sans le centre par
        ligne, `_draw_block` centrait chaque ligne sur le centre du
        RECTANGLE inscrit global — correct pour un ovale symétrique, mais
        pas pour une bulle asymétrique (queue qui tire le centre de masse
        d'un côté) : la ligne du bas débordait alors du bord opposé à la
        queue, mesuré sur "YOU'RE MY ONLY BLOOD RELATIVE" (texte collé au
        bord droit).
        """
        h, w = mask.shape[:2]
        y0c = max(0, min(h, int(round(y0))))
        y1c = max(y0c + 1, min(h, int(round(y1))))
        band = mask[y0c:y1c, :]
        if band.size == 0:
            return None
        cols = np.nonzero(np.count_nonzero(band > 0, axis=0))[0]
        if cols.size == 0:
            return None
        return float(cols.min() + cols.max()) / 2.0

    @staticmethod
    def _mask_row_span(mask: np.ndarray, y0: float, y1: float) -> float:
        """
        Largeur RÉELLE moyenne (en pixels du masque) de la bulle sur la bande
        de lignes [y0, y1) — mesurée comme le ratio de pixels solides.
        Contrairement à max(x)-min(x) qui donne une fausse grande largeur
        s'il y a deux pointes espacées (bulle de cri), le comptage strict
        reflète la vraie place utilisable. Retourne 0 si hors masque.
        """
        h = mask.shape[0]
        y0c = max(0, min(h, int(round(y0))))
        y1c = max(y0c + 1, min(h, int(round(y1))))
        band = mask[y0c:y1c, :]
        if band.size == 0:
            return 0.0
        return float(np.count_nonzero(band > 0)) / float(band.shape[0])

    def _wrap_text_by_mask(
        self, text: str, font: ImageFont.FreeTypeFont,
        inner_w: int, inner_h: int, line_h: int, spacing: int,
        bubble_mask: np.ndarray, mask_y_offset: float, mask_x_origin: float = 0.0,
    ) -> Tuple[List[str], List[int], List[float]]:
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

        Retourne aussi le centre-X (coordonnées GLOBALES, `mask_x_origin`
        déjà ajouté) de chaque ligne : `_draw_block` centrait auparavant
        TOUTES les lignes sur le centre du rectangle inscrit global, correct
        pour un ovale symétrique mais pas pour une bulle avec une queue —
        le centre de masse du masque est alors décalé, et une ligne large
        proche du bord opposé à la queue débordait du contour.
        """
        rough_w = max(10, int(inner_w * self.cfg.word_wrap_ratio))
        # Centre de référence = centre horizontal RÉEL du ballon, mesuré sur le
        # masque. `mask_x_origin + inner_w / 2` (l'ancienne formule) mélangeait
        # deux repères : l'origine est celle de la BBOX, la largeur celle de la
        # zone INTÉRIEURE, plus étroite — le centre obtenu était donc décalé
        # vers la gauche de la moitié de la marge, et tout le bloc avec lui.
        # Visible sur les bulles serrées de path-of-vengeance ch1, où le texte
        # sortait par le bord gauche.
        cols_all = np.nonzero(np.count_nonzero(bubble_mask > 0, axis=0))[0]
        if cols_all.size:
            default_center = mask_x_origin + float(cols_all.min() + cols_all.max()) / 2.0
        else:
            default_center = mask_x_origin + bubble_mask.shape[1] / 2.0
        words = re.sub(r"\s+", " ", text or "").strip().split()
        if not words:
            return [""], [rough_w], [default_center]

        # Largeur TYPIQUE du ballon : médiane des largeurs de bande sur sa
        # moitié centrale. C'est la référence du plancher ci-dessous — pas
        # `inner_w`, qui est le rectangle inscrit à l'aveugle et vaut ici 138 px
        # pour un ballon large de 202 px. Un plancher calé sur ce rectangle
        # laissait le texte se fragmenter en mots isolés dès que le bloc
        # grandissait et poussait ses lignes vers les pointes étroites de
        # l'ovale — « MAKE SURE YOU COME BACK SAFE, OKAY? » sortait en six
        # lignes d'un mot au lieu des trois du lettrage d'origine.
        mh = bubble_mask.shape[0]
        band = max(1, line_h)
        spans = [
            self._mask_row_span(bubble_mask, y, y + band)
            for y in range(int(mh * 0.25), max(int(mh * 0.25) + 1, int(mh * 0.75)), max(1, band // 2))
        ]
        spans = [s for s in spans if s > 0]
        typical_w = float(np.median(spans)) if spans else float(inner_w)
        floor_w = int(max(inner_w * self.cfg.word_wrap_ratio * 0.60,
                          typical_w * self.cfg.word_wrap_ratio * 0.70))

        def _wrap_with(n_lines_assumed: int) -> Tuple[List[str], List[int], List[float]]:
            total_h = n_lines_assumed * line_h + max(0, n_lines_assumed - 1) * spacing
            ys_local = max(0, (inner_h - total_h) // 2)

            def _row_metrics(idx: int) -> Tuple[int, float]:
                y0 = mask_y_offset + ys_local + idx * (line_h + spacing)
                w = self._mask_row_span(bubble_mask, y0, y0 + line_h)
                center = self._mask_row_center(bubble_mask, y0, y0 + line_h)
                # Plancher de largeur. `_mask_row_span` mesure la largeur
                # SOLIDE moyenne de la bande : près des pointes haute et basse
                # d'un ovale elle tombe à quelques dizaines de pixels, la ligne
                # se réduit à un mot, le bloc gagne une ligne, qui repousse la
                # ligne suivante encore plus près de la pointe — spirale.
                # Mesuré sur pov ch1 : « UNCLE WAS A PLAYER ON THE FRONT
                # LINES… » sortait en 7 lignes pour 8 mots et remplissait 32 %
                # de la bulle, alors que le rectangle INSCRIT (158 px de large,
                # donc garanti dans la bulle) en tenait 4. Le plancher est
                # exprimé sur ce rectangle inscrit : il ne peut pas faire
                # déborder le texte.
                # Plancher SEULEMENT — surtout pas de plafond : le rectangle
                # inscrit `inner_w` est volontairement étroit (il doit tenir
                # dans l'ovale à l'aveugle), et plafonner la largeur de ligne
                # dessus fait déclarer « ne tient pas » à toutes les tailles,
                # ce qui renvoie le texte au plancher de police. Vérifié :
                # l'ajout d'un plafond a fait tomber « YOU'RE MY ONLY BLOOD
                # RELATIVE » à une taille minuscule collée en haut à gauche.
                allowed_w = max(floor_w, int(w * self.cfg.word_wrap_ratio)) if w > 0 else floor_w
                center_g = (center + mask_x_origin) if center is not None else default_center
                # Le centre de bande dérive quand le masque de bulle est
                # asymétrique (queue, fusion partielle avec le fond) : mesuré
                # jusqu'à 57 % de `inner_w` d'écart sur les petites bulles, le
                # bloc se retrouvait plaqué contre un bord. On borne la dérive
                # sans toucher à la largeur.
                max_drift = inner_w * 0.20
                center_g = min(max(center_g, default_center - max_drift),
                               default_center + max_drift)
                return allowed_w, center_g

            lines: List[str] = []
            allowed: List[int] = []
            centers: List[float] = []
            current: List[str] = []
            current_w, current_c = _row_metrics(0)
            for word in words:
                test = ' '.join(current + [word])
                if self._line_extents(font, test)[1] <= current_w:
                    current.append(word)
                else:
                    # Césure AVANT de renvoyer le mot seul sur la ligne
                    # suivante. C'est ce chemin-ci qu'empruntent les bulles —
                    # `wrap_text` ne sert qu'aux zones rectangulaires — donc
                    # c'est ici que la césure doit agir pour lever le plafond
                    # qu'un mot long impose à toute la bulle.
                    placed = False
                    if self._line_extents(font, word)[1] > current_w:
                        room = current_w
                        if current:
                            room -= self._line_extents(font, ' '.join(current) + ' ')[1]
                        nxt = _row_metrics(len(lines) + 1)[0]
                        # Si la place restante est trop courte, on coupe sur une
                        # ligne NEUVE plutot que de renoncer.
                        if room <= 20:
                            if current:
                                lines.append(' '.join(current))
                                allowed.append(current_w)
                                centers.append(current_c)
                                current = []
                                current_w, current_c = _row_metrics(len(lines))
                            room = current_w
                        chunks = self._hyphenate_word(word, font, room, nxt or current_w)
                        if not chunks and current:
                            # La place restante ne permettait pas de couper. On
                            # vide la ligne et on retente sur une ligne NEUVE,
                            # à pleine largeur. Sans ça, le mot partait entier
                            # sur la ligne suivante et la faisait déborder — ce
                            # qui rendait le prédicat « ça tient » NON MONOTONE
                            # (mesuré : 24 px échouait alors que 28 px tenait,
                            # la césure ne se déclenchant qu'à partir de 28) et
                            # faisait manquer la bonne taille à la dichotomie.
                            lines.append(' '.join(current))
                            allowed.append(current_w)
                            centers.append(current_c)
                            current = []
                            current_w, current_c = _row_metrics(len(lines))
                            chunks = self._hyphenate_word(
                                word, font, current_w, nxt or current_w,
                            )
                        if chunks:
                            head, tail = chunks[0], chunks[1:]
                            lines.append(' '.join(current + [head]) if current else head)
                            allowed.append(current_w)
                            centers.append(current_c)
                            for piece in tail[:-1]:
                                current_w, current_c = _row_metrics(len(lines))
                                lines.append(piece)
                                allowed.append(current_w)
                                centers.append(current_c)
                            current = [tail[-1]]
                            current_w, current_c = _row_metrics(len(lines))
                            placed = True
                    if not placed:
                        if current:
                            lines.append(' '.join(current))
                            allowed.append(current_w)
                            centers.append(current_c)
                        current = [word]
                        current_w, current_c = _row_metrics(len(lines))
            if current:
                lines.append(' '.join(current))
                allowed.append(current_w)
                centers.append(current_c)
            return lines, allowed, centers

        # Point fixe : la hauteur du bloc décide d'où commence la première
        # ligne, donc de la bande de masque mesurée pour chaque ligne — mais
        # cette hauteur dépend du découpage qu'on est en train de calculer.
        # Une seule passe sur une estimation rectangulaire mesurait les lignes
        # du bas à des bandes trop hautes (donc trop larges) : elles sortaient
        # de l'ovale par la gauche et la droite.
        n = max(1, len(self.wrap_text(text, font, rough_w)))
        lines, allowed, centers = _wrap_with(n)
        # 3 passes ne suffisaient pas à stabiliser le point fixe sur les blocs
        # de 5 lignes et plus : on rendait alors un découpage calculé pour une
        # hauteur de bloc qui n'était pas la bonne.
        for _ in range(6):
            if len(lines) == n:
                break
            n = len(lines)
            lines, allowed, centers = _wrap_with(n)

        import os
        if os.environ.get('RENDER_DEBUG'):
            print(f"[RENDER_DEBUG] wrap_by_mask text={text[:30]!r} mask_shape={bubble_mask.shape if bubble_mask is not None else None} mask_nnz={int(np.count_nonzero(bubble_mask)) if bubble_mask is not None else None} mask_y_offset={mask_y_offset} inner_w={inner_w} inner_h={inner_h} -> lines={lines!r} allowed={allowed!r} centers={centers!r}")
        return (lines, allowed, centers) if lines else ([""], [rough_w], [default_center])

    # Poids du mot orphelin dans le rééquilibrage, à comparer au coefficient de
    # variation des largeurs (typiquement 0,10 à 0,40). Une ligne finale d'un
    # seul mot est le défaut de lettrage le plus visible : on lui donne un poids
    # du même ordre que le pire déséquilibre.
    POIDS_ORPHELIN = 0.30

    # Prix d'une ligne AU-DELÀ du découpage de la planche, en unités
    # logarithmiques de corps — même échelle que `_route_cost`.
    #
    # L'ancien budget était une contrainte DURE : dès que le bloc dépassait
    # `lignes_source + 1`, on redescendait la taille jusqu'à rentrer dedans, à
    # n'importe quel prix. Autrement dit λ = ∞. Mesuré sur le corpus 16
    # planches : **84 des 141 bulles `corps_petit` sortent de ce rattrapage**,
    # pas d'un défaut d'ajustement — elles ne butent sur rien (aucun blocage
    # relevé) et ne sont pas au plafond (4/84). Leur `cap_ratio` médian tombe à
    # 0,72.
    #
    # La police de rendu étant plus large que celle du studio, le même texte a
    # légitimement besoin de plus de lignes : le nombre de lignes de la planche
    # est une INDICATION, pas une contrainte. 0,12 se lit « une ligne de plus
    # est acceptable si elle rapporte plus de 12 % de corps ».
    PENALITE_LIGNE_SUP = float(os.environ.get("WEBTOON_PENALITE_LIGNE_SUP", "0.12"))

    def _rebalance_lines(
        self, lines: List[str], allowed: List[float], font: ImageFont.FreeTypeFont,
    ) -> List[str]:
        """Rééquilibre un pavé SANS changer son nombre de lignes.

        `wrap_text` et `_wrap_text_by_mask` remplissent gloutonnement : chaque
        ligne prend tout ce qu'elle peut et le reliquat tombe sur la dernière,
        d'où le mot ORPHELIN — mesuré sur 67 des 71 blocs fautifs du corpus,
        tous sur la route `box`. Un lettreur, lui, équilibre ses lignes.

        On déplace des mots d'une ligne vers sa voisine tant que ça rapproche
        les largeurs, en refusant tout déplacement qui ferait dépasser une ligne
        de la largeur qui lui est allouée — pour une bulle, c'est la largeur du
        ballon À SA hauteur, donc la contrainte reste celle de la forme.

        Le nombre de lignes ne bouge JAMAIS : la hauteur du bloc, et donc la
        taille de police déjà retenue par la dichotomie, restent valides. C'est
        ce qui permet de poser ce rééquilibrage sans toucher à la recherche de
        taille.
        """
        if len(lines) < 2 or len(lines) > 12:
            return lines
        mots = [ln.split() for ln in lines]
        if any(not m for m in mots):
            return lines          # ligne vide : mise en page voulue, on n'y touche pas

        def largeurs(mm):
            return [self._line_extents(font, " ".join(m))[1] for m in mm]

        def tient(mm):
            return all(w <= a for w, a in zip(largeurs(mm), allowed))

        def cout(mm):
            ws = largeurs(mm)
            moyenne = sum(ws) / len(ws)
            if moyenne <= 0:
                return 0.0
            ecart = (sum((w - moyenne) ** 2 for w in ws) / len(ws)) ** 0.5
            c = ecart / moyenne
            if len(mm[-1]) == 1:
                c += self.POIDS_ORPHELIN
            return c

        meilleur = cout(mots)
        for _ in range(6):
            bouge = False
            for i in range(len(mots) - 1):
                for cand in (self._mot_descendu(mots, i), self._mot_remonte(mots, i)):
                    if cand is None or not tient(cand):
                        continue
                    c = cout(cand)
                    if c < meilleur - 1e-9:
                        mots, meilleur, bouge = cand, c, True
            if not bouge:
                break
        return [" ".join(m) for m in mots]

    @staticmethod
    def _mot_descendu(mots, i):
        """Dernier mot de la ligne i poussé en tête de la ligne i+1."""
        if len(mots[i]) < 2:
            return None
        cand = [list(m) for m in mots]
        cand[i + 1].insert(0, cand[i].pop())
        return cand

    @staticmethod
    def _mot_remonte(mots, i):
        """Premier mot de la ligne i+1 remonté en fin de ligne i."""
        if len(mots[i + 1]) < 2:
            return None
        cand = [list(m) for m in mots]
        cand[i].append(cand[i + 1].pop(0))
        return cand

    def _layout_at_size(
        self, text: str, font_size: int, inner_w: int, inner_h: int,
        font_path: Optional[str], use_mask_wrap: bool,
        bubble_mask: Optional[np.ndarray], mask_y_offset: float, mask_x_origin: float = 0.0,
    ) -> Optional[Dict]:
        """Découpe + mesure du bloc de texte à une taille donnée."""
        font = self._load_font_from_path(font_path, font_size)
        if font is None:
            return None

        line_h, ascent = self._font_metrics(font)
        spacing = int(line_h * self.cfg.line_spacing_ratio)
        line_centers: Optional[List[float]] = None

        if use_mask_wrap and bubble_mask is not None:
            lines, allowed, line_centers = self._wrap_text_by_mask(
                text, font, inner_w, inner_h, line_h, spacing, bubble_mask, mask_y_offset, mask_x_origin,
            )
            # Chaque ligne est jugée sur la largeur de la bulle À SA hauteur.
            fits_width = all(
                self._line_extents(font, ln)[1] <= aw
                for ln, aw in zip(lines, allowed)
            )
        else:
            largeur_max = max(10, int(inner_w * self.cfg.word_wrap_ratio))
            lines = self.wrap_text(text, font, largeur_max)
            allowed = [float(largeur_max)] * len(lines)
            fits_width = all(
                self._line_extents(font, ln)[1] <= inner_w for ln in lines
            )

        # Rééquilibrage commun aux deux découpeurs. Les retours à la ligne
        # explicites (cartes System) sont une mise en page voulue : on n'y touche
        # pas.
        if chr(10) not in (text or ""):
            lines = self._rebalance_lines(lines, allowed, font)

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
            'line_centers': line_centers,
            # Les deux contraintes SÉPARÉMENT : savoir laquelle mord dit où est
            # la marge de manoeuvre. Une taille bloquée par la HAUTEUR ne peut
            # être débloquée qu'en enlevant une ligne ; bloquée par la LARGEUR,
            # c'est le découpage qui est en cause.
            'fits_h': total_h <= inner_h,
            'fits_w': fits_width,
            'fits': total_h <= inner_h and fits_width,
        }

    def _fit_font_hard(
        self, text: str, font_size: int, inner_w: int, inner_h: int,
        bubble_mask: Optional[np.ndarray] = None, shape_wrap: bool = False,
        font_path: Optional[str] = None, mask_y_offset: float = 0,
        max_font_size: Optional[int] = None, mask_x_origin: float = 0.0,
        target_lines: Optional[int] = None,
        length_ratio: Optional[float] = None,
        em_source: Optional[float] = None,
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
        # Contrainte qui a bloqué la taille juste au-dessus de la retenue.
        blocage: Optional[Dict] = None
        while lo <= hi:
            mid = (lo + hi) // 2
            layout = self._layout_at_size(
                text, mid, inner_w, inner_h, font_path, use_mask_wrap, bubble_mask, mask_y_offset, mask_x_origin,
            )
            if layout is None:
                return None
            if layout['fits']:
                best = layout
                lo = mid + 1
            else:
                if blocage is None or mid < blocage['size']:
                    blocage = {
                        'size': mid,
                        'hauteur': not layout['fits_h'],
                        'largeur': not layout['fits_w'],
                        'n_lines': len(layout.get('lines') or []),
                    }
                hi = mid - 1
        if best is not None and blocage is not None:
            best['blocage'] = blocage

        # « Tient » ne suffit pas comme critère : maximiser la taille produit
        # volontiers un empilement de mots isolés qui tient très bien dans la
        # hauteur mais ne ressemble pas à du lettrage. Mesuré sur « MAKE SURE
        # YOU COME BACK SAFE, OKAY? » : six lignes d'un mot là où la planche
        # d'origine en fait trois — la police de rendu est plus large que celle
        # du studio, donc à hauteur de casse égale il faut plus de largeur.
        #
        # Le découpage d'origine est connu : c'est le nombre de polygones de
        # ligne OCR. On redescend donc d'un cran en taille tant que le bloc a
        # plus de lignes que la planche d'origine (tolérance +1, la traduction
        # pouvant être plus longue que la source).
        if best is not None and target_lines and target_lines > 0:
            # Budget de lignes INDEXE SUR LE FOISONNEMENT.
            #
            # Un `+1` constant enfermait la traduction dans le decoupage de la
            # planche : le moteur epuisait la reduction de taille avant
            # d'envisager une ligne de plus. Mesure sur path-of-vengeance,
            # bulle « WELL... BUT FOR A SUPPLY RUN... » traduite par « Enfin...
            # Mais pour une mission de ravitaillement... » : 33 px sur 5 lignes
            # devenaient 16 px sur 4 lignes — plus de la moitie du corps perdue
            # alors qu'il restait de la place en hauteur.
            #
            # Un texte 17 % plus long a droit a ~17 % de lignes en plus. La
            # regle vaut pour n'importe quelle langue cible sans rien coder en
            # dur : elle se deduit du texte lui-meme.
            budget = int(target_lines) + 1
            if length_ratio and length_ratio > 1.0:
                budget = max(budget, int(math.ceil(target_lines * length_ratio)) + 1)
            def _a_orphelin(lay):
                ll = [l for l in (lay.get('lines') or []) if l and l.strip()]
                return len(ll) >= 2 and len(ll[-1].split()) == 1 and len(ll[-1]) <= 6

            # ARBITRAGE de taille par coût. Déclenché quand le découpage dépasse
            # le budget de lignes OU laisse un mot ORPHELIN : dans les deux cas,
            # descendre d'un ou deux crans peut donner une mise en page
            # nettement meilleure pour une perte de corps minime, et le coût
            # tranche. Une ligne de trop coûte `PENALITE_LIGNE_SUP`, un orphelin
            # est pénalisé dans `_rag_penalty` ; aucun ne coûte plus l'infini.
            if len(best.get('lines') or []) > budget or _a_orphelin(best):
                em = float(em_source) if em_source else (
                    float(max_font_size) / 1.05 if max_font_size else None)

                def _cout(lay):
                    if lay is None or not lay.get('fits'):
                        return float('inf')
                    c = 0.0
                    if em and em > 0 and lay.get('size'):
                        c += abs(math.log(float(lay['size']) / em))
                    sup = max(0, len(lay.get('lines') or []) - budget)
                    c += self.PENALITE_LIGNE_SUP * sup
                    c += self._rag_penalty(lay.get('lines'))
                    return c

                meilleur, cout_meilleur = best, _cout(best)
                # 8 crans : assez pour absorber un orphelin ou une ligne de trop
                # (chacun coûte ~0,15-0,30, soit 2-4 crans de corps), pas plus —
                # au-delà la perte de corps l'emporte de toute façon.
                bas = max(int(self.cfg.min_font_size), int(best['size']) - 8)
                resolu_depuis = 0
                for size in range(int(best['size']) - 1, bas - 1, -1):
                    cand = self._layout_at_size(
                        text, size, inner_w, inner_h, font_path, use_mask_wrap,
                        bubble_mask, mask_y_offset, mask_x_origin,
                    )
                    if cand is None:
                        break
                    c = _cout(cand)
                    if c < cout_meilleur:
                        meilleur, cout_meilleur = cand, c
                    # Arrêt : une fois budget respecté ET orphelin résorbé, on
                    # explore encore 2 crans (le coût peut encore baisser d'un
                    # cheveu) puis on s'arrête — inutile de rétrécir davantage.
                    if len(cand.get('lines') or []) <= budget and not _a_orphelin(cand):
                        resolu_depuis += 1
                        if resolu_depuis >= 2:
                            break
                if meilleur is not best:
                    meilleur['blocage'] = best.get('blocage')
                    meilleur['arbitrage_budget'] = True
                best = meilleur

        if best is not None:
            return best

        # Rien ne tient, même au plancher : on rend quand même, mais on le dit.
        fallback = self._layout_at_size(
            text, int(self.cfg.min_font_size), inner_w, inner_h,
            font_path, use_mask_wrap, bubble_mask, mask_y_offset, mask_x_origin,
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

    @staticmethod
    def _shrink_zone_away_from_siblings(
        zone: Tuple[int, int, int, int],
        sibling_boxes: Optional[List[Tuple[int, int, int, int]]],
    ) -> Tuple[int, int, int, int]:
        """
        Rétrécit la zone utile pour ne pas empiéter sur la bbox d'une bulle
        VOISINE.

        `_get_inner_zone` reste toujours dans SA PROPRE bbox — mais deux
        bulles réellement voisines (chaîne "*PANT*... / *PANT*...", queues qui
        se touchent) ont des bbox qui se chevauchent un peu elles-mêmes. Sans
        ce filtre, le texte de la première pouvait tomber dans la zone que la
        seconde recouvre ensuite de son propre remplissage blanc en la
        dessinant après — un texte lisible dans la planche source se
        retrouvait tronché à l'écran. Repousse la zone hors de CHAQUE
        chevauchement, sur l'axe de moindre pénétration (comme une résolution
        de collision AABB classique), plutôt que de l'abandonner en bloc.
        """
        zx1, zy1, zx2, zy2 = zone
        for sx1, sy1, sx2, sy2 in (sibling_boxes or []):
            ox1, oy1 = max(zx1, sx1), max(zy1, sy1)
            ox2, oy2 = min(zx2, sx2), min(zy2, sy2)
            if ox2 <= ox1 or oy2 <= oy1:
                continue
            zcx, zcy = (zx1 + zx2) / 2.0, (zy1 + zy2) / 2.0
            scx, scy = (sx1 + sx2) / 2.0, (sy1 + sy2) / 2.0
            # On garde celui des deux dégagements qui préserve le plus d'AIRE,
            # au lieu de trancher sur l'axe de moindre pénétration en pixels.
            # Mesuré sur la paire de bulles de cri de path-of-vengeance ch1 :
            # dégager en X ne gardait que 29 % de la zone, contre 50 % en
            # dégageant en Y — et c'est bien X que l'ancien critère choisissait.
            # « Le meilleur des deux » ne peut jamais faire moins bien que
            # l'ancien comportement, qui est toujours l'un des deux candidats.
            cand_x = (zx1, zy1, min(zx2, ox1), zy2) if zcx <= scx else (max(zx1, ox2), zy1, zx2, zy2)
            cand_y = (zx1, zy1, zx2, min(zy2, oy1)) if zcy <= scy else (zx1, max(zy1, oy2), zx2, zy2)

            def _area(z):
                return max(0, z[2] - z[0]) * max(0, z[3] - z[1])

            zx1, zy1, zx2, zy2 = cand_x if _area(cand_x) >= _area(cand_y) else cand_y
            if zx2 <= zx1 or zy2 <= zy1:
                # Chevauchement trop sévère pour être résolu proprement (une
                # bulle presque entièrement contenue dans l'autre) : on
                # revient à la zone d'origine plutôt que produire une zone
                # vide qui ferait échouer le rendu.
                return zone

        # Plancher d'aire.
        #
        # Ce retrait protège d'un cas réel — un texte qui tomberait là où une
        # bulle voisine viendra ensuite peindre son propre fond. Mais il
        # raisonne sur des BBOX, et deux bulles de cri dentelées ont des bbox
        # qui se recouvrent largement alors que leurs contours se touchent à
        # peine. Mesuré sur « TOO SLOW, KAZUKI! » (bulle de 297 px de haut) :
        # la hauteur utile tombait à 113 px, soit un texte rendu à la moitié du
        # corps d'origine. Déborder un peu sur la bbox d'une voisine coûte
        # moins cher que ça : au-delà de 40 % d'aire perdue, on renonce au
        # retrait.
        zx1, zy1, zx2, zy2 = int(zx1), int(zy1), int(zx2), int(zy2)
        kept = max(0, zx2 - zx1) * max(0, zy2 - zy1)
        original = max(1, (zone[2] - zone[0]) * (zone[3] - zone[1]))
        if kept < 0.60 * original:
            return zone
        return (zx1, zy1, zx2, zy2)

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
        skip_inpainting: bool = False,
    ) -> np.ndarray:
        """Efface puis réécrit. `text_regions` = polygones OCR (le texte),
        `mask_regions` = segmentation de la bulle."""
        erase_regions = text_regions or mask_regions

        if text_color_rgb is None and self.cfg.preserve_original_text_color:
            text_color_rgb = self.extract_original_text_color(img, x1, y1, x2, y2, erase_regions)

        if not skip_inpainting:
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
        exclude_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
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

        `exclude_boxes` : bbox des AUTRES détections de la même planche. Sur
        des cartouches de narration empilés de près, le padding de recherche
        peut fusionner deux cadres voisins en une seule composante malgré le
        renforcement de l'érosion (le "pont" entre eux peut être plus large
        que ce que l'érosion tranche). Signal direct et fiable : un contenant
        qui engloutit la bbox d'UNE AUTRE détection a mangé le cadre du
        voisin — on le rejette plutôt que d'y dessiner deux textes empilés.
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

            # 2 passes (~14px de rayon) : sur des cartouches empilés de près
            # (narration multi-boîtes), une seule passe (7px) ne suffisait pas
            # à trancher le pont fin entre deux cadres voisins — le padding
            # (0.6*bw / 0.9*bh) de l'un débordait alors dans le cadre suivant,
            # les deux composantes fusionnaient en une seule, et le texte des
            # deux détections finissait empilé dans le second cadre pendant
            # que le premier restait vide.
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            eroded = cv2.erode(similar, kernel, iterations=2)
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(eroded, 8)
            if n_labels < 2:
                return None
            biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = cv2.dilate((labels == biggest).astype(np.uint8) * 255, kernel, iterations=2)

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

        # Le composant touche le bord de la fenêtre de recherche PADDÉE (pas le
        # bord de l'image) : la vraie forme continue au-delà de ce qu'on a
        # regardé, donc ce n'est pas un contour fermé qu'on a trouvé — c'est une
        # fuite. Mesuré sur une bulle "snowman" posée juste au-dessus d'un
        # gouttière blanche de page : le fond blanc de la bulle et celui de la
        # gouttière ont la même teinte, la composante connexe fusionne les deux
        # et déborde largement sous la bulle réelle ; le texte, centré dans ce
        # contenant trop grand, tombe hors du contour visible. Un vrai
        # contenant (bulle, cartouche) a un pourtour fermé qui laisse une marge
        # avant le bord de la fenêtre — seule une fuite l'atteint pile.
        if bx1 <= cx1 + 1 and cx1 > 0:
            return None
        if bx2 >= cx2 - 1 and cx2 < w_img:
            return None
        if by1 <= cy1 + 1 and cy1 > 0:
            return None
        if by2 >= cy2 - 1 and cy2 < h_img:
            return None

        # Le contenant doit recouvrir la majeure partie de la détection, sinon
        # ce n'en est pas un. Recouvrement large plutôt qu'englobement strict :
        # la bbox YOLO elle-même peut déborder du cadre réel (mesuré sur une
        # bbox "System" qui remontait de 140px au-dessus du cadre visible à
        # cause d'un filigrane de scan chevauchant juste au-dessus) — exiger
        # un englobement parfait rejetait alors le VRAI cadre, pourtant trouvé
        # correctement, et le rendu retombait sur la bbox imprécise (texte
        # qui déborde du cadre en haut et en bas).
        ix1, iy1 = max(bx1, x1), max(by1, y1)
        ix2, iy2 = min(bx2, x2), min(by2, y2)
        inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter_area < 0.5 * bw * bh:
            return None
        # ...et rester crédible : au-delà, on a probablement attrapé le fond
        # de la case entière.
        if (bx2 - bx1) > bw * 4 or (by2 - by1) > bh * 5:
            return None
        # Aucun des deux ratios largeur/hauteur ne dépasse son plafond pris
        # isolément, mais leur PRODUIT peut quand même trahir un fond uni
        # attrapé en entier (mesuré sur une bulle SFX 489×366 dont le
        # "contenant" trouvé faisait 799×829 : 1.63x en largeur, 2.27x en
        # hauteur, chacun sous son plafond, mais 3.7x en aire — un vrai cadre
        # ne grossit pas la bbox source à ce point sur les deux axes à la fois).
        if (bx2 - bx1) * (by2 - by1) > 4.0 * bw * bh:
            return None
        # Un contenant collé aux DEUX bords gauche/droit de la planche alors
        # que la détection, elle, a une marge confortable des deux côtés : ce
        # n'est pas un cadre recadré par le bord de l'image, c'est un fond
        # uni (ciel, aplat de couleur) qui s'étend sur toute la largeur et
        # que la recherche a pris pour un cadre. Mesuré sur une case dont le
        # "contenant" trouvé faisait toute la largeur de la planche et
        # laissait un vide béant sous un texte pourtant correctement placé.
        edge_margin = 0.08 * w_img
        if bx1 <= 2 and bx2 >= w_img - 2 and x1 > edge_margin and (w_img - x2) > edge_margin:
            return None
        # Anciennement rejeté ici si le contenant n'était pas plus grand que la
        # détection dans au moins une dimension (sinon "pas d'intérêt à
        # l'utiliser"). Mais une bbox YOLO peut être PLUS GRANDE que le cadre
        # réel (cf. commentaire plus haut) : le contenant trouvé est alors
        # plus petit qu'elle sur les deux axes tout en étant le bon, et ce
        # garde-fou le rejetait à tort. Le recouvrement (check ci-dessus) et
        # le plafond de taille (check ci-dessous) suffisent à écarter un faux
        # positif.

        for ex1, ey1, ex2, ey2 in (exclude_boxes or []):
            # Chevauchement significatif avec la bbox d'une AUTRE détection :
            # pas seulement "contenue", un simple recouvrement partiel suffit
            # déjà à indiquer que le contenant déborde sur le cadre du voisin
            # (les deux cadres s'étant révélés adjacents à quelques pixels
            # près, pas franchement disjoints).
            ox1, oy1 = max(bx1, ex1), max(by1, ey1)
            ox2, oy2 = min(bx2, ex2), min(by2, ey2)
            if ox2 > ox1 and oy2 > oy1:
                overlap_area = (ox2 - ox1) * (oy2 - oy1)
                sibling_area = max(1, (ex2 - ex1) * (ey2 - ey1))
                if overlap_area > 0.35 * sibling_area:
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

        def _inset(mask: np.ndarray) -> np.ndarray:
            """Retire une marge intérieure au masque avant de l'utiliser pour
            le wrap.

            Le masque décrit le ballon TRAIT COMPRIS, bavures de segmentation
            incluses : mesuré sur « MAKE SURE YOU COME BACK SAFE, OKAY? », il
            couvrait 85 % de la bbox là où l'ovale réel en fait ~78 %. Les
            largeurs de ligne calculées dessus amenaient donc le texte au
            contact du contour, voire au-delà. Une marge proportionnelle à la
            taille de la bulle rétablit la respiration qu'un lettreur laisse
            toujours.
            """
            def _typical_span(m: np.ndarray) -> float:
                """Largeur solide médiane sur la moitié centrale — c'est elle
                qui décide de la place réellement offerte au texte."""
                h = m.shape[0]
                lo, hi = int(h * 0.25), max(int(h * 0.25) + 1, int(h * 0.75))
                band = m[lo:hi]
                if band.size == 0:
                    return 0.0
                per_row = np.count_nonzero(band > 0, axis=1)
                per_row = per_row[per_row > 0]
                return float(np.median(per_row)) if per_row.size else 0.0

            # Bulle de CRI : on met en page sur l'enveloppe convexe.
            #
            # Les pointes d'une bulle dentelée sont décoratives ; le lettreur
            # d'origine y fait déborder le texte, il ne le confine pas au cœur
            # plein. Mesuré sur les 30 bulles de path-of-vengeance ch1, la
            # SOLIDITÉ (aire / aire de l'enveloppe convexe) sépare les deux
            # familles sans ambiguïté : 0,852 à 0,863 pour les cinq bulles de
            # cri, 0,943 et plus pour toutes les autres — un écart de 0,08 sans
            # rien entre les deux. Sur ces formes, la largeur solide médiane ne
            # fait que 72 % de la bbox contre 89 % pour un ovale, d'où un texte
            # rendu à la moitié du corps d'origine.
            try:
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                )
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    area_c = cv2.contourArea(c)
                    hull = cv2.convexHull(c)
                    area_h = cv2.contourArea(hull)
                    if area_h > 0 and (area_c / area_h) < 0.90:
                        filled = np.zeros_like(mask)
                        cv2.drawContours(filled, [hull], -1, 255, -1)
                        mask = filled
            except Exception:
                pass

            base_span = _typical_span(mask)
            base_area = int(np.count_nonzero(mask))

            pad = max(3, int(round(min(box_w, box_h) * 0.06)))
            eroded = mask
            while pad >= 2:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
                candidate = cv2.erode(mask, kernel, iterations=1)
                # Une bulle très fine peut disparaître entièrement.
                if int(np.count_nonzero(candidate)) < 0.25 * base_area:
                    pad = pad // 2
                    continue
                # Effondrement de la largeur utile.
                #
                # Sur un ovale, éroder de `pad` retire ~2·pad à la largeur des
                # bandes : mesuré, 10 à 15 % de perte. Sur une bulle de CRI, le
                # contour en dents de scie fait que l'érosion referme les
                # pointes et étrangle la taille centrale — mesuré sur
                # path-of-vengeance ch1, la bande utile passait de 71-79 px à
                # 14 px, soit plus de 80 % de perte, ce qui forçait ensuite le
                # texte à un mot par ligne et à la moitié du corps d'origine.
                # La perte d'AIRE ne sépare pas ces deux cas (17 à 46 % sur
                # toutes les bulles de la planche, dentelées ou non) ; la perte
                # de LARGEUR, si.
                span = _typical_span(candidate)
                if base_span <= 0 or span >= 0.70 * base_span:
                    eroded = candidate
                    break
                pad = pad // 2
            else:
                return mask

            if int(np.count_nonzero(eroded)) < 0.25 * base_area:
                return mask
            return eroded

        if isinstance(raw_mask, np.ndarray) and raw_mask.size > 0:
            m = raw_mask[:, :, 0] if raw_mask.ndim == 3 else raw_mask
            if m.shape[:2] != (box_h, box_w):
                try:
                    m = cv2.resize(m, (box_w, box_h), interpolation=cv2.INTER_NEAREST)
                except Exception:
                    m = None
            # Un ballon remplit largement sa bbox ; un masque de lettres non.
            if m is not None and float(np.count_nonzero(m)) / float(box_w * box_h) >= 0.55:
                return _inset((m > 0).astype(np.uint8) * 255)

        # Tentée quelle que soit l'étiquette : c'est une MESURE, elle n'a pas
        # besoin que YOLO ait vu juste. L'appelant décidera ensuite, d'après la
        # forme obtenue, s'il y a lieu d'épouser le contour.
        derived = self._bubble_mask_from_image(crop_bgr)
        if derived is not None:
            return _inset(derived)

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

    # ─────────────────────────────────────────────────────────────────────────
    # INSERT TEXT (REWRITTEN FOR EXACT LINE-BY-LINE & BUBBLE SURFACE)
    # ─────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────
    # ROUTAGE DU RENDU PAR COÛT CONTINU
    # ─────────────────────────────────────────────────────────────────────
    #
    # Trois routes possibles, de la plus fidèle à la planche à la plus libre :
    #   exact_lines — le texte est redessiné ligne à ligne dans les polygones
    #                 OCR d'origine ; les coupures de la planche sont conservées
    #   anchor      — remise en page, mais bornée à l'enveloppe des polygones
    #   box         — remise en page dans la boîte (ou la forme du ballon)
    #
    # Ces routes étaient choisies par deux portes binaires testant
    # `probe.size < 0.6 * cap`, où `cap` dérive de `source_line_height`. Le
    # seuil était donc PROPORTIONNEL à la grandeur qu'il testait : baisser le
    # plafond abaissait le seuil d'autant, la porte basculait, et le rendu
    # changeait de route. Mesuré : `i-married #22` plafond 56 → 45 (baisse) mais
    # police 28 → 59 (double) ; `hellogin #4` plafond 48 → 101 (monte) mais
    # police 75 → 42 (baisse). La taille finale n'était pas monotone en son
    # propre plafond — un effet de falaise, pas un réglage.
    #
    # Chaque route est maintenant CHIFFRÉE dans une unité commune et on prend la
    # moins coûteuse. Une route qui se dégrade perd désormais progressivement.

    # Prix de l'infidélité à la planche, en unités logarithmiques de corps.
    # 0,10 se lit : « on accepte de perdre 10 % de corps de texte pour garder
    # les coupures de lignes de la planche d'origine ».
    ROUTE_INFIDELITE = {
        "exact_lines": 0.00,
        "anchor": 0.10,
        "box": 0.20,
    }
    # Un texte qui ne tient pas est rédhibitoire : 2,0 dépasse tout écart de
    # corps plausible (ln 2 = 0,69 pour un facteur deux).
    COUT_NE_TIENT_PAS = 2.0

    # Qualité du découpage en lignes, dans la même unité logarithmique.
    #
    # C'est ce terme qui JUSTIFIE la préférence pour les routes fidèles au lieu
    # de la postuler : `exact_lines` reprend les coupures de la planche, donc son
    # pavé est celui d'un lettreur ; `box` re-découpe gloutonnement et fabrique
    # les orphelins (mesuré : 25 % des blocs multi-lignes du corpus finissent sur
    # un mot seul).
    PENALITE_ORPHELIN = 0.15      # « éviter un orphelin vaut 15 % de corps »
    PENALITE_DESEQUILIBRE = 0.25  # multiplie le coefficient de variation

    @staticmethod
    def _rag_penalty(lines: Optional[List[str]]) -> float:
        """Coût du découpage en lignes d'un pavé.

        Deux défauts que l'œil voit immédiatement et qu'aucune mesure de taille
        n'attrape : le mot ORPHELIN seul sur la dernière ligne, et le
        DÉSÉQUILIBRE des largeurs, qui donne l'escalier au lieu du bloc.

        Longueurs en caractères plutôt qu'en pixels : à police constante dans un
        même pavé, c'est proportionnel, et ça évite de mesurer chaque ligne.
        """
        lignes = [l for l in (lines or []) if l and l.strip()]
        if len(lignes) < 2:
            return 0.0
        cout = 0.0
        derniere = lignes[-1].split()
        if len(derniere) == 1 and len(lignes[-1]) <= 6:
            cout += TextRenderer.PENALITE_ORPHELIN
        longueurs = [len(l) for l in lignes]
        moyenne = sum(longueurs) / len(longueurs)
        if moyenne > 0:
            ecart = (sum((x - moyenne) ** 2 for x in longueurs) / len(longueurs)) ** 0.5
            cout += TextRenderer.PENALITE_DESEQUILIBRE * (ecart / moyenne)
        return cout

    def _route_cost(self, probe: Optional[Dict], em_source: Optional[float],
                    route: str) -> float:
        """Coût d'une route de rendu. Plus bas = mieux, `inf` = impossible.

        `|ln(taille / taille_source)|` : symétrique, donc rendre au double ou à
        la moitié du corps de la planche coûte exactement pareil — ce que ne
        faisait aucun seuil, tous unilatéraux. L'infidélité de route s'ajoute
        dans la MÊME unité, ce qui rend l'arbitrage explicite et réglable.
        """
        if probe is None or not em_source or em_source <= 0:
            return float("inf")
        size = float(probe.get("size") or 0)
        if size <= 0:
            return float("inf")
        cout = abs(math.log(size / float(em_source)))
        cout += self.ROUTE_INFIDELITE.get(route, 0.0)
        # `exact_lines` reprend les coupures de la PLANCHE : son pavé est celui
        # du lettreur d'origine, pas celui du sonde, donc on ne lui impute pas
        # le découpage glouton de la sonde. Les routes qui remettent en page,
        # elles, sont jugées sur le découpage qu'elles produiront vraiment.
        if route != "exact_lines":
            cout += self._rag_penalty(probe.get("lines"))
        if not probe.get("fits", True):
            cout += self.COUT_NE_TIENT_PAS
        return cout

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
        sibling_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
        # Largeur de contour MESURÉE sur la planche d'origine, en pixels.
        # Doit être calculée en amont : ici l'image est déjà effacée, le texte
        # source n'existe plus.
        outline_width_px: Optional[int] = None,
        # Texte SOURCE, pour calibrer le budget de lignes sur le foisonnement
        # reel de la traduction. Passe explicitement plutot que devine depuis la
        # geometrie des polygones : c'est la seule donnee exacte.
        source_text: Optional[str] = None,
        # Verdict MESURE « y a-t-il un contour ferme autour de ce texte ? ».
        # Se calcule en amont (cf. `core.bubble_shape.has_closed_bubble`) : il
        # faut a la fois l'image d'origine, pour l'encre, et l'effacee, pour le
        # trait du ballon — or ici seule l'effacee existe.
        # None = indecidable, on garde le comportement historique.
        bubble_present: Optional[bool] = None,
    ) -> np.ndarray:
        if not text or not str(text).strip():
            return img

        labelled_bubble = str(class_name).lower().strip() == "bulle"
        ox1, oy1, ox2, oy2 = x1, y1, x2, y2

        # ── Couleurs ──
        text_color, outline_color, outline_width_auto = self.get_text_colors(
            img, ox1, oy1, ox2, oy2, class_name=class_name, text_color_override=text_color_rgb,
        )
        if stroke_color_rgb is not None: outline_color = stroke_color_rgb
        if stroke_width is not None: outline_width_auto = stroke_width

        # Le contour sort du résolveur de couleurs à 2 px quelle que soit la
        # taille du texte. C'est juste pour du dialogue (~20 px de corps), mais
        # ridicule sur un cartouche d'impact dont les lettres font 60 px et
        # portent un contour de 4 à 5 px : le texte réinjecté paraissait fin et
        # posé par-dessus l'image au lieu d'y appartenir. On l'indexe donc sur
        # la taille du texte d'origine.
        # Un cartouche posé sur du décor a TOUJOURS un contour sur la planche
        # d'origine : c'est lui qui le détache du dessin. Le résolveur de
        # couleurs le supprime dès que le contraste local est bon, mais « bon
        # au centre » ne veut rien dire sur un fond d'éclairs ou de flammes où
        # la luminosité change d'un bout à l'autre du cartouche.
        if str(class_name).lower() == "out_text" and outline_color is None:
            luma = 0.299 * text_color[0] + 0.587 * text_color[1] + 0.114 * text_color[2]
            outline_color = (0, 0, 0) if luma > 128 else (255, 255, 255)
            outline_width_auto = max(2, outline_width_auto)

        # Contour MESURÉ prioritaire sur le forfait, pour les cartouches.
        if (
            outline_color is not None
            and stroke_width is None
            and outline_width_px
            and str(class_name).lower() in ("out_text", "system")
        ):
            outline_width_auto = max(1, min(int(outline_width_px), 8))
        elif outline_color is not None and stroke_width is None:
            if source_line_height and source_line_height > 4:
                outline_width_auto = max(
                    outline_width_auto, int(round(float(source_line_height) * 0.075)),
                )

        if bg_color_rgb is not None:
            cv2.rectangle(img, (ox1, oy1), (ox2, oy2), bg_color_rgb[::-1], -1)

        # ── Style ──
        bw, bh = x2 - x1, y2 - y1
        if text_style == "dialogue": text_style = self.infer_text_style(text, bw, bh, class_name=class_name)
        if text_style == "system_card": text = self._format_system_card_text(text)
        if text_style not in ("whisper", "system_card"): text = text.upper()
        
        # Les cartouches out_text sont du texte d'IMPACT : sur la planche
        # d'origine ils sont gras, contourés, et posés sur du décor chargé.
        # Rendus en graisse normale ils paraissent à la fois plus petits et
        # moins lisibles que l'original — c'est ce que « certains out_text sont
        # difficiles à lire » désigne.
        if str(class_name).lower() == "out_text":
            font_hint = "bold"
        resolved_font_path = self._resolve_font_path(font_key, text_style, font_hint)

        # Cartouches et cartes System : police CONDENSÉE ET GRASSE.
        #
        # Comparée sur cartouches réels contre quatre candidates, `Allegre Sans`
        # est la seule à rendre le poids de la planche. Chiffres mesurés
        # (largeur par caractère rapportée à la hauteur de casse / part d'encre
        # dans la boîte du H) : Allegre 0,51 / 0,72 — CCDynamicDuo 0,53 / 0,77 —
        # Graphique 0,48 / 0,58 — TwoFisted 0,42 / 0,44 — CCWildWords (avant)
        # 0,80. Le cartouche « THE REALM OF A DEMIGOD, THE 9TH CIRCLE » tient
        # désormais en 2 lignes au lieu de 3, avec des lettres plus hautes.
        #
        # La GRAISSE compte autant que la largeur : les polices les plus proches
        # du ratio de la source (TwoFisted à 0,42) rendent mal parce qu'elles
        # sont maigres. Les bulles de dialogue gardent volontairement une police
        # plus large et plus ronde, plus lisible en petit corps.
        if str(class_name).lower() in ("out_text", "system"):
            caption_font = self._find_font_by_fragment("Al__gre_Sans")
            if caption_font:
                resolved_font_path = caption_font

        # ── Forme ──
        # `_container_box` cherche la BOÎTE qui contient le texte (cartouche,
        # panneau). Sur un cartouche out_text posé à même l'illustration il n'y
        # a pas de boîte : la recherche accroche alors une zone de contraste du
        # décor. Mesuré sur « THE COORDINATES OF TOKYO'S 23 WARDS… » (bbox
        # 423x284) : container retourné 276x424, plus étroit ET plus haut que
        # le cartouche, débordant vers le haut — le texte se retrouvait
        # découpé en neuf lignes étroites qui sortaient du panneau par le haut.
        # Pour cette classe, la bbox de la détection EST la zone de mise en
        # page.
        if str(class_name).lower() == "out_text":
            container = None
        else:
            container = self._container_box(img, x1, y1, x2, y2, exclude_boxes=sibling_boxes)
            # Le container doit être CENTRÉ sur le texte source, pas seulement le
            # contenir. `_container_box` attrape parfois un aplat de fond bien
            # plus grand que le texte (fond de case uni) : y centrer le texte le
            # déplace loin de sa position d'origine. Mesuré sur hellogin p02 #41 :
            # container (fond blanc de case) centré 176 px SOUS le texte, qui
            # sortait de sa bulle — le pire pour le confort de lecture. On le
            # rejette dès que son centre s'écarte de plus d'une hauteur de texte.
            if container is not None and text_regions:
                _pys = [q[1] for reg in text_regions for q in (reg.get('bbox') or [])]
                if _pys:
                    _src_cy = oy1 + (min(_pys) + max(_pys)) / 2.0
                    _src_h = max(8.0, float(max(_pys) - min(_pys)))
                    _cont_cy = (container[1] + container[3]) / 2.0
                    if abs(_cont_cy - _src_cy) > _src_h:
                        container = None
        if container is not None:
            x1, y1, x2, y2 = container
            mask_for_wrap = None
            has_mask_wrap = False
        elif bubble_present is False and not labelled_bubble:
            # Le veto ne contredit JAMAIS une detection etiquetee `bulle`.
            #
            # Le seuil de remplissage qui separe ballon et texte libre a ete
            # calibre sur une seule planche, ou les ballons plafonnaient a
            # 79,5 %. Sur le chapitre entier il ne generalise pas : 9 vraies
            # bulles remplissent 85 a 97 % de leur bbox — couronne herissee, ou
            # boite serree sur le ballon — et se faisaient donc prendre pour du
            # texte libre, ce qui les routait vers `_draw_exact_lines`, qui
            # ignore la forme du ballon.
            #
            # Le defaut MESURE que ce veto corrige est ailleurs : la teinte qui
            # invente une geometrie sur des cartouches `out_text`. C'est donc a
            # ce cas-la qu'on le reserve, et on fait confiance a YOLO la ou il
            # annonce une bulle.
            #
            # Aucun contour ferme autour de ce texte : il n'y a pas de ballon a
            # epouser. Sans ce veto, `_bubble_mask_from_image` rend quand meme
            # une « forme » sur du texte libre — mesure sur path-of-vengeance :
            # trois cartouches out_text sur six, dont un masque fantome de
            # 27,3 % de la boite. `is_bubble` passait alors a vrai et detournait
            # le texte de `_draw_exact_lines`, seul regime correct pour du texte
            # libre, vers un wrap dans une forme qui n'existe pas.
            mask_for_wrap = None
            has_mask_wrap = False
        else:
            mask_for_wrap = self._bubble_shape_mask(
                bubble_mask,
                img[max(0, y1):y2, max(0, x1):x2],
                max(1, x2 - x1), max(1, y2 - y1),
                is_bubble=labelled_bubble,
            )
            has_mask_wrap = mask_for_wrap is not None and self._is_non_rectangular(mask_for_wrap)

            # Validation du masque de forme contre le texte SOURCE.
            #
            # `_bubble_shape_mask` croît une région depuis le masque des lettres ;
            # sur une bulle ouverte ou fondue avec un aplat clair voisin (fond de
            # case), la région déborde dans cet aplat et le « ballon » obtenu est
            # centré loin du texte. Mesuré sur hellogin p02 #41 : forme centrée
            # 187 px SOUS le texte source, qui finissait rendu hors de sa bulle,
            # dans le vide — le pire défaut pour le confort de lecture (l'œil
            # cherche le texte).
            #
            # Le masque des LETTRES (`bubble_mask`) est la vérité : le lettreur a
            # posé son texte au centre du ballon. Si la forme dérive de plus de
            # 20 % de la hauteur par rapport à lui, on la remplace par une ellipse
            # inscrite dans la bbox, centrée sur le texte — le texte reste alors
            # dans sa bulle.
            if has_mask_wrap and isinstance(bubble_mask, np.ndarray) and bubble_mask.size > 0:
                _ysf, _xsf = np.nonzero(mask_for_wrap > 0)
                _ysl, _xsl = np.nonzero(bubble_mask > 0)
                if _ysf.size > 0 and _ysl.size > 0:
                    _fy = _ysf.mean() / float(mask_for_wrap.shape[0])
                    _ly = _ysl.mean() / float(bubble_mask.shape[0])
                    if abs(_fy - _ly) > 0.20:
                        _bh, _bw = max(2, y2 - y1), max(2, x2 - x1)
                        _ell = np.zeros((_bh, _bw), dtype=np.uint8)
                        cv2.ellipse(_ell, (_bw // 2, _bh // 2),
                                    (max(1, _bw // 2 - 2), max(1, _bh // 2 - 2)),
                                    0, 0, 360, 255, -1)
                        mask_for_wrap = _ell
                        has_mask_wrap = True
        is_bubble = has_mask_wrap

        angle = 0.0
        if angle_override is not None:
            angle = angle_override
        elif self.cfg.follow_source_text_angle:
            angle = self._source_text_angle(text_regions)
            if abs(angle) < float(self.cfg.min_text_angle_deg): angle = 0.0

        import os
        if os.environ.get('RENDER_DEBUG'):
            print(f"[RENDER_DEBUG] insert_text text={text[:30]!r} class={class_name} is_bubble={is_bubble} container={container} angle={angle} n_regions={len(text_regions or [])}")

        # ── Référence de comparaison : la route « boîte » ──
        # Toujours disponible, donc c'est elle qui sert d'alternative réelle aux
        # deux autres. La chiffrer AVANT les portes est ce qui permet de juger
        # chaque route contre une VRAIE option, et non contre un seuil dérivé
        # d'elle-même. Calculée seulement quand une porte peut se poser, c'est-
        # à-dire quand il y a des polygones OCR et un corps source mesuré.
        self._last_route_costs = {}
        em_source = (float(source_line_height) / 0.75
                     if source_line_height and source_line_height > 4 else None)
        cout_box = None
        if em_source and text_regions:
            _bx1, _by1, _bx2, _by2 = self._get_inner_zone(
                x1, y1, x2, y2, img.shape, bubble_mask=mask_for_wrap,
                shrink=self._shrink_ratio_for(is_bubble, has_mask_wrap),
            )
            if sibling_boxes:
                _bx1, _by1, _bx2, _by2 = self._shrink_zone_away_from_siblings(
                    (_bx1, _by1, _bx2, _by2), sibling_boxes)
            _bw = max(10, (_bx2 - _bx1) - 2 * self.cfg.padding_horizontal)
            _bh = max(10, (_by2 - _by1) - 2 * self.cfg.padding_vertical)
            _cap = int(em_source * 1.05)
            _probe_box = self._fit_font_hard(
                text, _cap, _bw, _bh,
                bubble_mask=mask_for_wrap if angle == 0.0 else None,
                shape_wrap=has_mask_wrap and angle == 0.0,
                font_path=resolved_font_path,
                mask_y_offset=(_by1 + self.cfg.padding_vertical) - y1,
                max_font_size=_cap,
                mask_x_origin=x1,
            )
            cout_box = self._route_cost(_probe_box, em_source, "box")
            self._last_route_costs["box"] = cout_box

        # === LINE-BY-LINE REPLACEMENT POUR TEXTE HORS-BULLE ===
        #
        # Ce chemin repose sur une hypothèse : les polygones OCR décrivent
        # fidèlement les lignes de la planche. Quand l'OCR en fusionne plusieurs
        # — deux régions pour six lignes réelles, mesuré sur la carte System
        # « FUELED BY REVENGE, I REACHED THE 6TH CIRCLE… » de
        # i-married-the-dragon — tout le texte doit tenir dans la hauteur de ces
        # deux régions : le corps s'effondre et les lignes débordent largement
        # du cadre. On vérifie donc d'abord que le texte tient dans l'enveloppe
        # des régions à une taille proche de celle d'origine ; sinon on remet en
        # page normalement dans la boîte.
        exact_lines_ok = bool(text_regions)
        if exact_lines_ok and source_line_height and source_line_height > 4:
            xs = [p[0] for r in text_regions for p in (r.get('bbox') or [])]
            ys = [p[1] for r in text_regions for p in (r.get('bbox') or [])]
            if xs and ys:
                env_w = max(1, int(max(xs) - min(xs)))
                env_h = max(1, int(max(ys) - min(ys)))
                cap = int((float(source_line_height) / 0.75) * 1.05)
                probe = self._fit_font_hard(
                    text, cap, env_w, env_h,
                    bubble_mask=None, shape_wrap=False,
                    font_path=resolved_font_path, max_font_size=cap,
                )
                cout = self._route_cost(probe, em_source, "exact_lines")
                self._last_route_costs["exact_lines"] = cout
                if cout_box is not None:
                    if cout > cout_box:
                        exact_lines_ok = False
                elif probe is None or probe.get('size', 0) < 0.6 * cap:
                    # Repli sur l'ancienne règle si la route « boîte » n'a pas
                    # pu être chiffrée : mieux vaut un seuil que rien.
                    exact_lines_ok = False

        if not is_bubble and container is None and exact_lines_ok:
            return self._draw_exact_lines(
                img, text, text_regions,
                text_color, outline_color, outline_width_auto,
                resolved_font_path, text_style, source_line_height, angle,
                ox1, oy1,
            )

        # === BUBBLE WRAP AVEC POLYGONE DE SURFACE ===
        use_locked_mode = bool(getattr(self.cfg, 'lock_text_to_ocr_regions', False))
        if bool(getattr(self.cfg, 'lock_text_system_only', True)) and str(class_name).lower() != 'system':
            use_locked_mode = False

        anchor_box = None
        zone_unshrunk: Optional[Tuple[int, int, int, int]] = None
        if use_locked_mode or (not is_bubble and container is None):
            anchor_box = self._compute_anchor_box_from_regions(ox1, oy1, ox2, oy2, text_regions)
            if anchor_box and container:
                cx1, cy1, cx2, cy2 = container
                anchor_box = (max(anchor_box[0], cx1), max(anchor_box[1], cy1),
                              min(anchor_box[2], cx2), min(anchor_box[3], cy2))
                if anchor_box[2] <= anchor_box[0] or anchor_box[3] <= anchor_box[1]: anchor_box = None

            # L'ancrage sur les polygones OCR ne vaut que si ces polygones
            # décrivent VRAIMENT le bloc de texte d'origine. Quand l'OCR fusionne
            # plusieurs lignes en une seule région — ou n'en rend que deux pour
            # six lignes réelles, mesuré sur la carte System « FUELED BY REVENGE,
            # I REACHED THE 6TH CIRCLE… » de i-married-the-dragon — la boîte
            # d'ancrage est bien plus BASSE que le texte, et tout doit tenir dans
            # cette hauteur : le corps s'effondre et le texte déborde du cadre.
            #
            # On vérifie donc que le texte peut tenir dans cette boîte à une
            # taille proche de celle de la planche. Sinon on abandonne l'ancrage
            # et on remet en page dans le cadre, comme pour n'importe quel autre
            # texte.
            if anchor_box is not None and source_line_height and source_line_height > 4:
                a_w = max(1, anchor_box[2] - anchor_box[0])
                a_h = max(1, anchor_box[3] - anchor_box[1])
                cap = int((float(source_line_height) / 0.75) * 1.05)
                probe = self._fit_font_hard(
                    text, cap, a_w, a_h,
                    bubble_mask=None, shape_wrap=False,
                    font_path=resolved_font_path, max_font_size=cap,
                )
                cout = self._route_cost(probe, em_source, "anchor")
                self._last_route_costs["anchor"] = cout
                _rejette = (cout > cout_box) if cout_box is not None else (
                    probe is None or probe.get('size', 0) < 0.6 * cap)
                if _rejette:
                    anchor_box = None
                    use_locked_mode = False

        if anchor_box is not None:
            ix1, iy1, ix2, iy2 = anchor_box
        else:
            use_locked_mode = False
            ix1, iy1, ix2, iy2 = self._get_inner_zone(
                x1, y1, x2, y2, img.shape, bubble_mask=mask_for_wrap,
                shrink=self._shrink_ratio_for(is_bubble, has_mask_wrap),
            )
            if sibling_boxes:
                zone_unshrunk = (ix1, iy1, ix2, iy2)
                ix1, iy1, ix2, iy2 = self._shrink_zone_away_from_siblings((ix1, iy1, ix2, iy2), sibling_boxes)

        tw, th = ix2 - ix1, iy2 - iy1
        if tw <= 0 or th <= 0:
            import os
            if os.environ.get('RENDER_DEBUG'):
                print(f"[RENDER_DEBUG] bail tw/th<=0 bbox=({ox1},{oy1},{ox2},{oy2}) container={container} is_bubble={is_bubble} anchor_box={anchor_box} tw={tw} th={th} text={text[:40]!r}")
            # ── HOOK DE MESURE (additif, ne change aucun comportement) ──
            self.last_layout_debug = {
                "bail": "tw_th_le_0", "text": str(text)[:60],
                "bbox": (ox1, oy1, ox2, oy2), "is_bubble": is_bubble,
                "tw": tw, "th": th,
            }
            return img

        inner_w = max(10, tw - 2 * self.cfg.padding_horizontal)
        inner_h = max(10, th - 2 * self.cfg.padding_vertical)

        if angle != 0.0:
            diag_w = self._rotated_line_width(text_regions, angle)
            if diag_w > 20:
                inner_w = max(20, int(diag_w))
                inner_h = max(10, int(max(inner_h, th)))

        fs = self.calculate_optimal_font_size(text, inner_w, inner_h, source_line_height)
        if text_style == "scream": fs = min(self.cfg.max_font_size, int(fs * 1.12))
        elif text_style == "whisper": fs = max(self.cfg.min_font_size, int(fs * 0.92))
        size_cap = int((float(source_line_height) / 0.75) * 1.05) if source_line_height and source_line_height > 4 else None

        layout = self._fit_font_hard(
            text, fs, inner_w, inner_h,
            bubble_mask=mask_for_wrap if angle == 0.0 else None,
            shape_wrap=has_mask_wrap and angle == 0.0,
            font_path=resolved_font_path,
            mask_y_offset=(iy1 + self.cfg.padding_vertical) - y1,
            max_font_size=size_cap,
            mask_x_origin=x1,
            target_lines=len(text_regions) if text_regions else None,
            length_ratio=(
                len(text) / max(1, len(source_text)) if source_text and text else None
            ),
        )

        # Le retrait « loin des voisines » est une précaution : il évite qu'un
        # texte tombe dans la zone qu'une bulle voisine va repeindre ensuite.
        # Mais quand deux bulles se chevauchent franchement, il ampute la zone
        # au point que RIEN ne tient plus — `_fit_font_hard` rend alors son
        # repli au plancher, dont le bloc est plus haut que la zone : centré, il
        # démarre AU-DESSUS de la bulle et le texte sort par le haut (mesuré sur
        # les paires « IT MIGHT JUST BE A NORMAL RUN » / « MAKE SURE YOU COME
        # BACK » et « YOU'RE MY ONLY BLOOD RELATIVE » / « OF COURSE I'M GOING
        # TO WORRY »). Déborder de sa propre bulle est pire que déborder sur la
        # zone d'une voisine : on reprend alors la zone complète.
        if (
            zone_unshrunk is not None
            and (layout is None or not layout.get('fits'))
            and (ix1, iy1, ix2, iy2) != zone_unshrunk
        ):
            fx1, fy1, fx2, fy2 = zone_unshrunk
            full_w = max(10, (fx2 - fx1) - 2 * self.cfg.padding_horizontal)
            full_h = max(10, (fy2 - fy1) - 2 * self.cfg.padding_vertical)
            retry = self._fit_font_hard(
                text, fs, full_w, full_h,
                bubble_mask=mask_for_wrap if angle == 0.0 else None,
                shape_wrap=has_mask_wrap and angle == 0.0,
                font_path=resolved_font_path,
                mask_y_offset=(fy1 + self.cfg.padding_vertical) - y1,
                max_font_size=size_cap,
                mask_x_origin=x1,
                target_lines=len(text_regions) if text_regions else None,
                length_ratio=(
                    len(text) / max(1, len(source_text)) if source_text and text else None
                ),
            )
            if retry is not None and retry.get('fits'):
                ix1, iy1, ix2, iy2 = zone_unshrunk
                inner_w, inner_h = full_w, full_h
                layout = retry

        if layout is None:
            import os
            if os.environ.get('RENDER_DEBUG'):
                print(f"[RENDER_DEBUG] bail layout=None bbox=({ox1},{oy1},{ox2},{oy2}) container={container} is_bubble={is_bubble} fs={fs} inner_w={inner_w} inner_h={inner_h} size_cap={size_cap} text={text[:40]!r}")
            # ── HOOK DE MESURE (additif, ne change aucun comportement) ──
            self.last_layout_debug = {
                "bail": "layout_none", "text": str(text)[:60],
                "bbox": (ox1, oy1, ox2, oy2), "is_bubble": is_bubble,
                "fs_estimate": fs, "inner_w": inner_w, "inner_h": inner_h,
                "size_cap": size_cap, "source_line_height": source_line_height,
            }
            return img

        # ── HOOK DE MESURE (additif, ne change aucun comportement) ──
        # Purement lecture de `layout` + recalcul des formules de `_draw_block`
        # (sans les appliquer) pour comparer le centre du bloc rendu au centre
        # de la zone utilisable, et estimer le taux de remplissage.
        try:
            top_aligned_dbg = bool(use_locked_mode or text_style == "system_card")
            lines_dbg = layout['lines']
            font_dbg = layout['font']
            line_h_dbg, spacing_dbg = layout['line_h'], layout['spacing']
            total_h_dbg = layout['total_h']
            left_dbg = ix1 + self.cfg.padding_horizontal
            top_dbg = iy1 + self.cfg.padding_vertical
            if top_aligned_dbg:
                ys_dbg = top_dbg
            elif self.cfg.vertical_align == 'top':
                ys_dbg = top_dbg
            elif self.cfg.vertical_align == 'bottom':
                ys_dbg = iy2 - self.cfg.padding_vertical - total_h_dbg
            else:
                ys_dbg = top_dbg + (inner_h - total_h_dbg) // 2
                # `_draw_block` applique ENSUITE ce recentrage optique (cf. la
                # note sur le « texte trop haut »). Sans le rejouer ici, le
                # `block_top` exposé serait la position AVANT correction —
                # décalage mesuré de 2-3 px, qui se serait retrouvé sur chaque
                # texte de l'éditeur manuel.
                ys_dbg = self._optical_center_y(
                    font_dbg, lines_dbg, line_h_dbg, spacing_dbg, top_dbg, inner_h, ys_dbg
                )

            line_centers_dbg = layout.get('line_centers')
            ink_area = 0.0
            # Géométrie d'encre PAR LIGNE. Exposée pour que l'éditeur Konva
            # aligne le texte sur l'encre réelle plutôt que sur une convention
            # d'ancrage : PIL ancre en haut de l'ascendante, Canvas sur la ligne
            # de base, et les deux ne coïncident pas (2 px d'écart mesurés).
            # En donnant `ink_top`/`lsb`, l'alignement devient exact et
            # indépendant de ces conventions. Additif : ne change rien au rendu.
            line_ink_dbg = []
            for li, ln in enumerate(lines_dbg):
                if not ln:
                    line_ink_dbg.append(None)
                    continue
                lsb_dbg, ink_w_dbg = self._line_extents(font_dbg, ln)
                bb = font_dbg.getbbox(ln)
                ink_h_dbg = max(0, bb[3] - bb[1])
                ink_area += float(ink_w_dbg) * float(ink_h_dbg)
                line_ink_dbg.append({
                    "lsb": int(lsb_dbg),
                    "ink_top": int(bb[1]),
                    "ink_w": int(ink_w_dbg),
                    "ink_h": int(ink_h_dbg),
                })

            if line_centers_dbg:
                block_cx = sum(line_centers_dbg) / len(line_centers_dbg)
            elif top_aligned_dbg or self.cfg.horizontal_align == 'left':
                block_cx = left_dbg + (layout['max_line_w'] / 2.0)
            elif self.cfg.horizontal_align == 'right':
                block_cx = left_dbg + inner_w - (layout['max_line_w'] / 2.0)
            else:
                block_cx = left_dbg + inner_w / 2.0
            block_cy = ys_dbg + total_h_dbg / 2.0
            zone_cx = (ix1 + ix2) / 2.0
            zone_cy = (iy1 + iy2) / 2.0
            usable_area = max(1, inner_w * inner_h)

            self.last_layout_debug = {
                "bail": None, "text": str(text)[:60],
                "bbox": (ox1, oy1, ox2, oy2),
                "usable_zone": (ix1, iy1, ix2, iy2),
                "inner_w": inner_w, "inner_h": inner_h,
                "is_bubble": is_bubble, "has_mask_wrap": has_mask_wrap,
                "container": container, "anchor_box": anchor_box,
                "use_locked_mode": use_locked_mode, "top_aligned": top_aligned_dbg,
                "mode": "anchor" if anchor_box is not None else "box",
                "route_costs": dict(self._last_route_costs),
                "angle": angle,
                # `block_top` et `line_centers` : coordonnées absolues réelles du
                # bloc dessiné. Exposées pour que l'éditeur Konva puisse replacer
                # le texte EXACTEMENT là où ce moteur l'a posé — sans elles, il
                # faudrait re-deviner la position, et les bulles rondes (dont
                # chaque ligne est recentrée sur la largeur disponible à son y)
                # divergeraient. Additif : aucune incidence sur le rendu.
                "block_top": ys_dbg,
                "line_centers": list(line_centers_dbg) if line_centers_dbg else None,
                "line_ink": line_ink_dbg,
                "ascent": font_dbg.getmetrics()[0],
                "descent": font_dbg.getmetrics()[1],
                # Bord gauche et largeur interne : PIL centre chaque ligne sur
                # `left + (inner_w - ink_w) / 2`. Sans ces deux valeurs, le
                # frontend devrait les redériver de usable_zone + padding.
                "left": left_dbg,
                "inner_w_used": inner_w,
                "source_line_height": source_line_height,
                "fs_estimate": fs, "font_size_final": layout['size'],
                "blocage": layout.get('blocage'),
                "n_lines": len(lines_dbg), "lines": list(lines_dbg),
                "line_h_px": line_h_dbg, "spacing_px": spacing_dbg,
                # Largeurs d'encre REELLES : le barème mesurait l'équilibre en
                # nombre de CARACTÈRES, alors que `_rebalance_lines` optimise
                # des pixels. Les deux divergent dès que la largeur moyenne des
                # glyphes change d'une ligne à l'autre.
                "line_widths_px": [self._line_extents(font_dbg, ln)[1]
                                   for ln in lines_dbg],
                "total_h_px": total_h_dbg, "max_line_w_px": layout['max_line_w'],
                "zone_center": (zone_cx, zone_cy),
                "block_center": (block_cx, block_cy),
                "dx": block_cx - zone_cx, "dy": block_cy - zone_cy,
                "ink_area": ink_area, "usable_area": usable_area,
                "fill_ratio": ink_area / usable_area,
            }
        except Exception as _dbg_e:
            self.last_layout_debug = {"bail": "debug_hook_error", "error": str(_dbg_e)}

        if angle != 0.0:
            return self._draw_rotated(
                img, layout, angle, text_regions, ox1, oy1, ox2, oy2,
                text_color, outline_color, outline_width_auto,
            )

        return self._draw_block(
            img, layout, ix1, iy1, ix2, iy2, inner_w, inner_h,
            text_color, outline_color, outline_width_auto,
            top_aligned=(use_locked_mode or text_style == "system_card"),
        )

    def _draw_exact_lines(
        self, img: np.ndarray, text: str, regions: List[Dict],
        text_color, outline_color, outline_width: int,
        font_path: Optional[str], text_style: str, source_line_height: Optional[float], angle: float,
        region_offset_x: int = 0, region_offset_y: int = 0,
    ) -> np.ndarray:
        """
        Place le texte ligne par ligne exactement dans les polygones OCR d'origine.
        Si le nombre de lignes traduites differe des polygones d'origine, on tente de les fusionner ou de les diviser.

        `regions` contient des polygones en coordonnées LOCALES au crop de la
        détection (comme partout ailleurs dans ce fichier — cf.
        `_compute_anchor_box_from_regions`, `_draw_rotated`), alors que `img`
        est la planche ENTIÈRE : sans `offset_x/offset_y` (= bbox globale de
        la détection), le texte se dessinait près du coin (0,0) de la page —
        donc hors de la bulle réelle, à un endroit qui pouvait sembler
        totalement vide dans le cas d'une carte de narration seule sur son
        segment de page (ni bulle ni container détecté).
        """
        lines = text.split('\n')
        
        # Si le LLM n'a pas renvoyé le texte avec des retours à la ligne,
        # on le découpe PROPORTIONNELLEMENT à la largeur de chaque polygone.
        # L'ancienne répartition en parts égales entassait les derniers mots
        # dans le dernier polygone (overflow mesuré à 38-41%).
        if len(lines) == 1 and len(regions) > 1:
            words = text.split()
            if words:
                # Mesurer la largeur de chaque polygone OCR
                widths = []
                for r in regions:
                    pts = r.get('bbox') if isinstance(r, dict) else None
                    if pts and len(pts) >= 3:
                        xs = [p[0] for p in pts]
                        widths.append(max(xs) - min(xs))
                    else:
                        widths.append(1)
                total_w = max(1, sum(widths))
                # Distribuer les mots proportionnellement à la largeur
                chunks = []
                word_idx = 0
                for i, w in enumerate(widths):
                    share = max(1, round(len(words) * w / total_w))
                    if i == len(widths) - 1:
                        # Dernier polygon : prend tout ce qui reste
                        chunk_words = words[word_idx:]
                    else:
                        chunk_words = words[word_idx:word_idx + share]
                    chunks.append(" ".join(chunk_words) if chunk_words else "")
                    word_idx += len(chunk_words)
                lines = chunks
        
        # Si on a plus de lignes de texte que de regions, on combine les dernieres
        if len(lines) > len(regions):
            excess = lines[len(regions)-1:]
            lines = lines[:len(regions)-1] + [" ".join(excess)]
        
        # S'il y a moins de lignes, on utilisera juste les premieres regions

        # Largeur du BLOC : enveloppe horizontale de tous les polygones.
        block_x1, block_x2 = None, None
        for r in regions:
            pts_r = r.get('bbox') if isinstance(r, dict) else None
            if not pts_r or len(pts_r) < 3:
                continue
            rxs = [p[0] + region_offset_x for p in pts_r]
            lo, hi = int(min(rxs)), int(max(rxs))
            block_x1 = lo if block_x1 is None else min(block_x1, lo)
            block_x2 = hi if block_x2 is None else max(block_x2, hi)
        block_w = max(1, (block_x2 - block_x1)) if block_x1 is not None else 1

        # UNE SEULE TAILLE pour tout le bloc, ET un découpage recalculé à cette
        # taille.
        #
        # Ajuster chaque ligne à son propre polygone donnait des corps
        # différents dans le même cartouche : mesuré sur « IT WOULD GO QUIET,
        # ONLY TO ERUPT AGAIN WITHOUT WARNING. », la 3e ligne sortait deux fois
        # plus petite que la 1re. Aucun lettreur ne fait ça.
        #
        # Mais prendre simplement la plus petite des tailles ne marche pas non
        # plus : la répartition des mots par largeur de polygone ne retrouve pas
        # les coupures d'origine, une ligne se retrouve trop chargée, et sa
        # taille tire tout le cartouche vers le bas. On cherche donc la plus
        # grande taille à laquelle le texte, redécoupé sur la largeur du BLOC,
        # tient encore dans le nombre de lignes disponibles.
        fitted = self._fit_block_lines(
            text, lines, regions, region_offset_y, block_w, font_path,
            source_line_height,
        )
        block_fs: Optional[int] = None
        rewrapped = False
        if fitted is not None:
            block_fs, lines = fitted
            rewrapped = True
        else:
            for i, line in enumerate(lines):
                if i >= len(regions) or not line:
                    continue
                fs_i = self._fit_line_font_size(
                    line, regions[i], region_offset_x, region_offset_y,
                    block_w, font_path, source_line_height,
                )
                if fs_i is None:
                    continue
                block_fs = fs_i if block_fs is None else min(block_fs, fs_i)

        for i, line in enumerate(lines):
            if i >= len(regions): break
            region = regions[i]

            # Recup bbox du polygone pour cette ligne
            pts = region.get('bbox')
            if not pts or len(pts) < 3: continue

            xs = [p[0] + region_offset_x for p in pts]
            ys = [p[1] + region_offset_y for p in pts]
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))

            rw, rh = max(1, x2 - x1), max(1, y2 - y1)

            # Largeur disponible = celle du BLOC, pas celle du polygone de
            # cette ligne. Le polygone est serré sur le texte d'origine ; comme
            # la police de rendu est plus large que celle du studio, la
            # contrainte de largeur mordait avant la contrainte de hauteur et
            # chaque ligne rétrécissait — les cartouches sortaient nettement
            # plus petits que l'original, donc difficiles à lire. La hauteur du
            # polygone, elle, reste la vraie référence : c'est elle qui
            # conserve le corps de texte de la planche.
            # Les cartouches ont le DROIT DE DÉBORDER de la largeur d'origine.
            #
            # Notre police de lettrage est plus large par caractère que celle
            # des studios : à hauteur de casse égale il faut plus de place, donc
            # plus de lignes, donc un corps plus petit — les cartouches
            # sortaient systématiquement plus maigres que l'original. Puisque
            # ces textes sont posés sur du décor et non enfermés dans une bulle,
            # les laisser dépasser un peu rend la taille et le poids de la
            # planche, ce qui compte davantage que de respecter au pixel près
            # l'encombrement d'origine.
            rw = max(rw, int(block_w * self.CAPTION_WIDTH_ALLOWANCE))
            
            # Recherche dichotomique pour trouver la taille qui rentre (largeur ET hauteur)
            lo = self.cfg.min_font_size
            hi = self.cfg.max_font_size
            if source_line_height and source_line_height > 4:
                em_source = float(source_line_height) / 0.75
                hi = min(hi, int(em_source * 1.05))
            
            best_fs = block_fs if block_fs is not None else lo

            font = self._load_font_from_path(font_path, best_fs)
            if not font: continue
            
            # Dessiner la ligne
            offset_x, ink_w = self._line_extents(font, line)
            _bb = font.getbbox(line)
            line_h = (_bb[3] - _bb[1])
            # Décalage du haut de l'ENCRE dans le cadratin. `draw.text()` place
            # son `y` au haut du CADRATIN, alors que `line_h` ci-dessus est une
            # hauteur d'ENCRE : sans cette compensation, chaque ligne était
            # dessinée `_bb[1]` px trop bas, soit 20 à 30 % du corps pour des
            # capitales. L'offset horizontal, lui, était déjà compensé par
            # `offset_x` — c'était une asymétrie, pas un choix.
            offset_y = _bb[1]
            
            # Centrer la ligne au milieu de son polygone — SAUF si on a
            # redécoupé le texte nous-mêmes : les polygones décrivent alors les
            # coupures de la planche d'origine, pas les nôtres, et centrer
            # chacune de nos lignes sur un polygone qui ne lui correspond plus
            # décale le bloc et le rend irrégulier (mesuré sur « WHEN I WAS
            # TEN, JUST A MONTH BEFORE ENTERING THE ROYAL ACADEMY, » : lignes
            # poussées vers la droite, alignement en escalier). Dans ce cas le
            # bon repère est le centre du BLOC, commun à toutes les lignes ;
            # les ordonnées, elles, restent celles des polygones, ce qui
            # préserve l'interligne de la planche.
            if rewrapped and block_x1 is not None:
                center_x = (block_x1 + block_x2) / 2.0
                xp = int(round(center_x - ink_w / 2.0)) - offset_x
            else:
                xp = x1 + (rw - ink_w) // 2 - offset_x
            # Position VERTICALE : sur la bande d'encre source quand elle a
            # été mesurée (cf. `_prepare_render_style`), sinon centrage dans le
            # polygone. Le polygone contient de la place de jambage inutilisée
            # par des capitales ; s'y centrer remontait le texte.
            _iy0 = region.get('ink_y0') if isinstance(region, dict) else None
            _iy1 = region.get('ink_y1') if isinstance(region, dict) else None
            if _iy0 is not None and _iy1 is not None and _iy1 > _iy0:
                _cy = (float(_iy0) + float(_iy1)) / 2.0 + region_offset_y
                yp = int(round(_cy - line_h / 2.0)) - offset_y
            else:
                yp = y1 + (rh - line_h) // 2 - offset_y

            # Dessiner
            if outline_color is not None and outline_width > 0:
                img = self._draw_text_with_outline_pil(img, line, xp, yp, font, text_color, outline_color, outline_width)
            else:
                img = self._draw_text_pil(img, line, xp, yp, font, text_color)

        # ── HOOK DE MESURE (additif, ne change aucun comportement) ──
        # Ce chemin sortait sans renseigner `last_layout_debug`, alors que
        # `insert_text` le renseigne sur tous les siens : le banc de mesure
        # perdait 38 zones sur 332, dont 25 cartouches out_text — précisément
        # là où le corps de texte s'écarte le plus de la planche d'origine.
        try:
            drawn = [ln for i, ln in enumerate(lines) if i < len(regions) and ln]
            self.last_layout_debug = {
                "bail": None, "mode": "exact_lines", "text": str(text)[:60],
                "route_costs": dict(getattr(self, "_last_route_costs", {}) or {}),
                "font_size_final": block_fs,
                "source_line_height": source_line_height,
                "n_lines": len(drawn), "lines": drawn,
                "n_regions": len(regions), "block_w": block_w,
                "rewrapped": bool(rewrapped), "angle": angle,
            }
        except Exception as _dbg_e:
            self.last_layout_debug = {"bail": "debug_hook_error", "error": str(_dbg_e)}

        return img

    def _fit_block_lines(
        self, text: str, current_lines: List[str], regions: List[Dict],
        region_offset_y: int, block_w: int, font_path: Optional[str],
        source_line_height: Optional[float],
    ) -> Optional[Tuple[int, List[str]]]:
        """(taille, lignes) : plus grande taille où le texte, redécoupé sur la
        largeur du bloc, tient dans les lignes disponibles.

        Renvoie None si le texte porte déjà ses propres retours à la ligne (on
        respecte alors le découpage fourni) ou si aucune taille ne convient.
        """
        if '\n' in (text or '') or len(regions) < 2:
            return None
        flat = " ".join((text or "").split())
        if not flat:
            return None

        heights = []
        for r in regions:
            pts = r.get('bbox') if isinstance(r, dict) else None
            if not pts or len(pts) < 3:
                heights.append(None)
                continue
            ys = [p[1] + region_offset_y for p in pts]
            heights.append(max(1, int(max(ys)) - int(min(ys))))
        if not any(h for h in heights):
            return None

        lo = int(self.cfg.min_font_size)
        hi = int(self.cfg.max_font_size)
        if source_line_height and source_line_height > 4:
            # Plafond relevé pour les cartouches : autoriser le débordement en
            # LARGEUR ne servait à rien tant que la taille restait bornée à celle
            # de la planche (mesuré : aucun changement visible). Ce qui manquait
            # n'était pas la place mais le CORPS.
            hi = min(hi, int((float(source_line_height) / 0.75) * self.CAPTION_SIZE_ALLOWANCE))
        hi = max(lo, hi)

        best: Optional[Tuple[int, List[str]]] = None
        while lo <= hi:
            mid = (lo + hi) // 2
            font = self._load_font_from_path(font_path, mid)
            if not font:
                break
            wrapped = self.wrap_text(flat, font, max(10, int(block_w * 0.98)))
            ok = len(wrapped) <= len(regions)
            if ok:
                for idx, ln in enumerate(wrapped):
                    rh = heights[idx] if idx < len(heights) else None
                    if not rh:
                        continue
                    bb = font.getbbox(ln)
                    if (bb[3] - bb[1]) > rh * 1.10:
                        ok = False
                        break
            if ok:
                best = (mid, wrapped)
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    def _fit_line_font_size(
        self, line: str, region: Dict, region_offset_x: int, region_offset_y: int,
        block_w: int, font_path: Optional[str], source_line_height: Optional[float],
    ) -> Optional[int]:
        """Plus grande taille à laquelle `line` tient dans son polygone.

        Largeur autorisée : celle du BLOC (le polygone d'une ligne courte ne
        doit pas rapetisser tout le cartouche). Hauteur autorisée : celle du
        polygone de CETTE ligne, +10 % — le polygone OCR est un peu plus
        généreux que l'encre qu'il contient, et le rapport hauteur-de-casse /
        cadratin diffère d'une police à l'autre.
        """
        pts = region.get('bbox') if isinstance(region, dict) else None
        if not pts or len(pts) < 3:
            return None
        ys = [p[1] + region_offset_y for p in pts]
        xs = [p[0] + region_offset_x for p in pts]
        rh = max(1, int(max(ys)) - int(min(ys)))
        rw = max(max(1, int(max(xs)) - int(min(xs))),
                 int(block_w * self.CAPTION_WIDTH_ALLOWANCE))

        lo = int(self.cfg.min_font_size)
        hi = int(self.cfg.max_font_size)
        if source_line_height and source_line_height > 4:
            # Plafond relevé pour les cartouches : autoriser le débordement en
            # LARGEUR ne servait à rien tant que la taille restait bornée à celle
            # de la planche (mesuré : aucun changement visible). Ce qui manquait
            # n'était pas la place mais le CORPS.
            hi = min(hi, int((float(source_line_height) / 0.75) * self.CAPTION_SIZE_ALLOWANCE))
        hi = max(lo, hi)

        best = lo
        while lo <= hi:
            mid = (lo + hi) // 2
            f = self._load_font_from_path(font_path, mid)
            if not f:
                break
            _, ink_w = self._line_extents(f, line)
            bb = f.getbbox(line)
            ink_h = bb[3] - bb[1]
            if ink_w <= rw and ink_h <= rh * 1.25:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    def _draw_text_with_outline_pil(self, img, text, x, y, font, text_color, outline_color, outline_width):
        pad = outline_width * 2 + 5
        offset_x, ink_w = self._line_extents(font, text)
        line_h = (font.getbbox(text)[3] - font.getbbox(text)[1])
        
        rx1, ry1 = max(0, x - pad), max(0, y - pad)
        rx2, ry2 = min(img.shape[1], x + ink_w + pad*2), min(img.shape[0], y + line_h + pad*2)
        if rx2 <= rx1 or ry2 <= ry1: return img
        
        crop = img[ry1:ry2, rx1:rx2]
        crop_pil = ImageUtils.cv2_to_pil(crop)
        draw = ImageDraw.Draw(crop_pil)
        
        draw.text(
            (x - rx1, y - ry1), text, font=font,
            fill=text_color, stroke_width=outline_width, stroke_fill=outline_color,
        )
        img[ry1:ry2, rx1:rx2] = ImageUtils.pil_to_cv2(crop_pil)
        return img
        
    def _draw_text_pil(self, img, text, x, y, font, text_color):
        pad = 5
        offset_x, ink_w = self._line_extents(font, text)
        line_h = (font.getbbox(text)[3] - font.getbbox(text)[1])
        
        rx1, ry1 = max(0, x - pad), max(0, y - pad)
        rx2, ry2 = min(img.shape[1], x + ink_w + pad*2), min(img.shape[0], y + line_h + pad*2)
        if rx2 <= rx1 or ry2 <= ry1: return img
        
        crop = img[ry1:ry2, rx1:rx2]
        crop_pil = ImageUtils.cv2_to_pil(crop)
        draw = ImageDraw.Draw(crop_pil)
        
        draw.text((x - rx1, y - ry1), text, font=font, fill=text_color)
        img[ry1:ry2, rx1:rx2] = ImageUtils.pil_to_cv2(crop_pil)
        return img

    @staticmethod
    def _optical_center_y(
        font, lines: List[str], line_h: int, spacing: int,
        top: int, inner_h: int, fallback: int,
    ) -> int:
        """Ordonnée de la 1re ligne pour que l'ENCRE du bloc soit centrée.

        `font.getbbox` donne l'étendue verticale réelle des glyphes par rapport
        à l'origine du texte : pour des capitales, le haut de l'encre est
        nettement sous le haut du cadratin et il n'y a pas de descendante. Sans
        cette correction le bloc est systématiquement rendu trop haut.
        """
        try:
            drawn = [ln for ln in lines if ln]
            if not drawn:
                return fallback
            first, last = drawn[0], drawn[-1]
            top_ink = float(font.getbbox(first)[1])
            bot_ink = float(font.getbbox(last)[3])
            n_gaps = len(lines) - 1
            ink_h = n_gaps * (line_h + spacing) + (bot_ink - top_ink)
            if ink_h <= 0 or ink_h > inner_h * 3:
                return fallback
            return int(round(top + (inner_h - ink_h) / 2.0 - top_ink))
        except Exception:
            return fallback

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
        line_centers = layout.get('line_centers')
        if line_centers is not None and len(line_centers) != len(lines):
            line_centers = None

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
            #
            # Centrage OPTIQUE et non métrique : `total_h` compte des cadratins
            # complets (ascendante + descendante), or un texte de manhwa est
            # tout en capitales et son encre n'occupe que la moitié haute du
            # cadratin. Centrer le cadratin laissait donc l'encre visiblement
            # au-dessus du centre de la bulle — le défaut « texte trop haut »
            # visible sur presque toutes les bulles. On centre l'encre réelle,
            # mesurée sur la première et la dernière ligne.
            ys = top + (inner_h - total_h) // 2
            ys = self._optical_center_y(font, lines, line_h, spacing, top, inner_h, ys)

        # Zone à convertir en PIL : elle doit contenir TOUT le texte. Avec le
        # wrap sur masque, une ligne peut légitimement être plus large que
        # `inner_w` (la bulle est plus large que le rectangle inscrit) et
        # déborder des deux côtés — la découper ici la tronquerait au rendu.
        pad = max(4, outline_width * 2 + layout['size'] // 2)
        if line_centers is not None:
            # Chaque ligne est centrée sur SON propre centre de masque (peut
            # varier ligne à ligne sur une bulle asymétrique) : la marge
            # symétrique `overflow_x` autour du bloc ne suffit plus, il faut
            # l'étendue réelle mesurée par ligne.
            half_widths = [self._line_extents(font, ln)[1] / 2.0 for ln in lines]
            line_x1s = [c - hw for c, hw in zip(line_centers, half_widths)]
            line_x2s = [c + hw for c, hw in zip(line_centers, half_widths)]
            rx1 = max(0, int(min([left, ix1] + line_x1s)) - pad)
            rx2 = min(img.shape[1], int(max([ix2, left + inner_w] + line_x2s)) + pad)
        else:
            overflow_x = max(0, (layout['max_line_w'] - inner_w + 1) // 2)
            rx1 = max(0, min(left, ix1) - overflow_x - pad)
            rx2 = min(img.shape[1], max(ix2, left + inner_w) + overflow_x + pad)
        ry1 = max(0, min(ys, iy1) - pad)
        ry2 = min(img.shape[0], max(iy2, ys + total_h) + pad)
        if rx2 <= rx1 or ry2 <= ry1:
            import os
            if os.environ.get('RENDER_DEBUG'):
                print(f"[RENDER_DEBUG] _draw_block bail rect empty rx1={rx1} ry1={ry1} rx2={rx2} ry2={ry2} ys={ys} total_h={total_h} lines={lines!r} ix1={ix1} iy1={iy1} ix2={ix2} iy2={iy2}")
            return img

        crop = img[ry1:ry2, rx1:rx2]
        crop_pil = ImageUtils.cv2_to_pil(crop)
        draw = ImageDraw.Draw(crop_pil)

        for i, line in enumerate(lines):
            if not line:
                continue
            offset_x, ink_w = self._line_extents(font, line)

            if line_centers is not None and self.cfg.horizontal_align not in ('left', 'right') and not top_aligned:
                xp = int(round(line_centers[i] - ink_w / 2.0))
            elif self.cfg.horizontal_align == 'left' or top_aligned:
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
            import os
            if os.environ.get('RENDER_DEBUG'):
                print(f"[RENDER_DEBUG] _draw_rotated bail block_w={block_w} total_h={total_h} lines={lines!r}")
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
            import os
            if os.environ.get('RENDER_DEBUG'):
                print(f"[RENDER_DEBUG] _draw_rotated bail offscreen cx={cx} cy={cy} rw={rw} rh={rh} px1={px1} py1={py1} img_shape={img.shape}")
            return img

        rotated = rotated.crop((sx1, sy1, sx1 + (dx2 - dx1), sy1 + (dy2 - dy1)))

        base = ImageUtils.cv2_to_pil(img[dy1:dy2, dx1:dx2]).convert('RGBA')
        base.alpha_composite(rotated)
        img[dy1:dy2, dx1:dx2] = ImageUtils.pil_to_cv2(base.convert('RGB'))
        return img

