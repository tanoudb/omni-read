"""
═══════════════════════════════════════════════════════════════════════════════
WEBTOON TRANSLATOR V5 PREMIUM - CONFIGURATION CENTRALISÉE
═══════════════════════════════════════════════════════════════════════════════
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS & DIRECTORIES
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).parent.parent.absolute()
ASSETS_DIR = BASE_DIR / "assets"
MODEL_DIR = ASSETS_DIR / "models"
CACHE_DIR = ASSETS_DIR / "cache"
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
LOGS_DIR = BASE_DIR / "logs"

YOLO_MODEL_PATH = MODEL_DIR / "manhwa_v3.pt"  # Nouveau modèle 4 classes
OCR_CACHE_DIR = CACHE_DIR / "ocr_weights"
TRANSLATION_CACHE_DIR = CACHE_DIR / "translation_models"
INPAINTING_CACHE_DIR = CACHE_DIR / "inpainting_models"


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _default_segmentation_backend() -> str:
    explicit = os.environ.get("WEBTOON_SEGMENTATION_BACKEND")
    if explicit is not None and explicit.strip():
        return explicit.strip().lower()
    sam2_ckpt = OCR_CACHE_DIR / "sam2_b.pt"
    return "sam2" if sam2_ckpt.exists() else "hybrid"


def _discover_font_paths() -> List[str]:
    """Découvre les polices du projet + chemins optionnels via env."""
    fonts_dir = ASSETS_DIR / "fonts"
    font_exts = {".ttf", ".otf", ".ttc"}

    discovered = []
    if fonts_dir.exists():
        for font_file in fonts_dir.rglob("*"):
            if font_file.is_file() and font_file.suffix.lower() in font_exts:
                discovered.append(str(font_file))

    env_fonts = os.environ.get("WEBTOON_FONT_PATHS", "").strip()
    if env_fonts:
        for raw_path in env_fonts.split(os.pathsep):
            candidate = raw_path.strip()
            if candidate and Path(candidate).exists():
                discovered.append(candidate)

    # fallback de secours uniquement si aucune police projet trouvée
    if not discovered and os.name == 'nt':
        fallback_windows = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/times.ttf",
            "C:/Windows/Fonts/calibri.ttf",
        ]
        discovered.extend([p for p in fallback_windows if Path(p).exists()])

    # dedupe en conservant l'ordre
    unique_paths = []
    seen = set()
    for path in discovered:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)
    return unique_paths

# ═══════════════════════════════════════════════════════════════════════════════
# DEVICE & PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PerformanceConfig:
    device: str = "cuda"
    use_fp16: bool = True
    aggressive_cleanup: bool = True
    max_batch_size: int = 8
    prefetch_images: bool = False
    num_workers: int = 4
    enable_cache: bool = True
    cache_max_size_mb: int = 500


# ═══════════════════════════════════════════════════════════════════════════════
# DETECTION - YOLO
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DetectionConfig:
    """Configuration détection YOLO"""
    
    # SLICING ADAPTATIF
    enable_adaptive_slicing: bool = True
    base_window_height: int = 2048
    overlap_ratio: float = 0.30
    auto_calibrate_window: bool = False
    min_window_height: int = 1024
    max_window_height: int = 4096
    max_height: int = _env_int("WEBTOON_MAX_HEIGHT", 0)
    
    # MULTI-SCALE
    enable_multi_scale: bool = True
    detection_scales: List[float] = field(default_factory=lambda: [1.0, 0.75, 0.5])
    
    multi_scale_fusion: str = "weighted"
    scale_weights: Dict[float, float] = field(default_factory=lambda: {
        1.0: 1.0,
        0.75: 0.8,
        0.5: 0.6
    })
    
    # ── CONFIDENCE - YOLO v3 (4 classes) ──
    # names: ['System', 'bulle', 'out_text', 'sfx']
    confidence_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'System': 0.55,
        'bulle': 0.70,
        'out_text': 0.30,    # Texte hors bulle (baissé pour capturer plus)
        'sfx': 0.50          # Effets sonores
    })
    
    # NMS - YOLO v3
    nms_iou_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'System': 0.5,
        'bulle': 0.4,
        'out_text': 0.6,
        'sfx': 0.5
    })
    
    multi_scale_nms_iou: float = 0.6
    inter_class_iou_threshold: float = 0.7

    # Padding noir optionnel (hack legacy) autour de l'image avant détection
    use_black_padding: bool = _env_bool("WEBTOON_USE_BLACK_PADDING", False)
    black_padding_ratio: float = float(os.environ.get("WEBTOON_BLACK_PADDING_RATIO", "0.03"))
    
    # ── FILTRAGE BORDURES - DÉSACTIVÉ ──
    # Le filtrage des bordures supprimait des détections valides
    # dans les zones d'overlap du sliding window. La NMS suffit
    # pour éliminer les doublons.
    filter_border_detections: bool = False   # CHANGÉ: True → False
    border_margin_px: int = 50
    
    min_box_area: int = 300
    max_box_area: int = 800000
    min_box_ratio: float = 0.1
    max_box_ratio: float = 10.0
    
    # ── CLASSES À TRADUIRE - YOLO v3 ──
    # bulle = bulles de dialogue (à traduire)
    # out_text = texte hors bulle (à traduire)
    # sfx = effets sonores (ne pas traduire)
    # System = UI/titre (à traduire)
    translatable_classes: List[str] = field(default_factory=lambda: ['bulle', 'out_text'])


# ═══════════════════════════════════════════════════════════════════════════════
# OCR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OCRConfig:
    """Configuration OCR - VL1.5 + RapidOCR"""
    backend = os.environ.get("WEBTOON_OCR_BACKEND", "paddleocr-vl-v1.5").strip().lower()
    primary_backend: str = os.environ.get("WEBTOON_OCR_PRIMARY", "paddleocr-vl-v1.5").strip().lower()
    fallback_backends: List[str] = field(default_factory=lambda: [
        b.strip().lower()
        for b in os.environ.get("WEBTOON_OCR_FALLBACKS", "rapidocr-ppocrv5").split(",")
        if b.strip()
    ])
    fallback_min_confidence: float = float(os.environ.get("WEBTOON_OCR_FALLBACK_MIN_CONF", "0.72"))
    use_fp16: bool = True
    use_vl15: bool = _env_bool("WEBTOON_USE_VL15", True)
    vl15_min_free_vram_gb: float = float(os.environ.get("WEBTOON_VL15_MIN_FREE_VRAM_GB", "4.0"))
    
    # PREPROCESSING
    enable_preprocessing: bool = True
    auto_resize: bool = True
    min_text_height: int = 20
    max_resize_factor: int = 3
    resize_interpolation: str = "lanczos"
    enable_contrast_enhancement: bool = False
    enable_denoising: bool = False
    convert_to_rgb: bool = True
    
    # ── VALIDATION - SEUILS ASSOUPLIS ──
    min_confidence: float = 0.1
    min_text_length: int = 2
    
    enable_spell_check: bool = False
    remove_isolated_chars: bool = True
    
    filter_numeric_only: bool = True
    filter_special_chars_only: bool = True
    max_numeric_ratio: float = 0.8


# ═══════════════════════════════════════════════════════════════════════════════
# SEGMENTATION - YOLO PROMPTS → PIXEL MASKS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SegmentationConfig:
    enable_precise_masks: bool = _env_bool("WEBTOON_ENABLE_PRECISE_MASKS", True)
    backend: str = _default_segmentation_backend()  # hybrid|sam2|ocr_regions
    sam2_checkpoint: str = os.environ.get("WEBTOON_SAM2_CHECKPOINT", str(OCR_CACHE_DIR / "sam2_b.pt"))
    use_multimask: bool = _env_bool("WEBTOON_SEGMENTATION_MULTIMASK", True)
    min_component_area: int = int(os.environ.get("WEBTOON_SEGMENTATION_MIN_COMPONENT", "24"))
    mask_dilate_kernel: int = int(os.environ.get("WEBTOON_SEGMENTATION_DILATE", "9"))
    bbox_prompt_margin: int = int(os.environ.get("WEBTOON_SEGMENTATION_PROMPT_MARGIN", "4"))


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSLATION - NLLB
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TranslationConfig:
    backend: str = os.environ.get("WEBTOON_TRANSLATION_BACKEND", "local_llm").strip().lower()

    # NLLB (legacy)
    model_name: str = os.environ.get("WEBTOON_NLLB_MODEL", "facebook/nllb-200-distilled-600M")

    # Local LLM (aucune API externe)
    llm_model_name: str = os.environ.get("WEBTOON_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    llm_require_cuda: bool = _env_bool("WEBTOON_LLM_REQUIRE_CUDA", True)
    llm_max_new_tokens: int = int(os.environ.get("WEBTOON_LLM_MAX_NEW_TOKENS", "512"))
    llm_temperature: float = float(os.environ.get("WEBTOON_LLM_TEMPERATURE", "0.0"))
    llm_top_p: float = float(os.environ.get("WEBTOON_LLM_TOP_P", "1.0"))
    llm_repetition_penalty: float = float(os.environ.get("WEBTOON_LLM_REPETITION_PENALTY", "1.05"))
    llm_prompt_template: str = os.environ.get(
        "WEBTOON_LLM_PROMPT_TEMPLATE",
        "Tu es un traducteur expert manga/webtoon/manhwa EN→FR. "
        "Règles : français naturel jamais littéral. Conserve le ton émotionnel. "
        "Les onomatopées (HUFF, AHH, BOOM, CRASH...) restent EXACTEMENT en original, ne jamais traduire. "
        "Si c'est un watermark/crédit/scanlation/URL/@handle, renvoie le texte inchangé. "
        "Ne commence jamais par Hello/I am/Here is. "
        "Respecte la ponctuation et effets stylistiques (!!!, ...). "
        "Réponds uniquement avec la traduction.\n"
        "TEXT:\n{text}\nTRADUCTION:"
    )
    

    use_fp16: bool = True  # CRITIQUE pour tes 8 Go de RAM
    use_bitsandbytes: bool = _env_bool("WEBTOON_USE_BITSANDBYTES", False)
    bnb_4bit: bool = _env_bool("WEBTOON_BNB_4BIT", True)
    bnb_8bit: bool = _env_bool("WEBTOON_BNB_8BIT", False)
    
    source_lang: str = "en"
    auto_detect_source_lang: bool = _env_bool("WEBTOON_AUTO_DETECT_SOURCE_LANG", True)
    fallback_source_lang: str = os.environ.get("WEBTOON_FALLBACK_SOURCE_LANG", "en").strip().lower() or "en"
    target_lang: str = "fr"
    
    lang_codes: Dict[str, str] = field(default_factory=lambda: {
        'en': 'eng_Latn',
        'fr': 'fra_Latn',
        'ko': 'kor_Hang',
        'ja': 'jpn_Jpan',
        'es': 'spa_Latn',
        'de': 'deu_Latn',
        'it': 'ita_Latn',
        'pt': 'por_Latn',
        'ru': 'rus_Cyrl',
        'zh': 'zho_Hans'
    })
    
    max_length: int = int(os.environ.get("WEBTOON_TRANSLATION_MAX_LENGTH", "640"))
    num_beams: int = int(os.environ.get("WEBTOON_TRANSLATION_NUM_BEAMS", "7"))
    early_stopping: bool = _env_bool("WEBTOON_TRANSLATION_EARLY_STOPPING", True)
    
    enable_context_grouping: bool = _env_bool("WEBTOON_ENABLE_CONTEXT_GROUPING", True)
    context_distance_threshold: int = int(os.environ.get("WEBTOON_CONTEXT_DISTANCE_THRESHOLD", "420"))
    max_group_size: int = int(os.environ.get("WEBTOON_MAX_GROUP_SIZE", "7"))
    group_separator: str = " | "
    
    enable_cache: bool = True
    cache_file: str = "translation_cache_v5.json"
    
    skip_numeric_only: bool = True
    skip_single_char: bool = True
    skip_if_no_letters: bool = True
    preserve_ellipsis: bool = True
    preserve_emphasis: bool = True

    # Glossaire/overrides manuels pour cohérence de style
    forced_translations: Dict[str, str] = field(default_factory=lambda: {
        "MISO.": "Miso.",
        "GHISLAIN PERDIUM.": "GHISLAIN PERDIUM.",
        "YUNHO! HELP ME!": "YUNHO! HELP ME!",
        "LOOK HERE, MRS. YUJIN JUNG!": "Regardez ici, Mme YUJIN JUNG !"
    })


# ═══════════════════════════════════════════════════════════════════════════════
# RENDERING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RenderingConfig:
    inpainting_method: str = "simple_fill"
    margin_px: int = 5
    inpaint_radius: int = 7
    inpaint_method: str = "telea"
    background_detection: str = "median_border"
    background_sample_ratio: float = 0.1
    
    font_paths: List[str] = field(default_factory=_discover_font_paths)
    
    enable_dynamic_sizing: bool = True
    target_fill_ratio: float = 0.85
    min_font_size: int = 12
    max_font_size: int = 80
    font_size_step: int = 2
    max_iterations: int = 15
    
    horizontal_align: str = "center"
    vertical_align: str = "center"
    line_spacing_ratio: float = 0.20
    word_wrap_ratio: float = 0.90
    padding_horizontal: int = 8
    padding_vertical: int = 6
    
    auto_text_color: bool = True
    luminosity_threshold: int = 128
    default_text_color: Tuple[int, int, int] = (0, 0, 0)
    default_outline_color: Tuple[int, int, int] = (255, 255, 255)
    enable_outline: bool = True
    outline_width: int = 2
    outline_method: str = "stroke"
    enable_shadow: bool = False
    shadow_offset: Tuple[int, int] = (2, 2)
    shadow_blur: int = 3

    preserve_original_text_color: bool = _env_bool("WEBTOON_PRESERVE_TEXT_COLOR", True)
    auto_style_typesetting: bool = _env_bool("WEBTOON_AUTO_STYLE_TYPESETTING", True)
    lock_text_to_ocr_regions: bool = _env_bool("WEBTOON_LOCK_TEXT_TO_OCR_REGIONS", True)
    lock_text_system_only: bool = _env_bool("WEBTOON_LOCK_TEXT_SYSTEM_ONLY", True)
    precise_masks: bool = _env_bool("WEBTOON_PRECISE_MASKS", True)
    inpainting_model_id: str = os.environ.get("WEBTOON_INPAINTING_MODEL_ID", "dreMaz/AnimeMangaInpainting")
    inpainting_model_path: Path = INPAINTING_CACHE_DIR / "dreMaz--AnimeMangaInpainting"


# ═══════════════════════════════════════════════════════════════════════════════
# FILTERS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FilterConfig:
    enable_watermark_filter: bool = True
    
    watermark_patterns: List[str] = field(default_factory=lambda: [
        r'asurascans?\.com',
        r'reaper-?scans?\.com',
        r'flamescans?\.org',
        r'\.com',
        r'\.net',
        r'\.org',
        r'www\.',
        r'http',
        r'chapter\s*\d+',
        r'page\s*\d+',
        r'read\s+on',
        r'scan\s+by'
    ])
    
    enable_sfx_filter: bool = True
    
    sfx_patterns: List[str] = field(default_factory=lambda: [
        r'^(.)\1{2,}$',
        r'^[!?.*#@\-_]{2,}$',
    ])
    
    # ── FILTRAGE BORDS - ASSOUPLI ──
    # Sur les images longues (14400px), le titre en haut et les crédits
    # en bas sont filtrés, mais 5% de 14400px = 720px c'est beaucoup.
    # On réduit pour ne filtrer que le strict bord.
    filter_top_edge: bool = True
    top_edge_threshold: float = 0.02    # Réduit de 0.05 → 0.02
    filter_bottom_edge: bool = True
    bottom_edge_threshold: float = 0.98  # Réduit de 0.95 → 0.98


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LoggingConfig:
    level: str = "INFO"
    console_format: str = "%(levelname)s | %(message)s"
    file_format: str = "%(asctime)s | %(levelname)s | %(module)s | %(message)s"
    enable_file_logging: bool = True
    log_file: str = "webtoon_v5.log"
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 3
    log_statistics: bool = True
    log_timings: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION GLOBALE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    YOLO_MODEL_PATH: Path = YOLO_MODEL_PATH 
    INPUT_DIR: Path = INPUT_DIR
    OUTPUT_DIR: Path = OUTPUT_DIR
    LOGS_DIR: Path = LOGS_DIR
    OCR_CACHE_DIR: Path = OCR_CACHE_DIR
    TRANSLATION_CACHE_DIR: Path = TRANSLATION_CACHE_DIR
    INPAINTING_CACHE_DIR: Path = INPAINTING_CACHE_DIR
    
    
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def __post_init__(self):
        for directory in [INPUT_DIR, OUTPUT_DIR, LOGS_DIR, MODEL_DIR, CACHE_DIR,
                 OCR_CACHE_DIR, TRANSLATION_CACHE_DIR, INPAINTING_CACHE_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Config':
        return cls(**data)


config = Config()