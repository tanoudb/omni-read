from .detector import YOLODetector, Detection
from .segmenter import SmartSegmenter
from .ocr import OCREngine
from .renderer import TextRenderer
from .translation import NLLBTranslator
# Gemini (Gemini/Google) translator
from .translator_gemini import GeminiTranslator
# QCheck post-render auto-repair
try:
    from .qcheck import QCheckEngine
except ImportError:
    QCheckEngine = None
# Alias pour la compatibilité si nécessaire
Translator = NLLBTranslator

__all__ = [
    'YOLODetector',
    'Detection',
    'SmartSegmenter',
    'OCREngine',
    'TextRenderer',
    'Translator',
    'NLLBTranslator',
    'GeminiTranslator',
    'QCheckEngine',
]