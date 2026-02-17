"""
Package des backends OCR
Expose les backends disponibles et l'interface abstraite
"""

from .base import OCRBackend
from .paddleocr_vl_v15 import PaddleOCRVLV15Backend
from .rapidocr_ppocrv5 import RapidOCRPPOCRv5Backend

__all__ = [
    'OCRBackend',
    'PaddleOCRVLV15Backend',
    'RapidOCRPPOCRv5Backend',
]
