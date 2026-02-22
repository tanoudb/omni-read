"""
OCR engine: PaddleOCR-VL 1.5 (Transformers) + RapidOCR PP-OCRv5 fallback.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Callable

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from utils import ImageUtils, TextFilter
from .backends import OCRBackend, PaddleOCRVLV15Backend, RapidOCRPPOCRv5Backend


BACKEND_REGISTRY = {
    "paddleocr-vl-v1.5": PaddleOCRVLV15Backend,
    "rapidocr-ppocrv5": RapidOCRPPOCRv5Backend,
    "rapidocr": RapidOCRPPOCRv5Backend,
    "ppocr-v5": RapidOCRPPOCRv5Backend,
}


class OCREngine:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.cfg = config.ocr
        self.primary_backend: Optional[OCRBackend] = None
        self.fallback_backends: List[OCRBackend] = []

        self.text_filter = TextFilter(
            watermark_patterns=config.filters.watermark_patterns,
            sfx_patterns=config.filters.sfx_patterns,
        )

        self._load_backends_chain()

    def _load_backends_chain(self):
        primary_name = (getattr(self.cfg, "backend", None) or getattr(self.cfg, "primary_backend", None) or "paddleocr-vl-v1.5").strip().lower()
        fallback_names = [
            str(name).strip().lower()
            for name in getattr(self.cfg, "fallback_backends", ["rapidocr-ppocrv5"])
            if str(name).strip()
        ]
        fallback_names = [name for name in fallback_names if name != primary_name]

        loaded: List[OCRBackend] = []
        for name in [primary_name] + fallback_names:
            backend_class = BACKEND_REGISTRY.get(name)
            if backend_class is None:
                continue
            try:
                backend = backend_class()
                backend.load(self.device)
                loaded.append(backend)
                print(f"✅ Backend OCR chargé: {backend.name}")
            except Exception as exc:
                print(f"⚠️ Backend OCR {name} indisponible: {exc}")

        if not loaded:
            raise RuntimeError("Aucun backend OCR disponible (VL1.5/RapidOCR)")

        self.primary_backend = loaded[0]
        self.fallback_backends = loaded[1:]

    def preprocess_image(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = img.shape[:2]
        upscale_factor = 1.0

        if h < 80:
            upscale_factor = 150 / max(1, h)
            img = cv2.resize(img, (int(w * upscale_factor), 150), interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]
        elif h < 100:
            upscale_factor = 120 / max(1, h)
            img = cv2.resize(img, (int(w * upscale_factor), 120), interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]

        if h < 64:
            extra = 64 / max(1, h)
            upscale_factor *= extra
            img = cv2.resize(img, (max(1, int(w * extra)), 64), interpolation=cv2.INTER_CUBIC)

        if self.cfg.auto_resize:
            img = ImageUtils.smart_resize(
                img,
                min_height=self.cfg.min_text_height,
                max_factor=self.cfg.max_resize_factor,
                interpolation=self.cfg.resize_interpolation,
            )

        return img, upscale_factor

    def post_process_text(self, text: str) -> str:
        if not text:
            return ""

        text = self.text_filter.clean_text(text)
        text = re.sub(r"\b1\.(?=\s+[A-Z])", "I.", text)
        text = re.sub(r"\bI\.(?=\s+THE\b)", "I,", text)
        text = re.sub(r"(?<=[A-Z])\s+1\s+(?=[A-Z])", " I ", text)
        text = re.sub(r"\s+", " ", text).strip()

        if self.cfg.remove_isolated_chars:
            words = text.split()
            words = [w for w in words if len(w) > 1 or w.isalnum()]
            text = " ".join(words)

        return text

    def is_valid_text(self, text: str, confidence: float) -> Tuple[bool, Optional[str]]:
        if confidence < self.cfg.min_confidence:
            return False, "low_confidence"

        if len(text.strip()) < self.cfg.min_text_length:
            return False, "too_short"

        should_skip, reason = self.text_filter.should_skip(
            text,
            min_length=self.cfg.min_text_length,
            max_numeric_ratio=self.cfg.max_numeric_ratio,
        )
        if should_skip:
            return False, reason

        if self.cfg.filter_numeric_only and self.text_filter.is_numeric_only(text, self.cfg.max_numeric_ratio):
            return False, "numeric_only"

        if self.cfg.filter_special_chars_only and self.text_filter.is_special_chars_only(text):
            return False, "special_chars_only"

        return True, None

    def _choose_best(self, candidates: List[Tuple[str, float, List[Dict]]]) -> Tuple[str, float, List[Dict]]:
        best_text = ""
        best_conf = 0.0
        best_regions: List[Dict] = []

        for text, conf, regions in candidates:
            clean = self.post_process_text(text)
            if not clean:
                continue
            valid, _ = self.is_valid_text(clean, conf)
            if not valid and conf < best_conf:
                continue
            if conf >= best_conf:
                best_text, best_conf, best_regions = clean, float(conf), regions or []

        return best_text, best_conf, best_regions

    @staticmethod
    def _preview_text(value: str, max_len: int = 140) -> str:
        txt = (value or "").replace("\n", " ").strip()
        if len(txt) <= max_len:
            return txt
        return txt[:max_len] + "..."

    def get_runtime_diagnostics(self) -> Dict:
        details: Dict = {
            "device": self.device,
            "primary": self.primary_backend.name if self.primary_backend else "none",
            "fallbacks": [b.name for b in self.fallback_backends if b],
        }

        backend_infos = []
        for backend in [self.primary_backend, *self.fallback_backends]:
            if backend is None:
                continue
            info = {"name": backend.name}
            getter = getattr(backend, "get_runtime_info", None)
            if callable(getter):
                try:
                    extra = getter()
                    if isinstance(extra, dict):
                        info.update(extra)
                except Exception as exc:
                    info["runtime_info_error"] = str(exc)
            backend_infos.append(info)

        details["backends"] = backend_infos
        return details

    def extract_text(self, img: np.ndarray) -> Tuple[Optional[str], float, bool, Optional[str], List[Dict], float]:
        results = self.extract_batch([img])
        return results[0]

    def extract_batch(
        self,
        crops: List[np.ndarray],
        debug_hook: Optional[Callable[[str], None]] = None,
    ) -> List[Tuple[Optional[str], float, bool, Optional[str], List[Dict], float]]:
        if not crops:
            return []

        if debug_hook:
            debug_hook(f"[OCR] crops reçus: {len(crops)}")

        processed: List[np.ndarray] = []
        upscale_factors: List[float] = []
        for crop_idx, crop in enumerate(crops):
            img_proc, up = self.preprocess_image(crop)
            processed.append(img_proc)
            upscale_factors.append(up)
            if debug_hook:
                h0, w0 = crop.shape[:2]
                h1, w1 = img_proc.shape[:2]
                debug_hook(
                    f"[OCR][crop {crop_idx}] input={w0}x{h0} dtype={crop.dtype} -> preproc={w1}x{h1} upscale={up:.2f}"
                )

        all_backend_outputs: List[List[Tuple[str, float, List[Dict]]]] = [[] for _ in processed]

        backends: List[OCRBackend] = []
        if self.primary_backend is not None:
            backends.append(self.primary_backend)
        backends.extend(self.fallback_backends)

        for backend_idx, backend in enumerate(backends):
            if backend is None:
                continue
            if debug_hook:
                debug_hook(f"[OCR] backend[{backend_idx}] actif: {backend.name}")
            batch_reader = getattr(backend, "read_batch", None)
            if callable(batch_reader):
                outputs = batch_reader(processed)
            else:
                outputs = [backend.read_text(img) for img in processed]

            if debug_hook:
                for i, out in enumerate(outputs):
                    try:
                        text, conf, regions = out
                        preview = self._preview_text(text)
                        debug_hook(
                            f"[OCR][crop {i}][{backend.name}] brut: conf={float(conf):.3f} regions={len(regions or [])} text='{preview}'"
                        )
                    except Exception as exc:
                        debug_hook(f"[OCR][crop {i}][{backend.name}] brut: format inattendu ({exc})")

            for i, out in enumerate(outputs):
                if i >= len(all_backend_outputs):
                    break
                all_backend_outputs[i].append(out)

            if backend_idx == 0:
                min_conf = float(getattr(self.cfg, "fallback_min_confidence", 0.72))
                if all(
                    (self._choose_best([cand])[0] and self._choose_best([cand])[1] >= min_conf)
                    for cand in outputs
                ):
                    break

        final_results: List[Tuple[Optional[str], float, bool, Optional[str], List[Dict], float]] = []
        for i, candidates in enumerate(all_backend_outputs):
            text, confidence, regions = self._choose_best(candidates)
            is_valid, skip_reason = self.is_valid_text(text, confidence) if text else (False, "empty")
            if not is_valid:
                if debug_hook:
                    debug_hook(
                        f"[OCR][crop {i}] skip={skip_reason} conf={float(confidence):.3f} text='{self._preview_text(text)}'"
                    )
                final_results.append((None, float(confidence), False, skip_reason, [], upscale_factors[i]))
            else:
                if debug_hook:
                    debug_hook(
                        f"[OCR][crop {i}] valid conf={float(confidence):.3f} regions={len(regions or [])} text='{self._preview_text(text)}'"
                    )
                final_results.append((text, float(confidence), True, None, regions, upscale_factors[i]))

        return final_results

    def predict_full_image(self, image_path: Path) -> List[Dict]:
        return []

    def get_backend_name(self) -> str:
        names = []
        if self.primary_backend:
            names.append(self.primary_backend.name)
        names.extend([b.name for b in self.fallback_backends if b])
        return " -> ".join(names) if names else "none"

    def __del__(self):
        for backend in [self.primary_backend, *self.fallback_backends]:
            if backend:
                try:
                    backend.unload()
                except Exception:
                    pass