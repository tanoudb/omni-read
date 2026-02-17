"""
Backend PaddleOCR-VL v1.5 via Transformers.
"""

from __future__ import annotations

from typing import Tuple, List, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image

from config import config
from .base import OCRBackend


class PaddleOCRVLV15Backend(OCRBackend):
    def __init__(self):
        self.processor = None
        self.model = None
        self.device = "cpu"

    @property
    def name(self) -> str:
        return "PaddleOCR-VL-v1.5-TF"

    @staticmethod
    def _normalize_box(raw: Any) -> List[List[int]]:
        try:
            arr = np.array(raw, dtype=np.float32)
        except Exception:
            return []

        if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] >= 2:
            return [[int(p[0]), int(p[1])] for p in arr]
        if arr.ndim == 1 and arr.shape[0] >= 4:
            x1, y1, x2, y2 = [int(v) for v in arr[:4]]
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        return []

    @staticmethod
    def _extract_entries(node: Any, out: List[Dict]) -> None:
        if node is None:
            return

        if isinstance(node, dict):
            text = ""
            for key in ("text", "generated_text", "transcription", "content", "label"):
                value = node.get(key)
                if isinstance(value, str) and value.strip():
                    text = value.strip()
                    break

            confidence = 0.0
            for key in ("score", "confidence", "prob", "conf"):
                value = node.get(key)
                if isinstance(value, (int, float)):
                    confidence = float(value)
                    break

            bbox = None
            for key in ("bbox", "box", "polygon", "poly", "points"):
                if key in node:
                    bbox = node.get(key)
                    break

            if text:
                out.append({
                    "text": text,
                    "conf": confidence,
                    "bbox": PaddleOCRVLV15Backend._normalize_box(bbox),
                })

            for value in node.values():
                PaddleOCRVLV15Backend._extract_entries(value, out)
            return

        if isinstance(node, (list, tuple)):
            for item in node:
                PaddleOCRVLV15Backend._extract_entries(item, out)

    def _has_enough_vram_for_vl15(self) -> bool:
        if not torch.cuda.is_available():
            return False
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024 ** 3)
            return free_gb > float(getattr(config.ocr, "vl15_min_free_vram_gb", 4.0))
        except Exception:
            return True

    def load(self, device: str) -> None:
        from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModel

        if not bool(getattr(config.ocr, "use_vl15", True)):
            raise RuntimeError("WEBTOON_USE_VL15=false")

        use_cuda = (device == "cuda") and self._has_enough_vram_for_vl15()
        self.device = "cuda" if use_cuda else "cpu"

        model_name = "PaddlePaddle/PaddleOCR-VL"
        cache_dir = str(config.TRANSLATION_CACHE_DIR)

        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
        model_kwargs = {
            "cache_dir": cache_dir,
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }

        try:
            self.model = AutoModelForVision2Seq.from_pretrained(model_name, **model_kwargs)
        except Exception:
            self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.model = self.model.to(self.device)
        self.model.eval()

    def _predict_one(self, img: np.ndarray) -> Tuple[str, float, List[Dict]]:
        if self.model is None or self.processor is None:
            return "", 0.0, []

        if img.ndim == 3 and img.shape[2] == 3:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            rgb = img

        image = Image.fromarray(rgb)

        try:
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                generated = self.model.generate(**inputs, max_new_tokens=1024)

            decoded = self.processor.batch_decode(generated, skip_special_tokens=True)
            parsed = []
            self._extract_entries(decoded, parsed)

            if not parsed:
                text = str(decoded[0]).strip() if decoded else ""
                return text, (0.80 if text else 0.0), []

            texts = [x["text"] for x in parsed if x.get("text")]
            confs = [float(x.get("conf", 0.8)) for x in parsed if x.get("text")]
            regions = [{"bbox": x.get("bbox", []), "text": x.get("text", ""), "conf": float(x.get("conf", 0.8))} for x in parsed if x.get("text")]

            if not texts:
                return "", 0.0, []

            return " ".join(texts), float(sum(confs) / max(1, len(confs))), regions
        except Exception:
            return "", 0.0, []

    def read_text(self, img: np.ndarray) -> Tuple[str, float, List[Dict]]:
        return self._predict_one(img)

    def read_batch(self, images: List[np.ndarray]) -> List[Tuple[str, float, List[Dict]]]:
        return [self._predict_one(img) for img in images]

    def unload(self) -> None:
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
