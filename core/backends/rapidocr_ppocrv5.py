"""
Backend RapidOCR (PP-OCRv5) via ONNX Runtime CUDA.
"""

from __future__ import annotations

from typing import Tuple, List, Dict, Any

import cv2
import numpy as np

from .base import OCRBackend


class RapidOCRPPOCRv5Backend(OCRBackend):
    def __init__(self):
        self.engine = None
        self.cuda_enforced = False
        self.cuda_force_error = ""

    @property
    def name(self) -> str:
        return "RapidOCR-PP-OCRv5"

    def load(self, device: str) -> None:
        from rapidocr import RapidOCR, EngineType, OCRVersion

        params = {
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Cls.engine_type": EngineType.ONNXRUNTIME,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
        }
        self.engine = RapidOCR(params=params)
        self._force_cuda_provider()

    @staticmethod
    def _collect_onnx_sessions(root: Any) -> List[Any]:
        sessions: List[Any] = []
        visited = set()

        def _walk(obj: Any, depth: int = 0):
            if obj is None or depth > 6:
                return
            obj_id = id(obj)
            if obj_id in visited:
                return
            visited.add(obj_id)

            if hasattr(obj, "set_providers") and hasattr(obj, "get_providers"):
                sessions.append(obj)

            if hasattr(obj, "__dict__"):
                for _, value in obj.__dict__.items():
                    if isinstance(value, (str, int, float, bool, bytes)):
                        continue
                    _walk(value, depth + 1)

        _walk(root)
        return sessions

    def _force_cuda_provider(self):
        self.cuda_enforced = False
        self.cuda_force_error = ""
        if self.engine is None:
            return

        try:
            sessions = self._collect_onnx_sessions(self.engine)
            if not sessions:
                self.cuda_force_error = "no_onnx_sessions_found"
                return

            for sess in sessions:
                try:
                    sess.set_providers([
                        ("CUDAExecutionProvider", {"device_id": 0}),
                        "CPUExecutionProvider",
                    ])
                except Exception:
                    sess.set_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])

            self.cuda_enforced = True
        except Exception as exc:
            self.cuda_force_error = str(exc)

    def get_runtime_info(self) -> Dict:
        info: Dict[str, Any] = {
            "engine": "onnxruntime",
            "onnx_cuda_requested": True,
            "onnx_cuda_enforced": self.cuda_enforced,
        }
        if self.cuda_force_error:
            info["onnx_cuda_force_error"] = self.cuda_force_error

        try:
            import onnxruntime as ort
            info["onnx_available_providers"] = ort.get_available_providers()
        except Exception as exc:
            info["onnxruntime_error"] = str(exc)

        providers_found: List[str] = []

        for sess in self._collect_onnx_sessions(self.engine):
            try:
                p = sess.get_providers()
                if isinstance(p, list):
                    providers_found.extend([str(x) for x in p])
            except Exception:
                pass

        if providers_found:
            uniq = []
            seen = set()
            for p in providers_found:
                if p not in seen:
                    seen.add(p)
                    uniq.append(p)
            info["onnx_session_providers"] = uniq
            info["onnx_cuda_active"] = any("CUDAExecutionProvider" in p for p in uniq)
        else:
            info["onnx_session_providers"] = []
            info["onnx_cuda_active"] = False

        return info

    @staticmethod
    def _normalize_points(raw: Any) -> List[List[int]]:
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

    def _parse_result(self, result_obj: Any) -> Tuple[str, float, List[Dict]]:
        if not result_obj:
            return "", 0.0, []

        # RapidOCR >=3.x returns RapidOCROutput dataclass-like object
        if hasattr(result_obj, "txts") and hasattr(result_obj, "scores"):
            txts = list(getattr(result_obj, "txts", []) or [])
            scores = list(getattr(result_obj, "scores", []) or [])
            boxes = getattr(result_obj, "boxes", None)

            texts: List[str] = []
            confs: List[float] = []
            regions: List[Dict] = []

            for idx, text_raw in enumerate(txts):
                text = str(text_raw or "").strip()
                if not text:
                    continue
                conf = float(scores[idx]) if idx < len(scores) else 0.0

                pts: List[List[int]] = []
                if boxes is not None:
                    try:
                        pts = self._normalize_points(boxes[idx])
                    except Exception:
                        pts = []

                texts.append(text)
                confs.append(conf)
                regions.append({"bbox": pts, "text": text, "conf": conf})

            if texts:
                return " ".join(texts), float(sum(confs) / max(1, len(confs))), regions
            return "", 0.0, []

        lines = result_obj
        if isinstance(result_obj, tuple) and len(result_obj) >= 1:
            lines = result_obj[0]

        if not isinstance(lines, (list, tuple)):
            return "", 0.0, []

        texts: List[str] = []
        confs: List[float] = []
        regions: List[Dict] = []

        for item in lines:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue

            pts = self._normalize_points(item[0])

            text = ""
            conf = 0.0

            if isinstance(item[1], (list, tuple)) and len(item[1]) >= 1:
                text = str(item[1][0] or "").strip()
                if len(item[1]) > 1:
                    try:
                        conf = float(item[1][1])
                    except Exception:
                        conf = 0.0
            else:
                text = str(item[1] or "").strip()
                if len(item) > 2:
                    try:
                        conf = float(item[2])
                    except Exception:
                        conf = 0.0

            if not text:
                continue

            texts.append(text)
            confs.append(conf)
            regions.append({"bbox": pts, "text": text, "conf": conf})

        if not texts:
            return "", 0.0, []

        return " ".join(texts), float(sum(confs) / max(1, len(confs))), regions

    def read_text(self, img: np.ndarray) -> Tuple[str, float, List[Dict]]:
        if self.engine is None:
            return "", 0.0, []

        if img.ndim == 3 and img.shape[2] == 3:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            rgb = img

        try:
            output = self.engine(rgb)
            return self._parse_result(output)
        except Exception:
            return "", 0.0, []

    def read_batch(self, images: List[np.ndarray]) -> List[Tuple[str, float, List[Dict]]]:
        if self.engine is None:
            return [("", 0.0, []) for _ in images]

        results: List[Tuple[str, float, List[Dict]]] = []
        for img in images:
            results.append(self.read_text(img))
        return results

    def unload(self) -> None:
        self.engine = None
