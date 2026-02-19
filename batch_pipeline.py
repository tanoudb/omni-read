from __future__ import annotations

from dataclasses import dataclass
import gc
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional
import time

import cv2
import numpy as np
import torch

from config import config
from core import Detection, NLLBTranslator, OCREngine, SmartSegmenter, TextRenderer, YOLODetector
from pipeline import TranslationPipeline
from utils import MemoryManager, WebtoonLogger, model_context


@dataclass
class ImageResult:
    image_path: Path
    output_path: Path
    image: np.ndarray
    detections: list
    success: bool = True
    error: str = ""


class BatchPipeline(TranslationPipeline):
    def __init__(self, logger: WebtoonLogger, debug: bool = False):
        super().__init__(logger, debug=debug, lazy_models=True)
        self.debug = debug
        self._glossaire: list[str] = []

    def set_glossaire(self, termes: list[str]):
        self._glossaire = [t.strip() for t in termes if str(t).strip()]

    @staticmethod
    def _cleanup_cuda():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _to_detail(self, result: ImageResult) -> Dict:
        return {
            "image": result.image_path.name,
            "success": bool(result.success),
            "error": str(result.error or ""),
            "translated": int(sum(1 for d in result.detections if getattr(d, "text_translated", None))),
            "detections": int(len(result.detections)),
        }

    def _build_results(self, image_paths: List[Path], output_dir: Path) -> List[ImageResult]:
        results: List[ImageResult] = []
        for image_path in image_paths:
            output_path = output_dir / f"{image_path.stem}_translated.png"
            img = cv2.imread(str(image_path))
            if img is None:
                results.append(
                    ImageResult(
                        image_path=image_path,
                        output_path=output_path,
                        image=np.zeros((1, 1, 3), dtype=np.uint8),
                        detections=[],
                        success=False,
                        error="load_failed",
                    )
                )
            else:
                results.append(
                    ImageResult(
                        image_path=image_path,
                        output_path=output_path,
                        image=img,
                        detections=[],
                        success=True,
                        error="",
                    )
                )
        return results

    def process_chapter(
        self,
        image_paths: list[Path],
        output_dir: Path,
        progress_callback=None,
    ) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        image_paths = [Path(p) for p in image_paths]
        results = self._build_results(image_paths, output_dir)

        if not results:
            return {"success": 0, "errors": 0, "details": []}

        # PHASE 1: YOLO
        self.logger.phase("Detection (batch)", 1, 4)
        with model_context(lambda: YOLODetector(config.YOLO_MODEL_PATH, self.device)) as detector:
            for result in results:
                if not result.success:
                    continue
                try:
                    img = result.image
                    h, w = img.shape[:2]

                    use_black_padding = bool(
                        getattr(config.detection, "black_bars_enabled", getattr(config.detection, "use_black_padding", False))
                    )
                    pad_h = int(h * max(0.0, float(getattr(config.detection, "black_padding_ratio", 0.03)))) if use_black_padding else 0

                    if pad_h > 0:
                        black_bar_top = np.zeros((pad_h, w, 3), dtype=np.uint8)
                        black_bar_bot = np.zeros((pad_h, w, 3), dtype=np.uint8)
                        img_padded = np.vstack([black_bar_top, img, black_bar_bot])
                    else:
                        img_padded = img

                    max_h = int(getattr(config.detection, "max_height", 0) or 0)
                    detection_img = img_padded
                    detection_scale = 1.0

                    if max_h > 0 and img_padded.shape[0] > max_h:
                        detection_scale = img_padded.shape[0] / float(max_h)
                        resized_w = max(1, int(img_padded.shape[1] / detection_scale))
                        detection_img = cv2.resize(img_padded, (resized_w, max_h), interpolation=cv2.INTER_AREA)

                    detections = detector.detect(detection_img, logger=self.logger)
                    yolo_report = detector.get_last_debug_report()

                    if detection_scale != 1.0:
                        for det in detections:
                            det.bbox = [
                                float(det.bbox[0] * detection_scale),
                                float(det.bbox[1] * detection_scale),
                                float(det.bbox[2] * detection_scale),
                                float(det.bbox[3] * detection_scale),
                            ]

                    if pad_h > 0:
                        for det in detections:
                            new_y1 = max(0, int(det.bbox[1]) - pad_h)
                            new_y2 = min(h, int(det.bbox[3]) - pad_h)
                            det.bbox = [det.bbox[0], new_y1, det.bbox[2], new_y2]

                    detections = [d for d in detections if d.y2 > 0 and d.y1 < h]
                    translatable = detector.get_translatable_detections(detections)
                    translatable = [d for d in translatable if str(getattr(d, "class_name", "")).lower() != "sfx"]
                    translatable = self._sort_detections_reading_order(translatable)
                    result.detections = translatable

                    if self.debug:
                        image_stem = result.image_path.stem
                        self.save_debug_detections(img, detections, translatable, output_dir, image_stem)
                        self.save_debug_yolo_rejected(output_dir, image_stem, yolo_report if isinstance(yolo_report, dict) else {})

                except Exception as exc:
                    result.success = False
                    result.error = f"yolo_failed: {exc}"

        self._cleanup_cuda()

        # PHASE 2: OCR
        self.logger.phase("OCR (batch)", 2, 4)
        try:
            with model_context(lambda: OCREngine(device=self.device)) as ocr_engine, model_context(
                lambda: SmartSegmenter(logger=self.logger)
            ) as segmenter:
                crops: List[np.ndarray] = []
                crop_map: List[tuple[int, int]] = []

                for img_idx, result in enumerate(results):
                    if not result.success:
                        continue
                    for det_idx, det in enumerate(result.detections):
                        crop = result.image[det.y1:det.y2, det.x1:det.x2]
                        if crop.size == 0:
                            continue
                        crops.append(crop)
                        crop_map.append((img_idx, det_idx))

                def _ocr_debug_log(message: str):
                    self.logger.info(f"      {message}")

                batch_results = ocr_engine.extract_batch(crops, debug_hook=_ocr_debug_log) if crops else []

                for (img_idx, det_idx), ocr_result in zip(crop_map, batch_results):
                    result = results[img_idx]
                    if not result.success:
                        continue
                    det = result.detections[det_idx]

                    text, confidence, is_valid, skip_reason, text_regions, upscale_factor = ocr_result
                    det.ocr_upscale_factor = upscale_factor
                    det.ocr_confidence = confidence

                    if not is_valid or not text:
                        continue

                    det.text_original = text
                    det.text_regions = text_regions or []
                    det.ocr_lines = self._extract_ocr_lines_from_regions(det.text_regions)

                    if segmenter is not None:
                        det.mask_regions = segmenter.segment_detection(result.image, det, det.text_regions)
                    else:
                        det.mask_regions = det.text_regions

                    if self.debug:
                        self.save_debug_mask_bundle(result.image, output_dir, result.image_path.stem, det_idx + 1, det, det.mask_regions)

                if self.debug:
                    for result in results:
                        if result.success:
                            self.save_debug_ocr(output_dir, result.image_path.stem, result.detections)
        except Exception as exc:
            for result in results:
                if result.success:
                    result.success = False
                    result.error = f"ocr_failed: {exc}"

        self._cleanup_cuda()

        # PHASE 3: TRANSLATION
        self.logger.phase("Translation (batch)", 3, 4)
        try:
            with model_context(lambda: NLLBTranslator(self.device)) as translator:
                if hasattr(translator, "set_glossaire"):
                    translator.set_glossaire(self._glossaire)

                all_texts: List[str] = []
                text_map: List[tuple[int, int]] = []

                for img_idx, result in enumerate(results):
                    if not result.success:
                        continue
                    for det_idx, det in enumerate(result.detections):
                        text = (det.text_original or "").strip()
                        if not text:
                            continue
                        all_texts.append(text)
                        text_map.append((img_idx, det_idx))

                if all_texts:
                    try:
                        translations_map = translator.translate_page_json(all_texts)
                        map_ok = isinstance(translations_map, dict) and all(
                            (str(i) in translations_map or i in translations_map)
                            for i in range(len(all_texts))
                        )
                        if not map_ok:
                            raise RuntimeError("JSON LLM non indexé correctement")

                        for i, (img_idx, det_idx) in enumerate(text_map):
                            result = results[img_idx]
                            if not result.success:
                                continue
                            det = result.detections[det_idx]
                            det.text_translated = (
                                translations_map.get(str(i))
                                or translations_map.get(i)
                                or det.text_original
                            )
                    except Exception as global_exc:
                        self.logger.warning(f"Fallback traduction image par image: {global_exc}")
                        for result in results:
                            if not result.success:
                                continue
                            local_dets = [d for d in result.detections if (d.text_original or "").strip()]
                            if not local_dets:
                                continue
                            payload_texts = [d.text_original for d in local_dets]
                            local_map = translator.translate_page_json(payload_texts)
                            for idx, det in enumerate(local_dets):
                                det.text_translated = (
                                    local_map.get(str(idx))
                                    or local_map.get(idx)
                                    or det.text_original
                                )
        except Exception as exc:
            for result in results:
                if result.success:
                    result.success = False
                    result.error = f"translation_failed: {exc}"

        self._cleanup_cuda()

        # PHASE 4: RENDERING
        self.logger.phase("Rendering (batch)", 4, 4)
        total = len(results)
        with model_context(lambda: TextRenderer()) as renderer:
            for idx, result in enumerate(results, start=1):
                if not result.success:
                    if callable(progress_callback):
                        progress_callback(idx, total, False)
                    continue

                try:
                    img_translated = result.image.copy()
                    valid_detections = [d for d in result.detections if getattr(d, "text_translated", None)]

                    if self.debug:
                        self.save_debug_double_page_ocr(result.image, output_dir, result.image_path.stem, [d for d in result.detections if getattr(d, "text_original", None)])

                    for det_idx, det in enumerate(valid_detections, start=1):
                        det.text_style = renderer.infer_text_style(
                            det.text_translated,
                            det.x2 - det.x1,
                            det.y2 - det.y1,
                            class_name=det.class_name,
                        )
                        det.text_color_rgb = renderer.extract_original_text_color(
                            result.image,
                            det.x1,
                            det.y1,
                            det.x2,
                            det.y2,
                            getattr(det, "mask_regions", None) or getattr(det, "text_regions", None),
                        )
                        det.font_hint = renderer.detect_font_hint(
                            result.image,
                            det.x1,
                            det.y1,
                            det.x2,
                            det.y2,
                            getattr(det, "mask_regions", None) or getattr(det, "text_regions", None),
                        )

                        before_crop = None
                        if self.debug:
                            before_crop = img_translated[det.y1:det.y2, det.x1:det.x2].copy()

                        img_translated, _, _ = renderer.render_text_with_timing(
                            img_translated,
                            det.text_translated,
                            det.x1,
                            det.y1,
                            det.x2,
                            det.y2,
                            text_regions=getattr(det, "text_regions", None),
                            mask_regions=getattr(det, "mask_regions", None),
                            text_color_rgb=getattr(det, "text_color_rgb", None),
                            text_style=getattr(det, "text_style", "dialogue"),
                            font_hint=getattr(det, "font_hint", "regular"),
                            class_name=getattr(det, "class_name", ""),
                        )

                        if self.debug and before_crop is not None:
                            after_crop = img_translated[det.y1:det.y2, det.x1:det.x2].copy()
                            self.save_debug_render_bundle(output_dir, result.image_path.stem, det_idx, before_crop, after_crop, det)

                    cv2.imwrite(str(result.output_path), img_translated)

                    metadata_path = output_dir / f"{result.image_path.stem}_metadata.json"
                    metadata = {
                        "source": str(result.image_path),
                        "output": str(result.output_path),
                        "dimensions": {"width": int(result.image.shape[1]), "height": int(result.image.shape[0])},
                        "detections": [
                            {
                                "class": d.class_name,
                                "bbox": d.bbox,
                                "original": d.text_original,
                                "translated": d.text_translated,
                                "confidence": d.ocr_confidence,
                                "detection_confidence": d.score,
                            }
                            for d in result.detections
                            if d.text_translated
                        ],
                    }
                    with open(metadata_path, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=2)

                    if callable(progress_callback):
                        progress_callback(idx, total, True)

                except Exception as exc:
                    result.success = False
                    result.error = f"render_failed: {exc}"
                    if callable(progress_callback):
                        progress_callback(idx, total, False)

        self._cleanup_cuda()

        details = [self._to_detail(r) for r in results]
        success_count = sum(1 for r in results if r.success)
        error_count = len(results) - success_count

        return {
            "success": success_count,
            "errors": error_count,
            "details": details,
        }
