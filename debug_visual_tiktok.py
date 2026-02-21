"""
Script de debug visuel (format démo TikTok).

Génère 5 images étape par étape dans output_tiktok/<image_name>/ :
1) 01_original.png
2) 02_detection.png
3) 03_masks.png (masque combiné blanc sur fond noir)
4) 04_inpainted.png
5) 05_final.png
"""

import argparse
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from config import BASE_DIR, INPUT_DIR, config, LOGS_DIR
from core import Detection, NLLBTranslator, TextRenderer, YOLODetector
from pipeline import TranslationPipeline
from utils import init_logger, model_context


def _class_color_bgr(class_name: str) -> tuple:
    name = str(class_name or "").lower()
    if name in {"bulle", "bubble", "box", "small_text", "continuation"}:
        return (0, 255, 0)  # vert
    if name in {"out_text", "outer_text", "system"}:
        return (255, 0, 0)  # bleu
    if name in {"sfx"}:
        return (0, 0, 255)  # rouge
    return (255, 255, 255)


class TikTokVisualDebugger:
    def __init__(self):
        log_file = LOGS_DIR / "debug_visual_tiktok.log"
        self.logger = init_logger(log_file=log_file, level="INFO")
        self.pipeline = TranslationPipeline(self.logger, debug=False, lazy_models=True)

    def _save_detection_overlay(self, img: np.ndarray, detections: List[Detection], output_path: Path) -> None:
        canvas = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.55, min(canvas.shape[1] / 1200.0, 1.0))

        for idx, det in enumerate(detections, start=1):
            color = _class_color_bgr(det.class_name)
            cv2.rectangle(canvas, (det.x1, det.y1), (det.x2, det.y2), color, 3)
            label = f"#{idx} {det.class_name} {det.score:.0%}"
            (lw, lh), _ = cv2.getTextSize(label, font, scale, 2)
            y_top = max(0, det.y1 - lh - 10)
            cv2.rectangle(canvas, (det.x1, y_top), (det.x1 + lw + 10, det.y1), color, -1)
            cv2.putText(canvas, label, (det.x1 + 5, det.y1 - 6), font, scale, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imwrite(str(output_path), canvas)

    def _run_detection(self, img: np.ndarray) -> Dict[str, List[Detection]]:
        h, w = img.shape[:2]
        use_black_padding = getattr(config.detection, "use_black_padding", False)
        pad_ratio = float(getattr(config.detection, "black_padding_ratio", 0.03))
        pad_h = int(h * max(0.0, pad_ratio)) if use_black_padding else 0

        if pad_h > 0:
            black_bar_top = np.zeros((pad_h, w, 3), dtype=np.uint8)
            black_bar_bot = np.zeros((pad_h, w, 3), dtype=np.uint8)
            img_padded = np.vstack([black_bar_top, img, black_bar_bot])
        else:
            img_padded = img

        with model_context(lambda: YOLODetector(config.YOLO_MODEL_PATH, self.pipeline.device)) as detector:
            max_h = int(getattr(config.detection, "max_height", 0) or 0)
            detection_img = img_padded
            detection_scale = 1.0

            if max_h > 0 and img_padded.shape[0] > max_h:
                detection_scale = img_padded.shape[0] / float(max_h)
                resized_w = max(1, int(img_padded.shape[1] / detection_scale))
                detection_img = cv2.resize(img_padded, (resized_w, max_h), interpolation=cv2.INTER_AREA)

            detections = detector.detect(detection_img, logger=self.logger)

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

        translatable = [
            d for d in translatable
            if str(getattr(d, "class_name", "")).lower() != "sfx"
        ]
        translatable = self.pipeline._sort_detections_reading_order(translatable)

        return {
            "all": detections,
            "translatable": translatable,
        }

    def _run_ocr_and_masks(self, img: np.ndarray, detections: List[Detection]) -> List[Detection]:
        if not detections:
            return []

        if not self.pipeline._ensure_ocr_engine():
            self.logger.error("OCR engine non initialisé")
            return []

        if not self.pipeline._ensure_segmenter():
            self.logger.warning("Segmenter indisponible, fallback text_regions")

        crops: List[np.ndarray] = []
        crop_indices: List[int] = []

        for i, det in enumerate(detections):
            crop = img[det.y1:det.y2, det.x1:det.x2]
            if crop.size == 0:
                continue
            crops.append(crop)
            crop_indices.append(i)

        if not crops:
            return []

        batch_results = self.pipeline.ocr_engine.extract_batch(crops)

        for idx, ocr_result in zip(crop_indices, batch_results):
            det = detections[idx]
            text, confidence, is_valid, _skip_reason, text_regions, upscale_factor = ocr_result
            det.ocr_upscale_factor = upscale_factor
            det.ocr_confidence = confidence

            if not is_valid or not text:
                continue

            det.text_original = text
            det.text_regions = text_regions or []

            if self.pipeline.segmenter:
                det.mask_regions = self.pipeline.segmenter.segment_detection(img, det, det.text_regions)
            else:
                det.mask_regions = det.text_regions

        return [d for d in detections if d.text_original]

    def _translate(self, detections: List[Detection]) -> None:
        if not detections:
            return

        with model_context(lambda: NLLBTranslator(self.pipeline.device)) as translator:
            system_detections = [d for d in detections if str(getattr(d, "class_name", "")).lower() == "system"]
            regular_detections = [d for d in detections if str(getattr(d, "class_name", "")).lower() != "system"]

            if regular_detections:
                payload_texts = [d.text_original for d in regular_detections]
                translations_map = translator.translate_page_json(payload_texts)
                map_ok = isinstance(translations_map, dict) and all(
                    (str(i) in translations_map or i in translations_map)
                    for i in range(len(payload_texts))
                )
                if not map_ok:
                    translations_map = {str(i): translator.translate(txt) for i, txt in enumerate(payload_texts)}

                for det_idx, det in enumerate(regular_detections):
                    det.text_translated = (
                        translations_map.get(str(det_idx))
                        or translations_map.get(det_idx)
                        or det.text_original
                    )

            for det in system_detections:
                det.text_translated = translator.translate(det.text_original or "")

    def _build_combined_mask(self, img_shape: tuple, renderer: TextRenderer, detections: List[Detection]) -> np.ndarray:
        h_img, w_img = img_shape[:2]
        combined = np.zeros((h_img, w_img), dtype=np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

        for det in detections:
            regions = getattr(det, "mask_regions", None) or getattr(det, "text_regions", None)
            if not regions:
                continue

            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
            bubble_height = y2 - y1
            if bubble_height < getattr(renderer, "INPAINT_MIN_HEIGHT", 100):
                continue

            m = int(getattr(renderer, "CROP_MARGIN", 30))
            crop_x1 = max(0, x1 - m)
            crop_y1 = max(0, y1 - m)
            crop_x2 = min(w_img, x2 + m)
            crop_y2 = min(h_img, y2 + m)

            crop_h = crop_y2 - crop_y1
            crop_w = crop_x2 - crop_x1
            if crop_h <= 0 or crop_w <= 0:
                continue

            local_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)

            for region in regions:
                bbox_points = region.get("bbox") if isinstance(region, dict) else None
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
                    hull = cv2.convexHull(pts)
                    cv2.fillPoly(local_mask, [hull], 255)

            local_mask = cv2.dilate(local_mask, kernel, iterations=1)

            if np.sum(local_mask) == 0:
                continue

            roi = combined[crop_y1:crop_y2, crop_x1:crop_x2]
            combined[crop_y1:crop_y2, crop_x1:crop_x2] = np.maximum(roi, local_mask)

        return combined

    def run(self, image_path: Path, output_dir: Path) -> Dict[str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = output_dir / image_path.stem
        image_dir.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"Impossible de charger l'image: {image_path}")

        # 1) 01_original.png
        original_path = image_dir / "01_original.png"
        cv2.imwrite(str(original_path), img)

        # 2) 02_detection.png
        det_results = self._run_detection(img)
        all_detections = det_results["all"]
        translatable = det_results["translatable"]
        detection_path = image_dir / "02_detection.png"
        self._save_detection_overlay(img, all_detections, detection_path)

        if not translatable:
            empty_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            masks_path = image_dir / "03_masks.png"
            cv2.imwrite(str(masks_path), cv2.cvtColor(empty_mask, cv2.COLOR_GRAY2BGR))

            inpainted_path = image_dir / "04_inpainted.png"
            final_path = image_dir / "05_final.png"
            cv2.imwrite(str(inpainted_path), img)
            cv2.imwrite(str(final_path), img)
            return {
                "original": str(original_path),
                "detection": str(detection_path),
                "masks": str(masks_path),
                "inpainted": str(inpainted_path),
                "final": str(final_path),
            }

        valid = self._run_ocr_and_masks(img, translatable)
        self._translate(valid)

        renderer = TextRenderer()

        # 3) 03_masks.png (masque combiné global)
        combined_mask = self._build_combined_mask(img.shape, renderer, valid)
        masks_path = image_dir / "03_masks.png"
        cv2.imwrite(str(masks_path), cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR))

        # 4) 04_inpainted.png
        img_inpainted = img.copy()
        for det in valid:
            effective_regions = getattr(det, "mask_regions", None) or getattr(det, "text_regions", None)
            img_inpainted = renderer.inpaint_region(
                img_inpainted,
                det.x1,
                det.y1,
                det.x2,
                det.y2,
                text_regions=effective_regions,
            )
        inpainted_path = image_dir / "04_inpainted.png"
        cv2.imwrite(str(inpainted_path), img_inpainted)

        # 5) 05_final.png
        img_final = img_inpainted.copy()
        for det in valid:
            if not getattr(det, "text_translated", None):
                continue

            det.text_style = renderer.infer_text_style(
                det.text_translated,
                det.x2 - det.x1,
                det.y2 - det.y1,
                class_name=det.class_name,
            )
            det.text_color_rgb = renderer.extract_original_text_color(
                img,
                det.x1,
                det.y1,
                det.x2,
                det.y2,
                getattr(det, "mask_regions", None) or getattr(det, "text_regions", None),
            )
            det.font_hint = renderer.detect_font_hint(
                img,
                det.x1,
                det.y1,
                det.x2,
                det.y2,
                getattr(det, "mask_regions", None) or getattr(det, "text_regions", None),
            )

            img_final = renderer.insert_text(
                img_final,
                det.text_translated,
                det.x1,
                det.y1,
                det.x2,
                det.y2,
                text_regions=getattr(det, "mask_regions", None) or getattr(det, "text_regions", None),
                text_color_rgb=getattr(det, "text_color_rgb", None),
                text_style=getattr(det, "text_style", "dialogue"),
                font_hint=getattr(det, "font_hint", "regular"),
                class_name=getattr(det, "class_name", ""),
            )

        final_path = image_dir / "05_final.png"
        cv2.imwrite(str(final_path), img_final)

        self.pipeline._release_ocr_engine()
        self.pipeline._release_segmenter()

        return {
            "original": str(original_path),
            "detection": str(detection_path),
            "masks": str(masks_path),
            "inpainted": str(inpainted_path),
            "final": str(final_path),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug visuel TikTok (5 images étapes)")
    parser.add_argument("--image", type=Path, help="Image unique à traiter (optionnel)")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=INPUT_DIR,
        help="Dossier d'images à traiter en batch (défaut: input/)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=BASE_DIR / "output_tiktok",
        help="Dossier racine de sortie (défaut: output_tiktok/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    debugger = TikTokVisualDebugger()

    if args.image is not None:
        if not args.image.exists():
            raise FileNotFoundError(f"Image introuvable: {args.image}")

        outputs = debugger.run(args.image, args.output)
        print("\n✅ Debug visuel généré:")
        print(f"- original:  {outputs['original']}")
        print(f"- detection: {outputs['detection']}")
        print(f"- masks:     {outputs['masks']}")
        print(f"- inpainted: {outputs['inpainted']}")
        print(f"- final:     {outputs['final']}")
        return

    input_dir = args.input
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Dossier input introuvable: {input_dir}")

    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    image_files = sorted(
        [
            f for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
    )

    if not image_files:
        raise RuntimeError(f"Aucune image trouvée dans: {input_dir}")

    print(f"\n🚀 Batch TikTok debug: {len(image_files)} image(s) depuis {input_dir}")
    success = 0
    failed = 0

    for i, image_path in enumerate(image_files, start=1):
        print(f"\n[{i}/{len(image_files)}] {image_path.name}")
        try:
            outputs = debugger.run(image_path, args.output)
            print(f"   ✅ {outputs['final']}")
            success += 1
        except Exception as exc:
            print(f"   ❌ Erreur: {exc}")
            failed += 1

    print("\n📦 Résumé batch")
    print(f"- Succès: {success}")
    print(f"- Échecs: {failed}")
    print(f"- Dossier: {args.output}")


if __name__ == "__main__":
    main()
