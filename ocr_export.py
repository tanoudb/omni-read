#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
OCR EXPORTER — Extrait les textes OCR et les exporte en JSON pour Gemini

Lance le pipeline YOLO + OCR sans traduction, et sauvegarde les résultats
en fichiers JSON compatibles avec gemini_watcher.py.

Usage :
    python ocr_export.py                      # input/ → ocr_input/
    python ocr_export.py --input manhwa/ch01/  # dossier custom
    python ocr_export.py --image page01.jpg    # image unique
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import sys
import time
from pathlib import Path

from config import config, INPUT_DIR, OUTPUT_DIR, LOGS_DIR
from utils import init_logger, MemoryManager, model_context
from core import YOLODetector, OCREngine, Detection, SmartSegmenter


def export_ocr(image_path: Path, output_dir: Path, logger, debug: bool = False) -> dict:
    """Détection + OCR sur une image, retourne les résultats."""
    import cv2
    import numpy as np

    logger.info(f"\n📖 {image_path.name}")

    img = cv2.imread(str(image_path))
    if img is None:
        logger.error(f"   ❌ Impossible de charger {image_path}")
        return {"error": str(image_path)}

    h, w = img.shape[:2]
    device = MemoryManager.get_device()

    # Detection
    detector = YOLODetector(logger)
    detections = detector.detect(img, str(image_path))
    detector.release()
    logger.info(f"   🔍 {len(detections)} détections")

    # OCR
    ocr = OCREngine(device=device, logger=logger)
    results = []

    for idx, det in enumerate(detections):
        crop = img[det.y1:det.y2, det.x1:det.x2]
        if crop.size == 0:
            continue

        text, conf, regions = ocr.read_text(crop)
        if not text or text == "(none)":
            continue

        results.append({
            "id": f"{idx:03d}",
            "text": text,
            "confidence": round(conf, 3),
            "bbox": [det.x1, det.y1, det.x2, det.y2],
            "class": getattr(det, "class_name", "bulle"),
            "size": f"{det.x2-det.x1}x{det.y2-det.y1}",
        })

    ocr.unload()

    # Sauvegarder
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{image_path.stem}.json"

    data = {
        "image": image_path.name,
        "width": w,
        "height": h,
        "detections": results,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"   ✅ {len(results)} textes → {out_file}")
    return data


def main():
    parser = argparse.ArgumentParser(description="OCR Exporter pour Gemini Watcher")
    parser.add_argument("--input", "-i", type=Path, default=INPUT_DIR)
    parser.add_argument("--output", "-o", type=Path, default=Path("ocr_input"))
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logger = init_logger(level="INFO")
    logger.header("OCR EXPORTER → Gemini")

    if args.image:
        export_ocr(args.image, args.output, logger, args.debug)
    else:
        images = sorted(
            f for f in args.input.iterdir()
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        logger.info(f"📁 {len(images)} images dans {args.input}")
        for img_path in images:
            export_ocr(img_path, args.output, logger, args.debug)

    logger.header("✅ Export terminé")


if __name__ == "__main__":
    main()
