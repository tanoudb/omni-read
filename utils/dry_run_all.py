import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import cv2
import numpy as np
from config import config
from core import OCREngine, TextRenderer, SmartSegmenter, YOLODetector
from pipeline import TranslationPipeline
from utils import WebtoonLogger

def build_pipeline() -> TranslationPipeline:
    logger = WebtoonLogger("dry-run-all")
    p = TranslationPipeline.__new__(TranslationPipeline)
    p.logger = logger
    p.debug = False
    p.device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    p.detector = YOLODetector(config.YOLO_MODEL_PATH, p.device)
    p.detector_secondary = None
    sec = getattr(config, "YOLO_MODEL_PATH_SECONDARY", None)
    if sec and Path(sec).exists():
        p.detector_secondary = YOLODetector(sec, p.device)
    p.segmenter = SmartSegmenter(logger=logger)
    p.ocr_engine = OCREngine(device=p.device)
    return p

def run_dry_run_on_image(p, renderer, img_path, out_dir):
    img = cv2.imread(str(img_path))
    if img is None:
        return
    h, w = img.shape[:2]
    
    # -- Détection --
    max_h = int(getattr(config.detection, "max_height", 0) or 0)
    det_img, scale = img, 1.0
    if max_h > 0 and h > max_h:
        scale = h / float(max_h)
        det_img = cv2.resize(img, (max(1, int(w / scale)), max_h), interpolation=cv2.INTER_AREA)
    dets = p._detect_ensemble(det_img)
    if scale != 1.0:
        for d in dets:
            d.bbox = [float(v * scale) for v in d.bbox]
    dets = [d for d in p.detector.get_translatable_detections(dets)
            if str(d.class_name).lower() != "sfx"]
    dets = p._sort_detections_reading_order(dets)

    # -- OCR + masques --
    for d in dets:
        c = img[d.y1:d.y2, d.x1:d.x2]
        d.text = p.ocr_engine.recognize(c)
    
    keep = []
    for d in dets:
        txt = (d.text or "").strip()
        if len(txt) <= 1: continue
        d.text = txt
        mask = p.segmenter.segment(img, d)
        if mask is not None and mask.any():
            d.mask = mask
            keep.append((None, d))
            
    # -- Effacement --
    out = img.copy()
    for _, d in keep:
        out = renderer.erase_text_from_image(out, [d])
        
    # -- Rendu --
    for _, d in keep:
        if d.mask is not None:
            out = renderer.render_text(out, [d], d.text)

    # Write output
    out_path = out_dir / (img_path.stem + "_dryrun_translated.png")
    cv2.imwrite(str(out_path), out)

def main():
    target_series = [
        "hellogin",
        "i-married-the-dragon-i-killed",
        "path-of-vengeance",
        "the-wind-mage"
    ]
    
    manhwa_dir = Path("manhwa")
    out_base_dir = Path("tests/dry_run_out")
    
    p = build_pipeline()
    renderer = TextRenderer()
    
    for slug in target_series:
        serie_dir = manhwa_dir / slug
        if not serie_dir.exists():
            continue
            
        out_serie_dir = out_base_dir / slug
        
        for ch_dir in sorted(serie_dir.iterdir()):
            if not ch_dir.is_dir():
                continue
                
            out_ch_dir = out_serie_dir / ch_dir.name
            out_ch_dir.mkdir(parents=True, exist_ok=True)
            
            images = [f for f in ch_dir.iterdir() if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]]
            
            for img in sorted(images):
                print(f"Processing {img}...")
                try:
                    run_dry_run_on_image(p, renderer, img, out_ch_dir)
                except Exception as e:
                    print(f"Error on {img}: {e}")

if __name__ == '__main__':
    main()
