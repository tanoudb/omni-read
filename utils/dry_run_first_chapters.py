# -*- coding: utf-8 -*-
"""
Dry-run (zero API, zero traduction) sur le PREMIER chapitre de chaque serie
non encore traduite dans manhwa_trad/. On reinjecte le texte OCR original
(anglais/source) a la place d'une traduction, pour isoler et valider la
detection + OCR + effacement + rendu.

Usage:
    python utils/dry_run_first_chapters.py
    python utils/dry_run_first_chapters.py --serie hellogin
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import cv2
import numpy as np
from config import config
from core import OCREngine, TextRenderer, SmartSegmenter, YOLODetector
from pipeline import TranslationPipeline
from utils import WebtoonLogger

TARGET_SERIES = [
    "hellogin",
    "i-married-the-dragon-i-killed",
    "path-of-vengeance",
    "the-frontier-count's-10th-class-outcas",
    "the-wind-mage",
]


def build_pipeline() -> TranslationPipeline:
    logger = WebtoonLogger("dry-run-first-chapters")
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


def run_dry_run_on_image(p, renderer, img_path, out_dir, bubble_debug_dir=None):
    report = {
        "image": img_path.name,
        "error": None,
        "detections": 0,
        "translatable": 0,
        "ocr_kept": 0,
        "ocr_skip_reasons": {},
        "no_chirurgical_mask": 0,
        "render_errors": [],
        "ghost_risk_bboxes": [],
    }

    img = cv2.imread(str(img_path))
    if img is None:
        report["error"] = "imread_failed"
        return report
    h, w = img.shape[:2]

    # Detection
    max_h = int(getattr(config.detection, "max_height", 0) or 0)
    det_img, scale = img, 1.0
    if max_h > 0 and h > max_h:
        scale = h / float(max_h)
        det_img = cv2.resize(img, (max(1, int(w / scale)), max_h), interpolation=cv2.INTER_AREA)
    dets = p._detect_ensemble(det_img)
    report["detections"] = len(dets)
    if scale != 1.0:
        for d in dets:
            d.bbox = [float(v * scale) for v in d.bbox]
    dets = [d for d in p.detector.get_translatable_detections(dets)
            if str(d.class_name).lower() != "sfx"]
    dets = p._sort_detections_reading_order(dets)
    report["translatable"] = len(dets)

    # OCR + masks
    crops, idxs = [], []
    for i, d in enumerate(dets):
        c = img[d.y1:d.y2, d.x1:d.x2]
        crops.append(c)
        idxs.append(i)

    results = p.ocr_engine.extract_batch(crops) if crops else []
    keep = []
    for i, res in zip(idxs, results):
        reason, _ = p._apply_ocr_result(img, dets[i], res)
        if not reason:
            keep.append(dets[i])
        else:
            report["ocr_skip_reasons"][reason] = report["ocr_skip_reasons"].get(reason, 0) + 1
    report["ocr_kept"] = len(keep)

    # Style / couleur / police -- calcules sur l'image ORIGINALE (avant
    # effacement), comme le fait la production (pipeline._prepare_render_style).
    # Reprendre le texte OCR comme "traduction" puisqu'on ne traduit pas ici.
    for d in keep:
        d.text_translated = d.text_original
        try:
            TranslationPipeline._prepare_render_style(renderer, img, d)
        except Exception as e:
            report["render_errors"].append({
                "bbox": [d.x1, d.y1, d.x2, d.y2],
                "text": str(d.text_original)[:60],
                "error": f"prepare_style {type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            })

    # Erase
    out, _, ghost_risk_indices = p._run_pre_inpainting(img, keep, renderer)
    for idx in ghost_risk_indices:
        if idx < len(keep):
            d = keep[idx]
            report["ghost_risk_bboxes"].append([d.x1, d.y1, d.x2, d.y2])

    # Render -- meme appel que la production (insert_text + sibling_boxes
    # pour eviter que deux bulles voisines se chevauchent au rendu).
    for d in keep:
        if getattr(d, 'chirurgical_mask', None) is None:
            report["no_chirurgical_mask"] += 1
        try:
            out = renderer.insert_text(
                out,
                d.text_translated,
                int(d.x1), int(d.y1), int(d.x2), int(d.y2),
                text_regions=getattr(d, 'text_regions', None),
                text_color_rgb=getattr(d, 'text_color_rgb', None),
                text_style=getattr(d, 'text_style', 'dialogue'),
                font_hint=getattr(d, 'font_hint', 'regular'),
                class_name=str(d.class_name),
                bubble_mask=getattr(d, 'mask_binary', None),
                source_line_height=getattr(d, 'source_line_height', None),
                sibling_boxes=[(o.x1, o.y1, o.x2, o.y2) for o in keep if o is not d],
            )
        except Exception as e:
            report["render_errors"].append({
                "bbox": [d.x1, d.y1, d.x2, d.y2],
                "text": str(d.text_original)[:60],
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            })

    # Write output
    out_path = out_dir / (img_path.stem + "_dryrun_translated.png")
    cv2.imwrite(str(out_path), out)
    report["output"] = str(out_path)

    # Debug : crop avant/apres par bulle, pour verifier effacement + placement
    # sans depiler des captures de 60000px. Marge de 15% autour de la bbox.
    if bubble_debug_dir is not None:
        bubble_debug_dir.mkdir(parents=True, exist_ok=True)
        for i, d in enumerate(keep):
            bw, bh = d.x2 - d.x1, d.y2 - d.y1
            mx, my = max(20, int(bw * 0.25)), max(20, int(bh * 0.25))
            x1, y1 = max(0, d.x1 - mx), max(0, d.y1 - my)
            x2, y2 = min(w, d.x2 + mx), min(out.shape[0], d.y2 + my)
            before = img[y1:y2, x1:x2]
            after = out[y1:y2, x1:x2]
            if before.size == 0 or after.size == 0:
                continue
            gap = np.full((before.shape[0], 8, 3), 255, dtype=np.uint8)
            side_by_side = np.hstack([before, gap, after])
            fname = f"{img_path.stem}_b{i:03d}_x{d.x1}_y{d.y1}.png"
            cv2.imwrite(str(bubble_debug_dir / fname), side_by_side)

    del img, det_img, dets, keep, out, crops, idxs
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return report


def first_chapter_dir(series_dir: Path) -> Path | None:
    chapter_dirs = sorted(d for d in series_dir.iterdir() if d.is_dir())
    return chapter_dirs[0] if chapter_dirs else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serie", type=str, default=None, help="Ne traiter qu'une seule serie")
    parser.add_argument("--bubble-debug", action="store_true", help="Dump avant/apres par bulle")
    args = parser.parse_args()

    manhwa_dir = Path("manhwa")
    out_base_dir = Path("tests/dry_run_out")

    series_list = TARGET_SERIES
    if args.serie:
        series_list = [s for s in TARGET_SERIES if s == args.serie]
        if not series_list:
            print(f"Serie '{args.serie}' pas dans TARGET_SERIES: {TARGET_SERIES}")
            sys.exit(1)

    p = build_pipeline()
    renderer = TextRenderer()

    full_report = {}

    for slug in series_list:
        serie_dir = manhwa_dir / slug
        if not serie_dir.exists():
            print(f"[SKIP] {slug}: introuvable dans manhwa/")
            continue

        ch_dir = first_chapter_dir(serie_dir)
        if ch_dir is None:
            print(f"[SKIP] {slug}: aucun chapitre")
            continue

        out_ch_dir = out_base_dir / slug / ch_dir.name
        out_ch_dir.mkdir(parents=True, exist_ok=True)

        images = [f for f in ch_dir.iterdir() if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]]

        print(f"\n=== {slug} / {ch_dir.name} ({len(images)} pages) ===", flush=True)
        series_report = []
        for img in sorted(images):
            print(f"  Processing {img.name}...", flush=True)
            t0 = time.time()
            bdir = (out_ch_dir / "bubble_debug") if args.bubble_debug else None
            try:
                r = run_dry_run_on_image(p, renderer, img, out_ch_dir, bubble_debug_dir=bdir)
            except Exception as e:
                r = {"image": img.name, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
                print(f"    !! FATAL: {r['error']}", flush=True)
            r["elapsed_s"] = round(time.time() - t0, 2)
            if r.get("error"):
                print(f"    !! error: {r['error']}", flush=True)
            elif r.get("render_errors"):
                print(f"    !! {len(r['render_errors'])} render_errors", flush=True)
            series_report.append(r)

        full_report[slug] = {"chapter": ch_dir.name, "pages": series_report}

        report_path = out_base_dir / slug / "report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(full_report[slug], f, ensure_ascii=False, indent=2)
        print(f"  -> report: {report_path}")

    combined_path = out_base_dir / "first_chapters_report.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)
    print(f"\n[DONE] Rapport combine: {combined_path}")


if __name__ == '__main__':
    main()
