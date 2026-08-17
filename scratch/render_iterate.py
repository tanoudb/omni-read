# -*- coding: utf-8 -*-
"""
Dry-run de rendu (traduction désactivée) avec sauvegarde des crops
avant/après par bulle, pour itérer visuellement sur l'effacement +
la réinjection de texte sans dépendre de la traduction.

Usage:
    python scratch/render_iterate.py <image.jpg> <out_dir> [--margin PX]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"A:\omni read")))

import cv2
import numpy as np
from config import config
from core import OCREngine, TextRenderer, SmartSegmenter, YOLODetector
from pipeline import TranslationPipeline
from utils import WebtoonLogger


def build_pipeline() -> TranslationPipeline:
    logger = WebtoonLogger("render-iterate")
    p = TranslationPipeline.__new__(TranslationPipeline)
    p.logger = logger
    p.debug = False
    import torch
    p.device = "cuda" if torch.cuda.is_available() else "cpu"
    p.detector = YOLODetector(config.YOLO_MODEL_PATH, p.device)
    p.detector_secondary = None
    sec = getattr(config, "YOLO_MODEL_PATH_SECONDARY", None)
    if sec and Path(sec).exists():
        p.detector_secondary = YOLODetector(sec, p.device)
    p.segmenter = SmartSegmenter(logger=logger)
    p.ocr_engine = OCREngine(device=p.device)
    return p


def run(image_path: Path, out_dir: Path, margin: int = 10):
    out_dir.mkdir(parents=True, exist_ok=True)
    bubbles_dir = out_dir / "bubbles"
    bubbles_dir.mkdir(parents=True, exist_ok=True)

    p = build_pipeline()
    renderer = TextRenderer()

    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"Impossible de charger {image_path}")
    h, w = img.shape[:2]
    print(f"Image: {w}x{h}px")

    # ── Détection ──
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
    print(f"Détections traduisibles: {len(dets)}")

    # ── OCR + masques ──
    crops, idxs = [], []
    for i, d in enumerate(dets):
        c = img[d.y1:d.y2, d.x1:d.x2]
        crops.append(c)
        idxs.append(i)

    results = p.ocr_engine.extract_batch(crops) if crops else []
    sibling_boxes = [(d.x1, d.y1, d.x2, d.y2) for d in dets]
    keep = []
    for i, res in zip(idxs, results):
        reason, _ = p._apply_ocr_result(img, dets[i], res, sibling_boxes)
        if not reason and TranslationPipeline._is_render_noise_text(dets[i].text_original, dets[i].ocr_confidence):
            reason = "render_noise_or_watermark"
        if not reason:
            keep.append(dets[i])
    print(f"Textes OCR valides: {len(keep)}")

    # ── Effacement (inpainting) ──
    out, _, ghost_idx = p._run_pre_inpainting(img, keep, renderer)
    ghost_set = set(ghost_idx)
    # Etat APRES effacement mais AVANT reinjection du texte : c'est le seul
    # endroit ou l'on peut juger l'effacement sans que le nouveau texte le
    # recouvre (contour de bulle entame, residu, halo).
    erased = out.copy()
    cv2.imwrite(str(out_dir / "page_erased.png"), erased)

    # ── Réinjection texte (texte source, PAS de traduction) ──
    meta = []
    for i, d in enumerate(keep):
        try:
            d.text_translated = str(d.text_original)
            p._prepare_render_style(renderer, img, d)
            out = renderer.insert_text(
                out,
                d.text_translated,
                int(d.x1), int(d.y1), int(d.x2), int(d.y2),
                text_regions=getattr(d, 'text_regions', None),
                text_color_rgb=getattr(d, 'text_color_rgb', None),
                outline_width_px=getattr(d, 'measured_outline_px', None),
                text_style=getattr(d, 'text_style', 'dialogue'),
                font_hint=getattr(d, 'font_hint', 'regular'),
                class_name=str(d.class_name),
                bubble_mask=getattr(d, 'mask_binary', None),
                font_key=getattr(d, 'font_key', None),
                source_line_height=getattr(d, 'source_line_height', None),
                sibling_boxes=[(o.x1, o.y1, o.x2, o.y2) for o in keep if o is not d],
            )
        except Exception as e:
            print(f"[{i}] Exception rendu: {e}")

    # ── Écriture image pleine page ──
    cv2.imwrite(str(out_dir / "page_before.png"), img)
    cv2.imwrite(str(out_dir / "page_after.png"), out)

    # ── Crops avant/après par bulle (avec marge) ──
    for i, d in enumerate(keep):
        x1 = max(0, d.x1 - margin)
        y1 = max(0, d.y1 - margin)
        x2 = min(w, d.x2 + margin)
        y2 = min(h, d.y2 + margin)
        before_c = img[y1:y2, x1:x2]
        after_c = out[y1:y2, x1:x2]
        if before_c.size == 0 or after_c.size == 0:
            continue
        cv2.imwrite(str(bubbles_dir / f"{i:02d}_before.png"), before_c)
        cv2.imwrite(str(bubbles_dir / f"{i:02d}_after.png"), after_c)
        cv2.imwrite(str(bubbles_dir / f"{i:02d}_erased.png"), erased[y1:y2, x1:x2])
        meta.append({
            "index": i,
            "class": str(d.class_name),
            "score": float(getattr(d, 'score', 0.0)),
            "bbox": [int(d.x1), int(d.y1), int(d.x2), int(d.y2)],
            "ocr_text": getattr(d, 'text_original', '') or '',
            "ghost_risk": i in ghost_set,
            # Polygones de ligne OCR, en coordonnees LOCALES a la bbox.
            # Sans eux, toute analyse ulterieure doit deviner ou est le texte
            # dans la boite — et se trompe : la mesure des FX prenait la bbox
            # entiere pour de l'encre, d'ou des epaisseurs de trait de 67 px.
            "text_regions": [
                {"bbox": [[int(p[0]), int(p[1])] for p in (r.get('bbox') or [])]}
                for r in (getattr(d, 'text_regions', None) or [])
                if r.get('bbox')
            ],
        })

    with open(out_dir / "bubbles_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"OK: {len(meta)} bulles -> {out_dir}")

    del img, det_img, dets, keep, out, crops, idxs
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--margin", type=int, default=50)
    args = ap.parse_args()
    run(args.image, args.out_dir, args.margin)
