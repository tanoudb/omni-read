"""
═══════════════════════════════════════════════════════════════════════════════
PIPELINE - Orchestration complète de la traduction
═══════════════════════════════════════════════════════════════════════════════

Mode --debug : sauvegarde image annotée + crops OCR dans output/debug/
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time
import json
import gc
import re
from concurrent.futures import ThreadPoolExecutor

from config import config
from utils import MemoryManager, model_context, WebtoonLogger
from core import YOLODetector, OCREngine, NLLBTranslator, TextRenderer, Detection, SmartSegmenter

# Couleurs par classe pour le debug (v3 + legacy v2)
DEBUG_COLORS = {
    'bulle':        (0, 255, 0),     # Vert
    'out_text':     (0, 165, 255),   # Orange
    'sfx':          (255, 0, 255),   # Magenta
    'System':       (128, 128, 128), # Gris
    'Bubble':       (0, 255, 0),
    'Box':          (255, 0, 0),
    'Outer_Text':   (0, 165, 255),
    'Small_Text':   (0, 255, 255),
    'Continuation': (255, 0, 255),
}


class TranslationPipeline:
    """Pipeline complet de traduction manhwa"""
    
    def __init__(
        self,
        logger: WebtoonLogger,
        debug: bool = False,
        lazy_models: bool = False,
        strict_memory_cleanup: bool = False,
        shared_ocr_engine: Optional[OCREngine] = None,
        shared_translator: Optional[NLLBTranslator] = None,
    ):
        self.logger = logger
        self.device = MemoryManager.get_device()
        self.debug = debug
        self.lazy_models = lazy_models
        self.strict_memory_cleanup = strict_memory_cleanup
        self.ocr_engine = shared_ocr_engine
        self.shared_ocr_engine = shared_ocr_engine
        self.shared_translator = shared_translator
        self.segmenter = None
        self.detector = None  # Added YOLO persistent detector
        
        self.logger.info(f"🖥️  Device: {self.device}")
        
        if self.debug:
            self.logger.info(f"🐛 Mode DEBUG activé")
            
        MemoryManager.log_memory_status(self.logger)
        
        if not self.lazy_models:
            self._ensure_ocr_engine()
            self._ensure_segmenter()
            self._ensure_detector()  # Ensure YOLO is loaded once

    def _ensure_ocr_engine(self) -> bool:
        if self.ocr_engine is not None:
            return True
        try:
            self.ocr_engine = OCREngine(
                device=self.device
            )
            self.logger.info(f"   🔤 OCR initialisé: {self.ocr_engine.get_backend_name()}")
            return True
        except Exception as e:
            self.logger.error(f"Échec initialisation OCR: {e}")
            self.ocr_engine = None
            return False

    def _ensure_segmenter(self) -> bool:
        if self.segmenter is not None:
            return True
        try:
            self.segmenter = SmartSegmenter(logger=self.logger)
            return True
        except Exception as e:
            self.logger.error(f"Échec initialisation segmenter: {e}")
            self.segmenter = None
            return False

    def _ensure_detector(self):
        """Load YOLO once and keep it in memory."""
        if self.detector is not None:
            return True
        try:
            from core import YOLODetector
            from config import config
            self.detector = YOLODetector(config.YOLO_MODEL_PATH, self.device)
            self.logger.info(f"   🎯 YOLO loaded (persistent)")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load YOLO: {e}")
            self.detector = None
            return False

    def _release_ocr_engine(self):
        if self.shared_ocr_engine is not None:
            self.ocr_engine = self.shared_ocr_engine
            return
        self.ocr_engine = None

    def _release_segmenter(self):
        if self.segmenter is not None and hasattr(self.segmenter, 'release'):
            try:
                self.segmenter.release()
            except Exception as e:
                self.logger.debug(f"release segmenter ignoré: {e}")
        self.segmenter = None

    def _release_stage_memory(self, stage_name: str):
        if self.strict_memory_cleanup:
            self.logger.info(f"🧹 Cleanup mémoire strict après {stage_name}...")
            MemoryManager.cleanup_aggressive()

    def _force_unload_before_translation(self):
        self.logger.info("🧹 Déchargement total YOLO/SAM2 avant traduction...")
        self._release_segmenter()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
        MemoryManager.cleanup_aggressive()

    @staticmethod
    def _dedupe_overlap_detections(detections: List[Detection], iou_threshold: float = 0.55) -> List[Detection]:
        if not detections:
            return []

        by_class: Dict[str, List[Detection]] = {}
        for det in detections:
            by_class.setdefault(det.class_name, []).append(det)

        kept: List[Detection] = []
        for cls_name, class_dets in by_class.items():
            sorted_dets = sorted(class_dets, key=lambda item: item.score, reverse=True)
            while sorted_dets:
                best = sorted_dets.pop(0)
                kept.append(best)
                filtered: List[Detection] = []
                for item in sorted_dets:
                    iou = 0.0
                    try:
                        from utils import ImageUtils
                        iou = ImageUtils.calculate_iou(best.bbox, item.bbox)
                    except Exception:
                        iou = 0.0
                    if iou < iou_threshold:
                        filtered.append(item)
                sorted_dets = filtered

        return kept

    def _sort_detections_reading_order(self, detections: List[Detection]) -> List[Detection]:
        """Tri lecture naturelle: haut→bas, puis gauche→droite par ligne."""
        if not detections:
            return detections

        sorted_by_y = sorted(detections, key=lambda d: ((d.y1 + d.y2) / 2, d.x1))
        median_h = sorted([(d.y2 - d.y1) for d in sorted_by_y])[len(sorted_by_y) // 2]
        row_threshold = max(30, int(median_h * 0.55))

        rows: List[List[Detection]] = []
        current_row: List[Detection] = []
        current_row_y: Optional[float] = None

        for det in sorted_by_y:
            cy = (det.y1 + det.y2) / 2
            if current_row_y is None or abs(cy - current_row_y) <= row_threshold:
                current_row.append(det)
                current_row_y = cy if current_row_y is None else (current_row_y + cy) / 2
            else:
                rows.append(sorted(current_row, key=lambda d: d.x1))
                current_row = [det]
                current_row_y = cy

        if current_row:
            rows.append(sorted(current_row, key=lambda d: d.x1))

        ordered: List[Detection] = []
        for row in rows:
            ordered.extend(row)
        return ordered

    def _extract_text_with_retry(self, img: np.ndarray, det: Detection):
        """OCR principal + retry ciblé si confiance faible."""
        crop = img[det.y1:det.y2, det.x1:det.x2]
        if crop.size == 0:
            return None, 0.0, False, "empty_crop", [], 1.0, "none"

        text, confidence, is_valid, skip_reason, text_regions, upscale_factor = self.ocr_engine.extract_text(crop)
        if is_valid and confidence >= 0.45:
            return text, confidence, is_valid, skip_reason, text_regions, upscale_factor, "base"

        # Retry 1: crop élargi (récupère ponctuation/bords de lettres)
        h_img, w_img = img.shape[:2]
        margin = 8
        x1 = max(0, det.x1 - margin)
        y1 = max(0, det.y1 - margin)
        x2 = min(w_img, det.x2 + margin)
        y2 = min(h_img, det.y2 + margin)
        crop_expand = img[y1:y2, x1:x2]

        # Offset réel entre crop élargi et crop original
        off_x = det.x1 - x1
        off_y = det.y1 - y1

        def _shift_regions(regions, dx, dy):
            """Ramène les text_regions d'un crop décalé vers le crop original."""
            if not regions or (dx == 0 and dy == 0):
                return regions
            result = []
            for region in regions:
                if not isinstance(region, dict):
                    result.append(region)
                    continue
                pts = region.get('bbox')
                if pts:
                    pts = [[p[0] - dx, p[1] - dy] for p in pts]
                result.append({**region, 'bbox': pts})
            return result

        if crop_expand.size > 0:
            t2, c2, v2, r2, reg2, u2 = self.ocr_engine.extract_text(crop_expand)
            if v2 and c2 >= max(0.35, confidence):
                reg2 = _shift_regions(reg2, off_x, off_y)
                return t2, c2, v2, r2, reg2, u2, "expanded"

        # Retry 2: contraste CLAHE + sharpen (crop original → pas de décalage)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        enhanced = cv2.filter2D(enhanced, -1, sharpen_kernel)
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        t3, c3, v3, r3, reg3, u3 = self.ocr_engine.extract_text(enhanced_bgr)
        if v3 and c3 >= max(0.35, confidence):
            return t3, c3, v3, r3, reg3, u3, "clahe_sharpen"

        return text, confidence, is_valid, skip_reason, text_regions, upscale_factor, "base"

    @staticmethod
    def _compute_global_confidence(det_score: float, ocr_conf: float, lang_conf: float) -> float:
        """Score global [0..1] basé sur détection + OCR + langue."""
        det_score = min(1.0, max(0.0, float(det_score)))
        ocr_conf = min(1.0, max(0.0, float(ocr_conf)))
        lang_conf = min(1.0, max(0.0, float(lang_conf)))
        return round(0.35 * det_score + 0.45 * ocr_conf + 0.20 * lang_conf, 3)

    @staticmethod
    def _extract_ocr_lines_from_regions(text_regions: List[Dict]) -> List[str]:
        if not text_regions:
            return []

        ranked = []
        for region in text_regions:
            text = (region.get('text') or '').strip()
            poly = region.get('bbox')
            if not text or not poly or len(poly) < 3:
                continue
            try:
                ys = [float(p[1]) for p in poly]
                y_center = sum(ys) / max(1, len(ys))
            except Exception:
                y_center = 0.0
            ranked.append((y_center, text))

        ranked.sort(key=lambda x: x[0])
        lines: List[str] = []
        for _, txt in ranked:
            if txt not in lines:
                lines.append(txt)
        return lines

    @staticmethod
    def _is_render_noise_text(text: str, confidence: float) -> bool:
        value = str(text or "").strip()
        if not value:
            return True

        # Tokens alpha très faibles: bruit OCR typique
        alpha_tokens = re.findall(r"[A-Za-z']+", value)
        if not alpha_tokens:
            return True

        # Mélange lettres/chiffres ou répétitions anormales sur faible confiance
        if re.search(r"[A-Za-z]+\d+[A-Za-z]+", value) and float(confidence) < 0.92:
            return True

        weird = 0
        for tok in alpha_tokens:
            t = tok.strip("'")
            if len(t) < 4:
                continue
            vowels = sum(1 for c in t.lower() if c in "aeiouy")
            vowel_ratio = vowels / max(1, len(t))
            repeated = re.search(r"(.)\1{3,}", t.lower()) is not None
            mixed = any(c.islower() for c in t) and any(c.isupper() for c in t)
            if (repeated and (mixed or vowel_ratio < 0.45)) or (mixed and vowel_ratio < 0.25):
                weird += 1

            if len(alpha_tokens) == 1 and len(t) >= 6 and float(confidence) < 0.90:
                if re.search(r"[a-z]{2,}[A-Z]", t) or re.search(r"[A-Z]{2,}[a-z]{2,}", t):
                    weird += 1

        if weird >= max(1, len(alpha_tokens) // 2) and float(confidence) < 0.93:
            return True
        return False

    @staticmethod
    def _run_pre_inpainting(
        img: np.ndarray,
        detections: List[Detection],
        renderer: TextRenderer,
    ) -> Tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        out = img.copy()
        for det in detections:
            cls = str(getattr(det, 'class_name', '') or '').strip().lower()
            if cls == 'system':
                effective_regions = getattr(det, 'mask_regions', None) or getattr(det, 'text_regions', None)
            else:
                effective_regions = getattr(det, 'text_regions', None) or getattr(det, 'mask_regions', None)
            out = renderer.inpaint_region(
                out,
                det.x1,
                det.y1,
                det.x2,
                det.y2,
                text_regions=effective_regions,
                class_name=getattr(det, 'class_name', ''),
                chirurgical_mask=getattr(det, 'chirurgical_mask', None),
                bubble_mask=getattr(det, 'mask_binary', None),
            )
        return out, max(0.0, time.perf_counter() - t0)
    
    # ─────────────────────────────────────────────────────────────────────────
    # DEBUG : Visualisation détections
    # ─────────────────────────────────────────────────────────────────────────
    
    def save_debug_detections(self, img: np.ndarray, all_detections: List[Detection],
                              translatable_detections: List[Detection],
                              output_dir: Path, image_name: str):
        """
        Image debug LISIBLE :
        - Chaque détection numérotée avec couleur par classe
        - Épaisseur forte pour les traduisibles, fine pour les autres
        - Légende en haut avec code couleur
        - Score de confiance affiché
        """
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # ── Image annotée ──
        img_debug = img.copy()
        h_img, w_img = img_debug.shape[:2]
        
        # Police plus grande 
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.7, min(w_img / 700, 1.5))  # Adapte à la largeur
        
        for idx, det in enumerate(all_detections):
            color = DEBUG_COLORS.get(det.class_name, (255, 255, 255))
            is_translatable = det.class_name in config.detection.translatable_classes
            thickness = 4 if is_translatable else 2
            
            # Rectangle
            cv2.rectangle(img_debug, (det.x1, det.y1), (det.x2, det.y2), color, thickness)
            
            # Label : #numero classe score%
            label = f"#{idx} {det.class_name} {det.score:.0%}"
            label_size = cv2.getTextSize(label, font, font_scale, 2)[0]
            
            # Fond du label (plus grand, lisible)
            pad = 6
            cv2.rectangle(img_debug,
                          (det.x1, det.y1 - label_size[1] - 2 * pad),
                          (det.x1 + label_size[0] + 2 * pad, det.y1),
                          color, -1)
            
            cv2.putText(img_debug, label,
                        (det.x1 + pad, det.y1 - pad),
                        font, font_scale,
                        (255, 255, 255), 2, cv2.LINE_AA)
        
        # ── Légende en haut ──
        legend_h = 50
        legend = np.zeros((legend_h, w_img, 3), dtype=np.uint8)
        x_offset = 10
        
        # Compter par classe
        class_counts = {}
        for det in all_detections:
            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
        
        for cls_name, count in class_counts.items():
            color = DEBUG_COLORS.get(cls_name, (255, 255, 255))
            is_tr = cls_name in config.detection.translatable_classes
            marker = "✓" if is_tr else "✗"
            txt = f"{cls_name}:{count} [{marker}]"
            
            # Pastille couleur
            cv2.rectangle(legend, (x_offset, 10), (x_offset + 25, 40), color, -1)
            x_offset += 30
            
            cv2.putText(legend, txt, (x_offset, 32),
                        font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            txt_w = cv2.getTextSize(txt, font, 0.6, 1)[0][0]
            x_offset += txt_w + 20
        
        # Coller légende au-dessus de l'image
        img_debug = np.vstack([legend, img_debug])
        
        debug_path = debug_dir / f"{image_name}_detections.png"
        cv2.imwrite(str(debug_path), img_debug)
        self.logger.info(f"   🐛 Debug détections: {debug_path}")
        
        # ── Crops individuels ──
        crops_dir = debug_dir / f"{image_name}_crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        
        for i, det in enumerate(translatable_detections):
            crop = img[det.y1:det.y2, det.x1:det.x2]
            if crop.size > 0:
                crop_path = crops_dir / f"crop_{i:02d}_{det.class_name}_{det.score:.2f}.png"
                cv2.imwrite(str(crop_path), crop)
        
        self.logger.info(f"   🐛 {len(translatable_detections)} crops dans: {crops_dir}")
    
    def save_debug_ocr(self, output_dir: Path, image_name: str,
                       detections: List[Detection]):
        """Sauvegarde un résumé OCR en texte"""
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        ocr_path = debug_dir / f"{image_name}_ocr_results.txt"
        
        with open(ocr_path, 'w', encoding='utf-8') as f:
            f.write(f"OCR Results for {image_name}\n")
            f.write("=" * 60 + "\n\n")
            
            for i, det in enumerate(detections):
                f.write(f"[{i+1}] {det.class_name} (score={det.score:.2f})\n")
                f.write(f"    bbox: [{det.x1}, {det.y1}, {det.x2}, {det.y2}]\n")
                f.write(f"    size: {det.x2-det.x1}x{det.y2-det.y1}px\n")
                f.write(f"    OCR text: \"{det.text_original or '(none)'}\" \n")
                f.write(f"    OCR conf: {det.ocr_confidence:.2f}\n")
                f.write(f"    Traduction NLLB: \"{getattr(det, 'text_nllb_raw', None) or '(none)'}\" \n")
                f.write(f"    Traduction finale: \"{det.text_translated or '(none)'}\" \n")
                f.write("\n")
        
        self.logger.info(f"   🐛 Résultats OCR: {ocr_path}")

    def save_debug_yolo_rejected(self, output_dir: Path, image_name: str, report: Dict) -> None:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        out_path = debug_dir / "yolo_rejected.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"YOLO Reject Report for {image_name}\n")
            f.write("=" * 80 + "\n\n")

            shape = report.get("image_shape", {}) if isinstance(report, dict) else {}
            if shape:
                f.write(f"Image shape: {shape.get('width', '?')}x{shape.get('height', '?')}\n\n")

            scale_stats = report.get("scale_debug_stats", {}) if isinstance(report, dict) else {}
            if scale_stats:
                f.write("[Scale stats]\n")
                for scale_key in sorted(scale_stats.keys()):
                    st = scale_stats.get(scale_key, {})
                    f.write(
                        f"- scale {scale_key}: raw={st.get('raw', 0)} kept={st.get('kept', 0)} "
                        f"reject_conf={st.get('threshold_reject', 0)} reject_border={st.get('border_reject', 0)}\n"
                    )
                    by_cls = st.get("threshold_reject_by_class", {}) or {}
                    if by_cls:
                        f.write(f"  reject_conf_by_class={by_cls}\n")
                f.write("\n")

            def _dump_events(title: str, events: List[Dict], max_items: int = 2000):
                f.write(f"[{title}] count={len(events)}\n")
                for idx, item in enumerate(events[:max_items], start=1):
                    f.write(f"{idx:04d}. {item}\n")
                f.write("\n")

            _dump_events("Raw detections", report.get("raw_detections", []) or [])
            _dump_events("Confidence rejects", report.get("confidence_rejects", []) or [])
            _dump_events("NMS per class rejects", report.get("nms_per_class_rejects", []) or [])
            _dump_events("NMS multi-scale rejects", report.get("nms_multi_scale_rejects", []) or [])
            _dump_events("Containment rejects", report.get("containment_rejects", []) or [])
            _dump_events("Geometry rejects", report.get("geometry_rejects", []) or [])
            _dump_events("Final detections", report.get("final_detections", []) or [])

        self.logger.info(f"   🐛 Rejets YOLO: {out_path}")

    @staticmethod
    def _wrap_debug_text(text: str, max_chars: int = 52) -> List[str]:
        words = (text or "").split()
        if not words:
            return [""]

        lines: List[str] = []
        current: List[str] = []
        current_len = 0
        for word in words:
            add_len = len(word) + (1 if current else 0)
            if current and (current_len + add_len) > max_chars:
                lines.append(" ".join(current))
                current = [word]
                current_len = len(word)
            else:
                current.append(word)
                current_len += add_len

        if current:
            lines.append(" ".join(current))
        return lines

    @staticmethod
    def _estimate_debug_text_color(img: np.ndarray, det: Detection) -> Tuple[int, int, int]:
        fallback = DEBUG_COLORS.get(getattr(det, 'class_name', ''), (255, 255, 255))
        text_regions = getattr(det, 'text_regions', None) or []

        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
        crop = img[y1:y2, x1:x2]
        if crop.size == 0 or not text_regions:
            return fallback

        h, w = crop.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for region in text_regions:
            pts = region.get('bbox') if isinstance(region, dict) else None
            if not pts:
                continue
            arr = np.array(pts, dtype=np.int32)
            if arr.ndim != 2 or arr.shape[0] < 3:
                continue
            arr[:, 0] = np.clip(arr[:, 0], 0, max(0, w - 1))
            arr[:, 1] = np.clip(arr[:, 1], 0, max(0, h - 1))
            cv2.fillPoly(mask, [arr], 255)

        if np.sum(mask) == 0:
            return fallback

        pixels = crop[mask > 0]
        if pixels.size == 0:
            return fallback

        bgr = np.median(pixels.reshape(-1, 3), axis=0)
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    def save_debug_double_page_ocr(self, img: np.ndarray, output_dir: Path, image_name: str,
                                   detections: List[Detection]):
        """
        Génère une vue debug double-page orientée crops (lisibilité maximale):
        - gauche: crop OCR zoomé, surligné avec régions OCR
        - droite: feuille blanche, retranscription OCR encadrée en couleur
        - indicateurs couleur + confiance visibles sur chaque ligne
        """
        if img is None or img.size == 0:
            return

        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        ordered = [d for d in self._sort_detections_reading_order(detections) if getattr(d, 'text_original', None)]
        if not ordered:
            return

        margin = 16
        gap = 16
        header_h = 56
        row_h = 232
        left_w = 920
        right_w = 920
        total_w = margin + left_w + gap + right_w + margin
        total_h = header_h + margin + len(ordered) * (row_h + gap)

        page = np.full((total_h, total_w, 3), 245, dtype=np.uint8)

        cv2.putText(page, "OCR Debug (Crop-based Double Page)", (margin, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(page, "Left: OCR crop highlighted | Right: OCR transcript + color/confidence",
                    (margin, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)

        y_cursor = header_h + margin
        for idx, det in enumerate(ordered, start=1):
            color = self._estimate_debug_text_color(img, det)
            class_name = str(getattr(det, 'class_name', 'text'))

            lx1, ly1 = margin, y_cursor
            lx2, ly2 = lx1 + left_w, y_cursor + row_h
            rx1, ry1 = lx2 + gap, y_cursor
            rx2, ry2 = rx1 + right_w, y_cursor + row_h

            # Cartes de fond colorées (léger tint)
            tint = np.array(color, dtype=np.float32)
            left_tint = np.clip((0.10 * tint + 0.90 * np.array([255, 255, 255])), 0, 255).astype(np.uint8)
            right_tint = np.clip((0.06 * tint + 0.94 * np.array([255, 255, 255])), 0, 255).astype(np.uint8)

            page[ly1:ly2, lx1:lx2] = left_tint
            page[ry1:ry2, rx1:rx2] = right_tint
            cv2.rectangle(page, (lx1, ly1), (lx2, ly2), color, 3)
            cv2.rectangle(page, (rx1, ry1), (rx2, ry2), color, 3)

            # Crop OCR
            crop = img[det.y1:det.y2, det.x1:det.x2].copy()
            if crop.size > 0:
                for region in getattr(det, 'text_regions', None) or []:
                    raw = region.get('bbox') if isinstance(region, dict) else None
                    if not raw:
                        continue
                    pts = np.array(raw, dtype=np.int32)
                    if pts.ndim != 2 or pts.shape[0] < 3:
                        continue
                    pts[:, 0] = np.clip(pts[:, 0], 0, max(0, crop.shape[1] - 1))
                    pts[:, 1] = np.clip(pts[:, 1], 0, max(0, crop.shape[0] - 1))
                    cv2.polylines(crop, [pts], True, color, 2, cv2.LINE_AA)

                inner_pad = 14
                slot_w = max(10, left_w - 2 * inner_pad)
                slot_h = max(10, row_h - 2 * inner_pad - 28)
                ch, cw = crop.shape[:2]
                scale = min(slot_w / max(1, cw), slot_h / max(1, ch))
                if scale <= 0:
                    scale = 1.0
                nw = max(1, int(cw * scale))
                nh = max(1, int(ch * scale))
                crop_rs = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA)

                px = lx1 + inner_pad + (slot_w - nw) // 2
                py = ly1 + 30 + inner_pad + (slot_h - nh) // 2
                page[py:py + nh, px:px + nw] = crop_rs

            cv2.putText(page, f"#{idx}  {class_name}  conf={det.ocr_confidence:.0%}",
                        (lx1 + 12, ly1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            # Panneau transcript (droite)
            text = (det.text_original or "").strip() or "(none)"
            lines = self._wrap_debug_text(text, max_chars=66)

            cv2.putText(page, "OCR transcript", (rx1 + 12, ry1 + 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (25, 25, 25), 1, cv2.LINE_AA)

            # carré couleur + score (en haut à droite de la carte)
            sw = 18
            cv2.rectangle(page, (rx2 - 170, ry1 + 8), (rx2 - 170 + sw, ry1 + 8 + sw), color, -1)
            cv2.rectangle(page, (rx2 - 170, ry1 + 8), (rx2 - 170 + sw, ry1 + 8 + sw), (0, 0, 0), 1)
            cv2.putText(page, f"{det.ocr_confidence:.0%}", (rx2 - 145, ry1 + 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (20, 20, 20), 1, cv2.LINE_AA)

            text_y = ry1 + 52
            for line in lines[:6]:
                cv2.putText(page, line, (rx1 + 14, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.58, (15, 15, 15), 1, cv2.LINE_AA)
                text_y += 28

            y_cursor += row_h + gap

        double_page = page
        out_path = debug_dir / f"{image_name}_ocr_double_page.png"
        cv2.imwrite(str(out_path), double_page)
        self.logger.info(f"   🐛 Debug double-page OCR: {out_path}")

    def save_debug_mask_bundle(
        self,
        img: np.ndarray,
        output_dir: Path,
        image_name: str,
        index: int,
        det: Detection,
        mask_regions: Optional[List[Dict]],
        mask_binary: Optional[np.ndarray] = None,
    ) -> None:
        """Sauvegarde crop original + masque segmenté pour debug fin."""
        debug_dir = output_dir / "debug" / f"{image_name}_pipeline"
        debug_dir.mkdir(parents=True, exist_ok=True)

        crop = img[det.y1:det.y2, det.x1:det.x2]
        if crop.size == 0:
            return

        cv2.imwrite(str(debug_dir / f"{index:02d}_crop.png"), crop)

        h, w = crop.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for region in mask_regions or []:
            pts = region.get('bbox') if isinstance(region, dict) else None
            if not pts:
                continue
            arr = np.array(pts, dtype=np.int32)
            if arr.ndim != 2 or arr.shape[0] < 3:
                continue
            arr[:, 0] = np.clip(arr[:, 0], 0, max(0, w - 1))
            arr[:, 1] = np.clip(arr[:, 1], 0, max(0, h - 1))
            cv2.fillPoly(mask, [arr], 255)

        cv2.imwrite(str(debug_dir / f"{index:02d}_mask.png"), mask)
        if mask_binary is not None and isinstance(mask_binary, np.ndarray) and mask_binary.size > 0:
            if mask_binary.shape[:2] != (h, w):
                mask_binary = cv2.resize(mask_binary, (w, h), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(debug_dir / f"{index:02d}_mask_bin.png"), mask_binary)

    def save_debug_render_bundle(
        self,
        output_dir: Path,
        image_name: str,
        index: int,
        before_crop: np.ndarray,
        after_crop: np.ndarray,
        det: Detection,
    ) -> None:
        debug_dir = output_dir / "debug" / f"{image_name}_pipeline"
        debug_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(debug_dir / f"{index:02d}_render_before.png"), before_crop)
        cv2.imwrite(str(debug_dir / f"{index:02d}_render_after.png"), after_crop)

        text_path = debug_dir / f"{index:02d}_texts.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(f"class: {det.class_name}\n")
            f.write(f"bbox: {det.bbox}\n")
            f.write(f"ocr: {det.text_original or ''}\n")
            f.write(f"translation: {det.text_translated or ''}\n")
            f.write(f"style: {getattr(det, 'text_style', 'dialogue')}\n")
            f.write(f"mask_regions: {len(getattr(det, 'mask_regions', []) or [])}\n")

    def save_debug_inpaint_only_bundle(
        self,
        output_dir: Path,
        image_name: str,
        index: int,
        det: Detection,
        cleaned_crop: np.ndarray,
    ) -> None:
        debug_dir = output_dir / "debug" / f"{image_name}_pipeline"
        debug_dir.mkdir(parents=True, exist_ok=True)

        if cleaned_crop is None or cleaned_crop.size == 0:
            return

        h, w = cleaned_crop.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cls = str(getattr(det, 'class_name', '') or '').strip().lower()
        if cls == 'system':
            mask_regions = getattr(det, 'mask_regions', None) or getattr(det, 'text_regions', None) or []
        else:
            mask_regions = getattr(det, 'text_regions', None) or getattr(det, 'mask_regions', None) or []
        for region in mask_regions:
            pts = region.get('bbox') if isinstance(region, dict) else None
            if not pts:
                continue
            arr = np.array(pts, dtype=np.int32)
            if arr.ndim != 2 or arr.shape[0] < 3:
                continue
            arr[:, 0] = np.clip(arr[:, 0], 0, max(0, w - 1))
            arr[:, 1] = np.clip(arr[:, 1], 0, max(0, h - 1))
            cv2.fillPoly(mask, [arr], 255)

        if getattr(det, 'chirurgical_mask', None) is not None:
            cmask = det.chirurgical_mask
            if isinstance(cmask, np.ndarray) and cmask.size > 0:
                if cmask.shape[:2] != (h, w):
                    cmask = cv2.resize(cmask, (w, h), interpolation=cv2.INTER_NEAREST)
                mask = cv2.bitwise_or(mask, cmask.astype(np.uint8))

        overlay = cleaned_crop.copy()
        if np.sum(mask) > 0:
            color = np.zeros_like(overlay)
            color[:, :] = (0, 0, 255)
            alpha = (mask.astype(np.float32) / 255.0) * 0.35
            alpha = np.expand_dims(alpha, axis=2)
            overlay = (overlay.astype(np.float32) * (1.0 - alpha) + color.astype(np.float32) * alpha).astype(np.uint8)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)

        cv2.imwrite(str(debug_dir / f"{index:02d}_inpaint_only.png"), cleaned_crop)
        cv2.imwrite(str(debug_dir / f"{index:02d}_inpaint_mask_overlay.png"), overlay)
        cv2.imwrite(str(debug_dir / f"{index:02d}_inpaint_mask.png"), mask)

    def save_debug_render_overview(
        self,
        output_dir: Path,
        image_name: str,
        original_img: np.ndarray,
        translated_img: np.ndarray,
        detections: List[Detection],
    ) -> None:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        h, w = original_img.shape[:2]
        panel_w = max(640, min(1400, w))
        panel_h = max(360, min(900, h))

        def _fit(img_src: np.ndarray) -> np.ndarray:
            return cv2.resize(img_src, (panel_w, panel_h), interpolation=cv2.INTER_AREA)

        top_left = _fit(original_img)
        top_right = _fit(translated_img)

        overlay = original_img.copy()
        for det in detections:
            color = DEBUG_COLORS.get(getattr(det, 'class_name', ''), (255, 255, 255))
            cv2.rectangle(overlay, (det.x1, det.y1), (det.x2, det.y2), color, 2)
        bottom_left = _fit(overlay)

        blend = cv2.addWeighted(original_img, 0.35, translated_img, 0.65, 0)
        bottom_right = _fit(blend)

        canvas = np.zeros((panel_h * 2, panel_w * 2, 3), dtype=np.uint8)
        canvas[0:panel_h, 0:panel_w] = top_left
        canvas[0:panel_h, panel_w:panel_w * 2] = top_right
        canvas[panel_h:panel_h * 2, 0:panel_w] = bottom_left
        canvas[panel_h:panel_h * 2, panel_w:panel_w * 2] = bottom_right

        labels = [
            ("ORIGINAL", 16, 28),
            ("TRANSLATED", panel_w + 16, 28),
            ("DETECTIONS", 16, panel_h + 28),
            ("OVERLAY", panel_w + 16, panel_h + 28),
        ]
        for text, lx, ly in labels:
            cv2.putText(canvas, text, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        out_path = debug_dir / "_render_debug.png"
        cv2.imwrite(str(out_path), canvas)
        self.logger.info(f"   🐛 Debug render global: {out_path}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # TRAITEMENT IMAGE UNIQUE
    # ─────────────────────────────────────────────────────────────────────────
    
    def process_image(self, image_path: Path, output_dir: Path) -> Dict:
        """Pipeline complet pour une image"""
        self.logger.header(f"📸 {image_path.name}")
        
        start_time = time.time()
        image_stem = image_path.stem
        
        # Charger image
        img = cv2.imread(str(image_path))
        if img is None:
            self.logger.error(f"Impossible de charger {image_path}")
            return {'success': False, 'error': 'load_failed'}
        
        h, w = img.shape[:2]
        self.logger.info(f"📏 {w}x{h}px\n")
        
        stats = {
            'image': image_path.name,
            'width': w,
            'height': h,
            'detections': 0,
            'translatable': 0,
            'translated': 0,
            'skipped': 0,
            'skip_reasons': {},
            'time_seconds': 0
        }
        timings = {
            'yolo_seconds': 0.0,
            'sam2_seconds': 0.0,
            'ocr_seconds': 0.0,
            'llm_seconds': 0.0,
            'llm_generation_seconds': 0.0,
            'inpainting_seconds': 0.0,
            'text_render_seconds': 0.0,
        }
        
        # ─────────────────────────────────────────────────────────────────
        # PHASE 1 : DETECTION
        # ─────────────────────────────────────────────────────────────────
        
        self.logger.phase("Detection", 1, 4)
        
        detections = []
        
        #─ Padding noir optionnel haut/bas pour les bords ──
        # Désactivé par défaut (WEBTOON_USE_BLACK_PADDING=false)
        use_black_padding = bool(
            getattr(config.detection, 'black_bars_enabled', getattr(config.detection, 'use_black_padding', False))
        )
        pad_h = int(h * max(0.0, float(getattr(config.detection, 'black_padding_ratio', 0.03)))) if use_black_padding else 0

        if pad_h > 0:
            black_bar_top = np.zeros((pad_h, w, 3), dtype=np.uint8)
            black_bar_bot = np.zeros((pad_h, w, 3), dtype=np.uint8)
            img_padded = np.vstack([black_bar_top, img, black_bar_bot])
            self.logger.info(f"   🔲 Barres noires actives: +{pad_h}px haut/bas ({w}x{img_padded.shape[0]}px)")
        else:
            img_padded = img
            self.logger.info("   🔲 Barres noires: désactivées")
        
        yolo_t0 = time.perf_counter()
        yolo_report: Dict = {}

        max_h = int(getattr(config.detection, 'max_height', 0) or 0)
        detection_img = img_padded
        detection_scale = 1.0

        if max_h > 0 and img_padded.shape[0] > max_h:
            detection_scale = img_padded.shape[0] / float(max_h)
            resized_w = max(1, int(img_padded.shape[1] / detection_scale))
            detection_img = cv2.resize(img_padded, (resized_w, max_h), interpolation=cv2.INTER_AREA)
            self.logger.info(f"   ↕️ Limite hauteur active: {img_padded.shape[0]} -> {max_h}px")

        detections = self.detector.detect(detection_img, logger=self.logger)
        yolo_report = self.detector.get_last_debug_report()

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
        translatable_detections = self.detector.get_translatable_detections(detections)
        timings['yolo_seconds'] += max(0.0, time.perf_counter() - yolo_t0)

        # Garde-fou: ne jamais traduire les SFX, même si la config change.
        pre_filter_count = len(translatable_detections)
        translatable_detections = [
            d for d in translatable_detections
            if str(getattr(d, 'class_name', '')).lower() != 'sfx'
        ]
        if len(translatable_detections) != pre_filter_count:
            self.logger.info(
                f"   🔇 SFX exclus de la traduction: {pre_filter_count - len(translatable_detections)}"
            )

        translatable_detections = self._sort_detections_reading_order(translatable_detections)
        
        stats['detections'] = len(detections)
        stats['translatable'] = len(translatable_detections)
        
        #─ DEBUG : sauvegarder visualisation ──
        if self.debug:
            self.save_debug_detections(img, detections, translatable_detections,
                                       output_dir, image_stem)
            self.save_debug_yolo_rejected(output_dir, image_stem, yolo_report if isinstance(yolo_report, dict) else {})
        
        self.logger.info(f"\n✅ {len(translatable_detections)} zones à traduire")
        
        if not translatable_detections:
            self.logger.warning("Aucune zone traduisible détectée")
            stats['time_seconds'] = time.time() - start_time
            return stats
        
        # ─────────────────────────────────────────────────────────────────
        # PHASE 2 : OCR
        # ─────────────────────────────────────────────────────────────────
        
        self.logger.phase("OCR", 2, 4)

        if not self._ensure_ocr_engine():
            self.logger.error("OCR engine non initialisé — arrêt de l'image")
            stats['time_seconds'] = time.time() - start_time
            return {'success': False, 'error': 'ocr_init_failed', **stats}

        if not self._ensure_segmenter():
            self.logger.warning("Segmenter indisponible — fallback OCR regions")

        if not self.ocr_engine:
            self.logger.error("OCR engine non initialisé — saut de l'OCR")
        else:
            self.logger.info(f"   🔤 Backend OCR: {self.ocr_engine.get_backend_name()}")
            try:
                ocr_diag = self.ocr_engine.get_runtime_diagnostics()
                self.logger.info(f"   🔎 OCR runtime: {ocr_diag}")
            except Exception as exc:
                self.logger.warning(f"   ⚠️  OCR runtime diagnostics indisponible: {exc}")

            crops: List[np.ndarray] = []
            crop_indices: List[int] = []
            for i, det in enumerate(translatable_detections):
                crop = img[det.y1:det.y2, det.x1:det.x2]
                if crop.size == 0:
                    self.logger.info(f"      ⚠️  Crop vide (det={i})")
                    stats['skipped'] += 1
                    continue
                crops.append(crop)
                crop_indices.append(i)

            self.logger.info(f"   📦 OCR batch: {len(crops)} crops envoyés")

            def _ocr_debug_log(message: str):
                self.logger.info(f"      {message}")

            ocr_t0 = time.perf_counter()
            batch_results = self.ocr_engine.extract_batch(crops, debug_hook=_ocr_debug_log)
            timings['ocr_seconds'] += max(0.0, time.perf_counter() - ocr_t0)

            for idx, ocr_result in zip(crop_indices, batch_results):
                det = translatable_detections[idx]
                text, confidence, is_valid, skip_reason, text_regions, upscale_factor = ocr_result

                det.ocr_upscale_factor = upscale_factor
                det.ocr_confidence = confidence

                if not is_valid or not text:
                    text_preview = (text or "").replace("\n", " ")
                    if len(text_preview) > 120:
                        text_preview = text_preview[:120] + "..."
                    self.logger.info(
                        f"      ⚠️  Ignoré: reason={skip_reason} conf={confidence:.2f} det={idx} text='{text_preview}'"
                    )
                    stats['skipped'] += 1
                    stats['skip_reasons'][skip_reason] = stats['skip_reasons'].get(skip_reason, 0) + 1
                    continue

                # Filtre de Bruit : confiance OCR < 0.85 ET texte < 4 caractères
                # (artefacts OCR typiques : "ii", "HOnn", etc.)
                if float(confidence) < 0.85 and len((text or "").strip()) < 4:
                    stats['skipped'] += 1
                    stats['skip_reasons']['ocr_noise_short'] = (
                        stats['skip_reasons'].get('ocr_noise_short', 0) + 1
                    )
                    self.logger.info(
                        f"      ⚠️  Ignoré (bruit court): conf={confidence:.2f} "
                        f"len={len((text or '').strip())} text='{(text or '').strip()}'"
                    )
                    continue

                det.text_original = text
                raw_regions = text_regions or []
                uf = float(upscale_factor) if upscale_factor and float(upscale_factor) > 1.0 else 1.0
                if uf != 1.0:
                    remapped = []
                    for region in raw_regions:
                        if not isinstance(region, dict):
                            remapped.append(region)
                            continue
                        pts = region.get('bbox')
                        if pts:
                            pts = [[p[0] / uf, p[1] / uf] for p in pts]
                        remapped.append({**region, 'bbox': pts})
                    det.text_regions = remapped
                else:
                    det.text_regions = raw_regions
                det.ocr_lines = self._extract_ocr_lines_from_regions(det.text_regions)

                if self.segmenter:
                    seg_t0 = time.perf_counter()
                    seg_regions, seg_binary, seg_backend = self.segmenter.segment_detection(img, det, det.text_regions)
                    det.mask_regions = seg_regions
                    det.mask_binary = seg_binary
                    # Construire masques chirurgical (OCR ∩ SAM2) — si possible
                    try:
                        import cv2 as _cv2
                        if det.mask_binary is None:
                            # fallback: OCR-only mask (dilated)
                            h_det = max(1, det.y2 - det.y1)
                            w_det = max(1, det.x2 - det.x1)
                            ocr_mask = np.zeros((h_det, w_det), dtype=np.uint8)
                            for region in det.text_regions or []:
                                pts = region.get('bbox') if isinstance(region, dict) else None
                                if not pts:
                                    continue
                                arr = np.array(pts, dtype=np.int32)
                                if arr.ndim != 2 or arr.shape[0] < 3:
                                    continue
                                arr[:, 0] = np.clip(arr[:, 0], 0, max(0, w_det - 1))
                                arr[:, 1] = np.clip(arr[:, 1], 0, max(0, h_det - 1))
                                _cv2.fillPoly(ocr_mask, [arr], 255)
                            kernel = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (11, 11))
                            ocr_mask_dilated = _cv2.dilate(ocr_mask, kernel, iterations=1)
                            det.chirurgical_mask = ocr_mask_dilated
                        else:
                            # det.mask_binary is crop-local mask; build OCR mask same shape
                                h_det = max(1, det.y2 - det.y1)
                                w_det = max(1, det.x2 - det.x1)
                                # debug shapes
                                try:
                                    mb_shape = det.mask_binary.shape
                                except Exception:
                                    mb_shape = None
                                print(f"[DEBUG CHIR] mask_binary.shape={mb_shape} expected=({h_det},{w_det})")

                                # Always build OCR mask with detection dimensions
                                ocr_mask = np.zeros((h_det, w_det), dtype=np.uint8)
                                for region in det.text_regions or []:
                                    pts = region.get('bbox') if isinstance(region, dict) else None
                                    if not pts:
                                        continue
                                    arr = np.array(pts, dtype=np.int32)
                                    if arr.ndim != 2 or arr.shape[0] < 3:
                                        continue
                                    arr[:, 0] = np.clip(arr[:, 0], 0, max(0, ocr_mask.shape[1] - 1))
                                    arr[:, 1] = np.clip(arr[:, 1], 0, max(0, ocr_mask.shape[0] - 1))
                                    _cv2.fillPoly(ocr_mask, [arr], 255)
                                kernel = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (11, 11))
                                ocr_mask_dilated = _cv2.dilate(ocr_mask, kernel, iterations=1)
                                # Intersection avec mask binaire SAM2 (resize mask_binary if needed)
                                try:
                                    mask_binary_local = det.mask_binary
                                    if mask_binary_local is None:
                                        det.chirurgical_mask = ocr_mask_dilated
                                    else:
                                        if getattr(mask_binary_local, 'ndim', 0) == 3:
                                            mask_binary_local = mask_binary_local[:, :, 0]
                                        if mask_binary_local.shape[:2] != (h_det, w_det):
                                            try:
                                                mask_binary_resized = _cv2.resize(mask_binary_local, (w_det, h_det), interpolation=_cv2.INTER_NEAREST)
                                                print(f"[DEBUG CHIR] resized mask_binary from {mb_shape} to {(h_det,w_det)}")
                                            except Exception:
                                                mask_binary_resized = mask_binary_local
                                        else:
                                            mask_binary_resized = mask_binary_local
                                        # ensure uint8
                                        mask_binary_resized = (mask_binary_resized > 0).astype(np.uint8) * 255
                                        det.chirurgical_mask = _cv2.bitwise_and(ocr_mask_dilated, mask_binary_resized)
                                except Exception:
                                    det.chirurgical_mask = ocr_mask_dilated
                        # closing pour boucher trous entre lettres
                        kernel_close = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (5, 5))
                        if getattr(det, 'chirurgical_mask', None) is not None:
                            det.chirurgical_mask = _cv2.morphologyEx(det.chirurgical_mask, _cv2.MORPH_CLOSE, kernel_close)
                    except Exception:
                        det.chirurgical_mask = getattr(det, 'mask_binary', None)
                    det.seg_backend = seg_backend
                    timings['sam2_seconds'] += max(0.0, time.perf_counter() - seg_t0)
                else:
                    det.mask_regions = det.text_regions
                    det.mask_binary = None
                    det.seg_backend = "none"
                    # Construire un chirurgical_mask minimal depuis text_regions (même sans segmenter)
                    try:
                        import cv2 as _cv2
                        h_det = max(1, det.y2 - det.y1)
                        w_det = max(1, det.x2 - det.x1)
                        ocr_mask = np.zeros((h_det, w_det), dtype=np.uint8)
                        for _region in det.text_regions or []:
                            _pts = _region.get('bbox') if isinstance(_region, dict) else None
                            if not _pts:
                                continue
                            _arr = np.array(_pts, dtype=np.int32)
                            if _arr.ndim != 2 or _arr.shape[0] < 3:
                                continue
                            _arr[:, 0] = np.clip(_arr[:, 0], 0, max(0, ocr_mask.shape[1] - 1))
                            _arr[:, 1] = np.clip(_arr[:, 1], 0, max(0, ocr_mask.shape[0] - 1))
                            _cv2.fillPoly(ocr_mask, [_arr], 255)
                        if np.sum(ocr_mask) > 0:
                            _k = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (7, 7))
                            det.chirurgical_mask = _cv2.dilate(ocr_mask, _k, iterations=1)
                        else:
                            det.chirurgical_mask = None
                    except Exception:
                        det.chirurgical_mask = None

                if self.debug:
                    self.save_debug_mask_bundle(
                        img,
                        output_dir,
                        image_stem,
                        idx + 1,
                        det,
                        det.mask_regions,
                        getattr(det, 'mask_binary', None),
                    )

                self.logger.info(f"      ✓ \"{text}\" ({confidence:.0%}) [{len(det.text_regions)} régions]")
        
        # Filtrer détections sans texte
        valid_detections = [d for d in translatable_detections if d.text_original]

        # Sécurité rendu: ignorer les artefacts OCR (pas d'inpainting ni texte)
        cleaned_for_render: List[Detection] = []
        for det in valid_detections:
            if self._is_render_noise_text(det.text_original, det.ocr_confidence):
                stats['skipped'] += 1
                stats['skip_reasons']['ocr_noise_render'] = stats['skip_reasons'].get('ocr_noise_render', 0) + 1
                self.logger.info(
                    f"      ⚠️  Zone ignorée (bruit OCR) conf={det.ocr_confidence:.2f} text='{(det.text_original or '')[:80]}'"
                )
                continue
            cleaned_for_render.append(det)
        valid_detections = cleaned_for_render
        
        self.logger.info(f"\n✅ {len(valid_detections)} textes extraits")
        
        if not valid_detections:
            self.logger.warning("Aucun texte valide extrait")
            # Debug : sauvegarder résultats OCR même si aucun valide
            if self.debug:
                self.save_debug_ocr(output_dir, image_stem, translatable_detections)
            self._release_ocr_engine()
            self._release_segmenter()
            self._release_stage_memory("OCR")
            stats['time_seconds'] = time.time() - start_time
            stats['timings'] = {k: round(v, 3) for k, v in timings.items()}
            return stats

        renderer = TextRenderer()

        # Release OCR and segmenter (not needed for translation)
        self._release_ocr_engine()
        self._release_segmenter()

        # LIGHT cleanup (not aggressive — keep YOLO in memory)
        gc.collect()

        self.logger.info("🧹 OCR/segmenter released")

        vram = MemoryManager.get_vram_usage()
        if vram:
            self.logger.info(f"   💾 VRAM: {vram['allocated_gb']:.2f} GB")

        # ─────────────────────────────────────────────────────────────────
        # PHASE 3 : TRADUCTION
        # ─────────────────────────────────────────────────────────────────
        
        self.logger.phase("Translation", 3, 4)
        
        # Traductions pour tous les textes
        texts_to_translate = [d.text_original for d in valid_detections]
        
        if texts_to_translate:
            llm_t0 = time.perf_counter()
            translator_cm = None
            translator = self.shared_translator
            if translator is None:
                translator_cm = model_context(lambda: NLLBTranslator(self.device))
                translator = translator_cm.__enter__()
            try:
                llm_gen_before = float(getattr(translator, 'get_generation_seconds_total', lambda: 0.0)())

                # Detect source language for all detections
                for det in valid_detections:
                    src_text = det.text_original or ""
                    detected_lang, lang_conf = translator.detect_source_language_with_confidence(src_text)
                    det.source_lang_detected = detected_lang
                    det.source_lang_confidence = lang_conf
                    det.global_confidence = self._compute_global_confidence(det.score, det.ocr_confidence, lang_conf)

                system_detections = [d for d in valid_detections if str(getattr(d, 'class_name', '')).lower() == 'system']
                regular_detections = [d for d in valid_detections if str(getattr(d, 'class_name', '')).lower() != 'system']
                translation_mode = str(getattr(config.translation, 'translation_mode', 'hybrid')).lower()

                if regular_detections:
                    self.logger.info(f"\n   🌍 Traduction page entière ({len(regular_detections)} bulles)")
                    payload_texts = [d.text_original for d in regular_detections]

                    translations_map = translator.translate_page_json(payload_texts)

                    map_ok = isinstance(translations_map, dict) and all(
                        (str(i) in translations_map or i in translations_map)
                        for i in range(len(payload_texts))
                    )
                    if not map_ok:
                        self.logger.warning("      ⚠️  JSON LLM non indexé, fallback bulle par bulle")
                        translations_map = {
                            str(i): translator.translate(txt)
                            for i, txt in enumerate(payload_texts)
                        }

                    for det_idx, det in enumerate(regular_detections):
                        det.text_nllb_raw = (translations_map.get(str(det_idx)) or translations_map.get(det_idx) or det.text_original)
                        det.text_translated = det.text_nllb_raw

                for det in system_detections:
                    lines = [ln.strip() for ln in getattr(det, 'ocr_lines', []) if ln and ln.strip()]
                    if len(lines) >= 2:
                        title_tr = translator.translate(lines[0]).strip()
                        body_tr = translator.translate(" ".join(lines[1:])).strip()
                        det.text_nllb_raw = f"{title_tr}\n{body_tr}"
                        det.text_translated = det.text_nllb_raw
                    else:
                        det.text_nllb_raw = translator.translate(det.text_original or "")
                        det.text_translated = det.text_nllb_raw
                
                cache_stats = translator.get_cache_stats()
                if cache_stats:
                    self.logger.info(f"\n   💾 Cache: {cache_stats['entries']} entrées, hit rate={cache_stats['hit_rate']}")
            finally:
                llm_gen_after = float(getattr(translator, 'get_generation_seconds_total', lambda: 0.0)())
                timings['llm_generation_seconds'] += max(0.0, llm_gen_after - llm_gen_before)
                if translator_cm is not None:
                    translator_cm.__exit__(None, None, None)
            timings['llm_seconds'] += max(0.0, time.perf_counter() - llm_t0)
        
        stats['translated'] = len(valid_detections)
        
        self.logger.info(f"\n✅ {len(valid_detections)} traductions")
        
        # ── DEBUG : sauvegarder résultats OCR + traduction ──
        if self.debug:
            self.save_debug_ocr(output_dir, image_stem, translatable_detections)
            self.save_debug_double_page_ocr(img, output_dir, image_stem, valid_detections)
        
        # ─────────────────────────────────────────────────────────────────
        # PHASE 4 : RENDERING
        # ─────────────────────────────────────────────────────────────────
        
        self.logger.phase("Rendering", 4, 4)

        img_translated = img.copy()

        inpaint_backend = "lama" if getattr(renderer, 'lama', None) is not None else "cv2-telea"
        self.logger.info(f"   🩹 Inpainting backend: {inpaint_backend}")
        
        for i, det in enumerate(valid_detections):
            if not det.text_translated:
                continue

            det.text_style = renderer.infer_text_style(
                det.text_translated,
                det.x2 - det.x1,
                det.y2 - det.y1,
                class_name=det.class_name,
            )
            det.text_color_rgb = renderer.extract_original_text_color(
                img,
                det.x1, det.y1, det.x2, det.y2,
                getattr(det, 'mask_regions', None) or getattr(det, 'text_regions', None),
            )
            det.font_hint = renderer.detect_font_hint(
                img,
                det.x1, det.y1, det.x2, det.y2,
                getattr(det, 'mask_regions', None) or getattr(det, 'text_regions', None),
            )
            
            self.logger.info(
                f"   [{i+1}/{len(valid_detections)}] bbox=({det.x1},{det.y1},{det.x2},{det.y2}) class={det.class_name}"
            )

            before_crop = None
            if self.debug:
                before_crop = img_translated[det.y1:det.y2, det.x1:det.x2].copy()

            effective_regions = getattr(det, 'mask_regions', None) or getattr(det, 'text_regions', None)

            # Inpainting
            inpaint_t0 = time.perf_counter()
            img_translated = renderer.inpaint_region(
                img_translated,
                det.x1, det.y1, det.x2, det.y2,
                text_regions=effective_regions,
                class_name=getattr(det, 'class_name', ''),
            )
            timings['inpainting_seconds'] += max(0.0, time.perf_counter() - inpaint_t0)

            # Text rendering
            render_t0 = time.perf_counter()
            img_translated = renderer.insert_text(
                img_translated,
                det.text_translated,
                det.x1, det.y1, det.x2, det.y2,
                text_regions=effective_regions,
                text_color_rgb=getattr(det, 'text_color_rgb', None),
                text_style=getattr(det, 'text_style', 'dialogue'),
                font_hint=getattr(det, 'font_hint', 'regular'),
                class_name=getattr(det, 'class_name', ''),
            )
            timings['text_render_seconds'] += max(0.0, time.perf_counter() - render_t0)

            if self.debug and before_crop is not None:
                after_crop = img_translated[det.y1:det.y2, det.x1:det.x2].copy()
                self.save_debug_render_bundle(output_dir, image_stem, i + 1, before_crop, after_crop, det)

        if self.debug:
            self.save_debug_render_overview(output_dir, image_stem, img, img_translated, valid_detections)
        
        # Sauvegarder
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_image_path = output_dir / f"{image_stem}_translated.png"
        cv2.imwrite(str(output_image_path), img_translated)
        
        self.logger.info(f"\n💾 {output_image_path.name}")
        
        # Métadonnées
        metadata_path = output_dir / f"{image_stem}_metadata.json"
        metadata = {
            'source': str(image_path),
            'output': str(output_image_path),
            'dimensions': {'width': w, 'height': h},
            'stats': stats,
            'detections': [
                {
                    'class': d.class_name,
                    'bbox': d.bbox,
                    'original': d.text_original,
                    'translated': d.text_translated,
                    'confidence': d.ocr_confidence,
                    'detection_confidence': d.score,
                }
                for d in valid_detections if d.text_translated
            ]
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        stats['time_seconds'] = time.time() - start_time
        stats['timings'] = {k: round(v, 3) for k, v in timings.items()}
        self.logger.info(
            "📈 Bench | "
            f"YOLO={stats['timings']['yolo_seconds']:.2f}s | "
            f"OCR={stats['timings']['ocr_seconds']:.2f}s | "
            f"LLM={stats['timings']['llm_seconds']:.2f}s | "
            f"INPAINT={stats['timings'].get('inpainting_seconds', 0.0):.2f}s | "
            f"TEXT={stats['timings'].get('text_render_seconds', 0.0):.2f}s"
        )
        self.logger.info(f"⏱️  {stats['time_seconds']:.1f}s")
        
        return stats

    # ─────────────────────────────────────────────────────────────────────────
    # TRAITEMENT BATCH
    # ─────────────────────────────────────────────────────────────────────────
    
    def process_directory(self, input_dir: Path, output_dir: Path) -> Dict:
        """Traite toutes les images d'un dossier"""
        image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        image_files = [
            f for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        if not image_files:
            self.logger.error(f"Aucune image dans {input_dir}")
            return {'success': False, 'error': 'no_images'}
        
        self.logger.header(f"🚀 TRADUCTION BATCH - {len(image_files)} IMAGES")
        self.logger.stat("Input", str(input_dir))
        self.logger.stat("Output", str(output_dir))
        
        global_stats = {
            'total_images': len(image_files),
            'processed': 0,
            'failed': 0,
            'total_detections': 0,
            'total_translated': 0,
            'total_skipped': 0,
            'total_time_seconds': 0,
            'results': []
        }
        
        start_time = time.time()
        
        for i, img_path in enumerate(image_files):
            self.logger.info(f"\n{'═'*80}")
            self.logger.info(f"IMAGE {i+1}/{len(image_files)}")
            self.logger.info(f"{'═'*80}")
            
            try:
                stats = self.process_image(img_path, output_dir)
                
                if stats.get('success', True):
                    global_stats['processed'] += 1
                    global_stats['total_detections'] += stats.get('detections', 0)
                    global_stats['total_translated'] += stats.get('translated', 0)
                    global_stats['total_skipped'] += stats.get('skipped', 0)
                else:
                    global_stats['failed'] += 1
                
                global_stats['results'].append(stats)
                
            except Exception as e:
                self.logger.error(f"Erreur: {e}")
                global_stats['failed'] += 1
                global_stats['results'].append({
                    'image': img_path.name,
                    'success': False,
                    'error': str(e)
                })
            
            if config.performance.aggressive_cleanup:
                MemoryManager.cleanup_medium()
        
        global_stats['total_time_seconds'] = time.time() - start_time
        
        self.logger.header("🎉 TRAITEMENT TERMINÉ")
        self.logger.summary({
            'Images traitées': f"{global_stats['processed']}/{global_stats['total_images']}",
            'Détections': global_stats['total_detections'],
            'Traductions': global_stats['total_translated'],
            'Ignorées': global_stats['total_skipped'],
            'Échecs': global_stats['failed'],
            'Temps total': f"{global_stats['total_time_seconds']:.1f}s",
            'Temps moyen': f"{global_stats['total_time_seconds'] / max(1, global_stats['processed']):.1f}s/image"
        })
        
        summary_path = output_dir / "summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(global_stats, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"\n📊 {summary_path}")
        
        return global_stats