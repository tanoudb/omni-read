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

from config import config
from utils import MemoryManager, model_context, WebtoonLogger
from core import YOLODetector, OCREngine, NLLBTranslator, TextRenderer, Detection, SmartSegmenter

try:
    from translator_gemini import GeminiTranslator
except ImportError:
    GeminiTranslator = None

try:
    from core.qcheck import QCheckEngine
except ImportError:
    QCheckEngine = None

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
        series_db=None,  # Mode Série
    ):
        self.logger = logger
        self.device = MemoryManager.get_device()
        self.debug = debug
        self.lazy_models = lazy_models
        self.strict_memory_cleanup = strict_memory_cleanup
        self.ocr_engine = shared_ocr_engine
        self.shared_ocr_engine = shared_ocr_engine
        self.shared_translator = shared_translator
        self.series_db = series_db  # Mode Série
        # Si mode série actif et glossaire présent, exposer les entrées
        # dans la config de traduction pour forcer les traductions NLLB.
        try:
            if self.series_db and hasattr(self.series_db, 'glossary'):
                # glossary.entries est déjà en MAJUSCULES normalisées
                forced = getattr(config.translation, 'forced_translations', None)
                if forced is None:
                    config.translation.forced_translations = dict(self.series_db.glossary.entries)
                else:
                    # Merge without overwriting existing config entries
                    merged = dict(forced)
                    merged.update(self.series_db.glossary.entries)
                    config.translation.forced_translations = merged
        except Exception:
            pass
        self.segmenter = None
        self.detector = None  # Added YOLO persistent detector
        self.detector_secondary = None  # Ensemble multi-modèles (désactivé par défaut)

        self.logger.info(f"🖥️  Device: {self.device}")

        if self.debug:
            self.logger.info("🐛 Mode DEBUG activé")
            
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
        """Load YOLO once and keep it in memory (+ modèle secondaire pour ensemble)."""
        if self.detector is not None:
            return True
        try:
            self.detector = YOLODetector(config.YOLO_MODEL_PATH, self.device)
            self.logger.info(f"   🎯 YOLO loaded (persistent): {config.YOLO_MODEL_PATH.name}")

            self.detector_secondary = None
            sec_path = getattr(config, 'YOLO_MODEL_PATH_SECONDARY', None)
            if sec_path and Path(sec_path).exists() and Path(sec_path) != Path(config.YOLO_MODEL_PATH):
                try:
                    self.detector_secondary = YOLODetector(sec_path, self.device)
                    self.logger.info(f"   🎯 YOLO secondaire (ensemble): {Path(sec_path).name}")
                except Exception as e:
                    self.logger.warning(f"   ⚠️ YOLO secondaire indisponible: {e}")
                    self.detector_secondary = None
            else:
                self.logger.info("   🎯 Détection mono-modèle (ensemble désactivé)")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load YOLO: {e}")
            self.detector = None
            self.detector_secondary = None
            return False

    def _detect_ensemble(self, image, logger=None) -> List[Detection]:
        """
        Détection (1 modèle par défaut, 2 si un modèle secondaire est configuré),
        suivie d'un dédoublonnage en DEUX temps :

        1. par classe, seuil permissif — deux modèles/passes différents ne
           prédisent jamais la même boîte au pixel près ;
        2. INTER-classes — c'est celui qui manquait. Une même boîte de narration
           sortait en "bulle" pour un modèle et en "out_text" pour l'autre ;
           comme le dédoublonnage regroupait par classe, les deux survivaient,
           étaient OCRisées, traduites et rendues séparément → deux (voire
           trois) textes superposés dans la même bulle.

        `detect()` applique déjà une résolution inter-classes, mais en INTERNE,
        donc une fois par modèle : elle ne voit jamais les boîtes de l'autre
        modèle. D'où la reprise ici, après fusion.
        """
        dets = self.detector.detect(image, logger=logger)

        if getattr(self, 'detector_secondary', None) is not None:
            dets_secondary = self.detector_secondary.detect(image, logger=logger)
            dets = dets + dets_secondary

        before = len(dets)
        dets = self._dedupe_overlap_detections(
            dets,
            iou_threshold=float(getattr(config.detection, 'ensemble_dedupe_iou', 0.35)),
        )
        dets = self._dedupe_cross_class_detections(
            dets,
            iou_threshold=float(getattr(config.detection, 'inter_class_iou_threshold', 0.5)),
        )
        if logger and before != len(dets):
            logger.info(f"      → dédoublonnage fusion: {before} → {len(dets)} zones")
        return dets

    @staticmethod
    def _dedupe_cross_class_detections(
        detections: List[Detection], iou_threshold: float = 0.5,
    ) -> List[Detection]:
        """
        Supprime les doublons entre classes DIFFÉRENTES : deux boîtes qui se
        recouvrent fortement décrivent la même zone de texte, quelle que soit
        l'étiquette. On garde celle dont le score est le plus élevé.

        Le recouvrement est mesuré à la fois en IoU et en taux de contenance
        (intersection / plus petite boîte) : une petite boîte "out_text" incluse
        dans une grande "bulle" a un IoU faible mais reste un doublon — la
        traduire séparément écrirait deux fois le même texte.
        """
        if len(detections) < 2:
            return list(detections)

        from utils import ImageUtils

        ordered = sorted(detections, key=lambda d: float(d.score), reverse=True)
        kept: List[Detection] = []
        for det in ordered:
            duplicate = False
            for keep in kept:
                if keep.class_name == det.class_name:
                    continue  # déjà traité par le dédoublonnage intra-classe
                try:
                    iou = float(ImageUtils.calculate_iou(keep.bbox, det.bbox))
                except Exception:
                    iou = 0.0

                inter_w = max(0, min(keep.x2, det.x2) - max(keep.x1, det.x1))
                inter_h = max(0, min(keep.y2, det.y2) - max(keep.y1, det.y1))
                inter = inter_w * inter_h
                area_a = max(1, (keep.x2 - keep.x1) * (keep.y2 - keep.y1))
                area_b = max(1, (det.x2 - det.x1) * (det.y2 - det.y1))
                containment = inter / float(min(area_a, area_b))

                if iou >= iou_threshold or containment >= 0.80:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(det)
        return kept

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

    # ─────────────────────────────────────────────────────────────────────────
    # OCR → DÉTECTION : chemin UNIQUE partagé par tous les modes
    #
    # Ces trois helpers existent parce que le post-traitement OCR était
    # dupliqué entre `process_image` et `_process_chapters_mega_batch`, et que
    # chaque correctif n'avait été appliqué qu'à une copie : le mode chapitre
    # ne remettait pas les polygones OCR à l'échelle et ne segmentait pas du
    # tout. Un seul chemin, donc un seul endroit à corriger.
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _rescale_regions_from_ocr(regions: Optional[List[Dict]], upscale_factor) -> List[Dict]:
        """
        Ramène les polygones OCR dans le repère du crop d'origine.

        L'OCR agrandit les petits crops avant lecture (cf. preprocess_image) et
        renvoie ses polygones dans les coordonnées AGRANDIES. Sans cette
        division, le masque d'effacement est jusqu'à ~2.3x trop grand et
        décalé : bavures d'inpainting sur le dessin, et texte original laissé
        en place là où le masque n'est pas tombé.
        """
        try:
            uf = float(upscale_factor or 1.0)
        except Exception:
            uf = 1.0
        if not regions or uf <= 1.0:
            return list(regions or [])

        rescaled: List[Dict] = []
        for region in regions:
            if not isinstance(region, dict):
                rescaled.append(region)
                continue
            pts = region.get('bbox')
            if pts:
                pts = [[p[0] / uf, p[1] / uf] for p in pts]
            rescaled.append({**region, 'bbox': pts})
        return rescaled

    @staticmethod
    def _ocr_mask_from_regions(
        regions: Optional[List[Dict]], h_det: int, w_det: int,
        crop_bgr: Optional[np.ndarray] = None, dilate: int = 3,
    ) -> Optional[np.ndarray]:
        """
        Masque des LETTRES d'origine, en coordonnées locales bbox.

        Les polygones OCR délimitent des LIGNES entières, pas des glyphes.
        Les remplir pleins puis les dilater de 11 px soudait les lignes entre
        elles : sur une boîte de narration de 4 lignes, le masque couvrait 95 %
        de la boîte. LaMa recevait donc un trou de la taille du bloc de texte
        et reconstruisait une texture — d'où un fantôme flou du texte anglais
        visible sous la traduction.

        On redescend donc à l'encre : à l'intérieur de chaque polygone, on ne
        garde que les pixels qui s'écartent du fond local, puis on dilate juste
        assez pour attraper l'anticrénelage.

        Le seuil est calculé LIGNE PAR LIGNE. Un seuil global était dominé par
        la ligne la plus contrastée : sur une carte System holographique, dont
        les lignes du haut sont en blanc semi-transparent et celle du bas en
        blanc franc, seule la ligne du bas était masquée — les deux autres
        restaient visibles sous la traduction.
        """
        polygons = np.zeros((h_det, w_det), dtype=np.uint8)
        per_line: List[np.ndarray] = []
        for region in regions or []:
            pts = region.get('bbox') if isinstance(region, dict) else None
            if not pts:
                continue
            arr = np.array(pts, dtype=np.int32)
            if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
                continue
            arr[:, 0] = np.clip(arr[:, 0], 0, max(0, w_det - 1))
            arr[:, 1] = np.clip(arr[:, 1], 0, max(0, h_det - 1))
            line = np.zeros((h_det, w_det), dtype=np.uint8)
            cv2.fillPoly(line, [arr], 255)
            cv2.fillPoly(polygons, [arr], 255)
            per_line.append(line)

        if int(np.count_nonzero(polygons)) == 0:
            return None

        ink = None
        if crop_bgr is not None and crop_bgr.size > 0 and per_line:
            try:
                gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
                if gray.shape[:2] != (h_det, w_det):
                    gray = cv2.resize(gray, (w_det, h_det), interpolation=cv2.INTER_AREA)
                gray_i = gray.astype(np.int16)
                search_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

                accumulated = np.zeros((h_det, w_det), dtype=np.uint8)
                for line in per_line:
                    inside = gray[line > 0]
                    if inside.size <= 32:
                        continue

                    # Séparation par seuil d'Otsu, PUIS le côté MINORITAIRE
                    # est l'encre — sans supposer si l'encre est plus sombre
                    # ou plus claire que le fond. Une estimation par
                    # « le fond est le côté clair » (percentile haut, écart
                    # mesuré vers le bas) est correcte pour du texte noir sur
                    # bulle blanche (le cas très majoritaire ici), mais
                    # s'inverse sur une carte System à texte BLANC sur fond
                    # teal : le fond y est plus sombre que l'encre, donc
                    # « le clair = fond » désigne alors les LETTRES comme
                    # fond, et le seuil qui en découle ne masque plus que la
                    # bande de fond visible entre/autour des lettres — les
                    # lettres elles-mêmes restaient intactes. Mesuré sur une
                    # de ces cartes : masque non-vide mais qui manquait
                    # entièrement le remplissage blanc des glyphes.
                    try:
                        otsu_thr, _ = cv2.threshold(
                            inside, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                        )
                    except Exception:
                        continue
                    frac_below = float(np.mean(inside < otsu_thr))
                    ink_is_dark_side = frac_below <= 0.5
                    minority_frac = frac_below if ink_is_dark_side else (1.0 - frac_below)
                    # Pas de séparation nette (quasi-uniforme, ou l'inverse :
                    # presque tout est "encre") → ce seuillage n'est pas
                    # concluant pour cette ligne, on la laisse aux polygones
                    # pleins (repli plus bas).
                    if not (0.03 <= minority_frac <= 0.55):
                        continue

                    # On cherche un peu AU-DELÀ du polygone : les polices script
                    # ont des jambages et fioritures qui sortent du rectangle
                    # OCR et laissaient un résidu coloré.
                    search = cv2.dilate(line, search_kernel)
                    if ink_is_dark_side:
                        candidate = ((gray_i < otsu_thr) & (search > 0)).astype(np.uint8)
                    else:
                        candidate = ((gray_i >= otsu_thr) & (search > 0)).astype(np.uint8)

                    # ...mais on ne garde que ce qui touche le polygone : sinon
                    # un bout de dessin voisin serait pris pour du texte.
                    n_labels, labels = cv2.connectedComponents(candidate, 8)
                    if n_labels <= 1:
                        continue
                    touching = np.unique(labels[line > 0])
                    keep = np.isin(labels, touching[touching > 0])
                    accumulated |= keep.astype(np.uint8) * 255

                # Trop peu d'encre trouvée : seuillage non concluant (dégradé,
                # texte contourné). On reprend les polygones pleins.
                if np.count_nonzero(accumulated) >= 0.02 * np.count_nonzero(polygons):
                    ink = accumulated
            except Exception:
                ink = None

        mask = ink if ink is not None else polygons
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        return cv2.dilate(mask, kernel, iterations=1)

    def _build_masks_for_detection(self, img: np.ndarray, det: Detection) -> float:
        """
        Renseigne mask_regions / mask_binary / chirurgical_mask.
        Retourne le temps passé (pour le bench).
        """
        t0 = time.perf_counter()
        h_det = max(1, det.y2 - det.y1)
        w_det = max(1, det.x2 - det.x1)

        seg_regions, seg_binary, seg_backend = det.text_regions, None, "none"
        if self.segmenter is not None:
            try:
                seg_regions, seg_binary, seg_backend = self.segmenter.segment_detection(
                    img, det, det.text_regions,
                )
            except Exception as exc:
                self.logger.debug(f"segmentation ignorée: {exc}")
                seg_regions, seg_binary, seg_backend = det.text_regions, None, "error"

        det.mask_regions = seg_regions or det.text_regions
        det.mask_binary = seg_binary
        det.seg_backend = seg_backend

        ocr_mask = self._ocr_mask_from_regions(
            det.text_regions, h_det, w_det,
            crop_bgr=img[max(0, det.y1):det.y2, max(0, det.x1):det.x2],
        )
        if ocr_mask is None:
            det.chirurgical_mask = None
            return max(0.0, time.perf_counter() - t0)

        bubble = det.mask_binary
        if bubble is not None:
            if getattr(bubble, 'ndim', 0) == 3:
                bubble = bubble[:, :, 0]
            if bubble.shape[:2] != (h_det, w_det):
                try:
                    bubble = cv2.resize(bubble, (w_det, h_det), interpolation=cv2.INTER_NEAREST)
                except Exception:
                    bubble = None

        if bubble is not None:
            bubble = (bubble > 0).astype(np.uint8) * 255
            intersection = cv2.bitwise_and(ocr_mask, bubble)
            # L'intersection borne l'effacement à l'intérieur de la bulle (elle
            # évite de manger le trait de contour ou le dessin voisin). Mais si
            # le masque de bulle est trop serré, elle peut presque tout annuler
            # — auquel cas on n'effacerait plus rien et le texte original
            # resterait sous la traduction. On garde alors le masque OCR seul.
            if int(np.sum(intersection)) > 0.30 * int(np.sum(ocr_mask)):
                ocr_mask = intersection

        # Fermeture douce : boucher les trous entre les lettres d'un mot, sans
        # souder les lignes entre elles (elles sont espacées d'une dizaine de
        # pixels — un noyau plus large recrée le pavé plein qu'on veut éviter).
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        det.chirurgical_mask = cv2.morphologyEx(ocr_mask, cv2.MORPH_CLOSE, kernel_close)
        return max(0.0, time.perf_counter() - t0)

    def _apply_ocr_result(
        self, img: np.ndarray, det: Detection, ocr_result,
    ) -> Tuple[Optional[str], float]:
        """
        Applique un résultat OCR à une détection : filtres de bruit, remise à
        l'échelle des polygones, lignes, masques.

        Retourne (skip_reason ou None, secondes de segmentation).
        """
        text, confidence, is_valid, skip_reason, text_regions, upscale_factor = ocr_result

        det.ocr_upscale_factor = upscale_factor
        det.ocr_confidence = confidence
        det.text_original = ""

        if not is_valid or not text:
            return (skip_reason or "invalid"), 0.0

        # Artefacts OCR typiques : "ii", "HOnn"... courts ET peu confiants.
        if float(confidence) < 0.85 and len((text or "").strip()) < 4:
            return "ocr_noise_short", 0.0

        det.text_original = text
        det.text_regions = self._rescale_regions_from_ocr(text_regions, upscale_factor)
        det.ocr_lines = self._extract_ocr_lines_from_regions(det.text_regions)
        seg_seconds = self._build_masks_for_detection(img, det)
        return None, seg_seconds

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
    ) -> Tuple[np.ndarray, float, List[int]]:
        """
        Retourne aussi les INDEX (dans `detections`) des détections dont
        l'effacement est resté imparfait (fantôme résiduel) — `renderer`
        accumule ce signal dans `qcheck_flags` (cf. `_apply_erasure_by_group`),
        ici associé à la détection précise plutôt qu'à la page entière, pour
        que le rapport QCheck pointe la bonne bulle/cartouche.
        """
        t0 = time.perf_counter()
        out = img.copy()
        ghost_risk_indices: List[int] = []
        for idx, det in enumerate(detections):
            # Toujours les polygones OCR : `mask_regions` est la segmentation de
            # la bulle entière, l'utiliser comme masque d'effacement repeindrait
            # tout l'intérieur au lieu des seules lettres.
            effective_regions = getattr(det, 'text_regions', None) or getattr(det, 'mask_regions', None)
            flags_before = len(renderer.qcheck_flags)
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
            if len(renderer.qcheck_flags) > flags_before:
                ghost_risk_indices.append(idx)
        return out, max(0.0, time.perf_counter() - t0), ghost_risk_indices

    @staticmethod
    def _measure_source_line_height(regions: Optional[List[Dict]]) -> Optional[float]:
        """
        Hauteur de ligne du texte ORIGINAL (médiane des polygones OCR).

        Sert de garde-fou au dimensionnement : sans elle, le renderer remplit
        70 % de la boîte quoi qu'il arrive, ce qui grossit un « FINALLY… »
        discret jusqu'à saturer sa bulle. Un letterer conserve le corps de
        texte de la planche.
        """
        heights: List[float] = []
        for region in regions or []:
            pts = region.get('bbox') if isinstance(region, dict) else None
            if not pts or len(pts) < 3:
                continue
            try:
                ys = [float(p[1]) for p in pts]
            except Exception:
                continue
            h = max(ys) - min(ys)
            if h > 2:
                heights.append(h)
        if not heights:
            return None
        heights.sort()
        return float(heights[len(heights) // 2])

    @staticmethod
    def _prepare_render_style(renderer: TextRenderer, img: np.ndarray, det: Detection) -> None:
        """Style, couleur, graisse et taille source — calculés sur l'image ORIGINALE."""
        regions = getattr(det, 'mask_regions', None) or getattr(det, 'text_regions', None)

        det.text_style = renderer.infer_text_style(
            det.text_translated,
            det.x2 - det.x1, det.y2 - det.y1,
            class_name=det.class_name,
        )
        det.text_color_rgb = renderer.extract_original_text_color(
            img, det.x1, det.y1, det.x2, det.y2, regions,
        )
        det.font_hint = renderer.detect_font_hint(
            img, det.x1, det.y1, det.x2, det.y2, regions,
        )
        # Mesurée sur les polygones OCR (serrés autour des lettres), pas sur le
        # masque de segmentation qui couvre toute la bulle.
        det.source_line_height = TranslationPipeline._measure_source_line_height(
            getattr(det, 'text_regions', None)
        )

    @staticmethod
    def _write_output_image(out_dir: Path, stem: str, img: np.ndarray) -> Path:
        """
        Écrit la planche traduite.

        Défaut PNG : sans perte, c'est la meilleure qualité possible. Le fichier
        est gros (~35 Mo pour une planche fusionnée de 43 000 px) mais il ne
        rajoute aucune génération de compression par-dessus le JPEG source.
        `IMWRITE_PNG_COMPRESSION=6` réduit la taille sans toucher aux pixels
        (OpenCV utilise 1 par défaut, soit le plus rapide et le plus gros).

        WEBTOON_OUTPUT_FORMAT=jpg + WEBTOON_OUTPUT_QUALITY=95 pour des fichiers
        ~6x plus légers, au prix d'une passe de compression supplémentaire.
        Le WebP est refusé au-delà de 16383 px : c'est sa limite de format, et
        ces planches font bien plus haut.
        """
        fmt = str(getattr(config.rendering, 'output_format', 'png')).lower().strip()
        quality = int(getattr(config.rendering, 'output_quality', 95))

        if fmt == 'webp' and max(img.shape[:2]) > 16383:
            fmt = 'png'

        if fmt in ('jpg', 'jpeg'):
            path = out_dir / f"{stem}_translated.jpg"
            params = [
                cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, quality)),
                cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
            ]
        elif fmt == 'webp':
            path = out_dir / f"{stem}_translated.webp"
            params = [cv2.IMWRITE_WEBP_QUALITY, max(1, min(101, quality))]
        else:
            path = out_dir / f"{stem}_translated.png"
            params = [cv2.IMWRITE_PNG_COMPRESSION, 6]

        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img, params)
        return path

    MEGA_BATCH_SIZE = 5  # Nombre de chapitres à accumuler avant 1 appel Gemini

    def _process_chapters_mega_batch(self, chapter_map, output_dir):
        """
        Traite les chapitres en mode mega-batch :
        - Détection + OCR par chapitre (local, pas de quota API)
        - Traduction groupée de N chapitres en 1 appel Gemini
        - Rendering par chapitre (local)
        """
        import time
        import cv2
        import numpy as np
        from pathlib import Path
        from config import config
        from core import TextRenderer

        try:
            from core.translator_gemini import GeminiTranslator
        except ImportError:
            GeminiTranslator = None

        all_images_count = sum(len(imgs) for imgs in chapter_map.values())

        global_stats = {
            'total_images': all_images_count,
            'processed': 0,
            'failed': 0,
            'total_detections': 0,
            'total_translated': 0,
            'total_skipped': 0,
            'total_time_seconds': 0,
            'results': [],
        }
        start_time = time.time()

        # Structure pour stocker tout entre les phases
        chapter_data = {}

        for ch_name, imgs in sorted(chapter_map.items()):
            out_ch_dir = output_dir / ch_name
            out_ch_dir.mkdir(parents=True, exist_ok=True)
            self.logger.header(f"🚀 PHASE 1 (Detect+OCR): {ch_name} - {len(imgs)} IMAGES")

            ch_info = {
                'images': [],
                'detections': {},
                'valid_texts': [],
                'out_dir': out_ch_dir,
            }

            if not self._ensure_ocr_engine():
                self.logger.error("OCR engine non initialisé")
                chapter_data[ch_name] = ch_info
                continue
            self._ensure_segmenter()

            def _ocr_debug_log(message):
                self.logger.info(f"      {message}")

            total_dets = 0
            sorted_imgs = sorted(imgs)
            for img_idx, img_path in enumerate(sorted_imgs, start=1):
                # Progression explicite : sur un chapitre de 136 pages, les
                # logs de détection se ressemblent tous et rien n'indiquait
                # où on en était ni si ça avançait encore. Un blocage passait
                # ainsi pour de la lenteur pendant de longues minutes.
                self.logger.info(f"   📄 [{img_idx}/{len(sorted_imgs)}] {img_path.name}")
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        self.logger.error(f"Impossible de charger {img_path}")
                        ch_info['images'].append((img_path, 0, 0))
                        ch_info['detections'][img_path] = []
                        continue

                    h, w = img.shape[:2]
                    # On ne conserve QUE les dimensions : garder les images
                    # décodées de 5 chapitres jusqu'à la phase de rendu faisait
                    # plus d'1.5 Go de RAM (une planche webtoon fusionnée pèse
                    # ~90 Mo décodée). Elles sont rechargées au rendu.
                    ch_info['images'].append((img_path, w, h))

                    use_black_padding = bool(
                        getattr(config.detection, 'black_bars_enabled',
                                getattr(config.detection, 'use_black_padding', False))
                    )
                    pad_h = int(h * max(0.0, float(
                        getattr(config.detection, 'black_padding_ratio', 0.03)
                    ))) if use_black_padding else 0

                    if pad_h > 0:
                        img_padded = np.vstack([
                            np.zeros((pad_h, w, 3), dtype=np.uint8),
                            img,
                            np.zeros((pad_h, w, 3), dtype=np.uint8),
                        ])
                    else:
                        img_padded = img

                    max_h = int(getattr(config.detection, 'max_height', 0) or 0)
                    detection_img = img_padded
                    detection_scale = 1.0
                    if max_h > 0 and img_padded.shape[0] > max_h:
                        detection_scale = img_padded.shape[0] / float(max_h)
                        resized_w = max(1, int(img_padded.shape[1] / detection_scale))
                        detection_img = cv2.resize(
                            img_padded, (resized_w, max_h), interpolation=cv2.INTER_AREA
                        )

                    dets = self._detect_ensemble(detection_img, logger=self.logger)

                    if detection_scale != 1.0:
                        for det in dets:
                            det.bbox = [
                                float(det.bbox[0] * detection_scale),
                                float(det.bbox[1] * detection_scale),
                                float(det.bbox[2] * detection_scale),
                                float(det.bbox[3] * detection_scale),
                            ]

                    if pad_h > 0:
                        for det in dets:
                            new_y1 = max(0, int(det.bbox[1]) - pad_h)
                            new_y2 = min(h, int(det.bbox[3]) - pad_h)
                            det.bbox = [det.bbox[0], new_y1, det.bbox[2], new_y2]

                    dets = [d for d in dets if d.y2 > 0 and d.y1 < h]
                    translatable = self.detector.get_translatable_detections(dets)
                    translatable = [
                        d for d in translatable
                        if str(getattr(d, 'class_name', '')).lower() != 'sfx'
                    ]
                    translatable = self._sort_detections_reading_order(translatable)

                    ch_info['detections'][img_path] = translatable
                    total_dets += len(translatable)

                    if self.debug:
                        self.save_debug_detections(img, dets, translatable, out_ch_dir, img_path.stem)

                    # ── OCR + segmentation, image par image ──
                    # Fait ici (et non en un gros lot après toutes les images)
                    # pour que `img` puisse être libérée tout de suite.
                    crops = []
                    crop_indices = []
                    for di, det in enumerate(translatable):
                        crop = img[det.y1:det.y2, det.x1:det.x2]
                        if crop.size == 0:
                            continue
                        crops.append(crop)
                        crop_indices.append(di)

                    if crops:
                        try:
                            ocr_results = self.ocr_engine.extract_batch(
                                crops, debug_hook=_ocr_debug_log if self.debug else None,
                            )
                        except Exception as exc:
                            self.logger.error(f"OCR échoué sur {img_path.name}: {exc}")
                            ocr_results = [("", 0.0, False, 'ocr_error', [], 1.0)] * len(crops)

                        for det_idx, ocr_res in zip(crop_indices, ocr_results):
                            det = translatable[det_idx]
                            # Chemin partagé : remise à l'échelle des polygones
                            # OCR + segmentation + masque chirurgical. Aucun de
                            # ces trois traitements n'était appliqué ici avant.
                            skip_reason, _seg_s = self._apply_ocr_result(img, det, ocr_res)
                            if skip_reason or not det.text_original:
                                continue
                            if self._is_render_noise_text(det.text_original, det.ocr_confidence):
                                det.text_original = ""
                                continue
                            ch_info['valid_texts'].append(
                                (img_path, det_idx, det.text_original,
                                 getattr(det, 'class_name', '') or '')
                            )

                except Exception as e:
                    self.logger.error(f"Erreur detection/OCR {img_path}: {e}")
                    ch_info['images'].append((img_path, 0, 0))
                    ch_info['detections'][img_path] = []
                finally:
                    img = None
                    gc.collect()

            self.logger.info(
                f"✅ Détection: {total_dets} zones — OCR: {len(ch_info['valid_texts'])} textes valides"
            )
            global_stats['total_detections'] += total_dets
            chapter_data[ch_name] = ch_info

        try:
            self._release_ocr_engine()
            self._release_segmenter()
        except Exception:
            pass

        # PHASE 2 : TRADUCTION MEGA-BATCH
        self.logger.header("🌐 PHASE 2 : TRADUCTION MEGA-BATCH")

        translator = self.shared_translator
        translator_cm = None

        if translator is None:
            backend = str(getattr(config.translation, 'backend', 'local_llm')).lower()
            if backend == "gemini" and GeminiTranslator is not None:
                series_name = None
                try:
                    if getattr(self, 'input_dir', None):
                        p = Path(self.input_dir)
                        if re.match(r'^(chap|chapter|chapitre)\b', p.name.lower()):
                            series_name = p.parent.name
                        else:
                            series_name = p.name
                except Exception:
                    pass
                if not series_name and self.series_db is not None:
                    try:
                        series_name = (
                            getattr(self.series_db, 'name', None)
                            or getattr(self.series_db, 'title', None)
                            or (self.series_db.get('name') if isinstance(self.series_db, dict) else None)
                            or "default"
                        )
                    except Exception:
                        series_name = "default"
                series_name = series_name or "default"
                translator = GeminiTranslator(
                    self.device, series_db=self.series_db, series_name=series_name,
                )
            else:
                from utils import model_context
                from core import NLLBTranslator

                def _create_trans():
                    t = NLLBTranslator(self.device)
                    try:
                        setattr(t, 'series_db', self.series_db)
                    except Exception:
                        pass
                    return t

                translator_cm = model_context(_create_trans)
                translator = translator_cm.__enter__()

        is_gemini = hasattr(translator, 'translate_mega_batch')  # défini AVANT le try pour éviter NameError
        try:

            if is_gemini:
                ch_names = [
                    ch for ch in sorted(chapter_data.keys())
                    if chapter_data[ch]['valid_texts']
                ]

                for batch_start in range(0, len(ch_names), self.MEGA_BATCH_SIZE):
                    batch_chs = ch_names[batch_start:batch_start + self.MEGA_BATCH_SIZE]

                    chapters_payload = {}
                    class_map_payload = {}

                    for ch_name in batch_chs:
                        texts = [entry[2] for entry in chapter_data[ch_name]['valid_texts']]
                        classes = [entry[3] for entry in chapter_data[ch_name]['valid_texts']]
                        chapters_payload[ch_name] = texts
                        class_map_payload[ch_name] = classes

                    total_texts = sum(len(t) for t in chapters_payload.values())
                    self.logger.info(
                        f"   🌐 MEGA-BATCH: {len(batch_chs)} chapitres, "
                        f"{total_texts} textes → 1 appel API"
                    )
                    self.logger.info(f"      Chapitres: {', '.join(batch_chs)}")

                    results = translator.translate_mega_batch(
                        chapters_payload, class_map=class_map_payload,
                    )

                    for ch_name in batch_chs:
                        ch_results = results.get(ch_name, {})
                        valid_texts = chapter_data[ch_name]['valid_texts']
                        dets_map = chapter_data[ch_name]['detections']

                        for out_idx, (img_path, det_idx, orig, _cls) in enumerate(valid_texts):
                            det = dets_map.get(img_path, [])[det_idx]
                            tr = ch_results.get(str(out_idx), "")
                            if hasattr(translator, 'get_font_key'):
                                det.font_key = translator.get_font_key(f"{ch_name}_{out_idx:03d}")
                            if tr:
                                det.text_nllb_raw = tr
                                det.text_translated = tr
                                global_stats['total_translated'] += 1
                            else:
                                det.text_nllb_raw = orig
                                det.text_translated = orig

            else:
                for ch_name in sorted(chapter_data.keys()):
                    valid_texts = chapter_data[ch_name]['valid_texts']
                    if not valid_texts:
                        continue

                    payload = [entry[2] for entry in valid_texts]
                    self.logger.info(f"   📝 {ch_name}: {len(payload)} textes (local)")

                    translations_map = {}
                    TCH = 1000
                    offset = 0
                    while offset < len(payload):
                        block = payload[offset:offset + TCH]
                        res = translator.translate_page_json(block)
                        for k, v in res.items():
                            try:
                                idx = int(k) + offset
                            except Exception:
                                continue
                            translations_map[idx] = v
                        offset += TCH

                    dets_map = chapter_data[ch_name]['detections']
                    for out_idx, (img_path, det_idx, orig, _cls) in enumerate(valid_texts):
                        det = dets_map.get(img_path, [])[det_idx]
                        tr = translations_map.get(out_idx, "")
                        if tr:
                            det.text_nllb_raw = tr
                            det.text_translated = tr
                            global_stats['total_translated'] += 1
                        else:
                            det.text_nllb_raw = orig
                            det.text_translated = orig

            try:
                if hasattr(translator, '_save_cache'):
                    translator._save_cache()
            except Exception:
                pass

        finally:
            if translator_cm is not None:
                translator_cm.__exit__(None, None, None)

        # PHASE 3 : RENDERING (par chapitre, local)
        self.logger.header("🎨 PHASE 3 : RENDERING")
        renderer = TextRenderer()

        for ch_name in sorted(chapter_data.keys()):
            ch_info = chapter_data[ch_name]
            out_ch_dir = ch_info['out_dir']

            self.logger.info(f"   🎨 Rendering: {ch_name}")
            qcheck_report: List[Dict] = []

            for img_path, w, h in ch_info['images']:
                dets = ch_info['detections'].get(img_path, [])
                # Rechargée ici : les images ne sont plus gardées en mémoire
                # entre les phases (cf. phase 1).
                img = cv2.imread(str(img_path))
                if img is None:
                    global_stats['failed'] += 1
                    global_stats['results'].append({
                        'image': img_path.name, 'success': False, 'error': 'load_failed',
                    })
                    continue

                ghost_risk_indices: List[int] = []
                try:
                    valid_dets_for_inpaint = [d for d in dets if getattr(d, 'text_translated', None)]
                    img_translated, _inpaint_sec, ghost_risk_indices = self._run_pre_inpainting(img, valid_dets_for_inpaint, renderer)
                except Exception as exc:
                    self.logger.warning(f"   ⚠️  Inpainting échoué sur {img_path.name}: {exc}")
                    img_translated = img.copy()

                # QCheck : chaque détection dont l'effacement est resté
                # imparfait (fantôme résiduel, cf. `_apply_erasure_by_group`)
                # — aucun signal fiable ici pour réparer automatiquement sans
                # risquer pire (le fond n'est pas sûr à diffuser, par
                # construction), donc on se contente de le SIGNALER pour une
                # retouche manuelle ciblée plutôt que de relire toute la page.
                for idx in ghost_risk_indices:
                    if idx >= len(dets):
                        continue
                    d = dets[idx]
                    qcheck_report.append({
                        'page': img_path.name,
                        'class': getattr(d, 'class_name', ''),
                        'bbox': [d.x1, d.y1, d.x2, d.y2],
                        'type': 'ghost_residual',
                        'detail': "Effacement imparfait (LaMa a échoué, fond jugé trop texturé pour diffuser) — vérifier visuellement.",
                    })

                for det in dets:
                    if not getattr(det, 'text_translated', None):
                        continue

                    self._prepare_render_style(renderer, img, det)
                    img_translated = renderer.insert_text(
                        img_translated,
                        det.text_translated,
                        det.x1, det.y1, det.x2, det.y2,
                        text_regions=getattr(det, 'text_regions', None),
                        text_color_rgb=getattr(det, 'text_color_rgb', None),
                        text_style=getattr(det, 'text_style', 'dialogue'),
                        font_hint=getattr(det, 'font_hint', 'regular'),
                        class_name=getattr(det, 'class_name', ''),
                        bubble_mask=getattr(det, 'mask_binary', None),
                        font_key=getattr(det, 'font_key', None),
                        source_line_height=getattr(det, 'source_line_height', None),
                        sibling_boxes=[
                            (d.x1, d.y1, d.x2, d.y2) for d in dets if d is not det
                        ],
                    )

                # ── QCheck auto-repair post-render ──────────────────────────
                img_to_save = img_translated
                if QCheckEngine is not None:
                    try:
                        _translated_dets = [d for d in dets if getattr(d, 'text_translated', None)]
                        if _translated_dets:
                            qcheck_engine = QCheckEngine(
                                font_paths=getattr(renderer.cfg, 'font_paths', [])
                            )
                            img_fixed, qcheck_issues = qcheck_engine.check_and_repair(
                                img_before=img,
                                img_after=img_translated,
                                detections=_translated_dets,
                                renderer=renderer,
                            )
                            if qcheck_issues:
                                summary = QCheckEngine.format_summary(qcheck_issues)
                                self.logger.info(f"      {summary}")
                                if any(i.repaired for i in qcheck_issues):
                                    img_to_save = img_fixed
                                QCheckEngine.save_report(
                                    qcheck_issues,
                                    out_ch_dir / 'qcheck_report.json',
                                    page_name=img_path.stem,
                                )
                                # Ajouter à l'ancien rapport pour compatibilité
                                for issue in qcheck_issues:
                                    if not issue.repaired:
                                        qcheck_report.append({
                                            'page': img_path.name,
                                            'class': issue.class_name,
                                            'bbox': list(issue.bbox),
                                            'type': issue.issue_type,
                                            'detail': issue.detail,
                                        })
                    except Exception as qcheck_err:
                        self.logger.warning(f"      ⚠️  QCheck ignoré: {qcheck_err}")

                output_path = self._write_output_image(out_ch_dir, img_path.stem, img_to_save)
                global_stats['processed'] += 1
                global_stats['results'].append({
                    'image': img_path.name,
                    'success': True,
                    'output': output_path.name,
                    'detections': len(dets),
                    'translated': sum(1 for d in dets if getattr(d, 'text_translated', None)),
                })

                img = None
                img_translated = None
                img_to_save = None
                gc.collect()

            if qcheck_report:
                qcheck_path = out_ch_dir / 'qcheck_report.json'
                with open(qcheck_path, 'w', encoding='utf-8') as f:
                    json.dump(qcheck_report, f, ensure_ascii=False, indent=2)
                self.logger.warning(
                    f"   🔍 QCheck: {len(qcheck_report)} détection(s) à revérifier manuellement "
                    f"dans {ch_name} → {qcheck_path.name}"
                )

        global_stats['total_time_seconds'] = time.time() - start_time

        # `gemini_stats` n'était défini que dans la branche du dessus : si le
        # traducteur n'exposait pas `stats()`, le récapitulatif plus bas levait
        # un NameError en toute fin de run, avant l'écriture de summary.json —
        # tout le travail était perdu.
        gemini_stats: Dict = {}
        if is_gemini and hasattr(translator, 'stats'):
            try:
                gemini_stats = translator.stats() or {}
            except Exception as exc:
                self.logger.warning(f"   ⚠️  Stats Gemini indisponibles: {exc}")
            self.logger.info("\n   📊 Gemini Stats:")
            self.logger.info(f"      API calls: {gemini_stats.get('api_calls', 0)}")
            self.logger.info(f"      Cache hits: {gemini_stats.get('cache_hits', 0)}")
            self.logger.info(f"      Fallbacks: {gemini_stats.get('fallback_count', 0)}")

        self.logger.header("🎉 TRAITEMENT TERMINÉ")
        self.logger.summary({
            'Chapitres': len(chapter_data),
            'Images traitées': f"{global_stats['processed']}/{global_stats['total_images']}",
            'Détections': global_stats['total_detections'],
            'Traductions': global_stats['total_translated'],
            'Échecs': global_stats['failed'],
            'Temps total': f"{global_stats['total_time_seconds']:.1f}s",
            'Appels API Gemini': gemini_stats.get('api_calls', '?') if is_gemini else 'N/A (local)',
        })

        summary_path = output_dir / "summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(global_stats, f, ensure_ascii=False, indent=2)

        return global_stats
    
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

        detections = self._detect_ensemble(detection_img, logger=self.logger)
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

                # Chemin partagé avec le mode chapitre : filtres de bruit,
                # remise à l'échelle des polygones OCR, segmentation et masque
                # chirurgical. Ce bloc était auparavant dupliqué et divergent
                # entre les deux modes.
                skip_reason, seg_seconds = self._apply_ocr_result(img, det, ocr_result)
                timings['sam2_seconds'] += seg_seconds

                if skip_reason:
                    stats['skipped'] += 1
                    stats['skip_reasons'][skip_reason] = stats['skip_reasons'].get(skip_reason, 0) + 1
                    preview = (ocr_result[0] or "").replace("\n", " ")[:120]
                    self.logger.info(
                        f"      ⚠️  Ignoré: reason={skip_reason} "
                        f"conf={det.ocr_confidence:.2f} det={idx} text='{preview}'"
                    )
                    continue

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

                self.logger.info(
                    f"      OK \"{det.text_original}\" ({det.ocr_confidence:.0%}) "
                    f"[{len(det.text_regions)} régions, seg={getattr(det, 'seg_backend', '?')}]"
                )
        
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
                def _create_trans():
                    backend = str(getattr(config.translation, 'backend', 'local_llm')).lower()
                    if backend == "gemini" and GeminiTranslator is not None:
                        # Try to infer series_name from pipeline input_dir first
                        series_name = None
                        try:
                            if getattr(self, 'input_dir', None):
                                p = Path(self.input_dir)
                                # If path looks like a chapter (chapitre XXX), use parent folder as series
                                if re.match(r'^(chap|chapter|chapitre)\b', p.name.lower()):
                                    series_name = p.parent.name
                                elif p.is_file():
                                    series_name = p.parent.name
                                else:
                                    series_name = p.name
                        except Exception:
                            series_name = None

                        # Fallback to series_db metadata if available
                        if not series_name and self.series_db is not None:
                            try:
                                if hasattr(self.series_db, 'name'):
                                    series_name = getattr(self.series_db, 'name')
                                elif hasattr(self.series_db, 'title'):
                                    series_name = getattr(self.series_db, 'title')
                                elif isinstance(self.series_db, dict):
                                    series_name = self.series_db.get('name') or self.series_db.get('title')
                            except Exception:
                                series_name = None

                        series_name = (series_name or "default")
                        return GeminiTranslator(self.device, series_db=self.series_db, series_name=series_name)
                    t = NLLBTranslator(self.device)
                    try:
                        setattr(t, 'series_db', self.series_db)
                    except Exception:
                        pass
                    return t

                translator_cm = model_context(_create_trans)
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
                        # Récupérer le font_key choisi par le LLM
                        if hasattr(translator, 'get_font_key'):
                            det.font_key = translator.get_font_key(str(det_idx))

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
                    # Les cartes System ont leur propre police dans FONT_MAP.
                    # Cette boucle ne passait pas par la sélection de font_key
                    # (réservée à `regular_detections`) : elles retombaient sur
                    # l'heuristique par mots-clés de dossier.
                    det.font_key = "SYSTEM"
                
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

            self._prepare_render_style(renderer, img, det)

            self.logger.info(
                f"   [{i+1}/{len(valid_detections)}] bbox=({det.x1},{det.y1},{det.x2},{det.y2}) class={det.class_name}"
            )

            before_crop = None
            if self.debug:
                before_crop = img_translated[det.y1:det.y2, det.x1:det.x2].copy()

            # Le masque d'effacement doit être celui du TEXTE (polygones OCR),
            # pas celui de la bulle : `mask_regions` vient de la segmentation et
            # couvre tout l'intérieur de la bulle — l'utiliser repeindrait la
            # bulle entière.
            effective_regions = getattr(det, 'text_regions', None) or getattr(det, 'mask_regions', None)

            # Inpainting
            inpaint_t0 = time.perf_counter()
            img_translated = renderer.inpaint_region(
                img_translated,
                det.x1, det.y1, det.x2, det.y2,
                text_regions=effective_regions,
                class_name=getattr(det, 'class_name', ''),
                chirurgical_mask=getattr(det, 'chirurgical_mask', None),
                bubble_mask=getattr(det, 'mask_binary', None),
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
                bubble_mask=getattr(det, 'mask_binary', None),
                font_key=getattr(det, 'font_key', None),
                source_line_height=getattr(det, 'source_line_height', None),
                sibling_boxes=[
                    (d.x1, d.y1, d.x2, d.y2) for d in valid_detections if d is not det
                ],
            )
            timings['text_render_seconds'] += max(0.0, time.perf_counter() - render_t0)

            if self.debug and before_crop is not None:
                after_crop = img_translated[det.y1:det.y2, det.x1:det.x2].copy()
                self.save_debug_render_bundle(output_dir, image_stem, i + 1, before_crop, after_crop, det)

        if self.debug:
            self.save_debug_render_overview(output_dir, image_stem, img, img_translated, valid_detections)

        # QCheck auto-repair post-render : filet de sécurité, non destructif
        # (ne remplace l'image que si au moins une réparation a réellement eu
        # lieu), mêmes catégories que le mode série/mega-batch.
        img_to_save = img_translated
        if QCheckEngine is not None and valid_detections:
            try:
                qcheck_engine = QCheckEngine(font_paths=getattr(renderer.cfg, 'font_paths', []))
                img_fixed, qcheck_issues = qcheck_engine.check_and_repair(
                    img_before=img,
                    img_after=img_translated,
                    detections=valid_detections,
                    renderer=renderer,
                )
                if qcheck_issues:
                    self.logger.info(f"   {QCheckEngine.format_summary(qcheck_issues)}")
                    if any(i.repaired for i in qcheck_issues):
                        img_to_save = img_fixed
                    QCheckEngine.save_report(
                        qcheck_issues, output_dir / 'qcheck_report.json', page_name=image_stem,
                    )
            except Exception as qcheck_err:
                self.logger.warning(f"   ⚠️  QCheck ignoré: {qcheck_err}")

        # Sauvegarder
        output_image_path = self._write_output_image(output_dir, image_stem, img_to_save)

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
        # Store the current input_dir to allow series name inference elsewhere
        self.input_dir = input_dir
        image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        image_files = [
            f for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        # Si pas d'images au niveau racine, chercher dans les sous-dossiers (chapitres)
        if not image_files:
            # Découvrir les sous-dossiers contenant des images
            chapter_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
            all_images = []
            chapter_map = {}
            for ch in chapter_dirs:
                imgs = [f for f in ch.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
                if imgs:
                    chapter_map[ch.name] = imgs
                    all_images.extend(imgs)

            if not all_images:
                self.logger.error(f"Aucune image dans {input_dir}")
                return {'success': False, 'error': 'no_images'}

            # Traiter chapitre par chapitre (mega-batch mode)
            return self._process_chapters_mega_batch(chapter_map, output_dir)
        
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