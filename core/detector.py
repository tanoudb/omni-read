"""
═══════════════════════════════════════════════════════════════════════════════
DETECTOR PREMIUM - YOLO avec slicing adaptatif & multi-scale
═══════════════════════════════════════════════════════════════════════════════

FIX v2: containment filter intra-classe + inter-classe
"""

import numpy as np
import cv2
import torch
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from collections import defaultdict

from config.settings import config
from utils.image_utils import ImageUtils
from utils.filters import GeometricFilter


class Detection:
    """Classe pour stocker une détection"""
    
    def __init__(self, class_name: str, bbox: List[float], score: float, 
                 scale: float = 1.0, metadata: Optional[Dict] = None):
        self.class_name = class_name
        self.bbox = bbox
        self.score = score
        self.scale = scale
        self.metadata = metadata or {}
        
        self.text_original: Optional[str] = None
        self.text_translated: Optional[str] = None
        self.text_nllb_raw: Optional[str] = None
        self.ocr_confidence: float = 0.0
        self.text_regions: List[Dict] = []  # bbox OCR pour inpainting précis
        self.ocr_upscale_factor: float = 1.0
    @property
    def x1(self) -> int:
        return int(self.bbox[0])
    
    @property
    def y1(self) -> int:
        return int(self.bbox[1])
    
    @property
    def x2(self) -> int:
        return int(self.bbox[2])
    
    @property
    def y2(self) -> int:
        return int(self.bbox[3])
    
    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    def __repr__(self):
        return f"Detection({self.class_name}, score={self.score:.2f}, bbox={self.bbox})"


class YOLODetector:
    """Détecteur YOLO premium avec slicing adaptatif et multi-scale"""
    
    def __init__(self, model_path: Path, device: str = 'cuda'):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.cfg = config.detection
        self.geo_filter = GeometricFilter(
            top_threshold=config.filters.top_edge_threshold,
            bottom_threshold=config.filters.bottom_edge_threshold
        )
        self.last_debug_report: Dict = {}
        
        self._load_model()
    
    def _load_model(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
            if self.device == 'cuda':
                self.model.to('cuda')
                self.model.model.float()
                # Note: fp16 disabled for YOLO due to dtype mismatch with uint8 inputs
                # YOLO's fp16 implementation requires proper input format handling
            self.model.model.eval()
        except Exception as e:
            raise RuntimeError(f"Erreur chargement YOLO: {e}")

    def _recover_model_after_dtype_error(self):
        """Recharge proprement le modèle en FP32 après une erreur dtype Ultralytics."""
        try:
            if self.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self._load_model()
        try:
            if hasattr(self.model, 'predictor'):
                self.model.predictor = None
        except Exception:
            pass

    @staticmethod
    def _is_dtype_mismatch_error(error: Exception) -> bool:
        message = str(error)
        patterns = [
            'float != struct c10::Half',
            'expected mat1 and mat2 to have the same dtype',
            'Expected all tensors to be on the same device',
        ]
        return any(pattern in message for pattern in patterns)
    
    # ─────────────────────────────────────────────────────────────────────────
    # SLICING ADAPTATIF
    # ─────────────────────────────────────────────────────────────────────────
    
    def calculate_window_size(self, image_width: int) -> int:
        if not self.cfg.auto_calibrate_window:
            return self.cfg.base_window_height
        base_width = 800
        base_height = self.cfg.base_window_height
        ratio = image_width / base_width
        window_height = int(base_height * ratio)
        return max(self.cfg.min_window_height, min(window_height, self.cfg.max_window_height))
    
    @staticmethod
    def _det_key(det: Detection) -> Tuple[str, int, int, int, int, int]:
        return (
            str(det.class_name),
            int(round(det.score * 10000)),
            int(det.x1),
            int(det.y1),
            int(det.x2),
            int(det.y2),
        )

    def sliding_window_detect(
        self,
        image: np.ndarray,
        scale: float = 1.0,
        logger=None,
        debug_stats: Optional[Dict] = None,
        conf_reject_events: Optional[List[Dict]] = None,
    ) -> List[Detection]:
        h, w = image.shape[:2]
        
        if self.cfg.enable_adaptive_slicing:
            window_height = self.calculate_window_size(w)
        else:
            window_height = self.cfg.base_window_height
        
        if scale != 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            image_scaled = cv2.resize(image, (new_w, new_h))
        else:
            image_scaled = image
            new_h, new_w = h, w
        
        overlap = int(window_height * self.cfg.overlap_ratio)
        step = window_height - overlap
        detections = []
        scale_key = f"{float(scale):.2f}"

        if debug_stats is not None and scale_key not in debug_stats:
            debug_stats[scale_key] = {
                'raw': 0,
                'kept': 0,
                'threshold_reject': 0,
                'threshold_reject_by_class': defaultdict(int),
                'border_reject': 0,
            }
        
        for y in range(0, new_h, step):
            y_end = min(y + window_height, new_h)
            window = image_scaled[y:y_end, :]

            try:
                results = self.model.predict(
                    window,
                    conf=0.15,
                    verbose=False,
                    device=self.device,
                )
            except RuntimeError as pred_error:
                if not self._is_dtype_mismatch_error(pred_error):
                    raise
                if logger:
                    logger.warning("      ⚠️  Dtype/device mismatch YOLO détecté, reload modèle et retry...")
                self._recover_model_after_dtype_error()
                results = self.model.predict(
                    window,
                    conf=0.15,
                    verbose=False,
                    half=False,
                    device=self.device,
                )
            
            for result in results:
                for box in result.boxes:
                    x1, y1_local, x2, y2_local = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = result.names[cls_id]
                    if debug_stats is not None:
                        debug_stats[scale_key]['raw'] += 1

                    conf_threshold = float(self.cfg.confidence_thresholds.get(cls_name, 0.25))
                    
                    if conf < conf_threshold:
                        if scale != 1.0:
                            rej_x1 = x1 / scale
                            rej_x2 = x2 / scale
                            rej_y1 = (y1_local + y) / scale
                            rej_y2 = (y2_local + y) / scale
                        else:
                            rej_x1 = x1
                            rej_x2 = x2
                            rej_y1 = y1_local + y
                            rej_y2 = y2_local + y

                        if conf_reject_events is not None:
                            conf_reject_events.append({
                                'stage': 'confidence',
                                'class_name': cls_name,
                                'scale': float(scale),
                                'score': round(float(conf), 4),
                                'threshold': round(float(conf_threshold), 4),
                                'bbox': [int(rej_x1), int(rej_y1), int(rej_x2), int(rej_y2)],
                            })
                        if debug_stats is not None:
                            debug_stats[scale_key]['threshold_reject'] += 1
                            debug_stats[scale_key]['threshold_reject_by_class'][cls_name] += 1
                        continue
                    
                    if scale != 1.0:
                        x1 = x1 / scale
                        x2 = x2 / scale
                        y1_global = (y1_local + y) / scale
                        y2_global = (y2_local + y) / scale
                    else:
                        y1_global = y1_local + y
                        y2_global = y2_local + y
                    
                    bbox = [float(x1), float(y1_global), float(x2), float(y2_global)]
                    
                    if self.cfg.filter_border_detections:
                        margin = self.cfg.border_margin_px
                        too_close_top = (y > 0) and ((y1_local) < margin)
                        too_close_bottom = (y_end < new_h) and ((window_height - y2_local) < margin)
                        if too_close_top or too_close_bottom:
                            if debug_stats is not None:
                                debug_stats[scale_key]['border_reject'] += 1
                            continue
                    
                    detections.append(Detection(
                        class_name=cls_name, bbox=bbox, score=conf,
                        scale=scale, metadata={'window_y': y}
                    ))
                    if debug_stats is not None:
                        debug_stats[scale_key]['kept'] += 1

        if logger and debug_stats is not None:
            stats = debug_stats.get(scale_key, {})
            by_class = dict(stats.get('threshold_reject_by_class', {}))
            logger.info(
                f"      [scale {scale_key}] raw={stats.get('raw', 0)} | kept={stats.get('kept', 0)} "
                f"| rejet_conf={stats.get('threshold_reject', 0)} | rejet_border={stats.get('border_reject', 0)}"
            )
            if by_class:
                logger.info(f"      [scale {scale_key}] rejet_conf_par_classe={by_class}")
        
        return detections
    
    # ─────────────────────────────────────────────────────────────────────────
    # MULTI-SCALE
    # ─────────────────────────────────────────────────────────────────────────
    
    def multi_scale_detect(self, image: np.ndarray, logger=None, debug_stats: Optional[Dict] = None) -> List[Detection]:
        all_detections = []
        for scale in self.cfg.detection_scales:
            all_detections.extend(
                self.sliding_window_detect(
                    image,
                    scale=scale,
                    logger=logger,
                    debug_stats=debug_stats,
                    conf_reject_events=self.last_debug_report.setdefault('confidence_rejects', []),
                )
            )
        return all_detections
    
    # ─────────────────────────────────────────────────────────────────────────
    # NMS
    # ─────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def _recover_truncated_boxes(
        detections: List[Detection], raw_detections: List[Detection],
        min_containment: float = 0.92, max_growth: float = 2.2,
    ) -> None:
        """Réétend une boîte finale TRONQUÉE, d'après les boîtes brutes.

        Le NMS ne garde que le meilleur SCORE, or le score ne dit rien de la
        complétude. Mesuré sur the-frontier-count ch1 : la même bulle sortait
        en [113,5463,339,5589] (h=126, score 0,909) et en [111,5373,340,5591]
        (h=218, score 0,903) ; le NMS gardait la tronquée, et la ligne
        « I PREPARED » — hors boîte — n'était ni lue, ni effacée, ni
        retraduite. Le texte anglais restait visible dans la bulle avec la
        traduction dessous, dans une autre police.

        Corriger ça DANS le NMS ne marche pas : agrandir une gagnante en cours
        de boucle élargit mécaniquement les comparaisons suivantes, ici comme
        aux étapes de NMS d'après, et le voisinage se fait avaler — essayé, et
        path-of-vengeance tombait de 37 à 34 détections, trois paires de bulles
        soudées. On agit donc APRÈS toutes les étapes de suppression, sur les
        boîtes finales, sans influencer aucune décision.

        Une boîte n'est étendue que vers une boîte brute qui la CONTIENT
        presque entièrement (92 %) et qui reste de taille comparable : c'est la
        signature « deux vues du même objet, l'une coupée », que deux bulles
        voisines n'ont jamais (mesuré : 0,31 de containment).
        """
        for det in detections:
            try:
                dx1, dy1, dx2, dy2 = (float(v) for v in det.bbox)
                area_det = max(1.0, (dx2 - dx1) * (dy2 - dy1))
                best_box = None
                best_area = area_det
                for raw in raw_detections:
                    if raw.class_name != det.class_name:
                        continue
                    rx1, ry1, rx2, ry2 = (float(v) for v in raw.bbox)
                    area_raw = max(1.0, (rx2 - rx1) * (ry2 - ry1))
                    if area_raw <= best_area or area_raw > max_growth * area_det:
                        continue
                    inter = (
                        max(0.0, min(dx2, rx2) - max(dx1, rx1))
                        * max(0.0, min(dy2, ry2) - max(dy1, ry1))
                    )
                    if (inter / area_det) < min_containment:
                        continue
                    best_box, best_area = [rx1, ry1, rx2, ry2], area_raw
                if best_box is None:
                    continue
                # Ne jamais s'étendre SUR une autre détection : la boîte
                # élargie serait ensuite fusionnée avec elle par le
                # dédoublonnage du pipeline, et on perdrait une bulle — deux
                # textes dans la même, l'autre laissée en anglais. Mesuré :
                # sans cette garde, path-of-vengeance passait de 37 à 35
                # détections.
                ex1, ey1, ex2, ey2 = best_box
                clash = False
                for other in detections:
                    if other is det or other.class_name != det.class_name:
                        continue
                    ox1, oy1, ox2, oy2 = (float(v) for v in other.bbox)
                    area_other = max(1.0, (ox2 - ox1) * (oy2 - oy1))
                    inter_new = (
                        max(0.0, min(ex2, ox2) - max(ex1, ox1))
                        * max(0.0, min(ey2, oy2) - max(ey1, oy1))
                    )
                    inter_old = (
                        max(0.0, min(dx2, ox2) - max(dx1, ox1))
                        * max(0.0, min(dy2, oy2) - max(dy1, oy1))
                    )
                    if inter_new > inter_old and (inter_new / area_other) > 0.25:
                        clash = True
                        break
                if not clash:
                    det.bbox = best_box
            except Exception:
                continue

    def nms_per_class(self, detections: List[Detection], debug_events: Optional[List[Dict]] = None) -> List[Detection]:
        by_class = defaultdict(list)
        for det in detections:
            by_class[det.class_name].append(det)
        
        kept = []
        for cls_name, dets in by_class.items():
            dets = sorted(dets, key=lambda x: x.score, reverse=True)
            iou_thresh = self.cfg.nms_iou_thresholds.get(cls_name, 0.5)
            
            keep = []
            while dets:
                best = dets.pop(0)
                keep.append(best)
                remaining = []
                for d in dets:
                    iou = ImageUtils.calculate_iou(best.bbox, d.bbox)
                    if iou < iou_thresh:
                        remaining.append(d)
                    elif debug_events is not None:
                        debug_events.append({
                            'stage': 'nms_per_class',
                            'class_name': cls_name,
                            'iou': round(float(iou), 3),
                            'threshold': float(iou_thresh),
                            'kept_score': round(float(best.score), 3),
                            'removed_score': round(float(d.score), 3),
                            'removed_bbox': [int(d.x1), int(d.y1), int(d.x2), int(d.y2)],
                        })
                dets = remaining
            kept.extend(keep)
        
        return kept
    
    def multi_scale_nms(self, detections: List[Detection], debug_events: Optional[List[Dict]] = None) -> List[Detection]:
        if not detections:
            return []
        detections = sorted(detections, key=lambda x: x.score, reverse=True)
        kept = []
        iou_thresh = self.cfg.multi_scale_nms_iou
        while detections:
            best = detections.pop(0)
            kept.append(best)
            remaining = []
            for d in detections:
                iou = ImageUtils.calculate_iou(best.bbox, d.bbox)
                if iou < iou_thresh:
                    remaining.append(d)
                elif debug_events is not None:
                    debug_events.append({
                        'stage': 'nms_multi_scale',
                        'iou': round(float(iou), 3),
                        'threshold': float(iou_thresh),
                        'kept_scale': float(getattr(best, 'scale', 1.0)),
                        'removed_scale': float(getattr(d, 'scale', 1.0)),
                        'kept_score': round(float(best.score), 3),
                        'removed_score': round(float(d.score), 3),
                        'removed_bbox': [int(d.x1), int(d.y1), int(d.x2), int(d.y2)],
                    })
            detections = remaining
        return kept
    
    # ─────────────────────────────────────────────────────────────────────────
    # CONTAINMENT + CONFLICT RESOLUTION
    # ─────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def _containment_ratio(inner_bbox, outer_bbox):
        """Quelle fraction de inner est contenue dans outer (0-1)"""
        ix1 = max(inner_bbox[0], outer_bbox[0])
        iy1 = max(inner_bbox[1], outer_bbox[1])
        ix2 = min(inner_bbox[2], outer_bbox[2])
        iy2 = min(inner_bbox[3], outer_bbox[3])
        
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        
        intersection = (ix2 - ix1) * (iy2 - iy1)
        inner_area = (inner_bbox[2] - inner_bbox[0]) * (inner_bbox[3] - inner_bbox[1])
        return intersection / inner_area if inner_area > 0 else 0.0
    
    def remove_contained_boxes(self, detections: List[Detection], debug_events: Optional[List[Dict]] = None) -> List[Detection]:
        """
        Supprime les bbox contenues dans d'autres (TOUTES classes confondues).
        
        Si bbox A est contenue à >75% dans bbox B :
        - Garder B (la plus grande), supprimer A
        
        Ceci gère les cas comme :
        - [10] "OR HAS SUBPAR ADMINISTRATION" (grande) contient [14] "OR HAS SUBPAR" (petite)
        - [4] "A TRASH GAME" (grande) contient [11] "UNMML:" (bout de bulle)
        """
        if len(detections) < 2:
            return detections
        
        to_remove = set()
        
        for i in range(len(detections)):
            if i in to_remove:
                continue
            for j in range(len(detections)):
                if i == j or j in to_remove:
                    continue
                
                # Est-ce que j est contenu dans i ?
                ratio = self._containment_ratio(detections[j].bbox, detections[i].bbox)
                if ratio > 0.75:
                    # Ce filtre arbitrait UNIQUEMENT sur la géométrie : le score
                    # n'entrait jamais dans la décision. Une détection médiocre
                    # et mal placée pouvait donc avaler une détection
                    # excellente. Mesuré sur la bulle ronde de
                    # i-married-the-dragon : un `bulle` à 0,337, décalé vers la
                    # droite et ratant le tiers gauche du cercle, absorbait un
                    # `out_text` à 0,721 qui, lui, encadrait correctement le
                    # texte. Résultat : 79 px de texte hors de la boîte
                    # retenue, jamais effacés — les résidus « HAT » et « DON'T »
                    # visibles dans le rendu final.
                    #
                    # On INVERSE plutôt que de refuser : refuser laisserait les
                    # deux boîtes vivre, et leur IoU (0,306 mesuré) est sous le
                    # seuil inter-classes de 0,5, donc rien ne les
                    # départagerait ensuite — le texte serait traduit et rendu
                    # DEUX fois, superposé. C'est le défaut que documente déjà
                    # `config/settings.py::inter_class_iou_threshold`.
                    #
                    # Le facteur 1,5 : dans le cas légitime que ce filtre vise
                    # — un fragment redondant à l'intérieur d'une vraie
                    # détection — le conteneur est au moins aussi confiant que
                    # le fragment. Un conteneur nettement MOINS confiant est la
                    # signature de l'inverse. Ici le rapport vaut 2,1.
                    if detections[j].score > detections[i].score * 1.5:
                        to_remove.add(i)
                        if debug_events is not None:
                            debug_events.append({
                                'stage': 'containment_inverse',
                                'ratio': round(float(ratio), 3),
                                'keeper_class': detections[j].class_name,
                                'keeper_score': round(float(detections[j].score), 3),
                                'removed_class': detections[i].class_name,
                                'removed_score': round(float(detections[i].score), 3),
                                'removed_bbox': [int(detections[i].x1), int(detections[i].y1),
                                                 int(detections[i].x2), int(detections[i].y2)],
                            })
                        break

                    # j est contenu dans i → supprimer j (le plus petit)
                    to_remove.add(j)
                    if debug_events is not None:
                        debug_events.append({
                            'stage': 'containment',
                            'ratio': round(float(ratio), 3),
                            'keeper_class': detections[i].class_name,
                            'keeper_score': round(float(detections[i].score), 3),
                            'removed_class': detections[j].class_name,
                            'removed_score': round(float(detections[j].score), 3),
                            'removed_bbox': [int(detections[j].x1), int(detections[j].y1), int(detections[j].x2), int(detections[j].y2)],
                        })
        
        kept = [d for i, d in enumerate(detections) if i not in to_remove]
        
        return kept
    
    def resolve_inter_class_conflicts(self, detections: List[Detection]) -> List[Detection]:
        """Résout les conflits IoU entre classes différentes"""
        to_remove = set()
        
        for i in range(len(detections)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(detections)):
                if j in to_remove:
                    continue
                if detections[i].class_name == detections[j].class_name:
                    continue
                
                iou = ImageUtils.calculate_iou(detections[i].bbox, detections[j].bbox)
                if iou > self.cfg.inter_class_iou_threshold:
                    if detections[i].score > detections[j].score:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break
        
        return [d for i, d in enumerate(detections) if i not in to_remove]
    
    # ─────────────────────────────────────────────────────────────────────────
    # FILTRAGE
    # ─────────────────────────────────────────────────────────────────────────
    
    def filter_detections(
        self,
        detections: List[Detection],
        image_shape: Tuple[int, int],
        debug_events: Optional[List[Dict]] = None,
    ) -> List[Detection]:
        filtered = []
        for det in detections:
            if not ImageUtils.is_valid_bbox(
                det.x1, det.y1, det.x2, det.y2, image_shape,
                min_area=self.cfg.min_box_area, max_area=self.cfg.max_box_area,
                min_ratio=self.cfg.min_box_ratio, max_ratio=self.cfg.max_box_ratio
            ):
                if debug_events is not None:
                    debug_events.append({
                        'stage': 'geometry',
                        'reason': 'invalid_bbox',
                        'class_name': det.class_name,
                        'score': round(float(det.score), 3),
                        'bbox': [int(det.x1), int(det.y1), int(det.x2), int(det.y2)],
                    })
                continue
            if config.filters.filter_top_edge and self.geo_filter.is_on_top_edge(det.y1, image_shape[0]):
                if debug_events is not None:
                    debug_events.append({
                        'stage': 'geometry',
                        'reason': 'top_edge',
                        'class_name': det.class_name,
                        'score': round(float(det.score), 3),
                        'bbox': [int(det.x1), int(det.y1), int(det.x2), int(det.y2)],
                    })
                continue
            if config.filters.filter_bottom_edge and self.geo_filter.is_on_bottom_edge(det.y2, image_shape[0]):
                if debug_events is not None:
                    debug_events.append({
                        'stage': 'geometry',
                        'reason': 'bottom_edge',
                        'class_name': det.class_name,
                        'score': round(float(det.score), 3),
                        'bbox': [int(det.x1), int(det.y1), int(det.x2), int(det.y2)],
                    })
                continue
            filtered.append(det)
        return filtered
    
    # ─────────────────────────────────────────────────────────────────────────
    # PIPELINE PRINCIPAL
    # ─────────────────────────────────────────────────────────────────────────
    
    def detect(self, image: np.ndarray, logger=None) -> List[Detection]:
        h, w = image.shape[:2]
        scale_debug_stats: Dict[str, Dict] = {}
        nms_debug: List[Dict] = []
        containment_debug: List[Dict] = []
        geometry_debug: List[Dict] = []
        multi_nms_debug: List[Dict] = []
        confidence_debug: List[Dict] = []
        self.last_debug_report = {
            'image_shape': {'width': int(w), 'height': int(h)},
            'scale_debug_stats': {},
            'raw_detections': [],
            'final_detections': [],
            'confidence_rejects': confidence_debug,
            'nms_per_class_rejects': nms_debug,
            'nms_multi_scale_rejects': multi_nms_debug,
            'containment_rejects': containment_debug,
            'geometry_rejects': geometry_debug,
        }
        
        if logger:
            logger.info(f"   🔍 Détection sur {w}x{h}px")
        
        # 1. Multi-scale
        if self.cfg.enable_multi_scale:
            if logger:
                logger.info(f"      Multi-scale: {self.cfg.detection_scales}")
            detections = self.multi_scale_detect(image, logger=logger, debug_stats=scale_debug_stats)
        else:
            detections = self.sliding_window_detect(
                image,
                scale=1.0,
                logger=logger,
                debug_stats=scale_debug_stats,
                conf_reject_events=confidence_debug,
            )

        raw_detections = list(detections)
        self.last_debug_report['raw_detections'] = [
            {
                'class_name': d.class_name,
                'score': round(float(d.score), 4),
                'scale': float(getattr(d, 'scale', 1.0)),
                'bbox': [int(d.x1), int(d.y1), int(d.x2), int(d.y2)],
            }
            for d in raw_detections
        ]
        self.last_debug_report['scale_debug_stats'] = {
            str(k): {
                'raw': int(v.get('raw', 0)),
                'kept': int(v.get('kept', 0)),
                'threshold_reject': int(v.get('threshold_reject', 0)),
                'threshold_reject_by_class': dict(v.get('threshold_reject_by_class', {})),
                'border_reject': int(v.get('border_reject', 0)),
            }
            for k, v in scale_debug_stats.items()
        }
        
        if logger:
            logger.info(f"      → {len(detections)} détections brutes")
            if detections:
                by_scale = defaultdict(int)
                for det in detections:
                    by_scale[f"{float(getattr(det, 'scale', 1.0)):.2f}"] += 1
                logger.info(f"      📐 Brutes par scale: {dict(sorted(by_scale.items()))}")
        
        # 2. NMS par classe
        detections = self.nms_per_class(detections, debug_events=nms_debug)
        if logger:
            logger.info(f"      → {len(detections)} après NMS par classe")
            if nms_debug:
                logger.info(f"      🔎 Rejets NMS classe: {len(nms_debug)}")
                for event in nms_debug[:5]:
                    logger.info(
                        f"         - {event['class_name']} conf {event['removed_score']:.2f} rejeté "
                        f"(iou={event['iou']:.2f} ≥ {event['threshold']:.2f}) bbox={event['removed_bbox']}"
                    )
        
        # 3. NMS multi-échelle
        if self.cfg.enable_multi_scale:
            detections = self.multi_scale_nms(detections, debug_events=multi_nms_debug)
            if logger:
                logger.info(f"      → {len(detections)} après NMS multi-échelle")
                if multi_nms_debug:
                    logger.info(f"      🔎 Rejets NMS multi-scale: {len(multi_nms_debug)}")
                    for event in multi_nms_debug[:5]:
                        logger.info(
                            f"         - scale {event['removed_scale']:.2f} conf {event['removed_score']:.2f} rejeté "
                            f"(iou={event['iou']:.2f} ≥ {event['threshold']:.2f}) bbox={event['removed_bbox']}"
                        )
                if detections:
                    by_scale_after = defaultdict(int)
                    for det in detections:
                        by_scale_after[f"{float(getattr(det, 'scale', 1.0)):.2f}"] += 1
                    logger.info(f"      📐 Après NMS multi-scale: {dict(sorted(by_scale_after.items()))}")
        
        # 4. Suppression containment (NOUVEAU - toutes classes)
        before = len(detections)
        detections = self.remove_contained_boxes(detections, debug_events=containment_debug)
        if logger and before != len(detections):
            logger.info(f"      → {len(detections)} après suppression containment ({before - len(detections)} doublons)")
            for event in containment_debug[:5]:
                logger.info(
                    f"         - containment: {event['removed_class']} conf {event['removed_score']:.2f} rejeté "
                    f"(ratio={event['ratio']:.2f}) bbox={event['removed_bbox']}"
                )
        
        # 5. Conflits inter-classes
        before = len(detections)
        detections = self.resolve_inter_class_conflicts(detections)
        if logger and before != len(detections):
            logger.info(f"      → {len(detections)} après résolution conflits")
        
        # 6. Filtrage géométrique
        before = len(detections)
        detections = self.filter_detections(detections, (h, w), debug_events=geometry_debug)
        if logger and before != len(detections):
            logger.info(f"      → {len(detections)} après filtrage géométrique ({before - len(detections)} supprimés)")
            for event in geometry_debug[:5]:
                logger.info(
                    f"         - géométrie: {event['class_name']} conf {event['score']:.2f} rejeté "
                    f"({event['reason']}) bbox={event['bbox']}"
                )
        
        # 7. Récupération des boîtes TRONQUÉES
        before_boxes = [list(d.bbox) for d in detections]
        self._recover_truncated_boxes(detections, raw_detections)
        if logger:
            grown = sum(1 for a, d in zip(before_boxes, detections) if list(d.bbox) != a)
            if grown:
                logger.info(f"      → {grown} boîte(s) tronquée(s) réétendue(s)")

        # Stats
        if logger:
            by_class = defaultdict(int)
            for det in detections:
                by_class[det.class_name] += 1
            logger.info(f"\n      📊 Détections par classe:")
            for cls_name in sorted(by_class.keys()):
                translatable = "✓" if cls_name in self.cfg.translatable_classes else "✗"
                logger.info(f"         {cls_name:15s}: {by_class[cls_name]:2d} [{translatable}]")

        self.last_debug_report['final_detections'] = [
            {
                'class_name': d.class_name,
                'score': round(float(d.score), 4),
                'scale': float(getattr(d, 'scale', 1.0)),
                'bbox': [int(d.x1), int(d.y1), int(d.x2), int(d.y2)],
            }
            for d in detections
        ]
        
        return detections

    def get_last_debug_report(self) -> Dict:
        return self.last_debug_report or {}
    
    def get_translatable_detections(self, detections: List[Detection]) -> List[Detection]:
        return [d for d in detections if d.class_name in self.cfg.translatable_classes]
    
    def __del__(self):
        if self.model is not None:
            del self.model
            self.model = None