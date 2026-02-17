"""
═══════════════════════════════════════════════════════════════════════════════
DETECTOR PREMIUM - YOLO avec slicing adaptatif & multi-scale
═══════════════════════════════════════════════════════════════════════════════

FIX v2: containment filter intra-classe + inter-classe
"""

import numpy as np
import cv2
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
    
    def sliding_window_detect(self, image: np.ndarray, scale: float = 1.0) -> List[Detection]:
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
        
        for y in range(0, new_h, step):
            y_end = min(y + window_height, new_h)
            window = image_scaled[y:y_end, :]
            
            results = self.model.predict(window, conf=0.15, verbose=False, half=False, device='cuda')
            
            for result in results:
                for box in result.boxes:
                    x1, y1_local, x2, y2_local = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = result.names[cls_id]
                    
                    if conf < self.cfg.confidence_thresholds.get(cls_name, 0.25):
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
                            continue
                    
                    detections.append(Detection(
                        class_name=cls_name, bbox=bbox, score=conf,
                        scale=scale, metadata={'window_y': y}
                    ))
        
        return detections
    
    # ─────────────────────────────────────────────────────────────────────────
    # MULTI-SCALE
    # ─────────────────────────────────────────────────────────────────────────
    
    def multi_scale_detect(self, image: np.ndarray) -> List[Detection]:
        all_detections = []
        for scale in self.cfg.detection_scales:
            all_detections.extend(self.sliding_window_detect(image, scale=scale))
        return all_detections
    
    # ─────────────────────────────────────────────────────────────────────────
    # NMS
    # ─────────────────────────────────────────────────────────────────────────
    
    def nms_per_class(self, detections: List[Detection]) -> List[Detection]:
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
                dets = [d for d in dets if ImageUtils.calculate_iou(best.bbox, d.bbox) < iou_thresh]
            kept.extend(keep)
        
        return kept
    
    def multi_scale_nms(self, detections: List[Detection]) -> List[Detection]:
        if not detections:
            return []
        detections = sorted(detections, key=lambda x: x.score, reverse=True)
        kept = []
        iou_thresh = self.cfg.multi_scale_nms_iou
        while detections:
            best = detections.pop(0)
            kept.append(best)
            detections = [d for d in detections if ImageUtils.calculate_iou(best.bbox, d.bbox) < iou_thresh]
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
    
    def remove_contained_boxes(self, detections: List[Detection]) -> List[Detection]:
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
                    # j est contenu dans i → supprimer j (le plus petit)
                    to_remove.add(j)
        
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
    
    def filter_detections(self, detections: List[Detection], image_shape: Tuple[int, int]) -> List[Detection]:
        filtered = []
        for det in detections:
            if not ImageUtils.is_valid_bbox(
                det.x1, det.y1, det.x2, det.y2, image_shape,
                min_area=self.cfg.min_box_area, max_area=self.cfg.max_box_area,
                min_ratio=self.cfg.min_box_ratio, max_ratio=self.cfg.max_box_ratio
            ):
                continue
            if config.filters.filter_top_edge and self.geo_filter.is_on_top_edge(det.y1, image_shape[0]):
                continue
            if config.filters.filter_bottom_edge and self.geo_filter.is_on_bottom_edge(det.y2, image_shape[0]):
                continue
            filtered.append(det)
        return filtered
    
    # ─────────────────────────────────────────────────────────────────────────
    # PIPELINE PRINCIPAL
    # ─────────────────────────────────────────────────────────────────────────
    
    def detect(self, image: np.ndarray, logger=None) -> List[Detection]:
        h, w = image.shape[:2]
        
        if logger:
            logger.info(f"   🔍 Détection sur {w}x{h}px")
        
        # 1. Multi-scale
        if self.cfg.enable_multi_scale:
            if logger:
                logger.info(f"      Multi-scale: {self.cfg.detection_scales}")
            detections = self.multi_scale_detect(image)
        else:
            detections = self.sliding_window_detect(image, scale=1.0)
        
        if logger:
            logger.info(f"      → {len(detections)} détections brutes")
        
        # 2. NMS par classe
        detections = self.nms_per_class(detections)
        if logger:
            logger.info(f"      → {len(detections)} après NMS par classe")
        
        # 3. NMS multi-échelle
        if self.cfg.enable_multi_scale:
            detections = self.multi_scale_nms(detections)
            if logger:
                logger.info(f"      → {len(detections)} après NMS multi-échelle")
        
        # 4. Suppression containment (NOUVEAU - toutes classes)
        before = len(detections)
        detections = self.remove_contained_boxes(detections)
        if logger and before != len(detections):
            logger.info(f"      → {len(detections)} après suppression containment ({before - len(detections)} doublons)")
        
        # 5. Conflits inter-classes
        before = len(detections)
        detections = self.resolve_inter_class_conflicts(detections)
        if logger and before != len(detections):
            logger.info(f"      → {len(detections)} après résolution conflits")
        
        # 6. Filtrage géométrique
        before = len(detections)
        detections = self.filter_detections(detections, (h, w))
        if logger and before != len(detections):
            logger.info(f"      → {len(detections)} après filtrage géométrique ({before - len(detections)} supprimés)")
        
        # Stats
        if logger:
            by_class = defaultdict(int)
            for det in detections:
                by_class[det.class_name] += 1
            logger.info(f"\n      📊 Détections par classe:")
            for cls_name in sorted(by_class.keys()):
                translatable = "✓" if cls_name in self.cfg.translatable_classes else "✗"
                logger.info(f"         {cls_name:15s}: {by_class[cls_name]:2d} [{translatable}]")
        
        return detections
    
    def get_translatable_detections(self, detections: List[Detection]) -> List[Detection]:
        return [d for d in detections if d.class_name in self.cfg.translatable_classes]
    
    def __del__(self):
        if self.model is not None:
            del self.model
            self.model = None