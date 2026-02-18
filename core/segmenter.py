"""
Segmentation guidée par les bounding boxes YOLO.

Objectif:
- utiliser YOLO comme prompt spatial
- produire des masques précis (format polygones relatifs à la bbox)
- supporter un mode multi-masques pour bulles fusionnées

Le backend SAM2 est optionnel. Sans SAM2, un backend hybride léger est utilisé:
OCR regions + raffinage morphologique/threshold.
"""

from __future__ import annotations

from typing import List, Dict, Optional
from pathlib import Path
import gc

import cv2
import numpy as np

from config import config


class SmartSegmenter:
    """Segmenter principal avec fallback robuste sans dépendance lourde."""

    def __init__(self, logger=None):
        self.cfg = config.segmentation
        self.logger = logger
        self._sam2_available = False
        self._sam_model = None

        if self.cfg.backend.lower() == "sam2":
            self._sam2_available = self._try_init_sam2()
            if not self._sam2_available and self.logger:
                self.logger.warning("   ⚠️  SAM2 indisponible, fallback segmentation hybride")
        elif self.cfg.backend.lower() not in {"hybrid", "ocr_regions"} and self.logger:
            self.logger.warning(f"   ⚠️  Backend segmentation inconnu: {self.cfg.backend} (fallback hybrid)")

    def release(self):
        """Libère explicitement les ressources GPU/CPU liées au segmenter."""
        self._sam_model = None
        self._sam2_available = False
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _try_init_sam2(self) -> bool:
        """Initialise un backend SAM via ultralytics (checkpoint SAM/SAM2)."""
        try:
            from ultralytics import SAM

            candidate = (self.cfg.sam2_checkpoint or "").strip()
            if candidate:
                model_ref = candidate
            else:
                model_ref = "sam2_b.pt"

            if candidate and not Path(candidate).exists() and self.logger:
                self.logger.warning(
                    f"   ⚠️  Checkpoint SAM2 introuvable ({candidate}), tentative auto avec {model_ref}"
                )

            self._sam_model = SAM(model_ref)
            if self.logger:
                self.logger.info(f"   🧠 Segmenter SAM chargé: {model_ref}")
            return True
        except Exception as exc:
            self._sam_model = None
            if self.logger:
                self.logger.warning(f"   ⚠️  Chargement SAM échoué: {exc}")
            return False

    def _predict_sam2_mask(self, crop_bgr: np.ndarray, seed_mask: np.ndarray) -> Optional[np.ndarray]:
        if not self._sam2_available or self._sam_model is None:
            return None

        h, w = crop_bgr.shape[:2]
        margin = max(0, int(self.cfg.bbox_prompt_margin))
        bbox = [margin, margin, max(margin + 1, w - margin), max(margin + 1, h - margin)]

        try:
            try:
                import torch
                predict_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except Exception:
                predict_device = 'cpu'

            results = self._sam_model.predict(source=crop_bgr, bboxes=[bbox], verbose=False, device=predict_device)
        except Exception:
            return None

        if not results:
            return None

        masks_obj = getattr(results[0], "masks", None)
        if masks_obj is None:
            return None

        data = getattr(masks_obj, "data", None)
        if data is None:
            return None

        try:
            mask_array = data.detach().cpu().numpy()
        except Exception:
            try:
                mask_array = np.array(data)
            except Exception:
                return None

        if mask_array.ndim == 2:
            mask_array = mask_array[np.newaxis, ...]
        if mask_array.ndim != 3 or mask_array.shape[0] == 0:
            return None

        candidate_masks: List[np.ndarray] = []
        for idx in range(mask_array.shape[0]):
            m = (mask_array[idx] > 0.5).astype(np.uint8) * 255
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            candidate_masks.append(m)

        if not candidate_masks:
            return None

        if self.cfg.use_multimask:
            fused = np.zeros((h, w), dtype=np.uint8)
            for m in candidate_masks:
                # conserver les masques utiles (surface non négligeable)
                if int(np.sum(m > 0)) < self.cfg.min_component_area:
                    continue
                fused = cv2.bitwise_or(fused, m)

            if np.sum(fused) > 0:
                return fused

        if np.sum(seed_mask) == 0:
            return max(candidate_masks, key=lambda x: int(np.sum(x > 0)))

        def overlap_score(mask: np.ndarray) -> float:
            inter = np.sum((mask > 0) & (seed_mask > 0))
            union = np.sum((mask > 0) | (seed_mask > 0))
            return float(inter) / float(max(1, union))

        best = max(candidate_masks, key=overlap_score)
        return best

    @staticmethod
    def _clamp_polygon(points: np.ndarray, width: int, height: int) -> np.ndarray:
        points[:, 0] = np.clip(points[:, 0], 0, max(0, width - 1))
        points[:, 1] = np.clip(points[:, 1], 0, max(0, height - 1))
        return points

    def _build_mask_from_ocr_regions(
        self,
        crop_h: int,
        crop_w: int,
        text_regions: Optional[List[Dict]],
    ) -> np.ndarray:
        mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        if not text_regions:
            return mask

        for region in text_regions:
            raw = region.get("bbox")
            if not raw:
                continue
            pts = np.array(raw, dtype=np.int32)
            if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
                continue
            pts = self._clamp_polygon(pts, crop_w, crop_h)
            hull = cv2.convexHull(pts)
            cv2.fillPoly(mask, [hull], 255)

        return mask

    def _refine_with_threshold(self, crop_bgr: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

        dark = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            25,
            12,
        )
        bright = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            25,
            12,
        )

        text_like = cv2.bitwise_or(dark, bright)

        if np.sum(seed_mask) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            grown_seed = cv2.dilate(seed_mask, kernel, iterations=1)
            text_like = cv2.bitwise_and(text_like, grown_seed)

        k = min(5, max(3, int(self.cfg.mask_dilate_kernel) | 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        refined = cv2.morphologyEx(text_like, cv2.MORPH_CLOSE, kernel, iterations=1)
        refined = cv2.dilate(refined, kernel, iterations=1)

        return refined

    def _mask_to_regions(self, mask: np.ndarray) -> List[Dict]:
        if np.sum(mask) == 0:
            return []

        regions: List[Dict] = []

        if self.cfg.use_multimask:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for label_id in range(1, num_labels):
                area = int(stats[label_id, cv2.CC_STAT_AREA])
                if area < self.cfg.min_component_area:
                    continue

                comp_mask = np.where(labels == label_id, 255, 0).astype(np.uint8)
                contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if contour.shape[0] < 3:
                        continue
                    epsilon = max(0.5, 0.01 * cv2.arcLength(contour, True))
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    poly = approx.reshape(-1, 2).astype(int).tolist()
                    if len(poly) >= 3:
                        regions.append({"bbox": poly, "conf": 1.0, "source": "seg_multi"})
        else:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if contour.shape[0] < 3:
                    continue
                epsilon = max(0.5, 0.01 * cv2.arcLength(contour, True))
                approx = cv2.approxPolyDP(contour, epsilon, True)
                poly = approx.reshape(-1, 2).astype(int).tolist()
                if len(poly) >= 3:
                    regions.append({"bbox": poly, "conf": 1.0, "source": "seg_single"})

        return regions

    def segment_detection(
        self,
        image_bgr: np.ndarray,
        detection,
        text_regions: Optional[List[Dict]],
    ) -> List[Dict]:
        """
        Retourne des régions masque relatives à la bbox de détection.
        """
        try:
            if not self.cfg.enable_precise_masks:
                return text_regions or []

            x1, y1, x2, y2 = detection.x1, detection.y1, detection.x2, detection.y2
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                return text_regions or []

            seed_mask = self._build_mask_from_ocr_regions(crop.shape[0], crop.shape[1], text_regions)

            backend = (self.cfg.backend or "hybrid").lower()
            if backend == "ocr_regions":
                refined = seed_mask
            elif backend == "sam2":
                sam_mask = self._predict_sam2_mask(crop, seed_mask)
                if sam_mask is None or np.sum(sam_mask) == 0:
                    refined = self._refine_with_threshold(crop, seed_mask)
                else:
                    if np.sum(seed_mask) > 0:
                        refined = cv2.bitwise_or(sam_mask, seed_mask)
                    else:
                        refined = sam_mask
            else:
                refined = self._refine_with_threshold(crop, seed_mask)

            regions = self._mask_to_regions(refined)

            if not regions and text_regions:
                return text_regions

            return regions
        finally:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
