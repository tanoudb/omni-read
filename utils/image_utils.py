"""
═══════════════════════════════════════════════════════════════════════════════
IMAGE UTILITIES - Manipulation et preprocessing images
═══════════════════════════════════════════════════════════════════════════════
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from typing import Tuple, Optional, List
from pathlib import Path


class ImageUtils:
    """Utilitaires images premium"""
    
    # ─────────────────────────────────────────────────────────────────────────
    # CONVERSION
    # ─────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def cv2_to_pil(img: np.ndarray) -> Image.Image:
        """OpenCV (BGR) → PIL (RGB)"""
        if len(img.shape) == 2:  # Grayscale
            return Image.fromarray(img)
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    @staticmethod
    def pil_to_cv2(img: Image.Image) -> np.ndarray:
        """PIL (RGB) → OpenCV (BGR)"""
        img_array = np.array(img)
        if len(img_array.shape) == 2:  # Grayscale
            return img_array
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    @staticmethod
    def to_grayscale(img: np.ndarray) -> np.ndarray:
        """Convertit en niveaux de gris"""
        if len(img.shape) == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def to_rgb(img: np.ndarray) -> np.ndarray:
        """Convertit en RGB"""
        if len(img.shape) == 3 and img.shape[2] == 3:
            return img
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # ─────────────────────────────────────────────────────────────────────────
    # PREPROCESSING INTELLIGENT
    # ─────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def smart_resize(img: np.ndarray, min_height: int = 20, 
                     max_factor: int = 3, interpolation: str = 'lanczos') -> np.ndarray:
        """
        Resize intelligent : seulement si texte trop petit
        """
        h, w = img.shape[:2]
        
        if h >= min_height:
            return img  # Déjà assez grand
        
        # Calculer facteur optimal
        factor = min(min_height / h, max_factor)
        factor = max(1.0, factor)
        
        if factor == 1.0:
            return img
        
        new_w = int(w * factor)
        new_h = int(h * factor)
        
        # Choisir interpolation
        interp_map = {
            'nearest': cv2.INTER_NEAREST,
            'linear': cv2.INTER_LINEAR,
            'cubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }
        interp = interp_map.get(interpolation, cv2.INTER_LANCZOS4)
        
        return cv2.resize(img, (new_w, new_h), interpolation=interp)
    
    @staticmethod
    def minimal_preprocessing(img: np.ndarray, for_ocr: bool = True) -> np.ndarray:
        """
        Preprocessing MINIMAL pour TrOCR
        TrOCR préfère images naturelles, pas binarisées
        """
        # Juste convertir en RGB si nécessaire
        if for_ocr:
            img = ImageUtils.to_rgb(img)
        
        return img
    
    @staticmethod
    def enhance_contrast_light(img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """CLAHE léger (optionnel, souvent inutile pour TrOCR)"""
        if len(img.shape) == 3:
            img = ImageUtils.to_grayscale(img)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        return clahe.apply(img)
    
    # ─────────────────────────────────────────────────────────────────────────
    # DÉTECTION COULEUR
    # ─────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def detect_background_color(img: np.ndarray, x1: int, y1: int, 
                               x2: int, y2: int, method: str = 'median_border',
                               sample_ratio: float = 0.1) -> Tuple[int, int, int]:
        """
        Détecte la couleur de fond d'une zone
        
        Methods:
        - median_border: médiane des pixels des bords
        - mean: moyenne de toute la zone
        - mode: couleur la plus fréquente
        """
        crop = img[y1:y2, x1:x2]
        
        if crop.size == 0:
            return (255, 255, 255)  # Blanc par défaut
        
        h, w = crop.shape[:2]
        
        if method == 'median_border':
            # Échantillonner les bords
            margin_h = max(1, int(h * sample_ratio))
            margin_w = max(1, int(w * sample_ratio))
            
            top = crop[0:margin_h, :].reshape(-1, 3)
            bottom = crop[h-margin_h:h, :].reshape(-1, 3)
            left = crop[:, 0:margin_w].reshape(-1, 3)
            right = crop[:, w-margin_w:w].reshape(-1, 3)
            
            all_pixels = np.concatenate([top, bottom, left, right])
            color = np.median(all_pixels, axis=0)
            
        elif method == 'mean':
            color = np.mean(crop.reshape(-1, 3), axis=0)
            
        elif method == 'mode':
            # Trouver couleur la plus fréquente
            pixels = crop.reshape(-1, 3)
            unique, counts = np.unique(pixels, axis=0, return_counts=True)
            color = unique[counts.argmax()]
        
        else:
            color = np.median(crop.reshape(-1, 3), axis=0)
        
        return tuple(map(int, color))
    
    @staticmethod
    def auto_text_color(bg_color: Tuple[int, int, int], 
                       threshold: int = 128) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        Choisit automatiquement la couleur du texte selon le fond
        
        Returns:
            (text_color, outline_color)
        """
        # Calculer luminosité
        luminosity = sum(bg_color) / 3
        
        if luminosity > threshold:
            # Fond clair → texte noir, contour blanc
            return (0, 0, 0), (255, 255, 255)
        else:
            # Fond sombre → texte blanc, contour noir
            return (255, 255, 255), (0, 0, 0)
    
    # ─────────────────────────────────────────────────────────────────────────
    # INPAINTING
    # ─────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def simple_fill(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                   color: Optional[Tuple[int, int, int]] = None,
                   margin: int = 5) -> np.ndarray:
        """
        Remplissage simple avec couleur (meilleur que cv2.inpaint pour webtoon)
        """
        if color is None:
            color = ImageUtils.detect_background_color(img, x1, y1, x2, y2)
        
        # Appliquer marges
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(img.shape[1], x2 + margin)
        y2 = min(img.shape[0], y2 + margin)
        
        # Remplir
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        
        # Léger blur sur les bords pour transition douce
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (x1-2, y1-2), (x2+2, y2+2), 255, -1)
        img = cv2.GaussianBlur(img, (5, 5), 0, dst=img, borderType=cv2.BORDER_DEFAULT)
        
        return img
    
    @staticmethod
    def cv2_inpaint(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                   radius: int = 7, method: str = 'telea') -> np.ndarray:
        """
        Inpainting OpenCV (peut laisser des artefacts)
        """
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        method_flag = cv2.INPAINT_TELEA if method == 'telea' else cv2.INPAINT_NS
        
        return cv2.inpaint(img, mask, radius, method_flag)
    
    # ─────────────────────────────────────────────────────────────────────────
    # VALIDATION
    # ─────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def is_valid_bbox(x1: int, y1: int, x2: int, y2: int, img_shape: Tuple[int, int],
                     min_area: int = 400, max_area: int = 500000,
                     min_ratio: float = 0.1, max_ratio: float = 10.0) -> bool:
        """Valide une bounding box"""
        h, w = img_shape[:2]
        
        # Vérifier bounds
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            return False
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        # Aire
        area = (x2 - x1) * (y2 - y1)
        if area < min_area or area > max_area:
            return False
        
        # Ratio
        width = x2 - x1
        height = y2 - y1
        ratio = width / height if height > 0 else 0
        
        if ratio < min_ratio or ratio > max_ratio:
            return False
        
        return True
    
    # ─────────────────────────────────────────────────────────────────────────
    # GEOMETRY
    # ─────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
        """Calcule l'Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def distance_between_boxes(box1: List[float], box2: List[float]) -> float:
        """Calcule la distance entre centres de 2 boxes"""
        center1_x = (box1[0] + box1[2]) / 2
        center1_y = (box1[1] + box1[3]) / 2
        center2_x = (box2[0] + box2[2]) / 2
        center2_y = (box2[1] + box2[3]) / 2
        
        return np.sqrt((center2_x - center1_x)**2 + (center2_y - center1_y)**2)
    
    # ─────────────────────────────────────────────────────────────────────────
    # I/O
    # ─────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def load_image(path: Path) -> Optional[np.ndarray]:
        """Charge une image avec gestion d'erreur"""
        try:
            img = cv2.imread(str(path))
            if img is None:
                return None
            return img
        except Exception:
            return None
    
    @staticmethod
    def save_image(img: np.ndarray, path: Path) -> bool:
        """Sauvegarde une image"""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), img)
            return True
        except Exception:
            return False


if __name__ == "__main__":
    """Test image utilities"""
    print("═" * 80)
    print("IMAGE UTILITIES TEST")
    print("═" * 80)
    
    # Test resize
    test_img = np.zeros((10, 50, 3), dtype=np.uint8)
    resized = ImageUtils.smart_resize(test_img, min_height=20)
    print(f"\n✓ Resize: {test_img.shape} → {resized.shape}")
    
    # Test couleur
    test_img = np.ones((100, 100, 3), dtype=np.uint8) * 200
    bg_color = ImageUtils.detect_background_color(test_img, 0, 0, 100, 100)
    text_color, outline = ImageUtils.auto_text_color(bg_color)
    print(f"✓ Background: {bg_color}")
    print(f"✓ Text color: {text_color}")
    
    # Test IOU
    box1 = [0, 0, 100, 100]
    box2 = [50, 50, 150, 150]
    iou = ImageUtils.calculate_iou(box1, box2)
    print(f"✓ IOU: {iou:.2f}")
    
    print("\n✅ Image utils OK\n")
