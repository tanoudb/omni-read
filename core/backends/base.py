"""
Interface abstraite pour les backends OCR
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import numpy as np


class OCRBackend(ABC):
    """Interface abstraite que tous les backends OCR doivent implémenter"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Nom du backend (ex: 'PaddleOCR-VL-v1.5', 'RapidOCR-PP-OCRv5')"""
        pass
    
    @abstractmethod
    def load(self, device: str) -> None:
        """
        Charge le modèle OCR
        
        Args:
            device: 'cuda' ou 'cpu'
        """
        pass
    
    @abstractmethod
    def read_text(self, img: np.ndarray) -> Tuple[str, float, List[Dict]]:
        """
        Extrait le texte d'une image
        
        Args:
            img: Image au format numpy array (BGR)
            
        Returns:
            Tuple contenant:
            - text: Texte extrait (str)
            - confidence: Score de confiance moyen (float, 0.0-1.0)
            - text_regions: Liste de dictionnaires contenant:
                * 'bbox': Coordonnées de la boîte englobante
                * 'text': Texte de cette région
                * 'conf': Confiance pour cette région
        """
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Décharge le modèle et libère la mémoire"""
        pass
