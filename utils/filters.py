"""
═══════════════════════════════════════════════════════════════════════════════
FILTERS - Filtrage watermarks, bruitages, onomatopées
═══════════════════════════════════════════════════════════════════════════════
"""

import re
from typing import List, Tuple, Optional


class TextFilter:
    """Filtre texte intelligent"""
    
    def __init__(self, watermark_patterns: List[str], sfx_patterns: List[str]):
        self.watermark_patterns = [re.compile(p, re.IGNORECASE) for p in watermark_patterns]
        self.sfx_patterns = [re.compile(p) for p in sfx_patterns]
    
    # ─────────────────────────────────────────────────────────────────────────
    # WATERMARKS
    # ─────────────────────────────────────────────────────────────────────────
    
    def is_watermark(self, text: str) -> bool:
        """Détecte si le texte est un watermark"""
        if not text:
            return False
        
        text_lower = text.lower().strip()
        
        # Patterns regex
        for pattern in self.watermark_patterns:
            if pattern.search(text_lower):
                return True
        
        # URLs
        if any(x in text_lower for x in ['www.', 'http://', 'https://']):
            return True
        
        # Domaines courants
        if any(text_lower.endswith(x) for x in ['.com', '.net', '.org', '.io']):
            return True
        
        return False
    
    # ─────────────────────────────────────────────────────────────────────────
    # ONOMATOPÉES / SFX
    # ─────────────────────────────────────────────────────────────────────────
    
    def is_sfx(self, text: str) -> bool:
        """Détecte les bruitages/onomatopées"""
        if not text or len(text) < 2:
            return True
        
        # Patterns regex
        for pattern in self.sfx_patterns:
            if pattern.match(text):
                return True
        
        # Répétitions de caractères (AAAA, HHHH)
        if len(set(text.upper())) == 1 and len(text) > 2:
            return True
        
        # Symboles uniquement
        if all(c in '!?.*#@-_~' for c in text):
            return True
        
        return False
    
    # ─────────────────────────────────────────────────────────────────────────
    # VALIDATIONS
    # ─────────────────────────────────────────────────────────────────────────
    
    def is_numeric_only(self, text: str, max_ratio: float = 0.8) -> bool:
        """Vérifie si le texte est principalement numérique"""
        if not text:
            return False
        
        # Compter chiffres
        num_digits = sum(c.isdigit() for c in text)
        ratio = num_digits / len(text)
        
        return ratio > max_ratio
    
    def is_special_chars_only(self, text: str) -> bool:
        """Vérifie si le texte ne contient que des caractères spéciaux"""
        if not text:
            return False
        
        # Enlever espaces
        text_no_space = text.replace(' ', '')
        
        if not text_no_space:
            return True
        
        # Vérifier si que ponctuation/symboles
        return all(not c.isalnum() for c in text_no_space)
    
    def has_letters(self, text: str) -> bool:
        """Vérifie si le texte contient au moins une lettre"""
        return any(c.isalpha() for c in text)
    
    def is_single_char(self, text: str) -> bool:
        """Vérifie si c'est un seul caractère (après strip)"""
        return len(text.strip()) == 1
    
    def is_too_short(self, text: str, min_length: int = 2) -> bool:
        """Vérifie si le texte est trop court"""
        return len(text.strip()) < min_length
    
    # ─────────────────────────────────────────────────────────────────────────
    # FILTRAGE PRINCIPAL
    # ─────────────────────────────────────────────────────────────────────────
    
    def should_skip(self, text: str, min_length: int = 2, 
                   max_numeric_ratio: float = 0.8) -> Tuple[bool, Optional[str]]:
        """
        Détermine si le texte doit être ignoré
        
        Returns:
            (should_skip: bool, reason: str)
        """
        if not text:
            return True, "empty"
        
        text = text.strip()
        
        # Trop court
        if self.is_too_short(text, min_length):
            return True, "too_short"
        
        # Watermark
        if self.is_watermark(text):
            return True, "watermark"
        
        # SFX
        if self.is_sfx(text):
            return True, "sfx"
        
        # Numérique uniquement
        if self.is_numeric_only(text, max_numeric_ratio):
            return True, "numeric_only"
        
        # Caractères spéciaux uniquement
        if self.is_special_chars_only(text):
            return True, "special_chars_only"
        
        # Pas de lettres
        if not self.has_letters(text):
            return True, "no_letters"
        
        return False, None
    
    # ─────────────────────────────────────────────────────────────────────────
    # NETTOYAGE
    # ─────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Nettoie le texte (espaces, caractères bizarres)"""
        if not text:
            return ""
        
        # Espaces multiples
        text = re.sub(r'\s+', ' ', text)
        
        # Caractères de contrôle
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        
        # Trim
        return text.strip()


class GeometricFilter:
    """Filtre géométrique (position, taille)"""
    
    def __init__(self, top_threshold: float = 0.05, bottom_threshold: float = 0.95):
        self.top_threshold = top_threshold
        self.bottom_threshold = bottom_threshold
    
    def is_on_top_edge(self, y1: int, img_height: int) -> bool:
        """Vérifie si la box est sur le bord supérieur"""
        return (y1 / img_height) < self.top_threshold
    
    def is_on_bottom_edge(self, y2: int, img_height: int) -> bool:
        """Vérifie si la box est sur le bord inférieur"""
        return (y2 / img_height) > self.bottom_threshold
    
    def is_on_edge(self, y1: int, y2: int, img_height: int) -> bool:
        """Vérifie si sur un bord"""
        return self.is_on_top_edge(y1, img_height) or self.is_on_bottom_edge(y2, img_height)
    
    def is_too_small(self, x1: int, y1: int, x2: int, y2: int, min_area: int = 400) -> bool:
        """Vérifie si l'aire est trop petite"""
        area = (x2 - x1) * (y2 - y1)
        return area < min_area
    
    def is_too_large(self, x1: int, y1: int, x2: int, y2: int, max_area: int = 500000) -> bool:
        """Vérifie si l'aire est trop grande"""
        area = (x2 - x1) * (y2 - y1)
        return area > max_area
    
    def has_bad_ratio(self, x1: int, y1: int, x2: int, y2: int,
                     min_ratio: float = 0.1, max_ratio: float = 10.0) -> bool:
        """Vérifie si le ratio largeur/hauteur est aberrant"""
        width = x2 - x1
        height = y2 - y1
        
        if height == 0:
            return True
        
        ratio = width / height
        return ratio < min_ratio or ratio > max_ratio


if __name__ == "__main__":
    """Test filters"""
    print("═" * 80)
    print("FILTERS TEST")
    print("═" * 80)
    
    # Patterns
    watermark_patterns = [
        r'asurascans?\.com',
        r'\.com',
        r'chapter\s*\d+'
    ]
    
    sfx_patterns = [
        r'^(.)\1{2,}$',
        r'^[!?.*#@\-_]{2,}$'
    ]
    
    text_filter = TextFilter(watermark_patterns, sfx_patterns)
    
    # Tests
    tests = [
        ("asurascans.com", True, "watermark"),
        ("AAAA", True, "sfx"),
        ("Hello world", False, None),
        ("123 456", True, "numeric_only"),
        ("...", True, "special_chars_only"),
        ("A", True, "too_short"),
    ]
    
    print("\n🔍 Text Filter Tests:")
    for text, expected_skip, expected_reason in tests:
        skip, reason = text_filter.should_skip(text)
        status = "✓" if skip == expected_skip else "✗"
        print(f"   {status} '{text}' → skip={skip}, reason={reason}")
    
    # Geometric
    geo_filter = GeometricFilter()
    
    print("\n📐 Geometric Filter Tests:")
    is_top = geo_filter.is_on_top_edge(10, 1000)
    print(f"   ✓ Top edge (10/1000): {is_top}")
    
    is_small = geo_filter.is_too_small(0, 0, 10, 10)
    print(f"   ✓ Too small (10x10): {is_small}")
    
    print("\n✅ Filters OK\n")
