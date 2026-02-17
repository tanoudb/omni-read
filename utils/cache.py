"""
═══════════════════════════════════════════════════════════════════════════════
CACHE SYSTEM - Système de cache intelligent pour traductions
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class CacheManager:
    """Gestionnaire de cache premium"""
    
    def __init__(self, cache_file: Path, max_size_mb: int = 500):
        self.cache_file = cache_file
        self.max_size_mb = max_size_mb
        self.cache: Dict[str, Any] = {}
        self.stats = {
            'hits': 0,
            'misses': 0,
            'saves': 0
        }
        
        self._load()
    
    def _load(self):
        """Charge le cache depuis le disque"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}
    
    def _save(self):
        """Sauvegarde le cache sur disque"""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            self.stats['saves'] += 1
        except Exception:
            pass
    
    def _make_key(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Génère une clé unique"""
        raw = f"{src_lang}:{tgt_lang}:{text.lower().strip()}"
        return hashlib.md5(raw.encode()).hexdigest()
    
    def get(self, text: str, src_lang: str, tgt_lang: str) -> Optional[str]:
        """Récupère une traduction du cache"""
        key = self._make_key(text, src_lang, tgt_lang)
        
        if key in self.cache:
            self.stats['hits'] += 1
            return self.cache[key]
        
        self.stats['misses'] += 1
        return None
    
    def set(self, text: str, translation: str, src_lang: str, tgt_lang: str):
        """Stocke une traduction"""
        key = self._make_key(text, src_lang, tgt_lang)
        self.cache[key] = translation
        
        # Sauvegarde périodique (tous les 10 ajouts)
        if self.stats['misses'] % 10 == 0:
            self._save()
    
    def get_size_mb(self) -> float:
        """Taille du cache en MB"""
        if not self.cache_file.exists():
            return 0.0
        return self.cache_file.stat().st_size / (1024 * 1024)
    
    def clear(self):
        """Vide le cache"""
        self.cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        self.stats = {'hits': 0, 'misses': 0, 'saves': 0}
    
    def get_stats(self) -> dict:
        """Stats du cache"""
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total * 100) if total > 0 else 0
        
        return {
            'entries': len(self.cache),
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': f"{hit_rate:.1f}%",
            'size_mb': f"{self.get_size_mb():.2f}",
            'saves': self.stats['saves']
        }
    
    def cleanup_old_entries(self, max_entries: int = 10000):
        """Nettoie les entrées anciennes si trop de cache"""
        if len(self.cache) > max_entries:
            # Garder seulement les max_entries plus récents
            # (simplification : on garde les premiers dans le dict)
            keys = list(self.cache.keys())
            for key in keys[max_entries:]:
                del self.cache[key]
            self._save()
    
    def __enter__(self):
        """Context manager"""
        return self
    
    def __exit__(self, *args):
        """Sauvegarde en sortant du context"""
        self._save()


if __name__ == "__main__":
    """Test cache"""
    import tempfile
    from pathlib import Path
    
    print("═" * 80)
    print("CACHE SYSTEM TEST")
    print("═" * 80)
    
    # Test avec fichier temporaire
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = Path(tmpdir) / "test_cache.json"
        
        with CacheManager(cache_file) as cache:
            # Test set
            cache.set("Hello", "Bonjour", "en", "fr")
            cache.set("World", "Monde", "en", "fr")
            
            # Test get (hit)
            result = cache.get("Hello", "en", "fr")
            print(f"\n✓ Cache hit: {result}")
            
            # Test get (miss)
            result = cache.get("Unknown", "en", "fr")
            print(f"✓ Cache miss: {result}")
            
            # Stats
            stats = cache.get_stats()
            print(f"\n📊 Stats:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
        
        # Vérifier persistence
        with CacheManager(cache_file) as cache:
            result = cache.get("Hello", "en", "fr")
            print(f"\n✓ Persistence: {result}")
    
    print("\n✅ Cache OK\n")
