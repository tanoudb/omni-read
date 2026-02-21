"""
═══════════════════════════════════════════════════════════════════════════════
MEMORY MANAGER - Gestion optimale RAM/VRAM
═══════════════════════════════════════════════════════════════════════════════
"""

import gc
import torch
import psutil
from typing import Optional, Any
from contextlib import contextmanager


class MemoryManager:
    """Gestionnaire mémoire GPU/CPU ultra-optimisé"""
    
    @staticmethod
    def get_device() -> str:
        """Détecte le device optimal"""
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():  # Apple Silicon
            return 'mps'
        return 'cpu'
    
    @staticmethod
    def get_ram_usage() -> dict:
        """Stats RAM système"""
        mem = psutil.virtual_memory()
        return {
            'total_gb': mem.total / (1024**3),
            'used_gb': mem.used / (1024**3),
            'available_gb': mem.available / (1024**3),
            'percent': mem.percent
        }
    
    @staticmethod
    def get_vram_usage() -> Optional[dict]:
        """Stats VRAM GPU"""
        if not torch.cuda.is_available():
            return None
        
        return {
            'allocated_gb': torch.cuda.memory_allocated() / (1024**3),
            'reserved_gb': torch.cuda.memory_reserved() / (1024**3),
            'max_allocated_gb': torch.cuda.max_memory_allocated() / (1024**3)
        }
    
    @staticmethod
    def cleanup_light():
        """Nettoyage léger (entre images)"""
        gc.collect()
    
    @staticmethod
    def cleanup_medium():
        """Nettoyage moyen (entre phases)"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def cleanup_aggressive():
        """Nettoyage agressif (après modèle lourd)"""
        # Multiple GC passes
        for _ in range(3):
            gc.collect()
        
        # Clear CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Force numpy cleanup
        import ctypes
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except:
            pass
    
    @staticmethod
    def optimize_model_memory(model: Any, use_fp16: bool = True) -> Any:
        """Optimise la mémoire d'un modèle"""
        if use_fp16 and torch.cuda.is_available():
            model = model.half()
        
        model.eval()  # Mode évaluation (désactive dropout, etc.)
        
        return model
    
    @staticmethod
    def log_memory_status(logger):
        """Log l'état mémoire"""
        ram = MemoryManager.get_ram_usage()
        logger.info(f"   RAM: {ram['used_gb']:.1f}/{ram['total_gb']:.1f} GB ({ram['percent']:.1f}%)")
        
        vram = MemoryManager.get_vram_usage()
        if vram:
            logger.info(f"   VRAM: {vram['allocated_gb']:.2f} GB allocated")


@contextmanager
def model_context(model_loader, cleanup_level: str = 'aggressive'):
    """
    Context manager pour charger/décharger modèles automatiquement
    
    Usage:
        with model_context(lambda: load_yolo()) as model:
            results = model.predict(image)
        # Model automatiquement détruit ici
    """
    model = None
    try:
        # Load
        model = model_loader()
        yield model
    finally:
        # Cleanup
        if model is not None:
            del model
        
        if cleanup_level == 'light':
            MemoryManager.cleanup_light()
        elif cleanup_level == 'medium':
            MemoryManager.cleanup_medium()
        else:
            MemoryManager.cleanup_aggressive()


@contextmanager
def memory_profiler(logger, operation_name: str):
    """
    Profile la consommation mémoire d'une opération
    
    Usage:
        with memory_profiler(logger, "Detection"):
            detect_bubbles(image)
    """
    # Avant
    ram_before = MemoryManager.get_ram_usage()
    vram_before = MemoryManager.get_vram_usage()
    
    try:
        yield
    finally:
        # Après
        ram_after = MemoryManager.get_ram_usage()
        vram_after = MemoryManager.get_vram_usage()
        
        ram_delta = ram_after['used_gb'] - ram_before['used_gb']
        logger.debug(f"{operation_name} - RAM: {ram_delta:+.2f} GB")
        
        if vram_before and vram_after:
            vram_delta = vram_after['allocated_gb'] - vram_before['allocated_gb']
            logger.debug(f"{operation_name} - VRAM: {vram_delta:+.2f} GB")


class BatchProcessor:
    """Traitement par batch pour économiser mémoire"""
    
    @staticmethod
    def create_batches(items: list, batch_size: int) -> list:
        """Divise une liste en batches"""
        return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    
    @staticmethod
    def auto_batch_size(total_items: int, max_memory_gb: float = 2.0) -> int:
        """Calcule la taille de batch optimale"""
        ram = MemoryManager.get_ram_usage()
        available_gb = ram['available_gb']
        
        # Si peu de RAM disponible, réduire batch size
        if available_gb < 4:
            return 1
        elif available_gb < 8:
            return 2
        else:
            return 4


if __name__ == "__main__":
    """Test memory manager"""
    print("═" * 80)
    print("MEMORY MANAGER TEST")
    print("═" * 80)
    
    device = MemoryManager.get_device()
    print(f"\n🖥️  Device: {device}")
    
    ram = MemoryManager.get_ram_usage()
    print(f"\n💾 RAM:")
    print(f"   Total: {ram['total_gb']:.2f} GB")
    print(f"   Used: {ram['used_gb']:.2f} GB")
    print(f"   Available: {ram['available_gb']:.2f} GB")
    
    if device == 'cuda':
        vram = MemoryManager.get_vram_usage()
        print(f"\n🎮 VRAM:")
        print(f"   Allocated: {vram['allocated_gb']:.2f} GB")
        print(f"   Reserved: {vram['reserved_gb']:.2f} GB")
    
    print("\n✅ Memory manager OK\n")
