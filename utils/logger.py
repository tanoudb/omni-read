"""
═══════════════════════════════════════════════════════════════════════════════
LOGGER PREMIUM - Logging coloré avec timing et statistiques
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import time
from functools import wraps

# Couleurs ANSI
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'


class ColoredFormatter(logging.Formatter):
    """Formatter avec couleurs"""
    
    FORMATS = {
        logging.DEBUG: Colors.CYAN + "%(levelname)s" + Colors.RESET + " | %(message)s",
        logging.INFO: Colors.GREEN + "%(levelname)s" + Colors.RESET + " | %(message)s",
        logging.WARNING: Colors.YELLOW + "%(levelname)s" + Colors.RESET + " | %(message)s",
        logging.ERROR: Colors.RED + "%(levelname)s" + Colors.RESET + " | %(message)s",
        logging.CRITICAL: Colors.BG_RED + Colors.WHITE + "%(levelname)s" + Colors.RESET + " | %(message)s",
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class WebtoonLogger:
    """Logger premium pour traducteur"""
    
    def __init__(self, name: str = "WebtoonV5", log_file: Optional[Path] = None,
                 level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.logger.handlers = []  # Clear handlers
        
        # Console handler avec couleurs
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter())
        self.logger.addHandler(console_handler)
        
        # File handler si demandé
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_format = "%(asctime)s | %(levelname)s | %(module)s | %(message)s"
            file_handler.setFormatter(logging.Formatter(file_format))
            self.logger.addHandler(file_handler)
        
        # Stats
        self.stats = {
            'start_time': None,
            'phase_times': {},
            'current_phase': None
        }
    
    # ─────────────────────────────────────────────────────────────────────────
    # LOGGING STANDARD
    # ─────────────────────────────────────────────────────────────────────────
    
    def debug(self, msg: str):
        self.logger.debug(msg)
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)
    
    def critical(self, msg: str):
        self.logger.critical(msg)
    
    # ─────────────────────────────────────────────────────────────────────────
    # LOGGING PREMIUM
    # ─────────────────────────────────────────────────────────────────────────
    
    def header(self, text: str, width: int = 80):
        """Affiche un header stylé"""
        line = "═" * width
        self.info(f"\n{line}")
        self.info(f"{text.center(width)}")
        self.info(f"{line}\n")
    
    def section(self, text: str, width: int = 80):
        """Affiche une section"""
        line = "─" * width
        self.info(f"\n{line}")
        self.info(f"  {text}")
        self.info(f"{line}")
    
    def phase(self, name: str, number: int, total: int):
        """Annonce une phase"""
        self.info(f"\n🎯 PHASE {number}/{total} : {name.upper()}")
        
        # Timer
        if self.stats['current_phase']:
            self.end_phase()
        
        self.stats['current_phase'] = name
        self.stats['phase_times'][name] = {'start': time.time()}
    
    def end_phase(self):
        """Termine une phase et log le temps"""
        if self.stats['current_phase']:
            phase = self.stats['current_phase']
            elapsed = time.time() - self.stats['phase_times'][phase]['start']
            self.stats['phase_times'][phase]['duration'] = elapsed
            self.info(f"✅ {phase} terminé en {elapsed:.2f}s")
            self.stats['current_phase'] = None
    
    def progress(self, current: int, total: int, prefix: str = ""):
        """Barre de progression"""
        percent = (current / total) * 100
        bar_length = 40
        filled = int(bar_length * current / total)
        bar = "█" * filled + "░" * (bar_length - filled)
        self.info(f"{prefix}[{bar}] {current}/{total} ({percent:.1f}%)")
    
    def stat(self, key: str, value):
        """Log une statistique"""
        self.info(f"   {key}: {value}")
    
    def start_timer(self):
        """Démarre le timer global"""
        self.stats['start_time'] = time.time()
    
    def end_timer(self):
        """Termine et affiche le temps total"""
        if self.stats['start_time']:
            elapsed = time.time() - self.stats['start_time']
            self.header(f"⏱️  TEMPS TOTAL: {elapsed:.2f}s")
    
    def summary(self, data: dict):
        """Affiche un résumé"""
        self.section("📊 RÉSUMÉ")
        for key, value in data.items():
            self.info(f"   {key}: {value}")


# ═══════════════════════════════════════════════════════════════════════════════
# DECORATORS
# ═══════════════════════════════════════════════════════════════════════════════

def timing_decorator(logger: WebtoonLogger):
    """Décorateur pour timer une fonction"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            logger.debug(f"▶ {func.__name__} started")
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.debug(f"✓ {func.__name__} completed in {elapsed:.2f}s")
            return result
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# INSTANCE GLOBALE
# ═══════════════════════════════════════════════════════════════════════════════

# Sera initialisé dans main.py
logger = None

def init_logger(log_file: Optional[Path] = None, level: str = "INFO") -> WebtoonLogger:
    """Initialise le logger global"""
    global logger
    logger = WebtoonLogger(log_file=log_file, level=level)
    return logger


if __name__ == "__main__":
    """Test logger"""
    test_logger = WebtoonLogger()
    
    test_logger.header("TEST LOGGER V5")
    test_logger.info("Info message")
    test_logger.warning("Warning message")
    test_logger.error("Error message")
    
    test_logger.section("Test Section")
    test_logger.stat("Images", 42)
    test_logger.stat("GPU", "CUDA")
    
    test_logger.phase("Detection", 1, 4)
    time.sleep(0.5)
    test_logger.end_phase()
    
    test_logger.progress(7, 10, "Processing: ")
    
    test_logger.summary({
        'Total images': 10,
        'Translated': 8,
        'Skipped': 2
    })
