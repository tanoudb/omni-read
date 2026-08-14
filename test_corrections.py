# -*- coding: utf-8 -*-
"""
Test rapide des corrections d'effacement/rendu.
Lance le pipeline sur 1 chapitre de the_cleaner avec --force.
"""
import sys, os

# UTF-8 Windows
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

os.environ['FLAGS_allocator_strategy'] = 'auto_growth'

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from pathlib import Path
from config import config
from utils import init_logger
from pipeline import TranslationPipeline

SERIE = "hellogin"
CHAPITRE = "Chapitre 001"
INPUT_DIR = Path("manhwa") / SERIE / CHAPITRE
OUTPUT_DIR = Path("out_manhwa_test") / SERIE / CHAPITRE

def main():
    print(f"\n{'='*60}")
    print(f"  TEST CORRECTIONS - {SERIE} / {CHAPITRE}")
    print(f"{'='*60}\n")
    
    if not INPUT_DIR.exists():
        print(f"[ERROR] Input non trouvé : {INPUT_DIR}")
        sys.exit(1)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger = init_logger("test_corrections")
    
    pipeline = TranslationPipeline(
        logger=logger,
        debug=True,
        lazy_models=False,
    )
    
    results = pipeline.process_directory(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
    )
    
    print(f"\n{'='*60}")
    print(f"  RESULTATS")
    print(f"{'='*60}")
    
    # Chercher le qcheck report
    qcheck_path = OUTPUT_DIR / "qcheck_report.json"
    if qcheck_path.exists():
        import json
        with open(qcheck_path, 'r') as f:
            report = json.load(f)
        
        ghosts = sum(1 for r in report if r.get('type') in ('ghost', 'ghost_residual'))
        overflows = sum(1 for r in report if r.get('type') == 'overflow')
        contrasts = sum(1 for r in report if r.get('type') == 'contrast')
        total = len(report)
        
        print(f"  QCheck total   : {total}")
        print(f"  - Ghosts       : {ghosts} ({100*ghosts/max(1,total):.0f}%)")
        print(f"  - Overflows    : {overflows} ({100*overflows/max(1,total):.0f}%)")
        print(f"  - Contrasts    : {contrasts} ({100*contrasts/max(1,total):.0f}%)")
    else:
        print("  Pas de QCheck report trouvé")
    
    print(f"\n  Output: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
