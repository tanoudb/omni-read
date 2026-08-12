"""
═══════════════════════════════════════════════════════════════════════════════
WEBTOON TRANSLATOR V5 PREMIUM
═══════════════════════════════════════════════════════════════════════════════

Usage:
    python main.py                  # Traite input/ → output/
    python main.py --debug          # Mode debug (sauvegarde détections)
    python main.py --input custom/  # Dossier custom
    python main.py --image test.png # Image unique
"""

import argparse
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()  # charge .env (GEMINI_API_KEY, etc.) avant de lire la config
except ImportError:
    pass

from config import config, INPUT_DIR, OUTPUT_DIR, LOGS_DIR
from utils import init_logger, MemoryManager
from pipeline import TranslationPipeline

# Mode Série (optionnel)
try:
    from utils.series_db import SeriesDB
except ImportError:
    SeriesDB = None

import os
os.environ['FLAGS_allocator_strategy'] = 'auto_growth'

# Ensure stdout/stderr use UTF-8 (helps on Windows consoles with cp1252)
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    else:
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
except Exception:
    pass

def print_banner():
    """Affiche le banner de démarrage"""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║               🚀 WEBTOON TRANSLATOR V5🚀                                 ║
║                                                                           ║
║               Architecture Modulaire                                      ║
║                    Traduction Manhwa Universelle                          ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Point d'entrée principal"""
    
    parser = argparse.ArgumentParser(
        description='Webtoon Translator V5 Premium',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--input', '-i',
        type=Path,
        default=INPUT_DIR,
        help=f'Dossier input (défaut: {INPUT_DIR})'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=OUTPUT_DIR,
        help=f'Dossier output (défaut: {OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--image',
        type=Path,
        help='Traiter une image unique'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Niveau de logging (défaut: INFO)'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Désactiver le cache de traduction'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Mode debug: sauvegarde image annotée des détections + crops dans output/debug/'
    )

    parser.add_argument(
        '--api',
        action='store_true',
        help='Traduction via API Gemini (nécessite GEMINI_API_KEY). Rapide, haute qualité.'
    )
    
    parser.add_argument(
        '--show-config',
        action='store_true',
        help='Afficher la configuration et quitter'
    )

    parser.add_argument(
        '--translation-mode',
        choices=['hybrid', 'hybrid_quality', 'nllb', 'qwen'],
        help='Mode traduction: hybrid (NLLB→Qwen), hybrid_quality (NLLB + correction Qwen), nllb, qwen'
    )

    # ── Mode Série ──
    parser.add_argument(
        '--series',
        type=str,
        nargs='?',  # Makes the argument optional
        const='',   # Default value if no argument is provided
        help='Slug de la série (ex: tbate, solo-leveling). Active le mode série avec glossaire et contexte.'
    )

    # Alias français `--serie` pour compatibilité avec les commandes utilisateur
    parser.add_argument(
        '--serie',
        dest='series',
        nargs='?',
        const='',
        help='Alias français pour --series (ex: --serie tbate).'
    )

    parser.add_argument(
        '--chapter',
        type=int,
        default=None,
        help='Numéro du chapitre (utilisé avec --series pour le contexte narratif)'
    )

    parser.add_argument(
        '--init-series',
        type=str,
        default=None,
        metavar='NOM',
        help='Initialise une nouvelle série. Ex: --series tbate --init-series "The Beginning After The End"'
    )
    
    args = parser.parse_args()
    
    # Banner
    print_banner()
    
    # Show config
    if args.show_config:
        print(f"\nDevice: {MemoryManager.get_device()}")
        print(f"FP16: {config.performance.use_fp16}")
        print(f"\nDetection:")
        print(f"  Adaptive slicing: {config.detection.enable_adaptive_slicing}")
        print(f"  Max height: {getattr(config.detection, 'max_height', 0)}")
        print(f"  Scales: {config.detection.detection_scales}")
        print(f"  Black padding: {config.detection.use_black_padding} (ratio={config.detection.black_padding_ratio})")
        print(f"\nOCR: {config.ocr.backend}")
        print(f"  OCR primary: {getattr(config.ocr, 'primary_backend', config.ocr.backend)}")
        print(f"  OCR fallbacks: {getattr(config.ocr, 'fallback_backends', [])}")
        print(f"  OCR fallback min conf: {getattr(config.ocr, 'fallback_min_confidence', 0.72)}")
        print(f"  Use VL1.5: {getattr(config.ocr, 'use_vl15', True)}")
        print(f"  Source lang: {config.translation.source_lang}")
        print(f"\nSegmentation:")
        print(f"  Precise masks: {config.segmentation.enable_precise_masks}")
        print(f"  Backend: {config.segmentation.backend}")
        print(f"  Multi-mask: {config.segmentation.use_multimask}")
        print(f"\nTranslation:")
        print(f"  Mode: {getattr(config.translation, 'translation_mode', 'hybrid')}")
        print(f"  {config.translation.source_lang.upper()} → {config.translation.target_lang.upper()}")
        print(f"  Cache: {config.translation.enable_cache}")
        print(f"  BitsAndBytes: {config.translation.use_bitsandbytes} (4bit={config.translation.bnb_4bit}, 8bit={config.translation.bnb_8bit})")
        print(
            f"  Context grouping: {config.translation.enable_context_grouping} "
            f"(distance={config.translation.context_distance_threshold}, max_group={config.translation.max_group_size})"
        )
        print(f"\nRendering:")
        print(f"  Inpainting: {config.rendering.inpainting_method}")
        print(f"  Font size: {config.rendering.min_font_size}-{config.rendering.max_font_size}")
        print(f"  Preserve text color: {config.rendering.preserve_original_text_color}")
        print(f"  Auto style typesetting: {config.rendering.auto_style_typesetting}")
        print(f"  Lock text to OCR regions: {config.rendering.lock_text_to_ocr_regions} (system_only={config.rendering.lock_text_system_only})")
        return
    
    # Appliquer arguments
    if args.no_cache:
        config.translation.enable_cache = False

    # Mode traduction
    import os
    has_gemini_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

    if getattr(args, 'api', False):
        config.translation.translation_mode = "gemini"
        config.translation.backend = "gemini"
    elif args.translation_mode:
        config.translation.translation_mode = args.translation_mode
    else:
        # Force NLLB default for 0 API cost
        config.translation.translation_mode = "nllb"
        config.translation.backend = "nllb"

    if config.translation.translation_mode in {"hybrid", "hybrid_quality", "qwen"}:
        config.translation.backend = "local_llm"
    elif config.translation.translation_mode == "nllb":
        config.translation.backend = "nllb"

    # Si --series fourni sans valeur, demander le nom maintenant (avant sélection input)
    if args.series == '':
        args.series = input("Nom de la série : ").strip()
        if not args.series:
            print("Erreur: Aucun nom de série fourni.")
            sys.exit(1)

    # Initialiser logger
    log_file = LOGS_DIR / "webtoon_v5.log"
    logger = init_logger(log_file=log_file, level=args.log_level)
    
    logger.header("WEBTOON TRANSLATOR V5 PREMIUM")
    
    # Vérifications
    if args.image:
        if not args.image.exists():
            logger.error(f"Image introuvable: {args.image}")
            sys.exit(1)
        input_path = args.image
        mode = "single"
    else:
        # En mode série, input_path = manhwa/<slug>
        if args.series and SeriesDB is not None:
            raw_series = args.series.strip().lower()
            slug = raw_series.replace(" ", "_")
            manhwa_base = Path("manhwa")
            manhwa_dir = manhwa_base / slug
            
            # Fuzzy match si introuvable
            if not manhwa_dir.exists() and manhwa_base.exists():
                search_term = raw_series.replace("_", " ").replace("-", " ")
                for d in manhwa_base.iterdir():
                    if d.is_dir():
                        d_name = d.name.lower().replace("_", " ").replace("-", " ")
                        if search_term in d_name:
                            manhwa_dir = d
                            args.series = d.name # met à jour le nom exact pour la suite
                            slug = d.name
                            break

            if manhwa_dir.exists():
                input_path = manhwa_dir
                mode = "batch"
            else:
                logger.error(f"Dossier manhwa/{slug} (ou similaire) introuvable.")
                sys.exit(1)
        else:
            if not args.input.exists():
                logger.error(f"Dossier/Fichier input introuvable: {args.input}")
                sys.exit(1)
            input_path = args.input
            mode = "single" if input_path.is_file() else "batch"
    
    # Vérifier modèle YOLO
    if not config.YOLO_MODEL_PATH.exists():
        logger.error(f"Modèle YOLO introuvable: {config.YOLO_MODEL_PATH}")
        logger.info("Placez le modèle manhwa_v2.pt dans assets/models/")
        sys.exit(1)
    
    # Stats système
    logger.section("SYSTÈME")
    logger.stat("Device", MemoryManager.get_device())
    
    ram = MemoryManager.get_ram_usage()
    logger.stat("RAM", f"{ram['available_gb']:.1f} GB disponible")
    
    vram = MemoryManager.get_vram_usage()
    if vram:
        logger.stat("VRAM", f"{vram['allocated_gb']:.2f} GB allouée")
    
    # Configuration
    logger.section("CONFIGURATION")
    logger.stat("Mode", mode)
    logger.stat("Debug", "OUI" if args.debug else "non")
    logger.stat("Input", str(input_path))
    logger.stat("Output", str(args.output))
    logger.stat("Adaptive slicing", "ON" if config.detection.enable_adaptive_slicing else "OFF")
    logger.stat("Multi-scale", str(config.detection.detection_scales))
    logger.stat("OCR", str(config.ocr.backend))
    logger.stat("Segmentation", f"{config.segmentation.backend} (precise={config.segmentation.enable_precise_masks})")
    logger.stat("Translation", f"{config.translation.source_lang.upper()} → {config.translation.target_lang.upper()}")
    logger.stat("Translation mode", str(getattr(config.translation, 'translation_mode', 'hybrid')))
    logger.stat("Cache", "Activé" if config.translation.enable_cache else "Désactivé")
    
    # Créer pipeline avec mode debug
    logger.info("")

    # ── Mode Série ──
    series_db = None

    if args.series and SeriesDB is not None:
        series_dir = Path("data/series")
        series_db = SeriesDB(series_dir, args.series, logger=logger)

        if args.init_series:
            series_db.init_series(name=args.init_series)
            logger.info(f"   📚 Série initialisée: {args.init_series}")
        else:
            status = series_db.get_status()
            logger.info(f"   📚 Série: {status['series']}")
            logger.info(f"   📖 Glossaire: {status['glossary_entries']} entrées")
            logger.info(f"   👤 Personnages: {status['characters']}")

        # Recherche automatique des chapitres et images
        slug = args.series.strip().lower().replace(" ", "_")
        manhwa_dir = Path("manhwa") / slug
        chapitres = []
        total_images = 0
        if manhwa_dir.exists():
            for ch in manhwa_dir.iterdir():
                if ch.is_dir():
                    chapitres.append(ch.name)
                    images = [f for f in ch.iterdir() if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg"}]
                    total_images += len(images)
            logger.info(f"   📂 Dossier manhwa/{slug} trouvé.")
            logger.info(f"   📑 Chapitres: {len(chapitres)}")
            logger.info(f"   🖼️  Images totales: {total_images}")
            logger.info(f"   Liste chapitres: {chapitres}")
        else:
            logger.warning(f"   ⚠️  Dossier manhwa/{slug} introuvable.")

        # Demande confirmation avant traduction
        confirm = input(f"Lancer la traduction pour {len(chapitres)} chapitres et {total_images} images ? (o/n) : ").strip().lower()
        if confirm != "o":
            logger.info("Traduction annulée.")
            sys.exit(0)

        # Définir le dossier de sortie en mode série
        args.output = Path("manhwa_trad") / slug
        args.output.mkdir(parents=True, exist_ok=True)

        if args.chapter is not None:
            series_db.start_chapter(args.chapter)

        logger.stat("Mode série", f"{args.series}" + (f" ch.{args.chapter}" if args.chapter else ""))
    elif args.series and SeriesDB is None:
        logger.warning("⚠️  Mode série demandé mais module series_db non trouvé")

    pipeline = TranslationPipeline(logger, debug=args.debug, series_db=series_db)
    
    # Traiter
    logger.start_timer()
    
    try:
        if mode == "single":
            stats = pipeline.process_image(input_path, args.output)
            if not stats.get('success', True):
                logger.error("Échec du traitement")
                sys.exit(1)
        else:
            stats = pipeline.process_directory(input_path, args.output)
            if stats.get('failed', 0) > 0:
                logger.warning(f"{stats['failed']} images ont échoué")
    
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Interruption utilisateur")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        logger.end_timer()

    # ── Finaliser le chapitre série ──
    if series_db:
        series_db.finalize_chapter()
        warnings = series_db.run_consistency_check()
        if warnings:
            logger.warning(f"   ⚠️  Incohérences détectées:")
            for w in warnings:
                logger.warning(f"      {w}")
    
    logger.header("✅ TERMINÉ")
    
    if args.debug:
        logger.info(f"🐛 Fichiers debug dans: {args.output / 'debug'}")


if __name__ == "__main__":
    main()