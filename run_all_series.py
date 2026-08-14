# -*- coding: utf-8 -*-
"""
=======================================================================
RUN ALL SERIES -- Traite automatiquement toutes les series manhwa
=======================================================================

Usage:
    python run_all_series.py               # Traite toutes les series non traduites
    python run_all_series.py --dry-run     # Affiche ce qui serait traite sans lancer
    python run_all_series.py --serie nom   # Traite une serie specifique
    python run_all_series.py --force       # Force le re-traitement meme si output existe

Objectif : zero cout API (Qwen local ou NLLB).
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path

# Compat Windows UTF-8 -- doit etre avant tout print()
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    else:
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
except Exception:
    pass

# Forcer mode traduction local (zero API)
os.environ.setdefault("WEBTOON_FORCE_LOCAL", "1")
os.environ['FLAGS_allocator_strategy'] = 'auto_growth'

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


MANHWA_DIR = Path("manhwa")
OUTPUT_DIR = Path("manhwa_trad")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def discover_series():
    """
    Scanne manhwa/ et retourne la liste des series avec leurs chapitres.
    Structure attendue :
        manhwa/<serie_slug>/<Chapitre XXX>/image.png
    """
    series = {}
    if not MANHWA_DIR.exists():
        return series

    for series_dir in sorted(MANHWA_DIR.iterdir()):
        if not series_dir.is_dir():
            continue

        slug = series_dir.name
        chapters = {}

        # Recherche chapitres (sous-dossiers)
        for ch_dir in sorted(series_dir.iterdir()):
            if not ch_dir.is_dir():
                continue
            images = [
                f for f in ch_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            ]
            if images:
                chapters[ch_dir.name] = sorted(images)

        # Recherche images directement dans le dossier serie
        if not chapters:
            direct_images = [
                f for f in series_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            ]
            if direct_images:
                chapters["__root__"] = sorted(direct_images)

        if chapters:
            series[slug] = {
                'path': series_dir,
                'chapters': chapters,
                'total_images': sum(len(imgs) for imgs in chapters.values()),
            }

    return series


def is_chapter_done(slug: str, ch_name: str) -> bool:
    """Retourne True si le chapitre a deja un output traduit."""
    if ch_name == "__root__":
        out_dir = OUTPUT_DIR / slug
    else:
        out_dir = OUTPUT_DIR / slug / ch_name

    if not out_dir.exists():
        return False

    for f in out_dir.iterdir():
        if f.is_file() and "_translated" in f.stem and f.suffix.lower() in IMAGE_EXTENSIONS:
            return True

    return False


def print_discovery_report(series: dict, force: bool = False):
    """Affiche un rapport de la decouverte."""
    total_images = 0
    total_todo = 0
    total_done = 0

    print("\n" + "=" * 70)
    print("[SERIES] Detectees dans manhwa/")
    print("=" * 70)

    for slug, info in sorted(series.items()):
        print(f"\n  [SERIE] {slug} ({info['total_images']} images)")
        for ch_name, imgs in sorted(info['chapters'].items()):
            done = is_chapter_done(slug, ch_name)
            if done and not force:
                status = "[OK]   "
            elif done and force:
                status = "[REDO] "
            else:
                status = "[TODO] "
            print(f"         {ch_name:45s} {len(imgs):3d} pages  {status}")
            total_images += len(imgs)
            if done and not force:
                total_done += len(imgs)
            else:
                total_todo += len(imgs)

    print("\n" + "=" * 70)
    print(f"  Total: {total_images} images | a traiter: {total_todo} | deja traduit: {total_done}")
    print("=" * 70 + "\n")


def run_series(slug: str, series_info: dict, force: bool, dry_run: bool, logger=None) -> dict:
    """
    Lance le pipeline sur une serie complete.
    Retourne un dict de stats.
    """
    from config import config, LOGS_DIR
    from utils import init_logger
    from pipeline import TranslationPipeline

    # Forcer mode Qwen local (zero API)
    config.translation.translation_mode = "qwen"
    config.translation.backend = "local_llm"

    stats = {
        'slug': slug,
        'total_chapters': 0,
        'processed_chapters': 0,
        'skipped_chapters': 0,
        'total_images': 0,
        'processed_images': 0,
        'failed_images': 0,
        'errors': [],
        'start_time': time.time(),
        'end_time': None,
    }

    chapters = series_info['chapters']
    chapters_to_process = {}

    for ch_name, imgs in sorted(chapters.items()):
        stats['total_chapters'] += 1
        stats['total_images'] += len(imgs)
        if is_chapter_done(slug, ch_name) and not force:
            print(f"  [SKIP] {ch_name} : deja traduit")
            stats['skipped_chapters'] += 1
            continue
        chapters_to_process[ch_name] = imgs

    if not chapters_to_process:
        print(f"  [OK] {slug} : tous les chapitres deja traduits")
        stats['end_time'] = time.time()
        return stats

    if dry_run:
        for ch_name, imgs in sorted(chapters_to_process.items()):
            print(f"  [DRY] Traiterait: {ch_name} ({len(imgs)} images)")
        stats['end_time'] = time.time()
        return stats

    # Initialiser le logger par serie
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"series_{slug}_{int(time.time())}.log"
    if logger is None:
        logger = init_logger(log_file=log_file, level='INFO')

    logger.header(f"SERIE: {slug}")

    # Verifier modele YOLO
    if not config.YOLO_MODEL_PATH.exists():
        msg = f"Modele YOLO introuvable: {config.YOLO_MODEL_PATH}"
        logger.error(msg)
        stats['errors'].append(msg)
        stats['end_time'] = time.time()
        return stats

    # Creer le pipeline
    pipeline = TranslationPipeline(logger, debug=False)

    for ch_name, imgs in sorted(chapters_to_process.items()):
        if ch_name == "__root__":
            out_dir = OUTPUT_DIR / slug
        else:
            out_dir = OUTPUT_DIR / slug / ch_name

        out_dir.mkdir(parents=True, exist_ok=True)
        logger.header(f"Chapitre: {ch_name} -- {len(imgs)} images")

        src_dir = imgs[0].parent
        try:
            ch_stats = pipeline.process_directory(src_dir, out_dir)
            stats['processed_chapters'] += 1
            stats['processed_images'] += ch_stats.get('processed', 0)
            stats['failed_images'] += ch_stats.get('failed', 0)
        except Exception as e:
            err_msg = f"{ch_name}: {e}"
            logger.error(f"  Erreur chapitre {ch_name}: {e}")
            stats['errors'].append(err_msg)
            import traceback
            traceback.print_exc()

    stats['end_time'] = time.time()
    elapsed = (stats['end_time'] or time.time()) - stats['start_time']
    logger.header(f"[DONE] {slug} termine en {elapsed:.0f}s")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Traduction automatique de toutes les series manhwa (zero API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help="Afficher les series a traiter sans lancer le pipeline",
    )
    parser.add_argument(
        '--force', '-f', action='store_true',
        help="Re-traiter meme les chapitres deja traduits",
    )
    parser.add_argument(
        '--serie', '--series', type=str, default=None,
        help="Traiter uniquement cette serie (slug, ex: rise-of-the-dragon-overlord)",
    )
    parser.add_argument(
        '--list', action='store_true',
        help="Lister les series disponibles et quitter",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  WEBTOON TRANSLATOR -- RUN ALL SERIES")
    print("  Mode : Zero API (Qwen local)")
    print("=" * 70 + "\n")

    # Decouvrir les series
    series = discover_series()

    if not series:
        print(f"[ERROR] Aucune serie trouvee dans {MANHWA_DIR.absolute()}")
        print("   Structure attendue: manhwa/<serie>/<Chapitre XXX>/image.png")
        sys.exit(1)

    # Filtrer si --serie specifie
    if args.serie:
        slug = args.serie.strip().lower().replace(" ", "_")
        if slug not in series:
            slug_dash = slug.replace("_", "-")
            if slug_dash in series:
                slug = slug_dash
            else:
                print(f"[ERROR] Serie '{args.serie}' introuvable. Series disponibles:")
                for s in sorted(series.keys()):
                    print(f"   - {s}")
                sys.exit(1)
        series = {slug: series[slug]}

    print_discovery_report(series, force=args.force)

    if args.list:
        sys.exit(0)

    if args.dry_run:
        print("[DRY RUN] Aucun traitement lance.\n")
        for slug, info in sorted(series.items()):
            run_series(slug, info, force=args.force, dry_run=True)
        sys.exit(0)

    # Compter ce qui reste a faire
    total_to_do = 0
    for slug, info in series.items():
        for ch_name in info['chapters']:
            if not is_chapter_done(slug, ch_name) or args.force:
                total_to_do += len(info['chapters'][ch_name])

    if total_to_do == 0:
        print("[OK] Toutes les series sont deja traduites. Utilisez --force pour retraiter.")
        sys.exit(0)

    print(f"[INFO] {total_to_do} images a traduire (mode Qwen local, zero API).")
    try:
        confirm = input("Lancer ? (o/n) : ").strip().lower()
        if confirm not in ("o", "y", "oui", "yes"):
            print("Annule.")
            sys.exit(0)
    except KeyboardInterrupt:
        print("\nAnnule.")
        sys.exit(0)

    # Lancer la traduction
    all_stats = []
    global_start = time.time()

    for slug, info in sorted(series.items()):
        print(f"\n[START] Traitement: {slug}")
        try:
            s = run_series(slug, info, force=args.force, dry_run=False)
            all_stats.append(s)
        except KeyboardInterrupt:
            print("\n[WARN] Interruption. Sauvegarde du rapport partiel...")
            break
        except Exception as e:
            print(f"[ERROR] Erreur fatale sur {slug}: {e}")
            import traceback
            traceback.print_exc()
            all_stats.append({'slug': slug, 'errors': [str(e)]})

    # Rapport final
    elapsed = time.time() - global_start
    total_processed = sum(s.get('processed_images', 0) for s in all_stats)
    total_failed = sum(s.get('failed_images', 0) for s in all_stats)
    total_errors = sum(len(s.get('errors', [])) for s in all_stats)

    print("\n" + "=" * 70)
    print("[DONE] TRAITEMENT TERMINE")
    print("=" * 70)
    print(f"  Series traitees : {len(all_stats)}")
    print(f"  Images reussies : {total_processed}")
    print(f"  Images echouees : {total_failed}")
    print(f"  Erreurs         : {total_errors}")
    print(f"  Temps total     : {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("=" * 70)

    # Sauvegarder le rapport
    report_path = OUTPUT_DIR / "run_all_series_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'elapsed_seconds': elapsed,
                'total_processed': total_processed,
                'total_failed': total_failed,
                'series': all_stats,
            }, f, ensure_ascii=False, indent=2)
        print(f"\n[REPORT] {report_path}")
    except Exception as e:
        print(f"[WARN] Rapport non sauvegarde: {e}")


if __name__ == "__main__":
    main()
