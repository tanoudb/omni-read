from pathlib import Path
import re
from utils.series_db import SeriesDB

def process_series_mode(logger):
    """
    Handles the series mode: prompts for series name, initializes or loads series,
    and prepares for translation.
    """
    series_name = input("Nom de la série : ").strip()
    if not series_name:
        logger.error("Nom de série invalide.")
        return

    # Initialize or load the series database
    series_slug = re.sub(r"[^a-zA-Z0-9_]+", "_", series_name.lower()).strip("_")
    series_path = Path("data/series") / series_slug
    series_db = SeriesDB(series_path, series_slug, logger=logger)

    # Display glossary terms and chapter information
    glossary_terms = series_db.get_termes_glossaire()
    logger.info(f"Glossaire: {len(glossary_terms)} termes")

    chapters_path = Path("manhwa") / series_name
    if chapters_path.exists():
        chapters = [p.name for p in chapters_path.iterdir() if p.is_dir()]
        logger.info(f"Chapitres existants: {chapters}")
    else:
        logger.info("Aucun chapitre existant trouvé.")

    # Prompt to start translation
    start_translation = input("Lancer la traduction ? (o/n) : ").strip().lower()
    if start_translation != "o":
        logger.info("Traduction annulée.")
        return

    # Start translation process (placeholder for actual implementation)
    logger.info("Démarrage de la traduction...")
    # Here you would call the translation pipeline for each chapter

    logger.info("Mode série terminé.")