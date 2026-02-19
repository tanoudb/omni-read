from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import time
import traceback

from batch_pipeline import BatchPipeline
from utils import init_logger
from oeuvre_manager import OeuvreManager


def tri_naturel(s: str) -> list:
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", str(s))]


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}


def _format_duree(secondes: int) -> str:
    heures = secondes // 3600
    minutes = (secondes % 3600) // 60
    return f"{heures}h {minutes}min"


def _append_erreur_log(erreurs_log: Path, nom_chap: str, image_name: str, exc: Exception):
    erreurs_log.parent.mkdir(parents=True, exist_ok=True)
    contenu = (
        f"[{datetime.now().isoformat()}]\n"
        f"chapitre: {nom_chap}\n"
        f"image: {image_name}\n"
        f"erreur: {repr(exc)}\n"
        f"traceback:\n{traceback.format_exc()}\n"
        f"{'-' * 80}\n"
    )
    with erreurs_log.open("a", encoding="utf-8") as f:
        f.write(contenu)


def _load_progression(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_progression(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _clear_progression(path: Path):
    _save_progression(path, {})


def main():
    parser = argparse.ArgumentParser(description="WEBTOON TRANSLATOR — MODE SÉRIE")
    parser.add_argument("--dry-run", action="store_true", help="Simulation sans traduire")
    parser.add_argument("--debug", action="store_true", help="Active les artefacts debug du pipeline")
    args = parser.parse_args()

    print("WEBTOON TRANSLATOR — MODE SÉRIE")

    nom_oeuvre = input("Nom de l'œuvre : ").strip()
    oeuvre = OeuvreManager.charger_ou_creer(nom_oeuvre)
    progression_path = oeuvre.base_dir / "progression.json"
    erreurs_log_path = oeuvre.base_dir / "erreurs.log"

    termes = oeuvre.get_termes_glossaire()
    print(f"Glossaire actuel : {len(termes)} termes → {termes}")
    if input("Ajouter des termes au glossaire ? (o/n) ").strip().lower() == "o":
        oeuvre.ajouter_termes_glossaire_interactif()

    base_input = Path("manhwa") / oeuvre.slug
    if not base_input.exists() or not base_input.is_dir():
        print(f"Erreur : dossier introuvable {base_input}")
        raise SystemExit(1)

    chapitres = sorted([p for p in base_input.iterdir() if p.is_dir()], key=lambda p: tri_naturel(p.name))
    if not chapitres:
        print("Erreur : aucun chapitre trouvé")
        raise SystemExit(1)

    a_traduire = []
    total_images = 0

    for chapitre in chapitres:
        nom_chap = chapitre.name
        images = sorted([p for p in chapitre.iterdir() if _is_image(p)], key=lambda p: tri_naturel(p.name))

        if oeuvre.chapitre_deja_traduit(nom_chap):
            print(f"⚠️  {nom_chap} déjà traduit")
            if input("Refaire ? (o/n) ").strip().lower() != "o":
                continue

        a_traduire.append((nom_chap, chapitre, images))
        total_images += len(images)

    if not a_traduire:
        print("Aucun chapitre à traduire.")
        raise SystemExit(0)

    chap_names = ", ".join(nom for nom, _, _ in a_traduire)
    print(f"Chapitres à traduire : {chap_names} ({total_images} images total)")

    if args.dry_run:
        estimation_sec = total_images * 45
        print(f"Temps estimé : {_format_duree(estimation_sec)}")
        print(f"Glossaire : {len(oeuvre.get_termes_glossaire())} termes")
        print("MODE SIMULATION — rien ne sera traduit")
        raise SystemExit(0)

    if input("Lancer la traduction ? (o/n) ").strip().lower() != "o":
        raise SystemExit(0)

    logger = init_logger(level="INFO")
    pipeline = BatchPipeline(logger, debug=args.debug)
    if hasattr(pipeline, "set_glossaire"):
        pipeline.set_glossaire(oeuvre.get_termes_glossaire())

    base_output = Path("manhwa_trad") / oeuvre.slug
    chapitres_traduits = 0
    images_ok_total = 0
    images_erreur_total = 0
    t0_total = time.perf_counter()

    for idx_chap, (nom_chap, chap_dir, images) in enumerate(a_traduire, start=1):
        print("═══════════════════════════════")
        print(f"  {nom_chap}  ({idx_chap}/{len(a_traduire)})")
        print("═══════════════════════════════")

        output_dir = base_output / nom_chap
        output_dir.mkdir(parents=True, exist_ok=True)

        t_debut = time.perf_counter()
        progression = _load_progression(progression_path)
        images_deja_ok = set()
        if progression.get("chapitre") == nom_chap:
            images_deja_ok = {
                str(name)
                for name in progression.get("images_reussies", [])
                if isinstance(name, str) and name.strip()
            }

        image_names_disponibles = {img.name for img in images}
        images_deja_ok = {name for name in images_deja_ok if name in image_names_disponibles}

        if images_deja_ok:
            last_ok = str(progression.get("derniere_image_reussie") or "")
            print(f"⚠️  Progression détectée pour {nom_chap} ({len(images_deja_ok)} images déjà réussies)")
            if last_ok:
                print(f"Dernière image réussie : {last_ok}")
            reprendre = input("Reprendre depuis la dernière image réussie ? (o/n) ").strip().lower() == "o"
            if not reprendre:
                images_deja_ok = set()
                _clear_progression(progression_path)

        compteur_ok = len(images_deja_ok)
        compteur_erreur = 0

        images_a_traiter = [img for img in images if img.name not in images_deja_ok]

        for idx_img, image_path in enumerate(images, start=1):
            if image_path.name in images_deja_ok:
                print(f"  [{idx_img}/{len(images)}] {image_path.name}... ⏭️ (déjà traité)")

        def _progress_callback(index: int, total: int, success: bool):
            nonlocal compteur_ok
            if index < 1 or index > len(images_a_traiter):
                return
            image_name = images_a_traiter[index - 1].name
            status = "✅" if success else "❌"
            print(f"  [{index}/{total}] {image_name}... {status}")

            if success:
                images_deja_ok.add(image_name)
                compteur_ok += 1
                _save_progression(
                    progression_path,
                    {
                        "chapitre": nom_chap,
                        "derniere_image_reussie": image_name,
                        "images_reussies": sorted(images_deja_ok, key=tri_naturel),
                        "mis_a_jour_le": datetime.now().isoformat(),
                    },
                )

        if images_a_traiter:
            chapter_stats = pipeline.process_chapter(
                images_a_traiter,
                output_dir,
                progress_callback=_progress_callback,
            )

            details = chapter_stats.get("details", []) if isinstance(chapter_stats, dict) else []
            for detail in details:
                if not bool(detail.get("success", False)):
                    compteur_erreur += 1
                    images_erreur_total += 1
                    image_name = str(detail.get("image") or "unknown")
                    error_msg = str(detail.get("error") or "unknown_error")
                    print(f"  ❌ ERREUR : {image_name} -> {error_msg}")
                    _append_erreur_log(erreurs_log_path, nom_chap, image_name, RuntimeError(error_msg))

        oeuvre.marquer_chapitre_traduit(nom_chap, compteur_ok)
        _clear_progression(progression_path)
        duree = time.perf_counter() - t_debut
        print(f"✅ {nom_chap} terminé — {compteur_ok} images en {duree:.0f}s")
        if compteur_erreur:
            print(f"⚠️  {nom_chap}: {compteur_erreur} image(s) en erreur (voir {erreurs_log_path})")
        chapitres_traduits += 1
        images_ok_total += compteur_ok

    duree_totale = int(time.perf_counter() - t0_total)
    print("\nRÉSUMÉ FINAL")
    print("-" * 60)
    print(f"{'Chapitres traduits':<28} | {chapitres_traduits}")
    print(f"{'Images réussies':<28} | {images_ok_total}")
    print(f"{'Images en erreur':<28} | {images_erreur_total}")
    print(f"{'Durée totale':<28} | {duree_totale}s")
    print("-" * 60)
    print(f"🎉 Série terminée — {chapitres_traduits} chapitres traduits")


if __name__ == "__main__":
    main()
