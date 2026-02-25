#!/usr/bin/env python3
"""Slice / select PNG files from a `manhwa_trad` work.

Usage examples:
  python scripts/slice_manhwa_trad.py --work "the_cleaner" --root manhwa_trad
  python scripts/slice_manhwa_trad.py --work "the_cleaner" --copy --out out_dir
  python scripts/slice_manhwa_trad.py --work "the_cleaner" --manifest manhwa_manifest.json
"""
import argparse
import json
import shutil
from pathlib import Path
from typing import List
try:
    from PIL import Image
except Exception:
    Image = None


def find_pngs_in_chapter(chapter_path: Path) -> List[Path]:
    return sorted([p for p in chapter_path.iterdir() if p.is_file() and p.suffix.lower() == '.png'])


def collect_work(root: Path, work_name: str):
    work_dir = root / work_name
    if not work_dir.exists() or not work_dir.is_dir():
        raise FileNotFoundError(f"Oeuvre introuvable: {work_dir}")

    chapters = [p for p in sorted(work_dir.iterdir()) if p.is_dir()]
    result = {}
    for chap in chapters:
        pngs = find_pngs_in_chapter(chap)
        result[chap.name] = [str(p) for p in pngs]
    return result


def copy_pngs(manifest: dict, root: Path, out: Path, work_name: str):
    for chap_name, files in manifest.items():
        dest_chap = out / work_name / chap_name
        dest_chap.mkdir(parents=True, exist_ok=True)
        for f in files:
            src = Path(f)
            if not src.exists():
                print(f"Ignoré (manquant): {src}")
                continue
            shutil.copy2(src, dest_chap / src.name)


def pad_chapter_name(name: str) -> str:
    return name.zfill(3) if name.isdigit() else name


def slice_image_to_parts(src: Path, height: int, dest_dir: Path, chapter_padded: str, part_start: int) -> int:
    if Image is None:
        raise RuntimeError('Pillow non installé. Installez-le avec: pip install Pillow')

    img = Image.open(src)
    w, h = img.size
    y = 0
    part_idx = part_start
    while y < h:
        bottom = min(y + height, h)
        crop = img.crop((0, y, w, bottom))
        part_idx += 1
        name = f"{chapter_padded}_part{part_idx:03d}.png"
        out_path = dest_dir / name
        crop.save(out_path)
        y += height
    return part_idx


def slice_and_copy(manifest: dict, out: Path, work_name: str, height: int):
    for chap_name, files in manifest.items():
        dest_chap = out / work_name / chap_name
        dest_chap.mkdir(parents=True, exist_ok=True)
        chapter_padded = pad_chapter_name(chap_name)
        part_counter = 0
        for f in files:
            src = Path(f)
            if not src.exists():
                print(f"Ignoré (manquant): {src}")
                continue
            try:
                part_counter = slice_image_to_parts(src, height, dest_chap, chapter_padded, part_counter)
            except Exception as e:
                print(f"Erreur en découpant {src}: {e}")


def main():
    p = argparse.ArgumentParser(description="Sélectionne les PNGs d'une oeuvre dans manhwa_trad")
    p.add_argument('--work', required=True, help='Nom du dossier de l\'oeuvre (ex: the_cleaner)')
    p.add_argument('--root', default='manhwa_trad', help='Dossier racine contenant les oeuvres')
    p.add_argument('--copy', action='store_true', help='Copier les PNGs vers --out (préserve structure)')
    p.add_argument('--out', default='out_manhwa', help='Dossier de sortie pour la copie')
    p.add_argument('--height', type=int, default=1280, help='Hauteur des parts en pixels (défaut: 1280)')
    p.add_argument('--manifest', help='Fichier JSON pour écrire la liste des PNGs')
    p.add_argument('--print', dest='do_print', action='store_true', help='Imprimer la liste trouvée')
    args = p.parse_args()

    root = Path(args.root)
    work = args.work

    try:
        manifest = collect_work(root, work)
    except FileNotFoundError as e:
        print(e)
        return

    if args.manifest:
        with open(args.manifest, 'w', encoding='utf-8') as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2)
        print(f"Manifest écrit: {args.manifest}")

    if args.do_print or not args.manifest:
        for chap, files in manifest.items():
            print(f"Chapitre: {chap} — {len(files)} PNG(s)")
            for f in files:
                print(f"  {f}")

    if args.copy:
        out = Path(args.out)
        try:
            slice_and_copy(manifest, out, work, args.height)
            print(f"Découpage et copie terminés dans: {out / work} (parts hauteur {args.height}px)")
        except RuntimeError as e:
            print(e)
            print("Pour copier sans découper, réexécutez sans --copy ou installez Pillow.")


if __name__ == '__main__':
    main()
