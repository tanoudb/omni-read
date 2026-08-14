"""
Redécoupe les images "_partNN" trop hautes en morceaux plus petits.

Les strips webtoon fusionnées ("Chapitre 001_merged_partNN.jpg") font parfois
40 000 à 55 000 px de haut. L'OCR traite chaque détection d'un même fichier
dans le MÊME lot envoyé au worker PaddleOCR-VL — sur une image aussi haute,
le nombre de détections par lot grimpe (60-90 zones vs 5-15 sur une page
normale), et le worker plante plus souvent par manque de mémoire
("worker_error"). Redécouper en morceaux plus petits réduit le lot par appel
et rend la progression visible plus tôt (moins d'attente entre deux lignes de
log).

Usage : python utils/resplit_tall_images.py <dossier_chapitre> [--target-height 8000] [--threshold 12000]
"""
import argparse
import re
from pathlib import Path

import cv2


def resplit_chapter_dir(chapter_dir: Path, target_height: int = 8000, threshold: int = 12000) -> int:
    """
    Redécoupe tous les fichiers `*_partNN.<ext>` du dossier plus hauts que
    `threshold`, puis renumérote l'ENSEMBLE des fichiers du dossier pour
    préserver l'ordre de lecture (les morceaux issus d'un même fichier
    doivent rester consécutifs et dans le bon sens).

    Retourne le nombre de fichiers dans le dossier après redécoupage.
    """
    pattern = re.compile(r'^(.*_part)(\d+)(\.\w+)$', re.IGNORECASE)
    files = sorted(
        [f for f in chapter_dir.iterdir() if f.is_file() and pattern.match(f.name)],
        key=lambda f: int(pattern.match(f.name).group(2)),
    )
    if not files:
        return 0

    prefix = pattern.match(files[0].name).group(1)
    ext = pattern.match(files[0].name).group(3)

    # Ordre final des morceaux (chemins temporaires si issus d'un split, sinon
    # le fichier d'origine tel quel).
    ordered_chunks = []

    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            ordered_chunks.append(f)
            continue
        h = img.shape[0]
        if h <= threshold:
            ordered_chunks.append(f)
            continue

        n_chunks = max(2, round(h / target_height))
        chunk_h = -(-h // n_chunks)  # ceil
        tmp_paths = []
        for i in range(n_chunks):
            y0 = i * chunk_h
            y1 = min(h, y0 + chunk_h)
            if y0 >= h:
                break
            piece = img[y0:y1]
            tmp_path = f.with_name(f"{f.stem}__split{i:02d}{ext}")
            cv2.imwrite(str(tmp_path), piece, [cv2.IMWRITE_JPEG_QUALITY, 100])
            tmp_paths.append(tmp_path)
        f.unlink()
        ordered_chunks.extend(tmp_paths)

    # Renumérotation finale séquentielle (part01, part02, ...).
    final_paths = []
    for i, chunk_path in enumerate(ordered_chunks, start=1):
        final_name = f"{prefix}{i:02d}{ext}"
        final_path = chunk_path.with_name(final_name)
        if chunk_path != final_path:
            # Nom cible potentiellement déjà pris par un fichier de la boucle
            # (ex: part02 original devient le nom cible d'un split de part01) :
            # passe par un nom intermédiaire unique pour ne rien écraser avant
            # que la boucle ait fini de renommer tout le monde.
            tmp = chunk_path.with_name(f"__tmp_{i:03d}{ext}")
            chunk_path.rename(tmp)
            final_paths.append((tmp, final_path))
        else:
            final_paths.append((chunk_path, final_path))

    for tmp, final_path in final_paths:
        tmp.rename(final_path)

    return len(final_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("chapter_dir", type=Path)
    parser.add_argument("--target-height", type=int, default=8000)
    parser.add_argument("--threshold", type=int, default=12000)
    args = parser.parse_args()

    n = resplit_chapter_dir(args.chapter_dir, args.target_height, args.threshold)
    print(f"{n} fichiers dans {args.chapter_dir} après redécoupage")
