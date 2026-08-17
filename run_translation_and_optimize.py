import os
import subprocess
import shutil
import sys
from pathlib import Path
from PIL import Image

def process_series():
    # La clé était écrite EN CLAIR dans ce fichier et réécrasait `.env` à
    # chaque exécution. On la lit désormais depuis l'environnement ; si elle
    # n'y est pas, on laisse le `.env` existant tranquille au lieu de
    # l'écraser avec une valeur en dur.
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if api_key:
        with open(".env", "w", encoding="utf-8") as f:
            f.write(f"GEMINI_API_KEY={api_key}\n")
    elif not Path(".env").exists():
        print(
            "Aucune clé: définissez GEMINI_API_KEY dans l'environnement, "
            "ou créez un fichier .env contenant GEMINI_API_KEY=...",
            flush=True,
        )
        return

    failed_series = []

    series_map = {
        "path-of-vengeance": "path_of_vengeance",
        "i-married-the-dragon-i-killed": "i_married_a_dragon",
        "hellogin": "hello_gin"
    }

    base_out = Path("traductions_propres")
    base_out.mkdir(exist_ok=True)

    for slug, nice_name in series_map.items():
        print(f"=== Traitement de {slug} ===", flush=True)
        # 1. Lancer la traduction
        cmd = [r".venv311\Scripts\python.exe", "main.py", "--series", slug, "--api"]
        try:
            print(f"Lancement: {' '.join(cmd)}", flush=True)
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=sys.stdout, stderr=subprocess.STDOUT, text=True)
            # Send 'o\n' when requested. Since we redirect stdout to sys.stdout, we can't easily wait for the prompt.
            # Wait! If we redirect stdout, we must write 'o\n' to stdin immediately and close it.
            p.stdin.write("o\n")
            p.stdin.close()
            p.wait()
            # Le code de sortie n'était jamais lu : « Traduction terminée. »
            # s'affichait même après un plantage, et le dossier PARTIEL était
            # ensuite optimisé et recopié sans le moindre avertissement.
            # Mesuré le 2026-08-15 : i-married-the-dragon-i-killed s'est
            # arrêté au rendu de part03 (3 des 5 fichiers absents), et rien ne
            # le signalait nulle part.
            if p.returncode != 0:
                print(
                    f"[ECHEC] {slug}: main.py a quitté avec le code {p.returncode}. "
                    f"La sortie manhwa_trad/{slug} est probablement INCOMPLÈTE — "
                    f"voir logs/webtoon_v5.log avant de l'utiliser.",
                    flush=True,
                )
                failed_series.append(slug)
            else:
                print("Traduction terminée.", flush=True)
        except Exception as e:
            print(f"Erreur lors de la traduction de {slug}: {e}", flush=True)
            failed_series.append(slug)
            continue
        
        # 2. Optimiser les images (redimensionnement et WebP)
        src_dir = Path("manhwa_trad") / slug
        dst_dir = base_out / nice_name
        
        if not src_dir.exists():
            print(f"Attention: {src_dir} n'existe pas. Skip.", flush=True)
            continue
            
        print(f"Optimisation des images de {slug} vers {dst_dir}...", flush=True)
        
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    src_path = Path(root) / file
                    
                    # Compute relative path to preserve chapter structure
                    rel_path = src_path.relative_to(src_dir)
                    dst_path = dst_dir / rel_path
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Change extension to .webp
                    dst_path = dst_path.with_suffix('.webp')
                    
                    try:
                        with Image.open(src_path) as img:
                            # Convert to RGB to ensure webp compatibility
                            if img.mode not in ('RGB', 'RGBA'):
                                img = img.convert('RGBA' if 'A' in img.mode else 'RGB')
                                
                            width, height = img.size
                            if width > 1080:
                                new_width = 1080
                                new_height = int((new_width / width) * height)
                                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                            
                            img.save(dst_path, "WEBP", quality=85)
                            print(f"Optimisé: {dst_path.name}", flush=True)
                    except Exception as e:
                        print(f"Erreur sur l'image {src_path}: {e}", flush=True)
        print(f"Série {slug} optimisée avec succès.", flush=True)

    if failed_series:
        print("\n" + "=" * 60, flush=True)
        print("SERIES EN ECHEC — sortie incomplète, NE PAS uploader tel quel :", flush=True)
        for slug in failed_series:
            print(f"  - {slug}", flush=True)
        print("=" * 60, flush=True)
        return

    print("\nMission accomplie: dossiers créés et remplis avec les fichiers WebP prêts à être uploadés.", flush=True)

if __name__ == '__main__':
    process_series()
