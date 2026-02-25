from PIL import Image
import math
import os

# ==== CONFIGURATION ====

input_path = r"A:\omni read\manhwa\the_cleaner\chapitre 001\Chapitre 001_merged_part01.jpg"   # mets ton image ici
output_folder = "decoupe"      # dossier de sortie
target_width = 1080            # format TikTok 3:4
target_height = 1440

# ========================

# Créer le dossier si non existant
os.makedirs(output_folder, exist_ok=True)

# Charger l'image
img = Image.open(input_path)
w, h = img.size

# Si ton image n'est pas en 1080px de large, on la resize proprement
if w != target_width:
    new_height = int((target_width / w) * h)
    img = img.resize((target_width, new_height), Image.LANCZOS)
    w, h = img.size

# Calcul du nombre de morceaux verticaux
num_parts = math.ceil(h / target_height)

print(f"Largeur finale : {w}px  |  Hauteur : {h}px")
print(f"Découpage en {num_parts} parties…")

# Découpage
for i in range(num_parts):
    top = i * target_height
    bottom = min(h, top + target_height)
    
    crop = img.crop((0, top, w, bottom))
    crop.save(f"{output_folder}/part_{i+1}.png")

print("Découpage terminé !")
