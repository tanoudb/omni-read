import os
from PIL import Image

# --- CONFIGURATION ---
# Ton chemin spécifique :
DOSSIER_SOURCE = r"A:\omni read\output\test_inpainting_restore\fusion"
NOM_SORTIE = "fusion_finale_longue.png"

def fusionner_images():
    # 1. Vérifier si le dossier existe
    if not os.path.exists(DOSSIER_SOURCE):
        print(f"Erreur : Le dossier est introuvable -> {DOSSIER_SOURCE}")
        return

    # 2. Lister et trier les fichiers PNG
    fichiers = [f for f in os.listdir(DOSSIER_SOURCE) if f.lower().endswith('.png')]
    fichiers.sort() # Trie par ordre : 0001, 0005, 0009...

    if not fichiers:
        print(f"Aucune image PNG trouvée dans : {DOSSIER_SOURCE}")
        return

    print(f"Lecture de {len(fichiers)} images...")

    # 3. Charger les images
    images = []
    largeur_max = 0
    hauteur_totale = 0

    for f in fichiers:
        img = Image.open(os.path.join(DOSSIER_SOURCE, f))
        images.append(img)
        # On calcule la taille finale au fur et à mesure
        if img.width > largeur_max:
            largeur_max = img.width
        hauteur_totale += img.height

    # 4. Créer le canevas vide (RGBA pour la transparence)
    image_finale = Image.new('RGBA', (largeur_max, hauteur_totale))

    # 5. Assemblage vertical
    y_offset = 0
    for img in images:
        image_finale.paste(img, (0, y_offset))
        y_offset += img.height

    # 6. Sauvegarder à la racine du projet
    image_finale.save(NOM_SORTIE)
    print(f"\n--- SUCCÈS ---")
    print(f"L'image longue est générée : {os.path.abspath(NOM_SORTIE)}")

if __name__ == "__main__":
    fusionner_images()