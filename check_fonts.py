"""
Script: Vérifier les polices Windows disponibles
Aide à déboguer les problèmes de polices manquantes
"""

from pathlib import Path

FONTS_DIR = Path("C:/Windows/Fonts")

print("=" * 80)
print("🔍 VÉRIFICATION POLICES WINDOWS")
print("=" * 80)

fonts_to_check = [
    "impact.ttf",
    "impactbd.ttf",
    "arialbd.ttf",
    "arial.ttf",
    "ariblk.ttf",
    "calibrib.ttf",
    "cour.ttf",
    "times.ttf",
    "georgia.ttf",
    "verdanab.ttf",
    "framd.ttf",      # Franklin Gothic Medium
    "comic.ttf",      # Comic Sans (si existant)
    "comicbd.ttf",    # Comic Sans Bold (si existant)
]

print("\n📋 Polices disponibles:\n")

available = []
missing = []

for font in fonts_to_check:
    path = FONTS_DIR / font
    if path.exists():
        available.append(font)
        print(f"   ✅ {font:20s} ({path})")
    else:
        missing.append(font)
        print(f"   ❌ {font:20s} (manquant)")

print(f"\n📊 Résumé:")
print(f"   ✅ Disponibles: {len(available)}")
print(f"   ❌ Manquantes: {len(missing)}")

# Lister toutes les polices TTF disponibles
print(f"\n🎨 Toutes les polices TTF du système:\n")

all_ttf = sorted(FONTS_DIR.glob("*.ttf"))
print(f"   Trouvées: {len(all_ttf)} polices .ttf\n")

for i, font_path in enumerate(all_ttf[:20], 1):  # Afficher les 20 premières
    print(f"   {i:2d}. {font_path.name}")

if len(all_ttf) > 20:
    print(f"\n   ... et {len(all_ttf) - 20} autres")

# Recommandations
print("\n" + "=" * 80)
print("✅ POLICES RECOMMANDÉES POUR MANGA (garanties d'exister):")
print("=" * 80)

recommended = [
    ("impact.ttf", "Gras, lisible, style manga parfait"),
    ("arialbd.ttf", "Sans-serif gras, très lisible"),
    ("ariblk.ttf", "Arial Black, ultra gras"),
    ("georgia.ttf", "Serif élégant (fallback)"),
    ("verdanab.ttf", "Verdana Bold, lisible"),
]

for font, desc in recommended:
    path = FONTS_DIR / font
    status = "✅" if path.exists() else "⚠️"
    print(f"\n{status} {font}")
    print(f"   → {desc}")

print("\n" + "=" * 80)
