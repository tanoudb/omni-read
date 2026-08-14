import os
import sys
from pathlib import Path
import time

# Forcer l'utilisation de configurations légères pour le test
os.environ['WEBTOON_TRANSLATION_MODE'] = 'nllb'
os.environ['WEBTOON_USE_FP16'] = '1'

# Importer les composants
try:
    from core import NLLBTranslator
    # Patch de la classe NLLBTranslator pour ne pas charger le modèle et retourner l'anglais
    def dummy_load_model(self):
        print("🔧 Patch DUMMY NLLB: Bypass du chargement du modèle")
        pass
        
    def dummy_translate_batch(self, texts, **kwargs):
        print(f"🔧 Patch DUMMY NLLB: Traduction simulée de {len(texts)} textes (retourne la source)")
        return texts
        
    def dummy_translate(self, text, **kwargs):
        return text

    NLLBTranslator._load_model = dummy_load_model
    NLLBTranslator.translate_batch = dummy_translate_batch
    NLLBTranslator.translate = dummy_translate
    
except Exception as e:
    print(f"Erreur de patch NLLB: {e}")

# Lancer via main
from main import main

if __name__ == "__main__":
    test_input = "manhwa/i-married-the-dragon-i-killed/Chapitre 001"
    test_output = "output/test_dummy"
    print(f"\n🚀 Lancement du test sur {test_input}")
    print(f"📁 Sortie prévue dans {test_output}\n")
    
    sys.argv = [
        "main.py", 
        "--input", test_input, 
        "--translation-mode", "nllb", 
        "--debug", 
        "--output", test_output
    ]
    main()
