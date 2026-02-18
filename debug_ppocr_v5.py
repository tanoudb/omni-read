"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                   🔍 DEBUG PP-OCRv5 - SCRIPT DE TEST                         ║
║                                                                               ║
║ Teste PaddleOCR v5 sur un crop réel pour:                                    ║
║ 1. Vérifier qu'il détecte les régions OCR précises ✅                        ║
║ 2. Obtenir confiance par région (fallback strategy)                          ║
║ 3. Vérifier que les coordonnées sont exploitables                            ║
║ 4. Comparer avec PaddleOCR-VL si besoin                                      ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import cv2
import json
import tempfile
from pathlib import Path

# Configuration Paddle
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

CROP_PATH = r"A:\manwha trad v2\output\debug\image_crops\crop_00_bulle_0.92.png"

if not os.path.exists(CROP_PATH):
    print(f"❌ Fichier test manquant: {CROP_PATH}")
    print("Créez d'abord des crops avec: python main.py --debug")
    sys.exit(1)

print("=" * 80)
print("🔍 DEBUG PP-OCRv5 - Test sur crop réel")
print("=" * 80)

# Charger le crop
crop = cv2.imread(CROP_PATH)
h, w = crop.shape[:2]
print(f"\n📸 Crop dimensions: {w}x{h}px")

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1: PP-OCRv5 STANDARD (pas VL)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("📝 ÉTAPE 1: PP-OCRv5 (Standard OCR)")
print("=" * 80)

print("\n⏳ Chargement PP-OCRv5...")
try:
    from paddleocr import PaddleOCR
    
    # PP-OCRv5: OCR standard (pas vision-language)
    ocr_v5 = PaddleOCR(
        lang='en',
        use_textline_orientation=True
    )
    print("✅ PP-OCRv5 chargé!")
    
except Exception as e:
    print(f"❌ Erreur PP-OCRv5: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Prédiction
print("\n🚀 Prédiction PP-OCRv5...")
results_v5 = ocr_v5.ocr(CROP_PATH, cls=True)

if not results_v5 or not results_v5[0]:
    print("❌ Aucun résultat PP-OCRv5")
    sys.exit(1)

print(f"✅ Résultat obtenu")

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2: PARSER LES RÉSULTATS PP-OCRv5
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("📊 RÉSULTATS PP-OCRv5 (Parsing)")
print("=" * 80)

texts = []
confidences = []
regions = []

print("\n▶ Extraction des détections:")

for i, line_result in enumerate(results_v5[0]):
    # PP-OCRv5 retourne: [bbox_points, (text, confidence)]
    bbox_points = line_result[0]  # 4 points [[x,y], [x,y], [x,y], [x,y]]
    text = line_result[1][0]       # Texte
    conf = float(line_result[1][1])  # Confiance

    text = text.strip()
    
    # Convertir bbox à coords
    xs = [p[0] for p in bbox_points]
    ys = [p[1] for p in bbox_points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    
    print(f"\n   [{i}] Texte: '{text}'")
    print(f"       Confiance: {conf:.3f}")
    print(f"       Bbox: x1={x1:.0f}, y1={y1:.0f}, x2={x2:.0f}, y2={y2:.0f}")
    print(f"       Points: {bbox_points}")
    
    if text and conf >= 0.0:  # Garder même confiance basse pour debug
        texts.append(text)
        confidences.append(conf)
        regions.append({
            'text': text,
            'conf': conf,
            'bbox_points': bbox_points,
            'bbox_rect': [float(x1), float(y1), float(x2), float(y2)]
        })

print(f"\n📊 Résumé PP-OCRv5:")
print(f"   Lignes détectées: {len(texts)}")
print(f"   Textes: {texts}")
print(f"   Confiances: {[f'{c:.3f}' for c in confidences]}")

if texts:
    combined = " ".join(texts)
    avg_conf = sum(confidences) / len(confidences)
    print(f"\n✅ Texte combiné: '{combined}'")
    print(f"✅ Confiance moyenne: {avg_conf:.3f}")
else:
    print("❌ Aucun texte extrait")

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3: STRATÉGIE FALLBACK (confiance basse → VL)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("🔄 ÉTAPE 3: Évaluation Fallback")
print("=" * 80)

confidence_threshold = 0.5  # Seuil pour basculer à VL

if texts:
    avg_conf_v5 = sum(confidences) / len(confidences)
    print(f"\n▶ Confiance moyenne V5: {avg_conf_v5:.3f}")
    print(f"▶ Seuil fallback: {confidence_threshold}")
    
    if avg_conf_v5 >= confidence_threshold:
        print(f"✅ ACCEPTÉ: Utiliser PP-OCRv5 (confiance {avg_conf_v5:.1%} ≥ {confidence_threshold:.1%})")
        use_v5 = True
    else:
        print(f"⚠️  FALLBACK: Essayer VL (confiance {avg_conf_v5:.1%} < {confidence_threshold:.1%})")
        use_v5 = False
else:
    print(f"⚠️  FALLBACK: Aucun résultat V5, essayer VL")
    use_v5 = False

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4: VL COMME FALLBACK
# ═══════════════════════════════════════════════════════════════════════════════

if not use_v5:
    print("\n" + "=" * 80)
    print("🔄 ÉTAPE 4: Fallback à PaddleOCR-VL v1.5")
    print("=" * 80)
    
    print("\n⏳ Chargement PaddleOCR-VL...")
    try:
        from paddleocr import PaddleOCRVL
        
        ocr_vl = PaddleOCRVL(pipeline_version="v1.5", device="gpu:0")
        print("✅ VL chargé")
        
        print("\n🚀 Prédiction VL...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_img = os.path.join(tmp_dir, "test.png")
            cv2.imwrite(tmp_img, crop)
            
            results_vl = ocr_vl.predict(tmp_img, use_ocr_for_image_block=True)
            
            if results_vl:
                result_vl = results_vl[0]
                
                # Accéder aux données VL
                if hasattr(result_vl, 'json'):
                    result_json = result_vl.json
                    result_data = result_json.get('res', {})
                    
                    parsing_list = result_data.get('parsing_res_list', [])
                    layout_boxes = result_data.get('layout_det_res', {}).get('boxes', [])
                    
                    # Map confiance
                    conf_by_bbox = {}
                    for box in layout_boxes:
                        coord = tuple(box.get('coordinate', []))
                        score = float(box.get('score', 0.95))
                        conf_by_bbox[coord] = score
                    
                    vl_texts = []
                    vl_confs = []
                    
                    for item in parsing_list:
                        text = item.get('block_content', '').strip()
                        if text:
                            bbox = item.get('block_bbox', [0, 0, 0, 0])
                            bbox_tuple = tuple(bbox)
                            conf = conf_by_bbox.get(bbox_tuple, 0.95)
                            
                            vl_texts.append(text)
                            vl_confs.append(conf)
                            
                            print(f"\n   VL: '{text}' (conf={conf:.3f})")
                    
                    if vl_texts:
                        vl_combined = " ".join(vl_texts)
                        vl_avg_conf = sum(vl_confs) / len(vl_confs)
                        print(f"\n✅ VL Texte: '{vl_combined}'")
                        print(f"✅ VL Confiance: {vl_avg_conf:.3f}")
                        
                        # Utiliser VL
                        texts = vl_texts
                        confidences = vl_confs
                    else:
                        print("❌ VL n'a rien extrait")
    
    except Exception as e:
        print(f"❌ Erreur VL: {e}")
        import traceback
        traceback.print_exc()

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 5: RÉSULTAT FINAL
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("📋 RÉSULTAT FINAL")
print("=" * 80)

print(f"\n📝 Textes: {texts}")
print(f"📊 Confiances: {[f'{c:.3f}' for c in confidences]}")

if texts:
    final_text = " ".join(texts)
    final_conf = sum(confidences) / len(confidences) if confidences else 0.0
    print(f"\n✅ TEXTE FINAL: '{final_text}'")
    print(f"✅ CONFIANCE FINALE: {final_conf:.3f}")
    
    # Sauvegarder en JSON
    output = {
        'success': True,
        'ocr_engine': 'PP-OCRv5' if use_v5 else 'PaddleOCR-VL-fallback',
        'text': final_text,
        'confidence': final_conf,
        'regions': regions,
        'raw_texts': texts,
        'raw_confidences': confidences
    }
else:
    print("\n❌ Aucun texte extrait (PP-OCRv5 ET VL ont échoué)")
    output = {
        'success': False,
        'error': 'No text extracted'
    }

# Sauvegarder résultat
result_file = Path("debug_ppocr_v5_result.json")
with open(result_file, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n💾 Résultat sauvegardé: {result_file}")

print("\n" + "=" * 80)
print("✅ DEBUG TERMINÉ")
print("=" * 80)