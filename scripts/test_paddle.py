import os
import json
import paddle
from paddleocr import PaddleOCRVL
import time

# Désactive la vérification réseau inutile pour gagner quelques secondes
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# 1. Initialisation optimisée
# use_queues=True permet de charger les images en parallèle de l'analyse
# Deux pipelines : un rapide (pas d'analyse lourde des blocs d'image) et
# un lourd (fallback) activé seulement si nécessaire.
pipeline_fast = PaddleOCRVL(
    device="gpu:0",
    use_ocr_for_image_block=False,
    use_queues=True,
)
# pipeline_heavy will be created lazily on first need to avoid OOM at startup
pipeline_heavy = None

def _create_heavy_pipeline(prefer_gpu=True):
    global pipeline_heavy
    if pipeline_heavy is not None:
        return pipeline_heavy
    # Try GPU first, fall back to CPU on MemoryError
    try:
        if prefer_gpu:
            print("Creating heavy pipeline on GPU...")
            pipeline_heavy = PaddleOCRVL(
                device="gpu:0",
                use_ocr_for_image_block=True,
                use_queues=True,
            )
        else:
            raise MemoryError("force CPU")
    except MemoryError:
        try:
            print("GPU OOM or unavailable — creating heavy pipeline on CPU...")
            pipeline_heavy = PaddleOCRVL(
                device="cpu",
                use_ocr_for_image_block=True,
                use_queues=False,
            )
        except Exception as e:
            print(f"Impossible de créer le pipeline lourd: {e}")
            pipeline_heavy = None
    except Exception as e:
        print(f"Erreur lors de la création du pipeline lourd: {e}")
        pipeline_heavy = None
    return pipeline_heavy

# 2. Dossiers (Vérifiez bien ces chemins !)
input_folder = r"A:\omni read\output\debug\Chapitre 001_merged_part01_crops"
# On définit bien output_file ici
output_file = r"A:\omni read\test\resultats_complets.json"
# Fichier texte formatté lisible (taille / texte / confiance)
output_text_file = r"A:\omni read\test\resultats_complets.txt"

# 3. Liste des fichiers
files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) 
         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf'))]

if not files:
    print(f"Erreur : Aucun fichier trouvé dans {input_folder}")
else:
    tous_les_resultats = []
    print(f"--- Début de l'analyse de {len(files)} fichiers ---")
    print("Note : utilisation d'un pipeline rapide + fallback intelligent si nécessaire.")

    # Utilisation d'un traitement par lots pour limiter l'empreinte mémoire
    BATCH = 8
    formatted_lines = []

    def _extract_items(data):
        # Normalise différentes structures de sortie possibles
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for k in ("items", "results", "predictions", "bubbles", "lines"):
                if k in data and isinstance(data[k], (list, tuple)):
                    return data[k]
            # fallback: try common flat keys
            if any(k in data for k in ("bbox", "text", "confidence", "conf", "score")):
                return [data]
        return []

    def _get_text_conf_bbox(item):
        # Text
        text = None
        for k in ("text", "ocr_text", "transcription", "pred", "sentence"):
            if k in item:
                text = item[k]
                break
        # Confidence
        conf = None
        for k in ("conf", "confidence", "score", "ocr_conf"):
            if k in item:
                try:
                    conf = float(item[k])
                except Exception:
                    conf = None
                break
        # BBox
        bbox = None
        for k in ("bbox", "box", "boxes"):
            if k in item:
                bbox = item[k]
                break
        # Some outputs store bbox as dict
        if isinstance(bbox, dict) and all(x in bbox for x in ("x1", "y1", "x2", "y2")):
            bbox = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
        return text, conf, bbox

    def _format_item(text, conf, bbox):
        # compute size if bbox is available
        if bbox and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            try:
                w = int(abs(bbox[2] - bbox[0]))
                h = int(abs(bbox[3] - bbox[1]))
                size = f"size: {w}x{h}px"
            except Exception:
                size = "size: (unknown)"
        else:
            size = "size: (unknown)"
        txt = text if text is not None else "(none)"
        conf_str = f"{conf:.2f}" if conf is not None else "0.00"
        return f"{size}\n    OCR text: \"{txt}\"\n    OCR conf: {conf_str}"

    # 4. Prédiction par petits lots (fast pass), puis fallback par image si nécessaire
    total_start = time.perf_counter()
    for start in range(0, len(files), BATCH):
        batch_files = files[start:start+BATCH]

        # fast batch
        t0 = time.perf_counter()
        results_fast = pipeline_fast.predict(batch_files)
        t_batch = time.perf_counter() - t0

        # approx per-image time for fast pass
        per_image_fast = t_batch / max(1, len(batch_files))

        for i, res in enumerate(results_fast):
            img_path = batch_files[i]
            # basic data extraction
            data_fast = getattr(res, 'json', None) or res
            todos_fast = _extract_items(data_fast)

            # compute maximum confidence seen in fast results
            max_conf = None
            if todos_fast:
                confs = []
                for item in todos_fast:
                    _, conf, _ = _get_text_conf_bbox(item)
                    if conf is not None:
                        confs.append(conf)
                if confs:
                    max_conf = max(confs)

            # Decide si fallback nécessaire: pas d'items ou confiance faible
            NEED_FALLBACK = (not todos_fast) or (max_conf is None) or (max_conf < 0.45)

            total_time_for_image = per_image_fast
            used_fallback = False
            data_final = data_fast

            if NEED_FALLBACK:
                used_fallback = True
                t1 = time.perf_counter()
                # fallback sur l'image seule (coûteux)
                try:
                    res_heavy = pipeline_heavy.predict([img_path])[0]
                    data_heavy = getattr(res_heavy, 'json', None) or res_heavy
                except Exception:
                    data_heavy = None
                t_fallback = time.perf_counter() - t1
                total_time_for_image += t_fallback
                # prefer heavy data when available
                if data_heavy:
                    data_final = data_heavy

            # Normalise le format et écrit l'entrée formatée
            todos = _extract_items(data_final)
            if not todos and isinstance(data_final, dict):
                todos = [data_final]

            if not todos:
                formatted_lines.append(_format_item(None, None, None) + f"\n    time: {total_time_for_image:.3f}s")
            else:
                for item in todos:
                    text, conf, bbox = _get_text_conf_bbox(item)
                    formatted_lines.append(_format_item(text, conf, bbox) + f"\n    time: {total_time_for_image:.3f}s" + (" (fallback)" if used_fallback else ""))

            tous_les_resultats.append(data_final)
            nom_img = os.path.basename(img_path)
            print(f"[{start + i + 1}/{len(files)}] Terminé : {nom_img} - time: {total_time_for_image:.3f}s{' (fallback)' if used_fallback else ''}")

    total_elapsed = time.perf_counter() - total_start
    formatted_lines.insert(0, f"TOTAL_TIME: {total_elapsed:.3f}s")

    # 5. Sauvegarde finale
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(tous_les_resultats, f, ensure_ascii=False, indent=4)

    # 6. Écriture du fichier texte formaté (lisible)
    with open(output_text_file, 'w', encoding='utf-8') as f:
        for line in formatted_lines:
            f.write(line + "\n\n")

    print(f"\n--- ANALYSE TERMINÉE ---")
    print(f"Fichier JSON généré : {output_file}")
    print(f"Fichier texte généré : {output_text_file}")