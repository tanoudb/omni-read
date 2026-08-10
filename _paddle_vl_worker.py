#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
PaddleOCR PP-OCRv5 Server — Worker subprocess persistant

Tourne dans .venv_paddleocr. Charge PP-OCRv5 (server models) UNE FOIS,
reçoit des crops en base64 via stdin, renvoie le texte OCR via stdout.

IMPORTANT: On utilise PaddleOCR (PP-OCRv5 classique), PAS PaddleOCRVL.
PaddleOCRVL est un doc parser (layout + VLM) qui ne marche PAS sur des
petits crops de bulles manga. PP-OCRv5 est un vrai OCR det+rec qui marche
parfaitement sur des crops individuels.

Protocole stdin/stdout :
  → {"cmd":"ping"}
  → {"cmd":"ocr_batch", "images":[{"id":0,"b64":"..."}]}
  → {"cmd":"shutdown"}
  ← RESULT:{...}
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import json
import base64
import time
import traceback
import os
import gc
import tempfile

import numpy as np

# ─── Suppress noise ─────────────────────────────────────────────────────────
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
os.environ.setdefault("PADDLE_LOG_LEVEL", "error")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# ─── Globals ─────────────────────────────────────────────────────────────────
_ocr = None
_tmp_dir = tempfile.mkdtemp(prefix="pocr_")


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_model() -> bool:
    """Load PaddleOCR PP-OCRv5 (server models) — classic det+rec, NOT VL."""
    global _ocr

    t0 = time.time()
    sys.stderr.write("[paddle_worker] Loading PaddleOCR PP-OCRv5 server...\n")
    sys.stderr.flush()

    try:
        from paddleocr import PaddleOCR

        # PP-OCRv5 server models — meilleure précision que mobile
        # Pas besoin de orientation/unwarping sur des crops de bulles
        _ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_server_det",
            text_recognition_model_name="PP-OCRv5_server_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="gpu:0",
        )

        elapsed = time.time() - t0
        sys.stderr.write(f"[paddle_worker] PP-OCRv5 loaded in {elapsed:.1f}s\n")
        sys.stderr.flush()
        return True

    except Exception as exc:
        sys.stderr.write(f"[paddle_worker] PP-OCRv5 server failed: {exc}\n")
        sys.stderr.flush()

        # Fallback: essayer sans spécifier les modèles (default)
        try:
            sys.stderr.write("[paddle_worker] Trying default PaddleOCR()...\n")
            sys.stderr.flush()
            from paddleocr import PaddleOCR
            _ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            elapsed = time.time() - t0
            sys.stderr.write(f"[paddle_worker] Default PaddleOCR loaded in {elapsed:.1f}s\n")
            sys.stderr.flush()
            return True
        except Exception as exc2:
            sys.stderr.write(f"[paddle_worker] All load strategies failed: {exc2}\n")
            sys.stderr.write(traceback.format_exc())
            sys.stderr.flush()
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def decode_b64(b64_str: str):
    import cv2
    raw = base64.b64decode(b64_str)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def save_temp(img, idx: int) -> str:
    import cv2
    path = os.path.join(_tmp_dir, f"crop_{idx}.png")
    cv2.imwrite(path, img)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# OCR SINGLE CROP
# ═══════════════════════════════════════════════════════════════════════════════

def ocr_crop(img, crop_id: int) -> dict:
    """
    OCR un seul crop avec PaddleOCR PP-OCRv5.

    Retourne: {"id", "text", "confidence", "regions": [{"bbox", "text", "conf"}]}
    """
    result = {"id": crop_id, "text": "", "confidence": 0.0, "regions": []}

    try:
        # PaddleOCR.predict() accepte un path ou un ndarray
        # Sur un crop, il fait det → rec directement
        tmp_path = save_temp(img, crop_id)
        output = _ocr.predict(tmp_path)

        texts = []
        scores = []
        regions = []

        for res in output:
            # res est un objet Result — accéder aux données
            rd = None

            # Méthode 1: attribut .json
            if hasattr(res, 'json'):
                rd = res.json
            elif isinstance(res, dict):
                rd = res
            else:
                try:
                    rd = dict(res)
                except Exception:
                    pass

            if rd is None:
                continue

            # Unwrap 'res' key si présent
            if isinstance(rd, dict) and 'res' in rd:
                rd = rd['res']

            if not isinstance(rd, dict):
                continue

            # PaddleOCR 3.x: rec_texts, rec_scores, dt_polys/rec_polys
            rec_texts = rd.get('rec_texts', [])
            rec_scores = rd.get('rec_scores', [])
            polys = rd.get('rec_polys', rd.get('dt_polys', None))

            if rec_texts:
                for i, t in enumerate(rec_texts):
                    t = str(t).strip()
                    if not t:
                        continue
                    texts.append(t)
                    sc = float(rec_scores[i]) if i < len(rec_scores) else 0.85
                    scores.append(sc)

                    if polys is not None and i < len(polys):
                        try:
                            pts = np.array(polys[i], dtype=np.int32).tolist()
                            regions.append({"bbox": pts, "text": t, "conf": sc})
                        except Exception:
                            pass
            else:
                # Single result format
                t = rd.get('rec_text', rd.get('text', ''))
                if t:
                    t = str(t).strip()
                    texts.append(t)
                    sc = float(rd.get('rec_score', rd.get('score', 0.85)))
                    scores.append(sc)

        result["text"] = " ".join(t for t in texts if t)
        result["confidence"] = sum(scores) / max(1, len(scores)) if scores else 0.0
        result["regions"] = regions

        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    except Exception as exc:
        result["error"] = str(exc)
        shape = getattr(img, "shape", "?")
        sys.stderr.write(f"[paddle_worker] OCR error crop {crop_id} (shape={shape}): {exc}\n")
        sys.stderr.flush()

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# GPU HOUSEKEEPING
# ═══════════════════════════════════════════════════════════════════════════════

def _try_free_gpu_memory():
    """
    Libère le cache mémoire GPU de Paddle entre les lots. Ce worker tourne en
    persistant sur des centaines/milliers de crops d'affilée sur une longue
    session — sans ce nettoyage périodique, la fragmentation mémoire peut
    s'accumuler et finir par déclencher des erreurs d'allocation étranges
    en cours de route (observé après une longue session de tests).
    """
    try:
        import paddle
        paddle.device.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

_batches_since_cleanup = 0
_GPU_CLEANUP_EVERY_N_BATCHES = 5


def handle_batch(images_data: list) -> list:
    global _batches_since_cleanup
    results = []

    for item in images_data:
        cid = item.get("id", 0)

        img = None
        if item.get("b64"):
            try:
                img = decode_b64(item["b64"])
            except Exception:
                pass
        elif item.get("path"):
            import cv2
            img = cv2.imread(item["path"])

        if img is None or img.size == 0:
            results.append({"id": cid, "text": "", "confidence": 0.0, "regions": [],
                            "error": "decode_failed"})
            continue

        r = ocr_crop(img, cid)
        results.append(r)

    _batches_since_cleanup += 1
    if _batches_since_cleanup >= _GPU_CLEANUP_EVERY_N_BATCHES:
        _batches_since_cleanup = 0
        _try_free_gpu_memory()
    else:
        gc.collect()
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# EVENT LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def send(data: dict):
    out = "RESULT:" + json.dumps(data, ensure_ascii=False) + "\n"
    try:
        # Write as UTF-8 bytes to avoid Windows cp1252 encoding errors
        sys.stdout.buffer.write(out.encode("utf-8"))
        sys.stdout.buffer.flush()
    except Exception:
        # Fallback to text write if buffer unavailable
        sys.stdout.write(out)
        sys.stdout.flush()


def main():
    t0 = time.time()
    ok = load_model()
    lt = time.time() - t0

    if not ok:
        send({"status": "error", "message": "load_failed"})
        sys.exit(1)

    send({"status": "ready", "engine": "PP-OCRv5", "load_time": round(lt, 1)})

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue

        try:
            cmd = json.loads(line)
        except json.JSONDecodeError as e:
            send({"status": "error", "message": f"json_error: {e}"})
            continue

        act = cmd.get("cmd", "")

        if act == "ping":
            send({"status": "pong", "engine": "PP-OCRv5"})

        elif act == "ocr_batch":
            imgs = cmd.get("images", [])
            t0 = time.time()
            try:
                res = handle_batch(imgs)
                send({
                    "status": "ok",
                    "results": res,
                    "elapsed": round(time.time() - t0, 3),
                    "count": len(res),
                })
            except Exception as exc:
                # Filet de sécurité : une erreur pas rattrapée par ocr_crop()
                # (ex: gc.collect() qui pète sur un état Paddle corrompu) ne
                # doit pas faire mourir tout le worker en pleine série —
                # le process appelant redémarrera de toute façon en le
                # voyant répondre "error", mais autant rester vivant si on
                # le peut encore.
                sys.stderr.write(f"[paddle_worker] batch fatal error: {exc}\n")
                sys.stderr.write(traceback.format_exc())
                sys.stderr.flush()
                send({"status": "error", "message": f"batch_error: {exc}"})
                _try_free_gpu_memory()

        elif act == "shutdown":
            send({"status": "shutdown"})
            break

        else:
            send({"status": "error", "message": f"unknown: {act}"})

    import shutil
    try:
        shutil.rmtree(_tmp_dir, ignore_errors=True)
    except Exception:
        pass

    sys.stderr.write("[paddle_worker] Exiting.\n")
    sys.stderr.flush()


if __name__ == "__main__":
    main()