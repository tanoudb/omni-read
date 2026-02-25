"""
═══════════════════════════════════════════════════════════════════════════════
PaddleOCR PP-OCRv5 Backend — Subprocess-isolated, persistent worker

Le worker tourne dans .venv_paddleocr avec PaddlePaddle GPU + PP-OCRv5 server.
Communication JSON via stdin/stdout (pas de socket, pas de HTTP).

Pourquoi PP-OCRv5 et pas PaddleOCR-VL ?
  - PaddleOCR-VL = doc parser (layout + VLM) → renvoie VIDE sur des crops
  - PP-OCRv5 = OCR classique (det + rec) → marche parfaitement sur des crops
  - PP-OCRv5 server models = meilleure précision que mobile
  - Tourne sur GPU via PaddlePaddle CUDA → rapide

Le nom de classe reste PaddleOCRVLV15Backend pour compatibilité avec
le BACKEND_REGISTRY existant.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .base import OCRBackend


# ─── Configuration ───────────────────────────────────────────────────────────

_VENV_CANDIDATES = [
    Path(".venv_paddleocr/Scripts/python.exe"),   # Windows
    Path(".venv_paddleocr/bin/python"),            # Linux
    Path(".venv_paddleocr/bin/python3"),
]

_WORKER_CANDIDATES = [
    Path(__file__).parent / "_paddle_vl_worker.py",
    Path(__file__).parent.parent / "_paddle_vl_worker.py",
    Path("_paddle_vl_worker.py"),
]

WORKER_STARTUP_TIMEOUT = 180   # PP-OCRv5 server models are large
BATCH_TIMEOUT_PER_CROP = 10
BATCH_TIMEOUT_BASE = 30
MAX_CROP_DIMENSION = 1024
JPEG_QUALITY = 92


class PaddleOCRVLV15Backend(OCRBackend):
    """
    Backend OCR via PP-OCRv5 dans un worker subprocess (.venv_paddleocr).
    
    Nom de classe gardé "VLV15" pour compatibilité avec BACKEND_REGISTRY.
    En réalité utilise PP-OCRv5 server (det+rec classique).
    """

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._python_path: Optional[Path] = None
        self._worker_path: Optional[Path] = None
        self._ready = False
        self._load_count = 0
        self._total_crops_processed = 0
        self._total_ocr_seconds = 0.0

    @property
    def name(self) -> str:
        return "PaddleOCR-VL-v1.5"  # Keep for compat with existing code

    # ═══════════════════════════════════════════════════════════════════════════
    # LIFECYCLE
    # ═══════════════════════════════════════════════════════════════════════════

    def load(self, device: str) -> None:
        self._python_path = self._find_venv_python()
        if self._python_path is None:
            raise RuntimeError(
                "PaddleOCR: .venv_paddleocr introuvable. "
                "Vérifiez que .venv_paddleocr existe à la racine du projet."
            )
        self._worker_path = self._find_worker_script()
        if self._worker_path is None:
            raise RuntimeError(
                "PaddleOCR: _paddle_vl_worker.py introuvable. "
                "Placez-le dans backends/ ou à la racine du projet."
            )
        self._start_worker()

    def unload(self) -> None:
        self._shutdown_worker()

    # ═══════════════════════════════════════════════════════════════════════════
    # OCR INTERFACE
    # ═══════════════════════════════════════════════════════════════════════════

    def read_text(self, img: np.ndarray) -> Tuple[str, float, List[Dict]]:
        results = self.read_batch([img])
        return results[0] if results else ("", 0.0, [])

    def read_batch(self, images: List[np.ndarray]) -> List[Tuple[str, float, List[Dict]]]:
        if not images:
            return []

        if not self._ensure_worker():
            return [("", 0.0, []) for _ in images]

        # Encode crops
        batch_payload = []
        for idx, img in enumerate(images):
            b64 = self._encode_image(img)
            h, w = img.shape[:2]
            batch_payload.append({"id": idx, "b64": b64, "w": w, "h": h})

        timeout = BATCH_TIMEOUT_BASE + BATCH_TIMEOUT_PER_CROP * len(images)

        response = self._send_command({
            "cmd": "ocr_batch",
            "images": batch_payload,
        }, timeout=timeout)

        if response is None or response.get("status") != "ok":
            error_msg = (response or {}).get("message", "worker_error")
            print(f"⚠️ PaddleOCR batch error: {error_msg}")
            return [("", 0.0, []) for _ in images]

        raw_results = response.get("results", [])
        elapsed = response.get("elapsed", 0)
        self._total_ocr_seconds += elapsed
        self._total_crops_processed += len(images)

        result_map = {r["id"]: r for r in raw_results if isinstance(r, dict)}

        output: List[Tuple[str, float, List[Dict]]] = []
        for idx in range(len(images)):
            r = result_map.get(idx, {})
            text = str(r.get("text", "")).strip()
            conf = float(r.get("confidence", 0.0))
            regions = r.get("regions", [])

            normalized = []
            for reg in regions:
                if isinstance(reg, dict) and "bbox" in reg:
                    normalized.append({
                        "bbox": reg["bbox"],
                        "text": reg.get("text", text),
                        "conf": reg.get("conf", conf),
                    })

            output.append((text, conf, normalized))

        return output

    # ═══════════════════════════════════════════════════════════════════════════
    # SUBPROCESS MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════

    def _find_venv_python(self) -> Optional[Path]:
        search_roots = [
            Path.cwd(),
            Path(__file__).resolve().parent.parent.parent,
            Path(__file__).resolve().parent.parent,
        ]
        for root in search_roots:
            for candidate in _VENV_CANDIDATES:
                full = root / candidate
                if full.exists():
                    return full
        return None

    def _find_worker_script(self) -> Optional[Path]:
        for candidate in _WORKER_CANDIDATES:
            if candidate.is_absolute() and candidate.exists():
                return candidate
            for root in [Path.cwd(), Path(__file__).resolve().parent.parent.parent]:
                full = root / candidate
                if full.exists():
                    return full
        return None

    def _start_worker(self):
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return

            print(f"🚀 PaddleOCR-VL: démarrage worker subprocess...")
            print(f"   Python: {self._python_path}")
            print(f"   Worker: {self._worker_path}")

            try:
                self._process = subprocess.Popen(
                    [str(self._python_path), str(self._worker_path)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    cwd=str(Path.cwd()),
                    env=self._build_env(),
                )
            except Exception as exc:
                raise RuntimeError(f"PaddleOCR: échec lancement worker: {exc}")

            self._stderr_thread = threading.Thread(
                target=self._stderr_reader, daemon=True,
            )
            self._stderr_thread.start()

            ready_response = self._read_result(timeout=WORKER_STARTUP_TIMEOUT)
            if ready_response is None or ready_response.get("status") != "ready":
                self._kill_worker()
                error = (ready_response or {}).get("message", "timeout/no_response")
                raise RuntimeError(f"PaddleOCR: worker non prêt: {error}")

            self._ready = True
            self._load_count += 1
            engine = ready_response.get("engine", "unknown")
            print(f"✅ PaddleOCR-VL worker prêt (PID={self._process.pid}, engine={engine})")

    def _build_env(self) -> dict:
        import os
        env = os.environ.copy()
        env["GLOG_minloglevel"] = "2"
        env["FLAGS_allocator_strategy"] = "auto_growth"
        env["PADDLE_LOG_LEVEL"] = "error"
        env["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        return env

    def _ensure_worker(self) -> bool:
        if self._process is not None and self._process.poll() is None:
            return True
        print("⚠️ PaddleOCR worker mort, redémarrage...")
        try:
            self._start_worker()
            return True
        except Exception as exc:
            print(f"❌ PaddleOCR: échec redémarrage: {exc}")
            return False

    def _shutdown_worker(self):
        with self._lock:
            if self._process is None:
                return
            try:
                if self._process.poll() is None:
                    self._write_command({"cmd": "shutdown"})
                    try:
                        self._process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self._process.terminate()
                        try:
                            self._process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            self._process.kill()
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None
            self._ready = False

    def _kill_worker(self):
        if self._process:
            try:
                self._process.kill()
            except Exception:
                pass
            self._process = None
            self._ready = False

    # ═══════════════════════════════════════════════════════════════════════════
    # IPC
    # ═══════════════════════════════════════════════════════════════════════════

    def _write_command(self, cmd: dict):
        if self._process is None or self._process.stdin is None:
            return
        line = json.dumps(cmd, ensure_ascii=False) + "\n"
        self._process.stdin.write(line)
        self._process.stdin.flush()

    def _read_result(self, timeout: float = 60) -> Optional[dict]:
        if self._process is None or self._process.stdout is None:
            return None

        result_holder = [None]

        def _reader():
            try:
                while True:
                    line = self._process.stdout.readline()
                    if not line:
                        break
                    line = line.strip()
                    if line.startswith("RESULT:"):
                        result_holder[0] = json.loads(line[7:])
                        break
            except Exception:
                pass

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if t.is_alive():
            print(f"⚠️ PaddleOCR: timeout ({timeout}s)")
            return None

        return result_holder[0]

    def _send_command(self, cmd: dict, timeout: float = 60) -> Optional[dict]:
        with self._lock:
            if not self._ensure_worker():
                return None
            self._write_command(cmd)
            return self._read_result(timeout=timeout)

    def _stderr_reader(self):
        try:
            while self._process and self._process.stderr:
                line = self._process.stderr.readline()
                if not line:
                    break
                sys.stderr.write(f"  [VL] {line}")
                sys.stderr.flush()
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════════════════════
    # IMAGE ENCODING
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _encode_image(img: np.ndarray) -> str:
        h, w = img.shape[:2]
        if max(h, w) > MAX_CROP_DIMENSION:
            scale = MAX_CROP_DIMENSION / max(h, w)
            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            ok, buf = cv2.imencode(".png", img)
            if not ok:
                return ""
        return base64.b64encode(buf.tobytes()).decode("ascii")

    # ═══════════════════════════════════════════════════════════════════════════
    # DIAGNOSTICS
    # ═══════════════════════════════════════════════════════════════════════════

    def get_runtime_info(self) -> Dict[str, Any]:
        alive = self._process is not None and self._process.poll() is None
        return {
            "backend": "PaddleOCR-VL-v1.5",
            "isolation": "subprocess",
            "worker_alive": alive,
            "worker_pid": self._process.pid if self._process and alive else None,
            "python_path": str(self._python_path) if self._python_path else None,
            "total_crops_processed": self._total_crops_processed,
            "total_ocr_seconds": round(self._total_ocr_seconds, 2),
            "avg_seconds_per_crop": round(
                self._total_ocr_seconds / max(1, self._total_crops_processed), 3
            ),
            "load_count": self._load_count,
        }

    def __del__(self):
        try:
            self._shutdown_worker()
        except Exception:
            pass