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
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .base import OCRBackend


# ─── Configuration ───────────────────────────────────────────────────────────

# `PADDLE_VENV` permet de pointer un AUTRE environnement que celui par défaut,
# pour comparer des versions de PaddleOCR sur le banc `scratch/ocr_bench.py`
# sans toucher au venv qui fait tourner la production.
import os as _os

_VENV_NAME = _os.environ.get("PADDLE_VENV", "").strip() or (
    ".venv_paddle_next" if Path(".venv_paddle_next").exists() else ".venv_paddleocr"
)

_VENV_CANDIDATES = [
    Path(f"{_VENV_NAME}/Scripts/python.exe"),   # Windows
    Path(f"{_VENV_NAME}/bin/python"),            # Linux
    Path(f"{_VENV_NAME}/bin/python3"),
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
        # RLock et non Lock : `_send_command` prend le verrou puis appelle
        # `_ensure_worker()`, qui appelle `_start_worker()` — lequel reprend ce
        # même verrou. Avec un Lock non réentrant, ce chemin se bloquait
        # DÉFINITIVEMENT (le thread s'attendait lui-même), et il n'est
        # emprunté que lorsque le worker est déjà mort au moment de l'envoi :
        # d'où un pipeline qui tournait normalement puis se figeait pour
        # toujours au deuxième crash consécutif du worker (le premier
        # redémarrage, lui, passe par `read_batch` HORS verrou et aboutit).
        self._lock = threading.RLock()
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

        # Transport par FICHIER : seul le CHEMIN passe par le pipe.
        #
        # Faire transiter les crops (base64, plusieurs Mo) dans la commande et
        # les résultats dans la réponse remplissait le tampon de pipe de l'OS
        # (~64 Ko) dans les deux sens. Interblocage observé et confirmé par
        # pile d'exécution : client bloqué dans `_write_command`, worker bloqué
        # dans `send()` — chacun attendant que l'autre draine son tuyau, alors
        # qu'aucun des deux ne peut plus lire. Avec un fichier, la ligne
        # échangée fait quelques centaines d'octets et ne peut plus saturer
        # quoi que ce soit.
        cmd, tmp_paths = self._build_batch_command(batch_payload)
        try:
            response = self._send_command(cmd, timeout=timeout)

            if response is None or response.get("status") != "ok":
                error_msg = (response or {}).get("message", "worker_error")
                print(f"⚠️ PaddleOCR batch error: {error_msg} — relance du worker + 1 nouvelle tentative")
                # Le process a pu crasher EN COURS de traitement du lot (au lieu
                # d'être déjà mort quand on l'a vérifié) : poll()/pipe cassé ne
                # sont pas toujours détectés à temps par _ensure_worker(). On le
                # force à mourir proprement, on en relance un neuf, et on retente
                # UNE fois plutôt que de rendre tout le lot vide définitivement.
                self._kill_worker()
                if self._ensure_worker():
                    response = self._send_command(cmd, timeout=timeout)

                if response is None or response.get("status") != "ok":
                    error_msg = (response or {}).get("message", "worker_error")
                    print(f"❌ PaddleOCR: échec du lot même après relance ({error_msg})")
                    return [("", 0.0, []) for _ in images]
                print("✅ PaddleOCR: lot retraité avec succès après relance")

            raw_results = self._extract_results(response)
        finally:
            for p in tmp_paths:
                try:
                    Path(p).unlink()
                except Exception:
                    pass

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

    def _build_batch_command(self, batch_payload: List[Dict]) -> Tuple[dict, List[str]]:
        """
        Écrit le lot dans un fichier temporaire et renvoie (commande, fichiers
        à nettoyer). La commande ne contient que des CHEMINS, jamais les
        données — c'est ce qui garantit que le pipe ne peut plus saturer.

        Si l'écriture du fichier échoue (disque plein, droits), on retombe sur
        l'ancien transport en ligne : dégradé mais fonctionnel, plutôt que de
        perdre le lot.
        """
        try:
            tmp_dir = Path(tempfile.gettempdir()) / "omniread_ocr_ipc"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            token = uuid.uuid4().hex
            payload_path = tmp_dir / f"batch_{token}.json"
            result_path = tmp_dir / f"result_{token}.json"
            with open(payload_path, "w", encoding="utf-8") as fh:
                json.dump(batch_payload, fh, ensure_ascii=False)
            cmd = {
                "cmd": "ocr_batch",
                "payload_path": str(payload_path),
                "result_path": str(result_path),
            }
            return cmd, [str(payload_path), str(result_path)]
        except Exception as exc:
            print(f"⚠️ PaddleOCR: transport fichier indisponible ({exc}), repli sur transport en ligne")
            return {"cmd": "ocr_batch", "images": batch_payload}, []

    @staticmethod
    def _extract_results(response: dict) -> List[Dict]:
        """Résultats depuis le fichier pointé par la réponse, ou en ligne."""
        result_path = response.get("result_path")
        if result_path:
            try:
                with open(result_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception as exc:
                print(f"⚠️ PaddleOCR: lecture résultat impossible ({exc})")
                return []
        return response.get("results", [])

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
        """
        Termine le worker et ATTEND sa mort confirmée avant de rendre la main.

        `.kill()` seul ne fait qu'ENVOYER le signal — il ne garantit pas que
        le process (et surtout le contexte CUDA qu'il tenait) soit vraiment
        libéré au moment où l'appelant relance un nouveau worker juste après.
        Mesuré : un blocage total du pipeline (15 min sans la moindre ligne de
        log, worker PID à 0% CPU / mémoire quasi nulle — bloqué avant même
        d'avoir chargé le moindre modèle) après un DEUXIÈME crash consécutif.
        Hypothèse la plus probable : le nouveau worker attendait un contexte
        GPU encore accaparé par l'ancien process, tué mais pas encore
        vraiment éteint. `wait()` avec un délai borné, et fermeture explicite
        des pipes pour ne pas laisser un thread lecteur bloqué indéfiniment
        dessus, réduisent ce risque sans le garantir à 100%.
        """
        if self._process:
            proc = self._process
            pid = getattr(proc, 'pid', None)
            self._process = None
            self._ready = False
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=10)
            except Exception:
                pass
            # `proc.kill()` ne suffit pas toujours : un worker bloqué en
            # écriture sur un pipe plein a été observé SURVIVANT à kill(),
            # restant en vie à retenir sa VRAM (plusieurs Go sur une carte qui
            # n'en a que 6) pendant que le pipeline en démarrait un nouveau.
            # `taskkill /F /T` tue l'arbre de process pour de bon.
            if pid is not None and proc.poll() is None:
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(pid)],
                        capture_output=True, timeout=15,
                    )
                except Exception:
                    pass
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass
            # `proc.wait()` confirme que Windows a réclamé le process, mais le
            # driver GPU (WDDM) peut relâcher la VRAM associée de façon
            # asynchrone, avec un léger retard après ce point. Sur une carte à
            # 6 Go partagée entre YOLO + RapidOCR (process principal) et ce
            # worker, un redémarrage immédiat peut retenter d'allouer avant que
            # cette VRAM soit vraiment libre — le nouveau worker reste alors
            # bloqué en initialisation CUDA sans jamais répondre. Petite marge
            # avant de rendre la main à l'appelant (qui relance juste après).
            time.sleep(2.0)

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

        # Référence CAPTURÉE, pas relue dynamiquement via `self._process` dans
        # le thread : sur un timeout, ce thread reste en vie (daemon, jamais
        # interrompu — un `readline()` bloquant ne se laisse pas annuler) et
        # continue sa boucle. S'il relisait `self._process.stdout` à chaque
        # itération, il basculerait sur le worker SUIVANT dès qu'un redémarrage
        # réassigne `self._process` — et pourrait alors intercepter la vraie
        # réponse du nouveau worker, que personne ne lit plus jamais côté
        # appelant (déjà reparti sur un timeout). Repéré comme cause probable
        # d'un blocage total de 15 min sans aucune ligne de log après un
        # deuxième crash consécutif du worker.
        stdout = self._process.stdout
        result_holder = [None]

        def _reader():
            try:
                while True:
                    line = stdout.readline()
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
        # Référence capturée à l'entrée du thread, pas relue dynamiquement via
        # `self._process` (même raison que `_read_result` : ce process peut
        # être remplacé par un redémarrage pendant que ce thread tourne
        # encore, et il se mettrait alors à lire le stderr du MAUVAIS worker).
        proc = self._process
        if proc is None or proc.stderr is None:
            return
        try:
            while True:
                line = proc.stderr.readline()
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