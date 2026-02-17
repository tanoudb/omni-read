from __future__ import annotations

import csv
import gc
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import TranslationPipeline
from utils import MemoryManager
from config import config
from core import OCREngine, NLLBTranslator


app = FastAPI(title="Webtoon Translator Lazy API", version="1.28.2")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProcessRequest(BaseModel):
    input_path: str = Field(..., description="Path image input")
    output_dir: str = Field(..., description="Path dossier output")
    debug: bool = Field(default=False)


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    logs: List[dict]
    next_offset: int
    result: dict | None = None
    error: str | None = None


@dataclass
class JobState:
    status: str = "queued"
    logs: List[dict] = field(default_factory=list)
    result: dict | None = None
    error: str | None = None


class ApiJobLogger:
    def __init__(self, push_line):
        self.push_line = push_line
        self.stats = {
            "start_time": None,
            "phase_times": {},
            "current_phase": None,
        }

    def debug(self, msg: str):
        self.push_line("DEBUG", msg)

    def info(self, msg: str):
        self.push_line("INFO", msg)

    def warning(self, msg: str):
        self.push_line("WARNING", msg)

    def error(self, msg: str):
        self.push_line("ERROR", msg)

    def critical(self, msg: str):
        self.push_line("CRITICAL", msg)

    def header(self, text: str, width: int = 80):
        line = "═" * width
        self.info(f"\n{line}")
        self.info(text.center(width))
        self.info(f"{line}\n")

    def section(self, text: str, width: int = 80):
        line = "─" * width
        self.info(f"\n{line}")
        self.info(f"  {text}")
        self.info(line)

    def phase(self, name: str, number: int, total: int):
        self.info(f"\n🎯 PHASE {number}/{total} : {name.upper()}")

    def end_phase(self):
        pass

    def progress(self, current: int, total: int, prefix: str = ""):
        self.info(f"{prefix}{current}/{total}")

    def stat(self, key: str, value):
        self.info(f"   {key}: {value}")

    def start_timer(self):
        self.stats["start_time"] = datetime.utcnow().timestamp()

    def end_timer(self):
        if self.stats["start_time"]:
            elapsed = datetime.utcnow().timestamp() - self.stats["start_time"]
            self.info(f"⏱️  TEMPS TOTAL: {elapsed:.2f}s")

    def summary(self, data: dict):
        self.section("📊 RÉSUMÉ")
        for key, value in data.items():
            self.info(f"   {key}: {value}")


jobs: Dict[str, JobState] = {}
jobs_lock = threading.Lock()
benchmark_lock = threading.Lock()
BENCHMARK_CSV = PROJECT_ROOT / "benchmark_results.csv"
WARM_OCR_ENGINE: OCREngine | None = None
WARM_TRANSLATOR: NLLBTranslator | None = None


def _cuda_vram_snapshot() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            allocated = float(torch.cuda.memory_allocated() / (1024 ** 3))
            reserved = float(torch.cuda.memory_reserved() / (1024 ** 3))
            try:
                peak = float(torch.cuda.max_memory_allocated() / (1024 ** 3))
            except Exception:
                peak = allocated
            return {
                "cuda": True,
                "allocated_gb": round(allocated, 3),
                "reserved_gb": round(reserved, 3),
                "peak_allocated_gb": round(peak, 3),
            }
    except Exception:
        pass
    return {"cuda": False}


def _warm_models():
    global WARM_OCR_ENGINE

    device = "cuda"
    try:
        import torch
        if not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"

    if WARM_OCR_ENGINE is None:
        WARM_OCR_ENGINE = OCREngine(device=device)

    # LLM warm-up is intentionally deferred per job to avoid startup crash
    # on constrained VRAM systems.


def _strict_cuda_cleanup():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _append_benchmark_result(result: dict, fallback_image: Path):
    if not isinstance(result, dict):
        return

    timings = result.get("timings") or {}

    row = {
        "image": str(result.get("image") or fallback_image.name),
        "yolo_seconds": float(timings.get("yolo_seconds", 0.0) or 0.0),
        "sam2_seconds": float(timings.get("sam2_seconds", 0.0) or 0.0),
        "ocr_seconds": float(timings.get("ocr_seconds", 0.0) or 0.0),
        "llm_seconds": float(timings.get("llm_seconds", 0.0) or 0.0),
        "total_seconds": float(result.get("time_seconds", 0.0) or 0.0),
    }

    fieldnames = [
        "image",
        "yolo_seconds",
        "sam2_seconds",
        "ocr_seconds",
        "llm_seconds",
        "total_seconds",
    ]

    with benchmark_lock:
        BENCHMARK_CSV.parent.mkdir(parents=True, exist_ok=True)
        write_header = not BENCHMARK_CSV.exists() or BENCHMARK_CSV.stat().st_size == 0
        with BENCHMARK_CSV.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def _push_log(job_id: str, level: str, message: str):
    entry = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "level": level,
        "message": message,
    }
    with jobs_lock:
        job = jobs.get(job_id)
        if job is not None:
            job.logs.append(entry)


def _run_job(job_id: str, req: ProcessRequest):
    global WARM_TRANSLATOR

    with jobs_lock:
        jobs[job_id].status = "running"

    logger = ApiJobLogger(lambda level, message: _push_log(job_id, level, message))

    try:
        _warm_models()

        input_path = Path(req.input_path)
        output_dir = Path(req.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not input_path.exists():
            raise FileNotFoundError(f"Input introuvable: {input_path}")

        pipeline = TranslationPipeline(
            logger=logger,
            debug=req.debug,
            lazy_models=True,
            strict_memory_cleanup=False,
            shared_ocr_engine=WARM_OCR_ENGINE,
            shared_translator=WARM_TRANSLATOR,
        )

        if WARM_TRANSLATOR is None and str(getattr(config.translation, 'backend', 'nllb')).lower() == 'local_llm':
            before = _cuda_vram_snapshot()
            _push_log(job_id, "INFO", f"VRAM avant chargement LLM: {before}")
            try:
                WARM_TRANSLATOR = NLLBTranslator(device=pipeline.device)
            except Exception as exc:
                _push_log(job_id, "WARNING", f"Warm LLM échoué, fallback lazy: {exc}")
                WARM_TRANSLATOR = None
            after = _cuda_vram_snapshot()
            _push_log(job_id, "INFO", f"VRAM après chargement LLM: {after}")
            pipeline.shared_translator = WARM_TRANSLATOR

        try:
            import torch
            if torch.cuda.is_available():
                pipeline.device = 'cuda'
                _push_log(job_id, "INFO", "GPU détecté: YOLO/SAM2 forcés sur device=cuda")
            else:
                _push_log(job_id, "WARNING", "GPU non détecté: fallback cpu")
        except Exception as exc:
            _push_log(job_id, "WARNING", f"Détection GPU échouée: {exc}")

        result = pipeline.process_image(input_path, output_dir)
        _append_benchmark_result(result, input_path)

        _strict_cuda_cleanup()
        MemoryManager.cleanup_aggressive()

        with jobs_lock:
            jobs[job_id].status = "done"
            jobs[job_id].result = result

    except Exception as exc:
        _push_log(job_id, "ERROR", f"Erreur: {exc}")
        _push_log(job_id, "ERROR", traceback.format_exc())
        with jobs_lock:
            jobs[job_id].status = "failed"
            jobs[job_id].error = str(exc)
    finally:
        _strict_cuda_cleanup()
        MemoryManager.cleanup_aggressive()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "lazy_loading": True,
        "strict_memory_cleanup": False,
        "warm_ocr": WARM_OCR_ENGINE is not None,
        "warm_llm": WARM_TRANSLATOR is not None,
    }


@app.on_event("startup")
def on_startup():
    try:
        _warm_models()
    except Exception:
        pass


@app.post("/jobs")
def create_job(req: ProcessRequest):
    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = JobState(status="queued")

    thread = threading.Thread(target=_run_job, args=(job_id, req), daemon=True)
    thread.start()

    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str, offset: int = 0):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job introuvable")

        logs = job.logs[offset:]
        return JobStatusResponse(
            job_id=job_id,
            status=job.status,
            logs=logs,
            next_offset=len(job.logs),
            result=job.result,
            error=job.error,
        )
