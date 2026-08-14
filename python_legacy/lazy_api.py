from __future__ import annotations

import csv
import gc
import json
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


def _extract_bubbles_and_output_from_metadata(input_path: Path, output_dir: Path) -> tuple[list[dict], str | None]:
    metadata_path = output_dir / f"{input_path.stem}_metadata.json"
    if not metadata_path.exists():
        return [], None

    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except Exception:
        return [], None

    raw_detections = metadata.get("detections") if isinstance(metadata, dict) else None
    output_path = metadata.get("output") if isinstance(metadata, dict) else None

    if not isinstance(raw_detections, list):
        return [], str(output_path) if output_path else None

    bubbles: list[dict] = []
    for detection in raw_detections:
        if not isinstance(detection, dict):
            continue

        raw_bbox = detection.get("bbox")
        if not (isinstance(raw_bbox, list) and len(raw_bbox) >= 4):
            continue

        try:
            bbox = [
                float(raw_bbox[0]),
                float(raw_bbox[1]),
                float(raw_bbox[2]),
                float(raw_bbox[3]),
            ]
        except Exception:
            continue

        bubble = {
            "id": str(uuid.uuid4()),
            "bbox": bbox,
            "class": str(detection.get("class") or "bulle"),
            "source_text": str(detection.get("original") or ""),
            "translated_text": str(detection.get("translated") or ""),
            "detection_confidence": (
                float(detection.get("detection_confidence"))
                if detection.get("detection_confidence") is not None
                else None
            ),
            "ocr_confidence": (
                float(detection.get("confidence"))
                if detection.get("confidence") is not None
                else None
            ),
        }
        bubbles.append(bubble)

    return bubbles, str(output_path) if output_path else None


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
        bubbles, output_path = _extract_bubbles_and_output_from_metadata(input_path, output_dir)
        if isinstance(result, dict):
            result["bubbles"] = bubbles
            if output_path:
                result["output"] = output_path
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

# --- Nouveaux endpoints pour UI étape par étape ---

class DetectRequest(BaseModel):
    image_path: str
    classes: List[str] = []
    debug: bool = False

class OcrRequest(BaseModel):
    image_path: str
    bubbles: List[dict]

class TranslateRequest(BaseModel):
    bubbles: List[dict]
    cache_enabled: bool = True
    return_llm_debug: bool = False
    glossary: dict = {}

class RenderRequest(BaseModel):
    image_path: str
    bubbles: List[dict]
    text_only: bool = False
    skip_inpainting: bool = False

class CacheRequest(BaseModel):
    enabled: bool

@app.post("/cache")
def cache_toggle(req: CacheRequest):
    # This is a stub for the cache toggle. In a real scenario, this would
    # toggle a global or session-level cache setting in the backend.
    # For now, it just echoes back the state to satisfy the UI MVP contract.
    return {"enabled": req.enabled}

@app.post("/detect")
def detect_api(req: DetectRequest):
    _warm_models()
    pipeline = TranslationPipeline(logger=ApiJobLogger(lambda l, m: None), lazy_models=True)
    if not pipeline._ensure_detector():
        raise HTTPException(500, "Detector init failed")
    
    import cv2
    import uuid
    img = cv2.imread(req.image_path)
    if img is None:
        raise HTTPException(404, "Image not found")
        
    dets = pipeline._detect_ensemble(img)
    bubbles = []
    h, w = img.shape[:2]
    for d in dets:
        bubbles.append({
            "id": str(uuid.uuid4()),
            "bbox": {"x": float(d.x1), "y": float(d.y1), "w": float(d.x2 - d.x1), "h": float(d.y2 - d.y1)},
            "class": getattr(d, 'class_name', 'bulle'),
            "source_text": "",
            "translated_text": "",
            "llm_input_index": None,
            "llm_output_index": None,
            "detection_confidence": getattr(d, 'score', 0.0),
            "ocr_confidence": None,
            "errors": []
        })
    return {
        "page": {"width": w, "height": h},
        "bubbles": bubbles,
        "errors": []
    }

@app.post("/ocr")
def ocr_api(req: OcrRequest):
    _warm_models()
    pipeline = TranslationPipeline(logger=ApiJobLogger(lambda l, m: None), lazy_models=True, shared_ocr_engine=WARM_OCR_ENGINE)
    if not pipeline._ensure_ocr_engine():
        raise HTTPException(500, "OCR init failed")
        
    import cv2
    img = cv2.imread(req.image_path)
    if img is None:
        raise HTTPException(404, "Image not found")
        
    crops = []
    for b in req.bubbles:
        bbox = b["bbox"]
        x1, y1 = int(bbox["x"]), int(bbox["y"])
        x2, y2 = x1 + int(bbox["w"]), y1 + int(bbox["h"])
        crops.append(img[max(0,y1):y2, max(0,x1):x2])
        
    ocr_results = pipeline.ocr_engine.extract_batch(crops)
    
    out_bubbles = []
    for b, res in zip(req.bubbles, ocr_results):
        text, conf, valid, reason, regions, upscale = res
        b["source_text"] = text if valid else ""
        b["ocr_confidence"] = conf
        if not valid:
            b.setdefault("errors", []).append({"code": reason, "message": "OCR failed"})
        out_bubbles.append(b)
        
    return {"bubbles": out_bubbles, "errors": []}

@app.post("/translate")
def translate_api(req: TranslateRequest):
    _warm_models()
    from core import NLLBTranslator
    
    texts = [b.get("source_text", "") for b in req.bubbles]
    
    global WARM_TRANSLATOR
    if WARM_TRANSLATOR is None:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        WARM_TRANSLATOR = NLLBTranslator(device=device)
        
    translator = WARM_TRANSLATOR
    if not translator:
        raise HTTPException(500, "Translator init failed")

    translated = translator.translate_batch(texts)
    
    # Naive glossary post-processing
    if req.glossary:
        for i, (src, tgt) in enumerate(zip(texts, translated)):
            for g_src, g_tgt in req.glossary.items():
                if g_src.lower() in src.lower():
                    if g_tgt.lower() not in tgt.lower():
                        translated[i] = f"{tgt} ({g_tgt})"
    
    out_bubbles = []
    parsed_mapping = []
    for i, (b, t) in enumerate(zip(req.bubbles, translated)):
        b["translated_text"] = t
        b["llm_input_index"] = i
        b["llm_output_index"] = i
        if "errors" not in b:
            b["errors"] = []
        out_bubbles.append(b)
        parsed_mapping.append({
            "input_index": i,
            "output_index": i,
            "bubble_id": b.get("id", str(i))
        })
        
    llm_debug = {
        "payload": {"items": [{"index": i, "text": txt} for i, txt in enumerate(texts)]},
        "raw_response": json.dumps({str(i): t for i, t in enumerate(translated)}),
        "parsed_mapping": parsed_mapping
    } if req.return_llm_debug else None
        
    return {
        "bubbles": out_bubbles,
        "llm_debug": llm_debug,
        "errors": []
    }

@app.post("/render")
def render_api(req: RenderRequest):
    import time
    start_time = time.time()
    
    _warm_models()
    from core import TextRenderer
    import cv2
    import uuid
    from pathlib import Path
    
    img = cv2.imread(req.image_path)
    if img is None:
        raise HTTPException(404, "Image not found")
        
    renderer = TextRenderer()
    out_img = img.copy()
    
    for b in req.bubbles:
        bbox = b["bbox"]
        x1, y1 = int(bbox["x"]), int(bbox["y"])
        x2, y2 = x1 + int(bbox["w"]), y1 + int(bbox["h"])
        text = b.get("translated_text") or b.get("source_text") or ""
        
        if text:
            style = b.get("text_style", {})
            color = style.get("color", "#000000")
            color_bgr = (0,0,0)
            if color.startswith("#") and len(color) == 7:
                color_bgr = (int(color[5:7], 16), int(color[3:5], 16), int(color[1:3], 16))
                
            stroke_color = style.get("stroke_color")
            stroke_color_bgr = None
            if stroke_color and stroke_color.startswith("#") and len(stroke_color) == 7:
                stroke_color_bgr = (int(stroke_color[5:7], 16), int(stroke_color[3:5], 16), int(stroke_color[1:3], 16))
                
            bg_color = style.get("bg_color")
            bg_color_bgr = None
            if bg_color and bg_color.startswith("#") and len(bg_color) == 7:
                bg_color_bgr = (int(bg_color[5:7], 16), int(bg_color[3:5], 16), int(bg_color[1:3], 16))
                
            stroke_width = style.get("stroke_width")
            angle = style.get("angle")
            
            chirurgical_mask = None
            mask_strokes = b.get("mask_strokes", [])
            if mask_strokes and not (req.skip_inpainting or req.text_only):
                # Construire le masque chirurgical local à la bbox (w, h)
                import numpy as np
                bw, bh = x2 - x1, y2 - y1
                if bw > 0 and bh > 0:
                    chirurgical_mask = np.zeros((bh, bw), dtype=np.uint8)
                    for stroke in mask_strokes:
                        size = int(stroke.get("size", 10))
                        points = stroke.get("points", [])
                        for i in range(len(points) - 1):
                            pt1 = points[i]
                            pt2 = points[i+1]
                            # Les points viennent du canvas global, on les ramène au repère de la bbox
                            cx1 = int(pt1["x"]) - x1
                            cy1 = int(pt1["y"]) - y1
                            cx2 = int(pt2["x"]) - x1
                            cy2 = int(pt2["y"]) - y1
                            cv2.line(chirurgical_mask, (cx1, cy1), (cx2, cy2), 255, size)
                            cv2.circle(chirurgical_mask, (cx1, cy1), size // 2, 255, -1)
                        if len(points) == 1:
                            cx = int(points[0]["x"]) - x1
                            cy = int(points[0]["y"]) - y1
                            cv2.circle(chirurgical_mask, (cx, cy), size // 2, 255, -1)
                
            out_img = renderer.render_text(
                out_img, text, x1, y1, x2, y2,
                text_color_rgb=color_bgr,
                class_name=b.get("class", ""),
                stroke_color_rgb=stroke_color_bgr,
                stroke_width=stroke_width,
                bg_color_rgb=bg_color_bgr,
                angle_override=angle,
                skip_inpainting=req.skip_inpainting or req.text_only,
                chirurgical_mask=chirurgical_mask
            )
            
    # Sauvegarde dans le dossier previews
    base_path = Path(req.image_path)
    previews_dir = base_path.parent.parent / "previews" if base_path.parent.name == "originals" else base_path.parent / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)
    
    preview_filename = f"{base_path.stem}_translated.png"
    preview_path = previews_dir / preview_filename
    cv2.imwrite(str(preview_path), out_img)
    
    total_ms = int((time.time() - start_time) * 1000)
    
    return {
        "preview_path": str(preview_path).replace("\\", "/"),
        "timings": {
            "text_render_ms": total_ms,
            "inpaint_ms": 0,
            "total_ms": total_ms
        },
        "errors": []
    }

# --- Batch Processing Endpoints ---

batch_status = {
    "status": "idle", # idle, running, done, error
    "total": 0,
    "processed": 0,
    "current_file": "",
    "errors": []
}

class BatchRequest(BaseModel):
    input_dir: str
    output_dir: str

def _run_batch(input_dir: str, output_dir: str):
    import glob
    from pathlib import Path
    
    global batch_status
    batch_status["status"] = "running"
    batch_status["total"] = 0
    batch_status["processed"] = 0
    batch_status["current_file"] = ""
    batch_status["errors"] = []
    
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    files = []
    for ext in extensions:
        files.extend(glob.glob(str(Path(input_dir) / "**" / ext), recursive=True))
        
    batch_status["total"] = len(files)
    
    if len(files) == 0:
        batch_status["status"] = "done"
        return
        
    _warm_models()
    pipeline = TranslationPipeline(logger=ApiJobLogger(lambda l, m: None), lazy_models=True, shared_ocr_engine=WARM_OCR_ENGINE)
    
    for f in files:
        if batch_status["status"] == "error":
            break
            
        batch_status["current_file"] = f
        try:
            import cv2
            img = cv2.imread(f)
            if img is not None:
                result = pipeline.process_image(img)
                if result:
                    out_path = Path(output_dir) / Path(f).relative_to(Path(input_dir))
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_path), result.output)
            batch_status["processed"] += 1
        except Exception as e:
            batch_status["errors"].append({"file": f, "error": str(e)})
            batch_status["processed"] += 1

    batch_status["status"] = "done"

@app.post("/batch")
def start_batch(req: BatchRequest):
    global batch_status
    if batch_status["status"] == "running":
        raise HTTPException(400, "A batch is already running")
        
    import threading
    thread = threading.Thread(target=_run_batch, args=(req.input_dir, req.output_dir), daemon=True)
    thread.start()
    return {"message": "Batch started"}

@app.get("/batch/status")
def get_batch_status():
    return batch_status

# --- Export Endpoints ---

class ExportRequest(BaseModel):
    input_dir: str
    output_path: str
    format: str = "cbz" # cbz, zip, webp
    watermark_text: str = ""
    
@app.post("/export")
def export_api(req: ExportRequest):
    import glob
    import zipfile
    from pathlib import Path
    import cv2
    import numpy as np
    
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    files = []
    for ext in extensions:
        files.extend(glob.glob(str(Path(req.input_dir) / "**" / ext), recursive=True))
        
    files.sort()
    
    if not files:
        raise HTTPException(400, "No images found in input_dir")
        
    if req.format in ["cbz", "zip"]:
        # Create archive
        with zipfile.ZipFile(req.output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for i, f in enumerate(files):
                img = cv2.imread(f)
                if img is None:
                    continue
                    
                # Add watermark to first page
                if i == 0 and req.watermark_text:
                    h, w = img.shape[:2]
                    cv2.putText(img, req.watermark_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(img, req.watermark_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
                    
                # Convert back to bytes
                is_success, buffer = cv2.imencode(".jpg", img)
                if is_success:
                    arcname = Path(f).name
                    zf.writestr(arcname, buffer.tobytes())
                    
        return {"message": f"Exported {len(files)} files to {req.output_path}"}
    else:
        raise HTTPException(400, "Unsupported format. Use cbz or zip.")

