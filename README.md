# Webtoon Translator V5

Pipeline local de traduction automatique pour images manga/manhwa/webtoon.
Il détecte les zones texte, extrait l'OCR, traduit avec LLM local, puis recompose le texte traduit dans l'image finale.

## Features

- Détection multi-échelle YOLO avec sliding window adaptatif
- Filtrage containment + conflits inter-classes pour réduire les doublons
- OCR principal/fallback avec RapidOCR (ONNX Runtime CUDA)
- Traduction locale avec Qwen (bitsandbytes 4-bit) ou NLLB
- Traduction page entière indexée + fallback par bulle
- Inpainting local (Simple-LaMa / fallback OpenCV)
- Rendu typographique avec adaptation de taille, wrapping, et centrage
- Mode debug avec artefacts intermédiaires et logs détaillés

## Prérequis

- Python 3.11
- Windows 10/11 ou Linux
- GPU NVIDIA recommandé (6 Go+ VRAM)
- CUDA 12.x recommandé

## Installation

### 1) Cloner et créer l'environnement

```powershell
git clone <URL_DU_REPO>
cd "omni read"
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) (Optionnel) dépendances dev

```powershell
pip install -r requirements-dev.txt
```

## Usage

### Run standard

```powershell
.\run_phase1.ps1
```

### Run image unique

```powershell
python main.py --image "image test\image1.png" --output "output" --debug
```

### Afficher la config active

```powershell
python main.py --show-config
```

## Configuration (variables d'environnement)

### Traduction LLM locale

- WEBTOON_TRANSLATION_BACKEND=local_llm
- WEBTOON_LLM_MODEL=Qwen/Qwen2.5-3B-Instruct
- WEBTOON_USE_BITSANDBYTES=true
- WEBTOON_BNB_4BIT=true
- WEBTOON_BNB_8BIT=false

### OCR

- WEBTOON_USE_VL15=false
- WEBTOON_OCR_PRIMARY=paddleocr-vl-v1.5
- WEBTOON_OCR_FALLBACKS=rapidocr-ppocrv5

### Segmentation / rendu

- WEBTOON_SEGMENTATION_BACKEND=hybrid
- WEBTOON_ENABLE_PRECISE_MASKS=true

## Performance

Mesures observées sur image longue en mode debug (GPU CUDA):

- Détection YOLO: ~2.5s à 3.5s
- OCR batch: ~1.6s à 3.0s
- LLM total: ~19s à 30s (génération pure plus faible)
- Inpainting: poste dominant (~14s à 22s)
- Total pipeline: ~49s à 60s selon image et texte

Les benchmarks sont aussi archivés dans benchmark_results.csv.

## Structure utile

- main.py : point d'entrée CLI
- pipeline.py : orchestration complète
- core/detector.py : détection YOLO
- core/ocr.py : OCR orchestrator
- core/translator.py : traduction LLM/NLLB
- core/renderer.py : inpainting + rendu final
- python_legacy/lazy_api.py : API FastAPI asynchrone

## Roadmap

- Stabiliser le mapping index->traduction lorsque le LLM dévie
- Réduire le coût inpainting pour revenir vers ~40s/image
- Améliorer qualité FR sur textes ambigus et OCR bruité
- Ajouter tests automatiques non-régression (prompt + mapping)
- Ajouter CI lint + checks de packaging

