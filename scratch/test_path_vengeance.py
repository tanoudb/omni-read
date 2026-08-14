import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"a:\omni read")))
from core.renderer import TextRenderer
from pipeline import TranslationPipeline
from utils import WebtoonLogger
from utils.dry_run_fast import build_pipeline, run_dry_run_on_image

p = build_pipeline()
renderer = TextRenderer()
img_path = Path(r"a:\omni read\manhwa\path-of-vengeance\Chapitre 001\Chapitre 001_merged_part01.jpg")
out_dir = Path(r"a:\omni read\tests\dry_run_out\path-of-vengeance\Chapitre 001")
out_dir.mkdir(parents=True, exist_ok=True)
run_dry_run_on_image(p, renderer, img_path, out_dir)
