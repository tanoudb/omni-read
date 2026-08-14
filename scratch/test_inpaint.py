import cv2
import sys
from pathlib import Path
sys.path.insert(0, str(Path(r"a:\omni read")))
from core.renderer import TextRenderer
from pipeline import TranslationPipeline
from utils import WebtoonLogger
from config import config

p = TranslationPipeline.__new__(TranslationPipeline)
p.logger = WebtoonLogger("test")
p.device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

img = cv2.imread(str(Path(r"a:\omni read\tests\dry_run_out\path-of-vengeance\Chapitre 001\Chapitre 001_merged_part01_dryrun_translated.png")))
print("Image shape:", img.shape)
