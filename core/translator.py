"""
═══════════════════════════════════════════════════════════════════════════════
TRANSLATOR v3 - NLLB avec nettoyage texte OCR + GPU FIX
═══════════════════════════════════════════════════════════════════════════════

FIX MAJEUR: Nettoyage du texte OCR AVANT traduction.
Les OCR manga produisent souvent des artefacts :
  - Mots collés : "IWANTED" → "I WANTED"
  - Ponctuation collée : "STRENGTHI" → "STRENGTH!"  
  - Underscores : "HIM_" → "HIM."

GPU FIX: Force explicitement le modèle sur CUDA avec vérification

Config: model_name dans settings.py
  - "facebook/nllb-200-distilled-600M"  (rapide, ~1.5GB VRAM)
  - "facebook/nllb-200-distilled-1.3B"  (meilleur, ~3GB VRAM)
"""

import re
import json
import os
import numpy as np
import torch
from typing import List, Optional, Tuple
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from utils import CacheManager, ImageUtils

try:
    import torch
except ImportError:
    raise RuntimeError("PyTorch requis: pip install torch")


# ═════════════════════════════════════════════════════════════════════════════
# NETTOYAGE TEXTE OCR
# ═════════════════════════════════════════════════════════════════════════════
def should_skip_translation(self, text: str) -> bool:
    if not text:
        return True
    
    # Skip si coréen (pas de lettres latines)
    if not any(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz' for c in text):
        return True
    
    # ... reste du code
def clean_ocr_text(text: str) -> str:
    """
    Nettoie le texte brut OCR avant traduction.
    Corrige les artefacts typiques sur du texte manga.
    """
    if not text:
        return text

    # Normaliser les guillemets typographiques
    text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")

    # Retirer des guillemets externes parasites (cas OCR: ''TEXT'' ou "TEXT")
    for _ in range(3):
        stripped = text.strip()
        if len(stripped) >= 2 and (
            (stripped[0] == stripped[-1] and stripped[0] in {'"', "'"})
            or (stripped.startswith("''") and stripped.endswith("''"))
            or (stripped.startswith('""') and stripped.endswith('""'))
        ):
            text = stripped[1:-1]
            continue
        break
    
    # Retirer underscores (OCR lit _ au lieu de . ou espace)
    text = re.sub(r'_+$', '.', text)
    text = re.sub(r'_+', ' ', text)
    
    # Fix mots collés avec I majuscule en début
    # "IWANTED" → "I WANTED", "ICOULD" → "I COULD"
    text = re.sub(r'\bI([A-Z]{2,})', r'I \1', text)
    
    # Fix I parasite en fin de mot majuscule
    # "STRENGTHI" → "STRENGTH!", "SURVIVEI" → "SURVIVE!"  
    text = re.sub(r'([A-Z]{3,})I\b', r'\1!', text)
    
    # Fix ; → , (OCR confond souvent)
    text = text.replace(';', ',')
    
    # Fix : en fin de phrase → .
    text = re.sub(r':\s*$', '.', text)

    # Fix fréquent PP-OCR: "1." reconnu à la place de "I."
    # ex: "YOU'RE TELLING ME THAT 1. THE BEST..." -> "... I. THE BEST..."
    text = re.sub(r'\b1\.(?=\s+[A-Z])', 'I.', text)

    # OCR ponctuation: "I. THE" est souvent "I, THE"
    text = re.sub(r'\bI\.(?=\s+THE\b)', 'I,', text)

    # Un "1" isolé entre mots majuscules est souvent un "I"
    text = re.sub(r'(?<=[A-Z])\s+1\s+(?=[A-Z])', ' I ', text)

    # Corrections OCR ciblées (webtoon)
    text = re.sub(r'\bDALIGHTER\b', 'DAUGHTER', text, flags=re.IGNORECASE)
    text = re.sub(r'\bDAIGTHER\b', 'DAUGHTER', text, flags=re.IGNORECASE)
    text = re.sub(r'\bWHERE\s+WE\s+ARE\?', 'WHERE ARE WE?', text, flags=re.IGNORECASE)
    
    # Nettoyer espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


class TranslationGroup:
    def __init__(self, detections: list):
        self.detections = detections
        self.combined_text: Optional[str] = None
        self.translation: Optional[str] = None
    
    def get_center(self) -> Tuple[float, float]:
        if not self.detections:
            return (0, 0)
        centers = [((d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2) for d in self.detections]
        return (
            sum(c[0] for c in centers) / len(centers),
            sum(c[1] for c in centers) / len(centers)
        )


class NLLBTranslator:
    """Traducteur NLLB/LLM local avec nettoyage OCR et GPU fix"""

    SINGLE_WORD_NON_NAME_STOPLIST = {
        "perhaps", "later", "hello", "look", "wait", "please", "help",
        "where", "what", "when", "why", "how", "there", "here",
        "yes", "no", "stop", "start", "wake", "time", "moment"
    }

    MULTI_WORD_NON_NAME_STOPLIST = {
        "oh", "there", "here", "please", "help", "wait", "look",
        "yes", "no", "what", "when", "where", "why", "how",
        "stop", "start", "go", "come", "now", "again"
    }

    LANGUAGE_STOPWORDS = {
        "en": {"the", "and", "you", "are", "what", "there", "here", "oh", "this", "that"},
        "fr": {"le", "la", "les", "et", "vous", "que", "est", "pas", "une", "des"},
        "es": {"el", "la", "los", "las", "y", "que", "una", "por", "para", "está"},
        "de": {"der", "die", "das", "und", "ist", "nicht", "ein", "eine", "mit", "ich"},
        "it": {"il", "lo", "la", "gli", "le", "e", "che", "non", "una", "con"},
        "pt": {"o", "a", "os", "as", "e", "que", "não", "uma", "com", "para"},
    }
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.cfg = config.translation
        self.backend = getattr(self.cfg, 'backend', 'nllb')
        self.tokenizer = None
        self.model = None
        self.llm_backend = "transformers"
        self.gguf_model_path: Optional[Path] = None
        self.generation_seconds_total: float = 0.0
        self.last_page_system_prompt: str = ""
        self.last_page_user_prompt: str = ""
        self.last_page_payload_lines: List[str] = []
        self.cache = None
        self.name_memory_file = config.TRANSLATION_CACHE_DIR / "name_memory_v1.json"
        self.name_memory = self._load_name_memory()
        
        if self.cfg.enable_cache:
            cache_file = config.TRANSLATION_CACHE_DIR / self.cfg.cache_file
            self.cache = CacheManager(cache_file, max_size_mb=config.performance.cache_max_size_mb)
        
        self._load_model()

    def _load_name_memory(self) -> dict:
        try:
            if self.name_memory_file.exists():
                with open(self.name_memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data if isinstance(data, dict) else {}
        except Exception:
            pass
        return {}

    def _save_name_memory(self):
        try:
            self.name_memory_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.name_memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.name_memory, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    @staticmethod
    def _normalize_text_key(text: str) -> str:
        normalized = re.sub(r"\s+", " ", text.strip().upper())
        return normalized

    @staticmethod
    def _post_process_french(text: str) -> str:
        if not text:
            return text
        text = re.sub(r"\bJe y\b", "J'y", text)
        text = re.sub(r"\bje y\b", "j'y", text)
        text = re.sub(r"\bde déchets\b", "d'ordure", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _is_mostly_uppercase(text: str) -> bool:
        letters = [ch for ch in text if ch.isalpha()]
        if len(letters) < 6:
            return False
        upper_count = sum(1 for ch in letters if ch.isupper())
        return (upper_count / max(1, len(letters))) >= 0.75

    @staticmethod
    def _normalize_case_for_translation(text: str) -> str:
        """Normalise les textes OCR en MAJUSCULES pour améliorer la traduction."""
        if not text:
            return text

        if not NLLBTranslator._is_mostly_uppercase(text):
            return text

        normalized = text.lower()
        normalized = re.sub(r"\bi\b", "I", normalized)

        # Majuscule sur le premier caractère alphabétique
        chars = list(normalized)
        for idx, ch in enumerate(chars):
            if ch.isalpha():
                chars[idx] = ch.upper()
                break
        normalized = ''.join(chars)
        return normalized

    @staticmethod
    def _name_key(text: str) -> Optional[str]:
        if not text:
            return None
        cleaned = re.sub(r"[^A-Za-z'\-\s]", " ", text.upper())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return None
        words = cleaned.split()
        if 2 <= len(words) <= 4 and all(len(w) >= 2 for w in words):
            if all(re.fullmatch(r"[A-Z][A-Z'\-]*", w) for w in words):
                return " ".join(words)
        return None

    @classmethod
    def _looks_like_uppercase_name_sequence(cls, text: str) -> bool:
        tokens = re.findall(r"[A-Z][A-Z'\-]+", text.upper())
        if not (2 <= len(tokens) <= 4):
            return False

        # Si le groupe contient des mots usuels de dialogue, ce n'est pas un nom.
        if any(tok.lower() in cls.MULTI_WORD_NON_NAME_STOPLIST for tok in tokens):
            return False

        # Évite de classer des segments trop courts comme des noms.
        # Ex: "OH THERE" -> rejeté, "GHISLAIN PERDIUM" -> accepté.
        if any(len(tok) < 3 for tok in tokens):
            return False

        return True

    def _detect_source_language(self, text: str) -> str:
        return self._detect_source_language_with_confidence(text)[0]

    def _detect_source_language_with_confidence(self, text: str) -> Tuple[str, float]:
        fallback = self.cfg.fallback_source_lang if self.cfg.fallback_source_lang in self.cfg.lang_codes else self.cfg.source_lang

        if not self.cfg.auto_detect_source_lang:
            return self.cfg.source_lang, 1.0

        if not text:
            return fallback, 0.4

        # Scripts non-latins (detection robuste)
        if re.search(r"[\uAC00-\uD7AF]", text):
            return "ko", 0.99
        if re.search(r"[\u3040-\u30FF]", text):
            return "ja", 0.99
        if re.search(r"[\u4E00-\u9FFF]", text):
            return "zh", 0.99
        if re.search(r"[\u0400-\u04FF]", text):
            return "ru", 0.99

        # Latin: heuristique légère par stopwords
        words = re.findall(r"[A-Za-zÀ-ÿ']+", text.lower())
        if not words:
            return fallback, 0.4

        scores = {}
        for lang, stopwords in self.LANGUAGE_STOPWORDS.items():
            scores[lang] = sum(1 for w in words if w in stopwords)

        best_lang = max(scores, key=scores.get)
        sorted_scores = sorted(scores.values(), reverse=True)
        best_score = sorted_scores[0] if sorted_scores else 0
        second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0
        if best_score >= 1:
            margin = max(0, best_score - second_score)
            confidence = min(0.95, 0.60 + 0.15 * best_score + 0.08 * margin)
            return best_lang, float(confidence)

        # Si latin sans signal fort, on garde l'anglais (cas OCR webtoon le plus fréquent)
        return ("en" if "en" in self.cfg.lang_codes else fallback), 0.5

    def detect_source_language(self, text: str) -> str:
        """API publique pour debug/inspection de la langue source détectée."""
        return self._detect_source_language_with_confidence(text)[0]

    def detect_source_language_with_confidence(self, text: str) -> Tuple[str, float]:
        """Retourne (langue, confiance) pour le texte OCR."""
        return self._detect_source_language_with_confidence(text)

    @staticmethod
    def _is_single_proper_name(text: str) -> bool:
        stripped = text.strip()
        if not re.fullmatch(r"[A-Z][a-z]{2,20}[.!?]?", stripped):
            return False
        token = re.sub(r"[.!?]$", "", stripped).lower()
        if token in NLLBTranslator.SINGLE_WORD_NON_NAME_STOPLIST:
            return False
        return True
    
    def _load_model(self):
        if self.backend == 'local_llm':
            self._load_local_llm_model()
            return

        try:
            from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
            
            model_name = self.cfg.model_name
            print(f"⏳ Chargement NLLB: {model_name}...")
            
            self.tokenizer = NllbTokenizer.from_pretrained(
                model_name,
                cache_dir=str(config.TRANSLATION_CACHE_DIR),
                trust_remote_code=True
            )
            
            # ✅ FIX GPU: Forcer dtype correctement
            dtype = torch.float16 if self.device == 'cuda' and self.cfg.use_fp16 else torch.float32
            quantization_config = None

            if self.device == 'cuda' and self.cfg.use_bitsandbytes:
                try:
                    from transformers import BitsAndBytesConfig
                    if self.cfg.bnb_4bit:
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=dtype,
                            bnb_4bit_quant_type='nf4',
                            bnb_4bit_use_double_quant=True,
                        )
                        print("   Quantization: bitsandbytes 4-bit (nf4)")
                    elif self.cfg.bnb_8bit:
                        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                        print("   Quantization: bitsandbytes 8-bit")
                    else:
                        print("   Quantization: bitsandbytes activé mais aucun mode sélectionné")
                except Exception as quant_error:
                    print(f"⚠️  BitsAndBytes indisponible ({quant_error}) -> fallback FP16/FP32")
                    quantization_config = None
            
            print(f"   Dtype: {dtype}")
            print(f"   Device: {self.device}")

            base_model_kwargs = {
                'cache_dir': str(config.TRANSLATION_CACHE_DIR),
                'trust_remote_code': True,
                'use_safetensors': False,
                'low_cpu_mem_usage': False,
            }

            if quantization_config is not None:
                base_model_kwargs['quantization_config'] = quantization_config
            else:
                base_model_kwargs['torch_dtype'] = dtype

            def _find_local_bin_snapshot() -> Optional[Path]:
                model_cache = config.TRANSLATION_CACHE_DIR / f"models--{model_name.replace('/', '--')}" / "snapshots"
                if not model_cache.exists():
                    return None
                for snapshot in sorted(model_cache.iterdir(), reverse=True):
                    if not snapshot.is_dir():
                        continue
                    if (snapshot / 'config.json').exists() and (snapshot / 'pytorch_model.bin').exists():
                        return snapshot
                return None

            load_sources = [
                (model_name, {**base_model_kwargs, 'local_files_only': True}),
            ]

            local_snapshot = _find_local_bin_snapshot()
            if local_snapshot is not None:
                load_sources.append((str(local_snapshot), {**base_model_kwargs, 'local_files_only': True}))

            load_sources.append((model_name, {**base_model_kwargs, 'local_files_only': False}))

            last_error = None
            for source, kwargs in load_sources:
                try:
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(source, **kwargs)
                    print(f"   Chargement NLLB OK depuis: {source}")
                    break
                except Exception as load_error:
                    last_error = load_error
                    print(f"⚠️  Échec chargement depuis {source}: {load_error}")

            if self.model is None and last_error is not None:
                raise last_error

            if self.device == 'cuda' and torch.cuda.is_available() and quantization_config is None:
                self.model = self.model.to('cuda')
            else:
                self.model = self.model.to('cpu')
                self.device = 'cpu'

            model_device = next(self.model.parameters()).device
            print(f"   Model device: {model_device}")
            
            self.model.eval()
            print(f"✅ NLLB chargé ! ({model_name})")
            print(f"✅ Model running on: {next(self.model.parameters()).device}\n")
            
        except Exception as e:
            raise RuntimeError(f"Erreur chargement NLLB: {e}")

    def _load_local_llm_model(self):
        model_name = self._select_llm_model_name()
        gguf_path = self._resolve_gguf_model_path(model_name)
        if gguf_path is not None:
            self._load_local_gguf_model(gguf_path)
            return

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

            self.llm_backend = "transformers"
            force_low_vram_mode = model_name.endswith("2.5-3B-Instruct")
            print(f"⏳ Chargement LLM local: {model_name}...")

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(config.TRANSLATION_CACHE_DIR),
                trust_remote_code=True,
            )

            dtype = torch.float16 if self.device == 'cuda' and self.cfg.use_fp16 else torch.float32
            quantization_config = None

            require_cuda = bool(getattr(self.cfg, 'llm_require_cuda', True))
            if self.device == 'cuda' and not torch.cuda.is_available():
                message = "CUDA demandé pour le LLM mais torch ne voit pas de GPU (build CPU ou drivers absents)."
                if require_cuda:
                    raise RuntimeError(message)
                print(f"⚠️  {message} Fallback CPU activé.")
                self.device = 'cpu'

            if self.device == 'cuda' and (self.cfg.use_bitsandbytes or force_low_vram_mode):
                try:
                    use_4bit = self.cfg.bnb_4bit or force_low_vram_mode
                    if use_4bit:
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=dtype,
                            bnb_4bit_quant_type='nf4',
                            bnb_4bit_use_double_quant=True,
                        )
                        print("   Quantization LLM: bitsandbytes 4-bit (nf4)")
                    elif self.cfg.bnb_8bit:
                        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                        print("   Quantization LLM: bitsandbytes 8-bit")
                except Exception as quant_error:
                    print(f"⚠️  BitsAndBytes indisponible ({quant_error}) -> fallback FP16/FP32")
                    quantization_config = None

            model_kwargs = {
                'cache_dir': str(config.TRANSLATION_CACHE_DIR),
                'trust_remote_code': True,
                'low_cpu_mem_usage': True,
            }

            if quantization_config is not None:
                model_kwargs['quantization_config'] = quantization_config
                model_kwargs['device_map'] = 'auto'
            else:
                model_kwargs['torch_dtype'] = dtype

            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

            if self.device == 'cuda' and quantization_config is None and torch.cuda.is_available():
                self.model = self.model.to('cuda')

            self.model.eval()
            print(f"✅ LLM local chargé ! ({model_name})")
            print(f"✅ Model running on: {next(self.model.parameters()).device}\n")

        except Exception as e:
            raise RuntimeError(f"Erreur chargement LLM local: {e}")

    def _resolve_gguf_model_path(self, model_name: str) -> Optional[Path]:
        candidate = Path(str(model_name).strip())
        if candidate.suffix.lower() == ".gguf":
            if candidate.exists() and candidate.is_file():
                return candidate
            relative = Path(__file__).resolve().parents[1] / candidate
            if relative.exists() and relative.is_file():
                return relative
        return None

    def _load_local_gguf_model(self, model_path: Path):
        try:
            from llama_cpp import Llama
        except Exception as exc:
            raise RuntimeError(f"llama-cpp-python indisponible pour GGUF: {exc}")

        self.llm_backend = "gguf"
        self.gguf_model_path = model_path

        n_ctx = int(max(2048, int(getattr(self.cfg, 'max_length', 640)) * 2))
        n_threads = max(1, (os.cpu_count() or 8) // 2)
        n_gpu_layers = int(os.environ.get("WEBTOON_GGUF_N_GPU_LAYERS", "-1")) if (self.device == 'cuda' and torch.cuda.is_available()) else 0

        print(f"⏳ Chargement LLM GGUF: {model_path}...")
        self.model = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        self.tokenizer = None
        print(f"✅ LLM GGUF chargé ! ({model_path.name})")
        print(f"✅ GGUF runtime: n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers}\n")

    @staticmethod
    def _gguf_extract_content(response: dict) -> str:
        if not isinstance(response, dict):
            return ""
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first.get("message"), dict) else {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        text = first.get("text")
        return str(text).strip() if text is not None else ""

    def _select_llm_model_name(self) -> str:
        forced_model = os.environ.get("WEBTOON_LLM_MODEL", "").strip()
        if forced_model:
            return forced_model
        cfg_model = str(getattr(self.cfg, 'llm_model_name', '') or '').strip()
        if cfg_model:
            return cfg_model
        if self.device != 'cuda' or not torch.cuda.is_available():
            return "Qwen/Qwen2.5-3B-Instruct"
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024 ** 3)
            if free_gb > 4.0:
                return "Qwen/Qwen2.5-7B-Instruct"
        except Exception:
            pass
        return "Qwen/Qwen2.5-3B-Instruct"

    def _build_llm_prompt(self, text: str, source_lang_code: str) -> str:
        source_lang = source_lang_code or self.cfg.source_lang
        target_lang = self.cfg.target_lang
        template = getattr(self.cfg, 'llm_prompt_template', None) or (
            "Translate from {source_lang} to {target_lang}. Output only the translation.\n"
            "TEXT:\n{text}\nTRANSLATION:"
        )
        return template.format(source_lang=source_lang, target_lang=target_lang, text=text)

    @staticmethod
    def _extract_llm_translation(raw_output: str, prompt: str) -> str:
        if not raw_output:
            return ""
        content = raw_output[len(prompt):] if raw_output.startswith(prompt) else raw_output
        content = content.strip()

        if "TRANSLATION:" in content:
            content = content.split("TRANSLATION:", 1)[1].strip()

        marker_match = re.split(r"\n\s*(Human|User|Assistant|System)\s*:\s*", content, maxsplit=1)
        if marker_match:
            content = marker_match[0].strip()

        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            return ""

        meta_re = re.compile(
            r"^(hello\b|hi\b|sure\b|here\s+is\b|here's\b|i\s*(am|'m)\b|as\s+an\b|translation\s*:)",
            flags=re.IGNORECASE,
        )
        while lines and meta_re.match(lines[0]):
            lines.pop(0)
        if not lines:
            return ""

        first_line = lines[0]
        if first_line.startswith(('"', "'")) and first_line.endswith(('"', "'")) and len(first_line) >= 2:
            first_line = first_line[1:-1].strip()

        return first_line

    def _translate_with_local_llm(self, source_text: str, source_lang_code: str) -> str:
        prompt = self._build_llm_prompt(source_text, source_lang_code)

        if self.llm_backend == "gguf":
            import time

            temperature = float(getattr(self.cfg, 'llm_temperature', 0.0))
            top_p = float(getattr(self.cfg, 'llm_top_p', 1.0))
            max_tokens = int(getattr(self.cfg, 'llm_max_new_tokens', 220))
            repeat_penalty = float(getattr(self.cfg, 'llm_repetition_penalty', 1.05))

            messages = [
                {
                    'role': 'system',
                    'content': (
                        'You are a translation engine. Return only the translated text on a single line. '
                        'Never start with Hello/I am/Here is. '
                        'If the input is onomatopoeia/SFX or watermark/credits/URL/@handle, return it unchanged.'
                    ),
                },
                {'role': 'user', 'content': prompt},
            ]

            t0 = time.perf_counter()
            response = self.model.create_chat_completion(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                repeat_penalty=repeat_penalty,
            )
            self.generation_seconds_total += max(0.0, time.perf_counter() - t0)

            raw = self._gguf_extract_content(response)
            translation = self._extract_llm_translation(raw, prompt)
            return translation or source_text

        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {
                    'role': 'system',
                    'content': (
                        'You are a translation engine. Return only the translated text on a single line. '
                        'Never start with Hello/I am/Here is. '
                        'If the input is onomatopoeia/SFX or watermark/credits/URL/@handle, return it unchanged.'
                    ),
                },
                {'role': 'user', 'content': prompt},
            ]
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors='pt',
            )
            if isinstance(inputs, torch.Tensor):
                input_ids = inputs
                attention_mask = torch.ones_like(input_ids)
                inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        else:
            inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=self.cfg.max_length)

        if self.device == 'cuda' and torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}

        generate_kwargs = {
            'max_new_tokens': int(getattr(self.cfg, 'llm_max_new_tokens', 220)),
            'repetition_penalty': float(getattr(self.cfg, 'llm_repetition_penalty', 1.05)),
            'pad_token_id': self.tokenizer.eos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }

        temperature = float(getattr(self.cfg, 'llm_temperature', 0.0))
        top_p = float(getattr(self.cfg, 'llm_top_p', 1.0))
        if temperature > 0:
            generate_kwargs['do_sample'] = True
            generate_kwargs['temperature'] = temperature
            generate_kwargs['top_p'] = top_p
        else:
            generate_kwargs['do_sample'] = False

        gen_t0 = torch.cuda.Event(enable_timing=True) if (self.device == 'cuda' and torch.cuda.is_available()) else None
        gen_t1 = torch.cuda.Event(enable_timing=True) if (self.device == 'cuda' and torch.cuda.is_available()) else None
        wall_t0 = torch.cuda.Event(enable_timing=False) if False else None

        if gen_t0 is not None and gen_t1 is not None:
            gen_t0.record()
        else:
            import time
            _wall_start = time.perf_counter()

        with torch.no_grad():
            generated = self.model.generate(**inputs, **generate_kwargs)

        if gen_t0 is not None and gen_t1 is not None:
            gen_t1.record()
            torch.cuda.synchronize()
            self.generation_seconds_total += max(0.0, float(gen_t0.elapsed_time(gen_t1)) / 1000.0)
        else:
            import time
            self.generation_seconds_total += max(0.0, time.perf_counter() - _wall_start)

        input_token_count = int(inputs['input_ids'].shape[-1])
        new_tokens = generated[0][input_token_count:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        translation = self._extract_llm_translation(raw, prompt)
        return translation or source_text

    def _translate_page_with_local_llm(self, texts: List[str]) -> dict:
        numbered_lines = [f"{idx}. {txt}" for idx, txt in enumerate(texts)]
        user_prompt = "\n".join(numbered_lines)
        system_prompt = (
            "Tu es un traducteur expert manga/webtoon/manhwa EN→FR.\n"
            "Règles : français naturel jamais littéral. Conserve le ton émotionnel.\n"
            "Les onomatopées (HUFF, AHH, BOOM, CRASH...) restent EXACTEMENT en original, ne jamais traduire.\n"
            "Si c'est un watermark/crédit/scanlation/URL/@handle, renvoie le texte inchangé.\n"
            "Ne commence jamais par Hello/I am/Here is.\n"
            "Respecte la ponctuation et effets stylistiques (!!!, ...).\n"
            "Réponds uniquement en JSON strict de la forme {\"0\":\"...\",\"1\":\"...\"}.\n"
            "Ne mets aucun texte hors JSON."
        )

        self.last_page_payload_lines = list(numbered_lines)
        self.last_page_system_prompt = system_prompt
        self.last_page_user_prompt = user_prompt

        if self.llm_backend == "gguf":
            import time

            t0 = time.perf_counter()
            response = self.model.create_chat_completion(
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ],
                temperature=float(getattr(self.cfg, 'llm_temperature', 0.0)),
                top_p=float(getattr(self.cfg, 'llm_top_p', 1.0)),
                max_tokens=int(getattr(self.cfg, 'llm_max_new_tokens', 512)),
                repeat_penalty=float(getattr(self.cfg, 'llm_repetition_penalty', 1.05)),
            )
            self.generation_seconds_total += max(0.0, time.perf_counter() - t0)
            raw = self._gguf_extract_content(response).strip()
            return self._parse_page_translation_output(raw, texts)

        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ]
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors='pt',
            )
            if isinstance(inputs, torch.Tensor):
                input_ids = inputs
                attention_mask = torch.ones_like(input_ids)
                inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        else:
            prompt = f"{system_prompt}\n\n{user_prompt}"
            inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=self.cfg.max_length)

        if self.device == 'cuda' and torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}

        generate_kwargs = {
            'max_new_tokens': int(getattr(self.cfg, 'llm_max_new_tokens', 512)),
            'repetition_penalty': float(getattr(self.cfg, 'llm_repetition_penalty', 1.05)),
            'pad_token_id': self.tokenizer.eos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'do_sample': False,
            'temperature': float(getattr(self.cfg, 'llm_temperature', 0.0)),
        }

        if self.device == 'cuda' and torch.cuda.is_available():
            gen_t0 = torch.cuda.Event(enable_timing=True)
            gen_t1 = torch.cuda.Event(enable_timing=True)
            gen_t0.record()
            with torch.no_grad():
                generated = self.model.generate(**inputs, **generate_kwargs)
            gen_t1.record()
            torch.cuda.synchronize()
            self.generation_seconds_total += max(0.0, float(gen_t0.elapsed_time(gen_t1)) / 1000.0)
        else:
            import time
            t0 = time.perf_counter()
            with torch.no_grad():
                generated = self.model.generate(**inputs, **generate_kwargs)
            self.generation_seconds_total += max(0.0, time.perf_counter() - t0)

        input_token_count = int(inputs['input_ids'].shape[-1])
        raw = self.tokenizer.decode(generated[0][input_token_count:], skip_special_tokens=True).strip()
        return self._parse_page_translation_output(raw, texts)

    def get_last_page_payload_debug(self) -> dict:
        return {
            'system_prompt': self.last_page_system_prompt or '',
            'user_prompt': self.last_page_user_prompt or '',
            'payload_lines': list(self.last_page_payload_lines or []),
        }

    @staticmethod
    def _clean_llm_value(value: str) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
        text = text.strip(" \t\r\n,;")
        for _ in range(4):
            if len(text) >= 2 and (
                (text[0] == text[-1] and text[0] in {'"', "'"})
                or (text.startswith("''") and text.endswith("''"))
                or (text.startswith('""') and text.endswith('""'))
            ):
                text = text[1:-1].strip()
                continue
            break
        text = re.sub(r'^index\s*[:=]\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\d+\s*[\.:\-\)]\s*', '', text)
        return text.strip()

    @staticmethod
    def _extract_json_candidate(raw: str) -> str:
        content = (raw or "").strip()
        if content.startswith("```"):
            content = re.sub(r'^```(?:json)?\s*', '', content, flags=re.IGNORECASE)
            content = re.sub(r'\s*```$', '', content).strip()

        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1 and end > start:
            return content[start:end + 1]
        return content

    def _parse_page_translation_output(self, raw: str, source_texts: List[str]) -> dict:
        expected = len(source_texts)
        mapped: dict = {}

        json_candidate = self._extract_json_candidate(raw)
        parsed = None
        try:
            parsed = json.loads(json_candidate)
        except Exception:
            parsed = None

        if isinstance(parsed, dict):
            for key, value in parsed.items():
                key_str = str(key).strip().lower()
                value_str = self._clean_llm_value(value)

                key_num = re.fullmatch(r'\d+', str(key).strip())
                if key_num:
                    mapped[str(int(key_num.group(0)))] = value_str
                    continue

                # Cas: {"index": "0. ..."}
                if key_str == 'index':
                    m = re.match(r'\s*(\d+)\s*[\.:\-\)]\s*(.+)$', str(value).strip())
                    if m:
                        mapped[str(int(m.group(1)))] = self._clean_llm_value(m.group(2))
                    elif expected == 1:
                        mapped['0'] = value_str
                    continue

                # Cas fragments: value contient "2. texte" ou '"index": "2. texte"'
                m = re.search(r'"?index"?\s*:\s*"?\s*(\d+)\s*[\.:\-\)]\s*(.+?)"?\s*$', str(value), flags=re.IGNORECASE)
                if m:
                    mapped[str(int(m.group(1)))] = self._clean_llm_value(m.group(2))
                    continue

                m = re.match(r'\s*(\d+)\s*[\.:\-\)]\s*(.+)$', str(value).strip())
                if m:
                    mapped[str(int(m.group(1)))] = self._clean_llm_value(m.group(2))

        # Fallback texte brut ligne par ligne (si JSON incomplet/malformé)
        if len(mapped) < expected:
            lines = [ln.strip() for ln in (raw or '').splitlines() if ln.strip()]
            for ln in lines:
                candidate = self._clean_llm_value(ln)
                m = re.match(r'^(\d+)\s*[\.:\-\)]\s*(.+)$', candidate)
                if m:
                    mapped[str(int(m.group(1)))] = self._clean_llm_value(m.group(2))
                else:
                    m = re.search(r'^"(\d+)"\s*:\s*"(.+)"$', candidate)
                    if m:
                        mapped[str(int(m.group(1)))] = self._clean_llm_value(m.group(2))

        # Remplir les index manquants avec la source nettoyée
        final_map = {}
        for i, src in enumerate(source_texts):
            tr = self._clean_llm_value(mapped.get(str(i), ''))
            final_map[str(i)] = tr if tr else src
        return final_map

    def translate_page_json(self, texts: List[str]) -> dict:
        if not texts:
            return {}
        clean_inputs = [clean_ocr_text((text or '').strip()) for text in texts]
        if self.backend == 'local_llm':
            return self._translate_page_with_local_llm(clean_inputs)

        # fallback NLLB: traduction unitaire + map JSON indexée
        out = {}
        for i, text in enumerate(clean_inputs):
            out[str(i)] = self.translate(text)
        return out

    def get_generation_seconds_total(self) -> float:
        return float(self.generation_seconds_total)
    
    def group_detections_by_context(self, detections: list) -> List[TranslationGroup]:
        if not self.cfg.enable_context_grouping or not detections:
            return [TranslationGroup([d]) for d in detections]
        
        sorted_dets = sorted(detections, key=lambda d: d.y1)
        groups = []
        current_group = [sorted_dets[0]]
        
        for det in sorted_dets[1:]:
            last_det = current_group[-1]
            distance = ImageUtils.distance_between_boxes(last_det.bbox, det.bbox)
            
            if (distance < self.cfg.context_distance_threshold and 
                len(current_group) < self.cfg.max_group_size):
                current_group.append(det)
            else:
                groups.append(TranslationGroup(current_group))
                current_group = [det]
        
        if current_group:
            groups.append(TranslationGroup(current_group))
        
        return groups
    
    def should_skip_translation(self, text: str) -> bool:
        if not text:
            return True
        text = text.strip()
        if self.cfg.skip_numeric_only:
            if all(c.isdigit() or c.isspace() or c in '.,;:' for c in text):
                return True
        if self.cfg.skip_single_char and len(text) == 1:
            return True
        if self.cfg.skip_if_no_letters:
            if not any(c.isalpha() for c in text):
                return True
        return False
    
    def translate(self, text: str) -> str:
        if not text or self.should_skip_translation(text):
            return text
        
        # ★ NETTOYAGE OCR ★
        source_text = clean_ocr_text(text.strip())

        # Translation Memory des noms
        name_key = self._name_key(source_text)
        if name_key and name_key in self.name_memory:
            return self.name_memory[name_key]

        # Glossaire forcé
        forced_map = getattr(self.cfg, 'forced_translations', {}) or {}
        forced_key = self._normalize_text_key(source_text)
        forced_translation = forced_map.get(forced_key)
        if forced_translation:
            return forced_translation

        # Noms propres simples (ex: "Miso.") : conserver tel quel
        if self._is_single_proper_name(source_text):
            return source_text

        translation_input = self._normalize_case_for_translation(source_text)
        source_lang_code = self._detect_source_language(source_text)

        if source_lang_code == self.cfg.target_lang:
            return source_text

        # Heuristique: noms propres/entités courtes en MAJUSCULES -> conserver
        # ex: "GHISLAIN PERDIUM."
        source_words = re.findall(r"[A-Z][A-Z'\-]+", source_text.upper())
        word_count = len(source_text.split())
        if source_words and len(source_words) <= 3 and 2 <= word_count <= 4:
            alpha_ratio = sum(c.isalpha() for c in source_text) / max(1, len(source_text))
            if (
                alpha_ratio > 0.65
                and source_text.upper() == source_text
                and self._looks_like_uppercase_name_sequence(source_text)
            ):
                if name_key:
                    self.name_memory[name_key] = source_text
                    self._save_name_memory()
                return source_text
        
        # Cache (sur texte nettoyé)
        if self.cache:
            cached = self.cache.get(source_text, source_lang_code, self.cfg.target_lang)
            if cached:
                # Ne pas garder un cache "identique à la source" pour une vraie traduction.
                # Permet de corriger les anciennes mauvaises sorties (ex: "OH THERE" -> "OH THERE").
                if (
                    source_lang_code != self.cfg.target_lang
                    and cached.strip().lower() == source_text.strip().lower()
                    and any(c.isalpha() for c in source_text)
                ):
                    cached = None
                else:
                    return cached
        
        try:
            if self.backend == 'local_llm':
                translation = self._translate_with_local_llm(translation_input, source_lang_code)
            else:
                src_lang = self.cfg.lang_codes.get(source_lang_code, 'eng_Latn')
                tgt_lang = self.cfg.lang_codes.get(self.cfg.target_lang, 'fra_Latn')

                self.tokenizer.src_lang = src_lang
                inputs = self.tokenizer(
                    translation_input, return_tensors="pt", padding=True,
                    truncation=True, max_length=self.cfg.max_length
                )

                # ✅ FIX GPU: Mettre inputs sur le même device que le modèle
                if self.device == 'cuda' and torch.cuda.is_available():
                    inputs = {k: v.to('cuda') for k, v in inputs.items()}

                forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)

                with torch.no_grad():
                    generated = self.model.generate(
                        **inputs,
                        max_length=self.cfg.max_length,
                        num_beams=self.cfg.num_beams,
                        early_stopping=self.cfg.early_stopping,
                        forced_bos_token_id=forced_bos_token_id
                    )

                translation = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

            # Règle de reformulation (out_text rhétorique)
            # "YOU'RE TELLING ME THAT I, THE ..." -> "Vous me dites que moi, ..."
            m = re.match(r"^YOU'?RE\s+TELLING\s+ME\s+THAT\s+I[,\.]\s+(.+)$", source_text.upper())
            if m and self.cfg.target_lang == 'fr':
                tail = m.group(1).strip()
                tail_translation = self.translate(tail)
                tail_translation = tail_translation[0].lower() + tail_translation[1:] if tail_translation else tail_translation
                if tail_translation:
                    punct = '.' if source_text.strip().endswith('.') else ''
                    translation = f"Vous me dites que moi, {tail_translation.rstrip('.!?')}{punct}"

            # Garde-fou: éviter une traduction anormalement courte
            # (souvent une hallucination/résumé sur phrases OCR longues)
            src_len = len(source_text)
            tr_len = len(translation)
            if src_len >= 24 and tr_len < max(12, int(src_len * 0.40)):
                return source_text
            
            # ✅ NOUVEAU: Post-traitement pour corriger traductions bizarres
            # "Miso." → "C'est le Miso." est faux, remettre en "Miso."
            if source_text.lower().strip() == "miso." and translation.lower().startswith("c'est le"):
                translation = "Miso."

            if self.cfg.target_lang == 'fr':
                translation = self._post_process_french(translation)
            
            if self.cache:
                self.cache.set(source_text, translation, source_lang_code, self.cfg.target_lang)

            if name_key:
                self.name_memory[name_key] = translation
                self._save_name_memory()
            
            return translation
            
        except Exception as e:
            print(f"⚠️ Erreur traduction: {e}")
            return text
    
    def translate_group(self, group: TranslationGroup) -> str:
        texts = [d.text_original for d in group.detections if d.text_original]
        if not texts:
            return ""
        
        if len(texts) == 1:
            translation = self.translate(texts[0])
            for det in group.detections:
                if det.text_original:
                    det.text_translated = translation
            group.translation = translation
            return translation
        
        combined = self.cfg.group_separator.join(texts)
        group.combined_text = combined
        translation = self.translate(combined)
        group.translation = translation
        
        translated_parts = translation.split(self.cfg.group_separator)
        if len(translated_parts) != len(texts):
            translated_parts = [self.translate(t) for t in texts]
        
        dets_with_text = [d for d in group.detections if d.text_original]
        for det, trans in zip(dets_with_text, translated_parts):
            det.text_translated = trans.strip()
        
        return translation
    
    
    def translate_batch(self, texts: List[str]) -> List[str]:
        """
        ✅ SIMPLIFIÉ: Traduction INDIVIDUELLE robuste
        
        La traduction par batch crée trop de problèmes :
        - Séparation échoue souvent
        - Force des traductions bizarres (ex: "Miso" → "C'est le Miso")
        - Plus lent que traduction individuelle
        
        On traduit simplement bulle par bulle.
        Le cache réutilise les traductions = pas de perte de contexte.
        
        Args:
            texts: Liste des textes à traduire
            
        Returns:
            Liste des traductions
        """
        if not texts:
            return []
        
        results = []
        for i, text in enumerate(texts):
            trans = self.translate(text)
            results.append(trans)
            print(f"      [{i+1}/{len(texts)}] \"{text[:40]}...\" → \"{trans[:40]}...\"")
        
        return results
    
    def get_cache_stats(self) -> dict:
        if self.cache:
            return self.cache.get_stats()
        return {}
    
    def __del__(self):
        if self.cache:
            self.cache._save()
        self._save_name_memory()
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None