from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import re
import torch

from config import config
from utils import CacheManager


class NLLBCT2Translator:
    """Traducteur NLLB-200 via CTranslate2 (batch, cache, fallback safe)."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.cfg = config.translation
        self.model = None
        self.tokenizer = None
        self.available = False
        self.backend_type = "none"  # ct2 | hf

        self.cache: Optional[CacheManager] = None
        if self.cfg.enable_cache:
            cache_file = config.TRANSLATION_CACHE_DIR / self.cfg.nllb_cache_file
            self.cache = CacheManager(cache_file, max_size_mb=config.performance.cache_max_size_mb)

    def load_model(self) -> None:
        if self.available:
            return

        try:
            import ctranslate2
            from transformers import AutoTokenizer
        except Exception as exc:
            self._load_hf_fallback(reason=f"CT2 indisponible: {exc}")
            return

        model_dir = Path(self.cfg.nllb_ct2_model_dir)
        if not model_dir.exists() or not model_dir.is_dir():
            self._load_hf_fallback(reason="Modèle CT2 introuvable")
            return

        tokenizer_source = model_dir if (model_dir / "tokenizer.json").exists() else self.cfg.nllb_source_model

        device = "cuda" if (self.device == "cuda") else "cpu"
        compute_type = str(getattr(self.cfg, "nllb_ct2_compute_type", "int8") or "int8").strip().lower()

        try:
            self.model = ctranslate2.Translator(
                str(model_dir),
                device=device,
                compute_type=compute_type,
                inter_threads=1,
                intra_threads=0,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                cache_dir=str(config.TRANSLATION_CACHE_DIR),
                use_fast=True,
                trust_remote_code=True,
            )

            self.backend_type = "ct2"
            self.available = True
        except Exception as exc:
            self._load_hf_fallback(reason=f"Échec chargement CT2: {exc}")

    def _load_hf_fallback(self, reason: str) -> None:
        from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

        model_name = str(getattr(self.cfg, "model_name", "facebook/nllb-200-3.3B"))
        print(f"[NLLB] CT2 fallback -> HF ({reason})")
        print(f"[NLLB] Loading HF fallback model: {model_name}")

        self.tokenizer = NllbTokenizer.from_pretrained(
            model_name,
            cache_dir=str(config.TRANSLATION_CACHE_DIR),
            trust_remote_code=True,
        )

        loaded = False
        gpu_attempt_failed = False
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig

                qcfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True,
                )
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    cache_dir=str(config.TRANSLATION_CACHE_DIR),
                    trust_remote_code=True,
                    quantization_config=qcfg,
                    device_map="auto",
                )
                loaded = True
                print("[NLLB] HF fallback loaded in 4-bit (bitsandbytes)")
            except Exception as exc:
                gpu_attempt_failed = True
                print(f"[NLLB] 4-bit load failed: {exc}")

        if not loaded and not gpu_attempt_failed:
            try:
                dtype = torch.float16 if (self.device == "cuda" and torch.cuda.is_available()) else torch.float32
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    cache_dir=str(config.TRANSLATION_CACHE_DIR),
                    trust_remote_code=True,
                    torch_dtype=dtype,
                )
                if self.device == "cuda" and torch.cuda.is_available():
                    self.model = self.model.to("cuda")
                else:
                    self.model = self.model.to("cpu")
                loaded = True
                print(f"[NLLB] HF fallback loaded in dtype={dtype}")
            except Exception as exc:
                print(f"[NLLB] HF GPU/auto load failed: {exc}")

        if not loaded:
            print("[NLLB] Falling back to CPU load for stability")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                cache_dir=str(config.TRANSLATION_CACHE_DIR),
                trust_remote_code=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            self.model = self.model.to("cpu")
            print("[NLLB] HF fallback loaded on CPU fp32")

        self.backend_type = "hf"
        self.available = True

    @staticmethod
    def _clean_translation(text: str) -> str:
        value = (text or "").strip()
        value = re.sub(r"\s+", " ", value).strip()
        return value

    def _translate_batch_uncached(self, texts: List[str], src_lang: str, tgt_lang: str) -> List[str]:
        assert self.model is not None and self.tokenizer is not None

        if self.backend_type == "hf":
            self.tokenizer.src_lang = src_lang
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(getattr(self.cfg, "max_length", 640)),
            )
            if self.device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=int(getattr(self.cfg, "max_length", 640)),
                    num_beams=int(getattr(self.cfg, "num_beams", 4)),
                    early_stopping=bool(getattr(self.cfg, "early_stopping", True)),
                )
            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            return [self._clean_translation(x) for x in decoded]

        max_len = int(getattr(self.cfg, "nllb_max_decoding_length", 320))
        beam_size = int(getattr(self.cfg, "nllb_beam_size", 4))

        source_tokens = []
        for text in texts:
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            body_tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
            src = [src_lang] + body_tokens + ["</s>"]
            source_tokens.append(src)

        target_prefix = [[tgt_lang] for _ in source_tokens]
        results = self.model.translate_batch(
            source_tokens,
            target_prefix=target_prefix,
            beam_size=beam_size,
            max_decoding_length=max_len,
            return_scores=False,
        )

        translations: List[str] = []
        for result in results:
            hypo_tokens = list(result.hypotheses[0]) if getattr(result, "hypotheses", None) else []
            filtered_tokens = [tok for tok in hypo_tokens if tok not in {tgt_lang, src_lang, "</s>", "<s>"}]
            token_ids = self.tokenizer.convert_tokens_to_ids(filtered_tokens)
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            translations.append(self._clean_translation(text))
        return translations

    def translate_batch(self, texts: List[str], src_lang: str = "eng_Latn", tgt_lang: str = "fra_Latn") -> List[str]:
        if not texts:
            return []

        self.load_model()

        out = [""] * len(texts)
        to_compute = []
        to_compute_indices = []

        for i, text in enumerate(texts):
            src_text = (text or "").strip()
            if not src_text:
                out[i] = src_text
                continue

            if self.cache is not None:
                cached = self.cache.get(src_text, src_lang, tgt_lang)
                if cached:
                    out[i] = cached
                    continue

            to_compute.append(src_text)
            to_compute_indices.append(i)

        if to_compute:
            computed = self._translate_batch_uncached(to_compute, src_lang=src_lang, tgt_lang=tgt_lang)
            for idx, translated in zip(to_compute_indices, computed):
                src_text = texts[idx]
                value = translated or src_text
                out[idx] = value
                if self.cache is not None:
                    self.cache.set(src_text, value, src_lang, tgt_lang)

        for i, original in enumerate(texts):
            if not out[i]:
                out[i] = original
        return out

    def get_cache_stats(self) -> dict:
        if self.cache is None:
            return {}
        return self.cache.get_stats()

    def __del__(self):
        if self.cache is not None:
            try:
                self.cache._save()
            except Exception:
                pass
