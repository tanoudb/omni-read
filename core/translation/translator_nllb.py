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
        self.runtime_device: str = "unknown"

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
            self.runtime_device = device
            self.available = True
            print(f"[NLLB] Runtime backend: ct2 | device={self.runtime_device}")
        except Exception as exc:
            self._load_hf_fallback(reason=f"Échec chargement CT2: {exc}")

    def _hf_model_device(self) -> str:
        if self.model is None:
            return "unknown"
        try:
            if hasattr(self.model, "device") and self.model.device is not None:
                return str(self.model.device)
            first_param = next(self.model.parameters(), None)
            if first_param is not None:
                return str(first_param.device)
        except Exception:
            pass
        return "unknown"

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

        cuda_requested = self.device == "cuda"
        cuda_available = torch.cuda.is_available()

        if cuda_requested and cuda_available:
            try:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    cache_dir=str(config.TRANSLATION_CACHE_DIR),
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="cuda",
                    low_cpu_mem_usage=True,
                )
                print("[NLLB] HF fallback loaded on CUDA fp16 (device_map=cuda)")
            except Exception as fp16_exc:
                print(f"[NLLB] HF CUDA fp16 load failed: {fp16_exc}")
                try:
                    from transformers import BitsAndBytesConfig

                    qcfg = BitsAndBytesConfig(load_in_8bit=True)
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name,
                        cache_dir=str(config.TRANSLATION_CACHE_DIR),
                        trust_remote_code=True,
                        quantization_config=qcfg,
                        device_map="cuda",
                        low_cpu_mem_usage=True,
                    )
                    print("[NLLB] HF fallback loaded on CUDA 8-bit (device_map=cuda)")
                except Exception as int8_exc:
                    raise RuntimeError(
                        "NLLB HF fallback CUDA load failed (fp16 and 8-bit). "
                        f"fp16={fp16_exc} | int8={int8_exc}"
                    )
        else:
            if cuda_requested and not cuda_available:
                print("[NLLB] CUDA indisponible: fallback HF sur CPU")
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
        self.runtime_device = self._hf_model_device()
        self.available = True
        print(f"[NLLB] Runtime backend: hf | device={self.runtime_device}")

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
            target_device = self._hf_model_device()
            if target_device.startswith("cuda") and torch.cuda.is_available():
                inputs = {k: v.to(target_device) for k, v in inputs.items()}

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
