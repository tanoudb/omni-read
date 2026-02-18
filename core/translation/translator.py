from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import re
import json
import time

import torch

from config import config

from ..translator import NLLBTranslator as LegacyTranslator
from ..translator import clean_ocr_text
from .translator_nllb import NLLBCT2Translator


class NLLBTranslator:
    """
    Façade compatible pipeline:
    - qwen  : comportement legacy local LLM
    - nllb  : NLLB CT2 seul
    - hybrid: NLLB CT2 -> polish Qwen
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.cfg = config.translation
        self.mode = str(getattr(self.cfg, "translation_mode", "hybrid") or "hybrid").strip().lower()
        if self.mode not in {"hybrid", "nllb", "qwen"}:
            self.mode = "hybrid"

        self.last_page_system_prompt: str = ""
        self.last_page_user_prompt: str = ""
        self.last_nllb_inputs: List[str] = []
        self.last_nllb_outputs: List[str] = []
        self._generation_seconds_total: float = 0.0

        self.qwen: Optional[LegacyTranslator] = None
        self.nllb: Optional[NLLBCT2Translator] = None
        self.nllb_ready = False

        if self.mode in {"hybrid", "qwen"}:
            self.qwen = self._init_qwen_translator()

        if self.mode in {"hybrid", "nllb"}:
            self.nllb = NLLBCT2Translator(device=device)
            try:
                self.nllb.load_model()
                self.nllb_ready = True
                print(f"✅ NLLB CT2 prêt ({self.cfg.nllb_ct2_model_dir})")
            except Exception as exc:
                self.nllb_ready = False
                print(f"⚠️  NLLB CT2 indisponible: {exc}")

        if self.mode == "nllb" and not self.nllb_ready:
            print("⚠️  Mode nllb demandé mais CT2 indisponible. Fallback qwen.")
            self.mode = "qwen"
            if self.qwen is None:
                self.qwen = self._init_qwen_translator()

        if self.mode == "hybrid" and (not self.nllb_ready):
            print("⚠️  Mode hybrid: NLLB indisponible, fallback qwen uniquement.")

    def _init_qwen_translator(self) -> LegacyTranslator:
        original_backend = self.cfg.backend
        try:
            self.cfg.backend = "local_llm"
            translator = LegacyTranslator(device=self.device)
            return translator
        finally:
            self.cfg.backend = original_backend

    def _heuristic_detect_lang(self, text: str) -> Tuple[str, float]:
        if not text:
            return self.cfg.source_lang, 0.4
        if re.search(r"[\uAC00-\uD7AF]", text):
            return "ko", 0.99
        if re.search(r"[\u3040-\u30FF]", text):
            return "ja", 0.99
        if re.search(r"[\u4E00-\u9FFF]", text):
            return "zh", 0.99
        return "en", 0.6

    def detect_source_language_with_confidence(self, text: str) -> Tuple[str, float]:
        if self.qwen is not None:
            return self.qwen.detect_source_language_with_confidence(text)
        return self._heuristic_detect_lang(text)

    def detect_source_language(self, text: str) -> str:
        return self.detect_source_language_with_confidence(text)[0]

    def _translate_with_nllb_map(self, texts: List[str]) -> Dict[str, str]:
        if not texts:
            return {}

        if not self.nllb_ready or self.nllb is None:
            if self.qwen is not None:
                return self.qwen.translate_page_json(texts)
            return {str(i): txt for i, txt in enumerate(texts)}

        src_lang = self.cfg.lang_codes.get(self.cfg.source_lang, "eng_Latn")
        tgt_lang = self.cfg.lang_codes.get(self.cfg.target_lang, "fra_Latn")

        translated = self.nllb.translate_batch(texts, src_lang=src_lang, tgt_lang=tgt_lang)
        self.last_nllb_inputs = list(texts)
        self.last_nllb_outputs = list(translated)
        return {str(i): translated[i] if i < len(translated) else texts[i] for i in range(len(texts))}

    def _polish_with_qwen(self, nllb_texts: List[str]) -> Dict[str, str]:
        if not nllb_texts:
            return {}
        if self.qwen is None:
            return {str(i): txt for i, txt in enumerate(nllb_texts)}

        tokenizer = self.qwen.tokenizer
        model = self.qwen.model
        if tokenizer is None or model is None:
            return {str(i): txt for i, txt in enumerate(nllb_texts)}

        system_prompt = getattr(self.cfg, "llm_polish_system_prompt", "")
        payload_obj = {str(idx): txt for idx, txt in enumerate(nllb_texts)}
        user_prompt = (
            "POLISH_JSON_INPUT:\n"
            f"{json.dumps(payload_obj, ensure_ascii=False)}\n\n"
            "Retourne UNIQUEMENT un JSON avec exactement les mêmes clés."
        )

        self.last_page_system_prompt = system_prompt
        self.last_page_user_prompt = user_prompt

        if hasattr(tokenizer, 'apply_chat_template'):
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ]
            inputs = tokenizer.apply_chat_template(
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
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=self.cfg.max_length)

        if self.device == 'cuda' and torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}

        generate_kwargs = {
            'max_new_tokens': int(getattr(self.cfg, 'llm_max_new_tokens', 512)),
            'repetition_penalty': float(getattr(self.cfg, 'llm_repetition_penalty', 1.05)),
            'pad_token_id': tokenizer.eos_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'do_sample': False,
        }

        t0 = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(**inputs, **generate_kwargs)
        self._generation_seconds_total += max(0.0, time.perf_counter() - t0)

        input_token_count = int(inputs['input_ids'].shape[-1])
        raw = tokenizer.decode(generated[0][input_token_count:], skip_special_tokens=True).strip()

        mapped = self.qwen._parse_page_translation_output(raw, nllb_texts)

        def _is_suspicious_polish(base: str, polished: str) -> bool:
            txt = (polished or "").strip()
            if not txt:
                return True

            low = txt.lower()
            if low in {"version polie", "version polie.", "texte final", "texte"}:
                return True

            if re.search(r"(^|\n)\s*\d+\s*[\.:\-\)]\s+", txt):
                return True

            if "{" in txt or "}" in txt or '"0"' in txt:
                return True

            base_len = max(1, len((base or "").strip()))
            txt_len = len(txt)
            if txt_len > max(260, int(base_len * 2.2)):
                return True

            if txt_len < max(2, int(base_len * 0.25)) and base_len >= 20:
                return True

            return False

        final = {}
        for i, base in enumerate(nllb_texts):
            polished = str(mapped.get(str(i), "") or "").strip()
            if _is_suspicious_polish(base, polished):
                print(f"[HYBRID] polish rejeté idx={i}, fallback NLLB")
                final[str(i)] = base
            else:
                final[str(i)] = polished

            preview_base = (base or "").replace("\n", " ").strip()
            preview_polish = (final[str(i)] or "").replace("\n", " ").strip()
            if len(preview_base) > 120:
                preview_base = preview_base[:120] + "..."
            if len(preview_polish) > 120:
                preview_polish = preview_polish[:120] + "..."
            print(f"[HYBRID] NLLB[{i}] {preview_base}")
            print(f"[HYBRID] QWEN[{i}] {preview_polish}")
        return final

    def get_last_page_payload_debug(self) -> dict:
        if self.mode == "qwen" and self.qwen is not None:
            return self.qwen.get_last_page_payload_debug()

        return {
            'system_prompt': self.last_page_system_prompt,
            'user_prompt': self.last_page_user_prompt,
            'nllb_inputs': list(self.last_nllb_inputs),
            'nllb_outputs': list(self.last_nllb_outputs),
            'mode': self.mode,
        }

    def translate_page_json(self, texts: List[str]) -> dict:
        if not texts:
            return {}

        clean_inputs = [clean_ocr_text((text or '').strip()) for text in texts]

        if self.mode == "qwen":
            if self.qwen is None:
                return {str(i): t for i, t in enumerate(clean_inputs)}
            return self.qwen.translate_page_json(clean_inputs)

        if self.mode == "nllb":
            return self._translate_with_nllb_map(clean_inputs)

        nllb_map = self._translate_with_nllb_map(clean_inputs)
        nllb_texts = [nllb_map.get(str(i), clean_inputs[i]) for i in range(len(clean_inputs))]

        try:
            polished_map = self._polish_with_qwen(nllb_texts)
        except Exception as exc:
            print(f"⚠️  Polish Qwen échoué, fallback NLLB: {exc}")
            polished_map = {str(i): nllb_texts[i] for i in range(len(nllb_texts))}

        merged = {}
        for i, original in enumerate(clean_inputs):
            base = nllb_map.get(str(i), original)
            polish = str(polished_map.get(str(i), "") or "").strip()
            merged[str(i)] = polish if polish else base
        return merged

    def translate(self, text: str) -> str:
        if not text:
            return text
        mapped = self.translate_page_json([text])
        return mapped.get("0", text)

    def translate_batch(self, texts: List[str]) -> List[str]:
        mapped = self.translate_page_json(texts)
        return [mapped.get(str(i), texts[i]) for i in range(len(texts))]

    def get_generation_seconds_total(self) -> float:
        total = self._generation_seconds_total
        if self.qwen is not None:
            total += float(getattr(self.qwen, 'get_generation_seconds_total', lambda: 0.0)())
        return float(total)

    def get_cache_stats(self) -> dict:
        data = {}
        if self.qwen is not None:
            data['qwen'] = self.qwen.get_cache_stats()
        if self.nllb is not None:
            data['nllb'] = self.nllb.get_cache_stats()

        # Compat pipeline legacy: attend {'entries': ..., 'hit_rate': ...}
        def _to_int(value, default=0):
            try:
                return int(value)
            except Exception:
                return default

        def _to_percent(value) -> float:
            if value is None:
                return 0.0
            if isinstance(value, (int, float)):
                return float(value)
            txt = str(value).strip().replace('%', '')
            try:
                return float(txt)
            except Exception:
                return 0.0

        qwen_stats = data.get('qwen', {}) if isinstance(data.get('qwen', {}), dict) else {}
        nllb_stats = data.get('nllb', {}) if isinstance(data.get('nllb', {}), dict) else {}

        entries = _to_int(qwen_stats.get('entries', 0), 0) + _to_int(nllb_stats.get('entries', 0), 0)
        hit_rate = max(_to_percent(qwen_stats.get('hit_rate', 0.0)), _to_percent(nllb_stats.get('hit_rate', 0.0)))

        return {
            'entries': entries,
            'hit_rate': f"{hit_rate:.1f}%",
            'qwen': qwen_stats,
            'nllb': nllb_stats,
        }

    def __del__(self):
        try:
            if self.qwen is not None:
                del self.qwen
        except Exception:
            pass
        try:
            if self.nllb is not None:
                del self.nllb
        except Exception:
            pass
