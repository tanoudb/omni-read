from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import re
import json
import time

import torch

from config import config

from ..translator import NLLBTranslator as LegacyTranslator
from ..translator import clean_ocr_text
from utils.prompts import (
    PAGE_QWEN_SYSTEM,
    PAGE_HYBRID_QUALITY_SYSTEM,
    POLISH_DEFAULT_SYSTEM,
    format_json_payload,
    format_polish_user_payload,
)
from .translator_nllb import NLLBCT2Translator


class NLLBTranslator:
    """
    Façade compatible pipeline:
    - qwen  : comportement legacy local LLM
    - nllb  : NLLB CT2 seul
    - hybrid: NLLB CT2 -> polish Qwen
    """

    _SFX_TOKENS = {
        "AH", "AAH", "WAAH", "WAAAH", "UGH", "URGH", "ARGH", "ERGH", "KRGH", "KHOFF",
        "HUFF", "PANT", "GASP", "SOB", "SNIFF", "HMPH", "GRR", "BAM", "BOOM", "CRASH",
        "BANG", "THUD", "SNAP", "TAP", "CLAP", "WHAM", "WHOOSH",
    }

    _TOKENIZATION_LEXICON = {
        "I", "A", "AN", "THE", "THIS", "THAT", "THERE", "HERE", "YOU", "YOUR", "YOURS", "ME", "MY",
        "MINE", "WE", "OUR", "OURS", "HE", "HIS", "SHE", "HER", "HERS", "IT", "ITS", "THEY", "THEM",
        "THEIR", "THEIRS", "IS", "ARE", "WAS", "WERE", "BE", "BEEN", "BEING", "DO", "DID", "DONE", "DOES",
        "HAVE", "HAS", "HAD", "WILL", "WOULD", "CAN", "COULD", "SHALL", "SHOULD", "MAY", "MIGHT", "MUST",
        "NOT", "NO", "YES", "TO", "OF", "IN", "ON", "AT", "FOR", "FROM", "WITH", "WITHOUT", "BY", "AS",
        "AND", "OR", "BUT", "SO", "IF", "THEN", "WHEN", "WHERE", "WHY", "HOW", "WHAT", "WHO", "WHOM",
        "WHICH", "FIRST", "SECOND", "THIRD", "LOOK", "WAIT", "HELP", "PLEASE", "SORRY", "THANK", "THANKS",
        "NOW", "LATER", "AGAIN", "NEVER", "ALWAYS", "SOMETHING", "NOTHING", "EVERYTHING", "SOMEONE", "ANYONE",
        "EVERYONE", "WANT", "WANTED", "NEED", "NEEDED", "KNOW", "KNEW", "THINK", "THOUGHT", "GO", "GOING",
        "COME", "CAME", "TAKE", "TOOK", "GIVE", "GAVE", "MAKE", "MADE", "GET", "GOT", "FIND", "FOUND",
        "LIKE", "LOVE", "HATE", "GOOD", "BAD", "BEST", "WORST", "TIME", "DAY", "NIGHT", "MAN", "WOMAN",
        "BOY", "GIRL", "SON", "DAUGHTER", "FATHER", "MOTHER", "BROTHER", "SISTER", "FRIEND", "ENEMY",
    }

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.cfg = config.translation
        self.mode = str(getattr(self.cfg, "translation_mode", "hybrid") or "hybrid").strip().lower()
        if self.mode not in {"hybrid", "nllb", "qwen", "hybrid_quality"}:
            self.mode = "hybrid"

        self.last_page_system_prompt: str = ""
        self.last_page_user_prompt: str = ""
        self.last_nllb_inputs: List[str] = []
        self.last_nllb_outputs: List[str] = []
        self._generation_seconds_total: float = 0.0
        self._glossaire: list[str] = []

        self.qwen: Optional[LegacyTranslator] = None
        self.nllb: Optional[NLLBCT2Translator] = None
        self.nllb_ready = False

        if self.mode in {"hybrid", "qwen", "hybrid_quality"}:
            self.qwen = self._init_qwen_translator()

        if self.mode in {"hybrid", "nllb", "hybrid_quality"}:
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

        if self.mode in {"hybrid", "hybrid_quality"} and (not self.nllb_ready):
            print("⚠️  Mode hybrid: NLLB indisponible, fallback qwen uniquement.")

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        if not text:
            return False
        return bool(re.search(r"[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uAC00-\uD7AF]", text))

    @staticmethod
    def _normalize_for_compare(text: str) -> str:
        base = re.sub(r"\s+", " ", str(text or "")).strip().lower()
        return re.sub(r"[\W_]+", "", base)

    @classmethod
    def _looks_joined_ocr_block(cls, text: str) -> bool:
        value = str(text or "").strip()
        if not value or " " in value:
            return False
        if len(value) < 10:
            return False
        if not re.fullmatch(r"[A-Za-z']+", value):
            return False
        if cls._is_likely_sfx(value):
            return False
        letters = [c for c in value if c.isalpha()]
        if not letters:
            return False
        upper_ratio = sum(1 for c in letters if c.isupper()) / float(len(letters))
        return upper_ratio >= 0.8

    @classmethod
    def _split_joined_upper_token(cls, token: str) -> str:
        raw = str(token or "").strip()
        if not raw:
            return raw

        upper = raw.upper()
        n = len(upper)
        max_len = 16
        dp: List[Optional[Tuple[int, List[str]]]] = [None] * (n + 1)
        dp[0] = (0, [])

        for i in range(n):
            state = dp[i]
            if state is None:
                continue
            base_penalty, base_parts = state
            for j in range(min(n, i + max_len), i, -1):
                seg = upper[i:j]
                if seg in cls._TOKENIZATION_LEXICON:
                    score = base_penalty + max(0, 4 - len(seg))
                elif len(seg) <= 2:
                    continue
                elif re.fullmatch(r"[A-Z]{3,}", seg):
                    score = base_penalty + 8 + len(seg)
                else:
                    continue

                next_state = dp[j]
                if next_state is None or score < next_state[0]:
                    dp[j] = (score, base_parts + [seg])

        end_state = dp[n]
        if end_state is None:
            return raw

        parts = end_state[1]
        if len(parts) < 2:
            return raw

        joined = " ".join(parts).strip()
        if len(joined.replace(" ", "")) != len(upper):
            return raw
        return joined

    @classmethod
    def _pre_tokenize_ocr_text(cls, text: str) -> str:
        value = str(text or "").strip()
        if not cls._looks_joined_ocr_block(value):
            return value
        split = cls._split_joined_upper_token(value)
        return split if split else value

    def _adaptive_page_max_tokens(self, texts: List[str], fallback: int = 512) -> int:
        hard_cap = int(getattr(self.cfg, "llm_max_new_tokens", fallback))
        count = max(1, len(texts))
        total_chars = sum(len(str(t or "")) for t in texts)
        estimate = int(total_chars * 0.95)
        floor = 96 if count <= 2 else 160
        dynamic_cap = min(hard_cap, max(floor, estimate, count * 12))
        return int(dynamic_cap)

    def _aggressive_french_retry(self, source_text: str, fallback_text: str) -> str:
        src = str(source_text or "").strip()
        if not src:
            return fallback_text

        if self.qwen is not None:
            system_prompt = (
                "Tu traduis UNIQUEMENT en français naturel. "
                "Interdiction absolue de recopier l'anglais source sauf onomatopées (SFX) ou URLs. "
                "Conserve le sens et réponds uniquement avec la traduction finale."
            )
            user_prompt = format_json_payload({"0": src})
            mapped = self._qwen_generate_json_map(system_prompt, user_prompt, [src])
            candidate = str(mapped.get("0", "") or "").strip()
            if candidate and self._normalize_for_compare(candidate) != self._normalize_for_compare(src):
                return candidate

        if self.nllb_ready and self.nllb is not None:
            try:
                out = self.nllb.translate_batch([src], src_lang="eng_Latn", tgt_lang="fra_Latn")
                candidate = str(out[0] if out else "").strip()
                if candidate and self._normalize_for_compare(candidate) != self._normalize_for_compare(src):
                    return candidate
            except Exception:
                pass

        return fallback_text

    @classmethod
    def _is_likely_sfx(cls, text: str) -> bool:
        if not text:
            return False
        tokens = re.findall(r"[A-Za-z]+", text.upper())
        if not tokens:
            return False
        return all(tok in cls._SFX_TOKENS or re.search(r"(.)\1{2,}", tok) for tok in tokens)

    @classmethod
    def _rule_based_override(cls, source_text: str) -> Optional[str]:
        src = (source_text or "").strip()
        if not src:
            return src
        upper = src.upper()

        if cls._is_likely_sfx(src):
            return src

        if re.fullmatch(r"\(NONE\)", upper):
            return "(none)"

        if re.fullmatch(r"HONEY[!?\.\s]*", upper):
            return "Chéri !"

        if re.fullmatch(r"BUT\s+DA+A+D[!?\.\s]*", upper):
            return "Mais papa !!!"

        if re.fullmatch(r"\.*\s*AGENT\s+101[!?\.\s]*", upper):
            return "…Agent 101."

        return None

    @classmethod
    def _should_repair_with_single_pass(cls, source_text: str, translated_text: str) -> bool:
        src = (source_text or "").strip()
        out = (translated_text or "").strip()
        if not src:
            return False
        if cls._is_likely_sfx(src):
            return False
        if cls._contains_cjk(out):
            return True
        if not out:
            return True

        src_norm = re.sub(r"\s+", " ", src).strip().lower()
        out_norm = re.sub(r"\s+", " ", out).strip().lower()
        src_words = re.findall(r"[A-Za-z']+", src)

        # Si la sortie recopie l'anglais source sur une phrase non triviale,
        # on tente une traduction unitaire plus fiable.
        if out_norm == src_norm and len(src_words) >= 3:
            return True
        return False

    def _repair_single_translation(self, source_text: str, current_text: str) -> str:
        if self.qwen is None:
            return current_text
        try:
            src_norm = re.sub(r"\s+", " ", (source_text or "").strip()).lower()
            attempts = 2
            best = str(current_text or "").strip()

            for _ in range(attempts):
                repaired = str(self.qwen.translate(source_text) or "").strip()
                if not repaired:
                    continue
                if self.cfg.target_lang == "fr" and self._contains_cjk(repaired):
                    continue
                rep_norm = re.sub(r"\s+", " ", repaired).lower()
                src_words = re.findall(r"[A-Za-z']+", source_text or "")
                if rep_norm == src_norm and len(src_words) >= 3:
                    best = repaired
                    continue
                return repaired

            return best if best else current_text
        except Exception:
            return current_text

    def _qwen_generate_json_map(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback_texts: List[str],
    ) -> Dict[str, str]:
        expected = len(fallback_texts)
        fallback = {str(i): fallback_texts[i] for i in range(expected)}

        if self.qwen is None:
            return fallback

        model = getattr(self.qwen, "model", None)
        tokenizer = getattr(self.qwen, "tokenizer", None)
        if model is None:
            return fallback

        self.last_page_system_prompt = system_prompt
        self.last_page_user_prompt = user_prompt

        raw = ""
        page_max_tokens = self._adaptive_page_max_tokens(fallback_texts, fallback=512)

        try:
            if getattr(self.qwen, "llm_backend", "transformers") == "gguf":
                t0 = time.perf_counter()
                response = model.create_chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=float(getattr(self.cfg, "llm_temperature", 0.0)),
                    top_p=float(getattr(self.cfg, "llm_top_p", 1.0)),
                    max_tokens=page_max_tokens,
                    repeat_penalty=float(getattr(self.cfg, "llm_repetition_penalty", 1.05)),
                )
                self._generation_seconds_total += max(0.0, time.perf_counter() - t0)
                raw = self.qwen._gguf_extract_content(response).strip()
            else:
                if tokenizer is None:
                    return fallback

                if hasattr(tokenizer, "apply_chat_template"):
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                    inputs = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    )
                    if isinstance(inputs, torch.Tensor):
                        input_ids = inputs
                        attention_mask = torch.ones_like(input_ids)
                        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
                else:
                    prompt = f"{system_prompt}\n\n{user_prompt}"
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.cfg.max_length)

                if self.device == "cuda" and torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}

                generate_kwargs = {
                    "max_new_tokens": page_max_tokens,
                    "repetition_penalty": float(getattr(self.cfg, "llm_repetition_penalty", 1.05)),
                    "pad_token_id": tokenizer.eos_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "do_sample": False,
                }

                t0 = time.perf_counter()
                with torch.no_grad():
                    generated = model.generate(**inputs, **generate_kwargs)
                self._generation_seconds_total += max(0.0, time.perf_counter() - t0)

                input_token_count = int(inputs["input_ids"].shape[-1])
                raw = tokenizer.decode(generated[0][input_token_count:], skip_special_tokens=True).strip()
        except Exception:
            return fallback

        try:
            extract_json = getattr(self.qwen, "_extract_json_candidate", None)
            json_candidate = extract_json(raw) if callable(extract_json) else (raw or "").strip()
            parsed = json.loads(json_candidate)
            if not isinstance(parsed, dict):
                return fallback

            clean_value = getattr(self.qwen, "_clean_llm_value", None)
            result: Dict[str, str] = {}
            for i in range(expected):
                key = str(i)
                value = parsed.get(key, fallback.get(key, ""))
                if isinstance(value, dict):
                    value = value.get("fr") or value.get("text") or value.get("translation") or ""
                if callable(clean_value):
                    txt = clean_value(value)
                else:
                    txt = str(value or "").strip()
                if not txt:
                    txt = fallback.get(key, "")
                if self.cfg.target_lang == "fr" and self._contains_cjk(txt):
                    txt = fallback.get(key, "")
                result[key] = txt
            return result
        except Exception:
            return fallback

    def translate_page_qwen(self, texts: List[str]) -> Dict[str, str]:
        if not texts:
            return {}
        clean_inputs = [clean_ocr_text((text or "").strip()) for text in texts]
        payload = {str(i): clean_inputs[i] for i in range(len(clean_inputs))}

        system_prompt = PAGE_QWEN_SYSTEM
        user_prompt = format_json_payload(payload)

        mapped = self._qwen_generate_json_map(system_prompt, user_prompt, clean_inputs)
        final: Dict[str, str] = {}
        for i, src in enumerate(clean_inputs):
            key = str(i)
            forced = self._rule_based_override(src)
            candidate = forced if forced is not None else mapped.get(key, src)
            if forced is None and self._should_repair_with_single_pass(src, candidate):
                candidate = self._repair_single_translation(src, candidate)
            final[key] = candidate
        return final

    def translate_page_hybrid_quality(self, texts: List[str]) -> Dict[str, str]:
        if not texts:
            return {}

        clean_inputs = [clean_ocr_text((text or "").strip()) for text in texts]
        nllb_map = self._translate_with_nllb_map(clean_inputs)
        nllb_results = [nllb_map.get(str(i), clean_inputs[i]) for i in range(len(clean_inputs))]
        for i, src in enumerate(clean_inputs):
            forced = self._rule_based_override(src)
            if forced is not None:
                nllb_results[i] = forced

        if self.qwen is None:
            return {str(i): nllb_results[i] for i in range(len(nllb_results))}

        payload = {
            str(i): {
                "en": clean_inputs[i],
                "fr": nllb_results[i],
            }
            for i in range(len(clean_inputs))
        }

        system_prompt = PAGE_HYBRID_QUALITY_SYSTEM
        user_prompt = format_json_payload(payload)

        corrected = self._qwen_generate_json_map(system_prompt, user_prompt, nllb_results)
        final: Dict[str, str] = {}
        for i, src in enumerate(clean_inputs):
            key = str(i)
            forced = self._rule_based_override(src)
            if forced is not None:
                final[key] = forced
            else:
                final[key] = corrected.get(key, nllb_results[i])
        return final

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

    def set_glossaire(self, termes: list[str]):
        self._glossaire = [t.strip() for t in termes if t.strip()]

    def _masquer_termes(self, texte: str) -> tuple[str, dict]:
        """Remplace les termes du glossaire par des tokens neutres.
        Retourne le texte modifié et le dictionnaire de restauration.
        Trie les termes par longueur décroissante pour éviter les remplacements partiels."""
        tokens = {}
        termes_tries = sorted(self._glossaire, key=len, reverse=True)
        for i, terme in enumerate(termes_tries):
            token = f"__GTERM{i}__"
            pattern = re.compile(re.escape(terme), re.IGNORECASE)
            if pattern.search(texte):
                tokens[token] = terme
                texte = pattern.sub(token, texte)
        return texte, tokens

    def _restaurer_termes(self, texte: str, tokens: dict) -> str:
        """Restaure les tokens en termes originaux."""
        for token, terme in tokens.items():
            texte = texte.replace(token, terme)
        return texte

    def _polish_with_qwen(self, nllb_texts: List[str]) -> Dict[str, str]:
        if not nllb_texts:
            return {}
        if self.qwen is None:
            return {str(i): txt for i, txt in enumerate(nllb_texts)}

        tokenizer = self.qwen.tokenizer
        model = self.qwen.model
        if tokenizer is None or model is None:
            return {str(i): txt for i, txt in enumerate(nllb_texts)}

        system_prompt = getattr(self.cfg, "llm_polish_system_prompt", POLISH_DEFAULT_SYSTEM)
        if self._glossaire:
            termes_str = ", ".join(self._glossaire)
            system_prompt = (f"Termes à NE PAS modifier (noms propres, titres, attaques) : {termes_str}\n\n" + system_prompt)
        payload_obj = {str(idx): txt for idx, txt in enumerate(nllb_texts)}
        user_prompt = format_polish_user_payload(payload_obj)

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
        clean_inputs = [self._pre_tokenize_ocr_text(text) for text in clean_inputs]

        if self.mode == "qwen":
            return self.translate_page_qwen(clean_inputs)

        if self.mode == "hybrid_quality":
            return self.translate_page_hybrid_quality(clean_inputs)

        # Masquer les termes du glossaire
        tous_tokens = {}
        textes_masques = []
        for texte in clean_inputs:
            texte_masque, tokens = self._masquer_termes(texte)
            textes_masques.append(texte_masque)
            tous_tokens.update(tokens)
        clean_inputs = textes_masques

        if self.mode == "nllb":
            nllb_map = self._translate_with_nllb_map(clean_inputs)
            return {
                str(i): self._restaurer_termes(nllb_map.get(str(i), clean_inputs[i]), tous_tokens)
                for i in range(len(clean_inputs))
            }

        nllb_map = self._translate_with_nllb_map(clean_inputs)
        nllb_texts = [nllb_map.get(str(i), clean_inputs[i]) for i in range(len(clean_inputs))]

        # Restaurer les termes dans les sorties NLLB
        nllb_texts = [self._restaurer_termes(t, tous_tokens) for t in nllb_texts]

        try:
            polished_map = self._polish_with_qwen(nllb_texts)
        except Exception as exc:
            print(f"⚠️  Polish Qwen échoué, fallback NLLB: {exc}")
            polished_map = {str(i): nllb_texts[i] for i in range(len(nllb_texts))}

        merged = {}
        for i, original in enumerate(clean_inputs):
            base = nllb_texts[i] if i < len(nllb_texts) else self._restaurer_termes(nllb_map.get(str(i), original), tous_tokens)
            polish = str(polished_map.get(str(i), "") or "").strip()
            merged[str(i)] = polish if polish else base

        if str(getattr(self.cfg, "target_lang", "fr")).lower() == "fr":
            for i, src in enumerate(clean_inputs):
                key = str(i)
                out = str(merged.get(key, src) or "").strip()
                if self._is_likely_sfx(src):
                    continue
                if self._normalize_for_compare(src) and self._normalize_for_compare(src) == self._normalize_for_compare(out):
                    print(f"[FR_RETRY_ALERT] bloc={i} sortie identique à la source, seconde passe agressive")
                    merged[key] = self._aggressive_french_retry(src, out)
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
