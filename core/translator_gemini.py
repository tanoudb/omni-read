"""
GeminiTranslator — Drop-in pour NLLBTranslator.
SDK: google-genai (nouveau). Modèle: gemini-2.5-flash (free tier).
1 requête par page (toutes bulles d'un coup), glossaire auto, mémoire, cache.
"""

import json, os, re, time, hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from utils.gemini_prompt import FONT_MAP, PromptBank

STATE_DIR = Path("data/gemini_state")
CACHE_FILE = Path("cache/gemini_cache.json")

SYSTEM = (
    "Tu es un expert en LOCALISATION de manhwa EN→FR.\n"
    "RÈGLES DE FER :\n"
    "0. FRANÇAIS UNIQUEMENT. Toute trace d'anglais = erreur fatale.\n"
    "1. Jamais de mots hybrides inventés. Vrais mots français uniquement.\n"
    "2. SENS > MOT-À-MOT : 'I WILL.'→'C'est promis.' | 'HOLD STILL'→'Ne bouge pas' | "
    "'HONEY!'→'Chéri(e) !' | 'DAD!!'→'Papa !!' | 'CHILD ABUSE'→'maltraitance'\n"
    "3. Mots collés OCR (ABOUTOUR) → sépare mentalement et traduis.\n"
    "4. SFX/onomatopées pures (KRGH, WAAAH, BOOM, GAAAAH) → recopie tels quels.\n"
    "5. Noms propres → recopie tels quels.\n"
    "6. TU entre proches/famille, VOUS en contexte formel/hiérarchique.\n"
    "7. AUCUNE hallucination. Ne rajoute RIEN absent du texte source."
)

SFX = {
    "AH","AAH","WAAH","WAAAH","UGH","URGH","ARGH","ERGH","KRGH","KHOFF",
    "HUFF","PANT","GASP","SOB","SNIFF","HMPH","GRR","BAM","BOOM","CRASH",
    "BANG","THUD","SNAP","TAP","CLAP","WHAM","WHOOSH","GAAAAH",
}

# Structured output schema — force Gemini à renvoyer ce format exact
FONT_KEYS = list(FONT_MAP.keys())

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "traductions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "fr": {"type": "string"},
                    "font_key": {"type": "string", "enum": FONT_KEYS}
                },
                "required": ["id", "fr", "font_key"]
            }
        },
        "state_update": {
            "type": "object",
            "properties": {
                "summary_update": {"type": "string"},
                "relationship_changes": {"type": "array", "items": {"type": "string"}},
                "entity_discovery": {"type": "object"}
            }
        }
    },
    "required": ["traductions"]
}


class GeminiTranslator:

    def __init__(self, device: str = "cuda", series_db=None, series_name: str = "default", source_lang: str = "en"):
        self.device = device
        self.backend = "gemini"
        self.series_db = series_db
        self.source_lang = (source_lang or "en").lower()
        # Nettoie le nom de la série pour éviter les problèmes de chemin
        self.series_name = re.sub(r"[^\w\s-]", "", (series_name or "")).strip().replace(" ", "_") or "default"
        self.generation_seconds_total = 0.0
        self._last_ts = 0.0
        # Stats
        self.api_request_count = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self._session_start = time.perf_counter()

        # API key
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY manquant !\n"
                "   set GEMINI_API_KEY=ta-cle    (Windows)\n"
                "   export GEMINI_API_KEY='...'   (Linux)\n"
                "   https://aistudio.google.com/apikey"
            )

        self.model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

        # Nouveau SDK google-genai
        try:
            from google import genai
            from google.genai import types
            self._types = types
        except ImportError:
            raise RuntimeError("pip install -U google-genai --break-system-packages")

        self._client = genai.Client(api_key=self.api_key)

        # State: créer (ou réutiliser) un dossier spécifique par série
        orig_dir = STATE_DIR / self.series_name
        # Check if the exact directory already exists
        existed = orig_dir.exists()

        # If not, try to find a similar existing directory (ignore separators/case)
        base_state_dir = orig_dir
        if not existed and STATE_DIR.exists():
            try:
                def _norm(name: str) -> str:
                    return re.sub(r"[^a-z0-9]", "", name.lower())

                for p in STATE_DIR.iterdir():
                    if p.is_dir() and _norm(p.name) == _norm(self.series_name):
                        base_state_dir = p
                        existed = True
                        break
            except Exception:
                pass

        # Ensure directory exists (will be reused if we matched an existing one)
        base_state_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = base_state_dir / "global_state.json"
        self._intrigue_file = base_state_dir / "intrigue.txt"

        # Debug logs sur l'association de la série et le chemin utilisé
        try:
            print(f"[DEBUG] 📂 État Gemini lié à l'œuvre : {self.series_name}")
            print(f"[DEBUG] 📂 Chemin : {str(base_state_dir)}")
            if not existed:
                print(f"[DEBUG] ✨ Nouveau dossier d'état créé pour cette œuvre.")
        except Exception:
            pass
        self._state = self._load_json(self._state_file, {
            "personnages": {}, "organisations": {}, "tutoiement": {}
        })

        # Cache
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._cache = self._load_json(CACHE_FILE, {})
        self._cache_dirty = False

        print("   Gemini pret | {} | series={} | {} en cache | {} persos connus".format(
            self.model_name, self.series_name, len(self._cache), len(self._state.get('personnages', {}))
        ))

    # ── IO ────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_json(path: Path, default):
        if path.exists():
            try: return json.loads(path.read_text("utf-8"))
            except Exception: pass
        return default

    def _save_state(self):
        self._state_file.write_text(json.dumps(self._state, ensure_ascii=False, indent=2), "utf-8")

    def _save_cache(self):
        if self._cache_dirty:
            CACHE_FILE.write_text(json.dumps(self._cache, ensure_ascii=False), "utf-8")
            self._cache_dirty = False

    def _get_intrigue(self) -> str:
        return self._intrigue_file.read_text("utf-8").strip() if self._intrigue_file.exists() else ""

    def _update_intrigue(self, summary: str):
        """Consolide l'intrigue globale en demandant à Gemini de fusionner l'ancien
        résumé et les nouveaux éléments en un résumé consolidé (éviter explosion tokens).
        """
        old = self._get_intrigue()
        if not summary:
            return
        try:
            types = self._types
            # Build a compact prompt asking Gemini to merge summaries conservatively
            prompt = (
                f"Voici le RESUME global actuel :\n{old}\n\n"
                f"Nouveaux faits :\n{summary}\n\n"
                "Fusionne ces informations et renvoie un résumé global consolidé, concis (3-5 phrases)."
            )
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=PromptBank.BASE_SYSTEM,
                    temperature=0.0,
                    max_output_tokens=2000,
                    response_mime_type="text/plain",
                ),
            )
            consolidated = getattr(response, "text", "") or ""
            consolidated = consolidated.strip()
            if not consolidated:
                # fallback to simple append if SDK returned nothing parseable
                combined = f"{old}\n---\n{summary}" if old else summary
            else:
                combined = consolidated
            # Keep a rolling window (50k chars)
            if len(combined) > 50000:
                combined = combined[-50000:]
            self._intrigue_file.write_text(combined.strip(), "utf-8")
        except Exception:
            # Best-effort fallback
            combined = f"{old}\n---\n{summary}" if old else summary
            if len(combined) > 50000:
                combined = combined[-50000:]
            self._intrigue_file.write_text(combined.strip(), "utf-8")

    # ── CONTEXT ───────────────────────────────────────────────────────────

    def _build_context(self) -> str:
        return self._get_compact_context()

    def _get_compact_context(self, max_chars: int = 3000) -> str:
        """Return a compact, relevant context string for the current prompt.

        Selects the last part of the global intrigue plus a concise listing of
        characters and tutoiement state truncated to `max_chars`.
        """
        parts: List[str] = []
        intr = self._get_intrigue()
        if intr:
            # keep only the tail (most recent) of the intrigue
            tail = intr[-max_chars:] if len(intr) > max_chars else intr
            parts.append(f"RESUME :\n{tail}")

        persos = self._state.get("personnages", {})
        if persos:
            lines = []
            for n, v in persos.items():
                small = json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else str(v)
                lines.append(f"  {n}: {small}")
            parts.append("PERSONNAGES :\n" + "\n".join(lines[:30]))

        tuto = self._state.get("tutoiement", {})
        if tuto:
            parts.append("TU/VOUS :\n" + "\n".join(f"  {k}: {v}" for k, v in list(tuto.items())[:30]))

        ctx = "\n\n".join(parts)
        return ctx[:max_chars]

    # ── API CALL ──────────────────────────────────────────────────────────

    def _rate_wait(self):
        elapsed = time.time() - self._last_ts
        # Reduce wait slightly to 4.1s (safe for ~15 req/min free tier)
        if elapsed < 4.1:
            time.sleep(4.1 - elapsed)
        self._last_ts = time.time()

    def _call(self, prompt: str, attempt: int = 0) -> Optional[dict]:
        self._rate_wait()
        types = self._types
        t0 = time.perf_counter()
        try:
            # Estimate input tokens (will be adjusted if SDK reports real usage)
            est_in = int(len(prompt) / 4)
            self.tokens_in += est_in

            # Config standard
            self.api_request_count += 1
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=PromptBank.BASE_SYSTEM,
                    temperature=0.3,
                    top_p=0.95,
                    max_output_tokens=16384,
                    response_mime_type="application/json",
                    response_json_schema=RESPONSE_SCHEMA,
                ),
            )
            self.generation_seconds_total += time.perf_counter() - t0

            raw = response.text if hasattr(response, "text") else ""
            # Try to extract token usage info from the SDK response
            t_in, t_out = self._extract_tokens_from_response(response, prompt, raw)
            if t_in:
                self.tokens_in += (t_in - est_in)
            if t_out:
                self.tokens_out += t_out
            else:
                self.tokens_out += int(len(raw) / 4)

            parsed = self._parse_json(raw)

            if parsed and "traductions" in parsed:
                return parsed

            if attempt < 2:
                print(f"      JSON invalide, retry {attempt+1}/2")
                time.sleep(2 ** attempt)
                return self._call(prompt, attempt + 1)
            return None

        except Exception as exc:
            self.generation_seconds_total += time.perf_counter() - t0
            err = str(exc).lower()
            if ("429" in str(exc) or "quota" in err or "resource_exhausted" in err) and attempt < 4:
                wait = min(60, 2 ** (attempt + 2))
                print(f"      Rate limit, pause {wait}s...")
                time.sleep(wait)
                return self._call(prompt, attempt + 1)
            if attempt < 2:
                time.sleep(2 ** attempt)
                return self._call(prompt, attempt + 1)
            print(f"      Gemini erreur: {exc}")
            return None

    @staticmethod
    def _parse_json(raw: str) -> Optional[dict]:
        text = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
        text = re.sub(r"\n?```\s*$", "", text).strip()
        try: return json.loads(text)
        except Exception: pass
        depth = start = 0
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0: start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try: return json.loads(text[start:i+1])
                    except Exception: pass
        return None

    def _extract_tokens_from_response(self, response, prompt: str, raw: str) -> Tuple[int, int]:
        """Try to extract token usage info from the SDK response. Returns (in, out) or (0,0) if unknown."""
        tin = 0
        tout = 0
        try:
            if hasattr(response, "token_usage"):
                tu = response.token_usage
                tin = getattr(tu, "input_tokens", getattr(tu, "prompt_tokens", 0) or 0)
                tout = getattr(tu, "output_tokens", getattr(tu, "completion_tokens", 0) or 0)
            if hasattr(response, "usage"):
                usage = response.usage
                tin = tin or getattr(usage, "prompt_tokens", 0) or tin
                tout = tout or getattr(usage, "completion_tokens", 0) or tout
            if hasattr(response, "metadata") and isinstance(response.metadata, dict):
                md = response.metadata
                tin = tin or int(md.get("input_tokens", md.get("prompt_tokens", 0) or 0))
                tout = tout or int(md.get("output_tokens", md.get("completion_tokens", 0) or 0))
        except Exception:
            pass
        return int(tin or 0), int(tout or 0)

    # ── SFX ───────────────────────────────────────────────────────────────

    def should_skip_translation(self, text: str) -> bool:
        if not text or not text.strip(): return True
        tokens = re.findall(r"[A-Za-z]+", text.upper())
        if not tokens: return True
        dialogue = {"I","YOU","WE","HE","SHE","THE","A","AN","AND","BUT","MY","YOUR",
                    "OUR","WILL","OKAY","OK","PLEASE","SO","IS","IT","TO","NOT","DO",
                    "THAT","THIS","WHAT","HOW","WHY","WHO","WELL","JUST","NEVER"}
        if all(t in SFX or re.search(r"(.)\1{2,}", t) for t in tokens):
            if not any(t in dialogue for t in tokens): return True
        return False

    # ── CACHE ─────────────────────────────────────────────────────────────

    def _ckey(self, t): return hashlib.md5(t.strip().lower().encode()).hexdigest()
    def _cget(self, t):
        key = self._ckey(t)
        val = self._cache.get(key)
        if val is not None:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        return val
    def _cset(self, t, tr):
        self._cache[self._ckey(t)] = tr
        self._cache_dirty = True

    # ── TRANSLATE PAGE — TOUT EN 1 REQUETE ────────────────────────────────

    def translate_page_json(self, texts: List[str]) -> dict:
        if not texts: return {}

        try:
            from ocr import clean_ocr_text
        except ImportError:
            clean_ocr_text = lambda t: t

        clean = [clean_ocr_text((t or "").strip()) for t in texts]
        result: Dict[str, str] = {}
        to_send: List[tuple] = []

        for i, t in enumerate(clean):
            if not t or self.should_skip_translation(t):
                result[str(i)] = t; continue
            cached = self._cget(t)
            if cached:
                result[str(i)] = cached; continue
            to_send.append((i, t))

        if not to_send:
            return result

        print(f"      Gemini: {len(to_send)} textes a traduire ({len(result)} en cache/SFX)")

        ctx = self._build_context()
        numbered = "\n".join(f'{idx}: {txt}' for idx, txt in to_send)

        prompt = (
            f"{ctx}\n\n"
            f"{PromptBank.LANG_RULES.get(self.source_lang, '')}\n\n"
            f"{PromptBank.TYPO_RULES}\n\n"
            f"TEXTES A TRADUIRE (id: texte source) :\n{numbered}\n\n"
            "Pour chaque item, renvoie un objet {id, fr, font_key} où `font_key` est l'une des clés suivantes: "
            f"{', '.join(FONT_KEYS)}.\n"
            "Renvoie un JSON au format exact demandé. Inclut aussi un objet `state_update`\n"
            "contenant `summary_update`, `relationship_changes` et `entity_discovery` si pertinent.\n"
            "Ne répète pas l'ancien résumé; fournis seulement les faits nouveaux dans `summary_update`.\n"
        )

        parsed = self._call(prompt)

        if parsed and "traductions" in parsed:
            trad_map: Dict[str, str] = {}
            font_map: Dict[str, str] = {}
            for item in parsed.get("traductions", []):
                if not isinstance(item, dict):
                    continue
                id_str = str(item.get("id", ""))
                fr_text = item.get("fr", "")
                font_key = item.get("font_key", "STANDARD") or "STANDARD"
                if font_key not in FONT_MAP:
                    font_key = "STANDARD"
                trad_map[id_str] = fr_text
                font_map[id_str] = font_key

            for local_idx, (global_idx, orig) in enumerate(to_send):
                fr = trad_map.get(str(global_idx)) or trad_map.get(str(local_idx), "")
                result[str(global_idx)] = fr if fr else orig
                if fr:
                    self._cset(orig, fr)
                    # Optionally persist font choice per-translation in state
                    # store font selection plus color metadata to avoid white-on-white defaults
                    fk = font_map.get(str(global_idx), "STANDARD")
                    # Preserve existing color if present (use original color), otherwise do not force a color
                    prev = self._state.get("fonts", {}).get(str(global_idx))
                    orig_color = None
                    if isinstance(prev, dict):
                        orig_color = prev.get("color")
                    entry = {"font_key": fk}
                    if orig_color:
                        entry["color"] = orig_color
                    self._state.setdefault("fonts", {})[str(global_idx)] = entry

            # State update handling (differential)
            state_upd = parsed.get("state_update", {}) or {}
            summary_update = state_upd.get("summary_update")
            if summary_update:
                self._update_intrigue(summary_update)

            rel_changes = state_upd.get("relationship_changes") or []
            if rel_changes:
                lst = self._state.setdefault("relationship_changes", [])
                if isinstance(rel_changes, list):
                    lst.extend(rc for rc in rel_changes if isinstance(rc, str))

            entity_disc = state_upd.get("entity_discovery") or {}
            if isinstance(entity_disc, dict):
                for k, v in entity_disc.items():
                    if isinstance(v, dict):
                        self._state.setdefault(k, {}).update(v)
            self._save_state()

            ok = sum(1 for it in parsed.get("traductions", []) if it.get("fr"))
            print(f"      {ok}/{len(to_send)} traduits")
        else:
            print("      Echec Gemini - textes laisses en anglais")
            for gi, orig in to_send:
                result[str(gi)] = orig

        self._save_cache()
        return result

    # ── TRANSLATE SINGLE ──────────────────────────────────────────────────

    def translate(self, text: str) -> str:
        if not text or self.should_skip_translation(text): return text
        try:
            from ocr import clean_ocr_text
            text = clean_ocr_text(text.strip())
        except ImportError:
            text = text.strip()
        cached = self._cget(text)
        if cached: return cached
        r = self.translate_page_json([text])
        return r.get("0", text)

    # ── COMPAT NLLBTranslator ─────────────────────────────────────────────

    def detect_source_language_with_confidence(self, text: str) -> Tuple[str, float]:
        return ("en", 0.95)

    def get_generation_seconds_total(self) -> float:
        return self.generation_seconds_total

    def get_cache_stats(self) -> dict:
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total else 0.0
        return {
            "entries": len(self._cache),
            "hits": int(self.cache_hits),
            "misses": int(self.cache_misses),
            "hit_rate": f"{hit_rate:.1f}%",
        }

    def get_last_page_payload_debug(self) -> dict:
        return {"system_prompt": PromptBank.BASE_SYSTEM, "user_prompt": "", "payload_lines": []}

    def __del__(self):
        try:
            self._save_cache()
        except Exception:
            pass

        try:
            print("\n========================================")
            print("       📊 RAPPORT DEBUG GEMINI")
            print("========================================")
            print(f"  Requêtes API : {int(self.api_request_count)}")
            print(f"  Cache Hits  : {int(self.cache_hits)}")
            print(f"  Cache Misses: {int(self.cache_misses)}")
            print(f"  Tokens In   : {int(self.tokens_in)}")
            print(f"  Tokens Out  : {int(self.tokens_out)}")
            print(f"  Temps Total : {round(self.generation_seconds_total,1)}s")
            print("========================================")
        except Exception:
            pass