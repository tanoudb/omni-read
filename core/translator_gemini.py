"""
═══════════════════════════════════════════════════════════════════════════════
GeminiTranslator V3 — Mega-Batch + Fallback Cascade + Color Resolver

Nouveautés V3 :
- Fallback 429 : gemini-2.5-flash → gemini-2.5-flash-lite → gemini-3-flash → gemma-3-27b-it
- Mega-Batch : accumule 5-10 chapitres par requête (1 ticket RPD)
- Nettoyage OCR intégré dans le prompt
- IDs composites pour le multi-chapitre
- System formatting : retours à la ligne automatiques
═══════════════════════════════════════════════════════════════════════════════
"""

import json, os, re, time, hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from utils.gemini_prompt import FONT_MAP, PromptBank

STATE_DIR = Path("data/gemini_state")
CACHE_FILE = Path("cache/gemini_cache.json")

# ── Modèle principal et cascade de fallback ──────────────────────────────

MODEL_CASCADE = [
    {
        "name": "gemini-2.5-flash",
        "rpd": 20,
        "tpm": 250_000,
        "description": "Principal — 20 RPD, 250K TPM",
    },
    {
        "name": "gemini-2.5-flash-lite",
        "rpd": 30,
        "tpm": 250_000,
        "description": "Fallback 1 — Plus léger",
    },
    {
        "name": "gemini-3-flash",
        "rpd": 50,
        "tpm": 500_000,
        "description": "Fallback 2 — Nouvelle génération",
    },
    {
        "name": "gemma-3-27b-it",
        "rpd": 14_400,
        "tpm": 1_000_000,
        "description": "Fallback ultime — 14 400 RPD, quota quasi-illimité",
    },
]

DEFAULT_MODEL = MODEL_CASCADE[0]["name"]

# ── System prompt enrichi V3 ─────────────────────────────────────────────

SYSTEM = (
    "Tu es un ADAPTATEUR professionnel de manhwa (EN/KO → FR).\n"
    "Ton travail est identique à celui d'un studio de doublage.\n"
    "RÈGLES DE FER :\n"
    "0. FRANÇAIS UNIQUEMENT. Toute trace d'anglais = erreur fatale.\n"
    "1. FRANÇAIS ORAL : Utilise des contractions naturelles (T'as, J'en peux plus, C'est quoi ça).\n"
    "2. SENS > MOT-À-MOT : 'I WILL.'→'Compte sur moi.' | 'HOLD STILL'→'Bouge pas !' | "
    "'HONEY!'→'Chéri(e) !' | 'DAD!!'→'Papa !!' | 'YOU BASTARD!'→'Enfoiré !'\n"
    "3. NETTOYAGE OCR : Mots collés → sépare. Bruits (677777) → ignore. Lettres confondues → déduis.\n"
    "4. SFX/onomatopées pures (KRGH, WAAAH, BOOM, GAAAAH) → recopie tels quels.\n"
    "5. Noms propres → recopie tels quels.\n"
    "6. TU entre proches/famille, VOUS en contexte formel/hiérarchique.\n"
    "7. AUCUNE hallucination. Ne rajoute RIEN absent du texte source.\n"
    "8. ÉCRANS SYSTÈME : Insère \\n après les deux-points. (JOB: PRIEST → CLASSE :\\nPRÊTRE)"
)

SFX = {
    "AH", "AAH", "WAAH", "WAAAH", "UGH", "URGH", "ARGH", "ERGH", "KRGH",
    "KHOFF", "HUFF", "PANT", "GASP", "SOB", "SNIFF", "HMPH", "GRR", "BAM",
    "BOOM", "CRASH", "BANG", "THUD", "SNAP", "TAP", "CLAP", "WHAM", "WHOOSH",
    "GAAAAH", "SLASH", "FWOOSH", "CRACK",
}

# ── Schema JSON structuré ────────────────────────────────────────────────

FONT_KEYS = list(FONT_MAP.keys())

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "traductions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id":       {"type": "string"},
                    "fr":       {"type": "string"},
                    "font_key": {"type": "string", "enum": FONT_KEYS},
                },
                "required": ["id", "fr", "font_key"],
            },
        },
        "state_update": {
            "type": "object",
            "properties": {
                "summary_update":       {"type": "string"},
                "relationship_changes": {"type": "array", "items": {"type": "string"}},
                "entity_discovery": {
                    "type": "object",
                    "properties": {
                        # Listes d'objets à champs fixes plutôt qu'un dict à clés
                        # libres : "additionalProperties" n'est pas supporté par le
                        # schéma structuré Gemini, et en pratique un "object" libre
                        # reste vide (le modèle ne le remplit pas de façon fiable).
                        # Le pattern array-of-objects-à-champs-requis est celui qui
                        # marche déjà très bien pour "traductions".
                        "glossary": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source": {"type": "string"},
                                    "fr": {"type": "string"},
                                },
                                "required": ["source", "fr"],
                            },
                        },
                        "personnages": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "nom": {"type": "string"},
                                    "description": {"type": "string"},
                                },
                                "required": ["nom", "description"],
                            },
                        },
                    },
                },
            },
        },
    },
    "required": ["traductions"],
}


class GeminiTranslator:
    """
    Traducteur Gemini V3 avec :
    - Cascade de fallback sur erreur 429
    - Mega-batch multi-chapitres
    - Prompt enrichi (doublage, OCR cleaning, system format)
    """

    def __init__(
        self,
        device: str = "cuda",
        series_db=None,
        series_name: str = "default",
        source_lang: str = "en",
    ):
        self.device = device
        self.backend = "gemini"
        self.series_db = series_db
        self.series_name = series_name
        self.source_lang = source_lang

        # Stats
        self.cache_hits = 0
        self.cache_misses = 0
        self.generation_seconds_total = 0.0
        self.api_calls = 0
        # Textes du dernier lot restés en langue source, (index, texte).
        # Permet à l'appelant de refuser de valider une page partielle.
        self.last_untranslated: List[Tuple[int, str]] = []
        self.fallback_count = 0

        # Modèle courant (commence par le principal)
        self._current_model_idx = 0

        # State & Cache
        self._state = self._load_state()
        self._cache = self._load_cache()
        self._cache_dirty = False
        self._last_font_map: Dict[str, str] = {}  # composite_id -> font_key (dernier appel mega-batch)

        # Init SDK
        self._client = None
        self._init_client()

    # ── INIT ──────────────────────────────────────────────────────────────

    def _init_client(self):
        """Initialise le client google-genai."""
        try:
            from google import genai
            from google.genai import types

            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                print("⚠️ GEMINI_API_KEY non défini. Traduction Gemini désactivée.")
                self._client = None
                return

            self._client = genai.Client(api_key=api_key)
            self._genai_types = types
            print(f"✅ Gemini initialisé — Modèle: {self.current_model_name}")
        except ImportError:
            print("⚠️ google-genai non installé. pip install google-genai")
            self._client = None

    @property
    def current_model_name(self) -> str:
        return MODEL_CASCADE[self._current_model_idx]["name"]

    @property
    def current_model_info(self) -> dict:
        return MODEL_CASCADE[self._current_model_idx]

    # ── FALLBACK CASCADE ──────────────────────────────────────────────────

    def _escalate_model(self) -> bool:
        """
        Passe au modèle suivant dans la cascade.
        Retourne False si on a épuisé tous les modèles.
        """
        if self._current_model_idx < len(MODEL_CASCADE) - 1:
            self._current_model_idx += 1
            self.fallback_count += 1
            model_info = self.current_model_info
            print(
                f"⚠️ Fallback → {model_info['name']} "
                f"({model_info['description']})"
            )
            return True
        return False

    def _reset_model(self):
        """Revient au modèle principal (après succès ou nouveau batch)."""
        self._current_model_idx = 0

    # ── APPEL API AVEC RETRY + CASCADE ────────────────────────────────────

    def _call(self, prompt: str, max_retries: int = 3) -> Optional[dict]:
        """
        Appel API Gemini avec :
        1. Retry classique (3 essais par modèle)
        2. Cascade de fallback sur 429 / ResourceExhausted
        """
        if self._client is None:
            # Échouer BRUYAMMENT. L'avertissement de construction (« clé non
            # définie ») se perd dans le défilement d'un run long ; ici on est
            # au moment précis où une traduction est perdue.
            print(
                "   ❌ TRADUCTION IMPOSSIBLE : aucun client Gemini "
                "(GEMINI_API_KEY absente ou invalide). Les textes de ce lot "
                "resteront en langue source."
            )
            return None

        original_model_idx = self._current_model_idx

        while self._current_model_idx < len(MODEL_CASCADE):
            model_name = self.current_model_name

            for attempt in range(max_retries):
                try:
                    t0 = time.perf_counter()

                    config = self._genai_types.GenerateContentConfig(
                        temperature=0.15,
                        top_p=0.95,
                        # 5-10 chapitres/appel (~250-1500 bulles) peuvent facilement
                        # dépasser 16K tokens en sortie et tronquer le JSON, ce qui
                        # laissait des bulles entières sans traduction. Gemini 2.5
                        # Flash supporte jusqu'à 65536 tokens de sortie.
                        max_output_tokens=65_536,
                        system_instruction=SYSTEM,
                        response_mime_type="application/json",
                        response_schema=RESPONSE_SCHEMA,
                    )

                    response = self._client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=config,
                    )

                    elapsed = time.perf_counter() - t0
                    self.generation_seconds_total += elapsed
                    self.api_calls += 1

                    raw = response.text or ""
                    parsed = self._parse_json(raw)

                    if parsed:
                        # Succès — on peut revenir au modèle principal prochain coup
                        if self._current_model_idx != original_model_idx:
                            print(f"   ✅ Succès avec {model_name} ({elapsed:.1f}s)")
                        return parsed

                except Exception as e:
                    err_str = str(e).lower()

                    # Erreur 429 / quota → cascade immédiate
                    if "429" in err_str or "resource" in err_str or "quota" in err_str:
                        print(f"   ⚡ Quota épuisé sur {model_name} (429)")
                        break  # Sort de la boucle retry → passe au modèle suivant

                    # Erreur réseau → retry après backoff
                    if attempt < max_retries - 1:
                        wait = 2 ** (attempt + 1)
                        print(f"   ⚠️ Erreur Gemini (essai {attempt+1}/{max_retries}): {e}")
                        print(f"   ⏳ Retry dans {wait}s...")
                        time.sleep(wait)
                    else:
                        print(f"   ❌ Échec sur {model_name} après {max_retries} essais: {e}")

            # Passer au modèle suivant
            if not self._escalate_model():
                print("   ❌ Tous les modèles épuisés. Traduction impossible.")
                break

        # Reset pour le prochain batch
        self._current_model_idx = original_model_idx
        return None

    # ── JSON PARSING ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(raw: str) -> Optional[dict]:
        """Parse JSON brut depuis la réponse Gemini."""
        if not raw:
            return None

        # Nettoyer les marqueurs markdown
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Tenter d'extraire un objet JSON
            match = re.search(r"\{[\s\S]*\}", cleaned)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return None

    # ── STATE / CACHE ─────────────────────────────────────────────────────

    def _state_path(self) -> Path:
        return STATE_DIR / self.series_name / "state.json"

    def _intrigue_path(self) -> Path:
        return STATE_DIR / self.series_name / "intrigue.txt"

    def _load_state(self) -> dict:
        p = self._state_path()
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_state(self):
        p = self._state_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_cache(self) -> dict:
        if CACHE_FILE.exists():
            try:
                return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_cache(self):
        if self._cache_dirty:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            CACHE_FILE.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._cache_dirty = False

    def _update_intrigue(self, summary_update: str):
        p = self._intrigue_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        existing = p.read_text(encoding="utf-8") if p.exists() else ""
        if summary_update and summary_update not in existing:
            p.write_text(
                existing + "\n" + summary_update if existing else summary_update,
                encoding="utf-8",
            )

    # ── CONTEXTE DYNAMIQUE ────────────────────────────────────────────────

    def _build_context(self) -> str:
        """Construit le contexte (glossaire, intrigue, état) pour le prompt."""
        parts = [PromptBank.BASE_SYSTEM]

        # Intrigue
        intrigue = ""
        ip = self._intrigue_path()
        if ip.exists():
            try:
                intrigue = ip.read_text(encoding="utf-8").strip()
            except Exception:
                pass
        if intrigue:
            # Garder les 800 derniers caractères max
            if len(intrigue) > 800:
                intrigue = "..." + intrigue[-800:]
            parts.append(f"═══ CONTEXTE NARRATIF ═══\n{intrigue}")

        # Glossaire depuis state
        glossary = self._state.get("glossary", {})
        if glossary:
            entries = [f"  {k} → {v}" for k, v in glossary.items()]
            parts.append("═══ GLOSSAIRE ═══\n" + "\n".join(entries))

        # Personnages connus
        chars = self._state.get("personnages", {})
        if chars:
            char_lines = []
            for name, info in chars.items():
                if isinstance(info, dict):
                    char_lines.append(f"  {name}: {info.get('role', '?')} ({info.get('genre', '?')})")
                else:
                    char_lines.append(f"  {name}: {info}")
            parts.append("═══ PERSONNAGES ═══\n" + "\n".join(char_lines))

        return "\n\n".join(parts)

    # ── SKIP / CACHE ──────────────────────────────────────────────────────

    def should_skip_translation(self, text: str) -> bool:
        """Détecte les SFX purs et textes à ne pas traduire."""
        t = text.strip().upper()
        if not t:
            return True
        tokens = re.split(r"[\s!?.…]+", t)
        tokens = [tk for tk in tokens if tk]
        if all(tk in SFX for tk in tokens):
            return True
        # Bruit numérique pur
        if re.match(r"^[\d\s.,]+$", t):
            return True
        # Répétition de caractères sans sens
        if re.match(r"^(.)\1{4,}$", t):
            return True
        return False

    def _ckey(self, t: str) -> str:
        return hashlib.md5(t.strip().lower().encode()).hexdigest()

    def _cget(self, t: str) -> Optional[str]:
        key = self._ckey(t)
        val = self._cache.get(key)
        if val is not None:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        return val

    def _cset(self, t: str, tr: str):
        self._cache[self._ckey(t)] = tr
        self._cache_dirty = True

    # ── POST-PROCESSING SYSTEM ────────────────────────────────────────────

    @staticmethod
    def _format_system_text(text: str) -> str:
        """
        Force les retours à la ligne après les deux-points pour les écrans System.
        "CLASSE : PRÊTRE" → "CLASSE :\nPRÊTRE"
        """
        if not text:
            return text

        # Ajouter \n après : s'il n'y en a pas déjà
        formatted = re.sub(r":\s+(?!\n)", ":\n", text)
        return formatted.strip()

    # ── DÉTECTION LANGUE (compat interface Translator) ────────────────────

    def detect_source_language_with_confidence(self, text: str) -> Tuple[str, float]:
        """
        Gemini infère la langue depuis le contexte du prompt lui-même (pas de
        détection statistique locale nécessaire) : on retourne juste la langue
        source configurée avec confiance maximale, pour rester compatible avec
        l'interface de Translator (utilisée par pipeline.py pour le scoring de
        confiance globale, pas pour le choix de traduction).
        """
        return self.source_lang, 1.0

    def detect_source_language(self, text: str) -> str:
        return self.source_lang

    # ── TRADUCTION PAGE (API PRINCIPALE) ──────────────────────────────────

    def translate_page_json(
        self,
        texts: List[str],
        class_names: Optional[List[str]] = None,
        chapter_id: str = "",
    ) -> dict:
        """
        Traduit une liste de textes en une requête.
        Retourne {str(index): traduction_fr}.
        """
        if not texts:
            return {}

        if class_names is None:
            class_names = [""] * len(texts)

        # Nettoyage basique
        clean = [(t or "").strip() for t in texts]
        result: Dict[str, str] = {}
        to_send: List[Tuple[int, str, str]] = []  # (index, text, class_name)

        for i, (t, cls) in enumerate(zip(clean, class_names)):
            if not t or self.should_skip_translation(t):
                result[str(i)] = t
                continue
            cached = self._cget(t)
            if cached:
                result[str(i)] = cached
                continue
            to_send.append((i, t, cls))

        if not to_send:
            return result

        print(f"      Gemini V3: {len(to_send)} textes à traduire ({len(result)} en cache/SFX)")

        # Construire le prompt avec IDs
        if chapter_id:
            numbered = "\n".join(
                f"{chapter_id}_{idx:03d}: {txt}" for idx, txt, _ in to_send
            )
            is_mega = True
        else:
            numbered = "\n".join(
                f"{idx}: {txt}" for idx, txt, _ in to_send
            )
            is_mega = False

        ctx = self._build_context()
        prompt = PromptBank.build_full_prompt(
            numbered_texts=numbered,
            context=ctx,
            source_lang=self.source_lang,
            is_mega_batch=is_mega,
        )

        # Appel API avec fallback cascade
        self.last_untranslated = []
        parsed = self._call(prompt)

        if not (parsed and "traductions" in parsed):
            # Sans cette branche, les index de `to_send` n'apparaissaient nulle
            # part dans le résultat : l'appelant ne pouvait pas distinguer
            # « non traduit » de « absent », et rendait l'anglais sans le
            # savoir.
            for idx, orig, _cls in to_send:
                result[str(idx)] = orig
                self.last_untranslated.append((idx, orig))
            print(
                f"   ❌ {len(to_send)} texte(s) NON TRADUIT(S) : l'appel API "
                f"n'a rien rendu d'exploitable."
            )
            for idx, orig in self.last_untranslated[:5]:
                print(f"        #{idx:02d} {orig[:60]!r}")
            return result

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

            for idx, orig, cls in to_send:
                # Essayer les deux formats d'ID
                if chapter_id:
                    composite_id = f"{chapter_id}_{idx:03d}"
                    fr = trad_map.get(composite_id, "")
                else:
                    fr = trad_map.get(str(idx), "")

                # Post-traitement System
                if cls.lower() in ("system", "system_card", "sys") and fr:
                    fr = self._format_system_text(fr)

                if fr:
                    result[str(idx)] = fr
                    self._cset(orig, fr)
                else:
                    # On garde le texte source pour ne pas rendre une bulle
                    # VIDE — mais on le COMPTE et on le signale. C'est ce
                    # repli, jusque-là muet, qui laissait passer de l'anglais
                    # dans une planche française sans que rien ne l'indique.
                    result[str(idx)] = orig
                    self.last_untranslated.append((idx, orig))

                # Stocker le choix de font
                fk = font_map.get(
                    f"{chapter_id}_{idx:03d}" if chapter_id else str(idx),
                    "STANDARD",
                )
                fonts = self._state.setdefault("fonts", {})
                fonts[str(idx)] = {"font_key": fk}

            # Mettre à jour state
            su = parsed.get("state_update") or {}
            summary = su.get("summary_update", "")
            if summary:
                self._update_intrigue(summary)

            entities = su.get("entity_discovery") or {}
            if isinstance(entities, dict):
                for k, v in entities.items():
                    if isinstance(v, dict):
                        self._state.setdefault(k, {}).update(v)

            rels = su.get("relationship_changes") or []
            if isinstance(rels, list) and rels:
                lst = self._state.setdefault("relationship_changes", [])
                lst.extend(r for r in rels if isinstance(r, str))

            self._save_state()

        if self.last_untranslated:
            print(
                f"   ⚠️ {len(self.last_untranslated)}/{len(texts)} texte(s) "
                f"restés en langue source dans ce lot."
            )

        self._save_cache()
        return result

    # ── MEGA-BATCH (MULTI-CHAPITRES) ──────────────────────────────────────

    def translate_mega_batch(
        self,
        chapters: Dict[str, List[str]],
        class_map: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Dict[str, str]]:
        """
        Traduit plusieurs chapitres en une seule requête API.

        Args:
            chapters: {"CH01": [texte1, texte2, ...], "CH02": [...], ...}
            class_map: {"CH01": [class1, class2, ...], ...}

        Returns:
            {"CH01": {"0": "trad1", "1": "trad2"}, "CH02": {...}, ...}
        """
        if not chapters:
            return {}

        if class_map is None:
            class_map = {}

        # Combiner tous les textes avec IDs composites
        all_to_send: List[Tuple[str, int, str, str]] = []  # (ch_id, idx, text, class)
        results: Dict[str, Dict[str, str]] = {ch: {} for ch in chapters}

        for ch_id, texts in chapters.items():
            classes = class_map.get(ch_id, [""] * len(texts))
            for i, (t, cls) in enumerate(zip(texts, classes)):
                t = (t or "").strip()
                if not t or self.should_skip_translation(t):
                    results[ch_id][str(i)] = t
                    continue
                cached = self._cget(t)
                if cached:
                    results[ch_id][str(i)] = cached
                    continue
                all_to_send.append((ch_id, i, t, cls))

        if not all_to_send:
            return results

        total_chapters = len(chapters)
        print(
            f"      Gemini V3 MEGA-BATCH: {len(all_to_send)} textes "
            f"de {total_chapters} chapitres en 1 requête"
        )

        # Construire le prompt
        numbered = "\n".join(
            f"{ch_id}_{idx:03d}: {txt}"
            for ch_id, idx, txt, _ in all_to_send
        )

        ctx = self._build_context()
        prompt = PromptBank.build_full_prompt(
            numbered_texts=numbered,
            context=ctx,
            source_lang=self.source_lang,
            is_mega_batch=True,
        )

        # Appel API
        parsed = self._call(prompt)

        trad_map: Dict[str, str] = {}
        font_map: Dict[str, str] = {}

        if parsed and "traductions" in parsed:
            for item in parsed.get("traductions", []):
                if not isinstance(item, dict):
                    continue
                id_str = str(item.get("id", ""))
                trad_map[id_str] = item.get("fr", "")
                font_map[id_str] = item.get("font_key", "STANDARD")

            # State update
            su = parsed.get("state_update") or {}
            summary = su.get("summary_update", "")
            if summary:
                self._update_intrigue(summary)
            entities = su.get("entity_discovery") or {}
            if isinstance(entities, dict):
                glossary_items = entities.get("glossary") or []
                if isinstance(glossary_items, list):
                    glossary = self._state.setdefault("glossary", {})
                    for item in glossary_items:
                        if isinstance(item, dict) and item.get("source") and item.get("fr"):
                            glossary[str(item["source"])] = str(item["fr"])

                perso_items = entities.get("personnages") or []
                if isinstance(perso_items, list):
                    personnages = self._state.setdefault("personnages", {})
                    for item in perso_items:
                        if isinstance(item, dict) and item.get("nom") and item.get("description"):
                            personnages[str(item["nom"])] = str(item["description"])
            self._save_state()

        # ── Retry ciblé pour les bulles manquées (LLM qui dévie de l'ID
        # attendu, ou JSON tronqué sur un batch trop gros). Sans ça, une
        # bulle manquante se traduisait par le texte anglais original
        # simplement redessiné — "bulle pas traduite". On ne retente que
        # si le 1er appel a au moins partiellement réussi (sinon la cascade
        # de modèles est déjà épuisée, retenter identique ne sert à rien).
        missing = [
            (ch_id, idx, orig, cls)
            for (ch_id, idx, orig, cls) in all_to_send
            if not trad_map.get(f"{ch_id}_{idx:03d}", "")
        ]
        if missing and parsed:
            print(f"      ⚠️  {len(missing)}/{len(all_to_send)} bulles manquantes dans la réponse — retry ciblé")
            retry_numbered = "\n".join(
                f"{ch_id}_{idx:03d}: {txt}" for ch_id, idx, txt, _ in missing
            )
            retry_prompt = PromptBank.build_full_prompt(
                numbered_texts=retry_numbered,
                context=ctx,
                source_lang=self.source_lang,
                is_mega_batch=True,
            )
            retry_parsed = self._call(retry_prompt)
            if retry_parsed and "traductions" in retry_parsed:
                for item in retry_parsed.get("traductions", []):
                    if not isinstance(item, dict):
                        continue
                    id_str = str(item.get("id", ""))
                    if item.get("fr"):
                        trad_map[id_str] = item.get("fr", "")
                        font_map[id_str] = item.get("font_key", "STANDARD")

        still_missing = 0
        for ch_id, idx, orig, cls in all_to_send:
            composite_id = f"{ch_id}_{idx:03d}"
            fr = trad_map.get(composite_id, "")

            if cls.lower() in ("system", "system_card", "sys") and fr:
                fr = self._format_system_text(fr)

            if not fr:
                still_missing += 1
            results[ch_id][str(idx)] = fr if fr else orig
            if fr:
                self._cset(orig, fr)

        if still_missing:
            print(f"      ❌ {still_missing} bulle(s) restées non traduites après retry (texte original conservé)")

        # Exposé pour le rendu : font_key choisi par le LLM, par ID composite.
        # (Le résultat "results" garde son contrat existant — juste le texte —
        # pour ne rien casser côté appelants qui ne s'intéressent qu'à ça.)
        self._last_font_map.update(font_map)

        self._save_cache()
        return results

    def get_font_key(self, composite_id: str) -> Optional[str]:
        """font_key choisi par le LLM pour cet ID (dernier mega-batch), ou None."""
        fk = self._last_font_map.get(composite_id)
        return fk if fk in FONT_MAP else None

    def translate(self, text: str) -> str:
        """
        Traduction d'un seul texte (compat interface Translator), utilisée en
        fallback quand translate_page_json ne renvoie pas un JSON indexé
        exploitable, ou pour les cartes System traitées titre/corps séparément.
        """
        if not text or not text.strip():
            return text
        result = self.translate_page_json([text])
        return (result.get("0") or result.get(0) or text) if isinstance(result, dict) else text

    # ── STATS ─────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        return {
            "model": self.current_model_name,
            "api_calls": self.api_calls,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "generation_seconds": round(self.generation_seconds_total, 2),
            "fallback_count": self.fallback_count,
        }

    def get_cache_stats(self) -> dict:
        """Compat interface Translator (pipeline.py logue entries/hit_rate)."""
        total = self.cache_hits + self.cache_misses
        hit_rate = f"{(100.0 * self.cache_hits / total):.1f}%" if total else "0.0%"
        return {"entries": len(self._cache), "hit_rate": hit_rate}