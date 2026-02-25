#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
GEMINI WATCHER — Traducteur Webtoon 100% autonome via API Gemini

Surveille un dossier, détecte les fichiers OCR JSON, traduit tout par chapitre
via l'API Gemini, gère glossaire + mémoire + intrigue automatiquement.

Usage :
    export GEMINI_API_KEY="ta-clé"
    python gemini_watcher.py                          # surveille ocr_input/
    python gemini_watcher.py --input ocr_input/       # dossier custom
    python gemini_watcher.py --once chapter_001.json   # traitement unique
    python gemini_watcher.py --batch ocr_input/        # traite tout d'un coup

Structure attendue des fichiers OCR JSON :
    {
        "image": "chapter_001_part01.jpg",
        "detections": [
            {"id": "001_01", "text": "HELLO WORLD", "bbox": [x1,y1,x2,y2], "class": "bulle"},
            ...
        ]
    }

    OU simplement une liste :
    [
        {"id": "p01_b01", "text": "HELLO WORLD"},
        ...
    ]

Fichiers générés :
    data/gemini_state/global_state.json  — Glossaire auto-évolutif
    data/gemini_state/intrigue.txt       — Résumé narratif
    translated/                          — JSONs traduits
    cache/gemini_cache.json              — Cache traductions
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_INPUT_DIR = Path("ocr_input")
DEFAULT_OUTPUT_DIR = Path("translated")
DEFAULT_STATE_DIR = Path("data/gemini_state")
DEFAULT_CACHE_DIR = Path("cache")
DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_RPM = 14  # plan gratuit = 15 RPM, on prend 14 par sécurité
MAX_TEXTS_PER_REQUEST = 80  # au-delà, split en sous-batches
MAX_RETRIES = 3


# ═══════════════════════════════════════════════════════════════════════════
# RATE LIMITER
# ═══════════════════════════════════════════════════════════════════════════

class RateLimiter:
    def __init__(self, rpm: int = DEFAULT_RPM, min_delay: float = 1.0):
        self.rpm = rpm
        self.min_delay = min_delay
        self._ts: list[float] = []

    def wait(self):
        now = time.time()
        self._ts = [t for t in self._ts if now - t < 60]
        if len(self._ts) >= self.rpm:
            sleep_for = 60 - (now - self._ts[0]) + 0.5
            if sleep_for > 0:
                print(f"   ⏳ Rate limit: pause {sleep_for:.1f}s")
                time.sleep(sleep_for)
        elif self._ts:
            gap = now - self._ts[-1]
            if gap < self.min_delay:
                time.sleep(self.min_delay - gap)
        self._ts.append(time.time())


# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL STATE (glossaire + intrigue)
# ═══════════════════════════════════════════════════════════════════════════

class GlobalState:
    def __init__(self, state_dir: Path):
        self.dir = state_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = self.dir / "global_state.json"
        self.intrigue_file = self.dir / "intrigue.txt"
        self.data = self._load()

    def _load(self) -> dict:
        if self.file.exists():
            try:
                return json.loads(self.file.read_text("utf-8"))
            except Exception:
                pass
        return {"personnages": {}, "lieux": {}, "organisations": {},
                "relations": {}, "tutoiement": {}, "chapitres_traduits": []}

    def save(self):
        self.file.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), "utf-8")

    def get_intrigue(self) -> str:
        if self.intrigue_file.exists():
            return self.intrigue_file.read_text("utf-8").strip()
        return ""

    def update_intrigue(self, summary: str):
        old = self.get_intrigue()
        combined = f"{old}\n\n---\n{summary}" if old else summary
        if len(combined) > 3000:
            combined = combined[-3000:]
        self.intrigue_file.write_text(combined.strip(), "utf-8")

    def merge_entities(self, new: dict):
        for cat in ("personnages", "lieux", "organisations", "relations", "tutoiement"):
            incoming = new.get(cat, {})
            if isinstance(incoming, dict):
                existing = self.data.setdefault(cat, {})
                for k, v in incoming.items():
                    if k not in existing:
                        existing[k] = v
                    elif isinstance(existing[k], dict) and isinstance(v, dict):
                        existing[k].update(v)
        self.save()

    def mark_done(self, chapter_id: str):
        done = self.data.setdefault("chapitres_traduits", [])
        if chapter_id not in done:
            done.append(chapter_id)
            self.save()

    def is_done(self, chapter_id: str) -> bool:
        return chapter_id in self.data.get("chapitres_traduits", [])

    def build_context(self) -> str:
        parts = []
        intrigue = self.get_intrigue()
        if intrigue:
            parts.append(f"RÉSUMÉ DE L'HISTOIRE :\n{intrigue}")
        persos = self.data.get("personnages", {})
        if persos:
            lines = []
            for name, info in persos.items():
                if isinstance(info, dict):
                    lines.append(f"  - {name} ({', '.join(f'{k}: {v}' for k,v in info.items())})")
                else:
                    lines.append(f"  - {name}: {info}")
            parts.append("PERSONNAGES :\n" + "\n".join(lines))
        tuto = self.data.get("tutoiement", {})
        if tuto:
            parts.append("TU/VOUS :\n" + "\n".join(f"  - {k}: {v}" for k,v in tuto.items()))
        orgs = self.data.get("organisations", {})
        if orgs:
            parts.append("ORGANISATIONS :\n" + "\n".join(f"  - {k}: {v}" for k,v in orgs.items()))
        return "\n\n".join(parts) if parts else "(Premier chapitre — aucun contexte)"


# ═══════════════════════════════════════════════════════════════════════════
# CACHE
# ═══════════════════════════════════════════════════════════════════════════

class TranslationCache:
    def __init__(self, cache_file: Path):
        self.file = cache_file
        self.data: dict = {}
        if self.file.exists():
            try:
                self.data = json.loads(self.file.read_text("utf-8"))
            except Exception:
                pass

    def key(self, text: str) -> str:
        return hashlib.md5(text.strip().lower().encode()).hexdigest()

    def get(self, text: str) -> Optional[str]:
        return self.data.get(self.key(text))

    def set(self, text: str, tr: str):
        self.data[self.key(text)] = tr

    def save(self):
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.file.write_text(json.dumps(self.data, ensure_ascii=False), "utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# JSON PARSER
# ═══════════════════════════════════════════════════════════════════════════

def parse_json_response(raw: str) -> Optional[dict]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    pass
                start = None
    return None


# ═══════════════════════════════════════════════════════════════════════════
# SFX DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

SFX_TOKENS = {
    "AH","AAH","WAAH","WAAAH","UGH","URGH","ARGH","ERGH","KRGH","KHOFF",
    "HUFF","PANT","GASP","SOB","SNIFF","HMPH","GRR","BAM","BOOM","CRASH",
    "BANG","THUD","SNAP","TAP","CLAP","WHAM","WHOOSH","GAAAAH",
}

def is_sfx_only(text: str) -> bool:
    tokens = re.findall(r"[A-Za-z]+", text.upper())
    if not tokens:
        return True
    return all(t in SFX_TOKENS or re.search(r"(.)\1{2,}", t) for t in tokens)


# ═══════════════════════════════════════════════════════════════════════════
# PROMPT
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
Tu es un expert en LOCALISATION de manhwa/webtoon anglais → français.

RÈGLES DE FER :
0. FRANÇAIS UNIQUEMENT. Toute trace d'anglais = erreur fatale.
1. PURETÉ LINGUISTIQUE : Jamais de mots hybrides inventés. Uniquement de vrais mots français.
2. SENS > MOT-À-MOT : Traduis l'intention.
   "I WILL." → "C'est promis." | "HOLD STILL" → "Ne bouge pas" | "HONEY!" → "Chéri(e) !"
   "DAD!!" → "Papa !!" | "CHILD ABUSE" → "maltraitance"
3. MOTS COLLÉS OCR : Si des mots sont fusionnés (ABOUTOUR), sépare-les et traduis.
4. SFX/ONOMATOPÉES : Recopie les bruits purs tels quels (KRGH, WAAAH, BOOM).
5. NOMS PROPRES : Recopie tels quels.
6. TU/VOUS : "tu" entre proches, "vous" en contexte hiérarchique/formel.
7. AUCUNE HALLUCINATION : Ne rajoute RIEN absent du texte source."""


def build_prompt(texts: List[dict], context: str) -> str:
    numbered = "\n".join(f'{item["id"]}: {item["text"]}' for item in texts)
    return f"""{context}

TEXTES À TRADUIRE :
{numbered}

RÉPONDS UNIQUEMENT avec un JSON valide :
{{
  "traductions": [
    {{"id": "...", "fr": "traduction française"}}
  ],
  "nouveau_resume": "Résumé court (2-3 phrases) des événements de ce passage.",
  "nouvelles_entites": {{
    "personnages": {{"Nom": {{"genre": "M/F", "role": "desc"}}}},
    "organisations": {{"Nom": "desc"}},
    "tutoiement": {{"A↔B": "tu/vous"}},
    "lieux": {{"Nom": "desc"}}
  }}
}}"""


# ═══════════════════════════════════════════════════════════════════════════
# GEMINI CLIENT
# ═══════════════════════════════════════════════════════════════════════════

class GeminiClient:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, rpm: int = DEFAULT_RPM):
        try:
            import google.generativeai as genai
        except ImportError:
            print("❌ Module manquant : pip install google-generativeai --break-system-packages")
            sys.exit(1)

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model,
            system_instruction=SYSTEM_PROMPT,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                top_p=0.95,
                max_output_tokens=8192,
                response_mime_type="application/json",
            ),
        )
        self.limiter = RateLimiter(rpm=rpm)
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_requests = 0

    def translate_batch(self, texts: List[dict], context: str,
                        retries: int = 0) -> Optional[dict]:
        prompt = build_prompt(texts, context)
        self.limiter.wait()

        try:
            response = self.model.generate_content(prompt)
            self.total_requests += 1
            raw = response.text if hasattr(response, "text") else ""
            parsed = parse_json_response(raw)

            if parsed and "traductions" in parsed:
                return parsed

            if retries < MAX_RETRIES:
                print(f"   ⚠️  JSON invalide, retry {retries+1}/{MAX_RETRIES}")
                time.sleep(2 ** retries)
                return self.translate_batch(texts, context, retries + 1)

            print(f"   ❌ Échec après {MAX_RETRIES} tentatives")
            return None

        except Exception as exc:
            err = str(exc).lower()
            if "429" in str(exc) or "quota" in err or "resource_exhausted" in err:
                wait = min(60, 2 ** (retries + 2))
                print(f"   ⏳ Rate limit, pause {wait}s...")
                time.sleep(wait)
                if retries < MAX_RETRIES + 2:
                    return self.translate_batch(texts, context, retries + 1)
            elif retries < MAX_RETRIES:
                time.sleep(2 ** retries)
                return self.translate_batch(texts, context, retries + 1)

            print(f"   ❌ Erreur: {exc}")
            return None


# ═══════════════════════════════════════════════════════════════════════════
# OCR FILE PARSER
# ═══════════════════════════════════════════════════════════════════════════

def parse_ocr_file(path: Path) -> List[dict]:
    """Parse un fichier OCR JSON en liste de {id, text, ...}."""
    raw = json.loads(path.read_text("utf-8"))

    items = []

    # Format pipeline: {"detections": [...]} ou {"results": [...]}
    if isinstance(raw, dict):
        dets = raw.get("detections") or raw.get("results") or raw.get("texts") or []
        if not dets and "image" in raw:
            # Peut-être un seul fichier avec clé texte
            dets = raw.get("ocr_results", [])
        for d in dets:
            if isinstance(d, dict):
                text = d.get("text") or d.get("ocr_text") or d.get("text_original") or ""
                text = text.strip()
                if text and text != "(none)":
                    item_id = d.get("id") or d.get("index") or str(len(items))
                    items.append({
                        "id": str(item_id),
                        "text": text,
                        "bbox": d.get("bbox"),
                        "class": d.get("class") or d.get("class_name", "bulle"),
                    })

    # Format liste directe
    elif isinstance(raw, list):
        for i, d in enumerate(raw):
            if isinstance(d, dict):
                text = d.get("text", "").strip()
                if text and text != "(none)":
                    items.append({
                        "id": d.get("id", str(i)),
                        "text": text,
                        "bbox": d.get("bbox"),
                        "class": d.get("class", "bulle"),
                    })
            elif isinstance(d, str) and d.strip():
                items.append({"id": str(i), "text": d.strip()})

    return items


def group_by_chapter(files: List[Path]) -> Dict[str, List[Path]]:
    """Regroupe les fichiers par chapitre (préfixe commun)."""
    groups: Dict[str, List[Path]] = {}
    for f in sorted(files):
        # Extraire le chapitre du nom : chapter_001_part01.json → chapter_001
        stem = f.stem
        match = re.match(r"(.+?)(?:_part\d+|_p\d+|_\d{2,3})?$", stem, re.I)
        chapter = match.group(1) if match else stem
        groups.setdefault(chapter, []).append(f)
    return groups


# ═══════════════════════════════════════════════════════════════════════════
# MAIN TRANSLATOR
# ═══════════════════════════════════════════════════════════════════════════

def translate_chapter(
    chapter_id: str,
    files: List[Path],
    client: GeminiClient,
    state: GlobalState,
    cache: TranslationCache,
    output_dir: Path,
) -> dict:
    """Traduit un chapitre complet (plusieurs fichiers OCR regroupés)."""
    print(f"\n{'═'*60}")
    print(f"📖 Chapitre: {chapter_id} ({len(files)} fichier(s))")
    print(f"{'═'*60}")

    if state.is_done(chapter_id):
        print(f"   ⏭️  Déjà traduit, skip")
        return {"skipped": True}

    # Collecter tous les textes
    all_items: List[dict] = []
    file_boundaries: List[int] = []  # pour reconstruire par fichier

    for fpath in files:
        items = parse_ocr_file(fpath)
        file_boundaries.append(len(all_items))
        for item in items:
            # Prefix id avec le fichier pour unicité
            item["id"] = f"{fpath.stem}_{item['id']}"
            item["source_file"] = fpath.name
            all_items.append(item)

    print(f"   📝 {len(all_items)} textes collectés")

    if not all_items:
        print("   ⚠️  Aucun texte, skip")
        return {"empty": True}

    # Séparer SFX (pas besoin d'API) et texte à traduire
    to_translate: List[dict] = []
    sfx_results: Dict[str, str] = {}

    for item in all_items:
        if is_sfx_only(item["text"]):
            sfx_results[item["id"]] = item["text"]
        elif cache.get(item["text"]):
            sfx_results[item["id"]] = cache.get(item["text"])
        else:
            to_translate.append(item)

    print(f"   🔤 {len(to_translate)} à traduire | {len(sfx_results)} en cache/SFX")

    # Traduire par sous-batches
    all_translations: Dict[str, str] = dict(sfx_results)
    context = state.build_context()

    for batch_start in range(0, len(to_translate), MAX_TEXTS_PER_REQUEST):
        batch = to_translate[batch_start:batch_start + MAX_TEXTS_PER_REQUEST]
        batch_num = batch_start // MAX_TEXTS_PER_REQUEST + 1
        total_batches = (len(to_translate) + MAX_TEXTS_PER_REQUEST - 1) // MAX_TEXTS_PER_REQUEST

        print(f"\n   🌐 Batch {batch_num}/{total_batches} ({len(batch)} textes)")

        result = client.translate_batch(batch, context)

        if result and "traductions" in result:
            for tr in result["traductions"]:
                item_id = str(tr.get("id", ""))
                fr = tr.get("fr", "")
                if fr:
                    all_translations[item_id] = fr
                    # Find original text for cache
                    orig = next((b for b in batch if b["id"] == item_id), None)
                    if orig:
                        cache.set(orig["text"], fr)

            # Update state
            resume = result.get("nouveau_resume", "")
            if resume:
                state.update_intrigue(resume)
            entities = result.get("nouvelles_entites", {})
            if entities:
                state.merge_entities(entities)

            found = sum(1 for tr in result["traductions"] if tr.get("fr"))
            print(f"   ✅ {found}/{len(batch)} traduits")
        else:
            print(f"   ❌ Batch {batch_num} échoué — fallback un par un")
            for item in batch:
                # Ultra simple fallback prompt
                prompt = f'Traduis en français : "{item["text"]}"\nRéponds : {{"fr":"..."}}'
                try:
                    client.limiter.wait()
                    resp = client.model.generate_content(prompt)
                    parsed = parse_json_response(resp.text if hasattr(resp, "text") else "")
                    if parsed and parsed.get("fr"):
                        all_translations[item["id"]] = parsed["fr"]
                        cache.set(item["text"], parsed["fr"])
                except Exception:
                    all_translations[item["id"]] = item["text"]

    # Reconstruire le résultat par fichier source
    output_dir.mkdir(parents=True, exist_ok=True)
    output_data = []

    for item in all_items:
        fr = all_translations.get(item["id"], item["text"])
        output_data.append({
            "id": item["id"],
            "en": item["text"],
            "fr": fr,
            "bbox": item.get("bbox"),
            "class": item.get("class", "bulle"),
            "source_file": item.get("source_file", ""),
        })

    # Sauvegarder
    out_file = output_dir / f"{chapter_id}_translated.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "chapter": chapter_id,
            "total": len(output_data),
            "traductions": output_data,
        }, f, ensure_ascii=False, indent=2)

    state.mark_done(chapter_id)
    cache.save()

    translated_count = sum(1 for item in output_data if item["fr"] != item["en"])
    print(f"\n   📄 Sauvegardé: {out_file}")
    print(f"   📊 {translated_count}/{len(output_data)} traduits")

    return {"file": str(out_file), "total": len(output_data), "translated": translated_count}


# ═══════════════════════════════════════════════════════════════════════════
# WATCHER
# ═══════════════════════════════════════════════════════════════════════════

def watch_directory(
    input_dir: Path,
    output_dir: Path,
    client: GeminiClient,
    state: GlobalState,
    cache: TranslationCache,
    poll_interval: float = 5.0,
):
    """Surveille le dossier et traduit les nouveaux fichiers."""
    print(f"\n👁️  Surveillance de {input_dir}/ (Ctrl+C pour arrêter)")
    print(f"   Déposez des fichiers JSON OCR pour lancer la traduction.\n")

    processed = set()
    try:
        while True:
            json_files = sorted(input_dir.glob("*.json"))
            new_files = [f for f in json_files if f.name not in processed]

            if new_files:
                chapters = group_by_chapter(new_files)
                for chapter_id, files in chapters.items():
                    if not state.is_done(chapter_id):
                        translate_chapter(chapter_id, files, client, state, cache, output_dir)
                    for f in files:
                        processed.add(f.name)

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\n\n🛑 Arrêt du watcher")
        cache.save()


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Gemini Watcher — Traduction autonome de Webtoons",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", "-i", type=Path, default=DEFAULT_INPUT_DIR,
                        help=f"Dossier à surveiller (défaut: {DEFAULT_INPUT_DIR})")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Dossier sortie (défaut: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--once", type=Path, default=None,
                        help="Traduire un seul fichier JSON et quitter")
    parser.add_argument("--batch", action="store_true",
                        help="Traiter tous les fichiers du dossier input puis quitter")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Modèle Gemini (défaut: {DEFAULT_MODEL})")
    parser.add_argument("--rpm", type=int, default=DEFAULT_RPM,
                        help=f"Requêtes par minute max (défaut: {DEFAULT_RPM})")
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR,
                        help=f"Dossier état persistant (défaut: {DEFAULT_STATE_DIR})")
    parser.add_argument("--reset-state", action="store_true",
                        help="Réinitialiser glossaire et intrigue")
    args = parser.parse_args()

    # Banner
    print("""
╔═══════════════════════════════════════════════════════════════╗
║       🌐 GEMINI WATCHER — Traducteur Webtoon Autonome       ║
╚═══════════════════════════════════════════════════════════════╝""")

    # API key
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        print("\n❌ Variable GEMINI_API_KEY manquante !")
        print("   export GEMINI_API_KEY='ta-clé'")
        print("   Obtiens-la sur https://aistudio.google.com/apikey")
        sys.exit(1)

    # State
    state = GlobalState(args.state_dir)
    if args.reset_state:
        state.data = {"personnages": {}, "lieux": {}, "organisations": {},
                      "relations": {}, "tutoiement": {}, "chapitres_traduits": []}
        state.save()
        if state.intrigue_file.exists():
            state.intrigue_file.unlink()
        print("🗑️  État réinitialisé")

    # Cache
    cache = TranslationCache(DEFAULT_CACHE_DIR / "gemini_cache.json")

    # Client
    client = GeminiClient(api_key, model=args.model, rpm=args.rpm)

    print(f"\n📋 Config:")
    print(f"   Modèle:  {args.model}")
    print(f"   RPM:     {args.rpm}")
    print(f"   State:   {args.state_dir}")
    print(f"   Persos:  {len(state.data.get('personnages', {}))} connus")
    print(f"   Cache:   {len(cache.data)} entrées")

    # ── Mode unique ──
    if args.once:
        if not args.once.exists():
            print(f"\n❌ Fichier introuvable: {args.once}")
            sys.exit(1)
        args.output.mkdir(parents=True, exist_ok=True)
        translate_chapter(args.once.stem, [args.once], client, state, cache, args.output)
        cache.save()
        return

    # ── Mode batch ──
    if args.batch:
        args.input.mkdir(parents=True, exist_ok=True)
        json_files = sorted(args.input.glob("*.json"))
        if not json_files:
            print(f"\n⚠️  Aucun fichier JSON dans {args.input}/")
            return
        chapters = group_by_chapter(json_files)
        print(f"\n📚 {len(chapters)} chapitre(s) à traiter")
        args.output.mkdir(parents=True, exist_ok=True)
        for chapter_id, files in chapters.items():
            translate_chapter(chapter_id, files, client, state, cache, args.output)
        cache.save()
        print(f"\n✅ Terminé ! Résultats dans {args.output}/")
        return

    # ── Mode watcher ──
    args.input.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    watch_directory(args.input, args.output, client, state, cache)


if __name__ == "__main__":
    main()
