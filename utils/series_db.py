"""
═══════════════════════════════════════════════════════════════════════════════
SERIES DATABASE — Mode Série pour traduction cohérente

Architecture :
    data/series/<slug>/
    ├── series.json          # Métadonnées, personnages, monde
    ├── glossary.json        # Terme EN → FR verrouillé
    ├── consistency.json     # Auto-généré : phrases répétées, variantes
    └── chapters/
        ├── ch001.json       # Résumé, timeline, traductions
        └── ch002.json

Fonctionnalités :
    1. Glossaire persistant (noms, termes, expressions)
    2. Mémoire narrative (résumé épisode, timeline, mood)
    3. Vérification cohérence (noms, phrases récurrentes)
    4. Poids narratif (type de bulle → ton adapté)
    5. Contexte inter-chapitres (résumé chapitre précédent)
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Character:
    name: str                          # Nom principal (ex: "Arthur Leywin")
    aliases: List[str] = field(default_factory=list)  # ["Art", "ARTHUR"]
    gender: str = "?"                  # M / F / ?
    tone: str = "neutral"             # confident / timid / aggressive / comic / wise / neutral
    relationship: str = ""            # "MC", "antagonist", "love interest", "comic relief"
    notes: str = ""                   # Notes libres

    def matches(self, text: str) -> bool:
        """Vérifie si un texte contient ce personnage."""
        text_upper = text.upper()
        if self.name.upper() in text_upper:
            return True
        return any(alias.upper() in text_upper for alias in self.aliases)


@dataclass
class ChapterContext:
    chapter_number: int = 0
    title: str = ""
    summary: str = ""                  # Résumé narratif du chapitre
    timeline: str = ""                 # "Wedding → MC meurt → 8 ans plus tard"
    mood: str = ""                     # "tragic → hopeful"
    active_characters: List[str] = field(default_factory=list)
    previous_summary: str = ""         # Résumé du chapitre précédent
    translations: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # translations = {"0": {"en": "...", "fr": "..."}, ...}


@dataclass
class SeriesProfile:
    name: str = ""                     # "The Beginning After The End"
    slug: str = ""                     # "tbate"
    source_lang: str = "en"
    target_lang: str = "fr"
    genre: str = ""                    # "action / fantasy / romance"
    tone_general: str = ""             # "dark / comedic / dramatic"
    characters: List[Character] = field(default_factory=list)
    world_terms: Dict[str, str] = field(default_factory=dict)
    # world_terms = {"Cheongdo": "Cheongdo", "Chairman": "Président"}


# ═══════════════════════════════════════════════════════════════════════════
# GLOSSARY
# ═══════════════════════════════════════════════════════════════════════════

class Glossary:
    """Glossaire persistant EN → FR pour une série."""

    def __init__(self, path: Path):
        self.path = path
        self.entries: Dict[str, str] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.entries = {k.upper().strip(): v for k, v in data.items()}
            except Exception:
                self.entries = {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)

    def lookup(self, text: str) -> Optional[str]:
        """Cherche une traduction exacte dans le glossaire."""
        key = text.upper().strip()
        return self.entries.get(key)

    def lookup_fuzzy(self, text: str) -> List[Tuple[str, str]]:
        """Cherche les termes du glossaire CONTENUS dans le texte."""
        text_upper = text.upper()
        matches = []
        for key, value in self.entries.items():
            if key in text_upper:
                matches.append((key, value))
        return sorted(matches, key=lambda x: len(x[0]), reverse=True)

    def add(self, en: str, fr: str):
        """Ajoute/met à jour une entrée."""
        self.entries[en.upper().strip()] = fr
        self.save()

    def add_batch(self, entries: Dict[str, str]):
        """Ajoute plusieurs entrées d'un coup."""
        for en, fr in entries.items():
            self.entries[en.upper().strip()] = fr
        self.save()

    def remove(self, en: str):
        key = en.upper().strip()
        self.entries.pop(key, None)
        self.save()

    def get_prompt_block(self, texts: List[str], max_entries: int = 30) -> str:
        """Génère un bloc de glossaire pour le prompt LLM.
        
        Ne retourne que les entrées PERTINENTES pour les textes donnés.
        """
        relevant = set()
        all_text = " ".join(texts).upper()

        for key, value in self.entries.items():
            # Entrée exacte ou contenue dans un des textes
            if key in all_text:
                relevant.add((key, value))

        if not relevant:
            return ""

        lines = sorted(relevant, key=lambda x: x[0])[:max_entries]
        block = "GLOSSAIRE (traductions OBLIGATOIRES, ne PAS modifier) :\n"
        for en, fr in lines:
            block += f'  "{en}" → "{fr}"\n'
        return block


# ═══════════════════════════════════════════════════════════════════════════
# CONSISTENCY CHECKER
# ═══════════════════════════════════════════════════════════════════════════

class ConsistencyChecker:
    """Vérifie la cohérence des traductions dans la série."""

    def __init__(self, path: Path):
        self.path = path
        self.phrase_memory: Dict[str, str] = {}
        # phrase_memory = {"I WILL KILL YOU": "Je vais te tuer"}
        self.name_variants: Dict[str, List[str]] = {}
        # name_variants = {"ARTHUR": ["Arthur", "arthur", "ARTHUR"]}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.phrase_memory = data.get("phrases", {})
                    self.name_variants = data.get("names", {})
            except Exception:
                pass

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({
                "phrases": self.phrase_memory,
                "names": self.name_variants,
            }, f, ensure_ascii=False, indent=2)

    def record_translation(self, en: str, fr: str):
        """Enregistre une traduction pour vérification future."""
        key = en.upper().strip()
        if len(key) >= 8:  # Seulement les phrases assez longues
            self.phrase_memory[key] = fr

    def record_name(self, name: str, variant: str):
        """Enregistre une variante de nom."""
        key = name.upper().strip()
        if key not in self.name_variants:
            self.name_variants[key] = []
        if variant not in self.name_variants[key]:
            self.name_variants[key].append(variant)

    def check_phrase(self, en: str) -> Optional[str]:
        """Vérifie si cette phrase a déjà été traduite (flashback, running gag)."""
        key = en.upper().strip()
        return self.phrase_memory.get(key)

    def check_name_consistency(self, translations: Dict[str, str]) -> List[str]:
        """Détecte des incohérences de noms dans un lot de traductions."""
        warnings = []
        for name_key, variants in self.name_variants.items():
            found_in_trad = set()
            for _idx, fr_text in translations.items():
                for variant in variants:
                    if variant in fr_text:
                        found_in_trad.add(variant)
            if len(found_in_trad) > 1:
                warnings.append(
                    f"⚠️ Nom '{name_key}' a {len(found_in_trad)} variantes dans ce chapitre: {found_in_trad}"
                )
        return warnings

    def get_repeated_phrases_block(self, texts: List[str], max_entries: int = 10) -> str:
        """Génère un bloc de phrases récurrentes pour le prompt."""
        relevant = {}
        for text in texts:
            key = text.upper().strip()
            if key in self.phrase_memory:
                relevant[text] = self.phrase_memory[key]

        if not relevant:
            return ""

        block = "PHRASES DÉJÀ TRADUITES (réutilise la même traduction) :\n"
        for en, fr in list(relevant.items())[:max_entries]:
            block += f'  "{en}" → "{fr}"\n'
        return block


# ═══════════════════════════════════════════════════════════════════════════
# NARRATIVE WEIGHT (POIDS NARRATIF)
# ═══════════════════════════════════════════════════════════════════════════

class NarrativeAnalyzer:
    """Classifie le type narratif de chaque bulle pour adapter le ton."""

    # Patterns pour détecter le type
    EMOTION_PATTERNS = [
        (r"\b(please|s'il\s+vous?\s+pla[iî]t|je\s+t'en\s+prie)\b", "supplication"),
        (r"\b(i'?m?\s+so\s+(happy|sorry|scared|sad))\b", "emotion_intense"),
        (r"\b(love|hate|miss|forgive)\b", "emotion"),
        (r"[!]{2,}", "exclamation"),
        (r"\.{3,}", "hesitation"),
    ]

    ACTION_PATTERNS = [
        (r"\b(kill|die|fight|attack|destroy|run|escape)\b", "action"),
        (r"\b(blood|wound|bleed|hurt|pain)\b", "violence"),
    ]

    HUMOR_PATTERNS = [
        (r"\b(child\s+abuse|maltraitance|ridicul|idiot|stupid)\b", "humor_exaggeration"),
        (r"\b(daaad|moooom|papaaaa|mamaaaan)\b", "humor_childish"),
    ]

    LORE_PATTERNS = [
        (r"\b(chairman|president|clan|group|organization|kingdom)\b", "lore"),
        (r"\b(prophecy|legend|ancient|power|ability)\b", "lore_fantasy"),
    ]

    @classmethod
    def classify(cls, text: str) -> Tuple[str, str]:
        """
        Classifie une bulle.
        
        Returns:
            (category, subcategory) parmi:
            - ("emotion", "supplication" | "intense" | "standard")
            - ("action", "violence" | "standard")
            - ("humor", "exaggeration" | "childish")
            - ("lore", "worldbuilding" | "standard")
            - ("dialogue", "standard")
        """
        text_lower = text.lower()

        for pattern, subcat in cls.HUMOR_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return "humor", subcat

        for pattern, subcat in cls.EMOTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return "emotion", subcat

        for pattern, subcat in cls.ACTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return "action", subcat

        for pattern, subcat in cls.LORE_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return "lore", subcat

        return "dialogue", "standard"

    @classmethod
    def get_tone_instruction(cls, category: str, subcategory: str, character: Optional[Character] = None) -> str:
        """Génère une instruction de ton pour le LLM."""
        base_tones = {
            ("emotion", "supplication"): "Ton suppliant, émouvant. Phrases courtes, sincères.",
            ("emotion", "emotion_intense"): "Émotion forte. Ponctuation expressive, vocabulaire senti.",
            ("emotion", "emotion"): "Ton émotionnel mais retenu.",
            ("emotion", "exclamation"): "Ton vif, énergie dans la voix.",
            ("emotion", "hesitation"): "Ton hésitant, points de suspension naturels.",
            ("action", "action"): "Ton sec, direct. Phrases percutantes.",
            ("action", "violence"): "Ton dur, cru. Vocabulaire de combat.",
            ("humor", "humor_exaggeration"): "Ton exagéré, dramatique-comique. Le personnage surjoue.",
            ("humor", "humor_childish"): "Ton enfantin, capricieux. Vocabulaire simple et plaintif.",
            ("lore", "lore"): "Ton informatif mais naturel. Termes du monde à respecter.",
            ("lore", "lore_fantasy"): "Ton solennel ou mystérieux selon contexte.",
            ("dialogue", "standard"): "Ton conversationnel naturel.",
        }

        tone = base_tones.get((category, subcategory), "Ton naturel.")

        if character:
            char_tones = {
                "confident": " Le personnage est sûr de lui, parle avec autorité.",
                "timid": " Le personnage est timide, phrases douces et hésitantes.",
                "aggressive": " Le personnage est agressif, vocabulaire dur et menaçant.",
                "comic": " Le personnage est comique, ton léger et exagéré.",
                "wise": " Le personnage est sage, ton posé et réfléchi.",
                "childish": " Le personnage est un enfant, vocabulaire simple et spontané.",
            }
            tone += char_tones.get(character.tone, "")

        return tone


# ═══════════════════════════════════════════════════════════════════════════
# SERIES DATABASE (gestionnaire principal)
# ═══════════════════════════════════════════════════════════════════════════

class SeriesDB:
    """Gestionnaire principal du mode série."""

    def __init__(self, series_dir: Path, slug: str, logger=None):
        self.base_dir = series_dir / slug
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.slug = slug
        self.logger = logger

        # Fichiers
        self.profile_path = self.base_dir / "series.json"
        self.glossary_path = self.base_dir / "glossary.json"
        self.consistency_path = self.base_dir / "consistency.json"
        self.chapters_dir = self.base_dir / "chapters"
        self.chapters_dir.mkdir(parents=True, exist_ok=True)

        # Modules
        self.profile = self._load_profile()
        self.glossary = Glossary(self.glossary_path)
        self.consistency = ConsistencyChecker(self.consistency_path)
        self.analyzer = NarrativeAnalyzer()

        # Chapitre courant
        self.current_chapter: Optional[ChapterContext] = None

        # Ensure files exist (create minimal defaults)
        try:
            if not self.profile_path.exists():
                # human-readable name from slug
                self.profile.name = str(self.slug).replace('_', ' ').title()
                self.save_profile()
            if not self.glossary_path.exists():
                self.glossary.save()
            if not self.consistency_path.exists():
                self.consistency.save()
        except Exception:
            # non-blocking: best-effort file creation
            pass
    def _load_profile(self) -> SeriesProfile:
        if self.profile_path.exists():
            try:
                with open(self.profile_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                characters = [Character(**c) for c in data.get("characters", [])]
                return SeriesProfile(
                    name=data.get("name", ""),
                    slug=data.get("slug", self.slug),
                    source_lang=data.get("source_lang", "en"),
                    target_lang=data.get("target_lang", "fr"),
                    genre=data.get("genre", ""),
                    tone_general=data.get("tone_general", ""),
                    characters=characters,
                    world_terms=data.get("world_terms", {}),
                )
            except Exception:
                pass
        return SeriesProfile(slug=self.slug)

    def save_profile(self):
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.profile.name,
            "slug": self.profile.slug,
            "source_lang": self.profile.source_lang,
            "target_lang": self.profile.target_lang,
            "genre": self.profile.genre,
            "tone_general": self.profile.tone_general,
            "characters": [asdict(c) for c in self.profile.characters],
            "world_terms": self.profile.world_terms,
        }
        with open(self.profile_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def init_series(self, name: str, genre: str = "", tone: str = ""):
        """Initialise une nouvelle série (première fois)."""
        self.profile.name = name
        self.profile.slug = self.slug
        self.profile.genre = genre
        self.profile.tone_general = tone
        self.save_profile()

        # Glossaire par défaut (expressions universelles manhwa)
        default_glossary = {
            "HONEY": "Chéri(e)",
            "DEAR": "Chéri(e)",
            "DAD": "Papa",
            "DAAAD": "Papaaaa",
            "MOM": "Maman",
            "MOOOOM": "Mamaaaan",
            "SIS": "Frangine",
            "BRO": "Frérot",
            "CHILD ABUSE": "C'est de la maltraitance !",
            "LOOK FORWARD TO IT": "Tu verras bien.",
            "I WILL": "J'y compte bien.",
            "SO PLEASE": "Alors s'il te plaît...",
            "OKAY": "D'accord.",
            "I SEE": "Je vois.",
            "IS THAT SO": "Ah bon ?",
            "LET'S GO": "Allons-y.",
            "WHAT DO YOU MEAN": "Qu'est-ce que tu veux dire ?",
            "HOW DARE YOU": "Comment oses-tu !",
            "SHUT UP": "Tais-toi !",
            "YOU BASTARD": "Espèce de salaud !",
        }
        self.glossary.add_batch(default_glossary)
        if self.logger:
            self.logger.info(f"   📚 Série initialisée: {name} ({self.slug})")
            self.logger.info(f"   📖 Glossaire: {len(default_glossary)} entrées par défaut")

    # ── Chapitre ──

    def start_chapter(self, chapter_number: int, previous_summary: str = ""):
        """Démarre un nouveau chapitre."""
        self.current_chapter = ChapterContext(
            chapter_number=chapter_number,
            previous_summary=previous_summary,
        )

        # Charger si existe déjà
        ch_path = self.chapters_dir / f"ch{chapter_number:03d}.json"
        if ch_path.exists():
            try:
                with open(ch_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.current_chapter.summary = data.get("summary", "")
                    self.current_chapter.timeline = data.get("timeline", "")
                    self.current_chapter.mood = data.get("mood", "")
                    self.current_chapter.translations = data.get("translations", {})
            except Exception:
                pass

        # Charger le résumé du chapitre précédent
        if chapter_number > 1 and not previous_summary:
            prev_path = self.chapters_dir / f"ch{chapter_number - 1:03d}.json"
            if prev_path.exists():
                try:
                    with open(prev_path, "r", encoding="utf-8") as f:
                        prev = json.load(f)
                        self.current_chapter.previous_summary = prev.get("summary", "")
                except Exception:
                    pass

        if self.logger:
            self.logger.info(f"   📖 Chapitre {chapter_number} chargé")
            if self.current_chapter.previous_summary:
                self.logger.info(f"   📝 Résumé ch. précédent disponible")

    def save_chapter(self):
        """Sauvegarde le chapitre courant."""
        if not self.current_chapter:
            return
        ch_path = self.chapters_dir / f"ch{self.current_chapter.chapter_number:03d}.json"
        data = {
            "chapter_number": self.current_chapter.chapter_number,
            "title": self.current_chapter.title,
            "summary": self.current_chapter.summary,
            "timeline": self.current_chapter.timeline,
            "mood": self.current_chapter.mood,
            "active_characters": self.current_chapter.active_characters,
            "previous_summary": self.current_chapter.previous_summary,
            "translations": self.current_chapter.translations,
        }
        ch_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ch_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def record_translation(self, index: str, en: str, fr: str):
        """Enregistre une traduction dans le chapitre courant + consistency."""
        if self.current_chapter:
            self.current_chapter.translations[str(index)] = {"en": en, "fr": fr}
        self.consistency.record_translation(en, fr)

    # ── Personnages ──

    def find_character(self, text: str) -> Optional[Character]:
        """Trouve le personnage qui parle (heuristique basée sur le texte)."""
        for char in self.profile.characters:
            if char.matches(text):
                return char
        return None

    def add_character(self, name: str, **kwargs):
        """Ajoute un personnage à la série."""
        char = Character(name=name, **kwargs)
        self.profile.characters.append(char)
        self.save_profile()

    # ── Construction du prompt contextuel ──

    def build_context_prompt(self, texts: List[str]) -> str:
        """
        Construit le BLOC DE CONTEXTE à injecter dans le prompt LLM.
        
        Combine : glossaire + personnages + résumé + ton + phrases récurrentes.
        """
        blocks = []

        # 1. Identité série
        if self.profile.name:
            series_line = f"SÉRIE: {self.profile.name}"
            if self.profile.genre:
                series_line += f" ({self.profile.genre})"
            if self.profile.tone_general:
                series_line += f" — Ton général: {self.profile.tone_general}"
            blocks.append(series_line)

        # 2. Contexte narratif (résumé chapitre précédent + courant)
        if self.current_chapter:
            if self.current_chapter.previous_summary:
                blocks.append(f"CHAPITRE PRÉCÉDENT: {self.current_chapter.previous_summary}")
            if self.current_chapter.summary:
                blocks.append(f"CE CHAPITRE: {self.current_chapter.summary}")
            if self.current_chapter.timeline:
                blocks.append(f"TIMELINE: {self.current_chapter.timeline}")
            if self.current_chapter.mood:
                blocks.append(f"AMBIANCE: {self.current_chapter.mood}")

        # 3. Personnages actifs
        char_lines = []
        for char in self.profile.characters:
            line = f"  - {char.name}"
            if char.gender != "?":
                line += f" ({char.gender})"
            if char.tone != "neutral":
                line += f" — ton: {char.tone}"
            if char.relationship:
                line += f" — rôle: {char.relationship}"
            char_lines.append(line)
        if char_lines:
            blocks.append("PERSONNAGES:\n" + "\n".join(char_lines))

        # 4. Glossaire pertinent
        glossary_block = self.glossary.get_prompt_block(texts)
        if glossary_block:
            blocks.append(glossary_block)

        # 5. Phrases récurrentes (flashbacks, running gags)
        repeat_block = self.consistency.get_repeated_phrases_block(texts)
        if repeat_block:
            blocks.append(repeat_block)

        # 6. Termes du monde
        if self.profile.world_terms:
            relevant_terms = {}
            all_text = " ".join(texts).upper()
            for en, fr in self.profile.world_terms.items():
                if en.upper() in all_text:
                    relevant_terms[en] = fr
            if relevant_terms:
                lines = [f'  "{k}" → "{v}"' for k, v in relevant_terms.items()]
                blocks.append("TERMES DU MONDE (ne pas traduire):\n" + "\n".join(lines))

        if not blocks:
            return ""

        return "\n\n".join(blocks)

    def build_bubble_annotations(self, texts: List[str]) -> Dict[int, Dict[str, str]]:
        """
        Pour chaque bulle, génère des annotations :
        - type narratif (action, émotion, humour, lore)
        - personnage probable
        - instruction de ton
        
        Returns: {index: {"type": "humor", "tone_hint": "...", "character": "..."}}
        """
        annotations = {}
        for i, text in enumerate(texts):
            cat, subcat = self.analyzer.classify(text)
            char = self.find_character(text)
            tone = self.analyzer.get_tone_instruction(cat, subcat, char)

            ann = {
                "type": f"{cat}/{subcat}",
                "tone_hint": tone,
            }
            if char:
                ann["character"] = char.name
                ann["character_tone"] = char.tone

            # Glossaire exact match → forcer
            exact = self.glossary.lookup(text.strip())
            if exact:
                ann["forced_translation"] = exact

            # Phrase déjà traduite (flashback)
            repeat = self.consistency.check_phrase(text)
            if repeat:
                ann["repeated_phrase"] = repeat

            annotations[i] = ann

        return annotations

    # ── Auto-génération du résumé (post-traduction) ──

    def auto_generate_summary(self, translations: Dict[str, Dict[str, str]]) -> str:
        """
        Génère automatiquement un résumé du chapitre à partir des traductions.
        Utilisé comme contexte pour les chapitres suivants.
        
        Note: Ce résumé sera affiné par le LLM si disponible, sinon heuristique.
        """
        all_fr = [v.get("fr", "") for v in translations.values() if v.get("fr")]
        if not all_fr:
            return ""

        # Heuristique simple : concaténer les bulles principales
        # (exclure onomatopées, phrases < 5 mots)
        meaningful = [
            t for t in all_fr
            if len(t.split()) >= 4 and not re.match(r"^[A-Z!.?]+$", t)
        ]

        if len(meaningful) > 10:
            # Garder début, milieu, fin
            summary_parts = meaningful[:3] + meaningful[len(meaningful)//2:len(meaningful)//2+2] + meaningful[-3:]
        else:
            summary_parts = meaningful

        return " [...] ".join(summary_parts[:8])

    # ── Finalisation chapitre ──

    def finalize_chapter(self):
        """Finalise le chapitre : sauvegarde, génère résumé, update consistency."""
        if not self.current_chapter:
            return

        # Auto-générer résumé si vide
        if not self.current_chapter.summary and self.current_chapter.translations:
            self.current_chapter.summary = self.auto_generate_summary(
                self.current_chapter.translations
            )

        self.save_chapter()
        self.consistency.save()
        self.glossary.save()

        if self.logger:
            self.logger.info(f"   ✅ Chapitre {self.current_chapter.chapter_number} finalisé")
            self.logger.info(f"   📖 {len(self.current_chapter.translations)} traductions enregistrées")
            self.logger.info(f"   📚 Glossaire: {len(self.glossary.entries)} entrées")

    # ── Rapport de cohérence ──

    def run_consistency_check(self) -> List[str]:
        """Lance une vérification de cohérence sur le chapitre courant."""
        warnings = []
        if self.current_chapter and self.current_chapter.translations:
            fr_map = {k: v.get("fr", "") for k, v in self.current_chapter.translations.items()}
            warnings.extend(self.consistency.check_name_consistency(fr_map))
        return warnings

    # ── Utilitaire d'affichage ──

    def get_status(self) -> Dict[str, Any]:
        return {
            "series": self.profile.name or self.slug,
            "characters": len(self.profile.characters),
            "glossary_entries": len(self.glossary.entries),
            "consistency_phrases": len(self.consistency.phrase_memory),
            "current_chapter": self.current_chapter.chapter_number if self.current_chapter else None,
        }
