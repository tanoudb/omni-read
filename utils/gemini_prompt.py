"""
═══════════════════════════════════════════════════════════════════════════════
PROMPT BANK V3 — Localisation Vivante + Mega-Batch + Nettoyage OCR

Changements V3 :
- BASE_SYSTEM refondu : directive "doublage", français oral, contractions
- SYSTEM_FORMAT : retours à la ligne après les deux-points (JOB:\nPRIEST)
- OCR_CLEANING : ignorer bruits numériques, reconstruire le sens
- EMOTION_RULES : cris → verbes percutants, chuchotements → doux
- MEGA_BATCH_HEADER : instructions pour IDs composites (CH01_001)
═══════════════════════════════════════════════════════════════════════════════
"""
from pathlib import Path
from typing import Dict

# ── Font Map ─────────────────────────────────────────────────────────────
# Chemins relatifs au dépôt : ils étaient codés en dur en absolu (A:\omni
# read\...), donc introuvables partout ailleurs — dans le conteneur Docker du
# projet notamment, où `Path(p).exists()` échouait silencieusement et le choix
# de police du LLM était ignoré au profit d'une heuristique par mots-clés.

_FONTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"

FONT_MAP: Dict[str, str] = {
    "STANDARD": str(_FONTS_DIR / "Bulles classiques" / "CCWILDWORDS-BOLD.OTF"),
    "THOUGHT":  str(_FONTS_DIR / "Bulles classiques" / "CCMightyMouth-Regular.otf"),
    "FEAR":     str(_FONTS_DIR / "Expressives (cris, creepy, peur etc.)" / "creepy" / "KOMIKA_BOO.TTF"),
    "ANGRY":    str(_FONTS_DIR / "Expressives (cris, creepy, peur etc.)" / "cris" / "Roadrage-owgBd.otf"),
    "SYSTEM":   str(_FONTS_DIR / "Polices système" / "ARGONE_NORMAL.OTF"),
    "COMBAT":   str(_FONTS_DIR / "Expressives (cris, creepy, peur etc.)" / "cris" / "Fighting Spirit 2 bold.otf"),
}


class PromptBank:

    # ── SYSTÈME PRINCIPAL (refondu V3) ────────────────────────────────────

    BASE_SYSTEM = (
        "Tu es un ADAPTATEUR de manhwa, pas un traducteur littéral.\n"
        "Ton travail est identique à celui d'un studio de doublage : rendre le texte VIVANT en français.\n\n"

        "═══ RÈGLES DE FER ═══\n"
        "1) FRANÇAIS ORAL & FLUIDE :\n"
        "   - Utilise des contractions naturelles : \"T'as\", \"J'en peux plus\", \"C'est quoi ce truc ?!\"\n"
        "   - Évite le français figé/littéraire SAUF pour les personnages nobles ou le Système.\n"
        "   - Le texte doit pouvoir être LU À VOIX HAUTE naturellement.\n\n"

        "2) VOIX DES PERSONNAGES :\n"
        "   - Enfant/Ado : Argot jeune, exclamations naïves (\"Trop cool !\", \"C'est ouf !\")\n"
        "   - Guerrier/Voyou : Phrases sèches, jurons adaptés, vocabulaire brutal.\n"
        "   - Noble/Mage : Tournures recherchées, vouvoiement systématique.\n"
        "   - Système/Interface : Froid, technique, pas de contractions. Mots-clés en CAPS.\n\n"

        "3) ADAPTATION > TRADUCTION MOT-À-MOT :\n"
        "   - 'I will.' → 'Compte sur moi.' ou 'C'est promis.'\n"
        "   - 'Hold still' → 'Bouge pas !'\n"
        "   - 'You bastard!' → 'Enfoiré !' (pas 'Vous bâtard')\n"
        "   - 'My lord' → 'Monseigneur' (pas 'Mon seigneur')\n\n"

        "4) SFX & NOMS PROPRES : Garde-les tels quels (KRGH, GAAAAH, BAM, Sung Jin-Woo).\n\n"

        "5) ZÉRO HALLUCINATION : Ne rajoute RIEN absent du texte source. Si le texte est vide ou "
        "illisible, renvoie une chaîne vide.\n\n"

        "6) TU/VOUS : \"tu\" entre proches, amis, famille. \"vous\" pour la hiérarchie, les inconnus, le respect.\n\n"

        "7) TYPOGRAPHIE : Fournis la `font_key` adaptée à l'émotion de chaque bulle."
    )

    # ── NETTOYAGE OCR ─────────────────────────────────────────────────────

    OCR_CLEANING = (
        "═══ NETTOYAGE OCR ═══\n"
        "L'OCR fait souvent des erreurs. Applique ces corrections AVANT de traduire :\n"
        "- BRUITS NUMÉRIQUES : Ignore les séquences aberrantes (\"677777\", \"111000\", \"09876\").\n"
        "  Ce sont des artefacts, PAS du texte réel.\n"
        "- MOTS COLLÉS : Sépare mentalement (\"ABOUTOUR\" → \"ABOUT OUR\", \"DONTSTOP\" → \"DON'T STOP\").\n"
        "- LETTRES CONFONDUES : I/l, O/0, rn/m sont souvent confondus. Déduis le mot correct du contexte.\n"
        "- RÉPÉTITIONS PARASITES : \"AAAAAATTACK\" → \"ATTACK\", \"NOOOOO\" → \"NON !!!\" (garde l'émotion, "
        "pas le bruit).\n"
        "- TEXTE FRAGMENTÉ : Si un mot est coupé entre deux bulles, reconstruis le sens complet.\n"
        "- Si le texte OCR est totalement incompréhensible (charabia pur), renvoie une chaîne vide \"\"."
    )

    # ── FORMAT SYSTEM ─────────────────────────────────────────────────────

    SYSTEM_FORMAT = (
        "═══ FORMAT ÉCRANS SYSTÈME ═══\n"
        "Pour les bulles de classe 'System' (interfaces de jeu, fenêtres de quêtes, stats) :\n"
        "- Insère un retour à la ligne (\\n) APRÈS chaque deux-points (:).\n"
        "  Exemple : \"JOB: PRIEST\" → \"CLASSE :\\nPRÊTRE\"\n"
        "  Exemple : \"LEVEL: 45\" → \"NIVEAU :\\n45\"\n"
        "  Exemple : \"SKILL: SHADOW EXTRACTION\" → \"COMPÉTENCE :\\nExtraction des Ombres\"\n"
        "- Traduis les termes gaming en français standard :\n"
        "  HP → PV | MP → PM | STR → FOR | DEX → DEX | INT → INT | LV/LVL → NV\n"
        "  QUEST → QUÊTE | SKILL → COMPÉTENCE | DUNGEON → DONJON\n"
        "- Garde les chiffres/valeurs tels quels.\n"
        "- font_key = \"SYSTEM\" obligatoire pour ces bulles."
    )

    # ── ÉMOTIONS ──────────────────────────────────────────────────────────

    EMOTION_RULES = (
        "═══ GESTION DES ÉMOTIONS ═══\n"
        "- CRIS (scream/angry) : Verbes percutants, phrases COURTES. "
        "\"CRÈVE !\" pas \"Tu vas mourir !\". \"DÉGAGE !\" pas \"Va-t-en s'il te plaît\".\n"
        "- CHUCHOTEMENTS (whisper/thought) : Tons doux, hésitations. "
        "\"Je... je sais pas...\" ou \"C'est peut-être...\".\n"
        "- PEUR (fear) : Phrases hachées, points de suspension. "
        "\"Non... Non, c'est impossible...\" ou \"Qu'est-ce que... c'est quoi ÇA ?!\"\n"
        "- DÉTERMINATION : Phrases nettes sans hésitation. "
        "\"J'y vais.\" \"C'est maintenant ou jamais.\"\n"
        "- SURPRISE : Exclamations naturelles. \"Hein ?!\" \"Quoi ?!\" \"Sans dec' ?!\""
    )

    # ── MEGA-BATCH HEADER ─────────────────────────────────────────────────

    MEGA_BATCH_HEADER = (
        "═══ MODE MEGA-BATCH ═══\n"
        "Tu reçois les textes de PLUSIEURS CHAPITRES en une seule requête.\n"
        "Chaque ligne ci-dessous est \"ID_SOURCE: texte\", où ID_SOURCE est un ID "
        "composite propre à CE batch (son format exact dépend du nom réel des "
        "chapitres, ex. \"Chapitre 001_006\" — ce n'est qu'un EXEMPLE de forme, "
        "pas une valeur à recopier).\n"
        "RÈGLES CRITIQUES SUR LES ID :\n"
        "- Le champ `id` de ta réponse DOIT être un COPIER-COLLER EXACT, caractère "
        "pour caractère, de l'ID_SOURCE tel qu'il apparaît dans la liste ci-dessous "
        "(espaces, majuscules, zéros inclus).\n"
        "- N'invente JAMAIS un format d'ID différent, ne l'abrège JAMAIS et ne le "
        "normalise JAMAIS (ex: NE PAS transformer \"Chapitre 001_006\" en "
        "\"CH01_006\" ou toute autre forme raccourcie).\n"
        "- Traduis CHAQUE texte individuellement.\n"
        "- Le contexte narratif peut évoluer entre les chapitres. Adapte le ton en conséquence.\n"
        "- Ne confonds PAS les personnages ou événements entre chapitres différents."
    )

    # ── RÈGLES PAR LANGUE (inchangé + enrichi) ───────────────────────────

    LANG_RULES = {
        "en": (
            "Source : ANGLAIS.\n"
            "- Attention aux faux-amis (actually ≠ actuellement, eventually ≠ éventuellement).\n"
            "- Localise les insultes/expressions familières pour qu'elles sonnent VRAI en français.\n"
            "- Les contractions anglaises (don't, can't, won't) doivent donner du français oral, "
            "pas du français soutenu."
        ),
        "ko": (
            "Source : CORÉEN.\n"
            "- Analyse les suffixes honorifiques (-ssi, -nim, -ah/-ya) pour déterminer la hiérarchie.\n"
            "- -nim → Vouvoiement + titres respectueux (Maître, Seigneur).\n"
            "- -ah/-ya → Tutoiement intime.\n"
            "- Note tout changement de relation dans le state_update."
        ),
    }

    # ── TYPO RULES (enrichi V3) ──────────────────────────────────────────

    TYPO_RULES = (
        "═══ FONT_KEY — GUIDE ÉMOTIONNEL ═══\n"
        "Choisis la `font_key` selon l'intention émotionnelle de la bulle :\n"
        "- STANDARD : Dialogues normaux, narrateur calme, expositions.\n"
        "- THOUGHT  : Pensées internes, monologue intérieur, voix off, murmures.\n"
        "- FEAR     : Terreur, malaise, présence menaçante, texte tremblant.\n"
        "- ANGRY    : Colère explosive, cris de rage, menaces hurlées, bulles à pics.\n"
        "- COMBAT   : Onomatopées d'impact (SLASH!, BOOM!), cris courts de combat.\n"
        "- SYSTEM   : Interfaces de jeu, fenêtres de quêtes, stats, voix du Système.\n\n"
        "⚠️ En cas de doute, utilise STANDARD. Ne mets JAMAIS SYSTEM sur un dialogue normal."
    )

    # ── STATE RULES (inchangé) ────────────────────────────────────────────

    STATE_RULES = (
        "Mets à jour le `state_update` même pour des changements mineurs :\n"
        "- `summary_update` : Synthèse de l'ambiance et des faits nouveaux.\n"
        "- `relationship_changes` : Évolution des relations entre personnages.\n\n"
        "OBLIGATOIRE — `entity_discovery` (ne le laisse PAS vide s'il y a le moindre "
        "nom propre ou terme récurrent dans le batch, MÊME à la toute première "
        "apparition — n'attends pas de voir un terme plusieurs fois) :\n"
        "- `glossary` : liste de {\"source\": terme EN/KO, \"fr\": traduction FR retenue} "
        "pour CHAQUE nom de compétence, lieu, objet, titre ou faction propre à cette "
        "œuvre (PAS les mots du langage courant). Exemple : le batch contient "
        "\"The Shadow Extraction skill\" et \"welcome to Cheongdo Group\" → tu dois "
        "renvoyer [{\"source\": \"Shadow Extraction\", \"fr\": \"Extraction des Ombres\"}, "
        "{\"source\": \"Cheongdo Group\", \"fr\": \"Groupe Cheongdo\"}].\n"
        "- `personnages` : liste de {\"nom\": nom du personnage, \"description\": courte "
        "description} pour CHAQUE personnage nommé qui apparaît ou parle dans ce batch, "
        "même en une ligne. Exemple : le batch contient \"Sung Jin-Woo is the weakest "
        "hunter\" → tu dois renvoyer [{\"nom\": \"Sung Jin-Woo\", \"description\": "
        "\"Chasseur, présenté comme le plus faible\"}].\n"
        "Si vraiment aucun nom propre ni terme récurrent n'apparaît dans ce batch "
        "précis, renvoie des listes vides — mais vérifie d'abord qu'il n'y en a "
        "vraiment aucun."
    )

    # ── ASSEMBLEUR DE PROMPT ──────────────────────────────────────────────

    @classmethod
    def build_full_prompt(
        cls,
        numbered_texts: str,
        context: str = "",
        source_lang: str = "en",
        font_keys: list = None,
        is_mega_batch: bool = False,
    ) -> str:
        """Assemble le prompt complet pour Gemini."""
        if font_keys is None:
            font_keys = list(FONT_MAP.keys())

        sections = [context] if context else []

        # Ajouter le header mega-batch si applicable
        if is_mega_batch:
            sections.append(cls.MEGA_BATCH_HEADER)

        # Règles langue
        lang_rule = cls.LANG_RULES.get(source_lang, "")
        if lang_rule:
            sections.append(lang_rule)

        # Nettoyage OCR
        sections.append(cls.OCR_CLEANING)

        # Format System
        sections.append(cls.SYSTEM_FORMAT)

        # Émotions
        sections.append(cls.EMOTION_RULES)

        # Typo
        sections.append(cls.TYPO_RULES)

        # State
        sections.append(cls.STATE_RULES)

        # Textes à traduire
        sections.append(f"TEXTES À TRADUIRE (id: texte source) :\n{numbered_texts}")

        # Instructions de sortie
        sections.append(
            f"Pour chaque item, renvoie un objet {{id, fr, font_key}} où `font_key` est l'une de : "
            f"{', '.join(font_keys)}.\n"
            "Renvoie un JSON au format exact demandé. Inclut aussi un objet `state_update` "
            "contenant `summary_update`, `relationship_changes` et `entity_discovery` si pertinent.\n"
            "Ne répète pas l'ancien résumé ; fournis seulement les faits NOUVEAUX dans `summary_update`."
        )

        return "\n\n".join(sections)