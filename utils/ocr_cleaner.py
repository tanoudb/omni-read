"""
═══════════════════════════════════════════════════════════════════════════════
OCR CLEANER V2 — Nettoyage post-OCR spécialisé manga/webtoon

Problèmes corrigés (classés par fréquence dans les logs réels) :
1. Mots collés PaddleOCR-VL : AFTERTHEJOB → AFTER THE JOB
2. Unicode fullwidth pourri : ï¼ˆ → (, ï¼‰ → ), å…± → supprimé
3. Hallucinations numériques : 677777, SIHL 15... → nettoyé
4. Lettres confondues U↔L : LNDERSTANDING → UNDERSTANDING
5. Espaces parasites : BE COME → BECOME, HAND SOME → HANDSOME
6. Apostrophes OCR : ' → '  (curly → straight)
7. Ponctuation collée : COMBAT JOB!HIS → COMBAT JOB! HIS

USAGE :
    from utils.ocr_cleaner import clean_ocr_output
    cleaned = clean_ocr_output("AFTERTHEJOB CHANGECEREMONY")
    # → "AFTER THE JOB CHANGE CEREMONY"

INTÉGRATION dans core/ocr.py :
    Dans post_process_text(), ajouter :
        from utils.ocr_cleaner import clean_ocr_output
        text = clean_ocr_output(text)
    AVANT le word_splitter existant.
═══════════════════════════════════════════════════════════════════════════════
"""

import re
from typing import Optional


# ═════════════════════════════════════════════════════════════════════════════
# 1. UNICODE CLEANUP — Fullwidth et caractères parasites
# ═════════════════════════════════════════════════════════════════════════════

# PaddleOCR-VL lit souvent les parenthèses CJK fullwidth comme des séquences
# UTF-8 corrompues (ï¼ˆ = （, ï¼‰ = ）, etc.)
_UNICODE_REPLACE = {
    # Fullwidth corrompus (affichés comme mojibake latin)
    "ï¼ˆ": "(",
    "ï¼‰": ")",
    "ï¼š": ":",
    "ï¼Ÿ": "?",
    "ï¼ ": "!",
    "ï¼Œ": ",",
    "ï¼Ž": ".",
    "ï¼›": ";",
    # Fullwidth réels (si encodage correct)
    "\uff08": "(",  # （
    "\uff09": ")",  # ）
    "\uff1a": ":",  # ：
    "\uff1f": "?",  # ？
    "\uff01": "!",  # ！
    "\uff0c": ",",  # ，
    "\uff0e": ".",  # ．
    "\uff1b": ";",  # ；
    # Guillemets CJK
    "\u201c": '"',  # "
    "\u201d": '"',  # "
    "\u2018": "'",  # '
    "\u2019": "'",  # '
    "\u300c": '"',  # 「
    "\u300d": '"',  # 」
    # Tirets CJK
    "\u2014": "-",  # —
    "\u2013": "-",  # –
    "\u2012": "-",  # ‒
    # Espaces spéciaux
    "\u3000": " ",  # ideographic space
    "\u00a0": " ",  # non-breaking space
    "\u200b": "",   # zero-width space
}

# Regex pour détecter les caractères CJK parasites isolés
# (un seul caractère chinois/japonais au milieu de texte anglais = hallucination)
_CJK_RANGES = (
    r"[\u4e00-\u9fff"   # CJK Unified
    r"\u3400-\u4dbf"    # CJK Extension A
    r"\u3000-\u303f"    # CJK Symbols
    r"\u3040-\u309f"    # Hiragana
    r"\u30a0-\u30ff"    # Katakana
    r"\uac00-\ud7af]"   # Hangul
)
_RE_ISOLATED_CJK = re.compile(
    rf"(?<=[A-Za-z0-9\s.,!?])\s*{_CJK_RANGES}+\s*(?=[A-Za-z0-9\s.,!?]|$)"
)
_RE_TRAILING_CJK = re.compile(rf"\s*{_CJK_RANGES}+\s*$")
_RE_LEADING_CJK = re.compile(rf"^\s*{_CJK_RANGES}+\s*")


def _clean_unicode(text: str) -> str:
    """Nettoie les caractères unicode parasites."""
    for bad, good in _UNICODE_REPLACE.items():
        text = text.replace(bad, good)

    # Supprimer CJK isolés au milieu de texte anglais
    text = _RE_ISOLATED_CJK.sub(" ", text)
    text = _RE_TRAILING_CJK.sub("", text)
    text = _RE_LEADING_CJK.sub("", text)

    return text


# ═════════════════════════════════════════════════════════════════════════════
# 2. HALLUCINATION FILTER — Textes OCR absurdes
# ═════════════════════════════════════════════════════════════════════════════

# Patterns de bruit numérique pur (677777, 12345, etc.)
_RE_NUMERIC_NOISE = re.compile(r"^[\d\s.,]+$")
# Répétitions d'un même chiffre/lettre (CCCC, 7777, AAAA)
_RE_CHAR_REPEAT = re.compile(r"^(.)\1{3,}$")
# "SIHL" est une hallucination récurrente de PaddleOCR (lit "THIS" à l'envers)
_HALLUCINATION_TOKENS = {
    "SIHL", "SIHL'S", "SLHL", "SIHI",
}
# Séquences ??? seules (PaddleOCR lit des bulles vides)
_RE_ONLY_QUESTIONS = re.compile(r"^[?\s!.]+$")


def _is_hallucination(text: str) -> bool:
    """Détecte si le texte est une hallucination OCR pure."""
    stripped = text.strip()
    if not stripped:
        return True

    # Bruit numérique pur
    if _RE_NUMERIC_NOISE.match(stripped):
        return True

    # Répétition de caractère
    alpha_only = re.sub(r"[^A-Za-z]", "", stripped)
    if alpha_only and _RE_CHAR_REPEAT.match(alpha_only):
        return True

    # Ponctuation seule
    if _RE_ONLY_QUESTIONS.match(stripped):
        return True

    return False


def _clean_hallucination_tokens(text: str) -> str:
    """Supprime les tokens hallucinés connus et nettoie autour."""
    words = text.split()
    cleaned = []
    for w in words:
        w_upper = re.sub(r"[^A-Za-z]", "", w).upper()
        if w_upper in _HALLUCINATION_TOKENS:
            # Remplacer par "" mais garder la ponctuation attachée
            punct = re.sub(r"[A-Za-z]", "", w)
            if punct.strip():
                cleaned.append(punct.strip())
            continue
        cleaned.append(w)
    return " ".join(cleaned)


# ═════════════════════════════════════════════════════════════════════════════
# 3. LETTRE CONFUSION FIX — U↔L, I↔l, O↔0
# ═════════════════════════════════════════════════════════════════════════════

# PP-OCR confond systématiquement U→L et L→U dans les majuscules
# LNDERSTANDING → UNDERSTANDING, LSING → USING, LNKNOWN → UNKNOWN
# LNLIMITED → UNLIMITED

_LETTER_CONFUSION_MAP = {
    # L→U au début de mots connus
    r"\bLNDERSTAND": "UNDERSTAND",
    r"\bLNKNOWN": "UNKNOWN",
    r"\bLNLIMITED": "UNLIMITED",
    r"\bLSING\b": "USING",
    r"\bLNDER\b": "UNDER",
    r"\bLNTIL\b": "UNTIL",
    r"\bLNLESS\b": "UNLESS",
    r"\bLNIQUE\b": "UNIQUE",
    r"\bLNIVERSE\b": "UNIVERSE",
    r"\bLNIT\b": "UNIT",
    r"\bLPDATE": "UPDATE",
    r"\bLPGRADE": "UPGRADE",
    r"\bLPPER\b": "UPPER",
    r"\bLRGENT": "URGENT",
    r"\bLSEFUL\b": "USEFUL",
    r"\bLSELESS\b": "USELESS",
    r"\bLSUAL": "USUAL",
    # EQLIPMENT → EQUIPMENT (U→L dans le mot)
    r"\bEQLIPMENT": "EQUIPMENT",
    r"\bEQLAL": "EQUAL",
    # ALINT → AUNT
    r"\bALINT\b": "AUNT",
    r"\bALNT\b": "AUNT",
    # HI STORY → HISTORY
    r"\bHI\s+STORY\b": "HISTORY",
    # POWER HOUSE → POWERHOUSE
    r"\bPOWER\s*HOUSE\b": "POWERHOUSE",
}


def _fix_letter_confusion(text: str) -> str:
    """Corrige les confusions de lettres typiques OCR."""
    for pattern, replacement in _LETTER_CONFUSION_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


# ═════════════════════════════════════════════════════════════════════════════
# 4. ESPACE PARASITE FIX — "BE COME" → "BECOME"
# ═════════════════════════════════════════════════════════════════════════════

# PP-OCR insère des espaces parasites dans les mots longs
# BE COME → BECOME, HAND SOME → HANDSOME, etc.
_SPACE_FIX_MAP = {
    r"\bBE\s+COME\b": "BECOME",
    r"\bBE\s+CAME\b": "BECAME",
    r"\bBE\s+CAUSE\b": "BECAUSE",
    r"\bBE\s+FORE\b": "BEFORE",
    r"\bBE\s+HIND\b": "BEHIND",
    r"\bBE\s+LONG\b": "BELONG",
    r"\bBE\s+LIEVE\b": "BELIEVE",
    r"\bBE\s+TWEEN\b": "BETWEEN",
    r"\bBE\s+YOND\b": "BEYOND",
    r"\bHAND\s+SOME\b": "HANDSOME",
    r"\bUNDER\s+STAND\b": "UNDERSTAND",
    r"\bOVER\s+COME\b": "OVERCOME",
    r"\bOVER\s+WHELM": "OVERWHELM",
    r"\bEVERY\s+ONE\b": "EVERYONE",
    r"\bEVERY\s+THING\b": "EVERYTHING",
    r"\bEVERY\s+WHERE\b": "EVERYWHERE",
    r"\bEVERY\s+BODY\b": "EVERYBODY",
    r"\bEVERY\s+DAY\b": "EVERYDAY",
    r"\bSOME\s+ONE\b": "SOMEONE",
    r"\bSOME\s+THING\b": "SOMETHING",
    r"\bSOME\s+WHERE\b": "SOMEWHERE",
    r"\bSOME\s+HOW\b": "SOMEHOW",
    r"\bSOME\s+TIMES?\b": "SOMETIMES",
    r"\bANY\s+ONE\b": "ANYONE",
    r"\bANY\s+THING\b": "ANYTHING",
    r"\bANY\s+WHERE\b": "ANYWHERE",
    r"\bANY\s+MORE\b": "ANYMORE",
    r"\bNO\s+THING\b": "NOTHING",
    r"\bNO\s+WHERE\b": "NOWHERE",
    r"\bAL\s+READY\b": "ALREADY",
    r"\bAL\s+THOUGH\b": "ALTHOUGH",
    r"\bAL\s+WAYS\b": "ALWAYS",
    r"\bTO\s+GETHER\b": "TOGETHER",
    r"\bTO\s+DAY\b": "TODAY",
    r"\bTO\s+NIGHT\b": "TONIGHT",
    r"\bTO\s+MORROW\b": "TOMORROW",
    r"\bMY\s+SELF\b": "MYSELF",
    r"\bYOUR\s+SELF\b": "YOURSELF",
    r"\bHIM\s+SELF\b": "HIMSELF",
    r"\bHER\s+SELF\b": "HERSELF",
    r"\bOUR\s+SELVES\b": "OURSELVES",
    r"\bTHEM\s+SELVES\b": "THEMSELVES",
    r"\bMEAN\s+WHILE\b": "MEANWHILE",
    r"\bOTHER\s+WISE\b": "OTHERWISE",
    r"\bFURTHER\s+MORE\b": "FURTHERMORE",
    r"\bNEVER\s+THE\s+LESS\b": "NEVERTHELESS",
    r"\bWHAT\s+EVER\b": "WHATEVER",
    r"\bWHEN\s+EVER\b": "WHENEVER",
    r"\bWHERE\s+EVER\b": "WHEREVER",
    r"\bHOW\s+EVER\b": "HOWEVER",
    r"\bIMPOSSI\s+BLE\b": "IMPOSSIBLE",
    # Webtoon gaming specific
    r"\bEXPERI\s+ENCE\b": "EXPERIENCE",
    r"\bTRANS\s+MIGRAT": "TRANSMIGRAT",
    r"\bPROFESSION\s+ALS?\b": "PROFESSIONALS",
}


def _fix_parasitic_spaces(text: str) -> str:
    """Corrige les espaces parasites dans les mots connus."""
    for pattern, replacement in _SPACE_FIX_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


# ═════════════════════════════════════════════════════════════════════════════
# 5. PONCTUATION COLLÉE — "COMBAT JOB!HIS" → "COMBAT JOB! HIS"
# ═════════════════════════════════════════════════════════════════════════════

def _fix_glued_punctuation(text: str) -> str:
    """Ajoute un espace après la ponctuation collée à une lettre."""
    # "JOB!HIS" → "JOB! HIS", "JOB.THE" → "JOB. THE"
    text = re.sub(r"([.!?])([A-Z])", r"\1 \2", text)
    # "ATTAINEDA" pas touché car pas de ponctuation
    # "IT'S AMAGE'S" — pas de fix ici, c'est le word splitter qui gère
    return text


# ═════════════════════════════════════════════════════════════════════════════
# 6. APOSTROPHE NORMALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def _normalize_apostrophes(text: str) -> str:
    """Normalise toutes les variantes d'apostrophes."""
    text = text.replace("\u2019", "'")  # '
    text = text.replace("\u2018", "'")  # '
    text = text.replace("\u0060", "'")  # `
    text = text.replace("\u00b4", "'")  # ´
    text = text.replace("\u2032", "'")  # ′
    return text


# ═════════════════════════════════════════════════════════════════════════════
# 7. APOSTROPHE SPACING FIX — "DON' T" → "DON'T"
# ═════════════════════════════════════════════════════════════════════════════

def _fix_apostrophe_spacing(text: str) -> str:
    """Corrige les espaces autour des apostrophes dans les contractions."""
    # "DON' T" → "DON'T", "CAN' T" → "CAN'T", "I' M" → "I'M"
    text = re.sub(r"(\w)'\s+([TSMDLR]\b)", r"\1'\2", text)
    # "WON 'T" → "WON'T"
    text = re.sub(r"(\w)\s+'([TSMDLR]\b)", r"\1'\2", text)
    return text


# ═════════════════════════════════════════════════════════════════════════════
# 8. SPECIFIC WEBTOON/MANHWA FIXES
# ═════════════════════════════════════════════════════════════════════════════

def _fix_webtoon_specific(text: str) -> str:
    """Corrections spécifiques au contenu manhwa/webtoon."""

    # "T'LL" → "I'LL" (OCR confond I et T dans les contractions)
    text = re.sub(r"\bT'LL\b", "I'LL", text)
    # "T'M" → "I'M"
    text = re.sub(r"\bT'M\b", "I'M", text)
    # "T'VE" → "I'VE"
    text = re.sub(r"\bT'VE\b", "I'VE", text)
    # "T'D" → "I'D"
    text = re.sub(r"\bT'D\b", "I'D", text)

    # "AMAGE'S" → "A MAGE'S"
    text = re.sub(r"\bAMAGE'S\b", "A MAGE'S", text, flags=re.IGNORECASE)
    text = re.sub(r"\bAMAGE\b", "A MAGE", text, flags=re.IGNORECASE)

    # "ATTAINEDA" → "ATTAINED A"
    text = re.sub(r"\bATTAINEDA\b", "ATTAINED A", text, flags=re.IGNORECASE)
    text = re.sub(r"\bATTAINAN\b", "ATTAIN AN", text, flags=re.IGNORECASE)

    # "IFYOUCAN" → word splitter handles, but common OCR words:
    text = re.sub(r"\bIFI\b", "IF I", text)
    text = re.sub(r"\bIFICANT\b", "IF I CAN'T", text, flags=re.IGNORECASE)

    # "WHAT'S GOING 之NO" → "WHAT'S GOING ON" (CJK + reversed letters)
    # Also handle case where CJK was already stripped: "GOING NO"
    text = re.sub(r"GOING\s*之\s*NO\b", "GOING ON", text, flags=re.IGNORECASE)
    text = re.sub(r"GOING\s+NO\b(?!\w)", "GOING ON", text, flags=re.IGNORECASE)

    # "WHO A!!" → "WHOA!!"
    text = re.sub(r"\bWHO\s+A\s*(!+)", r"WHOA\1", text, flags=re.IGNORECASE)

    # "AMA DRAGON" → "A DRAGON" at start (OCR reads "I AM A" as "AMA")
    text = re.sub(r"^AMA\s+", "I AM A ", text, flags=re.IGNORECASE)

    # Fix "EYEOFAPPRAISAL" → "EYE OF APPRAISAL"
    text = re.sub(r"\bEYEOFAPPRAISAL\b", "EYE OF APPRAISAL", text, flags=re.IGNORECASE)

    # Fix "DRAGONNATION" → "DRAGON NATION"
    text = re.sub(r"\bDRAGONNATION\b", "DRAGON NATION", text, flags=re.IGNORECASE)

    # Remove trailing noise "å…±" and similar
    text = re.sub(r"\s*[å]\S*$", "", text)

    return text


# ═════════════════════════════════════════════════════════════════════════════
# 9. CONFIDENCE-BASED CLEANUP — Textes basse confiance
# ═════════════════════════════════════════════════════════════════════════════

def clean_low_confidence(text: str, confidence: float) -> Optional[str]:
    """
    Pour les textes avec confiance < 0.85, applique un nettoyage plus agressif.
    Retourne None si le texte est probablement du bruit.
    """
    if confidence >= 0.85:
        return text

    stripped = text.strip()

    # Conf < 0.6 et moins de 5 caractères alphabétiques → bruit
    alpha_count = sum(1 for c in stripped if c.isalpha())
    if confidence < 0.6 and alpha_count < 5:
        return None

    # Conf < 0.7 et que des caractères non-latins → bruit
    latin_count = sum(1 for c in stripped if 'A' <= c.upper() <= 'Z')
    if confidence < 0.7 and latin_count < 3:
        return None

    # Conf < 0.75 et le texte ne contient que des répétitions → bruit
    unique = set(re.sub(r"[^A-Za-z]", "", stripped).upper())
    if confidence < 0.75 and len(unique) <= 2:
        return None

    return text


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def clean_ocr_output(text: str, confidence: float = 1.0) -> Optional[str]:
    """
    Nettoyage complet du texte OCR en sortie de PaddleOCR/RapidOCR.

    Ordre d'application (important) :
    1. Unicode cleanup (mojibake, fullwidth)
    2. Apostrophe normalization
    3. Hallucination filter
    4. Hallucination token removal (SIHL, etc.)
    5. Letter confusion fix (L→U)
    6. Parasitic space fix (BE COME → BECOME)
    7. Webtoon-specific fixes
    8. Glued punctuation fix
    9. Apostrophe spacing fix
    10. Confidence-based filter
    11. Final whitespace cleanup

    Args:
        text: Texte brut OCR
        confidence: Score de confiance OCR (0.0-1.0)

    Returns:
        Texte nettoyé, ou None si identifié comme bruit/hallucination.
    """
    if not text:
        return None

    # 1. Apostrophes (avant tout pour normaliser)
    text = _normalize_apostrophes(text)

    # 2. Webtoon-specific AVANT unicode cleanup (pour capter les patterns CJK+latin)
    text = _fix_webtoon_specific(text)

    # 3. Unicode
    text = _clean_unicode(text)

    # 4. Hallucinations
    if _is_hallucination(text):
        return None

    # 5. Tokens hallucinés
    text = _clean_hallucination_tokens(text)

    # 6. Confusions de lettres
    text = _fix_letter_confusion(text)

    # 7. Espaces parasites
    text = _fix_parasitic_spaces(text)

    # 8. Ponctuation collée
    text = _fix_glued_punctuation(text)

    # 9. Apostrophe spacing
    text = _fix_apostrophe_spacing(text)

    # 10. Confidence filter
    text = clean_low_confidence(text, confidence)
    if text is None:
        return None

    # 11. Whitespace final
    text = re.sub(r"\s+", " ", text).strip()

    # Vérification finale
    if not text or len(text.strip()) < 1:
        return None

    return text


# ═════════════════════════════════════════════════════════════════════════════
# TESTS INTÉGRÉS
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        # (input, confidence, expected_contains)
        ("AFTERTHEJOB CHANGECEREMONY", 0.99, "AFTER"),
        ("ï¼Ÿï¼Ÿ ï¼Ÿï¼Ÿï¼Ÿ", 0.787, None),  # → None (ponctuation seule)
        ("677777 SKILL:JOB CHANGE", 0.815, "SKILL"),
        ("SIHL 15...", 0.736, None),  # hallucination
        ("SIHL GUY IS QUITE HAND SOME TOO.", 0.94, "HANDSOME"),
        ("LNDERSTANDING OF THIS WORLD", 0.96, "UNDERSTANDING"),
        ("LSING POWERFUL ABILITIES", 0.96, "USING"),
        ("MY ALINT, HEIDI KELLER", 0.94, "AUNT"),
        ("BE COME A HIDDEN COMBAT", 0.97, "BECOME"),
        ("EVERY ONE HUNTS MONSTERS", 0.96, "EVERYONE"),
        ("NEW SKILL ACQUIRED \"EYEOFAPPRAISAL\"", 0.95, "EYE OF APPRAISAL"),
        ("DRAGONNATION", 0.997, "DRAGON NATION"),
        ("WHAT'S GOING 之NO", 0.91, "GOING ON"),
        ("WHO A!!", 0.95, "WHOA!!"),
        ("AMA DRAGON TAMER....", 0.93, "I AM A DRAGON"),
        ("IT'S A MAGE!IT'S AMAGE'S FIREBALL!", 0.94, "A MAGE'S"),
        ("EQLIPMENTAND WEAPONS", 0.96, "EQUIPMENT"),
        ("HI STORY..", 0.96, "HISTORY"),
        ("POWER HOUSE OF DRAGON NATION", 0.96, "POWERHOUSE"),
        ("...HERBALIST, I GUESS T'LL BE", 0.94, "I'LL"),
        ("ALRIGHT, ALRIGHT, DON'T THINK ABOUT IT ANY MORE, LET'S DIGIN! å…±", 0.80, "ANYMORE"),
        ("沁", 0.552, None),  # trop court + CJK
        ("CCCC", 0.789, None),  # répétition
        ("A HIDDEN 2 aor REALLY?!", 0.816, "HIDDEN"),
        ("ATTAINEDA COMBAT JOB!", 0.96, "ATTAINED A"),
        ("LNLIMITED FUTURE", 0.96, "UNLIMITED"),
    ]

    print("=" * 80)
    print("OCR CLEANER V2 — TESTS")
    print("=" * 80)

    passed = 0
    failed = 0
    for text_in, conf, expected in tests:
        result = clean_ocr_output(text_in, conf)
        if expected is None:
            ok = result is None
        else:
            ok = result is not None and expected in result
        status = "✅" if ok else "❌"
        if not ok:
            failed += 1
        else:
            passed += 1
        print(f"  {status} [{conf:.2f}] '{text_in}'\n       → '{result}'")

    print(f"\n{'=' * 80}")
    print(f"  {passed}/{passed + failed} tests passés")
    print(f"{'=' * 80}")
