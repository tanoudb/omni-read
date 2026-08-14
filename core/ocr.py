"""
OCR engine: PaddleOCR PP-OCRv5 (subprocess .venv_paddleocr) en PRIMARY.
RapidOCR PP-OCRv5 (ONNX, in-process) en FALLBACK si le subprocess échoue.

Les deux moteurs utilisent PP-OCRv5 mais:
  - PaddleOCR (subprocess) = PaddlePaddle GPU natif, server models → plus précis
  - RapidOCR (in-process) = ONNX Runtime, mobile models → plus rapide mais moins bon
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Callable

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from utils import ImageUtils, TextFilter
from .backends import OCRBackend, PaddleOCRVLV15Backend, RapidOCRPPOCRv5Backend


BACKEND_REGISTRY = {
    "paddleocr-vl-v1.5": PaddleOCRVLV15Backend,
    "rapidocr-ppocrv5": RapidOCRPPOCRv5Backend,
    "rapidocr": RapidOCRPPOCRv5Backend,
    "ppocr-v5": RapidOCRPPOCRv5Backend,
}


# ═══════════════════════════════════════════════════════════════════════════════
# WORD SPLITTER — Mots collés OCR (EVERYONEHUNTSMONSTERS → EVERYONE HUNTS MONSTERS)
# ═══════════════════════════════════════════════════════════════════════════════
#
# PP-OCRv5 rend souvent une ligne entière sans espaces quand la police BD serre
# les lettres. Ce texte part tel quel vers le traducteur, donc l'erreur se
# propage avant même le rendu.
#
# L'implémentation précédente s'appuyait sur une liste de ~840 mots écrite à la
# main. Un chapitre de manhwa utilise un vocabulaire bien plus large : sur le
# cas réel « EVERYONEHUNTSMONSTERS TOGAINEXPERIENCEPOINTS », aucun des mots
# (EVERYONE, HUNTS, MONSTERS, EXPERIENCE, POINTS...) n'y figurait, et rien
# n'était découpé. Une liste manuelle ne peut pas rattraper ça.
#
# On utilise donc `wordsegment` (déjà dans les dépendances) : segmentation par
# maximum de vraisemblance sur les unigrammes/bigrammes du corpus Google, soit
# 333 000 mots au lieu de 840. Il est en revanche trop confiant — il découpe
# volontiers un nom propre ou une onomatopée en syllabes plausibles — d'où les
# garde-fous ci-dessous.

_SEG_MIN_TOKEN_LEN = 6      # en dessous, le gain est nul et le risque réel
_SEG_MIN_AVG_PIECE = 2.6    # un découpage trop émietté est un faux positif
_VOWELS = frozenset("AEIOUY")

# Les seuls mots anglais de 1-2 lettres qu'on accepte comme morceau. Sans cette
# liste fermée, « KRGHAAAH » se découpe en « KR GH AAAH » : `kr` et `gh`
# existent dans un corpus web, avec une fréquence comparable à celle de `hunts`,
# donc un seuil de fréquence ne les distingue pas.
_SHORT_WORDS = frozenset({
    "A", "I", "AM", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "HE", "IF", "IN",
    "IS", "IT", "ME", "MY", "NO", "OF", "OK", "ON", "OR", "SO", "TO", "UP",
    "US", "WE",
    # Restes de contractions une fois l'apostrophe perdue en cours de
    # découpage (I'M → IM, YOU'LL → YOULL...). Trouvé sur "I'MWORRIEDABOUT" :
    # wordsegment proposait bien ['im','worried','about'], mais 'im' n'étant
    # pas dans cette liste, tout le découpage était rejeté en bloc — le mot
    # entier restait collé.
    "IM", "ID", "ILL", "IVE",
    # Abréviation de "Level", omniprésente dans les fiches de stats/monstres
    # de ce genre de manhwa ("GIANT HORNED ANTELOPE LV.7"). Même mécanisme :
    # wordsegment proposait ['giant','horned','antelope','lv'], rejeté en bloc
    # faute de 'lv' dans cette liste.
    "LV",
})

# Bruit du corpus `wordsegment` : ces suites de lettres y figurent comme
# "mot connu" (fragments d'URL/slang du corpus web d'origine) alors qu'elles
# ne sont jamais un mot anglais valide dans un manhwa — sans ce denylist,
# `_is_known_word` les laisse passer et bloque tout découpage ("INOT" reste
# collé au lieu de "I NOT", trouvé sur un cas réel).
# "ofcourse" : même mécanisme, trouvé sur path-of-vengeance ch1
# ("OFCOURSE I'MGOINGTO WORRY!" reste collé) — le corpus web contient assez
# d'occurrences du typo "ofcourse" en un seul mot (freq ~340k) pour que
# `_is_known_word` le valide tel quel et court-circuite le découpage, alors
# que `ws.segment('ofcourse')` proposerait correctement ['of', 'course'].
# Lot suivant, trouvé sur the-frontier-count's-10th-class-outcas ch1 (même
# mécanisme, vérifié un par un via `_is_known_word` avant ajout — chacun de
# ces "mots" est bien absent de tout dictionnaire anglais réel) :
# "ican"/"backto" ("ICAN NEVER GO BACKTO THOSE DAYS"), "fromme" ("A GIFT
# FROMME"), "bigbrother" ("BIGBROTHER! BIGBROTHER!"), "iwas" ("WHEN IWAS
# TEN"), "iam" ("WHO IAM"), "buti" ("BUTI HAVEN'T LOST"), "sizeof" ("THE
# SIZEOF THAT VESSEL"), "ata" ("CIRCULATES ATA SPEED").
_KNOWN_WORD_DENYLIST = frozenset({
    "inot", "whatis", "founda", "ihave", "duringthe", "ofcourse",
    "ican", "backto", "fromme", "bigbrother", "iwas", "iam", "buti",
    "sizeof", "ata",
})

_wordsegment_state: Dict[str, object] = {"loaded": False, "module": None}
_split_cache: Dict[str, Optional[List[str]]] = {}
_protected_state: Dict[str, object] = {}


def _wordsegment():
    """Charge `wordsegment` à la première utilisation (~0.5 s, ~87 Mo).

    Root cause investiguée (2026-08-14) : la quasi-totalité des mots collés
    listés comme "non rattrapés" (YOUCOMEBACK, BRINGBACKA, HEAVYWARRIOR,
    KUROIPLAYER, WHOWORRIESABOUT, SUPPLYRUN/DUTYAGAIN, MIGHTJUST/BEANORMAL,
    OFCOURSE/I'MGOINGTO...) segmentent CORRECTEMENT une fois `wordsegment`
    réellement chargé — le module n'était simplement PAS installé dans
    l'environnement `python` utilisé pour lancer le pipeline (présent dans
    requirements.txt mais absent du site-packages actif), alors qu'il l'est
    dans `.venv311`. L'`except Exception` ci-dessous avalait le
    `ModuleNotFoundError` en silence : tout le découpeur de mots collés
    tournait à vide sans qu'aucun log ne le signale. On log désormais
    l'échec une fois pour que ça ne se reproduise plus silencieusement.
    """
    if not _wordsegment_state["loaded"]:
        _wordsegment_state["loaded"] = True
        try:
            import wordsegment

            wordsegment.load()
            _wordsegment_state["module"] = wordsegment
        except Exception as exc:
            _wordsegment_state["module"] = None
            print(
                f"⚠️ wordsegment indisponible ({exc}) — le découpage des mots "
                f"collés OCR (YOUCOMEBACK, BRINGBACKA...) est DÉSACTIVÉ. "
                f"Vérifiez `pip show wordsegment` dans l'environnement utilisé "
                f"pour lancer le pipeline (`pip install wordsegment==1.3.1`)."
            )
    return _wordsegment_state["module"]


def _is_known_word(word: str) -> bool:
    ws = _wordsegment()
    lower = word.lower()
    return bool(ws) and lower in ws.UNIGRAMS and lower not in _KNOWN_WORD_DENYLIST


def _acceptable_pieces(pieces: List[str]) -> bool:
    """Un découpage n'est retenu que si CHAQUE morceau est un mot plausible."""
    if len(pieces) < 2:
        return False

    for piece in pieces:
        up = piece.upper()
        if len(up) <= 2:
            if up not in _SHORT_WORDS:
                return False
            continue
        # Un morceau sans voyelle n'est pas un mot anglais : c'est le signe
        # qu'on est en train d'émietter une onomatopée ou un sigle.
        if not (_VOWELS & set(up)):
            return False
        if not _is_known_word(piece):
            return False

    # Moyenne calculée en EXCLUANT les morceaux de la liste blanche des mots
    # courts : ce sont déjà des mots validés individuellement (pas une
    # supposition), les compter pénalise à tort un découpage par ailleurs
    # solide. Mesuré : "MYJOB" -> "MY"+"JOB" rejeté (moyenne 2.5 < 2.6) à
    # cause du seul "MY", alors que les DEUX morceaux sont des mots réels.
    # Les onomatopées ne profitent pas de cette exclusion : leurs fragments
    # courts ("KR", "GH"...) ne sont pas eux-mêmes dans cette liste blanche et
    # restent bloqués par le test morceau-par-morceau ci-dessus.
    long_pieces = [p for p in pieces if p.upper() not in _SHORT_WORDS]
    if not long_pieces:
        return True
    avg = sum(len(p) for p in long_pieces) / len(long_pieces)
    return avg >= _SEG_MIN_AVG_PIECE


def _segment_token(core: str, protected: frozenset) -> Optional[List[str]]:
    """
    Découpe `core` en mots. Retourne les morceaux DÉCOUPÉS SUR LA CHAÎNE
    D'ORIGINE (casse et ponctuation interne préservées), ou None.

    Limite connue, INVESTIGUÉE ET ABANDONNÉE (ne pas retenter la même piste) :
    un collage OCR peut coïncider avec une entrée du corpus wordsegment assez
    fréquente pour bloquer toute tentative de découpage ("THEDAY" freq=19k,
    "ABIT" freq=2M — tous deux restent collés). Un seuil sur la fréquence du
    mot entier, ou sur le RATIO fréquence-découpé/fréquence-entier, semblait
    une piste naturelle : mesuré et rejeté, aucun seuil ne sépare "theday"/
    "abit" (doivent être découpés) de "cannot"/"forgot"/"ago"/"thereby"
    (doivent rester collés) — le ratio de "cannot" (should NOT split) dépasse
    largement celui de "theday" ET "abit" (should split), dans le mauvais
    sens. Le signal n'existe simplement pas dans les fréquences lexicales
    seules ; accepté comme limite de l'approche statistique. Idem pour un
    garde-fou par PRÉFIXE ("my"/"the"/"a" + reste = mot connu) : "ANEW" (=
    "a"+"new", un vrai mot à ne jamais découper) a une fréquence du même
    ordre que "ABIT", donc indiscernable sur ce critère non plus.

    Autre cas limite, distinct : "MYJOB" (5 lettres) tombe sous
    `_SEG_MIN_TOKEN_LEN` (6) et n'est donc jamais soumis au découpage — seuil
    délibérément conservateur (cf. son commentaire) pour ne pas déchiqueter
    les onomatopées courtes ("BOOM", "CRACK"...), pas abaissé pour ce cas.
    """
    key = core.upper()

    # La protection est testée AVANT le cache : le résultat dépend du glossaire
    # passé en argument, alors que le cache n'est indexé que sur le token. Sans
    # ça, un token protégé lors d'un appel restait marqué « non découpable »
    # pour tous les appels suivants, glossaire ou pas.
    if key in protected:
        return None

    if key in _split_cache:
        return _split_cache[key]

    result: Optional[List[str]] = None
    letters = [c for c in core if c.isalnum()]
    joined_letters = "".join(letters)

    # Cas court-circuité : "I"/"A" collé au mot suivant ("AGIFT", "ITOLD",
    # "INOT") passe sous `_SEG_MIN_TOKEN_LEN` et n'est donc jamais tenté, alors
    # que c'est le collage le plus fréquent sur ce pronom/article d'une seule
    # lettre. Restreint au cas où le RESTE (sans le "I"/"A" initial) est déjà
    # un mot connu, pour ne jamais toucher un vrai mot en I/A ("ICE", "AGE",
    # "AND"...) ni une onomatopée courte ("AHHH", "IDK"...).
    is_short_prefix_glue = (
        len(letters) >= 3
        and letters[0].upper() in ("I", "A")
        and _is_known_word("".join(letters[1:]))
    )

    # Symétrique du cas ci-dessus, mais le pronom/article d'une lettre est
    # collé APRÈS le mot ("ATA" = "AT"+"A", "BUTI" = "BUT"+"I") plutôt
    # qu'avant. Trouvé sur the-frontier-count's-10th-class-outcas ch1
    # ("MANA CIRCULATES ATA SPEED", "BUTI HAVEN'T LOST"). Même garde-fou
    # (RESTE déjà un mot connu) pour ne jamais toucher un vrai mot en I/A
    # final ("AREA", "IDEA", "TAXI"...) : ces mots-là sont eux-mêmes des
    # mots connus, donc bloqués par `not _is_known_word(joined_letters)`
    # ci-dessous avant même d'atteindre ce test.
    is_short_suffix_glue = (
        len(letters) >= 3
        and letters[-1].upper() in ("I", "A")
        and _is_known_word("".join(letters[:-1]))
    )

    if (
        (
            len(core) >= _SEG_MIN_TOKEN_LEN
            and len(letters) >= _SEG_MIN_TOKEN_LEN
        )
        or is_short_prefix_glue
        or is_short_suffix_glue
    ) and not _is_known_word(joined_letters):
        ws = _wordsegment()
        if ws is not None:
            if is_short_prefix_glue:
                # `ws.segment` juge en probabilité globale : sur "ITOLD" il
                # préfère parfois "it"+"old" à "i"+"told", les deux étant des
                # mots connus. On a déjà vérifié que le reste après le seul
                # "I"/"A" initial est un mot connu — trancher directement sur
                # cette frontière-là plutôt que de laisser `segment` réémettre
                # une autre coupe, statistiquement plausible mais fausse ici.
                pieces = [letters[0], "".join(letters[1:])]
            elif is_short_suffix_glue:
                pieces = ["".join(letters[:-1]), letters[-1]]
            else:
                try:
                    pieces = ws.segment("".join(letters))
                except Exception:
                    pieces = []

            # `segment` peut perdre des caractères ; sans cette vérification on
            # réécrirait un texte différent de celui lu.
            if pieces and sum(len(p) for p in pieces) == len(letters) and _acceptable_pieces(pieces):
                positions = [i for i, c in enumerate(core) if c.isalnum()]
                out: List[str] = []
                cursor = 0
                for idx, piece in enumerate(pieces):
                    start = positions[cursor]
                    cursor += len(piece)
                    # Jusqu'au début du morceau suivant, pour ne perdre aucune
                    # apostrophe ou trait d'union interne.
                    end = positions[cursor] if cursor < len(positions) else len(core)
                    if idx == len(pieces) - 1:
                        end = len(core)
                    out.append(core[start:end].strip())
                if all(out) and "".join(c for p in out for c in p if c.isalnum()) == "".join(letters):
                    result = out

    _split_cache[key] = result
    return result


def _protected_words() -> frozenset:
    """
    Formes à ne jamais découper : les entrées du glossaire de la série (noms de
    personnages, lieux, compétences), qui y sont déjà en MAJUSCULES.

    Le pipeline recopie ce glossaire dans `config.translation.forced_translations`
    au démarrage quand le mode série est actif.
    """
    entries = getattr(config.translation, 'forced_translations', None) or {}
    signature = len(entries)
    if _protected_state.get('signature') != signature:
        _protected_state['signature'] = signature
        _protected_state['value'] = frozenset(
            str(k).upper().replace(" ", "") for k in entries if k
        )
    return _protected_state.get('value') or frozenset()


_CONTRACTION_GLUE = re.compile(
    r"^(?P<head>[A-Za-z]+'(?:re|ll|ve|t|s|d|m))(?P<tail>[A-Za-z]{2,})$",
    re.IGNORECASE,
)


def _split_after_contraction(core: str, protected: frozenset) -> str:
    """Détache le mot collé DERRIÈRE une contraction ("YOU'REMY" -> "YOU'RE MY").

    Repli utilisé seulement quand `_segment_token` a déjà renoncé sur le token
    entier : lui, il concatène les lettres sans l'apostrophe et sait très bien
    traiter "I'VEGOTIT" d'un bloc. On n'intervient donc que sur ce qu'il laisse
    passer — mesuré sur path-of-vengeance ch1 : "YOU'REMY ONLY BLOOD RELATIVE",
    "I'MGOINGTO WORRY!", "KAZUKI'SON SUPPLY RUN DUTY AGAIN".
    """
    m = _CONTRACTION_GLUE.match(core)
    if not m:
        return core
    tail = m.group("tail")
    tail_pieces = _segment_token(tail, protected)
    tail_out = " ".join(tail_pieces) if tail_pieces else tail
    return f"{m.group('head')} {tail_out}"


def _split_glued_words(text: str, protected: Optional[frozenset] = None) -> str:
    """
    Rétablit les espaces dans les mots collés par l'OCR.

    `protected` : formes en MAJUSCULES à ne jamais découper (noms propres du
    glossaire de la série). `wordsegment` découpe volontiers « SUNGJINWOO » en
    « sung jin woo » — souvent acceptable, mais destructeur sur un nom d'un seul
    tenant, et le glossaire est la seule source fiable pour le savoir.
    """
    if not text:
        return text

    # Ponctuation interne collée à un mot entier ("LEVEL:29", "LV.7",
    # "FOXFUR(8") : le découpeur ci-dessous n'agit que sur des tokens
    # délimités par des ESPACES, donc un token entier restait intact tant
    # qu'aucun espace n'y donnait prise. Mesuré sur deux séries différentes :
    # "GIANTHORNEDANTELOPELV.7" ne se découpait pas du tout, parce que le "7"
    # final restait collé à "LV" une fois le point ignoré pour la
    # segmentation — et "LV7" n'a pas la tête d'un mot valide (aucune
    # voyelle), donc le découpage entier était rejeté.
    # Lookbehind restreint aux LETTRES (pas aux chiffres) pour ne jamais
    # toucher un vrai nombre décimal ("3.5").
    text = re.sub(r'(?<=[A-Za-z])([.:])(?=[A-Za-z0-9])', r'\1 ', text)
    text = re.sub(r'(?<=[A-Za-z0-9])(\()(?=[A-Za-z0-9])', r' \1', text)


    protected = protected or frozenset()
    result: List[str] = []

    for word in text.split():
        leading = ""
        trailing = ""
        core = word

        while core and not core[0].isalnum():
            leading += core[0]
            core = core[1:]
        while core and not core[-1].isalnum():
            trailing = core[-1] + trailing
            core = core[:-1]

        if not core:
            result.append(word)
            continue

        pieces = _segment_token(core, protected)
        if not pieces:
            core = _split_after_contraction(core, protected)
        result.append(leading + (" ".join(pieces) if pieces else core) + trailing)

    text = " ".join(result)

    # Frontières lettre↔chiffre (ONLY8 → ONLY 8, I'MONLY8 → I'M ONLY 8)
    text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)

    return re.sub(r' {2,}', ' ', text).strip()


# ═══════════════════════════════════════════════════════════════════════════════
# WATERMARK STRIP — fix ciblé d'un bug dans utils/filters.py (hors périmètre)
# ═══════════════════════════════════════════════════════════════════════════════
#
# `TextFilter.strip_watermark_fragments` (utils/filters.py) termine par
# `.strip(" .-—|")` INCONDITIONNEL, même quand AUCUN pattern watermark n'a
# matché. Conséquence mesurée sur `scratch/render_out/pov_full_v0/bubbles_meta.json` :
# du texte de dialogue parfaitement propre perd sa ponctuation finale
# légitime — "AKINA..." -> "AKINA", "HEHE." -> "HEHE", "LOST AGAIN..." ->
# "LOST AGAIN", "*PANT*.." -> "*PANT*" — alors que l'OCR brut (avant tout
# post-traitement) contenait bien cette ponctuation (vérifié via
# `debug_hook` sur `OCREngine.extract_batch`, texte primaire PaddleOCR
# inchangé). `utils/filters.py` est hors périmètre pour cette tâche (un
# autre agent y travaille en parallèle), donc on ne le corrige pas
# directement : on réimplémente ici la même logique de suppression de
# fragments, mais on ne strippe les bords qu'APRÈS confirmation qu'un
# pattern watermark a réellement été substitué (sinon rien à nettoyer en
# bordure). Comportement inchangé sur les vrais watermarks ("CRAWLED BY
# MANHWACLAN.COM SHARDS OF OUR GOD..." -> "SHARDS OF OUR GOD" dans les
# deux versions).
def _strip_watermark_fragments_safe(text_filter, text: str) -> str:
    if not text:
        return text
    out = text
    matched = False
    for pattern in text_filter.watermark_patterns:
        if pattern.pattern.startswith('^') or pattern.pattern.endswith('$'):
            continue
        new_out = pattern.sub(' ', out)
        if new_out != out:
            matched = True
        out = new_out
    out = re.sub(r'\s+([.,:;])', r'\1', out)
    out = re.sub(r'\s{2,}', ' ', out).strip()
    if matched:
        # Un fragment a bien été retiré : nettoyer la ponctuation/tirets
        # orphelins qu'il a pu laisser en bordure (comportement original).
        out = out.strip(" .-—|")
    return out


class OCREngine:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.cfg = config.ocr
        self.primary_backend: Optional[OCRBackend] = None
        self.fallback_backends: List[OCRBackend] = []

        self.text_filter = TextFilter(
            watermark_patterns=config.filters.watermark_patterns,
            sfx_patterns=config.filters.sfx_patterns,
        )

        self._load_backends_chain()

    def _load_backends_chain(self):
        primary_name = (
            getattr(self.cfg, "backend", None)
            or getattr(self.cfg, "primary_backend", None)
            or "paddleocr-vl-v1.5"
        ).strip().lower()

        fallback_names = [
            str(name).strip().lower()
            for name in getattr(self.cfg, "fallback_backends", ["rapidocr-ppocrv5"])
            if str(name).strip()
        ]
        fallback_names = [name for name in fallback_names if name != primary_name]

        loaded: List[OCRBackend] = []
        for name in [primary_name] + fallback_names:
            backend_class = BACKEND_REGISTRY.get(name)
            if backend_class is None:
                continue
            try:
                backend = backend_class()
                backend.load(self.device)
                loaded.append(backend)
                print(f"✅ Backend OCR chargé: {backend.name}")
            except Exception as exc:
                print(f"⚠️ Backend OCR {name} indisponible: {exc}")

        if not loaded:
            raise RuntimeError("Aucun backend OCR disponible")

        self.primary_backend = loaded[0]
        self.fallback_backends = loaded[1:]

    def preprocess_image(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = img.shape[:2]
        upscale_factor = 1.0

        if h < 80:
            upscale_factor = 150 / max(1, h)
            img = cv2.resize(img, (int(w * upscale_factor), 150), interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]
        elif h < 100:
            upscale_factor = 120 / max(1, h)
            img = cv2.resize(img, (int(w * upscale_factor), 120), interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]

        if h < 64:
            extra = 64 / max(1, h)
            upscale_factor *= extra
            img = cv2.resize(img, (max(1, int(w * extra)), 64), interpolation=cv2.INTER_CUBIC)

        if self.cfg.auto_resize:
            h_before = img.shape[0]
            img = ImageUtils.smart_resize(
                img,
                min_height=self.cfg.min_text_height,
                max_factor=self.cfg.max_resize_factor,
                interpolation=self.cfg.resize_interpolation,
            )
            # smart_resize peut agrandir une seconde fois. Sans ce cumul, le
            # facteur renvoyé sous-estime l'agrandissement réel et l'appelant
            # ne ramène pas complètement les polygones OCR dans le repère du
            # crop d'origine (masque d'effacement décalé).
            if h_before > 0 and img.shape[0] != h_before:
                upscale_factor *= img.shape[0] / float(h_before)

        return img, upscale_factor

    def post_process_text(self, text: str) -> str:
        if not text:
            return ""
        # ★ NOUVEAU : OCR Cleaner V2 ★
        try:
            from utils.ocr_cleaner import clean_ocr_output
            result = clean_ocr_output(text)
            if result is None:
                return ""
            text = result
        except Exception:
            # If cleaner unavailable, continue with existing pipeline
            pass
        text = self.text_filter.clean_text(text)
        if getattr(config.filters, 'enable_watermark_filter', True):
            text = _strip_watermark_fragments_safe(self.text_filter, text)
        text = re.sub(r"\b1\.(?=\s+[A-Z])", "I.", text)
        text = re.sub(r"\bI\.(?=\s+THE\b)", "I,", text)
        text = re.sub(r"(?<=[A-Z])\s+1\s+(?=[A-Z])", " I ", text)
        # ── Word splitting (mots collés OCR) ──
        text = _split_glued_words(text, protected=_protected_words())
        text = re.sub(r"\s+", " ", text).strip()
        if self.cfg.remove_isolated_chars:
            words = text.split()
            words = [w for w in words if len(w) > 1 or w.isalnum()]
            text = " ".join(words)
        return text

    def is_valid_text(self, text: str, confidence: float) -> Tuple[bool, Optional[str]]:
        # Filtre confiance-aware via OCR Cleaner
        try:
            from utils.ocr_cleaner import clean_low_confidence
            text = clean_low_confidence(text, confidence)
            if text is None:
                return False, "low_conf_noise"
        except Exception:
            pass

        if confidence < self.cfg.min_confidence:
            return False, "low_confidence"
        if len(text.strip()) < self.cfg.min_text_length:
            return False, "too_short"
        should_skip, reason = self.text_filter.should_skip(
            text,
            min_length=self.cfg.min_text_length,
            max_numeric_ratio=self.cfg.max_numeric_ratio,
        )
        if should_skip:
            return False, reason
        if self.cfg.filter_numeric_only and self.text_filter.is_numeric_only(text, self.cfg.max_numeric_ratio):
            return False, "numeric_only"
        if self.cfg.filter_special_chars_only and self.text_filter.is_special_chars_only(text):
            return False, "special_chars_only"
        return True, None

    @staticmethod
    def _preview_text(value: str, max_len: int = 140) -> str:
        txt = (value or "").replace("\n", " ").strip()
        return txt if len(txt) <= max_len else txt[:max_len] + "..."

    def get_runtime_diagnostics(self) -> Dict:
        details: Dict = {
            "device": self.device,
            "primary": self.primary_backend.name if self.primary_backend else "none",
            "fallbacks": [b.name for b in self.fallback_backends if b],
        }
        backend_infos = []
        for backend in [self.primary_backend, *self.fallback_backends]:
            if backend is None:
                continue
            info = {"name": backend.name}
            getter = getattr(backend, "get_runtime_info", None)
            if callable(getter):
                try:
                    extra = getter()
                    if isinstance(extra, dict):
                        info.update(extra)
                except Exception as exc:
                    info["runtime_info_error"] = str(exc)
            backend_infos.append(info)
        details["backends"] = backend_infos
        return details

    def extract_text(self, img: np.ndarray) -> Tuple[Optional[str], float, bool, Optional[str], List[Dict], float]:
        results = self.extract_batch([img])
        return results[0]

    def extract_batch(
        self,
        crops: List[np.ndarray],
        debug_hook: Optional[Callable[[str], None]] = None,
    ) -> List[Tuple[Optional[str], float, bool, Optional[str], List[Dict], float]]:
        """
        Batch OCR:
        1) Preprocess all crops
        2) Primary (PaddleOCR PP-OCRv5 subprocess) processes all
        3) Fallback (RapidOCR) for any that failed
        """
        if not crops:
            return []

        if debug_hook:
            debug_hook(f"[OCR] crops reçus: {len(crops)}")

        # 1. Preprocessing
        processed: List[np.ndarray] = []
        upscale_factors: List[float] = []
        for i, crop in enumerate(crops):
            img_proc, up = self.preprocess_image(crop)
            processed.append(img_proc)
            upscale_factors.append(up)
            if debug_hook:
                h0, w0 = crop.shape[:2]
                h1, w1 = img_proc.shape[:2]
                debug_hook(f"[OCR][crop {i}] {w0}x{h0} → {w1}x{h1} up={up:.2f}")

        n = len(processed)

        # 2. Primary — all crops
        primary_results: List[Tuple[str, float, List[Dict]]] = [("", 0.0, [])] * n

        if self.primary_backend is not None:
            if debug_hook:
                debug_hook(f"[OCR] Primary: {self.primary_backend.name} — {n} crops")

            batch_reader = getattr(self.primary_backend, "read_batch", None)
            if callable(batch_reader):
                raw = batch_reader(processed)
            else:
                raw = [self.primary_backend.read_text(img) for img in processed]

            for i, out in enumerate(raw):
                if i < n:
                    primary_results[i] = out
                    if debug_hook:
                        text, conf, regions = out
                        debug_hook(
                            f"[OCR][crop {i}][{self.primary_backend.name}] "
                            f"conf={conf:.3f} text='{self._preview_text(text)}'"
                        )

        # 3. Fallback for failed crops
        failed = [i for i in range(n)
                  if not self.post_process_text(primary_results[i][0])
                  or primary_results[i][1] < 0.1]

        if failed and self.fallback_backends:
            fb = self.fallback_backends[0]
            if debug_hook:
                debug_hook(f"[OCR] Fallback: {fb.name} — {len(failed)}/{n} crops")

            fb_crops = [processed[i] for i in failed]
            batch_reader = getattr(fb, "read_batch", None)
            fb_raw = batch_reader(fb_crops) if callable(batch_reader) else [fb.read_text(img) for img in fb_crops]

            for idx_fb, orig_idx in enumerate(failed):
                if idx_fb >= len(fb_raw):
                    break
                fb_text, fb_conf, fb_regions = fb_raw[idx_fb]
                fb_clean = self.post_process_text(fb_text)
                if fb_clean:
                    primary_results[orig_idx] = (fb_clean, fb_conf, fb_regions)
                    if debug_hook:
                        debug_hook(
                            f"[OCR][crop {orig_idx}][{fb.name}] "
                            f"conf={fb_conf:.3f} text='{self._preview_text(fb_clean)}'"
                        )

        # 4. Final results
        final: List[Tuple[Optional[str], float, bool, Optional[str], List[Dict], float]] = []
        for i in range(n):
            text, confidence, regions = primary_results[i]
            clean = self.post_process_text(text)
            is_valid, skip_reason = self.is_valid_text(clean, confidence) if clean else (False, "empty")

            if not is_valid:
                if debug_hook:
                    debug_hook(f"[OCR][crop {i}] SKIP {skip_reason} conf={confidence:.3f} '{self._preview_text(clean)}'")
                final.append((None, float(confidence), False, skip_reason, [], upscale_factors[i]))
            else:
                if debug_hook:
                    debug_hook(f"[OCR][crop {i}] ✓ conf={confidence:.3f} '{self._preview_text(clean)}'")
                final.append((clean, float(confidence), True, None, regions, upscale_factors[i]))

        return final

    def predict_full_image(self, image_path: Path) -> List[Dict]:
        return []

    def get_backend_name(self) -> str:
        names = []
        if self.primary_backend:
            names.append(self.primary_backend.name)
        names.extend([b.name for b in self.fallback_backends if b])
        return " → ".join(names) if names else "none"

    def __del__(self):
        for backend in [self.primary_backend, *self.fallback_backends]:
            if backend:
                try:
                    backend.unload()
                except Exception:
                    pass