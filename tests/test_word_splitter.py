"""Batterie de tests du decoupeur de mots colles."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from core.ocr import _split_glued_words

# (entree, attendu)
DOIT_DECOUPER = [
    # cas reels observes sur rise_of_the_dragon_overlord
    ("EVERYONEHUNTSMONSTERS", "EVERYONE HUNTS MONSTERS"),
    ("TOGAINEXPERIENCEPOINTS", "TO GAIN EXPERIENCE POINTS"),
    ("ANDMATERIALSTOCRAFT", "AND MATERIALS TO CRAFT"),
    ("YEARSOLDER", "YEARS OLDER"),
    ("SHESUPPORTS", "SHE SUPPORTS"),
    ("WITHOUTCOMPLAINTORREGRET", "WITHOUT COMPLAINT OR REGRET"),
    ("JOBCHANGEINTENTIONSURVEY", "JOB CHANGE INTENTION SURVEY"),
    # cas que l'ancien dictionnaire gerait deja (non-regression)
    ("ABOUTOUR", "ABOUT OUR"),
    ("MYSONWILL", "MY SON WILL"),
    ("INHISPLACE", "IN HIS PLACE"),
    ("LIKEYOUCOULDCOME", "LIKE YOU COULD COME"),
    # casse et ponctuation preservees
    ("Everyonehunts", "Everyone hunts"),    # casse d'origine conservee telle quelle
    ("ABOUTOUR!", "ABOUT OUR!"),
    ("\"ABOUTOUR\"", "\"ABOUT OUR\""),
]

DOIT_LAISSER_INTACT = [
    "FORWARD",          # vrai mot
    "MONSTERS",         # vrai mot
    "EXPERIENCE",       # vrai mot
    "KRGHAAAH",         # onomatopee -> KR/GH sans voyelle
    "GAAAAH",           # onomatopee
    "BAM",              # trop court
    "HMPH",             # pas de decoupage plausible
    "I'M",              # trop court
    "SYSTEM",           # vrai mot
]

PROTEGES = frozenset({"SUNGJINWOO", "CHEONGDO"})

ok = fail = 0
print("--- doit decouper ---")
for src, expected in DOIT_DECOUPER:
    got = _split_glued_words(src)
    good = got == expected
    ok, fail = (ok + 1, fail) if good else (ok, fail + 1)
    print(f"  {'OK ' if good else 'ECHEC'} {src:28s} -> {got!r}" + ("" if good else f"   attendu {expected!r}"))

print("--- doit rester intact ---")
for src in DOIT_LAISSER_INTACT:
    got = _split_glued_words(src)
    good = got == src
    ok, fail = (ok + 1, fail) if good else (ok, fail + 1)
    print(f"  {'OK ' if good else 'ECHEC'} {src:28s} -> {got!r}")

print("--- noms propres proteges (glossaire) ---")
for src in ("SUNGJINWOO", "CHEONGDO"):
    got = _split_glued_words(src, protected=PROTEGES)
    good = got == src
    ok, fail = (ok + 1, fail) if good else (ok, fail + 1)
    print(f"  {'OK ' if good else 'ECHEC'} {src:28s} -> {got!r}")
    if src == "SUNGJINWOO":
        print(f"      (sans protection : {_split_glued_words(src)!r})")

print("--- phrase complete ---")
phrase = "EVERYONEHUNTSMONSTERS TOGAINEXPERIENCEPOINTS ANDMATERIALSTOCRAFT NEWEQUIPMENTANDWEAPONS."
print("  ", _split_glued_words(phrase))

print(f"\n{ok} OK / {fail} echecs")
sys.exit(1 if fail else 0)
