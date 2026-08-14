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
    # contraction collee au mot suivant (the-frontier-count's-10th-class-outcas,
    # ch1 p2/p3) : l'apostrophe interne au token n'etait ni prise en compte
    # dans le decoupage ("im" absent de la liste des mots courts autorises),
    # ni preservee dans la reconstruction.
    ("I'MWORRIEDABOUT", "I'M WORRIED ABOUT"),
    ("YOU'LLSEE", "YOU'LL SEE"),
    ("I'VEGOTIT", "I'VE GOT IT"),
    # ponctuation interne collee a un mot entier (rise_of_the_dragon_overlord
    # ET the-frontier-count's-10th-class-outcas, fiches de stats de monstre) :
    # aucun espace ne donnait prise au decoupeur, le token entier restait
    # intact malgre la ponctuation deja presente.
    ("GIANTHORNEDANTELOPELV.7", "GIANT HORNED ANTELOPE LV. 7"),
    ("LEVEL:29", "LEVEL: 29"),
    ("MATERIALS:FOXFUR(8", "MATERIALS: FOX FUR (8"),
    # decoupage rejete a tort par le seuil de longueur moyenne de morceau
    # (the-wind-mage, ch0) : "TO"/"WE" comptaient dans la moyenne au meme
    # titre qu'un morceau incertain, alors que ce sont deja des mots courts
    # valides (liste blanche) -- moyenne recalculee en les excluant.
    ("TOBEGIN", "TO BEGIN"),
    ("WEWALKED", "WE WALKED"),
    # "I"/"A" colle au mot suivant, en dessous de _SEG_MIN_TOKEN_LEN donc
    # jamais tente avant (the-frontier-count's-10th-class-outcas, ch1 p1) :
    # pronom/article d'une lettre, le collage le plus frequent.
    ("AGIFT", "A GIFT"),
    ("ITOLD", "I TOLD"),
    ("INOT", "I NOT"),
    ("WASSICK", "WAS SICK"),
    # bruit du corpus wordsegment (denylist) : ces suites de lettres y
    # figurent comme "mot connu" (the-frontier-count's-10th-class-outcas,
    # ch1 p2), ce qui bloquait tout decoupage a tort.
    ("IHAVE", "I HAVE"),
    ("DURINGTHE", "DURING THE"),
    # mot colle DERRIERE une contraction (path-of-vengeance ch1) : le token
    # contient une apostrophe interne, `_segment_token` renonce, et le repli
    # `_split_after_contraction` doit prendre le relais.
    ("YOU'REMY", "YOU'RE MY"),
    ("I'MGOINGTO", "I'M GOING TO"),
    ("I'VEGOTIT", "I'VE GOT IT"),
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
    # ne doit JAMAIS etre modifie par la pre-passe de ponctuation interne
    "3.5",              # nombre decimal
    "12:30",            # heure
    # vrais mots en I/A a ne jamais casser (regression du fix "I"/"A" colle)
    "ICE", "AGE", "AND", "AWAY", "IDEA", "ITEM", "ANEW",
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
