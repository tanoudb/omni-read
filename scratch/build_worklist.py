# -*- coding: utf-8 -*-
"""Construit la liste de vérification VISUELLE pour l'essaim, à partir d'un run
du barème. Cible : toute zone à défaut d'EFFACEMENT (residu/ghost/erase_spill),
toutes les zones System, plus un échantillon de zones conformes par série
(paranoïa : les métriques peuvent mentir). Sort scratch/worklist.json avec les
chemins ABSOLUS des crops before/erased/after."""
import json, io, os, sys
RUN = sys.argv[1] if len(sys.argv) > 1 else "final1"
ROOT = os.path.abspath(".")
def _slug(name): return "".join(c if c.isalnum() or c in "-_" else "-" for c in name).strip("-").lower()
rows = json.load(io.open("scratch/bareme/runs/%s/scores.json" % RUN, encoding="utf-8"))["rows"]
ERASE = {"residu", "ghost", "erase_spill"}
by_series = {}
for r in rows:
    by_series.setdefault(r["series"], []).append(r)

targets = []
seen = set()
def crop_paths(r):
    d = os.path.join(ROOT, "scratch", "bareme", "runs", RUN, "crops",
                     "%s__%s" % (_slug(r["series"]), r["page"]))
    return {k: os.path.join(d, "%03d_%s.png" % (r["index"], k)) for k in ("before", "erased", "after")}
def add(r, reason):
    key = (r["series"], r["page"], r["index"])
    if key in seen: return
    seen.add(key)
    cp = crop_paths(r)
    if not os.path.exists(cp["erased"]):  # crop manquant → ignore
        return
    targets.append({"series": r["series"], "page": r["page"], "index": r["index"],
                    "cls": r["class"], "text": (r.get("text") or "")[:60],
                    "defauts": r["defauts"], "reason": reason,
                    "residu_pct": round(r.get("residu_pct") or 0, 1),
                    "ghost": round(r.get("ghost_contrast") or 0, 1),
                    "spill": round(r.get("erase_spill_pct") or 0, 1),
                    "before": cp["before"], "erased": cp["erased"], "after": cp["after"]})

# 1) défauts d'effacement + System
for r in rows:
    if ERASE & set(r["defauts"]):
        add(r, "erase_defect")
    elif r["class"] == "System":
        add(r, "system")
# 2) échantillon conforme par série (stride déterministe, 5/série)
for s, rs in by_series.items():
    clean = [r for r in rs if not r["defauts"]]
    if not clean: continue
    step = max(1, len(clean) // 5)
    for r in clean[::step][:5]:
        add(r, "sample_clean")

json.dump({"run": RUN, "zones": targets}, io.open("scratch/worklist.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
from collections import Counter
c = Counter(t["reason"] for t in targets)
print("worklist: %d zones -> %s" % (len(targets), dict(c)))
print("series covered:", len(set(t["series"] for t in targets)))

# --- shard en chunks pour l'essaim (chaque agent lit son fichier) ---
NCH = 12
import math
sz = math.ceil(len(targets) / NCH)
os.makedirs("scratch/wl_chunks", exist_ok=True)
# purge anciens chunks
for old in os.listdir("scratch/wl_chunks"):
    if old.startswith("chunk_"):
        os.remove(os.path.join("scratch/wl_chunks", old))
nchunks = 0
for ci in range(0, len(targets), sz):
    chunk = targets[ci:ci + sz]
    json.dump(chunk, io.open("scratch/wl_chunks/chunk_%02d.json" % (nchunks), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    nchunks += 1
print("chunks: %d (~%d zones each)" % (nchunks, sz))
