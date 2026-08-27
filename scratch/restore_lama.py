# -*- coding: utf-8 -*-
"""Retelecharge et valide les deux chargeurs d'inpainting.

`lama_cleaner` supprime `big-lama.pt` quand `torch.jit.load` echoue, donc un
checkpoint corrompu disparait au premier usage et se retelecharge au suivant.
Ce script force ce cycle et DIT si chaque chargeur est reellement pret, au lieu
de laisser le pipeline se rabattre en silence sur cv2.inpaint.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(r"A:\omni read")))

import torch
from core.renderer import TextRenderer, LAMA_AVAILABLE

print(f"LAMA_AVAILABLE (simple-lama) : {LAMA_AVAILABLE}")
r = TextRenderer()
print()
print(f"self.lama (SimpleLama)          : {'PRET' if r.lama is not None else 'ABSENT'}")
print(f"anime_inpainter_ready (cleaner) : {getattr(r, 'anime_inpainter_ready', False)}")

ck = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "big-lama.pt"
if ck.exists():
    import hashlib
    h = hashlib.md5(ck.read_bytes()).hexdigest()
    print(f"\nbig-lama.pt : {ck.stat().st_size/1e6:.1f} Mo   md5 {h}")
    print("md5 attendu : e3aa4aaa15225a33ec84f9f4bc47e500")
    print("=> " + ("CONFORME" if h == "e3aa4aaa15225a33ec84f9f4bc47e500" else "NON CONFORME"))
else:
    print("\nbig-lama.pt : ABSENT")
