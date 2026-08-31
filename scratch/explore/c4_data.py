"""C4 - chargement du cache + verite terrain d'encre (reprise de d2_ink)."""
import sys, os, glob, pickle
sys.path.insert(0, '.')
import numpy as np
import cv2

CACHE = 'scratch/bareme/cache'

def keys():
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(CACHE, '*'))
                  if os.path.isdir(p))

_pages = {}
def load(key):
    if key in _pages:
        return _pages[key]
    d = pickle.load(open(os.path.join(CACHE, key, 'dets.pkl'), 'rb'))
    page = cv2.imread(os.path.join(CACHE, key, 'page.png'), cv2.IMREAD_COLOR)
    _pages[key] = (d, page)
    return d, page

def crop_of(page, item):
    x1, y1, x2, y2 = [int(v) for v in item['bbox']]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(page.shape[1], x2); y2 = min(page.shape[0], y2)
    return page[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

def iter_items(limit_pages=None):
    ks = keys()
    if limit_pages:
        ks = ks[:limit_pages]
    for k in ks:
        d, page = load(k)
        for i, it in enumerate(d['items']):
            c, box = crop_of(page, it)
            if c.size == 0:
                continue
            yield k, i, it, c
