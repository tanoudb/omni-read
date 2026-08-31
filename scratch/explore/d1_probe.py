import sys, pickle, numpy as np, cv2
sys.path.insert(0, '.')
from pipeline import TranslationPipeline as TP

CACHE = 'scratch/bareme/cache/30-years-have-passed-since-the-prologue__p01/dets.pkl'
d = pickle.load(open(CACHE, 'rb'))
items = d['items']
print('n items', len(items))
for i, it in enumerate(items):
    t = (it.get('text') or '')[:40].replace('\n', ' ')
    b = it['bbox']
    print(i, it.get('class_name'), b, (b[2]-b[0], b[3]-b[1]), repr(t))
