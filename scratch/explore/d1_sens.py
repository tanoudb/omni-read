import sys, pickle, numpy as np, cv2
sys.path.insert(0, '.')
from pipeline import TranslationPipeline as TP
CACHE = 'scratch/bareme/cache/30-years-have-passed-since-the-prologue__p01'
d = pickle.load(open(CACHE + '/dets.pkl','rb'))
it = d['items'][14]
page = cv2.imread(CACHE + '/page.png')
x1,y1,x2,y2 = it['bbox']; crop = page[y1:y2, x1:x2]; h,w = crop.shape[:2]
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
ocr = TP._ocr_mask_from_regions(it['text_regions'], h, w, crop_bgr=crop, dilate=3)
cm = it['chirurgical_mask']

# x max du masque chirurgical vs encre
xs_cm = np.nonzero(cm)[1]; xs_ocr = np.nonzero(ocr)[1]
print('chirurgical x', xs_cm.min(), '..', xs_cm.max(), ' | encre OCR x', xs_ocr.min(), '..', xs_ocr.max())
# profil colonne par tranche de 50 px
for a in range(0, w, 60):
    b = min(w, a+60)
    print(f'  x[{a:3d}:{b:3d}] encre={int(np.count_nonzero(ocr[:,a:b])):5d}  chirurgical={int(np.count_nonzero(cm[:,a:b])):5d}')

def bubble_mask(crop_bgr, tol=40):
    h,w = crop_bgr.shape[:2]
    if h<24 or w<24: return None
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    ref = float(np.median(g[h//3:2*h//3, w//3:2*w//3]))
    sim = (np.abs(g.astype(np.int16)-ref) < tol).astype(np.uint8)*255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    er = cv2.erode(sim,k,1)
    n,lab,st,_ = cv2.connectedComponentsWithStats(er,8)
    if n<2: return None
    big = 1+int(np.argmax(st[1:,cv2.CC_STAT_AREA]))
    m = cv2.dilate((lab==big).astype(np.uint8)*255,k,1)
    c,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not c: return None
    f = np.zeros_like(m); cv2.drawContours(f,c,-1,255,-1)
    fill = np.count_nonzero(f)/(w*h)
    return f, fill

print('\nsensibilite a la tolerance (40 = valeur du code) :')
for tol in (20,30,40,50,60,80,100,127):
    r = bubble_mask(crop, tol)
    if r is None: print(f'  tol={tol:3d} -> None'); continue
    f, fill = r
    e = cv2.erode(f, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)),1)
    cov = np.count_nonzero(cv2.bitwise_and(ocr,e))/np.count_nonzero(ocr)
    ok = 0.20<=fill<=0.98
    print(f'  tol={tol:3d} fill={fill:.3f} retenu_par_garde_fou_fill={ok}  couverture_encre={cov:.3f} -> guard applique={ok and cov>=0.5}')
