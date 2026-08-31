import sys, pickle, numpy as np, cv2
sys.path.insert(0, '.')
CACHE = 'scratch/bareme/cache/30-years-have-passed-since-the-prologue__p01'
d = pickle.load(open(CACHE + '/dets.pkl', 'rb'))
it = d['items'][14]
page = cv2.imread(CACHE + '/page.png')
x1, y1, x2, y2 = it['bbox']
crop = page[y1:y2, x1:x2]
h, w = crop.shape[:2]
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

B = 5
bd = w / B
def prof(m, label):
    cols = [int(np.count_nonzero(m[:, int(i*bd):int((i+1)*bd)])) for i in range(B)]
    print(f'{label:32s} total={int(np.count_nonzero(m)):6d} bands={cols}')

print('taille bande =', int(bd*h))
print('gray median par bande :', [int(np.median(gray[:, int(i*bd):int((i+1)*bd)])) for i in range(B)])
print('gray p10/p90 par bande:', [(int(np.percentile(gray[:, int(i*bd):int((i+1)*bd)],10)),
                                   int(np.percentile(gray[:, int(i*bd):int((i+1)*bd)],90))) for i in range(B)])

ref = float(np.median(gray[h//3:2*h//3, w//3:2*w//3]))
print('ref (mediane tiers central) =', ref)
similar = (np.abs(gray.astype(np.int16) - ref) < 40).astype(np.uint8) * 255
prof(similar, 'A. |gray-ref|<40')
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
er = cv2.erode(similar, k, 1)
prof(er, 'B. erode 7x7')
n, lab, stats, _ = cv2.connectedComponentsWithStats(er, 8)
print('n composantes =', n, ' 5 plus grandes aires =', sorted(stats[1:, cv2.CC_STAT_AREA])[-5:])
big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
m = (lab == big).astype(np.uint8)*255
prof(m, 'C. plus grande composante')
print('   bbox de la composante x,y,w,h =', stats[big][:4])
m2 = cv2.dilate(m, k, 1)
prof(m2, 'D. dilate 7x7')
cnts, _ = cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filled = np.zeros_like(m2); cv2.drawContours(filled, cnts, -1, 255, -1)
prof(filled, 'E. contours remplis')
print('fill ratio =', np.count_nonzero(filled)/(w*h))

# ou est l'encre du texte ? (colonne max du masque OCR complet)
from pipeline import TranslationPipeline as TP
ocr = TP._ocr_mask_from_regions(it['text_regions'], h, w, crop_bgr=crop, dilate=3)
ys, xs = np.nonzero(ocr)
print('encre OCR : x', xs.min(), '..', xs.max(), ' y', ys.min(), '..', ys.max())
# fraction de l'encre couverte par chaque etape
for name, mm in [('A similar', similar), ('B erode', er), ('C bigCC', m), ('E filled', filled)]:
    cov = np.count_nonzero(cv2.bitwise_and(ocr, mm))/max(1,np.count_nonzero(ocr))
    print(f'   couverture de l encre OCR par {name:10s}: {cov:.3f}')

# gray sur les pixels d encre a droite vs gauche
for i in range(B):
    sl = slice(int(i*bd), int((i+1)*bd))
    sub = ocr[:, sl] > 0
    if sub.sum():
        g = gray[:, sl][sub]
        print(f'  bande {i}: encre n={sub.sum():5d} gray med={np.median(g):5.1f}  fond med={np.median(gray[:, sl]):5.1f}')
cv2.imwrite('scratch/explore/d1_similar.png', similar)
cv2.imwrite('scratch/explore/d1_bigcc.png', m)
cv2.imwrite('scratch/explore/d1_gray.png', gray)
