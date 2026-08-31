import json, sys
import numpy as np

rows = [r for r in json.load(open('scratch/explore/d3_audit.json')) if 'CH' in r and r.get('CH')]
print('zones exploitables : %d / 632' % len(rows))
cls = {}
for r in rows:
    cls[r['cls']] = cls.get(r['cls'], 0) + 1
print('classes :', cls)


def col(key, sub):
    return np.array([r[key][sub] for r in rows if r.get(key)], float)


def p(name, a):
    print('  %-14s med %7.1f  p75 %7.1f  p90 %7.1f  p99 %7.1f  max %8.1f  moy %7.1f'
          % (name, np.median(a), np.percentile(a, 75), np.percentile(a, 90),
             np.percentile(a, 99), a.max(), a.mean()))


print('\n=== 1. COUVERTURE DE L\'ENCRE SOURCE (cover_pct, plus haut = mieux) ===')
for k in ('CH', 'OCR', 'FULL'):
    p(k, col(k, 'cover_pct'))

print('\n=== 2. RESIDU : zones ou > 15 %% de l\'encre source reste NON effacee ===')
for k in ('CH', 'OCR', 'FULL'):
    a = col(k, 'residu_pct')
    print('  %-5s  >15%%: %3d   >30%%: %3d   >50%%: %3d' %
          (k, (a > 15).sum(), (a > 30).sum(), (a > 50).sum()))

print('\n=== 3. SURFACE EFFACEE (%% de la bbox) ===')
for k in ('CH', 'OCR', 'FULL'):
    p(k, col(k, 'area_pct'))

print('\n=== 4. BAVURE : px effaces hors encre et hors halo 9px (spill_pct_area) ===')
for k in ('CH', 'OCR', 'FULL'):
    p(k, col(k, 'spill_pct_area'))

print('\n=== 5. STRUCTURE DETRUITE : bavure tombant sur un bord Canny (px) ===')
for k in ('CH', 'OCR', 'FULL'):
    p(k, col(k, 'struct_px'))

print('\n=== 6. MORSURE DU TRAIT DE BALLON (px de la bande de contour effaces) ===')
for k in ('CH', 'OCR', 'FULL'):
    a = np.array([r[k]['band_px'] for r in rows if r.get(k) and r[k]['band_px'] >= 0], float)
    print('  %-5s n=%d' % (k, len(a)), end=' ')
    p('', a)

print('\n=== 7. LE GARDE-FOU SE DECLENCHE-T-IL ? (CH vs OCR) ===')
eq = [r for r in rows if r.get('ch_eq_ocr') is True]
ne = [r for r in rows if r.get('ch_eq_ocr') is False]
print('  identiques         : %d (%.0f%%)' % (len(eq), 100.0 * len(eq) / len(rows)))
print('  differents         : %d (%.0f%%)' % (len(ne), 100.0 * len(ne) / len(rows)))
rm = np.array([r['ocr_minus_ch'] for r in ne], float)
rmi = np.array([r['ocr_minus_ch_ink'] for r in ne], float)
add = np.array([r['ch_minus_ocr'] for r in ne], float)
print('  px retires par le garde-fou : med %.0f p90 %.0f max %.0f' %
      (np.median(rm), np.percentile(rm, 90), rm.max()))
print('  dont px D\'ENCRE retires     : med %.0f p90 %.0f max %.0f  (total %.0f)' %
      (np.median(rmi), np.percentile(rmi, 90), rmi.max(), rmi.sum()))
print('  px ajoutes (fermeture 3x3)  : med %.0f' % np.median(add))
tot_rm = sum(r['ocr_minus_ch'] for r in ne)
tot_rmi = sum(r['ocr_minus_ch_ink'] for r in ne)
print('  BILAN : sur %d px retires par les garde-fous, %d (%.0f%%) etaient de l\'ENCRE'
      % (tot_rm, tot_rmi, 100.0 * tot_rmi / max(1, tot_rm)))

print('\n=== 8. LE GARDE-FOU GAGNE-T-IL ? (par zone, CH vs OCR) ===')
gain_band = [r for r in ne if r['OCR']['band_px'] - r['CH']['band_px'] > 50]
gain_struct = [r for r in ne if r['OCR']['struct_px'] - r['CH']['struct_px'] > 50]
lose_ink = [r for r in ne if r['CH']['residu_pct'] - r['OCR']['residu_pct'] > 10]
lose_ink_bad = [r for r in ne if r['CH']['residu_pct'] > 15 and r['OCR']['residu_pct'] <= 15]
print('  zones ou CH protege >50px de trait de ballon en plus : %d' % len(gain_band))
print('  zones ou CH sauve   >50px de structure en plus       : %d' % len(gain_struct))
print('  zones ou CH laisse >10 pts d\'encre en plus non effacee : %d' % len(lose_ink))
print('  zones ou CH bascule en RESIDU (>15%%) alors que OCR non : %d' % len(lose_ink_bad))
print('\n  --- pires pertes du garde-fou (residu cree) ---')
for r in sorted(ne, key=lambda r: r['OCR']['cover_pct'] - r['CH']['cover_pct'], reverse=True)[:12]:
    print('   %-42s #%-3d %-8s cover CH %5.1f%% -> OCR %5.1f%%  trait CH %4d OCR %4d  | %s'
          % (r['page'][:42], r['idx'], r['cls'], r['CH']['cover_pct'], r['OCR']['cover_pct'],
             r['CH']['band_px'], r['OCR']['band_px'], r['text'][:34]))
print('\n  --- meilleurs gains du garde-fou (trait/structure protege) ---')
for r in sorted(ne, key=lambda r: (r['OCR']['band_px'] - r['CH']['band_px']), reverse=True)[:12]:
    print('   %-42s #%-3d %-8s trait CH %5d -> OCR %5d  cover CH %5.1f OCR %5.1f | %s'
          % (r['page'][:42], r['idx'], r['cls'], r['CH']['band_px'], r['OCR']['band_px'],
             r['CH']['cover_pct'], r['OCR']['cover_pct'], r['text'][:34]))

print('\n=== 9. RISQUE DE LA BRANCHE DE REPLI REELLE (FULL, polygones pleins+11px) ===')
worse = [r for r in rows if r['FULL']['area_pct'] - r['CH']['area_pct'] > 20]
print('  zones ou FULL efface >20 pts de bbox de plus que CH : %d (%.0f%%)'
      % (len(worse), 100.0 * len(worse) / len(rows)))
a = np.array([r['FULL']['area_pct'] for r in rows], float)
print('  FULL couvre >60%% de la bbox : %d zones ; >80%% : %d zones'
      % ((a > 60).sum(), (a > 80).sum()))
b = np.array([r['FULL']['band_px'] - r['CH']['band_px'] for r in rows if r['CH']['band_px'] >= 0], float)
print('  trait de ballon en plus (FULL - CH) : med %.0f p90 %.0f max %.0f' %
      (np.median(b), np.percentile(b, 90), b.max()))
