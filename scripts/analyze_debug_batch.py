import csv
from pathlib import Path
import cv2
import numpy as np

def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1


def analyze_pair(mask_path, overlay_path, out_dir):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    overlay = cv2.imread(str(overlay_path), cv2.IMREAD_UNCHANGED)
    if mask is None or overlay is None:
        return None
    if mask.ndim == 3:
        mask_ch = mask[:, :, 0]
    else:
        mask_ch = mask
    h, w = mask_ch.shape[:2]
    total = h * w
    nonzero = int(np.sum(mask_ch > 0))
    nonzero_ratio = nonzero / total if total>0 else 0.0
    bbox = bbox_from_mask(mask_ch)
    bbox_area = 0
    bbox_frac = 0.0
    if bbox:
        bx1, by1, bx2, by2 = bbox
        bbox_area = max(0, (bx2 - bx1) * (by2 - by1))
        bbox_frac = bbox_area / total if total>0 else 0.0
    # visual
    vis = overlay.copy()
    mask_r = mask_ch
    if vis.shape[0:2] != mask_ch.shape[0:2]:
        mask_r = cv2.resize(mask_ch, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)
    # draw bbox
    if bbox:
        cv2.rectangle(vis, (bx1, by1), (bx2-1, by2-1), (0,255,0), 2)
    # overlay red
    red = np.zeros_like(vis); red[:] = (0,0,255)
    alpha = (mask_r.astype(np.float32)/255.0)[:,:,None]
    blended = (vis.astype(np.float32)*(1-alpha) + red.astype(np.float32)*alpha).astype(np.uint8)
    out_vis = out_dir / f"analysis_vis_{mask_path.stem}.png"
    cv2.imwrite(str(out_vis), blended)
    return {
        'mask': str(mask_path.name),
        'overlay': str(overlay_path.name),
        'shape_h': h,
        'shape_w': w,
        'nonzero': nonzero,
        'nonzero_ratio': nonzero_ratio,
        'bbox': bbox if bbox else '',
        'bbox_area': bbox_area,
        'bbox_frac': bbox_frac,
        'vis': str(out_vis.name),
    }


def main():
    temps = Path(r"A:\omni read\temps")
    out_dir = temps
    masks = sorted(temps.glob('real_mask_*.png'))
    rows = []
    paired = 0
    for m in masks:
        stem = m.stem.replace('real_mask_', '')
        o = temps / f"real_overlay_{stem}.png"
        if not o.exists():
            continue
        paired += 1
        info = analyze_pair(m, o, out_dir)
        if info:
            rows.append(info)
    # write csv
    csvp = out_dir / 'analysis_report.csv'
    with open(csvp, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['mask','overlay','shape_h','shape_w','nonzero','nonzero_ratio','bbox','bbox_area','bbox_frac','vis']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Processed {len(rows)} pairs. Report: {csvp}")

if __name__ == '__main__':
    main()
