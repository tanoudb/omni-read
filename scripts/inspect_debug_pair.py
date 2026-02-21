import sys
from pathlib import Path
import cv2
import numpy as np


def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1


def main(mask_path, overlay_path):
    mp = Path(mask_path)
    op = Path(overlay_path)
    if not mp.exists() or not op.exists():
        print(f"Missing file: {mp.exists()=}, {op.exists()=}")
        return

    mask = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
    overlay = cv2.imread(str(op), cv2.IMREAD_UNCHANGED)

    print(f"mask: {mp} -> shape={mask.shape} dtype={mask.dtype}")
    print(f"overlay: {op} -> shape={overlay.shape} dtype={overlay.dtype}")

    # Normalize mask to single channel
    if mask is None or overlay is None:
        print("Error reading images")
        return

    if mask.ndim == 3:
        mask_ch = mask[:, :, 0]
    else:
        mask_ch = mask

    unique = np.unique(mask_ch)
    print(f"mask unique values (sample up to 20): {unique[:20]}")
    print(f"mask nonzero count: {int(np.sum(mask_ch>0))}")

    bbox = bbox_from_mask(mask_ch)
    print(f"mask bbox: {bbox}")

    # Save a visualization: overlay red where mask>0 on overlay image
    vis = overlay.copy()
    if vis.shape[0:2] != mask_ch.shape[0:2]:
        # try to resize mask to overlay size
        try:
            mask_r = cv2.resize(mask_ch, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)
            print("Resized mask to match overlay shape")
        except Exception as e:
            print(f"Could not resize mask: {e}")
            return
    else:
        mask_r = mask_ch

    red = np.zeros_like(vis)
    red[:] = (0,0,255)
    alpha = (mask_r.astype(np.float32) / 255.0)[:,:,None]
    alpha = np.clip(alpha, 0.0, 1.0)
    blended = (vis.astype(np.float32) * (1-alpha) + red.astype(np.float32) * alpha).astype(np.uint8)

    outp = Path(r"A:\omni read\temps") / f"inspect_vis_{mp.stem}.png"
    cv2.imwrite(str(outp), blended)
    print(f"Wrote visualization: {outp}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: inspect_debug_pair.py <mask.png> <overlay.png>")
    else:
        main(sys.argv[1], sys.argv[2])
