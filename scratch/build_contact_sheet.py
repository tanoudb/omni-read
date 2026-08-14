# -*- coding: utf-8 -*-
"""Construit des planches-contact (grille avant/après) à partir des crops
générés par render_iterate.py, pour une revue visuelle rapide."""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def build(run_dir: Path, per_sheet: int = 10, cell_w: int = 260, cell_h: int = 260):
    bubbles_dir = run_dir / "bubbles"
    meta = json.loads((run_dir / "bubbles_meta.json").read_text(encoding="utf-8"))

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    def fit(img: Image.Image, w: int, h: int) -> Image.Image:
        img = img.copy()
        img.thumbnail((w, h), Image.LANCZOS)
        canvas = Image.new("RGB", (w, h), (40, 40, 40))
        canvas.paste(img, ((w - img.width) // 2, (h - img.height) // 2))
        return canvas

    sheets_dir = run_dir / "contact_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    n = len(meta)
    n_sheets = (n + per_sheet - 1) // per_sheet
    label_h = 20
    row_h = cell_h + label_h

    for s in range(n_sheets):
        chunk = meta[s * per_sheet:(s + 1) * per_sheet]
        sheet_w = cell_w * 2 + 20
        sheet_h = row_h * len(chunk) + 10
        sheet = Image.new("RGB", (sheet_w, sheet_h), (20, 20, 20))
        draw = ImageDraw.Draw(sheet)
        for row, item in enumerate(chunk):
            idx = item["index"]
            y0 = row * row_h + 5
            before_p = bubbles_dir / f"{idx:02d}_before.png"
            after_p = bubbles_dir / f"{idx:02d}_after.png"
            if before_p.exists():
                b = fit(Image.open(before_p).convert("RGB"), cell_w, cell_h)
                sheet.paste(b, (5, y0 + label_h))
            if after_p.exists():
                a = fit(Image.open(after_p).convert("RGB"), cell_w, cell_h)
                sheet.paste(a, (cell_w + 15, y0 + label_h))
            label = f"#{idx:02d} [{item['class']}] risk={item.get('ghost_risk', False)}  \"{item['ocr_text'][:40]}\""
            draw.text((5, y0), label, fill=(255, 255, 0), font=font)
            draw.text((5, y0 + label_h), "AVANT", fill=(0, 255, 0), font=font)
            draw.text((cell_w + 15, y0 + label_h), "APRES", fill=(0, 200, 255), font=font)
        out_path = sheets_dir / f"sheet_{s:02d}.png"
        sheet.save(out_path)
        print(f"Wrote {out_path} ({len(chunk)} bulles)")


if __name__ == "__main__":
    run_dir = Path(sys.argv[1])
    build(run_dir)
