"""
Converts the CraterDANet source repo's per-image CSV crater annotations
into YOLO label format.

Usage:
    git clone https://github.com/yizuifangxiuyh/Lunar_Crater_Detection_Data.git
    python3 build_dataset.py /path/to/Lunar_Crater_Detection_Data

Each source .txt file is a CSV with header "ID,X,Y,W,H,C_X,CY" where
(X, Y) is the top-left corner and (W, H) the width/height of a crater
bounding box, in that image's own pixel coordinates. Train and val images
in this dataset are NOT the same resolution (800x800 vs 1000x1000), so we
read each image's real size rather than assuming a fixed one.
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

from PIL import Image

OUT = Path(__file__).parent / "dataset"


def convert_split(src_root: Path, src_split: str, dst_split: str):
    imgs = sorted((src_root / "LRO_DATA" / src_split).glob("*.png"))
    n_boxes_total = 0
    n_bad = 0
    (OUT / "images" / dst_split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / dst_split).mkdir(parents=True, exist_ok=True)

    for img_path in imgs:
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        w_img, h_img = Image.open(img_path).size
        shutil.copy(img_path, OUT / "images" / dst_split / img_path.name)

        lines_out = []
        with open(txt_path) as f:
            reader = csv.reader(f)
            next(reader)  # header row
            for row in reader:
                if len(row) < 5:
                    continue
                _id, x, y, w, h = row[:5]
                x, y, w, h = float(x), float(y), float(w), float(h)
                cx, cy = (x + w / 2) / w_img, (y + h / 2) / h_img
                nw, nh = w / w_img, h / h_img
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < nw <= 1 and 0 < nh <= 1):
                    n_bad += 1
                    continue
                lines_out.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        n_boxes_total += len(lines_out)
        (OUT / "labels" / dst_split / (img_path.stem + ".txt")).write_text("\n".join(lines_out))

    print(f"{src_split} -> {dst_split}: {len(imgs)} images, {n_boxes_total} boxes, {n_bad} rejected")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    src_root = Path(sys.argv[1])
    convert_split(src_root, "train", "train")
    convert_split(src_root, "test", "val")

    (OUT / "data.yaml").write_text(
        f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: crater\n"
    )
    print(f"Wrote {OUT / 'data.yaml'}")
