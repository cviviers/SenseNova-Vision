# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Side-by-side image | ground truth | prediction panels for CelebAMask-HQ runs.

    python tools/evaluation/segment/visualize_celebamask.py --pred_dir results/celebamask_hq/val_binary/pred --limit 8
"""

import argparse
import glob
import os

import numpy as np
from PIL import Image

from tools.evaluation.segment.evaluate_celebamask import gt_label_map

# CelebAMask-HQ demo palette (index = class id).
PALETTE = np.array([
    [0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255],
    [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204],
    [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0],
], dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--dataset_root", default="/datasets/CelebAMask-HQ/CelebAMask-HQ")
    parser.add_argument("--output_dir", default=None, help="Defaults to <pred_dir>/../vis.")
    parser.add_argument("--limit", type=int, default=16)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(os.path.dirname(os.path.normpath(args.pred_dir)), "vis")
    os.makedirs(output_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.pred_dir, "*.png")), key=lambda p: int(os.path.basename(p)[:-4]))
    for path in paths[: args.limit]:
        image_id = int(os.path.basename(path)[:-4])
        image = Image.open(os.path.join(args.dataset_root, "CelebA-HQ-img", f"{image_id}.jpg")).convert("RGB")
        image = image.resize((512, 512), Image.BICUBIC)
        gt = Image.fromarray(PALETTE[gt_label_map(args.dataset_root, image_id)])
        pred = Image.fromarray(PALETTE[np.asarray(Image.open(path))])
        panel = Image.new("RGB", (1536, 512))
        for i, im in enumerate((image, gt, pred)):
            panel.paste(im, (512 * i, 0))
        panel.save(os.path.join(output_dir, f"{image_id}.jpg"), quality=90)
    print(f"wrote {min(len(paths), args.limit)} panels to {output_dir}")


if __name__ == "__main__":
    main()
