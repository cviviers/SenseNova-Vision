# COCO-Stuff: photo → palette and mask → photo fine-tuning

The COCO-Stuff counterpart of the CelebAMask-HQ two-mode setup ([`CELEBAMASK_HQ.md`](CELEBAMASK_HQ.md)):
one LoRA run mixes photo → palette segmentation and palette segmentation → photo with 50/50 sampling,
and the same evaluation scripts score both modes.

## Data

Read straight from the zips (nothing to extract), mounted in the container under `/datasets/segmentation/`:

| File | Content |
| --- | --- |
| `train2017.zip`, `val2017.zip` | COCO 2017 images (118,287 train / 5,000 val) |
| `stuffthingmaps_trainval2017.zip` | COCO-Stuff label maps (pixel value = class id − 1, 255 = unlabeled) |

`data/cocostuff.py` holds the 171 used classes (80 things + 91 stuff), the palette and the prompts.
COCO-Stuff has no official colour list, so the palette is generated deterministically by greedy
farthest-point sampling in RGB (closest pair 41.6 apart, median nearest neighbour 48), with black
reserved for unlabeled pixels; `data/configs/cocostuff/cocostuff_classes.json` lists it.

## Modes and prompts

| Mode | Input → target | Prompt |
| --- | --- | --- |
| `img2palette` | photo → COCO-Stuff palette map | `IMG2PALETTE_PROMPT`, starts "Task: photo to segmentation map." — fixed, 79 tokens, no colour list |
| `seg2img` | palette map → photo | `seg2img_prompt(map)`, starts "Task: segmentation map to photo." — colour key of the classes present in the input map (median 204 tokens) |

Differences from CelebAMask-HQ, and why:

* **Colour key.** Listing all 171 class colours would make every prompt ~2,800 tokens, about half of
  each training sequence. Photo → palette therefore uses a short fixed prompt (the palette is a fixed
  convention learned from the targets; listing the classes present would leak the answer). Mask → photo
  lists only the classes present in its input map, which are read from the input itself.
* **Image size.** COCO photos keep their aspect ratio: longest side 512, shortest ≥ 256 (e.g. 512×384).
  Targets are rendered at that exact size, so palette colours are never blended by resizing.
* **Metric protocol.** Unlabeled ground-truth pixels are ignored; mIoU averages the 171 classes.

## Training

```bash
docker exec sensenova-vision bash scripts/finetune_lora.sh \
  data/configs/cocostuff/cocostuff_img2palette_seg2img.yaml \
  --output_dir results/cocostuff_lora_i2p_s2i --total_steps 6000 --save_every 500
```

Every sample draws its mode independently (50/50) from train2017; val2017 stays held out. On an
RTX 5090 (QLoRA NF4, rank 16): ~2,250 tokens/s, 15.3 GiB peak, ~3 samples per step (~3.3 s), so
6,000 steps cover ~18k images (~15% of train2017). The log reports the image loss per mode
(`mse[img2palette]=… mse[seg2img]=…`), also kept in `train_log.jsonl`.

## Evaluation

```bash
docker exec sensenova-vision bash scripts/eval_seg_modes.sh cocostuff \
  results/cocostuff_lora_i2p_s2i/checkpoint-003000 results/cocostuff_eval/step3000 --limit 500
```

| Script | Measures |
| --- | --- |
| `inference/benchmark/eval_seg_img2palette.py --dataset cocostuff` | photo → palette on val2017: outputs snapped to the palette, mIoU over 171 classes (unlabeled ignored), pixel accuracy, per-class IoU, share of off-palette pixels (~9.5 s/image) |
| `inference/benchmark/eval_seg_seg2img.py --dataset cocostuff` | mask → photo: photo rate (model's own check), round-trip layout (photos re-segmented with the same checkpoint, non-photos scored as unlabeled, ratio to the same segmenter on the real photos), FID/KID (clean-fid) against the real photos at the same size, `sheet.jpg` (~13 s/image without the round trip) |

Both resume where they stopped, generate at the training image size, and apply a checkpoint in memory
(no merged copy). FID is biased below ~2k images, so compare checkpoints at equal `--limit` or use KID.
The untuned base model does not know this palette: photo → palette mIoU ≈ 0, and mask → photo returns a
recoloured copy of the map.

Full (all-weights) fine-tuning on multi-GPU machines and the LoRA/full comparison: [`FINETUNE.md`](FINETUNE.md).
