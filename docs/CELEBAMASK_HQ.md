# Face parsing on CelebAMask-HQ

Zero-shot face parsing of the CelebAMask-HQ validation split with
SenseNova-Vision, scored by IoU against the official 19-class label maps.

| File | Purpose |
| --- | --- |
| `inference/benchmark/batch_celebamask_segment.py` | Runs the model, writes `pred/<id>.png` label maps and `masks/<id>.npz` raw per-class masks |
| `tools/evaluation/segment/evaluate_celebamask.py` | Per-class IoU / F1, mIoU, confusion matrix -> `metrics.json` |
| `tools/evaluation/segment/visualize_celebamask.py` | image / ground truth / prediction panels |
| `scripts/run_celebamask_val.sh` | Segmentation + evaluation in one command (resumable) |

## Data and labels

- **Split:** CelebA identities 162771–182637, mapped through
  `CelebA-HQ-to-CelebA-mapping.txt` (the rule in CelebAMask-HQ's
  `g_partition.py`). This gives 2,993 val, 24,183 train and 2,824 test images.
- **Labels:** 0 background, 1 skin, 2 nose, 3 eye_g, 4 l_eye, 5 r_eye, 6 l_brow,
  7 r_brow, 8 l_ear, 9 r_ear, 10 mouth, 11 u_lip, 12 l_lip, 13 hair, 14 hat,
  15 ear_r, 16 neck_l, 17 neck, 18 cloth.
- **Ground truth:** label maps are rebuilt from the per-part PNGs in that order,
  as in CelebAMask-HQ's `g_mask.py`.
- **Predictions:** written at the annotation resolution (512×512). Inputs are
  resized from 1024 to 512 before inference (`--image_size`), which is about 4×
  faster than native resolution.
- **Left/right:** both the model and the annotations use the subject's side,
  checked on pilot images, so no swapping is applied. The metrics also report an
  L/R-merged mIoU.

## How the model is asked

The paper (Sec. 3, *Segmentation*; Appendix, Tables 12–13) uses two output
formats:

- **Generic (semantic/panoptic) segmentation**: the prompt lists categories;
  the model replies with a colour legend `<p>class<color>(R,G,B)</color></p>`
  and a colour-coded mask, decoded through the legend. This is how the ADE20K
  semantic mIoU in Table 3 is produced (`inference/benchmark/batch_panoptic_segment.py`).
- **Binary referring segmentation**: one target per prompt, white-on-black mask.
  Face-parsing data (LaPa) was trained **only** in this format (Table 12).

Both are implemented (`--method`):

| Method | Requests per image | Notes |
| --- | --- | --- |
| `binary` (default) | one mask per class (+ a yes/no question for gated classes) | In distribution for face parts |
| `generic` | one | The paper's semantic route; face parts are out of distribution for it |

Two details of the binary method matter for the score:

1. **Presence gating.** A binary request always returns a mask, even when the
   part is absent. In pilots, "hat" on a bare head covered 30–45% of the image.
   For eyeglasses, hat, earring, necklace and clothes (`--gate_classes`), the
   model is first asked a yes/no question in understanding mode (~0.3 s), and
   the class is segmented only on "yes".
2. **Merging overlapping masks** (`--merge`). `area` (the default) paints larger
   masks first, so small parts stay on top. `class_order` reproduces
   CelebAMask-HQ's own painting order. The raw masks are saved, so
   `--remerge_only --merge ...` rebuilds `pred/` without re-running the model.

## Pilot (first 5 val images, RTX 5090, NF4, 20 steps)

| Configuration | mIoU (18 face classes) | mean F1 | Pixel acc. | Time / image |
| --- | --- | --- | --- | --- |
| binary, gated, `--merge area` | 0.49 | 0.58 | 0.91 | ~70 s |
| same, BF16, 25 GiB budget (3.6 GiB of layers on CPU) | **0.50** | **0.60** | 0.87 | ~130 s |
| same, BF16, 30 GiB budget (no offload) | **0.50** | **0.60** | 0.87 | **~62 s** |
| binary, gated, `--merge class_order` | 0.34 | 0.44 | 0.86 | (same masks) |
| generic | 0.21 | 0.26 | 0.49 | ~9 s (one image drifted into keypoint text: 144 s) |

These are five images only, so treat them as a smoke test, not a benchmark result.
NF4 and BF16 agree on four of the five images. The differences come from one
image (id 7) where BF16's "inner mouth" mask covered the whole face of a subject
with a closed mouth (the ground truth has no mouth). That is the same absent-part
hallucination as "hat", not a precision effect. Inner mouth is annotated in only
~61% of images, so it is a candidate for presence gating too.

- **Good:** skin 0.86, neck 0.90, hair 0.87, eyes 0.78–0.82, nose 0.60.
- **Weak:** lips, inner mouth and eyebrows (0.03–0.35). The model's "lower lip"
  mask often covers both lips, and "upper lip" sometimes lands on the nose.
- In a separate single-image check, 20 diffusion steps matched 50.

## Running

Inside the container (`docker compose -f docker/docker-compose.yml up -d`; the
host datasets root is mounted read-only at `/datasets`, see `DATASETS_DIR` in
`docker/.env`):

```bash
# Pilot
docker exec -e SENSENOVA_DTYPE=nf4 sensenova-vision \
  bash scripts/run_celebamask_val.sh results/celebamask_hq/pilot --limit 5 --num_timesteps 20

# Full validation split (resumable: re-run the same command after an interruption).
# Fastest on a 32 GB card: BF16 with the whole model on the GPU (peaks at ~32.0 of 32.6 GB).
docker exec -e SENSENOVA_DTYPE=bf16 -e SENSENOVA_MAX_MEM_PER_GPU=30GiB sensenova-vision \
  bash scripts/run_celebamask_val.sh results/celebamask_hq/val_binary --num_timesteps 20
```

Estimated wall time for all 2,993 val images on one RTX 5090:

| Configuration | ~s / image | Full val split |
| --- | --- | --- |
| binary, BF16 (30 GiB, no offload), 20 steps | 62 | ~2.1 days |
| binary, NF4, 20 steps | 70 | ~2.5 days |
| binary, BF16 (25 GiB, offload), 20 steps | 130 | ~4.5 days |
| binary, NF4, 50 steps (model default) | 165 | ~5.5 days |
| generic, NF4, 20 steps | 9 | ~8 h |

To split across GPUs or machines, run one process per GPU with
`--num_shards N --shard_id i` and the same `output_dir`, then evaluate once:

```bash
python tools/evaluation/segment/evaluate_celebamask.py --pred_dir results/celebamask_hq/val_binary/pred
python tools/evaluation/segment/visualize_celebamask.py --pred_dir results/celebamask_hq/val_binary/pred --limit 20
```

## Outputs

```text
<output_dir>/
  pred/<id>.png          uint8 label map, 512x512, CelebAMask-HQ class ids (compare directly with GT)
  masks/<id>.npz         raw per-class binary masks (bit-packed) for re-merging
  log_shard<k>.jsonl     per-image time, presence answers (binary) or legend text (generic)
  config_shard<k>.json   run arguments and prompted classes
  metrics.json           per-class IoU/F1, mIoU variants, 19x19 confusion matrix
  vis/<id>.jpg           panels from visualize_celebamask.py
```

`evaluate_celebamask.py --save_gt_dir DIR` also writes the rebuilt ground-truth
label maps, so the IoU can be recomputed with other tools.

## Fine-tuning: photo → palette and mask → photo (one run, 50/50)

`data/configs/celebamask_hq/celebamask_img2palette_seg2img.yaml` trains two modes of
`CelebAMaskMultiTaskDataset` (`data/celebamask.py`) in a single LoRA run. Each sample draws its mode
independently with equal weight from the 24,183 train-split images; val/test stay held out.

| Mode | Input → target | Prompt (`data/celebamask.py`) |
| --- | --- | --- |
| `img2palette` | face photo → official-palette segmentation map | `IMG2PALETTE_PROMPT`, starts "Task: face photo to segmentation map." |
| `seg2img` | official-palette segmentation map → face photo | `SEG2IMG_PROMPT`, starts "Task: segmentation map to face photo." |

```bash
docker exec sensenova-vision bash scripts/finetune_lora.sh \
  data/configs/celebamask_hq/celebamask_img2palette_seg2img.yaml \
  --output_dir results/celebamask_lora_i2p_s2i --total_steps 6000 --save_every 500
```

On an RTX 5090 (QLoRA NF4, rank 16): ~3.4 s per step of 2 samples, ~2,100 tokens/s, 15.1 GiB peak;
one pass over the train split is ~12k steps. The log reports the image loss per mode
(`mse[img2palette]=… mse[seg2img]=…`) and `train_log.jsonl` in the output directory keeps it per
logging step.

### Evaluating a checkpoint

```bash
# both modes; the adapter is applied in place (no merged copy needed); "base" evaluates the base model
docker exec sensenova-vision bash scripts/eval_celebamask_modes.sh \
  results/celebamask_lora_i2p_s2i/checkpoint-003000 results/celebamask_eval/step3000 --limit 500
```

| Script | Measures |
| --- | --- |
| `inference/benchmark/eval_seg_img2palette.py --dataset celebamask` | photo → palette on the split: outputs snapped to the palette, mIoU over the 18 face classes plus the full `evaluate_celebamask.py` report (F1, L/R merged), share of off-palette pixels |
| `inference/benchmark/eval_seg_seg2img.py --dataset celebamask` | mask → photo: face rate, round-trip layout (the faces re-segmented with the same checkpoint's `img2palette` mode, non-faces scored as background, ratio to the same segmenter on the real photos), FID/KID (clean-fid) against the real photos of the same images, `sheet.jpg` |

`scripts/eval_celebamask_modes.sh` is a shortcut for `scripts/eval_seg_modes.sh celebamask …`; the same
scripts evaluate COCO-Stuff (see [`COCOSTUFF.md`](COCOSTUFF.md)). Untuned base model, 2 val images:
photo → palette mIoU 0.17, mask → photo outputs are copies of the map (face rate 0).

Both resume where they stopped. FID is biased below ~2k images, so compare checkpoints at the same
`--limit` (KID is the better small-sample number); the round trip shares weights between generator
and judge, so treat it as a consistency check alongside FID.
