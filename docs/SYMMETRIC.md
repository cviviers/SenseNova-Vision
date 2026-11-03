# Symmetric flow: one model for segmentation and synthesis

The other training modes treat segmentation and synthesis as two conditional tasks: a clean input in a cached prefix,
then a noisy target that only reads from it. The `symm` mode learns one flow over a photo and its colour-coded map
instead, in the spirit of symmetrical flow matching. The `symm_legend` mode does the same with the map in the
colour-legend format of `img2palette_legend` (below).

## The flow

With photo latent `z_x`, map latent `z_m`, independent noise for each, and one shared `t` drawn uniformly:

```
photo stream  (1 - t) z_x + t noise_x     noise level t
map stream    t z_m + (1 - t) noise_m     noise level 1 - t
```

At `t = 0` the photo is clean and the map is noise, which is where segmentation starts. At `t = 1` the map is clean and
the photo is noise, which is where synthesis starts. The model predicts `noise - data` for both streams, and both have a
loss. Each modality keeps its own noise, so synthesis stays one-to-many.

## The sequence

```
[photo ViT tokens, optional] [direction-free prompt] [ photo VAE tokens | map VAE tokens ]
                                                     one noise split, sees the prefix and itself
```

| Choice | Why |
|---|---|
| Only VAE tokens join the flow | ViT tokens have no time embedding, noise, or output head. They stay a clean prefix. |
| ViT prefix on half of the samples (`symm_vit_prob`) | Conditions segmentation on the understanding path. With it the photo never fully disappears, so this is a conditional flow; `symm_vit_prob: 0` is the pure one. |
| RoPE: photo 0, ViT 1, text from 2, map after the text | Keeps the source-to-target distance of the other modes. Adjacent ids cost most of the warm start. |
| Each token carries its true noise level | There is no adaLN: the time embedding is added per token, so `t = 0` matches what the model already knows. |
| Uniform `t` plus endpoints (`symm_endpoint_prob`), `--timestep_shift 1.0` | Both ends are task starts, and one-step readout sits exactly at `t = 0`, where uniform `t` rarely lands. Any other shift breaks noise levels `t` and `1 - t`, so the trainers refuse it. |
| One prompt that names both images | A directional prompt fails at the opposite end of the flow. |
| Map dequantization (`symm_beta`) | Flat colours are perturbed with uniform noise before encoding. |
| Learned stream identity (`StreamEmbedding`) | The photo is `c + u` and the map `c - u`, added to each token. A stream of pure noise has no content to tell what it should become: without this, training makes the model copy the clean stream at `t = 0`, even though the untrained warm start looks fine. It is saved in the adapter; `tools/export_full_checkpoint.py` and `tools/merge_lora.py` write it to `symm_embedding.safetensors`. |

`--symm_image_weight` scales the photo stream's loss (default 0.5). `symm_text_dropout` in the dataset YAML leaves out
the prompt and legend together in that share of samples, which trains the text-free branch that guidance needs at
inference (`--symm_cfg_scale`). The legend configs use 0.1.

## The legend variant (`symm_legend`)

```
[photo ViT, optional] [direction-free prompt + class list] [colour legend] [ photo VAE tokens | map VAE tokens ]
                                                           CE only with ViT    map in the legend's colours
```

The legend is the text of `img2palette_legend`: `<p>class<color>(R, G, B)</color></p>` for each class present, largest
first, with colours that belong to the legend position (a fixed farthest-point sequence), not to the class. It depends
only on the label map, so it can sit in the prefix that both directions share:

| Direction | Legend |
|---|---|
| Synthesis | Computed from the input map, which is drawn in the legend's colours. |
| Segmentation | Generated first, greedily, from the ViT prefix and the prompt; the map is then read through it. |

The text comes before the noise split, so it sees the photo only through the ViT prefix. The legend therefore has a loss
only in samples with that prefix, and segmentation always uses it. Everything else is the `symm` mode unchanged. Compared
with the official palettes, the legend's colours are far apart, and on COCO-Stuff the class list and legend carry the
taxonomy that the palette prompt leaves to the colours.

## Training

Set `mode_weights: {symm: 1.0}` or `{symm_legend: 1.0}` in a dataset YAML, as in
[celebamask_symm.yaml](../data/configs/celebamask_hq/celebamask_symm.yaml),
[celebamask_symm_legend.yaml](../data/configs/celebamask_hq/celebamask_symm_legend.yaml) and their COCO-Stuff and
Spike counterparts. A `symm_legend` run starts best from an adapter trained on `img2palette_legend`.

```bash
bash scripts/finetune_lora.sh data/configs/celebamask_hq/celebamask_symm.yaml --output_dir results/symm/<run> \
  --timestep_shift 1.0 --init_adapter results/spike/celeba-lora-v3/final
```

`--init_adapter` starts from an existing adapter with a fresh optimizer and step count. `--symm_embedding False` trains
without the stream identity, which is only useful to reproduce the failure described above. The full trainer accepts the
same YAML, also with `--timestep_shift 1.0`.

Read the per-stream curves, not the total. `image_t0.4` and `mask_t0.4` in `train_log.jsonl` are the loss of each stream
for `t` in that bin. Each stream's clean end has an irreducible floor near 1, because noise cannot be predicted from
data, so these totals are not comparable with the other modes.

## Evaluation

```bash
# Segmentation: one step from t = 0, without guidance
python -m inference.benchmark.eval_seg_img2palette --dataset celebamask --adapter <dir> --output_dir <out> --symm
# Synthesis: 25 steps from t = 1, with the photo check, round trip and FID of the other modes
python -m inference.benchmark.eval_seg_seg2img --dataset celebamask --adapter <dir> --output_dir <out> --symm
```

| Flag | Effect |
|---|---|
| `--symm_seg_steps`, `--symm_gen_steps` | Number of steps in each direction |
| `--symm_no_vit` | Segment without the ViT prefix |
| `--symm_recompute_known` | Compute the known stream from its data at every step instead of integrating it |
| `--symm_beta` | Map dequantization in synthesis (default: as in training) |
| `--decoder <decoder.pt>` | Read the map through a segmentation decoder instead of the VAE |
| `--symm_cfg_scale` | Text guidance on the created stream (1 = none); needs `symm_text_dropout` in training |

For a one-step baseline of a conventional model: `--num_timesteps 2 --cfg_text_scale 1.0`.

A `symm_legend` model takes `--format legend` in both scripts. Segmentation then reports the legend's class presence
against the ground truth (precision, recall, F1), as every legend evaluation does.

| Flag | Effect |
|---|---|
| `--legend_cache <dir>` | Keep generated legends per image and reuse them in later runs of the same model |
| `--oracle_legend` | Segment with the ground-truth legend: separates map quality from legend errors |

## Segmentation decoder

The VAE is trained on natural images and blurs flat colours, so a small decoder can read the map latent into classes
directly. It trains on one-step latents of a frozen model, half predicted at a uniform `t` and half ground truth, with
cross-entropy + Dice:

```bash
python -m train.train_seg_decoder --dataset celebamask --flow symm --adapter <dir> --output_dir <out> --size S
python -m train.train_seg_decoder --dataset celebamask --flow existing --adapter <dir> --output_dir <control> --size S
python -m inference.benchmark.eval_seg_img2palette --dataset celebamask --adapter <dir> --output_dir <eval> \
  --symm --decoder <out>/decoder.pt
```

Always train the `existing` control too. It is the same decoder on the same data without the symmetric flow, and it shows
how much of a gain comes from the decoder rather than from the flow. Evaluate it with `--decoder` and without `--symm`.
The decoder's architecture is a plain upsampling convolutional stack; sizes `S`, `M` and `L` differ in width and depth.

With `--legend` the decoder predicts legend slots instead of classes (0 = background or unlabeled, `k` = the `k`-th
legend entry, up to `--max_slots`), and evaluation maps slots to classes through the generated legend. The `existing`
control is then a model trained on `img2palette_legend`, evaluated with `--format legend --decoder` and without `--symm`:
its own legend, one conditional step, the same decoder.

Predicting a latent costs a pass of the whole model, which dominates training. `--cache_dir <dir>` predicts the latents
once (`--cache_predicted` per image, plus the ground truth) and trains from memory. On Spike:

```bash
bash scripts/spike_submit.sh <run> decoder celebamask --flow symm --legend --adapter /data/runs/<lora run>/final \
  --cache_dir /data/runs/<run>/cache --size S
```

## Checks

```bash
python -m train.test_symm --model_path results/tiny_model
```
