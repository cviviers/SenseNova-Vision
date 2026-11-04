# Symmetric flow: implementation review

Review of the modes `symm` and `symm_legend` on branch `symsense` at 09e4683, done on 2026-11-04. It covers the
dataset and packing code, the model changes, the LoRA trainer (and the full trainer where it shares code), the sampler,
the evaluation path and iREPA. It does not cover the segmentation decoder, which has not been trained yet.

## Summary

- **The implementation is correct as designed.** I found no bug that invalidates the trained runs or their evaluations.
  Training and inference agree on noise levels, velocity targets, sampler signs, sequence layout, RoPE ids, attention,
  loss masks and the text-free branch used for guidance.
- **Three small defects**, none of which changes current numbers: a shared temporary file in the legend cache, a silent
  fallback when the stream identity is missing, and a photo check judged by the model under test (section 3).
- **Four design points that probably cost accuracy**, each with a cheap test (section 4):
  1. Multi-step segmentation has to noise the photo it conditions on. The default sampler also produces photo states
     that training never shows.
  2. About 20% of the flow loss sits at the clean endpoints, where the target is mostly unpredictable noise.
  3. The learned stream identity barely moves from its initial value.
  4. Guidance removes the legend together with the prompt.
- **Evidence that the method works** (first 1,000 val images, paired bootstrap):
  - On COCO, v4 + symmetric fine-tuning at 10 guided steps is 0.005 below v4 at 50 steps, and the 95% interval
    includes zero.
  - Legends generated from the ViT prefix alone match v4's legends on both datasets.
  - CelebA at one step is 0.134 below v4. Its 10-step and 50-step results, and the from-base runs that give the clean
    comparison with v4, were still running at the time of writing.

## Update: 50-step focus (2026-11-04, late evening)

- **Focus:** inference at 50 steps. One- and ten-step inference waits until the symmetric model beats v4 at 50 steps.
  The work now concentrates on the from-base symmetric runs; the v4-start and iREPA variants wait.
- **The clean comparison:** COCO, trained from base with v4's recipe, first 1,000 val images, paired against v4 at
  50 steps (0.4782):

  | Setting | mIoU | Difference | 95% CI |
  |---|---|---|---|
  | 50 steps, guidance 2 | 0.4707 | −0.007 | −0.016 to +0.002 |
  | 50 steps, guidance 4 | 0.4720 | −0.006 | −0.015 to +0.003 |

  The two are level, with no gain for the symmetric objective. Legend F1 is 0.697, against v4's 0.705.

- **The design points at 50 steps:**

  | Point | At 50 steps |
  |---|---|
  | 4.1 The photo is noised during segmentation | The main point. Half of the steps run with the photo more than half noise, while v4 reads a clean photo at all 50. |
  | 4.2 Unlearnable loss at the clean endpoints | Holds. The endpoint share mainly serves one-step readout, so it can shrink. |
  | 4.3 The stream identity barely trains | Mostly irrelevant: it matters where a stream is pure noise, i.e. the first step. |
  | 4.4 Guidance removes the legend | Holds, low priority: guidance 2 and 4 score the same. |
  | 4.5 Sampler: grid, Heun, batched guidance | Drops; these are few-step and speed issues. |

- **Implemented in the working tree** (not yet committed; every option defaults to the old behaviour):
  - `symm_cond_prob` and `symm_cond_photo_share` (dataset YAML): conditional slices, which address 4.1.
  - `--symm_hold_known` (evaluation): the sampler keeps the known stream clean at every step, for models trained with
    slices.
  - `--symm_clean_weight` (both trainers): the loss weight of a stream at its clean endpoint, which addresses 4.2. The
    logs report those rows as `image_clean` and `mask_clean`.
  - Fixes for D1 (per-process temporary file) and D2 (a warning when `--symm` runs without a stream identity).
  - Tests for the slices, the weights, slice parity and the hold-known sampler. The CPU subset passes; the GPU checks
    have not run yet.
  - Configs `spike/{celebamask,cocostuff}_symm_legend_cond.yaml`: half the samples in slices, 5% endpoints.
- **Running on Spike:** coco-symm-legend at 50 steps with `--symm_recompute_known`, which tests the eval-only part of
  4.1, and CelebA's 50-step jobs.

## How this was checked

- I read every change between ae836ee and 09e4683 in the flow's path:
  - data: [dataset_base.py](../data/dataset_base.py), [seg_multitask.py](../data/seg_multitask.py), and the two dataset
    modules;
  - model: [bagel.py](../modeling/bagel/bagel.py), [modeling_utils.py](../modeling/bagel/modeling_utils.py),
    [qwen2_navit.py](../modeling/bagel/qwen2_navit.py);
  - training: [finetune_lora.py](../train/finetune_lora.py), [train_utils.py](../train/train_utils.py),
    [irepa.py](../train/irepa.py);
  - inference and evaluation: [symmetric.py](../inference/symmetric.py), the three files under
    [inference/benchmark](../inference/benchmark/).
- I traced one `symm_legend` sample through packing, the forward pass and the loss, and the same pair through inference:
  cache, joint query, sampler and readout.
- Numerical checks, all on CPU:
  - stream identity weights in 14 checkpoints (steps 500 to 2000);
  - BF16 rounding of the identity against the base model's latent-embedding weights;
  - per-stream loss curves from the logs of three finished training runs;
  - v4's legend presence F1, recomputed from its evaluation logs.
- Not re-run: `train/test_symm.py`, because the local GPU is reserved for other work. Its `check_parity`, which compares
  the training forward pass with the inference path on identical inputs (within 1%), is the main existing evidence that
  the layout matches.

## 1. What the implementation does

A training sample is:

```
[photo ViT, 50%] [direction-free prompt] [legend, symm_legend only] [ photo VAE stream | map VAE stream ]
 RoPE 1           RoPE from 2             CE only with ViT            RoPE 0              RoPE after the text
                  (prompt and legend dropped together in 10% of samples)   one "noise" split, full attention inside
```

- **Timesteps:** one t per sample, uniform in [0.001, 0.999], with 20% of samples at either end. The photo has noise
  level t and the map 1 − t, each with its own Gaussian noise.
- **Loss:** both streams predict `noise − data`, with the photo's loss weighted 0.5.
- **Stream identity:** a learned FP32 offset, `c + u` for the photo and `c − u` for the map, is added to every latent
  token.
- **Segmentation:**
  1. Generate the legend from the ViT prefix and the prompt.
  2. Cache the prefix.
  3. Integrate t from 0 to 1, with the clean photo and a noise map in one query.
  4. Read the map through the legend.
- **Synthesis:** integrate t from 1 to 0, starting from the dequantized map and a noise photo.

## 2. Verified correct

| Property | Where | Check |
|---|---|---|
| Noise levels t and 1 − t | [dataset_base.py:39](../data/dataset_base.py#L39), [:516](../data/dataset_base.py#L516); [bagel.py:250](../modeling/bagel/bagel.py#L250) | The map's logit is negated, and sigmoid(−x) = 1 − sigmoid(x). Both trainers refuse a timestep shift other than 1.0, which would break the identity. |
| Independent noise per stream | [bagel.py:249](../modeling/bagel/bagel.py#L249) | One `randn_like` over all latent tokens. |
| Targets and sampler signs | [bagel.py:302](../modeling/bagel/bagel.py#L302), [:886-892](../modeling/bagel/bagel.py#L886) | The photo's velocity is `noise − data`, applied with +dt. The map's noise level is 1 − t, so its update is −v·dt. Correct in both directions, including the formulas of `--symm_recompute_known`. |
| One-step readout | [symmetric.py:133](../inference/symmetric.py#L133), `generate_symmetric` | One step from t = 0 returns `noise − v`, the model's estimate of the clean map. |
| Layout, RoPE and attention parity | [seg_multitask.py:134](../data/seg_multitask.py#L134); [symmetric.py:44-77](../inference/symmetric.py#L44) | Same ids in both paths: photo 0, ViT 1, text from 2, map after the text. Both streams sit in one noise split during training and in one query at inference; the cache and query indexes in `prepare_vae_latent` are consistent. `check_parity` covers with and without ViT, with a legend, and with text dropped. |
| Loss masks | [seg_multitask.py:147-151](../data/seg_multitask.py#L147) | The legend has CE only with the ViT prefix, the prompt never does, and dropout removes prompt and legend together. The loss images have `enable_cfg = 0`, so conditioning dropout never removes a stream. |
| Guidance branch | [symmetric.py:95-98](../inference/symmetric.py#L95); [bagel.py:885](../modeling/bagel/bagel.py#L885) | The text-free query matches training's text-dropped layout: the map gets RoPE id 2, with or without ViT. Guidance applies to the created stream only. |
| Photo loss weight | [train_utils.py:67-100](../train/train_utils.py#L67) | MSE rows follow the latent tokens with a finite timestep, in order. Each symm sample's rows split in half, photo first, which holds because both images have the same size. |
| Stream identity storage | `lora_state_dict`; [seg_eval_common.py:234](../inference/benchmark/seg_eval_common.py#L234) | Saved in the adapter and restored from it, or from `symm_embedding.safetensors` in an exported model. |
| iREPA bookkeeping | [irepa.py:67-100](../train/irepa.py#L67) | Hidden states, padded images and token grids follow the same packing order. Only the head trains, and it is not used at inference. |
| Eval preprocessing | [seg_eval_common.py:122-130](../inference/benchmark/seg_eval_common.py#L122); `generate()` in eval_seg_seg2img.py | Photo and label are brought to the training size (label nearest), the synthesis input is dequantized as in training, and maps are read with v4's legend parser. |

## 3. Defects

| # | Defect | Effect | Fix |
|---|---|---|---|
| D1 | The legend cache writes every legend through the same `<key>.txt.tmp` ([seg_eval_common.py:379](../inference/benchmark/seg_eval_common.py#L379)). | Two jobs writing one cache can crash on `os.replace` or interleave writes. The final evaluations avoid this with separate caches. | Use a per-process name, e.g. `f"{path}.{os.getpid()}.tmp"`. |
| D2 | `_stream_embedding` returns 0 when no identity is loaded ([bagel.py:122](../modeling/bagel/bagel.py#L122)). | A symm adapter or merged model without its identity evaluates without any error. The streams then differ only by position and noise level. | In `symm_flow`, raise when `--symm` is set and `model.symm_embedding is None`. |
| D3 | The photo check uses the model under test as the judge ([eval_seg_seg2img.py:67](../inference/benchmark/eval_seg_seg2img.py#L67)). | Symm adapters change the understanding expert, so the judge differs between the models being compared. | Add a `--judge base` option. `results/symm/scripts/rejudge.py` already re-scores saved photos with the base model. |

## 4. Design points that may cost accuracy

### 4.1 Multi-step segmentation loses the photo along the way

- **Why:** training only sees the diagonal: photo at level t, map at 1 − t. The photo is clean only at t = 0.
  - One-step segmentation starts there and is consistent with training.
  - Every further step also noises the photo. At t = 0.5 the map is half formed and the photo half noise; the last
    steps refine the map against an almost pure-noise photo.
  - v4, by contrast, reads a clean cached photo at all 50 steps.
- **The default sampler adds a second problem.** It moves the photo with the model's own velocity
  ([bagel.py:886](../modeling/bagel/bagel.py#L886)).
  - At t = 0 that velocity is the expected noise minus the photo.
  - The expected noise given a clean photo is zero, so after one step the photo is about (1 − dt)·x0, with almost no
    noise.
  - Training never shows a photo at level t without its Gaussian noise, so from the second step on the photo is off
    the training distribution.
- **The existing alternative:** `--symm_recompute_known` sets the photo to (1 − t)·x0 + t·ε with one fixed ε at every
  step ([bagel.py:888-892](../modeling/bagel/bagel.py#L888)). That is exactly the training distribution. It has been
  run once, on an 8-image overfit check.
- **Tests (evaluation only):**
  1. Rerun the 10-step evaluation with `--symm_recompute_known`. This takes about 20 minutes per run on Spike with the
     cached legends.
  2. Compare the queued 10-step and 50-step results of the from-base runs. If 50 steps do no better than 10, the
     degrading photo is the likely limit.
- **Fix if confirmed: train with conditional slices.**
  - In a share of samples (say 30%), keep one stream clean and draw the other's noise level freely, with no loss on the
    clean stream.
  - Segmentation can then hold the photo clean at every step, as v4 does, while keeping joint attention and the
    diagonal for the rest.
  - The per-token timestep already supports this. The change is in `sample_symm_logit_t`, `pack_sequence`, and a
    hold-known option in `generate_symmetric`.

### 4.2 About 20% of the flow loss cannot be learned

- **The problem:** at a stream's clean end, its target `noise − data` is mostly the noise, which the input does not
  contain. The training logs show this floor:

  | Loss near the clean end, last 20 log windows | CelebA | COCO |
  |---|---|---|
  | Photo, t < 0.2 | 0.81 | 0.82 |
  | Map, t ≥ 0.8 | 0.73 | 0.70 |
  | Map's useful end (segmentation start), t < 0.2 | 0.19 | 0.37 |

- **Share of the loss:** weighted by the t distribution and the photo weight of 0.5, the two near-clean bins make up
  40–46% of the MSE term.
  - The exact endpoints alone make up roughly 20%: 10% of samples at each end, with a loss near 1.
  - Those rows add gradient noise and teach only the trivial −x0 part of the target.
- **The weighting is inverted at the synthesis endpoint:** the useful photo rows there have weight 0.5, and the
  unlearnable map rows have weight 1.0.
- **Fix (one line in [symm_row_weights](../train/train_utils.py#L93)):** give zero weight to a stream whose own noise
  level is at the clean endpoint (below 0.01). A smooth version would weight rows by min(1, level / 0.1). Masking the
  endpoint costs nothing that sampling uses, apart from the default sampler's first photo step in 4.1. The effect is
  probably modest, so it belongs in the next training run rather than a dedicated one.

### 4.3 The stream identity barely trains

| Checkpoint | RMS of u (initial value 0.001) | RMS of c (initial value 0) |
|---|---|---|
| Step 500, six runs | 0.0020–0.0034 | 0.0008–0.0017 |
| Step 1000, six runs | 0.0027–0.0040 | 0.0009–0.0019 |
| Step 2000, two CelebA runs | 0.0034–0.0042 | 0.0011–0.0020 |

- **Expected movement:** Adam moves a parameter with a consistently signed gradient by about the learning rate per step,
  roughly 0.1 over 500 steps here.
- **Observed movement:** 1–3% of that. The gradient on the identity has no consistent sign, so the model tells the
  streams apart mostly by other cues: the RoPE offset, the noise-level embedding and the content.
- **Size against the tokens:**
  - The photo–map difference 2u has norm ≈ 0.5, against a latent token norm of ≈ 90 (element RMS 1.0–2.2, estimated
    from the base model's `vae2llm`, time and position embeddings). That is about 0.6%.
  - After the BF16 cast of the token, 27% of the per-element differences round to zero, and the rest carry about 40%
    error. The direction survives on average.
- **Status:** not a bug. The 8-image overfit showed the identity is needed at least early in training.
- **Test:** one-step segmentation and synthesis on 100 images with the identity zeroed, and with the two ids swapped.
  This needs a flag of about ten lines.
  - If neither changes the output, the model does not use the identity.
  - If swapping breaks it, the identity carries the stream role.
- **Fix options:** a larger initial value (e.g. RMS 0.05) or its own learning rate.

### 4.4 Guidance removes the legend

- **What happens:** the text-free branch drops the prompt and the legend together
  ([seg_multitask.py:147](../data/seg_multitask.py#L147), [symmetric.py:96](../inference/symmetric.py#L96)).
  - Without the legend, that branch cannot know which colour stands for which class, except by guessing the legend order
    from the photo.
  - Guidance therefore amplifies "draw in this legend's colours" along with "follow the prompt".
- **Effect so far:**
  - Guidance 2 helped COCO at 10 steps: 0.4640 to 0.4735 for v4 + symmetric fine-tuning.
  - It hurt CelebA slightly at step 1000: 0.739 to 0.734 on 100 images.
- **Alternatives (both need retraining):** drop only the prompt in a share of samples and keep the legend in both
  branches, or guide on the ViT prefix instead. Low priority.

### 4.5 Sampler details

- **Step grid:** the sampler uses a uniform Euler grid in t ([bagel.py:867](../modeling/bagel/bagel.py#L867)). v4
  samples with a shifted grid that spends more steps where the target is noisy. At 5–10 steps, a grid concentrated at
  the created stream's noisy end, or a second-order (Heun) step, may help.
- **Guidance cost:** the conditional and text-free passes run one after the other
  ([bagel.py:880-885](../modeling/bagel/bagel.py#L880)). Guided 10-step segmentation costs 1.45 s per COCO image,
  against 0.83 s without guidance. Batching both passes into one forward, as the existing parallel guidance path does,
  would cut most of the difference.

### 4.6 iREPA target

- **Current design:** both streams align with the clean photo's DINOv2 features
  ([irepa.py:85](../train/irepa.py#L85)). For the photo stream this is standard representation alignment. For the map
  stream it asks the map's hidden states to carry the photo's features.
- **Results so far:** at steps 1000 and 2000, iREPA was level with the control or slightly below it (COCO at step 1000:
  −0.017 at one step, −0.021 at 10 steps).
- **If the final iREPA numbers stay below the control:** align only the photo stream, or lower the weight.

## 5. Results so far

First 1,000 val images, paired bootstrap against v4 at 50 steps (500 resamples).

| | COCO | CelebA |
|---|---|---|
| v4, 50 steps | 0.4782 | 0.7385 |
| v4 + symmetric fine-tuning, 1 step | 0.4454 (−0.033, CI −0.043 to −0.022) | 0.6042 (−0.134, CI −0.142 to −0.128) |
| 10 steps | 0.4640 (−0.014, CI −0.023 to −0.004) | still running |
| 10 steps, guidance 2 | 0.4735 (−0.005, CI −0.014 to +0.005) | still running |
| Legend presence F1: symmetric (ViT only) / v4 | 0.701 / 0.705 | 0.984 / 0.985 |

- **Legend:** v4 generates its legend from the photo's VAE and ViT tokens. The symmetric models see only the ViT
  prefix, yet their legends are as good, so the legend is not where the gap comes from.
- **COCO synthesis** (100 maps, v4 + symmetric fine-tuning):
  - 82% of outputs judged photos by its own model (see D3);
  - layout round trip 0.25, against a ceiling of 0.42.
- **Still running:** the from-base runs (the clean comparison with v4, at 50 guided steps), the CelebA multi-step
  results and the iREPA runs.

## 6. Recommended next steps

| # | Step | Kind | Cost | Answers |
|---|---|---|---|---|
| 1 | 10-step evaluation with `--symm_recompute_known` on both from-base runs | Evaluation on Spike | ~20 min per run | 4.1: does moving the photo with the model's velocity hurt? |
| 2 | Identity zeroed and swapped: one step and synthesis, 100 images | Evaluation and a small flag | < 1 h | 4.3: is the identity used? |
| 3 | Fix D1 to D3 | Code | Minutes | Robustness; comparable photo rates |
| 4 | Shifted step grid or Heun step, and batched guidance | Code and evaluation | A few hours | 4.5 |
| 5 | Next training run: endpoint loss mask and conditional slices, with a larger identity if step 2 says so | Training | ~20 h per run on one GPU | 4.1–4.3 |

Steps 1 and 2 need no retraining and answer the two largest open questions. Step 5 only fits before the Spike quota
ends on 2026-11-10 if it starts by about 2026-11-08.
