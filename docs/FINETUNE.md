# Fine-tuning recipes: LoRA and full

Two ways to fine-tune SenseNova-Vision on a dataset config — e.g. the two-mode (photo → palette,
mask → photo) configs for CelebAMask-HQ and COCO-Stuff ([`CELEBAMASK_HQ.md`](CELEBAMASK_HQ.md),
[`COCOSTUFF.md`](COCOSTUFF.md)). Both read the same YAMLs, train the same losses (flow-matching MSE on
target images, CE on target text) with the same conditioning dropout, and log the loss per mode.

| | LoRA (`scripts/finetune_lora.sh`) | Full (`scripts/finetune_full.sh`) |
| --- | --- | --- |
| Trainer | `train/finetune_lora.py`, single GPU | `train/pretrain_unified_navit.py`, FSDP over all GPUs |
| Trained | rank-16 adapters on every decoder linear of both experts (80.7M params, FP32) | all weights except the VAE (~14.6B, FP32 master weights) + EMA copy |
| Base weights | frozen; NF4 (`--quant nf4`, default) or BF16 (`--quant bf16`) | trained |
| Memory | NF4 ~15 GB, BF16 ~40 GB (one GPU) | ~290 GB of weights/grads/optimizer/EMA sharded per node, plus activations: ≥ 8 × 80 GB |
| Defaults | lr 1e-4, 300 steps, ~6-8k tokens per step | the release's recipe (`scripts/train_cv_unify.sh`): lr 2.5e-5 constant, EMA 0.995, 2 accumulation steps, 32-36k tokens per GPU per micro-step; 100 warmup updates and 2,000 micro-batch steps (~76 samples each on 8 GPUs; ~150k samples) |
| Checkpoint | `adapter_model.safetensors` (~320 MB) | FP32 `model.safetensors` + `ema.safetensors` (~58 GB each) + sharded optimizer |
| For evaluation | pass `--adapter <dir>` (applied in memory) | `tools/export_full_checkpoint.py` → model dir, pass `--model_path <dir>` |

## LoRA

```bash
bash scripts/finetune_lora.sh <config.yaml> --output_dir results/<run> --total_steps 6000 --save_every 500 \
  [--quant bf16] [--lora_rank 64 --lora_alpha 128]
```

## Full fine-tuning

```bash
# one node: uses every visible GPU; model sharded over the node's GPUs, replicated across nodes
bash scripts/finetune_full.sh <config.yaml> [--total_steps 2000] [--lr 2.5e-5] [...]
# multi-node: run on every node with WORLD_SIZE (nodes), RANK, MASTER_ADDR, MASTER_PORT set
```

* Starts from the released EMA weights (`--resume_from` / `--finetune_from_ema`), keeps the trained
  generation expert (`--copy_init_moe False`), and re-running the same command resumes from the run's
  latest checkpoint (model, optimizer, scheduler, data position).
* Stops after `--total_steps` and always writes a final checkpoint; also saves every `--save_every`
  steps and once at step 10 (an early sanity checkpoint). Budget ~120 GB of disk per checkpoint.
* Environment: `SENSENOVA_MODEL_DIR` (start weights), `SENSENOVA_RUN_DIR` (default `results/full_<config>`),
  `SENSENOVA_GPU_PER_NODE`, `SENSENOVA_NUM_SHARD` (GPUs one model copy is sharded over; default all GPUs
  of a node). Any trainer flag passed after the config overrides the defaults.
* Logs: `<run_dir>/log.txt`, `<run_dir>/train_log.jsonl` (per-mode losses, summed over all GPUs) and an
  offline wandb run.

Export a checkpoint and evaluate it:

```bash
python tools/export_full_checkpoint.py --checkpoint results/full_<config>/checkpoint/0001999 \
  --output_dir results/full_<config>/export_0001999            # BF16 EMA weights + base configs
python -m inference.benchmark.eval_seg_img2palette --dataset celebamask \
  --model_path results/full_<config>/export_0001999 --output_dir results/eval_full/img2palette --limit 500
python -m inference.benchmark.eval_seg_seg2img --dataset celebamask \
  --model_path results/full_<config>/export_0001999 --output_dir results/eval_full/seg2img --limit 500
```

### Smoke test on the target machine

`tools/make_tiny_model.py` builds a 2-layer model with the release's configs, tokenizer and VAE (~120M
params). Running the full recipe on it exercises everything — loading the start weights, FSDP, the
two-mode data, per-mode logging, checkpoint, resume, export, evaluation — in a few minutes on any GPU:

```bash
python tools/make_tiny_model.py results/tiny_model
SENSENOVA_MODEL_DIR=results/tiny_model SENSENOVA_RUN_DIR=results/tiny_full \
  bash scripts/finetune_full.sh data/configs/celebamask_hq/celebamask_img2palette_seg2img.yaml --total_steps 12
```

Validated this way on one RTX 5090. The full 14.6B recipe itself has not been run (it needs the
multi-GPU hardware above).
