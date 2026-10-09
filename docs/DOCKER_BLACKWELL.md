# Docker on NVIDIA Blackwell (RTX 5090 / sm_120)

The reference environment (`setup.sh`: PyTorch 2.5.1 + CUDA 12.4, flash-attn
2.6.3) has no kernels for Blackwell GPUs. `docker/` provides an image that runs
inference and LoRA fine-tuning on a single consumer Blackwell card, validated on
one RTX 5090 (32 GB).

| Component | Reference env | Blackwell image |
| --- | --- | --- |
| Base | conda, host CUDA 12.4 | `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`, Python 3.10 venv |
| PyTorch | 2.5.1+cu124 | 2.8.0+cu128 (sm_70 … sm_120) |
| flash-attn | 2.6.3 (source build) | 2.8.3 prebuilt wheel (cu12 / torch 2.8) |
| Other pins | `requirements.txt` + `constraints.txt` | same files; only torch/torchvision/triton, `nvidia-*` and sympy pins are dropped |
| Extras | — | `peft==0.15.2` for LoRA |

Requirements: an NVIDIA driver with CUDA 12.8+ support (R570+), Docker with the
NVIDIA Container Toolkit, and about 30 GB of disk for the model.

## 1. Build and start

```bash
# Download the weights once (29 GB).
hf download sensenova/SenseNova-Vision-7B-MoT --local-dir /path/to/SenseNova-Vision-7B-MoT

cp docker/.env.example docker/.env      # set MODEL_DIR, HOST_UID/HOST_GID
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up -d
docker exec sensenova-vision python /opt/sensenova/check_env.py
```

`check_env.py` should report `arch=sm_120` and run flash-attn, a bitsandbytes
NF4 matmul and flex-attention forward/backward on the GPU.

The repository is bind-mounted at `/workspace/SenseNova-Vision`, so host edits
apply without rebuilding. The model directory (`MODEL_DIR`) is mounted at
`/models/SenseNova-Vision-7B-MoT` (`MODEL_PATH`), the host datasets root
(`DATASETS_DIR`) read-only at `/datasets`, and the container runs as
`HOST_UID:HOST_GID` so outputs stay owned by you.

## 2. Inference on a 32 GB GPU

The BF16 checkpoint alone is 29.2 GB, so it cannot sit entirely on a 32 GB card
next to activations. The container picks one of two modes through environment
variables, which every entrypoint (`example`, `inference`, `interactive`, `demo`,
benchmark scripts) honours:

| Mode | Settings | Dense task (binary seg, 50 steps) | Peak GPU memory |
| --- | --- | --- | --- |
| BF16 + partial CPU offload (default) | `SENSENOVA_DTYPE=bf16 SENSENOVA_MAX_MEM_PER_GPU=25GiB` | ~25 s | ~28.5 GB |
| BF16, tighter offload | `SENSENOVA_MAX_MEM_PER_GPU=24GiB` | ~31 s | ~27.5 GB |
| BF16, no offload | `SENSENOVA_MAX_MEM_PER_GPU=30GiB` | ~13 s (measured at 27 GiB; ~4.5 s per 20-step mask at 512 px) | ~32.0 GB |
| NF4 quantized | `SENSENOVA_DTYPE=nf4` | ~15 s | ~13 GB |
| BF16 on 48 GB+ GPUs | `SENSENOVA_MAX_MEM_PER_GPU=80GiB` | no offload | — |

Peak figures include ~1.4 GB used by the desktop on the test machine. At 30 GiB
all 27.2 GiB of BF16 weights sit on the GPU (no offload) and dense tasks run
about 2x faster, but only ~0.6 GB stays free: fine for 512 px inputs, while
larger inputs or long panoptic/recon3d requests can run out of memory. Drop back
to 25 GiB if that happens.

Showcase: run all built-in examples (general understanding, binary/GCG/panoptic/
interactive segmentation, depth, normals, detection, multi-view 3D
reconstruction, camera pose):

```bash
docker exec sensenova-vision bash scripts/run_sensenova_vision.sh example
# -> examples/output/example_visualize/   (9m11s incl. model load on an RTX 5090, peak 28.7 GB)
```

Single requests, NF4 for speed:

```bash
docker exec -e SENSENOVA_DTYPE=nf4 sensenova-vision \
  bash scripts/run_sensenova_vision.sh inference bbox_detection \
  "bird, boat, person, cell phone, backpack, handbag" examples/images/5.jpg
```

Web demo (port `DEMO_PORT`, default 9001):

```bash
docker exec -it sensenova-vision bash scripts/run_sensenova_vision.sh demo
```

## 3. Single-GPU LoRA / QLoRA fine-tuning

`train/pretrain_unified_navit.py` is a full-parameter FSDP trainer sized for
16+ x 80 GB GPUs. `train/finetune_lora.py` reuses its data pipeline
(`PackedDataset`, the same JSONL formats and dataset groups) and losses
(cross-entropy on text + flow-matching MSE on generated images), but:

- stores the LLM decoder linears (~13B of ~14B parameters) in NF4
  (`--quant nf4`; `--quant bf16` needs 40 GB+),
- injects LoRA into the understanding expert, the generation expert, or both
  (`--lora_experts und|gen|both`; rank 16 on both = 80.7M trainable parameters),
- activation-checkpoints every decoder layer,
- saves `adapter_model.safetensors` + `adapter_config.json`.

Dataset YAMLs use the same keys as `data/configs/cv_unify/*.yaml`. A group can
declare its JSONLs inline under `datasets:` instead of editing
`data/dataset_info.py`; see `data/configs/lora_demo/glorp_lora.yaml`.

### Demo: teach the model a new concept

`tools/make_lora_demo_data.py` renders a synthetic dataset for a made-up object,
a *glorp* (teal hexagon with a white centre dot), surrounded by distractors that
share its colour or shape. It writes one bbox-detection JSONL (text target) and
one binary-segmentation JSONL (image target) in the repository's training
formats, plus a held-out test split.

```bash
# Fine-tune (generates datas/lora_demo on first use; 300 steps by default)
docker exec sensenova-vision bash scripts/finetune_lora.sh --output_dir results/lora_demo

# Fold the adapter into a regular model directory
docker exec sensenova-vision python tools/merge_lora.py \
  --adapter results/lora_demo/final --output_dir results/lora_demo/merged

# Compare base vs fine-tuned on the held-out split
docker exec sensenova-vision python tools/eval_lora_demo.py \
  --model_path /models/SenseNova-Vision-7B-MoT --output_dir results/lora_demo/eval_base
docker exec sensenova-vision python tools/eval_lora_demo.py \
  --model_path results/lora_demo/merged --output_dir results/lora_demo/eval_lora
```

Measured on an RTX 5090 with the defaults (NF4 base, rank-16 LoRA on both
experts, 300 steps, ~14 min, peak 15.6 GiB), evaluated in BF16 on the 24
held-out images:

| Model | Box F1@0.5 | Box mean best IoU | Mask mIoU |
| --- | --- | --- | --- |
| Base | 0.08 | 0.08 | 0.11 (outputs an all-white mask) |
| + LoRA (300 steps) | **0.95** | **0.95** | 0.08 |

The text task (detection) converges within ~50 steps. The image task does not
at this budget: the model produces near-perfect masks (IoU 0.99) on 2 of 24
images and an all-black mask on the rest, the usual early-training collapse of
flow-matching targets that are mostly background. 300 steps covers fewer than
one epoch of the segmentation data; give image-output tasks more steps (and/or
a larger share of the mix via `weight:`) when adapting dense outputs.

The merged directory is a drop-in `MODEL_PATH` for every inference entrypoint:

```bash
docker exec -e MODEL_PATH=/workspace/SenseNova-Vision/results/lora_demo/merged sensenova-vision \
  bash scripts/run_sensenova_vision.sh inference binary_seg "glorp" datas/lora_demo/test/images/00000.png
```

For face parsing on CelebAMask-HQ with IoU evaluation, see
[`CELEBAMASK_HQ.md`](CELEBAMASK_HQ.md).

For your own data, write JSONLs in the formats described in
[`train_data_prepare.md`](train_data_prepare.md), point a YAML at them and run
`bash scripts/finetune_lora.sh path/to/config.yaml --total_steps ... --output_dir ...`.
