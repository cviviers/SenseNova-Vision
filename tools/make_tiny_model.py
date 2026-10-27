# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Build a tiny SenseNova-Vision-shaped model dir (2 decoder layers, hidden 256, ~120M parameters) to smoke-test
the training recipes on any GPU: same configs and code paths as the release, real tokenizer and VAE, random
weights saved as ema.safetensors. Outputs are meaningless; only the mechanics are tested.

    python tools/make_tiny_model.py results/tiny_model
    SENSENOVA_MODEL_DIR=results/tiny_model SENSENOVA_RUN_DIR=results/tiny_full \
        bash scripts/finetune_full.sh data/configs/celebamask_hq/celebamask_img2palette_seg2img.yaml --total_steps 12
"""
import json, os, shutil, sys
import torch
from safetensors.torch import save_file
from modeling.bagel import Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
from modeling.autoencoder import load_ae

base, out = os.environ.get("MODEL_PATH", "/models/SenseNova-Vision-7B-MoT"), sys.argv[1]
os.makedirs(out, exist_ok=True)
for name in os.listdir(base):
    if name not in ("ema.safetensors", "llm_config.json", "vit_config.json") and os.path.isfile(f"{base}/{name}"):
        shutil.copy2(f"{base}/{name}", f"{out}/{name}")
llm = json.load(open(f"{base}/llm_config.json"))
llm.update(hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
           max_window_layers=2)
json.dump(llm, open(f"{out}/llm_config.json", "w"), indent=2)
vit = json.load(open(f"{base}/vit_config.json"))
vit.update(num_hidden_layers=3)
json.dump(vit, open(f"{out}/vit_config.json", "w"), indent=2)

# Build exactly as the inference loader does (inference/sensenova_vision.py), with real random init.
lc = Qwen2Config.from_json_file(f"{out}/llm_config.json"); lc.qk_norm = True; lc.tie_word_embeddings = False
lc.layer_module = "Qwen2MoTDecoderLayer"
vc = SiglipVisionConfig.from_json_file(f"{out}/vit_config.json"); vc.rope = False; vc.num_hidden_layers -= 1
_, vae_cfg = load_ae(local_path=f"{out}/ae.safetensors")
cfg = BagelConfig(visual_gen=True, visual_und=True, llm_config=lc, vit_config=vc, vae_config=vae_cfg,
                  latent_patch_size=2, max_latent_size=64, vit_max_num_patch_per_side=70, connector_act="gelu_pytorch_tanh")
model = Bagel(Qwen2ForCausalLM(lc), SiglipVisionModel(vc), cfg)
model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vc)
save_file({k: v.to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()}, f"{out}/ema.safetensors")
print("tiny model:", sum(p.numel() for p in model.parameters()) / 1e6, "M params ->", out)
