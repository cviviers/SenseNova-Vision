# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
"""Smoke check for the Blackwell image: imports plus one real kernel per backend."""

import torch


def main():
    import bitsandbytes as bnb
    import decord
    import fastevaluate  # noqa: F401
    import flash_attn
    import panopticapi  # noqa: F401
    import peft
    from flash_attn import flash_attn_varlen_func

    assert torch.cuda.is_available(), "CUDA is not visible inside the container"
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    print(f"gpu={torch.cuda.get_device_name()} arch={arch}")
    print(f"torch={torch.__version__} cuda={torch.version.cuda} arch_list={torch.cuda.get_arch_list()}")
    print(
        f"flash_attn={flash_attn.__version__} bitsandbytes={bnb.__version__} "
        f"peft={peft.__version__} decord={decord.__version__}"
    )
    assert arch in torch.cuda.get_arch_list(), f"torch has no kernels for {arch}"

    q = torch.randn(256, 28, 128, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, 128, 256], device="cuda", dtype=torch.int32)
    out = flash_attn_varlen_func(q, q, q, cu, cu, 128, 128, causal=True)
    print(f"flash_attn_varlen_func ok: {tuple(out.shape)}")

    lin = bnb.nn.Linear4bit(512, 512, compute_dtype=torch.bfloat16, quant_type="nf4").cuda()
    print(f"bnb nf4 matmul ok: {tuple(lin(torch.randn(4, 512, device='cuda', dtype=torch.bfloat16)).shape)}")

    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    mask = create_block_mask(causal, B=1, H=4, Q_LEN=512, KV_LEN=512, device="cuda")
    qf = torch.randn(1, 4, 512, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    torch.compile(flex_attention)(qf, qf, qf, block_mask=mask).sum().backward()
    print("flex_attention fwd/bwd ok")
    print("environment OK")


if __name__ == "__main__":
    main()
