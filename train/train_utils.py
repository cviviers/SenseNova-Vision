# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import logging
import os


def create_logger(logging_dir, rank, filename="log"):
    """
    Create a logger that writes to a log file and stdout.
    """
    if rank == 0 and logging_dir is not None:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(), 
                logging.FileHandler(f"{logging_dir}/{filename}.txt")
            ]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def get_latest_ckpt(checkpoint_dir):
    step_dirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    if len(step_dirs) == 0:
        return None
    step_dirs = sorted(step_dirs, key=lambda x: int(x))
    latest_step_dir = os.path.join(checkpoint_dir, step_dirs[-1])
    return latest_step_dir


def per_group_losses(loss_dict, data, groups):
    """Sum and token count of the image (mse) and text (ce) losses per sample group (e.g. a dataset's mode).

    Samples sit in the packed sequence in the same order as `groups`; each loss row belongs to the sample
    whose span (from sample_lens) contains its sequence position. Returns
    {group: {"mse": (sum, count), "ce": (sum, count), "samples": n}}.
    """
    import torch

    with torch.no_grad():
        ends = torch.cumsum(torch.as_tensor(data["sample_lens"], device="cpu"), 0)
        out = {}
        for key, index_key, reduce in (("mse", "mse_loss_indexes", lambda x: x.float().mean(-1)),
                                       ("ce", "ce_loss_indexes", lambda x: x.float())):
            values = loss_dict.get(key)
            if values is None or values.numel() == 0 or data.get(index_key) is None:
                continue
            sample_ids = torch.searchsorted(ends, data[index_key].cpu(), right=True)
            per_token = reduce(values.detach()).cpu()
            for sid in sample_ids.unique().tolist():
                sel = sample_ids == sid
                total, count = out.setdefault(groups[sid], {}).get(key, (0.0, 0))
                out[groups[sid]][key] = (total + float(per_token[sel].sum()), count + int(sel.sum()))
        for group in out:
            out[group]["samples"] = groups.count(group)
        return out


def accumulate_groups(window, batch_groups):
    """Add per_group_losses() output into a running {group: {"mse": [sum, n], "ce": [sum, n], "samples": n}}."""
    for group, vals in batch_groups.items():
        acc = window.setdefault(group, {"mse": [0.0, 0], "ce": [0.0, 0], "samples": 0})
        acc["samples"] += vals.get("samples", 0)
        for key in ("mse", "ce"):
            if key in vals:
                acc[key][0] += vals[key][0]
                acc[key][1] += vals[key][1]
    return window


def summarize_groups(window):
    """{group: {"mse": mean, "ce": mean, "samples": n}} from an accumulated window."""
    return {g: {k: acc[k][0] / acc[k][1] for k in ("mse", "ce") if acc[k][1]} | {"samples": acc["samples"]}
            for g, acc in sorted(window.items())}
