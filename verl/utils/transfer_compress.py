# verl/utils/transfer_compress.py
"""Lightweight tensor compression for inter-worker data transfer.

Supports three modes controlled by VERL_TRANSFER_COMPRESS:
  bf16  – cast all float32 batch tensors to bfloat16 before ray.put();
          cast back to float32 on the receiver side.  Safest 16-bit option:
          same exponent range as float32, so no overflow risk.  Recommended.
  fp16  – cast all float32 batch tensors to float16 before ray.put();
          cast back to float32 on the receiver side.  Same byte savings as
          bf16 but with narrower exponent range (risk of overflow on large values).
  int8  – per-tensor symmetric quantization with a float32 scale stored
          alongside each tensor.  Higher compression, needs validation.
  (unset / empty) – no compression (default).

Usage:
    export VERL_TRANSFER_COMPRESS=bf16
"""

import os
from typing import Dict

import torch


def get_compress_mode() -> str:
    return os.environ.get("VERL_TRANSFER_COMPRESS", "").lower()


def compress_batch(
    batch: Dict[str, torch.Tensor],
) -> tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    """Return (compressed_batch, metadata) where metadata is needed to decompress."""
    mode = get_compress_mode()
    if not mode:
        return batch, {}

    compressed = {}
    meta = {}

    for key, tensor in batch.items():
        if tensor.dtype != torch.float32:
            compressed[key] = tensor
            continue

        if mode == "bf16":
            compressed[key] = tensor.to(torch.bfloat16)
            meta[key] = {"orig_dtype": torch.float32, "mode": "bf16"}

        elif mode == "fp16":
            compressed[key] = tensor.to(torch.float16)
            meta[key] = {"orig_dtype": torch.float32, "mode": "fp16"}

        elif mode == "int8":
            scale = tensor.abs().max().clamp(min=1e-8) / 127.0
            q = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
            compressed[key] = q
            meta[key] = {"orig_dtype": torch.float32, "mode": "int8", "scale": scale}

        else:
            compressed[key] = tensor

    return compressed, meta


def decompress_batch(
    batch: Dict[str, torch.Tensor],
    meta: Dict[str, object],
) -> Dict[str, torch.Tensor]:
    """Reverse the compression applied by compress_batch."""
    if not meta:
        return batch

    out = {}
    for key, tensor in batch.items():
        if key not in meta:
            out[key] = tensor
            continue

        info = meta[key]
        if info["mode"] in ("bf16", "fp16"):
            out[key] = tensor.to(info["orig_dtype"])
        elif info["mode"] == "int8":
            out[key] = tensor.float() * info["scale"].to(tensor.device)
        else:
            out[key] = tensor

    return out


def compression_ratio(original: Dict[str, torch.Tensor],
                      compressed: Dict[str, torch.Tensor]) -> float:
    """Compute bytes(compressed) / bytes(original)."""
    def nbytes(d):
        return sum(t.element_size() * t.numel() for t in d.values())
    orig_b = nbytes(original)
    if orig_b == 0:
        return 1.0
    return nbytes(compressed) / orig_b


def probe_compress_event(context: str, orig_bytes: int, comp_bytes: int) -> None:
    """Emit a [transfer_probe] compress_stats event if probing is enabled."""
    probe_log = os.environ.get("VERL_TRANSFER_PROBE_LOG")
    if not probe_log:
        return
    import json, time
    event = {
        "event": "compress_stats",
        "context": context,
        "mode": get_compress_mode(),
        "orig_bytes": orig_bytes,
        "comp_bytes": comp_bytes,
        "ratio": comp_bytes / orig_bytes if orig_bytes else 1.0,
        "ts_ms": time.monotonic() * 1000,
    }
    line = f"[transfer_probe] {json.dumps(event)}\n"
    try:
        with open(probe_log, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass
