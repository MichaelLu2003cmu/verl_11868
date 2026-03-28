import itertools
import json
import os
import threading
import time
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from tensordict import TensorDict
except Exception:  # pragma: no cover
    TensorDict = None

_ENABLED = os.getenv("VERL_TRANSFER_PROBE", "0").lower() in {"1", "true", "yes", "on"}
_LOG_PATH = os.getenv("VERL_TRANSFER_PROBE_LOG", "").strip()
_LOCK = threading.Lock()
_SEQ = itertools.count()


def is_enabled() -> bool:
    return _ENABLED


def now_ns() -> int:
    return time.perf_counter_ns()


def ns_to_ms(delta_ns: int) -> float:
    return float(delta_ns) / 1e6


def _write_line(line: str) -> None:
    print(line, flush=True)
    if _LOG_PATH:
        with _LOCK:
            with open(_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")


def emit_event(event: str, **fields: Any) -> None:
    if not _ENABLED:
        return
    payload = {
        "event": event,
        "seq": next(_SEQ),
        "ts_unix_s": time.time(),
    }
    payload.update(fields)
    _write_line(f"[transfer_probe] {json.dumps(payload, sort_keys=True, ensure_ascii=True)}")


def emit_cpu_overhead(location: str, elapsed_ms: float, **fields: Any) -> None:
    emit_event("cpu_overhead", location=location, elapsed_ms=elapsed_ms, **fields)


def _is_dataproto_like(obj: Any) -> bool:
    return hasattr(obj, "batch") and hasattr(obj, "non_tensor_batch")


def estimate_nbytes(obj: Any) -> int:
    if obj is None:
        return 0
    if torch is not None and isinstance(obj, torch.Tensor):
        return int(obj.numel() * obj.element_size())
    if np is not None and isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if TensorDict is not None and isinstance(obj, TensorDict):
        return sum(estimate_nbytes(v) for _, v in obj.items())
    if _is_dataproto_like(obj):
        total = estimate_nbytes(getattr(obj, "batch", None))
        non_tensor_batch = getattr(obj, "non_tensor_batch", {}) or {}
        if isinstance(non_tensor_batch, dict):
            total += sum(estimate_nbytes(v) for v in non_tensor_batch.values())
        return total
    if isinstance(obj, dict):
        return sum(estimate_nbytes(v) for v in obj.values())
    if isinstance(obj, list | tuple):
        return sum(estimate_nbytes(v) for v in obj)
    return 0
