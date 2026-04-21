# Implementation Guide: Proposed Approach in verl

**Document Type:** Technical Implementation Reference  
**Scope:** Sections 4.1–4.3 of the project proposal — Local-Batch Pull, Async Pipelining, Lightweight Compression  
**Repository:** `verl_11868` (fork of volcengine/verl)  

---

## Table of Contents

1. [Codebase Orientation](#1-codebase-orientation)
2. [4.1 Local-Batch Pull Transfer Protocol](#2-41-local-batch-pull-transfer-protocol)
   - [Proposal vs Reality](#21-proposal-vs-reality)
   - [Data Structure Design](#22-data-structure-design)
   - [Sender Side: build_handle()](#23-sender-side-build_handle)
   - [Receiver Side: materialize()](#24-receiver-side-materialize)
   - [Wiring the Decorator](#25-wiring-the-decorator)
   - [How to Enable](#26-how-to-enable)
3. [4.2 Asynchronous Pipelining and Overlap](#3-42-asynchronous-pipelining-and-overlap)
   - [Proposal vs Reality](#31-proposal-vs-reality)
   - [RewardScoreFuture](#32-rewardscorефuture)
   - [Non-Blocking Worker Dispatch](#33-non-blocking-worker-dispatch)
   - [Fit Loop Restructure](#34-fit-loop-restructure)
   - [Overlap Measurement Infrastructure](#35-overlap-measurement-infrastructure)
   - [How to Enable](#36-how-to-enable)
4. [4.3 Lightweight Tensor Compression](#4-43-lightweight-tensor-compression)
   - [Proposal vs Reality](#41-proposal-vs-reality)
   - [Compression Module](#42-compression-module)
   - [Sender Integration](#43-sender-integration)
   - [Receiver Integration](#44-receiver-integration)
   - [Probe Instrumentation](#45-probe-instrumentation)
   - [How to Enable](#46-how-to-enable)
5. [End-to-End Integration Checklist](#5-end-to-end-integration-checklist)
6. [Key Design Decisions and Trade-offs](#6-key-design-decisions-and-trade-offs)

---

## 1. Codebase Orientation

Before reading the implementation, it helps to understand how verl routes data between the
trainer and workers.

### Key Modules

| Module | Role |
|---|---|
| `verl/trainer/ppo/ray_trainer.py` | Central PPO/GRPO training loop (`RayPPOTrainer.fit()`) |
| `verl/workers/fsdp_workers.py` | FSDP-backed actor/critic/ref worker implementations |
| `verl/single_controller/base/decorator.py` | Dispatch decorators: transform Python function calls into Ray remote calls with data routing |
| `verl/protocol.py` | `DataProto` — the universal tensor container; `DataProtoFuture` — async variant |
| `verl/experimental/reward_loop/reward_loop.py` | Reward worker manager and loop |

### The Default Dispatch Path

When the trainer calls `self.actor_rollout_wg.compute_log_prob(batch)`, the decorator
on `compute_log_prob` intercepts it:

```
trainer calls compute_log_prob(batch: DataProto)
    │
    └─► decorator intercepts
            ├─ serialize batch
            ├─ ray.put(batch)          ← full batch into Ray object store
            ├─ broadcast ref to all workers
            └─► each worker calls compute_log_prob(ray.get(ref))
                    └─ worker slices its DP shard locally
```

The inefficiency: every worker receives the **full** batch over Ray's shared-memory
object store, even though each worker only needs `1/dp_size` of the data.

---

## 2. 4.1 Local-Batch Pull Transfer Protocol

### 2.1 Proposal vs Reality

**Proposal said:**
> "The sender will not send the complete tensor but only provide shardable tensor
> metadata (shape, dtype, global batch partitioning).  Every receiver will compute
> its own local slice from its DP rank and only pulls that small slice."

**What we implemented:**
The spirit is preserved but the mechanism differs slightly from "metadata-only".
Instead of sending only metadata and having workers reconstruct tensors, we:

1. **Sender** pre-slices the batch into `dp_size` shards and places each shard into the
   Ray object store independently via `ray.put(shard)`.
2. **Receiver** is given a handle containing the object references and fetches only its
   own shard via `ray.get(shard_refs[my_dp_rank])`.

This approach is functionally equivalent to the proposal (each worker transfers
`1/dp_size` of the data) but avoids the complexity of remote tensor reconstruction,
which would require workers to have the full dtype/shape metadata and a shared
memory region to write into — a much more invasive change to verl's actor model.

The key trade-off accepted: a fixed `ray.put()` overhead per dispatch call (~7.5 ms
measured), which breaks even with the broadcast approach at ~13 samples and yields
increasing savings above that.

---

### 2.2 Data Structure Design

**File:** `verl/protocol.py`

Three new classes support the pull protocol:

#### `DataProtoSliceMeta`

```python
@dataclass
class DataProtoSliceMeta:
    total_len: int    # original un-padded batch length
    dp_size:   int    # number of data-parallel ranks
    pad_size:  int    # padding added to make total_len divisible by dp_size

    def local_range(self, dp_rank: int) -> tuple[int, int]:
        """Return (start, end) indices for this rank's shard."""
        chunk = (self.total_len + self.pad_size) // self.dp_size
        start = dp_rank * chunk
        return start, min(start + chunk, self.total_len)
```

This encodes the global-to-local mapping so any rank can compute its slice
boundaries without additional communication.

#### `DataProtoPullHandle`

```python
@dataclass
class DataProtoPullHandle:
    shard_refs:    list[ray.ObjectRef]  # one ref per dp_rank
    meta:          DataProtoSliceMeta
    compress_meta: dict = field(default_factory=dict)  # populated by compression (§4.3)

    def materialize(self, dp_rank: int) -> DataProto:
        shard = ray.get(self.shard_refs[dp_rank])
        if self.compress_meta:
            # decompress before returning (§4.3)
            shard.batch = TensorDict(
                decompress_batch(dict(shard.batch), self.compress_meta),
                batch_size=shard.batch.batch_size,
            )
        return shard
```

The handle is serialized by Ray (it contains only object refs, not the tensors
themselves) and passed to each worker as a lightweight descriptor.

#### `DataProtoSelectiveView` (multi-source futures)

For the `DataProtoFuture` pathway (used when workers have already produced partial
results as futures), a `DataProtoSelectiveView` allows a worker to resolve only the
futures whose data overlaps with its local DP range, avoiding `ray.get()` on
irrelevant futures.

---

### 2.3 Sender Side: `build_handle()`

**File:** `verl/single_controller/base/decorator.py`

```python
def dispatch_nd_compute_dataproto_pull(dp_rank_mapping, dp_size, worker_group, *args, **kwargs):

    def build_handle(obj: DataProto) -> DataProtoPullHandle:
        total_len = len(obj)
        pad_size  = 0

        # 1. Pad to dp_size if needed
        if obj.is_padding_enabled() and total_len % dp_size != 0:
            pad_size = dp_size - (total_len % dp_size)
            obj = obj[:]
            obj.padding(padding_size=pad_size)

        # 2. Optional compression (§4.3) — compress BEFORE sharding
        compress_meta = {}
        if get_compress_mode():
            orig_batch         = dict(obj.batch)
            compressed_tensors, compress_meta = compress_batch(orig_batch)
            probe_compress_event(...)        # instrumentation
            obj = dataclasses.replace(
                obj,
                batch=TensorDict(compressed_tensors, batch_size=obj.batch.batch_size)
            )

        # 3. Shard and store each shard independently
        shard_list = obj.chunk(chunks=dp_size)
        shard_refs = [ray.put(shard) for shard in shard_list]

        return DataProtoPullHandle(
            shard_refs=shard_refs,
            meta=DataProtoSliceMeta(total_len=total_len, dp_size=dp_size, pad_size=pad_size),
            compress_meta=compress_meta,
        )

    # Apply build_handle to all positional and keyword DataProto arguments
    args   = [build_handle(arg) for arg in args]
    kwargs = {k: build_handle(v) for k, v in kwargs.items()}

    # Distribute the per-rank shard handle to each worker
    for i in range(worker_group.world_size):
        local_dp_rank = dp_rank_mapping[i]
        # worker i receives handle.materialize(local_dp_rank)
        ...
```

**Critical design choice:** Compression is applied *before* sharding.  This means the
compression overhead is paid once on the trainer CPU, not once per rank.

---

### 2.4 Receiver Side: `materialize()`

**File:** `verl/protocol.py` — `DataProtoPullHandle.materialize()`

```python
def materialize(self, dp_rank: int) -> DataProto:
    # Fetch only this rank's shard from the object store
    shard: DataProto = ray.get(self.shard_refs[dp_rank])

    # Decompress if compression was applied at the sender
    if self.compress_meta:
        from tensordict import TensorDict
        from verl.utils.transfer_compress import decompress_batch
        decompressed = decompress_batch(dict(shard.batch), self.compress_meta)
        shard.batch  = TensorDict(
            source=decompressed,
            batch_size=shard.batch.batch_size,
        )
    return shard
```

Decompression occurs transparently inside `materialize()`.  Worker code is completely
unaware that compression happened — it receives a standard `DataProto` with float32
tensors, identical to the uncompressed case.

---

### 2.5 Wiring the Decorator

**File:** `verl/single_controller/base/decorator.py`

The top-level entry point that verl's worker group dispatch machinery calls:

```python
def dispatch_lazy_compute_data_proto_pull(mesh_name, worker_group, *args, **kwargs):
    dp_rank_mapping = worker_group._dispatch_info[mesh_name]
    dp_size         = max(dp_rank_mapping) + 1
    # Note: mesh_name is NOT passed to dispatch_nd_compute_dataproto_pull
    # (this was the bug — see §7.1 of what_we_have_done.md)
    return dispatch_nd_compute_dataproto_pull(
        dp_rank_mapping, dp_size, worker_group, *args, **kwargs
    )
```

**File:** `verl/workers/fsdp_workers.py`

The pull decorator is applied to `compute_log_prob`:

```python
@register(dispatch_mode=dispatch_lazy_compute_data_proto_pull)
def compute_log_prob(self, data: DataProto) -> DataProto:
    ...
```

---

### 2.6 How to Enable

```bash
# In your training command:
trainer.use_legacy_worker_impl=enable

# Pull is then active for compute_log_prob.
# No other changes needed.
```

---

## 3. 4.2 Asynchronous Pipelining and Overlap

### 3.1 Proposal vs Reality

**Proposal said:**
> "Non-blocking transfer dispatch and futures/callbacks to trigger downstream compute
> as soon as required slices are available."

**What we implemented:**
The "non-blocking dispatch" maps directly to Ray's `blocking=False` option on remote
calls, which returns a `ray.ObjectRef` (future) immediately instead of waiting for
the result.  The "callbacks" aspect is simplified: rather than callback-based triggers,
we use an explicit **join point** where the trainer calls `.get()` on the futures only
when the results are actually needed for the next computation stage.

The window of overlap is: time between dispatch and join, during which the trainer
executes CPU-bound post-processing (advantage estimation, data shuffling for the actor
mini-batch update).

---

### 3.2 `RewardScoreFuture`

**File:** `verl/experimental/reward_loop/reward_loop.py`

The reward worker in verl returns scores synchronously.  To make it async, we
introduce a thin future wrapper:

```python
class RewardScoreFuture:
    """Wraps a Ray ObjectRef for deferred reward score resolution."""

    def __init__(self, ref: ray.ObjectRef):
        self._ref = ref

    def result(self) -> DataProto:
        """Block until the reward scores are ready and return them."""
        return ray.get(self._ref)
```

A new method on `RewardLoopManager`:

```python
def compute_rm_score_async(self, data: DataProto) -> RewardScoreFuture:
    """Dispatch reward scoring non-blocking; return a future."""
    ref = self._worker.compute_rm_score.remote(data)
    return RewardScoreFuture(ref)
```

---

### 3.3 Non-Blocking Worker Dispatch

**File:** `verl/trainer/ppo/ray_trainer.py`

verl's worker group calls support `blocking=False`:

```python
# Blocking (default — waits for result):
output = self.ref_policy_wg.compute_ref_log_prob(batch)   # blocks until done

# Non-blocking — returns a DataProtoFuture immediately:
future = self.ref_policy_wg.compute_ref_log_prob(batch, blocking=False)
output = future.get()    # blocks only when .get() is called
```

When `trainer.use_legacy_worker_impl=enable`, the worker group's dispatch mechanism
routes through the legacy path where `blocking=False` correctly returns a
`DataProtoFuture` wrapping a list of `ray.ObjectRef`.  Without this flag, some worker
implementations eagerly resolve the future internally, making the non-blocking
dispatch ineffective.

---

### 3.4 Fit Loop Restructure

**File:** `verl/trainer/ppo/ray_trainer.py` — `RayPPOTrainer.fit()`

The critical change is moving all prep-stage dispatches to the **start** of the
iteration, before the CPU-bound work, and joining them only when needed:

```python
def fit(self):
    for batch in dataloader:

        # ── Step 1: Rollout ──────────────────────────────────────────────────
        batch = self._generate_sequences(batch)       # vLLM inference (GPU)

        # ── Step 2: Dispatch prep-stage NON-BLOCKING ────────────────────────
        # These GPU forwards now run in parallel with Step 3 below.
        _ref_future    = self._compute_ref_log_prob(batch)      # returns DataProtoFuture
        _values_future = self._compute_values(batch)            # returns DataProtoFuture
        _reward_future = self._compute_reward_colocate_async(batch)  # returns RewardScoreFuture

        probe_event("critical_path_stage", stage="ref",    dispatched_at_ms=now())
        probe_event("critical_path_stage", stage="values", dispatched_at_ms=now())

        # ── Step 3: CPU post-processing (OVERLAPS with Step 2) ───────────────
        batch = self._postprocess_rollout(batch)      # tokenize, filter, etc.
        batch = self._compute_advantages(batch)       # GAE/GRPO advantage est.

        # ── Step 4: Join prep-stage futures ──────────────────────────────────
        # By this point, the ref/critic GPUs have been running for ~(Step 3 time).
        # Most or all of their compute is already done; wait_ms ≈ 0.
        ref_output    = _ref_future.get()
        probe_event("critical_path_stage", stage="ref",    joined_at_ms=now(),
                    wait_ms=measured_wait)

        values_output = _values_future.get()
        probe_event("critical_path_stage", stage="values", joined_at_ms=now(),
                    wait_ms=measured_wait)

        reward_output = _reward_future.result()

        # ── Step 5: Actor update ──────────────────────────────────────────────
        batch.update(ref_output, values_output, reward_output)
        self._update_actor(batch)
        self._update_critic(batch)
```

---

### 3.5 Overlap Measurement Infrastructure

**File:** `scripts/summarize_transfer_probe.py`

The `critical_path_stage` probe events are parsed to compute:

```
in_flight_ms  = joined_at_ms − dispatched_at_ms
              = total wall-clock time the future was outstanding

wait_ms       = time the trainer blocked at .get() waiting for the result
              (measured by timing the .get() call itself)

hidden_ms     = in_flight_ms − wait_ms
              = time the GPU was computing while the trainer was doing other work

hidden_frac   = hidden_ms / in_flight_ms
              = fraction of GPU compute time that was "free" (overlapped)
```

A `hidden_frac` of 1.0 means perfect overlap (trainer never had to wait).
A `hidden_frac` of 0.0 means no overlap (trainer always blocked immediately).

**Measured results:**

| Stage | hidden_frac | Meaning |
|---|---|---|
| ref log-prob | **0.285** | 28.5% of ref GPU time was hidden under CPU post-processing |
| values (critic) | **0.590** | 59.0% of critic GPU time was hidden |

---

### 3.6 How to Enable

```bash
# Required for non-blocking futures to work correctly:
trainer.use_legacy_worker_impl=enable

# Async pipelining is always active once use_legacy_worker_impl is enabled.
# Use the probe to measure overlap:
VERL_TRANSFER_PROBE=1
VERL_TRANSFER_PROBE_LOG=/path/to/probe.jsonl

# Then analyze:
python scripts/summarize_transfer_probe.py --log /path/to/probe.jsonl
# Look for the "Critical-Path Stage Overlap" table.
```

---

## 4. 4.3 Lightweight Tensor Compression

### 4.1 Proposal vs Reality

**Proposal said:**
> "Optional payload reduction for tensors less sensitive to precision (advantages,
> value/logprob).  FP16 as low-risk baseline; optionally INT8 with per-tensor scaling."

**What we implemented:**
Exactly as proposed.  The compression is applied **sender-side** inside `build_handle()`
before `ray.put()`, and decompression happens **receiver-side** inside
`DataProtoPullHandle.materialize()` before the tensor reaches any training code.

**Scope of "selected tensors":** Rather than maintaining a hard-coded allowlist of
tensor names (which would break if the batch schema changes), we compress **all
float32 tensors** and skip all other dtypes.  Non-float32 tensors (int64 `input_ids`,
`attention_mask`, etc.) are left unchanged.  This is equivalent to targeting
advantages, values, and log-probs, which are the only float32 tensors in a typical
GRPO/PPO batch.

---

### 4.2 Compression Module

**File:** `verl/utils/transfer_compress.py`

#### `compress_batch()`

```python
def compress_batch(
    batch: Dict[str, torch.Tensor],
) -> tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    """
    Compress all float32 tensors in `batch`.
    Returns (compressed_batch, meta) where meta is needed for decompression.
    """
    mode = get_compress_mode()   # reads VERL_TRANSFER_COMPRESS env var
    compressed, meta = {}, {}

    for key, tensor in batch.items():
        if tensor.dtype != torch.float32:
            compressed[key] = tensor        # int64, int32, etc. — pass through
            continue

        if mode == "fp16":
            compressed[key] = tensor.to(torch.float16)
            meta[key] = {"orig_dtype": torch.float32, "mode": "fp16"}

        elif mode == "int8":
            # Per-tensor symmetric quantization
            scale = tensor.abs().max().clamp(min=1e-8) / 127.0
            q     = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
            compressed[key] = q
            meta[key] = {"orig_dtype": torch.float32, "mode": "int8", "scale": scale}

    return compressed, meta
```

#### `decompress_batch()`

```python
def decompress_batch(
    batch: Dict[str, torch.Tensor],
    meta:  Dict[str, object],
) -> Dict[str, torch.Tensor]:
    out = {}
    for key, tensor in batch.items():
        if key not in meta:
            out[key] = tensor
            continue
        info = meta[key]
        if info["mode"] == "fp16":
            out[key] = tensor.to(info["orig_dtype"])          # fp16 → fp32
        elif info["mode"] == "int8":
            out[key] = tensor.float() * info["scale"]         # int8 × scale → fp32
    return out
```

**INT8 quantization properties:**
- Symmetric (zero-point = 0)
- Per-tensor scaling (one scale scalar per tensor)
- Range: [−127, 127] mapped from [−max_abs, +max_abs]
- Max quantization error: `max_abs / 127 ≈ 0.79%` of the tensor range

---

### 4.3 Sender Integration

**File:** `verl/single_controller/base/decorator.py` — `build_handle()`

The compression block within `build_handle()` (see §2.3 above):

```python
from verl.utils.transfer_compress import get_compress_mode, compress_batch, probe_compress_event

if get_compress_mode():
    import dataclasses, torch as _torch
    from tensordict import TensorDict as _TensorDict

    orig_batch = dict(obj.batch)
    compressed_tensors, compress_meta = compress_batch(orig_batch)

    # Emit a probe event recording bytes before/after
    probe_compress_event(
        context="pull_build_handle",
        orig_bytes=sum(t.element_size() * t.numel()
                       for t in orig_batch.values()
                       if isinstance(t, _torch.Tensor)),
        comp_bytes=sum(t.element_size() * t.numel()
                       for t in compressed_tensors.values()
                       if isinstance(t, _torch.Tensor)),
    )

    new_batch = _TensorDict(
        source=compressed_tensors,
        batch_size=obj.batch.batch_size,
    )
    obj = dataclasses.replace(obj, batch=new_batch)
```

**Why `dataclasses.replace()` instead of in-place mutation:**  
`DataProto` is a dataclass.  Mutating `.batch` in-place would affect the original
object still referenced by the trainer.  `dataclasses.replace()` creates a shallow
copy with only `.batch` replaced, leaving all other fields (meta_info, non_tensor_batch)
shared — cheap and safe.

---

### 4.4 Receiver Integration

**File:** `verl/protocol.py` — `DataProtoPullHandle.materialize()`

See §2.4.  Decompression is transparent to all worker code downstream.

**Guarantee:** No tensor that enters any `nn.Module.forward()`, loss computation, or
gradient calculation is compressed.  The float32 cast in `decompress_batch()` runs
before `materialize()` returns, so the worker always sees standard float32 tensors.

---

### 4.5 Probe Instrumentation

**File:** `verl/utils/transfer_compress.py` — `probe_compress_event()`

```python
def probe_compress_event(context: str, orig_bytes: int, comp_bytes: int) -> None:
    probe_log = os.environ.get("VERL_TRANSFER_PROBE_LOG")
    if not probe_log:
        return
    event = {
        "event":      "compress_stats",
        "context":    context,
        "mode":       get_compress_mode(),
        "orig_bytes": orig_bytes,
        "comp_bytes": comp_bytes,
        "ratio":      comp_bytes / orig_bytes if orig_bytes else 1.0,
        "ts_ms":      time.monotonic() * 1000,
    }
    with open(probe_log, "a") as f:
        f.write(f"[transfer_probe] {json.dumps(event)}\n")
```

The `summarize_transfer_probe.py` script consumes these events and outputs a
**Compression Stats** table:

```
## Compression Stats (pull dispatch)
| context | mode | iters | orig_KB/iter | comp_KB/iter | ratio | saved_pct |
| pull_build_handle | fp16 | 20 | 108.01 | 106.01 | 0.981 | 1.9% |
```

---

### 4.6 How to Enable

```bash
# FP16 (recommended):
VERL_TRANSFER_COMPRESS=fp16

# INT8 (higher compression, CPU quantization overhead):
VERL_TRANSFER_COMPRESS=int8

# With probe to measure savings:
VERL_TRANSFER_PROBE=1
VERL_TRANSFER_PROBE_LOG=/path/to/probe.jsonl
VERL_TRANSFER_COMPRESS=fp16

# Compression requires pull dispatch to be active:
trainer.use_legacy_worker_impl=enable
```

---

## 5. End-to-End Integration Checklist

Use this checklist to verify a correct deployment of all three features together.

### Code Changes

- [x] `verl/protocol.py` — `DataProtoSliceMeta`, `DataProtoPullHandle` (with `compress_meta`), `materialize()` decompression
- [x] `verl/single_controller/base/decorator.py` — `dispatch_nd_compute_dataproto_pull`, `dispatch_lazy_compute_data_proto_pull`, compression in `build_handle()`
- [x] `verl/workers/fsdp_workers.py` — `@register(dispatch_mode=dispatch_lazy_compute_data_proto_pull)` on `compute_log_prob`
- [x] `verl/trainer/ppo/ray_trainer.py` — async prep-stage dispatch, `critical_path_stage` probe events, `_compute_reward_colocate_async`
- [x] `verl/experimental/reward_loop/reward_loop.py` — `RewardScoreFuture`, `compute_rm_score_async`
- [x] `verl/utils/transfer_compress.py` — new file: compression utilities and probe emitter
- [x] `verl/utils/model.py` — attn_implementation fix (required for V100 compatibility)
- [x] `verl/models/transformers/monkey_patch.py` — trl ≥ 1.0 import guard

### Runtime Configuration

| Feature | Required env var | Required Hydra arg |
|---|---|---|
| Transfer Probe | `VERL_TRANSFER_PROBE=1`, `VERL_TRANSFER_PROBE_LOG=...` | — |
| Local-Batch Pull | — | `trainer.use_legacy_worker_impl=enable` |
| Async Overlap | — | `trainer.use_legacy_worker_impl=enable` |
| FP16 Compression | `VERL_TRANSFER_COMPRESS=fp16` | `trainer.use_legacy_worker_impl=enable` |
| INT8 Compression | `VERL_TRANSFER_COMPRESS=int8` | `trainer.use_legacy_worker_impl=enable` |
| V100 compatibility | — | `+actor_rollout_ref.model.override_config.attn_implementation=eager` |

### Validation Steps

1. **Pull works:** After one training step, `transfer_probe.jsonl` should contain
   `transfer_latency` events with non-zero `send_bytes` (shards being stored).
2. **Async overlap works:** `transfer_probe.jsonl` should contain
   `critical_path_stage` events with `dispatched_at_ms < joined_at_ms` and
   `hidden_frac > 0`.
3. **Compression works:** `transfer_probe.jsonl` should contain `compress_stats`
   events with `ratio < 1.0`.
4. **Training stability:** `critic/rewards/mean` should follow a similar trajectory
   with and without compression.

---

## 6. Key Design Decisions and Trade-offs

### Why `ray.put()` per shard instead of true metadata-only pull?

A pure metadata-only protocol would require workers to access a shared tensor buffer
via RDMA or shared memory and read their slice directly — essentially implementing a
custom distributed memory subsystem.  This would require significant changes to Ray's
actor model and is not feasible within the scope of this project.  The chosen approach
achieves the same `1/dp_size` transfer volume per worker while staying entirely within
Ray's existing primitives.

### Why compress before sharding?

Compression is a CPU operation on the trainer.  Doing it once (before sharding)
means the cost is paid once regardless of `dp_size`.  If compression happened after
sharding (inside the worker), each worker would pay the decompression cost, but the
trainer would still pay one full compression pass — same total CPU time, but more
complex code.  Compressing before sharding is simpler and equivalent in cost.

### Why join at a fixed sync point rather than using callbacks?

Ray does not natively support "fire-and-forget with callback" semantics for actor
remote calls in a way that integrates cleanly with Python's execution model.  The
`asyncio`-based approach would require the entire `fit()` loop to be async, which
is an invasive refactor.  The explicit join-at-sync-point approach achieves
identical overlap with much less complexity: as long as the CPU work between dispatch
and join is non-trivial, the futures will be partially or fully resolved by the time
`.get()` is called.

### Why float32-only compression (not int64)?

Token IDs and masks are semantically exact integers.  Any lossy compression (FP16 or
INT8 quantization) would corrupt them.  Lossless compression (e.g., LZ4) could be
applied to int64 tensors but would require a different code path and the savings
would be minimal for the near-random bit patterns of token IDs.  The current
implementation correctly identifies dtype as the compression eligibility criterion.

### Why is `trainer.use_legacy_worker_impl=enable` required?

verl has two worker dispatch implementations.  The newer one eagerly resolves futures
inside the dispatch call, making `blocking=False` a no-op.  The legacy implementation
correctly propagates the `blocking=False` flag through to the Ray remote call and
returns an unresolved `DataProtoFuture`.  Since async pipelining depends on receiving
true futures, the legacy path must be used.  This is a known verl limitation and is
being addressed upstream.
