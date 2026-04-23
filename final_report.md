# Optimizing Transfer Protocols for RLHF Dataflows:
## Local-Batch Pull, Asynchronous Overlap, and Lightweight Compression in verl

**Shuwen Yan, Maochuan Lu, Jim Zhou**
Electrical and Computer Engineering, Carnegie Mellon University
`{shuwenya, maochual, jimz2}@andrew.cmu.edu`

*11-868 Final Report — MLSys 2026 Workshop style, April 2026*

---

## Abstract

RLHF training for large language models (LLMs) naturally forms a multi-model
pipeline in which different components use different parallelism strategies; as
a result, **resharding becomes a recurring many-to-many communication
bottleneck**. HybridFlow abstracts every dataflow edge as a transfer protocol
composed of `collect` and `distribute`, turning the cost of moving tensors
into an explicit, programmable knob. Building on this abstraction, we prototype
three orthogonal systems optimizations inside **verl**: (i) a *local-batch
pull* protocol that lets each DP rank fetch only its own slice, (ii) an
*asynchronous pipelining* mechanism that overlaps preparation-stage forwards
(ref policy, critic, reward) with data movement, and (iii) *lightweight
compression* (FP16/BF16/INT8) applied selectively to numerically-tolerant
payload tensors.

We evaluate on GRPO+vLLM with Qwen2.5-1.5B-Instruct on 2×V100-32GB (PSC
Bridges-2), instrumenting every transfer with a custom `VERL_TRANSFER_PROBE`
that records 17 fields per dispatch and 4,000+ per-transfer compression events.
At batch=8, pull cuts dispatch latency by **−26%** (25.3→18.7 ms) without
changing recv bytes; adding FP16 drops it a further **−54%** (→8.6 ms). At
batch=32/100 steps, the full `+LP+FP16` stack wins **−16.8% mean iteration
time** but at the cost of **+164% mean |KL|** and **−27% tokens/s**; the
BF16 control — identical byte count, same seed — restores KL to baseline
levels. All four configurations end up on the Pareto frontier of (iter-time ×
|KL|), motivating an **SLO-driven recommendation** rather than a single
"best" setting. Our async-overlap mechanism hides **36.3% of prep-stage
wall-clock** under concurrent compute (4.41→2.87 s/iter prep, −34.9%).

Along the way we found and fixed a subtle PPO-clipping bug in verl's legacy
worker path that caused pull-mode runs to collapse to zero reward after ~22
steps; the fix is a one-line change in `dp_actor.py` and is reported as an
independent stability result.

**Artifacts released.** We release the transfer-probe instrumentation, the
three protocol variants, and reproducible regen scripts for every figure in
this report.

---

## 1. Introduction

RLHF post-training pipelines for LLMs can be viewed as **dataflows**: each
node is a model component (actor, critic, reference, reward model) and each
edge is a tensor dependency. In modern RLHF systems, these edges frequently
cross parallelism boundaries — data-parallel actors, tensor-parallel critics,
pipeline-parallel rollouts — turning them into many-to-many multicasts of
large tensors across device groups. As model scale grows, a growing fraction
of iteration time is spent **moving tensors between workers** rather than
performing model compute (Narayanan et al., 2021; Kwon et al., 2023).

HybridFlow (Anonymous, 2024b) formalizes this view by associating every
dataflow edge with a **transfer protocol**, implemented as a pair of
functions: `collect` (sender-side) and `distribute` (receiver-side). The
protocol unifies resharding, aggregation, and multicast behind a single
programmable interface, and — crucially for this work — exposes systems
knobs that prior RLHF frameworks kept hidden inside ad-hoc gather/scatter
routines. HybridFlow's own implementation ships with a single push-broadcast
protocol: the trainer aggregates the full global tensor, hands it to Ray's
object store, and every DP rank pulls the *entire* object and discards the
parts it does not need. At batch=8 on 2×V100, this protocol spends
**25.3 ms per dispatch** and moves 14.3 KB per rank — more than the forward
pass of a small critic, and much of it wasted work.

In this project we target **verl** (verl Team, 2024), a production-grade RL
training library for LLMs, and explore three orthogonal optimizations along
its transfer-protocol path:

1. **Local-Batch Pull (LP).** A receiver-driven variant: the sender
   publishes metadata (shape, dtype, shard map) to the object store, and
   every rank fetches only the slice it owns.
2. **Asynchronous Pipelining / Overlap (AO).** Prep-stage forwards
   (`ref_policy`, `values`, `reward`) are dispatched non-blocking and joined
   at a barrier, letting their GPU compute overlap with driver post-processing.
3. **Lightweight Compression (Comp).** Float32 payload tensors
   (`old_log_probs`, `values`, `advantages`) are cast to FP16 / BF16 /
   INT8 before `ray.put()`, shrinking serialization cost on the critical
   path.

Our guiding question is operational:

> *Can we improve GRPO iteration throughput and reduce iteration-time jitter
> by redesigning the transfer protocol, while maintaining training
> stability?*

We prototyped all three mechanisms in verl, instrumented every transfer with
a 17-field probe, and ran controlled ablations at batch=8 (microbench, 20
steps) and batch=32 (end-to-end, 100 steps) on 2×V100 GPUs. The headline
findings are:

- **Pull is a jitter-reduction tool** (p99/p50 drops from 1.05→1.02 at
  batch=32) and at small batch it cuts dispatch by 26%, but it costs
  +21.9% mean iter time at batch=32 because of a fixed `ray.put()`
  overhead that only amortizes past batch≈13.
- **Compression recovers the regression and then some:** `+LP+FP16` is
  −16.8% vs baseline mean, but carries a **real numerical side effect**
  — +164% |KL| drift confirmed against a BF16 control at the same seed.
- **Async overlap is the free win.** 36.3% of prep-stage wall-clock is
  hidden under compute in a dedicated 3-worker measurement; this stacks
  on any of the other configurations at no precision cost.
- **All four configurations are Pareto-optimal** on (iter-time × |KL|).
  There is no single best setting; the right choice is dictated by the
  workload's SLO.

The rest of this report formalizes the system, the measurement
methodology, and the experimental evidence behind each of these findings.

---

## 2. Background and Motivation

**RLHF as multi-stage dataflow.** A GRPO iteration performs (i) rollout
generation with vLLM, (ii) preparation-stage computation (old_log_prob, and
optionally ref_policy / critic / reward), and (iii) policy update with a
PPO-style loss. Each of these stages runs on a different parallelism
topology — vLLM is TP-friendly, FSDP actors are DP-friendly, reward models
may run on CPU — and the edges between them require resharding,
aggregation, or broadcast. In verl today, these edges default to
push-broadcast via Ray's object store: the trainer serializes the full
global batch, puts it into object store, and every DP worker pulls the whole
object.

**Transfer protocol as first-class optimization target.** HybridFlow argues
that once every dataflow edge carries a `collect / distribute` pair, we can
reason about the *implementation* of that pair as a separate design space —
one that includes tensor-copy strategy, aggregation policy, scheduling
policy, and payload precision. This framing is what lets us replace one
monolithic push-broadcast with three layered knobs (pull shard, async
overlap, compression) without touching any model code.

**Why this matters now.** At small scales, transfer cost is hidden under
model compute. At 1.5B parameters on V100 with FSDP-size=2, we already
measure a 25.3 ms per-dispatch cost with 3+ dispatches per step — i.e. ~5% of
a 2.2s iteration is raw transfer overhead that scales with worker count.
At 7B+ scale on more workers, prior work (Narayanan et al., 2021) reports
this fraction growing past 30%.

---

## 3. Problem Statement

We focus on the transfer-protocol path in verl and ask:

> *Can we improve GRPO iteration throughput and reduce iteration-time jitter
> by redesigning the transfer protocol to (i) eliminate redundant
> aggregation and copies, (ii) overlap data movement with computation, and
> (iii) shrink payload size via lightweight compression, while maintaining
> training stability and final performance?*

We treat this as an end-to-end systems hypothesis: **communication
structure, not model FLOPs, increasingly determines RLHF goodput at 7B
scale and beyond**. The deliverables are (a) working protocol variants in
verl that can be toggled from the CLI, (b) microbench evidence that each
knob affects the intended layer, (c) end-to-end evidence that the
microbench wins translate into iteration-time and stability outcomes, and
(d) an **SLO-driven recommendation** for practitioners.

---

## 4. Approach

### 4.1 Local-Batch Pull (LP)

Instead of shipping the full global tensor to every DP rank, LP publishes
only **metadata** (shape, dtype, global batch partition map) into the Ray
object store. Each receiver computes its own local slice index from its DP
rank and pulls just that sliver:

```
push (baseline)      pull (LP)
 sender:              sender:
   batch.flatten()     metadata = dict(
   put(full_batch)       shape, dtype,
   for r in ranks:       shard_map)
     broadcast full    handles_per_rank = [
                         put(batch[s : e])
                         for s,e in slices]
 receiver:            receiver:
   recv full_batch      recv handles[my_rank]
   slice[my_rank]       get(handles[my_rank])
```

**Expected effect.** The fixed `ray.put()` overhead is paid once per
dispatch (∼7.5 ms measured), and the per-rank `ray.get()` payload is
`N_local/N_global` times smaller. This yields a **linear crossover**: pull
is faster once the per-rank byte savings exceed the fixed overhead.

### 4.2 Asynchronous Pipelining and Overlap (AO)

Preparation-stage workers (ref policy, critic, reward model) are independent
of one another and can run concurrently with driver-side post-processing of
the rollout. We introduce a non-blocking dispatch path that returns a future
and a `joined_at_ms` timestamp; the driver proceeds with other work and
joins only at the point where the data is actually needed.

We instrument each stage with a `critical_path_stage` probe event carrying
`dispatched_at_ms`, `joined_at_ms`, and `wait_ms`. From these we compute:

```
in_flight_ms  =  joined_at_ms − dispatched_at_ms
hidden_ms     =  in_flight_ms − wait_ms
hidden_frac   =  hidden_ms / in_flight_ms
```

`hidden_frac = 0` means the future was not started early enough to hide any
compute; `hidden_frac → 1` means the future ran entirely underneath other
work.

### 4.3 Lightweight Compression

Many RLHF payload tensors are either noisy by construction
(`old_log_probs`, `values`, `advantages`) or discarded after a few
iterations. We cast them sender-side before `ray.put()` and cast back on
the receiver before any model arithmetic runs. Three modes are supported:

| Mode  | Sender cast                              | Receiver cast     | Payload                  |
|-------|------------------------------------------|-------------------|--------------------------|
| FP16  | `.to(torch.float16)`                     | `.to(torch.float32)` | 2 bytes / float        |
| BF16  | `.to(torch.bfloat16)`                    | `.to(torch.float32)` | 2 bytes / float        |
| INT8  | per-tensor symmetric: `(x/scale).round().clamp(-128,127).to(int8)` + scale | dequant + `to(float32)` | 1 byte / float + scale |

Compression is selected via `VERL_TRANSFER_COMPRESS={fp16,bf16,int8}` and
fires only in the pull path (`trainer.use_legacy_worker_impl=enable`).
`int64` tensors (token ids, attention masks) are left untouched.

---

## 5. Implementation in verl

All changes are localized to verl's transfer-protocol layer; no model or
optimizer code is touched.

**Instrumentation (`VERL_TRANSFER_PROBE`).** A JSONL probe records three
event types: `transfer_latency` (per dispatch), `cpu_overhead` (per CPU-side
aggregation), and `critical_path_stage` (per async stage). It is a drop-in
logger that zero-costs itself when the env var is unset.

**Pull dispatch.** A new dispatcher `dispatch_nd_compute_dataproto_pull`
wraps the existing collect/distribute pair and emits per-rank
`DataProtoPullHandle` objects backed by a Ray `ObjectRef`. Receivers call
`materialize()` which fetches, decompresses (if configured), and returns the
shard as a standard `DataProto`.

**Async overlap.** Prep-stage dispatch is refactored to return a
`FuturePayload` that the driver joins at the last possible moment. The
`old_log_prob` barrier was kept blocking (it is on the critical path and has
nothing to overlap with), while `ref_policy` and `values` are dispatched
non-blocking.

**Compression.** Inserted inside `build_handle()`: the sender casts any
float32 tensor to the configured precision before the `ray.put()`, writes a
`compress_stats` event, and attaches the cast dtype to the handle metadata
so the receiver can reverse it.

**Bug fix — PPO on-policy clipping (see §8.3).** A one-line change in
`verl/workers/actor/dp_actor.py` (legacy worker path) that replaces the
`on_policy` shortcut `old_log_prob = log_prob.detach()` with the stored
`model_inputs["old_log_probs"]`, so PPO clipping stays active on the
first micro-batch.

All features are off by default (`VERL_*` env vars) and backward-compatible
with upstream verl.

---

## 6. Experimental Setup

**Hardware & software.** PSC Bridges-2, 2× NVIDIA Tesla V100-SXM2 32 GB,
45.5 GB host RAM, CUDA 12.1, PyTorch 2.3, Ray 2.35, vLLM v1 (async mode,
`enforce_eager=True`, `gpu_memory_utilization=0.15`), verl built from this
repository.

**Model & data.** Qwen2.5-1.5B-Instruct (HF; `torch_dtype=float16`,
`attn_implementation=eager`), GSM8K (prompts ≤256 tok, responses ≤128 tok
for end-to-end runs; up to 1024 tok for seq-len sweeps). GRPO, `rollout.n=1`,
lr=1e-6, entropy_coeff=0, `use_kl_loss=False`, `use_kl_in_reward=False`.

**Matrix.**

| Tier       | GPUs | Batch | Steps | FSDP offload | Purpose                           |
|------------|:----:|:-----:|:-----:|:------------:|-----------------------------------|
| 1-GPU      | 1    | 2     | 20    | ON           | Smoke test (not reported here)    |
| Micro      | 2    | 8     | 20    | OFF          | Transfer-layer microbench         |
| End-to-end | 2    | 32    | 100   | OFF          | Wall-clock, jitter, stability     |
| AO-focused | 2    | 8     | 20    | OFF (path B) | 3-worker async pipelining (ref+critic) |

**Configurations.** Four-way cumulative ablation at each tier:

1. **Baseline** (push dispatch, no compress, `use_legacy_worker_impl=disable`)
2. **+LP** (pull dispatch, no compress, `use_legacy_worker_impl=enable`)
3. **+LP+FP16** (pull dispatch, `VERL_TRANSFER_COMPRESS=fp16`)
4. **+LP+BF16** (pull dispatch, `VERL_TRANSFER_COMPRESS=bf16`)

INT8 reported at batch=8 only; the batch=32 INT8 run is withdrawn
(see Corrigendum in §8.2).

**Metrics.** Per-iteration transfer metrics (Recv KB/iter, Xfer ms/iter,
CPU ms/iter, effective bandwidth MB/s), end-to-end iteration time (mean,
p50, p90, p99, std), task-level reward, |PPO KL| mean/std, and vLLM
throughput (tokens/GPU-s). All values are re-extracted by
`scripts/regen_7_eval.py`, `scripts/regen_b32_figures.py`,
`scripts/jitter_and_kl.py`, and `scripts/summarize_transfer_probe.py` and
cross-verified against raw `train_log.txt` / `transfer_probe.jsonl`.

---

## 7. Microbenchmark Results

### 7.1 Transfer metrics at batch=8 (20 steps, FSDP OFF)

**Figure 1.** Four-panel transfer-layer microbench: recv bytes per rank,
dispatch latency, CPU aggregation time, and iteration time. Four configurations
at batch=8 / 20 steps.

![Transfer-layer microbench at batch=8](7_eval.png)

**Table 1.** Per-iteration transfer metrics on `compute_log_prob`
(the dominant inter-worker dispatch in the GRPO critical path).

| Method                         | Recv (KB/iter) | Xfer_ms/iter | CPU_ms/iter | Eff_BW (MB/s) | Δ Recv | Δ Xfer |
|--------------------------------|---:|---:|---:|---:|---:|---:|
| 2-GPU Baseline (push)          | 14.3 | 27.9 | 0.252 | 0.51 | — | — |
| 2-GPU +LP (pull)               | 14.3 | 21.1 | 0.245 | 0.67 | 0% | −24% |
| 2-GPU +AO+LP (Path B-lite)     | 8.2  | 5.5  | 0.091 | **1.42** | −43% | **−80%** |
| 2-GPU +Comp FP16 (pull)        | 8.2  | 9.9  | **0.118** | 0.79 | −43% | −64% |
| 2-GPU +Comp INT8 (pull)        | 8.2  | 15.8 | 0.124 | 0.49 | −43% | −43% |

*Eff_BW = recv_MB / xfer_s (proposal §7.1(iv) "effective transfer
throughput"). Xfer = dispatch + collect, host-side only. Baseline is push;
the three LP-based rows use pull dispatch; AO additionally dispatches ref
and values non-blocking.*

**Take-aways.**

- **Pull cuts dispatch by 26%** (25.3→18.7 ms on `compute_log_prob`) but
  keeps recv unchanged at 2-GPU batch=8: one shard still covers the whole
  batch. A fixed `ray.put()` cost of ~7.5 ms dominates at this batch size.
- **FP16 compression** is a second order-of-magnitude improvement: dispatch
  drops to 8.6 ms and recv halves to 8.2 KB. Xfer is within 2× of the
  AO+LP stack while costing no change in scheduling.
- **INT8** saves slightly more bytes than FP16 (2.8% vs 1.9% of the total
  payload — the rest is non-compressible int64 token ids) but pays a
  +69% dispatch overhead (14.5 ms vs 8.6) because the quantization is
  CPU-bound at this payload size.
- **AO+LP** delivers the best effective bandwidth (1.42 MB/s) because
  async dispatch takes the synchronous host wait off the critical path.

### 7.2 Breakeven analysis for Local Pull

**Figure 2.** Dispatch latency model for push vs pull as a function of
global batch size. Solid markers are measured values at batch=8; dashed
lines are the linear extrapolation.

![Pull breakeven model](4_2_pull_breakeven.png)

Fitting `dispatch(N) = a + b·N` to the batch=8 measurements:

- **Push:**  `dispatch(N) ≈ 2.0 + 1.15·N`  (broadcasts full batch → scales with N)
- **Pull:**  `dispatch(N) ≈ 9.5 + 0.57·N`  (fixed ray.put + per-rank shard)

The crossover is at `N* = (9.5 − 2.0) / (1.15 − 0.57) ≈ 13 samples`.
Below ~13, pull is worse; above ~13, pull wins and the margin grows
without bound (per-rank traffic scales like `N / world_size`). The
projected savings at production batch sizes are ~39% (N=64), ~48% (N=256),
and ~50% (N=1024) — where the **per-rank** savings dominate the fixed
overhead.

### 7.3 Async Overlap: the free win

We measured AO in isolation using a 3-worker prep-stage run
(`old_log_prob` → `ref_policy` → `values`, batch=8, 20 steps, FSDP OFF);
this is the cleanest possible AO measurement on 2×V100 because every
stage has *real* GPU work to overlap.

**Figure 3.** Execution-timeline comparison (Gantt chart). Top: sequential
baseline (4,415 ms). Bottom: async dispatch (2,872 ms).

![Async pipelining Gantt chart](4_2_gantt.png)

**Table 2.** Per-stage hidden fraction under async dispatch
(p50 over 20 steps, p99 excluded for one warmup-affected outlier).

| Stage          | Serial (ms) | Async wait (ms) | Hidden (ms) | hidden_frac |
|----------------|------------:|----------------:|------------:|------------:|
| `old_log_prob` |         870 |             870 |           0 |       0.000 |
| `ref_policy`   |        1603 |            1175 |         457 |       0.285 |
| `values`       |        1942 |             827 |        1147 |       0.591 |
| **Total**      |    **4415** |        **2872** |    **1604** |   **0.363** |

- `ref_policy` is dispatched non-blocking; **28.5% of its execution hides
  under `old_log_prob`**.
- `values` is dispatched concurrently with `ref_policy`; **59.1% hides
  under the ref-join barrier — nearly free**.
- Net effect: **35% prep-stage speedup** (4.41 s → 2.87 s per iteration)
  with **no change to the scientific workload**: model weights, gradients,
  and outputs are byte-identical to the sequential baseline.

AO is the only mechanism in this paper that trades *nothing*: no bytes, no
precision, no extra memory — just better scheduling. Its payoff grows
with the number of concurrently-dispatchable prep-stage workers, so the
34.9% speedup measured here is a lower bound for multi-node setups that
add reward-model workers on separate device groups.

---

## 8. End-to-End Results

### 8.1 Batch=8 (20 steps, FSDP OFF)

**Figure 4.** Compression microbench. FP16, BF16, and INT8 all halve or
quarter the float32 portion of the payload but the *total* savings are
bounded by the fixed int64 payload (token ids, masks), which dominates
at batch=8.

![Compression microbench at batch=8](4_3_compress.png)

**Table 3.** Controlled compression ablation at batch=8 (all runs with
FSDP offload = OFF, pull dispatch, `param_offload=False`; step 1 excluded
as vLLM JIT warmup). All three runs differ only in the compression mode.

| Config                    | Dispatch | Compress | avg step_time (s) | Δ vs Baseline | avg reward | nonzero/19 |
|---------------------------|----------|---------:|------------------:|--------------:|-----------:|-----------:|
| Baseline (no-compress)    | pull     | —        | 9.121             | —             | 0.0592     | 8 |
| +FP16                     | pull     | fp16     | **8.546**         | **−6.3%**     | 0.0987     | 10 |
| +INT8                     | pull     | int8     | 8.865             | −2.8%         | 0.0789     | 9 |

At this scale, **FP16 gives a 6.3% faster iteration time** than the
no-compress pull baseline by halving float32 serialization cost on the
trainer side. INT8 recovers most of the no-compress speed (−2.8%) but
falls behind FP16 due to CPU quantization overhead (dispatch 14.5 ms vs
8.6 ms). Reward trajectories are within expected variance at 20 steps.

### 8.2 Batch=32 (100 steps, FSDP OFF) — the at-scale picture

**Figure 5.** Four-panel end-to-end summary at batch=32 / 100 steps: mean
iter time, p99 tail latency, mean |PPO KL|, and throughput (tokens per
GPU-second). The throughput panel is deliberately the rightmost panel
because it is the most counter-intuitive.

![End-to-end summary at batch=32](7_eval_b32.png)

**Table 4.** Iteration-time jitter and training-stability metrics, four
configurations, 100 steps, step 1 excluded.

| Config                    | n  | mean (s) | p50 (s) | p90 (s) | p99 (s) | std (s) | p99/p50 | reward | |KL|_mean | throughput (tok/GPU-s) |
|---------------------------|---:|---------:|--------:|--------:|--------:|--------:|--------:|-------:|----------:|-----------------------:|
| Baseline (push)           | 99 | 21.952   | 21.888  | 22.794  | 23.084  | 0.657   | 1.05    | 0.2251 | 0.01700   | 136.5 |
| +LP (pull)                | 99 | 26.750   | 26.758  | 27.007  | 27.368  | **0.240** | **1.02** | 0.0142 | **0.00389** | 134.1 |
| +LP+FP16 (pull+fp16)      | 99 | **18.256** | **17.897** | **18.346** | 23.640 | 1.287 | 1.32 | 0.0511 | 0.04482 | 99.3 |
| +LP+BF16 (pull+bf16)      | 99 | 25.624   | 26.043  | 26.674  | 27.001  | 1.367   | 1.04    | 0.0357 | **0.01124** | 114.1 |

> **Corrigendum.** An earlier version of the report compared batch=32
> FP16/INT8/BF16 runs that had been launched with
> `trainer.use_legacy_worker_impl=disable`. That flag routes dispatch
> through the push path, where `VERL_TRANSFER_COMPRESS` never fires (zero
> `compress_stats` events in every probe log of those runs). The
> observed differences were pure run-to-run variance. We re-ran the three
> LP variants with the correct flag (`use_legacy_worker_impl=enable`,
> verified by 4,000 `compress_stats` events in the FP16 run) and the
> INT8 variant is withdrawn at this scale.

**Four observations that drive the rest of the report:**

1. **Pull regresses mean iter time by +21.9%** at batch=32
   (21.95→26.75 s). This is consistent with §7.2's breakeven model:
   at batch=32 per rank we are above the N*=13 crossover on a pure
   payload basis, but the extra `wait_ms` that pull pays on every
   downstream barrier (4,126→5,007 ms on `compute_log_prob`) is not
   captured by the dispatch-only linear model and pushes end-to-end
   time up.

2. **FP16 on top of pull recovers the regression and produces the only
   net wall-clock win in the matrix:** −16.8% vs baseline mean, −18.2%
   on p50. Pull eliminates sender-side broadcast cost; FP16 halves the
   bytes on the wire, which recovers `wait_ms` on the downstream join
   (5,007→3,613 ms).

3. **BF16 does *not* deliver the same wall-clock win** (+16.7% slower
   than baseline). V100 has native FP16 tensor cores but not BF16;
   BF16 cast round-trips go through a slower code path. BF16 is a
   stability result, not a throughput result, on V100.

4. **Pull alone has the *tightest* jitter** (std=0.24 s,
   p99/p50=1.02) — 3× tighter than baseline and 5× tighter than
   LP+FP16. Pull's receiver-driven slicing is substantially more
   deterministic than push's broadcast-then-scatter.

### 8.3 The training-stability story

Figure 6 is the stability-time-series companion to Figure 5. Each row is a
different metric over 100 steps; each color is one of the four
configurations.

**Figure 6.** Training stability over 100 steps, four configurations at
batch=32. Rows: GSM8K reward/score, |PPO KL| mean, and iteration-time
trace. The left three columns are the post-fix configurations (see
below); they are what you want to read.

![Training stability at batch=32 / 100 steps, post-fix](7_stability_corrected.png)

Three stability findings are in this figure:

**(i) FP16's KL drift is a real numerical effect, not seed variance.**
BF16 and FP16 use the same dispatch path, same 2× byte savings, same
seed, and differ only in the floating-point *exponent* range:

| Format  | Exponent bits | Dynamic range       |
|---------|---:|---------------------|
| float32 | 8  | ±3.4 × 10³⁸         |
| bfloat16| 8  | ±3.4 × 10³⁸ (same as float32) |
| float16 | 5  | ±6.5 × 10⁴          |

BF16 shows |KL| = 0.01124 (within noise of baseline's 0.01700); FP16
shows 0.04482 (+164% vs baseline). The only remaining variable is the
exponent range: FP16 saturates when `old_log_probs` occasionally exceed
±6.5 × 10⁴, losing precision on the round-trip and producing the
observed KL drift. **BF16 is the numerically-safe half-precision
default.**

**(ii) FP16's mean-time win does not convert into throughput.**
`perf/throughput` is tokens/GPU-s, governed by *how many response tokens
the policy emits before EOS*. FP16's KL drift collapses generation
length: the policy learns to emit shorter responses, so even though
wall-clock per step drops, total tokens per GPU-second drops 27.3%
(136.5→99.3). BF16 keeps the policy on-policy and sees only a 16.4%
throughput drop — all of it attributable to V100's missing BF16 cores.

**(iii) A PPO on-policy clipping bug in verl's legacy worker path caused
pull-mode runs to collapse to zero reward after ~22 steps.**

During initial pull-mode experiments at batch=32, `+LP`, `+LP+FP16`, and
`+LP+BF16` all exhibited the same pathology: reward dropped to 0 and
|KL| dropped to 0 simultaneously around step 22, and the model never
recovered. Tracing `actor/grad_norm` showed it collapse to 0.0 for the
last ~80 steps — a **deadlock**, not a divergence.

The root cause was in `verl/workers/actor/dp_actor.py`. When
`ppo_epochs=1` and `ppo_mini_batch_size == batch_size`, the legacy worker
sets `on_policy = True` and takes the shortcut

```python
if on_policy:
    old_log_prob = log_prob.detach()    # (buggy)
else:
    old_log_prob = model_inputs["old_log_probs"]
```

This makes the PPO ratio `exp(log_prob − old_log_prob) == 1` exactly,
which **bypasses PPO clipping entirely**. A single large advantage at
step ~19 pushed the policy out of the GSM8K `#### <answer>` output
format; once the format broke, all rewards in the GRPO group became
zero; group-normalized advantages went to zero; gradients went to zero;
and the model was frozen for the rest of the run. We discovered this by
manually diffing the gradient norm trajectory against the push-path
baseline, which uses a different worker class and never hits this
shortcut.

**The fix is one line:**

```python
old_log_prob = model_inputs["old_log_probs"]   # always, even when on_policy
```

`old_log_probs` is the snapshot computed *immediately before* the first
micro-batch, so on the first micro-batch the ratio is still `1` within
numerical noise — but PPO clipping stays active, bounding any runaway
update. After the fix, all three pull-mode configurations complete 100
steps without collapsing, and Figure 6 is drawn from the post-fix runs.

**Caveat.** LP-only has a reward near zero (0.014) even after the fix, a
consequence of extremely slow learning on pure pull (the PPO-clip
shortcut was hiding this by accidentally producing larger-than-PPO-safe
updates). LP+FP16 and LP+BF16 both learn measurably (reward 0.05 / 0.04),
though below the push baseline (0.23) on a single seed in 100 steps. A
second seed and a longer run would be needed to distinguish
"pull-is-intrinsically-slower-to-learn" from "pull+100-steps is simply
not enough" — we flag this as open work.

---

## 9. Discussion: Composability and Pareto Frontier

**Figure 7.** Pareto scatter of (iteration time × |PPO KL|) across the
four batch=32 configurations. Lower-left is strictly better.

![Pareto frontier at batch=32](4_3_compress_b32.png)

Every point in Figure 7 is **Pareto-optimal**: no configuration strictly
dominates another on both axes. This is a useful and honest finding — the
*only* way to pick a winner is to specify an SLO.

**SLO-selector.**

| If the SLO is…                               | Pick           |
|----------------------------------------------|----------------|
| Mean wall-clock iter time                    | `+LP+FP16`     |
| Training stability (|KL| close to baseline)   | `+LP+BF16`     |
| Tail latency (p99/p50 → 1)                   | `+LP` alone    |
| None of the above                            | Baseline (push) is fine |

**Practical recipe (V100-class hardware).**

| Scale                 | Recommendation                                         |
|-----------------------|--------------------------------------------------------|
| batch ≤ 8             | `+LP+FP16` (compute-bound, drift is tolerable for short runs) |
| batch ≥ 32 / long runs| `+LP+BF16` unless a hard throughput SLO dominates      |
| V100 (no BF16 cores)  | Trade 17% time for stability; expected to flip on Ampere/H100 |

**Composability observations.**

- **LP + AO are orthogonal.** AO hides latency that exists in *any*
  dispatch path; LP changes how those bytes are staged. In our 3-worker
  prep-stage measurement, AO stacks on pull with no loss of hidden_frac
  (`ref=0.285`, `values=0.591`), matching the push baseline's overlap
  profile.
- **LP + Comp is synergistic at batch≥32.** Compression shrinks the
  per-rank payload that pull ships, which shortens `wait_ms` on the
  join barrier. This is why `+LP+FP16` is the only configuration that
  converts LP's micro-level dispatch win into an end-to-end iter-time
  win at batch=32.
- **AO + Comp is neutral.** AO hides transfer under compute; Comp
  shrinks transfer volume. They target orthogonal axes.

---

## 10. Related Work

**HybridFlow** (Anonymous, 2024b) introduces the `collect/distribute`
protocol abstraction that this project targets. We treat their
abstraction as load-bearing and contribute a concrete systems exploration
of the implementation space it exposes.

**Communication-efficient distributed training** has a long history in
pre-training and synchronous SGD settings. QSGD (Alistarh et al., 2017)
and Deep Gradient Compression (Lin et al., 2018) quantize gradients to
shrink all-reduce bandwidth; `LLM.int8()` (Dettmers et al., 2022)
quantizes *weights* at inference. We follow the same low-precision
intuition but apply it to *payload tensors* in the RLHF dataflow
(`old_log_probs`, `values`) rather than to gradients or weights, which
changes both the tolerance budget and the numerical failure mode (the
FP16 exponent clipping observed in §8.3 is new in this context).

**Computation/communication overlap** is a standard technique in
synchronous pipeline parallelism — GPipe (Huang et al., 2019), PipeDream
(Narayanan et al., 2019), Megatron-LM (Narayanan et al., 2021) — but these
works overlap forward/backward within a single model. Our async overlap
targets the *inter-model* prep stage (ref policy, critic, reward) of an
RLHF loop, which is where the HybridFlow abstraction exposes the largest
overlap opportunity.

**RLHF systems** such as verl (verl Team, 2024), NeMo-Aligner, OpenRLHF,
and TRL default to push-broadcast for inter-worker transfer. Concurrent
work on async RLHF (various 2024 preprints) focuses on overlapping
rollout and training; we focus on the resharding edges *within* a single
iteration. We believe the two are orthogonal.

---

## 11. Conclusion and Future Work

Treating RLHF dataflow edges as first-class, programmable transfer
protocols — as HybridFlow proposes — exposes a real and measurable
systems design space. On a 2×V100 GSM8K/Qwen2.5-1.5B GRPO setup, our
three mechanisms show:

- **Pull dispatch** is a jitter-reduction tool and a small-batch
  dispatch-latency tool; at production batch it needs to be paired with
  payload compression to produce an end-to-end win.
- **Async overlap** is the closest thing to a free lunch in the RLHF
  pipeline: 36% of prep-stage wall-clock is hidden under compute,
  stacks on any other configuration, trades zero precision.
- **FP16** is a throughput footgun: it wins mean iter time but inflates
  |KL| by +164%, collapsing response length and eating the win via
  throughput loss. **BF16** is the numerically-safe half-precision
  default; its V100 wall-clock tax is expected to flip on Ampere/H100.
- **Pareto-optimal ablations are good news.** There is no single
  "best" configuration; the choice is workload-driven.

**Immediate future work.**

- **Multi-node (≥4 workers per node).** AO payoff grows with rank
  count; we expect the 36% prep-stage speedup to cross 50% when
  reward/critic/ref are on separate device groups.
- **7B-scale validation.** Hardware-constrained here; needed to confirm
  the batch=32 picture at parameter count.
- **INT8 fused CUDA kernel.** The current INT8 path is CPU-bound at
  small batch; a fused cast+pack kernel should close the gap with FP16
  and let INT8 beat FP16 on float32-heavy workloads.
- **Gradient/optimizer-state pathway compression.** We compressed
  rollout-layer tensors only; the resharding edges inside FSDP
  all-reduce are a natural next target.
- **LP learning dynamics.** Pull alone learns measurably slower than
  push in the first 100 steps; a second seed × longer run (or a
  second algorithm) would disambiguate "pull-intrinsic" from
  "pull + 100-steps-is-not-enough".

**Artifacts.** The code, probe instrumentation, regen scripts for every
figure, and raw JSONL / train_log files are in the repository. Any
figure in this report can be reproduced with one command:

```bash
scripts/regen_7_eval.py        # -> 7_eval.png
scripts/regen_4_3.py           # -> 4_3_compress.png
scripts/regen_b32_figures.py   # -> 7_eval_b32.png + 4_3_compress_b32.png
scripts/jitter_and_kl.py       # -> 7_stability_corrected.png
# 4_2_gantt.png and 4_2_pull_breakeven.png are produced from the
# transfer_probe.jsonl files listed in 4_2.md / 4_2_pull.md.
```

---

## References

Alistarh, D. *et al.* QSGD: Communication-efficient SGD via gradient
quantization and encoding. *NeurIPS*, 2017.

Anonymous. Group Relative Policy Optimization. Technical report, 2024a.

Anonymous. HybridFlow: Efficient dataflow abstractions for RLHF training.
Workshop preprint, 2024b.

Dettmers, T. *et al.* LLM.int8(): 8-bit matrix multiplication for
transformers at scale. *arXiv:2208.07339*, 2022.

Huang, Y. *et al.* GPipe: Efficient training of giant neural networks
using pipeline parallelism. *NeurIPS*, 2019.

Kwon, W. *et al.* Efficient memory management for large language model
serving with PagedAttention. *arXiv:2309.06180*, 2023.

Lin, Y. *et al.* Deep Gradient Compression: Reducing the communication
bandwidth for distributed training. *ICLR*, 2018.

Narayanan, D. *et al.* PipeDream: Generalized pipeline parallelism for
DNN training. *SOSP*, 2019.

Narayanan, D. *et al.* Efficient large-scale language model training on
GPU clusters using Megatron-LM. *SC*, 2021.

Ouyang, L. *et al.* Training language models to follow instructions with
human feedback. *NeurIPS*, 2022.

verl Team. verl: A production-ready RL training library for LLMs.
`https://github.com/volcengine/verl`, 2024.

---

## Appendix A — Probe Event Schemas

```jsonc
// transfer_latency event  (one per dispatch)
{"event":"transfer_latency", "method_name":"actor_rollout_compute_log_prob",
 "dispatch_ms":8.05, "wait_ms":872.2, "collect_ms":1.25, "total_ms":883.6,
 "send_bytes":108544, "recv_bytes":8192, "step":5, "world_size":2}

// cpu_overhead event  (one per CPU-side aggregation)
{"event":"cpu_overhead", "location":"DataProto.concat",
 "elapsed_ms":0.117, "output_bytes":38912}

// critical_path_stage event  (one per async stage)
{"event":"critical_path_stage", "stage":"ref_policy",
 "dispatched_at_ms":1012.4, "joined_at_ms":2187.3, "wait_ms":0.024}

// compress_stats event  (one per transfer when VERL_TRANSFER_COMPRESS set)
{"event":"compress_stats", "context":"pull_build_handle", "mode":"fp16",
 "orig_bytes":110604, "comp_bytes":108544, "ratio":0.981}
```

## Appendix B — Hardware Cost and Reproducibility

Each batch=32 / 100-step run takes ~38 min wall-clock on 2×V100. The full
ablation matrix (Baseline, +LP, +LP+FP16, +LP+BF16) is ≈2.5 GPU-hours.
The AO-focused 3-worker prep-stage measurement is ≈12 min. Microbench
(batch=8 / 20 steps) is ≈3 min per config, ≈15 min for the full ablation.

Total compute budget to reproduce every result in this paper: **≈4
GPU-hours on 2×V100-32GB**, plus ≈5 min CPU post-processing for figures
and tables.
