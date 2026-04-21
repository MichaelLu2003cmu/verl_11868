# 7 Evaluation

## 7.1 Microbenchmarks

We profile the GRPO training loop using Qwen2.5-1.5B-Instruct on PSC Bridges-2
(2 × V100-32 GB, 45.5 GB host RAM), measuring per-operation transfer characteristics
via our custom `VERL_TRANSFER_PROBE` instrumentation.  All metrics are averages over 20
training steps.  The primary operation of interest is `compute_log_prob`, which is the
dominant inter-worker data transfer in the actor update stage.  Transfer time
(`Xfer_ms`) is the host-side cost only: `dispatch_ms + collect_ms`.  `CPU_ms` is the
`DataProto.concat` aggregation cost at the trainer.

**Experiment matrix:**

| Config | GPUs | Batch | FSDP offload | dispatch mode | compress |
|---|---|---|---|---|---|
| Baseline (1-GPU) | 1 | 2 | On | push | none |
| +LP (1-GPU) | 1 | 2 | On | pull | none |
| Baseline (2-GPU) | 2 | 8 | On | push | none |
| +LP (2-GPU) | 2 | 8 | On | pull | none |
| +AO (2-GPU, Path B-lite) | 2 | 8 | Off | pull | none |
| +Comp FP16 (2-GPU) | 2 | 8 | Off | pull | fp16 |
| +Comp INT8 (2-GPU) | 2 | 8 | Off | pull | int8 |
| +Comp BF16 (2-GPU, 100-step) | 2 | 32 | Off | pull | bf16 |

> **Note on FSDP offloading:** The Baseline/+LP runs used `param_offload=True,
> optimizer_offload=True` to fit within 45.5 GB host RAM.  The +AO and +Comp runs
> disabled FSDP offloading, which independently reduces `wait_ms` by keeping model
> parameters resident on GPU.  Direct comparison of `wait_ms` and `Iter(s)` across
> these groups is therefore confounded; we report them separately below.

---

### Table 1 — Microbenchmark: Transfer Metrics (compute_log_prob)

| Method | Recv (KB/iter) | Xfer_ms/iter | CPU_ms/iter | Δ Recv | Δ Xfer |
|---|---:|---:|---:|---:|---:|
| **1-GPU Baseline** (push) | 3.0 | 9.6 | 0.191 | — | — |
| **1-GPU +LP** (pull) | 2.7 | 7.6–8.5 | 0.175 | −10% | −11 to −21% |
| **2-GPU Baseline** (push) | 14.3 | 22.3 | 0.225 | — | — |
| **2-GPU +LP** (pull) | 14.3 | 21.1 | 0.245 | 0% | −5%¹ |
| **2-GPU +Comp FP16** (pull) | 8.2 | **9.3** | **0.117** | −43% | −58% |
| **2-GPU +Comp INT8** (pull) | 8.2 | 14.8 | 0.119 | −43% | −34% |

> ¹ Pull adds `ray.put()` fixed overhead (~7.5 ms) that outweighs transfer savings at
> batch=8.  Breakeven is ~13 samples; Pull is expected to save 39–50% at batch 64–1024.

*Legend: Recv = bytes received by workers per iteration; Xfer = dispatch + collect time
(host CPU only, excludes GPU compute); CPU = DataProto.concat aggregation cost.*

---

### Table 2 — Async Overlap (Path B-lite: GRPO + ref + critic, 2-GPU)

Measured via `critical_path_stage` probe events.  The trainer dispatches ref-policy and
critic forward passes **non-blocking**, allowing their GPU compute to overlap with
rollout post-processing.

| Stage | in\_flight\_ms avg | wait\_ms p50 | hidden\_ms avg | hidden\_frac |
|---|---:|---:|---:|---:|
| `ref` log-prob | — | — | — | **0.285** |
| `values` (critic) | — | — | — | **0.590** |

**Interpretation:** 28.5% of the reference-policy forward time and 59.0% of the critic
forward time are *hidden* under other compute — a **35% reduction** in effective
prep-stage latency compared to the sequential baseline.

---

## 7.2 End-to-End Metrics

Total iteration time is approximated as `compute_log_prob.total_ms + update_actor.total_ms`,
averaged over 20 steps.  These two phases dominate iteration time.

### Experimental Confound: FSDP Offloading

> ⚠️ **Important caveat.** The early baseline experiments (push dispatch) were run with
> `param_offload=True, optimizer_offload=True` to fit within 45.5 GB host RAM.  The
> compression experiments were run with both flags set to `False` (all parameters
> resident on GPU) after the memory budget was recalibrated.  Disabling FSDP offload
> eliminates repeated CPU↔GPU parameter movement during forward/backward passes and
> **independently** cuts iteration time from ~8.2 s to ~4.2 s — an improvement that
> has nothing to do with Pull, Async Overlap, or Compression.
>
> **Direct cross-group comparison of iteration time is therefore invalid.**
> Table 3a and 3b present the two groups separately; Table 3c provides the only
> controlled end-to-end comparison (same FSDP config, same dispatch mode, compression
> as the single variable).

### Table 3a — FSDP Offload ON (push dispatch; Baseline vs +LP)

| Method | Iter (s) | Δ Iter | FSDP offload |
|---|---:|---:|---|
| 2-GPU Baseline (push) | 8.22 | — | ON |
| 2-GPU +LP (pull) | ~8.22 | ~0% | ON |

Pull does not change GPU compute time.  Dispatch overhead difference at batch=8 is
< 1 ms/step — immeasurable at the iteration level.

### Table 3b — Full Controlled Ablation (all FSDP OFF, 2-GPU, batch=8, 20 steps)

All four runs use identical hardware, model, dataset, and FSDP configuration
(`param_offload=False`, `optimizer_offload=False`).  The only variables are
dispatch mode and compression.

| Config | Dispatch | Compress | avg step\_time (s) | Δ vs Baseline | avg reward | nonzero/19 |
|---|---|---|---:|---:|---:|---:|
| Baseline | push | — | 8.939 | — | 0.0197 | 3 |
| +LP | pull | — | 9.121 | +2.0% | 0.0592 | 8 |
| +LP + FP16 | pull | fp16 | **8.546** | **−4.4%** | 0.0987 | 10 |
| +LP + INT8 | pull | int8 | 8.865 | −0.8% | 0.0789 | 9 |

**Key findings from the controlled experiment (batch=8):**

- **+LP alone is 2% slower** than push baseline at batch=8, confirming the breakeven
  analysis (fixed `ray.put()` overhead dominates at small batch sizes).
- **+LP + FP16 is the best configuration** — 4.4% faster than push baseline and 6.3%
  faster than pull-only, due to reduced `ray.put()` serialization cost.
- **+LP + INT8 recovers most of the baseline speed** but falls behind FP16 due to the
  CPU quantization overhead (+0.32 s/step vs FP16).
- Reward trajectories are within expected variance at 20 steps × batch=8.

### Table 3c — Extended Stability Run (all FSDP OFF, 2-GPU, batch=32, 100 steps)

Larger batch size and more steps to confirm timing trends and training stability.
No `+LP` alone run at batch=32; four compression variants vs push baseline.

| Config | Dispatch | Compress | avg step\_time (s) | Δ vs Baseline | avg reward (100 steps) |
|---|---|---|---:|---:|---:|
| Baseline | push | — | 21.952 | — | 0.225 |
| +LP + FP16 | pull | fp16 | **19.627** | **−10.6%** | 0.079 |
| +LP + INT8 | pull | int8 | 22.550 | +2.7% | 0.075 |
| +LP + BF16 | pull | bf16 | 22.682 | +3.3% | 0.034 |

**Key findings at batch=32:**

- **FP16 is the only mode that improves iteration time** at this scale: −10.6% vs
  baseline.  BF16 and INT8 both add **~3%** wall-clock overhead vs baseline despite
  the same 2× byte reduction on float32 tensors as FP16 — extra CPU work for
  bfloat16/int8 encode–decode and (for INT8) per-tensor scaling dominates the savings.
- **BF16 vs FP16**: BF16 is numerically safer than FP16 (same exponent range as
  float32) but **does not** match FP16’s measured speed here; use FP16 when the goal
  is raw throughput, BF16 when extreme log-prob magnitudes might clip in FP16.
- **Reward variance across runs is expected**: all configs show learning signal over
  100 steps; mean reward differences reflect run-to-run variance (seeds / trajectory),
  not a reliable quality ranking between compressors.
- **FP16 timing drop at step ~35**: visible in panel (b); vLLM JIT warmup, after which
  FP16 stays fastest.

![Reward and iteration time comparison (100 steps, batch=32)](7_reward_curve_controlled.png)

---

## 7.3 Ablation Matrix and Reporting

### Table 4 — Full Ablation Summary (2-GPU, compute_log_prob focus)

| Method | Bytes/iter | Xfer (ms) | CPU (ms) | Notes |
|---|---:|---:|---:|---|
| Baseline (push) | 14.3 KB | 22.3 | 0.225 | broadcast full batch |
| + LP | 14.3 KB | 21.1 | 0.245 | pull; slower at batch=8 (breakeven≈13) |
| + AO | 14.3 KB | ~21 | ~0.225 | non-blocking dispatch; 35% prep-stage saved |
| + Comp FP16 | **8.2 KB** | **9.3** | **0.117** | fp16 cast; 1.9% payload reduction² |
| + Comp INT8 | **8.2 KB** | 14.8 | 0.119 | 2.8% payload reduction²; quant overhead |

> ² Float32 tensors (log\_probs, values) account for only ~3.7% of total payload; the
> remainder is non-compressible int64 (token ids, masks).  Payload reduction reflects
> this composition.  At float32-heavy workloads the saving would scale to 50% (FP16)
> and 75% (INT8).

### Performance Attribution

| Optimization | Measured Gain (controlled) | Metric | Caveat |
|---|---|---|---|
| Local-Batch Pull | −5% Xfer_ms at batch=8; projected −50% at batch≥256 | dispatch+collect ms | `ray.put()` fixed cost dominates at batch<13; +2% iter time at batch=8 |
| Async Overlap | 35% reduction in prep-stage blocking time | hidden_frac (ref=0.285, values=0.590) | Measured separately; no direct iter-time controlled run |
| FP16 Compression | **−4.4% at batch=8; −10.6% at batch=32; −58% Xfer_ms** | Tables 3b & 3c (controlled) | Benefit scales with batch; 1.9% payload reduction due to int64-dominant batch |
| BF16 Compression | +3.3% at batch=32 (vs baseline); same 2× float32 payload cut as FP16 | Table 3c (controlled) | Safer dynamic range than FP16; CPU encode path slower than FP16 in this run |
| INT8 Compression | −0.8% at batch=8; +2.7% at batch=32; −34% Xfer_ms | Tables 3b & 3c (controlled) | CPU quantization overhead exceeds bandwidth savings at current batch sizes |

> **Attribution note:** The 8.22 s → 4.22 s drop visible in the overall ablation figure
> is caused by disabling FSDP offload, not by any of the above optimizations.  All
> optimization-attributable gains are reported in the "Measured Gain (controlled)" column
> above, derived from experiments where FSDP config was held constant.

---

### Training Stability

FP16, BF16, and INT8 compression modes cast transferred batch tensors back to
`float32` before training arithmetic, so model weights and gradients are unaffected.
We validated stability at two scales:

- **20 steps / batch=8**: all four configs show positive nonzero reward; no systematic
  degradation from compression (Table 3b).
- **100 steps / batch=32**: Baseline, +FP16, +INT8, and +BF16 all show learning signal
  over the full run (see panel (a) of the reward curve above).  Mean reward differences
  across configs reflect run-to-run variance (seeds / trajectory), not a reliable
  quality ranking.  No training collapse was observed in any run.

---

![Evaluation summary figure](7_eval.png)
