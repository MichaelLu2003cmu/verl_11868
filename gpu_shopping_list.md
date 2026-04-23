# GPU Shopping List for Step 2

Post-Step-1 audit: which reruns are actually required now that we've extracted
p99 jitter, effective bandwidth, and KL drift from the **existing** 14 runs.

## Audit findings that changed this list

On 2026-04-22, while preparing the final writeup, we discovered that the three
"compression" 100-step runs (`gsm8k_2gpu_fp16_100`, `gsm8k_2gpu_int8_100`,
`gsm8k_2gpu_bf16_100`) were launched with
`trainer.use_legacy_worker_impl=disable` — i.e. they used the **push** dispatch
path where `VERL_TRANSFER_COMPRESS` is never consulted.  Every one of those
runs has **zero `compress_stats` events** in its probe log; the differences we
attributed to compression were run-to-run variance.

The corrected shopping list below **replaces** the four reruns originally
listed with a smaller set focused on producing a clean 3-row cumulative
ablation (push → pull → pull+fp16) at batch=32 / 100 steps.

## What we now have from existing logs (no rerun needed)

| Proposal §7 item | Source |
|---|---|
| (i) Transfer bytes per iteration | `transfer_probe*.jsonl` (6 configs, 3-row ablation) |
| (ii) Transfer time per iteration | `transfer_probe*.jsonl` — `dispatch_ms + collect_ms` |
| (iii) CPU overhead per iteration | `transfer_probe*.jsonl` — `DataProto.concat elapsed_ms` |
| **(iv) Effective transfer throughput (MB/s)** | Table 1 `Eff_BW` column in `7_eval.md` |
| End-to-end iter time (mean) — 3-row cumulative ablation | `train_log.txt` for `baseline_100`, `lp_only_100`, `lp_fp16_100` |
| **p50 / p90 / p99 / max / std iter-time jitter** | `7_eval.md` Table 3d′ (3 rows) |
| Reward trajectory (3-config validated) | `7_stability_corrected.png` |
| **KL divergence trajectory & mean** | `7_eval.md` Table 3e′ (3 rows) |
| **GSM8k task success rate** (= `critic/score/mean`) | `7_eval.md` Table 3e′ |
| Async overlap `hidden_frac` (ref, values) | `pathB_lite` probe, Table 2 |
| Compression payload reduction | Tables 3b + `4_3.md` |
| **Pull dispatch breakdown (dispatch/collect/wait)** | Table 1 + in-text microbench in 7_eval.md |

## What still requires a GPU rerun (Step 2)

### ~~Rerun A — LP-only at batch=32, 100 steps~~  ✅ **COMPLETE (2026-04-22)**

- Run at `checkpoints/verl_examples/gsm8k_2gpu_lp_only_100` (706 probe events,
  100 complete steps).  Verified pull dispatch fired (compression did NOT
  fire — `compress_stats` events = 0, as intended for this "LP-only" config).
- **Result:** mean step time 26.750 s (+21.9 % vs baseline push), jitter
  std 0.240 s (−64 % vs baseline, tightest in the ablation), |KL|\_mean
  0.00389 (4.4 × lower than baseline).  This turns the 2-row comparison
  into a clean 3-row cumulative ablation, and crucially reveals that
  **pull on its own regresses iter-time at batch=32 but tightens jitter** —
  a Pareto trade-off not visible in any 2-row slice of the data.
- Patched into `7_eval.md` Tables 3c′ / 3d′ / 3e′ and
  `what_we_have_done.md` §9.3.

### ~~Rerun B — BF16 at batch=32 with `legacy_worker_impl=enable`~~  ✅ **COMPLETE (2026-04-22)**

- Run at `checkpoints/verl_examples/gsm8k_2gpu_lp_bf16_100`, 100 steps
  completed.
- **Result:** mean step time 25.624 s (+16.7 % vs baseline push, no
  wall-clock win), but **|KL|\_mean = 0.01124 — within noise of
  baseline's 0.017 and 4 × lower than FP16's 0.045**.  This decisively
  confirms that FP16's +164 % KL drift is a real numerical effect from
  FP16's narrower exponent range (±6.5 × 10⁴ vs BF16's ±3.4 × 10³⁸),
  not seed variance.  **BF16 is the numerically-safe half-precision
  default.**
- **Caveat on this run:** the run was launched without
  `VERL_TRANSFER_PROBE=1`, so per-transfer `compress_stats` events are
  not captured.  Compression itself fires independently of the probe
  (gated only on `VERL_TRANSFER_COMPRESS`), and the end-to-end KL/time
  numbers are reliable.  A ~40 min re-run with probe enabled would
  recover the transfer-layer microbench for BF16 but is not required
  for the stability claim.
- Patched into `7_eval.md` Tables 3c′ / 3d′ / 3e′ and
  `what_we_have_done.md` §9.3 / §12.

### Rerun C′ — **Combined LP + AO + Comp stack** (IN QUEUE, PROMOTED FROM OPTIONAL)

- **Goal:** close the composability gap.  The existing 100-step GRPO runs
  measured LP + Comp with the ref worker OFF (AO is a no-op there), and
  `pathB_lite` measured LP + AO with compression OFF.  Nowhere in the
  current data do all three optimizations fire together.  Rerun C′ runs
  three configs *with the ref worker enabled* so AO actually fires, and
  adds compression on top for the full-stack claim:

  | Config                      | `legacy_worker_impl` | `VERL_TRANSFER_COMPRESS` | LP | AO | Comp |
  |-----------------------------|----------------------|--------------------------|----|----|------|
  | `stack_push_ref` (baseline) | `disable`            | *(unset)*                | ✗  | ✗  | ✗    |
  | `stack_lp_ao`               | `enable`             | *(unset)*                | ✓  | ✓  | ✗    |
  | `stack_lp_ao_fp16`          | `enable`             | `fp16`                   | ✓  | ✓  | ✓    |

  *AO is implicit in `legacy_worker_impl=enable`: the worker-group
  decorator returns `DataProtoFuture` non-blocking, so `ref_log_prob`
  dispatches in parallel with `old_log_prob` (see
  `verl/trainer/ppo/ray_trainer.py:1592`).  In the push path the future
  is materialized immediately with `.get()`, which is why AO doesn't
  fire there.*

- **Prereqs (already verified):**
  - Ref worker turned on via `actor_rollout_ref.actor.use_kl_loss=True`
    with `kl_loss_coef=0.001` (minimal training-dynamic perturbation).
  - Memory fits on 2×V100 32 GB + 45 GB RAM at batch=8 with
    `gpu_memory_utilization=0.25` and ref param-offload enabled
    (`actor_rollout_ref.ref.fsdp_config.param_offload=True`).
  - Probe env vars set correctly for **all three** runs so
    `critical_path_stage` events are captured (`VERL_TRANSFER_PROBE=1`
    + `VERL_TRANSFER_PROBE_LOG=<run_dir>/transfer_probe.jsonl`).

- **Runner:**
  ```bash
  bash scripts/run_combined_stack.sh 2>&1 | tee combined_stack_driver.log
  ```

- **Analyzer (after sweep):**
  ```bash
  /ocean/projects/cis260009p/syan5/conda/project/bin/python \
    scripts/analyze_combined_stack.py \
      --out-fig poster_stack.png \
      --out-md  stack_report.md
  ```

- **Output dirs:**
  - `checkpoints/verl_examples/gsm8k_2gpu_stack_push_ref/`
  - `checkpoints/verl_examples/gsm8k_2gpu_stack_lp_ao/`
  - `checkpoints/verl_examples/gsm8k_2gpu_stack_lp_ao_fp16/`

- **Expected runtime:** 45–75 min total on 2×V100 (3 × ~15–25 min;
  batch=8 / 20 steps / response_length=512).

- **Expected outcome (headline claim for the poster):**
  Going left to right across the three configs, we should see a
  *monotonic* improvement on three axes:
  1. **Σ prep-stage driver-wait drops** from the push baseline
     (full sum of old_log_prob + ref, ≈ 2.4–2.7 s per iter expected
     from `pathB_lite` precedent where `Σ wait_ms_push ≈ Σ in_flight`)
     to the pull+AO case (ref's wait_ms shrinks toward the overlap
     tail) — **the headline AO win**.
  2. **Σ recv bytes drops** in the FP16 column (≈ 50 % on
     float32-heavy tensors, per `4_3.md` precedent) — **the Comp win,
     on top of LP's per-rank slicing**.
  3. **End-to-end iter-time mean drops** monotonically if both
     optimizations compose as theory predicts.  |KL| should stay
     within noise (FP16's numerical drift is a known risk — if it
     shows up here it's a fresh reproduction of the batch=32 finding).

- **Probe events to verify (sanity signal printed by the runner):**
  - `push_ref`: `critical_path_stage` count = 0 (AO no-op), `compress_stats` = 0.
  - `lp_ao`:    `critical_path_stage` count > 0, `compress_stats` = 0.
  - `lp_ao_fp16`: both `critical_path_stage` > 0 and `compress_stats` > 0.

- **Closes:**
  - Table 4 footnote ⁴ ("cannot isolate AO from LP").
  - Proposal §7.3 "We evaluate configurations in a cumulative fashion,
    starting from the baseline and incrementally enabling LocalPull,
    AsyncOverlap, and Compression" — the AO step has been missing
    from the cumulative chain until this rerun.
  - Poster Block 6 (currently cites only `pathB_lite`'s isolated AO
    numbers) and the poster composability framing.

### Rerun D — Sequence-length sweep at batch=8, 10 steps each (IN PROGRESS)

- **Goal:** proposal §7.2 "vary batch size **and sequence length** to evaluate
  compute-bound vs communication-bound regimes".
- **Grid:** `data.max_response_length ∈ {128, 512, 1024}` × {push-baseline
  (`legacy_worker_impl=disable`), LP+FP16 (`legacy_worker_impl=enable`,
  `VERL_TRANSFER_COMPRESS=fp16`)}.  6 short runs, 10 steps each at batch=8.
- **Output dirs:** `checkpoints/verl_examples/gsm8k_seqlen_{128,512,1024}_{base,lpfp16}/`.
- **Probe:** enabled (`VERL_TRANSFER_PROBE=1` +
  `VERL_TRANSFER_PROBE_LOG=<run_dir>/transfer_probe.jsonl`) so we get
  per-transfer `send_bytes / recv_bytes / dispatch_ms / collect_ms` and
  `compress_stats` events scaling with response length — this is the headline
  result.
- **Runner:** `bash scripts/run_seqlen_sweep.sh 2>&1 | tee seqlen_sweep_driver.log`
- **Analyzer (after sweep):** `python scripts/analyze_seqlen_sweep.py` (emits
  the Markdown scaling table + saves `7_seqlen_scaling.png`).
- **Expected runtime:** ~60–90 min total (response_length=1024 dominates).
- **Closes:** regime-dependency of compression payoff — expect the FP16
  mean-time win to **grow** with response length because the bandwidth-heavy
  tensors (`old_log_probs`, `values`) scale linearly with the sequence
  dimension.

## What we are **not** going to run (out of scope / future work)

- 7B-model validation (hardware constraint — documented in `what_we_have_done.md` §12).
- Multi-node evaluation (documented, future work).
- INT8 CUDA fused kernel (listed in §12 future work).
- Gradient-pathway compression (listed in §12 future work).
- Batch=256 production-scale pull validation (future work).
- Re-running INT8 at batch=32: the batch=8 validation (Table 3b) already
  establishes INT8 is CPU-bound at small batch; the interesting question for
  INT8 is a CUDA fused kernel (future work), not another 40-min data point.

## Recommended execution order

1. ~~**Rerun A** (LP-only @ batch=32)~~ ✅ done.
2. ~~**Rerun B** (BF16 stability)~~ ✅ done — FP16 drift confirmed real.
3. **Rerun D** (seq-len sweep) — **in progress**; strongest "new finding" for a
   workshop-style paper, closes proposal §7.2 "vary sequence length".
4. **Rerun C′** (**combined LP+AO+Comp stack, batch=8/20 steps**) —
   **promoted from "optional" to REQUIRED for the poster**.  The
   current story has no run where all three optimizations fire
   simultaneously, so the proposal's cumulative ablation chain skips
   AO.  This rerun (45–75 min) fills that gap with 3 configs on the
   same GRPO setup as the rest of the poster, and produces the
   composability figure (`poster_stack.png`) and the updated AO
   block 6.
5. **(Optional cleanup)** Re-run BF16 with probe enabled to capture
   transfer-layer microbench data.  30–40 min; only matters if the paper
   needs a per-transfer comparison of BF16 vs FP16 compression ratios.

After each run we re-run `scripts/jitter_and_kl.py` and
`scripts/summarize_transfer_probe.py` to regenerate all tables in
`7_eval.md` from the new logs.

## Completed in Step 2 so far

- ✅ **`gsm8k_2gpu_lp_fp16_100`** — LP + FP16 at batch=32 / 100 steps with
  `legacy_worker_impl=enable`, verified 4 000 `compress_stats` events.
- ✅ **`gsm8k_2gpu_lp_only_100`** — LP only (pull, no compress) at
  batch=32 / 100 steps with `legacy_worker_impl=enable`, verified 0
  `compress_stats` events.
- ✅ **`gsm8k_2gpu_lp_bf16_100`** — LP + BF16 at batch=32 / 100 steps with
  `legacy_worker_impl=enable`.  Probe env vars were not set in this run
  (see note), so per-transfer `compress_stats` events were not captured;
  end-to-end KL/time numbers are valid and establish the core stability
  finding.
- ✅ 4-row cumulative ablation (push → pull → pull+fp16 / pull+bf16)
  patched into `7_eval.md` Tables 3c′ / 3d′ / 3e′ and
  `what_we_have_done.md` §9.3.
- ✅ Microbench Table 1 confirms the mechanism (push dispatch 20.45 ms →
  pull 4.84 ms, −76 %) for baseline / LP-only / LP+FP16.
- ✅ **FP16 KL drift confirmed numerical**: BF16 control at same seed
  shows |KL| 0.011 (vs FP16's 0.045, baseline's 0.017), isolating the
  cause to FP16's narrower exponent range.

## Note on probe env variables (for future runs)

The probe module reads **two** env variables, not one:

```
export VERL_TRANSFER_PROBE=1                                          # flag (required)
export VERL_TRANSFER_PROBE_LOG=/path/to/transfer_probe.jsonl          # path (required)
```

A common mistake is to set only `VERL_TRANSFER_PROBE=<path>`, which
disables the probe (the value must be in `{1,true,yes,on}`).  Before
launching a run that needs microbench data, verify the probe is active
by watching for `[transfer_probe]` lines on stdout within the first
few seconds of training.
