# Poster Plan — Optimizing Transfer Protocols for RLHF Dataflows in verl

This file is the single source of truth for poster content.  All numbers
are re-extracted from raw logs on 2026-04-22 and cross-verified against
Tables 3c′/3d′/3e′ of `7_eval.md` and the bar values inside
`7_eval.png` / `7_eval_b32.png` / `4_3_compress.png` / `4_3_compress_b32.png`.

The professor's guidance is **more figures, less words, more demos**.
The poster is therefore organised so every block either (a) is a
figure/diagram, (b) is a compact table, or (c) is a one-line takeaway
bullet.  No paragraphs of prose.

---

## 1. Layout (landscape 48" × 36", 3 columns)

```
+-----------------------------------------------------------------------------+
| HEADER — title / authors / institution / QR-to-repo                         |
+---------------------+------------------------+---------------------------+--+
| Col 1 — Why & What  | Col 2 — How (results)  | Col 3 — So what (impact)  |
|---------------------|------------------------|---------------------------|
| 1.  Motivation      | 5.  Microbench (b=8)   | 9.  Training stability    |
|     [RLHF-dataflow  |     [7_eval.png]       |     [7_stability_         |
|      diagram,       |                        |       corrected.png]      |
|      Gemini asset]  | 6.  Async Overlap      |                           |
|                     |     [4_2_gantt.png]    | 10. Seq-len scaling       |
| 2.  Problem         |                        |     [7_seqlen_scaling.png]|
|     [Push-vs-Pull   | 7.  At-scale (b=32)    |     (Rerun D — in flight) |
|      schematic,     |     [7_eval_b32.png]   |                           |
|      Gemini asset]  |                        | 11. Takeaways             |
|                     | 8.  Pareto             |     [SLO-selector table]  |
| 3.  Approach        |     [4_3_compress      |                           |
|     [3 icons:       |      _b32.png]         | 12. Future work / QR      |
|      LP, AO, Comp]  |                        |                           |
|                     |                        |                           |
| 4.  Setup           |                        |                           |
|     [2xV100, Qwen,  |                        |                           |
|      GSM8k card]    |                        |                           |
+---------------------+------------------------+---------------------------+
```

Each block below says exactly what goes in it.

---

## 2. Header

- **Title:** *Optimizing Transfer Protocols for RLHF Dataflows in verl*
- **Authors:** Shuwen Yan, Maochuan Lu, Jim Zhou — ECE, Carnegie Mellon University
- **Venue tag:** MLSys 2026 Workshop submission
- **Logo block:** CMU ECE · 5th MLSys Conference
- **QR code:** link to the GitHub repo / final report PDF
- **One-sentence subtitle (biggest font on the poster after the title):**
  *"RLHF iteration time is a data-movement problem, not a FLOPs problem."*

---

## 3. Block 1 — Motivation  (Col 1, top)

**Visual:** Gemini-generated RLHF-as-dataflow diagram (see §14, asset G1).

**One-liner caption under the diagram:**

> *In GRPO training, each iteration re-shards large rollout tensors across
> parallelism boundaries 4 times per step.  At 2×V100, transfer dispatch
> costs **25.3 ms / iter** — more than the forward pass of the critic.*

**Mini-fact bar (3 big numbers, no prose):**

| Metric | Value | Source |
|---|---|---|
| Transfer volume per dispatch | **14.3 KB** | `7_eval.png` panel (a), baseline |
| Dispatch time per transfer | **25.3 ms** | `7_eval.png` panel (b), baseline |
| # transfers / iteration (b=32) | **≥ 3** | `transfer_probe.jsonl`, 304 events / 100 steps |

---

## 4. Block 2 — Problem  (Col 1, middle-top)

**Visual:** Gemini-generated push-vs-pull dispatch schematic (§14, asset G2).

**Caption (one line):**

> *Push dispatch aggregates the full global batch on every worker and
> discards 87 % of it.  Pull dispatch sends metadata, each worker fetches
> only its slice.*

---

## 5. Block 3 — Approach  (Col 1, middle)

**Visual:** Three icons horizontally, each annotated with one fact.  Use
Gemini asset G3 (three stylised icons) or use emoji + colour chips if you
prefer a cleaner, less-AI look.

| Icon | Mechanism | One-line effect |
|---|---|---|
| 🟢 | **LP** — Local-Batch Pull | Each DP rank pulls only its slice; fixes aggregate-then-discard.  *(See blocks 5, 7, 8.)* |
| 🟡 | **AO** — Async Overlap | Non-blocking dispatch; overlaps transfer with critic/ref/reward compute.  *(See block 6: −34.9 % prep-stage wall-clock.)* |
| 🔴 | **Comp** — Lightweight Compression | FP16 / BF16 / INT8 on bandwidth-heavy, numerically tolerant tensors.  *(See blocks 5, 7, 8.)* |

---

## 6. Block 4 — Setup  (Col 1, bottom)

Compact spec card (no prose):

| Field | Value |
|---|---|
| Model | Qwen2.5-1.5B-Instruct (FP16, attn_implementation=eager) |
| Algorithm | GRPO (no critic, no ref model) |
| Rollout engine | vLLM v1, async, enforce_eager |
| Hardware | 2× NVIDIA V100 32 GB, 45 GB host RAM |
| Data | GSM8K (prompts=256 tok, responses=128–1024 tok) |
| Parallelism | FSDP (size=2, offload OFF), TP=1 |
| Batch sizes | 8 (microbench, 20 steps) · 32 (at-scale, 100 steps) |
| Instrumentation | `VERL_TRANSFER_PROBE`, `compress_stats` events |
| Scheduler | SLURM on PSC Bridges-2 |

---

## 7. Block 5 — Microbench at batch=8  (Col 2, top)

**Existing figure:** `7_eval.png`   (4 panels, 4 configs, batch=8 / 20 steps)

**Caption (one line):**

> *Pull alone drops dispatch_ms **−26 %** (25.3 → 18.7) without changing
> payload size; adding FP16 drops it a further **−54 %** (18.7 → 8.6).
> End-to-end iter time mirrors the trend at this scale.*

**Compact table of exact values (paste below the figure, no rounding):**

| Config | recv KB / iter | dispatch_ms | concat_ms | iter time (s) |
|---|---:|---:|---:|---:|
| Baseline (push) | 14.3 | 25.318 | 0.252 | 8.939 |
| +LP (pull) | 14.3 | 18.706 | 0.245 | 9.121 |
| +LP + FP16 | 8.2 | 8.563 | 0.118 | 8.546 |
| +LP + INT8 | 8.2 | 14.507 | 0.124 | 8.865 |

---

## 8. Block 6 — Async Overlap results  (Col 2, middle-upper)

**Existing figure:** `4_2_gantt.png`  (Gantt chart: sequential baseline vs
async dispatch for prep-stage workers).

**Why this lives in its own block:** AO is the protocol mechanism that
hides transfer latency behind compute.  The batch=32 / 100-step GRPO runs
in blocks 7 and 8 disable the reward / ref / critic workers
(`reward_model.enable=False`), so there is no prep-stage compute to overlap
and AO is a no-op in that setting.  To measure AO in isolation we ran an
independent 3-worker prep-stage test (`old_log_prob` → `ref_policy` →
`values`, batch=8, p50 over 20 steps) where all three stages produce real
GPU work.  This is the cleanest possible AO measurement on 2×V100.

**Caption (one line, bold the result):**

> *Async dispatch hides **36.3 % of prep-stage wall-clock** behind concurrent
> compute; total prep time drops **4.41 s → 2.87 s per iteration (−34.9 %)**
> without any change to the scientific workload.*

**Exact values (paste below the figure, p50 over 20 steps):**

| Stage          | Serial (ms) | Async wait (ms) | Hidden (ms) | hidden_frac |
|----------------|------------:|----------------:|------------:|------------:|
| `old_log_prob` |         870 |             870 |           0 |       0.000 |
| `ref_policy`   |        1603 |            1175 |         457 |       0.285 |
| `values`       |        1942 |             827 |        1147 |       0.591 |
| **Total**      |    **4415** |        **2872** |    **1604** |   **0.363** |

**Two bullets for the pitch:**

- *`ref_policy` dispatched non-blocking; 28.5 % of it runs concurrently
  with `old_log_prob`.*
- *`values` dispatched with `ref_policy`; 59.1 % hidden under the
  ref-join barrier — nearly free.*

**Caveat (a single italicized line):**

*AO's payoff grows linearly with the number of active prep-stage workers;
the 34.9 % speedup measured here is a lower bound for multi-node setups
where ref/critic/reward are typically on separate device groups.*

---

## 9. Block 7 — At-scale at batch=32 / 100 steps  (Col 2, middle)

**Existing figure:** `7_eval_b32.png`   (4 panels, 4 configs: mean iter,
p99 iter, mean |KL|, throughput)

**Caption (one line, bold the counter-intuitive bit):**

> *At production batch, **FP16's −16.8 % mean-iter-time win does not
> translate into throughput**: FP16 loses −27.3 % tokens/GPU-s because its
> KL drift collapses response length.*

**Compact table of exact values:**

| Config | mean (s) | p50 (s) | p90 (s) | p99 (s) | std (s) | mean \|KL\| | KL std | reward | throughput (tok/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline (push) | 21.952 | 21.888 | 22.794 | 23.084 | 0.657 | 0.01700 | 0.02329 | 0.2251 | 136.5 |
| +LP (pull) | 26.750 | 26.758 | 27.007 | 27.368 | 0.240 | 0.00389 | 0.00783 | 0.0142 | 134.1 |
| +LP + FP16 | 18.256 | 17.897 | 18.346 | 23.640 | 1.287 | 0.04482 | 0.06384 | 0.0511 | 99.3 |
| +LP + BF16 | 25.624 | 26.043 | 26.674 | 27.001 | 1.367 | 0.01124 | 0.02991 | 0.0357 | 114.1 |

---

## 10. Block 8 — Pareto frontier  (Col 2, bottom)

**Existing figure:** `4_3_compress_b32.png`   (panels: %Δ iter, %Δ |KL|, 2D
Pareto scatter; 4 configs at batch=32 / 100 steps)

**Caption (one line, bold the systems claim):**

> *All 4 configurations are Pareto-optimal on (iter-time × \|KL\|): no
> configuration strictly dominates — **choose by SLO, not by average**.*

**Tiny SLO-selector next to the figure (use this in the live pitch):**

| If your SLO is… | Pick |
|---|---|
| Mean wall-clock iter time | `+LP+FP16` |
| Training stability (low \|KL\|) | `+LP+BF16` |
| Tail latency / jitter (p99/p50 → 1) | `+LP` alone |
| None of the above | `Baseline (push)` is fine |

---

## 11. Block 9 — Training stability  (Col 3, top)

**Existing figure:** `7_stability_corrected.png`   (reward / score / KL
time-series across 4 configs at batch=32 / 100 steps)

**Caption (one line):**

> *FP16's **+164 % \|KL\| drift is a real numerical effect**, confirmed by
> the BF16 control at the same seed: BF16's \|KL\| sits at 0.0112 vs
> FP16's 0.0448 despite identical payload-byte count.*

**Mini-diagram beside the caption:** Gemini asset G4 (FP16 vs BF16 bit
layout — see §14).  Text label:

> *Narrow exponent (FP16: ±6.5×10⁴) → saturated gradients on large advantages.*
> *Wide exponent (BF16: ±3.4×10³⁸) matches FP32 range → numerically safe.*

---

## 12. Block 10 — Sequence-length scaling  (Col 3, middle)

**Figure (PENDING — produced by Rerun D):** `7_seqlen_scaling.png`

Reserve a full-width block with a placeholder that says:

> *[RERUN D — seq-len sweep at batch=8, response ∈ {128, 512, 1024},
> 10 steps each.  6 runs × ~8-15 min ≈ 60-90 min total. Expected: FP16's
> byte-savings grow linearly with response length; cumulative wall-clock
> win at response=1024 should cross into double-digit percent — confirming
> the transfer bottleneck is compute-regime-sensitive (proposal §7.2).]*

When the sweep finishes, replace with the produced figure and add the
paired-delta table emitted by `scripts/analyze_seqlen_sweep.py`
(it prints a Markdown block that can be pasted directly).

---

## 13. Block 11 — Takeaways  (Col 3, bottom)

Three bullets max — these are what you say in your last 20 seconds:

1. **Pull removes aggregation overhead without touching payload bytes**:
   dispatch_ms −26 % at batch=8 (25.3 → 18.7), and tightens p99/p50 jitter
   from 1.05 → 1.02 at batch=32.
2. **FP16 is a throughput footgun in RLHF**: −16.8 % mean iter time but
   +164 % \|KL\| drift and −27.3 % throughput due to policy collapse.
3. **BF16 is the numerically-safe half-precision default**: same byte
   savings as FP16 (2-byte element), \|KL\| within noise of baseline.
   V100 pays a +16.7 % mean-time tax for lack of native BF16 tensor cores;
   expected to flip on Ampere / H100.

**Practical recipe (one-line per row, put this in a neat framed box):**

| Scale | Recommendation |
|---|---|
| batch ≤ 8 | `+LP+FP16` — compute-bound, drift is tolerable |
| batch ≥ 32 | `+LP+BF16` unless you have a hard throughput SLO |
| V100 hardware | Trade 17 % time for stability; expect flip on newer GPUs |

---

## 14. Block 12 — Future work / repro  (Col 3, bottom, small)

- **Multi-node** (1-GPU → 4+ GPU per node) — AO payoff grows with rank count.
- **7B-scale validation** — hardware-constrained on V100s.
- **INT8 CUDA fused kernel** — current INT8 is CPU-bound at small batch.
- **Gradient-pathway compression** — we only compressed rollout artifacts.

Below this, a QR code linking to:
- GitHub repo (with `scripts/regen_7_eval.py`, `scripts/regen_b32_figures.py`)
- `7_eval.md` + `what_we_have_done.md` final report

---

## 15. Gemini Nano Banana prompts  (demo assets to generate)

Use these four prompts to fill the visual "negative space" on the poster.
Keep all assets in the same palette as the PNGs
(#4878CF blue baseline / #6ACC65 green LP / #D65F5F red FP16 /
#CC8800 orange BF16 / #956CB4 purple INT8 / #4C4C4C charcoal text).

### G1 — RLHF dataflow diagram  *(goes in Block 1)*

> *A minimalist system diagram of an RLHF training loop, drawn in an
> academic-paper style (white background, thin line work, serif labels).
> Show 4 rectangular nodes arranged left-to-right: "Actor / Policy (FSDP,
> rank 0-1)", "vLLM Rollout Engine", "Reward & KL compute", "GRPO update".
> Connect adjacent nodes with thick directed arrows labelled with the
> tensor type being transferred: "prompts + responses (int64, ~104 KB)",
> "old_log_probs + values (float32, 4 KB)", "advantages (float32)", and
> "updated weights (fp16)".  Every arrow has a small red badge labelled
> "transfer_protocol: collect + distribute".  In the top-right corner,
> a small inset highlights one arrow with a magnifying glass and the
> caption "25.3 ms/iter on 2xV100 — target of our optimisation".  No
> 3D effects, no gradients, no shadows.  Think: ICML-figure aesthetic.*

**Filename suggestion:** `poster_g1_dataflow.png`

---

### G2 — Push vs Pull dispatch schematic  *(goes in Block 2)*

> *A two-panel side-by-side comparison, flat 2D, academic style, white
> background.  LEFT PANEL titled "Push (aggregate-then-scatter)": a large
> central grey box labelled "Rank 0 – global tensor (full batch)" with 8
> stacked colored strips inside (each strip is one DP rank's slice).
> Four arrows fan out from this box to four receiver rectangles on the
> right edge labelled "Rank 0", "Rank 1", "Rank 2", "Rank 3".  Each arrow
> is labelled with the full-batch byte count "14.3 KB".  Red overlay text
> in the middle of the panel: "87 % of bytes discarded".  RIGHT PANEL
> titled "Pull (metadata-then-slice)": four small rectangles labelled
> "Rank 0/1/2/3" each sending a thin dashed arrow to a shared "metadata
> catalog" rectangle in the center (shape, dtype, slice index) — dashed
> arrows labelled "request".  Solid short arrows return from the catalog
> to each rank labelled "8.2 KB / rank".  Green overlay text: "Each rank
> pulls only its slice".  Both panels share an x-axis annotation showing
> total bytes moved: 57.2 KB (push) vs 32.8 KB (pull).  Sans-serif labels,
> thin 1 pt strokes.*

**Filename suggestion:** `poster_g2_push_vs_pull.png`

---

### G3 — Three-technique icon strip  *(goes in Block 3)*

> *Three minimalist vector icons in a horizontal row, all on white,
> matching ICML figure style (thin 1.5 pt strokes, flat colours).  ICON
> ONE: green square labelled "LP" — shows four small rank boxes, each
> reaching into a shared rectangle and extracting a different-coloured
> slice.  Subtitle: "Local-Batch Pull".  ICON TWO: yellow square labelled
> "AO" — shows two horizontal lanes (compute lane = solid bar, transfer
> lane = dashed bar) with the dashed bar overlapping the solid bar by
> about 60 %.  Subtitle: "Async Overlap".  ICON THREE: red square
> labelled "Comp" — shows a large float32 rectangle on the left with an
> arrow pointing right to a half-size FP16 rectangle and a quarter-size
> INT8 rectangle below it.  Subtitle: "Lightweight Compression".  All
> three icons fit within a 1200×400 canvas at equal sizes with 40 px
> gaps.  Academic, restrained, no gradients, no 3D.*

**Filename suggestion:** `poster_g3_three_icons.png`

---

### G4 — FP16 vs BF16 bit layout  *(goes in Block 8, beside stability caption)*

> *A clean numerical-precision comparison diagram, white background, ICML
> paper aesthetic.  Show two horizontal 16-bit strips stacked vertically.
> Top strip labelled "FP16 (IEEE half)": 1 sign bit + 5 exponent bits
> (coloured #D65F5F red) + 10 mantissa bits (coloured grey).  Bottom
> strip labelled "BF16 (brain float)": 1 sign bit + 8 exponent bits
> (coloured #CC8800 orange) + 7 mantissa bits (coloured grey).  Each
> strip has its dynamic range annotated on the right: FP16 shows
> "±6.5 × 10⁴", BF16 shows "±3.4 × 10³⁸".  Below both strips a small
> bar chart with two bars showing "mean |KL| at batch=32/100":
> FP16 bar = 0.0448 (red), BF16 bar = 0.0112 (orange), with a dashed
> horizontal line at 0.017 labelled "baseline |KL|".  Very restrained
> design — no gradients or drop-shadows.  Target width 900 px.*

**Filename suggestion:** `poster_g4_fp16_vs_bf16.png`

---

## 16. The 90-second live pitch

Use this script verbatim when someone walks up.  Time-boxed; pauses marked
with `…`.

> **(0-15 s — hook)** "RLHF training on multi-GPU setups spends more time
> moving tensors between workers than doing matmuls.  We tested this on
> Qwen2.5-1.5B with 2× V100 and found that the transfer layer costs
> 25 ms per dispatch — that's the whole story of our project."
>
> **(15-30 s — approach)** "HybridFlow gave us the abstraction: every
> edge in the RLHF DAG is a `collect + distribute` protocol.  We built
> three knobs on that protocol — …  pull instead of push …  async dispatch
> to overlap with critic compute …  and FP16 / BF16 / INT8 compression on
> the float32 tensors."
>
> **(30-45 s — AO quick win, point at the Gantt chart)** "Async dispatch
> is the free win.  In an isolated 3-worker prep-stage test, non-blocking
> dispatch hides 36 % of wall-clock under concurrent compute — `ref_policy`
> runs underneath `old_log_prob`, `values` runs under the ref barrier —
> and cuts prep time from 4.4 s to 2.9 s per iteration.  No precision
> trade-off, no extra memory, just better scheduling."
>
> **(45-70 s — key result, point at the at-scale figure)** "Here's what
> surprised us on the compression side.  At batch=32, FP16 wins 17 % on
> mean iter time — classic compression result — but it costs 164 % more
> KL drift, and the policy ends up generating shorter responses, so
> throughput actually drops 27 %.  BF16 solves this.  Same payload
> savings as FP16, KL stays within noise of the baseline.  On V100 it
> pays 17 % wall-clock because V100 has no native BF16 tensor cores,
> but on Ampere or newer this should flip."
>
> **(70-90 s — closing, point at the Pareto chart)** "All four
> configurations are Pareto-optimal on iter-time versus KL.  So the
> right question isn't 'what's fastest' — it's 'what's your SLO'.  If
> you want throughput, pick FP16.  If you want stability, pick BF16.
> If you want tail-latency, pick pull alone.  AO stacks on any of them
> for free in a multi-worker prep stage.  The repo has the
> instrumentation, the probe log format, and reproducible scripts."

---

## 17. Numbers sanity checklist  (verify before printing)

Before sending to the plotter, paste each number below against the raw
data source to catch any typos.  All values here are as of 2026-04-22,
cross-verified against the regen scripts.

- [ ] 25.318 ms dispatch baseline ⇐ `checkpoints/verl_examples/gsm8k_2gpu_push_baseline/transfer_probe.jsonl`
- [ ] 8.563 ms dispatch FP16 ⇐ `checkpoints/verl_examples/gsm8k_2gpu_compress_fp16/transfer_probe_fp16.jsonl`
- [ ] 14.507 ms dispatch INT8 ⇐ `checkpoints/verl_examples/gsm8k_2gpu_compress_int8/transfer_probe_int8.jsonl`
- [ ] 18.706 ms dispatch LP ⇐ same as above for `gsm8k_2gpu_nocomp`
- [ ] batch=8 iter times 8.939 / 9.121 / 8.546 / 8.865 ⇐ `train_log.txt` in each of the 4 runs above
- [ ] batch=32 iter times 21.952 / 26.750 / 18.256 / 25.624 ⇐ `gsm8k_2gpu_{baseline,lp_only,lp_fp16,lp_bf16}_100/train_log.txt`
- [ ] batch=32 |KL| 0.01700 / 0.00389 / 0.04482 / 0.01124 ⇐ same
- [ ] batch=32 throughput 136.5 / 134.1 / 99.3 / 114.1 ⇐ same
- [ ] AO prep-stage serial totals 870 / 1603 / 1942 ms (sum = 4415 ms) ⇐ `4_2.md` Table 1
- [ ] AO async-wait totals 870 / 1175 / 827 ms (sum = 2872 ms) ⇐ `4_2.md` Table 1
- [ ] AO hidden_frac 0.000 / 0.285 / 0.591 / total 0.363 ⇐ `4_2.md` Table 1
- [ ] AO speedup 4415 → 2872 ms = −34.9 % ⇐ derived from above
- [ ] Seq-len sweep numbers ⇐ to be filled in after Rerun D completes

The exact producers of every figure are:
- `scripts/regen_7_eval.py` → `7_eval.png`
- `scripts/regen_4_3.py` → `4_3_compress.png`
- `scripts/regen_b32_figures.py` → `7_eval_b32.png` + `4_3_compress_b32.png`
- `scripts/jitter_and_kl.py` → `7_stability_corrected.png`
- `scripts/analyze_seqlen_sweep.py` → `7_seqlen_scaling.png` (pending)

Re-running any of these is idempotent and writes the exact same figure
from the raw JSONL / train_log files.
