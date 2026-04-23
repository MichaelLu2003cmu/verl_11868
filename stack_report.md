## Rerun C' - Combined LP + AO + Comp stack (batch=8, 20 steps, ref worker ENABLED)

Probe events skip step 1 (vLLM warmup).  Iter-time stats also skip step 1.
Mean is over steps 2..N.  `kl_mean_abs` averages `|actor/ppo_kl|` per step.

### End-to-end metrics
| Config | n | iter mean (s) | iter p50 | iter p99 | std | |KL|_mean | throughput |
|---|---:|---:|---:|---:|---:|---:|---:|
| push+ref baseline (ref ON, push) | 19 | 13.065 | 13.599 | 13.871 | 1.142 | 0.00000 | 125.7 |
| +LP+AO (pull + async, ref ON) | 19 | 12.903 | 13.258 | 13.748 | 0.839 | 0.00000 | 128.5 |
| +LP+AO+FP16 (full stack, ref ON) | 19 | 13.335 | 13.736 | 14.250 | 0.962 | 0.00000 | 125.0 |
| Existing push baseline (ref OFF) | 19 | 8.939 | 8.910 | 9.022 | 0.064 | 0.01561 | 103.3 |
| Existing pathB_lite (LP+AO, ref+critic) | 0 | nan | nan | nan | 0.000 | nan | nan |

### Prep-stage driver-blocking time (headline LP+AO metric)

*`total blocking` = `max(joined_at_ms)` across critical_path_stage events per step.  This is the ground-truth end-to-end driver blocking time for the prep stage in both push and pull modes: in push mode the ref computation completes inside `_compute_ref_log_prob()` before the probe marks dispatch (so `wait_ms`≈0 for ref), but `joined_at_ms` of the ref stage is still recorded after the computation finishes, giving the correct total.*

| Config | total blocking (ms, mean) | total blocking (ms, p50) | n_steps |
|---|---:|---:|---:|
| push+ref baseline (ref ON, push) | 1856.8 | 1863.5 | 19 |
| +LP+AO (pull + async, ref ON) | 1295.1 | 1295.8 | 19 |
| +LP+AO+FP16 (full stack, ref ON) | 1302.5 | 1309.9 | 19 |
| Existing push baseline (ref OFF) | 1223.5 | 1222.3 | 19 |
| Existing pathB_lite (LP+AO, ref+critic) | 2941.3 | 2924.2 | 19 |

### Per-stage breakdown (mean across steps)
| Config | stage | count | wait_ms mean | wait_ms p50 | in_flight_ms mean |
|---|---|---:|---:|---:|---:|
| push+ref baseline (ref ON, push) | `old_log_prob` | 19 | 1023.6 | 1033.6 | 1023.6 |
| push+ref baseline (ref ON, push) | `ref` | 19 | 0.0 | 0.0 | 0.1 |
| +LP+AO (pull + async, ref ON) | `old_log_prob` | 19 | 662.0 | 660.8 | 662.0 |
| +LP+AO (pull + async, ref ON) | `ref` | 19 | 626.9 | 625.4 | 627.0 |
| +LP+AO+FP16 (full stack, ref ON) | `old_log_prob` | 19 | 668.1 | 669.0 | 668.1 |
| +LP+AO+FP16 (full stack, ref ON) | `ref` | 19 | 628.5 | 633.6 | 628.6 |
| Existing push baseline (ref OFF) | `old_log_prob` | 19 | 1223.4 | 1222.2 | 1223.4 |
| Existing pathB_lite (LP+AO, ref+critic) | `old_log_prob` | 19 | 887.3 | 869.8 | 887.3 |
| Existing pathB_lite (LP+AO, ref+critic) | `ref` | 19 | 1206.2 | 1179.8 | 1210.5 |
| Existing pathB_lite (LP+AO, ref+critic) | `values` | 19 | 837.7 | 829.4 | 2044.5 |

### Composability: each optimization's contribution vs. push+ref baseline

| Config | iter_mean Delta% | iter_p50 Delta% | total blocking Delta% | |KL|_mean Delta% | recv_bytes Delta% |
|---|---:|---:|---:|---:|---:|
| push+ref baseline (ref ON, push) | +0.0% | +0.0% | +0.0% | +nan% | +0.0% |
| +LP+AO (pull + async, ref ON) | -1.2% | -2.5% | -30.2% | +nan% | -18.2% |
| +LP+AO+FP16 (full stack, ref ON) | +2.1% | +1.0% | -29.9% | +nan% | -18.2% |
| Existing push baseline (ref OFF) | -31.6% | -34.5% | -34.1% | +nan% | -49.7% |
| Existing pathB_lite (LP+AO, ref+critic) | +nan% | +nan% | +58.4% | +nan% | -86.3% |

### Transfer-layer probe (bytes + compression)
| Config | Sigma recv/iter (KB) | Sigma send/iter (KB) | Sigma xfer_ms/iter | compress events | mean compress ratio |
|---|---:|---:|---:|---:|---:|
| push+ref baseline (ref ON, push) | 9.0 | 0.0 | 11.38 | 0 | 0.000 |
| +LP+AO (pull + async, ref ON) | 7.4 | 55.4 | 14.44 | 0 | 0.000 |
| +LP+AO+FP16 (full stack, ref ON) | 7.4 | 55.4 | 17.19 | 20 | 0.967 |
| Existing push baseline (ref OFF) | 4.5 | 0.0 | 12.92 | 0 | 0.000 |
| Existing pathB_lite (LP+AO, ref+critic) | 1.2 | 16.8 | 73.00 | 0 | 0.000 |
