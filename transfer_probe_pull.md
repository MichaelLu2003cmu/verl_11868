# 4.1 Local-Batch Pull Transfer Protocol — Milestone Record

## Overview

This milestone implements and validates a **prototype of Local-Batch Pull Transfer Protocol (4.1)** in the verl RL training system.

The goal is to replace:

aggregate → split → send → collect → concat

with:

metadata → local slice computation → selective fetch

We focused on one transfer path:

- actor_rollout_compute_log_prob

---

## Approach

### Key Idea

- Avoid constructing global full-batch tensors
- Move slicing logic to receiver (dp_rank)
- Fetch only required shard(s)
- Reduce unnecessary concat / data movement

---

## Implementation

### Files Modified

- protocol.py
- decorator.py
- fsdp_workers.py (only one decorator change)

---

### Steps

#### Step 1 — Pull-style Dispatch (MVP)

- Replace eager split with shard references
- Receiver selects local shard via dp_rank

#### Step 2 — Selective Future Fetch

Added:

- PartitionSpec
- BatchInterval
- DataProtoSelectiveView

Key change:

- fetch only overlapping futures
- avoid ray.get(all futures)
- avoid full concat when unnecessary

---

#### Step 3 — Dispatch Integration

- New dispatch mode:
  make_nd_compute_dataproto_pull_dispatch_fn

Applied only to:
compute_log_prob

---

#### Step 4 — Materialization Hook

- Extended _materialize_futures(...)
- Converts selective view → DataProto
- No change to worker logic

---

## Results

### Baseline (Single GPU)

- dispatch_ms ≈ 7.65 ms
- collect_ms ≈ 1.96 ms
- recv_MB ≈ 0.003 MB
- DataProto.concat ≈ 0.19 ms
- Dominant cost: wait_ms

---

### 4.1 Pull Prototype (Single GPU)

20-step run:

- dispatch_ms ≈ 5.6 – 6.5 ms
- collect_ms ≈ 1.7 – 2.0 ms
- recv_MB ≈ 0.0024 – 0.0030 MB
- DataProto.concat ≈ 0.17 – 0.19 ms

2-step validation:

- dispatch_ms ≈ 6.1 ms
- collect_ms ≈ 1.7 – 4.3 ms
- recv_bytes ≈ 2704 bytes
- compute_log_prob executed successfully

---

## Interpretation

### Improvements

- Slight reduction in dispatch_ms
- Slight reduction in transfer size
- Protocol path executed correctly

### No Major Gains

- Runtime dominated by wait_ms
- Transfer size is tiny (KB scale)
- concat overhead already minimal

### Key Insight

This workload is compute-bound, not transfer-bound.

---

## Validation Summary

- compute_log_prob executed under new protocol
- selective fetch logic works
- no crash in materialization
- system stable across runs

---

## Limitations

- Single GPU (no communication bottleneck)
- Small batch size
- Only one path modified
- No visible end-to-end speedup

---

## Achievements

- Receiver-driven slicing (prototype)
- Selective future fetch
- Integrated into verl dispatch layer
- End-to-end validated

---

## Not Yet Done

- True worker-to-worker pull
- Multi-path adoption
- Transfer-bound evaluation
- Large-scale performance gain

---

## Next Step — Multi-GPU Evaluation

### Goal

Evaluate 4.1 under real transfer pressure.

---

### Experiment Plan

Run:

Baseline (original dispatch)
4.1 Pull (current implementation)

---

### Keep Constant

- model
- dataset
- batch size
- sequence length
- rollout config

---

### Change Only

- number of GPUs
- protocol version

---

### Progression

1. 2 GPU baseline
2. 2 GPU + 4.1
3. 4 GPU baseline
4. 4 GPU + 4.1

---

### Metrics

Path-level (compute_log_prob):

- dispatch_ms / iter
- collect_ms / iter
- recv_MB / iter
- total_ms / iter

System-level:

- iteration time
- throughput
- concat overhead
- wait_ms

---

### Stress Transfer

- increase batch size
- increase sequence length
- increase rollout.n

---

## Expected Outcome

Under multi-GPU:

- reduced dispatch overhead
- reduced transfer volume
- fewer concat operations
- improved efficiency when transfer-bound

---

## Summary

We implemented a working prototype of Local-Batch Pull Transfer Protocol (4.1).

- Correctness validated
- Minimal system changes
- No gain in current setup due to compute-bound workload

Next step: multi-GPU comparison to evaluate real impact.

- 2-GPU baseline command:
```bash
cd /ocean/projects/cis260009p/syan5/verl_11868 && \
ray stop --force || true; \
rm -rf /ocean/projects/cis260009p/syan5/verl_11868/checkpoints/verl_examples/gsm8k_2gpu_baseline && \
mkdir -p /ocean/projects/cis260009p/syan5/verl_11868/checkpoints/verl_examples/gsm8k_2gpu_baseline && \
rm -f /ocean/projects/cis260009p/syan5/verl_11868/checkpoints/verl_examples/gsm8k_2gpu_baseline/transfer_probe_2gpu_baseline.jsonl; \
env \
-u ROCR_VISIBLE_DEVICES \
-u HIP_VISIBLE_DEVICES \
-u PYTORCH_ALLOC_CONF \
-u PYTORCH_CUDA_ALLOC_CONF \
CUDA_VISIBLE_DEVICES=0,1 \
RAY_DISABLE_DASHBOARD=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_syan5 \
XDG_CACHE_HOME=/tmp/xdg_cache_syan5 \
TRITON_CACHE_DIR=/tmp/triton_cache_syan5 \
TMPDIR=/tmp \
VLLM_USAGE_STATS=0 \
HF_HUB_DISABLE_XET=1 \
HYDRA_FULL_ERROR=1 \
VERL_TRANSFER_PROBE=1 \
VERL_TRANSFER_PROBE_LOG=/ocean/projects/cis260009p/syan5/verl_11868/checkpoints/verl_examples/gsm8k_2gpu_baseline/transfer_probe_2gpu_baseline.jsonl \
/ocean/projects/cis260009p/syan5/conda/project/bin/python -m verl.trainer.main_ppo \
algorithm.adv_estimator=grpo \
data.train_files=/jet/home/syan5/data/gsm8k/train.parquet \
data.val_files=/jet/home/syan5/data/gsm8k/test.parquet \
data.train_batch_size=8 \
data.max_prompt_length=256 \
data.max_response_length=128 \
data.filter_overlong_prompts=False \
data.truncation=left \
data.trust_remote_code=True \
data.dataloader_num_workers=2 \
actor_rollout_ref.model.path=/ocean/projects/cis260009p/syan5/models/Qwen2.5-1.5B-Instruct \
actor_rollout_ref.model.tokenizer_path=/ocean/projects/cis260009p/syan5/models/Qwen2.5-1.5B-Instruct \
actor_rollout_ref.model.trust_remote_code=True \
actor_rollout_ref.model.use_remove_padding=False \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
+actor_rollout_ref.model.override_config.attn_implementation=eager \
actor_rollout_ref.actor.ppo_mini_batch_size=4 \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
actor_rollout_ref.actor.fsdp_config.fsdp_size=2 \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.mode=async \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.rollout.data_parallel_size=1 \
actor_rollout_ref.rollout.dtype=float16 \
actor_rollout_ref.rollout.enforce_eager=True \
actor_rollout_ref.rollout.gpu_memory_utilization=0.18 \
actor_rollout_ref.rollout.n=1 \
actor_rollout_ref.rollout.max_model_len=384 \
actor_rollout_ref.rollout.max_num_seqs=16 \
actor_rollout_ref.rollout.max_num_batched_tokens=512 \
actor_rollout_ref.rollout.enable_chunked_prefill=False \
actor_rollout_ref.rollout.enable_prefix_caching=False \
actor_rollout_ref.rollout.free_cache_engine=True \
actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024 \
actor_rollout_ref.rollout.agent.num_workers=1 \
trainer.n_gpus_per_node=2 \
trainer.nnodes=1 \
trainer.total_epochs=1 \
trainer.total_training_steps=20 \
trainer.val_before_train=False \
trainer.test_freq=-1 \
trainer.save_freq=-1 \
trainer.logger=console \
hydra.run.dir=/ocean/projects/cis260009p/syan5/verl_outputs_2gpu_baseline \
++ray_kwargs.ray_init._temp_dir=/tmp/ray_syan5 \
++ray_kwargs.ray_init.include_dashboard=False
```

- 2-GPU pull command
```bash
cd /ocean/projects/cis260009p/syan5/verl_11868 && \
ray stop --force || true; \
rm -rf /ocean/projects/cis260009p/syan5/verl_11868/checkpoints/verl_examples/gsm8k_2gpu_pull && \
mkdir -p /ocean/projects/cis260009p/syan5/verl_11868/checkpoints/verl_examples/gsm8k_2gpu_pull && \
rm -f /ocean/projects/cis260009p/syan5/verl_11868/checkpoints/verl_examples/gsm8k_2gpu_pull/transfer_probe_2gpu_pull.jsonl; \
env \
-u ROCR_VISIBLE_DEVICES \
-u HIP_VISIBLE_DEVICES \
-u PYTORCH_ALLOC_CONF \
-u PYTORCH_CUDA_ALLOC_CONF \
CUDA_VISIBLE_DEVICES=0,1 \
RAY_DISABLE_DASHBOARD=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_syan5 \
XDG_CACHE_HOME=/tmp/xdg_cache_syan5 \
TRITON_CACHE_DIR=/tmp/triton_cache_syan5 \
TMPDIR=/tmp \
VLLM_USAGE_STATS=0 \
HF_HUB_DISABLE_XET=1 \
HYDRA_FULL_ERROR=1 \
VERL_TRANSFER_PROBE=1 \
VERL_TRANSFER_PROBE_LOG=/ocean/projects/cis260009p/syan5/verl_11868/checkpoints/verl_examples/gsm8k_2gpu_pull/transfer_probe_2gpu_pull.jsonl \
/ocean/projects/cis260009p/syan5/conda/project/bin/python -m verl.trainer.main_ppo \
algorithm.adv_estimator=grpo \
data.train_files=/jet/home/syan5/data/gsm8k/train.parquet \
data.val_files=/jet/home/syan5/data/gsm8k/test.parquet \
data.train_batch_size=8 \
data.max_prompt_length=256 \
data.max_response_length=128 \
data.filter_overlong_prompts=False \
data.truncation=left \
data.trust_remote_code=True \
data.dataloader_num_workers=2 \
actor_rollout_ref.model.path=/ocean/projects/cis260009p/syan5/models/Qwen2.5-1.5B-Instruct \
actor_rollout_ref.model.tokenizer_path=/ocean/projects/cis260009p/syan5/models/Qwen2.5-1.5B-Instruct \
actor_rollout_ref.model.trust_remote_code=True \
actor_rollout_ref.model.use_remove_padding=False \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
+actor_rollout_ref.model.override_config.attn_implementation=eager \
actor_rollout_ref.actor.ppo_mini_batch_size=4 \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
actor_rollout_ref.actor.fsdp_config.fsdp_size=2 \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.mode=async \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.rollout.data_parallel_size=1 \
actor_rollout_ref.rollout.dtype=float16 \
actor_rollout_ref.rollout.enforce_eager=True \
actor_rollout_ref.rollout.gpu_memory_utilization=0.18 \
actor_rollout_ref.rollout.n=1 \
actor_rollout_ref.rollout.max_model_len=384 \
actor_rollout_ref.rollout.max_num_seqs=16 \
actor_rollout_ref.rollout.max_num_batched_tokens=512 \
actor_rollout_ref.rollout.enable_chunked_prefill=False \
actor_rollout_ref.rollout.enable_prefix_caching=False \
actor_rollout_ref.rollout.free_cache_engine=True \
actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024 \
actor_rollout_ref.rollout.agent.num_workers=1 \
trainer.n_gpus_per_node=2 \
trainer.nnodes=1 \
trainer.total_epochs=1 \
trainer.total_training_steps=20 \
trainer.val_before_train=False \
trainer.test_freq=-1 \
trainer.save_freq=-1 \
trainer.logger=console \
hydra.run.dir=/ocean/projects/cis260009p/syan5/verl_outputs_2gpu_pull \
++ray_kwargs.ray_init._temp_dir=/tmp/ray_syan5 \
++ray_kwargs.ray_init.include_dashboard=False
```