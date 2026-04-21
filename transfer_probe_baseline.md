# Transfer Probe Baseline (VERL PPO)

## Experiment Setup

- Repo: `verl_11868`
- Task: PPO (`algorithm.adv_estimator=grpo`)
- Device: single GPU (`trainer.n_gpus_per_node=1`, `trainer.nnodes=1`)
- Probe enabled:
  - `VERL_TRANSFER_PROBE=1`
  - `VERL_TRANSFER_PROBE_LOG=/ocean/projects/cis260009p/syan5/verl_11868/checkpoints/verl_examples/gsm8k/transfer_probe.jsonl`
- Run command:
```bash
cd /ocean/projects/cis260009p/syan5/verl_11868 && \
ray stop --force || true; \
rm -f /ocean/projects/cis260009p/syan5/verl_11868/checkpoints/verl_examples/gsm8k/transfer_probe.jsonl; \
env \
-u ROCR_VISIBLE_DEVICES \
-u HIP_VISIBLE_DEVICES \
-u PYTORCH_ALLOC_CONF \
-u PYTORCH_CUDA_ALLOC_CONF \
CUDA_VISIBLE_DEVICES=0 \
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
VERL_TRANSFER_PROBE_LOG=/ocean/projects/cis260009p/syan5/verl_11868/checkpoints/verl_examples/gsm8k/transfer_probe.jsonl \
/ocean/projects/cis260009p/syan5/conda/project/bin/python -m verl.trainer.main_ppo \
algorithm.adv_estimator=grpo \
data.train_files=/jet/home/syan5/data/gsm8k/train.parquet \
data.val_files=/jet/home/syan5/data/gsm8k/test.parquet \
data.train_batch_size=2 \
data.max_prompt_length=128 \
data.max_response_length=64 \
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
actor_rollout_ref.actor.ppo_mini_batch_size=1 \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.mode=async \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.rollout.dtype=float16 \
actor_rollout_ref.rollout.enforce_eager=True \
actor_rollout_ref.rollout.gpu_memory_utilization=0.18 \
actor_rollout_ref.rollout.n=1 \
actor_rollout_ref.rollout.max_model_len=192 \
actor_rollout_ref.rollout.max_num_seqs=8 \
actor_rollout_ref.rollout.max_num_batched_tokens=128 \
actor_rollout_ref.rollout.enable_chunked_prefill=False \
actor_rollout_ref.rollout.enable_prefix_caching=False \
actor_rollout_ref.rollout.free_cache_engine=True \
actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024 \
actor_rollout_ref.rollout.agent.num_workers=1 \
trainer.n_gpus_per_node=1 \
trainer.nnodes=1 \
trainer.total_epochs=1 \
trainer.total_training_steps=60 \
trainer.val_before_train=False \
trainer.test_freq=-1 \
trainer.save_freq=-1 \
trainer.logger=console \
hydra.run.dir=/ocean/projects/cis260009p/syan5/verl_outputs \
++ray_kwargs.ray_init._temp_dir=/tmp/ray_syan5 \
++ray_kwargs.ray_init.include_dashboard=False
```
- Statistics command:
```bash
python scripts/summarize_transfer_probe.py --log /ocean/projects/cis260009p/syan5/verl_11868/checkpoints/verl_examples/gsm8k/transfer_probe.jsonl
```

## Transfer Latency Baseline

| method_name | iters | dispatch_ms/iter | wait_ms/iter | collect_ms/iter | total_ms/iter | send_MB/iter | recv_MB/iter |
|---|---:|---:|---:|---:|---:|---:|---:|
| actor_rollout__query_collect_info | 1 | 0.005 | 1.354 | 0.001 | 1.643 | 0.000 | 0.000 |
| actor_rollout__query_dispatch_info | 1 | 0.026 | 1.398 | 0.004 | 1.786 | 0.000 | 0.000 |
| actor_rollout_compute_log_prob | 40 | 7.652 | 1381.174 | 1.957 | 1391.148 | 0.000 | 0.003 |
| actor_rollout_init_model | 1 | 0.005 | 19651.097 | 0.001 | 19651.496 | 0.000 | 0.000 |
| actor_rollout_load_checkpoint | 1 | 0.010 | 19301.636 | 0.001 | 19301.958 | 0.000 | 0.000 |
| actor_rollout_update_actor | 40 | 7.024 | 3975.616 | 0.029 | 3982.956 | 0.000 | 0.000 |
| actor_rollout_update_weights | 41 | 0.008 | 0.000 | 0.001 | 0.268 | 0.000 | 0.000 |

## CPU Overhead Baseline

| location | iters | elapsed_ms/iter | output_MB/iter |
|---|---:|---:|---:|
| DataProto.concat | 40 | 0.191 | 0.013 |

## Interpretation

- Main latency is concentrated in `wait_ms` (remote execution/worker-side compute), not `dispatch_ms` or `collect_ms`.
- Host-side aggregation overhead is low (`DataProto.concat` is about `0.191 ms/iter`).
- Measured transfer volume is small in this setup (e.g., `actor_rollout_compute_log_prob` has `recv_MB/iter ~= 0.003`).
- This table is a single-GPU baseline and can be used as the reference point for multi-GPU or larger-batch comparisons.
