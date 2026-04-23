#!/usr/bin/env bash
# Sequence-length sweep for Rerun D (proposal §7.2 "vary batch size and
# sequence length to evaluate compute-bound vs communication-bound regimes").
#
# Runs 6 configurations = 3 response-length points × {push-baseline, LP+FP16},
# each for 10 steps at batch=8.  Probe is enabled so we get transfer bytes /
# dispatch_ms / collect_ms per config.  Expected total runtime ~60-90 min.
#
# Results land in:
#   checkpoints/verl_examples/gsm8k_seqlen_<RESP>_<VARIANT>/
#     train_log.txt          # per-step metrics
#     transfer_probe.jsonl   # per-transfer microbench (VERL_TRANSFER_PROBE=1)
#
# Run as:
#   bash scripts/run_seqlen_sweep.sh 2>&1 | tee seqlen_sweep_driver.log
#
# Rerun of a single config: just re-run the matching inner block after
# `rm -rf <output_dir>`.

set -euo pipefail
cd /ocean/projects/cis260009p/syan5/verl_11868

unset ROCR_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

MODEL=/ocean/projects/cis260009p/syan5/models/Qwen2.5-1.5B-Instruct
DATA_TRAIN=/jet/home/syan5/data/gsm8k/train.parquet
DATA_VAL=/jet/home/syan5/data/gsm8k/test.parquet

# Response-length sweep.  Prompt length is fixed at 256.
#   max_model_len    = 256 + response_length
#   max_num_seqs     = batch_size (8)
#   max_num_batched_tokens should be ≥ max_model_len
RESPONSE_LENGTHS=(128 512 1024)

# 10 steps is enough to amortize vLLM warmup and give ~80-90 post-warmup
# transfer-probe events per config for averaging.
TOTAL_STEPS=10

# Variants: baseline (push, no compress) and LP+FP16 (pull, fp16).
# Same hyper-parameters as the 100-step cumulative ablation, just fewer steps
# and smaller batch.  batch=8 is chosen to keep all 6 runs fit into one
# allocation window.

run_one() {
  local resp_len=$1
  local variant=$2      # either "base" or "lpfp16"

  local max_model_len=$((256 + resp_len))
  local max_tokens=$((max_model_len > 1024 ? max_model_len : 1024))

  local exp_name="gsm8k_seqlen_${resp_len}_${variant}"
  local out_dir="checkpoints/verl_examples/${exp_name}"
  local probe_path="${out_dir}/transfer_probe.jsonl"

  if [[ -f "${out_dir}/train_log.txt" ]]; then
    echo "[skip] ${exp_name} already has train_log.txt (delete to rerun)"
    return 0
  fi

  mkdir -p "${out_dir}"

  # Probe: always on for this sweep (we want transfer-bytes scaling curves).
  export VERL_TRANSFER_PROBE=1
  export VERL_TRANSFER_PROBE_LOG="${probe_path}"

  local legacy_impl compress_arg
  if [[ "${variant}" == "base" ]]; then
    legacy_impl="disable"
    unset VERL_TRANSFER_COMPRESS
    compress_arg="(unset)"
  else
    legacy_impl="enable"
    export VERL_TRANSFER_COMPRESS="fp16"
    compress_arg="fp16"
  fi

  echo ""
  echo "=============================================="
  echo "=== ${exp_name}"
  echo "=== response_length=${resp_len}  variant=${variant}"
  echo "=== legacy_worker_impl=${legacy_impl}  compress=${compress_arg}"
  echo "=== max_model_len=${max_model_len}  max_num_batched_tokens=${max_tokens}"
  echo "=============================================="

  python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${DATA_TRAIN}" \
    data.val_files="${DATA_VAL}" \
    data.train_batch_size=8 \
    data.max_prompt_length=256 \
    data.max_response_length="${resp_len}" \
    data.shuffle=True \
    data.truncation=left \
    actor_rollout_ref.model.path="${MODEL}" \
    +actor_rollout_ref.model.override_config.attn_implementation=eager \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
    actor_rollout_ref.rollout.max_model_len="${max_model_len}" \
    actor_rollout_ref.rollout.max_num_seqs=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens="${max_tokens}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name=verl_examples \
    trainer.experiment_name="${exp_name}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    trainer.default_local_dir="${out_dir}" \
    trainer.resume_mode=auto \
    trainer.use_legacy_worker_impl="${legacy_impl}" \
    reward_model.enable=False \
    2>&1 | tee "${out_dir}/train_log.txt"

  # Post-run probe sanity check (prints 0-or-nonzero so the driver log is
  # self-documenting).
  local probe_events=$(grep -c '\[transfer_probe\]' "${probe_path}" 2>/dev/null || echo 0)
  local compress_events=$(grep -c '"event":"compress_stats"\|"event": "compress_stats"' "${probe_path}" 2>/dev/null || echo 0)
  echo "[post-run] ${exp_name}: probe_events=${probe_events}  compress_events=${compress_events}"
}

# Driver: run baseline first, then LP+FP16, for each seqlen.  Running same
# seqlen adjacently means the page cache for the data parquet stays warm and
# vLLM doesn't have to re-JIT for a new sequence length between variants.
for resp_len in "${RESPONSE_LENGTHS[@]}"; do
  run_one "${resp_len}" "base"
  run_one "${resp_len}" "lpfp16"
done

echo ""
echo "=============================================="
echo "=== SEQUENCE-LENGTH SWEEP COMPLETE"
echo "=============================================="
echo "Output dirs:"
for resp_len in "${RESPONSE_LENGTHS[@]}"; do
  for variant in base lpfp16; do
    printf "  %s\n" "checkpoints/verl_examples/gsm8k_seqlen_${resp_len}_${variant}"
  done
done
echo ""
echo "Next: run scripts/analyze_seqlen_sweep.py to produce the scaling table."
