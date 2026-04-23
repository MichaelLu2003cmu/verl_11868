#!/usr/bin/env bash
# Rerun the 3 LP-legacy 100-step configs after the stability fix in
# verl/workers/actor/dp_actor.py (always use stored old_log_probs instead
# of log_prob.detach() when on_policy=True).
#
# Original 100-step runs that COLLAPSED (grad_norm → 0 at step ~22):
#   gsm8k_2gpu_lp_only_100   legacy=enable  compress=(unset)
#   gsm8k_2gpu_lp_fp16_100   legacy=enable  compress=fp16
#   gsm8k_2gpu_lp_bf16_100   legacy=enable  compress=bf16
#
# The baseline (gsm8k_2gpu_baseline_100, legacy=disable) did NOT collapse
# and does NOT need rerunning — the fix only touches the legacy path.
#
# Config matches the original 100-step runs (extracted from train_log.txt):
#   model=Qwen2.5-1.5B-Instruct  batch=32  mini=8  max_prompt=256  max_resp=128
#   lr=1e-6  use_kl_in_reward=False  use_kl_loss=False  rollout.n=1
#   fsdp_size=2  param_offload=False  vllm async + float16  gpu_util=0.15
#
# Memory: 1.5B + fsdp=2 + NO ref worker ≈ 14 GB peak per GPU → fits 2×V100 16 GB.
# (Different from run_combined_stack.sh which enables ref → needs 0.5B.)
#
# Run (in a GPU session, e.g. on v002):
#   bash scripts/rerun_stability_100.sh 2>&1 | tee rerun_stability_driver.log
#
# After it finishes, regenerate the plot with the SAME command the original
# 7_stability_corrected.png was produced with:
#   /ocean/projects/cis260009p/syan5/conda/project/bin/python \
#     scripts/jitter_and_kl.py \
#     --logs checkpoints/verl_examples/gsm8k_2gpu_baseline_100/train_log.txt \
#            checkpoints/verl_examples/gsm8k_2gpu_lp_only_100/train_log.txt \
#            checkpoints/verl_examples/gsm8k_2gpu_lp_fp16_100/train_log.txt \
#            checkpoints/verl_examples/gsm8k_2gpu_lp_bf16_100/train_log.txt \
#     --labels "Baseline (push)" "+LP (pull)" "+LP+FP16" "+LP+BF16" \
#     --title "Training Stability — After on_policy fix (100 steps, batch=32, 2×V100)" \
#     --plot-out 7_stability_corrected.png

set -euo pipefail
cd /ocean/projects/cis260009p/syan5/verl_11868

unset ROCR_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export RAY_object_store_memory=4000000000

MODEL=/ocean/projects/cis260009p/syan5/models/Qwen2.5-1.5B-Instruct
DATA_TRAIN=/jet/home/syan5/data/gsm8k/train.parquet
DATA_VAL=/jet/home/syan5/data/gsm8k/test.parquet

BATCH_SIZE=32
MINI_BATCH=8
MAX_PROMPT=256
MAX_RESPONSE=128
TOTAL_STEPS=100
N_GPUS=2

run_one() {
  local tag=$1         # lp_only | lp_fp16 | lp_bf16
  local compress=$2    # "" | fp16 | bf16

  local exp_name="gsm8k_2gpu_${tag}_100"
  local out_dir="checkpoints/verl_examples/${exp_name}"

  # Force rerun: archive the old (collapsed) run before overwriting.
  if [[ -d "${out_dir}" ]]; then
    local stamp
    stamp=$(date +%Y%m%d_%H%M%S)
    echo "[archive] ${out_dir} → ${out_dir}.collapsed_${stamp}"
    mv "${out_dir}" "${out_dir}.collapsed_${stamp}"
  fi

  mkdir -p "${out_dir}"

  if [[ -n "${compress}" ]]; then
    export VERL_TRANSFER_COMPRESS="${compress}"
  else
    unset VERL_TRANSFER_COMPRESS || true
  fi

  echo ""
  echo "=============================================="
  echo "=== ${exp_name}  (compress=${compress:-unset})"
  echo "=== legacy=enable  batch=${BATCH_SIZE}  mini=${MINI_BATCH}  steps=${TOTAL_STEPS}"
  echo "=============================================="

  python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_files="${DATA_TRAIN}" \
    data.val_files="${DATA_VAL}" \
    data.train_batch_size="${BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT}" \
    data.max_response_length="${MAX_RESPONSE}" \
    data.shuffle=True \
    data.truncation=left \
    actor_rollout_ref.model.path="${MODEL}" \
    +actor_rollout_ref.model.override_config.torch_dtype=float16 \
    +actor_rollout_ref.model.override_config.attn_implementation=eager \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BATCH}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size="${N_GPUS}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.15 \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
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
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    trainer.default_local_dir="${out_dir}" \
    trainer.resume_mode=auto \
    trainer.use_legacy_worker_impl=enable \
    reward_model.enable=False \
    2>&1 | tee "${out_dir}/train_log.txt"

  # Collapse sanity check: how many steps had grad_norm=0 after step 20.
  local zero_grad_tail
  zero_grad_tail=$(grep -oE "actor/grad_norm:[0-9.]+" "${out_dir}/train_log.txt" \
      | tail -n 80 | grep -c "actor/grad_norm:0\.0" || true)
  echo "[post-run] ${exp_name}: zero-grad tail steps (of last 80) = ${zero_grad_tail}  (was 79 before fix)"
}

# Only rerun the legacy-path LP variants; baseline_100 is unaffected.
run_one "lp_only" ""
run_one "lp_fp16" "fp16"
run_one "lp_bf16" "bf16"

echo ""
echo "=============================================="
echo "=== STABILITY RERUN COMPLETE"
echo "=============================================="
echo "Regenerate 7_stability_corrected.png with:"
echo ""
echo "/ocean/projects/cis260009p/syan5/conda/project/bin/python \\"
echo "  scripts/jitter_and_kl.py \\"
echo "  --logs checkpoints/verl_examples/gsm8k_2gpu_baseline_100/train_log.txt \\"
echo "         checkpoints/verl_examples/gsm8k_2gpu_lp_only_100/train_log.txt \\"
echo "         checkpoints/verl_examples/gsm8k_2gpu_lp_fp16_100/train_log.txt \\"
echo "         checkpoints/verl_examples/gsm8k_2gpu_lp_bf16_100/train_log.txt \\"
echo "  --labels \"Baseline (push)\" \"+LP (pull)\" \"+LP+FP16\" \"+LP+BF16\" \\"
echo "  --title \"Training Stability — After on_policy fix\" \\"
echo "  --plot-out 7_stability_corrected.png"
