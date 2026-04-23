#!/usr/bin/env bash
# Rerun C' — combined LP + AO + Comp stack, 2×V100 16GB (0.5B model).
#
# Goal: show all three transfer-protocol optimizations fire simultaneously
# on the same pipeline, with an explicit push baseline to measure the AO win.
# Three configs (full spec per gpu_shopping_list.md §Rerun C'):
#
#   Config        legacy_worker_impl  VERL_TRANSFER_COMPRESS  LP  AO  Comp
#   stack_push_ref  disable           (unset)                  ✗   ✗   ✗
#   lp_ao           enable            (unset)                  ✓   ✓   ✗
#   lp_ao_fp16      enable            fp16                     ✓   ✓   ✓
#
# stack_push_ref is the AO baseline: push mode blocks the driver on every
# ref future (ray.get() inside _compute_ref_log_prob), so ref wait_ms is
# fully on the critical path.  Comparing push_ref → lp_ao gives the AO win
# (Σwait_ms drops as ref overlaps with old_log_prob).
# Comparing lp_ao → lp_ao_fp16 gives the incremental Comp contribution.
#
# NOTE: push_ref was originally omitted (2×V100 32GB + 1.5B model) because
# push mode materialises ref+actor all-gathers simultaneously → OOM for
# 1.5B fp16 (2×3 GB peaks + vLLM → >32 GB).  With 0.5B (2×1 GB peaks
# + vLLM 3.2 GB + Adam 1 GB ≈ 6 GB << 16 GB) it is safe.
#
# Key design decisions (each from a real crash investigation):
#   use_kl_in_reward=True  instead of  use_kl_loss=True
#       use_kl_loss retains ref log-probs through the entire actor
#       backward pass (needed for KL gradient), so ref+actor+optimizer
#       all peak simultaneously.  use_kl_in_reward puts ref in the
#       reward phase (before actor update), frees it, then actor update
#       runs with no ref in memory.  Both flags activate the ref worker
#       via need_reference_policy() — LP/AO/Comp behaviour is identical.
#
#   param_offload=False  (for both actor and ref)
#       param_offload=True + async rollout causes a CPU→GPU copy race:
#       actor_rollout_update_weights fires non-blocking (seq=2) while
#       FSDP immediately tries to all-gather from CPU → silent SIGKILL
#       at seq=3.  On 2×V100 with use_kl_in_reward the memory fits
#       without offloading (~14 GB peak per GPU, see budget below).
#
#   use_kl_in_reward does NOT affect the research claims:
#       LP, AO, and Comp are transfer-protocol mechanisms on tensor
#       edges between workers.  The KL flag only changes when the ref
#       output tensor is consumed by the driver; the probe events
#       (dispatch_ms, hidden_frac, compress_stats) are identical.
#
# Memory budget per GPU (2×V100 16GB, fsdp_size=2, Qwen2.5-0.5B-Instruct):
#   Actor FSDP shard (fp16)    ~0.5 GB
#   Ref   FSDP shard (fp16)    ~0.5 GB
#   Adam states (fp32)         ~1.0 GB
#   vLLM model+KV (gpu=0.20)   ~3.2 GB  (model ~1 GB + KV cache ~72 MB for b=8/seq=768)
#   Activations (b=8, seq=512) ~1 GB
#   ─────────────────────────────────
#   Total peak                ~6–7 GB  <<  16 GB
#   push_ref peak (actor+ref all-gather overlap): +2×1 GB = 8 GB  <<  16 GB  ✓
#
# NOTE: switched from 1.5B to 0.5B because the cluster allocates 16 GB V100s;
#   1.5B OOMs on 16 GB (actor+ref+vLLM peaks ~15-16 GB, no headroom for FSDP
#   all-gather).  0.5B saves ~6 GB total, fits easily.  LP/AO/Comp probe events
#   are model-agnostic so research claims are unaffected.
#
# Run:
#   bash scripts/run_combined_stack.sh 2>&1 | tee combined_stack_driver.log
#
# Re-run one config: rm -rf <output_dir> then re-invoke.

set -euo pipefail
cd /ocean/projects/cis260009p/syan5/verl_11868

unset ROCR_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

# Reduce CUDA allocator fragmentation (safe across PyTorch versions).
export PYTORCH_ALLOC_CONF=expandable_segments:True
export RAY_object_store_memory=4000000000   # 4 GB — prevent Ray eating host RAM

# Smaller model option for stability/memory:
# - Qwen2.5-1.5B-Instruct (default, larger)
# - Qwen-1_8B             (available locally, but larger; not recommended for 2×V100 + ref)
#
# If you have a smaller Qwen2.5 model (e.g. 0.5B / 0.5B-Instruct), point MODEL to it.
MODEL=/ocean/projects/cis260009p/syan5/models/Qwen2.5-0.5B-Instruct
DATA_TRAIN=/jet/home/syan5/data/gsm8k/train.parquet
DATA_VAL=/jet/home/syan5/data/gsm8k/test.parquet

TOTAL_STEPS=20
BATCH_SIZE=8   # must be divisible by async-rollout worker count (=8 on 2×V100)
N_GPUS=2

run_one() {
  local tag=$1           # lp_ao | lp_ao_fp16
  local legacy_impl=$2   # enable (pull)
  local compress=$3      # fp16 | "" (unset)

  local exp_name="gsm8k_2gpu_stack_${tag}"
  local out_dir="checkpoints/verl_examples/${exp_name}"
  local probe_path="${out_dir}/transfer_probe.jsonl"

  if [[ -f "${out_dir}/train_log.txt" ]]; then
    echo "[skip] ${exp_name} already has train_log.txt — delete dir to rerun"
    return 0
  fi

  mkdir -p "${out_dir}"

  export VERL_TRANSFER_PROBE=1
  export VERL_TRANSFER_PROBE_LOG="${probe_path}"

  if [[ -n "${compress}" ]]; then
    export VERL_TRANSFER_COMPRESS="${compress}"
    local compress_label="${compress}"
  else
    unset VERL_TRANSFER_COMPRESS || true
    local compress_label="(unset)"
  fi

  echo ""
  echo "=============================================="
  echo "=== ${exp_name}"
  echo "=== legacy_worker_impl=${legacy_impl}  compress=${compress_label}"
  echo "=== batch=${BATCH_SIZE}  gpus=${N_GPUS}  steps=${TOTAL_STEPS}"
  echo "=== ref_worker=ON (use_kl_in_reward=True)"
  echo "=============================================="

  python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_files="${DATA_TRAIN}" \
    data.val_files="${DATA_VAL}" \
    data.train_batch_size="${BATCH_SIZE}" \
    data.max_prompt_length=256 \
    data.max_response_length=512 \
    data.shuffle=True \
    data.truncation=left \
    actor_rollout_ref.model.path="${MODEL}" \
    +actor_rollout_ref.model.override_config.torch_dtype=float16 \
    +actor_rollout_ref.model.override_config.attn_implementation=eager \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size="${N_GPUS}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.20 \
    actor_rollout_ref.rollout.max_model_len=768 \
    actor_rollout_ref.rollout.max_num_seqs="${BATCH_SIZE}" \
    actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
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
    trainer.use_legacy_worker_impl="${legacy_impl}" \
    reward_model.enable=False \
    2>&1 | tee "${out_dir}/train_log.txt"

  # Sanity: verify probe/AO/compression fired as expected.
  local probe_events compress_events stage_events
  probe_events=$(grep -c '\[transfer_probe\]' "${probe_path}" 2>/dev/null || echo 0)
  compress_events=$(grep -c '"event":[[:space:]]*"compress_stats"' "${probe_path}" 2>/dev/null || echo 0)
  stage_events=$(grep -c '"event":[[:space:]]*"critical_path_stage"' "${probe_path}" 2>/dev/null || echo 0)
  echo ""
  echo "[post-run] ${exp_name}:"
  echo "           probe_events        = ${probe_events}"
  echo "           critical_path_stage = ${stage_events}   (expect 0 for push_ref; >0 for lp_ao + lp_ao_fp16)"
  echo "           compress_stats      = ${compress_events}   (expect 0 for push_ref + lp_ao; >0 for lp_ao_fp16)"
}

# =======================================================================
# Driver
# =======================================================================

# Config 0: Push baseline — ref ON, LP/AO/Comp all OFF.
#   Driver blocks on every ref future → ref wait_ms fully on critical path.
#   Gives the Σwait_ms upper bound needed to measure the AO win.
run_one "push_ref" "disable" ""

# Config 1: LP + AO, no compression.  Establishes the pull+AO baseline.
#   ref_compute_ref_log_prob dispatched non-blocking → AO fires.
run_one "lp_ao"           "enable"  ""

# Config 2: Full stack — LP + AO + FP16.  The only run where all three
# transfer-protocol optimizations fire in the same iteration.
run_one "lp_ao_fp16"      "enable"  "fp16"

echo ""
echo "=============================================="
echo "=== COMBINED-STACK SWEEP COMPLETE"
echo "=============================================="
for tag in push_ref lp_ao lp_ao_fp16; do
  printf "  checkpoints/verl_examples/gsm8k_2gpu_stack_%s/\n" "${tag}"
done
echo ""
echo "Next: /ocean/projects/cis260009p/syan5/conda/project/bin/python \\"
echo "        scripts/analyze_combined_stack.py \\"
echo "        --out-fig poster_stack.png --out-md stack_report.md"
