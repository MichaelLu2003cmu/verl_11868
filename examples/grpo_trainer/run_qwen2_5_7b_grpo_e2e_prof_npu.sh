set -x

# profiling configuration
PROFILE_STEPS="[2,4]"
PROFILE_RANKS_ALL=True
DISCRETE=False
DIST_PROFILER_ENABLE=${DIST_PROFILER_ENABLE:-False}

# profiling NPU options
SAVE_PATH="$HOME/profile_data"
LEVEL="level0"
CONTENTS=['npu','cpu']
ANALYSIS=True
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-7B-Instruct"}
RAY_TMPDIR=${RAY_TMPDIR:-"/tmp/ray_${USER}"}
DATA_HOME=${DATA_HOME:-"${RAY_DATA_HOME:-$HOME/data}"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_HOME}/gsm8k/train.parquet"}
VAL_FILE=${VAL_FILE:-"${DATA_HOME}/gsm8k/test.parquet"}

if [[ ! -d "${MODEL_PATH}" ]]; then
    model_parent_dir="$(dirname "${MODEL_PATH}")"
    model_base_name="$(basename "${MODEL_PATH}")"
    alt_base_name="${model_base_name/_/.}"
    alt_model_path="${model_parent_dir}/${alt_base_name}"

    if [[ -d "${alt_model_path}" ]]; then
        echo "[run] MODEL_PATH not found, fallback to: ${alt_model_path}"
        MODEL_PATH="${alt_model_path}"
    elif [[ "${MODEL_PATH}" != /* ]] && [[ "${MODEL_PATH}" == */* ]]; then
        # Allow Hugging Face model IDs like "Qwen/Qwen2.5-1.5B-Instruct".
        echo "[run] MODEL_PATH is not a local dir, treat as HF repo id: ${MODEL_PATH}"
    else
        echo "[run][error] MODEL_PATH directory not found: ${MODEL_PATH}"
        echo "[run][hint] Please verify model directory on this node."
        exit 1
    fi
fi

mkdir -p "${RAY_TMPDIR}"

# Avoid device-visibility env conflicts inside Ray workers.
# verl worker init forbids ROCR_VISIBLE_DEVICES with HIP/CUDA_VISIBLE_DEVICES.
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]] && [[ -n "${CUDA_VISIBLE_DEVICES:-}${HIP_VISIBLE_DEVICES:-}" ]]; then
    echo "[run] Detected ROCR_VISIBLE_DEVICES with HIP/CUDA_VISIBLE_DEVICES; unsetting ROCR_VISIBLE_DEVICES."
    unset ROCR_VISIBLE_DEVICES
fi

if [[ ! -f "${TRAIN_FILE}" ]] && [[ -n "${RAY_DATA_HOME:-}" ]]; then
    for candidate in \
        "${RAY_DATA_HOME}/data/gsm8k/train.parquet" \
        "${RAY_DATA_HOME}/dataset/gsm8k/train.parquet"; do
        if [[ -f "${candidate}" ]]; then
            echo "[run] TRAIN_FILE not found, fallback to: ${candidate}"
            TRAIN_FILE="${candidate}"
            break
        fi
    done
fi

if [[ ! -f "${VAL_FILE}" ]] && [[ -n "${RAY_DATA_HOME:-}" ]]; then
    for candidate in \
        "${RAY_DATA_HOME}/data/gsm8k/test.parquet" \
        "${RAY_DATA_HOME}/dataset/gsm8k/test.parquet"; do
        if [[ -f "${candidate}" ]]; then
            echo "[run] VAL_FILE not found, fallback to: ${candidate}"
            VAL_FILE="${candidate}"
            break
        fi
    done
fi

if [[ ! -f "${TRAIN_FILE}" ]]; then
    echo "[run][error] TRAIN_FILE not found: ${TRAIN_FILE}"
    echo "[run][hint] export TRAIN_FILE=/path/to/gsm8k/train.parquet"
    exit 1
fi

if [[ ! -f "${VAL_FILE}" ]]; then
    echo "[run][error] VAL_FILE not found: ${VAL_FILE}"
    echo "[run][hint] export VAL_FILE=/path/to/gsm8k/test.parquet"
    exit 1
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=4 \
    data.max_prompt_length=128 \
    data.max_response_length=64 \
    data.filter_overlong_prompts=False \
    data.truncation='left' \
    data.dataloader_num_workers=0 \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.tokenizer_path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.override_config.attn_implementation=eager \
    actor_rollout_ref.actor.optim.lr=5e-8 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.profiler.enable=${DIST_PROFILER_ENABLE} \
    actor_rollout_ref.actor.profiler.all_ranks=$PROFILE_RANKS_ALL \
    actor_rollout_ref.actor.profiler.tool_config.npu.discrete=$DISCRETE \
    actor_rollout_ref.actor.profiler.tool_config.npu.contents=$CONTENTS \
    actor_rollout_ref.actor.profiler.tool_config.npu.level=$LEVEL \
    actor_rollout_ref.actor.profiler.tool_config.npu.analysis=$ANALYSIS \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_model_len=256 \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=256 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.profiler.enable=${DIST_PROFILER_ENABLE} \
    actor_rollout_ref.ref.profiler.all_ranks=$PROFILE_RANKS_ALL \
    actor_rollout_ref.ref.profiler.tool_config.npu.discrete=$DISCRETE \
    actor_rollout_ref.ref.profiler.tool_config.npu.contents=$CONTENTS \
    actor_rollout_ref.ref.profiler.tool_config.npu.level=$LEVEL \
    actor_rollout_ref.ref.profiler.tool_config.npu.analysis=$ANALYSIS \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen2_5_7b_function_rm' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    ++ray_kwargs.ray_init._temp_dir="${RAY_TMPDIR}" \
    ++ray_kwargs.ray_init.include_dashboard=False \
    "$@"