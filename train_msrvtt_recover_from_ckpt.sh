#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_ROOT=${RUNS_ROOT:-"${PROJECT_ROOT}/runs"}
RUN_DIR="${RUNS_ROOT}/$(date +%Y%m%d_%H%M%S)_recover"

DATA_PATH=${DATA_PATH:-"/path/to/msrvtt"}
BEST_CKPT=${BEST_CKPT:-""}
CUDA_DEVICES=${CUDA_DEVICES:-"0,1,2,3"}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-6672}

if [[ -z "${BEST_CKPT}" ]]; then
  echo "BEST_CKPT is required."
  exit 1
fi

if [[ ! -f "${BEST_CKPT}" ]]; then
  echo "BEST_CKPT does not exist: ${BEST_CKPT}"
  exit 1
fi

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/ckpts"

(
  echo "==== GIT ===="
  git -C "${PROJECT_ROOT}" rev-parse HEAD || true
  git -C "${PROJECT_ROOT}" status -s || true
  echo "==== CKPT ===="
  echo "${BEST_CKPT}"
  echo "==== NV ===="
  nvidia-smi || true
  echo "==== PYTHON ===="
  python --version || true
  python -c "import torch; print('torch', torch.__version__)" || true
  torchrun --version || true
) > "${RUN_DIR}/logs/env.txt" 2>&1

CONDA_ENV="myexp"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

CMD=(
  torchrun
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_port="${MASTER_PORT}"
  main_task_retrieval.py
  --do_train
  --num_thread_reader=16
  --epochs=5
  --batch_size=128
  --n_display=50
  --train_csv "${DATA_PATH}/MSRVTT_train.9k.csv"
  --val_csv "${DATA_PATH}/MSRVTT_JSFUSION_test.csv"
  --data_path "${DATA_PATH}/MSR-VTT/MSRVTT_data.json"
  --features_path "${DATA_PATH}/MSRVTT_Videos"
  --output_dir "${RUN_DIR}/ckpts"
  --init_model "${BEST_CKPT}"
  --init_state_scope clip_only
  --lr 1e-4
  --lr_clip 3e-6
  --lr_new_modules 3e-4
  --warmup_proportion 0.05
  --max_words 32
  --max_frames 12
  --batch_size_val 16
  --datatype msrvtt
  --expand_msrvtt_sentences
  --feature_framerate 1
  --freeze_layer_num 0
  --slice_framepos 2
  --loose_type
  --linear_patch 2d
  --sim_header MUSE
  --train_scope clip_only
  --pcme_mode deterministic
  --muse_mix_target 0.0
  --muse_warmup_epochs 0
  --muse_ramp_epochs 0
  --pcme_prob_mix_target 0.0
  --pcme_prob_warmup_epochs 0
  --pcme_prob_ramp_epochs 0
  --pcme_enable_aux_loss false
  --pcme_aux_warmup_epochs 99
  --pcme_train_samples 4
  --pcme_eval_samples 16
  --pcme_logsigma_min -7.0
  --pcme_logsigma_max 7.0
  --pcme_logsigma_bias_init -5.0
  --pcme_alpha_init 1.0
  --pcme_beta_init 0.0
  --pcme_lambda_match 0.0
  --pcme_lambda_kl 0.0
  --pcme_lambda_unif 0.0
  --pcme_uniformity_t 2.0
  --pretrained_clip_name ViT-B/32
)

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

printf "%q " "${CMD[@]}" > "${RUN_DIR}/logs/cmd.txt"
echo >> "${RUN_DIR}/logs/cmd.txt"

echo "RUN_DIR=${RUN_DIR}"
set -x
"${CMD[@]}" 2>&1 | tee -a "${RUN_DIR}/logs/train.log"
echo "__TRAINING_DONE__ $(date -Is)" | tee -a "${RUN_DIR}/logs/train.log"
