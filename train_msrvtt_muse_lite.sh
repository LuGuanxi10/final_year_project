#!/usr/bin/env bash
set -euo pipefail

DATA_PATH=${DATA_PATH:-"/path/to/msrvtt"}
CUDA_DEVICES=${CUDA_DEVICES:-"0,1,2,3"}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-6662}

# Quick validation preset:
#   --pcme_train_samples 2 --pcme_eval_samples 8
# Formal training preset:
#   --pcme_train_samples 4 --pcme_eval_samples 16

CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} \
torchrun --nproc_per_node=${NPROC_PER_NODE} --master_port=${MASTER_PORT} \
main_task_retrieval.py --do_train --num_thread_reader=16 \
--epochs=5 --batch_size=128 --n_display=50 \
--train_csv ${DATA_PATH}/anns/MSRVTT_train.9k.csv \
--val_csv ${DATA_PATH}/anns/MSRVTT_JSFUSION_test.csv \
--data_path ${DATA_PATH}/MSR-VTT/anns/MSRVTT_data.json \
--features_path ${DATA_PATH}/Compressed_videos \
--output_dir ckpts/ckpt_msrvtt_muse_lite \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
--datatype msrvtt --expand_msrvtt_sentences \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0 --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header MUSE \
--pcme_train_samples 4 --pcme_eval_samples 16 \
--pcme_logsigma_min -7.0 --pcme_logsigma_max 7.0 \
--pcme_alpha_init 1.0 --pcme_beta_init 0.0 \
--pcme_lambda_match 1.0 --pcme_lambda_kl 5e-4 --pcme_lambda_unif 1e-3 \
--pcme_uniformity_t 2.0 \
--pretrained_clip_name ViT-B/32
