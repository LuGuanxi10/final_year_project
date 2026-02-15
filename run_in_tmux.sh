#!/usr/bin/env bash
set -euo pipefail

# 将日志全部写入文件（并且 tee 到终端）
exec > >(tee -a "ckpts/msrvtt_run/train_20251029_124814.log") 2>&1

echo "===== STARTING tmux-runner ====="
echo "Date: $(date)"
echo "Session: clip4clip_train"
echo "LOG_FILE: ckpts/msrvtt_run/train_20251029_124814.log"
echo "DATA_PATH: /root/autodl-tmp/Datasets/MSR-VTT"
echo

# 激活 conda 环境（确保使用正确的 python / site-packages）
if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1090
  source "/root/miniconda3/etc/profile.d/conda.sh"
else
  echo "WARNING: conda.sh not found at /root/miniconda3/etc/profile.d/conda.sh"
fi

conda activate "clip4clip" || { echo "Failed to activate conda env clip4clip"; exit 1; }

echo "Using python: $(which python)"
python -V
echo "Conda env packages (pip show ftfy):"
python -c "import sys, ftfy; print('python executable:', sys.executable); print('ftfy:', ftfy.__version__, ftfy.__file__)"

echo

# 防止 NCCL 在容器/虚拟化网络下的常见问题
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO

# 进入项目目录
cd "/root/autodl-tmp/CLIP4Clip_reproduce"

# （可选）先做一次小的环境检查（打印 torch/cuda 信息）
python - <<PY
import torch,sys
print('torch.__version__=', getattr(torch,'__version__',None))
print('torch.version.cuda=', getattr(torch.version,'cuda',None))
print('cuda available=', torch.cuda.is_available())
print('cuda device count=', torch.cuda.device_count())
if torch.cuda.is_available():
    try:
        print('device name 0:', torch.cuda.get_device_name(0))
    except Exception as e:
        print('get_device_name error:', e)
PY

# 实际训练命令（双 GPU）——根据需要修改超参
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2   main_task_retrieval.py   --do_train   --resume_model ckpts/msrvtt_run/pytorch_model.bin.0   --num_thread_reader=8   --epochs=5   --batch_size=64   --batch_size_val=64   --n_display=50   --train_csv "${DATA_PATH}/MSRVTT_train.9k.csv"   --val_csv "${DATA_PATH}/MSRVTT_JSFUSION_test.csv"   --data_path "${DATA_PATH}/MSRVTT_data.json"   --features_path "${DATA_PATH}/MSRVTT_Videos"   --output_dir ckpts/msrvtt_run   --lr 1e-4   --coef_lr 1e-3   --max_words 32   --max_frames 12   --datatype msrvtt   --expand_msrvtt_sentences   --feature_framerate 1   --freeze_layer_num 0   --slice_framepos 2   --loose_type   --linear_patch 2d   --sim_header meanP   --pretrained_clip_name ViT-B/32

echo "===== TRAINING FINISHED ====="
echo "Log saved to: ckpts/msrvtt_run/train_20251029_124814.log"
