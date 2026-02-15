#!/usr/bin/env bash
set -euo pipefail

# ===============================
# CLIP4Clip 多GPU 训练脚本（tmux）
# ===============================
SESSION_NAME=clip4clip_train
CONDA_BASE="${HOME}/miniconda3"     # 如果你的 conda 安装路径不同，请修改
CONDA_ENV="clip4clip"              # conda 环境名，按需修改

# 数据路径：如果你希望从外部传入 DATA_PATH，可以在运行前导出；否则使用下面默认值
export DATA_PATH=${DATA_PATH:-/root/autodl-tmp/Datasets/MSR-VTT}

# ✔ 修改你的 checkpoint 文件名（epoch 1 保存的）
RESUME_MODEL=ckpts/msrvtt_run/pytorch_model.bin.0
RESUME_OPT=ckpts/msrvtt_run/pytorch_opt.bin.0

# 日志路径（会创建）
LOG_DIR=ckpts/msrvtt_run
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

# 清理同名 tmux 会话（如果存在）
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

# 生成实际在 tmux 中运行的脚本（绝对路径）
LAUNCHER="$(pwd)/run_in_tmux.sh"
cat > "$LAUNCHER" <<EOF
#!/usr/bin/env bash
set -euo pipefail

# 将日志全部写入文件（并且 tee 到终端）
exec > >(tee -a "$LOG_FILE") 2>&1

echo "===== STARTING tmux-runner ====="
echo "Date: \$(date)"
echo "Session: $SESSION_NAME"
echo "LOG_FILE: $LOG_FILE"
echo "DATA_PATH: $DATA_PATH"
echo

# 激活 conda 环境（确保使用正确的 python / site-packages）
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  echo "WARNING: conda.sh not found at $CONDA_BASE/etc/profile.d/conda.sh"
fi

conda activate "$CONDA_ENV" || { echo "Failed to activate conda env $CONDA_ENV"; exit 1; }

echo "Using python: \$(which python)"
python -V
echo "Conda env packages (pip show ftfy):"
python -c "import sys, ftfy; print('python executable:', sys.executable); print('ftfy:', ftfy.__version__, ftfy.__file__)"

echo

# 防止 NCCL 在容器/虚拟化网络下的常见问题
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO

# 进入项目目录
cd "$(pwd)"

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
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
  main_task_retrieval.py \
  --do_train \
  --resume_model $RESUME_MODEL \
  --num_thread_reader=8 \
  --epochs=5 \
  --batch_size=64 \
  --batch_size_val=64 \
  --n_display=50 \
  --train_csv "\${DATA_PATH}/MSRVTT_train.9k.csv" \
  --val_csv "\${DATA_PATH}/MSRVTT_JSFUSION_test.csv" \
  --data_path "\${DATA_PATH}/MSRVTT_data.json" \
  --features_path "\${DATA_PATH}/MSRVTT_Videos" \
  --output_dir ckpts/msrvtt_run \
  --lr 1e-4 \
  --coef_lr 1e-3 \
  --max_words 32 \
  --max_frames 12 \
  --datatype msrvtt \
  --expand_msrvtt_sentences \
  --feature_framerate 1 \
  --freeze_layer_num 0 \
  --slice_framepos 2 \
  --loose_type \
  --linear_patch 2d \
  --sim_header meanP \
  --pretrained_clip_name ViT-B/32

echo "===== TRAINING FINISHED ====="
echo "Log saved to: $LOG_FILE"
EOF

chmod +x "$LAUNCHER"

# 创建 tmux 会话并在其中运行 launcher 脚本
tmux new-session -d -s "$SESSION_NAME" "bash '$LAUNCHER'"

echo "✅ 训练已在 tmux 会话启动中: $SESSION_NAME"
echo "👉 使用以下命令查看训练进度:"
echo "   tmux attach -t $SESSION_NAME"
echo "👉 如果你不想 attach，可以实时查看日志："
echo "   tail -f $LOG_FILE"
echo "📜 日志路径: $LOG_FILE"
