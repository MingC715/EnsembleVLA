#!/bin/bash
# Evaluate PyTorch checkpoint performance in RoboTwin simulation
#
# Usage: bash eval_pytorch_checkpoint.sh <task_name> <checkpoint_id> <gpu_id> [test_num] [seed]
# Example: bash eval_pytorch_checkpoint.sh click_alarmclock 30000 5 50 0
#!/bin/bash
# nohup bash eval_pytorch_checkpoint.sh click_alarmclock 30000 0 100 0 > eval_click_alarmclock_pytorch_30000_seed0.log 2>&1 &
# nohup bash eval_pytorch_checkpoint.sh beat_block_hammer 30000 5 100 0 > eval_beat_block_hammer_pytorch_30000_seed0_2.log 2>&1 &
# nohup bash eval_pytorch_checkpoint.sh dump_bin_bigbin 30000 2 100 0 > eval_dump_bin_bigbin_pytorch_30000_seed0.log 2>&1 &
# nohup bash eval_pytorch_checkpoint.sh handover_block 30000 5 100 0 > eval_handover_block_pytorch_30000_seed0.log 2>&1 &
# nohup bash eval_pytorch_checkpoint.sh move_playingcard_away 30000 6 100 0 > eval_move_playingcard_away_pytorch_30000_seed0_1.log 2>&1 &
# nohup bash eval_pytorch_checkpoint.sh open_laptop 30000 0 100 0 > eval_open_laptop_pytorch_30000_seed0.log 2>&1 &
# nohup bash eval_pytorch_checkpoint.sh stack_bowls_three 30000 6 100 0 > eval_stack_bowls_three_pytorch_30000_seed0.log 2>&1 &

# Run from this release tree: cd policy/pi05
# 新的评估命令
# nohup bash eval_pytorch_checkpoint.sh handover_block 1000 0 100 0 pi0_base_aloha_robotwin_full_pytorch > eval_handover_block_full_1000.log 2>&1 &

set -e

# Check arguments
if [ $# -lt 3 ]; then
    echo "Usage: bash eval_pytorch_checkpoint.sh <task_name> <checkpoint_id> <gpu_id> [test_num] [seed] [train_config_name]"
    echo "Example: bash eval_pytorch_checkpoint.sh click_alarmclock 30000 0 50 0"
    echo "Example: bash eval_pytorch_checkpoint.sh beat_block_hammer 1000 0 100 0 pi0_base_aloha_robotwin_full_pytorch"
    echo ""
    echo "Available tasks:"
    echo "  - beat_block_hammer"
    echo "  - click_alarmclock"
    echo "  - dump_bin_bigbin"
    echo "  - handover_block"
    echo "  - move_playingcard_away"
    echo "  - open_laptop"
    echo "  - stack_bowls_three"
    exit 1
fi

TASK_NAME=$1
CHECKPOINT_ID=$2
GPU_ID=$3
TEST_NUM=${4:-50}
SEED=${5:-0}
TRAIN_CONFIG_NAME=${6:-"pi0_base_aloha_robotwin_lora"}
TASK_CONFIG="demo_clean"
# CRITICAL FIX: MODEL_NAME should be TASK_NAME for correct checkpoint path
# The checkpoint is at: checkpoints/{TRAIN_CONFIG_NAME}/{TASK_NAME}/{CHECKPOINT_ID}_pytorch
MODEL_NAME="${TASK_NAME}"  # Use task name as model name

echo "============================================================"
echo "  Pi0.5 PyTorch Checkpoint Evaluation"
echo "============================================================"
echo "Task: $TASK_NAME"
echo "Checkpoint ID: $CHECKPOINT_ID"
echo "GPU: $GPU_ID"
echo "Test episodes: $TEST_NUM"
echo "Seed: $SEED"
echo "============================================================"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin

# Set environment variables
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
export CUROBO_TORCH_COMPILE=0
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export SAPIEN_OFFSCREEN_ONLY=1

# CRITICAL FIX: Disable torch.compile for evaluation to fix long-sequence bug (>400 steps)
# torch.compile causes state accumulation in repeated infer() calls
# This fixes 0% success rate on tasks like dump_bin_bigbin (600 steps)
export DISABLE_TORCH_COMPILE=1

# Base directory
PI05_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$PI05_DIR/../.." && pwd)"
CHECKPOINT_DIR="$BASE_DIR/policy/pi05/checkpoints/$TRAIN_CONFIG_NAME/$TASK_NAME/$CHECKPOINT_ID"
PYTORCH_DIR="${CHECKPOINT_DIR}_pytorch"

# Check if PyTorch checkpoint exists (支持两种目录结构)
# 1. .../checkpoint_id_pytorch/model.safetensors
# 2. .../checkpoint_id/checkpoint_id/model.safetensors (nested structure)
if [ -f "$PYTORCH_DIR/model.safetensors" ]; then
    echo "✓ Found PyTorch checkpoint: $PYTORCH_DIR"
elif [ -f "$CHECKPOINT_DIR/$CHECKPOINT_ID/model.safetensors" ]; then
    echo "✓ Found nested checkpoint: $CHECKPOINT_DIR/$CHECKPOINT_ID"
elif [ -f "$CHECKPOINT_DIR/model.safetensors" ]; then
    echo "✓ Found checkpoint: $CHECKPOINT_DIR"
else
    echo "✗ Error: Checkpoint not found"
    echo "  Tried: $PYTORCH_DIR"
    echo "  Tried: $CHECKPOINT_DIR/$CHECKPOINT_ID"
    echo "  Tried: $CHECKPOINT_DIR"
    exit 1
fi

# Move to RoboTwin root directory
cd "$BASE_DIR"

echo "============================================================"
echo "  Starting Evaluation"
echo "============================================================"
echo ""

# Run evaluation using the original eval.sh approach
# Note: model_name should be the task name for checkpoint path resolution
PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/pi05/deploy_policy.yml \
    --overrides \
    --task_name ${TASK_NAME} \
    --task_config ${TASK_CONFIG} \
    --train_config_name ${TRAIN_CONFIG_NAME} \
    --model_name ${TASK_NAME} \
    --ckpt_setting ${TASK_NAME} \
    --checkpoint_id ${CHECKPOINT_ID} \
    --seed ${SEED} \
    --policy_name pi05

echo ""
echo "============================================================"
echo "✓ Evaluation completed!"
echo "============================================================"
