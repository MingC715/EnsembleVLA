#!/bin/bash
# Ensemble Policy Evaluation Script
# 
# bash policy/Ensemble-Policy-easy/eval.sh <task> <policy1> <policy2> <gpu> [ckpt] [camera] [data_num] [seed] [test_num] [policy1_ckpt] [policy2_ckpt]
#
#   gpu:          GPU ID
#
# nohup bash policy/Ensemble-Policy-easy/eval.sh beat_block_hammer DP DP3 2 > beat_block_hammer_dp_dp3_eval.log 2>&1 &
# nohup bash policy/Ensemble-Policy-easy/eval.sh open_laptop DP DP3 6 > open_laptop_dp_dp3_eval.log 2>&1 &
# nohup bash policy/Ensemble-Policy-easy/eval.sh click_alarmclock DP DP3 7 > click_alarmclock_dp_dp3_eval.log 2>&1 &
# nohup bash policy/Ensemble-Policy-easy/eval.sh move_playingcard_away DP DP3 1 > move_playingcard_away_dp_dp3_eval.log 2>&1 &
# nohup bash policy/Ensemble-Policy-easy/eval.sh dump_bin_bigbin DP DP3 7 > dump_bin_bigbin_dp_dp3_eval.log 2>&1 &
# nohup bash policy/Ensemble-Policy-easy/eval.sh handover_block DP DP3 1 > handover_block_dp_dp3_eval.log 2>&1 &
# nohup bash policy/Ensemble-Policy-easy/eval.sh lift_pot DP DP3 1 > lift_pot_dp_dp3_eval.log 2>&1 &
#
# nohup bash policy/Ensemble-Policy-easy/eval.sh beat_block_hammer DP DP3 1 epoch_1 > beat_block_hammer_dp_dp3_eval_epoch_1.log 2>&1 &
# nohup bash policy/Ensemble-Policy-easy/eval.sh beat_block_hammer DP DP3 2 epoch_5 > beat_block_hammer_dp_dp3_eval_epoch_5.log 2>&1 &
# nohup bash policy/Ensemble-Policy-easy/eval.sh beat_block_hammer DP DP3 1 epoch_10 > beat_block_hammer_dp_dp3_eval_epoch_10.log 2>&1 &
#
# nohup bash policy/Ensemble-Policy-easy/eval.sh dump_bin_bigbin DP DP3 1 best L515 100 0 100 300 100 > dump_bin_bigbin_dp300_dp3_100_eval.log 2>&1 &
# nohup bash policy/Ensemble-Policy-easy/eval.sh handover_block DP DP3 7 best L515 100 0 100 300 100 > handover_block_dp300_dp3_100_eval.log 2>&1 &
# nohup bash policy/Ensemble-Policy-easy/eval.sh lift_pot DP DP3 2 best L515 100 0 100 300 100 > lift_pot_dp300_dp3_100_eval.log 2>&1 &

# nohup bash policy/Ensemble-Policy-easy/eval.sh dump_bin_bigbin DP DP3 0 epoch_1 L515 100 0 100 300 100 > dump_bin_bigbin_dp300_dp3_100_eval_epoch_1.log 2>&1 &
# nohup bash policy/Ensemble-Policy-easy/eval.sh handover_block DP DP3 0 epoch_1 L515 100 0 100 300 100 > handover_block_dp300_dp3_100_eval_epoch_1.log 2>&1 &
# nohup bash policy/Ensemble-Policy-easy/eval.sh lift_pot DP DP3 0 epoch_1 L515 100 0 100 300 100 > lift_pot_dp300_dp3_100_eval_epoch_1.log 2>&1 &
# nohup bash policy/Ensemble-Policy-easy/eval.sh click_alarmclock DP DP3 4 epoch_1 L515 100 0 100 100 3000 > click_alarmclock_dp100_dp3_3000_eval_epoch_1.log 2>&1 &


# nohup bash policy/Ensemble-Policy-easy/eval.sh dump_bin_bigbin DP DP3 0 epoch_5 L515 100 0 100 300 100 > dump_bin_bigbin_dp300_dp3_100_eval_epoch_5.log 2>&1 &
# nohup bash policy/Ensemble-Policy-easy/eval.sh handover_block DP DP3 0 epoch_5 L515 100 0 100 300 100 > handover_block_dp300_dp3_100_eval_epoch_5.log 2>&1 &
# nohup bash policy/Ensemble-Policy-easy/eval.sh lift_pot DP DP3 0 epoch_5 L515 100 0 100 300 100 > lift_pot_dp300_dp3_100_eval_epoch_5.log 2>&1 &
# nohup bash policy/Ensemble-Policy-easy/eval.sh click_alarmclock DP DP3 0 epoch_5 L515 100 0 100 100 3000 > click_alarmclock_dp100_dp3_3000_eval_epoch_5.log 2>&1 &

# nohup bash policy/Ensemble-Policy-easy/eval.sh dump_bin_bigbin DP DP3 5 epoch_10 L515 100 0 100 300 100 > dump_bin_bigbin_dp300_dp3_100_eval_epoch_10.log 2>&1 &
# nohup bash policy/Ensemble-Policy-easy/eval.sh handover_block DP DP3 5 epoch_10 L515 100 0 100 300 100 > handover_block_dp300_dp3_100_eval_epoch_10.log 2>&1 &
# nohup bash policy/Ensemble-Policy-easy/eval.sh lift_pot DP DP3 5 epoch_10 L515 100 0 100 300 100 > lift_pot_dp300_dp3_100_eval_epoch_10.log 2>&1 &
# nohup bash policy/Ensemble-Policy-easy/eval.sh click_alarmclock DP DP3 5 epoch_10 L515 100 0 100 100 3000 > click_alarmclock_dp100_dp3_3000_eval_epoch_10.log 2>&1 &

# nohup bash policy/Ensemble-Policy-easy/eval.sh dump_bin_bigbin DP DP3 0 epoch_20 L515 100 0 100 300 100 > dump_bin_bigbin_dp300_dp3_100_eval_epoch_20.log 2>&1 &
# nohup bash policy/Ensemble-Policy-easy/eval.sh handover_block DP DP3 6 epoch_20 L515 100 0 100 300 100 > handover_block_dp300_dp3_100_eval_epoch_20.log 2>&1 &
# nohup bash policy/Ensemble-Policy-easy/eval.sh lift_pot DP DP3 7 epoch_20 L515 100 0 100 300 100 > lift_pot_dp300_dp3_100_eval_epoch_20.log 2>&1 &
# nohup bash policy/Ensemble-Policy-easy/eval.sh dump_bin_bigbin DP DP3 1 epoch_50 L515 100 0 100 300 100 > dump_bin_bigbin_dp300_dp3_100_eval_epoch_50.log 2>&1 &
#
# nohup bash policy/Ensemble-Policy-easy/eval.sh beat_block_hammer DP PI05 1 > beat_block_hammer_dp_pi05_eval.log 2>&1 &
# nohup bash policy/Ensemble-Policy-easy/eval.sh click_alarmclock DP PI05 2 > click_alarmclock_dp_pi05_eval.log 2>&1 &
# nohup bash policy/Ensemble-Policy-easy/eval.sh click_alarmclock DP3 PI05 4 > click_alarmclock_dp3_pi05_eval.log 2>&1 &
# nohup bash policy/Ensemble-Policy-easy/eval.sh open_laptop DP PI05 3 > open_laptop_dp_pi05_eval.log 2>&1 &
# nohup bash policy/Ensemble-Policy-easy/eval.sh move_playingcard_away DP PI05 4 > move_playingcard_away_dp_pi05_eval.log 2>&1 &
# nohup bash policy/Ensemble-Policy-easy/eval.sh dump_bin_bigbin DP PI05 5 > dump_bin_bigbin_dp_pi05_eval.log 2>&1 &
# nohup bash policy/Ensemble-Policy-easy/eval.sh handover_block DP PI05 6 > handover_block_dp_pi05_eval.log 2>&1 &
# nohup bash policy/Ensemble-Policy-easy/eval.sh lift_pot DP PI05 7 > lift_pot_dp_pi05_eval.log 2>&1 &
#
# nohup bash policy/Ensemble-Policy-easy/eval.sh click_alarmclock DP PI05 3 best L515 100 0 100 100 30000 > click_alarmclock_dp100_pi05_30000_eval.log 2>&1 &
# nohup bash policy/Ensemble-Policy-easy/eval.sh handover_block DP PI05 6 best L515 100 0 100 300 30000 > handover_block_dp300_pi05_30000_eval.log 2>&1 &
# nohup bash policy/Ensemble-Policy-easy/eval.sh lift_pot DP PI05 7 best L515 100 0 100 600 30000 > lift_pot_dp600_pi05_30000_eval.log 2>&1 &
 

#  cd /path/to/EnsembleVLA-ICML2026

# nohup bash policy/Ensemble-Policy-easy/eval.sh click_alarmclock DP3 pi05 1 epoch_1 L515 100 0 100 3000 30000 > click_alarmclock_dp3_pi05_eval_epoch_1.log 2>&1 &


set -e

source $(conda info --base)/etc/profile.d/conda.sh
conda activate RoboTwin

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PID: $$"
echo ""

TASK=${1:-"beat_block_hammer"}
POLICY1=${2:-"DP"}
POLICY2=${3:-"DP3"}
GPU=${4:-0}
CKPT=${5:-"best"}
CAMERA=${6:-"L515"}
DATA_NUM=${7:-100}
SEED=${8:-0}
TEST_NUM=${9:-100}
POLICY1_CKPT=${10:-""}
POLICY2_CKPT=${11:-""}

if [ -z "${POLICY1_CKPT}" ]; then
    case ${POLICY1} in
        DP)
            POLICY1_CKPT=300
            ;;
        DP3)
            POLICY1_CKPT=3000
            ;;
        pi05|PI05)
            POLICY1_CKPT=30000
            ;;
        *)
            POLICY1_CKPT=300
            ;;
    esac
fi

if [ -z "${POLICY2_CKPT}" ]; then
    case ${POLICY2} in
        DP)
            POLICY2_CKPT=300
            ;;
        DP3)
            POLICY2_CKPT=3000
            ;;
        pi05|PI05)
            POLICY2_CKPT=30000
            ;;
        *)
            POLICY2_CKPT=3000
            ;;
    esac
fi

echo -e "\033[0;32m========================================"
echo "  Ensemble Policy Evaluation"
echo -e "========================================\033[0m"
echo ""
echo "Configuration:"
echo "  Task:                ${TASK}"
echo "  Policy 1:            ${POLICY1} (checkpoint: ${POLICY1_CKPT})"
echo "  Policy 2:            ${POLICY2} (checkpoint: ${POLICY2_CKPT})"
echo "  Ensemble Checkpoint: ${CKPT}"
echo "  Camera:              ${CAMERA}"
echo "  Data Num:            ${DATA_NUM}"
echo "  Seed:                ${SEED}"
echo "  Test Num:            ${TEST_NUM}"
echo "  GPU:                 ${GPU}"
echo ""

export CUDA_VISIBLE_DEVICES=${GPU}
export CUROBO_TORCH_COMPILE=0
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export SAPIEN_OFFSCREEN_ONLY=1

POLICY1_DIR="${POLICY1}"
POLICY2_DIR="${POLICY2}"
if [ "${POLICY1}" == "PI05" ]; then
    POLICY1_DIR="pi05"
fi
if [ "${POLICY2}" == "PI05" ]; then
    POLICY2_DIR="pi05"
fi

OUTPUT_DIR="./policy/Ensemble-Policy-easy/checkpoints/${TASK}_${POLICY1_DIR}${POLICY1_CKPT}_${POLICY2_DIR}${POLICY2_CKPT}"

echo -e "\033[0;34m>>> Checkpoint directory: ${OUTPUT_DIR}\033[0m"
echo -e "\033[0;34m>>> Starting evaluation...\033[0m"
echo ""

python policy/Ensemble-Policy-easy/eval.py \
    --config-name=ensemble_config \
    task.name=${TASK} \
    policy1.type=${POLICY1} \
    +policy1.checkpoint_num=${POLICY1_CKPT} \
    policy2.type=${POLICY2} \
    +policy2.checkpoint_num=${POLICY2_CKPT} \
    head_camera_type=${CAMERA} \
    expert_data_num=${DATA_NUM} \
    training.seed=${SEED} \
    eval.checkpoint_num=${CKPT} \
    eval.test_num=${TEST_NUM} \
    +output_dir=${OUTPUT_DIR}

echo ""
echo -e "\033[0;32m>>> Evaluation completed!\033[0m"
