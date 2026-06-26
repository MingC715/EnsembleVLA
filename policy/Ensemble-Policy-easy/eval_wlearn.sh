#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash policy/Ensemble-Policy-easy/eval_wlearn.sh \
#     <task> <policy1> <policy2> <gpu> [ensemble_ckpt] [camera] [data_num] \
#     [seed] [test_num] [policy1_ckpt] [policy2_ckpt] [output_dir] \
#     [composition_mode] [policy2_use_pytorch]

task=${1:-beat_block_hammer}
policy1=${2:-DP}
policy2=${3:-DP3}
gpu=${4:-0}
ensemble_ckpt=${5:-best}
camera=${6:-L515}
data_num=${7:-100}
seed=${8:-0}
test_num=${9:-100}
policy1_ckpt=${10:-300}
policy2_ckpt=${11:-3000}
output_dir=${12:-""}
composition_mode=${13:-""}
policy2_use_pytorch=${14:-""}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-RoboTwin}"

export CUDA_VISIBLE_DEVICES="${gpu}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${HOME}/.cache/torch_extensions}"
export CUROBO_TORCH_COMPILE=0
export HYDRA_FULL_ERROR=1
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export SAPIEN_OFFSCREEN_ONLY=1
export SAPIEN_DISABLE_OIDN=1
export DISABLE_TORCH_COMPILE=1

if [ -z "${output_dir}" ]; then
  if [ "${policy1}" = "DP" ] && { [ "${policy2}" = "DP3" ] || [ "${policy2}" = "dp3" ]; }; then
    output_dir="./best_checkpoint/dp+dp3/${task}/ensemble_checkpoint"
  elif [ "${policy1}" = "DP" ] && { [ "${policy2}" = "pi05" ] || [ "${policy2}" = "pi0.5" ] || [ "${policy2}" = "PI05" ]; }; then
    output_dir="./best_checkpoint/dp+pi0.5/${task}/ensemble_checkpoint"
  fi
fi

if [ -z "${output_dir}" ]; then
  candidates=(
    "./policy/Ensemble-Policy-easy/checkpoints/${task}_${policy1}${policy1_ckpt}_${policy2}${policy2_ckpt}_wlearn"
    "./policy/Ensemble-Policy-easy/checkpoints/${task}_${policy1}${policy1_ckpt}_${policy2}${policy2_ckpt}"
    "./policy/Ensemble-Policy-easy/checkpoints/${task}_${policy1}_${policy2}"
  )
  for candidate in "${candidates[@]}"; do
    if [ -d "${candidate}" ]; then
      output_dir="${candidate}"
      break
    fi
  done
fi

if [ -z "${output_dir}" ] || [ ! -d "${output_dir}" ]; then
  echo "Could not find an ensemble checkpoint directory for ${task}." >&2
  echo "Pass output_dir as the twelfth argument." >&2
  exit 1
fi

if [ "${ensemble_ckpt}" = "best" ] && [ ! -f "${output_dir}/best.pt" ]; then
  mapfile -t pt_files < <(find "${output_dir}" -maxdepth 1 -type f -name '*.pt' | sort)
  if [ "${#pt_files[@]}" -eq 1 ]; then
    ensemble_ckpt=$(basename "${pt_files[0]}" .pt)
  fi
fi

extra_args=()
if [ -n "${composition_mode}" ]; then
  extra_args+=(composition_mode="${composition_mode}")
fi
if [ -n "${policy2_use_pytorch}" ]; then
  extra_args+=(policy2.use_pytorch="${policy2_use_pytorch}")
fi

printf 'Task: %s\n' "${task}"
printf 'Policies: %s(%s) + %s(%s)\n' "${policy1}" "${policy1_ckpt}" "${policy2}" "${policy2_ckpt}"
printf 'GPU: %s\n' "${gpu}"
printf 'Ensemble checkpoint: %s\n' "${ensemble_ckpt}"
printf 'Checkpoint directory: %s\n' "${output_dir}"
if [ -n "${composition_mode}" ]; then printf 'Composition mode: %s\n' "${composition_mode}"; fi

python policy/Ensemble-Policy-easy/eval_wlearn.py \
  --config-name=ensemble_config_wlearn \
  task.name="${task}" \
  policy1.type="${policy1}" \
  +policy1.checkpoint_num="${policy1_ckpt}" \
  policy2.type="${policy2}" \
  +policy2.checkpoint_num="${policy2_ckpt}" \
  head_camera_type="${camera}" \
  expert_data_num="${data_num}" \
  training.seed="${seed}" \
  eval.checkpoint_num="${ensemble_ckpt}" \
  eval.test_num="${test_num}" \
  output_dir="${output_dir}" \
  "${extra_args[@]}"
