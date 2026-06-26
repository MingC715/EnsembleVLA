#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash policy/DP3/eval.sh <task_name> <task_config> <ckpt_setting> <expert_data_num> <seed> <gpu_id> [checkpoint_num] [ckpt_path]

policy_name=DP3
task_name=${1:?task_name is required}
task_config=${2:?task_config is required}
ckpt_setting=${3:?ckpt_setting is required}
expert_data_num=${4:?expert_data_num is required}
seed=${5:?seed is required}
gpu_id=${6:?gpu_id is required}
checkpoint_num=${7:-3000}
ckpt_path=${8:-""}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${script_dir}/../.." && pwd)"
log_dir="${root_dir}/log"
mkdir -p "${log_dir}"

log_file="${log_dir}/eval_DP3_${task_name}_ckpt${ckpt_setting}_N${expert_data_num}_seed${seed}_ckptnum${checkpoint_num}.log"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export HYDRA_FULL_ERROR=1
echo "GPU: ${gpu_id}"
echo "Log file: ${log_file}"

cd "${root_dir}"

python_args="--config policy/${policy_name}/deploy_policy.yml --overrides"
python_args="${python_args} --task_name ${task_name}"
python_args="${python_args} --task_config ${task_config}"
python_args="${python_args} --ckpt_setting ${ckpt_setting}"
python_args="${python_args} --expert_data_num ${expert_data_num}"
python_args="${python_args} --seed ${seed}"
python_args="${python_args} --checkpoint_num ${checkpoint_num}"
python_args="${python_args} --policy_name ${policy_name}"

if [ -n "${ckpt_path}" ]; then
  python_args="${python_args} --ckpt_path ${ckpt_path}"
fi

PYTHONWARNINGS=ignore::UserWarning \
nohup python -u script/eval_policy.py ${python_args} > "${log_file}" 2>&1 &

echo "Evaluation PID: $!"
echo "Tail log: tail -f ${log_file}"
