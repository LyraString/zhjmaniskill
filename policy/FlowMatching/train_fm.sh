#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
action_dim=${5}
gpu_id=${6}
resume_ckpt_path=${7:-null}  # 可选：指定恢复训练的 checkpoint 路径，例如 checkpoints/handover_block-demo_clean-50-42/300.ckpt

head_camera_type=D435

DEBUG=False
save_ckpt=True

alg_name=robot_fm_$action_dim
config_name=${alg_name}
addition_info=train
exp_name=${task_name}-robot_fm-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

zarr_path="${SCRIPT_DIR}/data/${task_name}-${task_config}-${expert_data_num}.zarr"

if [ ! -d "${zarr_path}" ]; then
    bash "${SCRIPT_DIR}/process_data.sh" "${task_name}" "${task_config}" "${expert_data_num}"
fi

python "${SCRIPT_DIR}/train.py" --config-name=${config_name}.yaml \
                            task.name=${task_name} \
                            task.dataset.zarr_path="${zarr_path}" \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            setting=${task_config} \
                            expert_data_num=${expert_data_num} \
                            head_camera_type=$head_camera_type \
                            training.resume_ckpt_path=${resume_ckpt_path}
                            # checkpoint.save_ckpt=${save_ckpt}
                            # hydra.run.dir=${run_dir} \
