#!/bin/bash

backend=robotwin
task_profile=
config_name=
positional=()
hydra_overrides=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --)
            shift
            hydra_overrides=("$@")
            break
            ;;
        --backend)
            backend="$2"
            shift 2
            ;;
        --task-profile)
            task_profile="$2"
            shift 2
            ;;
        --config-name)
            config_name="$2"
            shift 2
            ;;
        *)
            positional+=("$1")
            shift
            ;;
    esac
done

if [ ${#positional[@]} -lt 6 ]; then
    echo "Usage: $0 task_name task_config expert_data_num seed action_dim gpu_id [resume_ckpt_path] [--backend robotwin|maniskill] [--task-profile PROFILE] [--config-name CONFIG] [-- HYDRA_OVERRIDE ...]"
    exit 1
fi

task_name=${positional[0]}
task_config=${positional[1]}
expert_data_num=${positional[2]}
seed=${positional[3]}
action_dim=${positional[4]}
gpu_id=${positional[5]}
resume_ckpt_path=${positional[6]:-null}  # 可选：指定恢复训练的 checkpoint 路径，例如 checkpoints/handover_block-demo_clean-50-42/300.ckpt

head_camera_type=D435

DEBUG=False
save_ckpt=True

alg_name=robot_fpo_$action_dim
if [ -z "${config_name}" ]; then
    if [ "${backend}" = "maniskill" ]; then
        config_name=maniskill_fpo
    else
        config_name=${alg_name}
    fi
fi
addition_info=train
exp_backend=${backend}
exp_name=${task_name}-${exp_backend}_fpo-${addition_info}
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
FLOWMATCHING_DIR="${SCRIPT_DIR}/../FlowMatching"

zarr_path="${SCRIPT_DIR}/data/${task_name}-${task_config}-${expert_data_num}.zarr"

if [ ! -d "${zarr_path}" ]; then
    bash "${SCRIPT_DIR}/process_data.sh" "${task_name}" "${task_config}" "${expert_data_num}" --backend "${backend}"
fi

if [ "${resume_ckpt_path}" != "null" ] && [[ "${resume_ckpt_path}" != /* ]]; then
    resume_ckpt_path="${FLOWMATCHING_DIR}/${resume_ckpt_path}"
fi

hydra_task_name=${task_name}
if [ -n "${task_profile}" ]; then
    hydra_task_name=${task_profile}
fi

if [ "${backend}" = "maniskill" ]; then
    python "${SCRIPT_DIR}/train.py" --config-name=${config_name}.yaml \
                                task.name=${hydra_task_name} \
                                task.dataset.zarr_path="${zarr_path}" \
                                training.debug=$DEBUG \
                                training.seed=${seed} \
                                training.device="cuda:0" \
                                exp_name=${exp_name} \
                                logging.mode=${wandb_mode} \
                                training.resume_ckpt_path=${resume_ckpt_path} \
                                "${hydra_overrides[@]}"
else
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
                                training.resume_ckpt_path=${resume_ckpt_path} \
                                "${hydra_overrides[@]}"
fi
                            # checkpoint.save_ckpt=${save_ckpt}
                            # hydra.run.dir=${run_dir} \
