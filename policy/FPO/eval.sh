#!/bin/bash

policy_name=FPO # [TODO] 
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

backend=robotwin
task_profile=
ckpt_path=
episodes=1
max_episode_steps=50
positional=()
extra_eval_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --)
            shift
            extra_eval_args=("$@")
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
        --ckpt-path)
            ckpt_path="$2"
            shift 2
            ;;
        --episodes)
            episodes="$2"
            shift 2
            ;;
        --max-episode-steps)
            max_episode_steps="$2"
            shift 2
            ;;
        *)
            positional+=("$1")
            shift
            ;;
    esac
done

if [ "${backend}" = "maniskill" ]; then
    if [ ${#positional[@]} -lt 5 ]; then
        echo "Usage: $0 task_name task_config ckpt_setting expert_data_num gpu_id [seed] [--backend robotwin|maniskill] [--task-profile PROFILE] [--ckpt-path CKPT] [--episodes N] [--max-episode-steps N] [-- EVAL_OVERRIDE ...]"
        exit 1
    fi
else
    if [ ${#positional[@]} -lt 6 ]; then
        echo "Usage: $0 task_name task_config ckpt_setting expert_data_num seed gpu_id [--backend robotwin|maniskill] [--task-profile PROFILE] [--ckpt-path CKPT] [--episodes N] [--max-episode-steps N] [-- EVAL_OVERRIDE ...]"
        exit 1
    fi
fi

task_name=${positional[0]}
task_config=${positional[1]}
ckpt_setting=${positional[2]}
expert_data_num=${positional[3]}

if [ "${backend}" = "maniskill" ] && [ ${#positional[@]} -eq 5 ]; then
    seed=42
    gpu_id=${positional[4]}
else
    seed=${positional[4]}
    gpu_id=${positional[5]}
fi
DEBUG=False
# [TODO] add parameters here

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd "${ROOT_DIR}"

if [ "${backend}" = "maniskill" ]; then
    if [ -z "${ckpt_path}" ]; then
        echo "--ckpt-path is required when --backend maniskill"
        exit 1
    fi

    runtime_task_name="${task_name}"
    if [ -n "${task_profile}" ]; then
        runtime_task_name="${task_profile}"
    fi

    cmd=(
        python script/eval_policy.py
        --backend maniskill
        fpo
        "${ckpt_path}"
        "${runtime_task_name}"
        --episodes "${episodes}"
        --max-episode-steps "${max_episode_steps}"
        --seed "${seed}"
    )

    if [ -n "${task_profile}" ]; then
        cmd+=(--task-profile "${task_profile}")
    fi

    cmd+=("${extra_eval_args[@]}")
    PYTHONWARNINGS=ignore::UserWarning "${cmd[@]}"
else
    PYTHONWARNINGS=ignore::UserWarning \
    python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
        --overrides \
        --task_name ${task_name} \
        --task_config ${task_config} \
        --ckpt_setting ${ckpt_setting} \
        --expert_data_num ${expert_data_num} \
        --seed ${seed} \
        --policy_name ${policy_name} \
        "${extra_eval_args[@]}"
        # [TODO] add parameters here
fi
