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

has_extra_eval_arg() {
    local target="$1"
    local arg
    for arg in "${extra_eval_args[@]}"; do
        if [[ "${arg}" == "${target}" ]] || [[ "${arg}" == "${target}"=* ]]; then
            return 0
        fi
    done
    return 1
}

read_maniskill_eval_video_config() {
    local config_yaml="${SCRIPT_DIR}/diffusion_policy/config/maniskill_eval.yaml"
    python - "$config_yaml" <<'PY'
import sys
import yaml
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)

cfg = yaml.safe_load(path.read_text()) or {}
video = (((cfg.get("eval") or {}).get("video")) or {})
enabled = bool(video.get("enabled", False))
fps = int(video.get("fps", 10))
save_dir = str(video.get("save_dir", "eval_videos"))
print("1" if enabled else "0")
print(fps)
print(save_dir)
PY
}

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
    if ! [[ ${#positional[@]} -eq 2 || ${#positional[@]} -eq 3 || ${#positional[@]} -eq 5 || ${#positional[@]} -eq 6 ]]; then
        echo "Usage: $0 task_name gpu_id [seed] [--backend maniskill] [--task-profile PROFILE] [--ckpt-path CKPT] [--episodes N] [--max-episode-steps N] [-- EVAL_OVERRIDE ...]"
        echo "Legacy ManiSkill form is also accepted: $0 task_name task_config ckpt_setting expert_data_num gpu_id [seed] ..."
        exit 1
    fi
else
    if [ ${#positional[@]} -lt 6 ]; then
        echo "Usage: $0 task_name task_config ckpt_setting expert_data_num seed gpu_id [--backend robotwin|maniskill] [--task-profile PROFILE] [--ckpt-path CKPT] [--episodes N] [--max-episode-steps N] [-- EVAL_OVERRIDE ...]"
        exit 1
    fi
fi

task_name=${positional[0]}

if [ "${backend}" = "maniskill" ]; then
    task_config=
    ckpt_setting=
    expert_data_num=
    if [ ${#positional[@]} -eq 2 ]; then
        seed=42
        gpu_id=${positional[1]}
    elif [ ${#positional[@]} -eq 3 ]; then
        gpu_id=${positional[1]}
        seed=${positional[2]}
    elif [ ${#positional[@]} -eq 5 ]; then
        task_config=${positional[1]}
        ckpt_setting=${positional[2]}
        expert_data_num=${positional[3]}
        seed=42
        gpu_id=${positional[4]}
    else
        task_config=${positional[1]}
        ckpt_setting=${positional[2]}
        expert_data_num=${positional[3]}
        seed=${positional[4]}
        gpu_id=${positional[5]}
    fi
else
    task_config=${positional[1]}
    ckpt_setting=${positional[2]}
    expert_data_num=${positional[3]}
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

    if ! has_extra_eval_arg "--save-video-dir"; then
        mapfile -t eval_video_cfg < <(read_maniskill_eval_video_config)
        if [ ${#eval_video_cfg[@]} -eq 3 ] && [ "${eval_video_cfg[0]}" = "1" ]; then
            video_root="${ROOT_DIR}/${eval_video_cfg[2]}"
            task_label="${task_profile:-${runtime_task_name}}"
            ckpt_parent="$(basename "$(dirname "${ckpt_path}")")"
            ckpt_stem="$(basename "${ckpt_path}" .ckpt)"
            save_video_dir="${video_root}/fpo/${task_label}/${ckpt_parent}-${ckpt_stem}"
            cmd+=(--save-video-dir "${save_video_dir}")
            if ! has_extra_eval_arg "--video-fps"; then
                cmd+=(--video-fps "${eval_video_cfg[1]}")
            fi
        fi
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
