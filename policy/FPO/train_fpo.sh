#!/bin/bash

backend=robotwin
task_profile=
config_name=
positional=()
hydra_overrides=()
short_maniskill_syntax=false

has_override_prefix() {
    local prefix="$1"
    local override
    for override in "${hydra_overrides[@]}"; do
        if [[ "${override}" == "${prefix}"* ]]; then
            return 0
        fi
    done
    return 1
}

get_task_group_override() {
    local override
    for override in "${hydra_overrides[@]}"; do
        case "${override}" in
            task=*|+task=*)
                echo "${override#*=}"
                return 0
                ;;
        esac
    done
    return 1
}

resolve_task_group_zarr_path() {
    local task_group="$1"
    local task_yaml="${SCRIPT_DIR}/diffusion_policy/config/task/${task_group}.yaml"
    local raw_path=

    if [ ! -f "${task_yaml}" ]; then
        return 1
    fi

    raw_path=$(
        awk '
            $1 == "maniskill:" { in_maniskill = 1; next }
            in_maniskill && $1 == "dataset:" { in_maniskill = 0 }
            in_maniskill && $1 == "zarr_path:" { print $2; exit }
        ' "${task_yaml}"
    )

    if [ -z "${raw_path}" ]; then
        return 1
    fi

    raw_path="${raw_path%\"}"
    raw_path="${raw_path#\"}"

    if [[ "${raw_path}" = /* ]]; then
        echo "${raw_path}"
    else
        echo "${SCRIPT_DIR}/${raw_path}"
    fi
}

resolve_process_triplet_from_zarr_path() {
    local resolved_zarr_path="$1"
    local data_prefix="${SCRIPT_DIR}/data/"
    local base stem expert_num remainder task_cfg task_nm

    if [[ "${resolved_zarr_path}" != "${data_prefix}"* ]]; then
        return 1
    fi

    base="$(basename "${resolved_zarr_path}")"
    [[ "${base}" == *.zarr ]] || return 1

    stem="${base%.zarr}"
    expert_num="${stem##*-}"
    [[ "${expert_num}" =~ ^[0-9]+$ ]] || return 1

    remainder="${stem%-${expert_num}}"
    [[ "${remainder}" != "${stem}" ]] || return 1

    task_cfg="${remainder##*-}"
    task_nm="${remainder%-${task_cfg}}"
    task_nm="${task_nm%-}"

    if [[ -z "${task_nm}" || -z "${task_cfg}" ]]; then
        return 1
    fi

    printf '%s\t%s\t%s\n' "${task_nm}" "${task_cfg}" "${expert_num}"
}

resolve_task_profile_zarr_path() {
    local profile_name="$1"
    local raw_path=

    raw_path=$(
        PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}" python -c \
        "from sim.maniskill.task_profiles import resolve_task_profile; p = resolve_task_profile('${profile_name}'); print(p['dataset']['zarr_path'] if p and 'dataset' in p else '')" \
        2>/dev/null
    )

    if [ -z "${raw_path}" ]; then
        return 1
    fi

    if [[ "${raw_path}" = /* ]]; then
        echo "${raw_path}"
    else
        echo "${SCRIPT_DIR}/${raw_path}"
    fi
}

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ "${backend}" = "maniskill" ]; then
    if ! [[ ${#positional[@]} -eq 2 || ${#positional[@]} -eq 3 || ${#positional[@]} -eq 6 || ${#positional[@]} -eq 7 ]]; then
        echo "Usage: $0 seed gpu_id [resume_ckpt_path] [--backend maniskill] [--task-profile PROFILE] [--config-name CONFIG] [-- HYDRA_OVERRIDE ...]"
        echo "Legacy ManiSkill form is also accepted: $0 task_name task_config expert_data_num seed action_dim gpu_id [resume_ckpt_path] ..."
        exit 1
    fi
else
    if [ ${#positional[@]} -lt 6 ]; then
        echo "Usage: $0 task_name task_config expert_data_num seed action_dim gpu_id [resume_ckpt_path] [--backend robotwin|maniskill] [--task-profile PROFILE] [--config-name CONFIG] [-- HYDRA_OVERRIDE ...]"
        exit 1
    fi
fi

task_name=
task_config=
expert_data_num=
seed=
action_dim=
gpu_id=
resume_ckpt_path=null

if [ "${backend}" = "maniskill" ]; then
    if [ ${#positional[@]} -eq 2 ] || [ ${#positional[@]} -eq 3 ]; then
        short_maniskill_syntax=true
        seed=${positional[0]}
        gpu_id=${positional[1]}
        resume_ckpt_path=${positional[2]:-null}
    else
        task_name=${positional[0]}
        task_config=${positional[1]}
        expert_data_num=${positional[2]}
        seed=${positional[3]}
        action_dim=${positional[4]}
        gpu_id=${positional[5]}
        resume_ckpt_path=${positional[6]:-null}
    fi
else
    task_name=${positional[0]}
    task_config=${positional[1]}
    expert_data_num=${positional[2]}
    seed=${positional[3]}
    action_dim=${positional[4]}
    gpu_id=${positional[5]}
    resume_ckpt_path=${positional[6]:-null}
fi

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
task_group_override=
if task_group_override="$(get_task_group_override)"; then
    :
fi

exp_task_label=${task_name}
if [ -z "${exp_task_label}" ]; then
    if [ -n "${task_group_override}" ]; then
        exp_task_label=${task_group_override}
    elif [ -n "${task_profile}" ]; then
        exp_task_label=${task_profile}
    else
        exp_task_label=${config_name}
    fi
fi

exp_name=${exp_task_label}-${exp_backend}_fpo-${addition_info}
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
FLOWMATCHING_DIR="${SCRIPT_DIR}/../FlowMatching"

zarr_path="${SCRIPT_DIR}/data/${task_name}-${task_config}-${expert_data_num}.zarr"

if [ "${resume_ckpt_path}" != "null" ] && [[ "${resume_ckpt_path}" != /* ]]; then
    resume_ckpt_path="${FLOWMATCHING_DIR}/${resume_ckpt_path}"
fi

hydra_task_name=${task_name}
if [ -n "${task_profile}" ]; then
    hydra_task_name=${task_profile}
fi

inject_task_name=true
inject_dataset_zarr_path=true
resolved_dataset_zarr_path="${zarr_path}"
if [ -n "${task_group_override}" ]; then
    inject_task_name=false

    if has_override_prefix "task.dataset.zarr_path=" || has_override_prefix "+task.dataset.zarr_path="; then
        inject_dataset_zarr_path=false
    else
        resolved_task_zarr_path="$(resolve_task_group_zarr_path "${task_group_override}")"
        if [ -n "${resolved_task_zarr_path}" ]; then
            resolved_dataset_zarr_path="${resolved_task_zarr_path}"
        else
            inject_dataset_zarr_path=false
        fi
    fi
else
    if [ "${short_maniskill_syntax}" = true ]; then
        inject_task_name=false
        inject_dataset_zarr_path=false
    fi
    if has_override_prefix "task.name=" || has_override_prefix "+task.name="; then
        inject_task_name=false
    fi
    if has_override_prefix "task.dataset.zarr_path=" || has_override_prefix "+task.dataset.zarr_path="; then
        inject_dataset_zarr_path=false
    fi
fi

if [ "${short_maniskill_syntax}" = true ] && [ -n "${task_profile}" ] && [ -z "${task_group_override}" ] && [ "${inject_dataset_zarr_path}" = false ]; then
    hydra_task_name=${task_profile}
    inject_task_name=true
    if ! has_override_prefix "task.dataset.zarr_path=" && ! has_override_prefix "+task.dataset.zarr_path="; then
        resolved_profile_zarr_path="$(resolve_task_profile_zarr_path "${task_profile}")"
        if [ -n "${resolved_profile_zarr_path}" ]; then
            resolved_dataset_zarr_path="${resolved_profile_zarr_path}"
            inject_dataset_zarr_path=true
        fi
    fi
fi

process_target_zarr_path=
if [ "${inject_dataset_zarr_path}" = true ]; then
    process_target_zarr_path="${resolved_dataset_zarr_path}"
fi

if [ -n "${process_target_zarr_path}" ] && [ ! -d "${process_target_zarr_path}" ]; then
    process_task_name="${task_name}"
    process_task_config="${task_config}"
    process_expert_data_num="${expert_data_num}"

    if process_triplet="$(resolve_process_triplet_from_zarr_path "${process_target_zarr_path}")"; then
        IFS=$'\t' read -r process_task_name process_task_config process_expert_data_num <<< "${process_triplet}"
    fi

    bash "${SCRIPT_DIR}/process_data.sh" \
        "${process_task_name}" \
        "${process_task_config}" \
        "${process_expert_data_num}" \
        --backend "${backend}"
fi

maniskill_overrides=(
    training.debug=$DEBUG
    training.seed=${seed}
    training.device="cuda:0"
    exp_name=${exp_name}
    logging.mode=${wandb_mode}
    training.resume_ckpt_path=${resume_ckpt_path}
)

if [ "${inject_task_name}" = true ]; then
    maniskill_overrides=(task.name=${hydra_task_name} "${maniskill_overrides[@]}")
fi

if [ "${inject_dataset_zarr_path}" = true ]; then
    maniskill_overrides=(task.dataset.zarr_path="${resolved_dataset_zarr_path}" "${maniskill_overrides[@]}")
fi

if [ "${backend}" = "maniskill" ]; then
    python "${SCRIPT_DIR}/train.py" --config-name=${config_name}.yaml \
                                "${maniskill_overrides[@]}" \
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
