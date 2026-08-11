#!/bin/bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 seed gpu_id_or_list base_checkpoint [-- HYDRA_OVERRIDE ...]"
    echo "Use one GPU id (1) or a comma-separated distributed list (1,3)."
    exit 1
fi

seed="$1"
gpu_ids="$2"
base_checkpoint="$3"
shift 3
if [ "${1:-}" = "--" ]; then
    shift
fi
hydra_overrides=("$@")

if [ -z "${gpu_ids}" ] || [[ "${gpu_ids}" =~ [[:space:]] ]]; then
    echo "Invalid GPU id list '${gpu_ids}'. Use one id or comma-separated ids."
    exit 1
fi
IFS=',' read -r -a requested_gpu_ids <<< "${gpu_ids}"
declare -A seen_gpu_ids=()
for gpu_id in "${requested_gpu_ids[@]}"; do
    if ! [[ "${gpu_id}" =~ ^[0-9]+$ ]]; then
        echo "Invalid GPU id '${gpu_id}' in '${gpu_ids}'."
        exit 1
    fi
    if [[ -n "${seen_gpu_ids[${gpu_id}]:-}" ]]; then
        echo "Duplicate GPU id '${gpu_id}' in '${gpu_ids}'."
        exit 1
    fi
    seen_gpu_ids[${gpu_id}]=1
done
num_gpus=${#requested_gpu_ids[@]}

if [ ! -f "${base_checkpoint}" ]; then
    echo "Base checkpoint not found: ${base_checkpoint}"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base_checkpoint="$(realpath "${base_checkpoint}")"
python_bin="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${gpu_ids}"
export HYDRA_FULL_ERROR=1
cd "${SCRIPT_DIR}"

training_launcher=("${python_bin}")
if [ "${num_gpus}" -gt 1 ]; then
    training_launcher=(
        "${python_bin}" -m torch.distributed.run
        --standalone
        --nproc_per_node="${num_gpus}"
    )
    has_run_dir=false
    has_output_subdir=false
    for override in "${hydra_overrides[@]}"; do
        case "${override}" in
            hydra.run.dir=*|+hydra.run.dir=*) has_run_dir=true ;;
            hydra.output_subdir=*|+hydra.output_subdir=*) has_output_subdir=true ;;
        esac
    done
    if [ "${has_run_dir}" = false ]; then
        timestamp="$(date +%Y.%m.%d-%H.%M.%S)"
        hydra_overrides+=(
            "hydra.run.dir=${SCRIPT_DIR}/data/outputs/${timestamp}_resfit_ddp${num_gpus}_seed${seed}"
        )
    fi
    if [ "${has_output_subdir}" = false ]; then
        hydra_overrides+=("hydra.output_subdir=null")
    fi
    if [ -z "${NUMBA_NUM_THREADS:-}" ]; then
        available_cpu_threads="$(nproc)"
        per_process_numba_threads=$((available_cpu_threads / num_gpus))
        if [ "${per_process_numba_threads}" -lt 1 ]; then
            per_process_numba_threads=1
        fi
        export NUMBA_NUM_THREADS="${per_process_numba_threads}"
    fi
    echo "ResFiT distributed training enabled: GPUs=${gpu_ids}, processes=${num_gpus}"
    echo "NUMBA_NUM_THREADS per process: ${NUMBA_NUM_THREADS}"
fi

"${training_launcher[@]}" "${SCRIPT_DIR}/train.py" \
    --config-name=maniskill_resfit_handover_ur5_v2_multiview_delta_pose_actual_action_dit_patch.yaml \
    training.seed="${seed}" \
    training.device="cuda:0" \
    training.base_checkpoint="${base_checkpoint}" \
    "${hydra_overrides[@]}"
