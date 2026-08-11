#!/bin/bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 seed gpu_id base_checkpoint [-- HYDRA_OVERRIDE ...]"
    exit 1
fi

seed="$1"
gpu_id="$2"
base_checkpoint="$3"
shift 3
if [ "${1:-}" = "--" ]; then
    shift
fi

if ! [[ "${gpu_id}" =~ ^[0-9]+$ ]]; then
    echo "gpu_id must be one non-negative integer."
    exit 1
fi
if [ ! -f "${base_checkpoint}" ]; then
    echo "Base checkpoint not found: ${base_checkpoint}"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base_checkpoint="$(realpath "${base_checkpoint}")"
python_bin="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export HYDRA_FULL_ERROR=1
cd "${SCRIPT_DIR}"

"${python_bin}" "${SCRIPT_DIR}/train.py" \
    --config-name=maniskill_resfit_handover_ur5_v2_multiview_delta_pose_actual_action_dit_patch.yaml \
    training.seed="${seed}" \
    training.device="cuda:0" \
    training.base_checkpoint="${base_checkpoint}" \
    "$@"
