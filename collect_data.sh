#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"

if [ $# -lt 2 ]; then
    echo "Usage: $0 task_name task_config [gpu_id] [extra args passed to script/collect_data.py]"
    exit 1
fi

task_name=${1}
task_config=${2}
shift 2

gpu_id=
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
    gpu_id=${1}
    shift
fi

extra_args=("$@")

cd "${ROOT_DIR}"
"${ROOT_DIR}/script/.update_path.sh" > /dev/null 2>&1

# If caller passes a GPU id (arg 3), use it. Otherwise pick the GPU that has display attached.
if [ -z "${gpu_id}" ]; then
	# Query GPUs with display_active=Enabled, pick the first index; fallback to GPU 0.
	display_gpu=$(nvidia-smi --query-gpu=index,display_active --format=csv,noheader 2>/dev/null | \
		awk -F, '$2 ~ /Enabled/ {gsub(/ /, "", $1); print $1; exit}')
	if [ -z "${display_gpu}" ]; then
		display_gpu=0
	fi
	export CUDA_VISIBLE_DEVICES=${display_gpu}
else
	export CUDA_VISIBLE_DEVICES=${gpu_id}
fi

PYTHONWARNINGS=ignore::UserWarning \
python "${ROOT_DIR}/script/collect_data.py" "$task_name" "$task_config" "${extra_args[@]}"

backend=robotwin
for ((i=0; i<${#extra_args[@]}; i++)); do
    if [ "${extra_args[$i]}" = "--backend" ] && [ $((i + 1)) -lt ${#extra_args[@]} ]; then
        backend="${extra_args[$((i + 1))]}"
        break
    fi
done

if [ "${backend}" = "robotwin" ]; then
    rm -rf "${ROOT_DIR}/data/${task_name}/${task_config}/.cache"
fi
