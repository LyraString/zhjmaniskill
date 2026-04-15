#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"

backend=robotwin
task_profile=
gpu_id=
positional=()
extra_args=()

resolve_collect_target_from_profile() {
    local profile_name="$1"
    python - "$ROOT_DIR" "$profile_name" <<'PY'
import sys
from pathlib import Path

root_dir = sys.argv[1]
profile_name = sys.argv[2]
sys.path.insert(0, root_dir)

from sim.maniskill.task_profiles import resolve_task_profile

profile = resolve_task_profile(profile_name)
if profile is None:
    raise SystemExit(1)

env_id = str(profile["env_id"])
zarr_path = profile.get("dataset", {}).get("zarr_path")
if not zarr_path:
    raise SystemExit(1)

parts = Path(str(zarr_path)).name.removesuffix(".zarr").rsplit("-", 2)
if len(parts) != 3:
    raise SystemExit(1)

task_config = parts[1]
print(env_id)
print(task_config)
PY
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            backend="$2"
            extra_args+=("$1" "$2")
            shift 2
            ;;
        --task-profile)
            task_profile="$2"
            extra_args+=("$1" "$2")
            shift 2
            ;;
        --episode-num|--seed|--max-episode-steps|--action-type|--obs-mode|--control-mode|--render-mode|--sim-backend|--render-backend|--reward-mode|--robot-uids|--sim-freq|--control-freq|--collector|--teacher-zarr-path|--teacher-seed-path)
            extra_args+=("$1" "$2")
            shift 2
            ;;
        --save-only-success)
            extra_args+=("$1")
            shift
            ;;
        --*)
            extra_args+=("$1")
            shift
            ;;
        *)
            positional+=("$1")
            shift
            ;;
    esac
done

if [ "${backend}" = "maniskill" ] && [ -n "${task_profile}" ] && [ ${#positional[@]} -le 1 ]; then
    if [ ${#positional[@]} -eq 1 ]; then
        gpu_id=${positional[0]}
    fi
    mapfile -t collect_target < <(resolve_collect_target_from_profile "${task_profile}")
    if [ ${#collect_target[@]} -ne 2 ]; then
        echo "Failed to resolve collect target from ManiSkill task profile: ${task_profile}"
        exit 1
    fi
    task_name=${collect_target[0]}
    task_config=${collect_target[1]}
else
    if [ ${#positional[@]} -lt 2 ]; then
        echo "Usage: $0 task_name task_config [gpu_id] [extra args passed to script/collect_data.py]"
        echo "   or: $0 [gpu_id] --backend maniskill --task-profile PROFILE [extra args passed to script/collect_data.py]"
        exit 1
    fi

    task_name=${positional[0]}
    task_config=${positional[1]}
    if [ ${#positional[@]} -gt 2 ]; then
        gpu_id=${positional[2]}
    fi
fi

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

if [ "${backend}" = "robotwin" ]; then
    rm -rf "${ROOT_DIR}/data/${task_name}/${task_config}/.cache"
fi
