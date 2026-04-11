#!/bin/bash

backend=robotwin
task_profile=
positional=()
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

resolve_task_profile_zarr_path() {
    local profile_name="$1"
    python - "$ROOT_DIR" "$profile_name" <<'PY'
import sys

root_dir = sys.argv[1]
profile_name = sys.argv[2]
sys.path.insert(0, root_dir)

from sim.maniskill.task_profiles import resolve_task_profile

profile = resolve_task_profile(profile_name)
if profile is None:
    raise SystemExit(1)

zarr_path = profile.get("dataset", {}).get("zarr_path")
if zarr_path:
    print(str(zarr_path))
PY
}

resolve_process_triplet_from_zarr_path() {
    local zarr_path="$1"
    local zarr_name
    zarr_name="$(basename "$zarr_path")"
    zarr_name="${zarr_name%.zarr}"

    if [[ "$zarr_name" != *-* ]]; then
        return 1
    fi

    local suffix="${zarr_name##*-}"
    local prefix="${zarr_name%-*}"
    if [[ "$prefix" != *-* ]]; then
        return 1
    fi
    local task_config="${prefix##*-}"
    local task_name="${prefix%-*}"

    if [[ -z "$task_name" || -z "$task_config" || -z "$suffix" ]]; then
        return 1
    fi

    printf '%s\n%s\n%s\n' "$task_name" "$task_config" "$suffix"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            backend="$2"
            shift 2
            ;;
        --task-profile)
            task_profile="$2"
            shift 2
            ;;
        *)
            positional+=("$1")
            shift
            ;;
    esac
done

if [ "${backend}" = "maniskill" ] && [ -n "${task_profile}" ] && [ ${#positional[@]} -eq 0 ]; then
    profile_zarr_path="$(resolve_task_profile_zarr_path "${task_profile}")"
    if [ -z "${profile_zarr_path}" ]; then
        echo "Failed to resolve dataset.zarr_path from ManiSkill task profile: ${task_profile}"
        exit 1
    fi

    mapfile -t process_triplet < <(resolve_process_triplet_from_zarr_path "${profile_zarr_path}")
    if [ ${#process_triplet[@]} -ne 3 ]; then
        echo "Failed to infer process triplet from zarr path: ${profile_zarr_path}"
        exit 1
    fi

    task_name=${process_triplet[0]}
    task_config=${process_triplet[1]}
    expert_data_num=${process_triplet[2]}
else
    if [ ${#positional[@]} -lt 3 ]; then
        echo "Usage: $0 [--backend robotwin|maniskill] [--task-profile PROFILE] task_name task_config expert_data_num"
        echo "   or: $0 --backend maniskill --task-profile PROFILE"
        exit 1
    fi

    task_name=${positional[0]}
    task_config=${positional[1]}
    expert_data_num=${positional[2]}
fi

# python process_data.py $task_name $task_config $expert_data_num
python "${SCRIPT_DIR}/process_data.py" --backend "$backend" "$task_name" "$task_config" "$expert_data_num"
