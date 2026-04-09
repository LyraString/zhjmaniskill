#!/bin/bash

backend=robotwin
positional=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            backend="$2"
            shift 2
            ;;
        *)
            positional+=("$1")
            shift
            ;;
    esac
done

if [ ${#positional[@]} -lt 3 ]; then
    echo "Usage: $0 [--backend robotwin|maniskill] task_name task_config expert_data_num"
    exit 1
fi

task_name=${positional[0]}
task_config=${positional[1]}
expert_data_num=${positional[2]}

# python process_data.py $task_name $task_config $expert_data_num
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "${SCRIPT_DIR}/process_data.py" --backend "$backend" "$task_name" "$task_config" "$expert_data_num"
