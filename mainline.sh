#!/bin/bash

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

print_usage() {
    cat <<'EOF'
Usage:
  bash mainline.sh collect ...
  bash mainline.sh process-fm ...
  bash mainline.sh process-fpo ...
  bash mainline.sh train-fm ...
  bash mainline.sh train-fpo ...
  bash mainline.sh eval-fm ...
  bash mainline.sh eval-fpo ...

Examples:
  bash mainline.sh collect PickCube-v1 smoke_npz 0 --backend maniskill --task-profile pickcube_smoke --episode-num 1 --max-episode-steps 5
  bash mainline.sh process-fm PickCube-v1 smoke_npz 1 --backend maniskill
  bash mainline.sh train-fm PickCube-v1 smoke_npz 1 42 8 0 --backend maniskill --task-profile pickcube_smoke
  bash mainline.sh train-fpo PickCube-v1 smoke_npz 1 42 8 0 --backend maniskill --task-profile pickcube_smoke
  bash mainline.sh eval-fm PickCube-v1 smoke_npz dummy 1 42 0 --backend maniskill --task-profile pickcube_smoke --ckpt-path /ABS/PATH/TO/FM.ckpt
  bash mainline.sh eval-fpo PickCube-v1 smoke_npz dummy 1 42 0 --backend maniskill --task-profile pickcube_smoke --ckpt-path /ABS/PATH/TO/FPO.ckpt
EOF
}

if [ $# -lt 1 ]; then
    print_usage
    exit 1
fi

command="$1"
shift

case "$command" in
    collect)
        bash "${ROOT_DIR}/collect_data.sh" "$@"
        ;;
    process-fm)
        bash "${ROOT_DIR}/policy/FlowMatching/process_data.sh" "$@"
        ;;
    process-fpo)
        bash "${ROOT_DIR}/policy/FPO/process_data.sh" "$@"
        ;;
    train-fm)
        bash "${ROOT_DIR}/policy/FlowMatching/train_fm.sh" "$@"
        ;;
    train-fpo)
        bash "${ROOT_DIR}/policy/FPO/train_fpo.sh" "$@"
        ;;
    eval-fm)
        bash "${ROOT_DIR}/policy/FlowMatching/eval.sh" "$@"
        ;;
    eval-fpo)
        bash "${ROOT_DIR}/policy/FPO/eval.sh" "$@"
        ;;
    -h|--help|help)
        print_usage
        ;;
    *)
        echo "Unknown command: ${command}"
        print_usage
        exit 1
        ;;
esac
