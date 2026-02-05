#!/bin/bash

task_name=${1}
task_config=${2}
gpu_id=${3}

./script/.update_path.sh > /dev/null 2>&1

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
python script/collect_data.py $task_name $task_config
rm -rf data/${task_name}/${task_config}/.cache
