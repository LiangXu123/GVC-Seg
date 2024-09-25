#!/bin/bash
dataset_cfg=${1:-'configs/scannet200gvc.yaml'}
GPU_ID=${2:-'0,1,2,3,4'}
workers=${3:-8}
export PYTHONWARNINGS="ignore"
PYTHONPATH=./:$PYTHONPATH
export PYTHONPATH

python3 tools/GVC_pool.py --config $dataset_cfg  --GPU_ID $GPU_ID --workers $workers
