#!/bin/bash
uv='/home/fettigj/.local/bin/uv'

cd /research/hutchinson/workspace/fettigj/transformers2_electric_boogaloo
source /research/hutchinson/workspace/fettigj/transformers2_electric_boogaloo/.env

echo "Training the $1 architecture"
$uv run wandb login $WANDB_API_KEY
$uv run src/main.py --profiles $1 --run_name "${1}_sweep"
