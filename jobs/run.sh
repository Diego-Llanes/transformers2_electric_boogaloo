#!/bin/bash
cd /research/hutchinson/workspace/fettigj/transformers2_electric_boogaloo
uv='/home/fettigj/.local/bin/uv'

echo "Training the $1 architecture"

$uv run src/main.py --profiles $1 --run_name $1
