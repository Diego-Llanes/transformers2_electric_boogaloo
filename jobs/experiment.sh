#!/bin/bash
MODELS=(
	"transformer"
	"gru"
	"lstm"
	"rnn"
	"ins_transformer"
	"lev_transformer"
	"corr_transformer"
)

cd /research/hutchinson/workspace/fettigj/transformers2_electric_boogaloo/jobs/
for M in "${MODELS[@]}"; do
	condor_submit run.job -batch-name $M args=${M}
done;
