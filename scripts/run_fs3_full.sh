#!/bin/bash
# fs3 full experiment (plans/20261025.fs3_fullExperiment.md, v2 prompts with the two notes): valid (521) then invalid (195), base then CoT, one model,
# native thinking OFF. CoT uses --max-model-len 16384 (user decision), base 12288.
# usage: scripts/run_fs3_full.sh <gpu> <model> <short>      (Gemma: export HF_HUB_CACHE=<cache dir> first; --text-only is added automatically)
GPU=$1; MODEL=$2; SHORT=$3
cd "$(dirname "$0")/.."
PY=.venv-vllm/bin/python
export CUDA_VISIBLE_DEVICES=$GPU
EXTRA=""; case "$MODEL" in *gemma*) EXTRA="--text-only";; esac
mkdir -p outputs/fs3_full
run() {  # subset, prompt file, label, max_tokens, max_model_len, tag-args
  $PY scripts/run_zs_cot.py --model $MODEL --prompt-file $2 --exp-label $3 --subset $1 $6 --max-tokens $4 --max-model-len $5 --gpu-mem 0.7 $EXTRA > outputs/fs3_full/${SHORT}_$1_$3.log 2>&1
}
run valid   fs3_prompt.txt     fs3     2048 12288 ""
run valid   fs3_cot_prompt.txt fs3_cot 4096 16384 ""
run invalid fs3_prompt.txt     fs3     2048 12288 "--tag invalid"
run invalid fs3_cot_prompt.txt fs3_cot 4096 16384 "--tag invalid"
touch outputs/fs3_full/${SHORT}.done
