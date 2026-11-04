#!/bin/bash
# zs3 full experiment (plans/20261025.zs3_fullExperiment.md): valid (521) then invalid (195), base then prompted CoT, one model, native thinking OFF.
# usage: scripts/run_zs3_full.sh <gpu> <model> <short>      (Gemma: export HF_HUB_CACHE=<cache dir> first; --text-only is added automatically)
GPU=$1; MODEL=$2; SHORT=$3
cd "$(dirname "$0")/.."
PY=.venv-vllm/bin/python
export CUDA_VISIBLE_DEVICES=$GPU
EXTRA=""; case "$MODEL" in *gemma*) EXTRA="--text-only";; esac
mkdir -p outputs/zs3_full
run() {  # subset, prompt file, label, max_tokens, tag-args
  $PY scripts/run_zs_cot.py --model $MODEL --prompt-file $2 --exp-label $3 --subset $1 $5 --max-tokens $4 --max-model-len 12288 --gpu-mem 0.7 $EXTRA > outputs/zs3_full/${SHORT}_$1_$3.log 2>&1
}
run valid   zs3_prompt.txt     zs3     2048 ""
run valid   zs3_cot_prompt.txt zs3_cot 4096 ""
run invalid zs3_prompt.txt     zs3     2048 "--tag invalid"
run invalid zs3_cot_prompt.txt zs3_cot 4096 "--tag invalid"
touch outputs/zs3_full/${SHORT}.done
