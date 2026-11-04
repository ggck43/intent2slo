#!/bin/bash
# Full fs3 / fs3 CoT runs for the Mistral / Granite / Llama models (plans/20261025.fs3_modelsExperiment.md).
# Per model: valid zs3, valid zs3 CoT, invalid zs3, invalid zs3 CoT. Native thinking OFF (these chat templates have no thinking branch).
# Before every run it waits until the GPU has enough free memory (the machine is shared); after every run it checks the number of
# predictions and, on a failure, moves the incomplete run dir to outputs/runs_failed/ and retries ONCE. Failures are listed in
# outputs/models_fs3_full/FAILED.txt (never silent).
# usage: scripts/run_models_zs3_full.sh <gpu> <hf cache dir> <model> <short> [<model> <short> ...]
GPU=$1; CACHE=$2; shift 2
cd "$(dirname "$0")/.."
PY=.venv-vllm/bin/python
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_CACHE=$CACHE
LOG=outputs/models_fs3_full; mkdir -p $LOG outputs/runs_failed
free_gpu() {  # wait until >= 72000 MiB are free on the visible GPU (0.7 x 96 GB = 66 GB needed)
  until [ "$(nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')" -ge 72000 ]; do sleep 30; done
}
run_one() {  # model short subset prompt-file label max-tokens max-model-len tag-args expected-count
  local MODEL=$1 SHORT=$2 S=$3 F=$4 L=$5 T=$6 M=$7 TAG=$8 N=$9
  EXTRA=""; case "$MODEL" in *Ministral*) EXTRA="--text-only";; esac
  for TRY in 1 2; do
    free_gpu
    START=$(date +%s)
    $PY scripts/run_zs_cot.py --model $MODEL --prompt-file $F --exp-label $L --subset $S $TAG --max-tokens $T --max-model-len $M --gpu-mem 0.7 $EXTRA \
        > $LOG/${SHORT}_${S}_${L}_try${TRY}.log 2>&1
    SLUG=$(basename $MODEL | tr 'A-Z' 'a-z' | tr -d '.')   # same rule as the runner (Qwen3.5-4B -> qwen35-4b)
    DIR=$(ls -dt outputs/runs/${L}_$( [ "$S" = invalid ] && echo invalid_ )${SLUG}_2* 2>/dev/null | head -1)
    if [ -n "$DIR" ] && [ "$(wc -l < $DIR/predictions.jsonl 2>/dev/null)" = "$N" ] && [ $(stat -c %Y $DIR/predictions.jsonl) -ge $START ]; then return 0; fi
    echo "FAILED ${SHORT} ${S} ${L} try${TRY} $(date +%H:%M)" >> $LOG/FAILED.txt
    [ -n "$DIR" ] && [ $(stat -c %Y $DIR) -ge $START ] && mv "$DIR" outputs/runs_failed/
  done
  return 1
}
while [ $# -ge 2 ]; do
  MODEL=$1; SHORT=$2; shift 2
  run_one $MODEL $SHORT valid   fs3_prompt.txt     fs3     2048 12288 ""              521
  run_one $MODEL $SHORT valid   fs3_cot_prompt.txt fs3_cot 4096 16384 ""              521
  run_one $MODEL $SHORT invalid fs3_prompt.txt     fs3     2048 12288 "--tag invalid" 195
  run_one $MODEL $SHORT invalid fs3_cot_prompt.txt fs3_cot 4096 16384 "--tag invalid" 195
  touch $LOG/${SHORT}.done
done
touch $LOG/ALL_${GPU}.done
