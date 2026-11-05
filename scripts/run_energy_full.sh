#!/bin/bash
# Full energy campaign: 3 models per GPU, one after the other, nothing else on the GPU.
# usage: scripts/run_energy_full.sh <run_id> <gpu> <port> <model> [<model> ...]      (Ministral models use the cache in $MINISTRAL_CACHE)
RUN=$1; GPU=$2; PORT=$3; shift 3
cd "$(dirname "$0")/.."
OUT=outputs/energy/$RUN; mkdir -p $OUT/logs
for M in "$@"; do
  TAG=$(basename $M | tr 'A-Z' 'a-z' | tr -d '.')
  CACHE=""; case "$M" in *Ministral*) CACHE="--hf-cache $MINISTRAL_CACHE";; esac
  echo "$(date +%H:%M:%S) start $M on GPU $GPU" >> $OUT/logs/chain_gpu$GPU.log
  .venv-vllm/bin/python scripts/energy_probe2.py --model $M --gpu $GPU --port $PORT --out-dir $OUT $CACHE > $OUT/logs/$TAG.log 2>&1
  echo "$(date +%H:%M:%S) end $M exit $?" >> $OUT/logs/chain_gpu$GPU.log
done
touch $OUT/logs/chain_gpu$GPU.done
