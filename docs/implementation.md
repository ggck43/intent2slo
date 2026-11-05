# Implementation details

## Generation (accuracy, validation and hallucination runs; RQ1 to RQ3)
- **Serving.** vLLM 0.31.0 (https://github.com/vllm-project/vllm) on NVIDIA H100 NVL GPUs (96 GB). Weights in BF16, except Ministral 3, which is run with its stored FP8 weights.
- **Decoding.** Greedy (temperature 0, top-p 1), seed 0, one run per configuration. Outputs are deterministic for a fixed batch size (64 requests per batch); vLLM output depends on the batch size.
- **Output limits.** 2,048 tokens for `zs` and `fs`, 4,096 for `zs-cot` and `fs-cot`. The context length (`max_model_len`) is 12,288 tokens, and 16,384 for the `fs-cot` prompts. The share of answers that hit the limit is reported in the analysis and is above 5% only for a few small models with reasoning.
- **Answer extraction.** The answer is the last top-level JSON object (with a list `slos`) in the response; text before it, such as reasoning, is ignored. Every response is stored with its token counts and its finish reason.
- **Native thinking is off for all models.** For Qwen3.5 the chat template pre-fills an empty thinking block when thinking is off. Gemma 4 is run text-only (no image inputs). Ministral 3, Granite 4.1 and Llama have no native thinking mode.
- **Prompts** are in `prompts/` (`zs3`, `zs3_cot`, `fs3`, `fs3_cot`; paper names `zs`, `zs-cot`, `fs`, `fs-cot`). The `fs3` prompt carries a short note above the examples (names and numbers are invented) and one before the real input.
- **Llama prompt date.** The Llama chat templates insert a "Cutting Knowledge Date / Today Date" line into the system turn. Llama 3.1 uses a fixed date (26 Jul 2024); Llama 3.2 uses the date of the run.
- **Runner.** `scripts/run_zs_cot.py`, launched by `scripts/run_zs3_full.sh`, `run_fs3_full.sh`, `run_models_zs3_full.sh` and `run_models_fs3_full.sh`.
- **Reproducibility.** Single greedy runs, no repetition, so the results carry no sampling variance. The runs are not bit-for-bit reproducible across hardware, drivers or batch sizes.

## Energy measurement (RQ4)
- **Server.** One `vllm serve` process per model on its own GPU: `--dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.7 --max-num-seqs 128 --seed 0 --generation-config vllm --enable-prefix-caching --enable-prompt-tokens-details`. Requests go through the OpenAI-compatible HTTP interface; the answer is generated with the same prompts and token limits as above.
- **Energy.** GPU energy is the difference of the NVML total-energy counter (`nvidia-ml-py`) over a request or a measurement window. A 0.25 s power trace is recorded in parallel as a cross-check. Idle power is measured with the GPU empty before the server starts and with the model loaded before and after the measurement phases. Host CPU and memory energy is not measured, so all values are lower bounds of the full system.
- **Load generator.** Closed loop: N requests are always in flight; a new request starts when one finishes. N = 1 (one request at a time), 16 and 64.
- **Contamination check.** Before and during each window, the GPU is checked for processes other than the server; such a window would be rejected (0 of 144 were).
- **Tool.** `scripts/energy_probe2.py`; launcher `scripts/run_energy_full.sh`; analysis helpers `src/energy_analysis.py`. See `rq4_energy/DERIVATIONS.md` for how each number of the paper is derived.
