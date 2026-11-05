# RQ4: where every number comes from

All values are **GPU energy only** (NVIDIA H100 NVL, NVML energy counter). Host CPU and memory, networking and cooling are not included, so the values are lower bounds of the energy of a full deployment. In the tables the setups are `zs`, `fs`, `zs-cot`, `fs-cot` (code names `zs3`, `fs3`, `zs3_cot`, `fs3_cot`). Every table below is built from a file in `results/`; column names are given in `code font`.

## 1. What was measured
- Six models (Qwen3.5-4B and 9B, Gemma-4-E2B and E4B, Ministral-3-3B and 8B), each on its own GPU with no other process on it (checked before and during every measurement window; 0 of 144 windows contaminated). Server: vLLM 0.31.0 with prefix caching on (`docs/implementation.md`).
- Three load levels per model and setup: one request at a time, and closed loops of 16 and 64 concurrent requests. No intent was repeated inside a load level, so no complete prompt was served from the prefix cache; the shared instructions and examples are cached (16 to 39% of the prompt tokens; a request has 6,300 to 9,200 prompt tokens including the snapshot).
- Sample per model and setup: 60 valid intents at one request at a time (12 per complexity level), 20 of them repeated in two more passes (noise check), 40 invalid intents (10 per category), 96 distinct valid intents at 16 concurrent, 256 at 64 concurrent. Total 24 configurations x (60 + 40 + 40 + 96 + 256) = **11,808 requests** in 144 windows.
- Energy of one request at a time = NVML counter difference over the request. In the closed loops the energy per intent is the energy of the steady-state part of the window (from the completion of the N-th request to the start of the last request), taken from the integrated 0.25 s power trace and scaled so that the integral over the whole window equals the NVML counter reading (column `counter / trace scale` in `E-02_closed_loop_windows.csv`). On the 48 closed-loop windows the counter differs from the integrated trace by 0.15% to 3.56%.
- The sample size for one request at a time (60 intents) was chosen from an earlier pilot campaign of 16 configurations that is not released: the repeat-to-repeat variation of one intent was 0.2% to 3.2%, the spread across intents (driven by answer length) reached a coefficient of variation of 0.48, and a 95% interval of +-10% of the mean needed a median of 32 intents (at most 90).

## 2. Energy per intent, one request at a time (Wh)
**Whole valid set (521 intents): the values of Fig. 5 and of the text ("0.044 to 0.148 Wh without reasoning, 0.18 to 0.58 Wh with it").** For each model the measured requests give a linear cost model E = a + c x (generated tokens) (section 4). The energy of an intent in the accuracy runs is the model's a + c x its generated tokens; the table is the mean over the 521 valid intents. Source: `E-04_energy_and_accuracy_per_cell.csv`, column `Wh per intent (1 at a time)`.

| Model | zs | fs | zs-cot | fs-cot |
|---|---:|---:|---:|---:|
| Qwen3.5-4B | 0.085 | 0.086 | 0.584 | 0.469 |
| Qwen3.5-9B | 0.145 | 0.148 | 0.560 | 0.148 |
| Gemma-4-E2B | 0.072 | 0.074 | 0.511 | 0.500 |
| Gemma-4-E4B | 0.096 | 0.118 | 0.527 | 0.527 |
| Ministral-3-3B | 0.046 | 0.044 | 0.227 | 0.184 |
| Ministral-3-8B | 0.093 | 0.087 | 0.340 | 0.300 |

Without reasoning the range is 0.044 (Ministral-3-3B, `fs`) to 0.148 (Qwen3.5-9B, `fs`). With reasoning it is 0.184 (Ministral-3-3B, `fs-cot`) to 0.584 (Qwen3.5-4B, `zs-cot`), **leaving out Qwen3.5-9B `fs-cot` (0.148)**: that model usually answers the `fs-cot` prompt directly without writing out its reasoning (mean generated tokens 195.6 for `fs` and 195.4 for `fs-cot`, against 714.7 for `zs-cot`), so its cost equals that of `fs`.

**Measured on the 60-intent sample** (`E-01_valid_intents_one_at_a_time.csv`, column `mean Wh`):

| Model | zs | fs | zs-cot | fs-cot |
|---|---:|---:|---:|---:|
| Qwen3.5-4B | 0.088 | 0.094 | 0.598 | 0.521 |
| Qwen3.5-9B | 0.153 | 0.160 | 0.517 | 0.161 |
| Gemma-4-E2B | 0.075 | 0.086 | 0.490 | 0.547 |
| Gemma-4-E4B | 0.099 | 0.134 | 0.576 | 0.576 |
| Ministral-3-3B | 0.047 | 0.050 | 0.213 | 0.212 |
| Ministral-3-8B | 0.096 | 0.096 | 0.350 | 0.318 |

The whole-set estimate and the 60-intent measurement differ because the 60 intents are a level-balanced sample, not the whole set: the sample mean is 7.6% below to 16.5% above the whole-set estimate, depending on the configuration. The paper reports the whole-set estimate.

**Invalid intents** (`E-01_invalid_intents.csv`, 40 intents, mean Wh): rejections are short answers.

| Model | zs | fs | zs-cot | fs-cot |
|---|---:|---:|---:|---:|
| Qwen3.5-4B | 0.040 | 0.030 | 0.519 | 0.259 |
| Qwen3.5-9B | 0.067 | 0.052 | 0.314 | 0.220 |
| Gemma-4-E2B | 0.060 | 0.084 | 0.285 | 0.286 |
| Gemma-4-E4B | 0.035 | 0.045 | 0.374 | 0.268 |
| Ministral-3-3B | 0.026 | 0.025 | 0.207 | 0.123 |
| Ministral-3-8B | 0.054 | 0.038 | 0.322 | 0.217 |

## 3. What reasoning costs and buys
Energy multiplier = Wh per intent of `X-cot` / Wh per intent of `X`. F1 points = strict F1 of `X-cot` minus strict F1 of `X`, in points. Both come from `E-04_energy_and_accuracy_per_cell.csv` (columns `Wh per intent (1 at a time)` and `F1`; the same differences are in `E-04_price_of_cot.csv`).

| Model | zs -> zs-cot: energy x | F1 points | fs -> fs-cot: energy x | F1 points |
|---|---:|---:|---:|---:|
| Qwen3.5-4B | 6.90 | -2.6 | 5.44 | +1.9 |
| Qwen3.5-9B | 3.85 | +2.1 | 1.00 | +0.8 |
| Gemma-4-E2B | 7.09 | +25.9 | 6.76 | +28.4 |
| Gemma-4-E4B | 5.47 | +15.7 | 4.45 | +7.2 |
| Ministral-3-3B | 4.92 | +7.0 | 4.18 | +13.3 |
| Ministral-3-8B | 3.65 | +0.7 | 3.46 | +0.2 |

The multiplier is 3.46 to 7.09 (the paper: 3.5 to 7.1), leaving out Qwen3.5-9B `fs` (1.00; see section 2). The F1 gain is 25.9 and 28.4 points for Gemma-4-E2B and 13.3 points for Ministral-3-3B (`fs`); for Qwen3.5 and Ministral-3-8B it is at most 2.1 points (Qwen3.5-9B, `zs`), and Qwen3.5-4B `zs` loses 2.6 points.

## 4. Generated tokens set the energy
Cost model per model, fitted on the 400 requests of one request at a time (valid and invalid intents, four setups): E = a + c x (generated tokens). Source: `E-03_cost_model_per_model.csv`. A term for the uncached prompt length cannot be separated, because the uncached prompt varies by only a few percent between requests.

| Model | requests | a (J per request) | c (J per generated token) | R2 | cross-validated R2 |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B | 400 | 44.3 | 1.478 | 0.9997 | 0.9997 |
| Qwen3.5-9B | 400 | 84.9 | 2.496 | 0.9997 | 0.9997 |
| Gemma-4-E2B | 400 | 19.4 | 1.346 | 0.9963 | 0.9960 |
| Gemma-4-E4B | 400 | 55.3 | 2.115 | 0.9982 | 0.9981 |
| Ministral-3-3B | 400 | 26.3 | 0.884 | 0.9983 | 0.9982 |
| Ministral-3-8B | 400 | 61.2 | 1.675 | 0.9989 | 0.9989 |

The cross-validated R2 is 0.996 to 0.9997 (the paper states R2 >= 0.996). c is 0.88 to 2.50 J per generated token; the total energy per generated token including the fixed part is 0.88 to 2.97 J (`E-01`, column `J per generated token`). The larger model of a family costs 1.6 to 1.9 times more per token. Ministral weights are stored in FP8 and run as stored, so its low cost per token is not a like-for-like comparison.

Share of the energy of a request that is spent on generating the answer = c x (mean generated tokens) / (a + c x (mean generated tokens)), from `E-03_energy_shares.csv`, column `generation share` (82% to 99%):

| Model | zs | fs | zs-cot | fs-cot |
|---|---:|---:|---:|---:|
| Qwen3.5-4B | 0.87 | 0.87 | 0.98 | 0.98 |
| Qwen3.5-9B | 0.85 | 0.85 | 0.95 | 0.85 |
| Gemma-4-E2B | 0.93 | 0.93 | 0.99 | 0.99 |
| Gemma-4-E4B | 0.85 | 0.88 | 0.97 | 0.97 |
| Ministral-3-3B | 0.85 | 0.84 | 0.97 | 0.96 |
| Ministral-3-8B | 0.83 | 0.82 | 0.95 | 0.95 |

## 5. Load: energy and latency
**Energy per intent at 64 concurrent requests (J)** (`E-02_energy_per_intent_and_batching_saving.csv`; 33 to 186 J = 0.009 to 0.052 Wh):

| Model | zs | fs | zs-cot | fs-cot |
|---|---:|---:|---:|---:|
| Qwen3.5-4B | 59 | 62 | 180 | 142 |
| Qwen3.5-9B | 94 | 97 | 156 | 98 |
| Gemma-4-E2B | 33 | 38 | 92 | 98 |
| Gemma-4-E4B | 61 | 66 | 115 | 126 |
| Ministral-3-3B | 49 | 56 | 137 | 125 |
| Ministral-3-8B | 92 | 100 | 169 | 186 |

**Saving from 1 to 64 concurrent requests** = J per intent at 1 request / J per intent at 64 (same file, column `saving 1 -> 64`); 3.2 to 20.2, median 6.1:

| Model | zs | fs | zs-cot | fs-cot |
|---|---:|---:|---:|---:|
| Qwen3.5-4B | 5.4 | 5.5 | 11.9 | 13.2 |
| Qwen3.5-9B | 5.8 | 5.9 | 12.0 | 5.9 |
| Gemma-4-E2B | 8.2 | 8.1 | 19.3 | 20.2 |
| Gemma-4-E4B | 5.8 | 7.3 | 18.0 | 16.4 |
| Ministral-3-3B | 3.5 | 3.2 | 5.6 | 6.1 |
| Ministral-3-8B | 3.7 | 3.4 | 7.4 | 6.2 |

The saving per generated token (J per generated token at 1 request, `E-01`, divided by the steady-state J per intent at 64 divided by the mean generated tokens of the window, `E-02_closed_loop_windows.csv`) is 2.9 to 19.3, so the saving does not come from different answer lengths at the two load levels.

**95th-percentile latency (s)** (`E-05_capacity_and_marginal_energy.csv`, column `p95 latency (s)`). One request at a time: 0.8 to 3.2 s without reasoning, 2.1 to 20.4 s with reasoning.

| Model | zs | fs | zs-cot | fs-cot |
|---|---:|---:|---:|---:|
| Qwen3.5-4B | 1.4 | 1.4 | 16.8 | 16.3 |
| Qwen3.5-9B | 2.1 | 2.1 | 13.5 | 2.1 |
| Gemma-4-E2B | 2.3 | 2.4 | 15.3 | 20.4 |
| Gemma-4-E4B | 2.4 | 3.2 | 14.7 | 14.6 |
| Ministral-3-3B | 0.8 | 0.8 | 3.2 | 3.5 |
| Ministral-3-8B | 1.3 | 1.3 | 4.6 | 4.2 |

At 64 concurrent requests: 10.0 to 27.5 s without reasoning, 24.5 to 71.2 s with reasoning.

| Model | zs | fs | zs-cot | fs-cot |
|---|---:|---:|---:|---:|
| Qwen3.5-4B | 16.5 | 16.9 | 71.2 | 47.9 |
| Qwen3.5-9B | 27.3 | 27.5 | 57.2 | 27.9 |
| Gemma-4-E2B | 10.0 | 11.0 | 24.5 | 27.8 |
| Gemma-4-E4B | 18.3 | 19.4 | 31.9 | 33.3 |
| Ministral-3-3B | 13.1 | 14.9 | 58.0 | 36.4 |
| Ministral-3-8B | 25.6 | 27.3 | 45.2 | 45.7 |

## 6. A dedicated GPU: daily energy at 10,000 intents per day
Assumptions (`E-05_daily_projection.csv`, for 1,000, 10,000 and 100,000 intents per day): the GPU is powered all day; its idle draw is the measured power with the model loaded (101 to 108 W); it runs at the highest tested concurrency (1, 16 or 64) whose p95 latency stays within 30 s; and it is loaded to at most 50% of its measured throughput (more GPUs are added if needed). Daily energy = idle draw x 24 h x GPUs + (marginal energy per intent) x intents, where the marginal energy is (mean power - idle power) x window length / requests. Idle share = idle energy / daily energy. A 60 s latency bound is in `E-05_projection_with_a_60_s_latency_bound.csv`.

Result at **10,000 intents per day, all 24 configurations** (the last column is section 7):

| Model | setup | N used | GPUs | GPU kWh/day | idle share | g CO2e per intent (PUE 1.56, world 0.458) |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B | zs | 64 | 1 | 2.72 | 0.955 | 0.194 |
| Qwen3.5-4B | fs | 64 | 1 | 2.72 | 0.952 | 0.195 |
| Qwen3.5-4B | zs-cot | 1 | 2 | 9.36 | 0.554 | 0.669 |
| Qwen3.5-4B | fs-cot | 16 | 1 | 3.11 | 0.835 | 0.222 |
| Qwen3.5-9B | zs | 64 | 1 | 2.78 | 0.929 | 0.199 |
| Qwen3.5-9B | fs | 64 | 1 | 2.79 | 0.927 | 0.199 |
| Qwen3.5-9B | zs-cot | 16 | 1 | 3.17 | 0.814 | 0.227 |
| Qwen3.5-9B | fs-cot | 64 | 1 | 2.79 | 0.926 | 0.199 |
| Gemma-4-E2B | zs | 64 | 1 | 2.50 | 0.971 | 0.179 |
| Gemma-4-E2B | fs | 64 | 1 | 2.51 | 0.968 | 0.180 |
| Gemma-4-E2B | zs-cot | 64 | 1 | 2.64 | 0.923 | 0.188 |
| Gemma-4-E2B | fs-cot | 64 | 1 | 2.64 | 0.920 | 0.189 |
| Gemma-4-E4B | zs | 64 | 1 | 2.66 | 0.952 | 0.190 |
| Gemma-4-E4B | fs | 64 | 1 | 2.67 | 0.947 | 0.191 |
| Gemma-4-E4B | zs-cot | 16 | 1 | 2.94 | 0.861 | 0.210 |
| Gemma-4-E4B | fs-cot | 16 | 1 | 2.95 | 0.857 | 0.211 |
| Ministral-3-3B | zs | 64 | 1 | 2.64 | 0.962 | 0.189 |
| Ministral-3-3B | fs | 64 | 1 | 2.66 | 0.957 | 0.190 |
| Ministral-3-3B | zs-cot | 16 | 1 | 2.94 | 0.864 | 0.211 |
| Ministral-3-3B | fs-cot | 16 | 1 | 2.96 | 0.861 | 0.211 |
| Ministral-3-8B | zs | 64 | 1 | 2.76 | 0.931 | 0.197 |
| Ministral-3-8B | fs | 64 | 1 | 2.77 | 0.926 | 0.198 |
| Ministral-3-8B | zs-cot | 16 | 1 | 3.05 | 0.842 | 0.218 |
| Ministral-3-8B | fs-cot | 16 | 1 | 3.06 | 0.838 | 0.219 |

- For the 23 configurations that need one GPU, the idle share is 81% to 97% (median 93%) of 2.5 to 3.2 kWh per day (the paper's sentence).
- **Qwen3.5-4B `zs-cot` is the exception**: its answers are long, so only serial operation (one request at a time) meets the 30 s bound, which needs 2 GPUs (9.36 kWh per day, 55% idle). With a 60 s bound it needs 1 GPU (3.29 kWh per day, 79% idle).

## 7. Carbon
CO2e per intent = (GPU kWh per day x PUE x grid carbon intensity) / (intents per day), at 10,000 intents per day (`E-05_co2e_scenarios_at_10k_intents_per_day.csv`). PUE 1.56 is the industry average of the Uptime Institute Global Data Center Survey 2024; the file also holds 1.09 (a self-reported hyperscale fleet value) and 1.80 (an assumption for older sites) as sensitivity cases. Grid intensities are the 2025 annual values of Ember's yearly electricity data (`ember_grid_factors_2025.csv`, downloaded 2026-10-26; world 0.458 kg CO2e per kWh in the paper, 0.4585 in the file). Minimum and maximum over the **24** configurations, PUE 1.56:

| Grid (Ember 2025) | kg CO2e per kWh | g CO2e per intent, min | max |
|---|---:|---:|---:|
| France | 0.0415 | 0.016 | 0.061 |
| EU | 0.2092 | 0.082 | 0.305 |
| United States | 0.3844 | 0.150 | 0.561 |
| World | 0.4585 | 0.179 | 0.669 |
| Australia | 0.5246 | 0.205 | 0.766 |
| India | 0.6705 | 0.262 | 0.979 |

The paper's range for the world grid (0.18 to 0.67 g per intent) covers all 24 configurations, including Qwen3.5-4B `zs-cot` with its 2 GPUs (0.669 g); the 23 single-GPU configurations give 0.18 to 0.23 g. The location of the data center changes the result about 16-fold (0.979 g in India against 0.061 g in France at the maximum). These are operational emissions of the GPU energy only; embodied carbon and water are not included.

## 8. Notes on how the paper states some ranges
- The idle-share range (81 to 97%, 2.5 to 3.2 kWh) is for the 23 single-GPU configurations, while the CO2e range (0.18 to 0.67 g) is for all 24; see sections 6 and 7.
- "At most 2 points" of F1 gain for Qwen3.5 and Ministral-3-8B is the largest gain, +2.1 points (Qwen3.5-9B, `zs`).

## 9. Rebuilding the tables
`build_tables_from_raw_data.py` rebuilds every CSV in `results/` (except the grid factors) from the raw measurement files and the accuracy runs. The raw files are released with the camera-ready archive. Run on the authors' raw data it reproduces all 15 tables byte for byte.

## Citations
- NVML energy counter as a measurement source: Chung, J.-W. et al. *The ML.ENERGY Benchmark: Toward Automated Inference Energy Measurement and Optimization*. arXiv:2505.06371 (2025).
- Operational emissions = energy x PUE x grid intensity: Patterson, D. et al. *Carbon Emissions and Large Neural Network Training*. arXiv:2104.10350 (2021).
- PUE: Uptime Institute. *Uptime Institute Global Data Center Survey 2024*, Keynote Report 146M (2024).
- Grid intensity: Ember. *Yearly Electricity Data*, carbon intensity of electricity, 2025 values (https://ember-energy.org/data/yearly-electricity-data/; accessed 2026-10-26).
