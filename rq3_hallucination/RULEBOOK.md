# Hallucination rulebook

This is the rulebook that the paper refers to for RQ3. It says what counts as a **hallucinated** SLO, what counts as a **mistake**, and how the labels are assigned to the answers of the ten models. The code that applies it is listed at the end.

## 1. Definitions

The *sources* of an answer are the words of the intent, the numbers written in the intent, and the telemetry snapshot of the cluster. Only the final answer of a model is judged (text written before the final JSON, such as reasoning, is ignored).

- A **hallucinated** SLO asserts a requirement or a value that the sources do not support, or that contradicts them. Example: a placement on a layer that the operator never named; a number that is not the requested metric's value in the snapshot.
- A **mistake** is an SLO that differs from the ground truth but asserts nothing that the sources contradict. Example: a value off in its last digit, `<` for `<=`, an empty value.
- A value that the intent and the snapshot support but that differs from the dataset's ground truth is a mistake, not a hallucination.

Every hallucination is also an error against the ground truth, but not the other way round. When one SLO has several faults, the strongest label wins: hallucination before mistake.

An SLO is judged in three steps: (1) does it assert something? (2) do the sources support it? (3) is the difference only a convention or a form?

## 2. Labels

| Label | Kind | Meaning |
|---|---|---|
| Hallucination | Requirement | The SLO asks for something the text does not: an unnamed layer or metric, a wrong direction, an extra `highest`/`lowest`, a name that has no link to the schema, the snapshot or the intent. |
| Hallucination | Value | Resource, direction and metric are supported, but the number is not: not in the intent and not in the snapshot, the real number of another node, layer or metric, a wrong number stated in the intent, an arithmetic slip in a unit conversion. |
| Mistake | Convention | A different but defensible convention: decimal GB (10^9) instead of 2^30 bytes; `<` for `<=`; placement `any`; the extremum qualifier dropped while the number is right; the right number with a different unit label from the allowed units. |
| Mistake | Abstention | `null` where a value exists. The output states nothing false. |
| Mistake | Format | A wrong field, a unit label outside the schema's units, a name that exists in the schema or the snapshot, an unparseable output, a reversed comparator with the right extremum and the right value. |
| Mistake | Omission | A ground-truth SLO that is missing (response level). Incompleteness is not invention. |
| Mistake | Accepted invalid (text-supported) | For an invalid intent: the model should have rejected it but asserted only what the text states. |

Optional tag (kept in the label files, not analysed in the paper): *Intrinsic* (contradicts something the sources state) or *Extrinsic* (adds what the sources do not contain).

## 3. Decision table

| Case | Label / kind |
|---|---|
| Extra `highest`/`lowest` on a plain threshold | Hallucination / Requirement |
| Constraint for a layer or metric the text never names | Hallucination / Requirement |
| The intent says lowest, the output says highest (or the reverse) | Hallucination / Requirement |
| A name with no link to the schema, the snapshot or the intent | Hallucination / Requirement |
| A number that is neither in the intent nor in the snapshot | Hallucination / Value |
| The real snapshot number of another node, layer or metric | Hallucination / Value |
| A wrong number stated in the intent | Hallucination / Value |
| An arithmetic slip in a unit conversion | Hallucination / Value |
| Qualifier dropped and the number is wrong | Hallucination / Value |
| Decimal GB instead of 2^30; `<` for `<=`; placement `any`; qualifier dropped with the right number; the right number with another allowed unit label | Mistake / Convention |
| Right `highest`/`lowest` and value, only the comparator reversed | Mistake / Format |
| `value: null` where a value exists | Mistake / Abstention |
| Wrong field, a unit label outside the schema's units, a name found in the schema or the snapshot, unparseable output | Mistake / Format |
| A value for the layer that the intent names, supported by the snapshot, where the dataset's ground truth uses another scope | Mistake |

## 4. Valid intents: read the meaning first

For valid intents the labels are assigned by deterministic rules in code. A constraint that is malformed on the surface is first repaired, and the repaired constraint is judged with the rules above. The surface fault itself stays a mistake (Format). The repairs are:

- metric name variants (for example `storage_util_percent`, `memory_capacity_bytes`) map to the schema metric; a type name in the metric slot is read from the type slot;
- the type is taken from the metric; a comparator `lowest`/`highest` becomes the matching comparator plus the extremum qualifier; `below`/`above` become `<=`/`>=`;
- a missing or foreign unit becomes the default unit of the metric; a placement value such as "cloud side" becomes `cloud`.

If the metric cannot be read with certainty (for example a bare `memory` without a unit, or a type name used as the metric), the constraint is labelled without the repair.

## 5. Invalid intents: judged against the text

The ground truth of an invalid intent is "reject", so it says nothing about what a returned constraint should be. Each distinct returned constraint is therefore judged from the **intent words only**:

- **Mistake** if everything the constraint asserts (placement, metric family and kind, direction, number) is stated or implied by the text. The model should have rejected the intent but invented nothing. An empty or `any` placement carries no claim.
- **Hallucination / Requirement** if it names a placement, a metric, a kind (capacity or utilization), a threshold-or-extremum direction or an extremum that the text does not state. A superlative supports an extremum only when it refers to that metric.
- **Hallucination / Value** if placement, metric and direction are stated but the number is not in the text (a converted unit, for example GB to bytes, is fine).
- A garbled constraint (for example a metric name in the unit slot) is judged by what it means.
- For a conflicting intent, one of the two halves, or a merged `both`, is stated by the text: Mistake. For vague words ("sensible", "enough", "good") a chosen number or metric is invented: Hallucination.

How it is applied: an LLM (Claude Sonnet 5.5, a model that is not among the evaluated models) read each of the 35 distinct invalid intent texts once, under this rubric, and wrote a per-intent specification (layers, metrics, directions, numbers, concepts outside the schema). The specifications are in `scripts/judge_invalid_v2.py` (the `SPEC` list). They were then applied automatically to every returned constraint.

## 6. The hallucination rate

The **hallucination rate** η is the share of *parseable* responses that contain at least one hallucinated SLO. One unsupported requirement suffices to misconfigure a deployment, so the response is the unit. Unparseable responses are excluded from the denominator and reported separately (1.2% of the 28,640 responses). η is reported over all 716 intents, over the 521 valid and over the 195 invalid intents.

Omissions and mistakes do not count towards η. A rejected invalid intent (empty SLO set) contains no SLO and therefore no hallucination.

## 7. Scope and limits

- Scope: 10 models × 4 setups (`zs`, `fs`, `zs-cot`, `fs-cot`) × 716 intents = 28,640 responses.
- The labels are assigned by code (valid intents) and by the LLM-written specifications (invalid intents). A stratified sample of the labels was checked by one human annotator; the sample, the protocol and the responses will be released with the camera-ready archive.
- The rules are our interpretation of "unsupported by the sources". The Intent Engine paper does not publish flag rules or code for its hallucination categories, so the two sets of numbers are not directly comparable.
- The per-constraint label files are not part of this repository yet; they will be released with the camera-ready archive.

## 8. Code

| Step | File |
|---|---|
| Flags for every emitted constraint | `src/halluc.py` |
| Labels for valid intents (rules, meaning-first repair) | `src/halluc_verdict.py`, `src/halluc_v2.py`, `scripts/halluc_workload_v2.py` (shared helpers: `scripts/halluc_workload.py`) |
| Labels for invalid intents (specifications and their application) | `scripts/judge_invalid_v2.py`, then `scripts/merge_judged_labels.py` (puts them into the constraint file) |
| One row per response and the rate η | `scripts/halluc19_responses.py`, `src/halluc19.py` |
| Tests | `tests/test_halluc.py`, `tests/test_halluc_verdict.py`, `tests/test_halluc19.py` |

In the label files of the code the class "Label error" appears for some valid-intent rows; for this rulebook and every rate it counts as a mistake. It never counts towards η.
