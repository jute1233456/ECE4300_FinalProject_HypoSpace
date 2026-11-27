# simple-reflection-dpo-v3-additive

Additive update to the 130-pair training set used for `dpo-reflection-v2-ep1`.
The 130 base rows are unchanged and come first; 30 new pairs reviewed from the
35 states that completed after the v2 snapshot are appended. The three v2
exclusions remain excluded.

## Files

- `dpo.jsonl` (160 rows) — TRL conversational format; 130 base + 30 new.
- `pair_manifest.jsonl` (160 rows) — base provenance + new labels/sources/hashes.
- `judgments.jsonl` (183 rows) — all reviewed states (v2 148 + v3 35).
- `skipped.jsonl` (20 rows) — v2 15 + v3 5.
- `manifest.json` — base/increment fingerprints, exclusions, statistics.
- `SHA256SUMS`.

## Increment

- 35 new complete states, 140 qwen candidates (all stop, non-empty, no thinking), all with DS answers.
- 30 pairs / 5 skips; all pairs chosen=helpful/high, rejected=harmful/high.
- Domains: {'boolean': 14, '3d': 10, 'causal': 6}.
- Judgments: {'harmful': 91, 'helpful': 83, 'uncertain': 1}.

## Caveats

- Additive only; the 130-row base and its 3 documented exclusions are preserved.
- Reflection stage only; no final-answer preference.
- No train/validation/test split.
- DS answers come from reconstructed stage inputs and are recorded continuations.
- Most chosen answers are DS reflections; this is a distillation-style preference set.
