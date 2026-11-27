# simple-reflection-dpo-v2

Superset of `simple-reflection-dpo-v1`: all 59 v1 pairs are unchanged, plus 74
pairs reviewed from the 86 states that became complete after the v1 snapshot.
Built offline from the frozen v1 snapshot and the v2 increment snapshot; no
collection job, GPU, queue or pause state was touched.

## What this is

Each state is one reflection prompt (system + user, verbatim, including the
original benchmark question quoted as data and the prior-attempt history).
Five answers were reviewed per state under anonymised labels A-E: four fresh
`sft-pruned-v1-ep1` candidates (`final_text` only) and one historical
DeepSeek reflection answer from `training/sources/ds22-sampled19-canonical.jsonl`
(reconstructed stage input; provenance weaker than the qwen answers).

Each answer was graded helpful / harmful / neutral / uncertain by offline
reading, with a quoted evidence span and an explanation. A state yields at
most one pair: chosen = high-confidence helpful, rejected = high-confidence
harmful. States without such a pair are listed in `skipped.jsonl`.

## Files

- `dpo.jsonl` (133 rows) — trainer-facing. Schema:
  `{"prompt": [{"role","content"}...], "chosen": [{"role":"assistant","content"}], "rejected": [...]}`.
  `prompt` is exactly the frozen `state.messages`; chosen/rejected content is
  exactly `response.final_text` (qwen) or `ds_answers[0].text` (DS), no edits.
- `judgments.jsonl` (148 rows) — every label's grade, evidence, explanation,
  confidence and resolved source.
- `pair_manifest.jsonl` (133 rows) — labels, sources, text/prompt SHA-256 and
  the pairing rationale.
- `skipped.jsonl` (15 rows) — states without a high/high pair.
- `manifest.json` — snapshot fingerprints, counts, statistics, checks.
- `SHA256SUMS` — file digests.

## Counts

- reviewed states: 148 (62 v1 + 86 increment); pairs: 133; skipped: 15
- judgments: {'helpful': 311, 'harmful': 379, 'uncertain': 50}
- pairs by domain: {'boolean': 53, '3d': 49, 'causal': 31}
- chosen source: {'ds': 118, 'qwen': 15}; rejected source: {'qwen': 133}

## Composition warning

Most chosen answers are the historical DS answer (118 of 133); only 15
pairs choose a qwen (`sft-pruned-v1-ep1`) answer, and all 133 rejected answers
are qwen. The set therefore mostly teaches "DS reflection > ep1 reflection",
not "ep1 reflection > ep1 reflection". Treat it as a distillation-style
preference set, not as evidence that the ep1 checkpoint is good.

## Caveats

- Reflection stage only: the assistant text is a Review / Constraint check /
  Next direction, not a final benchmark answer.
- `no_split`: no train/validation/test assignment is made here. If a split is
  needed, split by `family_id`/`case_id` to avoid leakage.
- DS answers come from reconstructed inputs and are recorded continuations,
  not freshly sampled candidates under the ep1 protocol.
- These are model outputs judged by an offline reader, not human labels and
  not reward-model scores.
