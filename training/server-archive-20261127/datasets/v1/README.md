# simple-reflection-dpo-v1

Reflection-stage preference pairs built offline from the frozen snapshot
`/opt/data/private/ECE4300-dpo-datasets-20261127/simple-reflection-v1/`
(snapshot_utc 2026-11-26T23:23:05Z). No collection job, GPU, queue or pause state was touched.

## What this is

Each state is one reflection prompt (system + user, verbatim, including the
original benchmark question quoted as data and the prior-attempt history).
Five answers were reviewed per state under anonymised labels A-E:

- four fresh candidates from `sft-pruned-v1-ep1` (final_text only), and
- one historical DeepSeek reflection answer from
  `training/sources/ds22-sampled19-canonical.jsonl` (reconstructed stage input;
  treat its provenance as weaker than the qwen answers).

Each answer was graded helpful / harmful / neutral / uncertain by offline
reading, with a quoted evidence span and a short explanation. A state yields
at most one pair: chosen = a high-confidence helpful answer, rejected = a
high-confidence harmful answer. States with no high-confidence harmful answer
are skipped and listed in `skipped.jsonl`.

## Files

- `dpo.jsonl` (59 rows) — trainer-facing. Schema:
  `{"prompt": [{"role","content"}...], "chosen": [{"role":"assistant","content"}], "rejected": [...]}`.
  `prompt` is exactly `state.messages`; chosen/rejected content is exactly
  `response.final_text` (qwen) or `ds_answers[0].text` (DS), with no edits.
- `judgments.jsonl` (62 rows) — every label's grade, evidence, explanation,
  confidence, and resolved source metadata.
- `pair_manifest.jsonl` (59 rows) — chosen/rejected labels, sources, text and
  prompt SHA-256, and the short pairing rationale.
- `skipped.jsonl` (3 rows) — states with no high-confidence harmful answer.
- `manifest.json` — snapshot fingerprints, counts, class/source statistics,
  format notes and the check list.
- `SHA256SUMS` — file digests.

## Counts

- reviewed states: 62; pairs: 59; skipped: 3
- judgments: {'helpful': 126, 'harmful': 140, 'uncertain': 44}
- pairs by domain: {'boolean': 22, '3d': 24, 'causal': 13}
- chosen source: {'ds': 50, 'qwen': 9}; rejected source: {'qwen': 59}

## Composition warning

59 pairs: 50 choose the historical DS answer and 9 choose a qwen
(`sft-pruned-v1-ep1`) answer; all 59 rejected answers are qwen. The set
therefore mostly teaches "DS reflection > ep1 reflection", not
"ep1 reflection > ep1 reflection". Treat it as a distillation-style
preference set, not as evidence that the ep1 checkpoint is good. See
`pair_manifest.jsonl` for per-pair sources.

## Caveats

- Reflection stage only: the assistant text is a Review / Constraint check /
  Next direction, not a final benchmark answer.
- `no_split`: no train/validation/test assignment is made here. The 62 states
  cover a subset of benchmark cases; if a split is needed, split by
  `family_id`/`case_id` to avoid leakage.
- DS answers come from reconstructed inputs (`input_reconstructed=true`);
  their prompts match the frozen prompt but were not generated online.
- DS answers are the recorded continuation from the original DS run, not a
  freshly sampled candidate under the ep1 protocol; they may be conditioned on
  that run's branch. Treat them as teacher-style targets, not a like-for-like
  sample of the same policy.
- These are SFT-model outputs judged by an offline reader, not human labels
  and not reward-model scores.
