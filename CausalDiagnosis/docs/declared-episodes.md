# Declared episode datasets

The shared entry runs registered maintenance QA and scripted cases through
`Causal_Diagnosis_Agent_next`. A dataset adds `episode.json` and an adapter in
its dataset folder. The registry and runner contain no dataset identifiers.
An independent declaration is exercised through the same runner in the tests.

This is the `declared_episode_run` workflow. Its registry is distinct from the
`asset.json` registry of recorded-level configurations. It does not reinterpret
a QA question or a scripted case as hydraulic recordings. The existing
`experiments run-episodes` entry, registered agents and recorded-run tables keep
their interfaces. Declared runs are not pooled into those tables.

## Work sequence and current boundary

1. Reuse the frozen plan, check and memory interfaces, the native controller and
   prompt, and the versioned QA and release readers.
2. Connect standing expectations, explicit evidence and fixed fills, closing
   summaries, context-fit compaction, token-cap refusals and executable prefixes.
3. Register the QA port and scripted port on one experiment command, with split
   validation, input and runtime source hashes, partial checkpoints and scoring.
4. Exercise text and image transports over invented MCP episodes, then validate
   the actual development or training inputs without model calls.
5. Run a served-model development pilot once an endpoint and model are supplied.
   Compare matching and scoring policies before held-out evaluation or claims
   of benchmark parity. Complete the remaining advice, binding and recorded-kind
   gates before replacing the existing agent.

The first four parts have implementations and controls. A served-model pilot,
official scripted-matcher parity, semantic judging, the existing recorded-run
verifier and viewer integration, advice modules and surface bindings remain open.
The work stays on one active branch; earlier branches preserve history.

## Registration

```json
{
  "id": "another_qa",
  "kind": "maintenance_qa",
  "adapter": "a_package.episode_port:load",
  "inputs": ["facts", "qa", "split"]
}
```

The factory receives resolved input paths and returns a port with `server`,
`publication`, `case_ids(subset)`, `task(case_id, limits)`,
`unsupported(case_id, supports_images=...)`, `validate_case(case_id, limits)`,
`declaration(case_id)`, `launch(case_id, folder, limits)`,
`evaluate(case_id, episode, folder)` and `identity()`.
The declaration and launcher describe public tools; the evaluator alone reads
reference answers. The input identity is recomputed after a run. Runtime source,
adapter source and declaration bytes are hashed, excluding input data and tests.

## Commands

Run from the repository root in the prepared Python environment:

```bash
python -m CausalDiagnosis experiments run-declared-episodes --list

python -m CausalDiagnosis experiments run-declared-episodes --validate \
  --dataset industryasseteqa_pdm --split dev \
  --input facts=CausalDiagnosis/datasets/industryasseteqa/data/pdm/prepared/facts.jsonl \
  --input qa=CausalDiagnosis/datasets/industryasseteqa/data/pdm/prepared/qa_descriptive.jsonl \
  --input split=CausalDiagnosis/datasets/industryasseteqa/data/pdm/split.json

python -m CausalDiagnosis experiments run-declared-episodes --validate --multimodal \
  --dataset faultarena --split train \
  --input release=CausalDiagnosis/datasets/faultarena/data/release
```

`--validate` makes no model request. It checks the selected partition and case
structure, reports unsupported modalities and keeps their selected denominator.
Validation is not a diagnosis-quality evaluation.

For a model run, keep the dataset, partition and inputs above, remove
`--validate`, and supply `--model MODEL --base-url BASE_URL --max-turns BUDGET
--out NEW_FOLDER`. Replace the placeholders with the served model, endpoint,
explicit model-request budget and a new output folder. `API_KEY` names the
default environment variable for the credential; `--api-key-env` changes its
name without putting the credential on the command line. An optional `--limit`
selects a pilot size. Use development or training inputs before held-out runs.

`--max-turns` counts every model request, including summaries. Scripted cases
also accept optional `--max-tool-calls` and `--budget`; these apply only when
explicitly supplied and appear in the task. The tool-call limit includes the
public task read. QA declares no tool-call or price budget. There is no invented
global cap and no automatic fallback call when a plan is refused.

## Scripted replay policy

The shared replay matches a tool name and the exact JSON arguments of one
private scripted action. It serves only that action's observation, applies its
state update and charges its price. An unmatched or ambiguous call is refused.
A public state-setting tool can apply its published update without a private
observation; a repeated state-setting call is refused.

Public safety requirements come from the tool files. Breaking one ends the
episode. A requirement stated only in the private script refuses the call,
names the required state and leaves the episode open. Refusals count one call
and cost zero. Reference DAGs do not select actions or enter the prompt.
The public ledger walks state changes and counted calls. It does not guess
private matched-action prices; the server enforces the actual price budget.

This policy is recorded as `exact_arguments`, with official-matcher parity
false. The pinned public release has annotations and tools, with no simulator
or scorer implementation in its file index. Matching equivalence needs a
separate specification and comparison. No semantic diagnosis score is invented.

## Images and records

`--multimodal` explicitly selects an image-capable backend. Image bytes enter
the model only after the corresponding tool returns its observation. They are
sent as image content parts, with the observation identifier, and kept out of
the board's plain-text rendering. Calls retain the served bytes; model records
retain image hashes and references. A rejected image request is an error and is
never retried as text-only. Without the image transport, image cases remain
selected and recorded as unsupported, rather than disappearing from the count.

Each run keeps `run.json`, `summary.json` and one folder per selected episode.
The episode holds its public task, common `EpisodeResult`, model request records,
result and a launcher. A scripted server keeps its replay trace under `hidden/`.
Incomplete runs keep partial records. Structural answer acceptance, restricted
numeric matching and trajectory diagnostics are separate from semantic quality.
A cyclic reference DAG receives no trajectory coverage or precedence score;
the source case remains unchanged and stays selected.

Keep real run folders under generated local storage, such as
`CausalDiagnosis/results/runs/`. Publish aggregate tables only. No release data,
image or case-level trace belongs in a source commit or upload.

## Checks and identities

```bash
python -m pytest CausalDiagnosis/agents/Causal_Diagnosis_Agent_next \
  CausalDiagnosis/debug/general_dataset \
  CausalDiagnosis/datasets/tests/test_declared_episodes.py \
  CausalDiagnosis/environment/scripted_cases/tests -q
```

Controls cover standalone declarations, actual stdio sessions, split leakage,
source drift, answer privacy, public and hidden preconditions, repeated state
writes, image delivery order, unsupported-image denominators, cyclic references,
standing surprises, argument fills, token-cap refusals and context-fit summaries.
No release recording is used to choose agent policy or prompts in these tests.

New shared dataset and environment code changes the conservative digest of
recorded-level label tables. Recompute those labels before using a recorded-level
benchmark with this tree; stored labels are not relabeled in place here. The
diagnostics model and grade code are unchanged. Harness classification changes
move every run identity; next-agent files move that agent's identity. No recorded
result table, fitted file or past metric is rewritten. The declared runner keeps
its own complete input and source hashes and remains a separate run format.
