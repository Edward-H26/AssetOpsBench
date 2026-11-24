# Dataset generalization development

The descriptive maintenance QA baseline lives in
`CausalDiagnosis/debug/general_dataset/versions/v001/industryasseteqa/`.
It is independent of the maintenance-loop debug versions and does not replace
the registered agents, shared declarations or frozen general-agent interfaces.

The earlier implementation used different general-agent records and board
semantics. Its controller, parser, memory and prompt code must be adapted to the
official frozen interfaces before integration. Copying those files over the
frozen package would discard the current interface contracts.

## General-agent migration

The official general-agent package now implements the typed readers and model
reply parser in its `plan/readers.py` and `plan/reply_parser.py`. Their reusable
logic comes from the earlier implementation, adapted to the current signatures
and records. Frozen dataclass fields, serialization conventions, declarations
and existing-agent code are unchanged.

| Area | Reused behavior | Adaptation to the current interface |
| --- | --- | --- |
| Typed values | Integral counts, finite numbers, named values and copied JSON containers | Every reader keeps its required argument-name parameter and returns a named reason |
| Plan heads | Candidate values, rank normalization and named JSON refusals | `why` is required and refusals use `reason` and `fault` |
| Step prefixes | Valid steps before the first malformed step remain available | `Unusable.kinds` retains the tool names of the unusable suffix |
| Tool ownership | Tools come from the joined declaration | Advice tools cannot be named by model steps |
| Input integrity | Duplicate keys, nonfinite values and decoder limits are refused | UTF-8 surrogate halves and unsupported text controls receive named refusals |

The parser resolves fixed omissions and leaves facts-dependent omissions to the
live check. The native, no-advice execution path now joins listed schemas to a
declaration, checks a copy of the public ledger, executes a valid prefix, records
answers on the board, and rechecks remaining steps against the live state. The
prompt, schema, answer-form readers and memory use declared roles and keys.
The QA contract and kind rules remain in the development adapter.

Use `--agent general` for this path or `--agent json` for the independent JSON
baseline. Both use isolated MCP sessions and the same public evidence and scoring.
The general path stores common episode records under `episode` in `result.json`,
with requests, responses, plans, tool calls, refusals and the accepted commitment.
An unsuccessful submission is a refusal and cannot commit or end the task. Request
budget exhaustion and endpoint failures retain partial output and stay incomplete.

The backend receives a JSON schema on planning and applicable orientation turns.
The transport sets `strict` to true as the frozen reply interface specifies;
local parsing and live checking remain authoritative. An endpoint that refuses
`response_format` fails visibly rather than silently changing the request mode.
For QA, the accepted structured submission supplies the final answer; no extra
closing-summary request is made. The explicit `--max-turns` budget counts every
model request. The current executor processes tool calls in order.

Controls cover complete MCP QA episodes, refusal and correction, a blocked write
without evidence, endpoint failure, source privacy, and renamed tools with an
unrelated answer field. These are stand-in model controls, not model benchmarks.
Frozen fields and signatures, registered agents and their fixtures are unchanged.
Argument fills, surface bindings and advice modules are refused explicitly by this
runner until their implementations are integrated. Automatic context-fit
compaction, parallel scheduling, registered-kind parity, formal driver registration
and the full general-agent acceptance gate remain outstanding. The memory helpers
alone do not establish those features. This version's output remains
`maintenance_qa_development` and is separate from formal experiment records.

## Local inputs

Prepared inputs can be kept at
`CausalDiagnosis/datasets/industryasseteqa/data/pdm/prepared/`.
This is local storage under the existing generated-data policy. It has no asset
file or package registration. Input data and extracted publication code stay out
of Git and normal source uploads; their hashes remain in the build and run
records. Existing downloaded inputs need no new download for inspection.

Run commands from the repository root in the project's prepared environment:

```bash
python -m CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa --help
python -m CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa inspect --facts CausalDiagnosis/datasets/industryasseteqa/data/pdm/prepared/facts.jsonl --qa CausalDiagnosis/datasets/industryasseteqa/data/pdm/prepared/qa_descriptive.jsonl --qa-id QA_ID
python -m pytest CausalDiagnosis/debug/general_dataset/versions/v001/industryasseteqa/tests -q
```

Use an actual question identifier for `QA_ID`. The inspect command releases only
the task and public fact fields. Reference answers and private labels remain
evaluation-only inputs. The run command requires an explicit model endpoint,
question identifiers, a new output directory and a planning-turn budget. Keys
come from an environment variable rather than command arguments.

## Grouped evaluation and served-model pilot

Keep all questions from one machine in the same partition. The split command
requires public `asset_id` on every fact, hashes the input files, and records both
asset and question membership. The run command validates complete coverage and
disjointness before selecting questions. The default development fraction is a
pilot setting, not a published benchmark split. Keep the test partition untouched
while choosing prompts and settings.

```bash
python -m CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa split --facts CausalDiagnosis/datasets/industryasseteqa/data/pdm/prepared/facts.jsonl --qa CausalDiagnosis/datasets/industryasseteqa/data/pdm/prepared/qa_descriptive.jsonl --out CausalDiagnosis/datasets/industryasseteqa/data/pdm/split.json
python -m CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa run --agent general --facts CausalDiagnosis/datasets/industryasseteqa/data/pdm/prepared/facts.jsonl --qa CausalDiagnosis/datasets/industryasseteqa/data/pdm/prepared/qa_descriptive.jsonl --split CausalDiagnosis/datasets/industryasseteqa/data/pdm/split.json --subset dev --limit 5 --model MODEL --base-url http://127.0.0.1:8000/v1 --api-key-env API_KEY --max-turns 8 --out CausalDiagnosis/results/pdm_general_pilot
```

Replace `MODEL` with the served model name and set `API_KEY` in the environment.
Use a new output directory on each run. After the development pilot, run the JSON
baseline on the same selected identifiers, model, settings and request budget.
Inspect incomplete episodes before expanding to the held-out partition.
`run.json` records selected identifiers, split identity, input and runtime code
hashes. `scores.csv` keeps structural acceptance and provenance separate from the
restricted numeric-answer metric. Numeric matching is not a general semantic
quality score, and acceptance is not correctness.

The next integration work is the formal driver and dataset registration, after
satisfying the general-agent gates and validating a served-model pilot. FaultArena
requires its own declared tool surface, scripted-case semantics and replay
controls. It is a separate port; completing this QA path does not implement it.

FaultArena release preparation is available in
`versions/v001/faultarena/README.md`. Its downloader pins the release and its audit
checks partition integrity, case references, DAGs and images without publishing
case contents. The public task and tool projection remains separate from private
replay data. This is preparation for the missing scripted replay prerequisite,
not a runnable FaultArena agent port.
