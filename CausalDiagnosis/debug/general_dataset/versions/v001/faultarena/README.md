# FaultArena preparation

This adapter prepares the release from
[faultarena/FaultArena](https://huggingface.co/datasets/faultarena/FaultArena),
pinned in `download.py`. It does not yet implement the scripted-case simulator,
register an asset or provide an agent experiment command. The general-agent task
page requires the shared replay, role table and scripted-case action server before
the full port and framework move. Those components are absent from the current
checkout. Downloading the release does not satisfy that prerequisite.

The release is an anonymous review package. Its `LICENSE` provides no final public
license and does not grant redistribution. Keep source records, images and run
traces in generated local storage. Do not commit or upload them as source.

## Local download

From the repository root, using the project's Python environment:

```powershell
python -m CausalDiagnosis.debug.general_dataset.versions.v001.faultarena download --out CausalDiagnosis/datasets/faultarena/data/release --include-images
python -m CausalDiagnosis.debug.general_dataset.versions.v001.faultarena audit --release CausalDiagnosis/datasets/faultarena/data/release --out CausalDiagnosis/datasets/faultarena/data/audit.json
```

Both output destinations must be new. No Hugging Face login or GPU is required
for the pinned public revision. `data/all.jsonl` embeds the complete cases, so
duplicated individual files under `cases/` are not downloaded. Official train and
test files, tool definitions, metadata, README and license are retained. The image
flag also fetches the image files; without it, the audit reports missing images.
Running multimodal cases as text-only is not equivalent to the original task.

`source.json` records the revision, file sizes and SHA-256 identities. The audit
verifies those bytes, partition disjointness and coverage, embedded case identity,
tool references, action identifiers, graph references, graph cycles and referenced
image availability. A nonzero exit with a written report means the audit found
source issues. It leaves the release unchanged and does not remove cases.

`FaultArenaRelease.task()` releases only the case identifier, initial query and
initial state. `tools()` reads the public individual tool contracts, including
state changes omitted from the aggregate tool file. Hidden action arguments,
observations, DAGs and reference answers belong to the simulator or evaluator;
they must never be copied into a planning prompt. Public projections and release
controls use invented cases in tests.

## Next implementation dependency

Build the shared scripted replay on invented cases first: match calls, check
public and case-specific preconditions, apply state changes, return only the
matched observation, and account for counted calls and refusals. The defaults of
`CausalDiagnosis/datasets/README.md` govern unstated power requirements and repeat
state-setting calls. Do not infer per-case action choices from hidden annotations
for the public planner.

Then add the FaultArena matcher, declared tool roles, public kind ledger and
submission adapter. Check image handling and preserve the official test partition.
Source DAG issues need explicit accounting before trajectory-based scoring; do not
silently repair references or exclude problematic cases from the denominator.
The source includes observations and tool contracts, but these files alone do not
establish an equivalent matching algorithm or official diagnosis-quality scorer.
