# experiments

What each table of the paper runs, the program that plays the episodes, and the checks that a finished run is real and can be replayed.

## What lives here

| Folder | What it holds |
| --- | --- |
| `table_definitions/` | which presets, agents and seeds make up each table (`experiment_tables.py`), the pre-registered analysis (`registered_analysis.json`), the training variants, and the program that writes the training split of a dataset |
| `episode_runners/` | the episode runner, which plays episodes with an agent or a reference policy through the tool servers and verifies each one, and the runner of the reference policies without servers |
| `cluster_runs/` | the job lists of the cluster, one line per run, the tables a job list reads before it starts, and the checks of the cluster run order |
| `run_verification/` | the ledger of a run (where, when and with which code and model it ran), the record check of a tree and the serving file of record, the replay check, the digests of every text an agent reads, and the checks of synced and finished runs |

## How to use it

```bash
# one scripted episode through both tool servers, without a model
python -m CausalDiagnosis experiments run-episodes --config static --agent scripted --seeds 1

# the job lists of the tables
python -m CausalDiagnosis experiments grid-job-lists write

# replay a finished run on this code
python -m CausalDiagnosis experiments replay-check --help
```

Run folders go to `results/runs/`. The runs of record happen on the cluster through `cluster_jobs/`, whose README leads from a fresh clone to the synced run folders.
`CausalDiagnosis/docs/experiments.md` documents the runner with its options and output files, the judge, the statistics, the provenance of a run and the training.

## How to add to it

A table gets its runs in `table_definitions/experiment_tables.py` and its builder under `evaluation/`. `python -m CausalDiagnosis experiments grid-job-lists write` then writes the lines of its job list.
A run of record plays from a clean clone of a pushed commit, so a change here takes a new commit, and the rows whose code the change touches are played again at that commit. The start check in `CausalDiagnosis/docs/experiments.md` lists the rules a run of record passes.
