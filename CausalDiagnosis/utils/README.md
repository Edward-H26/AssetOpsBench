# utils

Small helpers that every folder may use. Nothing here imports another folder of CausalDiagnosis.

## What lives here

| File | What it does |
| --- | --- |
| `paths.py` | the one place that knows where things are: the folders of the repository, every generated location, and the lookup of a preset by name |
| `json_digest.py` | the canonical digest of a JSON value |
| `atomic_writes_and_locks.py` | writes that never leave a half-written file, and file locks |
| `download_retry.py` | the one retry policy of every fetch from outside the machine: a timeout that a slow link passes, retries counted only while nothing arrives, and a download that goes on from the bytes it kept |
| `llm_backends.py` | the model client of every agent and of the judge, built on AssetOpsBench's client |
| `frameworks.py` | the registry of the frameworks under `frameworks/`: it reads each declaration as data and loads a framework only when a run names it |

## How to use it

```bash
python -m CausalDiagnosis utils paths GRIDS
```

prints a location by its name, so that a shell script and the Python code read one place.

## How to add to it

A helper belongs here only when two or more folders use it. A new generated location gets a name in `paths.py`, an ignore rule in `CausalDiagnosis/.gitignore`, and is then checked by `experiments/run_verification/tests/test_generated_locations.py`.
