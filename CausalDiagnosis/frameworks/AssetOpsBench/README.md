# AssetOpsBench

This folder holds everything of AssetOpsBench (Patel et al., arXiv 2506.03828v4) that CausalDiagnosis runs: the two agents of
the paper, the tool servers the AssetOpsBench rows of the tables play, the judge, the hydraulic scenarios of the dataset, the
reproduction of the paper's granite row and the cluster scripts of these parts. `__init__.py` is its declaration, which
`CausalDiagnosis/utils/frameworks.py` reads as data (`CausalDiagnosis/frameworks/README.md`). No file outside `frameworks/`
imports a module of this folder. The core reaches each part by the name the declaration gives it.

| folder | what it holds | how the core reaches it |
| --- | --- | --- |
| `agents/` | the paper-era Plan-Execute and Agent-As-Tool agents and IBM's ReActXen library | `AGENTS`: `plan_execute`, `agent_as_tool`, `scripted` |
| `tool_servers/` | the servers `iot`, `fmsr`, `wo` and `task` behind the metering layer, their tool contract and briefing, and the recorded history engine they serve from | `SURFACES`: `aob` |
| `LLM_Judges/` | the judge of an episode, the judge of finished runs and the truncation pass of the judge | `JUDGES`: `local` |
| `scenarios/` | the download of the hydraulic scenarios, their anchors on the recordings of `hyd_1` and the preset `anchored.json` | `SCENARIO_SETS`: `anchored`, `CONFIGS`: `scenarios` |
| `reproduction/` | the reproduction of the paper's granite row with the two paper-era agents (`reproduction/README.md`) | `RUN_KINDS`: `aob_repro` |
| `cluster_jobs/` | the reproduction job, the conformance job of the metering layer and the CouchDB service of every job | `CausalDiagnosis/cluster_jobs/functions.sh` sources `cluster_jobs/job_services.sh` |

The environment that the AssetOpsBench rows play lies in `tool_servers/`, not in `CausalDiagnosis/environment/`: the servers
publish what the recorded history engine serves, and the core asks them through the surface `aob`. The core puts the fork's
`src/` on `sys.path`, which the servers and the agents of this folder rely on.

The commands of the group `assetopsbench` start the programs of this folder: `python -m CausalDiagnosis assetopsbench <command>`,
with the commands of `COMMANDS` in `__init__.py`.

The scenario file is not stored in the repository. `scenarios/download_scenarios.py` writes `scenarios_17.json` into the data
folder of `hyd_1` (`CausalDiagnosis/datasets/hyd_1/data/`), or where `HYD_SCENARIOS_PATH` points, and only the preset `anchored`
reads it.
