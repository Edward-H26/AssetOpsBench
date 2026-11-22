# Reproducing the granite row of AssetOpsBench with the paper-era agents

This folder plays the two paper-era agents of the AssetOpsBench paper (Patel et al., arXiv 2506.03828v4), the Agent-As-Tool
port of `CausalDiagnosis/frameworks/AssetOpsBench/agents/Agent_As_Tool_Paper` and the Plan-Execute port of `CausalDiagnosis/frameworks/AssetOpsBench/agents/Plan_Execute_Paper`, on the paper's scenarios over
the servers and the data of IBM's public branches, and checks each port against the paper's granite-3-3-8b-instruct
values: the steps of Table 3 and the judge scores of Figure 5. The check is `acceptance_check.py`. No result is stored here. The
report of a run stays beside its synced run folder, outside the fork.

| Paper | Written by | File of this reproduction | What it holds |
|---|---|---|---|
| Table 3 and Figure 5, Agent-As-Tool | `acceptance_check.py` | `results/acceptance.md` | the verdict of the port, the steps per task kind with their interval against the paper's values, and task completion over the paper's 141 scenarios as a bracket against the paper's 35 percent |
| Table 3 and Figure 5, Plan-Execute | `acceptance_check.py --port plan_execute_paper` | `results/acceptance_plan_execute_paper.md` | the verdict of the port, the steps per task kind against the paper's values, the judge rates beside the paper's counts, and the plan length against Table 12 |

## Files

- `run_folder_names.py`, `scenario_file.py` and `scenario_sets.py` holds the pinned scenario file (Hugging Face commit d925adb and its SHA-256), the 120 runnable ids by
  family, the ten ids of the pilot mode, the family and task kind of an id, the names of the two ports and the file names
  of a run.
- `fetch_public_servers_and_data.py` fetches the servers and the data of IBM's public branches by commit into a folder outside the repository,
  builds the scenario file and records what it changed in `LAYER.json`. Nothing of those branches is stored in the fork.
- `run_scenarios/` plays the ports on a list of scenario ids, n runs each, and writes a run folder.
- `score_runs.py` scores a run folder with the judge and the efficiency counts and writes `scores.csv` into it. With
  `--judge-model` it requires `--judge-base-url`, so that no trajectory leaves for a default endpoint.
- `reproduction_tables.py` holds the paper's granite values that the check compares with and reads scored run folders of one model.
- `acceptance_check.py` checks a port against the paper on scored runs and writes its report.
- `tests/` plays each port on one scenario with a scripted completion endpoint and a stand-in MCP server, the scorer on a
  fixture, the reader of scored runs and the acceptance rule on hand-made `scores.csv` files, without network.
  `test_layer_b.py` runs layer B's functions without a network, and with `AOB_LAYER_B` naming a built folder also the
  real servers over the real data on `fake_couchdb.py`.
- `CausalDiagnosis/frameworks/AssetOpsBench/cluster_jobs/run_reproduction.sh` runs the pilot mode or the full reproduction on the cluster, as section 3 of `CausalDiagnosis/cluster_jobs/README.md` shows.

## Runnable subset

The paper's 141 scenarios are the Hugging Face config `scenarios` of `ibm-research/AssetOpsBench` at commit
`d925adba1f1f05b54e44503ad01299c999af37ef`, the only commit with exactly 141 rows. Later commits rewrote the
characteristic form of id 112 and added rows. `fetch_public_servers_and_data.py` builds the file the driver reads from that one, with the forms
of 621 and 622, empty at that commit, from commit 5e25bb7, and the driver refuses any file whose SHA-256 differs
(`--allow-unpinned` overrides this and run.json records it).

The servers and the data come from IBM's public branches, Apache-2.0: `main-0.x` at c6b9675 and `data/tsfm-data-couchdb` at
d6ab803. The work order CSV files of `archive/legacy-tmp` are the same bytes. A scenario counts as runnable when every
asset, period or file its text names exists there and a tool reaches it, which says nothing about how well a model
answers it. The released fork's own data would run 63.

| Family | Rows | Runnable ids | What blocks the rest |
|---|---|---|---|
| IoT | 20 | 1, 2, 3, 4, 5, 6, 9, 12, 47 | The telemetry of Chiller 3, of both air handling units, of Chiller 6 outside June 2020 and of Chiller 9 in June 2020 is on neither commit this layer fetches. The forms of 3, 9, 12 and 47 expect a file pointer, which the IoT agent of the paper returned, while the server returns pages of up to 1,000 records. |
| FMSR | 20 (22 in the runner of the paper) | 101 to 120 | All run, on the failure mode list and the mapping tool of `main-0.x`. |
| TSFM | 23 | 201 to 223 | All run. 223 names `chiller9_tsad-small.csv`, which is on neither commit this layer fetches, and reads `chiller9_tsad.csv` under that name. |
| Work order | 36 | 400 to 435 | All run. 411 and 424 ask for computed bundles that the paper's agent got from generated code, so an agent has to reason them out. No tool of `main-0.x` reads the alert or anomaly failure code maps, so 415 to 419, 421, 430 and 433 to 435 are answered by reasoning, where the agent of the paper called recommendation functions. |
| Multi-agent | 42 (40 in the runner of the paper) | 502, 506, 507, 510, 511, 514, 515, 517, 518, 519, 601 to 622 | Ids 501, 503, 504, 505, 508, 509, 512, 513, 516 and 520 need Chiller 6 data of the week of 2020-04-27, which is on neither commit this layer fetches. 622 names `/tmp/tsad_conformal.csv`, which is on neither commit this layer fetches, and plays without it, which its form allows. |

`fetch_public_servers_and_data.py` changes the public data in two places, both listed in `LAYER.json`: the alias of 223, and the timestamps of the
Chiller 9 recordings, written with a space and a UTC offset, which it rewrites to the local time form of Chiller 6, since the
IoT server compares these strings. The public readings of the Chiller 6 June 2020 file and of `chiller9_tsad.csv` differ
from the rows IBM's published run logs return for the same timestamps, so scenarios that ask for a value are compared
with care.

`fetch_public_servers_and_data.py` also takes a private file of telemetry rows (`--recovered-rows` or `AOB_RECOVERED_ROWS`), which the fork does
not hold. The runs on that file, `--set recovered`, are the reproduction of record, and the runs on the public readings
alone, `--set runnable`, 120 scenarios, stand beside them. Without the file a reader can rerun the public set alone. With
it, in a folder of its own, its rows load after the public data and replace every public row of an
asset in every month they touch, so no reply mixes the two, `LAYER.json` counts what was dropped and what public spans remain, its assets join what the IoT server lists, and `--set recovered` adds the scenarios whose assets and days it
covers, at most 19, as `RECOVERED_UNLOCKS` in `scenario_sets.py` lists them. Answers then read those rows instead of the
public readings at those timestamps, so the numbers of the two sets are not the same quantity. `reproduction_tables.py` refuses to pool
runs with and without the file, and `LAYER.json` holds its hash and counts and no row. The TSFM tools still read the public
`chiller9_tsad.csv` while the IoT tools read the rows, so the two can disagree on one timestamp. The rows file holds no real
Chiller 6 rows after June 7, so 9 and 12 read the first week of June only, and 10 loses its first row. 433 and 434 read
Chiller 9 from 2020-04-27 only, and Chiller 6 lists one sensor fewer. These are the price of never mixing the two series in
a month. 43 and 45 stay out either way.

The pilot mode plays ids 5, 12, 101, 112, 217, 222, 400, 412, 502 and 609 once with both ports, 20 episodes. Id 12 is the
context stress case, since it returns the 2,896 records of the June 2020 telemetry inline. The full run plays the 120 ids
once with both ports, 240 episodes, one run per scenario as the paper ran them, and the recovered set plays the ids its
file covers the same way, 139 ids and 278 episodes when it covers all of them. Single-agent tasks are the IoT, FMSR, TSFM
and work order ids (88), multi-agent tasks the end-to-end ids (32), the split of Table 3 of AssetOpsBench.

## What each number is

- The port of the paper's Agent-As-Tool (`--agents agent_as_tool_paper`) is IBM's ReAct supervisor of
  `archive/legacy-tmp` on a vendored ReActXen with the paper's prompt and 28 examples, over the paper's five agents, IoT on
  the iot and utilities servers, TSFM on tsfm and utilities, FMSR, WO, a RuleLogicAgent that answers the failure reply of
  the paper's tools since its package is not public, and ReActXen's JSONReader (`CausalDiagnosis/frameworks/AssetOpsBench/agents/Agent_As_Tool_Paper/agent_builders.py`, `CausalDiagnosis/frameworks/AssetOpsBench/agents/Agent_As_Tool_Paper/react_tool_adapter.py` and `CausalDiagnosis/frameworks/AssetOpsBench/agents/Agent_As_Tool_Paper/sub_agent_registry.py`). Its model
  calls are raw text completions with the paper's decoding. Its steps are the supervisor's ReAct counter, the count IBM's
  runner logged, and its attempted MCP calls are kept as `tool_calls`.
- The port of the paper's Plan-Execute (`--agents plan_execute_paper`) plays the planner, the reviewer and the summarizer
  of AgentHive on the branch `archive/legacy-tmp` over step agents that are the agents of the Agent-As-Tool port for the
  servers of the run, with their examples, on the same completion route. Its code is `CausalDiagnosis/frameworks/AssetOpsBench/agents/Plan_Execute_Paper/`. Its
  steps are the executed plan steps, which is how this reproduction reads the steps that the paper prints for this design,
  and the summarization step is not counted. The length of the plan that ran is kept as `plan_steps`, compared with
  Table 12 of AssetOpsBench.
- Both ports normalize what passes between the model and a tool and leave the vendored ReActXen as IBM releases it. Every
  tool answer, every refusal of a tool's argument model and the text of every exception that a call of an MCP tool raises
  reach an agent on one line, each JSON value without indentation and one space apart and any other text with its line
  breaks replaced by spaces, since ReActXen truncates an overlong scratchpad only by its lines that open with Observation
  and raises once an answer over many lines keeps it over its limit. The trajectory keeps every answer as the server sent
  it. A string argument loses its outer quotes, which ReActXen keeps from a key=value input such as `work_type="CM"`,
  layer by layer, as `""CM""` gives `CM`, but keeps a pair whose quote also stands inside the text, as `"pump" is "bad"`
  passes as written, and each comma item then loses its own quotes by the same rule, so
  `evidence_ids="obs/1/a", "obs/2/b"` reaches the server as `obs/1/a, obs/2/b`
  (`CausalDiagnosis/frameworks/AssetOpsBench/agents/Agent_As_Tool_Paper/react_tool_adapter.py`). The reproduction lists
  no CausalDiagnosis MCP, so both ports play it with IBM's code, prompts, examples and limits, as on every row of the gym
  without the MCP, while the rows of the gym with the MCP change the examples written for the port, the reply cap, the
  step limits and the served context (`CausalDiagnosis/docs/experiments.md`).
- Both ports run against the six MCP servers of `main-0.x`, under the names of `DEFAULT_SERVER_PATHS`. `--agents` names
  either port or both, and both play by default.
- A stdio MCP server inherits only HOME, LOGNAME, PATH, SHELL, TERM and USER from its client. Each server therefore starts
  from a launcher file in a private temporary folder that sets the CouchDB settings of the driver's environment, and that
  puts the tree of `main-0.x` first on the path, since its servers use relative imports, and points the FMSR server's
  generate tools at the agent's model and endpoint (`FMSR_MODEL_ID=litellm_proxy/<model>`).
- Before every episode the driver drops the layer's CouchDB databases and loads them again with the loaders of
  `main-0.x`, the reset the fork's scenario suite runner makes with its manifest. The reload is timed apart and left out
  of the runtime.
- Runtime is the wall clock of one `runner.run(question)`, the start of the MCP server processes included. A run that takes
  longer than 1,800 seconds is stopped and counts as an error (`--episode-timeout`). A run of either port without a result
  keeps the step count its runner reached.
- The judge is AssetOpsBench's rubric judge through the fork's `Evaluator`, on the trajectory record the runner persisted and on
  the pinned scenario file. Every episode is judged under two protocols, the paper's five calls at temperature 0.3 per
  trajectory, averaged (section 5.2), and the single protocol of the rig's judge, one call at temperature 0, kept in
  `judge_paper.json` and `judge_single.json`. The paper's printed numbers lie on multiples of one over the number of
  scenarios, and the code of the paper took the majority of five for Agent-As-Tool and one call at temperature 0 for
  Plan-Execute, so the check reads that rule for each port. The judge reads the whole trajectory of the run, every tool
  output included, since `CausalDiagnosis/frameworks/AssetOpsBench/LLM_Judges/full_trajectory_scorer.py` passes it without the released cut at 8,000
  characters. A call that overflows the judge's context
  is a failed call. A failed call counts as false on every criterion with the denominator kept, and the failed calls are
  counted apart. A run that left no record scores false without a judge call.
- Every mean of this reproduction carries a 95 percent interval from a bootstrap that resamples whole scenarios
  (`stats.cluster_bootstrap` of `CausalDiagnosis`).

## The judge and its backend

AssetOpsBench's own scorer and backends run unchanged. What the scorer needs beyond them lives in two files beside the
rig's drivers, whose code `run_ledger/` hashes into the event of every scoring. Each answers the need to run the agent and
the judge in one process or what the judge reads.

| File | What it does | Reason |
|---|---|---|
| `CausalDiagnosis/frameworks/AssetOpsBench/LLM_Judges/full_trajectory_scorer.py` | The released rubric scorer with the trajectory passed uncut, so a call that overflows the judge's context counts as a failed call | the released cut at 8,000 characters hides the later steps of a long trajectory from the judge. The code of the paper also passed its trajectory uncut, and that trajectory held the steps of the supervisor or the plan with the answers of the domain agents and not the outputs of their tools |
| `CausalDiagnosis/utils/llm_backends.py` | The chat backend binds the endpoint it is given, sends `extra_body` only when given, reads `reasoning_content` when the content is empty and records the finish reason | the agent and the judge run in one process against two endpoints, the judge's thinking is switched off through `extra_body`, and a server with a reasoning parser can return a reply as reasoning only |

One released file keeps two changes. `src/agent/plan_execute/executor.py` starts a server as `python -I` and writes the
standard error of that server to the error log of the episode. The launchers of the rig run only in an isolated interpreter,
and the server check before the first episode of the ports and the scripted wiring check of the rig start their servers
through this executor. Every other released file runs as IBM releases it.

## Commands

Tests, from the fork root:

```bash
PATH=$PWD/.venv/bin:$PATH python -m pytest CausalDiagnosis/frameworks/AssetOpsBench/reproduction -q -p no:cacheprovider
```

The scenario file, once, anywhere outside the repository:

```bash
F=$HOME/aob_all_utterance_d925adb.jsonl
U=https://huggingface.co/datasets/ibm-research/AssetOpsBench/resolve
curl -L -o $F $U/d925adba1f1f05b54e44503ad01299c999af37ef/data/scenarios/all_utterance.jsonl
shasum -a 256 $F
```

The hash must be `ed87bdbd38c150c28130b5ebbfb60fc9a8c974a05514db919eed2a92690b5bd0`. Then layer B, once, with Git and the
network, also outside the repository. `$L/scenarios.jsonl` is the file the driver and `score_runs.py` read:

```bash
L=$HOME/aob_layer_b
PATH=$PWD/.venv/bin:$PATH python -m CausalDiagnosis.frameworks.AssetOpsBench.reproduction.fetch_public_servers_and_data --scenarios $F --out $L
```

On the cluster, after section 0 of `CausalDiagnosis/cluster_jobs/README.md` (the clone of the fork under the project folder, with the cluster
scripts in its `CausalDiagnosis/cluster_jobs` folder and those of this framework in `CausalDiagnosis/frameworks/AssetOpsBench/cluster_jobs`), a pilot mode run of twenty episodes goes inside a new serving job that serves granite and
the stand-in judge on one H100. `WORK` is the project folder on the cluster, `FORK` the clone and `J` the folder of
the judge model:

```bash
WORK=/insomnia001/home/$USER/causal_diagnosis
FORK=$WORK/AssetOpsBench
H=$FORK/CausalDiagnosis/cluster_jobs
A=$FORK/CausalDiagnosis/frameworks/AssetOpsBench/cluster_jobs
J=<judge model folder>
export JUDGE_SGLANG_EXTRA="--mem-fraction-static 0.45 --cuda-graph-max-bs 16 --reasoning-parser qwen3"
E=ALL,CONTEXT_LENGTH=131072
E=$E,CO_SERVE_JUDGE=1,JUDGE_MODEL=$J,AOB_MODE=pilot
bash $H/submit.sh serve_model.sbatch --export=$E,RUN_AFTER_SERVE=$A/run_reproduction.sh
```

The full run is the same with `AOB_MODE=full`. A serving job that is already up takes the script as a step,
`srun --jobid=<job> --overlap --export=ALL,AOB_MODE=pilot bash $A/run_reproduction.sh`, and a CPU job of its own works
when a CPU node reaches the GPU node's port, `sbatch --export=ALL,AOB_MODE=full $A/run_reproduction.sh`.
The script installs couchdb3 2.0.2 (the pin of `uv.lock`) into the fork's `.venv` when absent,
downloads the scenario file into the data folder, builds layer B there once, pulls `docker://couchdb:3.5` once into
`$WORK/images`, starts CouchDB on node local disk, runs the driver, scores the run (with the judge under both protocols when
`JUDGE_MODEL` is set) and checks each port the run played, writing the reports into
`$WORK/runs/aob_repro/<mode>_<job id>/results/`. `AOB_AGENTS` names the ports to play, both by default. The log carries
`ACCEPTANCE_EXIT` and `PE_ACCEPTANCE_EXIT` with the exit code of each check and ends with `CD_AOB_REPRO_DONE`.

The exit code of a check is 0 when every check that decides passes, 1 when one fails, and 3 when the runs cannot decide,
which holds for an unfinished run, for a list of ids, for more than one run per scenario and for the pilot. A pilot run
checks the setup and decides nothing. Runs that cannot decide are no outcome of the check, so the job is run once more from
the same commit with no file of a port changed. After a second such verdict in a row no third job runs until the
conditions it names are read and their cause is fixed outside the ports.

On the laptop, after `bash CausalDiagnosis/cluster_jobs/4_after_run/sync_runs.sh` of section 4 of `CausalDiagnosis/cluster_jobs/README.md`, a new scoring with another judge, which keeps
the earlier verdicts of every episode as `judge_<protocol>.<model>.json`, and the check of a synced run come from the fork
root:

```bash
R=<absolute path of the synced run folder>
PATH=$PWD/.venv/bin:$PATH python -m CausalDiagnosis.frameworks.AssetOpsBench.reproduction.score_runs $R --scenarios $L/scenarios.jsonl \
  --judge-model <name> --judge-base-url <url>
PATH=$PWD/.venv/bin:$PATH python -m CausalDiagnosis.frameworks.AssetOpsBench.reproduction.acceptance_check $R --out $R/results
PATH=$PWD/.venv/bin:$PATH python -m CausalDiagnosis.frameworks.AssetOpsBench.reproduction.acceptance_check $R --port plan_execute_paper --out $R/results
```

The synced run folder sits at `CausalDiagnosis/cluster_jobs/runs_synced/aob_repro/<mode>_<job id>/aob_<set>_<model>_<time>/`, and the rig's
`tables_from_runs/` and `leaderboard.py` leave it alone, since they read only the run folders directly under
`CausalDiagnosis/cluster_jobs/runs_synced`.

## Caveats

- CouchDB under Apptainer, the couchdb3 install and the start of the six released servers follow the image's and the
  fork's documented layout as `CausalDiagnosis/frameworks/AssetOpsBench/cluster_jobs/run_reproduction.sh` sets them up. The driver lists the tools of every server before
  the first episode and stops when a server does not start.
- Context length. A June 2020 telemetry page of 1,000 records adds tens of thousands of tokens, so a context of 32,768
  tokens is likely to overflow. The default of `CausalDiagnosis/cluster_jobs/env.example.sh` serves granite with 131,072 tokens. With the co-served
  judge holding 0.45 of the GPU memory, SGLang may size a smaller cache than that, and then either the judge moves to a
  separate job or the context drops to 65,536. run.json records the endpoint's own report of its context length.
- The judge is a stand-in for llama-4-maverick. Every trajectory record stays in the run
  folder, so `score_runs.py` can rescore with maverick through an endpoint that serves it without rerunning the agents.
- The TSFM tools need torch, transformers and granite-tsfm in the fork's `.venv`, and they run on CPU beside the model's
  GPU.
- The data are close to Table 5 of AssetOpsBench and not equal, as `fetch_public_servers_and_data.py` loads them: 4,249 work orders against 4,392,
  6,256 events against 6,929, 1,466 alerts against 1,995.
- The FMSR server of `main-0.x` returns a fixed list for chillers and air handling units and makes its generate and mapping
  calls on the agent's own model, where the tool of the paper and that server's default use llama-3-3-70b-instruct whatever
  the agent's model.
- A pooled rate over 120 scenarios cannot be set against the paper's rates. Figure 5a rests on 141 scenarios, so the check
  of the Agent-As-Tool port reads task completion as a bracket over the 141, and the granite labels of Figure 5b lie on a
  denominator of 84, 22, 27 and 26 of 84, inferred from the two decimals printed on arXiv v1 page 8, since the paper
  states no such denominator and names no such set. So no judge value decides the check of the Plan-Execute port.
