# The cluster jobs of this tree

Whoever runs the jobs follows this page alone, from the first login on the cluster to the tables of a run of record. Every command runs in the login shell of the cluster, unless its block starts on the laptop, where it runs at the fork root. Every job goes through `submit.sh`, which prints one line `SUBMITTED` with the name of the script, the id of the job and the path of its log. `H` names the folder `CausalDiagnosis/cluster_jobs` of the clone that a job plays. Its numbered folders follow the run order, `0_setup`, `1_checks`, `2_serve_model`, `3_run` and `4_after_run`, and the section of the same number describes each, while `grids` holds the grid files. `submit.sh` finds a job script of any numbered folder by its file name.

Names on this page follow the code and the texts that an agent reads, and the [glossary](../docs/README.md#glossary) of the docs README gives the term of the paper for each of them, such as cumulative layer study for ladder and call counter for metering layer.

## What this tree is

A run is a run of record when it starts from a clone of a pushed commit and the start check of the driver finds no reason against it. The check asks that the tree equal its commit, that the commit stand on a branch of origin, that the data of record pass their pins, that the generated tables be those of the tree, and that the serving of each model equal the tracked serving file `CausalDiagnosis/experiments/table_definitions/serving_of_record.json`. `run_ledger/ record-check` asks the rules that need no run line and no endpoint, and prints `RECORD` with the commit, or `DEVELOPMENT` with one `REASON` line for each rule that fails. Any other run carries the stamp development, and no number of such a run is quoted. The commit that a run records is its record, and a fix becomes a new pushed commit whose rows play again.

## Before the first job

The cluster answers `ssh insomnia` after the VPN of the university, and Slurm works in the login shell that this session opens. Every job writes under one folder of your own on the scratch filesystem, `ROOT`, which the settings file makes readable by you alone and refuses in the home filesystem. The SGLang image, the agent model, the judge and the rows file of the reproduction lie in the group readable folders of the table below, so whoever runs the jobs downloads nothing large. The first clone serves to make `env.sh`, the settings file, from its template, and git ignores `env.sh`.

| Shared folder | What it holds |
| --- | --- |
| `<SHARED>/images` | the SGLang image, read only |
| `<SHARED>/models` | the agent model and the judge, read only |
| `<SHARED>/recovered_rows` | the rows file of the reproduction of record, read only |

`SHARED` of `env.sh` names that folder, and the value of this cluster comes with the message that names the commit.

```bash
git clone https://github.com/Edward-H26/AssetOpsBench.git
H=$PWD/AssetOpsBench/CausalDiagnosis/cluster_jobs
cp $H/env.example.sh $H/env.sh
```

`env.sh` holds values alone and sources `functions.sh` beside it, which holds the functions of every job, so an `env.sh` made before `functions.sh` existed stops every submission until it is made again from the template once. Set the values below in `env.sh`. The template leaves them empty, since they differ between people and between clusters, and the values of this cluster come with the message that names the commit. A judge reached through a remote route sets `API_KEY` in `env.sh` and nowhere else. Three settings are optional. `NODES_OF_RECORD` takes the line that `check_nodes.sh read` prints, and `submit.sh` then sends a job of record to those nodes alone. `GPU_OF_RECORD` and `GPU_MIB_OF_RECORD` name one card model, its name and its memory as a `NODE_GPU` line of that command prints them, and once `env.sh` names them, every serving job and every training job checks its card against them and stops with `GPU_NOT_OF_RECORD` on another card.

| Setting | What it holds |
| --- | --- |
| `ROOT` | a folder of your own on the scratch filesystem, readable by you alone |
| `ACCOUNT` | the account of every job |
| `PARTITION_CPU` | the partition of a job without a card |
| `PARTITION_GPU` | the partition of a job with a card, one without preemption |
| `PARTITION_GPU_RESUMABLE` | the partitions of a job that resumes by itself after a requeue |
| `SHARED` | the group readable folder with the image and the models, or empty |
| `EXCLUDE_NODES` | the nodes that no job may take |
| `NODES_OF_RECORD` | optional, the nodes whose digest check passed, the only ones a job of record then takes |
| `GPU_OF_RECORD` | optional, the name of the card that every job with a card must hold, as `nvidia-smi` prints it |
| `GPU_MIB_OF_RECORD` | optional, the memory of that card in MiB, as `nvidia-smi` prints it without the unit |

Both models are named by their Hugging Face id, `MODEL` and `JUDGE_MODEL` of `env.sh`, the judge served under the name after the slash, and the tracked serving file `CausalDiagnosis/experiments/table_definitions/serving_of_record.json` pins the Hub commit of each as `weights_commit`, the sha that the model API of Hugging Face gives, beside the served name, the launch dtype, the engine version and the launch flags. Every endpoint file repeats the revision of its model as `weights_commit`. A run of record keeps `CONTEXT_LENGTH`, `AGENT_DTYPE` and `SGLANG_EXTRA` of the template and the rewritten chat template of the agent model, since the launch flags they make are the ones the serving file declares, and a job whose flags differ stops with `SERVING_DIFFERS` before its first line when it is a job of record and otherwise plays its runs with the stamp development. The judge differs from the agent model, since the evaluator of AssetOpsBench refuses a model that judges its own episodes.

| Model | Name | Revision |
| --- | --- | --- |
| the agent model | `ibm-granite/granite-3.3-8b-instruct` | the `weights_commit` of the serving file, which `HUB_COMMIT.txt` of its model folder repeats |
| the judge | `Qwen/Qwen3.8-27B-FP8` | the `weights_commit` of the serving file, which `HUB_COMMIT.txt` of its model folder repeats |

## 0. Setup

```bash
bash $H/0_setup/clone_commit.sh <full hash of the pushed commit>
```

`clone_commit.sh` makes the clone of one pushed commit under the folder of clones and prints one line `H=` with the folder `CausalDiagnosis/cluster_jobs` of the new clone. Paste that line, since every later job of the commit runs from that clone. A clone is never updated in place, and a new commit takes a new clone. `setup.sh` then submits the seven jobs that a first run needs, each through `submit.sh`, and `setup.sh status` prints one line for each job, its closing line, `PENDING`, `RUNNING`, `FAILED` with the last line of its log, or `never submitted`, and `SETUP_OK` once all seven have closed. The image and the environment start at once, the agent model, the judge and the data job follow the environment, since the two downloads fetch with the uv that it installs, the tables job follows the data, and the digest label of the clone follows the tables. A failed job takes the jobs after it out of the queue, since `setup.sh` submits every job that waits for another with `--kill-on-invalid-dep=yes`, and `setup.sh status` names each of them `FAILED`. Once no job of the last chain is pending or running, a second `setup.sh` submits again every job that has not closed and skips the others, and while one still is, it stops with status 4 and names that job. Every fetch from outside the cluster is made again after it fails, under the one rule of `CausalDiagnosis/utils/download_retry.py` whose settings have their defaults in `functions.sh`, waits at most `FETCH_TIMEOUT` seconds for its next bytes wherever its tool takes a timeout, and keeps on scratch what it fetched, the packages of uv and the layers of an image under `CACHE`, so that a second `setup.sh` after a broken link fetches only what is missing. With the SGLang image of `SHARED` readable the image job pulls no SGLang image, and a model job downloads nothing when the copy of `SHARED` holds the commit that the serving file pins. The image job also pulls the images that the frameworks of the tree start, the CouchDB image of the AssetOpsBench rows among them unless `SHARED` holds a readable copy of it, so that no job pulls an image while it holds a card. A model download takes the commit that the serving file pins and nothing newer, so the job refuses a model without a pin, stops on a shared copy of another commit, and stops when the files it fetched come from another commit. The environment job builds one environment for each lock file, so a second clone with the same lock reuses it. The data job runs the one fetch command of the data of record, `python -m CausalDiagnosis.datasets.hyd_1.download_and_build --download`, which downloads the archive of [UCI dataset 447](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems) once into `DATA`, builds the files of record into the data folder of the clone and checks them against the pins of `CHECKSUMS.txt`. The same job fetches the scenario file of the anchored preset at the pinned revision of the AssetOpsBench dataset into that folder, where the runs of that preset read it. No step of the setup reads that file, so when its fetch fails the job prints one warning that names its dataset, `ibm-research/AssetOpsBench`, goes on, and lists that dataset after `CD_DATA_DONE` among the optional datasets missing, and a run of the anchored preset then stops at its start with the command that fetches the file. The tables job computes the generated tables on a CPU node before any job with a card, and a second run computes no table again but a label table whose engine code or asset has changed since it was written. The digest job writes the digests of the clone into the folder `DIGESTS` under the id of the tree, and every job that plays lines with a served model compares the digests of its node with that label before its first line. The logs of the seven jobs end with `CD_IMAGE_DONE`, `CD_DOWNLOAD_DONE` for each model, `CD_SETUP_DONE`, `CD_DATA_DONE`, `CD_TABLES_DONE` and `CD_DIGESTS_DONE`.

```bash
bash $H/0_setup/setup.sh
bash $H/0_setup/setup.sh status
```

The judge model is downloaded at the commit that the serving file pins, and into a copy of one's own when the shared copy records no commit, since such a copy may hold another chat template than the pin. The download writes `FILES.sha256`, one hash per file of the model folder, and checks every large file against the listing of the Hub at the pin. The endpoint file of every serving job then names the weights commit, the hash of `FILES.sha256`, the hash of the chat template and the card, and `run.json` records them for every run. The first judge job of a clone runs `JUDGE_SMOKE` on one grid folder. Every judge job, the smoke and each judging of grid folders before its first pass, names `JUDGE_THINKING` in its log, the number of replies that hold thinking although the request of the judge turns thinking off, and any count above 0 stops the job.

Every serving job reads the server's `/server_info` once and writes four lines into the endpoint file: `engine_version`, the `dtype` the server reports beside the `dtype_launch` it was started with (`AGENT_DTYPE`, bfloat16 for the agent model), its `quantization` and `kv_cache_dtype`, and the launch flags in public form with their hash, without the host, the port, the key or a path of the machine. The log prints the precision once, as the line `ENGINE_PRECISION`, and a job whose server names no precision stops before it writes the endpoint file. Before its first line, a job that plays lines with a served model compares its endpoint file with the serving file through `run_ledger/ serving --check`, and a job of record stops with `SERVING_DIFFERS` on any difference, while the precision that the server reports is recorded in `run.json` and compared with nothing.

The line `SUBMITTED` names the log of the job. The first line of every log names the host, the commit of the tree and whether a file of the tree differs from that commit, and the last line is the closing line of the job or the reason it stopped.

## 1. Checks

The folder `1_checks` holds the checks of a tree and of the nodes. `run_suite.sbatch` runs the suite of a tree in a clean environment as a CPU job, `check_tables.sh` checks that runs leave the generated tables as they are, `check_nodes.sh` labels and checks the digests on the nodes of the partition and prints the nodes of record, `check_endpoint.sbatch` checks a served model from another node with and without the key, and `platform_record.sh` writes a platform record of the agent and the judge inside a serving job. No step of the path needs one of them, and the section Optional tools gives each.

## 2. Serving a model

The folder `2_serve_model` holds `serve_model.sbatch`, which serves a model on one card behind the key of the job and then runs on the same node the script that `RUN_AFTER_SERVE` names, a script of `3_run` or `platform_record.sh` of `1_checks`, or serves the nodes of the partition with `SERVE_REACH=nodes`.

## 3. Runs

The folder `3_run` holds the scripts that play inside a serving job, `run_lines.sh` with its runner `run_grid.sbatch`, which plays the baseline policies alone as a CPU job as well, `run_judge.sh`, `run_gepa_online.sh` and the training job `train_lora.sbatch`. The grid files that `run_lines.sh` plays lie in `grids`. The scripts of the rows of a framework lie in the folder `cluster_jobs` of that framework, for the AssetOpsBench rows `CausalDiagnosis/frameworks/AssetOpsBench/cluster_jobs`: `check_metering_layer.sh` runs the conformance test of the metering layer as a CPU job, `run_reproduction.sh` plays the reproduction of AssetOpsBench inside a serving job, and `job_services.sh` with `start_couchdb.sh` gives every job the function that starts CouchDB, since `functions.sh` sources the `job_services.sh` of every framework. One development row, the two further tables on development combinations and the registered order follow.

### One development row

```bash
E=ALL,RUN_AFTER_SERVE=$H/3_run/run_lines.sh,GRID_ROW=causal+mcp
E=$E,GRID_STARTS=development,GRID_PARALLEL=1
bash $H/submit.sh serve_model.sbatch --export=$E
bash $H/4_after_run/job_record.sh <folder that CD_RUN_LINES_DONE names>
```

One development row plays one row of the main table on the development combinations, inside a serving job of the agent model that ends when its line has played. `GRID_ROW` names the row by its agent, `plan_execute`, `agent_as_tool` or `causal`, with `+mcp` for the row with the CausalDiagnosis MCP, and the job plays the grid file of one line that `grid_job_lists.py row` prints. A grid file of `grids` in `GRID_FILE` plays in the same way, and git ignores every file named `local_*.txt`. `GRID_PARALLEL` names the runs in flight of the lines without the wall clock, and the lines with the wall clock play after them one at a time, so that one run alone is alive while a wall-clock episode plays. Before its first line the job checks the digest label of the clone (`OPENING_DIGESTS_OK`), compares the endpoint file with the serving file, and times one chat call (`OPENING_CALL`). The job ends with `CD_RUN_LINES_DONE` and the folder of its runs, and `job_record.sh` writes the accounting record of the job beside the runs once the job has ended. Whoever runs the jobs reads the `run.json` of each run, which carries the stamp development, its `scores.csv` and the `CALL_LATENCY` lines of the log.

A line of a grid file may also name one start state with `--start-state`, which plays with `GRID_STARTS=seeds`, or scenarios of the preset `anchored` with `--scenario-ids`, each one word with the levels or the ids joined by `/`, which `GRID_EXTRA` carries as well through the comma list of `--export`. A requeued job and a job with `GRID_PENDING=1` count a finished run for a line only when the run played the episodes that the line plays under the job, with the same start set, start cap, share and selectors, those of the line and of `GRID_EXTRA` alike, and the same seeds wherever the seeds choose the episodes. The reference runs of a grid job play the preset of each line without the selector of the line, while a selector of `GRID_EXTRA` reaches them as every other run, so a reference on the start state or the scenarios of a line names its agent on a line of its own, such as `--config base --agent oracle --start-state 3/100/0/130@2`, and the job plays with `GRID_REFERENCE=0`. No cell of a table reads a run of a selector.

```bash
E=ALL,RUN_AFTER_SERVE=$H/3_run/run_lines.sh,GRID_ROW=causal+mcp
E=$E,GRID_STARTS=seeds,GRID_SEEDS=1,GRID_EXTRA=--start-state=3/100/0/130@2
bash $H/submit.sh serve_model.sbatch --export=$E
```

### The two further tables on development combinations

The environment ladder and the training table can be tried on development combinations before any run of record, and every run of such a trial carries the stamp development. The ladder plays its rungs as the part `ladder`.

```bash
E=ALL,RUN_AFTER_SERVE=$H/3_run/run_lines.sh,GRID_PART=ladder
bash $H/submit.sh serve_model.sbatch --export=$E,GRID_STARTS=development
```

The trial of the training path plays the development split of the asset and writes its training files into a folder of its own, which `HYD_TRAINING` names on every line of the trial. `HYD_TRAINING` names the training folder of one asset, `<ROOT>/training` for the rig by default, and a run refuses data, adapters and blocks of another asset. The training environment is built once, and the smoke job of the trainer then makes one optimizer step on made-up examples into a folder of its own, before any sampling.

```bash
bash $H/submit.sh setup_train_env.sbatch
bash $H/submit.sh train_lora.sbatch --export=ALL,SMOKE=1
T=ALL,HYD_TRAINING=<ROOT>/training_trial
N=<runs in flight>
E=$T,RUN_AFTER_SERVE=$H/3_run/run_lines.sh,GRID_PART=sampling,GRID_PARALLEL=$N
E=$E,GRID_STARTS=training,GRID_EXTRA=--training-split=development
bash $H/submit.sh serve_model.sbatch --export=$E
RD=CausalDiagnosis.training.rejection_sampling_examples
D="python -m $RD --search \$RUNS/grid_*"
D="$D --agent <agent> --training-fold <fold>"
bash $H/submit.sh cpu_job.sbatch --export=$T -- bash -c "$D"
E=$T,AGENT=<agent>,ARM=filtered,FOLD=<fold>
bash $H/submit.sh train_lora.sbatch --export=$E
```

`N` names the runs in flight of a job of the training path, and a loop job of record of the prompt optimizer stops without it. The sampling lines play the training combinations of the development split, the data job writes the examples of one agent and fold from those runs, and each LoRA adapter trains on one card for one agent and one fold. The training job checks its card as a serving job does, and the trainer writes the card into the training manifest beside the adapter. The loops of the prompt optimizer use the judge as their reflection model. The judge runs in a serving job of its own with `SERVE_REACH=nodes`, so that the loop jobs on other nodes reach it, and every loop job names that job in `REFLECTION_JOB` and takes its key. `GEPA_SPLIT=development` plays the loops of the trial on the development split.

```bash
J=Qwen/Qwen3.8-27B-FP8
E=ALL,MODEL=$J,JUDGE_MODEL=$J,SERVE_REACH=nodes
bash $H/submit.sh serve_model.sbatch --export=$E
E=$T,JUDGE_MODEL=$J,RUN_AFTER_SERVE=$H/3_run/run_gepa_online.sh
E=$E,GRID_PARALLEL=$N
E=$E,GEPA_SPLIT=development,REFLECTION_JOB=<job id>
bash $H/submit.sh serve_model.sbatch --export=$E
touch <ROOT>/endpoints/<job id of the reflection model>.stop
```

A loop job plays at most `N` loops at once. The loops of record play a preset with the wall clock, on which a loop job plays one loop at a time at the runs in flight `RUNS_IN_FLIGHT` of `experiment_tables.py`, so the loops can also be launched as separate jobs, each naming its loops with `GEPA_AGENTS`, `GEPA_FOLDS` and `GEPA_CARDS`. The log of such a job names both numbers, the one loop in flight and the `N` it was given. Once the last loop job has ended, the last line of the block writes the stop file of the job of the reflection model, and that job ends. A requeued loop job runs under the same id and in the same folder, so every loop takes up its work with `--resume` where it stopped, and a finished loop plays never again. The job ends with status 5 when a loop failed and 4 when an export failed, and `scontrol requeue <job id>` sends its failed loops on. The evaluation lines of the trial then play the part `training` on the development split, with the adapters of the trial served beside the agent model.

```bash
E=$T,SERVE_ADAPTERS=1,RUN_AFTER_SERVE=$H/3_run/run_lines.sh,GRID_PART=training
E=$E,GRID_PARALLEL=$N,GRID_EXTRA=--training-split=development
bash $H/submit.sh serve_model.sbatch --export=$E
```

### The registered order

A run of record plays on a clone of a pushed commit once `setup.sh status` prints `SETUP_OK`, and no step by hand stands between the setup and the parts below. Every job that plays lines asks `run_ledger/ record-check` first and prints its lines into the log. On `RECORD` the runs of a part on the start set of a table land in `<ROOT>/runs/record/<tree>`, and on `DEVELOPMENT` they land in `<ROOT>/runs/development/<tree>` with the stamp development. A job submitted with `RUN_KIND=record` stops on `DEVELOPMENT` instead, and the start check of the driver stamps development any run that fails a rule of its own line, such as a preset with the wall clock at other runs in flight than `RUNS_IN_FLIGHT` of `experiment_tables.py`. A job that plays lines with the wall clock and co-serves the judge on its card stops with `WALL_CLOCK_WITH_JUDGE` before its first line. When `env.sh` names `NODES_OF_RECORD`, `submit.sh` leaves out every other node of the partition for a job of record and `run_lines.sh` refuses a host that the setting does not name, and when it names `GPU_OF_RECORD`, every job with a card checks the card. No script of this folder counts in the code identity of a run, so an edit of a script splits no group of finished runs in the tables, while the commit still pins every script and a job on a tree whose scripts differ from it plays no run of record. Every run of `run_episodes/run_record.py`, the loops of the prompt optimizer among them, records in `run.json` under `granted` the processors and the memory that Slurm gave its job, while the reproduction records none, and the tables pool the runs of a preset with the wall clock only when their grants agree, so every serving job of one wall-clock table keeps the `-c` and `--mem` of `serve_model.sbatch` and runs on one kind of node, since the processors are counted on the node and two kinds of node can grant two counts for one request. A run of a preset with the wall clock that a later job takes up under another grant than its first job is left out of the tables.

The runs of record follow in this order. The baseline policies play the test set on every preset of `grid_agents.txt`, the presets of the clock mode `generated_tokens` among them, as one CPU job. `N` is the number of runs in flight of the lines without the wall clock, and the job of the baseline policies plays at `N` runs in flight on `N` cores. Inside every job the lines without the wall clock play first at `GRID_PARALLEL` runs in flight, the log printing `PHASE_FREE` at the start and the end of that phase, and the lines with the wall clock then play one at a time under `PHASE_WALL`, at the runs in flight `RUNS_IN_FLIGHT` of `experiment_tables.py` that the pushed commit declares. The log names the card at the start and the end of the grid in `GPU_START` and `GPU_END` lines. The part `main` then plays the six rows of the main table with the healthy control in one serving job, followed by the reproduction of record, the part `companion`, the companion row, the part `environment_token`, the six rows on the full environment under the clock mode `generated_tokens`, and the part `environment`, the three rows with the CausalDiagnosis MCP on the full environment as released, which the agents table compares with the rows without it. Every part plays on a server without LoRA adapters but the part `training`, which the training table plays beside the adapters with `SERVE_ADAPTERS=1`, so the two rows of every comparison of record play on one serving setup, and the lines without training of the part `training` serve the training table alone. The judge scores the folders of record, the rescoring and the offline table of the CausalDiagnosis MCP run as CPU jobs, the offline table into `offline_mcp.csv` of the runs root of the tree, since no documented command writes a tracked file of the clone it runs from, and the environment ladder, the training table and the ablations follow by rank, each judged and rescored in the same way. The training table plays the steps of the training path above on the registered split, without `GRID_EXTRA`, `GEPA_SPLIT` and the folder of the trial, with the filtered and the unfiltered examples of every agent and fold and with `N` of this section. The loop job stops when a job of record names no `GRID_PARALLEL`. Every part is a grid file that `grid_job_lists.py part` prints inside its job, and `GRID_PARALLEL` names the runs in flight of its lines without the wall clock.

```bash
J=Qwen/Qwen3.8-27B-FP8
N=<runs in flight>
E=ALL,GRID_REFERENCE=only,GRID_STARTS=sampled,GRID_PARALLEL=$N
bash $H/submit.sh run_grid.sbatch -c $N --export=$E
P=ALL,RUN_AFTER_SERVE=$H/3_run/run_lines.sh,GRID_PARALLEL=$N
bash $H/submit.sh serve_model.sbatch --export=$P,GRID_PART=main
E=ALL,CO_SERVE_JUDGE=1,JUDGE_MODEL=$J,AOB_MODE=recovered
A=${H%/cluster_jobs}/frameworks/AssetOpsBench/cluster_jobs
E=$E,RUN_AFTER_SERVE=$A/run_reproduction.sh
bash $H/submit.sh serve_model.sbatch --export=$E
bash $H/submit.sh serve_model.sbatch --export=$P,GRID_PART=companion
bash $H/submit.sh serve_model.sbatch --export=$P,GRID_PART=environment_token
bash $H/submit.sh serve_model.sbatch --export=$P,GRID_PART=environment
E=ALL,MODEL=$J,JUDGE_MODEL=$J,RUN_AFTER_SERVE=$H/3_run/run_judge.sh
bash $H/submit.sh serve_model.sbatch --export=$E,JUDGE_FOLDERS=<grid folders>
RS=CausalDiagnosis.experiments.run_verification.rescore_under_verifier_scope
bash $H/submit.sh cpu_job.sbatch -- python -m $RS <run folders>
O=CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.offline_table
OT="python -m $O --out \$RUNS/offline_mcp.csv"
bash $H/submit.sh cpu_job.sbatch -- bash -c "$OT"
bash $H/submit.sh serve_model.sbatch --export=$P,GRID_PART=layers
bash $H/submit.sh serve_model.sbatch --export=$P,GRID_PART=layers_token
E=$P,GRID_PENDING=1
bash $H/submit.sh serve_model.sbatch --export=$E,GRID_PART=ladder
bash $H/submit.sh serve_model.sbatch --export=$P,GRID_PART=ablations
bash $H/submit.sh serve_model.sbatch --export=$P,GRID_PART=misdiagnosis_price
```

The environment ladder plays in two steps. Its four middle rungs play as two serving jobs submitted at once, each on a card of its own at the same runs in flight. The part `layers` plays them under the wall clock and the part `layers_token` plays their rungs with a clock under the clock mode `generated_tokens`, so no run of that clock shares the card of a wall-clock run. Once the job of `layers` has ended, the part `ladder` plays the three rows without the CausalDiagnosis MCP on the full environment, the baselines of the training table, as a serving job of its own at the same runs in flight, while the top rung of the ladder is the part `environment`. With `GRID_PENDING=1` a job plays only the pairs of a line and a replicate that no finished run of a grid folder of `RUNS` covers, so the twelve lines that `layers` played are left out. A finished run, for the switch as for `pending_grid_lines.py`, is a run folder whose `run.json` the driver wrote at the end of the run, with one row of `scores.csv` for every planned episode, the configuration, agent, switches and replicate of the line and the code identity of the clone, so a run that the time limit cut is no finished run, and `pending_grid_lines.py` prints its pair with `--resume` and its run folder, so that the job takes the run up from its last finished episode. A part that another part has played in part, and a part submitted again, takes `GRID_PENDING=1` once the jobs whose runs it reads have ended, so that no pair is played twice into one `RUNS` root.

A grid job on the test set plays the share of it that `GRID_FRACTION` names, 0.2 unless it is set, which is replicate 1, the one default that `start_state_seeds.py` holds for `pending_grid_lines.py` and the job. A whole share of the five replicates (0.2, 0.4, 0.6, 0.8 or 1) plays the replicates it covers as the runs of a line, so a later job with a larger share plays the missing replicates alone. A share below one replicate is a cut look, replicate 1 on a seeded prefix of the start combinations, whose runs carry `--fraction` and stand in rows of their own. Any other share is refused with the two nearest whole shares, and `GRID_REPLICATES` names whole replicates by hand in place of the share, never together with it.

The folders of record then reach the laptop and the tables as in section 4.

## 4. After a run

```bash
bash $H/4_after_run/job_record.sh <grid folder>
RC=CausalDiagnosis.experiments.cluster_runs.cluster_run_checks
bash $H/submit.sh cpu_job.sbatch -- python -m $RC suspect <grid folder>
J=Qwen/Qwen3.8-27B-FP8
E=ALL,MODEL=$J,JUDGE_MODEL=$J,RUN_AFTER_SERVE=$H/3_run/run_judge.sh
bash $H/submit.sh serve_model.sbatch --export=$E,JUDGE_FOLDERS=<grid folder>
RS=CausalDiagnosis.experiments.run_verification.rescore_under_verifier_scope
bash $H/submit.sh cpu_job.sbatch -- python -m $RS <run folder>
V=CausalDiagnosis.experiments.run_verification.verify_results_folder
E=ALL,CPU_COUCHDB=1,CPU_TABLES=1
bash $H/submit.sh cpu_job.sbatch --export=$E -- python -m $V --certify <run>
```

Every serving job writes the name, the memory, the uuid and the driver of its card into its endpoint file, and samples the card every five seconds into `gpu-<job id>.csv` beside the log of its server. After a job, `job_record.sh` writes its accounting record and copies the logs of the job from `LOGS` into the folder `job_logs` of its grid folder, the Slurm log and for a serving job the logs of its servers and the file of its GPU sampler, so the logs travel with the runs. The suspect check of `cluster_run_checks.py` names every run of a job that was suspended or requeued. Its overlap check, which a grid job runs at its end beside the suspect check, names every run alive during an episode of a preset with the wall clock and every such episode of an agent with a model endpoint whose flush of the model server's cache has no record, did not answer 200 or left cached tokens in its first request, the log then prints `OVERLAP_EXIT 1`, and a job that plays to its end ends with status 7, or with 5 when a run failed as well. The judge scores the runs after the grid, in a serving job of the judge alone over the grid folders that `JUDGE_FOLDERS` names one by one. A judging needs no number of its own, since the judge reads the whole trajectory and cuts only what passes the context of the served judge. Pass after pass, the job judges every episode without a judge record with `JUDGE_WORKERS` processes, and a second submission of the same folders continues where the first stopped. The log of the judge ends with `JUDGE_LEFT 0`, `JUDGE_FAILED_CALLS 0` and `CD_JUDGE_DONE`, and the job ends with status 1 while an episode that a pass can judge holds no judge record or a record holds a failed call, so the same submission runs again until both counts are 0. `JUDGE_NOT_JUDGED` counts the episodes that no pass can judge, a static episode without an answer and an episode whose run left no trajectory, and the lines after it name them and the episodes with a failed call. Such an episode leaves the status alone, and a table scores it false on every criterion. The rescoring is one CPU job over the runs it names. The replay certificate of one run folder is a CPU job of a clone of the commit that the run records, on the platform of the run, with CouchDB when the run played on CouchDB, and the job first computes the tables of the grid files as the tables job does, so that the replay hashes the tables the runs recorded. The certificate job ends with status 0 when the replay equals the run and writes the certificate into the run folder, with 1 when they differ, with 3 when the replay passed on another store kind or another platform, which gives no certificate, and with 5 when the replay passed on a tree that is no clone of the run's commit, such as an upload, where no certificate can be written, so that the run is certified on a clone of its commit.

```bash
CJ=CausalDiagnosis/cluster_jobs
bash $CJ/4_after_run/sync_runs.sh <ROOT>/runs/<kind>/<tree>
export PATH=$PWD/.venv/bin:$PATH PYTHONPATH=$PWD:$PWD/src
T=<folder of the tables, outside the fork>
TA=CausalDiagnosis.evaluation.result_tables.tables_from_runs
VR=CausalDiagnosis.experiments.run_verification.verify_results_folder
python -m $TA --runs <folder of the runs> --out $T/tables \
  --readme $T/README.md --published $T/runs \
  --fraction <the share the runs played, 1 for the tables of record>
python -m $VR --runs $CJ/runs_synced --results $T
cp $CJ/runs_synced/offline_mcp.csv CausalDiagnosis/results/offline_mcp.csv
```

On the laptop, `sync_runs.sh` copies the runs of one folder under `ROOT` and checks every file against the hashes of the cluster, and a synced copy writes its own events into `local_events.json` beside the ledger, which remains the cluster's byte for byte. The tables are built outside the fork from the synced folders, and `verify_results_folder.py` checks them against the ledgers of the runs. The tracked `CausalDiagnosis/results/offline_mcp.csv` is the published copy of the offline table, which the paper tables read, and it changes only by a deliberate step before a commit, the last line above, which copies over it the table that the cluster job wrote into the runs root and that `sync_runs.sh` brought with the runs.

## How many jobs a cut takes

A grid job plays the runs of its cut without the wall clock with `GRID_PARALLEL` runs in flight, and its wall-clock runs one at a time after them. Its hours are the episodes of the cut times the seconds of one episode, over 3600 times the runs in flight, and a job whose runs are fewer than the runs in flight takes the episodes of its slowest run times their seconds, over 3600. The episodes of a cut are its runs times the initial states of one run. The initial states come from the `run.json` of a trial run, and the seconds of one episode from its `CALL_LATENCY` lines. In the part `training` an agent without training plays every initial state of a replicate and the model of a fold the initial states that its fold holds out, so the episodes of a line depend on its fold. The table gives the structure of each part of `grid_job_lists.py`.

| Cut | Part | Lines | Replicates | Runs |
| --- | --- | --- | --- | --- |
| the six rows of the main table and the healthy control | `main` | 12 | 5 | 60 |
| the companion row | `companion` | 3 | 5 | 15 |
| the rungs of the environment ladder above the first | `ladder` | 15 | 5 | 75 |
| the four middle rungs of the environment ladder | `layers` | 12 | 5 | 60 |
| the evaluation of the training table | `training` | 33 | 5 | 165 |
| the ablations | `ablations` | 49 | 5 | 245 |
| the optional pilot | `pilot` | 3 | 1 | 3 |
| the optional development gate | `gate` | 2 | 1 | 2 |
| the sampling of the training table | `sampling` | 102 | 1 | 102 |
| the six rows on the full environment under the clock mode `generated_tokens` | `environment_token` | 6 | 5 | 30 |
| the middle rungs with a clock under the clock mode `generated_tokens` | `layers_token` | 9 | 5 | 45 |
| the ablation L1, the misdiagnosis price at the budget | `misdiagnosis_price` | 3 | 5 | 15 |
| the three rows with the CausalDiagnosis MCP on the full environment, as released | `environment` | 3 | 5 | 15 |

The runs in flight and the memory of the job of the baseline policies are options of the submit line. A job that the time limit ends writes its record and its checks before it stops, and the same part submitted again with `GRID_PENDING=1` plays the pairs of lines and replicates that no finished run covers, so a cut that does not fit one job splits by its lines.

## Sharing, provenance and house rules

Each person works under a `ROOT` of their own with their own clones, environments and serving jobs, and the shared folders are read only. Everyone plays the same commit, so compare the commit of the clone with the line below before the first job. Every run folder carries `ledger.json`, which records the command, the job, the code with the full SHA-256 of every file of the lists that `run.json` holds, the model and the judge with their endpoint files, the data and the seeds, and a tool that changes a finished run appends an event to the ledger. The tables read, per asset, no development run whose identity differs from that of the latest start of its agent, and a folder of runs of record holds no other run. A fault goes to the keeper of the findings with the command, the log and the commit.

```bash
git -C $H rev-parse HEAD
```

## Files of this folder

| File | What it does |
| --- | --- |
| `README.md` | this page |
| `.gitignore` | `env.sh`, `local_*.txt`, `runs_synced/` and `uploads/` |
| `env.example.sh` | the template of `env.sh`, the values of the settings |
| `functions.sh` | the functions that every job sources through `env.sh`, the folder of the scripts, the root of the tree, the folder of the grid files, and the rule of every fetch from outside the cluster with its cache |
| `submit.sh` | submits a job script of this folder or of a numbered step folder with the settings of `env.sh` |
| `cpu_job.sbatch` | any command of the tree that needs no card |
| `0_setup/clone_commit.sh` | the clone of one pushed commit, never updated in place |
| `0_setup/setup.sh` | the one setup command of a clone: the image, both models, the environment, the data, the tables and the digest label of the clone as a chain of seven jobs, and `setup.sh status` |
| `0_setup/pull_image.sbatch` | the SGLang image and the images that the frameworks of the tree start, a CPU job |
| `0_setup/download_model.sbatch` | a model from Hugging Face into the models folder, a CPU job |
| `0_setup/setup_env.sbatch` | the Python environment of the lock of the tree, a CPU job |
| `0_setup/fetch_data.sbatch` | the data of record from its official source, a CPU job |
| `0_setup/precompute_tables.sbatch` | the tables that every run reads, a CPU job and the one process that writes them on a clone: every other job sets `CD_TABLES_READ_ONLY=1` and stops with `TABLES_MISSING` on a missing table |
| `0_setup/setup_train_env.sbatch` | the training environment, a CPU job |
| `0_setup/requirements-train.txt` | the pinned packages of the training environment |
| `0_setup/upload_fork.sh` | an upload of a working tree with its manifest, for code that is not pushed, whose runs are development runs since the tree has no commit; the upload carries no `env.sh`, so the `env.sh` of the first clone is copied into its folder `CausalDiagnosis/cluster_jobs` before its first job |
| `1_checks/run_suite.sbatch` | the suite of a tree in a clean environment, a CPU job |
| `1_checks/check_nodes.sh` | optional, the label of record of the digests, the digest check of every node of the partition and the nodes that passed |
| `1_checks/check_tables.sh` | the check that runs leave the generated tables as they are |
| `1_checks/check_endpoint.sbatch` | the check of a served model from another node, with and without the key |
| `1_checks/platform_record.sh` | an optional platform record of the agent and the judge inside a serving job |
| `2_serve_model/serve_model.sbatch` | a model on one card behind the key of the job, then the script of `RUN_AFTER_SERVE` |
| `3_run/run_lines.sh` | the lines of a grid file inside a serving job, with one CouchDB |
| `3_run/run_grid.sbatch` | the runner of the lines, and the baseline policies alone as a CPU job |
| `3_run/run_judge.sh` | the judge after a grid inside a serving job of the judge alone |
| `3_run/run_gepa_online.sh` | the online loops of GEPA inside a serving job |
| `3_run/train_lora.sbatch` | the LoRA fine tuning of one agent and fold on the filtered or the unfiltered examples, on one card |
| `4_after_run/job_record.sh` | the accounting record of a finished job beside its runs |
| `4_after_run/sync_runs.sh` | the copy of the run folders to a laptop, with hashes on both sides |
| `grids/grid_agents.txt` | the 100 run lines of the main table, the healthy control, the companion row, the environment ladder and the ablations, then the rows of the full environment and of the ladder under the clock mode `generated_tokens`, the ablation L1, the misdiagnosis price at the budget, and last the three rows with the CausalDiagnosis MCP on the full environment, written by `grid_job_lists.py` |
| `grids/grid_pilot.txt` | the 3 run lines of the optional pilot of the thinking rate, written by `grid_job_lists.py` |
| `grids/grid_gate.txt` | the 2 run lines of the optional development gate, written by `grid_job_lists.py` |
| `grids/grid_training_sampling.txt` | the 102 run lines of the sampling of the training table, written by `grid_job_lists.py` |
| `grids/grid_training_evaluation.txt` | the 33 run lines of the evaluation of the training table, written by `grid_job_lists.py` |

## Optional tools

The steps below stay runnable on a clone, and no run of record needs them or reads what they print. The suite runs plain and with the digest tests as a CPU job. The judge job with `JUDGE_RECORD_ONLY=1` writes the endpoint file of the judge into `PLATFORM` and judges nothing, and the platform record, which needs that file, writes the platform of the agent and of the judge from a serving job. The latency probe of the AssetOpsBench servers runs as a CPU job. `SUITE_COUNT` is the number of tests that `python -m pytest --collect-only -q src CausalDiagnosis benchmarks` collects on the Mac for the same commit, and the suite job stops with `SUITE_COUNT_MISMATCH` before its first test when it collects another number, so that no test root goes missing in silence.

```bash
SC=ALL,SUITE_COUNT=<count on the Mac>
bash $H/submit.sh run_suite.sbatch --export=$SC
bash $H/submit.sh run_suite.sbatch --export=$SC,SUITE_DIGESTS=1
J=Qwen/Qwen3.8-27B-FP8
E=ALL,MODEL=$J,JUDGE_MODEL=$J,RUN_AFTER_SERVE=$H/3_run/run_judge.sh
bash $H/submit.sh serve_model.sbatch --export=$E,JUDGE_RECORD_ONLY=1
E=ALL,RUN_AFTER_SERVE=$H/1_checks/platform_record.sh
bash $H/submit.sh serve_model.sbatch --export=$E
LP=CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.latency_probe
bash $H/submit.sh cpu_job.sbatch -- python -m $LP
```

The optional pilot and the gate play the development starts on the clone of the pushed commit at the runs in flight `RUNS_IN_FLIGHT` of `experiment_tables.py`, the number every run of record plays, and no run of record reads what they print. `cluster_run_checks.py cap` prints the start cap of the pilot, the tables job then computes the tables of the development starts under that cap, since no job of the pilot computes a table, and `cluster_run_checks.py pilot` prints the measured thinking rate and token rate beside the values the asset file declares, at which every run of record plays, with `PILOT_OK`, or the next patience of the failure card for one more step of the pilot. The pilot refuses with `PILOT_EPISODE_TOO_LONG` an episode that cannot finish inside one job. The pilot projects every run of record from the wall seconds of its runs against the share of the job limit that `PROJECTION_SHARE` of `experiment_tables.py` names. A run past that share continues in the next job from its last finished episode, and `PILOT_SLOW` names the jobs it needs. `cluster_run_checks.py gate` then reads the gate.

```bash
C=CausalDiagnosis.experiments.cluster_runs.cluster_run_checks
bash $H/submit.sh cpu_job.sbatch -- python -m $C cap
PC=ALL,GRID_STARTS=development,GRID_EXTRA=--start-cap=<cap>
bash $H/submit.sh precompute_tables.sbatch --export=$PC
E=ALL,RUN_AFTER_SERVE=$H/3_run/run_lines.sh,GRID_STARTS=development
E=$E,GRID_SEEDS=1,GRID_EXTRA=--start-cap=<cap>,GRID_PARALLEL=$N
bash $H/submit.sh serve_model.sbatch --export=$E,GRID_PART=pilot
bash $H/submit.sh cpu_job.sbatch -- python -m $C pilot <grid folder>
bash $H/submit.sh serve_model.sbatch --export=$E,GRID_PART=gate
bash $H/submit.sh cpu_job.sbatch -- python -m $C gate <grid folder>
```

No value that the pilot prints enters a run of record. The runs in flight, the thinking rate, the token rate and the patience of the failure card are the values the pushed commit declares, and that commit is the record of the runs. A fix that a check needs after the push becomes a new pushed commit, and the steps run again on a clone of it.

`check_nodes.sh label` writes the label of record of the digests, the label `current` of `CausalDiagnosis/experiments/run_verification/invariant_digests.json`, which the digest tests of the suite compare with, by a CPU job on a node with a card, and the file then reaches the laptop and the next commit. The digest check of the nodes computes the digests again on every node of the partition, compares them with that label and prints the nodes that passed, which `NODES_OF_RECORD` of `env.sh` may then name. A node that passed on another card than `GPU_OF_RECORD` stands in a `NODE_OTHER_GPU` line.

```bash
bash $H/1_checks/check_nodes.sh label
bash $H/1_checks/check_nodes.sh submit
bash $H/1_checks/check_nodes.sh read
```

On the laptop, at the fork root, the two lines below copy the file into the fork.

```bash
A=CausalDiagnosis/experiments/run_verification/invariant_digests.json
scp insomnia:<folder of that clone>/$A $A
```
