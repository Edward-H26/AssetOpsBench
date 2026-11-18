# The stage profile of an episode and its measurement jobs

This folder reads run folders as the driver and the servers write them, finished or not, with the export of the model engine where a serving job writes one, and says where the wall time of an episode goes: one row per episode, a summary per cell and a view per cluster job. It changes no file of a run folder, and no driver, server or table of record imports it. The sections below give every quantity in plain words with its formula, how to run the profile, the engine's split of a model call, what the files of this release cannot give, the record that every reported figure carries, and the measurement jobs, each of them a development job.

## Running the profile

```bash
export PATH=$PWD/.venv/bin:$PATH PYTHONPATH=$PWD:$PWD/src
python -m CausalDiagnosis performance profile-table <run folder> [<run folder> ...] --out <folder outside the fork>
```

The command runs from the fork root. It writes `episodes.csv`, one row per episode and one column per quantity with no nested value, the table that any later plot or page reads, `engine_calls.csv`, one row per model call that the join to the engine's export reached, and `cells.json`, one summary per cell, one view per cluster job and the summary of the join, and it prints them as a readable table. An output folder inside the fork is refused, since the outputs hold measured times of runs, which stay outside the fork as the run folders do. A run folder holds `run.json`, or, for a run that did not finish, its `ledger.json` or `scores.csv` without `run.json`; a folder that holds none of them is refused, since it holds no run. `--engine-export` names the export folder of the engine of the one job that served the run folders, or its files, and is refused unless the run folders lie in one grid folder or all outside one.

A run folder inside a grid folder `grid_<job id>` also brings the view of its job, which reads every run of that grid folder, and the GPU figures of its episodes. Both read the files that `$H/4_after_run/job_record.sh` writes beside the runs after the job, `job_record.txt` and the folder `job_logs` with the Slurm log of the job and its sampler file `gpu-<job id>.csv`, and `in_flight.tsv`, which the grid runner writes during the job. The job is read from the name of the grid folder, as `CausalDiagnosis/experiments/run_verification/check_grid_complete.py` reads it, so a copy of a grid folder keeps that name. `H` names the folder `CausalDiagnosis/cluster_jobs` of the clone, as in `CausalDiagnosis/cluster_jobs/README.md`, whose section on what follows a run gives `job_record.sh` and the sync of the run folders to the laptop with `sync_runs.sh`.

A cell is the asset, the agent, the preset, the clock mode, the model, the runs in flight and the serving identity of the agent's endpoint, its weights, dtype, quantization, engine, launch flags, context and card, each read from the run's own record and never from a default. The cells of one comparison therefore differ in what the comparison varies, and no cell pools two assets, two cards or two launches. A run without `run.json` takes its cell from what its folder records, as the view of a job below says, and `unfinished_run` is 1 on every row of such a run.

## The quantities of an episode

Every quantity is in seconds of wall time unless its name says otherwise, and `episode.json` is the state file of the episode.

| Quantity | Column | Formula | In plain words |
| --- | --- | --- | --- |
| episode total T | `total_s` | `wall_s` of `scores.csv` | the attempt from before the state file exists to after the harness finish; it leaves out the verifier, the judge and the report after the attempt and the one tool listing of a run, and with a replayed attempt it is the standing attempt alone |
| creation C | `creation_s` | `handover_wall` - `episode_started_at` of `episode.json` | the state file, the launchers, the store and, on the wall clock, the flush of the model server's cache |
| cache flush | `cache_flush_s` | `flush_seconds` of `cache_flush` in `episode.json` | part of C, empty off the wall clock and for an agent without a model |
| agent part P | `agent_part_s` | `episode_ended_at` - `handover_wall` | the part in which the thinking clock runs |
| end E | `end_s` | T - C - P | the harness finish and the class of the attempt, with the short work before the state file exists; `wall_s` is rounded to 0.1 s, so E carries that rounding, and a value slightly below zero is the rounding and no fault |
| model M | `model_s` | the length of the union of [`started_at`, `ended_at`] over the records of `llm_calls.json` whose `kind` is not `tokenize`, inside [`handover_wall`, `episode_ended_at`] | overlapping calls count once, and a failed token count generates nothing |
| failed model attempts | `failed_model_s` | `failed_model_s` of `scores.csv` | the failed attempts of model requests with their waits, inside M |
| counted tool service S | `tool_service_s` | the sum of `end_wall` - `wall` over the `enter` and `call` lines of the agent's counted calls in `transcript.jsonl` | the server side of the calls that step the environment |
| processor part of S | `tool_service_cpu_s` | the same sum of `end_thread_cpu` - `thread_cpu` | the thread time of the thread that served the calls |
| no_call steps | `no_call_service_s` | the same sum of `end_wall` - `wall` over the `no_call` steps | the steps that the driver's idle hook takes after a turn without a tool call, which no runner records as a tool call, so they lie in the rest |
| server starts R | `server_starts_s` | the sum of `r_wall` - `s_wall` over the `ready` lines of `transcript.jsonl` | each tool server process from the first statement of its launcher to its stdio loop |
| environment tools on the client Tc | `env_tools_client_s` | the sum of `duration_ms` / 1000 over the records of `trajectory.json` whose `server` is not `causal_diagnosis` | the client side of every tool call of the action servers |
| diagnostics server on the client Tm | `diagnostics_client_s` | the same sum over the records whose `server` is `causal_diagnosis` | the client side alone, since a call of the diagnostics server leaves no stamp on its server side |
| transport and hold | `transport_hold_s` | Tc - S | the stdio transport, the client library and the wait for the hold of the server process |
| rest | `rest_s` | P - M - Tc - Tm - R | the agent's own code, its prompt building and parsing, the token counts that leave no record, and the idle hook |
| counted calls without an end | `counted_calls_without_end` | the count of the agent's `enter` lines without their `call` line | a call whose service has no end, which adds nothing to S |

A counted call whose caller stamped it with a time of its own, as the harness finish is, has its end at its entry and no measured service, so it enters neither S nor the checks, and `stamped_steps` counts it. An episode folder that lacks `transcript.jsonl`, `llm_calls.json` or `trajectory.json`, as the folder of a baseline policy lacks the last two, has a `note` that names the file, and every quantity that needs the file stays empty, the rest and the checks over it among them, so that a missing file never reads as zero seconds; a cell sums each quantity over the episodes that hold it. Every quantity also stands as `share_` followed by its column without `_s`, the quantity over T, and C, E, M, Tc, Tm, R and the rest add up to T. Over a cell, a share is the sum of the quantity over the kept episodes over the sum of their totals, a ratio of sums, so that every episode weighs as its seconds do, and `cells.json` gives the median and the 10th and the 90th percentile of every quantity beside it.

The charged thinking is read against the horizon. `thinking_s` and `thinking_cycles` are the sums of `think_seconds` and `think_cycles` of `episode.json`, the thinking that the counted calls charged, and `horizon_cycles` and `think_seconds_per_cycle` are those of the episode's configuration. `thinking_share_of_horizon` is `thinking_cycles` over `horizon_cycles`, how much of the horizon the thinking took. `credited_starts_s`, `credited_calls_s`, `uncredited_starts_s` and `uncredited_calls_s` split `credited_s` and `uncredited_s` of `scores.csv`, and `failed_model_attempts` and `uncharged_after_last_s` are the columns of that name. These clock quantities come from `scores.csv` and not from `episode.json`: the servers write the state file at the entry of every counted call, before that call's credit, and the clock report that the driver reads at the end of the attempt catches the credits up without writing the file. When the harness finish steps the episode, its own entry writes the earlier credits and failures into the file, and the two files agree; when the agent's own last call ends the episode, the file lacks the credit of that call and a failed model attempt after it.

The model calls are counted and timed per episode. `model_calls`, `failed_model_calls`, `call_seconds_sum` and `call_seconds_median` describe the generation calls of `llm_calls.json`, `input_tokens` and `output_tokens` sum their tokens, and `output_tokens_per_call_second` is the output tokens over the seconds of the calls that report both, a rate that still holds the prefill; the engine's split below gives the decode alone. `cells.json` spreads the seconds of one call over every call of a cell.

The GPU figures are figures of an episode. `sampler_rows` counts the rows of the job's sampler whose time lies in [`started_at`, `started_at` + T], `gpu_busy_share` is the mean of `utilization.gpu` over them as a share of one, and `gpu_energy_j` is the sum of `power.draw` over them times `gpu_period_s`, the median gap between two rows of the sampler, in joules. One row stands for one period of the sampler, which is coarse against one model call, so no figure of a single call comes from these rows. The sampler prints the local time of its node without a zone, and the profile places it by the UTC offset of the `started` line of the endpoint file that the ledger of the run keeps. `gpu_note` says why the columns of an episode stay empty, for example a copy of the sampler that does not begin with the bytes whose SHA-256 `run.json` records.

`held_card_s` is T for an episode of a model agent at one run in flight, which holds the card alone, so that its card-seconds are at least T. A cell gives them per episode and per passed episode, the card-seconds over the passed episodes, beside its pass rate.

## The engine's split of a model call

With the four flags of the export in `SGLANG_EXTRA`, `--enable-metrics`, `--enable-request-time-stats-logging`, `--export-metrics-to-file` and `--export-metrics-to-file-dir` followed by a folder, the engine writes one JSON line per generation request into the files `sglang-request-metrics-<date and hour>.log` of that folder. The lines of the measurement jobs make the folder before the submit, so that nothing rests on the engine making it. Every stamp of a line is in epoch seconds of the wall clock of the serving node, and the queue time is a duration in seconds, so the stamps compare with those of `llm_calls.json`, which the run writes on the same node. A line also holds the request itself, with its prompt. The export folder therefore lies under `ROOT` on the cluster, which its owner alone reads, and never leaves it: the profile with `--engine-export` runs there as a CPU job, the reader keeps only the stamps and the token counts of a line, and only the output folder of the profile, which holds no prompt, is copied to the laptop.

No id of a response ties a line to a call, so the join is by time. Every line received while a call is open belongs to that call, since the retry policy of the client attempts a failed request again inside one recorded call: the stage times of the attempts add, the input tokens of the call are checked against every line, and its output tokens against the line received last, the attempt whose answer the client kept. A line received while no call is open belongs to no call of the client, as the engine's warm-up at its launch, the opening call of the job and a probe of the harness at a wall cut do not, and it is passed over and counted; a line received while two calls are open belongs to neither and is counted apart. A call without a line, or without the two stamps of its span, stays unjoined. A token count of the client leaves no line, since the engine writes the export only from the output of a generation. The time ties a line to its call only when one episode at a time reaches the server, so a job whose model runs play at more runs in flight than one is not joined, and the note says why. Neither is a job with a run folder that records no agent, as a folder with `scores.csv` and no ledger, since the calls of that run may have left lines. Without `--engine-export` the columns stay empty, and `engine_note` says why.

The seconds of the generation calls of an episode, `call_seconds_sum`, split into six parts that add up to them, and each part also stands as its share of them, `call_share_` followed by its column without `_s`, so that the six shares add up to one. A call whose lines agree with it gives its queue, its prefill, its decode and the rest of the call outside those three stages; a call without a line gives its seconds to the part of the unjoined calls, and a call whose lines differ from it in their tokens to the part of the differing calls, so that no stage is guessed for either and the time of neither hides under a stage.

| Quantity | Column | Formula | In plain words |
| --- | --- | --- | --- |
| queue | `queue_s` | `queue_time` of the lines | the wait of the attempts in the engine's queue before their first forward batch |
| prefill | `prefill_s` | `prefill_finished_time` - `forward_entry_time` | the forward passes over the prompt |
| decode | `decode_s` | `request_finished_ts` - `prefill_finished_time` | the generation of every token after the first |
| outside the three stages | `outside_stages_s` | the `seconds` of the call less its queue, prefill and decode | the rest of the call outside the three stages: the client, the transport and the waits between attempts, and two spans of the engine's own time that no stage holds, from the receipt of the request, `request_received_ts`, to its entry into the queue, `forward_entry_time` less `queue_time`, and from the end of the decode, `request_finished_ts`, to the sending of the answer, `response_sent_to_client_ts`; below zero when the engine ran two attempts of the call at once, and by up to 0.005 s when the stages fill nearly all of the call, since the client rounds the `seconds` of a call to 0.01 s |
| unjoined calls | `unjoined_calls_s` | the `seconds` of the calls without a line | the calls that no line of the export answers |
| differing calls | `differing_calls_s` | the `seconds` of the calls whose lines differ from them in their tokens | the calls that the join cannot trust |
| cached prompt tokens | `cached_prompt_tokens` | `cached_tokens` of the lines | the prompt tokens that the engine found in its prefix cache |
| decode tokens | `decode_tokens` | `completion_tokens` - 1 of each line | the tokens of the decode, the first coming from the prefill |
| decode rate | `decode_tokens_per_s` | the decode tokens over the decode seconds, a ratio of sums | the rate of the decode alone |
| calls joined | `engine_calls_joined` | the generation calls of the episode with a line | the calls that the time ties to the engine's record |
| calls unjoined | `engine_calls_unjoined` | the generation calls of the episode without a line | the calls that no line of the export answers |
| lines of retries | `engine_extra_lines` | the lines of each joined call beyond its first | the attempts that the retry policy added |
| input token mismatches | `engine_input_mismatches` | the joined calls whose `input_tokens` differ from the `prompt_tokens` of a line | the check of the join on the prompt |
| output token mismatches | `engine_output_mismatches` | the joined calls whose `output_tokens` differ from the `completion_tokens` of the line received last | the check of the join on the answer |

The stages, the cached and the decode tokens and the decode rate hold the calls that agree on both counts. A cell gives every part and every share as a ratio of sums over the episodes that the join reached, those without a joined call among them, with the spread of the prefill and of the decode of one call. `engine_calls.csv` holds the figures of every joined call with its place among the generation calls of its episode in `llm_calls.json`, a token count left out, its lines and both checks, and `cells.json` counts the lines that the join read, could not read, passed over and could not place. The launch flags are part of the serving identity, so a job with the flags of the export plays cells of its own, and the serving job, a development job, names the difference from the serving file and goes on.

## The checks and the episodes left out

The rest is a difference and no measured interval, so it holds only when the parts that it subtracts do not overlap. Three checks run on every kept episode whose folder holds the files that they read, each a column, and an episode that fails a check stays in the sums of its cell, which counts per check the episodes that fail it among those that it checked.

| Check | Column | What it finds |
| --- | --- | --- |
| a line inside a model call | `lines_in_model_spans` | the agent's counted calls, from the entry to the end of their service, and the server starts whose interval meets the inside of a model call, which would put one interval in two subtracted parts |
| a rest below zero | `rest_negative` | 1 when the rest is below zero |
| a start outside the opening | `late_server_starts` | the `ready` lines that are not the first of their server or that start after the entry of the first counted call |

The last check holds the assumption of the server starts: the rest subtracts R and the client seconds apart, which holds when every server starts before the first tool call, as a runner that opens one session per server when the episode starts does. The scripted wiring check plays the released Plan-Execute executor, which starts a server process for every call, so its starts lie inside its calls, `late_server_starts` is above zero and its rest reads below zero by that design; its profile shows the wiring, never the time of an agent.

An episode stays out of the profile with its reason in `left_out`, and its cell counts it under that reason: `clock_skew`, where the wall and the monotonic lengths of the agent's part differ by more than a second, so the wall stamps and the total cannot be compared; `no state file`; a stamp of the state file missing, as for an attempt that never reached its handover; `torn transcript`; `no wall_s`; and `no episode folder` or `no row in scores.csv` for a row and a folder that do not match.

## The view of a job

A serving job holds one card from its start to its end, the Elapsed of `job_record.txt`. Its stamps split that time into parts, each also given as its share of Elapsed:

| Part | Column | From the stamp to the stamp |
| --- | --- | --- |
| server start | `server_start_s` | the start of the job in `job_record.txt` to the `started` line of the endpoint file, once the model server answers |
| opening | `opening_s` | the `started` line to the first line of the grid runner in the Slurm log, after the opening checks have passed |
| opening call | `opening_call_s` | the seconds of the one timed chat call that the line `OPENING_CALL` of the log names |
| opening checks | `opening_check_s` | the opening less the opening call: the start of CouchDB, the digest check of the node and the check of the serving file |
| to the first episode | `to_first_episode_s` | the grid runner's first line to the start of the first model episode: the check of the tables, the start of the run and its start check, and the one tool listing of the run, and every run without the served model played before it |
| episodes | `episodes_span_s` | the start of the first model episode to the end of the last model episode, with every run without the served model played between them |
| after the last episode | `after_last_episode_s` | the end of the last model episode to the end of the job: the verifier and the report of the last model episode, the end of its run, every run without the served model played after it, and the checks and the record of the job |

sacct and the log print the local time of the cluster without a zone, so both are placed by the UTC offset of the `started` line. A part whose stamps the grid folder does not hold stays empty, and the notes of the job say why: a job that started more than once, a job record that sacct wrote while the job ran, a log without its line. The last three parts read the episodes of the model runs alone, so the time of a run that plays no model episode, as a baseline policy that a line of the grid file plays or a reference agent of the grid runner, lies in them, and the notes of the job name each such run with the parts that hold its episodes. The job's card-seconds per model episode are its Elapsed over its scored model episodes, the ones that the profile leaves out among them, since each held the card, and per passed model episode over the passed ones. The Elapsed holds the time of the runs without the served model too, and the notes of the job name them where it divides its card-seconds. A cell gives the card-seconds and both rates over the jobs that played it alone, the sums of their Elapsed, of their episodes and of their windows; it leaves its card-seconds undivided when one of those jobs leaves its own undivided, and its notes name that job or, where it divides them, each job whose Elapsed holds the time of runs without the served model. An episode in flight at a cut leaves a folder without a row of `scores.csv`; it is no scored episode, so it enters neither count, and its card time stays in the Elapsed.

A run that did not finish has no `run.json`, which the driver writes at the end of a run alone: the end of the job cut it, at its time limit or by a cancel, and then its process wrote no end line in `in_flight.tsv`, or it stopped before its end, as the driver stops a run whose last episodes all end in a harness error. The view names every such run in its notes, once the job has ended with which of the two the end line of its process shows, and reads its cell from what its folder records: the agent, its label, the preset and the model of the start record of its ledger, the runs in flight from `GRID_PARALLEL` of the environment that the start record copies, the serving identity of the endpoint record that the ledger keeps, and the asset and the clock mode of the configuration in its state files. Its finished episodes are read like any other, and its GPU figures stay empty, since the prefix check of the sampler needs the record that `run.json` keeps. Where its folder does not record a field of its cell, the windows in which its process was alive give no rate, the card-seconds of the job are not divided, and the notes name the field. A folder that holds `scores.csv` without a ledger records no agent, and such a run may have played the served model: its episodes enter no figure of the model runs, the card-seconds of the job are not divided, the join to the engine's export is not made, since its calls may have left lines, and `gpu_note` and `engine_note` say that the run records no agent. The process of a run is the one that the grid runner names after the line and the replicate of the grid file that the ledger of the run records, and a window in which a process that no run folder records was alive gives no rate either.

The rate of a job is read over the windows in which exactly N model runs of the job are alive by `in_flight.tsv`, N the `runs_in_flight` of its runs, so it measures the job at N streams of the served model, the time between two episodes of a stream included, and leaves the start and the end of the job to the parts above. A run of an agent without the served model, as a baseline policy that a line of the grid file plays, counts toward no window, so a window in which only such a run is alive is no window. The window that a cut leaves open at the end of the record closes at the earliest last episode end among the runs alive in it, and the notes say so: the episodes in flight at the cut have no span and lie after that end, so the window holds none of them, and the episode whose end sets the close counts as ending inside it. An open window with a run that finished no episode inside it gives no rate. The rate is given twice, per second and per card-hour. `ending_per_s` and `ending_per_card_hour` count the episodes that end inside the windows over their length, the usual count, whose two edges cancel in expectation. `shares_per_s` and `shares_per_card_hour` sum over the episodes the share of each episode's span that lies inside the windows, over the same length, which has no edge term. The count of the episodes that lie whole inside a window is not used: at each edge it drops the episode of every stream that straddles it, so it reads low at N above one and not at one, and it understates the speedup of N runs in flight against one. A job whose model runs play more than one cell gets no rate and no card-seconds, since its card time is shared between the cells, and the episodes left out of the profile enter neither rate, since they have no span.

## What the files of this release cannot give

The profile reads the files that the driver and the servers already write, with the engine's export, which leave these quantities out. They need the recorder of the stage times of a later release.

- The time to the first token: neither route of the model calls streams, and the export holds no stamp of the first token.
- The engine's split of the calls of a job at more runs in flight than one: neither route keeps the id of a response, and the time ties a line to its call only at one run in flight.
- The server side of a call of the diagnostics server: its line in the transcript holds no stamp.
- The wall start of a tool call on the client: `trajectory.json` holds a duration alone, so Tc and Tm enter as sums without a timeline.
- A GPU sample finer than the period of the sampler.
- The time of the verifier and of the report after the attempt: no stamp closes it.

## The record of a reported figure

Every figure names the card, `gpu_name` and `gpu_mib` of the endpoint file that `run.json` records under `serving`; the dtype and the engine, the line `ENGINE_PRECISION` of the job's log and `engine_version`, `dtype`, `quantization` and `kv_cache_dtype` of the endpoint file; the launch flags, `SGLANG_EXTRA` of the job and `launch_args_sha256`; the runs in flight N of the client, `runs_in_flight`, and the context of the endpoint, `context_length`; the warm-up rule, which plays no warm-up episode, so that the start of the engine, with any warm-up of its own, and the opening call stand before the first line; the repetitions, the episodes and the runs of the cell; and the baseline it is compared with. The figures of the jobs below are figures of development runs, which describe the speed of the serving and of the harness on the named card and never stand for a result of the tables.

## The measurement jobs

Every job below runs inside a serving job of the agent model with `RUN_AFTER_SERVE=$H/3_run/run_lines.sh`, plays a grid file of its own in `grids`, a file `grids/local_<name>.txt` that git ignores and that holds the lines shown, and is a development job. Each submit line starts with `RUN_KIND=development`, which the job reads, and which `submit.sh` reads as well, so that a job with a card is not held to `NODES_OF_RECORD`. A variable set before the command reaches the job, which is how `SGLANG_EXTRA`, with its spaces, reaches the launch of the model.

- The card. Every model job of this page plays on a node with an Intel processor, the kind of the nodes of the H100, the L40S and the L40 cards, so that every run computes on one processor kind, and no job of this page plays on an A6000 node. A job that follows no earlier one is submitted as three twins, one for each of the three card types, each with an export folder of its own, and once one of them starts, the others are cancelled with `scancel`. A job that follows an earlier one, as J3 follows J1, whose seconds project the length of its run, and J4 and J5 follow J3, names the card type on which that one played, so that every cell of one comparison plays on one card type and J3 plays on the card type that J1 measured; the endpoint file and `run.json` record the card, and since the cell key holds the name and the memory of the card, a job that met another card of the same type plays a cell of its own. A job with a served model opens with the digest check of its node against the label that the setup wrote. Until the commit that declares one numeric configuration in every process that computes, a node of another processor kind than the node that wrote the label can compute other digests, and the job then stops with `OPENING_DIGESTS_DIFFER` before its first line. `GPU_OF_RECORD` stays empty in `env.sh` for these jobs, since the check of the card would stop every job on another card.
- The tables. Every grid job needs the tables of its lines first, from the tables job with the same `GRID_FILE`, `GRID_STARTS` and `GRID_EXTRA` as the grid job, `bash $H/submit.sh precompute_tables.sbatch --export=ALL,GRID_FILE=<file>,GRID_STARTS=<start set>,GRID_EXTRA=<options>`.
- The export. `SGLANG_EXTRA` carries flags of SGLang's launch, not names of this tree: the four flags of the export that the section on the engine names, the same in every cell of one comparison, and the folder of the export, which the lines below make under `ROOT` before the submit, one for every job and every twin. `D` names the export folder of a job and is set anew before the lines of each job.
- One serving job per node at a time. Two serving jobs that each start CouchDB must not share a node at one time, since the second of them fails, so every job below is submitted after the one before it has ended, and twins are cancelled as soon as one of them starts.
- The time limit. A serving job has the limit of 12 hours of `-t 12:00:00` in `serve_model.sbatch`, the job limit that the tree declares for every serving job. 180 seconds before the limit its warning reaches the batch script of the job alone, which passes it through the script of `RUN_AFTER_SERVE` to the grid runner: the grid runner writes the checks of the job and ends, the script stops CouchDB, the batch script stops the model server and ends, and the end of the job then cuts the runs still alive, which get no signal of their own. A run that the limit cuts keeps its finished episodes and its rate, but it has no GPU utilization and no energy figure, since the prefix check of the sampler needs the `run.json` that a cut run never writes.
- After every job, `job_record.sh` of the grid folder in the login shell, then the profile of its run folders with the export of the twin that played, as a CPU job on the cluster, and the copy of the output folder alone to the laptop.

```bash
D=<ROOT>/engine_export/<name of the job>
F="--enable-metrics --enable-request-time-stats-logging --export-metrics-to-file --export-metrics-to-file-dir"
twins() { for card in h100 l40s l40; do mkdir -p "$D/$card"; RUN_KIND=development SGLANG_EXTRA="$F $D/$card" bash $H/submit.sh serve_model.sbatch --gres=gpu:$card:1 --export=$1; done; }
bash $H/submit.sh cpu_job.sbatch -- python -m CausalDiagnosis performance profile-table <run folders> --out <ROOT>/profiles/<name of the job> --engine-export $D/<card type that played>
```

J1, the check of the agent: the CausalDiagnosis agent with the MCP on the preset `deterioration`, the wall clock, on the five development starts that cover every faulty level of the rig, one episode each, with the flags of the export; five is the smallest cap of the development starts that the driver accepts, and a named start state on this preset is held to its fault rule, which no development start of the rig meets. The question of J1 is a property that the records of the job show: on these starts the agent leaves its first plan and calls the tools of the rig. Its profile answers yes when the run finished with its `run.json`, every one of its five episodes is kept, and each has `counted_calls` above zero, the agent's own calls of the action servers; `tool_calls` of `scores.csv` counts the harness finish as well and is no reading of this. Any other reading is a no. On a no, J2 to J5 are not submitted, and the run folders of J1 go to whoever mends the planning of the agent, as the evidence of the fault; this page offers no way around it. J1 plays first, and its profile also shows whether every model call joins its line of the export.

```bash
# grids/local_probe.txt: --config deterioration --servers aob --store couchdb --agent causal
D=<ROOT>/engine_export/<name of the job>
E=ALL,RUN_AFTER_SERVE=$H/3_run/run_lines.sh,GRID_FILE=local_probe.txt,GRID_PARALLEL=1
E=$E,GRID_STARTS=development,GRID_SEEDS=1,GRID_EXTRA=--start-cap=5
twins $E
```

J2, the baseline and its profile: the same agent and line on the paper's condition, the wall clock with one episode at a time and the cache of the model server flushed before each episode, on one replicate of the test set, with the flags of the export. It gives the pass rate and the episode seconds of one row with their count, the profile of this folder and the engine's split of its calls.

```bash
# grids/local_baseline.txt: --config deterioration --servers aob --store couchdb --agent causal
D=<ROOT>/engine_export/<name of the job>
E=ALL,RUN_AFTER_SERVE=$H/3_run/run_lines.sh,GRID_FILE=local_baseline.txt,GRID_PARALLEL=1,GRID_FRACTION=0.2
twins $E
```

J3 and J4, the first optimization, one run in flight against eight: the sampling lines of the CausalDiagnosis agent with the MCP of one fold on the preset `deterioration_thinking_off`, whose clock charges no wall seconds, so that the lines play side by side, with the flags of the export. Each cell is one serving job of its own, so that no cell inherits another's cache. J3 plays one line with `GRID_PARALLEL=1` on the card type of J1, whose seconds project the length of its run below, and J4 eight lines, the samples 1 to 8, with `GRID_PARALLEL=8` on the card type of J3; J3 plays first, and its run shows that such a line plays before J4 spends a card on eight of them. Their rates and the speedup of eight against one come from the view of each job, and the engine's split from J3 alone, since the join holds at one run in flight.

```bash
# grids/local_sampling_1.txt: --config deterioration_thinking_off --servers aob --store couchdb --agent causal --training-fold 1 --training-sample 1
# grids/local_sampling_8.txt: the same line with --training-sample 1 to 8, one line each
T=ALL,HYD_TRAINING=<ROOT>/training_trial
D=<ROOT>/engine_export/<name of the job>
C=<the card type of J1>
E=$T,RUN_AFTER_SERVE=$H/3_run/run_lines.sh,GRID_FILE=local_sampling_1.txt,GRID_PARALLEL=1
E=$E,GRID_STARTS=training,GRID_EXTRA=--training-split=development
mkdir -p "$D/$C"
RUN_KIND=development SGLANG_EXTRA="$F $D/$C" bash $H/submit.sh serve_model.sbatch --gres=gpu:$C:1 --export=$E
```

```bash
D=<ROOT>/engine_export/<name of the job>
C=<the card type of J3>
E=$T,RUN_AFTER_SERVE=$H/3_run/run_lines.sh,GRID_FILE=local_sampling_8.txt,GRID_PARALLEL=8
E=$E,GRID_STARTS=training,GRID_EXTRA=--training-split=development
mkdir -p "$D/$C"
RUN_KIND=development SGLANG_EXTRA="$F $D/$C" bash $H/submit.sh serve_model.sbatch --gres=gpu:$C:1 --export=$E
```

Each run of J3 and J4 plays the 45 training starts of its sample, one episode each, and both are sized by the rule with which the tree's pilot sizes a run: the seconds per planned episode of a finished run, its seconds from `started` to `ended` of `run.json` over its planned episodes, which hold every cost of its job between two episodes, times the episodes of the run, against three quarters of the job limit, 3/4 x 43,200 = 32,400 seconds, the rest of the job kept for its start and for episodes slower than the measured ones. J3 is submitted when J1, the same agent on the same model and on the card type of J3, took at most 32,400 / 45 = 720 seconds per planned episode. J1 plays another preset than J3, the preset `deterioration`, whose clock charges the wall seconds and before each of whose episodes the harness empties the cache of the model server, so its seconds project those of J3 across the two presets by the pilot's rule and do not measure them; when the run so projected passes three quarters of the job, neither J3 nor J4 is submitted. J4 is submitted when J3 meets the same bound. Its eight runs share one card, so its episodes last longer than those of J3 by the factor that J4 itself measures, and the quarter of the job that the rule keeps is the only room for that factor; a run of J4 that the limit cuts nevertheless keeps its finished episodes and its rate, and loses its GPU figures.

J5, optional: the prefix cache of the engine off, the line and the variables of J3 on the card type of J3, with `--disable-radix-cache`, a flag of SGLang's launch, added after the folder of the export, submitted after J3 has ended. Its lines make an export folder of its own, and its profile reads that folder with `--engine-export $D/$C`. Its rate and its cached prompt tokens against those of J3 give the effect of the cache at one run in flight.

```bash
D=<ROOT>/engine_export/<name of the job>
C=<the card type of J3>
E=$T,RUN_AFTER_SERVE=$H/3_run/run_lines.sh,GRID_FILE=local_sampling_1.txt,GRID_PARALLEL=1
E=$E,GRID_STARTS=training,GRID_EXTRA=--training-split=development
mkdir -p "$D/$C"
RUN_KIND=development SGLANG_EXTRA="$F $D/$C --disable-radix-cache" bash $H/submit.sh serve_model.sbatch --gres=gpu:$C:1 --export=$E
```
