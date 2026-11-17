# Hydraulic Rig Action Server

The action server in `CausalDiagnosis/mcp_servers/Environment_Actions_MCP/` wraps one episode of the hydraulic rig environment as twelve Model Context Protocol (MCP) tools. The eight reads are `get_asset_profile`, `read_sensors`, `get_current_time`, `wait_for_notification`, `summarize_changes_since`, `get_failure_modes`, `get_fix_history` and `lookup_fix`, and the four writes are `submit_diagnosis`, `declare_insufficient_evidence`, `create_work_order` and `finish`. No tool sets the time, following the System app of ARE, so the agent reads the clock with `get_current_time` and can only wait with `wait_for_notification`. The agent sees the briefing, the asset profile and the answers to its own calls, and nothing else. This page covers how the server keeps an episode between calls, how a runner starts it, the state file and the released evidence, the briefing with its three guidance levels, the twelve tools with their parameters, classes, prices and refusals, the tool contract, the clock modes as the server charges them, and the tests.

Related docs: [README.md](README.md) for the index of these pages, [environment.md](environment.md) for the environment behind the tools, [causal-diagnosis-mcp.md](causal-diagnosis-mcp.md) for the CausalDiagnosis MCP, which reads the released evidence, [experiments.md](experiments.md) for the drivers that start both servers, and [../mcp-servers.md](../../docs/mcp-servers.md) for the tool tables of AssetOpsBench's servers, the action server and the CausalDiagnosis MCP.

Names on this page follow the code and the texts that an agent reads, and the [glossary](README.md#glossary) of the docs README gives the term of the paper for each of them, such as task description for briefing, abstention for declaration of insufficient evidence, instruction level for guidance level and configuration for row.

---

## Overview

Each tool builds the action that `HydraulicRigEnv.step` takes, hands it to `episode.call` in `CausalDiagnosis/mcp_servers/Environment_Actions_MCP/episode_state.py`, and returns the JSON text of the answer envelope, refusals included. The server checks no argument itself. A missing argument or one of the wrong type reaches the environment, which refuses, logs and counts the call like any other bad action, so the planner loses a step and not the episode.

The server keeps no episode state in memory, because the Plan-Execute executor of AssetOpsBench starts a new server process for every tool listing and every tool call (`_list_tools` and `_call_tool` in `src/agent/plan_execute/executor.py`). The episode lives in `episode.json` instead. Every call loads that file, builds the environment from the rig data file, calls `reset(seed)`, replays the stored actions with their stored thinking cycles, applies the new action, writes the episode files back and answers. Replay is exact because the environment draws every random number from the generators it seeds from the seed at `reset`, the episode's own and the separate ones of the hazard and the maintenance history. The server takes no lock, since the executor runs one step at a time.

The server receives no argument and only a few environment variables, so it finds its episode through the location of its launcher. `CausalDiagnosis/mcp_servers/Environment_Actions_MCP/launcher.py` starts the server on the `episode.json` in its own folder, so a copy of it beside the state file of an episode binds a server process to that episode. No tool takes an episode identifier, and `pyproject.toml` declares no entry point for this server.

- **Seen by the agent.** The briefing, the asset profile with the public normal ranges and the tool table, and for each call the answer envelope with the clock, the counters, the open work orders, the notifications since the previous call and the result of an accepted action or the reason of a refusal.
- **Hidden from the agent.** The seed, the configuration beyond its public rules, the fault rule of the initial states, the hidden combination, the recording indices, the combination after a repair, whether a work order was necessary, whether a repair took, the expiry of an emergency repair, the drawn schedule and hazard drops (the processes block publishes their parameters, never their draws), the component of a failure, the final combination, the data path, and the `info` dict of `HydraulicRigEnv.step`, which carries the full log when an episode ends.
- **Field names.** A served value is named after the column of the `hyd_1` asset of AssetOpsBench with the statistic appended, such as `PS1_Pressure_bar_100Hz_mean`, and the motor power channel EPS1 reads as `P1_MotorPower_W_100Hz`. Values are the mean and the standard deviation of a channel over one recorded cycle at six significant digits, rounded inside the environment by `served_values` in `CausalDiagnosis/datasets/dataset_loader/evidence_source.py`, so the log, the answers and the released evidence agree.
- **Files out of reach.** No tool reads a file that the agent names. A runner given only the launchers of an episode registers none of the default AssetOpsBench servers, among them the utilities server, whose `json_reader` opens any JSON file. The episode folder holds the seed in `episode.json`, and after verification the hidden state under `hidden/`, so an agent with file or shell tools has to run without access to the episode and run folders.

---

## Starting the server

### Through the benchmark driver

With `--servers rig`, `EpisodeRunner.start` in `CausalDiagnosis/experiments/episode_runners/run_episodes/` prepares each episode folder in the steps below before the action server starts, while with `--servers aob` it starts the episode through `CausalDiagnosis/frameworks/AssetOpsBench/tool_servers/episode_start_and_replay.py` and writes the launchers of AssetOpsBench's servers and the task server.

1. `episode.start` writes `episode.json` and the first `released.json`, which holds the briefing and the rules and no event.
2. `write_launchers` in `CausalDiagnosis/experiments/episode_runners/episode_launchers.py` writes `hydraulic_rig_server.py` and, unless the CausalDiagnosis MCP is switched off, `causal_diagnosis_server.py`. Each launcher puts the fork's `src` at the front of the import path, turns off bytecode writing, and binds its server to the file it reads beside it. Every launcher is written with mode 0600, since the launcher of a metered server holds the random CouchDB password of the job. The launcher of the CausalDiagnosis MCP names the ranking model to run, the models file of the asset of the run and the initial combination of the episode, whose recorded block every fit leaves out, and the abstention rule only when it differs from the default.
3. The briefing of the released evidence is written to `question.txt`, the question the runner receives.
4. Before the first episode, `check_servers` lists the tools behind every launcher through the runner's own `_list_tools` and stops the run when a server offers other tools than the contract names.
5. For a Plan-Execute agent, `PlanExecuteRunner` receives the launcher paths through its `server_paths` argument, under the server names `hydraulic_rig` and `causal_diagnosis`.

The runner turns each registered path into a command in `_make_stdio_params`:

| Registered path | Command | Working directory |
| --- | --- | --- |
| A module of a package of the fork, with `__init__.py` in every folder from the fork root down to it, as `src/servers/<name>/main.py` of AssetOpsBench | `python -m <dotted module path>` | The fork root |
| Any other file, such as the launcher of an episode, inside or outside the fork | `python -I <launcher path>` | The working directory of the driver |

The command is the literal `python` found on `PATH`, so the driver has to run with the environment's `bin` first on `PATH`, where `python` is an interpreter with the mcp package. Inside the fork the module path is built from folder names, so the driver turns the dots and slashes of a run label into underscores and refuses an `--out` folder inside the fork with a dot in its path.

### Environment of the server process

`_make_stdio_params` sets no environment for the process, so the stdio client of the mcp package passes only its default variables. Model keys never reach the server process.

| Variable | Reaches the server | Note |
| --- | --- | --- |
| `HOME`, `LOGNAME`, `PATH`, `SHELL`, `TERM`, `USER` | Yes | The default set of the mcp stdio client on Linux and macOS |
| `HYD_FACTS_PATH` | No | `episode.json` records the data file as `facts_path` |
| `PYTHONPATH` | No | The launcher of the driver puts `src` on the import path itself |
| `API_KEY`, `TOKENROUTER_API_KEY`, `TOKENROUTER_BASE_URL` | No | The server calls no model, so model keys stay in the driver process |

### By hand

A test can start the server without the driver. From the fork root, with the data file built, create an episode and copy the launcher beside it. The process that creates the episode names the data file, and `episode.json` records it.

```bash
export EP=$(mktemp -d)                              # the episode folder
PATH=$PWD/.venv/bin:$PATH PYTHONPATH=src python - <<'PY'
import os
from pathlib import Path
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.datasets.hyd_1.records import default_facts_path
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode
episode.start(Path(os.environ["EP"]) / "episode.json", 0, EpisodeConfig(), default_facts_path())
PY
cp CausalDiagnosis/mcp_servers/Environment_Actions_MCP/launcher.py "$EP/server.py"
```

Any MCP client that starts `$EP/server.py` over stdio with an interpreter that imports `servers` now plays that episode. The copied launcher imports the installed package, so the client passes `PYTHONPATH` with `src` when the fork is not installed. This client lists the tools with their read only hints and sends one call:

```bash
PATH=$PWD/.venv/bin:$PATH PYTHONPATH=src python - <<'PY'
import asyncio, os, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server = StdioServerParameters(command=sys.executable, args=[os.path.join(os.environ["EP"], "server.py")],
                                   env={**os.environ, "PYTHONPATH": "src"})
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            print(sorted(t.name for t in tools))
            print({t.name: t.annotations.readOnlyHint for t in tools})
            print((await session.call_tool("read_sensors", {"sensor_group": "tier4", "cycles": 1})).content[0].text)

asyncio.run(main())
PY
```

The sensor group `tier4` does not exist, so the answer is a refusal that costs nothing, takes no time in the declared clock mode, and counts as a step:

```
['create_work_order', 'declare_insufficient_evidence', 'finish', 'get_asset_profile', 'get_current_time', 'get_failure_modes', 'get_fix_history', 'lookup_fix', 'read_sensors', 'submit_diagnosis', 'summarize_changes_since', 'wait_for_notification']
{'get_asset_profile': True, 'read_sensors': True, 'get_current_time': True, 'wait_for_notification': True, 'summarize_changes_since': True, 'get_failure_modes': True, 'get_fix_history': True, 'lookup_fix': True, 'submit_diagnosis': False, 'declare_insufficient_evidence': False, 'create_work_order': False, 'finish': False}
{"refused":"unknown sensor group","time":0,"budget_left":300,"steps_left":59,"open_work_orders":[],"episode_over":false,"notifications":[]}
```

---

## Episode files

The server reads and writes two files in the episode folder and appends to a third, and in the `generated_tokens` clock mode it reads a fourth, which the harness of the agent appends to:

| File | Written by | Read by | Content |
| --- | --- | --- | --- |
| `episode.json` | `episode.start` at creation, `episode.call` after every call | The action server and the driver | The state the server replays |
| `released.json` | The same two functions, through `write_released` | The CausalDiagnosis MCP, the verifier and the audit report of the driver | Everything the agent has been told |
| `transcript.jsonl` | `episode.start` at creation, `episode.call` after every call, and the wrapper of `CausalDiagnosis/mcp_servers/utils/transcript_hooks.py` for a tool listing and for a call that fails before a tool runs | The replay check and the measures of the drivers | One line per tool listing and per call, with the SHA-256 of the answer |
| `generated_tokens.jsonl` | The driver, empty at the start of an episode of the `generated_tokens` mode, and the harness of the agent after every model call, through `CausalDiagnosis/environment/Causal_Diagnosis_Environment/clock/token_clock.py` | `episode.call` and the state file sessions of the metering layer, at every call of that mode | One line per model call with the tokens it generated |

`episode.json`, state format 4, holds:

| Field | Content |
| --- | --- |
| `format` | `4`, checked on every load |
| `seed` | The seed of `reset` |
| `config` | The whole `EpisodeConfig` as JSON, read back with `from_json` |
| `facts_path` | The rig data file the server loads |
| `actions` | Every action passed to `HydraulicRigEnv.step`, refused ones included, since refusals count toward the step cap |
| `think_cycles` | One entry per action, the thinking cycles charged before it, which replay passes to `step` again |
| `think_seconds` | One entry per action, the seconds of thinking charged before it, the seconds of the chess clock since the previous counted call in the `generation_time` mode (environment.md), the tokens generated since the previous call times `seconds_per_token` in the `generated_tokens` mode and 0 in the other modes |
| `think_seconds_carried` | The seconds of thinking that have not yet made a cycle, carried to the next call |
| `think_tokens` | In the `generated_tokens` mode alone, one entry per action, the tokens generated since the previous call that the thinking time before it charged |
| `generated_tokens_read` | In the `generated_tokens` mode alone, the total of `generated_tokens.jsonl` that the last call read, from which the next call counts its tokens |
| `tool_calls` | Every call with its tool name, arguments and answer text, including calls after the end, which never reach the environment |
| `last_call_wall_time` | The wall clock of the entry of the last call, from which the next call's wall gap is measured |
| `episode_started_at` | The wall clock at the start of the episode |
| `called_at` | One entry per action, the wall clock of the entry of the call that made it |
| `think_seconds_monotonic` | One entry per action in the `generation_time` mode, the monotonic clock seconds between the entries of the previous call and of this one, 0 in the other modes and null where either reading is missing |
| `think_seconds_wall` | The same gaps on the wall clock. The monotonic clock stops while the machine sleeps and does not step when the wall clock is set, so the difference of the two is a sleep or a clock step, which the chess clock never charges |
| `last_call_monotonic_time` | The monotonic clock of the entry of the last call, from which the next call's monotonic gap is measured |
| `handover_wall`, `handover_mono` | The handover of the task to the agent on both clocks, the origin of the chess clock, null until `episode.hand_over` |
| `charged_until_mono` | The cursor of the chess clock |
| `transcript_cursor`, `failures_cursor` | The byte offsets of `transcript.jsonl` and `model_failures.jsonl` read so far |
| `credited_segments` | The credited segments of the starts of the servers and of the failed model attempts that end after the cursor |
| `started_servers` | The servers whose first start is credited |
| `open_service` | The entry of the counted call in service, which the next counted call credits from the end in its transcript line |
| `refused_processes` | The server processes whose ready line shows that their readings cannot be trusted, with the reason |
| `credited_seconds`, `uncredited_seconds` | Per kind, `starts` and `calls`, the seconds of the services credited and those left to be charged |
| `failed_model_seconds`, `failed_model_attempts` | The seconds and the count of the failed model attempts credited |

`released.json`, released format 3, holds:

| Field | Content |
| --- | --- |
| `format` | `2` |
| `asset_id` | `hyd_1` |
| `task_text` | The briefing, composed at the configured guidance level |
| `guidance_level` | 1, 2 or 3 |
| `rules` | `components`, `levels`, `healthy`, `sensor_groups` with the public field names, `statistics`, `prices`, `durations`, `downtime_rate`, `budget`, `horizon`, `max_steps`, `processes`, the parameters of the schedule, hazard, emergency, dependencies, maintenance history and failure cards, under their configuration field names `schedule`, `hazard`, `emergency`, `dependencies`, `priority`, `fix_history` and `failure`, `clock_mode`, `clock`, the clock block of the tool contract with the clock modes, the order of one cycle and the notifying events and the episode's `mode`, `seconds_per_cycle` and declared duration of every action tool, `tools`, the tool table with the price and the declared duration of every tool under this configuration (`price_value`, `duration_value`), and `normal_ranges`, keyed by the field of each channel mean, so that the CausalDiagnosis MCP prices a read from what the agent was told |
| `status` | `time`, `day` when the episode is anchored, `budget_left` when the episode has a budget, `steps_left`, `open_work_orders` and `episode_over` |
| `events` | The public view of every log entry after the reset, in order, system events and the end entry included, each with `t_start`, `t_end` and `class`. A scheduled drop, a drop of the stochastic hazard, an ineffective close and the expiry of an emergency repair have no public view, and a failure carries its time but not its component. A served maintenance history appears in its `fix_history` event, the source of the `hist/<n>` identifiers |
| `observations` | One entry per served cycle, `n`, `t`, `sensor_group` and `values`, the index the evidence identifiers of readings refer to |

The released evidence never holds the seed, the configuration beyond its public rules, the fault rule of the initial states, the data path, a recording index, the entry offset, the hidden combination, the combination after a repair, whether a work order was necessary, the drawn schedule and hazard drops (the processes block publishes their parameters, never their draws) or the final combination. `test_released_evidence_holds_only_what_the_agent_saw` in `CausalDiagnosis/mcp_servers/Environment_Actions_MCP/tests/test_server.py` checks the file against the hidden fields.

Both files go through `write_json_atomically` in `CausalDiagnosis/utils/atomic_writes_and_locks.py`, which writes a temporary file in the same folder and renames it over the target, so a server stopped while writing never leaves a torn file. [experiments.md](experiments.md) describes the files that the driver adds to the folder.

`CausalDiagnosis/mcp_servers/utils/transcript.py` appends to `transcript.jsonl` under a lock on a file beside it, so the lines of every server process of an episode are numbered in the order they were written. The line of a call holds the tool, the arguments as the server received them, whether the call took a step of the environment, the wall clock reading that the call charged thinking time with, in the `generated_tokens` mode the total of `generated_tokens.jsonl` that the call read under `generated_tokens`, the SHA-256 of the text it returned and the SHA-256 of `released.json` after it. The answer text itself remains in `tool_calls` of `episode.json`. `episode.call` and the state file sessions of the metering layer read the wall clock and the monotonic clock through `WALL_CLOCK` and `MONOTONIC_CLOCK` of `episode_state.py`, and the total of `generated_tokens.jsonl` through `GENERATED_TOKENS` in the `generated_tokens` mode. `recorded_clock` sets all three for the length of a block, so that a replay of a transcript serves each recorded call under its recorded reading and needs no ledger. Nothing in a served episode sets them.

### Evidence identifiers

The agent cites evidence by identifiers that the verifier checks against the released evidence. The forms are stated in the briefing at guidance level 2:

| Identifier | Refers to |
| --- | --- |
| `obs/<n>/<field>` | one served value: observation `n` of `released.json` and the field of a channel summary |
| `wo/<id>` | a work order, such as `wo/wo-1` |
| `alarm/<t>/<channel>` | an alarm of a change summary or of a notification, which names the first cycle at which the mean of a channel of a sensor group already read in the episode left its public normal range, and the field of that mean |
| `fmea/<source>/<code>` | a failure mode code served by `get_failure_modes`, with the source `industryasseteqa_hydraulic_facts` |
| `hist/<n>` | a past work order of the maintenance history, once `get_fix_history` served it |
| `pred/<n>` | a prediction of the CausalDiagnosis MCP, `pred/<n>` from `estimate_intervention_effect` and `pred/b<n>` for a repair step or the `repair_plan` of one of the candidate plans that `rank_root_causes` returns |

---

## The briefing

The question that opens an episode is its briefing, composed by `briefing(profile, config)` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/briefing.py` from the asset profile and the episode configuration. It is the `task_text` of the released evidence and the question that the planner, the argument filler and the summarizer of the Plan-Execute runner all read. The argument filler sees the question, the tool name, the parameter names with their JSON types and required flags, the task of the step and the earlier answers, but no tool description, so at guidance level 2 the question carries the legal value of every tool parameter, while the tool descriptions hold static guidance only and no number.

| Level | What the briefing states |
| --- | --- |
| 1 | The asset (its identifier, type, description and site), the date at the start of the episode when the episode is anchored, the fact that the performance monitor flagged the unit, and the objective as one question in the form of an IndustryAssetEQA question, which condition level each component is at and which work orders restore the unit, within the budget when the episode has one and within the horizon. `config.question` replaces that question when set. It then states the rule of the answer, exactly one diagnosis or one declaration of insufficient evidence, the work orders that restore the unit, and a finish with a summary. One sentence states that a diagnosis is scored against the hidden condition at the cycle it is submitted, one states that every reading is a recorded cycle and that the readings of unchanged condition levels repeat with the length of the live window, and one names `get_current_time`, `wait_for_notification` and the notifications every answer carries, since no tool sets the time. One sentence names each deterioration card the episode switches on, the scheduled transitions while the schedule card draws them and the random transitions while the hazard card draws drops or collateral damage, and leaves their parameters to the processes block of `get_asset_profile`. One sentence each then names the repair manual and every repair rule the episode switches on, among the maintenance history, the emergency mode, the dependency rule and the visit of consecutive orders, and none of them while every repair card is off. The rules themselves are left to `get_asset_profile` |
| 2 | Level 1 plus the public details: the condition levels with the healthy level of every component, every tool of the action server with its class, duration and price, the sensor groups with their channels and read prices, the two read tools of the history and the manual, the work order lifecycle as the row makes it (an order closes inside its call without a clock or the repair card, with its visit before the next call that creates none with the card alone, and after the lead time and the repairs of its visit with a clock), with the parts, the lead times, the repair downtimes, the downtime rate of the stopped rig, the repair charge, the visit charge, the mode with its legal values, the visits and the dependency rule in words with `lookup_fix` naming the dependencies and the priorities, the emergency lead, surcharge and hold, the rule that a diagnosis or a declaration precedes a work order, the evidence identifier forms, the notifications every answer carries with the rule that an alarm is raised once per channel in an episode, the downtime rates, the budget, the horizon, the step cap with the rule that a turn that calls no tool of the rig still takes one step (with the CausalDiagnosis MCP, after the rule that a call of the diagnostics server is no call of the rig and takes no step), the clock mode sentence, the ends of an episode with the failure rule, the scoring of an episode that the wall limit ends while every call is answered or while the reply is being written, and the rule that a refused call counts as a step. When the CausalDiagnosis MCP is registered, a sentence names its four tools and the `pred/<n>` form. The entry of `submit_diagnosis` states the misdiagnosis price while it is above zero, which a wrong diagnosis adds to the cost of the episode and which does not fall from the budget |
| 3 | Level 2 plus a numbered procedure: read sensors before diagnosing, rank the candidate root causes after each reading and, while the ranking says read first, take the reading it names before you diagnose (both only with the CausalDiagnosis MCP), declare insufficient evidence with `declare_insufficient_evidence`, naming the reading that would separate the remaining candidates, when the evidence cannot separate them, look up the repair manual of every diagnosed fault, create one work order per diagnosed fault, one after another, with no other call between them, so that one visit repairs every component after the components it depends on, and wait for the notifications of their closes (the visit and the wait only where an order outlasts its call, on a row with a clock or the repair card, and the components it depends on only with the dependency card), cite the evidence identifiers behind every diagnosis, declaration and work order, finish with a summary |

Nothing in the briefing depends on the seed, so episodes with the same configuration open with the same question. The briefing states the rules of the cards the configuration switches on and no rule of a card at its off value, so the briefing of the base row states no downtime, visit charge, emergency mode, dependency rule, failure rule, deterioration or misdiagnosis price, and an episode without a budget stop has no budget sentence. On a row with a clock the finish entry of level 2 adds that the rig then runs until the open work orders close, while the downtime accrues when its card is on, and that the episode is scored after the last close. The wording carries a version, the constant `BRIEFING_VERSION` of `CausalDiagnosis/environment/Causal_Diagnosis_Environment/briefing.py`, which the driver records in `run.json`, since the wording is part of the environment definition. The guidance level is ablation En7 (table A10 of [experiments.md](experiments.md)).

---

## Tools

`build_server` in `CausalDiagnosis/mcp_servers/Environment_Actions_MCP/main.py` registers the twelve tools, each with the `readOnlyHint` annotation that the tool contract gives it. The asset of the episode's state file gives `submit_diagnosis` one condition level per component and gives the descriptions the component names, their number and the asset's short noun, so a tool listing before any episode and every episode on hyd_1 carry the texts of the hydraulic rig. The table gives the tools of hyd_1.

| Tool | Parameters | Class | Duration | Price | Scored | An accepted call |
| --- | --- | --- | --- | --- | --- | --- |
| `get_asset_profile` | none | read | 0 | free | no | returns the asset profile |
| `read_sensors` | `sensor_group` string, `cycles` integer | read | `cycles` times `read_per_cycle` | the group's read price for every cycle that returns a reading | no | runs the rig for the cycles and returns the mean and std of every channel of the group for each cycle the rig ran, with the observation number and the cycle time |
| `get_current_time` | none | read | 0 in every clock mode | free | no | returns the current cycle, the horizon, `seconds_per_cycle`, `think_seconds_per_cycle`, `seconds_per_token` in the `generated_tokens` mode, the budget left when the episode has a budget and the day of an anchored episode without moving the clock, since no tool sets the time |
| `wait_for_notification` | `timeout` integer, empty for the cycles left before the horizon | read | the cycles until a notification is pending or the timeout is reached, one cycle in the `instant` mode | free, downtime accrues while its card is on | no | lets the rig run and returns the new time, the cycles waited and what ended the wait |
| `summarize_changes_since` | `cycle` integer | read | 0 | free | no | returns the cycles elapsed, the channel means of the sensor groups already read in the episode that left their normal range since that cycle with the first such cycle, and the work orders closed since then |
| `get_failure_modes` | `component` string, empty for all | read | 0 | free | no | returns the failure mode codes with severity, recommended action, indicating channels and evidence identifier |
| `get_fix_history` | none | read | 0 | free | no | returns the work orders on the rig before the episode, oldest first, each with its identifier `hist/<n>`, component, negative closing cycle, mode and outcome |
| `lookup_fix` | `component` string | read | 0 | free | no | returns the repair manual of the component, its standard fix with the charges, the lead time and the repair downtime, its priority, its dependencies, the indicating channels (`indicating_channels`, the public fields of the channels that indicate the component), the expected effect and, while the emergency mode is on, the emergency option. The descriptions of `lookup_fix` and `create_work_order` name the visit charge, the priority with the dependencies and the emergency mode only where the episode switches their card on, `create_work_order` advertises `mode` only with the emergency mode, and the time a tool states follows the clock of the episode |
| `submit_diagnosis` | `cooler`, `valve`, `pump`, `accumulator` integers, `evidence_ids` string | write | 0 | free | yes | records one condition level per component, once per episode and never after a declaration |
| `declare_insufficient_evidence` | `sensor_group` string, `cycles` integer, `evidence_ids` string | write | 0 | free | yes | records that the evidence cannot separate the remaining candidates and the reading that would, once per episode and never after a diagnosis, and it must cite a reading already read, or the verifier scores it as a failure |
| `create_work_order` | `component`, `prediction_id`, `evidence_ids`, `mode` strings | write | the order closes when its visit has repaired it, after the latest creation cycle plus lead time of the orders of its visit and the repairs before it in priority order, and on a row without a clock inside the call while the repair card is off, or with its visit at the next call that is not an order while it is on | the part plus the repair charge, times the surcharge for an emergency order, plus the visit charge for the order that opens a visit, at creation | yes | asks a technician to restore the component, after a diagnosis or a declaration and while no order on the component is open, in the mode `standard` or, while the emergency mode is on, `emergency` |
| `finish` | `summary` string | write | 0 | free | yes | ends the episode, and on a row with a clock the rig then runs until the open work orders close, with no further call, so that the episode is scored after the last close |

Every parameter is flat, typed and required in the advertised schema, with no default, since the planner and the argument filler see each parameter only as a name, a JSON type and a required flag. Each parameter is annotated as `str` with a plain validator that returns the value unchanged (`_as_sent`), so that FastMCP does not decode a string that looks like JSON and a value arrives exactly as the client sent it: a cycle count sent as `"2"` or `2.0` reaches the environment as sent and is read as 2, as the validation of AssetOpsBench's own tools reads it, while `"two"` or `2.5` is refused. The state file records the value as sent, and the log holds the integer. `_advertised` drops from the schema the null defaults that let an incomplete call through and marks every parameter required. `evidence_ids` is a comma separated string, possibly empty, that `evidence_list` turns into the list the environment stores. `submit_diagnosis` takes one level per component of the episode's asset as flat parameters, the four levels on hyd_1, and builds the diagnosis dict, and `declare_insufficient_evidence` takes the separating test as `sensor_group` and `cycles`. The `mode` of `create_work_order` is the one parameter with a default. A missing mode stays out of the action, so the environment applies `standard`.

Every answer is one JSON object, the envelope of [environment.md](environment.md): `time`, `day` when the episode is anchored, `budget_left` when the episode has a budget, `steps_left`, `open_work_orders`, `episode_over`, `notifications`, the public events since the previous call, and `result` for an accepted call or `refused` with the reason for a refused one. A refused call costs nothing and counts as one step, and in the `instant` clock mode it advances the clock by one cycle like every other step. The refusal reasons are those of the environment, listed in [environment.md](environment.md). A call after the episode has ended is answered by the server itself with the refusal `the episode is over; this call changed nothing` and recorded in `tool_calls` only, since the environment is not called.

| Failure | What the agent gets | What is recorded |
| --- | --- | --- |
| A malformed call, a call out of order, or a second commitment | A `refused` envelope with the reason | The action in `actions`, the call in `tool_calls`, and a `refused` entry in the log, counted as a step |
| A read or a work order the budget cannot cover | A `refused` envelope with the reason `budget` and `episode_over` true | The same, followed by the end entry of the log with the reason `budget` |
| A call after the episode has ended | A `refused` envelope with the counters and `episode_over` true | The call in `tool_calls` only |
| A call to a tool the server does not offer | The error result `Unknown tool: <name>` from FastMCP | Nothing |
| An argument outside the signature of the tool | The answer to the call without that argument | Nothing, since FastMCP drops the key before the tool runs |

The Plan-Execute executor ignores the error flag of a result, so a call to an unknown tool counts as a successful step of the plan. The driver writes the trajectory of the runner to `trajectory.json` in the episode folder, and the audit report `report.md` lists such steps with the text of their answer.

**Where a cost comes from.** Every cost of an episode is read from the log of the engine, on both tool surfaces, and the price of a read is the one term that differs by surface. The action server charges the read price of the sensor group for every cycle that returns a reading. The AssetOpsBench rows charge it for each returned cycle whose fields no group paid for earlier, so a second read of a cycle is free there. The normalized cost divides the cost of an episode by the cost of the oracle policy's episode on the same seed and configuration, which the driver plays in process on `HydraulicRigEnv`. The oracle policy reads no sensor, and the engine of the AssetOpsBench rows changes no charge of a diagnosis, a work order, a wait or a finish, so one reference holds on both surfaces, as `test_the_oracle_policy_costs_the_same_on_both_engines_on_every_preset` in `CausalDiagnosis/frameworks/AssetOpsBench/tool_servers/tests/test_recorded_history_env.py` checks.

---

## The tool contract

`CausalDiagnosis/environment/Causal_Diagnosis_Environment/tool_contract.json` declares every tool of both servers, format 2. Each entry holds the `server` (`hydraulic_rig` or `causal_diagnosis`), the `class` (`read` or `write`), the `parameters` with their JSON types, the `duration` as a rule in words with `cycles_per_unit`, the `price` as a rule in words, whether the verifier scores the tool, and the `aliases` of the tool (`inspect` for `read_sensors`, `diagnose` for `submit_diagnosis` and `replace` for `create_work_order`). The four tools of the CausalDiagnosis MCP are reads with no duration and no price, unscored.

`CausalDiagnosis/environment/Causal_Diagnosis_Environment/tool_contract.py` loads the file: `load_contract(asset_id=None)`, `tools_of(server, asset_id=None)`, `tool_class(name)`, `read_only_hint(name)`, `check_advertised(server, hints, parameters=None, asset_id=None)`, which returns the list of differences between what a server advertises and the contract, and `public_table(servers=None, asset_id=None, cards_off=())`, the rows the asset profile serves under `tools`. The `card_clauses` of a tool name the parameters and rule clauses that hold only while a rule card is on, and the table of an episode with that card off leaves them out. The class travels with each MCP tool as its `readOnlyHint` annotation, the environment logs it on every entry, and `test_advertised_tools_and_hints_match_the_contract` in `CausalDiagnosis/mcp_servers/Environment_Actions_MCP/tests/test_server.py` and `test_advertised_tools_match_the_tool_contract` in `CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/tests/test_server.py` compare the advertised tools, hints and parameters with the file, so that the declaration and the code cannot drift apart.

---

## Clock modes in the server

The environment advances the clock by the declared duration of every action in the `declared` mode and by one cycle per step in the `instant` mode, except for `get_current_time`, which never moves the clock, as [environment.md](environment.md) describes. The `generation_time` mode charges the agent's thinking time as well, by the chess clock that [environment.md](environment.md) describes. `episode.call` reads the entry of the call in the thread that serves it and writes the line `enter`, and `episode.step` reads the transcript lines and the failed model attempts that arrived since the previous counted call, moves the cursor by the credit of that call's service, charges the seconds from the cursor to the entry less the credited segments, adds them to `think_seconds_carried`, `think_cycles` charges one cycle for every full `think_seconds_per_cycle` of them and carries the rest, the cycles and the seconds are stored in `think_cycles` and `think_seconds` of the state file, and the cycles are passed to `step` before the action. The end of the service is read just before the line of the call enters the transcript, and the line `exit` follows. The metering layer of the AssetOpsBench rows charges its calls through the same `episode.step`, so both surfaces charge the same cycles for the same seconds. The thinking time counts from the handover (`episode.hand_over`), and a counted call of an episode that was never handed over is refused. Replay passes the stored cycles to `step` again, so a replayed episode reproduces the log whatever the wall clock of the replay. The `generated_tokens` mode charges the thinking time from the tokens of the agent's model in place of the wall clock. `episode.call` reads the total of `generated_tokens.jsonl` in the episode folder, and `episode.step` takes the tokens since `generated_tokens_read`, charges them times `seconds_per_token` through the same `think_cycles`, stores the tokens in `think_tokens` and the seconds in `think_seconds`, and records the wall clock of the call without charging it. A call of such an episode whose folder holds no ledger fails with an error that names the file. In the other two modes the stored cycles are zero. `call` takes an optional `now` and `monotonic` for a caller that stamps its call with a time of its own, whose service then credits nothing.

---

## Baseline policies through the tools

`CausalDiagnosis/mcp_servers/Environment_Actions_MCP/policy_through_tools.py` lets a policy of `CausalDiagnosis/environment/reference_policies/policies.py` play through the tools of the action server with the two functions below, on exactly what an agent registered with that server gets, while the model agents play the rows of the environment table through AssetOpsBench's servers and the task server.

- `tool_call(action)` maps an action of `HydraulicRigEnv.step` to its tool and arguments, the inverse of the twelve tools, with the evidence identifiers joined by commas and the mode of a work order sent only when the action names one.
- `play(policy, opening, send)` runs one episode: the policy starts from the opening envelope, whose result is the asset profile (`episode.opening(state)` gives it for a stored episode), and after that sees only the answers that `send` returns, until an answer says the episode is over.

The driver plays the baseline policies `oracle`, `do_nothing`, `greedy`, `random` and `declare` this way, each action sent through the runner's own `_call_tool`, which starts the server from the launcher of the episode like any Plan-Execute call. A shadow environment in the driver process is reset with the same seed and configuration, the oracle reads the hidden state from it, and after every call the shadow steps the same action with the thinking cycles the server stored, so the served log can be compared with the shadow's at the end (`replay_matches` in `scores.csv`). `test_do_nothing_and_greedy_through_the_tools_take_their_in_process_actions` plays the two policies through `episode.call` and compares each log with a direct run of the environment.

---

## Tests

```bash
PATH=$PWD/.venv/bin:$PATH python -m pytest CausalDiagnosis/mcp_servers/Environment_Actions_MCP -q
```

`CausalDiagnosis/mcp_servers/Environment_Actions_MCP/tests/test_server.py` covers the field names, the state file, replay, refusals, calls after the end, the released evidence, the thinking cycles of the `generation_time` mode and their replay, the briefing at every guidance level, the contract match with its twelve action tools, `tool_call`, the policies through the tools, the server over stdio, and the maintenance history, the repair manual and an emergency order over stdio. Some tests are skipped when a file or a package is missing:

| Missing | Effect |
| --- | --- |
| The rig data file, in a clone that states `ASSETS_WITHOUT_DATA=1` | The whole module is skipped |
| The mcp package | The module fails to import, since it imports `build_server` from `main.py`, which imports FastMCP at import time |
| `src/couchdb/scenarios_data/shared/iot/hydraulic_pump_1.json` | The test that compares the field names with the asset's own columns is skipped |

The stdio test copies `launcher.py` beside a temporary episode, starts it with the interpreter of the test and `src` on `PYTHONPATH`, and checks the twelve tool names, the read only hints, the advertised types, that every parameter is required without a default, and that malformed calls come back as refusals and enter `episode.json` as sent. No test states a value computed from the rig data.

`CausalDiagnosis/mcp_servers/Environment_Actions_MCP/tests/test_token_clock.py` covers the `generated_tokens` mode, with the tokens since the previous call charged at the rate of the asset file, the same cycles whatever the wall seconds between calls, the error of an episode without a ledger, the replay of a transcript under its recorded totals with no ledger, and the briefing that states the rule in that mode alone.
