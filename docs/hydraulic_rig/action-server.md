# Hydraulic Rig Action Server

The action server in `src/servers/hydraulic_rig/` wraps one episode of the hydraulic rig environment as nine Model Context Protocol (MCP) tools: five reads, `get_asset_profile`, `read_sensors`, `advance_time`, `summarize_changes_since` and `get_failure_modes`, and four writes, `submit_diagnosis`, `declare_insufficient_evidence`, `create_work_order` and `finish`. The agent sees the briefing, the asset profile and the answers to its own calls, and nothing else. This page covers how the server keeps an episode between calls, how a runner starts it, the state file and the released evidence, the briefing with its three guidance levels, the nine tools with their parameters, classes, prices and refusals, the tool contract, the clock modes as the server charges them, and the tests.

Related docs: [README.md](README.md) for the index of these pages, [environment.md](environment.md) for the environment behind the tools, [causal-diagnosis-server.md](causal-diagnosis-server.md) for the server that reads the released evidence, [benchmarks.md](benchmarks.md) for the drivers that start both servers, and [../mcp-servers.md](../mcp-servers.md) for every MCP server of the repository.

---

## Overview

Each tool builds the action that `HydraulicRigEnv.step` takes, hands it to `episode.call` in `src/servers/hydraulic_rig/episode.py`, and returns the JSON text of the answer envelope, refusals included. The server checks no argument itself. A missing argument or one of the wrong type reaches the environment, which refuses, logs and counts the call like any other bad action, so the planner loses a step and not the episode.

The server keeps no episode state in memory, because the Plan-Execute executor of AssetOpsBench starts a new server process for every tool listing and every tool call (`_list_tools` and `_call_tool` in `src/agent/plan_execute/executor.py`). The episode lives in `episode.json` instead. Every call loads that file, builds the environment from the rig data file, calls `reset(seed)`, replays the stored actions with their stored thinking cycles, applies the new action, writes the episode files back and answers. Replay is exact because the environment draws every random number from the generator it seeds at `reset`. The server takes no lock, since the executor runs one step at a time.

The server receives no argument and only a few environment variables, so it finds its episode through the location of its launcher. `src/servers/hydraulic_rig/launcher.py` starts the server on the `episode.json` in its own folder, so a copy of it beside the state file of an episode binds a server process to that episode. No tool takes an episode identifier, and `pyproject.toml` declares no entry point for this server.

- **Seen by the agent.** The briefing, the asset profile with the public healthy bands and the tool table, and for each call the answer envelope: the clock, the counters, the open work orders and the result of an accepted action or the reason of a refusal.
- **Hidden from the agent.** The seed, the configuration, the hidden combination, the recording indices, the combination after a repair, whether a work order was necessary, the schedule, the external event, the component of a failure, the final combination, the data path, and the `info` dict of `HydraulicRigEnv.step`, which carries the full log when an episode ends.
- **Field names.** A served value is named after the column of the `hyd_1` asset of AssetOpsBench with the statistic appended, such as `PS1_Pressure_bar_100Hz_mean`, and the motor power channel EPS1 reads as `P1_MotorPower_W_100Hz`. Values are the mean and the standard deviation of a channel over one recorded cycle at six significant digits, rounded inside the environment, so the log, the answers and the released evidence agree (`public_result` in `src/servers/hydraulic_rig/names.py`).
- **Files out of reach.** No tool reads a file that the agent names. A runner given only the launchers of an episode registers none of the default AssetOpsBench servers, among them the utilities server, whose `json_reader` opens any JSON file. The episode folder holds the seed in `episode.json`, and after verification the hidden state under `hidden/`, so an agent with file or shell tools has to run without access to the episode and run folders.

---

## Starting the server

### Through the benchmark driver

`EpisodeRunner.start` in `benchmarks/hydraulic_rig/run_episodes.py` prepares each episode folder before a server starts:

1. `episode.start` writes `episode.json` and the first `released.json`, which holds the briefing and the rules and no event.
2. `write_launchers` in `benchmarks/hydraulic_rig/common.py` writes `hydraulic_rig_server.py` and, unless the diagnostics server is switched off, `causal_diagnosis_server.py`. Each launcher puts the fork's `src` at the front of the import path, turns off bytecode writing, and binds its server to the file it reads beside it. The diagnostics launcher names the ranking model the server runs, and the abstention rule only when it differs from the default.
3. The briefing of the released evidence is written to `question.txt`, the question the runner receives.
4. Before the first episode, `check_servers` lists the tools behind every launcher through the runner's own `_list_tools` and stops the run when a server offers other tools than the contract names.
5. For a Plan-Execute agent, `PlanExecuteRunner` receives the launcher paths through its `server_paths` argument, under the server names `hydraulic_rig` and `causal_diagnosis`.

The runner turns each registered path into a command in `_make_stdio_params`:

| Registered path | Command | Working directory |
| --- | --- | --- |
| Inside the fork | `python -m <dotted module path>` | The fork root |
| Outside the fork | `python <launcher path>` | The working directory of the driver |

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

A test can start the server without the driver. From the fork root, with `HYD_FACTS_PATH` set, create an episode and copy the launcher beside it. `HYD_FACTS_PATH` matters only to the process that creates the episode.

```bash
export HYD_FACTS_PATH=~/rig-data/hyd_facts.jsonl   # the rig data file, see environment.md
export EP=$(mktemp -d)                              # the episode folder
PATH=$PWD/.venv/bin:$PATH PYTHONPATH=src python - <<'PY'
import os
from pathlib import Path
from envs.hydraulic_rig.config import EpisodeConfig
from envs.hydraulic_rig.data import default_facts_path
from servers.hydraulic_rig import episode
episode.start(Path(os.environ["EP"]) / "episode.json", 0, EpisodeConfig(), default_facts_path())
PY
cp src/servers/hydraulic_rig/launcher.py "$EP/server.py"
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
['advance_time', 'create_work_order', 'declare_insufficient_evidence', 'finish', 'get_asset_profile', 'get_failure_modes', 'read_sensors', 'submit_diagnosis', 'summarize_changes_since']
{'get_asset_profile': True, 'read_sensors': True, 'advance_time': True, 'summarize_changes_since': True, 'get_failure_modes': True, 'submit_diagnosis': False, 'declare_insufficient_evidence': False, 'create_work_order': False, 'finish': False}
{"refused":"unknown sensor group","time":0,"budget_left":300,"steps_left":59,"open_work_orders":[],"episode_over":false}
```

---

## Episode files

The server reads and writes two files in the episode folder:

| File | Written by | Read by | Content |
| --- | --- | --- | --- |
| `episode.json` | `episode.start` at creation, `episode.call` after every call | The action server and the driver | The state the server replays |
| `released.json` | The same two functions, through `write_released` | The diagnostics server, the verifier and the audit report of the driver | Everything the agent has been told |

`episode.json`, state format 2, holds:

| Field | Content |
| --- | --- |
| `format` | `2`, checked on every load |
| `seed` | The seed of `reset` |
| `config` | The whole `EpisodeConfig` as JSON, read back with `from_json` |
| `facts_path` | The rig data file the server loads |
| `actions` | Every action passed to `HydraulicRigEnv.step`, refused ones included, since refusals count toward the step cap |
| `think_cycles` | One entry per action, the thinking cycles charged before it, which replay passes to `step` again |
| `tool_calls` | Every call with its tool name, arguments and answer text, including calls after the end, which never reach the environment |
| `last_call_wall_time` | The wall clock of the last call, from which the next call's thinking time is measured |

`released.json`, released format 2, holds:

| Field | Content |
| --- | --- |
| `format` | `2` |
| `asset_id` | `hyd_1` |
| `task_text` | The briefing, composed at the configured guidance level |
| `guidance_level` | 1, 2 or 3 |
| `rules` | `components`, `levels`, `healthy`, `sensor_groups` with the public field names, `statistics`, `prices`, `durations`, `downtime_rate`, `budget`, `horizon`, `max_steps`, `fault_rule`, `clock_mode` and `healthy_bands`, keyed by the field of each channel mean |
| `status` | `time`, `day` when the episode is anchored, `budget_left`, `steps_left`, `open_work_orders` and `episode_over` |
| `events` | The public view of every log entry after the reset, in order, system events and the end entry included, each with `t_start`, `t_end` and `class`. A scheduled drop has no public view, and a failure carries its time but not its component |
| `observations` | One entry per served cycle, `n`, `t`, `sensor_group` and `values`, the index the evidence identifiers of readings refer to |

The released evidence never holds the seed, the configuration, the data path, a recording index, the hidden combination, the combination after a repair, whether a work order was necessary, the schedule, the external event or the final combination. `test_released_evidence_holds_only_what_the_agent_saw` in `src/servers/hydraulic_rig/tests/test_server.py` checks the file against the hidden fields.

Both files go through `write_json_atomically` in `src/servers/json_files.py`, which writes a temporary file in the same folder and renames it over the target, so a server stopped while writing never leaves a torn file. [benchmarks.md](benchmarks.md) describes the files that the driver adds to the folder.

### Evidence identifiers

The agent cites evidence by identifiers that the verifier checks against the released evidence. The forms are stated in the briefing at guidance level 2:

| Identifier | Refers to |
| --- | --- |
| `obs/<n>/<field>` | one served value: observation `n` of `released.json` and the field of a channel summary |
| `wo/<id>` | a work order, such as `wo/wo-1`, or `wo/wo-ext-1` for the technician's order |
| `alarm/<t>/<channel>` | an alarm of a change summary: the first cycle at which the channel mean left its public band, and the field of that mean |
| `fmea/<source>/<code>` | a failure mode code served by `get_failure_modes`, with the source `industryasseteqa_hydraulic_facts` |
| `pred/<n>` | a prediction stored by `estimate_intervention_effect` of the diagnostics server |

---

## The briefing

The question that opens an episode is its briefing, composed by `briefing(profile, config)` in `src/servers/hydraulic_rig/briefing.py` from the asset profile and the episode configuration. It is the `task_text` of the released evidence and the question that the planner, the argument filler and the summarizer of the Plan-Execute runner all read. The argument filler sees the question, the tool name, the parameter names with their JSON types and required flags, the task of the step and the earlier answers, but no tool description, so at guidance level 2 the question carries the legal value of every tool parameter, while the tool descriptions hold static guidance only and no number.

| Level | What the briefing states |
| --- | --- |
| 1 | The asset (its identifier, type, description and site), the date of the start when the episode is anchored, the fact that the performance monitor flagged the unit, and the objective as one question in the form of an IndustryAssetEQA question: which condition level each component is at and which work orders restore the unit, within the budget and the horizon. `config.question` replaces that question when set. It ends with the rule of the answer: exactly one diagnosis or one declaration of insufficient evidence, the work orders that restore the unit, and a finish with a summary. The rules themselves are left to `get_asset_profile` |
| 2 | Level 1 plus the public details: the condition levels with the healthy level of every component, the fault count rule in words, every tool of the action server with its class, duration and price, the sensor groups with their channels and read prices, the work order lifecycle with the parts, the lead times, the repair downtimes, the stopped rate and the repair charge, the rule that a diagnosis or a declaration precedes a work order, the evidence identifier forms, the downtime rates, the budget, the horizon, the step cap, the clock mode sentence, the ends of an episode with the failure rule, and the rule that a refused call counts as a step. When the diagnostics server is registered, a sentence names its four tools and the `pred/<n>` form |
| 3 | Level 2 plus a numbered procedure: read sensors before diagnosing, rank the candidates after each reading and declare insufficient evidence with the separating test the ranking names when it reports the evidence as insufficient (both only with the diagnostics server), create one work order per diagnosed fault and advance time until the orders close, cite the evidence identifiers behind every diagnosis and work order, finish with a summary |

Nothing in the briefing depends on the seed, so episodes with the same configuration open with the same question. The wording carries a version, `BRIEFING_VERSION`, which the driver records in `run.json`, since the wording is part of the environment definition. The guidance level is ablation A10 of [benchmarks.md](benchmarks.md).

---

## Tools

`build_server` in `src/servers/hydraulic_rig/main.py` registers the nine tools, each with the `readOnlyHint` annotation that the tool contract gives it:

| Tool | Parameters | Class | Duration | Price | Scored | An accepted call |
| --- | --- | --- | --- | --- | --- | --- |
| `get_asset_profile` | none | read | 0 | free | no | returns the asset profile |
| `read_sensors` | `sensor_group` string, `cycles` integer | read | `cycles` times `read_per_cycle` | `cycles` times the group's read price | no | runs the rig for the cycles and returns the mean and std of every channel of the group for each cycle the rig ran, with the observation number and the cycle time |
| `advance_time` | `cycles` integer | read | `cycles` times `advance_per_cycle` | free, downtime accrues | no | lets the rig run and returns the new time |
| `summarize_changes_since` | `cycle` integer | read | 0 | free | no | returns the cycles elapsed, the channel means that left their healthy band since that cycle with the first such cycle, and the work orders closed since then |
| `get_failure_modes` | `component` string, empty for all | read | 0 | free | no | returns the failure mode codes with severity, recommended action, indicating channels and evidence identifier |
| `submit_diagnosis` | `cooler`, `valve`, `pump`, `accumulator` integers, `evidence_ids` string | write | 0 | free | yes | records one condition level per component, once per episode and never after a declaration |
| `declare_insufficient_evidence` | `sensor_group` string, `cycles` integer, `evidence_ids` string | write | 0 | free | yes | records that the evidence cannot separate the remaining candidates and the reading that would, once per episode and never after a diagnosis |
| `create_work_order` | `component`, `prediction_id`, `evidence_ids` strings | write | the order closes after the lead time and the repair downtime | the part plus the repair charge, at creation | yes | asks a technician to restore the component, after a diagnosis or a declaration and while no order on the component is open |
| `finish` | `summary` string | write | 0 | free | yes | ends the episode, open work orders stay open |

Every parameter is flat, typed and required in the advertised schema, with no default, since the planner and the argument filler see each parameter only as a name, a JSON type and a required flag. Each parameter is annotated as `str` with a plain validator that returns the value unchanged (`_as_sent`), so that FastMCP does not decode a string that looks like JSON and a value arrives exactly as the client sent it: a cycle count sent as `"2"` or `2.0` is refused by the environment. `_advertised` drops from the schema the null defaults that let an incomplete call through and marks every parameter required. `evidence_ids` is a comma separated string, possibly empty, that `evidence_list` turns into the list the environment stores. `submit_diagnosis` takes the four levels as flat parameters and builds the diagnosis dict, and `declare_insufficient_evidence` takes the separating test as `sensor_group` and `cycles`.

Every answer is one JSON object, the envelope of [environment.md](environment.md): `time`, `day` when the episode is anchored, `budget_left`, `steps_left`, `open_work_orders`, `episode_over`, and `result` for an accepted call or `refused` with the reason for a refused one. A refused call costs nothing and counts as one step, and in the `instant` clock mode it advances the clock by one cycle like every other step. The refusal reasons are those of the environment, listed in [environment.md](environment.md). A call after the episode has ended is answered by the server itself with the refusal `the episode is over; this call changed nothing` and recorded in `tool_calls` only, since the environment is not called.

| Failure | What the agent gets | What is recorded |
| --- | --- | --- |
| A malformed call, a call out of order, or a second commitment | A `refused` envelope with the reason | The action in `actions`, the call in `tool_calls`, and a `refused` entry in the log, counted as a step |
| A read or a work order the budget cannot cover | A `refused` envelope with the reason `budget` and `episode_over` true | The same, followed by the end entry of the log with the reason `budget` |
| A call after the episode has ended | A `refused` envelope with the counters and `episode_over` true | The call in `tool_calls` only |
| A call to a tool the server does not offer | The error result `Unknown tool: <name>` from FastMCP | Nothing |
| An argument outside the signature of the tool | The answer to the call without that argument | Nothing, since FastMCP drops the key before the tool runs |

The Plan-Execute executor ignores the error flag of a result, so a call to an unknown tool counts as a successful step of the plan. The driver counts such steps from the trajectory of the runner.

---

## The tool contract

`src/servers/tool_contract_v2.json` declares every tool of both servers, format 2. Each entry holds the `server` (`hydraulic_rig` or `causal_diagnosis`), the `class` (`read` or `write`), the `parameters` with their JSON types, the `duration` as a rule in words with `cycles_per_unit`, the `price` as a rule in words, whether the verifier scores the tool, and the `aliases` the proof of concept used for it (`inspect` for `read_sensors`, `diagnose` for `submit_diagnosis`, `replace` for `create_work_order`, and `advance_time`). The four tools of the diagnostics server are reads with no duration and no price, unscored.

`src/servers/contract.py` loads the file: `load_contract`, `tools_of(server)`, `tool_class(name)`, `read_only_hint(name)`, `check_advertised(server, hints, parameters=None)`, which returns the list of differences between what a server advertises and the contract, and `public_table(servers=None)`, the rows the asset profile serves under `tools`. The class travels with each MCP tool as its `readOnlyHint` annotation, the environment logs it on every entry, and `test_advertised_tools_and_hints_match_the_contract` in the test file of each server compares the advertised tools, hints and parameters with the file, so that the declaration and the code cannot drift apart.

---

## Clock modes in the server

The environment advances the clock by the declared duration of every action in the `declared` mode and by one cycle per step in the `instant` mode, as [environment.md](environment.md) describes. The `generation_time` mode charges the agent's thinking time as well, and that time is measured here: `episode.call` takes the wall clock of the call, `think_cycles` converts the seconds since `last_call_wall_time` into cycles by dividing by `seconds_per_cycle` and rounding up, the result is stored in `think_cycles` of the state file and passed to `step` before the action, and `last_call_wall_time` moves to the call. `episode.start` sets the wall clock too, so the thinking time of the first call counts from the start of the episode. Replay passes the stored cycles to `step` again, so a replayed episode reproduces the log whatever the wall clock of the replay. In the other two modes the stored cycles are zero. `call` takes an optional `now` for tests.

---

## Reference policies through the tools

`src/servers/hydraulic_rig/policy_client.py` lets a policy of `src/envs/hydraulic_rig/policies.py` play through the tools, on exactly what a language model agent gets:

- `tool_call(action)` maps an action of `HydraulicRigEnv.step` to its tool and arguments, the inverse of the nine tools, with the evidence identifiers joined by commas.
- `play(policy, opening, send)` runs one episode: the policy starts from the opening envelope, whose result is the asset profile (`episode.opening(state)` gives it for a stored episode), and after that sees only the answers that `send` returns, until an answer says the episode is over.

The driver plays the reference policies `oracle`, `do_nothing` and `greedy` this way, each action sent through the runner's own `_call_tool`, which starts the server from the launcher of the episode like any Plan-Execute call. A shadow environment in the driver process is reset with the same seed and configuration, the oracle reads the hidden state from it, and after every call the shadow steps the same action with the thinking cycles the server stored, so the served log can be compared with the shadow's at the end (`replay_matches` in `scores.csv`). `test_do_nothing_and_greedy_through_the_tools_take_their_in_process_actions` plays the two policies through `episode.call` and compares each log with a direct run of the environment.

---

## Tests

```bash
export HYD_FACTS_PATH=~/rig-data/hyd_facts.jsonl
PATH=$PWD/.venv/bin:$PATH python -m pytest src/servers/hydraulic_rig -q
```

`src/servers/hydraulic_rig/tests/test_server.py` covers the field names, the state file, replay, refusals, calls after the end, the released evidence, the thinking cycles of the `generation_time` mode and their replay, the briefing at every guidance level, the contract match, `tool_call`, the policies through the tools, and the server over stdio. Some tests are skipped when a file or a package is missing:

| Missing | Effect |
| --- | --- |
| The rig data file, at `HYD_FACTS_PATH` or the default path | The whole module is skipped |
| The mcp package | The stdio test is skipped |
| `src/couchdb/scenarios_data/shared/iot/hydraulic_pump_1.json` | The test that compares the field names with the asset's own columns is skipped |

The stdio test copies `launcher.py` beside a temporary episode, starts it with the interpreter of the test and `src` on `PYTHONPATH`, and checks the nine tool names, the read only hints, the advertised types, that every parameter is required without a default, and that malformed calls come back as refusals and enter `episode.json` as sent. No test states a value computed from the rig data.
