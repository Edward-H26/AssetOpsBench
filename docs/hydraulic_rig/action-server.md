# Hydraulic Rig Action Server

The action server in `src/servers/hydraulic_rig/` wraps one episode of the hydraulic rig environment as five Model Context Protocol (MCP) tools. `measure` reads sensors, `submit` gives a diagnosis, `submit_insufficient_evidence` declares the evidence insufficient, `replace` replaces a component, and `finish` ends the episode. The agent sees only the data it has paid to read. This page covers how the server keeps an episode between calls, how a runner starts it, the files it writes, the question that opens an episode, the tools with their answers and refusals, and the tests.

Related docs: [README.md](README.md) for the index of these pages, [environment.md](environment.md) for the environment behind the tools, [causal-diagnosis-server.md](causal-diagnosis-server.md) for the server that reads the released answers, [benchmarks.md](benchmarks.md) for the drivers that start both servers, and [../mcp-servers.md](../mcp-servers.md) for every MCP server of the repository.

---

## Overview

Each tool builds the action that `TinyHydraulicEnv.step` takes, hands it to `episode.call` in `src/servers/hydraulic_rig/episode.py`, and returns the JSON text of the outcome, refusals included. The server checks no argument itself. A missing argument or one of the wrong type reaches the environment, which refuses, logs, and counts the call like any other bad action.

The server keeps no episode state in memory, because the Plan-Execute executor of AssetOpsBench starts a new server process for every tool listing and every tool call (`_list_tools` and `_call_tool` in `src/agent/plan_execute/executor.py`). The episode lives in `episode.json` instead. Every call loads that file, builds the environment from the rig data file, calls `reset(seed)`, replays the stored actions, applies the new one, writes the episode files back, and answers. Replay is exact because the environment draws every random number from the generator it seeds at `reset`. The server takes no lock, since the executor runs one step at a time.

The server receives no argument and only a few environment variables, so it finds its episode through the location of its launcher. `src/servers/hydraulic_rig/launcher.py` starts the server on the `episode.json` in its own folder, so a copy of it beside the state file of an episode binds a server process to that episode. No tool takes an episode identifier, and `pyproject.toml` declares no entry point for this server.

- **Seen by the agent.** The question that opens the episode and, for each call, the history entry the call added, its cost, the phase, the counters, and whether the episode is over.
- **Hidden from the agent.** The seed, the hidden combination, cycle numbers, the combination after a replacement, whether a replacement was needed, the end event of the log, the data path, and the `info` dict of `TinyHydraulicEnv.step`, which carries the full log when an episode ends.
- **Measurement names.** A field name is the column name of the `hyd_1` asset of AssetOpsBench with the statistic appended, such as `PS1_Pressure_bar_100Hz_std`, and the motor power channel EPS1 reads as `P1_MotorPower_W_100Hz`. Values are rounded to six significant digits, while the log of the environment keeps the full values (`public_result` in `src/servers/hydraulic_rig/names.py`).
- **Files out of reach.** No tool reads a file that the agent names. A runner given only the launchers of an episode registers none of the six default AssetOpsBench servers, among them the utilities server, whose `json_reader` opens any JSON file. The episode folder holds the seed in `episode.json` and, after scoring, the hidden combination in `log.json` and `score.json`, so an agent with file or shell tools has to run without access to the episode and run folders.

---

## Starting the server

### Through the benchmark driver

For every agent but the oracle, `benchmarks/hydraulic_rig/run_episodes.py` prepares each episode folder before a server starts:

1. `start_episode` in `benchmarks/hydraulic_rig/common.py` calls `episode.start`, which writes `episode.json` and the first `released.json`.
2. `write_launchers` writes `hydraulic_rig_server.py` and, unless the agent is `plan_execute_rig_only`, `causal_diagnosis_server.py` from the template `_LAUNCHER`. Each launcher puts the fork's `src` at the front of the import path, turns off bytecode writing, and binds its server to the file it reads beside it.
3. Before the first episode, `check_servers` lists the tools behind every launcher through the runner's own `_list_tools` and stops the run on an unexpected tool list.
4. For a Plan-Execute agent, `PlanExecuteRunner` receives the launcher paths through its `server_paths` argument, under the server names `hydraulic_rig` and `causal_diagnosis`.

The runner turns each registered path into a command in `_make_stdio_params`:

| Registered path | Command | Working directory |
| --- | --- | --- |
| Inside the fork | `python -m <dotted module path>` | The fork root |
| Outside the fork | `python <launcher path>` | The working directory of the driver |

The command is the literal `python` found on `PATH`, so the driver has to run where `python` is an interpreter with the mcp package, as under `uv run` or `benchmarks/hydraulic_rig/run.sh`. Inside the fork the module path is built from folder names, so the driver turns the dots and slashes of a run label into underscores and refuses an `--out` folder inside the fork with a dot in its path.

### Environment of the server process

`_make_stdio_params` sets no environment for the process, so the stdio client of the mcp package passes only its default variables. Model keys never reach the server process.

| Variable | Reaches the server | Note |
| --- | --- | --- |
| `HOME`, `LOGNAME`, `PATH`, `SHELL`, `TERM`, `USER` | Yes | The default set of the mcp stdio client on Linux and macOS |
| `HYD_FACTS_PATH` | No | `episode.json` records the data file as `facts_path` |
| `PYTHONPATH` | No | The launcher of the driver puts `src` on the import path itself |
| `API_KEY`, `TOKENROUTER_API_KEY`, `TOKENROUTER_BASE_URL` | No | The server calls no model, so model keys stay in the driver process |

### By hand

A test can start the server without the driver. From the fork root, in the project environment, create an episode and copy the launcher beside it. `HYD_FACTS_PATH` matters only to the process that creates the episode.

```bash
export HYD_FACTS_PATH=~/rig-data/hyd_facts.jsonl   # the rig data file, see environment.md
export EP=$(mktemp -d)                              # the episode folder
uv run python - <<'EOF'
import os
from pathlib import Path
from envs.hydraulic_rig.data import default_facts_path
from servers.hydraulic_rig import episode
episode.start(Path(os.environ["EP"]) / "episode.json", 0, episode.default_env_kwargs(), default_facts_path())
EOF
cp src/servers/hydraulic_rig/launcher.py "$EP/server.py"
```

Any MCP client that starts `$EP/server.py` over stdio with the interpreter of the project environment now plays that episode. This one lists the tools and sends one call:

```bash
uv run python - <<'EOF'
import asyncio, os, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server = StdioServerParameters(command=sys.executable, args=[os.path.join(os.environ["EP"], "server.py")])
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print(sorted(t.name for t in (await session.list_tools()).tools))
            print((await session.call_tool("measure", {"tier": 4})).content[0].text)

asyncio.run(main())
EOF
```

Tier 4 does not exist, so the answer is a refusal that costs nothing and counts as a step:

```
['finish', 'measure', 'replace', 'submit', 'submit_insufficient_evidence']
{"event":{"event":"refused","action":{"type":"measure","tier":4},"reason":"unknown tier"},"cost":0,"phase":"investigate","budget_left":150,"repairs_left":2,"steps_left":199,"episode_over":false}
```

---

## Episode files

The server reads and writes two files in the episode folder:

| File | Written by | Read by | Content |
| --- | --- | --- | --- |
| `episode.json` | `episode.start` at creation, `episode.call` after every call | The action server and the driver | The state the server replays |
| `released.json` | The same two functions, through `write_released` | The causal diagnosis server and the audit report of the driver | Everything the agent has been told |

`episode.json` holds:

| Field | Content |
| --- | --- |
| `format` | The state format, `1`, checked on every load |
| `seed` | The seed of `reset` |
| `env_params` | Every keyword argument of `TinyHydraulicEnv`, written out from its defaults by `default_env_kwargs` and the changes of the scenario: `prices`, `budget`, `max_repairs`, `min_faulty`, `max_faulty`, `max_steps`, `block_cap`, and `allowed` |
| `facts_path` | The rig data file the server loads |
| `task_text` | The task text of the scenario, present when the driver gives one |
| `actions` | Every action passed to `TinyHydraulicEnv.step`, refused ones included, since refusals count toward the step cap |
| `tool_calls` | Every call with its tool name, arguments, and answer text, including calls after the end, which never reach the environment |

In JSON the tier keys of the prices are strings and the `allowed` combinations are lists, which `decode_params` turns back into integer keys and tuples.

`released.json` holds:

| Field | Content |
| --- | --- |
| `format` | The released format, `1` |
| `asset_id` | `hyd_1` |
| `task_text` | The task text of the state file, or null |
| `rules` | `min_faulty`, `max_faulty`, `budget`, `max_repairs`, `max_steps`, `prices`, `components`, `levels`, `healthy`, and `tiers` with the channel names of each tier |
| `status` | `phase`, `budget_left`, `repairs_left`, `steps_left`, and `episode_over` |
| `events` | The history as the agent saw it, with measurements as the tool answers showed them |

Both files go through `write_json_atomically` in `src/servers/json_files.py`, which writes a temporary file in the same folder and renames it over the target with `os.replace`, so a server stopped while writing never leaves a torn file. [benchmarks.md](benchmarks.md) describes the files that the driver adds to the folder, such as `question.txt`, `log.json`, and `score.json`.

---

## The briefing

The question that opens an episode is its briefing, composed by `briefing` in `src/servers/hydraulic_rig/briefing.py` from the observation that `reset(seed)` returns. In the Plan-Execute runner the question is the only text about the episode that the planner, the argument filler, and the summarizer all read. The argument filler in particular sees the question, the tool name, the parameter names with their JSON types and required flags, the task of the step, and the earlier answers, but no tool description. The question carries every legal value and every number of the episode, while the tool descriptions hold static guidance only.

The briefing states:

- The four components and the public rule on the number of faulty components, such as "between 1 and 2 components start off their healthy level", worded with "exactly" when `min_faulty` equals `max_faulty`.
- The condition levels of every component, with its healthy level.
- Each tool with its arguments and price, the channel names that each tier returns, and the rule that a field is named after its channel and statistic.
- The rules: one submission of either kind, no replacement before it, the repair limit, the ways an episode ends, the step cap, the budget, that a refused call counts as a step, and that a call with a missing argument or one of the wrong type is refused.

The rules ask the agent to measure first, which the environment does not enforce, since it accepts a submission before any measurement.

Nothing in the observation at reset depends on the seed, so episodes with the same parameters open with the same question. The wording carries a version, `BRIEFING_VERSION`, which the driver records in `run.json`. The driver writes each question to `question.txt` and, when the causal diagnosis server is registered, appends a paragraph on its seven tools (`TOOLS_NOTE` in `benchmarks/hydraulic_rig/run_episodes.py`).

The task text is separate. The driver passes the task text of the scenario, a short complaint worded from the public rule, to `episode.start`, which stores it in `episode.json` and releases it in `released.json`. The agent reads it only through `get_symptom_report` of the causal diagnosis server.

---

## Tools

`build_server` in `src/servers/hydraulic_rig/main.py` registers the five tools:

| Tool | Arguments | Phase | Price | An accepted call |
| --- | --- | --- | --- | --- |
| `measure` | `tier`, integer | Investigate or repair | 1, 3, or 10 for tier 1, 2, or 3 | Reads the next recorded cycle and returns the mean, std, min, and max of every channel of the tier |
| `submit` | `cooler`, `valve`, `pump`, `accumulator`, integers | Investigate | Free | Records one condition level per component and opens the repair phase |
| `submit_insufficient_evidence` | `next_action`, string | Investigate | Free | Records that the evidence cannot separate the remaining candidates, with the one action that would, and opens the repair phase without a diagnosis |
| `replace` | `component`, string | Repair | The part plus 20 for downtime | Restores the component to its healthy level, and ends the episode when every component is healthy or the repair limit is used up |
| `finish` | None | Repair | Free | Ends the episode |

With the default price list, `DEFAULT_PRICES` in `src/envs/hydraulic_rig/env.py`, the parts cost 20 for the cooler, the valve, and the accumulator, and 30 for the pump. By default the budget is 150, the repair limit 2 replacements, and the step cap 200 steps. Every step counts toward the cap, refused calls and an accepted `finish` included, and an action that would exceed the budget ends the episode. Each measurement reads the next cycle of a window of 10 recorded cycles (`block_cap`), whatever its tier, and `repeated` turns true once the reads have gone through the whole window and start over. [environment.md](environment.md) describes the windows, the phases, and the scoring.

Every parameter is flat, typed, and required in the advertised schema, with no default, since the planner and the argument filler see each parameter only as a name, a JSON type, and a required flag. Each parameter is annotated as `str`, so that FastMCP does not decode a string that looks like JSON, such as `"[2]"`, and a validator that returns the value unchanged replaces the string check (`_as_sent`). A value arrives exactly as the client sent it, so a tier sent as `"2"` or `2.0` is refused. `_advertised_as_required` drops from the schema the null defaults that let an incomplete call through.

Every answer is one JSON object, such as the refusal that the client above prints:

| Field | Content |
| --- | --- |
| `event` | The history entry the call added, as the agent sees it, or null when the call added none, as for an accepted `finish` |
| `cost` | What the call was charged |
| `phase` | `investigate` before the submission and `repair` after it |
| `budget_left`, `repairs_left`, `steps_left` | The counters after the call |
| `episode_over` | Whether the episode has ended |
| `note` | Present only on a call after the end, which changes nothing and carries the counters and `episode_over` without `event` or `cost` |

The `event` of an answer takes one of four forms:

| `event` | Other keys |
| --- | --- |
| `measure` | `tier`, `cost`, `result` with one value per field name, and `repeated` |
| `submit` | `diagnosis`, `insufficient_evidence`, and `next_action` |
| `replace` | `component` and `cost` |
| `refused` | `action` as the environment received it, and `reason` |

A refused call costs nothing and counts as one step. A refused submission leaves the investigate phase open for another attempt.

| Reason | Cause |
| --- | --- |
| `unknown tier` | A tier other than the integers 1, 2, and 3, a missing tier included |
| `diagnosis level not in the component's levels` | A level that is not an integer among the levels of the component, a missing level included |
| `insufficient_evidence needs next_action` | A `next_action` that is empty, missing, or not a string |
| `already submitted` | A second submission of either kind |
| `diagnosis must be submitted before the first repair` | `replace` before the submission |
| `unknown component` | A component other than `cooler`, `valve`, `pump`, and `accumulator` |
| `finish is allowed after the diagnosis` | `finish` before the submission |
| `budget` | A measurement or replacement that costs more than the budget left, which also ends the episode |

---

## Failure records

Every call that reaches one of the five tools is recorded in `episode.json`, and the environment logs each refusal with its reason. Calls that never reach a tool leave no trace in the episode files, so the driver counts them from the trajectory of the runner.

| Failure | What the agent gets | What is recorded | Column of `scores.csv` |
| --- | --- | --- | --- |
| A malformed call, a call out of order, or a second submission | A `refused` event with the action as sent and the reason | The action in `actions`, the call in `tool_calls`, and a `refused` event in the log, counted as a step | `refusals` |
| An action the budget cannot cover | A `refused` event with the reason `budget`, and `episode_over` true | The same, followed by the end event of the log with the reason `budget` | `refusals` and `end_reason` |
| A call after the episode has ended | The counters, `episode_over`, and `note` | The call in `tool_calls` only, since the environment is not called | `calls_after_end` |
| A call to a tool the server does not offer | The error result `Unknown tool: <name>` from FastMCP | Nothing | `calls_reaching_no_tool` |
| An argument outside the signature of the tool | The answer to the call without that argument | Nothing, since FastMCP drops the key before the tool runs | None |

The Plan-Execute executor ignores the error flag of a result, so a call to an unknown tool counts as a successful step. For a run through the Plan-Execute runner, `calls_reaching_no_tool` is the number of tool steps of the plan on this server minus the calls in `episode.json`, so it also counts a step that failed before its call.

---

## Released answers

The causal diagnosis server reasons over the episode without touching the environment, so it has to see what the agent has been told and nothing more. `episode.json` would give it the seed, from which a replay recovers the hidden combination. The action server writes `released.json` beside the state file instead, and the launcher of the causal diagnosis server binds that server to the `released.json` in its own folder (`src/servers/causal_diagnosis/launcher.py`). Apart from its own `predictions.json`, that server reads nothing else in the episode folder.

`released.json` never holds the seed, the `allowed` combinations, the data path, a cycle number, the hidden combination, the combination after a replacement, or whether a replacement was needed. `test_released_answers_hold_only_what_the_agent_saw` in `src/servers/causal_diagnosis/tests/test_server.py` checks the released events against the history as the agent sees it, and the text of the file against these keys.

---

## Reference policies through the same interface

`src/servers/hydraulic_rig/policy_client.py` lets a policy that maps an observation to an action play through the tools, on exactly what a language model agent gets:

- `tool_call` maps an action of `TinyHydraulicEnv.step` to its tool and arguments, the inverse of the five tools.
- `play` runs one episode, in which the policy sees the observation at reset and after that only what the answers carry.
- `PublicClassifiers` keys the held-out prediction table of `GreedyPolicy` by measurements as the answers show them. Its constructor refuses a table in which renaming and rounding would merge two measurements of a tier, so the policy decides through the tools as it does in a direct run.

The driver plays the reference policies `random`, `greedy`, and `greedy_prior_1to2` of `policy_table` in `benchmarks/hydraulic_rig/run_policies.py` this way. `run_policy` in `benchmarks/hydraulic_rig/run_episodes.py` sends each action through the runner's own `_call_tool`, which starts the server from the launcher of the episode like any Plan-Execute call. The oracle reads the hidden state, which no tool releases, so it runs in process.

```bash
uv run python benchmarks/hydraulic_rig/run_episodes.py --scenario S1 --agent greedy_prior_1to2 --per-record 1
```

`test_greedy_policy_on_tool_answers_takes_its_in_process_actions` plays the greedy policy through `episode.call` on seeds 0 to 4, with and without the public rule on the number of faulty components, and compares each log with a direct run of the environment.

---

## Tests

```bash
export HYD_FACTS_PATH=~/rig-data/hyd_facts.jsonl
uv run pytest src/servers/hydraulic_rig -q
```

`src/servers/hydraulic_rig/tests/test_server.py` covers the field names, the round trip of the environment parameters through JSON, replay, refusals, calls after the end, the briefing, the atomic writes, `tool_call`, the greedy policy through the answers, and the server over stdio. Some tests are skipped when a file or a package is missing:

| Missing | Effect |
| --- | --- |
| The rig data file, at `HYD_FACTS_PATH` or the default path | The whole module is skipped |
| The mcp package | `test_server_offers_five_flat_tools_and_answers_over_stdio` is skipped |
| `src/couchdb/scenarios_data/shared/iot/hydraulic_pump_1.json` | `test_hyd1_field_names_are_the_asset_s_own_columns` is skipped |

The stdio test copies `launcher.py` beside a temporary episode, starts it with the interpreter of the test, and checks the five tool names, the advertised types, that every parameter is required without a default, and that malformed calls come back as refusals and enter `episode.json` as sent. It passes its own environment with `src/` at the front of `PYTHONPATH`, so the copy imports `servers` also in the pinned runtime of `requirements-smoke.txt`, where the project is not installed.
