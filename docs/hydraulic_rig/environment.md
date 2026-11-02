# Hydraulic Rig Environment

`src/envs/hydraulic_rig/` turns the public UCI hydraulic rig recordings into episodes of finding the faults of an asset and restoring it on a clock. This page covers the data, the episode configuration, the clock with its three modes and its hidden processes, the nine actions with their refusals, the log and the information barrier, the oracle, the reference policies, the verifier with its checks, failure classes, scores and reward, and the Gymnasium and OpenEnv wrappers. The environment uses the Python standard library and simulates nothing: every state of the rig is answered with recorded cycles.

Related docs: [README.md](README.md) for the index of these pages, [action-server.md](action-server.md) for the MCP tools that serve an episode, [causal-diagnosis-server.md](causal-diagnosis-server.md) for the diagnostics tools, [benchmarks.md](benchmarks.md) for the drivers and the result tables, and [../mcp-servers.md](../mcp-servers.md) for every MCP server of the repository.

---

## Overview

Every episode hides one combination of the condition levels of four components, the cooler, the valve, the pump and the accumulator, drawn from the seed within a public rule on the number of components off their healthy level. Time counts rig cycles from the start of the episode and advances only inside an action. The agent reads a sensor group for a number of cycles at a price and a duration, advances time, summarizes the changes since a cycle, looks up the asset profile and the failure modes, submits one diagnosis or one declaration of insufficient evidence, creates work orders that a technician executes after a lead time and a repair downtime, and finishes with a summary. Three hidden processes run on the clock: a degradation schedule that drops a component level by level, an external replacement by a technician, and a failure rule. Downtime accrues per cycle while a component is degraded and counts against the budget with the prices.

Every change of the hidden combination, by a repair, a scheduled drop or an external replacement, switches the served recording to the stable block of the new combination, so a reading after a repair is a recording of the repaired rig. The verifier compares the agent's writes with the oracle of the episode and scores the episode from the log and the hidden state. The answer text enters nowhere but the contradiction heuristic of the failure taxonomy.

---

## Data

The recordings are dataset 447 of the UCI Machine Learning Repository, "Condition monitoring of hydraulic systems" by N. Helwig, E. Pignanelli, and A. Schütze, DOI 10.24432/C5CW21, under the CC BY 4.0 license. The rig repeats a constant load cycle of 60 s and records 17 sensor channels and one row of condition labels per cycle. The rig and its faults are described in N. Helwig, E. Pignanelli, and A. Schütze, "Condition Monitoring of a Complex Hydraulic System using Multivariate Statistics," Proc. IEEE I2MTC, 2015, pp. 210-215, DOI 10.1109/I2MTC.2015.7151267.

| Component | Label | Levels | Healthy level |
| --- | --- | --- | --- |
| cooler | `cooler_condition` | 3, 20, 100 (%) | 100 |
| valve | `valve_condition` | 73, 80, 90, 100 (%) | 100 |
| pump | `pump_leakage` | 0, 1, 2 | 0 |
| accumulator | `accumulator_pressure` | 90, 100, 115, 130 (bar) | 130 |

The label row also holds `stable_flag`, which is 0 when the conditions were stable and 1 when static conditions might not have been reached yet. The four level sets give 144 combinations, and `HEALTHY_COMBO` in `src/envs/hydraulic_rig/env.py`, the combination with every component healthy, is `(100, 100, 0, 130)`. `COMPONENTS`, `LEVELS`, `HEALTHY`, `SENSORS` and `TIER_SENSORS` in `src/envs/hydraulic_rig/data.py` hold these facts.

### Local files

| File | Origin | Contents |
| --- | --- | --- |
| `hyd_facts.jsonl` | the IndustryAssetEQA facts file of the public EmbodiedQA repository, at commit `d7f42a703751cba077d99bae34e8c5fa4bcfa9fe` | one record per cycle with `cycle_index`, `labels`, and `features`: nine statistics per channel and three features that combine channels |
| `src/couchdb/scenarios_data/shared/iot/hydraulic_pump_1.json` | the AssetOpsBench asset `hyd_1` | the 17 channel means and a timestamp per cycle, without labels |

The environment reads the facts file only. A reading returns the mean and the standard deviation of every channel of the sensor group (`SERVED_STATISTICS` in `src/envs/hydraulic_rig/env.py`) under the field names of the `hyd_1` asset with the statistic appended, such as `PS1_Pressure_bar_100Hz_mean`, where the motor power channel EPS1 reads as `P1_MotorPower_W_100Hz` (`HYD1_FIELDS` in `src/servers/hydraulic_rig/names.py`). The `hyd_1` file serves as a cross check of the cycles and as the source of the field names.

| Variable | Default | Read by |
| --- | --- | --- |
| `HYD_FACTS_PATH` | `src/envs/hydraulic_rig/data/hyd_cycles.jsonl`, a generated file | `default_facts_path`, which the tests, the drivers, the wrappers and the model fit of the diagnostics server call, and `verify_data_facts.py`, which reads the variable itself |

The fork ships neither the facts file nor the generated release file. `DATA_DIR` in `src/envs/hydraulic_rig/data.py` names the folder `src/envs/hydraulic_rig/data/`, which git ignores except `src/envs/hydraulic_rig/data/CHECKSUMS.txt`, written by the first download.

```bash
# 1. fetch the facts file at a fixed commit
mkdir -p ~/rig-data
curl -L -o ~/rig-data/hyd_facts.jsonl https://raw.githubusercontent.com/ChathurangiShyalika/EmbodiedQA/d7f42a703751cba077d99bae34e8c5fa4bcfa9fe/data/outputs/hydraulic/hyd_facts.jsonl

# 2. point the environment at it
export HYD_FACTS_PATH=~/rig-data/hyd_facts.jsonl

# 3. check both data files
PATH=$PWD/.venv/bin:$PATH python benchmarks/hydraulic_rig/verify_data_facts.py
```

The facts file has the SHA-256 `7912b1b641254d54daffa40ea2fe84c64dedf50edb2d996951ad1f3d9a433e47` and the `hyd_1` file the SHA-256 `622bcd1a668a939a2590b1aaf560ecce641731b04d802271b39ef454835918b5`, both held by `verify_data_facts.py`. The EmbodiedQA repository has no license file, so the facts file serves only as a development input and a cross check, and the fork redistributes neither the file nor anything derived from it.

### Release path

The release path rebuilds the labels and the served statistics of every cycle from the CC BY 4.0 archive alone. The three scripts put `src/` on the import path themselves and use the standard library only, so they also run under a plain `python3 -I`.

```bash
# 1. download the archive into an empty folder and record its SHA-256
PATH=$PWD/.venv/bin:$PATH python benchmarks/hydraulic_rig/fetch_uci.py --download

# 2. compare the recomputed labels and statistics with the facts file
PATH=$PWD/.venv/bin:$PATH python benchmarks/hydraulic_rig/build_blocks.py <raw folder printed by step 1> --check

# 3. write the records and the blocks with the convention that step 2 reported
PATH=$PWD/.venv/bin:$PATH python benchmarks/hydraulic_rig/build_blocks.py <raw folder printed by step 1> --std-ddof <0 or 1>
```

`fetch_uci.py` downloads only with `--download` and only into an empty folder, by default `src/envs/hydraulic_rig/data/uci_hydraulic/`. It records the archive's SHA-256 in `src/envs/hydraulic_rig/data/CHECKSUMS.txt` and refuses an archive whose hash differs from the one recorded under the same name, or from `EXPECTED_SHA256` when that constant is set. `build_blocks.py --check` writes nothing and reports whether the facts file computes the standard deviation with ddof 0 or 1. A run with `--std-ddof` writes `src/envs/hydraulic_rig/data/hyd_cycles.jsonl`, the default of `default_facts_path`, with the mean, std, min and max of every channel in the layout of the facts file, and `src/envs/hydraulic_rig/data/blocks.json`, the stable block of every combination, unless `--out-dir` names another folder.

### Data checks

`verify_data_facts.py` reads the two files and writes nothing. It does not import the environment, so the tests can compare its blocks with those of `build_blocks`. Apart from the two hashes, each check is a relation that holds whatever the counts:

- Both files hold the same number of cycles, numbered without gaps from 0 in the facts file and from 1 in `hyd_1`.
- Every facts record has a cycle of 60 s, the 17 sensors, one label row, the feature names of every other record, and provenance to the UCI files.
- `hyd_1` cycle c matches facts record c - 1 on all 17 channel means within the tolerance of the script, and a pairing shifted by one cycle agrees on no PS1 mean.
- The labels take exactly the levels above, the facts file marks exactly the healthy levels as `cooler_ok`, `valve_ok`, `pump_ok`, and `accumulator_ok`, and the valve is at 100 in every cycle flagged 1.
- All 144 combinations occur, each with stable cycles that form one contiguous block.

`load_cycles` in `src/envs/hydraulic_rig/data.py` reads one `Cycle` per record, and `build_blocks` groups the stable cycles of each combination into one block and refuses a block that is not contiguous. Cycles flagged as not stable are never served.

---

## Configuration

`EpisodeConfig` in `src/envs/hydraulic_rig/config.py` holds every parameter an episode depends on besides its seed. Its defaults are the placeholders of the design, stated as design parameters until the session with the subject matter experts, and every run records the whole configuration in `run.json`.

| Field | Default | Meaning |
| --- | --- | --- |
| `asset_id` | `hyd_1` | the asset whose records the cycles are served as |
| `components` | cooler, valve, pump, accumulator | from `data.py` |
| `levels` | the level sets of the table above | per component |
| `healthy` | cooler 100, valve 100, pump 0, accumulator 130 | the level a repair restores |
| `sensor_groups` | `tier1`: TS1 to TS4, VS1, CE. `tier2`: tier1 plus FS1, FS2, CP. `tier3`: all 17 channels | `TIER_SENSORS` of `data.py`, served under the public field names |
| `prices` | `read`: tier1 1, tier2 3, tier3 10 per cycle. `parts`: cooler 20, valve 20, pump 30, accumulator 20. `repair_charge`: 20 | a read price must be positive, since a free reading would give the whole recording away |
| `durations` | `read_per_cycle` 1, `advance_per_cycle` 1, `lead_time` 3 for every component, `repair_downtime` cooler 1, valve 1, pump 2, accumulator 1 | in cycles |
| `downtime_rate` | cooler 100: 0, 20: 1, 3: 2. valve 100: 0, 90: 1, 80: 2, 73: 3. pump 0: 0, 1: 1, 2: 2. accumulator 130: 0, 115: 1, 100: 2, 90: 3. `stopped`: 5 | per cycle and level, the level keys spelled as JSON strings, and the stopped rate while a repair stops the rig |
| `budget` | 300 | prices, downtime and penalties together |
| `horizon` | 60 | cycles |
| `max_steps` | 60 | tool calls, refused ones included |
| `fault_rule` | `min_faulty` 1, `max_faulty` 2 | the public rule on the number of components off their healthy level at the start |
| `allowed` | null | an optional list of start combinations within the rule |
| `start` | null | `{"combination", "block_offset"}` for an anchored start, which the pilot uses |
| `schedule` | `mode` none, `components` 1, `first_drop` [5, 40], `interval` 10 | the degradation schedule, drawn in mode `seeded` |
| `external_events` | `mode` none, `probability` 0.3, `window` [5, 50] | the external replacement, drawn in mode `seeded` |
| `failure` | `patience` 10, `strict` false, `penalty` 50 | the failure rule |
| `clock_mode` | `declared` | `declared`, `generation_time` or `instant` |
| `seconds_per_cycle` | 60 | the conversion of thinking time in the `generation_time` mode |
| `guidance_level` | 1 | 1, 2 or 3, the level of the briefing |
| `question` | null | a question that replaces the question sentence of the briefing |
| `anchors` | null | `{"date0": <ISO date>}`, one recorded cycle per day, which puts a `day` into every answer |
| `verifier_scope` | `writes` | `writes` or `all`, recorded for the verifier and ignored by the environment |
| `diagnostics_server` | true | recorded for the drivers and the briefing. The profile lists the diagnostics tools only when it is true |

`validate` runs in the constructor and raises one `ValueError` that lists every problem at once: a free read price, a priced group without sensors, a group without a price, a missing part price, a non finite or negative number, a bool where an integer is expected, a reversed fault rule, a start outside the rule, an anchor that is not an ISO date, and so on. `to_json` turns every tuple into a list, and `from_json` refuses unknown keys and gives an equal configuration back. The helpers `n_faulty`, `within_fault_rule`, `is_combination`, `read_price`, `order_price`, `level_rate` and `date_of` are what the environment and the policies read. The presets of the drivers under `benchmarks/hydraulic_rig/configs/` lay their changes over these defaults, and the price tables under `benchmarks/hydraulic_rig/configs/prices/` replace the prices, the durations and the downtime rates together.

---

## Time

`src/envs/hydraulic_rig/clock.py` holds the clock and the hidden processes that run on it. Time counts cycles from 0 and advances only inside `step`.

| Clock mode | What a step advances |
| --- | --- |
| `declared` | the declared duration of the action: `cycles` times `read_per_cycle` for a read, `cycles` times `advance_per_cycle` for an advance, nothing for every other action |
| `generation_time` | the declared duration, plus the agent's thinking time before the action in cycles, which the action server measures as the wall clock since its last call divided by `seconds_per_cycle` and rounded up |
| `instant` | one cycle per step whatever the tool, refused steps included, the rule of agent environments in which a step is a unit of time |

The clock never passes the horizon. An advance is cut at the horizon and the episode ends with the reason `horizon`. In the `generation_time` mode a thinking time that alone reaches the horizon ends the episode before the action runs, and the action is refused with that reason.

### One cycle of the clock

`RigClock.advance_cycle` runs the processes of cycle t in this order, then moves the clock to t + 1:

1. The scheduled drops at t set the component to its target level and switch the served window to the new combination.
2. The external replacement at t restores its component, opens and closes the work order `wo-ext-1` with the source `external`, and switches the window.
3. Work orders. An order whose lead time has passed goes into repair, which stops the rig (the system event `work_order_executed`). An order whose repair downtime has passed closes. An order with a repair downtime of zero closes as the technician arrives.
4. The recording of the cycle: the next cycle of the window of the current combination, or none while the rig is stopped for a repair.
5. Downtime accrual: the stopped rate while the rig is stopped, else the sum over components of `downtime_rate[component][level]`.
6. The failure check: a component at its worst level for `patience` consecutive cycles without an open work order fails once. The system event `failure` adds `penalty` to the cost, and under `strict` it ends the episode.
7. The clock moves to t + 1, and every order in repair whose downtime ends there closes: the component becomes healthy and the window switches to the repaired combination.

`degradation_order` orders the levels of a component by their distance from the healthy level, which gives cooler 100, 20, 3, valve 100, 90, 80, 73, pump 0, 1, 2, and accumulator 130, 115, 100, 90. The worst level is the last one.

### Hidden processes

`reset` draws, from the generator seeded by the seed and in this order, the hidden combination (unless `start` names it), the served window, the schedule and the external event.

- **Schedule.** In mode `seeded`, `draw_schedule` picks `components` distinct components, the ones healthy at the start first, so that a scheduled episode changes, and among the faulty ones only when more components are asked for than are healthy. Each picked component gets a first drop cycle uniform in `first_drop` and then a drop every `interval` cycles down to its worst level. The drops are stored with their absolute target levels, so a repair between two drops does not change the later drops. In mode `none` the schedule is empty.
- **External event.** In mode `seeded`, `draw_external_event` decides with probability `probability` that a technician replaces one component faulty at the start, drawn among them, at a cycle uniform in `window`. The event closes a work order the agent never opened, `wo-ext-1`, which the change summary lists as closed with the source `external`. In mode `none` there is no event.
- **Failure rule.** The failure check above, with `patience`, `strict` and `penalty` from the configuration.

### Work orders

`WorkOrder` holds the id, the component, the creation cycle, the execution cycle (creation plus `lead_time`), the closing cycle (execution plus `repair_downtime`), the source (`agent` or `external`), the prediction identifier and the evidence identifiers the agent cited, the hidden flag `unnecessary`, the status (`open`, `in_repair`, `closed`) and the closing cycle once closed. Agent orders are numbered `wo-1`, `wo-2` and so on. An order is charged `parts[component]` plus `repair_charge` at creation, and the rig is stopped at the stopped downtime rate during the repair downtime. An order on a healthy component is executed, charged and marked unnecessary in the hidden state. The public view of an order, `WorkOrder.public`, holds `id`, `component`, `created_t`, `expected_close_t`, `status`, `source` and `closed_t`, never the flag.

### Windows

When an episode enters a combination, the environment serves the stable block of that combination through a window of `block_cap` consecutive cycles, 10 by default (`BLOCK_CAP` in `env.py`), entered at a seeded offset, so that the length of a block reveals nothing about the combination. Every change of the combination opens a new window. A read serves, for every advanced cycle in which the rig ran, the recording the rig played during that cycle, so a drop or a closure in the middle of a read shows in the later observations. Once the reads have gone through the whole window, the next cycle repeats one and the observation carries `repeated` true. An anchored start (`start.block_offset` k) places the window so that the k-th stable cycle of the block is the first one served, and k past the block end is clamped.

A block shorter than `block_cap` would be served through a shorter window and wrap earlier. The environment caps the window at the block length and does not check whether any block is that short, which is a fact of the data file.

---

## Episode interface

`HydraulicRigEnv(cycles, blocks, config=None, block_cap=BLOCK_CAP)` in `src/envs/hydraulic_rig/env.py` takes the output of `load_cycles` and `build_blocks` and an `EpisodeConfig`, the default when none is given. The start candidates are the combinations within the fault rule, restricted to `allowed` when it is set. The constructor raises `ValueError` when no candidate remains and computes the public healthy bands from the data.

`reset(seed)` seeds the episode's own `random.Random`, draws the hidden state as described above and returns the first envelope, whose `result` is the asset profile. Every later draw comes from the same generator, so a seed, a configuration and an action sequence reproduce the log byte for byte, which the action server relies on when it replays an episode.

`step(action, think_cycles=0)` takes a dict with a `type` key and returns `(envelope, cost, done, info)`. `cost` is everything the step charged: the price of the action, the downtime the step let accrue and any penalty. `info` holds `refused` after a refusal and, on the last step, `log` with the full log and its hidden state and `reason`, which no tool server may forward. A step after the end raises `RuntimeError`, an action that is not a dict raises `TypeError`, and so does a `think_cycles` that is not a non negative integer. `think_cycles` is charged before the action in the `generation_time` mode and ignored otherwise.

| Attribute or method | Content |
| --- | --- |
| `log` | the full log, hidden fields included, for the verifier only |
| `time`, `cost`, `steps`, `done`, `diagnosed`, `declared` | the clock, the cost so far, the steps taken, and the state of the episode |
| `initial`, `hidden`, `candidates`, `clock` | the start combination, the current combination, the start candidates and the `RigClock`, all hidden from agents |
| `window_start`, `entry_offset`, `combination_at(t)`, `cycle_at(t)` | the window of the current combination, the combination current at cycle t and the recording index played during cycle t, hidden |
| `healthy_bands` | the public band of every channel mean |
| `oracle()` | the oracle of the episode, see below |
| `public_profile()`, `public_event(entry)`, `public_events()`, `public_work_orders(closed_since=None)`, `envelope(result=None, refused=None)` | the agent facing views |

Every answer is an envelope:

| Field | Content |
| --- | --- |
| `time` | the current cycle |
| `day` | the ISO date of the current cycle, present only when `anchors` is set |
| `budget_left`, `steps_left` | the budget and the step cap left |
| `open_work_orders` | the public view of every open work order |
| `episode_over` | whether the episode has ended |
| `result` | the answer of an accepted action, absent on a refusal |
| `refused` | the reason of a refusal, absent on an accepted action |

The asset profile, the `result` of `get_asset_profile` and of the first envelope, holds `asset_id`, `asset` (type, description and site from the sample asset profiles of the repository), `components`, `levels`, `healthy`, `sensor_groups` with the public field names, `statistics`, `prices`, `durations`, `downtime_rate`, `budget`, `horizon`, `max_steps`, `fault_rule`, `clock_mode`, `seconds_per_cycle`, `tools` (the table of the tool contract for the action server and, when `diagnostics_server` is true, for the diagnostics server) and `healthy_bands`.

| End reason | When |
| --- | --- |
| `finish` | the agent finishes, at any time |
| `budget` | an action would take the cost above `budget`, so it is refused and the episode ends |
| `horizon` | the clock reaches `horizon` |
| `max_steps` | the step count reaches `max_steps` on a step that does not end the episode otherwise |
| `failure` | a failure event under the strict rule |

Recovery does not end the episode, since an end on recovery would tell the agent the hidden state. The agent finishes.

---

## Actions

Every action is a dict with a `type` and the arguments below. The class, the duration and the price of every action come from the tool contract in `src/servers/tool_contract_v2.json`, which [action-server.md](action-server.md) describes.

| `type` | Arguments | Class | Duration | Price | `result` |
| --- | --- | --- | --- | --- | --- |
| `get_asset_profile` | none | read | 0 | 0 | the asset profile |
| `read_sensors` | `sensor_group`, `cycles` (a positive integer) | read | `cycles` times `read_per_cycle` | `cycles` times `prices.read[sensor_group]` | `sensor_group`, `served` (one entry per cycle the rig ran: `n` the observation number, `t` the cycle, `repeated`, `values`) and `stopped`, the read cycles during which the rig was stopped for a repair and produced no observation |
| `advance_time` | `cycles` (a positive integer) | read | `cycles` times `advance_per_cycle` | 0, downtime accrues meanwhile | `time` |
| `summarize_changes_since` | `cycle` (an integer from 0 to the current time) | read | 0 | 0 | `since`, `time`, `cycles_elapsed`, `alarms` (one entry per channel mean that left its public band in any cycle the rig ran since `cycle`, with `channel` and `first_t`) and `work_orders_closed` (the public orders closed at or after `cycle`) |
| `get_failure_modes` | `component`, or an empty string for all | read | 0 | 0 | `component` (null for all), `association_source` and `failure_modes`, one entry per fault code of the knowledge base with `component`, `level`, `code`, `severity`, `action`, `channels` (the public fields of the channels that indicate the component) and `evidence_id` |
| `submit_diagnosis` | `diagnosis` (one integer level per component), `evidence_ids` (a list of strings, possibly empty) | write | 0 | 0 | the stored `diagnosis` and `evidence_ids` |
| `declare_insufficient_evidence` | `separating_test` (`sensor_group` and a positive `cycles`), `evidence_ids` | write | 0 | 0 | the stored `separating_test` and `evidence_ids` |
| `create_work_order` | `component`, `prediction_id` (a string, empty without a prediction), `evidence_ids` | write | 0 at creation, the order closes at the creation cycle plus `lead_time` plus `repair_downtime` | `parts[component]` plus `repair_charge`, charged at creation | `work_order`, the public view of the new order |
| `finish` | `summary` (a string) | write | 0 | 0 | `summary` |

A diagnosis or a declaration is accepted once per episode and locks the other. Reads and work orders stay allowed after a declaration, a diagnosis does not. A work order needs a diagnosis or a declaration first and one open order per component at most. `finish` is accepted at any time, and open work orders stay open and count as not repaired.

Any other malformed or out of order action that is a dict is refused: it is logged with its reason, costs nothing, changes no state and counts as a step, so a language model agent that emits a bad action loses a step and not the episode. In the `instant` mode a refused step advances the clock by one cycle like every other step. Only a refusal for lack of budget ends the episode.

| Reason | Cause |
| --- | --- |
| `unknown action type` | a `type` that is not one of the nine |
| `unknown sensor group` | a `sensor_group` outside `sensor_groups`, a missing one included |
| `cycles must be a positive integer` | a `cycles` of `read_sensors` or `advance_time` that is not an int at or above 1, a bool, a float or a string included |
| `cycle must be an integer between 0 and the current time` | a `cycle` of `summarize_changes_since` outside that range |
| `unknown component` | a `component` outside the four, in `get_failure_modes` or `create_work_order` |
| `a diagnosis or a declaration was already submitted` | a second diagnosis or declaration of either kind |
| `diagnosis must give one level per component` | a `diagnosis` that is not a dict with exactly the four components |
| `diagnosis level not in the component's levels` | a level that is not an integer among the levels of its component |
| `separating_test must name a sensor group and a positive number of cycles` | a declaration whose test is malformed |
| `evidence_ids must be a list of strings` | an `evidence_ids` that is not a list of strings |
| `a diagnosis or a declaration must precede a work order` | `create_work_order` before the commitment |
| `prediction_id must be a string, empty when there is no prediction` | a `prediction_id` that is not a string |
| `a work order on the <component> is open` | a second order on a component whose order has not closed |
| `summary must be a string` | a `finish` without a string summary |
| `the episode ended before the action ran` | the thinking time of the `generation_time` mode alone reached the horizon or a strict failure |
| `budget` | a read or a work order that costs more than the budget left, which also ends the episode |

### Served values and bands

A reading returns the mean and the standard deviation of every channel of the group under the public field names at six significant digits (`served_values` in `env.py`, through `public_result` of `src/servers/hydraulic_rig/names.py`). The rounding happens inside the environment, so the log, the envelope and the released evidence carry the same numbers.

The public healthy band of a channel mean is the interval between its 1st and 99th percentiles (`BAND_QUANTILES`) over the stable cycles of the healthy combination, computed on the served values when the environment is built and served in the profile under `healthy_bands`, keyed by the field of the channel mean. The bands are served values, not reported results. An alarm of `summarize_changes_since` is a channel mean outside its band in a cycle the rig ran since the given cycle, with the first such cycle. Alarms are computed from the recording the rig played and never from a label, and a cycle during which the rig was stopped produces none.

---

## Log and information barrier

Every log entry carries `event`, `step` (the step number of an agent action, null for a system event), `t_start`, `t_end`, `class` (`read`, `write` or `system`, from the tool contract) and `cost`, the entry's own charge. Agent entries also carry `think_cycles`.

| `event` | Other fields, hidden ones marked |
| --- | --- |
| `reset` | `seed` (hidden), `hidden` (the start combination, hidden), `schedule` (hidden), `external_events` (hidden), `config` (hidden, since it may hold `start` and `allowed`) |
| `profile` | none |
| `read` | `sensor_group`, `cycles`, `stopped`, `served`: per observation `n`, `t`, `cycle` (the recording index, hidden), `combination` (hidden), `repeated`, `values` |
| `advance` | `cycles` |
| `summary` | `since`, `alarms`, `work_orders_closed` |
| `failure_modes` | `component` |
| `submit_diagnosis` | `diagnosis`, `evidence_ids` |
| `declare_insufficient_evidence` | `separating_test`, `evidence_ids` |
| `work_order_created` | `id`, `component`, `prediction_id`, `evidence_ids`, `expected_close_t`, `unnecessary` (hidden) |
| `finish` | `summary` |
| `refused` | `action` as the environment received it, `reason` |
| `scheduled_drop` | `component`, `from_level`, `to_level`, the whole entry hidden |
| `external_replacement` | `component`, `work_order_id` |
| `work_order_executed` | `id` |
| `work_order_closed` | `id`, `component`, `source`, `combination_after` (hidden) |
| `failure` | `component` (hidden), with the penalty as `cost` |
| `end` | `reason`, `time`, `final` (hidden), `cost` (the episode total), `downtime_total`, `downtime_by_cycle` (`t` and `charge` per cycle with a charge), `played` (hidden, the recording index and the combination of every cycle the rig ran) |

`public_event` gives the agent facing view of an entry: it strips `HIDDEN_FIELDS` (`cycle`, `combination`, `combination_after`, `unnecessary`, `hidden`, `schedule`, `external_events`, `final`, `seed`, `config` and `played`) at every depth, returns None for a scheduled drop, which the agent has to notice through readings and alarms, and strips the component from a failure, so the agent learns that a failure happened and when but not where. `public_events` lists the public views of every entry after the reset. The envelopes, the asset profile, the public events and the briefing of the action server are the only agent facing texts. Everything else stays in `env.log` and, in a run folder, under `hidden/`, which no agent facing tool reads. An agent with file or shell tools has to run with the episode and run folders out of its reach, since the state file of the action server holds the seed, from which a replay recovers the hidden state.

---

## Oracle

`env.oracle()` in `src/envs/hydraulic_rig/oracle.py` writes the reference the verifier compares the agent's writes with, in the form of the Agents Research Environments verifier: `{"diagnosis": {component: level at the start}, "work_orders": [{"component", "window_end"}], "finish": true}`.

The work order set holds every component that is faulty at the start or dropped by the schedule before the episode ends (before the horizon while the episode runs), except a component the external event restores before any agent order on it. A drop at cycle s happens when the clock passes s, so an episode that ends at time s never saw it. `window_end` is the cycle at which the failure event would fire for the component, that is the cycle at which it reaches `patience` cycles at its worst level, or the horizon when it never would. An order closed at or before `window_end` averts the failure, since the closure is processed before the failure check of that cycle. Because the set depends on the agent's orders through the external event rule, the oracle is evaluated against the log when the episode ends.

---

## Reference policies

A policy in `src/envs/hydraulic_rig/policies.py` has `start(first_envelope)`, called with the envelope that `reset` returns, and `act(envelope)`, called with the answer to its previous action. It sees nothing else, which is what a language model agent gets through the tools. A harness that skips `start` may hand the reset envelope to `act`, which starts the policy itself. `run_episode(env, policy, seed)` plays one seeded episode to its end and returns the log.

- **Oracle.** `OraclePolicy(env, inseparable=False)` reads the hidden state through the environment: the start combination, the current levels, the schedule and the external event. It submits the true diagnosis, or, when `inseparable` says that the diagnosability grade of the start marks it inseparable, declares insufficient evidence with the cheapest sensor group as the separating test, since the verifier counts a commitment on such a start as no success. It orders every faulty component once it will not drop further, leaves a component to the technician when the technician comes before the component would fail and before the horizon, advances time until the orders close and finishes. It does not wait for a drop whose repair could not close before the horizon, and the oracle set of such an episode never holds that component. The oracle is the reference for the cost of an episode and the upper bound of the verifier under its success definition, and only the tests, the drivers and the gym wrappers construct it. The drivers look the grade up before the oracle plays, and the gym wrappers construct it without a grade.
- **Do nothing.** `DoNothingPolicy()` advances time to the horizon in one call and finishes if the episode is still open. It is the lower reference of the verifier.
- **Random.** `RandomPolicy(seed)` draws from the stream `random-policy:<seed>`, kept apart from the environment's generator so that its first decision is not tied to the hidden combination. Before the diagnosis it submits a random diagnosis with probability 0.15, or whenever no sensor group is affordable, and otherwise reads one cycle of a random affordable group. After the diagnosis it orders a random component without an open order with probability 0.4 when affordable, advances one to five cycles with probability 0.3, and finishes otherwise.
- **Greedy.** `GreedyPolicy(classifiers, use_max_faulty=False, confidence=0.95, min_gain_per_price=0.05)` in `src/envs/hydraulic_rig/greedy.py` keeps a belief over the 144 combinations, reads every served observation through the cross fitted classifiers, buys one cycle of the affordable sensor group with the largest expected information gain per unit price while keeping budget for its work orders, submits the most probable combination at the confidence it is constructed with, with the `obs/` identifiers of what it read as evidence, creates one work order per diagnosed fault, advances time until the orders close and finishes. The reference row sets `use_max_faulty=True` and starts from a uniform prior over the combinations within the fault rule of the asset profile. The ablation without the rule starts from a uniform prior over the 143 combinations other than the healthy one.

**Classifiers and split.** `TierClassifiers` in `src/envs/hydraulic_rig/classifiers.py` fits one linear discriminant analysis (LDA) per measurement tier and component on the served summaries, with standardized features, a pooled within class covariance shrunk by `SHRINKAGE` toward a scaled identity, and equal class priors. `stratified_folds` deals the 144 combinations into `N_FOLDS` folds stratified by level, and `cross_fitted_predictions` predicts the stable cycles of each fold with classifiers fitted on the other folds, refusing a split whose training set lacks a level. The split is by combination and not by cycle because a window can start anywhere in a block, so a split inside a block would put served cycles into the training set. `ServedClassifiers` keys the held out predictions by the values a reading serves (public names, mean and std at six significant digits) and refuses a table in which two cycles of a tier serve the same values, so the policy decides on tool answers exactly as on the recordings. A sensor group is read through the tier that serves the same channels.

**Observation model.** `observation_model` gives, for hypothesis h, tier t and component c, the held out confusion row of the level of c in h, counted over every block except that of h, with one added to each count. The policy assumes:

- A1: given the combination, the predicted levels of the four components are independent.
- A2: given the combination, readings at different tiers are independent.
- A3: within one tier, predictions on cycles of the same block are exchangeable draws from a Dirichlet multinomial centered on the row, so the next prediction follows an urn around the row. The concentration is fitted per tier and component by maximum likelihood over `ALPHA_GRID`.
- A4: a reading equal to an earlier one at the same tier adds no information.

**Decision rule.** The expected information gain of a tier is the mutual information in bits between the hypothesis and the predicted levels of the next reading at that tier. The policy keeps a reserve for its orders, the sum of the most expensive order prices the fault rule may need. It submits the most probable combination when the largest belief reaches `confidence`, when the window starts repeating, when no group is affordable within the reserve, or when the best ratio of gain to price falls below `min_gain_per_price`. Otherwise it reads the group with the best ratio.

**Known limits.** A2 ignores the correlation between tiers, whose channels overlap, so the belief at submission can be overconfident, and the same cycle read at two tiers counts twice. The rule looks one step ahead and prices information in bits, not in expected repair cost. The observation rows leave out only the hypothesis's own block: in an episode on combination g, the rows of other hypotheses that share a level with g still count the held out predictions on g, and the concentrations are fitted on all blocks. The diagnostics server of [causal-diagnosis-server.md](causal-diagnosis-server.md) uses a different observation model, the interventional effect model, and keeps a refit of these classifiers only as an ablation arm.

**Start states.** A start state is a combination, a window start and an entry offset, and a block of n cycles gives n minus `block_cap` plus 1 windows with `block_cap` offsets each. `start_states` in `src/envs/hydraulic_rig/start_states.py` lists every start state that `reset` can draw, and `seed_table` gives each one the smallest seed that reaches it, so an episode from any start state replays through the public `reset(seed)`. An anchored start fixes the state through `config.start` and needs no table.

---

## Verifier

`verify(log, released, oracle, cost_oracle, grade=None, scope="writes", predictions=None)` in `src/envs/hydraulic_rig/verifier.py` returns the verdict of one episode as a plain dict that `json.dumps` accepts. It reads no answer text for a score.

| Input | Content |
| --- | --- |
| `log` | `env.log`, with its hidden fields |
| `released` | the released evidence of the episode in format 2, written by the action server as `released.json` or composed in process by `released_from_env` |
| `oracle` | `env.oracle()`, with `reads` added for the scope `all` |
| `cost_oracle` | the cost of the oracle policy's episode on the same seed and configuration, which normalizes the cost |
| `grade` | the diagnosability grade of the start from the grade table of the diagnostics server, `{"separable": true, "cost": c}`, `{"separable": false, "cost": null}`, or None when unknown. An unknown grade counts as separable, since it is no evidence that a declaration was right |
| `scope` | `writes`, the default, or `all` |
| `predictions` | the records of `predictions.json` when the diagnostics server wrote one, for the `pred/` identifiers |

### Checks

The checks run in the order of the Agents Research Environments verifier, each a dict with `ok` and its evidence under `checks`:

| Check | Rule |
| --- | --- |
| `count_gate` | exactly one of `submit_diagnosis` and `declare_insufficient_evidence`, exactly one `finish`, and no write event outside the four write events, a refused action being no write. Under the scope `all` the counts of the read events have to equal the oracle policy's, which `read_counts` computes from the oracle policy's own log |
| `hard` | the submitted levels against the oracle's diagnosis, component by component, and the components of the agent's work orders against the oracle's set as unordered lists, since the recorded consequence of two repairs does not depend on their order |
| `timing` | every agent order on a component that has a scheduled drop in the log has a closure at or before the oracle's `window_end` for that component. Orders on unscheduled components are not timed |
| `causality` | the first diagnosis or declaration precedes every work order in the log |
| `outcome` | `recovered` is the final combination equal to the healthy one, read from the end event or followed through the system events when the log has no end yet. `unnecessary` counts the orders whose hidden flag is set |
| `provenance` | every evidence identifier cited in the diagnosis, the declaration and the work orders, and every non empty `prediction_id`, exists among the identifiers the episode issued: `obs/{n}/{field}` for every released value, `wo/{id}` for every work order, `alarm/{t}/{channel}` for every alarm of a change summary, `fmea/{source}/{code}` for the codes of every component whose failure modes were served, and `pred/{n}` for every stored prediction |

### Failure classes

For an episode that did not succeed, or whose provenance check failed, the verdict names one of the eight classes of the paper's failure taxonomy, decided in this order, and `none` otherwise. A failure outside the taxonomy, such as a commitment on an inseparable start, is `none` too.

| Class | Rule |
| --- | --- |
| `premature_diagnosis` | a diagnosis with no `read` event before it |
| `unmatched_evidence` | the provenance check failed |
| `unnecessary_replacement` | at least one order on a component that was healthy when the order was created |
| `abstention_on_separable` | a declaration while the start is separable or the grade unknown |
| `summary_contradicts_log` | the closing summary claims a recovery the final combination does not show, or the replacement of a component no agent order closed on |
| `repeated_malformed_call` | `REPEATED_REFUSALS` or more refused actions in the episode |
| `anchoring_on_first_anomaly` | the diagnosed faulty set equals the components that the first channel the agent saw outside its public band indicates, while the true faulty set differs |
| `recovery_without_diagnosis` | the rig recovered while the diagnosis is not all correct |

The contradiction heuristic, reported under `summary_check` and marked as a heuristic, splits the closing summary into sentences, skips sentences with a negation, reads a recovery claim when a sentence names the whole asset with a recovery word, and reads a replacement claim when a sentence names a component with a repair word. It is conservative on purpose, and the judge agreement study of [benchmarks.md](benchmarks.md) measures what it misses.

### Scores

| Field | Content |
| --- | --- |
| `diagnosis_per_component`, `diagnosis_all_correct` | the diagnosis against the start combination, per component and in all, None without a diagnosis |
| `recovered`, `unnecessary` | the outcome from the recordings |
| `pass` | correct diagnosis and recovered and no unnecessary order, Equation 2 of the paper, for the metric tables |
| `separable`, `abstained`, `abstention_correct`, `commitment_on_inseparable` | the grade's flag, whether the agent declared, whether it declared on an inseparable start, and whether it committed on one |
| `success` | `pass` on a separable or ungraded start, `abstention_correct` on an inseparable one |
| `cost`, `cost_prices`, `penalties`, `downtime_total`, `cost_oracle`, `cost_normalized` | the prices of the agent entries plus the penalties of the failure events plus the downtime of the end event, its parts, the oracle's cost and their ratio, None when the oracle's cost is unknown |
| `detection_delay` | per component, the cycles from its first scheduled drop to the agent's first order on it, None without both |
| `repair_delay` | per component, the cycles from the first released reading of a channel that indicates it outside its public band to the agent's closed order on it, None without both |
| `turns_to_correct_diagnosis` | the step of the diagnosis when it is all correct, else None |
| `prediction_verdict` | per order that names a prediction: whether the prediction exists, the recovery the diagnostics server predicted, its signed error against the recorded recovery, and the verification the server stored, if any |
| `end_reason`, `time_used`, `steps`, `refusals`, `failures` | the end event's reason and time, the agent steps, the refused actions and the components that failed |
| `first_anomaly`, `summary_check`, `hidden_start`, `final`, `grade`, `scope`, `all_checks_ok`, `verified` | the first channel seen outside its band with its source, the heuristic's reading, the hidden start and final combinations, the inputs and the overall flags |

`reward(verdict, horizon, w_c=0.2, w_t=0.1)` is the terminal reward of policy learning: one if `success` else zero, minus `w_c` times the normalized cost, minus `w_t` times the largest detection or repair delay over the horizon. It is 0 when the provenance check failed, whatever the outcome, and None when the verifier could not run or the cost of the oracle is unknown. Crediting `success` rather than `pass` gives a correct abstention the outcome term and denies it to a commitment on an inseparable start.

`pass_power_k(successes, n, k)` is the unbiased estimator of pass^k of tau bench: for a task solved c times in n trials, the number of ways to choose k successes among the c over the number of ways to choose k trials among the n, and the mean of that quantity over tasks when `successes` is a sequence.

`released_from_env(env)` composes the released evidence of an in process episode, the equivalent of the action server's `released.json`, from the public profile as the rules, the public events and the observations of the reads. `read_counts(log)` counts the read events by name.

The verifier refuses to run, with `verified` false and an `error`, on a log without a reset event, a log whose hidden fields were stripped, a work order without its hidden flag, an oracle without its diagnosis or its work orders, an oracle whose diagnosis differs from the log's hidden start, and released evidence that is not format 2 or lacks its events or its observations. A missing grade is not an error.

---

## Gym and OpenEnv

`HydraulicRigGym(config, facts_path, seed_offset=0)` in `src/envs/hydraulic_rig/gym_env.py` is the environment for policy learning with text observations. `gymnasium` is imported lazily: with it installed the class is a `gymnasium.Env` with `Text` observation and action spaces, without it a stand in base class with the same methods, so the fork adds no dependency.

- `reset(seed=None, options=None)` returns the briefing followed by the asset profile as JSON, and an info dict with the effective seed, which is the seed plus `seed_offset`. Without a seed the instance plays consecutive seeds from the offset, as a training loop that calls `reset()` expects. At every reset the oracle policy plays the same seed on a second environment instance, and the verifier's cost of that episode becomes the `cost_oracle` of the verdict.
- `step(action)` takes a dict or a JSON string of an action of this page and returns the answer envelope as JSON, the reward, the `terminated` and `truncated` flags and an info dict with the cost, the time and the step count. Anything else, unparsable text included, becomes an action of type `malformed`, which the environment refuses, logs and counts, so a policy that emits bad text loses a step and never raises into the trainer. The reward is 0.0 until the terminal step, then the verifier's reward with `grade` None and the configuration's `verifier_scope`, and the info dict then carries `end_reason`, `verdict`, `reward` and `cost_oracle`. `terminated` is true for an end by `finish`, `budget` or `failure`, `truncated` for an end by `horizon` or `max_steps`. A verifier that could not run gives a truncation with 0.0 in the tuple and None in the info dict, so a trainer can drop the episode instead of scoring it zero.
- The attribute `observation` keeps the last answer dict of the environment, so that an in process policy such as `OraclePolicy(gym.env)` or `DoNothingPolicy()` acts on the same episode as a text policy through `policy.act(gym.observation)`. The attribute `env` is the wrapped environment, whose log and oracle are hidden state for the verifier and never for the policy, as is `info["verdict"]`.
- The gym charges no thinking time: it calls `step` with `think_cycles` 0, so the `generation_time` mode behaves as `declared` under the gym. The cost of a step is reported in the info dict and never subtracted from the reward, so that the outcome term stays the signal.

`HydraulicRigOpenEnv(config=None, facts_path=None, seed_offset=0)` in `src/envs/hydraulic_rig/openenv_env.py` puts the gym behind the `reset`, `step` and `state` interface of the OpenEnv core, imported lazily from `openenv.core`, with pydantic stand ins of the same fields when the package is absent. `HydraulicRigAction(tool_name, arguments)` has the shape of the core's tool call action: the tool name is the action type and the arguments are the action's argument dict. `HydraulicRigObservation(text, done, reward, metadata)` carries the text of the gym, the reward (None at reset, 0.0 before the end, the verifier's reward at the end, None when the verifier could not run) and the cost and time in `metadata`. `HydraulicRigState(episode_id, step_count, seed, time, done, end_reason, verdict)` is the bookkeeping of the orchestrator, with the verdict and its hidden truth set at the end and never in an observation. The default episode id names the seed, so that a replay of a seed reports the same state. The arguments default to `EpisodeConfig()` and `default_facts_path()` because the OpenEnv server builds one instance per session without arguments, and `SUPPORTS_CONCURRENT_SESSIONS` is true. In production mode the same actions are the MCP tools of the action server, served without reward, step count or done flag, where the server advertises flat parameters for the Plan-Execute argument filler while the OpenEnv action carries the environment's own argument dict.

`examples/hydraulic_rig/rollout.py` plays the oracle policy and the do nothing policy from one seed through the gym and prints which policy earned the higher reward, the oracle and baseline check of OpenEnv's RFC 008, and no value:

```bash
export HYD_FACTS_PATH=~/rig-data/hyd_facts.jsonl
PATH=$PWD/.venv/bin:$PATH python examples/hydraulic_rig/rollout.py --seed 0
```

---

## Programmatic use

Run from the fork root with `HYD_FACTS_PATH` set and `src/` on the import path, as in `PATH=$PWD/.venv/bin:$PATH PYTHONPATH=src python`:

```python
from envs.hydraulic_rig.config import EpisodeConfig
from envs.hydraulic_rig.data import build_blocks, default_facts_path, load_cycles
from envs.hydraulic_rig.env import HydraulicRigEnv
from envs.hydraulic_rig.policies import OraclePolicy, run_episode
from envs.hydraulic_rig.verifier import released_from_env, reward, verify

cycles = load_cycles(default_facts_path())
blocks = build_blocks(cycles)
config = EpisodeConfig()  # the placeholder parameters, static episode
env = HydraulicRigEnv(cycles, blocks, config)

envelope = env.reset(0)                                   # the result is the asset profile
envelope, cost, done, info = env.step({"type": "read_sensors", "sensor_group": "tier1", "cycles": 2})
served = envelope["result"]["served"]                     # one entry per cycle the rig ran, with its values
guess = {"cooler": 100, "valve": 100, "pump": 1, "accumulator": 130}
envelope, cost, done, info = env.step({"type": "submit_diagnosis", "diagnosis": guess, "evidence_ids": [f"obs/{s['n']}/{f}" for s in served for f in s["values"]]})
envelope, cost, done, info = env.step({"type": "create_work_order", "component": "pump", "prediction_id": "", "evidence_ids": []})
envelope, cost, done, info = env.step({"type": "advance_time", "cycles": envelope["result"]["work_order"]["expected_close_t"] - env.time})
envelope, cost, done, info = env.step({"type": "finish", "summary": "Replaced the pump."})

reference = HydraulicRigEnv(cycles, blocks, config)       # the oracle policy on the same seed gives the cost reference
run_episode(reference, OraclePolicy(reference), 0)
verdict = verify(env.log, released_from_env(env), env.oracle(), reference.cost)
terminal_reward = reward(verdict, config.horizon)          # keep env.log, the oracle and the verdict away from agents
```

---

## Known limits

- The window of a combination is capped at its block length, so a block shorter than `block_cap` would wrap earlier than the others. Whether any block is that short is a fact of the data file that the environment does not check.
- The diagnosis is scored against the start combination. In a timed episode whose schedule drops a component after the start, the correct diagnosis names the start level, not the level at the time of the reading.
- An external replacement happens at its cycle even when the agent's own order already restored the component. The log then shows an external replacement of a healthy component without a change of the combination.
- A read that crosses the horizon serves only the cycles before it and is charged its declared price, since the clock never passes the horizon and the budget rule charges at the call.
- Alarms of the change summary are free and take no time, so after the first advanced cycle an agent sees every channel mean outside its band for nothing. The band is a property of the healthy recordings and was not tuned.
- The greedy reference policy counts the same cycle read at two tiers twice (assumptions A2 and A4), and its observation rows keep the leakage described above. The diagnostics server merges such readings into one observation.
- The gym charges no thinking time, and `info["verdict"]` holds the hidden truth, so a harness that shows the info dict to the policy would leak.

---

## Tests

```bash
export HYD_FACTS_PATH=~/rig-data/hyd_facts.jsonl
PATH=$PWD/.venv/bin:$PATH python -m pytest src/envs/hydraulic_rig -q
PATH=$PWD/.venv/bin:$PATH python -m pytest benchmarks/hydraulic_rig/tests/test_data_facts.py benchmarks/hydraulic_rig/tests/test_build_blocks.py benchmarks/hydraulic_rig/tests/test_fetch_uci.py -q
```

| Test file | Covers |
| --- | --- |
| `src/envs/hydraulic_rig/tests/test_config.py` | the defaults validate and round trip through JSON, unknown keys are refused, every problem is listed at once, the degradation order, the anchored dates |
| `src/envs/hydraulic_rig/tests/test_clock.py` | a scheduled drop switches the served block, an external replacement closes an order the agent never opened, a work order executes after the lead time, stops the rig and closes with the repaired block, reads during the repair downtime, downtime accrual, the failure rule, the oracle's set and window ends |
| `src/envs/hydraulic_rig/tests/test_env.py` | replay byte for byte, the hidden state from the seed only, the envelope and the profile, the information barrier, reads cycle by cycle, windows, healthy bands, every refusal, the write lock, the budget, the horizon, the step cap, finish, the end entry, the change summary, the failure modes, the three clock modes |
| `src/envs/hydraulic_rig/tests/test_policies.py` | the oracle recovers every candidate start, follows the schedule and the technician, finishes before a drop it could not repair, the do nothing policy, the random policy, the greedy policy |
| `src/envs/hydraulic_rig/tests/test_greedy.py` | the linear algebra of the classifiers on hand computed cases, the stratified folds, the cross fit, the observation model, the information gain, the served value lookup |
| `src/envs/hydraulic_rig/tests/test_start_states.py` | the seed table lists every start state once and every seed replays its start state |
| `src/envs/hydraulic_rig/tests/test_verifier.py` | every check, every failure class, the tri state grade, the delays, the cost, the reward and pass^k on synthetic logs, the scope `all`, the unverifiable inputs, and one integration test that plays the oracle policy and the do nothing policy on the environment |
| `src/envs/hydraulic_rig/tests/test_gym_env.py`, `src/envs/hydraulic_rig/tests/test_openenv_env.py` | determinism, the reward order, truncation at the horizon and at the step cap, malformed text, no hidden field in any observation text or in the released evidence, import and run without gymnasium and without openenv, the OpenEnv state contract |
| `benchmarks/hydraulic_rig/tests/test_data_facts.py`, `benchmarks/hydraulic_rig/tests/test_build_blocks.py`, `benchmarks/hydraulic_rig/tests/test_fetch_uci.py` | every check of `verify_data_facts.py` and the environment's blocks against its recomputation, `build_blocks.py` on synthetic raw files and on the facts file, `fetch_uci.py` without the network |

Without the data file, the tests that need it are skipped rather than failed, so a passing run without the file does not exercise the environment. The two tests that need gymnasium or openenv installed are skipped without the package. No test states a value computed from the rig data: the assertions are relations and equalities.
