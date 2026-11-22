# Hydraulic Rig Environment

`CausalDiagnosis/environment/Causal_Diagnosis_Environment/` turns the public UCI hydraulic rig recordings into episodes of finding the faults of an asset and restoring it on a clock. This page covers the data, the episode configuration, the six big layers with their small switches, the clock with its four modes, its hidden processes and its repair rules, the twelve actions with their refusals, the log and the information barrier, the oracle, the baseline policies, the verifier with its checks, failure classes, scores and reward, and the Gymnasium and OpenEnv wrappers. The engine, which is `env.py`, `episode_clock.py`, the layers and the verifier, uses no package outside the Python standard library and the repository, and the environment simulates no reading. The sampler gives the initial state and the rules of the layers give every later state, and every state is answered with recorded cycles.

Related docs: [README.md](README.md) for the index of these pages, [environment-actions-mcp.md](environment-actions-mcp.md) for the MCP tools that serve an episode, [causal-diagnosis-mcp.md](causal-diagnosis-mcp.md) for the tools of the CausalDiagnosis MCP, [experiments.md](experiments.md) for the drivers and the result tables, and [../mcp-servers.md](../../docs/mcp-servers.md) for every MCP server of the repository.

Names on this page follow the code and the texts that an agent reads, and the [glossary](README.md#glossary) of the docs README gives the term of the paper for each of them, such as sensor set for sensor group, abstention for declaration of insufficient evidence and configuration for preset and for row.

---

## Overview

Every episode starts in a hidden initial state, one combination of the condition levels of four components, the cooler, the valve, the pump and the accumulator, drawn from the seed within a rule on the number of components off their healthy level that only the sampler of initial states reads, so no agent is told it. Time counts rig cycles from the start of the episode and advances only inside an action, and no tool sets the time, following the System app of ARE. The agent reads a sensor group for a number of cycles at a price and a duration, reads the clock with `get_current_time`, waits with `wait_for_notification` until the next notification or a timeout, summarizes the changes since a cycle, looks up the asset profile, the failure modes, the maintenance history of the rig and the repair manual, submits one diagnosis or one declaration of insufficient evidence, creates standard or emergency work orders that a technician executes in visits after a lead time and a repair downtime, and finishes with a summary. A repair takes only when the components it depends on are at their healthy level at its close, the orders created one after another, with no other call between them, form one visit with one visit charge, and an emergency repair restores its component only for a limited hold. Two hidden processes run on the clock, the deterioration that drops a component level by level and the failure rule. Downtime accrues per cycle while a component is degraded and counts toward the cost of the episode, while the budget limits the prices alone.

Every change of the hidden combination, by a repair or a drop, switches the served recording to the stable block of the new combination, so a reading after a repair is a recorded cycle of the repaired combination. The verifier compares the agent's writes with the oracle of the episode and scores the episode from the log and the hidden state. The answer text enters nowhere but the contradiction heuristic of the failure taxonomy.

---

## Data

The recordings are dataset 447 of the UCI Machine Learning Repository, "Condition monitoring of hydraulic systems" by N. Helwig, E. Pignanelli, and A. Schütze, DOI 10.24432/C5CW21, under the CC BY 4.0 license. The rig repeats a constant load cycle of 60 s and records 17 sensor channels and one row of condition labels per cycle. The rig and its faults are described in N. Helwig, E. Pignanelli, and A. Schütze, "Condition Monitoring of a Complex Hydraulic System using Multivariate Statistics," Proc. IEEE I2MTC, 2015, pp. 210-215, DOI 10.1109/I2MTC.2015.7151267.

| Component | Label | Levels | Healthy level |
| --- | --- | --- | --- |
| cooler | `cooler_condition` | 3, 20, 100 (%) | 100 |
| valve | `valve_condition` | 73, 80, 90, 100 (%) | 100 |
| pump | `pump_leakage` | 0, 1, 2 | 0 |
| accumulator | `accumulator_pressure` | 90, 100, 115, 130 (bar) | 130 |

The label row also holds `stable_flag`, which is 0 when the conditions were stable and 1 when static conditions might not have been reached yet. The four level sets give 144 combinations, and `HEALTHY_COMBO` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/env.py`, the combination with every component healthy, is `(100, 100, 0, 130)`. The asset file `CausalDiagnosis/datasets/hyd_1/asset.json` holds these facts in its `components`, `levels`, `healthy` and `observation_kinds` tables, and `CausalDiagnosis/datasets/hyd_1/hyd_1_tables.py` serves them under the names `COMPONENTS`, `LEVELS`, `HEALTHY`, `SENSORS` and `TIER_SENSORS` to the tests of `hyd_1`.

### The data file

| File | Origin | Contents |
| --- | --- | --- |
| `CausalDiagnosis/datasets/hyd_1/data/hyd_cycles.jsonl` | built by `CausalDiagnosis/datasets/hyd_1/download_and_build.py` from the archive of UCI dataset 447 | one record per cycle with `cycle_index`, `labels`, and `features`: the mean, the standard deviation, the minimum and the maximum of every channel |
| `CausalDiagnosis/datasets/hyd_1/data/blocks.json` | built by the same command | the stable block of every combination, as the environment forms it from the records |
| `src/couchdb/scenarios_data/shared/iot/hydraulic_pump_1.json` | the AssetOpsBench asset `hyd_1` | the 17 channel means and a timestamp per cycle, without labels |

The environment reads `hyd_cycles.jsonl` only. A reading returns the mean and the standard deviation of every channel of the sensor group (the served statistics of the `statistics` table of `CausalDiagnosis/datasets/hyd_1/asset.json`) under the field names of the `hyd_1` asset with the statistic appended, such as `PS1_Pressure_bar_100Hz_mean`, where the motor power channel EPS1 reads as `P1_MotorPower_W_100Hz` (the `channels` table of `CausalDiagnosis/datasets/hyd_1/asset.json`, which `HYD1_FIELDS` in `CausalDiagnosis/datasets/hyd_1/former_field_names.py` serves to the tests). The `hyd_1` file serves as a cross check of the cycles and as the source of the field names.

The repository holds no recording. One command, run from the root of the repository, fetches the archive from the official page of the dataset, checks its SHA-256, builds both files and checks them. The command uses the standard library only, so it also runs under a plain `python3 -I`.

```bash
PATH=$PWD/.venv/bin:$PATH python -m CausalDiagnosis datasets download hyd_1 --download
```

The last line it prints is `DATA_OK hyd_1 CausalDiagnosis/datasets/hyd_1/data/hyd_cycles.jsonl`. `CausalDiagnosis/datasets/hyd_1/data/CHECKSUMS.txt`, the one file of that folder that git tracks, pins the SHA-256 of the archive and of the two built files. The command refuses an archive or a built file whose SHA-256 differs from its pin, names both values, and puts no file in the place of the data file. A second run finds the files in place and fetches nothing.

| Option | Effect |
| --- | --- |
| `--download` | fetches the archive from the official page, unless it is already in the folder of `--dest` |
| `--archive FILE` | builds from a copy of the archive that is already on the machine, under the same checks and without the network |
| `--check` | only checks the two built files against `CHECKSUMS.txt` |
| `--dest DIR` | the folder that holds the downloaded archive, by default `CausalDiagnosis/datasets/hyd_1/data/uci_hydraulic/` |

The build adds the samples of a cycle with exactly rounded sums, squares each deviation by one multiplication, and divides by the number of samples less one for the standard deviation, so that the bytes of the two files do not depend on the machine. The pinned checksums verify that on every build.

| Variable | Default | Read by |
| --- | --- | --- |
| `HYD_FACTS_PATH` | unset | `default_facts_path` of `CausalDiagnosis/datasets/dataset_loader/recordings.py`, the one function that names the data file of an asset, which is the file that the variable of `records.data_env` names when it is set and not empty, else the release file `CausalDiagnosis/datasets/hyd_1/data/hyd_cycles.jsonl` |

The variable is an override for development and no step of a normal run sets it. `load_cycles` in `CausalDiagnosis/datasets/dataset_loader/recordings.py` reads the release file only when its SHA-256 equals its pin, and it stops with an error that names the official page and the command when the file is missing or differs. A file that the variable names is read as given. Every file built from the records names the SHA-256 of the data file it was built from under `facts_sha256`, and its reader refuses a file built from another data file. The rule covers the fitted models and the grade table of the CausalDiagnosis MCP, the label table, the fixed initial states and the stored digests. The asset file states the official page and the command under `records.page` and `records.fetch`.

The IndustryAssetEQA facts file of the public EmbodiedQA repository, at commit `d7f42a703751cba077d99bae34e8c5fa4bcfa9fe`, holds the same cycles with their labels. The EmbodiedQA repository has no license file, so this repository holds neither the file nor anything derived from it. A reader who fetches that file can compare it with the rebuilt records, cycle by cycle, with `python -m CausalDiagnosis datasets build-features-and-blocks <extracted archive> --cross-check <file>`, and `--check` in its place compares a fresh build with the pinned checksums.

### Data checks

`verify_data_facts.py` reads the data file and the `hyd_1` file and writes nothing. It does not import the environment, so the tests can compare its blocks with those of `build_blocks`. Apart from the two hashes, the data file against its pin and the `hyd_1` file against the value the script holds, each check is a relation that holds whatever the counts:

- Both files hold the same number of cycles, numbered without gaps from 0 in the data file and from 1 in `hyd_1`.
- Every record of the data file has one label row and the feature names of every other record, with the mean, the standard deviation, the minimum and the maximum of each of the 17 channels.
- `hyd_1` cycle c matches record c - 1 of the data file on all 17 channel means within the tolerance of the script, and a pairing shifted by one cycle agrees on no PS1 mean.
- The labels take exactly the levels above, and the valve is at 100 in every cycle flagged 1.
- All 144 combinations occur, each with stable cycles that form one contiguous block.

`load_cycles` in `CausalDiagnosis/datasets/dataset_loader/recordings.py` reads one `Cycle` per record under the label keys of the asset file, and `build_blocks` groups the stable cycles of each combination into one block and refuses a block that is not contiguous. Cycles flagged as not stable are never served.

---

## Configuration

`EpisodeConfig` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/episode_config.py` holds every parameter an episode depends on besides its seed. The fields an asset decides take their defaults from the asset file `CausalDiagnosis/datasets/hyd_1/asset.json`, as design parameters, and every run records the whole configuration in `run.json`.

| Field | Default | Meaning |
| --- | --- | --- |
| `asset_id` | `hyd_1` | the asset whose profile under `CausalDiagnosis/datasets/<asset_id>/asset.json` supplies the tables, the channels the sensor groups draw from and the data file |
| `components` | cooler, valve, pump, accumulator | from the asset file `CausalDiagnosis/datasets/hyd_1/asset.json` |
| `levels` | the level sets of the table above | per component |
| `healthy` | cooler 100, valve 100, pump 0, accumulator 130 | the level a repair restores |
| `sensor_groups` | `tier1`: TS1 to TS4, VS1, CE. `tier2`: tier1 plus FS1, FS2, CP. `tier3`: all 17 channels | the `observation_kinds` of the asset file, served under the public field names of its `channels` table |
| `prices` | `read`: tier1 1, tier2 3, tier3 10 per cycle. `parts`: cooler 20, valve 20, pump 30, accumulator 20. `repair_charge`: 20. `visit`: 15. `misdiagnosis`: 15. `fix_kind`: cooler replace, valve replace, pump replace, accumulator service | a read price is a non-negative number, zero on the free base row, where the step cap bounds the reads and the cost unit is calls. The visit charge is paid once per visit, and the fix kind names the standard fix that `lookup_fix` reports, `replace` or `service`. The misdiagnosis price is the call-out that a wrong diagnosis wastes, which the environment adds to the hidden record at the write of a wrong diagnosis, outside the budget |
| `durations` | `read_per_cycle` 1, `advance_per_cycle` 1, `lead_time` 3 for every component, `repair_downtime` cooler 1, valve 1, pump 2, accumulator 1 | in cycles |
| `downtime_rate` | cooler 100: 0, 20: 1, 3: 2. valve 100: 0, 90: 1, 80: 2, 73: 3. pump 0: 0, 1: 1, 2: 2. accumulator 130: 0, 115: 1, 100: 2, 90: 3. `stopped`: 5 | per cycle and level, the level keys spelled as JSON strings, and the downtime rate of the stopped rig while a repair stops it |
| `dependencies` | pump depends on valve and accumulator, cooler, valve and accumulator on nothing | the components that must be at their healthy level when an order on a component closes, every component named once. The empty mapping switches the repair rule off, an ablation. A design parameter of the asset file of `hyd_1` |
| `priority` | valve 1, accumulator 2, cooler 3, pump 4 | the order of the repairs within a visit, smaller number first. Every dependency of a component needs a smaller number than the component, so a visit repairs a dependency first |
| `emergency` | `lead` 1, `surcharge` 2.0, `hold` 10 | the emergency mode of a work order, its lead time in cycles, the factor on the part and repair charges, and the cycles its restoration holds after the close |
| `budget` | 300 | the limit on the prices paid alone, since the downtime, the failure penalties and the misdiagnosis price of a wrong diagnosis go to a hidden record that the end entry of the log keeps for the verifier, null for an episode without a budget stop |
| `horizon` | 60 | cycles |
| `max_steps` | 60 | tool calls, refused ones included |
| `fault_rule` | `min_faulty` 1, `max_faulty` 2 | the rule on the number of components off their healthy level at the start of the episode, which only the sampler of initial states reads and no profile, briefing or released file states. A rule may start at zero, which admits the combination with every component at its healthy level, the initial combination of the control preset `healthy_control` |
| `allowed` | null | an optional list of initial combinations within the rule |
| `start` | null | `{"combination", "entry_offset"}` for an anchored initial state, which the anchored preset uses |
| `schedule` | `mode` none, `components` 1, `first_drop` [5, 40], `interval` 10 | the degradation schedule, drawn in mode `seeded` |
| `hazard` | `mode` scheduled, `base_rate` 0.01, `depth_factor` 1.0, `collateral` 0.0 | the degradation hazard, see Hidden processes. `scheduled` keeps the schedule above, and `stochastic` replaces it and needs `schedule.mode` none |
| `fix_history` | `healthy_probability` 0.4, `healthy_ago` [5, 80], `degraded_probability` 0.3, `degraded_ago` [5, 80] | the past work orders drawn at reset, at most one per component, see the maintenance history below |
| `failure` | `patience` 10, `strict` false, `penalty` 50 | the failure rule of the asset file, which big layer 5 turns on with the strict end |
| `clock_mode` | `declared` | `declared`, `generation_time`, `generated_tokens` or `instant` |
| `seconds_per_cycle` | 60 | the seconds of rig time that one recorded cycle stands for, which spaces the record timestamps of the AssetOpsBench rows |
| `think_seconds_per_cycle` | the value of the asset file, 10 | the seconds of the agent's thinking that make one cycle in the `generation_time` and `generated_tokens` modes, a declared value, which no preset sets to a number and the two thinking price presets scale by 0.5 and 2 |
| `seconds_per_token` | null, and in the `generated_tokens` mode the value of the asset file, 10 / 2048, so that 2,048 generated tokens make one cycle | the seconds of thinking that one token generated by the agent's model stands for in the `generated_tokens` mode, which no preset sets and `to_json` leaves out while it is null |
| `time_tools` | true | whether the task server of the AssetOpsBench rows lists the two time tools, the small switch `time_tools` of the real-time clock |
| `guidance_level` | 1 | 1, 2 or 3, the level of the briefing |
| `question` | null | a question that replaces the question sentence of the briefing |
| `anchors` | null | `{"date0": <ISO date>}`, the day of cycle 0 with one recorded cycle per day, which puts a `day` into every answer and on the AssetOpsBench rows stamps cycle 0 one cycle before the end of that day, with an optional `"asset_alias"`, another id that the AssetOpsBench tools accept for the asset |
| `verifier_scope` | `writes` | `writes` or `all`, recorded for the verifier and ignored by the environment |
| `diagnostics_server` | true | recorded for the drivers and the briefing. The profile lists the tools of the CausalDiagnosis MCP only when it is true |

`validate` runs in the constructor and raises one `ValueError` that lists every problem at once: a priced group without sensors, a group without a price, a missing part price, a non finite or negative number, a bool where an integer is expected, a reversed fault rule, an initial combination outside the rule, an anchor that is not an ISO date, a fix kind other than `replace` or `service`, a missing or negative misdiagnosis price, a priority that misses a component, a dependency whose priority number is not smaller than its component's, an emergency surcharge below one, a stochastic hazard over a seeded schedule, an `advance_per_cycle` other than 1, since a wait runs the clock one cycle at a time, the `generated_tokens` mode without a positive `seconds_per_token`, and so on. `to_json` turns every tuple into a list, and `from_json` refuses unknown keys, validates a stored configuration as the constructor does, and gives an equal configuration back. The helpers `n_faulty`, `within_fault_rule`, `is_combination`, `read_price`, `order_price` (with the mode, the surcharge included), `lead_time` (with the mode), `depends_on`, `level_rate` and `date_of` are what the environment and the policies read. The presets of the drivers under `CausalDiagnosis/configs/` compose the big layers of the section Layers over these defaults, and a price table under `CausalDiagnosis/configs/prices/` replaces the prices, and the durations and the downtime rates when the table gives them.

**Budget.** The budget limits the prices paid alone. An action whose price would take the prices paid above `budget` is refused and ends the episode, while the downtime, the failure penalties and the misdiagnosis price of a wrong diagnosis go to a hidden record in the end entry of the log, so the budget counter never moves with the hidden condition. The price of a read is the price of its group for every cycle that returns a reading, so a read that the horizon cuts or a repair stops pays for the cycles it served, and a timed read with a price plays first on a copy of the engine to learn that price, which the budget then decides on. A `budget` of null, as on the base row, has no budget stop, so the horizon and the step cap are the only limits.

**Cost unit.** `cost_unit` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/layers/layer_1_priced.py` gives the unit of every cost of an episode from its prices. The unit is money when a read, a part, the repair charge or the visit has a price, and calls when every price is zero, as on the base row, where the cost of an episode is its number of tool calls, refused ones included, so that the cost of the oracle policy, the denominator of the normalized cost, stays above zero. The verdict carries the unit as `cost_unit`. The loader refuses a composition whose unit does not fit it, by the values and for any asset, through `unit_problems` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/layers/__init__.py`. A cost in money needs a price on the work order of every component, since the oracle policy reads nothing and would pay nothing on an initial state with a fault whose order is free. A cost in calls needs no downtime rate, no penalty of a failure rule that is on and no misdiagnosis price, since a cost in calls leaves the hidden record out. Wherever the cost of the oracle policy on its initial state is zero, as on the initial state without a fault under a cost in money, the normalized cost and the reward of the episode are undefined, None in the verdict.

---

## Layers

The environment is six big layers, the rows of the environment figure and of the ladder of the environment table. The base is always on, and the five layers above it join one at a time in the order of the ladder. Every big layer is one module of `CausalDiagnosis/environment/Causal_Diagnosis_Environment/layers/`, which owns its fields of `EpisodeConfig`, its small switches with their off and on values, and the rules of the layer that the engine, the rig's action server and the metering layer of the AssetOpsBench rows call, so no rule is written twice. `OWNER` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/layers/__init__.py` gives every field of the configuration exactly one owner, and every key of the two tables that layers share, `prices` and `durations`. A small switch sets its fields to its off value, to its on value, which is the value of the asset file unless a table below says otherwise, or to another setting it takes, a mode name, a number or a mapping of the entries it replaces.

### 0 Base

The base holds the hidden fault combination of the initial state, the recordings that AssetOpsBench's `iot`, `fmsr` and `wo` servers or the rig's action server serve, the task tools and the verifier. The base has no small switch, and its fields are the asset with its tables, the fault rule and the list of the initial combinations, the horizon and the step cap of 60 each, the length of one recorded cycle, the guidance level and the question of the briefing, the anchor date, the verifier's scope, whether the CausalDiagnosis MCP is listed, the fix kind of each component and the one cycle that a waited cycle lasts. With every layer above it off, every tool is free, no budget stops the episode and time passes only while the agent waits with the rig's own tools.

### 1 Priced reads and writes

| Small switch | Fields | Off | On |
| --- | --- | --- | --- |
| `read_prices` | `prices.read` | 0 for every sensor group | 1, 3 and 10 per cycle for `tier1`, `tier2` and `tier3`, or a mapping of groups to prices |
| `part_prices` | `prices.parts` | 0 for every component | cooler 20, valve 20, pump 30 and accumulator 20 |
| `repair_charge` | `prices.repair_charge` | 0 | 20 per work order |
| `budget` | `budget` | null, so no budget stops the episode | 300, or a number |
| `misdiagnosis` | `prices.misdiagnosis` | 0 | 15, which the environment adds to the hidden record at a wrong diagnosis and which the briefing states |

`CausalDiagnosis/environment/Causal_Diagnosis_Environment/layers/layer_1_priced.py` holds the cost unit, the price of a read and of a work order, the budget test and the cheapest sensor group that covers a metered read.

### 2 Real-time clock

| Small switch | Fields | Off | On |
| --- | --- | --- | --- |
| `clock_mode` | `clock_mode` and `seconds_per_token` | `declared`, so thinking takes no time | `generation_time`, or the mode `generated_tokens` or `instant` |
| `think_seconds_per_cycle` | `think_seconds_per_cycle` | the value of the asset file | the value of the asset file, a number, or the mapping `{"scale": x}` that gives x times the value of the asset file for a positive x |
| `time_tools` | `time_tools` | the task server of the AssetOpsBench rows lists no time tool | the task server lists `get_current_time` and `wait_for_notification` |

`CausalDiagnosis/environment/Causal_Diagnosis_Environment/layers/layer_2_real_time.py` holds the test of a clock, the rate `seconds_per_token`, which a configuration takes from the asset file in the `generated_tokens` mode alone, and the cycles an action moves the clock in the `instant` mode. `CausalDiagnosis/environment/Causal_Diagnosis_Environment/clock/wall_clock.py` holds the thinking cycles of a call, which add up the seconds of the calls and carry the rest, and `CausalDiagnosis/environment/Causal_Diagnosis_Environment/clock/token_clock.py` the seconds of the tokens generated between two calls in the `generated_tokens` mode with the ledger of those tokens. The rig's action server lists the two time tools on every row as its wait, so the switch `time_tools` acts on the AssetOpsBench rows. Every ladder preset turns the time tools on wherever the clock runs.

### 3 Advanced real-time clock

The advanced real-time clock adds actions that last. A read takes time per cycle read, a work order waits its lead time and its repair stops the rig for the repair downtime, and every cycle accrues production loss at the downtime rate of the degraded components or of the stopped rig.

| Small switch | Fields | Off | On |
| --- | --- | --- | --- |
| `read_per_cycle` | `durations.read_per_cycle` | 0, so a read takes no time | 1 cycle per cycle read |
| `lead_time` | `durations.lead_time` | 0 for every component | 3 cycles for every part |
| `repair_downtime` | `durations.repair_downtime` | 0 for every component | 1 cycle, 2 for the pump |
| `downtime_rate` | `downtime_rate` | every rate 0, the downtime rate of the stopped rig included | 1 to 3 per cycle by component and level, at most 2 for the cooler and the pump, summed over the degraded components, and 5 per cycle while a repair stops the rig |

`CausalDiagnosis/environment/Causal_Diagnosis_Environment/layers/layer_3_advanced_real_time.py` holds the lead time of an order, the declared duration of a tool, the downtime rate of a level and the downtime charge of a cycle. The clock runs where the clock mode of layer 2 is on or a duration of layer 3 is, so a composition with layer 2 off and layers 3 to 5 on still has a clock.

### 4 Repair rules

| Small switch | Fields | Off | On |
| --- | --- | --- | --- |
| `dependencies` | `dependencies` | no component depends on another | a pump repair takes only while the valve and the accumulator are at their healthy level, or a mapping of components to the components they depend on |
| `priority` | `priority` | 1 for every component | valve 1, accumulator 2, cooler 3 and pump 4, or a mapping of components to numbers |
| `visit` | `prices.visit` | 0 | 15 per visit, the standard orders created one after another forming one visit |
| `emergency` | `emergency` | a hold of 0, so no emergency order | an order that arrives after 1 cycle at twice the price and holds its component for 10 cycles |
| `fix_history` | `fix_history` | no past order | one past order closed 5 to 80 cycles before the start of the episode for 40 percent of the healthy and 30 percent of the faulty components |

The dependencies need the priorities, since a visit repairs a dependency first only by its smaller priority number, and the emergency order needs the lead times of layer 3, since it arrives sooner than a standard order only where standard orders wait a lead time. The loader refuses a composition that turns either on without the switch it needs and names that switch and its big layer. `CausalDiagnosis/environment/Causal_Diagnosis_Environment/layers/layer_4_repair_rules.py` holds the dependencies of a close, the visit an order joins and the plan of its repairs, the hold of an emergency repair and its expiry, the maintenance history drawn at reset, the repair manual of `lookup_fix` and the cards the briefing states. On the AssetOpsBench rows the task server lists `lookup_fix` wherever this layer is on and the clock runs, the maintenance history is the work orders that `list_workorders` of the `wo` server returns as closed before the episode, and an emergency order is `generate_work_order` with the work type EM, which the metering layer turns into the engine's emergency mode.

### 5 Deterioration and failure

| Small switch | Fields | Off | On |
| --- | --- | --- | --- |
| `deterioration` | `schedule` and the mode, base rate and depth factor of `hazard` | `none`, so only repairs change a level | `hazard`, under which a component drops one level in a cycle with a chance of 0.01 times one plus 1.0 times the levels it sits below its healthy level, or `schedule`, under which one component drops one level at a cycle drawn in `first_drop` and one more every 10 cycles, or a mapping of a mode with parameters of its card |
| `collateral` | the collateral chance of `hazard` | 0 | 0, the value of the asset file, or a number, the chance that the replacement of a healthy component damages it one level |
| `failure` | `failure` | a patience of 0, so no component fails | a component at its worst level for 10 cycles fails, adds 50 to the hidden record and ends the episode once the cycle of the failure completes, so a repair that ends with that cycle still closes, and only a repair that closes by then averts it, or a mapping of `patience`, `penalty` and `strict`, or the mapping `{"scale": x}` that gives the card of the asset file with the strict end and the patience of the asset file times a positive x, rounded up and refused below one cycle or at or above the horizon of the asset file |

The on value of `failure` is the card of the asset file with the strict end, so turning layer 5 on in any composition gives the failure card with the strict end, while the card of the asset file has none. The collateral chance is 0 in every preset. `CausalDiagnosis/environment/Causal_Diagnosis_Environment/layers/layer_5_deterioration_and_failure.py` holds the mode of a configuration, the schedule and the uniforms of the hazard drawn at reset, the drops of a cycle, the collateral damage of a replacement, the failure check with its penalty and its strict end, and the cards the briefing states.

### Compositions and the ladder

A preset under `CausalDiagnosis/configs/` is a composition. Each preset names the big layers above the base that it turns on under `layers`, sets single small switches under `switches` and sets fields of the base directly, and `compose` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/layers/__init__.py` builds the `EpisodeConfig` from the asset's defaults with every switch of a named layer at its on value and every other switch at its off value. The loader refuses an unknown layer or switch, a setting that a switch does not take, a field of layers 1 to 5 set outside its switch and a switch turned on without the switch it needs, and each refusal names what to use. `without` drops a big layer and the settings of its switches, which gives the row below on the ladder.

| Preset | Big layers above the base | Small switches | Cost unit | What moves the clock |
| --- | --- | --- | --- | --- |
| `base.json` | none | none | calls | waits alone |
| `priced.json` | 1 | none | money | waits alone |
| `real_time.json` | 1 and 2 | none | money | waits and thinking |
| `advanced_real_time.json` | 1 to 3 | none | money | waits, thinking and every declared duration |
| `repair_rules.json` | 1 to 4 | none | money | waits, thinking and every declared duration |
| `deterioration.json` | 1 to 5 | none | money | waits, thinking and every declared duration |

Each row of the ladder differs from the row below in the fields of its big layer and the task tools that layer adds, the time tools on the real-time clock row and `lookup_fix` on the repair rules row, and a seed draws the same hidden initial state under every row. The deterioration row, all six layers with the random hazard rates and the failure card, is the setting of the training table. Every row runs on AssetOpsBench's own `iot`, `fmsr` and `wo` servers behind the metering layer of `CausalDiagnosis/frameworks/AssetOpsBench/tool_servers` with the task server, and `CausalDiagnosis/frameworks/AssetOpsBench/tool_servers/aob_tool_contract.json` names the tool that carries what a layer adds. The twelve tools of the action server, the actions that `HydraulicRigEnv.step` takes, serve the baseline policies, the gym and OpenEnv interface and training only, and a result that uses them says so. The briefing states the rules of the cards that are on and no rule of a card that is off, and `lookup_fix` offers the emergency option and `create_work_order` accepts the mode `emergency` only while the emergency mode is on, as [environment-actions-mcp.md](environment-actions-mcp.md) describes.

The development presets compose the same layers for the tests and the digests.

| Preset | Big layers above the base | Small switches and base fields |
| --- | --- | --- |
| `static.json` | 1, 3, 4 and 5 | the deterioration mode none, the time tools on and the failure card without the strict end |
| `timed.json` | 1, 3, 4 and 5 | as `static.json` with the deterioration mode schedule and `first_drop` from 5 to 30 |
| `anchored.json` | 1, 3, 4 and 5 | as `static.json` with the fault rule of any number of faulty components and the anchor date set as base fields |
| `static_zero_durations.json` | 1, 3, 4 and 5 | as `static.json` with the three durations of layer 3 off, the emergency order off and the time tools off, so no clock runs and the task server lists neither the time tools nor `lookup_fix` |

The supporting ablations of the small switches compose all six layers with one setting changed.

| Preset | Small switches |
| --- | --- |
| `deterioration_thinking_off.json` | the clock mode off, so thinking takes no time, the configuration of `deterioration.json` run with `--clock-mode declared` |
| `deterioration_durations_off.json` | the three durations of layer 3 off, and the emergency order off with them, since it needs their lead times |
| `deterioration_mode_none.json` | the deterioration mode none, the failure card kept |
| `deterioration_failure_off.json` | the failure card off |
| `deterioration_thinking_price_double.json` | `think_seconds_per_cycle` at a scale of 0.5, so a second of thinking moves the clock twice as far as on the deterioration row |
| `deterioration_thinking_price_half.json` | `think_seconds_per_cycle` at a scale of 2, so a second of thinking moves the clock half as far as on the deterioration row |
| `scheduled.json` | the deterioration mode schedule with `first_drop` from 5 to 30 in place of the random hazard rates |

Four presets repeat the rows of the ladder with layer 2 on in the clock mode `generated_tokens` and change nothing else, a second condition of the clock for the rows of the agents on all six layers and for the rungs of the ladder. The rows without layer 2 charge no thinking time and play the same under both clocks.

| Preset | The preset it repeats |
| --- | --- |
| `real_time_token_clock.json` | `real_time.json` |
| `advanced_real_time_token_clock.json` | `advanced_real_time.json` |
| `repair_rules_token_clock.json` | `repair_rules.json` |
| `deterioration_token_clock.json` | `deterioration.json` |

---

## Time

`CausalDiagnosis/environment/Causal_Diagnosis_Environment/clock/episode_clock.py` holds the clock and the hidden processes that run on it. Time counts cycles from 0 and advances only inside `step`.

| Clock mode | What a step advances |
| --- | --- |
| `declared` | the declared duration of the action: `cycles` times `read_per_cycle` for a read, one cycle at a time until a notification or the timeout for a wait, nothing for every other action |
| `generation_time` | the declared duration, plus the agent's thinking time before the action in cycles. The servers charge the seconds of the chess clock below since the last counted call, add them to the seconds carried from the calls before, charge one cycle for every full `think_seconds_per_cycle` and carry the rest to the next call in the state file |
| `generated_tokens` | the declared duration, plus the agent's thinking time before the action in cycles, counted from the tokens that the agent's model generated since the last call. The harness of the agent appends the tokens of every model call to `generated_tokens.jsonl` in the episode folder, and the servers charge the tokens since the last call times `seconds_per_token`, add those seconds to the seconds carried, charge one cycle for every full `think_seconds_per_cycle` and carry the rest. The wall clock seconds between calls are recorded and not charged |
| `instant` | one cycle per step for every tool but `get_current_time`, refused steps included, the rule of agent environments in which a step is a unit of time. A diagnosis is entered at the cycle it is submitted, before the cycle of its own step runs, so a drop inside that cycle does not enter the state it is scored against |

The clock never passes the horizon. A read or a wait is cut at the horizon and the episode ends with the reason `horizon`. In the `generation_time` and `generated_tokens` modes a thinking time that alone reaches the horizon ends the episode before the action runs, and the action is refused with that reason. The `generated_tokens` mode charges the tokens that the model generates and leaves out the time to read the prompt and the delays of serving, which the `generation_time` mode counts, so the cycles of an episode follow the text its model generates and not the speed or the load of the serving engine. A tool call of an episode in that mode whose folder holds no `generated_tokens.jsonl` fails with an error that names the file, since its harness reports no tokens.

The clock of the thinking time is a chess clock that charges the time between two counted calls, less the credited intervals below, and every instant it reads is read by the benchmark's own code (the driver, its tool servers and its model client) on the monotonic clock of the host, never by the agent. Its origin is the handover, which the driver stamps just before it hands the task to the agent (`episode.hand_over`), so the creation of the episode is never charged, and a call of an episode that was never handed over is refused. A counted call is one that reaches `env.step`. Its service starts when the thread that serves it holds the hold of its server process and ends just before its line enters the transcript. The call is charged the time from the cursor to the start of its service, less the credited segments, and the cursor then moves on by the thread time of the serving thread within the service, never more than the service lasted, so the rest of a service, such as a wait on the disk, a lock or the store, is charged at the next call. Two kinds of interval are credited from their start: the first start of each tool server, from the first statement of its launcher to the line `ready` that its main writes just before the stdio loop, for the thread time of the main thread in it, and each failed attempt of a model request (no connection, or a status of 429, 502 or 503) with the registered wait before the next attempt, which the benchmark's model client appends to `model_failures.jsonl` in the episode folder. A restart of a server adds nothing. A counted call of a server process whose `ready` line shows that its readings cannot be trusted is refused by name: a process not started with `python -I`, a trace or profile function, tracing of allocations, `-X dev` or `importtime`, a `sitecustomize` or `usercustomize` module, an environment variable that slows the interpreter, a launcher other than the one the driver wrote, or more threads than its kind starts with. The rule is in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/clock/thinking_clock.py`, and the state file keeps the handover, the cursor, the credited segments that still matter and the credited, uncredited and failed seconds that the reports read.

The thinking rate is one value, `think_seconds_per_cycle` of the asset file `CausalDiagnosis/datasets/hyd_1/asset.json`, which no preset sets to a number. The two thinking price presets scale it by 0.5 and by 2, so they follow the declared value. The `generated_tokens` mode has a second rate, `seconds_per_token` of the same file, which no preset sets, 10 / 2048 seconds, `think_seconds_per_cycle` over 2,048, so that 2,048 generated tokens are one cycle of thinking. The value is separate from `seconds_per_cycle`, which remains the 60 s of a recorded load cycle of the rig and spaces the timestamps of the recorded data that the AssetOpsBench rows serve. A call shorter than a cycle of thinking charges no cycle of its own, since its seconds wait in the state file until they add up to one, so the thinking cycles of an episode are its thinking seconds over `think_seconds_per_cycle` rounded down once. With a thinking rate of a few seconds the horizon of 60 cycles binds through thinking before the step cap of 60 calls for an agent whose calls take longer than a cycle of thinking on average.

Both rates are declared values of the asset file, as the prices and the durations are, and no run measures or sets them. Runs under different values never pool, since they are runs of different commits. The values decide whether a model agent can act in time, because the failure card, on in the full environment, ends the episode once a component has stayed at its worst level for the patience of the card, open work orders or not. The first model job of a cluster session prints, from one timed model call, the seconds of that call, the cycles it would charge and the projected hours of the job, and prints nothing that stops it.

### One cycle of the clock

`RigClock.advance_cycle` runs the processes of cycle t in this order, then moves the clock to t + 1:

1. The emergency repairs whose hold ends at t expire. The component returns to the level it had before the repair (the hidden event `emergency_expired`) and the window switches.
2. The scheduled drops at t set the component to its target level and switch the served window to the new combination. Under the stochastic hazard the hazard drops of cycle t take their place. A drop ends the hold of an emergency repair on its component without an expiry, since it sets an absolute level.
3. Work orders, taken in priority order. An order whose repair start has come goes into repair, which stops the rig (the system event `work_order_executed`). An order with a repair downtime of zero closes as the technician reaches it.
4. The recording of the cycle: the next cycle of the window of the current combination, or none while the rig is stopped for a repair.
5. Downtime accrual: the downtime rate of the stopped rig while it is stopped, else the sum over components of `downtime_rate[component][level]`.
6. The failure check: a component at its worst level for `patience` consecutive cycles fails once. The count runs while a work order on the component is open and starts again only when the component leaves its worst level. A repair averts the failure only by closing by the window end of its run at the worst level, which on the first run is the window end of the oracle. The system event `failure` adds `penalty` to the cost, and under `strict` it ends the episode once the cycle of the failure completes, so a repair that ends with that cycle still closes in step 7.
7. The clock moves to t + 1, and every order in repair whose downtime ends there closes, in priority order. The component becomes healthy and the window switches to the repaired combination, unless the repair rule of the work orders below makes the order ineffective.

`degradation_order` lists the levels of a component from its healthy level to its worst as the `degradation` table of the asset file orders them, which gives cooler 100, 20, 3, valve 100, 90, 80, 73, pump 0, 1, 2, and accumulator 130, 115, 100, 90. The worst level is the last one.

### Hidden processes

`reset` draws, from the generator seeded by the seed and in this order, the hidden combination (unless `start` names it), the served window and the schedule. The uniforms of the hazard and the maintenance history come from generators of their own, seeded with the seed plus 2000003 and with `fix-history:<seed>`, so they change none of these draws.

- **Schedule.** In mode `seeded`, `draw_schedule` picks `components` distinct components, the ones healthy at the start of the episode first, so that a scheduled episode changes, and among the faulty ones only when more components are asked for than are healthy. Each picked component gets a first drop cycle uniform in `first_drop` and then a drop every `interval` cycles down to its worst level. The drops are stored with their absolute target levels, so a repair between two drops does not change the later drops. In mode `none` the schedule is empty.
- **Hazard.** In the mode `scheduled`, the default, the schedule above drops the components. In the mode `stochastic` the schedule is off, and a component drops one level in cycle t when the uniform `drop[t][c]` is below `base_rate` times one plus `depth_factor` times the number of levels the component sits below its healthy level (`hazard_rate` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/layers/layer_5_deterioration_and_failure.py`). Running degraded makes the next drop likelier, so what the agent does changes what happens to the rig. The uniforms are drawn once at reset for every cycle up to the horizon and every component (`draw_hazard`), so the same seed with the same actions replays exactly, while another action sequence meets other drops. In either mode the replacement of a component that is healthy at the close damages it one level when its uniform `collateral[t][c]` is below `collateral`, off at 0. A hazard drop is the hidden event `hazard_drop` with its `cause`, `hazard` or `collateral`. The hazard leaves the failure rule unchanged.
- **Failure rule.** The failure check above, with `patience`, `strict` and `penalty` from the configuration.
- **Maintenance history.** For each component at its healthy level at the start of the episode, with probability `healthy_probability` one past standard order closed a number of cycles before the start of the episode drawn uniformly in `healthy_ago`, and for each degraded component, with probability `degraded_probability` one past order closed a number of cycles in `degraded_ago` before the start of the episode (`draw_fix_history` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/layers/layer_4_repair_rules.py`). Each entry holds the identifier `hist/<n>`, numbered from the oldest, the component, the closing cycle as a negative number, the mode and the outcome `closed`, and `get_fix_history` serves the list. The history is mildly informative by construction. The default age windows coincide, so the closing cycle of a past order tells nothing about the condition of its component, and only the two probabilities, a past order being likelier on a healthy component than on a degraded one, weigh the candidate root causes, as the maintenance record of a real rig carries that weak signal, while the low probabilities keep the record from settling a diagnosis without readings.

### Work orders

`WorkOrder` holds the id, the component, the creation cycle, the cycle its repair starts, the closing cycle, the source `agent`, the prediction identifier and the evidence identifiers the agent cited, the hidden flag `unnecessary`, the mode (`standard` or `emergency`), the visit, the status (`open`, `in_repair`, `closed`) and the closing cycle once closed. Agent orders are numbered `wo-1`, `wo-2` and so on. An order on a component that is at its healthy level, and not held there by an emergency repair, is executed, charged and marked unnecessary in the hidden state. The public view of an order, `WorkOrder.public`, holds `id`, `component`, `created_t`, `expected_close_t`, `status`, `source`, `closed_t`, `mode` and `visit`, never the flag.

**Visits.** The standard orders created one after another, with no other call of the environment between them, form one visit, numbered from 1, and an emergency order forms a visit of its own. The thinking time between the calls and the calls of the CausalDiagnosis MCP end no visit; every other call does, a refused one included (`servers/public_rules/visits.py`). The technician of a visit arrives at the latest creation cycle plus lead time of its orders and executes the repairs back to back in priority order, smaller number first, each with its own repair downtime (`visit_schedule` in `servers/public_rules/visits.py`, which the agent and the CausalDiagnosis MCP read too); an order that joins a visit after one of its repairs has begun is repaired after the repairs begun. The rig is stopped at the stopped downtime rate from the arrival to the last close, and the recording switches at each close. The order that opens a visit pays `prices.visit` on top of its part and repair charges, and every charge is paid at creation, so the agent sees it in `budget_left` at once. Orders with another call between them are different visits, which may overlap in time. Before the visits existed, two orders of one cycle each arrived after their own lead time and were repaired at the same time, so a visit now stops the rig for the sum of its repair downtimes. A later order of the same visit with a smaller priority number goes first, so the `expected_close_t` of an earlier order can move after its creation. The envelope shows the current value, and the creation entry of the log keeps the value at creation.

**Rows without a clock.** A row has no clock when its mode is `declared` and its read, lead and repair durations are all zero (`has_clock` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/layers/layer_2_real_time.py`), as on `base.json`, `priced.json` and `static_zero_durations.json`, so no cycle runs in which an order of zero lead time and zero repair downtime could close. The repair card decides what happens instead (`repair_card_on`). While it is off, with no dependency and no visit charge, the order closes inside its own `create_work_order` call, in priority order with any other due order, and the answer carries the close as a notification. While it is on, the orders created since the last call that was not an order form one visit, which closes in priority order before the next call that is not an order and before the end of the episode, so a dependency ordered after its dependent still takes first. Only a visit whose orders are still open can be joined, so an order after that call opens a visit of its own and pays its charge. An emergency order on such a row stays open until its lead has passed in waits.

**Repair rule.** When an agent order on a component closes, the component is restored only if every component it depends on (`dependencies`) is at its healthy level in that cycle. Otherwise the order is ineffective. Its price and its downtime are charged as usual, the level of the component does not change, and the hidden event `order_ineffective` names the degraded dependencies. The agent sees an ordinary closed order in the envelope and in the change summary and learns from the readings that the repair did not take. The closes of one cycle are taken in priority order, so a dependency repaired earlier in the same visit, or closing in the same cycle, counts as healthy. An order whose component is healthy, and not held there by an emergency repair, changes nothing and records no event. The empty mapping of `dependencies` switches the rule off.

**Emergency mode.** `create_work_order` with the mode `emergency` asks for a repair that arrives `emergency.lead` cycles after the creation, costs the part and repair charges times `emergency.surcharge` plus the charge of its own visit, and follows the repair rule like any order. Its restoration holds for `emergency.hold` cycles after the close. As the clock passes the end of the hold, the component returns to the level it had before the repair (the hidden event `emergency_expired`), unless a standard order on it closed effective in between, which makes the repair permanent. A scheduled drop or a hazard drop of the component during the hold ends the hold without an expiry, since a drop sets the level itself. The default mode is `standard`, also when the argument is missing or empty.

### Windows

The environment serves the first `BLOCK_CAP` stable recordings of every combination, ten (`CausalDiagnosis/datasets/dataset_loader/evidence_source.py`), so that the length of a block reveals nothing about the combination. At reset the episode draws one entry offset o, the same way for every combination, which cuts these recordings into two segments of `SEGMENT` recordings, five, the block positions o to o + 4 and o + 5 to o + 9 modulo ten. A combination met for the first time in the episode opens a window on its live segment, the second one, and a return to a combination played before resumes its window where the rig left it. The engine of the AssetOpsBench rows records the first segment of the initial combination before cycle 0, and no live window plays it. A read serves, for every advanced cycle in which the rig ran, the recording the rig played during that cycle, so a drop or a closure in the middle of a read shows in the later observations. A read of cycles already recorded returns at once and a read of cycles not yet recorded plays them on the rig at their declared duration. The rig records a cycle only by playing it, so every read here plays new cycles. On the advanced real-time row a read lasts one cycle per cycle read, and on the rows whose read duration is zero, the base, priced and real-time rows, the cycles come at once from the window at the current cycle, so a read never reaches past the present. No read of the rig is a re-read, since every read serves recordings that no earlier read served, so a read of a second group at the same cycle pays its own price per cycle. Once the reads have gone through the whole window, the next cycle repeats one, which the log marks with `repeated` true and no public view shows. An anchored initial state names its entry offset (`start.entry_offset`), and an offset past nine is clamped to nine. The anchored preset names the offset whose live segment opens with the recording of the scenario's date, five positions before it (`start_of` of `CausalDiagnosis/frameworks/AssetOpsBench/scenarios/scenario_anchors.py`), and a date whose recording lies past the tenth of its block is refused.

A combination with fewer than `BLOCK_CAP` stable recordings is refused when the source is built, since every entry offset must exist for every combination alike.

---

## Episode interface

`HydraulicRigEnv(cycles, blocks, config=None)` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/env.py` takes the output of `load_cycles` and `build_blocks` and an `EpisodeConfig`, the default when none is given. The candidate initial combinations, `candidates`, are the combinations within the fault rule, restricted to `allowed` when it is set. The constructor raises `ValueError` when none remains and computes the public normal ranges from the data.

`reset(seed)` seeds the episode's own `random.Random`, draws the hidden state as described above and returns the first envelope, whose `result` is the asset profile. Every later draw comes from the same generator, so a seed, a configuration and an action sequence reproduce the log byte for byte, which the action server relies on when it replays an episode.

`step(action, think_cycles=0)` takes a dict with a `type` key and returns `(envelope, cost, done, info)`. `cost` is everything the step charged: the price of the action, the downtime the step let accrue and any penalty. `info` holds `refused` after a refusal and, on the last step, `log` with the full log and its hidden state and `reason`, which no tool server may forward. A step after the end raises `RuntimeError`, an action that is not a dict raises `TypeError`, and so does a `think_cycles` that is not a non negative integer. `think_cycles` is charged before the action in the `generation_time` mode and ignored otherwise.

| Attribute or method | Content |
| --- | --- |
| `log` | the full log, hidden fields included, for the verifier only |
| `time`, `cost`, `steps`, `done`, `diagnosed`, `declared` | the clock, the cost so far, the steps taken, and the state of the episode |
| `initial`, `hidden`, `candidates`, `clock` | the initial combination, the current combination, the candidate initial combinations and the `RigClock`, all hidden from agents |
| `window_start`, `entry_offset`, `combination_at(t)`, `cycle_at(t)` | the window of the current combination, the combination current at cycle t and the recording index played during cycle t, hidden |
| `normal_ranges` | the public normal range of every channel mean |
| `fix_history` | the past orders drawn at reset, which `get_fix_history` serves |
| `oracle()` | the oracle of the episode, see below |
| `public_profile()`, `public_event(entry)`, `public_events()`, `public_work_orders(closed_since=None)`, `fix_manual(component)`, `envelope(result=None, refused=None, notifications=())` | the agent facing views, `fix_manual` being the answer of `lookup_fix` |

Every answer is an envelope:

| Field | Content |
| --- | --- |
| `time` | the current cycle |
| `day` | the ISO date of the current cycle, present only when `anchors` is set |
| `budget_left`, `steps_left` | the budget less the prices paid, which never moves with the downtime or a penalty and is absent without a budget, and the step cap left |
| `open_work_orders` | the public view of every open work order |
| `episode_over` | whether the episode has ended |
| `notifications` | the messages of the public events since the previous call in log order, which are the start of a repair, the close of a work order, a failure and the alarm of a channel of a sensor group already read, then the timeout of a wait that reached it and the end of the episode, each with `message_type`, `t`, `message`, `event` and the public fields of its event, in the message types of the notification system of ARE. Hidden events and the agent's own actions never notify. The `no_call` step of an iteration of the agent that made no counted call answers no agent, so the messages it lists stay pending and the next counted call carries them, a wait ending at once on them |
| `result` | the answer of an accepted action, absent on a refusal |
| `refused` | the reason of a refusal, absent on an accepted action |

The asset profile, the `result` of `get_asset_profile` and of the first envelope, holds `asset_id`, `asset` (type, description and site from the sample asset profiles of the repository), `components`, `levels`, `healthy`, `sensor_groups` with the public field names, `statistics`, `prices`, `durations`, `downtime_rate`, `budget`, `horizon`, `max_steps`, `processes` (the parameters of the schedule, the hazard, the emergency mode, the dependencies, the priorities, the maintenance history and the failure card under their configuration field names `schedule`, `hazard`, `emergency`, `dependencies`, `priority`, `fix_history` and `failure`, on or off as the episode's layers set them, never their draws), `clock_mode`, `seconds_per_cycle`, `think_seconds_per_cycle`, `clock` (the clock block of the tool contract, with the clock modes, the order of one cycle and the notifying events, and the episode's `mode`, `seconds_per_cycle`, `think_seconds_per_cycle` and the declared duration of every action tool under `durations`, and in the `generated_tokens` mode the rule of that mode under `modes` and `seconds_per_token`), `tools` (the table of the tool contract for the action server and, when `diagnostics_server` is true, for the CausalDiagnosis MCP, every row with `price_value`, the price of a call under this configuration, per cycle and sensor group for a read and per mode and component with the visit charge for a work order, and `duration_value`, its declared cycles per cycle read or waited, zero where the tool is free or takes no time, and while the emergency mode is off the `create_work_order` row lists no mode and its rules drop the emergency clauses) and `normal_ranges`.

| End reason | When |
| --- | --- |
| `finish` | the agent finishes, at any time. On a row with a clock the end entry comes after the run of the clock to the last close of the open work orders, or at the horizon or a strict failure when either comes first |
| `budget` | the price of an action would take the prices paid above `budget`, so it is refused and the episode ends, while the downtime and the penalties never stop an action |
| `horizon` | the clock reaches `horizon` |
| `max_steps` | the step count reaches `max_steps` on a step that does not end the episode otherwise |
| `failure` | a failure event under the strict rule |

Recovery does not end the episode, since an end on recovery would tell the agent the hidden state. The agent finishes.

---

## Actions

Every action is a dict with a `type` and the arguments below. The class, the duration and the price of every action come from the tool contract in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/tool_contract.json`, which [environment-actions-mcp.md](environment-actions-mcp.md) describes. No action sets the time, following the System app of ARE, so `get_current_time` reads the clock without moving it and `wait_for_notification` lets the rig run until the next public event or a timeout.

| `type` | Arguments | Class | Duration | Price | `result` |
| --- | --- | --- | --- | --- | --- |
| `get_asset_profile` | none | read | 0 | 0 | the asset profile |
| `read_sensors` | `sensor_group`, `cycles` (a positive integer) | read | `cycles` times `read_per_cycle` | `prices.read[sensor_group]` for every cycle that returns a reading | `sensor_group`, `served` (one entry per cycle the rig ran, with `n` the observation number, `t` the cycle and `values`) and `stopped`, the read cycles during which the rig was stopped for a repair and produced no observation |
| `get_current_time` | none | read | 0 in every clock mode | 0 | `time`, `horizon`, `seconds_per_cycle`, `think_seconds_per_cycle`, `seconds_per_token` in the `generated_tokens` mode, `budget_left` when the episode has a budget, and `day` when it is anchored |
| `wait_for_notification` | `timeout` (a positive integer, or empty for the cycles left before the horizon) | read | one cycle at a time until a notification is pending or the timeout is reached, one cycle in the `instant` mode | 0, downtime accrues meanwhile | `time`, `waited`, the cycles the wait took, and `ended_by`, which is `notification`, `timeout` or `stopped` |
| `summarize_changes_since` | `cycle` (an integer from 0 to the current time) | read | 0 | 0 | `since`, `time`, `cycles_elapsed`, `alarms` (one entry per channel mean of a sensor group already read in the episode that left its public normal range in any cycle the rig ran since `cycle`, with `channel` and `first_t`) and `work_orders_closed` (the public orders closed at or after `cycle`) |
| `get_failure_modes` | `component`, or an empty string for all | read | 0 | 0 | `component` (null for all), `association_source` and `failure_modes`, one entry per fault code of the knowledge base with `component`, `level`, `code`, `severity`, `action`, `channels` (the public fields of the channels that indicate the component) and `evidence_id` |
| `get_fix_history` | none | read | 0 | 0 | `history`, the past orders of the rig, oldest first, each with `id` (`hist/<n>`), `component`, `closed_t` (negative), `mode` and `outcome` |
| `lookup_fix` | `component` | read | 0 | 0 | the repair manual of the component, `component`, `standard` (`mode`, `fix`, `part`, `repair_charge`, `price`, `visit_charge`, `lead_time`, `repair_downtime`), `priority`, `dependencies`, `effect` in words and, while the emergency mode is on, `emergency` (`mode`, `price`, `visit_charge`, `lead_time`, `repair_downtime`, `hold`, `effect`) |
| `submit_diagnosis` | `diagnosis` (one integer level per component), `evidence_ids` (a list of strings, possibly empty) | write | 0 | 0 | the stored `diagnosis` and `evidence_ids` |
| `declare_insufficient_evidence` | `separating_test` (`sensor_group` and a positive `cycles`), `evidence_ids` | write | 0 | 0 | the stored `separating_test` and `evidence_ids` |
| `create_work_order` | `component`, `prediction_id` (a string, empty without a prediction), `evidence_ids`, `mode` (`standard`, the default, or `emergency` while the emergency mode is on, and absent from the tool table while it is off) | write | 0 at creation, the order closes when its visit has repaired it, see Work orders | `parts[component]` plus `repair_charge`, times `emergency.surcharge` for an emergency order, plus `prices.visit` for the order that opens a visit, charged at creation | `work_order`, the public view of the new order |
| `finish` | `summary` (a string) | write | 0 | 0 | `summary` |

A diagnosis or a declaration is accepted once per episode and locks the other. The environment accepts a declaration whatever it cites, and the verifier scores one that cites no reading released before it as a failure. Reads and work orders stay allowed after a declaration, a diagnosis does not. A work order needs a diagnosis or a declaration first and one open order per component at most. `finish` is accepted at any time. On a row with a clock the clock then runs to the last close of the open work orders with no further call (`_settle_open_orders` in `env.py`), so the downtime and the events of that span count, no alarm is logged in it, and the verifier scores the state after the last close. The horizon or a failure under the strict rule stops that run first and leaves the remaining orders open, and the end reason stays `finish`. The settle keeps a clock row fair to an agent that does not wait for its closes, while the downtime of the wait still counts. A row without a clock closes its orders as the paragraph on rows without a clock states. An order still open when the episode ends, at the step cap or the budget stop for instance, counts as not repaired, while a `finish` that is the last call the step cap allows still settles.

Any other malformed or out of order action that is a dict is refused: it is logged with its reason, costs nothing, changes no state and counts as a step, so a language model agent that emits a bad action loses a step and not the episode. In the `instant` mode a refused step advances the clock by one cycle like every other step. Only a refusal for lack of budget ends the episode.

An integer argument is read by its value, whatever JSON type writes it. An int that is no bool, a float without a fraction and a string that holds a decimal integer, with an optional sign and an optional fraction of zeros, such as `"2"`, `"2.0"` or `"-5"`, all give the integer, which the log entry and the answer hold. A value that writes no integer, such as a bool, null, `"two"`, `2.5` or `"1e2"`, is refused with the reason of its argument, where an empty `timeout` has its own meaning.

| Reason | Cause |
| --- | --- |
| `unknown action type` | a `type` that is not one of the twelve |
| `the server <server> has no tool <tool> in this episode` | a call of a tool that the row does not list on the server it is asked of, which the metering layer of the AssetOpsBench rows passes to the engine as the action `unlisted_tool`, so that it counts as a refused step |
| `unknown sensor group` | a `sensor_group` outside `sensor_groups`, a missing one included |
| `cycles must be a positive integer` | a `cycles` of `read_sensors` that writes no integer at or above 1 |
| `cycles must not exceed the horizon` | a `cycles` of `read_sensors` above the horizon on a read that the clock does not cover, under a zero `read_per_cycle` or in the `instant` mode, whose cycles come at once |
| `timeout must be a positive integer, or empty for the cycles left before the horizon` | a `timeout` of `wait_for_notification` that is neither empty nor writes an integer at or above 1 |
| `cycle must be an integer between 0 and the current time` | a `cycle` of `summarize_changes_since` that writes no integer in that range |
| `unknown component` | a `component` outside the four, in `get_failure_modes`, `lookup_fix` or `create_work_order`, an empty one in `lookup_fix` included |
| `unknown site <site_id>: the site of the asset is <site>` | on the AssetOpsBench rows, a `generate_work_order` at a `site_id` that is not the site of the asset, compared in upper case, which the metering layer passes to the engine as an order on no component: the engine's log keeps `unknown component`, and the answer and the call's entry in `tool_calls` name the site (`MeterConfig.site_refusal` in `metering_layer.py`) |
| `a diagnosis or a declaration was already submitted` | a second diagnosis or declaration of either kind |
| `each component must be given one condition level` | a `diagnosis` that is not a dict with exactly the four components |
| `a component's level must be one of its condition levels` | a level that writes no integer among the levels of its component |
| `sensor_group must name a sensor group and cycles must be a positive integer` | a declaration whose test is malformed, a `sensor_group` that is no string included |
| `evidence_ids must be evidence identifiers, separated by commas in a tool call` | an `evidence_ids` that is not a list of strings |
| `a diagnosis or a declaration must precede a work order` | `create_work_order` before the commitment |
| `prediction_id must be a string, empty when there is no prediction` | a `prediction_id` that is not a string |
| `mode must be standard or emergency` | a `mode` of `create_work_order` that is neither, a missing or empty mode being `standard` |
| `the emergency mode is off in this episode` | a `mode` of `emergency` while the emergency hold is 0, the off value of its card |
| `a work order on the <component> is open; it closes at cycle <close>` | a second order on a component whose order has not closed, `<close>` the close of that order as its visit schedules it at the refusal, which is the `expected_close_t` that the open work orders of the same answer list, and on a row without a clock whose repair card holds the order until the next call that is not an order, the current cycle |
| `summary must be a string` | a `finish` without a string summary |
| `the episode ended before the action ran` | the thinking time of the `generation_time` or `generated_tokens` mode alone reached the horizon or a strict failure |
| `budget` | a read or a work order that costs more than the budget left, which also ends the episode |

### Served values and normal ranges

A reading returns the mean and the standard deviation of every channel of the group under the public field names at six significant digits (`served_values` in `CausalDiagnosis/datasets/dataset_loader/evidence_source.py`, which `env.py` wraps for the default asset). The rounding happens inside the environment, so the log, the envelope and the released evidence carry the same numbers.

The public normal range of a channel mean is the interval from its smallest to its largest value (`BAND_QUANTILES`) over the stable cycles of the healthy combination, so no healthy recording stands outside it, computed on the served values when the environment is built and served in the profile under `normal_ranges`, keyed by the field of the channel mean. The normal ranges are served values, not reported results. An alarm of `summarize_changes_since` is a channel mean of a sensor group already read in the episode that is outside its normal range in a cycle the rig ran since the given cycle, with the first such cycle. A group never read raises no alarm, since a free summary of its channels would replace their priced reading. The environment also logs an `alarm` event the first time a channel of a group already read stands outside its normal range, once per channel at the cycle the change summary names, and every answer carries it among the notifications, so a wait ends at it. Alarms are computed from the recording the rig played and never from a label, and a cycle during which the rig was stopped produces none.

---

## Log and information barrier

Every log entry carries `event`, `step` (the step number of an agent action, null for a system event), `t_start`, `t_end`, `class` (`read`, `write` or `system`, from the tool contract) and `cost`, the entry's own charge. Agent entries also carry `think_cycles`.

| `event` | Other fields, hidden ones marked |
| --- | --- |
| `reset` | `seed` (hidden), `hidden` (the initial combination, hidden), `entry_offset` (hidden), `schedule` (hidden), `fix_history` (the drawn history, released only through `get_fix_history`, since no public view covers the reset), `config` (hidden, since it may hold `start` and `allowed`) |
| `profile` | none |
| `read` | `sensor_group`, `cycles`, `stopped`, `served`: per observation `n`, `t`, `cycle` (the recording index, hidden), `combination` (hidden), `repeated` (hidden), `values` |
| `current_time` | none |
| `wait` | `timeout`, the cycles asked or, when the timeout was empty, the cycles left before the horizon, and `ended_by` |
| `summary` | `since`, `alarms`, `work_orders_closed` |
| `failure_modes` | `component` |
| `fix_history` | `history`, the entries served |
| `fix_lookup` | `component` |
| `submit_diagnosis` | `diagnosis`, `evidence_ids` |
| `misdiagnosis` | the misdiagnosis price as `cost`, at the cycle of a wrong diagnosis right after its `submit_diagnosis` entry, the whole entry hidden |
| `declare_insufficient_evidence` | `separating_test`, `evidence_ids` |
| `work_order_created` | `id`, `component`, `mode`, `visit`, `visit_charge` (the part of the entry's cost that pays the visit), `prediction_id`, `evidence_ids`, `expected_close_t` (at creation), `unnecessary` (hidden) |
| `finish` | `summary` |
| `refused` | `action` as the environment received it, `reason` |
| `scheduled_drop` | `component`, `from_level`, `to_level`, the whole entry hidden |
| `work_order_executed` | `id` |
| `work_order_closed` | `id`, `component`, `source`, `combination_after` (hidden) |
| `order_ineffective` | `id`, `component`, `degraded_dependencies`, the whole entry hidden |
| `hazard_drop` | `component`, `from_level`, `to_level`, `cause` (`hazard` or `collateral`), the whole entry hidden |
| `emergency_expired` | `id`, `component`, `to_level`, `combination_after`, the whole entry hidden |
| `failure` | `component` (hidden), with the penalty as `cost` |
| `alarm` | `channel`, logged once per channel at the first cycle a channel of a sensor group already read stood outside its normal range, after every cycle of a wait and every action that leaves the episode running, so none in the run of the clock after a `finish` |
| `end` | `reason`, `time`, `final` (hidden), `cost` (the episode total, hidden), `ledger` (hidden, the downtime and the penalties, the misdiagnosis charge among them, that the budget counter leaves out), `downtime_total` (hidden), `downtime_by_cycle` (hidden, `t` and `charge` per cycle with a charge), `played` (hidden, the recording index and the combination of every cycle the rig ran), and `held` (hidden) while an emergency hold is in force at the end, the level it replaced for every component it holds |

`public_event` gives the agent facing view of an entry: it strips `HIDDEN_FIELDS` (`cycle`, `combination`, `combination_after`, `unnecessary`, `hidden`, `schedule`, `final`, `seed`, `config`, `played`, `repeated`, `entry_offset` and `held`) at the top level of an entry and in every served observation, returns None for a scheduled or hazard drop, an ineffective close and the expiry of an emergency repair (`HIDDEN_EVENTS`), which the agent has to notice through readings and alarms, and strips the component from a failure, so the agent learns that a failure happened and when but not where, and the cost, the ledger and the downtime from the end entry (`END_LEDGER_FIELDS`), which would tell the hidden condition. `public_events` lists the public views of every entry after the reset. The envelopes, the asset profile, the public events and the briefing of the action server are the only agent facing texts. Everything else stays in `env.log` and, in a run folder, under `hidden/`, which no agent facing tool reads. An agent with file or shell tools has to run with the episode and run folders out of its reach, since the state file of the action server holds the seed, from which a replay recovers the hidden state.

---

## Oracle

`env.oracle()` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/oracle.py` writes the reference the verifier compares the agent's writes with, in the form of the Agents Research Environments verifier: `{"diagnosis": {component: level at the start}, "work_orders": [{"component", "window_end"}], "finish": true}`.

The work order set holds every component that is faulty at the start or dropped by the schedule or the hazard before the episode ends (before the horizon while the episode runs). A drop at cycle s happens when the clock passes s, so an episode that ends at time s never saw it. `window_end` is the cycle at which the failure event would fire for the component, that is the cycle at which it reaches `patience` cycles at its worst level, or the horizon when it never would, which holds for every component while the failure rule is off. An order that closes at or before `window_end` and restores its component averts the failure, since the closure is processed before the failure check of that cycle, and an order that is open but not closed by then does not. `window_end` is the deadline of the failure card on the first run of the component at its worst level. A later run, which opens when the component comes back to its worst level, has a window end of its own, its first cycle plus `patience` less one, capped at the horizon. The timing check of the verifier reads the window end of the run at the worst level that is open at each closure, and the horizon for a closure outside every such run. Because the drops of the hazard depend on what the agent does, the oracle is evaluated against the log when the episode ends.

The set holds one order per component. Under the repair rule an agent that orders a component before its dependencies, or follows an emergency repair with a standard one, places more orders than the set holds. The hard check of the verifier compares the set of components ordered, so such a repeated order does not fail it.

---

## Baseline policies

A policy in `CausalDiagnosis/environment/reference_policies/policies.py` has `start(first_envelope)`, called with the envelope that `reset` returns, and `act(envelope)`, called with the answer to its previous action. It sees nothing else, which is what a language model agent gets through the tools. A harness that skips `start` may hand the reset envelope to `act`, which starts the policy itself. `run_episode(env, policy, seed)` plays one seeded episode to its end and returns the log.

- **Oracle.** `OraclePolicy(env)` reads the hidden state through the environment: the initial combination, the current levels, the schedule and the table of uniforms of the stochastic hazard. It submits the current combination in every episode, which the verifier scores at the cycle of the write, since a correct diagnosis with a recovered rig is a success whatever the diagnosability label says. It orders every faulty component once it will not drop further and its repair will take, in priority order within the visit of the cycle, waits with `wait_for_notification` until the orders close and finishes. Each repair has to close by the window end of its component, since an open order does not avert the failure, so a component that waits for a later drop of its own is ordered from the latest cycle whose order, after the lead time and the repair downtimes of its visit, still closes by its window end, unless the close could then come before the drop, and no cycle waited for the hazard moves a close of the visit past its window end. The oracle waits for a drop only when the drop comes before the cycle at which its orders would close without the wait, since a later drop never happens once the oracle has finished. A component whose dependency is degraded outside the visit, or drops before the component's close, waits until the dependency is repaired or has dropped for the last time before the oracle could finish. Under the stochastic hazard the oracle reads from the table of uniforms the first drop of every healthy component and treats that drop as a scheduled one, places a visit a cycle later whenever the table would drop one of its repaired components again before the last of its orders closes, the orders after the drops it waits for included, and repairs every degraded component at once and again after every drop, since every cycle waited is a chance of a new drop. It does not wait for a drop whose repair could not close before the horizon, and the oracle set of such an episode never holds that component. When this schedule, played to the end on a copy of the environment, would still end with a failure, the oracle plays instead the plan of order cycles of the faulty components that passes with no failure, the orders of one cycle, created one after another, forming one visit. Each component is then ordered no later than the last cycle at which its order alone still closes by its window end, the plans with the earliest last cycle come first, and among them the oracle takes one that meets the timing check over one that misses it and then the cheapest. The oracle is the reference for the cost of an episode and the best of the five baseline policies, with a reward at least that of every other baseline policy on every seed of the evaluation pool and of every test replicate, on every row of the ladder and under both modes of the deterioration layer, and only the tests, the drivers and the gym wrappers construct it, none of them with a grade.
- **Do nothing.** `DoNothingPolicy()` waits up to the horizon with `wait_for_notification`, waits again after every notification, and finishes if the episode is still open. It is the lower reference of the verifier.
- **Declaring.** `DeclarePolicy()` reads the cheapest sensor group for one cycle, cites that reading, declares insufficient evidence and finishes, whatever the reading says, with no model and no hidden state. It is the most a constant declaration can earn.
- **Random.** `RandomPolicy(seed)` draws from the stream `random-policy:<seed>`, kept apart from the environment's generator so that its first decision is not tied to the hidden combination. Before the diagnosis it submits a random diagnosis with probability 0.15, or whenever no sensor group is affordable, and otherwise reads one cycle of a random affordable group. After the diagnosis it orders a random component without an open order with probability 0.4 when its price and a visit charge are affordable, waits one to five cycles for a notification with probability 0.3, and finishes otherwise.
- **Greedy.** `GreedyPolicy(classifiers, confidence=0.95, min_gain_per_price=0.05)` in `CausalDiagnosis/environment/reference_policies/greedy_policy.py` keeps a belief over the 144 combinations, starting uniform over all of them, reads every served observation through the cross fitted classifiers, buys one cycle of the affordable sensor group with the largest expected information gain per unit price while keeping the budget of the work orders that its most probable combination needs, submits the most probable combination at the confidence it is constructed with, with the `obs/` identifiers of what it read as evidence, creates one work order per diagnosed fault, waits with `wait_for_notification` until the orders close and finishes. It reads only the tool answers and never the fault rule of the initial states, which stays with the sampler of initial states.

**Classifiers and split.** `TierClassifiers` in `CausalDiagnosis/environment/reference_policies/tier_classifiers.py` fits one linear discriminant analysis (LDA) per measurement tier and component on the served statistics of the tier's channels alone, the `served` list of the `statistics` table of the asset file (mean and std on `hyd_1`), so that no recorded statistic that a reading withholds enters a prediction, with standardized features, a pooled within class covariance shrunk by `SHRINKAGE` toward a scaled identity, and equal class priors. `stratified_folds` deals the 144 combinations into `N_FOLDS` folds stratified by level, and `cross_fitted_predictions` predicts the stable cycles of each fold with classifiers fitted on the other folds, refusing a split whose training set lacks a level. The split is by combination and not by cycle because a window can start anywhere in a block, so a split inside a block would put served cycles into the training set. `ServedClassifiers` keys the held out predictions by the values a reading serves (public names, mean and std at six significant digits) and refuses a table in which two cycles of a tier serve the same values, so the policy decides on tool answers exactly as on the recordings. A sensor group is read through the tier that serves the same channels.

**Observation model.** `observation_model` gives, for combination h, tier t and component c, the held out confusion row of the level of c in h, counted over every block except that of h, with one added to each count. The policy assumes:

- A1: given the combination, the predicted levels of the four components are independent.
- A2: given the combination, readings at different tiers are independent.
- A3: within one tier, predictions on cycles of the same block are exchangeable draws from a Dirichlet multinomial centered on the row, so the next prediction follows an urn around the row. The concentration is fitted per tier and component by maximum likelihood over `ALPHA_GRID`.
- A4: a reading equal to an earlier one at the same tier adds no information.

**Decision rule.** The expected information gain of a tier is the mutual information in bits between the hidden combination and the predicted levels of the next reading at that tier. The policy keeps a reserve for its orders, the work orders that its most probable combination needs, one order per faulty component at its part and repair charge, and the charge of the one visit that orders created one after another form. It submits the most probable combination when the largest belief reaches `confidence`, when the window starts repeating, when no group is affordable within the reserve, or when the best ratio of gain to price falls below `min_gain_per_price`. Otherwise it reads the group with the best ratio.

**Known limits.** A2 ignores the correlation between tiers, whose channels overlap, so the belief at submission can be overconfident, and the same cycle read at two tiers counts twice. The rule looks one step ahead and prices information in bits, not in expected repair cost. The observation rows leave out only the block of the combination they score: in an episode on combination g, the rows of other combinations that share a level with g still count the held out predictions on g, and the concentrations are fitted on all blocks. The CausalDiagnosis MCP of [causal-diagnosis-mcp.md](causal-diagnosis-mcp.md) uses a different observation model, the interventional effect model, and keeps a refit of these classifiers only as an ablation arm.

**Initial states.** An initial state is a combination, a window start and an entry offset, and every combination gives one window start, 0, and `BLOCK_CAP` offsets, whatever the length of its block. `start_states` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/start_state_seeds.py` lists every initial state that `reset` can draw, and `seed_table` gives each one the smallest seed that reaches it, so an episode from any initial state replays through the public `reset(seed)`. An anchored initial state fixes the state through `config.start` and needs no table.

---

## Verifier

`verify(log, released, oracle, cost_oracle, grade=None, scope="writes", predictions=None)` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/` returns the verdict of one episode as a plain dict that `json.dumps` accepts. It reads no answer text for a score.

A declaration of insufficient evidence is correct, and earns `success` and the credit of the reward, only when it cites a reading released before it and both the independent label and the E18 grade of the CausalDiagnosis MCP for the rules of the row mark the initial combination not diagnosable within the budget.

| Input | Content |
| --- | --- |
| `log` | `env.log`, with its hidden fields |
| `released` | the released evidence of the episode in format 3, written by the action server as `released.json` or composed in process by `released_from_env` |
| `oracle` | `env.oracle()`, with `reads` added for the scope `all`, the sensor reads per group of the greedy policy on the same seed and configuration |
| `cost_oracle` | the cost of the oracle policy's episode on the same seed and configuration, which normalizes the cost |
| `grade` | the diagnosability of the initial combination as `start_grade` in `CausalDiagnosis/environment/reference_policies/diagnosability_label.py` gives it, the independent label with a boolean `separable` and the E18 grade of the CausalDiagnosis MCP under `e18` with a boolean `separable`, or None for an initial combination outside the fault rule of the sampler, such as an anchored day. A grade of None counts as diagnosable within the budget under both, since it is no evidence that a declaration was right, while a drawn initial combination that the label table or the grade table lacks raises, and so does a grade without the E18 verdict |
| `scope` | `writes`, the default, or `all` |
| `predictions` | the records of `predictions.json` when the CausalDiagnosis MCP wrote one, for the `pred/` identifiers |

### Checks

The checks run in the order of the Agents Research Environments verifier, each a dict with `ok` and its evidence under `checks`:

| Check | Rule |
| --- | --- |
| `count_gate` | exactly one of `submit_diagnosis` and `declare_insufficient_evidence`, exactly one `finish`, and no write event outside the four write events, a refused action being no write. Under the scope `all` the read gate holds as well, so no sensor group is read more often than the greedy policy reads it on the same initial state, which `read_counts` computes from the greedy policy's own log, since the oracle policy knows the initial state and reads nothing. A read counts once whatever cycles it covers, since counting cycles would measure the tool shape of AssetOpsBench's history call and not the reading strategy, and a multi-cycle read still pays its price in the cost column. It counts only when it released a reading, so a free re-read and the listing and lookup calls of the AssetOpsBench rows stay out. The check reports `reads`, `reference_reads` and `read_gate` |
| `hard` | the submitted levels against `truth_at_write`, the hidden combination at the cycle of the write, component by component, the flag `faulty_set_right` of the table Scores, and the components of the agent's work orders against the oracle's set as sets, since the outcome of two repairs in the hidden state does not depend on their order and a repeated order after an ineffective or expired repair is a legal recovery |
| `timing` | every agent order on a component that degrades during the episode, one with a scheduled drop or a hazard drop in the log, closes by the window end of the run at the worst level open at its closure. The window end is the cycle at which the failure card would fail the component on that run, the oracle's `window_end` on the first run and the first cycle plus `patience` less one on a later run. A closure outside every run at the worst level, a closure in a run that opens after the component failed and every closure while the failure rule is off read the horizon. Orders on components without such a drop are not timed |
| `causality` | the first diagnosis or declaration precedes every work order in the log |
| `outcome` | `recovered` is the lasting combination equal to the healthy one. The lasting combination is the final combination of the end event, which on a row with a clock follows the run of a `finish` to the last close, with every component that an emergency hold still holds at its healthy level put back at the level the hold replaced. `held` lists the components of such holds, an empty list in every other episode, and `unnecessary` counts the orders whose hidden flag is set |
| `provenance` | every evidence identifier cited in the diagnosis, the declaration and the work orders, and every non empty `prediction_id`, exists among the identifiers the episode issued: `obs/{n}/{field}` for every released value, `wo/{id}` for every work order, `alarm/{t}/{channel}` for every alarm of a change summary or of a notification, `fmea/{source}/{code}` for the codes of every component whose failure modes were served, `hist/{n}` for every past order of a served maintenance history, and `pred/{n}` for every stored prediction |

Beside the checks, `worst_level_timing` gives the row of the timing check to every work order on a component that stands at its worst level, the last level of its degradation order in the asset file, at the start of the episode or at an event of the log, whether or not a drop degrades it during the episode, against the window end of its run at the worst level, the cycle at which the failure card would fail it and the oracle's `window_end` on the first run. An order closes no sooner than the lead time and the repair downtime that the asset declares after its creation. The rows enter neither pass, nor `all_checks_ok`, nor a score, and `scores.csv` reports their flag as `worst_level_timing_ok`.

### Failure classes

For an episode that did not succeed, or whose provenance check failed, the verdict names one of the twelve classes of the failure taxonomy, decided in this order, and `none` otherwise. A failure outside the taxonomy, such as a wrong commitment that no class describes, is `none` too.

| Class | Rule |
| --- | --- |
| `premature_diagnosis` | a diagnosis with no `read` event before it |
| `unmatched_evidence` | the provenance check failed |
| `unnecessary_replacement` | at least one order on a component that was healthy when the order was created |
| `repair_before_dependency` | at least one order closed ineffective (the hidden event `order_ineffective`) because a component it depends on was degraded from the creation of the order to its close, so a dependency that dropped after the order, by the hazard, the schedule or an expiry, is not charged (`_charged_ineffective` in `verifier/failure_class.py`) |
| `temporary_repair_expired` | an emergency repair expired before the end and no later order restored its component |
| `abstention_without_reading` | a declaration that cites no reading released before it, whatever the label says, since the agent decides that and an unknown label says nothing about it |
| `abstention_on_separable` | a declaration that cites such a reading while the label marks the initial combination diagnosable within the budget or the grade is unknown |
| `abstention_on_e18_separable` | a declaration that cites such a reading on an initial combination the label marks not diagnosable within the budget while the E18 grade of the CausalDiagnosis MCP marks it diagnosable within the budget |
| `summary_contradicts_log` | the closing summary claims a recovery the final combination does not show, the replacement of a component no agent order closed on, or that no alarm occurred while a released change summary or notification told of one |
| `repeated_malformed_call` | `REPEATED_REFUSALS` or more refused actions in the episode |
| `anchoring_on_first_anomaly` | the diagnosed faulty set equals the components that the first channel the agent saw outside its public normal range indicates, the channel furthest outside its normal range in normal range widths when one reading shows several, while the true faulty set differs |
| `recovery_without_diagnosis` | the rig recovered while the diagnosis is not all correct |

The two repair classes follow the unnecessary replacement, since all three are decided from the hidden fields of the agent's orders, and precede the summary check, which an agent that never saw a repair fail would trip as well when its summary claims the recovery.

The contradiction heuristic, reported under `summary_check` and marked as a heuristic, splits the closing summary into sentences, reads a claim that no alarm occurred before the negation filter, since such a claim is its own negation, skips the other sentences with a negation, reads a recovery claim when a sentence names the whole asset with a recovery word, and reads a replacement claim when a sentence names a component with a repair word. It is conservative on purpose, and the judge agreement study of [experiments.md](experiments.md) measures what it misses.

### Scores

| Field | Content |
| --- | --- |
| `diagnosis_per_component`, `diagnosis_all_correct` | the diagnosis against `truth_at_write`, the hidden combination at the cycle of its write, per component and in all, None without a diagnosis |
| `faulty_set_right` | whether the components that the diagnosis puts off their healthy level are exactly the components off their healthy level at the cycle of its write, whatever the levels, false without a diagnosis |
| `recovered`, `unnecessary` | the outcome in the hidden state, whether the levels that outlast the episode are the healthy ones, and the number of orders on a component that was healthy when the order was placed |
| `false_alarms` | the number of components at their healthy level at the cycle of the write that the first accepted diagnosis puts at another level, 0 without a diagnosis |
| `pass` | correct diagnosis and recovered and no unnecessary order, the pass of the metric tables |
| `graded_top1`, `graded_top1_n` | the share of the components faulty at the cycle of the write that the diagnosis names at their true level, 0 without a diagnosis or when the diagnosis puts a healthy component at another level, and the number of those components. With no faulty component at the write the share equals `diagnosis_all_correct` |
| `graded_pass`, `graded_pass_n` | the share of the components faulty at any time of the episode, at the start of the episode, at the write or after a drop, that are healthy at the end, where a component faulty at the write counts only when the diagnosis named it at its true level, 0 when the diagnosis puts a healthy component at another level or a work order was unnecessary, and the number of those components. With no faulty component at any time the share equals `pass`. On an initial state with at most one faulty component and no drop the two shares equal `diagnosis_all_correct` and `pass` |
| `separable`, `separable_e18`, `abstained`, `abstention_cites_reading`, `abstention_correct`, `commitment_on_inseparable` | the flag of the label and the flag of the E18 grade of the CausalDiagnosis MCP, whether the agent declared, whether the declaration cites a reading released before it, an `obs` identifier of a read entry that precedes it in the log, whether such a declaration falls on an initial combination both mark not diagnosable within the budget, and whether the agent committed on an initial combination the label marks not diagnosable within the budget, a flag that ablation En5 reads and that does not decide `success` |
| `success` | `pass` or `abstention_correct` in every episode, so a correct diagnosis with a recovered rig succeeds whatever the labels say, a declaration that cites a reading released before it succeeds on an initial combination the label and E18 both mark not diagnosable within the budget, and a wrong commitment, a declaration on an initial combination either marks diagnosable within the budget or on an ungraded initial combination and a declaration without such a reading do not |
| `read_gate` | under the scope `all`, whether every sensor group stayed within the sensor reads of the greedy policy on the same initial state, and None under the scope `writes`. It changes neither `pass` nor `success`, and `tables_from_runs/episode_fields.py` reads it for A7 and the En4 lines (`scope_pass`) |
| `cost`, `cost_unit`, `cost_prices`, `penalties`, `misdiagnosis_charge`, `downtime_total`, `cost_oracle`, `cost_normalized` | in money, the prices of the agent entries plus the penalties of the failure events and the misdiagnosis charge plus the downtime of the end event, and in calls, on a row where every price is zero, the number of tool calls, then the unit, the parts, the oracle's cost in the same unit and their ratio, None when the oracle's cost is unknown or zero, and the reward is None with it. `misdiagnosis_charge`, present only when the log holds a charge, is the part of `penalties` that a wrong diagnosis cost, and a table reads a verdict without it as no charge through `misdiagnosis_charge_of` |
| `detection_delay` | per component, the time to work order of the paper, the largest over its fault occurrences of the cycles from the onset of the occurrence to the agent's first order on the component at or after the onset and before the occurrence ends, the onset being the cycle at which the hidden record takes the component off its healthy level, by a deterioration transition, a scheduled or hazard drop, the expiry of an emergency repair or a new fault after a restoring close, and cycle zero for a fault present at the start (`fault_occurrences`, `fault_onset`), so an order placed before the fault does not count, None for a component that never left its healthy level or got no such order |
| `repair_delay` | per component, the time to repair of the paper, the largest over its fault occurrences that an agent close ended of the cycles from the onset of the occurrence, the one onset of `detection_delay`, to that close, a close that takes the component from below its healthy level to it, None for a component that never left its healthy level or whose occurrences no agent close ended |
| `unrestored` | the components with a fault occurrence that no restoring close of the agent ended, in the order of the components, empty when every fault was restored or none occurred; the reward then takes the delay at the horizon |
| `turns_to_correct_diagnosis` | the step of the diagnosis when it is all correct, else None |
| `prediction_verdict` | per order that names a prediction: whether the prediction exists, the recovery the CausalDiagnosis MCP predicted, its signed error against the recorded recovery, and the verification that the CausalDiagnosis MCP stored, if any |
| `end_reason`, `time_used`, `steps`, `refusals`, `failures` | the end event's reason and time, the environment steps, the refused actions and the components that failed |
| `first_anomaly`, `summary_check`, `hidden_start`, `write_t`, `truth_at_write`, `final`, `grade`, `scope`, `all_checks_ok`, `verified` | the first channel seen outside its normal range, the one furthest outside within a reading, with its source, the heuristic's reading, the hidden initial state, the cycle of the diagnosis and the hidden combination at that cycle, the final combination, the inputs and the overall flags |

`reward(verdict, horizon, w_c=0.2, w_t=0.1)` is the terminal reward of policy learning, an outcome term of one for a `pass`, `ABSTENTION_CREDIT` (one half) for a correct declaration and zero otherwise, minus `w_c` times the normalized cost up to `COST_CAP`, minus `w_t` times the largest detection or repair delay over the horizon up to one, the delay taken at the horizon when a fault occurrence of the episode was never restored by an agent close (`unrestored`). For nonnegative weights, each penalty then lies between 0 and `w_c * COST_CAP + w_t`, so while that sum stays below the smallest gap between two outcome levels, `outcome_gap`, a pass earns more than a correct declaration and a correct declaration more than a wrong answer whatever either episode cost, and `reward` refuses weights that break the bound or are negative. On the rig the default weights and the cap of 1.5 give 0.4, below the gap of one half. A provenance failure, a cited identifier that the released evidence does not hold, scores as a failed episode, with the outcome term at 0 and the cost and delay terms kept, so its reward never exceeds that of an honest failure at the same cost, and the reward is None when the verifier could not run or the cost of the oracle is unknown. The credit keeps the order of a repair, then a declaration after reading, then a wrong answer, so a policy that knows both labels earns no more than the oracle, and a wrong commitment earns no outcome in any episode.

`pass_power_k(successes, n, k)` is the unbiased estimator of pass^k of tau bench: for a task solved c times in n trials, the number of ways to choose k successes among the c over the number of ways to choose k trials among the n, and the mean of that quantity over tasks when `successes` is a sequence.

`released_from_env(env)` composes the released evidence of an in process episode, the equivalent of the action server's `released.json`, from the asset profile as the rules, the public events and the observations of the reads. `read_counts(log)` counts the sensor reads of a log per sensor group, each read that released a reading once.

The verifier refuses to run, with `verified` false and an `error`, on a log without a reset event, a log without an end event, since an episode is verified once it has ended, a log whose hidden fields were stripped, a log whose reset entry holds no prices, since the unit of its cost comes from them, a work order without its hidden flag, an oracle without its diagnosis or its work orders, an oracle whose diagnosis differs from the log's hidden initial state, and released evidence that is not format 3 or lacks its events or its observations. A missing grade is not an error, while a grade without its E18 verdict is.

---

## Gym and OpenEnv

`HydraulicRigGym(config, facts_path, seed_offset=0, label_of=start_grade, starts="development")` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/gymnasium_wrapper.py` is the environment for policy learning with text observations, and `label_of`, by default `start_grade` of `CausalDiagnosis/environment/reference_policies/diagnosability_label.py`, gives the verifier the diagnosability of every initial combination, the independent label with the E18 grade, which reads the models of the CausalDiagnosis MCP. `gymnasium` is imported lazily: with it installed the class is a `gymnasium.Env` with `Text` observation and action spaces, without it a stand in base class with the same methods. The package group `rl` of `pyproject.toml` installs `gymnasium` 1.0.0 and `openenv-core` 0.3.0 (`uv sync --group rl`), and a plain `uv sync` installs neither. The newest FastAPI that `openenv-core` accepts requires a later OpenTelemetry than the environment of the runs of record holds, so `[tool.uv] constraint-dependencies` keeps the nine OpenTelemetry pins of those runs and the lock resolves an older FastAPI for the group; the test `test_rl_group_lock.py` fails when a lock run moves any pin outside the group. The constraints may be lifted once the runs of record are done.

- The gym plays the development starts of its asset by default (`development_starts` of `CausalDiagnosis/datasets/splits/development_split.py`), within `config.allowed` when that is set, so a trainer never meets a start of the test set. A start fixed in `config.start` must be a development start, and an asset whose recordings admit no development split has none, so its gym needs `starts="test"`.
- `starts="test"` plays the configuration's own start set, the test set of a preset, in any run.
- `reset(seed=None, options=None)` seeds `np_random` through `gymnasium.Env.reset` when `gymnasium` is installed, as its environment checker asks, and returns the briefing followed by the asset profile as JSON, and an info dict with the effective seed, which is the seed plus `seed_offset`. Without a seed the instance plays consecutive seeds from the offset, as a training loop that calls `reset()` expects. At every reset the oracle policy plays the same seed on a second environment instance, and the verifier's cost of that episode becomes the `cost_oracle` of the verdict.
- `step(action)` takes a dict or a JSON string of an action of this page and returns the answer envelope as JSON, the reward, the `terminated` and `truncated` flags and an info dict with the cost, the time and the step count. Anything else, unparsable text included, becomes an action of type `malformed`, which the environment refuses, logs and counts, so a policy that emits bad text loses a step and never raises into the trainer. The reward is 0.0 until the terminal step, then the verifier's reward with the diagnosability of the initial combination as `grade`, by default `start_grade` of `CausalDiagnosis/environment/reference_policies/diagnosability_label.py`, whose label and grade tables are computed once per rules key, and the configuration's `verifier_scope`, and the info dict then carries `end_reason`, `verdict`, `reward` and `cost_oracle`. `terminated` is true for an end by `finish`, `budget` or `failure`, `truncated` for an end by `horizon` or `max_steps`. A verifier that could not run gives a truncation with 0.0 in the tuple and None in the info dict, so a trainer can drop the episode instead of scoring it zero.
- The attribute `observation` keeps the last answer dict of the environment, so that an in process policy such as `OraclePolicy(gym.env)` or `DoNothingPolicy()` acts on the same episode as a text policy through `policy.act(gym.observation)`. The attribute `env` is the wrapped environment, whose log and oracle are hidden state for the verifier and never for the policy, as are `info["verdict"]` and `info["cost"]`, whose cost of a step holds the downtime of the hidden condition.
- The gym charges no thinking time: it calls `step` with `think_cycles` 0, so the `generation_time` and `generated_tokens` modes behave as `declared` under the gym. The cost of a step is reported in the info dict and never subtracted from the reward, so that the outcome term stays the signal.

`HydraulicRigOpenEnv(config=None, facts_path=None, seed_offset=0, starts="development")` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/openenv_wrapper.py` puts the gym behind the `reset`, `step` and `state` interface of the OpenEnv core, imported lazily from `openenv.core`, with pydantic stand ins of the same fields when the package is absent. `HydraulicRigAction(tool_name, arguments)` has the shape of the core's tool call action: the tool name is the action type and the arguments are the action's argument dict. `HydraulicRigObservation(text, done, reward, metadata)` carries the text of the gym, the reward (None at reset, 0.0 before the end, the verifier's reward at the end, None when the verifier could not run) and the time in `metadata`, which carries no cost, since the cost of a step holds the downtime of the hidden condition. `HydraulicRigState(episode_id, step_count, seed, time, done, end_reason, verdict)` is the bookkeeping of the orchestrator, with the verdict and its ground truth set at the end and never in an observation. The default episode id names the seed, so that a replay of a seed reports the same state. The arguments default to `EpisodeConfig()` and `default_facts_path(config)` of `CausalDiagnosis/environment/Causal_Diagnosis_Environment/environment_factory.py`, the data file of the configured asset, because the OpenEnv server builds one instance per session without arguments, so a session plays the development starts, `starts` passes to the gym, and `SUPPORTS_CONCURRENT_SESSIONS` is true. In production mode the same actions are the MCP tools of the action server, served without reward, step count or done flag, where the server advertises flat parameters for the Plan-Execute argument filler while the OpenEnv action carries the environment's own argument dict.

`CausalDiagnosis/examples/run_one_episode.py` plays the oracle policy and the do nothing policy from one seed through the gym and prints which policy earned the higher reward, the oracle and baseline check of OpenEnv's RFC 008, and no value:

```bash
PATH=$PWD/.venv/bin:$PATH python -m CausalDiagnosis examples run-one-episode --seed 0
```

---

## Programmatic use

Run from the fork root with the data file built and `src/` on the import path, as in `PATH=$PWD/.venv/bin:$PATH PYTHONPATH=src python`:

```python
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import build_blocks, default_facts_path, load_cycles
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.gymnasium_wrapper import oracle_cost
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import released_from_env, reward, verify

cycles = load_cycles(default_facts_path())
blocks = build_blocks(cycles)
config = EpisodeConfig()  # the default parameters, static episode
env = HydraulicRigEnv(cycles, blocks, config)

envelope = env.reset(0)                                   # the result is the asset profile
envelope, cost, done, info = env.step({"type": "read_sensors", "sensor_group": "tier1", "cycles": 2})
served = envelope["result"]["served"]                     # one entry per cycle the rig ran, with its values
guess = {"cooler": 100, "valve": 100, "pump": 1, "accumulator": 130}
envelope, cost, done, info = env.step({"type": "submit_diagnosis", "diagnosis": guess, "evidence_ids": [f"obs/{s['n']}/{f}" for s in served for f in s["values"]]})
envelope, cost, done, info = env.step({"type": "create_work_order", "component": "pump", "prediction_id": "", "evidence_ids": []})
close_t = envelope["result"]["work_order"]["expected_close_t"]
while env.time < close_t:                                 # no tool sets the time, so the agent waits until the order closes
    envelope, cost, done, info = env.step({"type": "wait_for_notification", "timeout": close_t - env.time})
envelope, cost, done, info = env.step({"type": "finish", "summary": "Replaced the pump."})

cost_oracle = oracle_cost(HydraulicRigEnv(cycles, blocks, config), 0)  # the oracle policy on the same seed, as the drivers play it
verdict = verify(env.log, released_from_env(env), env.oracle(), cost_oracle)
terminal_reward = reward(verdict, config.horizon)          # keep env.log, the oracle and the verdict away from agents
```

---

## Known limits

- The diagnosis is scored against the hidden combination at the cycle of its write, so in a timed episode whose schedule drops a component before the write the correct diagnosis names the level at the write, and a drop after the write changes nothing the hard check reads.
- A read that crosses the horizon serves only the cycles before it and is charged for those alone, since the clock never passes the horizon.
- Alarms of the change summary are free and take no time once a sensor group has been read, so after one read of a group the agent sees for free every cycle since the given cycle in which a channel mean of that group left its normal range, the cycles it never read included. The normal range of a channel mean runs from its smallest to its largest value over the stable cycles of the healthy combination.
- The greedy baseline policy counts the same cycle read at two tiers twice (assumptions A2 and A4), and its observation rows keep the leakage described above. The CausalDiagnosis MCP merges such readings into one observation.
- The independent label plays the live recordings only where a clock runs on the row, and on a row without a clock it labels an initial combination on the record before cycle 0 alone. A row without a clock that lists a wait could still be made to play live recordings, which no registered row does, and both labels describe its record before any wait.
- The gym charges no thinking time, and `info["verdict"]` and `info["cost"]` hold the ground truth, so a harness that shows the info dict to the policy would leak.
- The repair rule checks the dependencies at the close, so a dependency that drops during a visit makes the repair of its dependent ineffective. In an episode in which the dependent waits at its worst level while a dependency drops, the only orders that take come after the dependency's last drop, and the dependent fails meanwhile.
- The dependencies, the priorities, the emergency option, the visit charge, the fix kinds, the maintenance history and the rates of the hazard are design parameters of the asset file.
- Under the stochastic hazard at the rates of the asset file the oracle policy passes on the seeds of `test_the_oracle_passes_in_the_stochastic_mode` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/tests/test_hazard.py`, and the oracle, which reads the drop of a healthy dependency from the table of uniforms, orders the dependent only after that drop. A component that the hazard drops again after the last order of its visit closes, while the oracle still waits for another drop, can still make the oracle order it twice.
- Under the stochastic hazard a dependency that drops between an order and its close makes the order ineffective with no error of the agent, and the verifier does not charge the class `repair_before_dependency` for it, since the class fires only when a blocking dependency was degraded from the creation of the order to its close (`_charged_ineffective` in `verifier/failure_class.py`).

---

## Tests

```bash
PATH=$PWD/.venv/bin:$PATH python -m pytest CausalDiagnosis/environment/Causal_Diagnosis_Environment -q
PATH=$PWD/.venv/bin:$PATH python -m pytest CausalDiagnosis/datasets/hyd_1/tests/test_verify_data_facts.py CausalDiagnosis/datasets/hyd_1/tests/test_build_features_and_blocks.py CausalDiagnosis/datasets/hyd_1/tests/test_download_and_build.py -q
```

| Test file | Covers |
| --- | --- |
| `CausalDiagnosis/environment/Causal_Diagnosis_Environment/tests/test_episode_config.py` | the defaults validate and round trip through JSON, unknown keys are refused, every problem is listed at once, the default dependencies and the refusal of a dependency with a larger priority number, the degradation order, the anchored dates |
| `CausalDiagnosis/environment/Causal_Diagnosis_Environment/tests/test_layers.py` | the registry of the six big layers and the small switches of each, every switch set alone and off alone, the switches that need another, the compositions and their refusals, the time tools and the repair manual by their switches, and every switch of a layer off giving the row below byte for byte |
| `CausalDiagnosis/environment/Causal_Diagnosis_Environment/tests/test_episode_clock.py` | a scheduled drop switches the served block, a work order executes after the lead time, stops the rig and closes with the repaired block, reads during the repair downtime, downtime accrual, the failure rule, the oracle's set and window ends |
| `CausalDiagnosis/environment/Causal_Diagnosis_Environment/tests/test_one_deadline.py` | an order open at the window end that no longer averts the failure, a repair closed by the window end that averts it and meets the timing check, an ineffective close that leaves the count running, the oracle that orders before the last drop when an order after it would close late, the sentence of the failure card on both surfaces, the oracle that plans its visits on a development start where one visit would end with a failure, and a repair that ends no run at the worst level and meets the timing check on development starts |
| `CausalDiagnosis/environment/Causal_Diagnosis_Environment/tests/test_env.py` | replay byte for byte, the hidden state from the seed only, the envelope and the profile, the information barrier, reads cycle by cycle, windows, a later window that shares the entry offset and draws nothing, one initial state per entry offset, the refusal of a combination with too few recordings, normal ranges, every refusal, the write lock, the budget, the horizon, the step cap, finish, the end entry, the change summary, the failure modes, the clock modes `declared`, `generation_time` and `instant` |
| `CausalDiagnosis/environment/Causal_Diagnosis_Environment/tests/test_open_order_refusal.py` | the refusal of a second order on a component whose order is open names the close that the answer lists, on the clocked presets and on the row without a clock whose repair card holds the order, the close after a later order of the visit moved it, and as controls a first order, an order on another component and an order after the close answered as before, and a second order accepted on the base row, whose orders close inside their call |
| `CausalDiagnosis/environment/Causal_Diagnosis_Environment/tests/test_policies.py` | the oracle recovers every candidate initial combination in one visit, follows the schedule with the repair rule on, finishes before a drop it could not repair and before a drop that would come after its orders close, the do nothing policy, the declaring policy that reads the cheapest group once and cites that reading, the random policy, the greedy policy |
| `CausalDiagnosis/environment/Causal_Diagnosis_Environment/tests/test_visits.py` | consecutive orders form one visit whatever the thinking between them on both thinking clocks and the declared one, a read or a refused call between two orders opens a second visit, an order after thinking joins with its own lead time, a late order is repaired after the repair begun, orders of one cycle keep the schedule of one cycle |
| `CausalDiagnosis/environment/Causal_Diagnosis_Environment/tests/test_repairs.py` | an order before its dependency closes ineffective and fires its class, the oracle passes with the dependencies on, one visit per cycle with one charge and closes in priority order, two visits in two cycles, the mode refusal, the maintenance history from its own generator and its rule, the manual and the history as free reads, `hist` identifiers in the provenance check, the emergency lead, surcharge, hold, expiry and class |
| `CausalDiagnosis/environment/Causal_Diagnosis_Environment/tests/test_greedy_policy.py` | the linear algebra of the classifiers on hand computed cases, the stratified folds, the cross fit, the observation model, the information gain, the served value lookup, the features restricted to the served statistics |
| `CausalDiagnosis/environment/Causal_Diagnosis_Environment/tests/test_hazard.py` | the same seed with the same actions replays the stochastic mode, another action sequence meets another drop on a seed constructed from the hazard table, the oracle passes in the stochastic mode, an order that the hazard drop of a dependency made ineffective is not charged and the oracle, which reads that drop from the table, orders after it, the oracle places a visit a cycle later when the table would drop a repaired component before the visit ends, and a collateral of 0 or 1 after the replacement of a healthy component |
| `CausalDiagnosis/environment/Causal_Diagnosis_Environment/tests/test_start_state_seeds.py` | the initial combinations of seeds 0 to 19 against the recorded table, the seed table lists every initial state once and every seed replays its initial state, the first replicate of the evaluation set as the draw of the master seed 42 but on a test combination where that draw replays the development episode, five distinct initial states per test combination in replicate order, no replicate on a rollout seed of the diagnosability label or on the initial state of the development episode, the six test combinations whose first replicate moved off its development episode, the rollout seeds of a configuration that allows fewer initial combinations than its fault rule as the label table records them, and the training initial states, every single fault initial state that no replicate plays |
| `CausalDiagnosis/environment/Causal_Diagnosis_Environment/tests/test_verifier.py` | every check, every failure class, the success rule on a correct commitment, a correct declaration and a wrong commitment on an initial combination the label marks not diagnosable within the budget and a declaration on an initial combination diagnosable within the budget, a declaration on an initial combination only E18 marks diagnosable within the budget and a grade without E18, a declaration without a reading released before it under every label, an unknown citation alone and beside a reading, the delays, the cost, the reward and pass^k on synthetic logs, the window of each run at the worst level for the timing check, the scope `all` with the read gate failing an agent that reads more of a group than the greedy policy and holding one that reads less, the unverifiable inputs, and one integration test that plays the oracle policy and the do nothing policy on the environment |
| `CausalDiagnosis/environment/Causal_Diagnosis_Environment/tests/test_misdiagnosis.py` | a wrong diagnosis on a priced preset charged once on the hidden record, every reply and the release unchanged by the charge, no charge for a true diagnosis, a declaration, no write or the oracle, a fault diagnosed on a healthy initial state charged on a priced row, the base row that charges nothing and leaves its verdict keys unchanged, both engines charging the same, a misdiagnosis price under a cost in calls refused, the truth at the write, and the briefing sentence of the charge with no text that calls the price uncharged |
| `CausalDiagnosis/datasets/tests/test_misdiagnosis_price.py` | the charge that reads the misdiagnosis price of the asset file, on the tiny asset |
| `CausalDiagnosis/environment/Causal_Diagnosis_Environment/tests/test_budget_ledger.py` | the budget counter that falls by the prices paid alone, the hidden record of the downtime and the failure penalties in the end entry, the verifier's cost as the prices plus that record, the budget stop on prices alone, and no agent facing view that carries a number of the record |
| `CausalDiagnosis/environment/Causal_Diagnosis_Environment/tests/test_graded_verdict.py` | the graded fields of the verdict with their denominators, half credit on two faults, the zero after a false alarm or an unnecessary order, a fault that appears during the episode, the exact match of the faulty set, and a diagnosis scored at the cycle of its submission in every clock mode |
| `CausalDiagnosis/environment/Causal_Diagnosis_Environment/tests/test_healthy_start.py` | the fault rule from zero faulty components, the control preset on the healthy combination at one entry offset per replicate, the pass of an agent that orders nothing, every work order counted as unnecessary, the false alarms in any episode, every baseline policy on the control, the briefing of the control equal to that of the base preset on both surfaces, no fold with an initial combination without a fault, and the undefined normalized cost where the oracle pays nothing |
| `CausalDiagnosis/environment/Causal_Diagnosis_Environment/tests/test_leaks.py` | a change summary that lists alarms only for the channels of the groups a charged read named, and the closing date of a past work order that tells nothing about whether its component is healthy |
| `CausalDiagnosis/frameworks/AssetOpsBench/tool_servers/tests/test_recorded_history_env.py` | the engine of the AssetOpsBench rows, the prehistory and the live cycles of the two segments of an initial combination, reads over the record, a free re-read and the charge of an uncovered group, the rows without a clock and those on the real-time and the advanced real-time clock, the order that closes in its call where no clock runs, a timed read priced by its dry run, the free AssetOpsBench read, the refusals, the replay of the stored actions, and the cost of the oracle policy equal on both engines on every preset |
| `CausalDiagnosis/frameworks/AssetOpsBench/tool_servers/tests/test_open_order_refusal_receipt.py` | the same refusal through the metering layer on the repair rules row, as the error of `generate_work_order` that the designs of the AssetOpsBench rows read and as the reason of the call's entry, and as a control a second order accepted on the base row |
| `CausalDiagnosis/environment/Causal_Diagnosis_Environment/tests/test_reward_cap.py` | the cap of the cost term of the reward, a higher outcome level that earns more whatever either episode cost, and the refusal of weights that break the bound |
| `CausalDiagnosis/environment/Causal_Diagnosis_Environment/tests/test_scripted_policies.py` | two scripted policies that read the hidden state as the oracle policy does, on the rows with visits and an emergency card or drops, each reading one cycle before its diagnosis, the path of emergency orders and the finish, which never passes and earns less than the oracle policy on every seed, and the path of one visit after the drops, verified on every seed and passing wherever every order closes and no level drops at or after its first order |
| `CausalDiagnosis/environment/Causal_Diagnosis_Environment/tests/test_diagnosability_label.py` | the diagnosability label, a table that is deterministic and keyed by the rules, the observation models that read every level or none, the gym and the verifier that read the label, the rollout seeds after the evaluation pool, the prehistory that each rollout reads, the label file under two writers, a table computed again when the code that wrote it differs, a code digest that skips the folders of other assets, the first diagnosis found by its event, the label table of the control, a row without a clock labeled on the record before cycle 0 alone, and a key of the label that tells a row with a clock from one without |
| `CausalDiagnosis/environment/Causal_Diagnosis_Environment/tests/test_time_tools.py` | the two time tools, a wait to its timeout, to a close and to the horizon, the current time that takes no cycle, every public event notified once and no hidden one, notifications from the thinking time, alarms only for the sensor groups the agent paid for, the clock that the contract and the profile declare, and the off values of the emergency and failure cards and of the budget |
| `CausalDiagnosis/environment/Causal_Diagnosis_Environment/tests/test_token_mode.py` | the rate of the asset file in the `generated_tokens` mode and none in every other mode, the refusal of a configuration of that mode without a rate, the clock table of an asset file that may hold the rate beside its own keys, the profile, the clock block and the time tool of an episode of that mode, no word of that mode in a preset of another mode, and the failure switch that scales the patience of the asset file |
| `CausalDiagnosis/environment/Causal_Diagnosis_Environment/tests/test_training_states.py` | the registered split of the rig, a model named by the fold it holds out, the training combinations of each model, no training initial state one repair away from a combination it is judged on, the training states at offsets that no test replicate plays, the training seeds and their stream, the replicate seeds of each fold, and the development combinations |
| `CausalDiagnosis/environment/Causal_Diagnosis_Environment/tests/test_gymnasium_wrapper.py`, `CausalDiagnosis/environment/Causal_Diagnosis_Environment/tests/test_openenv_wrapper.py` | determinism, the reward order, the credit of a declaration after reading below the outcome term of a correct commitment on an initial combination both labels mark not diagnosable within the budget and no credit where E18 marks it diagnosable within the budget, truncation at the horizon and at the step cap, malformed text, no hidden field in any observation text or in the released evidence, import and run without gymnasium and without openenv, the OpenEnv state contract |
| `CausalDiagnosis/datasets/hyd_1/tests/test_verify_data_facts.py`, `CausalDiagnosis/datasets/hyd_1/tests/test_build_features_and_blocks.py`, `CausalDiagnosis/datasets/hyd_1/tests/test_download_and_build.py` | every check of `verify_data_facts.py` and the environment's blocks against its recomputation, `build_features_and_blocks.py` on synthetic raw files, its check against the pinned checksums and its cross check, `download_and_build.py` from an archive on the machine and without the network |

A test session stops before its first test when the data file is missing or is not the pinned one. A clone that holds no data states it with `ASSETS_WITHOUT_DATA=1`, and the tests that need the recordings are then skipped, so a run under that switch does not exercise the environment. The two tests that need gymnasium or openenv installed are skipped without the package. No test states a value computed from the rig data: the assertions are relations and equalities.
