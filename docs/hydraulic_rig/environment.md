# Hydraulic Rig Environment

`src/envs/hydraulic_rig/` turns the public UCI hydraulic rig recordings into episodes of finding the fault and repairing the asset. This page covers the data, the episode interface, the actions and their prices, the scores, the scenario classes, and the reference policies. The environment uses the Python standard library only and simulates nothing.

Related docs: [README.md](README.md) for the index of these pages, [action-server.md](action-server.md) for the MCP tools that serve an episode, [causal-diagnosis-server.md](causal-diagnosis-server.md) for the diagnosis tools, [benchmarks.md](benchmarks.md) for the drivers and the studies, and [../mcp-servers.md](../mcp-servers.md) for every MCP server of the repository.

---

## Overview

Every episode hides one combination of the condition levels of four components: the cooler, the valve, the pump, and the accumulator. The agent pays to read sensors at three measurement tiers, submits a diagnosis first, and replaces components afterward. After a replacement the environment continues with the real recordings of the repaired combination. The score comes from the log alone: whether the diagnosis was right, whether the asset recovered, and what was spent.

The environment, its scorer, and its reference policies live in `src/envs/hydraulic_rig/`. The scripts that fetch and check the data, run the reference policies, and compute the scenario anchors live in `benchmarks/hydraulic_rig/`.

---

## Data

The recordings are dataset 447 of the UCI Machine Learning Repository, "Condition monitoring of hydraulic systems" by N. Helwig, E. Pignanelli, and A. Schütze, DOI 10.24432/C5CW21, under the CC BY 4.0 license. The rig repeats a constant load cycle of 60 s and records 17 sensor channels and one row of condition labels per cycle. The rig and its faults are described in N. Helwig, E. Pignanelli, and A. Schütze, "Condition Monitoring of a Complex Hydraulic System using Multivariate Statistics," Proc. IEEE I2MTC, 2015, pp. 210-215, DOI 10.1109/I2MTC.2015.7151267.

| Component | Label | Levels | Healthy level |
| --- | --- | --- | --- |
| cooler | `cooler_condition` | 3, 20, 100 (%) | 100 |
| valve | `valve_condition` | 73, 80, 90, 100 (%) | 100 |
| pump | `pump_leakage` | 0, 1, 2 | 0 |
| accumulator | `accumulator_pressure` | 90, 100, 115, 130 (bar) | 130 |

The label row also holds `stable_flag`, which is 0 when the conditions were stable and 1 when static conditions might not have been reached yet. The four level sets give 144 combinations, and `HEALTHY_COMBO`, the combination with every component healthy, is `(100, 100, 0, 130)`.

### Local files

| File | Origin | Contents |
| --- | --- | --- |
| `hyd_facts.jsonl` | the IndustryAssetEQA facts file of the public EmbodiedQA repository, at commit `d7f42a703751cba077d99bae34e8c5fa4bcfa9fe` | one record per cycle with `cycle_index`, `labels`, and `features`: nine statistics per channel and three features that combine channels |
| `src/couchdb/scenarios_data/shared/iot/hydraulic_pump_1.json` | the AssetOpsBench asset `hyd_1` | the 17 channel means and a timestamp per cycle, without labels |

The environment reads the facts file only and serves four of its nine statistics, the mean, std, min, and max of each channel, under names such as `Pressure_PS1_mean`. The `hyd_1` file serves as a cross-check. Its field names carry the unit and the sampling rate, such as `PS1_Pressure_bar_100Hz`, and the action server renames the served summaries after them.

| Variable | Default | Read by |
| --- | --- | --- |
| `HYD_FACTS_PATH` | `src/envs/hydraulic_rig/data/hyd_cycles.jsonl`, a generated file | `default_facts_path`, which the tests, the other scripts, the drivers, and the model fit of the causal diagnosis server call, and `verify_data_facts.py`, which reads the variable itself |

The fork ships neither the facts file nor the generated release file. `DATA_DIR` in `src/envs/hydraulic_rig/data.py` names the folder `src/envs/hydraulic_rig/data/`, which git ignores except `src/envs/hydraulic_rig/data/CHECKSUMS.txt`.

```bash
# 1. fetch the facts file at a fixed commit
mkdir -p ~/rig-data
curl -L -o ~/rig-data/hyd_facts.jsonl https://raw.githubusercontent.com/ChathurangiShyalika/EmbodiedQA/d7f42a703751cba077d99bae34e8c5fa4bcfa9fe/data/outputs/hydraulic/hyd_facts.jsonl

# 2. point the environment at it
export HYD_FACTS_PATH=~/rig-data/hyd_facts.jsonl

# 3. check both data files
uv run python benchmarks/hydraulic_rig/verify_data_facts.py
```

The facts file has 30,185,743 bytes and the SHA-256 `7912b1b641254d54daffa40ea2fe84c64dedf50edb2d996951ad1f3d9a433e47`. The `hyd_1` file has 1,880,617 bytes and the SHA-256 `622bcd1a668a939a2590b1aaf560ecce641731b04d802271b39ef454835918b5`. The EmbodiedQA repository has no license file, so the facts file serves only as a development input and a cross-check, and the fork redistributes neither the file nor anything derived from it.

### Release path

The release path rebuilds the labels and the served statistics of every cycle from the CC BY 4.0 archive alone.

```bash
# 1. download the archive into an empty folder and record its SHA-256
uv run python benchmarks/hydraulic_rig/fetch_uci.py --download

# 2. compare the recomputed labels and statistics with the facts file
uv run python benchmarks/hydraulic_rig/build_blocks.py <raw folder printed by step 1> --check

# 3. write the records and the blocks with the convention that step 2 reported
uv run python benchmarks/hydraulic_rig/build_blocks.py <raw folder printed by step 1> --std-ddof <0 or 1>
```

`fetch_uci.py` downloads only with `--download` and only into an empty folder, by default `src/envs/hydraulic_rig/data/uci_hydraulic/`. It records the archive's SHA-256 in `src/envs/hydraulic_rig/data/CHECKSUMS.txt` and refuses an archive whose hash differs from the one recorded under the same name, or from `EXPECTED_SHA256` when that constant is set. `build_blocks.py --check` writes nothing and reports whether the facts file computes the standard deviation with ddof 0 or 1. A run with `--std-ddof` writes `src/envs/hydraulic_rig/data/hyd_cycles.jsonl`, the default of `default_facts_path`, and `src/envs/hydraulic_rig/data/blocks.json`, the stable block of every combination, unless `--out-dir` names another folder.

### Data checks

`verify_data_facts.py` reads the two files and writes nothing. It does not import the environment, so the tests can compare its blocks with those of `build_blocks`. Apart from the two hashes, each check is a relation that holds whatever the counts:

- Both files hold the same number of cycles, numbered without gaps from 0 in the facts file and from 1 in `hyd_1`.
- Every facts record has a cycle of 60 s, the 17 sensors, one label row, the feature names of every other record, and provenance to the UCI files.
- `hyd_1` cycle c matches facts record c - 1 on all 17 channel means within 1e-9, and a pairing shifted by one cycle agrees on no PS1 mean.
- The labels take exactly the levels above, the facts file marks exactly the healthy levels as `cooler_ok`, `valve_ok`, `pump_ok`, and `accumulator_ok`, and the valve is at 100 in every cycle flagged 1.
- All 144 combinations occur, each with stable cycles that form one contiguous block.

---

## Episode

`load_cycles` reads one `Cycle` per record, and `build_blocks` groups the stable cycles of each combination into one block. Cycles flagged as not stable are never served.

When an episode enters a combination, the environment draws a window of `block_cap` consecutive cycles, 10 by default, from its block and an entry offset inside the window. Every combination is served through a window of the same length, so the number of distinct cycles reveals nothing about the combination. Each measurement, at any tier, returns the next cycle of the window, wrapping around inside it. After `block_cap` measurements, every further measurement repeats a cycle and carries `repeated` set to true. A replacement that changes the combination draws a new window and offset, and a replacement of a healthy component leaves both unchanged. An agent that has used up the window can tell a replacement of a healthy component, after which the cycles keep repeating, from one of a faulty component, which opens a new window.

`TinyHydraulicEnv(cycles, blocks)` takes the output of `load_cycles` and `build_blocks` and these keyword arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `prices` | `DEFAULT_PRICES` | the price table, checked at construction |
| `budget` | 150 | the total cost an episode may reach |
| `max_repairs` | 2 | replacements allowed after the diagnosis, where 0 ends the episode at the submission |
| `min_faulty`, `max_faulty` | 1, 2 | the rule on the number of components off their healthy level at the start |
| `max_steps` | 200 | the step cap, refusals included |
| `block_cap` | 10 | the cycles in a served window |
| `allowed` | `None` | an optional set of start combinations, such as one side of a split |

The constructor raises `ValueError` unless 1 <= `min_faulty` <= `max_faulty`. The start candidates are the combinations within the rule, restricted to `allowed` when it is given: 47 by default, the 10 combinations with one faulty component and the 37 with two.

`reset(seed)` seeds the episode's own `random.Random`, draws the hidden combination uniformly from the candidates, and returns the first observation. Every later draw, such as a window after a replacement, comes from the same generator, so a seed and an action sequence reproduce the log exactly.

`step(action)` takes a dict with a `type` key and returns `(observation, cost, done, info)`. A step after the end raises `RuntimeError`, and an action that is not a dict raises `TypeError`. Any other malformed or out-of-order action is refused: it is logged with a reason, costs nothing, changes no state, and counts as a step. Apart from the step cap, only a refusal for lack of budget ends the episode.

An episode has two phases in a fixed order, so that repair outcomes cannot be used to back out the diagnosis:

1. `investigate`: the agent measures and then submits a diagnosis.
2. `repair`: the agent may measure, replace up to `max_repairs` components, and finish.

| End reason | When |
| --- | --- |
| `finish` | the agent finishes in the repair phase |
| `recovered` | a replacement makes every component healthy |
| `repairs used up` | the replacements reach `max_repairs` without recovery |
| `budget` | an action would take the cost above `budget`, so it is refused |
| `no repair phase` | the agent submits while `max_repairs` is 0 |
| `step limit` | the step count reaches `max_steps` on a step that does not end the episode for another reason |

The action that reaches the step cap is still carried out and charged, and an agent that never submits ends there unsubmitted.

| Observation field | Content |
| --- | --- |
| `phase` | `investigate` or `repair` |
| `components`, `levels`, `healthy` | the four components, their possible levels, and the healthy level that a replacement restores |
| `tiers`, `prices` | the channels of each tier and the price table |
| `budget_left`, `repairs_left`, `steps_left` | what remains of the budget, the replacements, and the step cap |
| `min_faulty`, `max_faulty` | the rule on the number of faulty components |
| `history` | the agent's view of every `measure`, `replace`, `submit`, and `refused` event |

History entries show the tier, cost, result, and `repeated` flag of a measurement, the component and cost of a replacement, the submitted fields, and the action and reason of a refusal. They never show a condition label, the combination after a replacement, or whether a replacement was necessary, and they omit cycle numbers because the UCI data is public and labeled. `info` holds `refused` after a refusal that leaves the episode running, and the full log, hidden combination included, on the last step. A tool server may forward the observation, the cost, and the done flag, but never `info`. Agents must also run without access to the data files, where a measurement can be looked up, and to the episode folders and the result tables, which reveal the hidden combination.

---

## Actions and prices

| Action | Fields | Phase | Price |
| --- | --- | --- | --- |
| `measure` | `tier`, the int 1, 2, or 3 | both | the tier's price |
| `submit` | `diagnosis`, one level per component | `investigate` | 0 |
| `submit` | `insufficient_evidence` set to true and `next_action`, a non-empty string | `investigate` | 0 |
| `replace` | `component` | `repair` | part price plus downtime |
| `finish` | none | `repair` | 0 |

| Tier | Channels | Values returned | Price |
| --- | --- | --- | --- |
| 1 | TS1 to TS4 temperatures, VS1 vibration, and CE cooling efficiency | 24 | 1 |
| 2 | tier 1 plus FS1 and FS2 flows and CP cooling power | 36 | 3 |
| 3 | tier 2 plus PS1 to PS6 pressures, EPS1 motor power, and SE efficiency factor | 68 | 10 |

| Component | Part price | Price with downtime |
| --- | --- | --- |
| cooler | 20 | 40 |
| valve | 20 | 40 |
| pump | 30 | 50 |
| accumulator | 20 | 40 |

The prices are those of `DEFAULT_PRICES`, with a downtime charge of 20 per replacement. A submission ends the investigate phase, either with one int level from each component's level set or by declaring insufficient evidence with `next_action`, the one action that would separate the remaining candidates. The environment logs `next_action` without checking it. A replacement of a healthy component is allowed at the same price and counts as unnecessary.

Every refusal carries a reason, such as `unknown tier`, `already submitted`, or `diagnosis must be submitted before the first repair`. A bool, a float, or a string is refused as a tier or a level even when it equals a valid one. `_checked_prices` raises `ValueError` for an invalid price table when the environment is built, for example a tier without channels in `TIER_SENSORS`, a missing part price, or a measurement priced at zero.

---

## Scoring

`score(log)` in `src/envs/hydraulic_rig/scoring.py` computes every score deterministically from the episode log alone, so a log saved as JSON gives the same scores again.

| Event | Fields besides `event` |
| --- | --- |
| `reset` | `seed`, `hidden`, `budget`, `max_repairs`, `max_steps`, `prices` |
| `measure` | `tier`, `cost`, `cycle`, `repeated`, `result` |
| `submit` | `cost`, `diagnosis`, `insufficient_evidence`, `next_action` |
| `replace` | `component`, `cost`, `unnecessary`, `combo_after` |
| `refused` | `action`, `cost`, `reason` |
| `finish` | `cost` |
| `end` | `cost`, `reason` |

`hidden` is the start combination, and `unnecessary` is true when the replaced component was already healthy. Every step call writes exactly one `measure`, `submit`, `replace`, `refused`, or `finish` event.

In the formulas, h0 is the start combination, H is `HEALTHY_COMBO`, and S is the submit event if there is one. The diagnosis d is that of S when S exists and does not declare insufficient evidence, and None otherwise. R is the list of replace events, and the final combination f starts at h0 and sets `f[c] = H[c]` for the component c of each event in R.

| Score | Formula over the log |
| --- | --- |
| `submitted` | S exists |
| `insufficient_evidence` | S exists and declares insufficient evidence |
| `diagnosis_per_component` | `d[c] == h0[c]` for each component c, or None for each when d is None |
| `diagnosis_all_correct` | d is not None and `d[c] == h0[c]` for all four c, the Top-1 of the combination |
| `recovered` | `f == H` |
| `repairs_match_diagnosis` | d is not None and every replaced component c has `d[c] != H[c]` |
| `cost` | the sum of `cost` over all events |
| `cost_before_submit` | the sum of `cost` up to and including S, or None without S |
| `steps` | the number of `measure`, `replace`, `submit`, `refused`, and `finish` events |
| `steps_before_submit` | the same count up to and including S, or None without S |
| `measures_tier1`, `measures_tier2`, `measures_tier3` | the number of `measure` events at each tier, in both phases |
| `replacements`, `unnecessary_replacements` | the events in R, and those with `unnecessary` true |
| `refusals` | the number of `refused` events |
| `end_reason` | the `reason` of the `end` event |
| `initial`, `final` | h0 and f |

The diagnosis is scored against h0, the fault the agent was asked to find, and recovery against f. A submission of insufficient evidence is never scored as a diagnosis, so a Top-1 rate over all episodes counts it as incorrect, and the rate of `insufficient_evidence` belongs beside it. The cost of a correct diagnosis is `cost_before_submit` on the episodes with `diagnosis_all_correct` true, while `cost` includes the replacements.

`run_policies.py` writes these fields after `policy` and `seed` as one CSV row per episode, by default to the generated file `benchmarks/hydraulic_rig/results/v0_scores.csv`.

---

## The clock

The clock is optional and off by default, so every episode without it is the v0 episode above. It is turned on with
`TinyHydraulicEnv(..., clock=DEFAULT_CLOCK)`, or with `--clock` on `run_episodes.py`. Time is counted in 60-second rig cycles.

| Setting | Default | Meaning |
| --- | --- | --- |
| `durations.measure` | tier 1: 1, tier 2: 2, tier 3: 5 | cycles a measurement takes; the reading is taken when it starts |
| `lead_time` | 10 | cycles between `replace` (a work order) and the repair; the rig keeps running |
| `repair_cycles` | 5 | cycles the rig is stopped for the repair (downtime; nothing degrades) |
| `max_advance` | 500 | the largest `advance_time` call |
| `degradation` | 1 event between cycles 20 and 200 | hidden schedule; at each event one component falls to its next recorded level |
| `fault_cost_per_cycle` | 0 | charged for every running cycle with a component off its healthy level |

- **advance_time(cycles)** is a new action and tool. It only lets time pass. The server offers it only in clocked episodes.
- **The degradation schedule** is drawn at reset from a generator of its own (`clock-<seed>`), so a seed starts from the same
  combination and window with or without the clock. When an event is due, the component is drawn among those that can
  still fall and whose new combination is in `allowed`; if none can, the event is logged and skipped. The agent is not told;
  only the readings change, because the recordings of the new combination are served from then on.
- **The diagnosis** is scored against the combination at the submission (`diagnosis_target`), which equals the initial one
  when nothing degraded first. Recovery is still scored on the final combination.
- **New score fields** (clocked episodes only): `diagnosis_target`, `cycles`, `fault_cycles`, `downtime_cycles`, `degradations`.
- The agent sees the clock in the observation (`clock`, `clock_rules`), in the status of every tool answer, and in the briefing;
  it never sees the schedule.
- Within one combination the recordings are stable, so which cycle of the block is served does not depend on the time;
  time changes the state only through degradation and repairs.
- Not yet built: renamed tools (`read_sensors`, `create_work_order`), a technician's repair mid-episode, and degradation
  awareness in the causal diagnosis server, whose ranking still assumes that all readings come from one combination.

## Scenario classes

| Class | Start combinations | Rule in the observation | Steps of the oracle |
| --- | --- | --- | --- |
| S1 | the 10 combinations with one faulty component | `min_faulty` = `max_faulty` = 1 | 2: submit, then replace the faulty component |
| S2 | the 37 combinations with two faulty components | `min_faulty` = `max_faulty` = 2 | 3: submit, then replace both faulty components |

`CLASSES` in `benchmarks/hydraulic_rig/scenario_anchors.py` defines the classes, and `class_env` builds the environment of a class at the default prices, budget, and repair limit, with `allowed` set to its combinations. Each combination of a class is one record. The task text of S1 reads "Hydraulic unit hyd_1 was flagged by the performance monitor. Exactly one component is off its healthy level. Find it and restore the unit within the budget." The task text of S2 reads "Hydraulic unit hyd_1 was flagged by the performance monitor. Exactly two components are off their healthy levels. Find them and restore the unit within the budget." `task_text` in `src/servers/causal_diagnosis/adapter.py` words both from the rule, and neither names a component, a level, or a symptom.

A replacement moves an episode into the block of another combination, so a split into training and evaluation combinations must be closed under repair, which `closed_under_repair` checks. Training on S1 and holding out S2 passes: every fault in one component stays in training, and an evaluation episode is a new combination of known faults.

A start state is a combination, a window start, and an entry offset, and a block of n cycles gives n - `block_cap` + 1 windows with `block_cap` offsets each. `start_states` lists every start state that `reset` can draw, and `seed_table` gives each one the smallest seed that reaches it.

An admission anchor is a condition on the scores of a fixed policy that a scenario class must meet to be admitted. `scenario_anchors.py` plays every start state of a class once with each policy and checks six anchors at the default prices:

| Anchor | Policy | Condition |
| --- | --- | --- |
| 1 | `oracle` | Top-1, recovered, and resolved rates of 1, every episode at the reference cost |
| 2 | `do_nothing`, which submits the healthy combination and finishes | Top-1, recovered, and resolved rates of 0 |
| 3 | `replace_everything`, which submits a combination of the class drawn without measuring and replaces components in a seeded random order | resolved rate at most 0.20 |
| 4 | `random` | Top-1 rate at most 0.10 and resolved rate at most 0.05 |
| 5 | `greedy_prior_1to2` under each of the 10 fold seeds | Top-1 and resolved rates strictly between those of `random` and `oracle` |
| 6 | `symptom_only`, which submits one fixed combination without measuring and replaces the components it calls faulty | resolved rate at most 0.20 |

An episode is resolved when the diagnosis is Top-1 correct, the asset recovered, and no replacement was unnecessary. The reference cost is the oracle's cost, the sum of the replacement prices of the components that start off their healthy level. Records can hold different numbers of start states, so a start state of record r weighs `1 / (R * n_r)`, with R records in the class and n_r start states in r. Intervals are 95% percentile intervals over 10,000 bootstrap resamples of records.

```bash
# from the fork root, with HYD_FACTS_PATH set
uv run python benchmarks/hydraulic_rig/scenario_anchors.py
```

The script writes `benchmarks/hydraulic_rig/results/scenario_anchors_S1.csv` and `scenario_anchors_S2.csv`, generated files in a folder that git ignores, so the repository holds no anchor values.

---

## Reference policies

Each policy has `act(obs)`, and `run_episode(env, policy, seed)` plays one seeded episode to its end and returns the log. `policy_table` in `run_policies.py` builds a new policy for every episode, so any CSV row reruns on its own.

**Random.** `RandomPolicy(seed)` draws from the stream `random-policy:<seed>`, kept apart from the environment's so that its draws do not repeat the draw of the hidden combination. In the investigate phase it submits a random diagnosis with probability 0.15 or when no tier is affordable, and otherwise measures a random affordable tier. In the repair phase it draws a random component with probability 0.4 and replaces it when affordable, measures a random affordable tier, if there is one, with probability 0.3 or after an unaffordable draw, and finishes otherwise.

**Oracle.** `OraclePolicy(env)` reads `env.initial` and `env.hidden`, submits the truth without measuring, and replaces the faulty components. It gives the upper bound on diagnosis and recovery and the lowest cost of any episode that recovers, and it runs in process only, never through the action server.

**Greedy.** `GreedyPolicy(classifiers, use_max_faulty=False, confidence=0.95, min_gain_per_price=0.05)` keeps a belief over the 144 combinations and buys the measurement with the largest expected information gain per unit price. The reference row `greedy_prior_1to2` sets `use_max_faulty=True` and starts from a uniform prior over the combinations that satisfy the observation's rule on the number of faulty components. The ablation row `greedy` keeps the default and starts from a uniform prior over the 143 combinations other than the healthy one.

**Classifiers and split.** `TierClassifiers` fits one linear discriminant analysis (LDA) per tier and component on the served summaries, with standardized features, a pooled within-class covariance shrunk by `SHRINKAGE` toward a scaled identity, and equal class priors. `stratified_folds` deals the 144 combinations into `N_FOLDS` folds of 18, stratified by level, and `cross_fitted_predictions` predicts the stable cycles of each fold with classifiers fitted on the other folds. It refuses a split whose training set lacks a level. The split is by combination and not by cycle because a window can start anywhere in a block, so every stable cycle of a block can be served, and a split inside a block would put served cycles into the training set. The cycles flagged as not stable cannot stand in for them, since the valve is at 100 in all of them. Every cycle that an episode serves before the submission is predicted by classifiers fitted on the 126 combinations outside the fold of its start combination. The predictions form a lookup table keyed by the measured values, an evaluation device that needs the labels of every block.

**Observation model.** `observation_model` gives, for hypothesis h, tier t, and component c, the held-out confusion row of the level of c in h, counted over every block except that of h, with one added to each count. The policy assumes:

- A1: given the combination, the predicted levels of the four components are independent.
- A2: given the combination, measurements at different tiers are independent.
- A3: within one tier, predictions on cycles of the same block are exchangeable draws from a Dirichlet-multinomial centered on the row, so the next prediction follows a Polya urn around the row. The concentration is fitted per tier and component by maximum likelihood over `ALPHA_GRID`.
- A4: a measurement equal to an earlier one at the same tier adds no information.

**Decision rule.** The expected information gain of tier t is `IG(t) = H(O) - sum_h b(h) H(O | h)` in bits, with b the belief and O the predicted levels of the next measurement at tier t. The policy keeps a reserve for its repairs, the sum of the `repairs_left` most expensive replacement prices, which is 90 at the default prices with two repairs left. It submits the most probable combination when the largest belief reaches `confidence`, when the window starts repeating, when no tier is affordable within the reserve, or when the best ratio of gain to price falls below `min_gain_per_price`. At the default prices that threshold means paying at most 20 price units for one bit. Otherwise the policy measures the tier with the best ratio. After the submission it replaces the components it diagnosed as faulty, the most probable fault first, and finishes.

**Known limits.** A2 ignores the correlation between tiers, whose channels overlap, so the belief at submission can be overconfident. The rule looks one step ahead and prices information in bits, not in expected repair cost.

**Remaining leakage.** The observation rows leave out only the hypothesis's own block. In an episode on combination g, the rows of other hypotheses that share a level with g still count the held-out predictions on g, the held-out predictions on the other folds come from classifiers fitted with the cycles of g, and the concentrations are fitted on all blocks.

---

## Programmatic use

Run from the fork root with `uv run python` and `HYD_FACTS_PATH` set:

```python
from envs.hydraulic_rig.data import build_blocks, default_facts_path, load_cycles
from envs.hydraulic_rig.env import TinyHydraulicEnv
from envs.hydraulic_rig.policies import RandomPolicy, run_episode
from envs.hydraulic_rig.scoring import score

cycles = load_cycles(default_facts_path())
blocks = build_blocks(cycles)
env = TinyHydraulicEnv(cycles, blocks)  # default rule: one or two faulty components

obs = env.reset(0)
obs, cost, done, info = env.step({"type": "measure", "tier": 1})
summaries = obs["history"][-1]["result"]  # mean, std, min, and max of the channels of tier 1

guess = {"cooler": 100, "valve": 100, "pump": 1, "accumulator": 130}
obs, cost, done, info = env.step({"type": "submit", "diagnosis": guess})
obs, cost, done, info = env.step({"type": "replace", "component": "pump"})
if not done:
    obs, cost, done, info = env.step({"type": "finish"})
scores = score(env.log)  # the final info carries this log, so keep info away from agents

s1 = TinyHydraulicEnv(cycles, blocks, min_faulty=1, max_faulty=1)  # the 10 start combinations of S1
reference = score(run_episode(s1, RandomPolicy(1), 1))
```

---

## Limits and later releases

The environment has no clock, and a step counts an action, not time. A UCI cycle lasts 60 s while the `hyd_1` records are stamped one day apart, and the recordings repeat one constant load cycle, so no episode contains a load change. A clock with action durations and delayed results is planned for a later release.

---

## Tests

```bash
# 1. the environment, the scorer, the reference policies, and the start states
uv run pytest src/envs/hydraulic_rig -q

# 2. the data scripts, the policy runner, and the scenario anchors
uv run pytest -q benchmarks/hydraulic_rig/tests/test_data_facts.py benchmarks/hydraulic_rig/tests/test_build_blocks.py \
  benchmarks/hydraulic_rig/tests/test_fetch_uci.py benchmarks/hydraulic_rig/tests/test_run_policies.py \
  benchmarks/hydraulic_rig/tests/test_scenario_anchors.py
```

Without the data file, the tests that need it are skipped rather than failed, so a passing run without the data file does not exercise the environment.
