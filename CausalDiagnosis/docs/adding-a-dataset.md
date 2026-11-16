# Porting the Gym to Another Asset Class

The hydraulic rig gym separates rules from tables. A new asset class enters as data and a profile, which are a data file of its recordings and an asset file `CausalDiagnosis/datasets/<asset_id>/asset.json` with the tables of the asset and the rules it uses, which `CausalDiagnosis/datasets/dataset_loader/rule_cards.py` calls cards. A language model may draft the asset file and its rules from the documentation of the asset, the draft is used as written, and the user may edit or override any field, as the section Drafting the asset file shows. The only gate before use is the validation of the profile, which lists every problem at once. No test gate stands between a valid profile and a run, and the engine never calls a language model at run time. The hydraulic rig `hyd_1` is the first asset file, at `CausalDiagnosis/datasets/hyd_1/asset.json`, with its record loader in `CausalDiagnosis/datasets/hyd_1/records.py`.

The asset file, the rule library, the validation, and the record loader live in `CausalDiagnosis/datasets/`. The engine, the two servers, the agent, and the drivers read the tables from the profile of the episode, and `CausalDiagnosis/datasets/hyd_1/hyd_1_tables.py`, `CausalDiagnosis/datasets/hyd_1/former_field_names.py`, and `CausalDiagnosis/datasets/hyd_1/tests/knowledge_tables.py` serve them under their former names to the tests of hyd_1 only. The documents of the AssetOpsBench base row, the registry entry with one child per component, the failure modes, and the telemetry records, come from the profile through `CausalDiagnosis/frameworks/AssetOpsBench/tool_servers/profile_documents.py`. The engine configuration checks its sensor sets, `sensor_groups` of `EpisodeConfig`, against the channels of the profile of its `asset_id`, so a second asset of the recorded levels kind passes the checks of its own profile and of the engine configuration alike. The section on what still binds the code to hyd_1 lists every binding that remains.

The related pages are [README.md](README.md) for the index, [environment.md](environment.md) for the episode rules, [causal-diagnosis-mcp.md](causal-diagnosis-mcp.md) for the fit of the models of the CausalDiagnosis MCP, [experiments.md](experiments.md) for the presets and the drivers, and [challenges.md](challenges.md) for the reasons behind the design.

---

## The parts a port fills

The environment consists of the first four parts, and the CausalDiagnosis MCP and the agent of the fifth part stand outside it and reach it through the tools only. The asset layer defines four port interfaces, the class `AssetProfile` and the protocols `EvidenceSource` and `TruthModel` in `CausalDiagnosis/datasets/` and the protocol `DiagnosticsModel` in `CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/belief/diagnostics_model.py`, and for recorded condition levels `RecordingSource`, `LevelsTruth`, and `EffectDiagnostics` implement the three protocols.

| Part | What it holds | What a port supplies |
| --- | --- | --- |
| asset profile | the public tables of the asset, its rules, and the private fault rule of the initial states | the asset file |
| evidence source | the recordings, served cycle by cycle from the stable block of the current combination | the data file, or a generator that writes it, or an evidence kind that the package of the asset registers |
| rule library | the rule kinds with the keys of their parameters | a choice of rules and their parameters, and a plugin kind for a rule that no built in kind covers |
| tools, verifier, and truth model | the actions with their prices and durations, the checks, and the scores | nothing beyond the profile |
| CausalDiagnosis MCP and agent | the models of the CausalDiagnosis MCP and the planning loop of the agent, outside the environment | a fit of those models on the data of the asset |

---

## The port contract

A port adds files in two places and edits no other file of the repository.

| Place | What the port puts there |
| --- | --- |
| `CausalDiagnosis/datasets/<asset_id>/` | `asset.json`, `__init__.py`, the modules of the port, the script that fetches and builds the data as `download_and_build.py`, `data/CHECKSUMS.txt`, its presets in `presets/` and its tests in `tests/`, a package with an empty `__init__.py`; its run folders go to `CausalDiagnosis/results/runs/<asset_id>/`, which git ignores |
| the table of sources | the row of the dataset with its official page and the one command that fetches it, in every file that holds the table, the index page [README.md](README.md), the README of the repository and [dataset.md](dataset.md) |

Code that several ports share, such as an evidence kind, lives in a folder of its own under `CausalDiagnosis/datasets/` and follows the same rule. A server or an agent that a port brings is a new package beside the existing ones, and its name holds the asset id. Such a server is built through `transcribed` of `CausalDiagnosis/mcp_servers/utils/transcript_hooks.py`, so that every tool listing and every call writes one line of the transcript of the episode. The tests of the asset layer mount a made-up port from its folder alone and play it on both engines and behind both task servers, so a general path that needed a registration outside that folder fails there. The scripts of a port use the packages of the repository and add none, so a port edits neither `pyproject.toml` nor `uv.lock`.

The `__init__.py` of an asset folder imports from `CausalDiagnosis.datasets` alone and names every other object by a `"module:attribute"` string, which the engine resolves on first use, since the engine imports the package while it loads the profile. Every file a port generates, the data files and the tables fitted or computed from them, lies in the `data` folder of its asset folder, which git ignores but for `CHECKSUMS.txt`. The tables of what a port provides, file by file, and of what an evidence kind states are in [CausalDiagnosis/datasets/README.md](../datasets/README.md).

In return the engine holds to eight rules.

1. The engine finds an asset by its folder, imports the package, which registers the kinds of the port, and then validates the asset file. Validation lists every problem at once.
2. Validation refuses a table the engine cannot run, by name, as the section Validation lists.
3. No general path reads a name, a count, or a file of another asset. Both engines, the action server, the task server, the documents of the AssetOpsBench servers, the task descriptions that `briefing.py` of the action server and of the task server build, the tool table, and the verifier read the components, levels, sensor sets, channels, statistics, nouns, and the site from the asset file of the episode.
4. An episode from an initial state with no fault is valid from reset to verdict when the fault rule of the asset admits it.
5. A cycle for which the source holds no recording serves no reading and is charged no read price.
6. The data file is checked before its use, as the section The data file states, and every table built from the data names the hash of the data file it was built from.
7. The test set of an asset has one definition, which the sampler of initial states and the folds file both read.
8. The rules of the section Rules every asset follows hold on every asset.

The reference classifiers behind the label of the rig, the greedy baseline policy and the read ceiling of the verifier scope all are written for the hydraulic rig. A port brings its own label when it wants one, `LABELS` in its `__init__.py`, and without one every initial combination of the port counts as diagnosable within the budget. A run of a port refuses the greedy baseline policy and the verifier scope all at launch, by reason, since both run the reference classifiers of the rig.

The drivers under `CausalDiagnosis/` play any asset with a folder under `CausalDiagnosis/datasets/`. A run plays the asset that its preset names or that `--asset` names, over whose defaults `run_episodes/` composes the preset, and its run folders land under `CausalDiagnosis/results/runs/<asset_id>/` unless `--out` names another root. The table tools build the tables of one asset from a folder of its runs and stop by name on a folder whose runs name two assets. The grid files under `CausalDiagnosis/cluster_jobs/` hold the lines that `grid_job_lists.py` writes for the rig, so a port writes its own lines, each with `--asset`. A run of a port refuses at launch, by reason, the anchored scenarios, which map the scenarios of AssetOpsBench onto the dates of the recordings of the rig, and a run of the static agent refuses an asset of a kind other than recorded levels, since that agent shows the recorded fields of the first cycle that a read serves.

The models of the CausalDiagnosis MCP are fitted per asset. Its model code fits an asset of the recorded levels kind that serves the mean and the standard deviation and whose recorded combinations admit the development split, on which the form of the models is chosen. A row with the CausalDiagnosis MCP refuses any other asset at launch, by reason, and that asset plays the rows without the CausalDiagnosis MCP.

An action beyond replacement to the healthy level, such as a change of load or speed, a partial repair, or a repair that halts a degradation in progress, lies outside a port. Such an action needs a new tool, a transition in the clock, an outcome rule in the verifier, and a rule kind for its parameters, which are edits of the engine, since the environment replays only replacement to the healthy level.

---

## The asset file

The asset file is one JSON object with the fields below, all required. `AssetProfile` in `CausalDiagnosis/datasets/dataset_loader/profile_data.py` reads it, and the field `evidence_kind` decides which tables the rest of the file holds.

| Field | What it holds | Public |
| --- | --- | --- |
| `format` | the version of the file layout, which is `1` | yes |
| `asset_id` | the identifier of the asset, letters, digits, and underscores | yes |
| `evidence_kind` | `recorded_levels` for recordings under set condition levels, `scripted_cases` for scripted fault cases, or a kind that the package of the asset registers | yes |
| `components` | the component names, which the diagnosis tool takes as its arguments, so each is an identifier | yes |
| `levels` | the condition levels of each component, integers whose order fixes the order of the combinations | yes |
| `healthy` | the healthy level of each component | yes |
| `degradation` | the levels of each component from healthy to worst, the order along which a transition to a worse level, a drop in the code, moves a component one step | yes |
| `observation_kinds` | the sensor sets, each with its channels | yes |
| `channels` | the public field name of each channel and the quantity that opens its feature names | yes |
| `statistics` | the statistics recorded per channel and the subset that a reading serves, of which the first is the plain statistic that a telemetry record stores under the bare field | yes |
| `defaults` | the read prices, part prices, repair cost `repair_charge`, fix kinds, misdiagnosis price, durations, budget, horizon, step cap, and clock, whose rate per generated token `seconds_per_token` only an asset that plays the clock mode `generated_tokens` needs | yes |
| `rules` | the rules of the asset, each a kind with its parameters | yes, as the processes view |
| `sampler` | the fault rule of the initial states, the fewest and the most faulty components of an initial state | no, only the sampler of initial states reads it |
| `failure_modes` | the component, level, code, severity, and recommended action of every failure mode | yes |
| `knowledge` | the sources of the failure modes and of the associations, the channels that indicate each component, and the correlations behind them | yes |
| `records` | the keys of the evidence kind, for recorded levels the label column of each component and the value of the stable flag, then the environment variable that names the data file, the release folder and the release file, the official page of the data, and the one command that fetches and builds it | no, only the record loader reads it |
| `text` | the nouns that prompts and descriptions use for the asset, the whole asset patterns of the contradiction check, the registry file with its fallback entry, an example channel, and five optional sentences, `question`, `opening`, `scoring`, `level_meaning`, and `readings`, of which `question` may hold the marks `{components}` and `{limits}` and `readings` is required of an asset whose evidence kind declares no loop | yes |
| `scoring` | `program` when the verifier scores against the hidden state, `judge` when a judge scores free text | yes |

The answer of `get_asset_profile` comes from `HydraulicRigEnv.public_profile` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/env.py`, which holds neither `sampler` nor `records`, so no agent is told the fault rule of the initial states or the layout of the data file. Its processes block, which `processes_block` in `CausalDiagnosis/datasets/dataset_loader/profile_data.py` builds from the configuration of the episode, holds the parameters of the rules and never a draw, so which component moves to a worse level at which cycle remains hidden. `AssetProfile.default_config` builds the episode configuration from `defaults`, the rules, and `sampler`, and the configuration of hyd_1 is checked field by field against the digests of `EpisodeConfig()` stored in `CausalDiagnosis/experiments/run_verification/invariant_digests.json`. The four tests of `CausalDiagnosis/datasets/tests/test_digests.py` recompute their digests with `CausalDiagnosis/experiments/run_verification/invariant_digests.py` and compare them with the label `current` of that file. The tests run only with `ASSET_DIGESTS=1` and only against a label of record, one that names the commit its digests were computed on, and while the digest file holds none for the data file in use they skip and name the command that writes it in a clone of that commit, `python -m CausalDiagnosis experiments invariant-digests --facts <data file> --label current --write CausalDiagnosis/experiments/run_verification/invariant_digests.json --commit <commit>`. The fields that the profile leaves out of the configuration, such as the instruction level `guidance_level` and the verifier scope `verifier_scope`, belong to the preset of a run.

---

## Rules of the asset file

A rule of the asset file is a JSON object `{"kind": <name>, "parameters": {...}}`. The kinds and the keys of their parameters are declared in `CausalDiagnosis/datasets/dataset_loader/rule_cards.py` and nowhere else. A profile picks the rules it needs and fills their parameters, and a rule that only one asset needs takes a declared kind with parameters of its own. Each kind also declares where its parameters land in the episode configuration, so the configuration that a profile builds does not depend on the order of its rules or of their parameters.

| Kind | Parameters | Lands in | Evidence kinds |
| --- | --- | --- | --- |
| `fix_history` | `healthy_probability`, `healthy_ago`, `degraded_probability`, `degraded_ago` | `fix_history` | every kind that plays episodes |
| `emergency` | `lead`, `surcharge`, `hold` | `emergency` | every kind that plays episodes |
| `schedule` | `mode`, `components`, `first_drop`, `interval` | `schedule` | every kind that plays episodes |
| `hazard` | `mode`, `base_rate`, `depth_factor`, `collateral` | `hazard` | every kind that plays episodes |
| `dependencies` | `dependencies`, `priority` | `dependencies` and `priority` | every kind that plays episodes |
| `visits` | `charge` | `prices.visit`, right after `repair_charge` | every kind that plays episodes |
| `downtime` | `rates`, `stopped` | `downtime_rate`, with the production loss rate of the stopped asset last | every kind that plays episodes |
| `failure` | `patience`, `strict`, `penalty` | `failure` | every kind that plays episodes |

A profile states a rule of each kind at most once and may state none. A kind without a rule is off, and its fields of the episode configuration take the off parameters that `CausalDiagnosis/datasets/dataset_loader/rule_cards.py` declares for the kind, such as a mode of none, a zero charge, or an empty dependency mapping, so a base preset that switches every rule off selects none of them. The `emergency` rule is off when `hold`, the duration of the emergency repair, is 0, and the engine then refuses the emergency mode. The `failure` rule is off when `patience`, the time to failure at the worst level, is 0. The parameters of the rules of hyd_1 are design parameters, which `EpisodeConfig` in `CausalDiagnosis/environment/Causal_Diagnosis_Environment/episode_config.py` takes as its defaults. A kind that no built in kind covers is a plugin, a `RuleKind` with a name, the keys of its parameters, and the evidence kinds it applies to, which the package of the asset that needs it registers by name with `rules.register_kind`. A rule of a kind for recorded levels applies to every evidence kind that plays episodes, since its parameters land in the episode configuration. A plugin cannot take the name of a built in kind or of an earlier plugin.

---

## Validation

`validate` in `CausalDiagnosis/datasets/dataset_loader/profile_validation.py`, `CausalDiagnosis/datasets/dataset_loader/profile_registry.py` and `CausalDiagnosis/datasets/dataset_loader/profile_data.py` returns every problem of a raw profile at once, and an empty list means the profile may be used. The function checks the fields and their types, the consistency of the tables, such as a healthy level among the levels and a degradation order that starts at it, the channels of every sensor set, the failure modes against the levels, and every rule against its kind. Validation also refuses, by name, a component without a second level beside the healthy one, two components or a component and the asset type that share a failure mode class key, empty whole asset patterns, served statistics outside the recorded ones, an asset id that is the name of a module of `CausalDiagnosis/datasets`, and sensor sets without one set of every channel, since a read of every channel is charged to the one set that holds them all. The environment refuses an initial combination without a recording when it is built. A profile of a kind that plays episodes whose tables pass these checks is then built into an episode configuration, and the problems that `EpisodeConfig` finds come back in the same list. A table that cannot be read is reported once and the checks that read it are skipped, so one unreadable table gives one problem. `load_profile` and `AssetProfile.from_json` raise `ProfileError` with every problem, `get_profile` serves the asset file of an asset folder by its identifier, after the import of the package of the asset, and `register_profile` serves a draft under its identifier in place of a bundled file. The command below, run from the root of the fork, prints every problem of a draft or the word valid.

```bash
PYTHONPATH=src python -c "import json, sys; from CausalDiagnosis.datasets.dataset_loader.profile_validation import validate; print('\n'.join(validate(json.load(open(sys.argv[1])))) or 'valid')" CausalDiagnosis/datasets/<asset_id>/asset.json
```

---

## The data file

A data file of the recorded levels kind holds one JSON record per recorded cycle with `cycle_index`, `labels`, and `features`. The labels hold the column of each component that `records.label_keys` names and the flag `stable_flag`, whose stable value is `records.stable_flag_stable`. The features are named `<Quantity>_<Channel>_<statistic>`, with at least every statistic of `statistics.recorded` for every channel, so a channel or a statistic is named by letters and digits alone. `load_cycles` and `build_blocks` in `CausalDiagnosis/datasets/dataset_loader/recordings.py` read the file and group the stable cycles of each combination into one block, and a block that is not contiguous is refused. `RecordingSource` refuses a combination with fewer than `BLOCK_CAP` stable recordings, ten, and names the first such combination, since a starting offset within the recorded block, the `entry_offset` of an initial state, and the fit it names must exist for every combination alike. Five of the ten are the cycles that an episode shows from before its first cycle, and five are the cycles that an unchanged state replays during the episode. The release file is read only when its SHA-256 equals its line of the `CHECKSUMS.txt` beside it, and a missing or refused file is named with the page and the command of the asset. The CausalDiagnosis MCP needs every level of every component inside its fitting pool, since its interventional effect model is identified by recordings under set conditions.

An asset that offers statistics per condition level instead of recordings needs a generator that writes records in this layout, and its episodes are then simulated rather than replayed. The fidelity of such a generator has to be reported against recorded data of the same machine class. The tests of the asset layer carry a seeded generator of this kind in `CausalDiagnosis/datasets/tests/conftest.py` and the tiny asset file `CausalDiagnosis/datasets/tests/tiny_asset.json`, three components with their own channels, sensor sets, and statistics.

---

## Rules every asset follows

**What an agent is told.** The environment serves no table that links a channel to a component and no reference of the value a channel takes at a fault level, beyond what the documents of the asset state. A port adds no such table from the recordings. On hyd_1 the rows on the AssetOpsBench tool surface return the failure mode codes of a component class without channels, the asset profile returns under `normal_ranges` the normal range of every channel mean, and from the repair rules layer on the repair manual names the channels that indicate a component.

**What ends an episode.** An episode ends only on a condition that the task description or a tool description states. A call that the rules cannot serve is refused, the refusal names the reason, and the episode goes on. A requirement that no public text states ends no episode, and the refusal names the state the call needs. What a refused call is charged is a rule that the task description states. On hyd_1 the task description lists every end condition, and a refused call is charged nothing and counts as a step.

**Where the data comes from, and what is published.** Every dataset is named by its official page and by one command that fetches it and checks its SHA-256. No dataset file and no file derived from single records enters the repository. A run folder holds readings of single records and is no part of the repository. Of a run the repository publishes the aggregate tables. The table of data sources in [dataset.md](dataset.md) holds one row per dataset, and a port adds its row with its command.

The design questions of a port that touch one of these rules, each with the answer in force, are in [CausalDiagnosis/datasets/README.md](../datasets/README.md).

---

## Drafting the asset file

A language model may draft the asset file of a new asset from the documentation of the asset, and its draft is the default, used as written once the validation command prints valid. The user overrides any field at one of two scopes. An edit of `asset.json` changes the field for every run of the asset, and the validation runs again. A preset changes a field of the episode configuration for its own runs alone, since `CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.compose` builds a preset over the defaults of its asset. A field of the base layer that the preset sets, or an option of a layer that it sets under `switches`, replaces the default.

1. `PYTHONPATH=src python -m CausalDiagnosis datasets new-dataset-scaffold <asset_id> --component <name>=<levels from healthy to worst>` writes `CausalDiagnosis/datasets/<asset_id>/asset.json` beside an empty `__init__.py`, the package of the asset, and takes one `--component` per component. The draft holds the components with their levels, their healthy levels and their degradation orders, and every other field holds the mark `TO FILL: <path>`, an open field. The command then prints the open fields left and every problem of the validation. The scaffold refuses an identifier that is not letters, digits and underscores or that names a module or a folder of the asset layer, and a folder that exists, so it never overwrites an asset, which `CausalDiagnosis/datasets/tests/test_new_dataset_scaffold.py` checks on hyd_1.
2. A language model receives the prompt below, the documentation of the asset, the scaffolded asset file, and this page, and returns the asset file with every open field filled.
3. The validation command of the section Validation runs on the result, and every problem it prints goes back to the model until the command prints valid. The file is then the default of the asset.

```text
You fill the open fields of the asset file of a new asset of the hydraulic rig gym.
You receive the documentation of the asset, the asset file that the scaffold wrote, and the page CausalDiagnosis/docs/adding-a-dataset.md.
Replace every value "TO FILL: <path>" with a value in the form that the table The asset file of that page gives for the field at <path>.
Take every name, level, channel, failure mode, price, duration and limit from the documentation, and name the passage it rests on.
Where the documentation states no value for a field, choose one that the documentation supports and mark it as an assumption.
Choose the rules from the kinds of the table of the section Rules of the asset file alone, one rule per kind at most, with exactly the parameter keys that the table names, and leave out a kind that the documentation does not describe.
Write no value that only the recordings of the asset could give, such as a normal range or a threshold, since the engine computes those from the data file.
Leave every field that the scaffold filled as it stands.
Return the whole asset.json, then one line per field you filled with the passage of the documentation or the word assumption.
```

On a made-up asset `fan_1` with a belt at the levels 0, 1 and 2 and a bearing at the levels 0 and 1, the first step is `PYTHONPATH=src python -m CausalDiagnosis datasets new-dataset-scaffold fan_1 --component belt=0,1,2 --component bearing=0,1`. If the manual of the fan states a visit cost of 40, the draft holds the rule `{"kind": "visits", "parameters": {"charge": 40}}`, and every run of `fan_1` charges 40 for a visit. A user who sets the visit cost to 55 edits that rule and runs the validation again, and every later run of `fan_1` charges 55. A preset in `presets/` of the folder of the port, where every program finds it by its name, such as `{"name": "fan_1_short", "asset_id": "fan_1", "horizon": 20}`, plays the asset with a horizon of 20 cycles and leaves the asset file as it is.

---

## Steps of a port

1. Write the data script of the asset as `download_and_build.py` in its folder `CausalDiagnosis/datasets/<asset_id>/`, one command, run from the root of the repository, that downloads the source from its official page, checks it, and builds the release file, after the model of `CausalDiagnosis/datasets/hyd_1/download_and_build.py`. The `data/CHECKSUMS.txt` of the asset folder, beside the release file, holds one `sha256sum` line for every file the command downloads and every file it builds, and the row of the dataset joins the table of sources. An asset that offers statistics per condition level writes its data file with a generator instead, and a check script after the model of `CausalDiagnosis/datasets/hyd_1/verify_data_facts.py` tests relations and never values.
2. Draft the asset file, as the section Drafting the asset file shows, with the tables, the rules, the fault rule, the failure modes, the record layout, the texts, and the scoring mode, and fill the package `__init__.py` that the scaffold writes empty beside it. The package registers the evidence kind and the rule kinds that the asset brings and declares its label, `LABELS`, and its generated tables, `TABLES`, so an asset of the recorded levels kind that brings no rule kind and no label of its own leaves it empty.
3. Run the validation command until it prints valid.
4. Fit the models of the CausalDiagnosis MCP with `PYTHONPATH=src python -m CausalDiagnosis mcp_servers fit-models --asset-id <asset_id> --out <models file>`, compute the grades with `PYTHONPATH=src python -m CausalDiagnosis mcp_servers diagnosability-grade --asset-id <asset_id> --models <models file> --out <grades file>`, and read the report of `PYTHONPATH=src python -m CausalDiagnosis mcp_servers calibration-report --asset-id <asset_id> --models <models file> --out <report folder>` before any agent runs. The models file and the grades file of a port lie in the data folder of its asset folder, and `CausalDiagnosis/experiments/cluster_runs/precompute_grid_tables.py --asset <asset_id>` fits the models ahead wherever the model code can fit the asset.
5. Run the baseline policies with `CausalDiagnosis/experiments/episode_runners/run_reference_policies.py --asset <asset_id>` and confirm that the oracle earns a higher reward than the do nothing policy on every seed, which is the ordering check of OpenEnv's RFC 008.
6. Register the training split before any run, with the first command below from the root of the fork, `--profile` and `--facts` added for a draft. The command deals the single faults of the test combinations into K folds, writes the folds file and prints, for each held-out fold, the test combinations its model tests and the initial combinations it trains on. K is at least 2 and at most the number of single faults, so every fold holds one, a component with fewer faults than K is faulty in that many folds only, and a split that leaves a model without a training initial state is refused and writes nothing, as a split that leaves a fold without a test combination is. The folds file names the SHA-256 of the data file it was dealt from, and a run on the recordings of another data file refuses it with the command that deals it again. The training numbers of the asset go into its `configs/training/<asset_id>/rft.json`, which holds `samples_per_state`, the samples of one round per training initial state, `minimum.kept_per_training_state`, the kept episodes per training initial state below which no adapter trains, a ratio to the training initial states of each model and not a count, `minimum.extension_rounds`, the rounds of the same size added below it, and `folds`, the folds file of the registered split under `file` and that of the development split under `development_file`. The number of folds and the training initial states of each model follow from the split and are no fields of that file. The development combinations follow from the same asset files through `development_split.development_starts`, the recorded fault combinations outside the test set with the test combinations that move when a faulty level needs them, and the second command deals the same rule over them into the development split, which a trial of the training path reads before the runs of record with `--training-split development`. Every asset has a training folder of its own, the folder that `HYD_TRAINING` names, and a trial uses another, apart from the files of the runs of record.
7. Run the agents with `CausalDiagnosis/experiments/episode_runners/run_episodes/ --config <preset> --asset <asset_id>` and build the tables of the asset with `CausalDiagnosis/evaluation/result_tables/tables_from_runs/ --runs <runs of the asset> --out <results of the asset>/aggregate` from a folder of its own runs, since the table tools stop by name on a folder whose runs name two assets. The AssetOpsBench judge reads the trajectory and the closing answer that the runner of the agent records, `trajectory.json` and `answer.txt`, which the driver writes for every agent, and `CausalDiagnosis/frameworks/AssetOpsBench/LLM_Judges/episode_judge.py` cuts what the prompt prints by the one registered rule that [experiments.md](experiments.md) describes. An episode without `trajectory.json` is not judged, and a call that overflows the context of the judge model is a failed call of the judge.

```bash
PYTHONPATH=src python -m CausalDiagnosis experiments write-training-split --asset <asset_id> --k <K> --out <folds file>
PYTHONPATH=src python -m CausalDiagnosis experiments write-training-split --asset <asset_id> --k <K> --starts development --out <development folds file>
```

The hydraulic rig's `configs/training/hyd_1/folds.json` and `configs/training/hyd_1/folds_development.json` are the outputs of the two commands for hyd_1 with K = 2 and seed 0, which two tests of `CausalDiagnosis/experiments/table_definitions/tests/test_write_training_split.py` check byte for byte, and another test of that file runs the command on a made-up asset of three components and reads its split back.

Steps 4 to 7 run on any asset with a folder under `CausalDiagnosis/datasets/`, and a run of another asset refuses at launch, by reason, the parts that the next section binds to hyd_1. Scenario anchors, which place a scenario text on the recorded day that the scenario names, are optional and live with the presets, as `CausalDiagnosis/frameworks/AssetOpsBench/scenarios/scenario_anchors.py` does for hyd_1.

---

## What still binds the code to hyd_1

Every module outside `CausalDiagnosis/datasets/hyd_1/` that imports the package of hyd_1 stands in the table `BOUND` of `CausalDiagnosis/datasets/tests/test_import_guard.py` with its reason, and the table below adds the parts of the drivers that their module binds to the rig and the default asset of a call that names none.

| Where | What binds it to hyd_1 | What a port does |
| --- | --- | --- |
| `CausalDiagnosis/environment/reference_policies/tier_classifiers.py`, read by `CausalDiagnosis/environment/reference_policies/greedy_policy.py` and `CausalDiagnosis/environment/reference_policies/diagnosability_label.py` | the reference classifiers of the rig behind the label of its initial combinations, the greedy baseline policy and the read ceiling of the verifier scope all | brings its own label as `LABELS` or plays with every initial combination diagnosable within the budget, and a run of it refuses the greedy baseline policy and the scope all at launch |
| `CausalDiagnosis/datasets/hyd_1/hyd_1_tables.py`, `CausalDiagnosis/datasets/hyd_1/tests/knowledge_tables.py`, `CausalDiagnosis/datasets/hyd_1/former_field_names.py` | the tables of hyd_1 under the former names of the rig modules | nothing, since they serve the tests of the rig alone |
| `CausalDiagnosis/environment/Causal_Diagnosis_Environment/env.py` | `HEALTHY_COMBO` and the module level helpers, which read the recordings of the default asset | nothing, since an episode reads the healthy state and the recordings of the profile of the episode |
| `CausalDiagnosis/frameworks/AssetOpsBench/scenarios/scenario_anchors.py` | the anchored initial states of the rig, from its scenario records | a run of it refuses the anchored scenarios at launch |
| `CausalDiagnosis/datasets/hyd_1/download_and_build.py`, `CausalDiagnosis/datasets/hyd_1/build_features_and_blocks.py`, `CausalDiagnosis/datasets/hyd_1/verify_data_facts.py` | the fetch, the build and the checks of the recordings of the rig | its own data script in its folder `CausalDiagnosis/datasets/<asset_id>/` |
| `CausalDiagnosis/environment/Causal_Diagnosis_Environment/episode_config.py`, `CausalDiagnosis/experiments/table_definitions/training_variants/held_out_folds.py` | `DEFAULT_ASSET`, the asset of a configuration or a call that names none | names its asset, in the `asset_id` of its presets or with `--asset` |
