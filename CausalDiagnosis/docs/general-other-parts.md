# General agent: the parts outside the agent

[general-agent-plan.md](general-agent-plan.md) makes the agent general, and [general-agent-tasks.md](general-agent-tasks.md) cuts that work into tasks. This page lists what is still written for one dataset outside the agent: in the shared configuration, the environment, the reference policies, the diagnostics server, the action servers, the driver, the cluster jobs, the tables, the viewer and the training scripts. Each item says where it stands, the way out, a first estimate of the effort, and the task of the plan that it joins, or that it is a task of its own. [adding-a-dataset.md](adding-a-dataset.md) lists the bindings to the rig that a port meets; this page adds the way out and the size of each that lies outside the folder of the rig.

A path that starts with `CausalDiagnosis/` exists in the tree at the commit that added this page. An estimate is in builder hours, from the number of places an item touches, without checking and integration. It is an estimate and no result.

## The rule for every way out

A value of one asset is a field of the asset file `CausalDiagnosis/datasets/<asset_id>/asset.json` or of a file in the folder of that asset. A value of one framework is a field of its declaration, read through `CausalDiagnosis/utils/frameworks.py`. A value of one served model is a field of the serving file. Shared code reads the field and names no asset, and what an asset does not declare is refused by name and never filled with the value of another asset. A number that is a true constant (a standard, a unit, a bound of a protocol or a format) stays in the code with its source.

The rules of the task page hold for every item here: one owner per file, a control for every test, no test weakened, and a pull request that names every stored table and every run identity that it moves. Where an item reaches into a file that a card of the task page owns, its change there is a request to that owner, and where a card says that a file must not be touched, the item comes after that card.

## Shared configuration

| Written for one dataset today | Where | The way out | Builder hours, first estimate | Joins |
| --- | --- | --- | --- | --- |
| the price tables are keyed by the components and the sensor groups of the rig | `CausalDiagnosis/configs/prices/` | a price table belongs to an asset: a folder of prices in the folder of the asset, found by the asset id of the preset | 4 to 6 | a task of its own |
| the timed development preset states a window for the first drop that was worked out by hand from the levels, the interval, the lead times and the horizon of the rig | `CausalDiagnosis/configs/development/timed.json` | the preset states the rule, and the loader computes the window from the asset | 4 to 6 | a task of its own |

## Environment

| Written for one dataset today | Where | The way out | Builder hours, first estimate | Joins |
| --- | --- | --- | --- | --- |
| `DEFAULT_ASSET` names the rig in two modules, and a configuration that names no asset takes every default from the rig's file; many shared modules read the constant, among them the module level helpers of the engine | `CausalDiagnosis/environment/Causal_Diagnosis_Environment/episode_config.py`, `CausalDiagnosis/experiments/table_definitions/training_variants/held_out_folds.py`, and their readers, `CausalDiagnosis/environment/Causal_Diagnosis_Environment/env.py` among them | a configuration always names its asset; the command line fills it in one place or refuses a run without it | 10 to 14 | a task of its own, with the readers in the tables below |
| the classes of the environment carry the rig's name (`HydraulicRigEnv`, `HydraulicRigGym`, `HydraulicRigOpenEnv`, `HydraulicRigAction`, `HydraulicRigObservation`, `HydraulicRigState`, `RigClock`), though their code reads the profile | `env.py`, `gymnasium_wrapper.py`, `openenv_wrapper.py` and `clock/episode_clock.py` of the same folder | names of the evidence kind, the old names kept as aliases for one release | 6 to 8 | a task of its own |
| eight shared files outside the driver and the training scripts name or build the environment class of the rig directly, and one example builds its gym class by name | `CausalDiagnosis/environment/Causal_Diagnosis_Environment/start_state_seeds.py`, `CausalDiagnosis/environment/reference_policies/diagnosability_label.py`, `CausalDiagnosis/episode_viewer/run_review_pack.py`, `CausalDiagnosis/experiments/cluster_runs/precompute_grid_tables.py`, `CausalDiagnosis/experiments/run_verification/invariant_digests.py`, `CausalDiagnosis/experiments/run_verification/replay_check/episode_replay.py`, `CausalDiagnosis/experiments/table_definitions/write_training_split.py`, `CausalDiagnosis/mcp_servers/Environment_Actions_MCP/episode_state.py`, `CausalDiagnosis/examples/run_one_episode.py`; a search of the tree for the class names lists them | `make_env` of `CausalDiagnosis/environment/Causal_Diagnosis_Environment/environment_factory.py` everywhere, and where the class is the default of an argument, the caller hands the class in | 6 to 8 | a task of its own; `invariant_digests.py` is a file of task 13, and task 12 owns one function of `diagnosability_label.py` |
| the contract writes the parameters of the diagnosis out, one per component of the rig | `CausalDiagnosis/environment/Causal_Diagnosis_Environment/tool_contract.json` | the contract of a kind states the tools, and the parameters per component are filled from the components of the asset at load | 2 to 4 | task 1 |
| before any episode the action server lists the texts of the default asset, and an episode that names no asset falls back to it | `CausalDiagnosis/mcp_servers/Environment_Actions_MCP/main.py` | a server is started for one asset and lists nothing of another | 3 to 5 | task 1 |

## Reference policies and labels

| Written for one dataset today | Where | The way out | Builder hours, first estimate | Joins |
| --- | --- | --- | --- | --- |
| the classifiers import the records of the rig and build its profile, its components and its levels when the module loads, so a label of any asset would be fitted on the rig, and every module that imports them loads the rig | `CausalDiagnosis/environment/reference_policies/tier_classifiers.py`, read by `diagnosability_label.py` and `greedy_policy.py` beside it, by `asset_view.py` and `belief/discriminant_urn_ranking.py` of `CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/`, and by three files of the driver under `CausalDiagnosis/experiments/episode_runners/`: `run_reference_policies.py`, `run_episodes/reference_policy_episodes.py` and `run_episodes/command_line.py` | the classifiers take a profile and its recordings as arguments, and a label table lives in the data folder of the asset it was computed for | 10 to 14 | a task of its own; task 12 owns the two digest functions alone |
| the greedy policy and the classifiers address the sensor groups by the names and the numbers that the rig gives them, and two examples of a reimplemented design and one latency probe of the rows of AssetOpsBench's servers use the group names of the rig | `CausalDiagnosis/environment/reference_policies/greedy_policy.py`, `tier_classifiers.py` beside it, `CausalDiagnosis/frameworks/AssetOpsBench/agents/Agent_As_Tool_Paper/in_context_examples.py`, `CausalDiagnosis/frameworks/AssetOpsBench/tool_servers/latency_probe.py` | the groups of the profile in their declared order; the two examples and the probe name a made-up group, in a change of its own that moves texts which both designs send and names that move | 8 to 12 | a task of its own; the examples only after the hash proof of task 11, whose card keeps their texts byte for byte |

## Diagnostics server

| Written for one dataset today | Where | The way out | Builder hours, first estimate | Joins |
| --- | --- | --- | --- | --- |
| the fitted files beside the module serve the default asset, and another asset needs an option and paths of its own | `CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/fit_models.py` | the fitted files of every asset in its data folder, and no default asset | 4 to 6 | task 12 |
| the model code fits one evidence kind and refuses another by name | the same file | another kind brings another backend | none of its own | tasks 3 and 9 |
| one tool text speaks of the cycles of the rig | `tools/tool_texts.py` of the same server | the noun and the unit of time from the declaration | 1 to 2 | a task of its own |

## The rows of AssetOpsBench's servers and the two reimplemented designs

| Written for one dataset today | Where | The way out | Builder hours, first estimate | Joins |
| --- | --- | --- | --- | --- |
| the sample documents of the rows are written for the rig | `CausalDiagnosis/frameworks/AssetOpsBench/tool_servers/hyd_1_sample_documents.py` | the declaration of the framework lists, per asset, its sample documents and the name it has on these rows | 4 to 6 | a task of its own; the entry in the declaration is a request to task 11, which owns that file |
| the anchored starts are those of the rig: the module imports the records of the rig and builds its profile when it loads, and the declaration of the framework names the id of the rig for the scenario set | `CausalDiagnosis/frameworks/AssetOpsBench/scenarios/scenario_anchors.py`, `CausalDiagnosis/frameworks/AssetOpsBench/__init__.py` | the anchors of an asset as data that the declaration lists per asset; the module takes the asset of the preset and loads no asset when it loads | 4 to 6 | a task of its own; the entry in the declaration is a request to task 11, which owns that file |
| the specs and the examples that this repository wrote for the two designs are keyed by the server names and the tool names of the rig | the registry and the runners of the two designs, the files of task 11 | the role table; the texts of the examples stay byte for byte, as the card of task 11 says | in the estimate of task 11 | task 11 |

## Driver and cluster jobs

| Written for one dataset today | Where | The way out | Builder hours, first estimate | Joins |
| --- | --- | --- | --- | --- |
| the names of the two servers and of their tools are fixed sets | `CausalDiagnosis/experiments/episode_runners/servers_of_a_run.py` | the sets come from the contracts of the run's asset | 4 to 6 | task 13 |
| the driver names or builds the environment class of the rig directly, in eight files | `planned_episodes.py`, `reference_policy_episodes.py`, `play_one_episode.py`, `episode_scoring.py`, `static_agent.py` and `__init__.py` under `CausalDiagnosis/experiments/episode_runners/run_episodes/`, `run_reference_policies.py` and `run_configuration.py` in the folder above | `make_env` of `CausalDiagnosis/environment/Causal_Diagnosis_Environment/environment_factory.py` everywhere, and where the class is the default of an argument, the caller hands the class in | 5 to 7 | task 13 |
| the environment variables carry a prefix taken from the rig's id (`HYD_TRAINING`, `HYD_ENDPOINT_FILE`, `HYD_JUDGE_ENDPOINT_FILE`, `HYD_GRID_LINE` and others) | among them the run ledger, the run record and other files of the driver, the folder of AssetOpsBench's rows, the training scripts and `CausalDiagnosis/cluster_jobs/` | a neutral prefix, the old names read for one release; the variable of a data path stays the one that the asset file declares | 8 to 12 | a task of its own |
| two commands are bound to the package of the rig (`build-features-and-blocks`, `verify-data-facts`) | `CausalDiagnosis/__main__.py` | a dataset package declares its commands, and the entry takes the dataset id first, as the download command does | 3 to 4 | a task of its own |
| the example settings of the cluster jobs default to the names of one agent model and one judge model, which the serving file also holds and the page of the jobs repeats in its commands | `CausalDiagnosis/cluster_jobs/env.example.sh`, `CausalDiagnosis/cluster_jobs/README.md`, `CausalDiagnosis/experiments/table_definitions/serving_of_record.json` | the settings read the names from the serving file, and the page points to it | 2 to 3 | a task of its own |

## Tables, viewer and training

| Written for one dataset today | Where | The way out | Builder hours, first estimate | Joins |
| --- | --- | --- | --- | --- |
| the strata of one table are the fault counts of the rig, one fault and two | `CausalDiagnosis/evaluation/result_tables/tables_from_runs/episode_selection.py` | the strata from the fault rule of the asset's sampler | 3 to 4 | a task of its own |
| six table modules read `DEFAULT_ASSET` | in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/`: `__init__.py`, `agents_table.py`, `cell_statistics.py`, `comparison_split_tables.py`, `episode_files.py` and `training_tables.py` | the asset of the runs that a table reads | 6 to 8 | with the item on `DEFAULT_ASSET` above; task 13 owns none of these files |
| the viewer holds an example with the components of the rig and binds the tool names in code | `CausalDiagnosis/episode_viewer/walkthrough/formatting.py`, `episode_files.py` | the binding from the declaration, the example with made-up words | 2 to 3 | task 13 |
| two training entries build the environment class of the rig, and one of them reads the default asset | `CausalDiagnosis/training/gepa_prompt_optimization/loop_settings.py`, `CausalDiagnosis/training/rejection_sampling_examples.py` | `make_env` and the asset of the configuration | 4 to 6 | a task of its own |
| docstrings of shared modules name the rig, and the examples of one smoke run name its components | among them `CausalDiagnosis/datasets/dataset_loader/profile_data.py` and `CausalDiagnosis/training/train_lora.py` | name no asset, and made-up words in an example | 1 to 2 | the name scan of task 15 finds those that hold an asset id; a word of the asset without the id is found by reading |

## Numbers of a design

The weights of the reward, the numbers of the label and the numbers of the diagnostics server (samples, folds, cuts) are constants of the modules that use them. They belong to no dataset, so no rule above moves them. The way out is one settings file per part, each number with its reason, read by the code and written into every result as today. It is a task of its own, 6 to 10 builder hours as a first estimate, and it changes no number.

## What this page does not hold

The agent's own package, which the plan page covers. The files of the rig under its own folder, which are declared data and stay where they are. The numbers that are true constants with a source. This page was written from a search of the shared code for the words of the rig and for constants, defaults and literal comparisons; a value of the rig written as a bare number inside a function body is not found that way, so the owner of a file reads it in full.
