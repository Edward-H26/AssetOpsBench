# Assets and ports

The folder `CausalDiagnosis/datasets/` holds the asset layer and one folder per asset. An asset enters the environment as an asset file `CausalDiagnosis/datasets/<asset_id>/asset.json` and a package, and the asset file states the rules of the asset, which `CausalDiagnosis/datasets/dataset_loader/rule_cards.py` calls cards. [CausalDiagnosis/docs/adding-a-dataset.md](../docs/adding-a-dataset.md) describes the parts, the contract and the steps of a port. The page below holds the two tables of the contract that a port fills in file by file, and the design questions of a port that touch a rule of the hydraulic rig.

## How to use it

```bash
python -m CausalDiagnosis datasets download hyd_1 --download
python -m CausalDiagnosis datasets new-dataset-scaffold --help
```

Everything of one dataset lives in its folder `<asset_id>/`: the asset file `asset.json`, the package `__init__.py`, the data script `download_and_build.py`, the folder `data/`, which keeps only `CHECKSUMS.txt` in the repository, and the tests. `python -m CausalDiagnosis datasets download <asset_id>` runs the data script of any dataset. One file of the data folder of hyd_1 comes from a framework: `scenarios_17.json`, the hydraulic scenarios of AssetOpsBench, which the command `python -m CausalDiagnosis assetopsbench download-scenarios` writes there and the preset `anchored` reads. The shared code that reads any dataset is `dataset_loader/`, and `splits/` holds the rules that divide the starts of a dataset into development, training and test sets.

## What a port provides

| Item | Where | What it states |
| --- | --- | --- |
| Asset file | `asset.json` | every field of the section The asset file of `adding-a-dataset.md`, for any number of components, levels, channels, failure modes, and sensor sets, which `observation_kinds` holds |
| Records | `records` in the asset file | `data_env`, `release_dir`, `release_file`, `page`, `fetch`, and the keys of the evidence kind, which are `label_keys` and `stable_flag_stable` for recorded levels |
| Data | the script under `CausalDiagnosis/datasets/<asset_id>/` | `fetch` is one command, run from the root of the repository, that downloads the source from the official page named by `page`, checks it, and builds the release file. `data/CHECKSUMS.txt` holds one `sha256sum` line for every file the command downloads and every file it builds. No dataset file enters the repository |
| Statistics | `statistics` in the asset file | the statistics the data holds per channel and the ones a reading serves. The served statistic listed ahead of the others is the plain one. A telemetry record stores it under the bare field, the normal ranges of `normal_ranges` and the alarms read it, and the texts name it in the plural with an s appended |
| Texts | `text` in the asset file | the nouns, the whole asset patterns, the fallback entry with the site, and an example channel. Five sentences are optional, `question`, `opening`, `scoring`, `level_meaning`, and `readings`. An asset whose evidence kind declares no loop must state `readings` |
| Evidence kind | `__init__.py` | `recorded_levels` needs nothing. Another kind enters with `kinds.register_kind` and the fields of the next table |
| Rule kinds | `__init__.py` | a kind of rule that no built in kind covers, added with `rules.register_kind` |
| Labels | `__init__.py` | `LABELS = "module:function"`, the function that gives the diagnosability label of the initial combination of an episode in the form the verifier reads, a mapping with the boolean `separable` and, under `e18`, a second verdict with a boolean `separable` of its own, the verdict of a second rule, or `None`. The verifier credits a declaration only in an episode whose initial combination both verdicts mark not diagnosable within the budget, so a port with one rule states the same verdict under both. Without the declaration every initial combination counts as diagnosable within the budget |
| Generated tables | `__init__.py` | `TABLES`, a tuple of `"module:ATTRIBUTE"` strings, one per path of a table that code of the port computes on first use, each path defined through `dataset_packages.table_path`. A test session then reads and writes copies, in its own process and in the processes that its tests start |
| Accepted texts | `CausalDiagnosis/datasets/<asset_id>/accepted_texts.json` | the coincidences between a text an agent reads and a combination of the asset that the test of agent texts accepts, each with its `file`, `text`, `count` and `why` |

## What an evidence kind states

| Field of `kinds.EvidenceKind` | Meaning |
| --- | --- |
| `name` | the value of `evidence_kind` in the asset file |
| `plays_episodes` | true when the engine plays the episodes of the kind. The asset file then holds every table of the engine and builds an episode configuration |
| `replays_repairs` | true when the records show the asset after a change of its condition. When false, no configuration of the asset may switch a rule of the deterioration layer on, and from the close of the first work order on the engine plays no recording |
| `record_keys` and `check` | the keys of `records` that the kind needs beside the data keys, and a function that returns the problems of a raw asset file that only the kind can find |
| `loader` | `loader(path, profile)` returns the records, a list that carries `facts_sha256` and whose items each hold an integer `index`, and the blocks, a mapping from a combination to the indices of the records that show it |
| `source` | `source(profile, records, blocks)` returns an object with the protocol `EvidenceSource` of `evidence_source.py`. Its `play` may return `None` for a cycle without a recording, and an optional `fix_history(state)` returns the recorded work orders of an initial state |
| `truth` | `truth(profile)` returns an object with the protocol `TruthModel` of `ground_truth.py`. The truth of recorded levels is the default |
| `loop` | `loop(profile)` returns the number of recordings that an unchanged state replays and the number that an episode shows from before its first cycle. A kind whose source does not loop leaves it out |

The engine counts time in cycles on every asset, one record a cycle. An asset whose kind states a loop states no sentence on its readings, since the kind gives that sentence.

## Design questions of a port

A port of a further dataset must answer four design questions that touch a rule of the hydraulic rig. Each question states the rule on the rig, the situation on the new dataset, the options and the answer in force. An answer marked default follows the rule of the rig, stated for any asset, and holds until the question is settled.

### 1 Whether an agent receives a table that links channels to components

The question applies to the predictive maintenance dataset of IndustryAssetEQA and to any asset whose component names carry no meaning.

| | |
|---|---|
| Rule on the rig | The environment serves no table from sensors to components and no reference of the value a channel takes at a fault level. An agent can read the link from names, such as the channel of cooling efficiency and the component cooler. From the repair rules layer on, the repair manual names the indicating channels. AssetOpsBench serves no such table either. |
| Situation on the dataset | The source names its components comp1 to comp4 and gives them no meaning, so no agent can read the link from names. |
| Option A | An association table says which channel moves, and in which direction, before each component fails. A fixed rule estimates the table on development sessions only, and the asset profile serves it to every agent and to the CausalDiagnosis MCP. Rows without the CausalDiagnosis MCP can then reason from readings to components. The option needs a development set and one more hook in the extension points. |
| Option B | Nothing is published. Without the CausalDiagnosis MCP an agent can see that a machine is off and can name the component only by guessing. The comparison of agent designs on the dataset is then expected at the floor and is reported as uninformative, and the comparison with and without the CausalDiagnosis MCP measures who holds the link. |
| Option C | A table is written by hand from all machines. The option is not allowed, since test labels would shape public knowledge. |
| Answer in force | Default, option B, the rule of the rig stated for any asset. |
| Status | Open. The default is in force until the question is settled, before the port of the dataset is built. |

### 2 What follows a call that breaks a requirement that no public text states

The question applies to FaultArena, where cases hold a step that needs the machine on or off although the tool files do not say so.

| | |
|---|---|
| Rule on the rig | An episode ends only on a condition that the task description or a tool description states. On the rig the task description, which `CausalDiagnosis/environment/Causal_Diagnosis_Environment/briefing.py` builds, lists the finish call, the horizon, the step cap, a limit on the wall time of the agent, the budget where a configuration sets one, and the failure rule where it is strict. A call that the rules cannot serve is refused with a named message and counts one step. The rig holds no hidden requirement. |
| Earlier rule for the dataset | Any wrong power state ends the episode as a failure. Under that rule a correct call can end the episode for a reason that no agent could know. |
| Option a | A wrong power state ends the episode only when the tool files state the requirement. Otherwise the call is refused, is charged what the task description states for a refused call, and the refusal names the state the step needs. On the rig a refused call is charged nothing and counts one step. |
| Option b | The earlier rule holds. |
| Option c | The task description of each case lists the requirements of the case. The agent then learns which steps the script holds. |
| Option d | A matched call ignores an unstated requirement. The script can then answer that a lamp is lit while the power is off. |
| Related case | A power tool is called in the state it would produce, such as power off while the power is off. The call is refused under the same charge rule, and the refusal says that the machine is already in that state. Nothing unsafe happened. |
| Answer in force | Default, option a with the related case as written, the rule of the rig stated for any asset. The answer narrows the earlier rule for the dataset. |
| Status | Open. The default is in force until the question is settled, before the port of the dataset is built. |

### 3 What of the runs may be published

The question applies to FaultArena and to the three datasets of IndustryAssetEQA.

| | |
|---|---|
| Rule on the rig | Every dataset is named by its official page and one fetch command, and no dataset file and no file derived from single records enters a repository. Of a run the repository publishes the aggregate tables and no run folder. |
| Situation | The release of FaultArena grants no redistribution. A run folder holds fault reports, observation texts and reference answers. |
| Option a | Aggregate tables only. |
| Option b | Aggregate tables, and one row per case and agent with scores and counts, keyed by the case identifier, with no text. The decisions of the matcher per call are not published and enter a report as totals. |
| Option c | Full run folders. The license does not allow the option. |
| Answer in force | Default, option a. Option b needs the finding that a score row keyed by a case identifier holds no content of a dataset record. |
| Status | Open. The default is in force until the question is settled, before any run of a port is published. |

### 4 Where the files of the predictive maintenance dataset come from

The question applies to the predictive maintenance dataset of IndustryAssetEQA.

| | |
|---|---|
| Rule on the rig | Every dataset is named by its official page and by one command that fetches it and checks its SHA-256. |
| Situation on the dataset | The files of 100 machines over one year, the telemetry file and four more, were served by hosts of the publisher that no longer resolve. A copy is on Kaggle, republished by a third party, with the license field Unknown. An archived repository of the publisher holds a larger release of the same guide, 1,000 machines, as Git LFS files under the license CC BY 4.0 of the repository, whose scope over data is not stated. No official page grants a license for the data. |
| Option a | The files are fetched from the Kaggle copy, named as a third-party mirror, with a pinned version and SHA-256. The option departs from the rule of official pages. |
| Option b | The release of 1,000 machines is fetched from the archived repository of the publisher. The source is official. The files differ from the files of 100 machines, so the counts of a design made on those do not carry over. |
| Option c | The dataset is left out of the ports. |
| Answer in force | Default, no fetch command names an unofficial page, so the port of the dataset is not built until its source is named. |
| Status | Open. One of the three options is chosen before the port of the dataset is built. |

The rules that every asset follows, and the contract of a port with its rule of the three places, are in [CausalDiagnosis/docs/adding-a-dataset.md](../docs/adding-a-dataset.md).
