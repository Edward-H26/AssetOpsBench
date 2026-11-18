# General agent: the tasks

[general-agent-plan.md](general-agent-plan.md) is the design of the general agent. This page cuts it into tasks that a person and a coding agent can do alone and hand back. Each card says what to read, which files the task owns, what it builds, which tests it writes, and what done means as checks that a third person can run. A path that starts with `CausalDiagnosis/` exists in the tree at the commit that added this page. Every other path is relative to that folder, or to the folder named just before it; where it names a file that the tree at that commit lacks, a piece of the plan page adds the file or a task creates it, or, for the shared file of the core texts, the maintainers add it; `plans.json` is a file that a run writes into its folder.

## Setup

Follow the Quick start of [README.md](README.md): the slim environment, the data file of the rig, and the tests. Run every command from the root of the repository with `PATH=$PWD/.venv/bin:$PATH` in front, as that section does. Once the first piece of the plan page is in the tree, the tests of the scripted cases use made-up cases and need no release of FaultArena. A job with a served model has rules of its own for the node it runs on; read [the page of the cluster folder](../cluster_jobs/README.md) before the first one. No task below needs a served model before the gate. Several checks name the folder of the new package as `CausalDiagnosis/agents/*_next`, which the shell fills in, and some select tests by a word of their name with `-k`; on a tree without that folder such a command stops with an error, and a check that selects no test has not passed. A check has passed only when the tests that its card names stand in its output. On the tree at the commit that adds this page the test `test_no_module_of_the_package_is_loaded_under_two_names` of `CausalDiagnosis/utils/tests/test_import_root.py` is red for a cause outside every task; the card of step 0 names the cause, its mend is a change of its own that the maintainers add ahead of the pieces, and until then a check that runs `CausalDiagnosis/utils` shows that test red.

## Rules that hold for every task

1. A task owns its files and edits no other. A change in a file of another task is a pull request of its own that names the owner, or a request in the owner's pull request.
2. Every test has a control: it fails without the change it tests. The pull request names the control of each test.
3. No test is weakened or deleted to make it pass. A test that the plan retires stands in the test map with its reason.
4. Shared code names no dataset, and a rule is stated for any asset.
5. Every text of the repository (docs, comments, docstrings, test names, tool texts, commit messages) names no person, no clock time and no tracking number of the work, speaks in no first person, cites no line number and states no measured result. American English, no em dash and no en dash.
6. Nothing observed on a start of the test set shapes the agent. Development starts alone.
7. No cap, limit or timeout that no source gives.
8. The present agent and every setting of it stay playable and tested at every commit until the last change. A pull request that breaks one is not done.
9. No signature of the plan page changes without a pull request to the owner of step 0.
10. A pull request names every stored table and every run identity that it moves, by the section Identity and stored tables of the plan page.

## How work comes back

A task is done on a branch that starts at the commit which holds step 0, or at the commit which holds the task it needs, and comes back as one pull request. [CONTRIBUTING.md](../../CONTRIBUTING.md) describes how a fork and a pull request are made; where that file differs from this page on the commit a branch starts from or on the size of a pull request, this page holds for the tasks below. The pull request says what was built, shows the checks of the card with their output, names the control of each test and what it moves, and lists what is open. A checker who did not write the change reads it against the checks of the card. A maintainer of the repository merges it.

## Order

Step 0 comes first, and its two parts land apart. The freeze needs none of the four pieces of the plan page, which the tree at the commit that adds this page does not hold, and is built on that tree. The move follows once the first piece is in the tree. The packages run in parallel once the freeze is in the tree: tasks 2, 4, 5 and 6 need nothing else to start, and every other task waits as the tables below say. Of the two tables the column Waits for of the table of sizes holds: it says which part of step 0 and which pieces a task waits for, also where the table of packages or a card says step 0 or names less. Before the freeze the environment of the Setup section can be built and every card can be read; the cards of tasks 1, 3, 9, 12 and 13 name files that come with the pieces. The hashes of task 11 are taken on the commit that the branch of that task starts from.

| Package | Tasks | Starts | Waits for |
| --- | --- | --- | --- |
| declaration and kinds | 1, 2, 3 | after step 0; task 3 after task 1, since both edit the package of the made-up port | the first and second pieces for tasks 1 and 3 |
| core of the agent | 4, 5, 6, 7 | tasks 4, 5 and 6 after step 0, against the fixture declarations | task 7 needs the functions of tasks 4 and 5, task 2 for its episodes on the rig and task 3 for its episode on the made-up port; the tests of the map that come with the fourth piece wait for it |
| advice modules and binding | 8, 9, 10 | task 10 after step 0; the unit tests of tasks 8 and 9 on the fixtures | task 10 needs task 2 for the flags; the episodes of task 8 need tasks 2 and 7 and those of task 9 tasks 3 and 7; task 9 reads files of the first and second pieces |
| designs, identity, driver | 11, 12, 13 | task 12 after step 0; the hashes of task 11 first, on the commit that its branch starts from | task 11 needs the builder of task 1, and the third piece for its episodes on the made-up port; task 12 needs task 3 for one line; task 13 needs tasks 1 and 7 and the third piece, and the fourth piece for one test of the map |
| gate and last change | 15, then the blind port | after every other task | the blind port needs the last change |

No two packages own one file. There is no task 14: the folder move that it would hold is part of step 0.

**Size.** A first estimate of each task in builder hours, from the size of the code that it replaces or moves, without checking and integration. It is an estimate and no result.

| Task | Builder hours, first estimate | Waits for |
| --- | --- | --- |
| step 0, the freeze | 6 to 8 | nothing beyond the tree of this page |
| step 0, the move | 4 to 6 | the first piece |
| 1, the contracts and the builder of the declaration | 14 to 18 | step 0, both parts; the first and second pieces |
| 2, the rules of recorded levels | 20 to 28 | the freeze |
| 3, the rules of scripted cases, the shared loader, the advice answers | 6 to 8 | task 1; the first and second pieces |
| 4, the plan reader and the check | 24 to 32 | the freeze; the fourth piece for the tests of the map that come with it |
| 5, the working memory and the reading of answers | 20 to 27 | the freeze; the fourth piece for the tests of the map that come with it |
| 6, the prompt | 15 to 20 | the freeze |
| 7, the loop | 28 to 38 | tasks 4 and 5; task 2 for the episodes on the rig, task 3 for the episode on the made-up port; the fourth piece for the tests of the map that come with it |
| 8, the ranking module | 24 to 32 | the freeze; tasks 2 and 7 for the episodes |
| 9, the warrant module | 10 to 14 | the freeze; the first and second pieces for its reading; tasks 3 and 7 for the episodes |
| 10, the binding | 10 to 14 | the freeze; task 2 for the flags |
| 11, the two reimplemented designs | 14 to 20 | task 1; the third piece for the episodes on the made-up port |
| 12, identity and stored tables | 14 to 18 | step 0, both parts; task 3 for one line |
| 13, the driver, the viewer and the tables | 20 to 28 | tasks 1 and 7; the third piece; the fourth piece for one test of the map |
| 15, the gate and the last change | 20 to 28 | every other task |
| the blind port | not estimated | the last change |

The work reaches four states in this order. First, one episode of the rig on its own action server without the diagnostics server, through the driver: step 0 and tasks 1, 2, 4, 5, 6 and 7 with the agent's entry of task 13. Second, the six settings on the rows of AssetOpsBench's servers with the new agent beside the present one: tasks 8, 9 and 10, the readers of task 13 and the comparison of task 15. Third, a second kind: task 3. Last, the gate on three assets and the last change, after tasks 11 and 12.

**What a task moves.** A changed Python file under `CausalDiagnosis/environment/` or under `CausalDiagnosis/datasets/` outside an asset folder moves the label digest of every asset, and the tables of the labels are computed again. A changed Python file of the package of the diagnostics server moves the digest of the fitted tables. A changed file of the lists `harness`, `episode` or `scoring` moves the identity of every run, and a changed file of an agent list the identity of the runs of that agent. So:

| Tasks | Label tables | Fitted tables | Run identity |
| --- | --- | --- | --- |
| step 0, the freeze | stays | stays | every run |
| step 0, the move | moves | stays | every run |
| 1, 3 | moves | moves | every run |
| 2, 5, 10, 13 | stays | stays | every run |
| 4, 6, 7, 8, 9 | stays | stays | the runs of the new agent alone |
| 11 | stays | stays | the runs of the two reimplemented designs |
| 12 | moves | moves | every run |
| 15, the last change | stays | moves | every run |

## Step 0: the folder move and the freeze of the interfaces

**Goal.** Give FaultArena's own files their framework folder with the mechanism that keeps the asset registered, and put every interface of the plan page into the tree as signatures with their fixture declarations, so that every task builds and tests against them without waiting for another. It is built by the maintainers of the repository, and its two parts land apart: the freeze (part 2) first, on a tree without the pieces, and the move (part 1) once the first piece is in the tree.

**Read first.** The sections Folders, The declaration, Rules of a kind, The plan and Advice modules of the plan page; `CausalDiagnosis/utils/frameworks.py`; `CausalDiagnosis/datasets/dataset_loader/dataset_packages.py`; `CausalDiagnosis/experiments/run_verification/run_identity_file_lists.py`.

**Part 1, the move.** It is no mechanical commit alone: a bare move leaves the asset unregistered and outside every run identity.
- First check, before any file moves: plant one folder with one test under a framework folder, run pytest from the root and `CausalDiagnosis/utils/tests/test_import_root.py`. If the asset package appears under a second import name, stop. In the tree at the commit that adds this page one test of that file, `test_no_module_of_the_package_is_loaded_under_two_names`, fails for a cause outside the move: where the environment lies inside the root of the repository, it also counts an alias that a third-party package registers for its own modules. Its mend is a change of its own that the maintainers add ahead of the pieces, so this check is read on a tree in which that test passes before the plant.
- The field `DATASETS` in `CausalDiagnosis/utils/frameworks.py`; the mount in `CausalDiagnosis/datasets/__init__.py`; `frameworks/FaultArena/__init__.py` with `DATASETS`, `COMMANDS` and `LISTS` (the lists `analysis`, `harness`, `episode`, in that order, so that every moved file keeps its list) and a `README.md`; a test in `CausalDiagnosis/utils/tests/test_framework_boundary.py` that no file imports or names the asset package through the framework folder.
- The moves, both of folders that the first piece adds: `datasets/faultarena/` to `frameworks/FaultArena/datasets/faultarena/`, `mcp_servers/FaultArena_Actions_MCP/` to `frameworks/FaultArena/action_server/`. No `__init__.py` in `frameworks/FaultArena/datasets/` and none in the tests folder of the dataset.
- What follows the move, each a line that the first piece adds: the name of the action server in the package; the folder of the release in the asset file; the two scripts that find the root of the repository by counting parent folders; the entry of the action server in `CausalDiagnosis/utils/tests/test_command_table.py`; the command of the matcher audit, which leaves `CausalDiagnosis/__main__.py` for the declaration; the docs that name the old paths. The roots of the scan of the texts that an agent reads, in `CausalDiagnosis/agents/tests/test_agent_texts.py`, gain the two new folders, since the moved files leave the roots that the scan names.
- Every place that finds the datasets by listing `CausalDiagnosis/datasets/` asks `dataset_packages.asset_dirs()`: the tests that list asset files by folder and the data job `CausalDiagnosis/cluster_jobs/0_setup/fetch_data.sbatch`, which lists that folder once the first piece is in the tree and until then calls the data script of the rig by name. Every place that names the data folders of the datasets by a pattern gains the folders of the datasets of a framework: `CausalDiagnosis/.gitignore`, the generated locations of `CausalDiagnosis/utils/paths.py` with their test, and the data rule of `CausalDiagnosis/experiments/run_verification/upload_manifest.py`.
- The same commit rewrites the paths that this page and the plan page write relative to `CausalDiagnosis/` into full paths from the root, for the files that exist by then, so that `CausalDiagnosis/experiments/episode_runners/tests/test_paths.py` covers them.

**Part 2, the freeze.**
- `agents/utils/declaration.py` and `agents/utils/answer.py`, complete, and `agents/utils/answer_forms.py` as a stub with its signature, all three written into the list `harness`, so that no later task edits the lists file for them.
- The package `agents/Causal_Diagnosis_Agent_next/` with `plan/plan_schema.py`, `agent_options.py` and `episode_result.py` complete, and as stubs that hold every signature and docstring of the plan page and raise `NotImplementedError`: `plan/readers.py`, `plan/reply_parser.py`, `plan/plan_check.py`, `plan/rules.py`, `plan/reply_schema.py`, `schedule.py`, `working_memory/board.py`, `prompt/core_protocol.py`, `prompt/assembler.py`, `advice/__init__.py`, `advice/ranking.py`, `advice/warrants.py`, `controller.py`.
- `utils/public_rules/plan_rules.py` with `KindRules`, `Ledger`, `Walk`, `StepCost` and `Needs` as stubs.
- The standing step and the schema of a reply of the plan page are part of the freeze: the value `standing` of `StepRecord.status`; `Standing(clause, reason)` in `plan/plan_schema.py`; `standing(step, index, state) -> Standing | None` in `plan/rules.py`, which holds one rule function per row of the table of the check and so none for a call with nothing new; the field `stands` of `StepCost`; `planning_turn` with the keyword `stood=None`; on the board `holds_answer`, `repeated`, `add_note`, `retire_notes` and the field `reply_schema`; the four functions of `plan/reply_schema.py`; and `structured_output` as one more field of `Options` and one more record of `SWITCHES`, the twelfth switch, with no flag, among `OMITTED_AT_DEFAULT`, so that `to_json` writes it where a run turns it off.
- `agents/Causal_Diagnosis_Agent_next/tests/fixture_declarations.py`: four fixtures written as data, each a declaration with its listing: the rig's on its own action server; the rig's on a row of AssetOpsBench's servers, with a stub of the binding that holds its carried map and the rows it states; the made-up port's, from the rows of the plan page with the defaults of a row applied; and one with a single tool. The second is built on the preset base, whose task server lists no clock and no wait, and once more from the listing of a preset with a clock, so that a wait through a binding has its fixture. The rig's own holds the key `omitted` on the four rows that the plan page gives it, the second on those of them whose tools its row lists, and both hold no candidate arguments on the row of `declare_insufficient_evidence`.
- The agent lists. The rule of the list of the present agent is the whole folder `CausalDiagnosis/agents/` today, so every file added under the new package would join the identity of its runs. Step 0 narrows it to `CausalDiagnosis/agents/Causal_Diagnosis_Agent/` and gives the new package a list of its own, `agent:causal_next`. Everything else that a run of the present agent loads lies in the lists `harness` or `episode` already. A helper that the two agents alone share is written into both agent lists.
- The freeze touches no file under `CausalDiagnosis/environment/` or `CausalDiagnosis/mcp_servers/` and no file that exists under `CausalDiagnosis/agents/utils/`.
- A data class of the freeze may gain a field with a default, a function or a constructor a keyword parameter with a default, and a class a method, each through a pull request to the owner of step 0, and a reader of the agent's records ignores a key it does not know.

**Tests, each with its control.** Every fixture declaration builds and joins with its listing, the second on each of its two listings, a carried tool stands in the table where its carrier is listed, a withheld tool stays out of the agent's own table and in the listed one, and the tools of the role advice stand in the table (control: a listed tool without a row raises by name). In every declaration of the rig the table gives `declare_insufficient_evidence` no candidate arguments, and in the made-up port's and the single tool's it gives the answer write its required arguments that no fill writes (control: with the default applied whatever the contract states, that row of the rig gains its sensor group and its count). Two servers of a listing that offer one tool name raise by name (control: without the second server the table builds). Every name of the plan page imports. Every module that the driver loads to reach the present agent lies in its narrowed list or in `harness`, in each of its settings (control: a file added under the new package is in no list of a run of the present agent). The asset is found after the move (control: without the mount it is not). The test of the map under declaration.

**Done means.**
```bash
PATH=$PWD/.venv/bin:$PATH python -c "from CausalDiagnosis.datasets.dataset_loader import dataset_packages as p; print(p.asset_dirs()['faultarena'])"
PATH=$PWD/.venv/bin:$PATH python -m CausalDiagnosis faultarena
PATH=$PWD/.venv/bin:$PATH python -c "from CausalDiagnosis.agents.utils import declaration, answer"
PATH=$PWD/.venv/bin:$PATH python -m pytest CausalDiagnosis/utils/tests CausalDiagnosis/datasets CausalDiagnosis/agents -q
PATH=$PWD/.venv/bin:$PATH python -m pytest CausalDiagnosis/experiments/table_definitions/tests/test_code_lists.py CausalDiagnosis/experiments/episode_runners/tests/test_agent_files.py CausalDiagnosis/experiments/episode_runners/tests/test_paths.py CausalDiagnosis/experiments/run_verification/tests/test_generated_locations.py -q
```
The first command prints a folder under `frameworks/FaultArena/`, the second lists the matcher audit, and every test passes. The first two commands show the move. The freeze, which lands first and on a tree without the pieces, is shown by the third command, by the tests of `CausalDiagnosis/agents` in the fourth and by the fifth; on that tree the first two commands cannot pass yet, and `test_no_module_of_the_package_is_loaded_under_two_names` of `CausalDiagnosis/utils/tests/test_import_root.py` stays red in the fourth until its mend, which the maintainers add ahead of the pieces, is in the tree.

**Needs.** The freeze needs nothing beyond the tree of this page. The move needs the first piece of the plan page. **It moves.** The freeze moves the identity of every run. The move moves it once more and the label digest of every asset: the tables of the labels are computed again after it.

## Task 1: the contracts and the builder of the declaration

**Goal.** Every contract states roles, write classes and row keys, and `declaration_of` builds the declaration of every registered asset from files alone.

**Read first.** The section The declaration of the plan page; `CausalDiagnosis/environment/Causal_Diagnosis_Environment/tool_contract.py`; `CausalDiagnosis/frameworks/AssetOpsBench/tool_servers/aob_tool_contract.py`; `datasets/dataset_loader/tool_roles.py` of the first piece.

**Owns.** `CausalDiagnosis/environment/Causal_Diagnosis_Environment/tool_contract.json` and `tool_contract.py`; `CausalDiagnosis/frameworks/AssetOpsBench/tool_servers/aob_tool_contract.json` and `aob_tool_contract.py`; `datasets/dataset_loader/tool_roles.py`; in `CausalDiagnosis/datasets/dataset_loader/`: `evidence_kinds.py`, `rule_cards.py`, `dataset_packages.py`, `profile_data.py`, `profile_validation.py`; `datasets/dataset_loader/scripted_case_replay/interface.py` of the first piece; `mcp_servers/Causal_Diagnosis_MCP/backends/scripted.py` of the second; the asset files, packages and new contract files of the scripted assets and of the two made-up ports that the pieces bring, `bottler_1` under the tests of the replay and `filler_1` under the tests of the backends; `experiments/episode_runners/run_episodes/declarations.py`.

**Builds.** The keys of the rig's rows and the block `roles` as the plan page writes them. The role tables of the scripted assets as contract files named by `ACTION_CONTRACT`, with the block gone from their asset files and from the profile. The check of a contract's rows, with the keys equal to the fields of `ToolRow` that a contract may state. The field `plan_rules` of an evidence kind and `plan_rule` of a rule kind. `load_contract` writes the role advice and the kind of the record into each entry it builds from a record. The reader of a contract takes the rows of the servers of the run alone and refuses no row of a class that it does not know on a server outside the run. `declaration_of` with its refusals by name: a row that its keys do not fit, no write that ends the episode, a kind that names no rules, an advice tool whose kind is none of the four.

**Tests, each with its control.** For every registered asset the declaration builds, and with the listing of its servers the table holds a row for every listed tool (control: one row removed from a contract raises a `DeclarationError` that names the tool). A key on a write class that it does not fit is refused by name, one case per key. On the rows of AssetOpsBench's servers the listed table gives each role the tools of one server (control: without the block `roles` it does not). The tests of the replay and of the backends pass unchanged after the role tables moved.

**Done means.**
```bash
PATH=$PWD/.venv/bin:$PATH python -c "
from CausalDiagnosis.datasets.dataset_loader import dataset_packages as p
from CausalDiagnosis.experiments.episode_runners.run_episodes.declarations import declaration_of
for a in p.asset_ids(): print(a, len(declaration_of(a, None).rows))"   # None: the asset's own action server
PATH=$PWD/.venv/bin:$PATH python -m pytest CausalDiagnosis/datasets CausalDiagnosis/environment/Causal_Diagnosis_Environment CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP CausalDiagnosis/frameworks/AssetOpsBench/tool_servers CausalDiagnosis/experiments/episode_runners/tests -q
```
The first command prints one line per registered asset with its count of rows, and every test passes.

**Needs.** Step 0. **Must not touch.** The asset file of the rig; the present agent; the driver beyond its one new file; the parts of the built in kind `scripted_cases` in `evidence_kinds.py`, which task 3 sets after this task.

## Task 2: the rules of recorded levels

**Goal.** The prices, durations, visits, closes, paid units and verdicts that the present board and plan check compute for the rig stand behind `KindRules`, `Ledger` and `Walk`, so that no module of the agent reads a key of the profile.

**Read first.** The section Rules of a kind of the plan page; in `CausalDiagnosis/agents/Causal_Diagnosis_Agent/`: `situation_board/board.py`, `situation_board/asset_rules.py`, `situation_board/read_walk.py`, `plan/plan_check.py`; `CausalDiagnosis/utils/public_rules/repair_completion.py` and `visits.py`.

**Owns.** `utils/public_rules/plan_rules.py` after step 0, `utils/public_rules/recorded_levels.py`, and their tests under `CausalDiagnosis/utils/`.

**Builds.** The class of the kind with its ledger and walk, the rule functions of the cards `visits` and `dependencies`, the end guard with the note for a row that leaves no way to the close, the kind's own verdict on a repair by the normal ranges, the ranges that the reader of answers takes from the ledger, and the board sections of the kind, with the table of what the record holds and what the board holds of it. The walk states `nothing_new` for a call that would serve nothing that the ledger does not hold while the record cannot grow, by the test that the present agent applies before a read (`Board.new_cycles` and `Board.clock_runs`). `repair_completion.py` and `visits.py` are read as they stand.

**Tests, each with its control.** The tests of the map under kind rules, each on the ledger and no longer on a board of the agent (control: the rule it holds switched off). For every registered asset whose kind holds open targets, the close that the ledger estimates is the close that the environment's rule gives; an asset whose kind holds none is asserted to hold none. The walk states `nothing_new` for a series whose every recorded unit is paid where no time passes, and such a step has no price, no time and no call (control: with one recorded unit unpaid, and in the same state where time passes, the step has the cost of a call).

**Done means.**
```bash
PATH=$PWD/.venv/bin:$PATH python -m pytest CausalDiagnosis/utils -q
PATH=$PWD/.venv/bin:$PATH python -m pytest CausalDiagnosis/agents -q -k "public_rules"
git grep -n -w -e sensor_groups -e normal_ranges -- "CausalDiagnosis/agents/*_next/*" ":!CausalDiagnosis/agents/*_next/*tests/*"
```
The tests pass, the kept test of the package of public rules among them, and the last command finds no key of the profile in the code of the new package, its tests left out (run it for the present package first to see that it finds some).

**Needs.** Step 0. **Must not touch.** `repair_completion.py` and `visits.py` of the same folder, which the present agent and the diagnostics server read; the present agent.

## Task 3: the rules of scripted cases, the shared loader, the advice answers

**Goal.** The precondition card is walked by public rules, a dataset of scripted cases in the shared form brings no code, and the fields of an advice answer have one home.

**Read first.** The sections Rules of a kind and Advice modules of the plan page and, once the first two pieces are in the tree, `datasets/dataset_loader/scripted_case_replay/` (`interface.py`, `texts.py`, `rows.py`, `tests/bottler.py`) and `mcp_servers/Causal_Diagnosis_MCP/backends/__init__.py`.

**Owns.** `utils/public_rules/scripted_cases.py`, `preconditions.py`, `advice_answers.py`; in the replay `texts.py`, `rows.py`, the new `case_files.py` and `tests/bottler.py`; `mcp_servers/Causal_Diagnosis_MCP/backends/__init__.py`.

**Builds.** The class of the kind: the walk of the card, the state after every state write, the steps left from the receipt. The walk states `state_holds` for a state write whose value the state already has at that step, and gives that step no call and no reason of the card. The moved readers, each with one writer and one reader held by a test: the requirement of a call, the words of an answer, the sentence of the state at the start with `initial_state`, the refusal phrase. `case_files.load_records`, and the made-up port moved to the built in kind. One ordered exception to the rule of one owner: after task 1 has landed, this task gives the built in kind `scripted_cases` in `CausalDiagnosis/datasets/dataset_loader/evidence_kinds.py` its parts, as references to the replay and to the new loader, so that no reference names a module that does not exist yet. `advice_answers.py` with the tuple of the kinds, the fields per kind as tuples and the reader of warrants; the backends import them and keep their own names, and the replay drops its own copy of the kinds.

**Tests, each with its control.** The walk refuses a call in the wrong state and passes it after the state write (control: the card removed). A state write to the value that the state already has stands, and the calls after it are walked in that state (control: without the clause the walk gives that step the reason of the card). The state at the start is read from the task message of every case of the made-up port (control: the sentence changed in the briefing alone fails the test that holds writer and reader together). Every refusal text of the kind holds the refusal phrase. A copy of the made-up port under another asset id, mounted from a temporary folder, loads its own cases with no module of its own (control: the same copy with the kind of the port's old loader reads the wrong files). The tests of the backends pass unchanged.

**Done means.**
```bash
PATH=$PWD/.venv/bin:$PATH python -m pytest CausalDiagnosis/utils CausalDiagnosis/datasets CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP -q
PATH=$PWD/.venv/bin:$PATH python -m pytest CausalDiagnosis/agents -q -k "public_rules"
```

**Needs.** Step 0 and task 1. **Must not touch.** The answers that the action server of scripted cases writes; the present agent; the fixture of the made-up port of step 0, against which the unit tests of the other tasks are written.

## Task 4: the plan reader and the check

**Goal.** Any reply of a model becomes a plan or a refusal with named reasons, and the check refuses exactly what the table, the cards and the limits forbid, in the words of the prompt.

**Read first.** The sections The plan and The check of the plan page; `plan/reply_parser.py` and `plan/plan_check.py` of the present agent with their tests.

**Owns.** `agents/Causal_Diagnosis_Agent_next/plan/` and its tests, which stand in a folder `tests/` inside it.

**Builds.** The typed readers, the two boundary functions with their guard, the walk of the check with one function per rule, the fills, the values of omitted arguments, the cut of a plan at its first step that cannot be used, `standing` with its five clauses, the constants of the reasons, and the schema of a reply in `plan/reply_schema.py`. A rule judges its step from the steps before it and from the live ledger and reads no later step.

**Tests, each with its control.** The generator of the plan page on the fixture declarations, with its count of generated replies printed (control: a reader that skips its type test lets a generated reply reach the guard, and the test fails). Every reason of the check and of the reader has its sentence in the constants that the prompt prints (control: one sentence removed). A reason belongs to its step or to the whole plan as the table says. A candidate and a step that leave out an argument of a key `omitted` are read with its value, and a commit step that writes none of its candidate arguments and no rank follows rank 1 (control: without the key the check names the argument as required). A step of a commit write without candidate arguments that leaves a required argument out is refused for that argument and follows no candidate (control: with candidate arguments on its row the same step follows rank 1). One test per clause of `standing` on the fixture declarations, each with the one fact that falsifies it: an answer that the board does not hold, a status that shows a time, a unit that is not paid, an argument of the commitment with another value, a state with another value (control: the walk then counts the call). A commit write that differs from the commitment and a repair write on an open target are still refused, each with a reason that opens with the change. A plan that fits the step limit only because its standing steps count no call is accepted (control: with one call counted for each standing step it is refused). For every fixture declaration a generator builds replies from the schema of its table, every form, every optional key present and absent and the values at every bound, with texts that hold a visible character and integers within the range of the readers, and the reader reads each with no reason, with the count of generated replies printed (control: a lower bound of the schema set below the reader's gives a generated reply a reason). The schema holds no keyword that the generator does not write and no value of the facts (control: a planted keyword). A value of any type under `evidence` on a step whose row fills no argument from it cuts no plan (control: with the key read on every step a number there cuts the plan). The tests of the map under core that read the parser or the check.

**Done means.**
```bash
PATH=$PWD/.venv/bin:$PATH python -m pytest CausalDiagnosis/agents/*_next/plan -q
```
The output names the count of generated replies, and no test reaches the guard.

**Needs.** Step 0. **Must not touch.** The present agent; `agents/utils/declaration.py`.

## Task 5: the working memory and the reading of answers

**Goal.** The board keeps what an episode holds for any asset, and every answer of a tool is read into one shape without raising.

**Read first.** The sections Reading answers and The loop of the plan page; `situation_board/board.py` of the present agent; `CausalDiagnosis/agents/utils/tool_answers.py`; `CausalDiagnosis/frameworks/AssetOpsBench/tool_servers/metering_receipts.py`.

**Owns.** `agents/Causal_Diagnosis_Agent_next/working_memory/`; `agents/utils/answer_forms.py`; `CausalDiagnosis/agents/utils/tool_answers.py`; `CausalDiagnosis/frameworks/AssetOpsBench/tool_servers/metering_receipts.py`.

**Builds.** The board of the core with the compaction and recall of the present board under their present rules. `tool_answers.read` with the refusal marks. The reader of the receipt line moved into `tool_answers.py`, with the metering layer importing it; that file is the one home of the reader, and a server or a host that needs it imports it there, as the metering layer imports that file today. The render of the board is built on one function that yields its parts in order, so that another reader of the board takes the same parts. The three forms of an answer. The record of the calls behind `holds_answer` and `repeated`, the lines of the status on what the board holds, and the notes with their ties.

**Tests, each with its control.** The generator over the shape of an answer for `tool_answers.read` (control as in task 4). One reader of the receipt serves every writer, the present test kept. The compaction loses no refuted set and no cited identifier. The status names every answered call of the facts tool and of the knowledge tools and a call that was made twice with the same arguments (control: a second call with another argument is not named). Each tie of a note leaves at its change of state and not before, equal notes of plans stand once with their count, a note that a summary named stays, and the order of the entries that a summary replaces is exact after a retirement (control: without the counts of the turn marks following the retirement a planted case misorders). The tests of the map under core that read the board.

**Done means.**
```bash
PATH=$PWD/.venv/bin:$PATH python -m pytest CausalDiagnosis/agents CausalDiagnosis/frameworks/AssetOpsBench/tool_servers -q
```

**Needs.** Step 0. **Must not touch.** What the present agent and the two designs read through `tool_answers.py` and the receipt reader: both keep their names and their behavior.

## Task 6: the prompt

**Goal.** One assembler prints every turn from the core texts, the rendered clauses and the blocks of the modules, and no word of a dataset stands in the core.

**Read first.** The section The prompt of the plan page; `prompts.py` of the present agent.

**Owns.** `agents/Causal_Diagnosis_Agent_next/prompt/` and its tests, which stand in a folder `tests/` inside it.

**Builds.** The constants of the core texts as the plan page writes them, of which those of the shared file `agents/utils/core_texts.py` are imported once that file is in the tree, the rendered clauses, the four turns, the variants of the switches, the section of the standing steps of the planning turn, and `identifier_forms`.

**Tests, each with its control.** For every registered asset the system prompt and the four turns assemble and hold no word of another registered asset (control: a noun of another asset planted in a core constant). The prompt of an asset speaks of no write class, role or event that its table lacks. With a switch off, its block or sentence is absent and no other text changes. The prompts of the fixture declarations are kept as files that the test compares, so that a change of a text shows in review. `planning_turn` with standing steps prints their section after the section of the steps that did not run, and without them prints none (control: the keyword left out). EVIDENCE names the tools that fill an argument from evidence and leaves that sentence out on a table without one, and the sentence of a call with nothing new stands in the line of an observe tool with a series only where the vocabulary holds its terms. With the switch `structured_output` off no text changes. The tests of the map under prompt.

**Done means.**
```bash
PATH=$PWD/.venv/bin:$PATH python -m pytest CausalDiagnosis/agents/*_next/prompt -q
```

**Needs.** Step 0. **Must not touch.** The sentences of the rules, which are constants of task 4: the assembler prints them. The shared file of the core texts, but through a pull request to the maintainers.

## Task 7: the loop

**Goal.** One controller plays an episode of any declaration: it builds the table, asks for plans, runs the steps and keeps the records.

**Read first.** The section The loop of the plan page; `controller.py` and `candidate_plan_schedule.py` of the present agent.

**Owns.** In `agents/Causal_Diagnosis_Agent_next/`: `controller.py`, `schedule.py`, `agent_options.py`, `episode_result.py`, `advice/__init__.py`, and their tests.

**Builds.** The loop, the one step runner with the fills, the batch, the repeated wait, the key `each`, the trigger of the differential, the verification hook, the schedule over the walk of the kind, the registry of switches with its flags, the standing step, which the runner asks before every call, and the schema that the planning request and the orientation request carry, with its record on the board. The step runner runs one step for a caller other than the plan loop, on the same path as a step of a plan.

**Tests, each with its control.** An episode on the rig's declaration and one on the made-up port reach their end with tool plans (control: the end guard off lets the rig's episode end unverified). A refused call ends its batch. The differential is asked once, at the first turn at which the facts and the answer of the tool that states `orients` are on the board (the present test, kept). A step that leaves the argument of `each` empty makes one call per value. On the rig's fixture of the rows of AssetOpsBench's servers a stand-in whose every plan restates the opening reads, an observation with nothing new and the commitment ahead of its repair writes and of the write that ends the episode reaches its end, with standing steps in its records and no surprise from one (control: with `standing` answering None the episode ends at the step limit). A turn of standing steps alone takes one step and is no fallback, and the next request names the steps. An expectation on a standing step that the board as it stands fails is a surprise, and no write after it runs (control: with the judgment left out the write runs). The planning request and the orientation request carry the schema of the table and the two summary requests none, with the switch off no request carries one, and a stand-in endpoint that ignores the field is read as before (control: the field planted on a summary request). The tests of the map under episode and the rest of those under core.

**Done means.**
```bash
PATH=$PWD/.venv/bin:$PATH python -m pytest CausalDiagnosis/agents/*_next -q
```

**Needs.** Tasks 4 and 5; task 2 for the episodes on the rig; task 3 for the episode on the made-up port. **Must not touch.** The present agent.

## Task 8: the ranking module

**Goal.** Everything that reads a ranking lives in one module that asks for the tool of its kind and names no tool.

**Read first.** The section Advice modules of the plan page; in the present agent `plan/read_advice.py`, `plan/advice_lines.py`, `plan/repair_budget.py`, `plan/rule_controller.py`, `situation_board/troubleshooting_plan_fields.py`.

**Owns.** `agents/Causal_Diagnosis_Agent_next/advice/ranking.py`, its tests, which stand in a folder `tests/` beside it, and the stand-in model of the rule under those tests.

**Builds.** The module as the plan page lists its parts. The rule that plays without a model becomes a stand-in model of the tests: it writes its plan as the JSON object that a model would write, so every test that used it passes through the reader and the check.

**Tests, each with its control.** The tests of the map under ranking module, each also on the tiny asset of `CausalDiagnosis/datasets/tests/` (control: the part of the module it holds switched off). For every registered asset with nothing for the schedule to show, the board with the switch `parallel` is the board without it. The forms of a standing observe step are judged on the ranking that the board holds, which the module asks again only when it is stale (control: a ranking call for a standing step whose ranking is current).

**Done means.**
```bash
PATH=$PWD/.venv/bin:$PATH python -m pytest CausalDiagnosis/agents/*_next/advice -q -k "ranking"
```

**Needs.** Step 0 for the unit tests; tasks 2 and 7 for the episodes. **Must not touch.** The core: a need of the module that the core does not serve is a pull request to task 7.

## Task 9: the warrant module

**Goal.** A repair write gets its warrant from the plan assessment of any backend, and a closed repair its verdict from the outcome check where one is listed.

**Read first.** The section Advice modules of the plan page; `mcp_servers/Causal_Diagnosis_MCP/backends/` of the second piece and, in the replay of the first, the rule of the scripted row driver for a call of a plan assessment.

**Owns.** `agents/Causal_Diagnosis_Agent_next/advice/warrants.py` and its tests, which stand in the folder `tests/` beside it.

**Builds.** The module as the plan page lists its parts, with the two inputs of a plan assessment, `action` and `calls`.

**Tests, each with its control.** The tests of the map under warrant module. An episode on the made-up port with the scripted backend holds one call of the plan assessment before the repair write (control: without the module the write goes out unwarranted). A write that no warrant covers is marked.

**Done means.**
```bash
PATH=$PWD/.venv/bin:$PATH python -m pytest CausalDiagnosis/agents/*_next/advice -q -k "warrant"
```

**Needs.** Step 0 for the unit tests; tasks 3 and 7 for the episodes. **Must not touch.** The backends.

## Task 10: the binding of the rows of AssetOpsBench's servers

**Goal.** On a surface whose tools are not the asset's own, one class writes the arguments that the episode fixes and reads the answers, and never chooses a step.

**Read first.** The section Reading answers of the plan page; `CausalDiagnosis/frameworks/AssetOpsBench/agents/causal_diagnosis_agent_surface.py`.

**Owns.** That file, in which the binding is a second class beside the present surface class, and `CausalDiagnosis/frameworks/AssetOpsBench/tool_servers/tool_surface.py`, in which the surface gains the attribute `agent_binding` beside the present one.

**Builds.** The writers and readers of the present surface, one by one, on the interface of the binding; the carried map and the rows that it states without the facts, as the plan page lists them; the switch `binding`.

**Tests, each with its control.** The tests of the map under binding. The window of a read covers unread cycles in one run and never a read one. With the switch off the plan names the listed tools and the binding writes no argument. The binding states the facts tool among the needs of every tool whose arguments it writes from the facts or whose answer it reads with them, and a step on such a tool ahead of the facts tool is held back (control: with that need taken out for a knowledge read, its step passes ahead of the facts tool).

**Done means.**
```bash
PATH=$PWD/.venv/bin:$PATH python -m pytest CausalDiagnosis/frameworks/AssetOpsBench -q
```

**Needs.** Step 0; task 2 for the flags that the rules of the kind read. **Must not touch.** The present surface class, which the present agent plays until the last change.

## Task 11: the two reimplemented designs on the role table

**Goal.** Both designs build their sub-agents from the roles of the listed tools of any asset, and the rig keeps every byte of its texts.

**Read first.** The section The two reimplemented designs of the plan page; in `CausalDiagnosis/frameworks/AssetOpsBench/agents/`: `Agent_As_Tool_Paper/sub_agent_registry.py`, `Agent_As_Tool_Paper/sub_agent_runtime.py`, `Plan_Execute_Paper/runner.py`, `agent_runs.py`.

**Owns.** In that folder: `Agent_As_Tool_Paper/sub_agent_registry.py`, `sub_agent_runtime.py`, `supervisor_runner.py`, `text_completion_route.py`; `Plan_Execute_Paper/runner.py` and `workflow.py`; `agent_runs.py`; the declaration `CausalDiagnosis/frameworks/AssetOpsBench/__init__.py`; their tests.

**Builds.** First of all, on the commit that the branch of the task starts from, the hash of every text that both designs send on the rows of AssetOpsBench's servers. Then the role surface beside the present registry, built once per run from the listed rows of the declaration, the filter of tools, which keeps what the listing of an episode serves, the texts by rule, the served state writes, the record of refused calls, the seeds of sampled calls, and the surface that lets both designs play a kind that the engine does not play.

**Tests, each with its control.** The hashes before and after are equal. On the made-up port and on the rig: a sub-agent per role with exactly the tools of its role; none for a role without a tool; a state write served to the roles that the card names and to no other; a call outside the set logged and never served; both designs play an episode to its end with and without the server (control: without the role surface the entry of the third piece refuses them by name). Two plays of one episode with a sampled call send the same seed. The test of the map under designs.

**Done means.**
```bash
PATH=$PWD/.venv/bin:$PATH python -m pytest CausalDiagnosis/frameworks/AssetOpsBench/agents -q
```

**Needs.** Task 1 for the declaration. **Must not touch.** `in_context_examples.py` of the Agent-As-Tool folder, whose texts stay byte for byte; the present registry until the last change.

## Task 12: identity and stored tables

**Goal.** A table of an asset and the identity of a run cover the code that the asset can load and no other, so that adding a dataset moves nothing of another.

**Read first.** The section Identity and stored tables of the plan page; `CausalDiagnosis/experiments/run_verification/run_identity_file_lists.py`; `CausalDiagnosis/utils/atomic_writes_and_locks.py`; `CausalDiagnosis/experiments/table_definitions/tests/test_code_lists.py`.

**Owns.** `CausalDiagnosis/experiments/run_verification/run_identity_file_lists.py` after step 0; `CausalDiagnosis/experiments/run_verification/run_ledger/code_identity.py`; `utils/kind_files.py`; `CausalDiagnosis/utils/atomic_writes_and_locks.py`; the two digest functions in `CausalDiagnosis/environment/reference_policies/diagnosability_label.py` and `CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/fit_models.py`; `CausalDiagnosis/evaluation/result_tables/index_file.py`; `CausalDiagnosis/experiments/episode_runners/episode_launchers.py`. One ordered exception to the rule of one owner: after task 3 has landed, this task adds to the launcher text of the replay, a file of the first piece, the line that records the files a server process loaded.

**Builds.** The three families of lists, the digest by them, the code section of the results index by owner, and the record of the files that a server process loaded, which the gate reads.

**Tests, each with its control.** Planting moves nothing: before and after a made-up dataset is planted in a planted framework folder, the label digest, the fit digest and the identity of each agent on every other asset are equal. Positive controls: one byte added to a file of one asset moves that asset's digest and identity and nothing of another; one byte added to a file of a kind moves the assets of that kind alone.

**Done means.**
```bash
PATH=$PWD/.venv/bin:$PATH python -m pytest CausalDiagnosis/experiments/table_definitions/tests/test_code_lists.py CausalDiagnosis/experiments/episode_runners/tests/test_agent_files.py CausalDiagnosis/experiments/run_verification -q
```

**Needs.** Step 0. **Must not touch.** The two agent lists as step 0 left them; a file of an agent or of a design.

## Task 13: the driver, the viewer and the tables

**Goal.** The driver hands each design the declaration of the run and plays the new agent beside the present one, and every reader of the agent's records reads both forms until the last change.

**Read first.** The sections The loop and The last change of the plan page; in `CausalDiagnosis/experiments/episode_runners/run_episodes/`: `agent_kinds.py`, `agent_runs.py`, `play_one_episode.py`, `run_record.py`.

**Owns.** In that folder: `agent_kinds.py`, `agent_runs.py`, `command_line.py`, `runner_setup.py`, `play_one_episode.py`, `run_record.py`, `__init__.py`, and `kind_rows.py` of the third piece; `CausalDiagnosis/agents/__init__.py`; `CausalDiagnosis/episode_viewer/walkthrough/`; in `CausalDiagnosis/evaluation/result_tables/tables_from_runs/`: `plan_schedule_tables.py` and `episode_fields.py`; `CausalDiagnosis/experiments/run_verification/invariant_digests.py`; `CausalDiagnosis/frameworks/AssetOpsBench/tool_servers/digest_families.py`.

**Builds.** The agent `causal_next` in the driver, with the flags of the registry; the declaration handed to the three designs; the two refusals of the driver's entry for a scripted asset lifted; the readers of `plans.json` for the new records, each of which takes the steps of a plan from the steps of its record and never assumes that the plan itself holds steps; the rule that fills `first_plan_mode`; the digests of the agent's texts for every registered asset.

**Tests, each with its control.** The driver plays `causal_next` on the rig and on the made-up port (control: the agent name removed from the driver's table). A run that sets no option records the options that a run of the present agent records. The viewer places every call of a batch on its plan step for both record forms. It places no call on a standing step and shows the step with its reason (control: a standing step followed by a call of the same tool, which lands on the standing step where the status is not read). The tests of the map under driver and viewer.

**Done means.**
```bash
PATH=$PWD/.venv/bin:$PATH python -m CausalDiagnosis experiments run-episodes --config base --servers aob --store memory --agent causal_next --starts development --start-cap 1
PATH=$PWD/.venv/bin:$PATH python -m pytest CausalDiagnosis/experiments/episode_runners CausalDiagnosis/episode_viewer CausalDiagnosis/evaluation/result_tables -q
```
The first command needs a model endpoint or the stand-in of the tests; with neither it stops by name before its run folder.

**Needs.** Tasks 1 and 7; the third piece of the plan page, which brings `kind_rows.py`; the fourth piece for one test of the map. **Must not touch.** The path of the present agent through the driver and its flags.

## Task 15: the gate and the last change

**Goal.** Show that the acceptance of the plan page holds, then replace the present agent in one pull request.

**Read first.** The sections The last change and Acceptance of the plan page; every merged card.

**Owns.** The comparison of the two agents with its helper that translates a reply of the present schema; the stand-ins of the gate; the gate test; the scan of names with its allow-list under `CausalDiagnosis/utils/tests/`; the docs of the agent; the last change itself.

**Builds.** The stand-in that turns the scripted plan of a case into plans of the new schema, and the scripted replies of the setting without the server. The comparison on every development start. The gate of the six settings on three assets. The planted dataset. The test that every module loaded while an asset is played lies in its lists, for each setting. The scan of names. Then the last change, as the plan page lists it, the removal of the term for the summary from the rule of completing a repair and from the one line of the diagnostics server included.

**Tests, each with its control.** The comparison reports no difference outside the listed changes of the plan page (control: one listed change undone shows as a difference). The gate (control: the planted dataset with one row removed from its contract fails by name). The scan of names (control: an asset id planted in a file of the core).

**Done means.**
```bash
PATH=$PWD/.venv/bin:$PATH python -m pytest src CausalDiagnosis benchmarks -q -p no:cacheprovider
for agent in causal plan_execute agent_as_tool; do for flag in "" "--no-diagnostics"; do
  PATH=$PWD/.venv/bin:$PATH python -m CausalDiagnosis experiments run-episodes --config base --servers aob --store memory --agent $agent $flag --starts development --start-cap 1
done; done
```
The suite, which needs the environment of `uv sync` as the paragraph Tests of [README.md](README.md) says, passes after the last change, and each of the six commands ends with a verdict and no error on the rig; the gate test plays the same six on the made-up cases that the first piece brings for FaultArena and on the planted dataset.

**Needs.** Every other task. **Must not touch.** The present agent before the maintainers have read the comparison and agreed to the last change.

## The blind port

**Goal.** Show that the page on adding a dataset is enough. A coding agent that has seen none of the agent's code writes one further made-up dataset of scripted cases from [adding-a-dataset.md](adding-a-dataset.md) alone.

**Read first.** That page and nothing of the agent's code. **Owns and builds.** An asset file, a package, a contract, a tool table and cases in the shared form, in a framework folder of its own, and no other file.

**Done means.** The six settings reach their end on it with no line of code changed, and planting it moves no digest and no identity of another asset. A failure is a fault of the declaration or of that page, and both are mended in a pull request of their own.

**Needs.** The last change.

## Test map

The map is preliminary. Every test function of the present agent's tests has one first owner, set from its name and its docstring on the tree that holds the opening rule in the words of the check and the cut of a plan at its first unusable step. Thirteen names of this map, twelve under Core and one under Driver, are tests of the fourth piece of the plan page and stand in no file of a tree without it. The body decides: a builder confirms each test when re-expressing it, a test that asserts properties of two tasks is split by assertion, and the pull request names every row that moved. A test that the map does not hold belongs to the task whose code it tests. A re-expressed test keeps its name where its property is unchanged, asserts the same property through the interfaces of the plan page, and keeps its control. The standing step of the plan page changes the property of names of this map, which keep their owner and are re-expressed on the new property: `a_read_of_a_group_with_every_recorded_cycle_read_makes_no_call` and the last assertion of `a_read_with_no_unread_cycle_known_is_answered_by_the_server_where_a_clock_runs` assert that a read with nothing new stands where no time passes, and no longer that it is a surprise.

**Core, tasks 4, 5 and 7, by the code under test** (67): `a_kind_the_row_does_not_serve_is_named_nowhere_and_refused`, `no_listing_serves_every_tool_and_an_empty_listing_stops_the_episode`, `a_failed_expectation_is_a_surprise_that_triggers_a_second_plan`, `a_plan_the_check_refuses_goes_back_to_the_model_with_the_reasons`, `the_summary_request_leaves_out_the_fewest_oldest_entries_that_make_it_fit`, `a_recall_step_brings_collapsed_rows_back_on_the_next_board_only`, `an_entry_that_holds_an_identifier_the_summary_names_stays_word_for_word`, `an_entry_the_summary_names_stays_through_later_summaries_and_only_its_own_event_returns`, `a_reply_without_json_runs_nothing_and_its_error_goes_to_the_next_turn`, `the_parser_takes_the_last_json_object_and_names_schema_errors`, `the_plan_check_enforces_the_guards_and_the_rule_plan_follows_the_decision`, `consecutive_planning_requests_share_every_part_that_held_and_the_status_comes_last`, `a_plan_that_declares_before_any_reading_is_refused_and_a_read_before_it_lifts_the_refusal`, `a_declaration_of_the_controller_cites_a_reading_even_when_the_top_candidate_bears_on_none`, `no_tool_call_on_the_rig_precedes_the_first_planning_call_whose_board_holds_the_question_verbatim`, `compaction_collapses_only_what_the_request_needs_to_fit_and_loses_no_refuted_set`, `the_parser_splits_the_joined_components_of_an_order_step_into_one_order_each`, `a_wait_of_zero_cycles_reads_as_one_and_the_prompt_says_so`, `a_plan_over_the_step_limit_is_refused_and_the_prompt_states_the_limit`, `without_a_count_a_plan_is_bounded_by_the_budget_the_horizon_and_the_step_cap_alone`, `a_step_keeps_every_identifier_it_cites_and_the_prompt_asks_for_few`, `the_opening_asks_the_failure_modes_of_every_component_by_name`, `a_summary_of_the_working_memory_keeps_every_cycle_it_holds`, `the_board_and_the_check_without_the_server_take_the_model_candidates_and_the_bands`, `the_agent_package_holds_no_band_of_its_own`, `the_rule_only_switch_left_the_driver_and_the_options_of_a_run`, `the_result_names_how_the_run_ended`, `a_cut_run_returns_what_it_recorded_with_the_end_cancelled`, `a_call_without_an_answer_is_recorded_as_a_transport_error`, `the_request_settings_of_the_controller_hold_every_key_and_the_values_of_its_backend`, `the_controller_opens_its_servers_once_per_episode_through_the_session_module`, `an_episode_over_stdio_starts_one_process_per_server`, `a_transport_that_breaks_is_the_error_of_the_run`, `with_the_server_a_plan_may_diagnose_a_candidate_of_its_own`, `the_options_of_the_agent_hold_no_threshold_card`, `the_board_with_the_server_shows_every_section_of_the_board_without_it`, `a_reply_cut_at_the_token_cap_is_not_parsed_and_the_retry_says_so`, `an_incomplete_plan_object_is_reported_as_incomplete`, `levels_and_component_names_are_taken_as_the_profile_spells_them`, `one_reader_of_the_receipt_serves_the_controller_and_the_port`, `the_orientation_waits_for_the_failure_modes`, `without_the_server_readings_collapse_only_when_the_request_does_not_fit_and_keep_their_side`, `a_group_counts_every_channel_the_board_holds_for_its_cycle`, `an_episode_that_a_failure_ends_reads_its_end_reason`, `an_episode_the_environment_ends_at_a_no_call_step_gets_one_summary_and_no_finish`, `a_declaration_without_the_server_cites_the_newest_absorbed_reading`, `time_and_extent_wait_for_the_profile_on_every_surface`, `a_first_plan_with_an_extent_step_on_the_rig_ends_without_an_error`, `the_closing_summary_words_the_limits_as_the_board_does`, `an_answer_that_is_not_json_keeps_its_text`, `before_the_profile_the_request_states_the_rule_that_the_check_refuses_by`, `the_board_shows_the_notes_on_one_line_before_and_after_the_profile`, `before_the_profile_every_request_names_the_turns_refused_before_it`, `no_reply_raises_out_of_the_planning_call_before_or_after_the_profile`, `the_candidates_of_a_plan_before_the_profile_are_set_aside_and_the_episode_plays_on`, `a_mended_reason_names_the_change_and_the_steps_before_its_step_pass`, `every_reason_of_the_plan_check_and_the_parser_has_its_rule_in_the_texts_a_model_reads`, `a_plan_runs_up_to_its_first_step_that_the_controller_cannot_use`, `a_plan_before_the_profile_runs_its_opening_with_its_candidates_set_aside`, `after_a_cut_no_sentence_of_the_next_request_claims_a_step_that_did_not_run`, `each_reason_of_the_check_belongs_to_its_step_or_to_the_whole_plan`, `the_parser_cuts_a_plan_at_its_first_step_that_does_not_parse_and_refuses_a_broken_head_whole`, `without_the_server_candidates_that_share_an_indicated_group_get_one_read_and_none_without_channels`, `a_read_that_is_refused_ends_the_batch_and_the_reads_before_it_are_ranked_once`, `the_surprise_of_a_batch_is_the_first_expectation_that_fails`, `with_the_switch_off_the_agent_is_the_agent_without_the_mechanism`, `a_test_that_one_plan_alone_names_is_not_in_the_batch`.

**Declaration, step 0** (1): `the_table_names_every_kind_a_level_may_leave_out_and_the_listing_decides_it`.

**Episodes, task 7** (6): `a_test_first_plan_reads_diagnoses_orders_advances_verifies_and_finishes`, `an_act_now_plan_orders_the_repair_plan_of_the_top_branch_and_verifies_it_on_its_own_read`, `without_the_server_the_agent_calls_no_diagnostics_tool_and_diagnoses_repairs_and_verifies_on_the_bands`, `without_the_server_a_repair_the_bands_refute_is_recorded_and_the_next_plan_repairs_another_candidate`, `on_a_start_with_no_fault_the_agent_diagnoses_all_healthy_and_finishes_without_a_work_order`, `the_agent_repairs_two_components_in_one_visit_on_the_rows_with_lead_times`.

**Prompt, task 6** (8): `the_system_prompt_tells_that_a_turn_without_a_call_takes_a_step`, `the_planning_turn_names_a_count_only_when_one_is_set`, `the_board_names_the_time_tools_on_a_clock_row_only`, `every_prompt_of_an_episode_holds_the_task_message_with_its_block`, `the_texts_of_the_registered_asset_keep_their_words`, `the_texts_the_agent_builds_for_the_tiny_asset_hold_no_word_of_another_asset`, `the_texts_of_the_agent_name_the_plain_statistic_the_asset_serves`, `the_texts_of_the_schedule_on_the_tiny_asset_hold_no_word_of_another_asset`.

**Kind rules, task 2** (28): `the_tail_reads_the_verifying_group_forward_where_no_wait_is_served`, `the_agent_and_the_server_finish_a_repair_by_the_same_rule`, `a_manifest_without_prices_or_budget_prices_every_tool_at_zero_and_never_declares`, `a_read_pays_and_takes_only_the_cycles_not_yet_paid_or_recorded`, `a_count_of_cycles_takes_the_singular_only_for_one`, `a_read_of_a_group_with_every_recorded_cycle_read_makes_no_call`, `the_band_verification_follows_the_bands_the_environment_serves_and_not_a_copy_of_its_own`, `the_band_verdict_takes_the_bands_and_refutes_a_repair_when_any_channel_mean_lies_outside`, `on_the_min_to_max_bands_every_healthy_recording_passes_and_every_recording_of_a_fault_is_refuted`, `an_order_stays_open_until_the_surface_says_it_closed`, `the_close_of_a_visit_is_estimated_as_the_environment_schedules_it`, `a_read_takes_no_time_where_recorded_cycles_are_served`, `without_the_server_a_repair_is_not_judged_while_another_order_is_open`, `the_summary_gives_an_open_order_as_open_with_the_close_of_its_visit`, `every_place_that_counts_the_tail_of_a_repair_calls_the_one_rule`, `the_completion_of_a_repair_is_what_the_environment_charges`, `near_the_step_cap_one_plan_passes_and_the_finish_guard_agrees`, `only_a_read_of_a_group_that_indicates_the_repair_verifies_it`, `the_thinking_clocks_of_the_board_are_those_of_the_environment`, `an_open_window_is_priced_at_most_what_thinking_lets_the_engine_charge`, `where_the_board_knows_no_indicating_channel_only_the_group_of_every_channel_verifies`, `every_registered_asset_has_a_group_of_every_channel`, `a_second_order_on_a_component_is_refused_only_while_the_first_is_open`, `the_board_follows_the_run_of_orders_as_the_environment_does`, `the_plan_check_charges_one_visit_per_run_of_orders`, `no_place_groups_orders_by_their_creation_cycle`, `two_orders_with_thinking_between_them_form_one_visit_and_cost_what_the_board_counted`, `a_paid_cycle_is_priced_once_inside_one_plan`.

**Ranking module, task 8** (40): `the_rule_writes_no_kind_the_row_does_not_serve`, `a_read_is_judged_on_the_faulty_set_on_top_and_its_mass_summed_over_the_set`, `the_batch_reserves_the_verifying_read_the_agent_reserves_whatever_it_was_told`, `the_batch_reserves_the_verifying_read_of_every_repair_the_agent_waits_on`, `the_rule_that_follows_the_reads_of_the_server_passes_the_plan_check`, `the_rule_acts_on_the_cheapest_branch_and_the_advice_without_the_server_names_rank_one`, `every_read_is_a_step_of_a_model_plan_and_the_board_shows_the_read_batch_as_advice`, `the_controller_ranks_before_the_first_planning_turn_and_the_model_reads_the_advised_head`, `a_diagnosis_while_the_ranking_says_read_first_runs_with_the_advice_on_the_board`, `a_refuted_verdict_leads_to_the_cheapest_other_plan_and_the_ablation_keeps_the_surprise_only`, `the_reserve_holds_the_top_plan_and_a_visit_charge_without_the_fault_rule`, `the_advice_keeps_the_refuted_set_out_until_a_later_ranking_names_it_first`, `the_batch_rule_the_declaration_and_the_switches`, `an_arm_keeps_its_own_choice_and_the_registered_agent_follows_the_advice`, `a_ranking_without_advice_is_planned_as_before_under_every_setting`, `the_board_shows_the_advice_of_the_server_and_the_rank_call_asks_for_its_form`, `a_diagnosis_is_possible_when_every_ranking_failed`, `the_reserve_is_the_plan_of_the_first_candidate`, `the_prompt_says_of_the_advice_what_the_board_shows`, `a_diagnosis_below_rank_one_without_an_expectation_expects_the_candidate_on_top`, `a_candidate_without_a_fault_is_an_option_of_the_rule`, `after_a_refuted_plan_a_candidate_without_a_fault_is_no_option_of_backtracking`, `the_agent_reads_the_flag_under_its_new_name_alone`, `a_read_that_several_candidate_plans_share_is_scheduled_once_with_the_plans_it_serves`, `the_reads_of_a_batch_are_cut_where_they_no_longer_fit_together`, `a_plan_that_could_not_be_completed_after_the_named_reads_is_left_out_and_named`, `a_read_the_server_does_not_name_is_added_only_beside_the_repair_of_every_component`, `under_one_read_through_a_group_that_covers_every_test_the_board_is_the_board_without_the_switch`, `after_a_refuted_plan_the_schedule_lists_what_each_remaining_plan_needs_and_repeats_no_executed_step`, `the_rule_plans_the_scheduled_reads_as_one_run_and_follows_the_board_otherwise`, `under_the_advice_of_the_server_the_reads_of_its_rule_run_as_one_batch_where_it_names_more_than_one`, `consecutive_read_steps_run_with_one_ranking_after_the_last`, `the_ranking_after_a_batch_is_the_ranking_after_the_same_readings_one_by_one`, `after_a_refuted_plan_the_agent_reads_what_the_remaining_plans_share_before_the_next_repair`, `after_a_refuted_plan_with_no_other_plan_left_the_advice_names_what_the_ranking_still_holds_as_faulty`, `the_line_names_a_repair_only_when_the_budget_left_pays_it`, `after_a_commitment_with_no_plan_offered_the_advice_names_what_the_ranking_still_holds_as_faulty`, `while_a_candidate_plan_is_left_the_advice_names_the_plan_and_no_component`, `after_a_refuted_plan_the_line_leaves_out_the_refuted_set_that_the_board_and_the_prompt_keep_out`, `the_reserve_of_the_base_row_keeps_the_calls_the_environment_charges`.

**Warrant module, task 9** (6): `a_verified_prediction_names_the_most_probable_combination_after_the_repair_and_its_mass`, `the_closing_summary_names_the_most_probable_combination_after_the_repair_and_its_mass`, `a_consistent_verified_plan_stands_whatever_the_belief_top_shows_after_the_reading`, `the_working_memory_names_every_prediction_of_the_mcp_as_a_prediction`, `the_controller_verifies_a_repair_on_its_first_verifying_reading`, `a_refuted_plan_is_the_surprise_of_the_last_read_made_before_any_expectation`.

**Binding, task 10** (9): `the_causal_agent_plays_the_assetopsbench_rows_with_the_scripted_model`, `a_history_window_covers_unread_cycles_in_one_run_and_never_a_read_one`, `without_the_server_the_agent_plays_the_assetopsbench_clock_row_without_our_tools_on_the_bands`, `without_the_server_the_agent_plays_the_assetopsbench_real_time_row_without_our_tools_and_finishes_unverified`, `a_cycle_without_a_record_is_not_offered_or_priced_again`, `a_read_with_no_unread_cycle_known_is_answered_by_the_server_where_a_clock_runs`, `a_text_that_is_no_json_object_is_no_step_of_the_base_row`, `the_steps_left_of_the_surface_equal_the_environment_after_every_call_of_the_base_row`, `a_batch_on_a_row_that_serves_recorded_cycles_makes_one_ranking_and_verifies_its_repair`.

**Designs, task 11** (1): `the_task_agent_shows_and_describes_only_the_tools_it_holds`.

**Driver, task 13** (6): `the_driver_runs_the_causal_agent_through_the_openai_compatible_client_with_the_switches`, `the_causal_agent_holds_one_model_request_open_and_every_failed_attempt_reaches_the_ledger`, `the_causal_agent_keeps_one_model_request_open_at_a_time_and_the_endpoint_sees_two_when_two_are_open`, `the_driver_runs_the_causal_agent_on_the_assetopsbench_rows`, `the_driver_plays_the_causal_agent_without_the_server_and_refuses_the_switches_that_need_it`, `through_the_driver_a_model_that_reads_the_board_leaves_the_opening_on_the_development_starts`.

**Viewer, task 13** (1): `the_walkthrough_places_every_call_of_a_batch_on_its_plan_step`.

**Kept as they are** (34): every test of `test_agent_texts.py`, `test_mcp_sessions.py`, `test_model_adapters.py`, `test_no_real_uv.py`, `test_sessions_isolation.py`, `test_start_commands.py`, and two tests of `test_causal_diagnosis_agent.py`, `the_package_of_public_rules_reads_public_values_alone` and `no_text_states_the_visit_of_one_cycle`.

**Retired at the last change** (8), each with its reason: `the_texts_of_a_row_that_serves_every_kind_are_those_of_today` (no text is edited after it is written, so no pair of texts with and without a kind exists to compare); `the_mode_declare_is_refused_where_no_declaration_is_served` (the plan has no mode); `the_change_summary_is_a_counted_call_where_served_and_a_local_answer_elsewhere` (the controller no longer calls the change summary after a wait); `the_sentence_of_the_acting_kinds_holds_for_any_count_of_kinds` (the sentence that counts the acting kinds is gone, and the table has one line per listed tool); `the_rule_only_test_tool_plans_without_the_model_and_the_model_writes_the_summary_only` (the rule plays as a stand-in model of the tests of the ranking module); `the_server_names_agree_in_the_agent_the_port_and_the_drivers` (the agent names no server, and the test of the declaration holds the names); `a_wait_on_a_row_without_a_clock_is_refused_with_the_sentence_of_the_prompt` (a wait where no clock runs is no longer refused by the check); `the_role_text_names_every_end_of_an_episode_and_the_lead_time_of_its_row` (the role text no longer restates the rules of the row, which the listed descriptions and the task message state).

**Known to span tasks.** `a_manifest_without_prices_or_budget_prices_every_tool_at_zero_and_never_declares` (kind rules, and assertions of the ranking module, the check and the prompt); `with_the_switch_off_the_agent_is_the_agent_without_the_mechanism` (core, and assertions of the driver and the prompt); `the_controller_verifies_a_repair_on_its_first_verifying_reading` (warrant module with the server, kind rules without it); `time_and_extent_wait_for_the_profile_on_every_surface` (the half on the read of the time retires with change 4 of the plan page, the half on the extent stays, and the sentence of the surface goes to task 10); `without_a_count_a_plan_is_bounded_by_the_budget_the_horizon_and_the_step_cap_alone` (core: the reader and the options, and an assertion of the prompt); `the_texts_of_the_agent_name_the_plain_statistic_the_asset_serves` (prompt, and the read of the binding); `the_batch_rule_the_declaration_and_the_switches` (ranking module, and the flags and labels of the driver); `the_working_memory_names_every_prediction_of_the_mcp_as_a_prediction` (warrant module, and the sections of the ranking module); and six tests of the ranking module that assert on the schedule, which task 7 owns, and on its lines on the board, which the ranking module prints: `a_read_that_several_candidate_plans_share_is_scheduled_once_with_the_plans_it_serves`, `the_reads_of_a_batch_are_cut_where_they_no_longer_fit_together`, `a_plan_that_could_not_be_completed_after_the_named_reads_is_left_out_and_named`, `a_read_the_server_does_not_name_is_added_only_beside_the_repair_of_every_component`, `under_one_read_through_a_group_that_covers_every_test_the_board_is_the_board_without_the_switch`, `after_a_refuted_plan_the_schedule_lists_what_each_remaining_plan_needs_and_repeats_no_executed_step`.
