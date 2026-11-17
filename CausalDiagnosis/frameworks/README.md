# Frameworks

A framework is an agent ecosystem that CausalDiagnosis runs beside its own environment: its agents, the tool servers they play,
its judge, its scenarios and its cluster scripts. Each lives in one folder here, `frameworks/<Name>/`, and today there is one,
`AssetOpsBench/`.

## How a framework connects

The folder's `__init__.py` is its declaration. It assigns plain literals at module level (dicts, tuples, strings, numbers,
booleans), and `CausalDiagnosis/utils/frameworks.py` reads them with `ast.literal_eval` without importing the file. Code is
named by a string `"module:attribute"` and imported only by a run that asks for it. The fields:

| field | shape | what the core reads from it |
| --- | --- | --- |
| AGENTS | {name: {model_agent, route, surfaces, trained, paper_paradigm, modules, entry}} | the agent kinds of the driver; an agent without model_agent counts as a model agent |
| SURFACES | {name: {servers, state_file, launch_module, stores, store_environment, export_file, default_on_row_presets, action_server_name, covers, withheld_actions, levels, gateway, entry}} | the tool servers an agent plays, how they start, and which of the environment's actions they serve at each level (utils/frameworks.py ANSWER_ACTIONS, LEVELS) |
| JUDGES | {name: {protocols, entry}} | the judges of a run |
| SCENARIO_SETS | {name: {asset_id, env, entry}} | the scenarios a preset may anchor |
| RUN_KINDS | {name: {entry}} | the kinds of run the ledger records |
| CONFIGS | (folder, ...) | folders of presets read beside `CausalDiagnosis/configs/` |
| JOB_ENVIRONMENT | (variable, ...) | variables of the cluster jobs the ledger records |
| PROFILE_CHECKS, DIGEST_FAMILIES | {name: "module:attribute"} | the profile check of a surface, the digest families |
| COMMANDS | {command: module} | the commands of the group `python -m CausalDiagnosis <name in lower case>` |
| LISTS | {list: (path, ...)} | the core list (or agent:<name> of its own agents) each of its files lies in, first match wins |

A further framework connects through the same declaration. The framework planted in CausalDiagnosis/utils/tests/planted_framework declares two agents, a judge, a tool surface, a run kind, a profile check, a digest family, its lists and a command, and every reader of the core resolves them with no change of core code. A scan test checks that no core module imports or names a framework module outside its docstrings. An episode of a further framework, played, judged, replayed and entered in the ledger, is not yet proven by a test.

## The import rule

A framework may import every layer of the core. No file outside `frameworks/` imports a framework, and no framework imports
another. The core puts the fork's `src/` on `sys.path`, which a framework built on the fork's packages relies on.

## The limit of the format

The fields above are derived from one framework, AssetOpsBench. A later framework that needs a field the declaration does not
have changes core code, among it CausalDiagnosis/utils/frameworks.py of the harness list, so runs before and after that change differ in their code identity. A field this core does not know is ignored with one
message, so a declaration written for a later format does not stop an older core; a known field with a wrong shape stops the
registry, since the core would misread it.
