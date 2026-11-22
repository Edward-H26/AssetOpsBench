# General agent: the plan

The CausalDiagnosis agent of this repository is written for one kind of asset, a rig with components, condition levels, sensor groups and work orders. This page is the plan for one agent that plays any dataset from what the dataset declares, and for the two reimplemented designs on the same declaration. The agent is rewritten once, in a new package beside the present one, and replaces it in one last change when the acceptance at the end of this page holds. [general-agent-tasks.md](general-agent-tasks.md) cuts the work into task cards.

Nothing on this page is built yet except where it says so. A path that starts with `CausalDiagnosis/` names a file or a folder of the tree at the commit that added this page. Every other path is relative to that folder, or to the folder named just before it; where it names a file that the tree at that commit lacks, a piece named below adds the file or a task creates it, or, for the shared file of the core texts, the maintainers add it; `run.json` and `plans.json` are files that a run writes into its folder. Where this page writes a signature with an ellipsis, names a function without its parameters or names a file without its function (`check_plan`, the four turns of `prompt/assembler.py`, the eight functions of the compaction and recall of the board, the constructor of `agents/utils/answer.py`), step 0 of the task page fixes the parameter list, and the cards read it from the stubs.

## What the tree holds and what the plan builds on

**The present agent**, `CausalDiagnosis/agents/Causal_Diagnosis_Agent/`, asks a model for a plan as JSON, checks it, runs its steps and keeps a working memory, the board. It is general in its loop, its compaction and its records, and fitted to the rig everywhere else: a plan step has one of a fixed list of kinds (`plan/plan_schema.py`), each kind is bound to one tool (`tool_surfaces.py`), the two system prompts are fixed texts that name sensor groups, components and work orders (`prompts.py`), and the board, the plan check and the schedule compute prices, visits and closes of the rig (`situation_board/`, `plan/plan_check.py`, `candidate_plan_schedule.py`). The present agent stays as it is and stays tested until the last change of this plan. Runs play it meanwhile.

**The two reimplemented designs**, Plan-Execute and Agent-As-Tool under `CausalDiagnosis/frameworks/AssetOpsBench/agents/`, run one sub-agent per tool server of that framework, with texts and examples written for those servers.

**The pieces for scripted cases.** The plan builds on four pieces. Each is named here by what it holds, and the command shows whether a tree holds it. Run each command from the root of the repository with `PATH=$PWD/.venv/bin:$PATH` in front, as the Quick start of [README.md](README.md) does: a piece is in the tree when its import ends without an error, and the fourth piece when its command prints a count.

| Piece | What it holds | Check |
| --- | --- | --- |
| the environment of scripted cases | the replay of scripted fault cases (`datasets/dataset_loader/scripted_case_replay/`), the role table of an asset (`datasets/dataset_loader/tool_roles.py`), the shared action server (`mcp_servers/Scripted_Cases_Actions_MCP/`), the dataset FaultArena (`datasets/faultarena/`, `mcp_servers/FaultArena_Actions_MCP/`) and the made-up port `bottler_1` under the tests of the replay | `python -c "import CausalDiagnosis.datasets.dataset_loader.tool_roles"` |
| the backends of the diagnostics server | the advice kinds, the tool records and the scripted backend (`mcp_servers/Causal_Diagnosis_MCP/backends/`) | `python -c "import CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.backends"` |
| the driver's entry for a kind that the engine does not play | `experiments/episode_runners/run_episodes/kind_rows.py` | `python -c "import CausalDiagnosis.experiments.episode_runners.run_episodes.kind_rows"` |
| the mended planning call of the present agent | the opening rule stated in the words of the check, a plan that runs up to its first step that the controller cannot use, and a reply reader that never raises | `git grep -c parse_usable_plan -- CausalDiagnosis/agents/Causal_Diagnosis_Agent` |

At the commit that adds this page the tree holds none of the four pieces. The maintainers of the repository add them. The freeze of step 0 of the task page needs none of them; its move needs the first. Until all four checks pass, the files of the pieces that a card names under Read first are not in the tree, and the part of a task that reads them cannot start. Where a section below speaks of the scripted cases, of FaultArena, of a made-up port or of a backend of the diagnostics server, it describes the tree once the piece that brings them is there.

Once the pieces are in the tree, the reference policies and the scripted agent play a scripted asset. The CausalDiagnosis agent and the two reimplemented designs do not: the entry of the third piece refuses them by name on such an asset, and this plan lifts both refusals.

**What this plan does not hold yet.** Three parts are designed for the present agent and are not in the tree: a plan that the model writes as a program, with its checker and its host; the tools that such programs keep, within an episode and in a library across episodes; and a server through which another agent runs a program over the tools of an episode. Each becomes a task of its own after the first episode of the new agent. For them a data class of this page may gain a field with a default, a function or a constructor a keyword parameter with a default, and a class a method, each through a pull request to the owner of step 0, and a reader of the agent's records ignores a key it does not know.

## Layers

A dataset brings data. A kind brings the rules that follow from its data, as functions over public values. Whoever serves a tool brings its contract. A diagnostics backend brings its tools with their advice kinds. The agent holds none of the four: it reads one declaration, and its code names no tool, no server, no field of a profile and no word of a dataset.

| Layer | What it holds | Where | Who adds one |
| --- | --- | --- | --- |
| core | the turn loop, the plan schema and its reader, the plan check, the working memory with its compaction, the prompt assembler, the registry of switches, the call site of the advice modules | `agents/Causal_Diagnosis_Agent_next/` | nobody, for a new dataset |
| kind | the cost of a call, the ledger of what is open and what is paid, the rule functions of the kind's rule cards, the end guard, the verdict on a repair where no backend checks it | `utils/public_rules/` (standard library alone) | the author of a new evidence kind |
| wire | how an answer is read and, on a surface whose tools are not the asset's own, how arguments are written | the tree's own format in `CausalDiagnosis/agents/utils/`, any other in the folder of its framework | the author of a server with another format |
| module | what one advice kind adds to the board, the prompt, the check and the loop | `agents/Causal_Diagnosis_Agent_next/advice/` | the author of a new advice kind |
| declaration | the contract of every server, the vocabulary, the rule cards, the records of a backend | the contract files, the asset file and its package, the declaration of a framework | the author of a dataset or a server |

## The declaration

`agents/utils/declaration.py` holds the data classes. It imports the standard library alone and is shared code of the three designs and of the driver: it is no file of the agent's package, so a run of another design that reads it loads no file of the agent.

```python
@dataclass(frozen=True)
class ToolRow:                                   # one tool: a contract row, joined with the listing once an episode has opened its sessions
    name: str; server: str; role: str            # role: observe | knowledge | act | task | advice
    description: str = ""; schema: dict = field(default_factory=dict)   # empty in a Declaration; in a Table as the server listed them
    write: str | None = None                     # write class: repair | probe | state | answer; None for a read
    commits: bool = False; ends: bool = False
    sets: str | None = None; target: str | None = None; words: tuple = ()
    facts: bool = False                          # the tool whose answer publishes the public facts
    orients: bool = False                        # a knowledge read whose answer the differential waits for
    each: str | None = None                      # the argument that, left empty or left out, stands for one call per legal value of it
    withheld: bool = False                       # listed, and kept out of the table that the CausalDiagnosis agent plans on
    minimum: dict = field(default_factory=dict)  # argument -> the least count that a step may write for it
    after: tuple = ()                            # each item a tool name or a tuple of names, any one of which satisfies it
    values: dict = field(default_factory=dict)   # argument -> path of the facts whose keys or items are its legal values
    fills: dict = field(default_factory=dict)    # argument -> evidence | warrant | closing_summary | fixed:<value>
    candidate: tuple = ()                        # on a commit write: the arguments that state the answer
    omitted: dict = field(default_factory=dict)  # argument -> what it takes when a step or a candidate leaves it out: a path of the facts, or fixed:<value>
    waits: str | None = None                     # the argument up to which the tool lets time pass
    answer: str = "object"                       # the form of its answer: object | records | text
    series: str | None = None                    # the argument that names the series of an observation
    requires_observation: bool = False
    advice: str | None = None                    # the advice kind of a tool of the role advice
    call: dict = field(default_factory=dict)     # on an advice tool: input of its module -> argument of the tool

@dataclass(frozen=True)
class Vocabulary:
    noun: str; short_noun: str
    example_field: str | None = None             # a field name to give as an example
    identifiers: tuple = ()                      # the forms of the identifiers a step may cite, each (form, what it names)
    terms: dict = field(default_factory=dict)    # words of the kind: time_unit, series, target, repair, value, state

@dataclass(frozen=True)
class Declaration:                               # per run, built from files alone before the run folder exists
    asset_id: str; kind: str; surface: str | None  # the surface of the run; None on the asset's own action server
    vocabulary: Vocabulary
    rows: dict                                   # name -> ToolRow of the asset's own tools: the contract rows of the servers of the run; on a surface with a binding, the rows that the binding states and the rows of the diagnostics server
    listed_rows: dict                            # (server, name) -> ToolRow of the listed tools of a framework surface; {} where the surface is the asset's own server
    server_roles: dict                           # server -> role, the block roles of a contract
    cards: tuple                                 # the public rule cards in force, each {"kind", "parameters"}
    advice: dict                                 # tool -> advice record (kind, fields, warrant fields, example), from a backend or, for a tool without one, from the key advice of its row; {} where the run has no diagnostics server
    events: tuple                                # the names of the events that an answer can notify
    rules: object                                # the rules of the kind
    binding: object | None                       # the binding of the surface; None where every tool is called as listed
    def table(self, listing) -> "Table"          # the asset's own tools that the listing serves and that are not withheld
    def listed(self, listing) -> "Table"         # every listed tool under its listed name
    def to_json(self) -> dict                    # what run.json records of it

@dataclass(frozen=True)
class Table:                                     # per episode
    declaration: Declaration
    tools: dict                                  # name -> ToolRow with description and schema, in listing order
    def of_role(self, role) -> tuple; def of_write(self, write) -> tuple
    def commit_tools(self) -> tuple; def end_tool(self) -> str | None; def facts_tool(self) -> str | None
    def advice_kinds(self) -> tuple

class DeclarationError(Exception)                # a fault of a declaration, by name
```

A declaration holds nothing that exists only inside an episode. The listing of the servers, the state of a rule card at the start and the limits that the first answer shows belong to the episode: the controller builds the table once it has opened its sessions, and the ledger holds the rest. The rows of a surface with a binding need no facts: the binding states them when the declaration is built.

**Who states what.** `declaration_of(asset_id, servers, config=None, store=None) -> Declaration` in `experiments/episode_runners/run_episodes/declarations.py` is the one place that reads the asset layer, the contracts, the declaration of a framework and the backends for an agent. It reads files alone and starts no server. `servers` is the surface of the run as the option `--servers` names it, or `None` for the asset's own action server; `config` is the configuration of the run where the kind has one, from which the cards in force follow; `store` is the store of a framework surface. Its refusals come before the run folder exists and name what is wrong: a row whose keys do not fit, no write that ends the episode, a kind that names no rules, an advice tool whose kind is none of the four. The declaration is recorded in `run.json` under `declaration`.

| Field | Stated by | Where |
| --- | --- | --- |
| asset_id, kind | the dataset | the asset file: `asset_id`, `evidence_kind` |
| nouns, example field | the dataset | the asset file: `text` |
| terms of the kind | the kind | recorded levels: the block `asset_words` of the engine's tool contract; scripted cases: derived from the card and the contract |
| the rows of the tools | whoever serves the tool | the contract of its server |
| description and schema of a tool | the server that lists it | the listing at the opening of the sessions of each episode, which is no call |
| rule cards in force | the dataset | the asset file: `rules` |
| the state at the start | the card, or the episode | the card's `initial`; where it says `entry`, the task message |
| events | the server of the actions | the engine's contract: `notifying_events` |
| advice records | the backend | the tool table of a backend with records; the engine's contract for the four tools of the server on an asset of recorded levels |
| rules of the kind | the kind | a new field of the evidence kind, `plan_rules = "module:attribute"`, which names a class |
| binding | the framework | the object of its entry under `SURFACES`, attribute `agent_binding` |

**The terms of a kind** are the words that a rendered sentence needs and that are neither a tool name nor an argument name. The core texts print none by themselves.

| Term | Recorded levels | Scripted cases |
| --- | --- | --- |
| `time_unit` | cycle | none: no clock |
| `time_limit` | horizon | none |
| `series` | sensor group | none |
| `target` | component | the parameter that the key `target` of the repair write names |
| `repair` | work order | the name of the repair write |
| `value` | the plain statistic that the asset serves | none |
| `state` | none | the state of the precondition card |

The identifier forms are stated beside what makes the identifiers and joined by the assembler: `obs/<label or n>/<field>` for a value of an observation in the form `records`, `obs/<n>` for an observation in the form `text`, the forms of the ledger of the kind (for recorded levels `alarm/<time>/<field>` for an alarm and the identifier of a closed repair), the forms of the modules (the identifier of a warrant), and the identifiers that the answers of the knowledge tools carry. `identifier_forms(table, board, modules) -> tuple` of `prompt/assembler.py` returns them, each with what it names, for the prompt and for any other reader.

**The contract of a server** is a JSON file beside the server, in the form of `CausalDiagnosis/environment/Causal_Diagnosis_Environment/tool_contract.json`. Per tool it gains the optional keys `role`, `write`, `advice` and the row keys below, and per contract the optional block `roles`, a map from server to role. The asset file holds facts of the asset and is not touched.

| Row key | Fits | Meaning |
| --- | --- | --- |
| `sets`, `target`, `ends`, `words` | as the role table of the first piece has them | the value a state write sets, the parameter a repair write names, the write that ends the episode, the words of an answer |
| `commits` | an answer write | the write commits the answer of the episode; a write with `ends` commits by default |
| `candidate` | a write with `commits` | the arguments that state the answer; where no row of the contract states the key, by default the required arguments that no fill writes |
| `omitted` | any tool | what an argument takes when a step or a candidate leaves it out: the value at a path of the facts, or `fixed:<value>`; the reader writes it before the check, and an argument stands under one of `fills`, `each` and `omitted` at most |
| `facts` | a task read | its answer publishes the public facts |
| `orients` | a knowledge read | the differential waits for its answer |
| `each` | a read | an argument that, left empty or left out, stands for one call per legal value of it |
| `after` | any tool | tools that must stand earlier |
| `values` | any tool | the legal values of an argument, as a path of the facts |
| `fills` | any tool | arguments that the controller writes: `evidence`, `warrant`, `closing_summary`, `fixed:<value>`; a fill applies where the listed schema holds the argument |
| `minimum` | any tool | the least count that a step may write for an argument; a smaller one is refused by the check before any call |
| `withheld` | any tool | the tool is listed and stays out of the table that the CausalDiagnosis agent plans on; the two reimplemented designs see it as listed |
| `waits` | a task read | the argument up to which the tool lets time pass; a value of 0 is read as 1, and this reading comes before the check of `minimum`, which so refuses a negative count alone for that argument |
| `answer`, `series` | an observe, knowledge or act tool | the form of the answer (`object`, `records`, `text`; `text` by default on scripted cases) and the argument that names its series |
| `requires_observation` | an answer write | the write must cite an observation |
| `call` | an advice tool | which argument takes which input of its module: `detail` and `top` for a ranking, `action` or `calls` for a plan assessment, `warrant` for an outcome check |

`datasets/dataset_loader/tool_roles.py`, a file of the first piece, keeps the roles and write classes and checks the rows of a contract. An action server that the engine does not serve gets its contract as a file `tool_contract.json` beside the asset file, named by a new declaration of the package, `ACTION_CONTRACT`. The role table that the scripted assets of the first piece state in their asset files moves into such files.

**The table of an episode** has two forms. The asset's own table (`Declaration.table`) is what the CausalDiagnosis agent plans on: every row whose tool the listing serves and that does not state `withheld`, in listing order. A tool that a binding carries counts as served where the listed tool that carries it is listed. The listed table (`Declaration.listed`) is what the two reimplemented designs bind their sub-agents on, and what the agent plans on with the switch `binding` off: every listed tool under its listed name, with the role that the block `roles` gives its server where the contract states the block, else the role of its own row, and with the write class and row keys of its own row. A tool that writes must have a row of its own. A listed tool with no row and no role of its server raises a `DeclarationError` by name. Without a binding the two forms hold the same tools, but for the withheld ones. The table holds the tools of the role advice as well, for the modules alone: a step cannot name one, `known_tool` counts it as outside the table, and TOOLS prints no line for it. A tool name stands for one tool of a table: where two servers of a listing offer one name, `Declaration.listed` raises a `DeclarationError` by name. That is a limit of this plan, and a later answer is a name qualified by its server.

**The defaults of a row.** `declaration_of` completes a row that states none of these keys: the form of an answer of an observe, knowledge or act tool is the default of the kind (`KindRules.default_answer`: `object` for recorded levels, `text` for scripted cases), and where no row of a contract states `commits`, its answer write with `ends` commits. `Declaration.table` completes what needs the listing, under the same condition: where no row of a contract states `candidate`, the candidate arguments of its commit write are the required arguments of its listed schema that no fill writes. Where a row states the key, a commit write of the same contract without it holds no candidate arguments: a step on it gives its own arguments, and its line offers no rank. The default values of the data class are never read as the meaning of a missing key.

**The rig.** The rows of `CausalDiagnosis/environment/Causal_Diagnosis_Environment/tool_contract.json` gain these keys. Its other fields stay. The contract states `candidate` on `submit_diagnosis`, so `declare_insufficient_evidence` holds no candidate arguments and a step on it gives its own arguments, as the present agent declares.

| Tool | Role | Write | Row keys |
| --- | --- | --- | --- |
| `get_asset_profile` | task | | `facts` |
| `read_sensors` | observe | | `values {sensor_group: sensor_groups}`; `series sensor_group`; `answer records`; `minimum {cycles: 1}`; `omitted {cycles: fixed:1}` |
| `get_current_time` | task | | `withheld` |
| `wait_for_notification` | task | | `waits timeout`; `minimum {timeout: 1}`; `omitted {timeout: fixed:1}` |
| `summarize_changes_since` | task | | |
| `get_failure_modes` | knowledge | | `values {component: components}`; `orients`; `each component` |
| `get_fix_history` | knowledge | | |
| `lookup_fix` | knowledge | | `values {component: components}`; `withheld` |
| `submit_diagnosis` | task | answer | `commits`; `candidate` (one argument per component); `values {<component>: levels.<component>}`; `omitted {<component>: healthy.<component>}`; `fills {evidence_ids: evidence}` |
| `declare_insufficient_evidence` | task | answer | `commits`; `requires_observation`; `values {sensor_group: sensor_groups}`; `minimum {cycles: 1}`; `omitted {cycles: fixed:1}`; `fills {evidence_ids: evidence}` |
| `create_work_order` | act | repair | `target component`; `values {component: components}`; `after [[submit_diagnosis, declare_insufficient_evidence]]`; `fills {prediction_id: warrant, evidence_ids: evidence, mode: fixed:standard}` |
| `finish` | task | answer | `ends`; `fills {summary: closing_summary}` |
| `get_episode_evidence` | advice | | `advice evidence` |
| `rank_root_causes` | advice | | `advice ranking`; `call {detail: branch_detail, top: top_k}` |
| `estimate_intervention_effect` | advice | | `advice plan_assessment`; `call {action: action}` |
| `verify_outcome` | advice | | `advice outcome_check`; `call {warrant: prediction_id}` |

On the rows that run AssetOpsBench's servers, the contract `CausalDiagnosis/frameworks/AssetOpsBench/tool_servers/aob_tool_contract.json` gains the block `roles` (`iot` observe, `fmsr` knowledge, `wo` act, `task` task) and, on `generate_work_order`, the role act and the write class repair. Every tool that the task server lists has the role task in the listed table, so each role is the tools of one server there. The two withheld tools keep what the present agent does on the rig's own action server, where it calls neither. On the rows of AssetOpsBench's servers the binding states the row of every table tool it carries, with the listed tool that carries it: `read_sensors` by `history` of `iot`, `get_failure_modes` by `get_failure_modes` of `fmsr`, `get_fix_history` by `list_workorders` of `wo`, `create_work_order` by `generate_work_order` of `wo`, `stream_extent` by `stream_extent` of `iot` (role task, no argument for the model; the binding writes its arguments from the facts), and by the task server under their own names `get_asset_profile`, `submit_diagnosis`, `declare_insufficient_evidence` and `finish` on every row and `get_current_time`, `wait_for_notification` and `summarize_changes_since` where the row lists them. Its row of `get_failure_modes` keeps `orients` and holds no `each`, since the server behind it takes one component per call, so a plan names one call per component there (change 3 of the last change), while on the rig's own action server one step with the component left empty still reads them all. Its row of `get_current_time` is not withheld, since a read needs the clock there, and it carries no `lookup_fix`. A row that the binding states keeps every key of the same tool in the table above but those named here.

**Scripted cases.** The made-up port `bottler_1` of the first piece shows the form: `read_gauge` and `inspect_nozzle` observe with `words`; `lock_out` and `restore_supply` act, write class state, with `sets`; `swap_part` act, repair, `target part`; `test_run` act, probe; `file_report` task, answer, `ends`. Its defaults make the rows complete: an observe, knowledge or act tool answers in the form `text`, the write with `ends` commits, and its candidate arguments are its required arguments. Once the first piece is in the tree, FaultArena states thirteen rows in the same form.

## Rules of a kind

`utils/public_rules/plan_rules.py`. The package `CausalDiagnosis/utils/public_rules/` imports the standard library alone and holds no data at module level beyond constants and tuples of constants, which a test of the tree asserts. So the rules of a kind are a class, which `declaration_of` instantiates, and data that move there are tuples.

```python
class KindRules:                                 # one class per built in kind
    refusal_marks: tuple = ()                    # phrases that mark an answer in text form as a refusal
    default_answer: str = "object"               # the form of an answer of an observe, knowledge or act tool whose row states none
    def ledger(self, facts, flags, cards, task) -> Ledger    # facts: None until the facts tool is answered; task: the task message
class Ledger:                                    # the public state of one episode that the rules of the kind read
    def record(self, row, args, answer) -> None              # after every answered call; the answer of the facts tool hands over the facts
    def status(self) -> dict                                  # time, time_limit, budget_left, steps_left, over; None where the kind has none or no answer has shown it
    def limits(self) -> tuple                                 # the names of the limits that the episode has; () while none is known
    def walk(self) -> Walk                                    # a copy for one walk of a plan
    def open_targets(self) -> dict; def paid(self, series) -> tuple   # target -> when its repair closes, None where unknown; the units of a series that the episode has paid for, () where the kind counts none
    def pending(self) -> list                                 # repairs closed and not yet verified: key, targets, closed at
    def verifies(self, row, args, targets) -> bool            # whether an observation by this call can verify a repair of the targets
    def verdict(self, pending, observations) -> dict | None   # the kind's own verdict on a repair, None where it has none
    def targets(self, candidate) -> tuple                     # what a candidate asks to repair, () where the kind cannot tell
    def verifying_call(self, targets) -> tuple | None         # the cheapest observe call that can verify a repair of the targets
    def ranges(self) -> dict                                  # field -> (low, high) of the public normal ranges, {} where the kind has none
    def sections(self, vocabulary) -> list                    # (heading, header, rows) of what the board prints for the kind
class Walk:
    def step(self, row, args) -> StepCost
    def end(self) -> list[str]                                # what must precede the write that ends the episode, as reasons
    def needs(self, repairs) -> Needs                         # of completing these repair writes from here
    def totals(self) -> tuple                                 # price, time, calls of the walk so far
@dataclass(frozen=True)
class StepCost: price: float; time: int; calls: int; needs_facts: bool; reasons: tuple; stands: str | None = None   # "nothing_new" or "state_holds" where a call of the step could change nothing, else None
@dataclass(frozen=True)
class Needs: price: float; time: int; calls: int
```

The controller builds one ledger per episode, `declaration.rules.ledger(None, flags, declaration.cards, question)`, with the flags of the binding where one stands.

- **Recorded levels** (`utils/public_rules/recorded_levels.py`): the prices, durations, visits, closes, paid cycles and normal ranges that the present board and plan check compute, over `repair_completion.py` and `visits.py` of the same package as they stand. It reads the cards `visits` and `dependencies`. Its walk states `stands = "nothing_new"` for a call that would serve no unit that the ledger does not hold while the record cannot have grown since the latest answer, which holds where every recorded unit of the series is paid and either no time passes in the episode or time passes only inside calls that take time (the clock mode `declared`) and no such call came after the latest answer, an earlier step of the plan included, and such a step has no price, no time and no call. Its sections hold one table of what the record holds and what the board holds of it: for every series with a record, its first and its last recorded unit and how many of its recorded units are paid, closed where no time passes by one line which says that the record stays as it is.
- **Scripted cases** (`utils/public_rules/scripted_cases.py`, `preconditions.py`), over the precondition card, the texts and the briefing that the first piece brings: the walk of the precondition card, with the state after every state write and the requirement of every call, one call per step, and the steps left from the receipt line. Its ledger holds no open target and no pending repair. Its walk states `stands = "state_holds"` for a state write whose value the state already has at that step, with no call and no reason of the card for that step. Its one refusal mark is the phrase that the texts of the kind write into every answer to a call that was not carried out; the constant moves here and the texts import it. The value of the state at the start is the card's `initial`; where the card says `entry`, the ledger reads it from the task message, in which the briefing of the kind writes one sentence of the form `<state> is <value>.`, through `preconditions.initial_state(task, state, values)`, which stands beside the constant of that sentence. While the state is unknown, the check gives no reason on it.
- A rule kind of the asset layer names the function that reads its card in a walk by a new optional field, `plan_rule = "module:attribute"`. A card whose kind names none adds no rule to the check.
- **One shared loader.** `datasets/dataset_loader/scripted_case_replay/case_files.py`, `load_records(path, profile)`, reads the case file and the tool table that lie beside the asset file (`records.cases_file` and `records.tools_file`, by default `cases.json` and `tools.json`, in the form of the two files that `bottler_1` holds in the first piece). The built in kind `scripted_cases` takes it with the other parts of the replay. A dataset in that form states that kind, registers none and brings no module. A dataset whose release has another form keeps a kind and a loader of its own.

## Reading answers

An answer reaches the controller in one shape, whoever read it: `{"result" | "refused", "status": {...}, "notifications": [...], "observations": [...]}` (`agents/utils/answer.py`). `tool_answers.read(text, is_error, marks=()) -> dict` in `CausalDiagnosis/agents/utils/tool_answers.py` reads the tree's own format: a JSON object is the envelope of the action server, a last line that is a JSON object with the steps left is the receipt line, and a text answer that holds a refusal mark of the kind is a refusal. The reader of the receipt line moves there from the metering layer, which goes on importing it.

`agents/utils/answer_forms.py` turns a result into observations by the key `answer` of its row: `observations(form, result, ranges) -> (rows, note | None)`, with the ranges of `Ledger.ranges()`, so that this shared file names no key of the facts. A result that does not have the form gives no observation and a note that names the tool and the form.

| Form | Reads | Observation and identifier | Expectation a step can state |
| --- | --- | --- | --- |
| `object` | any JSON object | none; the result is kept whole | none |
| `records` | a list of served records, each with a number, a time, an optional label and named numeric values | one row per value, `obs/<label or n>/<field>`, marked inside or outside where the ledger holds a normal range for the field | `outside: [fields]` |
| `text` | a text, or an object with a text | one row per answer, `obs/<n>`, with the declared words that the text holds as whole words | `word: w` |

The core judges two things for every tool: a refused call is a surprise, and `no_event: [names]` fails when a notification with one of these event names arrives during the step. A standing step (The check) makes no call, so it is never refused and no notification arrives during it.

**The binding.** A tool that a server of the asset lists is called as listed, and a dataset with its own tools brings no code. A framework whose surface lists other tools than the asset's own declares a binding in its folder. It may write the arguments of the listed tool that carries a table tool, from the arguments of the step and from what the episode fixes; read the answer into the shape above; and state, without the facts and when the declaration is built, the row of every table tool it carries, with its description, its schema and the listed tool that carries it. A row that it states replaces the row of the engine's contract for that tool. The tools that it needs answered before it can write the arguments of another stand among those rows with the role task, so they are in the table from the first turn and join no batch of observe steps. Where it writes the arguments of a tool from the facts, or reads its answer with them, the facts tool stands among the needs of that tool, and `facts_first` holds a step on it back until the facts are answered, also where the facts tool stands earlier in the same plan, but for a knowledge read, which `facts_first` lets pass there. It never chooses among actions and never adds, drops or reorders a step.

```python
class AobBinding:                                # frameworks/AssetOpsBench/agents/causal_diagnosis_agent_surface.py, beside the present surface class
    flags: dict
    carried: dict                                # table tool -> (server, listed tool) that carries it; known without the facts
    def rows(self) -> dict                       # name -> ToolRow of every table tool it carries, with its description and schema; known without the facts
    def needs(self, tool) -> tuple
    def write(self, tool, args, view) -> (str, str, dict)       # (server, listed tool, arguments)
    def read(self, tool, args, text, is_error, view) -> dict     # the answer in the shape above
```

## The plan

```
{"follow": <rank of the candidate the plan follows, or null>,
 "why": "<one line>",
 "candidates": [{"<argument that states the answer>": <value>, ..., "cites": ["<identifier>", ...]}, ...],
 "steps": [{"tool": "<a tool of the table, or recall>", "args": {...}, "expect": {...}, "evidence": ["<identifier>", ...]}, ...]}
```

- A step is one call of one tool of the table, with the arguments of the tool's schema.
- An argument that the row fills is not written by the model: `evidence` (the identifiers the step cites that the board holds, with those the controller adds), `warrant` (the identifier of the warrant that covers the write), `closing_summary` (the text of the closing summary turn), `fixed:<value>`. A written value for such an argument is dropped with a note on the board.
- A step carries `evidence` only where its row fills an argument from `evidence`. On a step of another tool the reader does not read the key, whatever it holds, and the step holds no evidence.
- An argument that the row states under `omitted` and that a step or a candidate leaves out takes the value of the key. The reader writes it before the check, so the check sees every argument, and a candidate may name the arguments alone that differ from it.
- A step of a commit write with candidate arguments gives those arguments, or `"candidate": <rank>` under `args`; the reader lifts the rank into `Step.candidate`. Such a step that writes none of those arguments and no rank follows the candidate of rank 1: what the reply itself writes decides this, before any omitted value is written. A step of a commit write without candidate arguments gives the arguments of its own schema, and no rank is read on it.
- A step whose tool states `each` and that leaves that argument empty or leaves it out stands for one call per legal value, made in the order of the facts.
- The one step that calls no tool is `{"tool": "recall", "args": {"observations": [<first>, <last>], "series": <optional>}}`.
- `waits`: the controller calls the tool again with what is left while less time passed than asked, no expectation failed and the episode goes on.

`plan/plan_schema.py`: `Step(tool, args, expect, evidence, candidate=None)`, `Plan(follow, why, steps, candidates, raw)`, `Unusable(step, reasons)`, `Refusal(reasons, guard=None)`, `Parsed(plan, unusable, refusal)`, `Standing(clause, reason)`, `PlanError`, `OWN_STEPS = ("recall",)`. Exactly one of `Parsed.plan` and `Parsed.refusal` is set.

`plan/reply_parser.py`: `last_json_object(text)`, `parse_usable_plan(text, table, forms, plan_length) -> Parsed`, `parse_candidates(raw, table)`, `parse_differential(text) -> (entries, Refusal | None)`.

`plan/readers.py`: `read_name`, `read_count`, `read_number`, `read_text`, `read_list`, `read_object`, each `reader(value, ...) -> (value, None) | (None, reason)`.

`forms` is `{tool: {expectation key: reader}}`, the expectation keys that a step of each tool can state, each with the reader of its value. `expectation_forms(table, modules) -> dict` in `advice/__init__.py` builds it once per turn: for every row the key of its answer form (`outside`, a list of names; `word`, one of the row's words), `no_event` (a list of declared event names) where the declaration lists events, and the keys of every module's `forms(row)` (`top_rank`, a count of at least 1; `top_mass_at_least` and `recovery_at_least`, a number from 0 to 1; `effective_at_most`, a number of at least 1).

**The schema of a reply.** `plan/reply_schema.py` gives the form of a reply to the serving engine as a JSON schema: `plan_schema_of(table, forms, options) -> dict`, `differential_schema() -> dict`, `response_format(name, schema) -> dict`, which is `{"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": true}}`, and `digest(schema) -> str`. The schema of a plan is generated from the table, the forms and the options and from nothing of the facts, so one table has one schema. Its head holds `follow`, `why`, `candidates` and `steps`, of which `steps` alone is required, with at least one entry and, with the switch `plan_length`, at most that count. An entry of `steps` is one of the forms of the tools that a step can name and, where the switch `recall` offers it, of the own step: `tool` as that one name; `args` as the listed schema of the tool without the arguments that a fill writes, with the rank `candidate` where the row holds candidate arguments, with the least count of a key `minimum`, and with an argument required where the listed schema requires it and no key `omitted`, `each` or `candidate` covers it; `expect` with the expectation keys of `forms`, each in the shape of its reader; and `evidence` only where the row fills an argument from it. No form takes a key that it does not state. The generator writes the lower bounds of the readers and no bound that no listing, no row and no option states. The schema fixes the form of a reply and not every value: a text of blanks alone and an integer above the largest that the readers take pass it, and the reader refuses both as before. It writes the keywords `type`, `properties`, `required`, `additionalProperties`, `const`, `enum`, `anyOf`, `items`, `minItems`, `maxItems`, `minLength`, `minimum` and `maximum` and passes on no other keyword of a listed schema; an argument whose listed schema these keywords cannot write takes any JSON value, and the check judges it. The planning request and the orientation request carry their schema in the request field `response_format`, the two summary requests carry none, and with the switch `structured_output` off no request does. The controller adds the field to the request fields of the backend for that one call and takes it out again after the call. `Board.reply_schema` records the two digests, the count of the requests that carried a schema and whether the backend could carry the field. The reader stays the one way in for every reply, with a schema or without one: no code assumes that a reply fits the schema, and a backend that ignores the field is read as before. Where the engine refuses the field, the episode ends as an error with the engine's message, and no request is sent again without the schema, so that a run plays one condition.

## The check

`plan/plan_check.py`: `step_problems(plan, board, ledger, table, modules) -> list[(int | None, str)]` and `check_plan(...) -> list[str]`. A reason with step `None` belongs to the plan as a whole. The rule functions are in `plan/rules.py`, one per row below, each `rule(step, index, state) -> list[str]`, where `state` is `WalkState(table, board, walk, earlier, plan)` of `plan/plan_check.py`: the table, the board, the copy of the ledger as the steps before this one left it, those steps, and the plan. The check walks the plan on a copy of the ledger and reads four things: the table with its row keys, the rule cards in force through the `plan_rule` of their kinds, the limits in the status of the ledger, and the schema of every tool. A rule whose key no row states, whose card the asset does not hold, or whose limit the episode does not set prints no sentence and gives no reason. Every reason has its sentence in the prompt, in the same words, and one test holds the two together.

| Rule | Reads | Belongs to | Reason | Sentence of the prompt |
| --- | --- | --- | --- | --- |
| `known_tool` | the table | step | `t` is no tool of this episode, and the tools are `<names>` | no step that names a tool outside the table |
| `arguments` | the schema; the key `values` with the facts; the key `minimum` | step | `<a>` is no argument of `t`; `<a>` is a required argument of `t`; `<a>` must be of type `<T>`; `<a>` must be one of `<values>`; `<a>` must be at least `<n>` | no step whose arguments its tool does not take: an argument outside its line, a required argument left out, a value of another type, a count below the least that its line states, or a value outside those that the table or `<facts tool>` lists |
| `facts_first` | the key `facts`; `values`, and `omitted` where it names a path of the facts; a need of the binding that is the facts tool; the walk where the price or the time needs the facts; a knowledge read that stands after the facts tool in the same plan is not held back, and `arguments` judges its values at the call | step | leave this step out and plan it in the next request, once `<facts tool>` is answered, since `<its argument a, its price>` comes from that answer | before `<facts tool>` is answered, a step whose arguments or price need its answer, but for a knowledge read that stands after `<facts tool>` in the same plan |
| `order` | the key `after`; the needs of a binding other than the facts tool; an argument of a key `each` that the step leaves empty or leaves out, whose calls need the facts, so the facts tool must be answered or stand earlier in the plan | step | put `<x>` before this step, since `t` needs its answer | per row: no `<t>` without an earlier `<x>` |
| `single_commitment` | the key `commits`; the commitment on the board and earlier in the plan | step | leave this step out, since another commitment already stands (`<commitment>`) and the environment takes one | no commitment that differs from the one that stands: `<commit tools>` commits once in an episode, and a step that repeats the commitment makes no call |
| `end_last` | the key `ends` | step | `t` ends the episode, so it is the last step of a plan | `<end tool>` is the last step of a plan |
| `cites_observation` | the key `requires_observation` | step | `t` counts only when it cites an observation, and none is on the board or earlier in the plan | per row: no `<t>` before an observation it can cite |
| `candidate_rank` | the candidates on the board and in the plan | whole plan for `follow`, step for a rank | `follow <f>` is not among the `<n>` candidates shown; candidate `<r>` is not among the `<n>` candidates shown; no candidate exists yet, so list your candidates under candidates or write the arguments of `t` | no follow and no candidate rank outside the candidates shown |
| `candidate_values` | the key `candidate` | whole plan | candidate `<i>`: `<a>` must be one of `<values>`; candidate `<i>`: `<a>` is no argument that states the answer | no candidate of yours with a value that the table or `<facts tool>` does not list |
| `recall_rows` | the collapsed rows of the board | step | no row of observations `<a>` to `<b>` was collapsed, so nothing comes back | no recall of observations the board did not collapse |
| `expectation` | the forms of the tool's answer and the modules | step | `expect.<key>` is no expectation of `t`, and its expectations are `<keys>`; `expect.<key>` must be `<shape>` | per form, in the line of its tool |
| `open_repair` | the write class repair with `target`; the open targets of the ledger | step | leave this step out, since the `<repair word>` on the `<target word>` `<target>` that an earlier call began is still open at this step and the environment takes no second one on it; where the ledger knows the close, the reason ends with `; it closes at <time unit> <close>` | no repair write on a `<target word>` whose repair is still open |
| `precondition` | the card `precondition`; the state writes with `sets`; the state at the start | step | `t` needs `<state>` `<value>`, and `<state>` is `<current>` at this step; set it first with `<state write>` | no call while `<state>` has a value that its tool does not allow |
| `end_guard` | the keys `ends` and `target`; `Walk.end` | step | `<end tool>` comes before the close of a repair; or before the observation that verifies the repair, so call `<observe call>` at or after `<time>`, then `<end tool>` | no `<end tool>` before the close of every repair and the observation that verifies the last one, unless the limits leave no room for it |
| `budget` | the prices of the walk; the budget left | whole plan | end the plan before this step or make the steps up to it cheaper, since the plan's prices (`<p>`) exceed the budget left (`<b>`) and pass it at this step | no plan that `<the limits of the episode>` cannot pay |
| `time_limit` | the times of the walk; the time limit | whole plan | end the plan before this step or shorten the steps up to it, since the plan reaches `<time unit>` `<t>` but the episode ends at `<limit>` before `<end tool>` can run, and this step reaches it | the same sentence |
| `step_limit` | the calls of the walk; the steps left | whole plan | end the plan before this step, since the plan needs `<n>` counted calls but `<m>` steps are left, and this step passes them | the same sentence |

Step `k` of tool `t` is written `step k (t): ` ahead of every step reason. Before every call the standing clauses and the step rules are applied once more on the live ledger: a step that stands makes no call, and a reason of a rule there is a surprise.

**The opening rule** is `facts_first`, stated three times from one constant: on the board before the facts, as the reason, and as the sentence of the prompt. An asset without a facts tool prints none of the three.

**A plan runs up to its first step that the controller cannot use**, a step that does not read or that a rule marked step refuses. That step and the steps after it do not run and come back with their reasons. A reason marked whole plan runs nothing. A reply without a plan head, and a plan whose first step cannot be used, run nothing and take the step of a turn without a call.

**A standing step** is a step whose effect already stands on the board, so that a call for it could add nothing. It makes no call and does not stop the plan. `plan/rules.py` holds `standing(step, index, state) -> Standing | None` beside the rule functions, with the clause and the reason as its answer. The check asks it for every step of its walk, and the loop asks it again before every call. A step stands under one of five clauses and under no other:

| Clause | The step stands when | Reads | Reason |
| --- | --- | --- | --- |
| `episode_answer` | it calls the tool with the key `facts` or a tool of the role knowledge, and the board holds an answer of that tool for the same arguments or an earlier step of the plan calls it with them. Where the tool states `each` and the step leaves that argument empty, the step stands when every call it stands for is answered, and where some are, it makes the others alone | the key `facts`, the role, the key `each`; `Board.holds_answer` | STANDS_EPISODE |
| `surface_answer` | it calls a tool that a binding needs answered before another, the facts tool aside, the board holds its answer for the same arguments, and no time passes in the episode, which the status of the ledger tells by showing no time | `needs` of the binding; `Board.holds_answer`; `Ledger.status` | STANDS_SURFACE |
| `nothing_new` | the walk of the kind states it for the call: the call would serve nothing that the ledger does not hold, and the record cannot have grown since the latest answer | `StepCost.stands` of `Walk.step` | STANDS_NOTHING_NEW where no time passes and STANDS_NOTHING_RECORDED where time passes only inside calls that take time, with STANDS_COLLAPSED where rows of that series are collapsed and the run offers `recall` |
| `same_commitment` | it calls a commit write, a commitment by the same tool stands on the board or earlier in the plan, and every argument that no fill writes has the value it has there, once the reader has written the omitted values and the values of the candidate that a rank names | the keys `commits`, `candidate`, `fills` and `omitted`; `Board.commitment` | STANDS_COMMITMENT |
| `state_holds` | the walk of the kind states it for a state write: the state already has the value that the write sets | `StepCost.stands` of `Walk.step`; the key `sets` | STANDS_STATE |

The first clause rests on what the declaration says of a tool: the answer of the tool with the key `facts` and of a tool of the role knowledge holds for the whole episode, which is also how the board prints it. A tool whose answer can change during an episode is declared in another role. A step that `known_tool`, `arguments`, `expectation` or `candidate_rank` refuses does not stand. A standing step has no price, no time and no call in the walk, and no other rule of the table above gives it a reason. It is no observation that verifies a repair and none that a write can cite. Its record has the status `standing`, the clause under `observed["stands"]` and the reason as its note, and the last plan on the board shows the status. An expectation on a standing step is judged on the board as it stands, a form of an answer on the latest observation of its series that the board holds and a form of a module by the module: one that fails is a surprise with STANDS_EXPECTATION, so that no write after it runs. A commit write that the walk took as standing and that differs from the commitment when it is reached makes no call and is a surprise with the reason of `single_commitment`, and any other step that no longer stands when it is reached is called. A call of an observe tool that was made and served nothing that the board does not hold is a surprise with NOTHING_NEW_AFTER_CALL. A repair write never stands: it is called, or `open_repair` refuses it. A turn of standing steps alone makes no counted call and takes the step of a turn without a call, and it is no fallback. The next planning request names every standing step of the last plan with its reason (The four turns).

| Constant | Text |
| --- | --- |
| STANDS_EPISODE | `leave this step out, since the board already holds the answer of {call}, which holds for the whole episode` |
| STANDS_SURFACE | `leave this step out, since the board already holds the answer of {call} and no time passes in this episode, so the answer cannot change` |
| STANDS_NOTHING_NEW | `leave this step out, since the board already holds every recorded {time_unit} of {series} {name} and no {time_unit} passes in this episode, so the call would return nothing new` |
| STANDS_NOTHING_RECORDED | `leave this step out, since the board already holds every recorded {time_unit} of {series} {name} and on this row the record grows only inside calls that take time, none of which came since the latest answer, so the call would return nothing new` |
| STANDS_COLLAPSED | `; its rows of observations {first} to {last} stand collapsed in a summary row, and a {recall_step} step brings them back` |
| STANDS_COMMITMENT | `leave this step out, since the board already holds this commitment ({commitment}) and the environment takes one` |
| STANDS_STATE | `leave this step out, since {state} is already {value}` |
| STANDS_EXPECTATION | `{stands}; its expectation, held against the board as it stands, fails ({detail}), and no call of this step can change that, so state an expectation only on a step whose answer is new` |
| NOTHING_NEW_AFTER_CALL | `the call returned no {time_unit} that the board does not hold, since the record ends at {time_unit} {last}; plan this step again once the record has grown, or leave it out` |

`{call}` is the name of the tool with the arguments of the step, `{name}` the value of the argument that the key `series` names, `{commitment}` the commitment as the status of the board states it, `{state}` and `{value}` the state and the value of the key `sets`, `{stands}` the reason of the clause, `{detail}` what the judgment found, and `{recall_step}` the name of the own step. All of these constants but STANDS_STATE stand in the shared file `agents/utils/core_texts.py`, as do the reasons of `single_commitment` and `open_repair`, since the present agent prints them as well; STANDS_STATE is a constant of the plan package.

**No text raises.** Every place where a text that the agent did not write enters it has one boundary function that takes any text and returns: `parse_usable_plan` and `parse_differential` for a reply of the model, `tool_answers.read` and the `read` of a binding for an answer, the reader of a module for an advice answer. Inside, every value goes through a typed reader that tests the type before the use and returns the value or a named reason: `read_name` (a string that is one of the names given), `read_count` (an integer that is no boolean, a float of integral value, or a string of the digits 0 to 9 alone, at least the minimum given), `read_number` (an integer or a float that is finite and inside its range). One last guard at each boundary turns any exception into the refusal "the reply could not be read" (for an answer, "the answer of `<tool>` could not be read") and records the class of the exception in the record of the turn (`PlanRecord.guard`) or of the call (`ToolRecord.guard`). The proof is a generator and no list of cases: for every position of the plan schema it places odd values one at a time and in pairs (another type, a list, an object, null, a boolean, an empty value, a huge integer, a huge float, the spellings of NaN and infinity, a negative, a doubled sign, digits of another script, a very long string, deep nesting, a duplicate key, a key in another letter case), through the path that the driver takes, on the fixture declarations of step 0, before and after the facts, with and without the modules. No call raises, and no generated text reaches the guard.

## The prompt

`prompt/core_protocol.py` holds the texts below as named constants, and `prompt/assembler.py` builds the turns: `system_prompt(table, board, options, modules)`, which reads the facts and the limits of the ledger from the board, `orientation_turn(...)`, `planning_turn(board, ..., retry=None, not_run=None, stood=None)`, `earlier_turns_turn(...)`, `summary_turn(...)`. A block is core (a constant with slots), rendered (built from declared fields) or given by a module. No sentence is edited after it is written: with a part switched off, its block or sentence is left out. The system prompt changes once in an episode, when the facts arrive. The texts below that the present agent prints as well live in one shared file, `agents/utils/core_texts.py`, which the maintainers add with the prompt texts of the present agent: `prompt/core_protocol.py` imports them and holds the others. Until that file is in the tree, `prompt/core_protocol.py` holds every constant.

| Block | Kind | Printed when |
| --- | --- | --- |
| ROLE | core | always |
| TOOLS: the head and one line per tool of the table, the advice tools left out | rendered | always |
| EXPECTATIONS | core, with a rendered sentence on the events | always |
| EVIDENCE, with the identifier forms | core, rendered forms | always |
| CANDIDATES | core, rendered commitment | always |
| BOARD | rendered from the parts that the table and the kind have | always |
| the blocks of the advice modules | module | one per active module |
| PLAN, with RECALL and BATCH | core | always; RECALL and BATCH by their switches |
| RULES: the sentence of every rule in force | rendered | always |
| LOOP | core | always |

Slots in braces are nouns of the vocabulary, names of tools, names of limits, or a clause that the assembler renders. No other word depends on a dataset. Two slots are words of the plan form of the agent that prints a text, so that the present agent prints the same constants from the shared file: `{step_key}`, the key of a step that names what it calls, which is `tool` here, and `{plan_head}`, the keys of the head of a plan that the check judges as a whole, which is `follow or candidates` here.

ROLE
```text
You plan the work on one {noun} for a controller. The controller keeps a working memory, the board, and calls the tools that your plan names, and no other tool of the {short_noun}. The task of the episode stands above the board in every request. Each turn you receive the task and the board, and you return one JSON object with nothing after it.
```

TOOLS (head)
```text
The tools. Each line below is one tool that a step of your plan can call, with its arguments. Every tool has a role: {roles}. {writes} An argument that the controller writes is marked so, and you leave it out.
```

`{roles}` names the roles that the table holds, in the order observe, knowledge, act, task: "observe for the tools that read the {short_noun}", "knowledge for what is known about it", "act for the work on it", "task for the tools that read or answer the task". `{writes}` names the write classes that the table holds: "A tool that changes something is a write, and its line names its class: " then "a repair restores a part of the {short_noun}", "a probe changes it in order to test it", "a state write sets a public state", "an answer records your answer to the task".

EXPECTATIONS
```text
Expectations. A step may carry an expectation under "expect", a statement about its answer that you are willing to be stopped by. The controller checks it when the answer arrives, and an expectation that fails is a surprise. A call that the {short_noun} refuses is a surprise whatever the step expects. The line of a {step_key} names the expectations its steps can state, and a step without one is judged by its refusal alone.{events}
```

`{events}` is printed where the declaration lists events, with a space before it:

```text
Any step may also state {"no_event": [<names>]}, with names among {event names}: a notification of one of these events during the step is a surprise.
```

EVIDENCE
```text
Evidence. Every answer puts rows on the board, and every row has an identifier.{carried} Cite only identifiers that appear on the board; the controller drops any other and says so.
```

`{carried}` is printed where the table holds a tool that fills an argument from `evidence`, with a space before it, and `{evidence_steps}` names those tools:

```text
A step of {evidence_steps} carries "evidence", a list of the identifiers from the board that justify it, the few that moved you, and no other step carries it.
```

CANDIDATES
```text
Candidates and your answer. A candidate is one possible answer to the task: a value for each argument that states the answer, {candidate_arguments}. List your candidates best first under "candidates", and name the one your plan follows by its rank under "follow". The board keeps your latest list. {commitment}
```

BOARD
```text
The board. It shows, in this order: {parts}; and the status, with every limit that is left. Rows that you have planned on may collapse into one summary row, and a collapsed row stays citable by its identifier.
```

`{parts}` joins the clauses that hold: "what holds for the whole episode, which is the public facts once {facts_tool} is answered" (a table with a facts tool) with "and what the knowledge tools returned" (a table with a knowledge tool); "your summary of earlier turns, when one was written"; "the observations with their identifiers"; "your candidates"; "the open work on the {short_noun}" (a kind whose ledger holds open work); "the surprises"; "your last plan with the state of each step".

BOARD is followed by the first text below, and on a kind whose ledger holds the record of a series by the second, the sentence of that section of the kind:

```text
The status also states which answers the board holds for the whole episode and which calls were made more than once with the same arguments.
For every {series} with a record, it states how much of the record the board holds.
```

PLAN
```text
The plan. Return
{"follow": <rank of the candidate you follow, or null>, "why": "<one line>", "candidates": [{...}, ...], "steps": [{"tool": "<a tool>", "args": {...}, "expect": {...}}, ...]}
with at least one entry under "steps"{count}. A step is one call of one tool with the arguments of its line. "candidates" is optional. The why line names the identifiers that moved you.
```

`{count}` is " and at most {n} entries; a longer plan is refused and you are asked again" with the switch `plan_length`, else ", as many as {limits} pay; a plan they cannot pay is refused and you are asked again".

RECALL
```text
One more step calls no tool: {"tool": "recall", "args": {"observations": [<first>, <last>]{series}}}, with no expectation. It costs nothing: the controller puts the full rows of those observations, which the board collapsed into a summary row, back on the next board, once. A plan of recalls alone makes no call and still takes one step.
```

BATCH
```text
Consecutive steps of the role observe run as one batch: the controller makes the calls one after the other and checks every expectation after the last, so a batch pays for all of its calls before it sees any of them.
```

RULES (head)
```text
The controller refuses, whatever you write, only what the {short_noun} or the plan itself would fail:
```

followed by the sentences of the rules in force, joined by semicolons, in the order of the table of the check.

LOOP
```text
The controller runs your steps while every expectation holds and calls you again when one fails, with the surprise written on the board, and when the steps of your plan have run, so plan several steps and attach an expectation you are willing to be stopped by. The steps of a plan before its first step that the controller cannot use, a step that does not parse or that it refuses, run as you wrote them, and that step and the steps after it do not run and come back to you with the reasons. A reply that does not parse as a plan, a plan whose first step cannot be used, a plan whose {plan_head} the controller refuses, and a plan that {limits} cannot pay run nothing, and you are asked again with the reasons. A turn that calls no tool of the {short_noun} still takes one step.
```

LOOP is followed by

```text
A step whose effect already stands on the board makes no call and does not stop the plan: its status in your last plan is standing, the steps after it run, and the next request names it with what the board already holds, so leave such a step out. An expectation on such a step is held against the board as it stands, and one that fails is a surprise.
```

**Rendered texts.**

| Text | Template | Fields it reads |
| --- | --- | --- |
| a tool line | `- {name} ({role}{, class write}): args {...}. {description} {key sentences} {expectations}` | name, role, write of the row; description and schema of the listing |
| an argument | its type from the schema; "one of {values}" for an enum; "one of the {path} that {facts_tool} lists" for a key `values`, with the values themselves once the facts are on the board; "written by the controller" for a fill; "or empty, for one call per value" for the argument of a key `each`; "at least {n}" for a key `minimum`; "0 is read as 1" for the argument of a key `waits`; "left out, it is {value}" or "left out, it takes the value that {facts_tool} lists for it under {path}" for a key `omitted` | schema, `values`, `fills`, `each`, `minimum`, `waits`, `omitted`, facts |
| key sentences | `facts`: "Its answer holds the public facts of the episode." `commits`: "It commits your answer." `ends`: "It ends the episode." `sets`: "It sets {state} to {value}." `target`: "It repairs what {argument} names." `after`: "Only after {x}." `waits`: "The controller repeats the call until the time asked has passed." `words`: "Its answer holds one of the words {words}." `requires_observation`: "It must cite an observation." `candidate`: `Give these arguments, or "candidate": <rank> to take them from that candidate.` | the row keys |
| the requirement of a tool | "Needs {state} {value}." or "With {argument} {values}: needs {state} {value}." | the card `precondition` |
| the price and duration of a tool | the rule in the words of the contract | the rows of the tools in the facts, once they are on the board |
| the expectations of a tool | `records`: `expect {"outside": [<fields>]}: a listed field that the answer serves inside its normal range is a surprise; a name stands for the field of that name or for every field whose name begins with it and an underscore{example}, and a field that the answer did not serve is not judged`, with `{example}` ", such as {example_field}" where the vocabulary holds one. `text`: `expect {"word": <one of its words>}: holds when the answer holds exactly that one of its words`. Then the forms of the modules. A tool without an expectation prints no such part. | `answer`, `words`; the modules |
| the sentence of a call with nothing new | at the end of the line of an observe tool with the key `series`: "Where no {time_unit} passes in the episode, or where time passes only inside calls that take time and none came since the latest answer, a step of this {step_key} on a {series} whose every recorded {time_unit} is on the board makes no call and stands; elsewhere the record may have grown, and the call is made." | `series`; the terms `time_unit` and `series`, without which the sentence is not printed |
| `{candidate_arguments}` | the arguments under `candidate` of the first commit write that holds any, joined, with "each one of its values above" where their rows state values, and "; one you leave out takes the value that {facts_tool} lists for it under {path}" where the row states `omitted` for them | `candidate`, `values`, `omitted` |
| `{series}` | where a row states the key `series`: `, "series": <one {series term}>` inside the args of the recall, and after the block the sentence "Name one {series term} to bring back its rows alone." | `series`; the term `series` |
| `{commitment}` | "You commit your answer with {commit tools}, once in an episode." and "{end tool} ends the episode."; where one write does both, "You commit your answer with {tool}, which also ends the episode." | `commits`, `ends` |
| `{limits}` | the names of the limits the episode has, joined: "the budget", "the {time_limit term}", "the step limit"; "the limits of the episode" while the ledger knows none | `Ledger.limits` |

**The four turns.**

Orientation: the system prompt, the heading, the task, the facts and the knowledge answers as the board prints them, then

```text
No observation has been made yet. Write the differential: for every cause that could explain the task, what you expect to observe if it holds and the identifiers behind it. Set no probability. Return {"differential": [{"about": "<a cause, in the words of the tools and the board>", "expected": "<one line>", "cites": ["<identifier>", ...]}, ...]} and nothing after it.
```

Planning: the system prompt, the heading, the task, the board, then the first text below. Before the facts the second text follows it, and the board then reads the third. Candidates of a plan made before the facts are not kept, and the next request says so in one sentence. After a cut plan the request holds a section under the fourth text, with the steps that did not run and their reasons, closed by the fifth. After a plan that held standing steps the request holds one more section, after that one, under the sixth text, with one line per standing step in the form of the seventh, closed by the eighth.

```text
Objective: the task above, within the limits on the board. Return the plan alone, the one JSON object described above, with no text before it and nothing after it.
Before {facts_tool} is answered this plan holds only the opening reads, and the next request asks for the rest.
No answer of {facts_tool} yet, so the board holds nothing about the {short_noun}. A plan made before {facts_tool} is answered holds only the opening reads: {facts_tool} first, then {the knowledge tools}. Plan every other step in the next request, once its answer is on the board.
### Steps of your previous plan that did not run
The steps before them ran as far as the board shows. Plan from the board as it stands now, as the JSON object only.
### Steps of your previous plan that made no call
- step {number} ({step_name}): {reason}
Their effect already stood, so the controller made no call for them and went on with the steps after them. Plan from the board as it stands now, without these steps, as the JSON object only.
```

Summary of earlier turns: the request of the present agent, with its list of what the board keeps in general words:

```text
The board keeps the task, the public facts, what the knowledge tools returned, the open work, the advice, the refuted repairs, the latest observation of every series and the status.
```

Closing summary: every write of the episode with its arguments and what it returned, every verdict, the limits line, then

```text
Write the closing summary of the episode in a few sentences of plain prose: your answer and the evidence behind it, each write and what it returned, what each verification said, and the state of the {short_noun} as the answers show it. Claim only what an answer on the board supports. Return plain text, no JSON.
```

**Switches.** `agent_options.py` holds `Options`, a data class with one field per switch, `label()` and `to_json()` as the options of the present agent have them, and one registry, `SWITCHES`, a tuple of records: name, default, label suffix, flag, owner. The driver keeps the flags and the recorded options of today, and a switch that a module owns is refused by name on a run whose table lists no tool of that module's advice kind.

| Switch | Default | Label suffix | Owner | With it off or set |
| --- | --- | --- | --- | --- |
| `plan_length` | none | `len<N>` | core | the count sentence of PLAN names the count |
| `expectations` | on | `noexp` | core | a failed expectation is recorded and the plan runs on; LOOP opens with "The controller runs every step of your plan and calls you again when its steps have run" |
| `backtracking` | on | `noback` | core | a refuted repair is a surprise and is not recorded as a set |
| `expectation_tolerance` | 0.05 | none | core | the numeric forms of the modules |
| `parallel` | on | `seq` | core | no batch and no schedule section |
| `recall` | on | `norecall` | core | no RECALL block and no own step |
| `summary_turn` | on | `nosumturn` | core | no summary request at the compaction |
| `binding` | on | `raw` | core | the table is the listed table, and the binding writes no arguments; refused by name on a surface without a binding |
| `structured_output` | on | `noschema` | core | no request carries the schema of its reply |
| `branches` | on | `nobranch` | ranking module | no section of the candidate plans |
| `single_hypothesis` | off | `single` | ranking module | the first candidate plan alone |
| `reads` | `gain` | `e10`, `ec2` | ranking module | which reads the advice names |

## The loop

```
open one session per server; read the listing; build the table
build the ledger, the board, the modules
while the episode is not over:
    mark the turn
    ask for the differential once: at the start of the first turn in which the board holds the facts, where the table
        has a facts tool, and an answer of every tool whose row states orients; at the latest before the first step
        that is no opening read (a call of the facts tool, of a knowledge tool, or of a task read that lets no time pass),
        on what the board holds then
    every module: before_turn
    one model call, which carries the schema of a plan; parse_usable_plan; step_problems; cut the plan at its first step that cannot be used
    run the kept steps in order; a run of consecutive observe steps that do not stand is one batch:
        own step recall: no call
        a standing step: no call; status standing; its expectation judged on the board as it stands
        fills; module before_step; the write of the binding where one stands; the call; the reading of the answer
        ledger.record; board.record_call
        after a batch: module after_batch; the expectations of its steps; the verification of every pending repair
            that an observation of the batch verifies (module on_pending, else ledger.verdict)
        a refused call ends its batch; the first surprise ends the plan
        compact, as today
    a turn without a counted call takes one step
closing summary where the write that ends the episode did not carry it
```

`controller.py`: `AgentRunner(llm, server_paths, declaration, options=None)`, `async run(question) -> EpisodeResult`, `request_settings(llm, temperature=None)`. The runner, the options and the result of the present agent are named `CausalLoopRunner`, `CausalLoopOptions` and `CausalLoopResult`; the new package names them `AgentRunner`, `Options` and `EpisodeResult`.

`episode_result.py` keeps the four records of the present agent with these fields, which the driver, the viewer and the tables read:

```python
@dataclass
class ToolRecord:                                # one tool call, in the shape that the audit report reads
    step: int; server: str; tool: str; tool_args: dict; response: str
    error: str = None; duration_ms: float = None
    guard: str = None                            # the class of the exception where the last guard read the answer
@dataclass
class StepRecord:
    tool: str; args: dict; expect: dict          # tool in place of the present field kind
    status: str                                  # done, surprise, failed, skipped or standing: a step whose effect already stood and that made no call, with the clause under observed["stands"]
    observed: dict = field(default_factory=dict); note: str = ""
@dataclass
class PlanRecord:                                # a reader takes the steps of a plan from steps and never assumes that plan holds steps
    index: int; source: str; plan: dict; candidates: list
    after_surprise: bool = False; steps: list = field(default_factory=list); surprise: dict = None
    parse_errors: list = field(default_factory=list); check_problems: list = field(default_factory=list)
    schedule: dict = None
    cut: dict = None                             # {"step", "steps", "reasons"} of a plan cut at its first step that could not be used
    guard: str = None                            # the class of the exception where the last guard read the reply
@dataclass
class EpisodeResult:
    question: str; answer: str; plans: list; trajectory: list
    model_calls: int; surprises: list; revisions: int; fallbacks: int
    mode_of_first_plan: str                      # filled by the rule of change 5
    differential: list; board: dict; end_reason: str
    error: str = None
    server: bool = True                          # whether the diagnostics server was registered
    advice: tuple = ()                           # the advice kinds of the modules that were active
    end: str = None                              # why the runner stopped, with the four values of the present agent
```

The controller calls a tool of the asset only when a step names it. The calls it makes of its own are those of a module on an advice tool, which change nothing on the asset and count no step.

`working_memory/board.py`: `Board(table, ledger, question, options)`, which holds the ledger of the episode, with `record_call(step, call, answer, observations)`, `facts`, `knowledge`, `observations`, `candidates()`, `commitment`, `status`, `known_ids()`, `render(modules)`, `to_json(modules)`, and the compaction and recall of the present board under their present names (`compact`, `mark_turn`, `replaceable`, `without`, `apply_summary`, `recall`, `recallable`, `drop_unfit_recall`). For the standing steps and the notes it also holds `holds_answer(tool, args) -> bool` and `repeated() -> list`, both read from the calls that `record_call` recorded, `add_note(text, tie=None)` and `retire_notes(passed)`, and the field `reply_schema`. `schedule.py`: `schedule(board, ledger, plans, named, reserve, refuted) -> Schedule`, the present schedule of candidate plans over `Walk.step` and `Walk.needs`.

**What the status states.** From the calls that `record_call` recorded, the status of the board states the answered calls of the facts tool and of the knowledge tools, the answered calls that a binding needs first where no time passes, and every call of a step that was made more than once with the same arguments. The record of a series is stated by the section of the kind (Rules of a kind).

| Constant | Text |
| --- | --- |
| HELD_EPISODE | `On the board for the whole episode: {calls}.` |
| HELD_SURFACE | `Answered, and unchanged since no time passes in this episode: {calls}.` |
| REPEATED | `Called more than once with the same arguments: {calls}.` |
| REPEATED_ITEM | `{call} {count} times` |

**When a note leaves.** A note enters through `add_note` with the state it speaks of as its tie, and `retire_notes` takes it off the board once that state has passed: a note written before the facts leaves when the facts are answered, the note of a differential that could not be read when a differential is on the board, and the note of a failed call of an advice tool when an answer of that kind arrives. Notes of refused, cut or unread plans that are equal but for the number of their turn stand once, as the latest, which closes with `(seen before; earlier turns with the same reasons: {count})`. A note without a tie stays until a summary of earlier turns replaces it, a note that such a summary names stays, and the counts of the turn marks follow a retirement, so that the order of the entries that a summary replaces stays exact.

## Advice modules

A module serves one advice kind (`evidence`, `ranking`, `plan_assessment`, `outcome_check`); the backends of the second piece name these kinds. `advice/__init__.py` holds the call site and `modules_for(table, options) -> list`, one module per kind for which the table lists a tool and a module exists. A module never reads a tool name or writes an argument name: it asks for the tool of its kind and hands named inputs, and the key `call` of the row says which argument takes which. For a tool of a backend of the second piece the key is derived from its record: a plan assessment takes `calls`, the calls of the plan from the step on as JSON text, in its one required parameter, and an outcome check takes `warrant` in its one required parameter. The fields of an answer per kind, the reader of warrants and the tuple of the kinds move to `utils/public_rules/advice_answers.py` and the backends import them from there: `ADVICE_KINDS`, `fields_of(kind) -> tuple`, `warrant_fields_of(kind) -> tuple`, `warrants(kind, answer) -> list`, with the fields held as tuples of pairs.

```python
class Module:                                    # advice/ranking.py RankingModule, advice/warrants.py WarrantModule; no base class
    kinds: tuple                                 # the advice kinds it serves
    switches: tuple                              # the switches it owns
    def __init__(self, tools, table, options)    # tools: {kind: (tool name, advice record)}
    def blocks(self) -> list                     # its block of the system prompt and its sentence of BOARD
    def forms(self, row) -> dict                 # the expectation keys it judges for a step of this row, each with its shape
    def step_problems(self, step, index, state) -> list
    def candidates(self) -> list                 # the candidates it ranks, which take the first ranks
    def sections(self, board) -> list            # what it prints on the board, each with its place
    def known_ids(self) -> set
    def evidence_for(self, step, row) -> list    # identifiers it adds to the evidence of a write
    async def before_turn(self, ctx) -> None
    async def before_step(self, step, row, ctx) -> dict      # fills it writes (the warrant), notes
    async def after_batch(self, steps, answers, ctx) -> list # the judgment of its forms, one per step
    async def on_pending(self, pending, observation, ctx) -> dict | None   # its verdict on a closed repair
    def to_json(self) -> dict
```

`ctx` gives a module the board and the ledger to read and `await ctx.ask(kind, inputs)`, which calls the tool of that kind, records the call in the trajectory and counts no step. For a standing observe step, which is handled alone and makes no call, `after_batch` gets that step with None as its answer, and a module judges its forms on what the board holds.

**The ranking module** (kind `ranking`) holds everything that reads a ranking today: it asks the ranking before a turn when the observations, the closed repairs or the clock moved, before a commit write and before a repair is verified when it is stale, once after a batch, and for a standing step only when it is stale; its candidates take the first ranks; it judges `top_rank`, `top_mass_at_least` and `effective_at_most` on an observe step and `top_rank` on a commit write with candidate arguments, by default 1 for a commit by a ranked candidate; it prints the ranking, the advice with the gain per series, the schedule and the candidate plans; its rules refuse a `top_rank` outside the ranked candidates and a commit by a ranked candidate before any ranking; it owns the reserve for the repair and the advice lines. Its block is built from the sentences that the present prompt holds on the ranking, with every tool name replaced by the tool of the kind and every word of the rig by a term of the vocabulary or a field of the answer. For the rig it reads:

```text
Advice: the ranking. Before a planning turn in which the evidence or the clock moved, the controller asks rank_root_causes and shows the answer as advice. The ranking lists the candidate answers now with their posterior mass, the effective number of hypotheses of the belief, and the decision fields of the server's rule: read_first, the separating test with its price and expected loss, and the expected loss of acting now. Per sensor group it gives the information gain of one cycle and that gain per unit cost, which ranks the reads, the server's read batch, and the lines of advice: the reads to take while the ranking says read first, the repair plan to act on and, after a refuted plan, the other plan. The table of candidate plans, when present, walks each ranked candidate forward: its confirm test, its repair plan with the warrant that every repair write of the plan cites, the cycles to a restored rig, the predicted recovery if acted, and the expected loss and the cost both ways. The ranked candidates take the first ranks and yours follow them. It is advice: you decide every step, and the controller makes only those your plan names. You never write a probability of your own: every number of the ranking comes from the board. The advised reads are the read the server advises. The advised repair plan is the rank the server advises to commit, with one repair write for each of its faulty components. A step of the role observe can expect {"top_rank": <rank in the current ranking>, "top_mass_at_least": <mass>}: after the observation, the set of faulty components of the candidate at that rank should be the set on top of the new ranking, with at least that mass. A step of submit_diagnosis can expect {"top_rank": <the rank of the candidate on top when the write is made>}. The schedule on the board, when one is shown, lists the reads that the server names and the reads that several candidate plans share, each once with the plans it serves, and after a refuted plan what each remaining plan still needs.
```

**The warrant module** (kinds `plan_assessment` and `outcome_check`) keeps the warrants of the episode and issues none itself. Before a repair write that no warrant covers it asks the plan assessment once; the write goes out with the warrant in the argument that the row fills from `warrant`, or marked unwarranted. It judges `recovery_at_least` on a repair write where the warrant states a recovery. Where an outcome check is listed, it asks it with the warrant of a repair whose every write has closed and that an observation can verify, and a verdict that is not consistent refutes the repair. For the rig its block reads:

```text
Advice: warrants. A warrant is an identifier that the diagnostics server issues for a repair, with what it predicts of it, and the board lists every warrant of the episode. The controller writes the warrant of a repair step: where no warrant on the board covers the step, it first asks estimate_intervention_effect for one, and the write then goes out with the warrant or, where the server issued none, without one and marked so. A repair step can expect {"recovery_at_least": <probability>}: a warrant that predicts less is a surprise. Once a repair has closed and an observation that can verify it is on the board, the controller asks verify_outcome whether the outcome is consistent with the warrant. A repair whose outcome is not consistent is refuted: the board records it, and the advice names another plan.
```

On a backend whose plan assessment takes the calls of a plan, the second sentence names that tool and adds that the answer of the check stands on the board: the calls it would insert, the calls of the plan that break a published requirement, the calls it cannot check, and the cost of the plan.

**Without advice.** No module is built. The candidates are the model's own, an observation is judged by the forms of its answer and by `no_event`, a repair write goes out without a warrant, and a closed repair is judged by the kind where it has a verdict of its own and is not judged where it has none. For recorded levels the board then holds the sentence of the kind: "The controller judges a repair itself: on the first observation that can verify it after its close, a value outside its normal range refutes the repair." The kind `evidence` has no module.

## The two reimplemented designs

Both designs read one list of sub-agents per run, built from the listed rows: one sub-agent for every role that some listed tool carries, which takes every listed tool of its role and no other. `SubAgentSpec` gains `role` and `tools`, `role_surface(declaration) -> tuple` builds the specs once per run from the listed rows of the declaration, since both runners register their sub-agents before a session opens, and `Runtime.tools_of` keeps a listed tool only where the spec names it.

| Role | Sub-agent | Order |
| --- | --- | --- |
| observe | IoTAgent | first |
| task | TaskAgent | second |
| knowledge | FMSRAgent | third |
| act | WOAgent | fourth |
| advice | CausalDiagnosisAgent | fifth |

- A framework may bring a spec for one of its servers. Where the tools of a role are exactly the listed tools of one server for which a spec stands, the sub-agent is that spec, with every byte of its texts. On the rows of AssetOpsBench's servers this gives the five sub-agents of today.
- Elsewhere the texts follow one rule: the description "The `<name>` works with the `<role>` tools of the `<short noun>`: `<t1>`, `<t2>` and `<t3>`.", the request field "A request to the `<role>` tools of the `<short noun>`."; for the advice agent one phrase per advice tool by its kind.
- A state write is served also to the sub-agent of every role that holds a tool whose requirement the precondition card names.
- A call that names a tool outside the set of its sub-agent reaches no server: the sub-agent holds no such tool, and the runtime logs the attempt under `refused` in the record of that sub-agent.
- The step limits and decoding values of the present specs do not change. Every call that samples carries a seed: the first four bytes of the SHA-256 of `<seed of the episode>:<index of the call in the episode>` as a non-negative integer, sent by the completion route, which already takes a seed, and recorded with the call.
- The proof that the rig keeps its bytes: the hash of every text that both designs send on the rows of AssetOpsBench's servers (among them the examples of the supervisor with and without the diagnostics server, the description and the examples of each sub-agent, and the list of agents of the planner), taken before the change and after it, equal.

On those rows the three designs do not write the same calls: the two reimplemented designs write every argument of every listed call, and a step of the CausalDiagnosis agent names an action of the asset whose arguments the binding writes. The switch `binding` measures the difference.

## Identity and stored tables

Today the code digest of an asset's label tables covers every Python file under `CausalDiagnosis/environment/` and `CausalDiagnosis/datasets/` outside test folders and the folders of other assets, the digest of the fitted tables covers the whole package of the diagnostics server, and the identity of a run hashes whole folders by list (`CausalDiagnosis/experiments/run_verification/run_identity_file_lists.py`). So a file of another kind moves the tables and the runs of every asset.

The file lists gain three families, each identity for a run that uses it and bound for any other, as an agent list is today.

| Family | Files | Declared |
| --- | --- | --- |
| `asset:<asset_id>` | every file under the folder that holds the asset file, tests excluded | by rule, from the folders of the asset packages |
| `kind:<kind>` | the shared code that only assets of that kind load | `utils/kind_files.py`, a data module that the lists and the digest both read |
| `framework:<Name>` | every file of a framework folder that its declaration puts in no list of the core | by rule |

`identity_of(files, agent, asset_id)` takes the asset of the run and derives its kind and its frameworks. The digest of files leaves out, beside the folders of other assets, every path of a kind list of another kind. The code section of the results index records the analysis and rendering files that lie in an asset folder or a framework folder under a key of that asset or framework, so a dataset that brings such a file adds a key of its own and changes no other entry.

Two tests prove it. Planting moves nothing: a made-up dataset is planted in a planted framework folder, and the label digest, the fit digest and the identity of each agent on every other asset stay, with positive controls (one byte added to a file of an asset moves that asset alone). Every module loaded while an asset is played lies in its lists, for each of the six settings; this second test belongs to the gate, since it plays the new agent.

Every change that moves a digest or an identity lands before the first run that a table reads. A pull request names what it moves.

## Folders

FaultArena's own files, which the first piece adds under `datasets/faultarena/` and `mcp_servers/FaultArena_Actions_MCP/`, move to a framework folder of their own, and the code of the kind stays shared.

```
frameworks/FaultArena/
    __init__.py                  the declaration, read as data: DATASETS, COMMANDS, LISTS
    README.md
    datasets/faultarena/         the dataset
    action_server/               the shared action server of scripted cases under the asset's name
```

A framework declares a folder of datasets by a new field of plain names, `DATASETS`, read by `CausalDiagnosis/utils/frameworks.py`, and `CausalDiagnosis/datasets/__init__.py` adds every declared folder to the search path of its package. The package of the asset keeps its import name, so every reference to it keeps its text. No folder between the framework and the dataset holds an `__init__.py`, and a test refuses the second import name. Every place that finds the datasets by listing the folder `CausalDiagnosis/datasets/` asks the registry of asset packages. The move is part of step 0 of the task page.

## The last change

The new agent is written in `agents/Causal_Diagnosis_Agent_next/` and the driver knows it as the agent `causal_next`, so both agents play the same starts through the same driver. What replaces something is written beside it: the binding beside the present surface class, the role surface of the two designs beside the present registry. Every setting of the present agent stays playable and tested at every commit.

Before the last change the two agents are compared on development starts alone, and nothing observed on a start of the test set shapes the agent. With a stand-in model, through the driver: with the diagnostics server both agents play the rule of the present agent, which for the new agent is a stand-in model that writes its plans as JSON; without the server both play one scripted model whose replies a helper translates into the new schema. Per episode the writes on the asset, in order, and the verdict are equal, and the call counts differ only by the changes below. With the served model the maintainers read a paired table per start; no value of it enters a public text.

What changes on the rig on purpose:

1. A step names a tool, and every prompt text changes.
2. The controller no longer calls the change summary after a wait. Closes and alarms reach the ledger through the notifications of every answer. The public rule of completing a repair and the one line of the diagnostics server that reads it lose the term for the summary in the last change.
3. On the rows of AssetOpsBench's servers the failure modes are read by one planned call per component.
4. A read of the time before the facts, and a wait where no clock runs, are no longer refused by the check.
5. The plan has no mode. The column `first_plan_mode` is filled by rule: `test_first` where the first step of the first plan that is no facts or knowledge read has the role observe, else the name of that step's tool.
6. `plans.json`: a step record holds `tool` in place of `kind`, the differential holds `about`, `expected` and `cites`, a plan holds `follow` in place of `branch`, and the board record holds the sections of the ledger and of the modules under their names.
7. The label of a run gains `raw` for the switch `binding`.
8. The core gives a step no expectation that the plan did not write. The default of a commit by a ranked candidate stays in the ranking module.
9. The estimate before an order is asked by the warrant module; the call and its place in the trajectory are the same.
10. On a surface with a binding, a step on a tool whose arguments the binding writes from the facts, or whose answer it reads with them, is held back until the facts tool is answered, but for a knowledge read that stands after the facts tool in the same plan. The present check lets a knowledge read of such a tool stand ahead of the facts tool, and its call then goes out with an argument left empty and its answer is read without the facts.

The last change is one pull request: the new package takes the name of the present one, the present package and its tests leave, the binding and the role surface replace what stood beside them, the readers of `plans.json` follow, and the docs of the agent are rewritten.

## Acceptance

1. **The six settings reach their end on three assets.** The three designs, each with and without the diagnostics server, play the rig on the rows of AssetOpsBench's servers and on its own action server, FaultArena, and a planted dataset, with a stand-in for the model. An episode reaches its end when it ends by the write that ends it, its row of `scores.csv` holds a verdict and no error, the run folder passes the checks of a run folder, and with the server at least one call of every active module's advice kind is in the trajectory.
2. **A made-up second dataset with no line of code changed.** The test copies a dataset in the shared form under another asset id into a planted framework folder and asserts, in this order: the difference of the tree holds new files under that folder and no changed file; the dataset registers and its declaration builds; the six settings reach their end on it; planting it moves no digest and no identity of another asset.
3. **No dataset name in shared code.** One test scans string constants and identifiers of every Python file outside tests for the asset id of every registered asset and the folder name of every framework that declares datasets, with an allow-list file in which every entry has a file, a name, a count and a reason, so that the exception cannot grow silently. The agent's package, `agents/utils` and `utils/public_rules` hold no entry.
4. **Every test for every registered asset.** The tests of the agent that are narrowed to the assets whose episodes the engine plays hold for every registered asset again, each in the form that fits a kind without that notion.
5. **The blind port.** After the last change, a coding agent that has seen none of the agent's code writes one further made-up dataset of scripted cases from [adding-a-dataset.md](adding-a-dataset.md) alone, and it plays the six settings with no line of code changed. Its failures are faults of the declaration or of that page, and both are mended.
6. **No text raises**, as the check section states it.
