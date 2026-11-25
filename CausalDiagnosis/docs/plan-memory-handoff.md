# Plan and working-memory handoff

## Scope

The work of tasks 4 and 5 follows `general-agent-tasks.md` and the frozen
interfaces. Its implementation belongs to the next agent's `plan/` and
`working_memory/` packages, `agents/utils/answer_forms.py`,
`agents/utils/tool_answers.py` and the shared receipt-reader import in the
metering layer. Existing-agent behavior and `agents/utils/declaration.py`
remain compatibility boundaries.

The prompt and ranking module, environment layers, cluster serving and jobs,
viewer, training configuration and training dependencies have separate owners.
Follow-up work here reads their interfaces and adds local contract tests; it
does not replace their implementation. A missing capability in one of those
packages is an integration dependency, rather than a reason to fill its stub
from this work. Earlier adapter and controller commits remain available as
integration scaffolding, not as replacements for those packages.

## Checks to maintain

| Requirement | Implementation | Evidence |
| --- | --- | --- |
| Typed replies, named refusals and valid prefixes | `plan/readers.py`, `plan/reply_parser.py` | `plan/tests/test_declared_reading.py` |
| Schema generated from the table, without facts | `plan/reply_schema.py` | `plan/tests/test_reply_schema.py` |
| Copied walk, standing clauses, defaults and scoped limits | `plan/plan_check.py`, `plan/rules.py`, `plan/preparation.py` | `plan/tests/test_check.py` |
| Public check sentences reach the prompt before and after facts | Rule constants consumed by the prompt assembler | `plan/tests/test_prompt_contract.py`, with a removed-sentence control |
| Public answer forms and retained call identity | `agents/utils/answer_forms.py`, `working_memory/board.py` | `working_memory/tests/test_memory.py` |
| Malformed observation metadata cannot enter memory | `agents/utils/answer_forms.py`, `working_memory/board.py` | Generated row mutations and normalized-answer controls in the memory tests |
| Escaped Unicode cannot break memory or record encoding | `agents/utils/tool_answers.py` | Nested result, status, notification, observation and receipt controls, with a complete-pair control |
| Observation prerequisites survive compaction and reject unrelated references | `plan/plan_check.py` | Collapsed knowledge-read and empty-observation controls in the check tests |
| Compaction, recall, tied notes and chronology | `working_memory/compaction.py`, `working_memory/board.py` | Memory tests with retirement, citation and no-progress controls |
| One receipt reader and compatible legacy readers | `agents/utils/tool_answers.py` | Memory tests and the existing metering tests |

Run the focused contracts from the prepared environment:

```bash
python -m pytest CausalDiagnosis/agents/Causal_Diagnosis_Agent_next/plan \
  CausalDiagnosis/agents/Causal_Diagnosis_Agent_next/working_memory \
  CausalDiagnosis/agents/Causal_Diagnosis_Agent_next/tests/test_frozen_interfaces.py -q
```

The task 5 compatibility check remains:

```bash
python -m pytest CausalDiagnosis/agents \
  CausalDiagnosis/frameworks/AssetOpsBench/tool_servers -q
```

The second command includes existing-agent and metering behavior. The test
fixtures keep generated tables in temporary copies; a focused pass alone is
not a substitute for this compatibility check.

Check the default generated model table before an isolated server launch too.
A fixture's newly fitted copy does not make an older default model compatible
with the server's reader. Test snapshots must provide matching default tables
for their child processes. Refresh formal experiment tables through the cluster
preflight workflow rather than replacing them with a test artifact.

## Dataset generality and integration dependencies

The registered PdM descriptive QA and scripted-case ports remain interface
fixtures for this work. Keep their input hashes, grouped partitions, private
references and unsupported-modality accounting intact. See
`declared-episodes.md` for the separate entry and its matching policy.
Use synthetic rows for malformed-input and compaction controls. Do not choose
reader rules, prompts or policy from held-out answers.

The prototype's state-condition interpreter lives in
`CausalDiagnosis/utils/public_rules/scripted_conditions.py`. Both its public
walk and private replay import that one interpreter. Public rules import only
standard-library or sibling rule modules, never the private environment.
Moving the prototype interpreter out of the environment changes the conservative
label digest; regenerate formal label tables through the preflight workflow.

The prompt owner consumes the named reasons and board parts. The ranking owner
supplies module forms, candidates, evidence identifiers and public sections.
Contract tests can use stand-ins for these interfaces without implementing the
modules. Complete recorded-kind, binding and advice episodes depend on their
respective implementations and the shared driver. Serving a real model and
running the cluster review are separate integration steps. Model quality,
official matcher equivalence and the final replacement gate are not established
by the reader or memory tests.
