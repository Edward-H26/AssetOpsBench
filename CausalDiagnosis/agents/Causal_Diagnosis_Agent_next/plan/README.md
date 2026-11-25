# Declared plan reader and check

The package reads a model reply through `parse_usable_plan` or
`parse_differential`. Expected input errors produce named refusals. Unexpected
exceptions produce a refusal with its exception class in `Refusal.fault`.
Neither boundary dispatches a tool.

`plan_schema_of` derives reply shapes from the joined table, expectation forms
and options. It reads no episode facts. The schema generator tests visit each
form, optional property and declared bound. Malformed input generators visit
positions alone and in pairs. Their output prints the generated reply counts.
The controls weaken a type reader, lower a schema bound and plant a keyword.

`inspect_plan` walks copies of the kind ledger and returns prepared steps,
standing clauses, fill notes and reasons. Step reasons stop at the first
unusable step; whole-plan reasons leave no executable prefix. The frozen
`step_problems` and `check_plan` functions expose these reasons in their declared
forms. The live board, ledger, facts and model plan stay unchanged.
Collapsed observations still meet an observation prerequisite, including when
the observation came from a knowledge tool rather than the observe role.
Knowledge identifiers alone, without observation rows, do not meet it.

## Work sequence

1. Preserve prior history and start from the merged interface freeze. Keep
   dataset adapters and agent foundations together on the active working branch.
2. Port the typed readers, parser, reply schemas and copied-ledger check.
3. Exercise defaults, candidate ranks, evidence fills, standing clauses and
   step-versus-plan refusals against the frozen declarations.
4. Port answer normalization and working memory, then run both acceptance suites.
5. Maintain local contract tests for the controller, prompt, modules and kind
   interfaces as their owners integrate them. Use the QA and scripted ports as
   generality controls. Keep follow-up implementation within tasks 4 and 5;
   `CausalDiagnosis/docs/plan-memory-handoff.md` lists the ownership boundaries
   and the remaining integration dependencies.

## Integration still required

The advice package's frozen `expectation_forms` entry point delegates to
`plan.expectations.expectation_forms`. Module forms join the answer forms and
event forms without reading facts. Advice-enabled execution remains unsupported.

The prompt must print the active clauses in `messages.RULE_SENTENCES` and explain
typed-reader refusals. The read-only tests in `tests/test_prompt_contract.py`
check every public rule sentence against the actual planning request, before
and after facts, with a removed-sentence control. Detailed reader explanations
remain part of the prompt owner's integration. The prompt package is outside
this package's ownership.

The controller must apply the notes returned by `inspect_plan`, use only its
executable prefix, and recheck standing and step legality before a live call.
It supplies controller fills and judges standing expectations against retained
observations or the responsible module. It records a parser fault in the plan
record and never dispatches a refusal. The controller now supports standing
expectations, evidence and fixed fills, closing summaries and backend-driven
context compaction. A cut prefix never resumes its unusable suffix after earlier
answers arrive. Registered QA and scripted episodes use the shared driver of
`CausalDiagnosis/docs/declared-episodes.md`. Bindings, warrant fills, advice modules
and complete recorded-kind episodes remain integration work.

Kind implementations must supply the public `Ledger` and `Walk` interfaces.
The doubles in `tests/fakes.py` verify the check without replacing a real kind.
Mapped tests that depend on those kinds or the controller remain integration
work. The native controller tests exercise no-call standing expectations and a
facts-dependent cut whose suffix must not resume after the facts arrive.

## Checks and identity

Run from the repository root in its slim environment:

```bash
PATH=$PWD/.venv/bin:$PATH python -m pytest CausalDiagnosis/agents/Causal_Diagnosis_Agent_next/plan -q
```

Plan-package changes move the identity of new-agent runs. They do not change
label tables or fitted tables. Shared answer-reader changes belong to the
working-memory handoff and move every run identity. Stored results are not
rewritten or presented as fresh runs.
