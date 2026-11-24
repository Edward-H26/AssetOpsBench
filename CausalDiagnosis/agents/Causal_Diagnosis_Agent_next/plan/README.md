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

## Work sequence

1. Preserve the prior implementation on its existing branch and start from the
   merged interface freeze.
2. Port the typed readers, parser, reply schemas and copied-ledger check.
3. Exercise defaults, candidate ranks, evidence fills, standing clauses and
   step-versus-plan refusals against the frozen declarations.
4. Port answer normalization and working memory, then run both acceptance suites.
5. Integrate the prompt and controller when their implementations are available,
   re-express the remaining mapped tests and run the episode gates.

## Integration still required

The advice package's frozen `expectation_forms` entry point must delegate to
`plan.expectations.expectation_forms`, or provide equivalent callable readers
with a `schema` attribute. Module forms join the answer forms and event forms
without reading facts.

The prompt must print the active clauses in `messages.RULE_SENTENCES` and explain
typed-reader refusals. A cross-package test must compare every emitted reason
with its prompt sentence. The prompt package is outside this package's ownership.

The controller must apply the notes returned by `inspect_plan`, use only its
executable prefix, and recheck standing and step legality before a live call.
It supplies controller fills and judges standing expectations against retained
observations or the responsible module. It records a parser fault in the plan
record and never dispatches a refusal. The controller and driver remain separate
integration dependencies; their frozen stubs cannot run an episode.

Kind implementations must supply the public `Ledger` and `Walk` interfaces.
The doubles in `tests/fakes.py` verify the check without replacing a real kind.
Mapped tests that depend on those kinds or the controller remain integration
work, including no-call standing expectations and live changes before dispatch.

## Checks and identity

Run from the repository root in its slim environment:

```bash
PATH=$PWD/.venv/bin:$PATH python -m pytest CausalDiagnosis/agents/Causal_Diagnosis_Agent_next/plan -q
```

Plan-package changes move the identity of new-agent runs. They do not change
label tables or fitted tables. Shared answer-reader changes belong to the
working-memory handoff and move every run identity. Stored results are not
rewritten or presented as fresh runs.
