# Public working memory and answer reading

The board retains public facts, knowledge, observations, exact call arguments,
commitments, refuted answers, notifications and tied notes. Text and JSON views
share the ordered `_parts` iterator, with the status last. Refused calls do not
become retained answers. Repeated calls include refused attempts; different
arguments form different groups. Repeated plans and clock ticks do not reset
the consecutive-turn stall counter.

`tool_answers.read` normalizes envelopes, plain text, refusal marks and final-line
receipts. `json_object` and `split_receipt` preserve their existing behavior.
The metering layer imports the shared receipt reader, so existing controller
and framework imports resolve to that same function. Unexpected reader errors
carry their class under `guard`; expected malformed input returns a named
refusal without a guard.

The three answer forms read objects, numeric records with public ranges, and
text. The board attaches the episode's observation number and declared words
to text rows. A form error becomes a tool-named note and is not retained as a
successful answer.

Compaction retains original observation rows and their identifiers. Recall is
scoped to one following board. Summary-cited identifiers pin their entries;
retiring notes shifts the turn boundaries used to order summary entries.
Compaction changes no paid-unit state or live kind ledger.

## Integration still required

The controller supplies the kind ledger, marks planning turns and absorbed
observations, assigns model candidates and the differential, and retires notes
when their tied state passes. Facts, candidate lists, differentials and successful
advice answers retire their corresponding ties on the board. Other state changes
are reported through `retire_notes(passed)`.

The controller supplies request-fit callbacks and the earlier-turn summarizer
to `compact`, and drops recalled rows when its actual context cannot hold them.
The reused native controller transfers an unexpected answer-reader `guard` to
the call record and adds the called tool's name to the refusal. Descriptive QA
development episodes exercise this controller over isolated MCP sessions.
Complete mapped episodes still need the remaining kind, binding, advice and
controller integrations. These development controls do not establish a formally
registered dataset port or a served-model evaluation.

## Checks and controls

```bash
PATH=$PWD/.venv/bin:$PATH python -m pytest CausalDiagnosis/agents CausalDiagnosis/frameworks/AssetOpsBench/tool_servers -q
```

Controls use different call arguments, new observations on every planning turn,
unrelated note ties, refused opening calls, removal of summary entries from a
copy, and an injected unexpected receipt-reader exception. The receipt identity
test fails if the metering layer keeps a separate implementation. The existing
agent's receipt and episode tests remain in place.

These shared-reader changes move every run identity. Label tables and fitted
tables stay unchanged; no stored table or recorded run is regenerated here.
