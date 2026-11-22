"""The situation board of the CausalDiagnosis agent (CausalDiagnosis/docs/experiments.md, section The driver, agent causal).

The model never reads the transcript. It reads this board, rendered as compact tables: the evidence with the identifiers
the verifier accepts, the ranking with its decision fields and troubleshooting plans, the timeline of alarms and closed
work orders, the open orders with their prediction identifiers, the commitment made, the budget and the cycles left, the
plan in progress and the surprises so far, and since version 7 the fix history, the expected elimination of every
sensor group with the read batch, and the repair plans a verification refuted. Every field comes from a tool answer the
agent paid for or could read for free, so the board holds nothing hidden. The ablation switches of the runner decide what
the board shows of the troubleshooting plans: all of them, the first one, or none.

The server only advises: the read batch, the reads to take while the ranking says read first,
the repair plan to act on and, after a refuted plan, the other plan are advice, and the controller reads, diagnoses and
orders only what a plan names. Before the profile is read the board holds nothing about the asset, since the model plans the
opening too, and the question stands above the board in every prompt (prompts.py). In both modes the board shows the failure
modes, the channels outside their normal range in the latest cycle of every sensor group read, and the candidates the model
listed in its latest plan; with the diagnostics server the ranking and its advice follow them, and without it a repair is
verified on the normal ranges (band_verdict).

Idle compaction: after every executed step, while no model call is pending, the controller
calls compact(fits). With the server, evidence rows that a ranking has already absorbed collapse into one summary row per sensor
group and cycle range, which keeps the identifiers outside their normal range, and every refuted plan keeps one line. Only while
the next planning request passes the compaction trigger, 90 percent of the served context (fits), does the latest observation
collapse too, and then one summary turn replaces the notes, the notifications, the surprises but the latest and the timeline
older than the latest ranking, except the entries that hold an identifier the summary names.
There is no budget and no keep count of the agent's own, and the absorbed identifiers stay citable.
"""

from . import asset_rules
from . import text_formatting
from . import troubleshooting_plan_fields
from . import board
from . import read_walk

from .asset_rules import (
    band_verdict, channels_outside, durations_of, order_prices, profile_level, read_prices, visit_closes)
from .text_formatting import combination_text, confirmation_text, fmt, n_cycles, scored_combination_text, table
from .troubleshooting_plan_fields import branch_prediction_id, repair_steps
from .board import Board, THINKING_CLOCKS
from .read_walk import ReadWalk
