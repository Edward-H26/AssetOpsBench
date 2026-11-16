"""The plan of the CausalDiagnosis agent: its schema, the parser of the model's reply, the check of a plan against the
environment's rules and the rule's own plan (CausalDiagnosis/docs/experiments.md, section The driver, agent causal).

A plan is one JSON object: the branch followed, the mode, one line of why and at most plan_length steps when a count is set, each with a kind,
its arguments, the expectation the controller checks after the step and the evidence identifiers behind it. Without the
diagnostics server it also lists the model's own candidates, the combinations it ranks, since no server ranks them. The
parser takes the last complete JSON object of the reply, since a model that reasons before it answers restates the plan
last (preet2026simplicity). The model plans every step, the free reads of the opening included,
and the server only advises it: the check refuses, before any tool is called, what the plan schema or the environment
rule out, what the budget, the horizon or the step cap cannot pay, a declaration with no reading to cite and
a finish before the read that verifies a repair, and names every reason so the model can repair the plan in the next
turn. The rest is advice on the board (advice below): the read to take while the ranking says read first,
the repair plan to act on, the rank the server advises,
and after a refuted plan the cheapest other plan. The rule's plan, the greedy controller of E10 written as a
plan, is an internal test tool that follows that advice without a model.
"""

from . import plan_schema
from . import reply_parser
from . import repair_budget
from . import read_advice
from . import rule_controller
from . import advice_lines
from . import plan_check

from .plan_schema import ESTIMATE, OPENING_KINDS, Plan, PlanError, Step
from .reply_parser import parse_candidates, parse_differential, parse_plan, parse_step
from .repair_budget import (
    _fits_beside_reserve, act_options, order_reserve, order_reserve_calls, read_affordable, repair_options,
    reserve_completion)
from .read_advice import ARMS, batch_reads, exploration, ranking_detail
from .rule_controller import WHY_SCHEDULE, rule_evidence, rule_plan
from .advice_lines import advice
from .plan_check import check_plan
