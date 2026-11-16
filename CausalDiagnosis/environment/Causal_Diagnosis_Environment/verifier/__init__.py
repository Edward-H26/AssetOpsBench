"""The verifier of an episode: the agent's write events against the oracle of the episode, in the order of the ARE
verifier, then the failure class, the scores and the reward (CausalDiagnosis/docs/environment.md).

Inputs are the full event log with its hidden fields (`env.log`), the released evidence of the episode (released.json
format 3, the public view of the same log), the oracle of the episode (`env.oracle()`), the cost of the oracle policy's
episode and, when known, the separability of the start, the independent label of diagnosability_label.py with the server's E18
grade nested, as separability.start_grade gives it. The verifier reads no answer text for a score. The closing
summary enters one place only, the contradiction heuristic of the failure taxonomy, which the judge agreement study covers.

The asset of an episode is the one that the configuration of its reset entry names, and the truth model of that asset
(CausalDiagnosis/datasets/dataset_loader/ground_truth.py) gives every asset fact the verifier reads: the components and their healthy levels, the
components a channel indicates, the identifiers of the failure modes and the words of the contradiction heuristic.

Checks run in the order of ARE: the count gate, the hard checks, the timing check, the causality check, the outcome in the
hidden state after the work orders and the provenance check. Reads are never matched under the default scope, since many reading strategies lead to
the same writes; the scope "all" of ablation A7 adds a read gate to the count gate, a ceiling per sensor group at the sensor
reads of the greedy policy on the same start, since the oracle policy knows the start and reads nothing.

The diagnosis states the condition of each component at the cycle of its write, so it is scored against `truth_at_write`,
the combination after every system event that the log holds before the `submit_diagnosis` entry, which equals the hidden
start h(t0) whenever nothing changed the rig before the write, as in every static episode.

Two outcome flags. `pass` is Equation 2 of the paper (correct diagnosis, recovered, no unnecessary order) and feeds the
metric tables. `success` adds the abstention rule through a tri-state label: a `pass` is a success on
every start, and a declaration is one on a start that the label and the server's E18 grade both mark inseparable when it
cites a reading released before it, so neither a wrong commitment, nor a declaration on a start either
marks separable or on an unlabelled start, nor a declaration without such a reading is.
The reward gives a pass the whole outcome term and a correct declaration ABSTENTION_CREDIT of it, so a repair stays above a
declaration after reading and that above a wrong answer, and it withholds the term from a provenance failure.

A verdict is a plain dict that json.dumps accepts. A log the verifier cannot read (no reset event, no end event, hidden
fields stripped, an oracle that disagrees with the log's start) gives a verdict with `verified` false, and `reward` returns
None for it, since a missing grade is not a wrong answer. An episode is verified once it has ended, by the environment or by
a finish, so the levels the outcome reads are always those of an end state.
"""

from . import episode_log
from . import episode_costs
from . import episode_reward
from . import failure_class
from . import summary_consistency
from . import episode_checks
from . import delays
from . import verdict

from .episode_log import (
    COMMIT_EVENTS, RELEASED_FORMAT, _asset_truth, _combination_after, _reset_config, _time, released_from_env)
from .episode_costs import MISDIAGNOSIS_EVENT, _costs, misdiagnosis_charge, misdiagnosis_charge_of
from .episode_reward import ABSTENTION_CREDIT, COST_CAP, good, outcome_gap, pass_power_k, reward
from .failure_class import FAILURE_CLASSES, REPEATED_REFUSALS, _charged_ineffective
from .summary_consistency import _asset_claims
from .episode_checks import read_counts
from .delays import fault_occurrences, fault_onset, restored_by_agent
from .verdict import verify
