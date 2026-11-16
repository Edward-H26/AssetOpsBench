"""The belief of the diagnostics server over the hidden state of the asset, filtered over time (version 7,
CausalDiagnosis/docs/causal-diagnosis-mcp.md, section The filter).

The hypothesis space is every combination of the profile. The filter tracks augmented states x = (h, e, m): h is the
combination now, e the point of the replay at which the recorded block of h was entered, counted in recorded cycles observed
before it, which fixes the readings that share the latent block offset of the joint_block likelihood, and m the rule memory,
which holds the draw of the scheduled degradation rule (each scheduled component, its first drop cycle and its level at the
start) and, for a component under an emergency hold, the level it returns to and the cycle of expiry. The prior is uniform
over the combinations, times the likelihood of the released fix history under the history rule, times the draw of the
schedule rule. The readings of negative cycles, recorded before the episode began, update the prior before any cycle runs.
Every cycle up to the current one then replays in the order the rig served it: the closures
stamped with the cycle whose repair had a downtime, which closed at the end of the cycle before, then what the rig served
while the clock stood at the cycle, the readings of a read of zero duration and, on a row without a clock or at a finish,
the closures settled there, in the order they were released, and for a cycle the clock ran the predict step of its other
kernels in the order of RigClock.advance_cycle, the expiry of the emergency holds, the scheduled drops or the drops of the
stochastic hazard and the closures of repairs without a downtime in priority order, followed by
the readings recorded during the cycle. Every reading updates the weights by the joint_block likelihood of the episode's
fits, which score each hypothesis without the block of the start and without its own, so a reading before
cycle 0 and a live one share the block offset of the block they were both recorded in, and the offset of a hypothesis has
the block covariance of that hypothesis's fit. A kernel that changes h restarts the block offset at the model mean, so a
combination entered twice starts a fresh offset.
States with equal (h, e, m) carry equal Kalman states and are merged. The change summary's alarms and the failure events are
not read, so the belief is a posterior on a subset of the released evidence. Every quantity comes from the released evidence
and the rule parameters of the asset view, never from a hidden event, a draw, a seed or the fault rule of the starts.
"""

from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.served_recordings import shows

from . import released_episode
from . import rule_cards
from . import augmented_states
from . import rule_card_kernels
from . import belief_filter

from .released_episode import Episode, Reading, merge_observations, parse_released
from .rule_cards import Rules
from .augmented_states import States
from .rule_card_kernels import close_kernel, expiry_kernel, hazard_kernel, schedule_kernel
from .belief_filter import Filter, log_density
