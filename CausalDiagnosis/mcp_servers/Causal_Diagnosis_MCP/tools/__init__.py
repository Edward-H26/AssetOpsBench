"""The four tools of the diagnostics server as plain methods over the released evidence of one episode and the fits of
that episode (version 7, CausalDiagnosis/docs/causal-diagnosis-mcp.md), testable without the mcp package.

No belief is stored between calls: every call replays the released evidence from cycle 0 through the filter, so that a
server process started per tool call, as the Plan-Execute runner does, answers exactly as a long lived one would. The
asset enters through the view that asset_model.profile_view builds from the released rules. The only files the tools
write are predictions.json beside the released evidence, where estimate_intervention_effect and the troubleshooting plans
of rank_root_causes store their predictions and verify_outcome reads them back, and the empty lock file beside it, which
every store holds while it adds a record. The server that serves the tools appends
every answer to the transcript of the episode (servers/transcript.py). Every threshold in a result traces to a
price, a duration, a downtime rate or the declared error level, and the assumptions of the result say how. The wall time of
every ranking goes to the server's log on stderr, where the job's log keeps it. Where the clock of an episode counts the
seconds between two calls of the action servers, the seconds of a call to this server lie between two such calls and are
counted with them.
"""

from . import tool_texts
from . import episode_reading
from . import fitted_models
from . import rank_root_causes
from . import intervention_effect
from . import causal_tools

from .tool_texts import A_BLOCK, A_COMPACT, A_E1, A_RULE
from .fitted_models import MODELS_PATH, Models, fit_episode, load_models
from .rank_root_causes import COMPACT_LEFT_OUT
from .causal_tools import (
    CausalTools, DEFAULT_ABSTENTION_CUT, DEFAULT_ALPHA, DEFAULT_TOP_K, IG_EPSILON, branch_previews, channel_field,
    decide, left_in_episode, profile_view, read_batch, repaired_components, verifying_read_price,
    write_json_atomically)
