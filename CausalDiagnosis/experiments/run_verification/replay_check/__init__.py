"""The replay check of a finished run and its certificate (CausalDiagnosis/docs/experiments.md).

A replay plays every episode folder of a run again in a scratch folder through the servers of the tree that runs it, with no model
and no agent code. It starts the episode through the start function of the driver under the wall clock and the monotonic clock
readings that the transcript of the episode recorded at its start, then feeds every line of the transcript in order, a listing as
a tools/list request and a call as a tools/call request with the recorded arguments, through the request handlers that the MCP
package calls for a client, one server object per recorded server process. A call that changes the episode runs under the wall
clock reading of its line, and a call of a server that only reads the released evidence runs against the version of the evidence
that its line names by hash. The servers of this tree write their own transcript, which must equal the recorded one line by line
in every field but the process id, both read in the order of service, with the entry and the exit of each call beside its line,
since a line that another server process wrote while a call was served stands between them in the record and before or after them
in the replay. Once the verifier of this tree has scored the scratch folder, the files the episode leaves must equal the recorded
ones as parsed JSON, the state file without its data path, the byte offset of the transcript and the processes it names by their
ids, with the counted call in service named by the place of its entry in that order, and the score without the judge, and the
score cells must equal those of scores.csv. A run replays when every episode does, the tool listings of its run.json equal those
of the servers of this tree, the data it recorded equals the data here, and the command its ledger recorded composes in this tree
the configuration its run.json records. A copy that rescore_under_verifier_scope.py scored again holds the command of its run's ledger and the
state file of every episode as the run played them, and the replay composes that command and scores every episode under the scope
that the copy's record of the rescoring names. A replay counts only on the store kind the run played on and on the platform it
recorded, and elsewhere it is advisory. So a replay shows that the environment, the servers and the verifier of this tree answer
the recorded calls as they were answered, while the file lists of the run decide the agent side.

certify writes the certificate of a run that replays and counts into its run folder, with an event of its ledger, and of a run
whose only differences are generated tables, the tables that the tree computes from the recordings and its own code, each of which
the certificate names. The commit of a run is its record: the certificate is written by a clone of the run's own commit, binds
that commit and the platform of the replay, and certified() reads a run as replayed only when both equal the commit and the
platform that the run's run.json recorded. A fault found later is fixed in a new commit, whose rows are played again.

usage:
  python -m CausalDiagnosis.experiments.run_verification.replay_check certify RUN_FOLDER
"""

from . import run_record_checks
from . import episode_replay
from . import run_replay
from . import certificate
from . import command_line

from .run_record_checks import RECORDED_DATA, asset_of, changed_files, config_problem
from .episode_replay import RELEASED_FILE, _feed_lines, feed, feed_no_call, replay_episode, server_object
from .run_replay import JUDGE_CELLS, STATIC_MODE, replay_run
from .certificate import ADVISORY, TOOL, certificate_name, certified, certify
