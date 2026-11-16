"""A walkthrough of one episode for a reader who was not there. It shows how the episode started and what the agent saw
at step one, every tool call with the clock, the budget and the model calls so far, what the environment did out of the
agent's sight, the verdict, and one paragraph in plain words (the walkthrough line of CausalDiagnosis/docs/README.md).

    PATH=$PWD/.venv/bin:$PATH python -m CausalDiagnosis.episode_viewer.walkthrough <episode folder> [--out file.md] [--note text]

Every value is read from the files of the episode folder and of its run folder, and nothing is recomputed from the rig
data. Parts 1, 2 and 5 read the public files only (episode.json, released.json, question.txt, trajectory.json,
llm_calls.json, plans.json, plan.txt, and the run settings of run.json and scores.csv), while parts 3 and 4 read hidden/.
A hidden value therefore never appears before part 3, which tests/test_walkthrough.py checks by rendering an episode
without its hidden folder.

The model calls are placed between the tool calls with the structure of the runner that made them. The Plan-Execute
runner plans, fills the arguments of each tool step with one call and summarizes (src/agent/plan_execute/executor.py).
The CausalDiagnosis controller plans before the first call of every plan, the reads of the opening included, orients
once the profile is read and summarizes before finish, and its prompts carry their own headings (CausalDiagnosis/agents/Causal_Diagnosis_Agent/prompts.py). The controller's
plan steps are matched to its tool calls in the order its runner makes them (CausalDiagnosis/agents/Causal_Diagnosis_Agent/agent_options.py, CausalDiagnosis/agents/Causal_Diagnosis_Agent/controller.py and CausalDiagnosis/agents/Causal_Diagnosis_Agent/episode_result.py), and a walk
that cannot account for every call leaves the plan column out and says so.

On the rows that run AssetOpsBench's iot, fmsr and wo servers and the task server, the walk follows the tool
that each action kind binds there (CausalDiagnosis/agents/Causal_Diagnosis_Agent/tool_surfaces.py), and every answer is read as AssetOpsBench's text followed,
from the priced row on, by the receipt line of the metering layer (CausalDiagnosis/frameworks/AssetOpsBench/tool_servers/metering_layer.py).
"""

from . import formatting
from . import controller_walk
from . import episode_files
from . import call_placement
from . import tool_answers
from . import part_plain_words
from . import part_start
from . import part_verdict
from . import controller_process
from . import part_steps
from . import part_hidden
from . import page

from .formatting import PARTS, effective_count, joined, long_levels, num, plural
from .episode_files import ACTION_SERVERS, Episode, RIG_TOOLS, SERVER_NAMES, aob_envelope
from .call_placement import plan_column
from .tool_answers import describe_answer, diagnostics_answer, notifications_text
from .part_start import gateway_folder
from .part_verdict import check_rows
from .controller_process import branch_text
from .part_hidden import combination_after, hidden_row, hidden_rows, start_row
from .page import FOR_PEOPLE, files_read, main, render
