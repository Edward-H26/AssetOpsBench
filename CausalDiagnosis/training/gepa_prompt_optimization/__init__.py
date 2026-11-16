"""The online loop of GEPA over one tunable block of text, the same for the three agents of the training table, AssetOpsBench's
Plan-Execute, the Agent-As-Tool of the AssetOpsBench paper and the CausalDiagnosis agent, each with the
CausalDiagnosis MCP (CausalDiagnosis/docs/experiments.md, section Training in the environment).

The command runs from the fork root with the fork's virtual environment first on PATH, and the rows of the training table pass --config
deterioration with --servers aob and, on the cluster, --store couchdb.
  python -m CausalDiagnosis.training.gepa_prompt_optimization --model NAME --base-url URL --reflection-model NAME --reflection-base-url URL
      --training-fold H [--agent plan_execute|agent_as_tool|causal] [--training-variant as_released|ft] [--card verifier|score_only]
      [--config deterioration] [--servers rig|aob] [--store memory|couchdb] [--block 3] [--window 2] [--children 1] [--max-episodes 240]
      [--first-seed 0] [--master-seed 42] [--seed-pool 8] [--gepa-seed 0] [--facts FILE] [--out CausalDiagnosis/results/runs]
  python -m CausalDiagnosis.training.gepa_prompt_optimization export <run folder> [--training-root DIR]

Candidates. A candidate holds one text, the tunable block, of any length, which the agent receives
appended to the task message of the episode, the briefing, an empty line and the block, the same position for the three agents
(training.with_task_block). The starting candidate is the empty block, so it plays the agent without a block and is the reference.
No candidate changes a prompt of an agent, its system prompt included, or the threshold card of the CausalDiagnosis agent. An
episode of a candidate plays through EpisodeRunner of run_episodes/ with the block in place for the episode. --training-variant
ft plays the agent with the filtered adapter of the loop's fold, the loop whose final block the ft_gepa variant of that fold plays.

The loop. A loop belongs to one fold of the asset's registered split, --training-fold, the fold its model holds out. The
stream serves the training seeds of that model in seed order (start_states.training_stream), the start states of the
combinations it trains on that no replicate of the test set plays, under seeds that are no reserved seed and no label rollout,
the pool that the fine tuning of the fold samples from as well. Per training start state the stream passes over every seed that
a round of that sampling may play (training.sampled_per_state), so no block is scored on an episode that an adapter of the fold
may have trained on, and the loop on the base weights and the loop on the adapter read the same stream. A loop plays training
start states alone: the comparison of its final block with the starting block comes from the evaluation runs of the variants,
which play the test set once. Round r takes the next --block seeds of the stream, and every member of the front and every child
proposed after the round before play each of them, paired by seed. The score of a stream episode is the verifier's reward with
the declaration credit set to zero (STREAM_DECLARATION_CREDIT), the pass alone in the outcome term, so that the loop cannot climb
the stream by declaring, and an episode that the verifier could not score enters no mean and is counted apart, since the reward
is negative for most failures and any fixed stand in would rank such an episode against them. An episode whose hidden log
reaches a single fault of the fold the loop's model holds out is left out: it scores nothing, no reflection call reads it and the
record counts it, since it plays a combination that model is judged on. The mean score of the incumbent fixed after the round
before on the block is the point of the online curve, so every point rests on episodes that no step of the optimizer has seen. A
child joins the front when its mean on the block exceeds its parent's. Over their episodes of the last --window rounds the front
then keeps the candidates that lead some stratum, a stratum of tables_from_runs/ or the whole block, and drops a candidate that
another strictly dominates, as the Pareto selection of GEPA does over its tasks, and the incumbent is the member with the best
mean. The reflection model then proposes --children children, each from a parent drawn from the front with weights equal to
the strata it leads, and a parent whose episodes of the round were all left out proposes none. A child is proposed only when the
next round can pay for it, and the loop stops before a round that the budget --max-episodes or the stream cannot pay. The gepa
package is not used, since its optimize call works on a fixed split of training and validation sets and cannot score a block
before it reflects on it.

Reflection. Per episode of the parent on the block, the reflection model reads the question, the agent's outputs as the run
folder keeps them for every agent (its written plan, plan.txt, or the supervisor's steps of Agent-As-Tool, then the closing answer) and the
feedback. Under --card verifier the feedback is the feedback card of the episode (feedback_card of gepa_prompt_optimization/feedback_card.py, the flags
and the outcome of the verdict, the public trajectory and the messages of the diagnostics server, never a hidden field or a
quantity of the oracle policy), and under --card score_only it is the score alone, the control that tells a gain of the verifier's card from the strength of the reflection model. Both cards show
the score of the stream, the reward with the declaration credit set to zero, the number the selection uses, so the reflection
model never reads a credit for declaring that the loop does not pay. A mutation sends the
meta prompt of GEPA (appendix C of the GEPA paper) with the current block, empty at the start, and one closing paragraph that says
where the block goes and that the instruction comes within ``` blocks. A proposal without a block or equal
to its parent is asked again with its problems, proposal_attempts attempts in all, and a refusal at the last records it without an
episode. A proposal whose every attempt ends on an error of the endpoint is unanswered, a refused text being an answer, and after
unanswered_proposals of them in a row, both registered numbers of the asset, the loop saves the status failed and the command exits
with status 1. The loop saves its whole state after each round with the proposals for the next. With --resume the command takes
up the unfinished loop of the same identity under --out, the settings of the loop, its asset and its code: the episodes of the
round that was cut short move to interrupted/ inside the run folder, the record counts them and their seeds play again, and a
finished loop is never resumed. The reflection model has its own endpoint, the judge model on the judge endpoint of the job,
asked at temperature zero without thinking, so a reply holds the answer alone, with no cap on the length of a reply. An
endpoint that no longer answers its health probe ends the program with ServiceDown, and --resume takes up its last saved state.

Files. The run folder holds run.json with the mode gepa_online, which the agent tables, the ablation tables and the
leaderboard leave out, scores.csv with the GEPA call of every episode, ledger.json, the episode folders call<n>_seed<s>,
gepa_online.json with the rounds of the curve, the candidates and the episodes, the block of every
candidate under candidates/ and every call of the reflection model under reflection/. The export writes the final block of a
finished loop to blocks/<name>.txt of the training folder with its record, gepa_score_only_<agent>_f<fold> for a loop under the
score-only card on the agent without training, gepa_<agent>_f<fold> for a
loop on the agent without training and gepa_ft_<agent>_f<fold> for a loop on the adapter of its fold. The record carries the counts
that the table of what each agent trained on prints, the stream episodes left out among them. The tests play the scripted model
through the servers with a fake reflection model.
"""

from CausalDiagnosis.environment.Causal_Diagnosis_Environment import verifier  # noqa: E402

from . import run_folders
from . import candidate_proposals
from . import stream_scores
from . import online_loop
from . import episode_player
from . import run_loop
from . import loop_settings
from . import block_export

from .run_folders import MODE, RESULT, already_finished, as_stored, folder_to_resume, loop_identity
from .candidate_proposals import (
    BLOCK, CUT_REPLY, MUTABLE, NO_BLOCK, PROMPT_META, RETRY_HEADING, agent_outputs, proposal, seed_texts)
from .stream_scores import improves
from .online_loop import FEEDBACK, FORMAT, LoopFailed, OnlineLoop
from .episode_player import playing
from .run_loop import main, numbers, run_online
from .loop_settings import (
    GEPA_NUMBERS, build_parser, driver_settings, gepa_meta, loop_stream, training_stream, with_registered_numbers)
