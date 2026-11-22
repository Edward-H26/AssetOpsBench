"""Run an agent or a reference policy on hydraulic rig episodes through the two MCP servers and verify every episode from
its log (CausalDiagnosis/docs/experiments.md).

usage, from the fork root with the fork's virtual environment first on PATH (the runner starts every tool server as python):
  python -m CausalDiagnosis.experiments.episode_runners.run_episodes [--config static|timed|anchored|<json>] [--agent AGENT] [--seeds N] [--first-seed S]
      [--model NAME] [--base-url URL] [--clock-mode declared|generation_time|instant|generated_tokens] [--guidance 1|2|3] [--no-diagnostics]
      [--servers rig|aob] [--store memory|couchdb]
      [--prices default|full_tier_high|sme|<json>] [--verifier-scope writes|all] [--judge none|local] [--judge-model NAME]
      [--judge-base-url URL] [--judge-protocol single|paper] [--ranking-model joint_block|joint|discriminant_urn]
      [--abstention-rule decision|fixed_cut] [--abstention-cut P] [--no-grade] [--out CausalDiagnosis/results/runs]
      [--causal-no-branches] [--causal-single-hypothesis] [--causal-plan-length N] [--causal-no-expectations]
      [--causal-expectation-tolerance T] [--causal-e10-reads] [--causal-no-backtrack] [--causal-sequential] [--causal-no-structured-output]
      [--training-variant as_released|gepa|gepa_score_only|ft|ft_gepa|ft_unfiltered] [--training-fold H] [--starts training --training-sample K]
      [--start-state LEVELS@OFFSET] [--scenario-ids ID/ID]

Agents. agent_as_tool runs the paper-era Agent-As-Tool of AssetOpsBench (CausalDiagnosis/frameworks/AssetOpsBench/agents/Agent_As_Tool_Paper): IBM's ReAct
supervisor on a vendored ReActXen with the paper's prompt, 28 examples, 10 steps and one reflection round, one ReAct
sub-agent per server of the row, raw text completions on the endpoint of --base-url with the paper's decoding, and its stop
when the environment ends the episode. It plays AssetOpsBench's servers only, and records the supervisor's ReAct counter
in react_steps beside the attempted tool calls. plan_execute runs the paper-era Plan-Execute of AssetOpsBench
(CausalDiagnosis/frameworks/AssetOpsBench/agents/Plan_Execute_Paper): AgentHive's planner with its reviewer, the plan's steps on ReActXen agents that are the
Agent-As-Tool port's sub-agents with their examples, and a summarization step, on the same completion route and over
AssetOpsBench's servers only. scripted runs AssetOpsBench's released Plan-Execute runner unchanged with a fixed plan that
calls every tool of both servers, a wiring check without a model. causal runs the CausalDiagnosis agent of
CausalDiagnosis/agents/Causal_Diagnosis_Agent, which reaches the model through AssetOpsBench's own OpenAI compatible backend at the endpoint
--base-url names (TokenRouter by default) with the key in API_KEY, the only key the drivers read, over the action servers and the diagnostics
server, or with --no-diagnostics over the action servers alone, the switch of the agents table that the released agents share:
the model plans every step, the opening included, in plans of several steps with an expectation checked after every
step, the server's ranking and advice inform the plans without taking a step, a new plan follows a surprise, and every closed
work order is verified on a reading a plan named, by verify_outcome with the server and on the normal ranges without it. The
--causal switches are its ablations, those that change the server's advice refused with --no-diagnostics, and the agent label
carries one suffix per switch (causal_nobranch, causal_single, causal_len1, causal_noexp, causal_e10, causal_ec2,
causal_noback, causal_seq).
oracle, do_nothing, greedy and declare are the reference policies of the environment, each action sent as a tool call through
the action server while a shadow environment in this process gives the oracle its hidden state; the served log is
compared with the shadow's at the end. The oracle against do_nothing is the ordering check of OpenEnv RFC 008. static
answers the question without the gym, in one model call over the question, the features of the recorded cycle the gym
would have served first (which the gym charges for) and the failure modes; its answer is judged by episode_judge.py against the
oracle's reference answer and parsed for the components it calls faulty and repaired (ablation A3).

Servers. Without --servers a model agent on a row of the environment table or on an arm of its rows plays aob, since every
row runs on AssetOpsBench's servers, and every other run rig (default_servers). --servers rig registers
the rig's action server. --servers aob registers AssetOpsBench's own iot, fmsr and wo
servers behind the metering layer of CausalDiagnosis/frameworks/AssetOpsBench/tool_servers, the AssetOpsBench rows of the environment table,
over the store --store names: the memory store on the laptop, or the CouchDB of a cluster job. The
driver creates the episode's databases before the first call and, after the verifier read the episode, exports the
documents to aob_documents.json in the episode folder and drops the databases; run.json records the store under
backend. The reference policies play through the action server only.

The diagnostics server. Its launcher names the combination the episode started in, whose block every fit of the server
leaves out and which the server uses for nothing else, and no tool result, released file or log the agent
reads names it.

Every episode folder holds what the agent could see (the state file, the released evidence, the predictions, the
question, the trajectory, the model calls, the audit report) and under hidden/ what no agent facing tool reads (the
oracle, the verdict, the scores and the full log). The run folder gets run.json, scores.csv and summary.md, which counts
episodes and files and states no score, and ledger.json, which run_ledger/ writes before the first episode and closes at
the end. EpisodeRunner is the machinery behind main(); gepa_prompt_optimization/ drives it too.

Training in the environment, the training table (training_variants/). Every run records its --training-variant in run.json, as_released
unless the option names another. gepa, gepa_score_only and ft_gepa append the final block of an online loop of GEPA to the task
message of the
agent, the briefing, an empty line and the block, the same position for the three agents, while question.txt and the judge keep the
briefing alone. ft, ft_gepa and ft_unfiltered send the model as <model>:<adapter>, the LoRA adapter that the serving job loads beside
the model, and run.json keeps the base model with the adapter apart. The adapter and the block are read under the training folder of
HYD_TRAINING and checked against their records before the first episode. Every trained variant has one model per fold of the
asset's registered split: --training-fold names the fold the model holds out, one of the folds of
configs/training/<asset_id>/folds.json, and the run plays the test episodes of that fold alone (start_states.fold_replicate_seeds),
while the agent without training plays all of them in one line. --starts training plays the training start states of the model
of --training-fold under sample --training-sample, one seed each (start_states.training_seeds), at the sampling temperature of
the asset's rft.json in place of the temperature the agent asks for, in the mode training, whose runs are training data and enter
no result table. --starts development plays the development starts, the recorded fault combinations outside the test set
(development_split.development_starts), under the --seeds smallest seeds of each, or --start-cap of them, on the preset whose fault rule and allowed starts are replaced
by those of the development starts and whose pinned start is dropped, so no test combination is drawn. --training-split development plays the training path of a
trial before the runs of record on the same widened preset, with the folds of the development split (folds_development.json beside
folds.json), so its sampling, its trained variants and the files they read lie on development combinations alone. --start-state
plays every seed of --starts seeds at the one start state it names, as a preset whose field start names it plays, and
--scenario-ids plays the scenarios of the anchored preset whose ids it names alone; no cell of a table reads a run of either.
Every agent plays every start that its preset allows, the test set among them, in any run, and every run that the start check
does not register carries the stamp development in run.json. The commit a run records is its record.
"""

from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import FMSR_SERVER, IOT_SERVER, TASK_SERVER, WO_SERVER  # noqa: E402

from . import agent_kinds
from . import model_backends
from . import reference_policy_episodes
from . import episode_scoring
from . import static_agent
from . import run_record
from . import play_one_episode
from . import harness_rule
from . import agent_runs
from . import planned_episodes
from . import start_check
from . import runner_setup
from . import command_line

from .agent_kinds import (
    AGENTS, AGENT_ENTRY_MODULES, DEFAULT_MODEL, MODEL_AGENTS, PAPER_ERA_AGENTS, POLICY_AGENTS, ROUTER_BASE_URL,
    SCRIPTED_AGENTS, agent_label_of)
from .model_backends import CONTEXT_READERS, RecordingBackend, full_group_of, router_backend
from .reference_policy_episodes import HydraulicRigEnv, RIG_SERVER, greedy_reference, open_sessions, oracle_reference
from .episode_scoring import (
    STATE_FILE, episode_reward, grade_of_start, run_judge, tool_counts, verdict_row, verify_episode)
from .static_agent import first_read, full_sensor_group, static_context, static_prompt
from .run_record import (
    BRIEFING_VERSION, RESUME_NONE, ROOT, agent_settings, clock_totals, default_out, ledger, loaded_files,
    loaded_outside_lists, loaded_paths, lora_adapters_of, run_folder, run_label, run_meta, serving_started,
    start_states, write_run)
from .play_one_episode import (
    CUT_FACTS, DEFAULT_RANKING_MODEL, DRIVER_STATE_FIELDS, EPISODE_WALL_LIMIT_S, EpisodeRunner, WallLimit,
    check_servers, get_profile, open_tool_calls, update_state, with_task_block, write_server_baseline)
from .harness_rule import (
    ATTEMPT_FOLDER, HARNESS_FILE, SERVER_CHECK_FOLDER, WALL_LIMIT, cut_replies, diagnosis_refusals, failed_calls,
    finish_stamp, run_limited)
from .agent_runs import CAUSAL_SERVER, causal_surface, diagnostics_calls, record_causal, run_causal
from .planned_episodes import (
    capped_starts, chosen_seeds, development_config, episode_plan, played_config, replicate_of, start_state_of, training_checks,
    training_sample_of)
from .start_check import (
    FILLER_REASON, RUNS_IN_FLIGHT, SEED_STARTS, START_SETS, registration, runs_in_flight, stamped_inputs, table_presets,
    tree_reasons)
from .runner_setup import (
    ContextNotCounted, build_runner, causal_options, default_servers, diagnostics_models, is_marker,
    registered_servers, require_context_count, store_settings)
from .command_line import (
    FILLER_JUDGE, FILLER_OUT, FILLER_STARTS, RESUME_FILLER, build_parser, configure, hold_tables, load_config,
    load_world, main)
