"""The online loop of GEPA (gepa_prompt_optimization/). The protocol runs on a fake player whose rewards depend on
the candidate's block, so the order of scoring and reflection, the curve, the admission, the front, the budget, the episodes
left out, the two feedback cards, the refusals, the cap and the leak guard are checked without a model. The stream of a loop passes
over every seed of the sampling of its fold. The loop then plays
the CausalDiagnosis agent with the scripted model through the servers in process, the block reaching the task message and no
prompt constant of the three agents changing, main() runs it against stand-in endpoints of the agent and of the reflection model,
and the export hands the final block to the variants of run_episodes/. Every reflection model is a fake, and run folders go
under pytest's tmp_path."""
import contextlib
import csv
import hashlib
import importlib.util
import json
import re
import types
from pathlib import Path
from unittest import mock

import pytest

from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import CAUSAL_SERVER, METERED_SERVERS, RIG_SERVER, registered_servers
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import COLUMNS, episode_name
from CausalDiagnosis.experiments.episode_runners.run_configuration import load_config, load_world
from CausalDiagnosis.utils.json_files import read_json, write_json
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path

if not Path(default_facts_path()).exists():
    pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states", allow_module_level=True)
if importlib.util.find_spec("mcp") is None:
    pytest.skip("needs the mcp package", allow_module_level=True)

from CausalDiagnosis.training import gepa_prompt_optimization as g  # noqa: E402
from CausalDiagnosis.experiments.episode_runners import run_episodes  # noqa: E402
from CausalDiagnosis.agents.Causal_Diagnosis_Agent import controller  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper import in_context_examples as port_examples, agent_builders, react_tool_adapter, sub_agent_registry  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents.ReActXen.agents.assessment_agent import assessment_prompt  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents.ReActXen.agents.react.prompts import agent_scratchpad_probe, fewshots, metaplanprompts, systemprompts  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents.ReActXen.agents.reviewer_agent import reviewer_prompt  # noqa: E402
from CausalDiagnosis.agents.Causal_Diagnosis_Agent import prompts as prompts_module  # noqa: E402
from agent.plan_execute import executor as executor_module, planner as planner_module, runner as pe_runner_module  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Plan_Execute_Paper import prompts as pe_port_prompts, runner as pe_port_runner, workflow as pe_port_workflow  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import start_state_seeds as start_states  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv  # noqa: E402
from llm import LLMBackend  # noqa: E402
from CausalDiagnosis.utils.llm_backends import ChatResult  # noqa: E402
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tests.test_causal_diagnosis_agent import SUMMARY_REPLY, InProcessServers, ScriptedPlanner, act_now, sessions_in_process  # noqa: E402
from CausalDiagnosis.training.tests.test_gepa_feedback_card import HIDDEN_KEYS, HIDDEN_VALUES, TRAJECTORY, _poisoned_verdict  # noqa: E402
from CausalDiagnosis.training.tests.test_rejection_sampling_examples import VISITS_OTHER  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.tests.test_run_episodes import _stand_in_endpoint  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.training_variants import CONFIG, DEFAULT_ASSET, FOLD_CONVENTION, FOLD_CONVENTION_KEY, FOLDS, TRAINING_SPLIT_KEY, VISITS  # noqa: E402

FIRST_BLOCK = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
BETTER = " Prefer the branch whose repair the evidence supports."
LOOP_ARGS = ["--model", "m", "--reflection-model", "r", "--reflection-base-url", "http://x", "--training-fold", "1"]
# the asset and the fold of every loop of the fake player, the registered number of proposals without an answer that fails it,
# and the registered attempts of one proposal
LOOP_OF = {"asset_id": DEFAULT_ASSET, "training_fold": 1, "unanswered_proposals": CONFIG["gepa"]["unanswered_proposals"],
           "proposal_attempts": CONFIG["gepa"]["proposal_attempts"]}
# the samples per training start state that the rounds of the sampling may play, from the registered numbers of the rig
SAMPLED = (1 + CONFIG["minimum"]["extension_rounds"]) * CONFIG["samples_per_state"]


PORT_CONSTANTS = {port_examples: ("PAPER_SUPERVISOR", "TASK_SUPERVISOR", "CAUSAL_SUPERVISOR", "IOT_RIG", "IOT_REPRO", "TSFM_REPRO",
                                   "FMSR_RIG", "FMSR_REPRO", "WO_RIG", "WO_REPRO", "TASK_TOOLS", "CAUSAL_TOOLS"),
                  # the shown names of the tools, which every prompt of a port holds
                  sub_agent_registry: ("PAPER", "OURS", "SURFACES", "SUPERVISOR_STEPS", "REFLECTIONS"),
                  react_tool_adapter: ("RESERVED", "ENGINE_WORDS"),
                  # the paper-era Plan-Execute that plan_execute plays
                  pe_port_prompts: ("INVALID_PLAN", "PLANNER", "REVIEWER", "SUMMARIZER"),
                  pe_port_workflow: ("PLANS", "REVIEW_ATTEMPTS", "SUMMARY_ATTEMPTS", "END_OF_RESPONSE", "NO_ANSWER"),
                  pe_port_runner: ("AGENT_HIVE", "STEP_AGENT_STEPS", "REFLECTIONS", "STEP_AGENT_OPTIONS")}
PORT_TEMPLATES = (systemprompts, fewshots, metaplanprompts, agent_scratchpad_probe, reviewer_prompt, assessment_prompt)


def prompt_constants():
    """The SHA-256 of every prompt constant of the three agents, which no candidate may change: the CausalDiagnosis agent's and
    Plan-Execute's prompts, and for the Agent-As-Tool port its examples, tool descriptions, sub-agent registry and step limits
    (repr of the structured ones) and every text of ReActXen's prompt templates."""
    texts = {"causal system prompt": prompts_module.SYSTEM_PROMPT, "causal repair rule": prompts_module.REPAIR_RULE,
             "plan prompt": planner_module._PLAN_PROMPT, "argument prompt": executor_module._ARG_RESOLUTION_PROMPT,
             "summary prompt": pe_runner_module._SUMMARIZE_PROMPT}
    for module, names in PORT_CONSTANTS.items():
        texts.update({f"{module.__name__}.{name}": repr(getattr(module, name)) for name in names})
    for module in PORT_TEMPLATES:
        texts.update({f"{module.__name__}.{name}": value for name, value in vars(module).items()
                      if isinstance(value, str) and not name.startswith("__")})
    return {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in texts.items()}


class FakeReflection(LLMBackend):
    """The reflection model without a model. A mutation returns the current block, read from the first block of the meta prompt,
    with one more sentence, unless reply, a function of the current block, says otherwise."""

    def __init__(self, reply=None):
        self._model_id = "fake-reflection"
        self.log, self.prompts = [], []
        self.reply = reply or (lambda current: f"```\n{current}{BETTER}\n```")

    def generate(self, prompt, temperature=0.0):
        self.prompts.append(prompt)
        self.log.append(("reflect", len(self.prompts)))
        return self.reply(FIRST_BLOCK.search(prompt).group(1).strip())


def better(candidate, seed):
    """0.8 for a block that carries the sentence of the fake reflection, 0.2 for any other."""
    return 0.8 if BETTER.strip() in candidate["texts"][g.BLOCK] else 0.2


PLAN_TEXT = ("Differential: [{\"component\": \"pump\"}]\n\nPlan 1 (model, branch 1, mode act_now): the pump\n"
             "  1. diagnose {\"rank\": 1} expect {\"top_rank\": 1} -> surprise\n  refused by the plan check: step 1 (diagnose) is refused\n")


class FakePlayer:
    """A scored episode folder per call in place of an episode, with the reward reward(candidate, seed), None for an episode
    that the verifier could not score, and a verdict that holds every hidden field when poisoned."""

    def __init__(self, run_dir, reward, log, poisoned=False):
        self.run_dir, self.reward, self.log, self.poisoned = Path(run_dir), reward, log, poisoned

    def __call__(self, candidate, seed, call):
        ep = self.run_dir / episode_name(seed, call=call)
        (ep / "hidden").mkdir(parents=True)
        verdict = _poisoned_verdict() if self.poisoned else {
            "checks": {"count_gate": {"ok": True}, "hard": {"ok": False}}, "pass": False, "success": False, "failure_class": "wrong_diagnosis",
            "abstained": False, "unnecessary": 0, "cost": 40, "end_reason": "finish", "steps": 4, "refusals": 0}
        (ep / "hidden" / "verdict.json").write_text(json.dumps(verdict), encoding="utf-8")
        (ep / "trajectory.json").write_text(json.dumps(TRAJECTORY), encoding="utf-8")
        (ep / "question.txt").write_text("the question of the episode", encoding="utf-8")
        (ep / "answer.txt").write_text("the closing summary", encoding="utf-8")
        (ep / "plan.txt").write_text(PLAN_TEXT, encoding="utf-8")
        reward = self.reward(candidate, seed)
        row = {k: None for k in COLUMNS}
        row.update({"run": "fake", "seed": seed, "reward": "" if reward is None else reward, "separable": True, "n_faulty": 1, "error": "",
                    "gepa_call": call})
        self.log.append(("play", call, candidate["id"], seed))
        return row, ep


def _loop(tmp_path, reward=better, poisoned=False, reflection=None, feedback="verifier", stream=(1, 2, 3, 4, 5, 6), block=2,
          max_episodes=10, player=FakePlayer):
    """One loop of the fake player and the fake reflection model into tmp_path. Returns the loop, its record, the reflection
    model and the log of the plays and the reflection calls in their order."""
    reflection = reflection or FakeReflection()
    loop = g.OnlineLoop(player(tmp_path, reward, reflection.log, poisoned), reflection, tmp_path, feedback, stream, block=block,
                        max_episodes=max_episodes, reflection_model=reflection.model_id, horizon=HORIZON,
                        **LOOP_OF)
    return loop, loop.run(), reflection, reflection.log


HORIZON = load_config("real_time")[1].horizon


class DeclaringPlayer(FakePlayer):
    """The fake player with the verdict of a correct declaration after one reading, which the verifier rewards with the declaration credit
    and the loop scores without it."""

    def __call__(self, candidate, seed, call):
        row, ep = super().__call__(candidate, seed, call)
        verdict = {"verified": True, "pass": False, "abstained": True, "abstention_correct": True, "success": True, "cost_normalized": 1.0,
                   "checks": {"provenance": {"ok": True}}, "detection_delay": {}, "repair_delay": {}, "unrestored": [], "failure_class": "none"}
        (ep / "hidden" / "verdict.json").write_text(json.dumps(verdict), encoding="utf-8")
        row["reward"] = g.verifier.reward(verdict, HORIZON)
        return row, ep


def test_both_cards_show_the_score_the_selection_uses_so_a_correct_declaration_earns_no_credit_there(tmp_path):
    """The number on the verifier's card and on the score-only card is
    the score of the stream, the reward with the declaration credit set to zero, on every episode the reflection reads, and not the
    reward of the scores row, which keeps the declaration credit for the tables."""
    for feedback in g.FEEDBACK:
        reflection = FakeReflection()
        loop = g.OnlineLoop(DeclaringPlayer(tmp_path / feedback, better, reflection.log), reflection, tmp_path / feedback, feedback,
                            (1, 2, 3, 4), block=2, max_episodes=6, reflection_model=reflection.model_id, horizon=HORIZON,
                            **LOOP_OF)
        record = loop.run()
        shown = [float(x) for x in re.findall(r"^reward ([^,\n]+)", reflection.prompts[0], re.M)]
        read = [e for e in record["episodes"] if e["candidate"] == "c000" and e["round"] == 1]
        assert len(shown) == len(read) == 2 and shown == [e["score"] for e in read]
        assert all(e["score"] < e["reward"] for e in read)  # the row's reward carries the credit of the declaration, the card does not


def test_the_loop_scores_a_declaring_episode_without_the_declaration_credit(tmp_path):
    """A correct declaration earns 0.5 of the outcome term from the verifier and
    0 in the loop, so a candidate cannot climb the stream by declaring, while the row keeps the full reward for the tables."""
    reflection = FakeReflection()
    loop = g.OnlineLoop(DeclaringPlayer(tmp_path, better, reflection.log), reflection, tmp_path, "verifier", (1, 2), block=2,
                        max_episodes=2, reflection_model=reflection.model_id, horizon=HORIZON, **LOOP_OF)
    record = loop.run()
    full, in_loop = round(0.5 - 0.2 * 1.0, 12), round(0.0 - 0.2 * 1.0, 12)
    assert [(e["reward"], e["score"]) for e in record["episodes"]] == [(full, in_loop)] * 2
    assert record["rounds"][0]["mean_score"] == in_loop and record["rounds"][0]["n"] == 2
    from CausalDiagnosis.experiments.table_definitions import registered_analysis as registered

    assert registered.reward_numbers()["stream_declaration_credit"] == 0.0 and not hasattr(g, "STREAM_DECLARATION_CREDIT")


def test_the_candidate_is_one_block_empty_at_the_start_and_of_any_length_for_every_agent():
    """GEPA edits one block of the same position for the three agents and never a system prompt, the starting candidate,
    the empty block, is the agent without a block, and no length limits the block: the meta prompt names none and a long
    proposal is taken."""
    assert g.MUTABLE == (g.BLOCK,) and g.seed_texts() == {g.BLOCK: ""} and g.FORMAT == 2
    assert "task_block_max_chars" not in CONFIG and not hasattr(g, "candidate_problems")
    long = "x" * 100_000
    assert g.proposal({"texts": {g.BLOCK: ""}}, g.BLOCK, f"```\n{long}\n```") == ({g.BLOCK: long}, [])
    prompt = g.PROMPT_META.format(current="", examples="# Example 1")
    assert "appended to the task it receives" in prompt and "at most" not in prompt and "characters" not in prompt
    assert FIRST_BLOCK.search(prompt).group(1).strip() == "" and "mark" not in prompt
    # one parser for the three agents, the loop's defaults the registered numbers of configs/training/hyd_1/rft.json
    parser = g.build_parser()
    for agent in ("plan_execute", "agent_as_tool", "causal"):
        a = parser.parse_args([*LOOP_ARGS, "--agent", agent])
        assert g.driver_settings(a).agent == agent and g.gepa_meta(a, DEFAULT_ASSET)["agent"] == agent and g.gepa_meta(a, DEFAULT_ASSET)["components"] == [g.BLOCK]
    defaults = parser.parse_args(LOOP_ARGS)
    # the parser leaves the numbers of a loop out, and the command takes them from the rft.json of the asset of the preset
    assert all(getattr(defaults, name) is None for name in g.GEPA_NUMBERS)
    g.with_registered_numbers(defaults, DEFAULT_ASSET)
    registered = CONFIG["gepa"]
    assert (defaults.max_episodes, defaults.block, defaults.window, defaults.children, defaults.gepa_seed, defaults.card) == (
        registered["max_episodes"], registered["block"], registered["window"], registered["children"], registered["gepa_seed"], registered["card"])
    with pytest.raises(SystemExit):
        parser.parse_args([*LOOP_ARGS, "--training-variant", "gepa"])
    # a loop belongs to one fold of the registered split, which it names, and the settings and run.json carry it; a fold the split
    # does not name is refused before any episode
    with pytest.raises(SystemExit):
        parser.parse_args(LOOP_ARGS[:-2])
    beyond = parser.parse_args([*LOOP_ARGS[:-1], str(max(fold for fold, _models in FOLDS.values()) + 1)])
    with pytest.raises(SystemExit, match="no fold of the registered split"):
        run_episodes.training_checks(g.driver_settings(beyond), load_config(beyond.config)[1])
    assert g.driver_settings(defaults).training_fold == g.gepa_meta(defaults, DEFAULT_ASSET)["training_fold"] == 1


def test_the_loop_command_refuses_a_fold_that_the_split_does_not_name_before_any_episode(tmp_path):
    """The parser takes any integer, so the command checks the fold against the split of the asset of its preset and stops by
    name, before it plays an episode or writes a run folder."""
    folds = tuple(sorted({fold for fold, _models in FOLDS.values()}))
    beyond = max(folds) + 1
    with pytest.raises(SystemExit, match=re.escape(f"names the folds {folds}, got {beyond}")):
        g.main([*LOOP_ARGS[:-1], str(beyond), "--out", str(tmp_path / "runs")])
    assert not (tmp_path / "runs").exists()


def test_each_block_is_scored_before_the_reflection_reads_it_and_the_curve_scores_the_incumbent_fixed_before_it(tmp_path):
    loop, record, reflection, log = _loop(tmp_path)
    assert [entry[1] for entry in log if entry[0] == "play"] == list(range(1, 11))
    # the reflection after round r comes after every episode of round r and before the first episode of round r + 1
    assert log.index(("reflect", 1)) == 2 and log.index(("reflect", 2)) == 7 and record["reflection_calls"] == 2
    rounds = record["rounds"]
    assert [(r["round"], r["rollouts"], r["incumbent"], r["mean_score"], r["n"]) for r in rounds] == [
        (1, 0, "c000", 0.2, 2), (2, 2, "c000", 0.2, 2), (3, 6, "c001", 0.8, 2)]
    # c001 earns 0.8 on the second block, yet the point of that round is the incumbent fixed before the block
    assert rounds[1]["means"] == {"c000": 0.2, "c001": 0.8} and rounds[1]["admitted"] == ["c001"] and rounds[1]["chosen"] == "c001"
    assert rounds[1]["front"] == ["c001"] and rounds[2]["members"] == ["c001", "c002"] and rounds[2]["rejected"] == ["c002"]
    assert {c["id"]: c["status"] for c in record["candidates"]} == {"c000": "start", "c001": "admitted", "c002": "rejected"}
    assert {c["id"]: c["component"] for c in record["candidates"]} == {"c000": None, "c001": g.BLOCK, "c002": g.BLOCK}
    assert record["episodes_played"] == 10 and record["rollouts_to_best"] == 6 and record["final_candidate"] == "c001"
    # a loop plays the stream alone: every episode is a stream episode, and the record holds no held-out comparison
    assert {e["seed"] for e in record["episodes"]} == {1, 2, 3, 4, 5, 6} and {e["phase"] for e in record["episodes"]} == {"stream"}
    assert not [key for key in (*record, *record["settings"]) if "heldout" in key] and record[VISITS] == 0
    saved = read_json(tmp_path / g.RESULT)
    assert record["status"] == "finished" and {k: v for k, v in saved.items() if k != "state"} == record and saved["state"]["calls"] == 10
    with open(tmp_path / "scores.csv", newline="", encoding="utf-8") as f:
        assert [r["gepa_call"] for r in csv.DictReader(f)] == [str(n) for n in range(1, 11)]
    assert (tmp_path / "candidates" / "c000_task_block.txt").read_text(encoding="utf-8") == ""
    assert (tmp_path / "candidates" / "c001_task_block.txt").read_text(encoding="utf-8") == BETTER.strip()
    assert [c["characters"] for c in record["candidates"]] == [0, len(BETTER.strip()), len(BETTER.strip() + BETTER)]
    assert record["settings"]["components"] == [g.BLOCK] and "task_block_max_chars" not in record["settings"]


def test_the_score_only_control_shows_the_reward_alone_and_every_input_and_output_alike(tmp_path):
    prompts = {}
    for feedback in g.FEEDBACK:
        _, _, reflection, _ = _loop(tmp_path / feedback, feedback=feedback, stream=(1, 2, 3, 4), max_episodes=6)
        assert len(reflection.prompts) == 1
        prompts[feedback] = reflection.prompts[0]
    card, score = prompts["verifier"], prompts["score_only"]
    assert "check count_gate: ok" in card and "tool calls:" in card and "reward 0.2, cost 40" in card
    assert "## Feedback\nreward 0.2" in score
    for text in ("check ", "tool calls:", "failure class", "diagnostics server messages", "cost 40"):
        assert text not in score, text

    def before_feedback(prompt):
        return [part.split("## Feedback")[0] for part in prompt.split("# Example ")[1:]]

    assert len(before_feedback(card)) == 2 and before_feedback(card) == before_feedback(score)
    assert "### question\nthe question of the episode" in score and "Plan 1 (model, branch 1, mode act_now)" in score
    assert "refused by the plan check: step 1 (diagnose) is refused" in score and "summary: the closing summary" in score


def test_the_outputs_of_agent_as_tool_are_the_supervisor_steps_without_the_sub_agents_replies(tmp_path):
    ep = tmp_path / "call00001_seed000001"
    steps = [{"step": 1, "thought": "Read the pump first.", "action": "IoTAgent", "action_input": "pump pressure of the last hour",
              "observation": "PS1 155 bar", "state": "Valid Action"},
             {"step": 2, "thought": "The pump is degraded.", "action": "Finish", "action_input": "The pump is degraded.",
              "observation": "", "state": "Valid Action"}]
    write_json(ep / "react_trajectory.json", {"trajectroy_log": steps, "sub_agents": [{"agent": "IoTAgent", "answer": "PS1 155 bar"}]})
    write_json(ep / "llm_calls.json", [{"agent": "supervisor", "kind": "thought", "reply": "a reply the steps already hold"}])
    (ep / "answer.txt").write_text("The pump is degraded.", encoding="utf-8")
    outputs = g.agent_outputs(ep)
    assert outputs == ("plan:\nstep 1: Read the pump first.\n  IoTAgent: pump pressure of the last hour (Valid Action)\n"
                       "step 2: The pump is degraded.\n  Finish: The pump is degraded. (Valid Action)\nsummary: The pump is degraded.")
    assert "155 bar" not in outputs and "already hold" not in outputs


def test_a_raised_agent_as_tool_run_falls_back_to_the_supervisor_replies(tmp_path):
    ep = tmp_path / "call00001_seed000001"
    write_json(ep / "llm_calls.json", [{"agent": "supervisor", "kind": "thought", "reply": " Read the pump first."},
                                       {"agent": "IoTAgent", "kind": "action", "reply": "a sub-agent's call"}])
    assert g.agent_outputs(ep) == "plan:\nthought:  Read the pump first.\nsummary: "


def test_no_hidden_field_and_no_seed_reaches_any_reflection_prompt(tmp_path):
    for feedback in g.FEEDBACK:
        _, record, reflection, _ = _loop(tmp_path / feedback, poisoned=True, feedback=feedback, stream=(424242, 424243, 424244, 424245),
                                         max_episodes=6)
        assert reflection.prompts and record["episodes"][0]["seed"] == 424242  # the record keeps the seed, and no prompt reads it
        for prompt in reflection.prompts:
            assert [k for k in HIDDEN_KEYS if k in prompt] == [] and [v for v in HIDDEN_VALUES if v in prompt] == []


def test_a_refused_proposal_goes_back_once_with_its_problems_and_never_plays(tmp_path):
    reflection = FakeReflection(reply=lambda current: "no block here")
    _, record, reflection, _ = _loop(tmp_path, reflection=reflection, block=1, max_episodes=4)
    assert {c["id"]: (c["status"], c["component"]) for c in record["candidates"]} == {
        "c000": ("start", None), "c001": ("refused", g.BLOCK), "c002": ("refused", g.BLOCK)}
    problem = g.NO_BLOCK
    assert all(c["problems"] == [problem] for c in record["candidates"][1:])
    assert len(reflection.prompts) == 4 and g.RETRY_HEADING not in reflection.prompts[0]
    assert g.RETRY_HEADING in reflection.prompts[1] and problem in reflection.prompts[1]
    assert {e["candidate"] for e in record["episodes"]} == {"c000"} and all(r["members"] == ["c000"] for r in record["rounds"])
    assert sorted(p.name for p in (tmp_path / "reflection").iterdir()) == [
        "c001_attempt1.json", "c001_attempt2.json", "c002_attempt1.json", "c002_attempt2.json"]
    assert not (tmp_path / "candidates" / "c001_task_block.txt").exists()


class CutReflection(FakeReflection):
    """A reflection model whose replies stop at the token limit inside the block."""

    def generate_with_usage(self, prompt, temperature=0.0):
        self.prompts.append(prompt)
        self.log.append(("reflect", len(self.prompts)))
        return ChatResult(text="```\nan instruction that the cap cut", finish_reason="length")


def test_a_reply_cut_at_the_token_limit_asks_for_a_shorter_text(tmp_path):
    _, record, reflection, _ = _loop(tmp_path, reflection=CutReflection(), stream=(1, 2, 3, 4), max_episodes=6)
    assert record["candidates"][1]["status"] == "refused" and record["candidates"][1]["problems"] == [g.CUT_REPLY]
    assert g.CUT_REPLY in reflection.prompts[1] and read_json(tmp_path / "reflection" / "c001_attempt2.json")["finish_reason"] == "length"


def test_a_reply_without_a_block_or_with_the_parent_unchanged_is_refused_and_the_last_block_is_read():
    parent = {"texts": {g.BLOCK: "Read the cheapest group first."}}
    assert g.proposal(parent, g.BLOCK, "no block here") == (None, [g.NO_BLOCK])
    assert g.proposal(parent, g.BLOCK, "```text\nRead the cheapest group first.\n```")[1] == ["the reply repeats the current task block unchanged"]
    texts, problems = g.proposal(parent, g.BLOCK, f"A draft.\n```\nfirst\n```\nThe final one.\n```text\nRead it.{BETTER}\n```")
    assert problems == [] and texts == {g.BLOCK: f"Read it.{BETTER}"}


def test_an_episode_that_the_verifier_could_not_score_enters_no_mean_and_is_counted_apart(tmp_path):
    def reward(candidate, seed):
        return None if seed == 2 else better(candidate, seed)

    _, record, _, _ = _loop(tmp_path, reward=reward, stream=(1, 2, 3, 4), max_episodes=6)
    first = record["rounds"][0]
    assert (first["mean_score"], first["n"], first["unscored"]) == (0.2, 1, 1)
    assert [e["score"] for e in record["episodes"] if e["seed"] == 2] == [None]
    assert not g.improves(None, 0.2) and g.improves(0.3, None) and not g.improves(0.2, 0.2) and g.improves(-0.1, -0.4)


class LeftOutPlayer(FakePlayer):
    """The fake player whose episodes on the seeds of left_out reach a single fault of fold 1, the fold the loop's model holds out,
    with a question and a plan of their own."""
    left_out = ()

    def __call__(self, candidate, seed, call):
        row, ep = super().__call__(candidate, seed, call)
        if seed in self.left_out:
            write_json(ep / "hidden" / "log.json", VISITS_OTHER)
            (ep / "question.txt").write_text("the question of an episode left out", encoding="utf-8")
            (ep / "plan.txt").write_text("a plan of an episode left out", encoding="utf-8")
        return row, ep


def test_a_stream_episode_that_met_a_held_out_fault_scores_nothing_and_reaches_no_reflection(tmp_path):
    """An episode whose hidden log reaches a single fault of the fold the loop's model holds out has no score and enters no mean,
    no reflection prompt holds its question or its outputs, the round and the record count it, and the budget counts it as played.
    A parent whose episodes of a round were all left out proposes no child, so the reflection model is first asked a round later."""

    class OneOut(LeftOutPlayer):
        left_out = (2,)

    _, record, reflection, _ = _loop(tmp_path / "one", stream=(1, 2, 3, 4), max_episodes=6, player=OneOut)
    assert [(e["seed"], e["score"]) for e in record["episodes"] if e["left_out"]] == [(2, None)]
    assert record[VISITS] == 1 and record["episodes_played"] == 6
    first = record["rounds"][0]
    assert (first["mean_score"], first["n"], first["unscored"], first["left_out"]) == (0.2, 1, 0, 1)
    assert len(reflection.prompts) == 1 and reflection.prompts[0].count("# Example ") == 1 and "left out" not in reflection.prompts[0]

    class BothOut(LeftOutPlayer):
        left_out = (1, 2)

    _, record, _, log = _loop(tmp_path / "both", stream=(1, 2, 3, 4, 5, 6), max_episodes=8, player=BothOut)
    plays = [entry for entry in log if entry[0] == "play"]
    assert record["rounds"][0]["left_out"] == 2 and record["reflection_calls"] == 1 and log.index(("reflect", 1)) > log.index(plays[3])
    assert [c["id"] for c in record["candidates"]] == ["c000", "c001"]


def test_the_stream_starts_after_every_seed_a_sampling_round_may_play():
    """On the real data, per fold: the stream serves the training start states of the fold's model in seed order, no test start
    state and no reserved seed, and passes over, per state, every seed that a round of the sampling may play, (1 + extension_rounds)
    times samples_per_state, a count computed from the registered numbers. It shares no seed with training_seeds of those samples,
    and its first seed of a state is the next seed of that state. The loops on the base weights and on the adapter of a fold read
    the same stream."""
    a = g.build_parser().parse_args(LOOP_ARGS)
    settings = g.driver_settings(a)
    assert not hasattr(settings, "diagnostics_table")  # one table serves every row, whose fits leave out the start's block
    with pytest.raises(SystemExit):
        g.build_parser().parse_args([*LOOP_ARGS, "--diagnostics-table", "unread"])
    facts_path, cycles, blocks = load_world()
    _name, config = load_config(a.config)
    env = HydraulicRigEnv(cycles, blocks, config)
    test_set = start_states.sampled_combination_seeds(env, 42, 8, start_states.REPLICATES)

    def state_of(seed):
        env.reset(seed)
        return start_states.start_state(env)

    played = {state_of(seed) for seed in test_set}

    def stream_of(fold, variant):
        loop = g.build_parser().parse_args([*LOOP_ARGS[:-1], str(fold), "--training-variant", variant, "--max-episodes", "40"])
        return g.loop_stream(g.with_registered_numbers(loop, DEFAULT_ASSET), config, cycles, blocks)

    for fold in (1, 2):
        stream = stream_of(fold, "as_released")
        assert len(stream) == 40 and (fold == 2 or stream_of(fold, "ft") == stream)
        # the stream is the pool of the fold's fine tuning: its training states alone, in seed order, no test start state and no
        # reserved seed, and the model of the fold trains on every stream combination
        states = start_states.training_start_states(env, FOLDS, fold)
        assert stream == sorted(stream) and not set(stream) & (set(test_set) | start_states.reserved_seeds(env))
        assert {state_of(seed) for seed in stream} <= set(states) and not set(states) & played
        assert all(fold in FOLDS[state_of(seed)[0]][1] for seed in stream)
        # the last sample a round may play and the next one, per state: the samples of a state are its free seeds in seed order, so a
        # stream whose seeds of each state start at the next sample shares no seed with the samples 1 to SAMPLED
        last = dict(zip(states, start_states.training_seeds(env, FOLDS, SAMPLED, fold)))
        following = dict(zip(states, start_states.training_seeds(env, FOLDS, SAMPLED + 1, fold)))
        firsts = {}
        for seed in stream:
            firsts.setdefault(state_of(seed), seed)
        assert firsts and all(firsts[state] == following[state] > last[state] for state in firsts)
    # the loop plays the preset of training, all six big layers, on AssetOpsBench's servers without a switch, the rule of
    # run_episodes.default_servers, and --servers rig names the rig
    assert (settings.agent, settings.config, settings.servers, settings.store, settings.starts) == ("causal", "deterioration", "aob", "memory", "sampled")
    assert registered_servers(settings.servers, True) == (*METERED_SERVERS, CAUSAL_SERVER)
    rig = g.driver_settings(g.build_parser().parse_args([*LOOP_ARGS, "--servers", "rig"]))
    assert registered_servers(rig.servers, True) == (RIG_SERVER, CAUSAL_SERVER)


def test_playing_puts_the_block_in_place_and_restores_it_after_an_error():
    runner = types.SimpleNamespace(task_block="")
    with pytest.raises(RuntimeError, match="episode failed"):
        with g.playing(runner, {"texts": {g.BLOCK: "Read first."}}):
            assert runner.task_block == "Read first."
            raise RuntimeError("episode failed")
    assert runner.task_block == ""
    with g.playing(runner, {"texts": g.seed_texts()}):
        assert runner.task_block == ""


@contextlib.contextmanager
def _servers_in_process():
    """The runner's tool calls answered in process, one InProcessServers per episode folder, through the stand-in of the session
    module of CausalDiagnosis/agents/Causal_Diagnosis_Agent/tests/test_causal_diagnosis_agent.py, without the stdio tool listing."""
    servers = {}

    def servers_of(path):
        ep_dir = Path(path).parent
        if ep_dir not in servers:
            servers[ep_dir] = InProcessServers(ep_dir)
        return servers[ep_dir]

    with mock.patch.object(controller, "open_sessions", sessions_in_process(servers_of)), \
            mock.patch.object(run_episodes.play_one_episode, "check_servers", lambda launchers, row=None: None):
        yield


def _llm_prompts(ep_dir):
    return [call["prompt"] for call in read_json(ep_dir / "llm_calls.json")]


def test_the_loop_plays_the_causal_agent_with_each_block_on_its_task_message_and_no_prompt_changes(tmp_path, monkeypatch):
    a = g.build_parser().parse_args(["--model", "scripted-causal", "--reflection-model", "fake-reflection", "--reflection-base-url",
                                     "http://127.0.0.1:9/v1", "--config", "static", "--block", "1", "--max-episodes", "5",
                                     "--training-fold", "1", "--out", str(tmp_path / "runs")])
    g.with_registered_numbers(a, DEFAULT_ASSET)  # as main does, since the parser leaves the registered numbers out
    settings = g.driver_settings(a)
    facts_path, cycles, blocks = load_world()
    config_name, config = load_config(a.config)
    planner = ScriptedPlanner(act_now)
    runner = run_episodes.EpisodeRunner(config, config_name, "causal", (RIG_SERVER, CAUSAL_SERVER), cycles, blocks, facts_path, "gepa-test",
                                        {"judge": None, "grade": True, "ranking_model": run_episodes.DEFAULT_RANKING_MODEL},
                                        backend=run_episodes.RecordingBackend(planner), scripted=planner, model=planner.model_id,
                                        causal_options=run_episodes.causal_options(settings))
    stream = start_states.training_stream(HydraulicRigEnv(cycles, blocks, config), FOLDS, 3, 1)
    reflection = FakeReflection()
    before = prompt_constants()
    monkeypatch.setattr(run_episodes.start_check, "registration", lambda *args: [])  # a loop of record, the one that the registered split plays
    with _servers_in_process():
        run_dir, loop = g.run_loop.run_online(a, ["gepa_online.py"], settings, runner, config, None, stream, reflection)
    assert prompt_constants() == before and runner.task_block == ""
    assert loop.attempts == g.run_loop.numbers(DEFAULT_ASSET).proposal_attempts  # the registered attempts of a proposal reach the loop
    record, meta, ledger = read_json(run_dir / g.RESULT), read_json(run_dir / "run.json"), read_json(run_dir / "ledger.json")
    assert meta["mode"] == g.MODE and meta["gepa"]["card"] == "verifier" and meta["gepa"]["reflection_model"] == "fake-reflection"
    assert meta["gepa"]["agent"] == "causal" and meta["gepa"]["training_variant"] == "as_released" and meta["training_variant"] == "as_released"
    assert meta["gepa"]["training_fold"] == meta["training_fold"] == record["settings"]["training_fold"] == 1
    # the loop asks the start check before its run.json, so a loop of record carries no stamp
    assert meta[TRAINING_SPLIT_KEY] == "registered" and "stamp" not in meta and "stamp_reasons" not in meta
    assert meta["episodes_planned"] == 5 and meta["seeds"] == stream and "thresholds" not in meta
    names = [e["episode"] for e in record["episodes"]]
    assert ledger["status"] == "finished" and ledger["configuration"]["mode"] == g.MODE and sorted(ledger["episodes"]) == sorted(names)
    assert [c["component"] for c in record["candidates"]] == [None, g.BLOCK, g.BLOCK]
    by_candidate = {}
    for e in record["episodes"]:
        by_candidate.setdefault(e["candidate"], []).append(run_dir / e["episode"])
        verdict = read_json(run_dir / e["episode"] / "hidden" / "verdict.json")
        assert verdict["verified"] and e["score"] is not None
    # the block of a candidate reaches the agent appended to the question of its orientation, question.txt keeps the briefing alone,
    # and every prompt of every candidate opens with the same system prompt, that of the surface of the run's asset
    surface = run_episodes.causal_surface(runner.config.asset_id, runner.store)
    system = prompts_module.system_prompt(run_episodes.causal_options(settings)["plan_length"], surface=surface)
    for cid, folders in by_candidate.items():
        block = (run_dir / "candidates" / f"{cid}_task_block.txt").read_text(encoding="utf-8")
        for ep in folders:
            question = (ep / "question.txt").read_text(encoding="utf-8")
            prompts = _llm_prompts(ep)
            assert prompts and all(p.startswith(system) for p in prompts) and not (block and block in question)
            assert f"{question}\n\n{block}" in prompts[0] if block else question in prompts[0]
            assert (ep / "task_block.txt").exists() == bool(block)
    assert all(BETTER.strip() not in " ".join(_llm_prompts(ep)) for ep in by_candidate["c000"])
    with open(run_dir / "scores.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["gepa_call"] for r in rows] == [str(n) for n in range(1, 6)] and {r["agent"] for r in rows} == {"causal_scripted-causal"}
    # the reflection read the episode cards of the block, and no prompt names a hidden field or an episode folder
    assert len(reflection.prompts) == 2 and all("# Example 1" in p and "check " in p and "## Feedback" in p for p in reflection.prompts)
    for prompt in reflection.prompts:
        assert [k for k in HIDDEN_KEYS if k != "seed" and k in prompt] == [] and "seed0" not in prompt and "hidden/" not in prompt
    assert {e["phase"] for e in record["episodes"]} == {"stream"} and not [key for key in record if "heldout" in key]


class Reached(Exception):
    """Raised by a stand-in of the step after a check, so that a test sees that a loop passed the check."""


def test_a_loop_on_the_registered_split_plays_only_as_a_run_of_record(tmp_path, monkeypatch):
    """A loop of GEPA on the registered split plays the training starts of its fold, which lie on test combinations of the other
    folds: a loop whose start check gives a reason is refused by name before its run folder exists and pointed to the development
    split, while a loop of a trial with the same reason and a loop of record on the registered split pass to their run.json."""
    def reached(*args, **kwargs):
        raise Reached

    monkeypatch.setattr(run_episodes, "run_meta", reached)
    runner, config = types.SimpleNamespace(config_name="static", facts_path=None), load_config("static")[1]
    words = ["--model", "m", "--reflection-model", "r", "--reflection-base-url", "http://127.0.0.1:9/v1", "--training-fold", "1",
             "--out", str(tmp_path / "runs")]
    registered, trial = g.build_parser().parse_args(words), g.build_parser().parse_args([*words, "--training-split", "development"])

    def online(a):
        g.run_loop.run_online(a, ["gepa_online.py"], g.driver_settings(a), runner, config, None, [], FakeReflection())

    monkeypatch.setattr(run_episodes.start_check, "registration", lambda *args: ["a reason of the start check"])
    with pytest.raises(SystemExit, match="a loop of GEPA on the registered split .*a reason of the start check.*--training-split development"):
        online(registered)
    with pytest.raises(Reached):
        online(trial)
    monkeypatch.setattr(run_episodes.start_check, "registration", lambda *args: [])  # a loop of record
    with pytest.raises(Reached):
        online(registered)
    assert not (tmp_path / "runs").exists()


def test_main_runs_the_loop_through_the_endpoints_of_the_agent_and_of_the_reflection_model(tmp_path, monkeypatch):
    agent_server, agent_requests = _stand_in_endpoint(fixed_reply=SUMMARY_REPLY)
    reflection_server, reflection_requests = _stand_in_endpoint(fixed_reply=f"```\n{BETTER.strip()}\n```")
    monkeypatch.setenv("API_KEY", "stand-in-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")

    def no_test_set(*args):
        raise AssertionError("a loop chooses no seed of the test set")

    monkeypatch.setattr(run_episodes.planned_episodes, "chosen_seeds", no_test_set)
    monkeypatch.setattr(run_episodes.start_check, "registration", lambda *args: [])  # a loop of record, the one that the registered split plays
    asked, stream_of = [], g.loop_settings.training_stream
    monkeypatch.setattr(g.loop_settings, "training_stream", lambda env, split, count, fold, first_seed, master_seed, pool, skip_per_state: asked.append(
        (count, fold, first_seed, master_seed, pool, skip_per_state)) or stream_of(env, split, count, fold, first_seed, master_seed, pool, skip_per_state))
    try:
        with _servers_in_process():
            run_dir = g.main(["--model", "vendor/stand-in-1.0", "--base-url", f"http://127.0.0.1:{agent_server.server_address[1]}/v1/",
                              "--reflection-model", "stand-in-reflection",
                              "--reflection-base-url", f"http://127.0.0.1:{reflection_server.server_address[1]}/v1/",
                              "--config", "static", "--block", "1", "--max-episodes", "3", "--training-fold", "2", "--out", str(tmp_path / "runs")])
    finally:
        agent_server.shutdown()
        reflection_server.shutdown()
    assert run_dir.name.startswith("static_gepa_online_verifier_causal_as_released_f2_vendor_stand-in-1_0_")
    assert asked == [(3, 2, 0, 42, 8, SAMPLED)]
    env = HydraulicRigEnv(*load_world()[1:], load_config("static")[1])
    allowed, reserved = set(start_states.training_start_states(env, FOLDS, 2)), start_states.reserved_seeds(env)
    streamed = [e["seed"] for e in read_json(run_dir / g.RESULT)["episodes"] if e["phase"] == "stream"]

    def state_of(seed):
        env.reset(seed)
        return start_states.start_state(env)

    assert streamed and all(state_of(seed) in allowed and seed not in reserved for seed in streamed)
    meta, record, ledger = read_json(run_dir / "run.json"), read_json(run_dir / g.RESULT), read_json(run_dir / "ledger.json")
    assert meta["mode"] == g.MODE and meta["model"] == "vendor/stand-in-1.0" and meta["starts"] == "sampled" and meta["training_fold"] == 2
    # the loop plays training start states alone: no seed of the test set was chosen, and the record holds no held-out comparison
    assert {e["phase"] for e in record["episodes"]} == {"stream"} and not [key for key in record if "heldout" in key]
    assert meta["gepa"][FOLD_CONVENTION_KEY] == FOLD_CONVENTION
    assert meta["gepa"]["reflection_model"] == "stand-in-reflection" and meta["gepa"]["reflection_request"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert meta["gepa"]["reflection_max_tokens"] is None
    # the reflection model is asked once, at temperature zero, without thinking and with no reply cap, on its own endpoint
    assert len(reflection_requests) == 1 and record["reflection_calls"] == 1
    body = reflection_requests[0][1]
    assert body["model"] == "stand-in-reflection" and body["temperature"] == 0.0 and body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body.get("max_tokens") is None
    assert agent_requests and all(request[1]["model"] == "vendor/stand-in-1.0" and "max_tokens" not in request[1] for request in agent_requests)  # no reply cap of the harness on the chat route
    assert record["episodes_played"] == 3 and [c["status"] for c in record["candidates"]][0] == "start"
    assert record["candidates"][1]["component"] == g.BLOCK and record["candidates"][1]["status"] in ("admitted", "rejected")
    assert ledger["status"] == "finished" and sorted(ledger["episodes"]) == sorted(e["episode"] for e in record["episodes"])


ADAPTER = {"name": "rft_causal_f1", "arm": "filtered", "fold": 1, "weights_sha256": "ab" * 32}


def _finished_loop(root, card="verifier", status="finished", agent="causal", variant="as_released", block="Read the cheapest group first.",
                   fold=1, left_out=(False, False, False, False), proposals=(2, 2)):
    """A finished loop of one fold: four stream episodes, of which the loop left out those that left_out marks, and the proposals
    of the loop with those the reflection model answered."""
    run = root / f"loop_{card}_{status}_{agent}_{variant}_f{fold}"
    write_json(run / "run.json", {"mode": g.MODE, "gepa": {"card": card, "agent": agent, "training_variant": variant, "training_fold": fold,
                                                           FOLD_CONVENTION_KEY: FOLD_CONVENTION},
                                  "training_fold": fold, "adapter": ADAPTER if variant == "ft" else None})
    episodes = [{"call": n, "episode": episode_name(n, call=n), "candidate": c, "phase": "stream", "left_out": out}
                for n, (c, out) in enumerate(zip(("c000", "c000", "c001", "c003"), left_out), 1)]
    write_json(run / g.RESULT, {"status": status, "final_candidate": "c003", "episodes": episodes, "episodes_played": 4,
                                "settings": {"training_fold": fold}, "proposals": proposals[0], "proposals_answered": proposals[1]})
    (run / "candidates").mkdir(parents=True)
    (run / "candidates" / "c003_task_block.txt").write_text(block, encoding="utf-8")
    return run


def test_the_export_hands_the_final_block_with_its_record_to_the_variants_and_refuses_another_loop(tmp_path):
    training = tmp_path / "training"
    target = g.main(["export", str(_finished_loop(tmp_path)), "--training-root", str(training)])
    assert target == training / "blocks" / "gepa_causal_f1.txt" and target.read_text(encoding="utf-8") == "Read the cheapest group first."
    record = read_json(target.with_suffix(".json"))
    assert record["agent"] == "causal" and record["base_variant"] == "as_released" and record["candidate"] == "c003" and record["fold"] == 1
    assert record["card"] == "verifier" and record[FOLD_CONVENTION_KEY] == FOLD_CONVENTION
    assert record["sha256"] == hashlib.sha256(b"Read the cheapest group first.").hexdigest() and record["characters"] == 30
    # the block names the run of its loop with the SHA-256 of its run.json, and the stamp of that run, none for a registered run
    loop = target.parent.parent.parent / "loop_verifier_finished_causal_as_released_f1"
    assert record["built_from"] == [{"run": loop.name, "run_json_sha256": hashlib.sha256((loop / "run.json").read_bytes()).hexdigest()}]
    assert record["stamp"] is None and record[TRAINING_SPLIT_KEY] == "registered"
    # the block of a loop of a trial names the development split, which a run of the registered split refuses
    trial = _finished_loop(tmp_path / "trial")
    write_json(trial / "run.json", {**read_json(trial / "run.json"), TRAINING_SPLIT_KEY: "development"})
    exported = g.main(["export", str(trial), "--training-root", str(tmp_path / "trial_training")])
    assert read_json(exported.with_suffix(".json"))[TRAINING_SPLIT_KEY] == "development"
    # what the table of what each agent trained on counts of the loop: the candidates that played on the stream, c002 refused, the
    # stream episodes and those the loop left out
    assert (record["candidates_played"], record["stream_episodes"], record[VISITS]) == (3, 4, 0)
    assert record["adapter"] is None and not [key for key in record if "heldout" in key]
    ft = g.main(["export", str(_finished_loop(tmp_path, variant="ft")), "--training-root", str(training)])
    assert ft.name == "gepa_ft_causal_f1.txt" and read_json(ft.with_suffix(".json"))["adapter"] == ADAPTER
    assert g.main(["export", str(_finished_loop(tmp_path, fold=2)), "--training-root", str(training)]).name == "gepa_causal_f2.txt"
    unfolded = _finished_loop(tmp_path / "unfolded")
    meta = read_json(unfolded / "run.json")
    write_json(unfolded / "run.json", {**meta, "gepa": {k: v for k, v in meta["gepa"].items() if k != "training_fold"}})
    # the score-only loop on the base weights exports under its own name and leaves the verifier's block alone
    control = g.main(["export", str(_finished_loop(tmp_path, card="score_only", block="Read the score.")), "--training-root", str(training)])
    assert control.name == "gepa_score_only_causal_f1.txt" and control.read_text(encoding="utf-8") == "Read the score."
    assert read_json(control.with_suffix(".json"))["card"] == "score_only"
    # a loop whose run.json does not state the fold convention was played under the earlier fold naming
    earlier = _finished_loop(tmp_path / "earlier")
    meta = read_json(earlier / "run.json")
    write_json(earlier / "run.json", {**meta, "gepa": {k: v for k, v in meta["gepa"].items() if k != FOLD_CONVENTION_KEY}})
    refused = ((_finished_loop(tmp_path, card="score_only", variant="ft"), "comes from a loop under the card 'score_only' on 'ft'"),
               (_finished_loop(tmp_path, card="another"), "is no finished online loop under a feedback card"),
               (_finished_loop(tmp_path, status="running"), "is no finished online loop under a feedback card"),
               (_finished_loop(tmp_path / "other", block="Another block."), "holds another block"),
               (unfolded, "and the fold None, which no block"),
               (earlier, "was written under the earlier fold convention"))
    for loop, message in refused:
        with pytest.raises(SystemExit, match=re.escape(message)):
            g.main(["export", str(loop), "--training-root", str(training)])
    assert target.read_text(encoding="utf-8") == "Read the cheapest group first."


def test_the_export_counts_the_stream_episodes_that_the_loop_left_out(tmp_path):
    """The gepa and ft_gepa columns train on the stream, so the record counts the stream episodes that the loop left out, those
    whose hidden log reaches a single fault of the fold the loop's model holds out."""
    loop = _finished_loop(tmp_path, left_out=(False, True, False, True))
    record = read_json(g.main(["export", str(loop), "--training-root", str(tmp_path / "training")]).with_suffix(".json"))
    assert record[VISITS] == 2


class FailingReflection(FakeReflection):
    """A reflection model whose endpoint raises on the calls of raises, counted from 1, and answers as FakeReflection otherwise."""

    def __init__(self, raises, reply=None):
        super().__init__(reply)
        self.raises, self.asked = set(raises), 0

    def generate_with_usage(self, prompt, temperature=0.0):
        self.asked += 1
        if self.asked in self.raises:
            self.log.append(("reflect", "error"))
            raise ConnectionError("the reflection endpoint is down")
        return super().generate_with_usage(prompt, temperature)


def test_a_loop_fails_after_two_proposals_in_a_row_without_an_answer(tmp_path, monkeypatch):
    """A reflection endpoint that raises on both attempts of two proposals in a row fails the loop: the result file says failed,
    no stream episode plays after the second proposal, the export refuses the folder, and main exits with status 1 and keeps the
    folder for --resume."""
    reflection = FailingReflection(raises=range(1, 5))
    with pytest.raises(g.LoopFailed, match="answered none of the last 2 proposals"):
        _loop(tmp_path / "loop", reflection=reflection)
    saved = read_json(tmp_path / "loop" / g.RESULT)
    plays = [entry for entry in reflection.log if entry[0] == "play"]
    assert saved["status"] == "failed" and (saved["proposals"], saved["proposals_answered"]) == (2, 0) and len(plays) == 4
    assert reflection.log[-1] == ("reflect", "error") and saved["state"]["unanswered_in_a_row"] == 2
    write_json(tmp_path / "loop" / "run.json", {"mode": g.MODE, "training_fold": 1, "adapter": None,
                                                "gepa": {"card": "verifier", "agent": "causal", "training_variant": "as_released",
                                                         "training_fold": 1, FOLD_CONVENTION_KEY: FOLD_CONVENTION}})
    with pytest.raises(SystemExit, match="is no finished online loop"):
        g.main(["export", str(tmp_path / "loop"), "--training-root", str(tmp_path / "training")])

    def down(*args):
        raise g.LoopFailed("the reflection model answered none of the last 2 proposals")

    monkeypatch.setenv("API_KEY", "stand-in-key")
    monkeypatch.setattr(g.run_loop, "run_online", down)
    with pytest.raises(SystemExit) as stopped:
        g.main([*LOOP_ARGS, "--config", "static", "--out", str(tmp_path / "runs")])
    # a SystemExit with a message ends the command with status 1
    assert isinstance(stopped.value.code, str) and stopped.value.code.startswith("FAILED causal f1 as_released") and "--resume" in stopped.value.code


def test_a_refused_text_is_an_answer_and_one_unanswered_proposal_does_not_fail_the_loop(tmp_path):
    """An error of the endpoint on both attempts, then a reply without a block on both attempts, a refused text and so an answer,
    then errors again: the count of proposals in a row without an answer never reaches two, and the loop finishes."""
    reflection = FailingReflection(raises=(1, 2, 5, 6), reply=lambda current: "no block here")
    _, record, _, _ = _loop(tmp_path, reflection=reflection, stream=(1, 2, 3, 4, 5, 6, 7, 8), max_episodes=20)
    assert record["status"] == "finished" and (record["proposals"], record["proposals_answered"]) == (3, 1)
    assert [c["status"] for c in record["candidates"]] == ["start", "refused", "refused", "refused"]


class PolicyReflection(FakeReflection):
    """A reflection model behind the one retry policy of the tree (CausalDiagnosis.utils.model_request_retry.post_with_retries), as the chat backend sends:
    each attempt takes the next of failures, an exception that fails it, and answers as FakeReflection once they are used up."""

    def __init__(self, failures):
        super().__init__()
        self.failures, self.sent = list(failures), 0

    def generate_with_usage(self, prompt, temperature=0.0):
        from CausalDiagnosis.utils import model_request_retry as retry

        def send(timeout):
            self.sent += 1
            if self.failures:
                raise self.failures.pop(0)
            return FakeReflection.generate_with_usage(self, prompt, temperature)

        return retry.post_with_retries(send)


class Unavailable(Exception):
    """An attempt that the service answers with 503, as the SDK raises it."""

    status_code = 503


def test_the_reflection_requests_name_the_probe_of_their_endpoint_so_a_dead_endpoint_ends_the_program(tmp_path, monkeypatch):
    """The reflection requests name the health probe of the reflection endpoint (judge.endpoint_probe). On an endpoint that no
    longer answers, the first attempt of the first proposal ends the program with ServiceDown and the loop keeps the state it saved
    after the round, status running, for --resume, where it once played on until the registered number of proposals in a row had
    gone unanswered. The control: a first attempt that the service answers with 503 while the probe answers is sent again, and
    every proposal is answered at its first attempt."""
    from CausalDiagnosis.experiments.episode_runners import attempt_error_classes as failures
    import httpx
    from CausalDiagnosis.utils import model_request_retry as retry

    monkeypatch.setattr(retry.time, "sleep", lambda seconds: None)
    health = {"status": None}
    monkeypatch.setattr(failures, "endpoint_health", lambda base_url, key, timeout_s: {"ok": health["status"] is not None, **health})

    def loop(folder, reflection):
        return g.OnlineLoop(FakePlayer(folder, better, reflection.log, False), reflection, folder, "verifier", (1, 2, 3, 4, 5, 6), block=2,
                            max_episodes=10, reflection_model=reflection.model_id, reflection_endpoint="http://127.0.0.1:1/v1",
                            horizon=HORIZON, **LOOP_OF)

    dead = PolicyReflection([httpx.ConnectError("refused")] * 8)
    with pytest.raises(retry.ServiceDown):
        loop(tmp_path / "dead", dead).run()
    saved = read_json(tmp_path / "dead" / g.RESULT)
    assert saved["status"] == "running" and saved["proposals"] == 0 and dead.sent == 1 and len(dead.log) == 2
    assert all(entry[0] == "play" for entry in dead.log)
    health["status"] = 200
    live = PolicyReflection([Unavailable()])
    record = loop(tmp_path / "live", live).run()
    assert record["status"] == "finished" and record["reflection_calls"] == record["proposals"] == live.sent - 1


def test_the_export_refuses_a_loop_of_which_no_proposal_was_answered(tmp_path):
    """A finished loop whose one proposal the reflection model never answered keeps the starting block, which no variant plays as
    a block of GEPA."""
    loop = _finished_loop(tmp_path, proposals=(1, 0))
    with pytest.raises(SystemExit, match="answered none of the 1 proposals"):
        g.main(["export", str(loop), "--training-root", str(tmp_path / "training")])


class Allocation(Exception):
    """The end of an allocation, which stops a job in the middle of a round."""


class StoppingPlayer(FakePlayer):
    """The fake player that stops the job at call stop_at, after it wrote the folder of that episode."""
    stop_at = None

    def __call__(self, candidate, seed, call):
        row, ep = super().__call__(candidate, seed, call)
        if call == self.stop_at:
            raise Allocation(f"stopped at call {call}")
        return row, ep


def test_a_resumed_loop_ends_as_the_uninterrupted_loop(tmp_path):
    """A loop stopped at its fifth episode, in the middle of its second round, and taken up from its result file plays the same
    rounds, candidates, final candidate and seeds as a loop that was never stopped. The episodes of the round that was cut short
    lie in interrupted/1 of the run folder, the record names them, and their seeds play again."""
    _, whole, _, _ = _loop(tmp_path / "whole")

    class StopsAtFive(StoppingPlayer):
        stop_at = 5

    with pytest.raises(Allocation):
        _loop(tmp_path / "cut", player=StopsAtFive)
    reflection = FakeReflection()
    resumed = g.OnlineLoop(FakePlayer(tmp_path / "cut", better, reflection.log), reflection, tmp_path / "cut", "verifier", (1, 2, 3, 4, 5, 6),
                           block=2, max_episodes=10, reflection_model=reflection.model_id, horizon=HORIZON, **LOOP_OF)
    resumed.restore(read_json(tmp_path / "cut" / g.RESULT))
    record = resumed.run()
    for key in ("rounds", "candidates", "final_candidate", "start_candidate", "rollouts_to_best", "episodes_played", "stream_seeds"):
        assert record[key] == whole[key], key
    assert [e["seed"] for e in record["episodes"]] == [e["seed"] for e in whole["episodes"]]
    cut = ["call00003_seed000003", "call00004_seed000003", "call00005_seed000004"]
    assert [(i["after_round"], i["episodes"], i["folder"]) for i in record["interrupted"]] == [(1, cut, "interrupted/1")]
    assert sorted(p.name for p in (tmp_path / "cut" / "interrupted" / "1").iterdir()) == cut


def test_resume_takes_only_an_unfinished_loop_of_the_same_identity(tmp_path):
    """loop_identity reads which loop a folder holds from run.json: another fold, another card and another code identity are other
    loops. --resume takes up an unfinished loop of the same identity alone, never a finished one, and two unfinished loops of one
    identity stop the command."""
    meta = {"config_name": "deterioration", "agent": "causal", "model": "m", "training_variant": "as_released", "training_fold": 1,
            "gepa": {"card": "verifier", "block": 3}, "code": {"identity": "abc"}}
    identity = g.as_stored(g.loop_identity(meta))
    others = [g.as_stored(g.loop_identity(other)) for other in ({**meta, "training_fold": 2}, {**meta, "gepa": {**meta["gepa"], "card": "score_only"}},
                                                                  {**meta, "code": {"identity": "def"}})]
    assert all(other != identity for other in others)
    out = tmp_path / "runs"
    for name, status, of in (("a_other_fold", "running", others[0]), ("b_other_card", "failed", others[1]),
                             ("c_other_code", "running", others[2]), ("d_finished", "finished", identity), ("e_cut", "running", identity)):
        write_json(out / name / g.RESULT, {"status": status, "identity": of})
    assert g.folder_to_resume(out, identity) == out / "e_cut" and g.folder_to_resume(tmp_path / "none", identity) is None
    # a requeued job starts every loop with --resume: a finished loop alone plays nothing again, beside an unfinished one it waits
    assert g.already_finished(out, identity) is None
    done = tmp_path / "done"
    write_json(done / "d_finished" / g.RESULT, {"status": "finished", "identity": identity})
    assert g.already_finished(done, identity) == done / "d_finished" and g.already_finished(done, others[0]) is None
    write_json(out / "f_failed" / g.RESULT, {"status": "failed", "identity": identity})
    with pytest.raises(SystemExit, match="found 2 unfinished loops"):
        g.folder_to_resume(out, identity)


def test_a_loop_resumes_after_its_serving_job_moved_the_model_and_the_reflection_model_to_new_addresses(tmp_path):
    """A requeued loop job reaches the served model and the reflection model at new URLs, so the identity of a loop holds their
    names and not their endpoints: --resume finds the unfinished loop of the first job, and another model is still another loop."""
    meta = {"config_name": "deterioration", "agent": "causal", "model": "granite", "endpoint": "http://10.0.0.1:31000/v1",
            "gepa": {"reflection_model": "judge", "reflection_endpoint": "http://10.0.0.2:32000/v1", "card": "verifier"}}
    moved = {**meta, "endpoint": "http://10.0.0.9:39000/v1", "gepa": {**meta["gepa"], "reflection_endpoint": "http://10.0.0.8:38000/v1"}}
    out = tmp_path / "out"
    write_json(out / "loop_a" / g.RESULT, {"status": "running", "identity": g.as_stored(g.loop_identity(meta))})
    assert g.folder_to_resume(out, g.as_stored(g.loop_identity(moved))) == out / "loop_a"
    for other in ({**moved, "model": "another"}, {**moved, "gepa": {**moved["gepa"], "reflection_model": "another"}}):
        assert g.folder_to_resume(out, g.as_stored(g.loop_identity(other))) is None


def test_the_attempts_of_a_proposal_are_the_registered_number_that_the_loop_reads(tmp_path):
    """The attempts of one proposal are proposal_attempts of the gepa block of the asset's rft.json, two on the rig, which
    training.numbers reads and run_online hands the loop, and no literal of the loop: under three attempts a proposal without a
    block is asked three times, and a proposal whose first two attempts end on an error of the endpoint is answered by its third,
    where two attempts leave it unanswered."""
    assert CONFIG["gepa"]["proposal_attempts"] == 2 and g.run_loop.numbers(DEFAULT_ASSET).proposal_attempts == 2
    for attempts in (2, 3):
        refused = FakeReflection(reply=lambda current: "no block here")
        folder = tmp_path / f"refused_{attempts}"
        record = g.OnlineLoop(FakePlayer(folder, better, refused.log), refused, folder, "verifier", (1, 2, 3, 4, 5, 6), block=1,
                              max_episodes=4, reflection_model=refused.model_id, horizon=HORIZON,
                              **{**LOOP_OF, "proposal_attempts": attempts}).run()
        assert [c["status"] for c in record["candidates"]] == ["start", "refused", "refused"] and len(refused.prompts) == 2 * attempts
        assert sorted(p.name for p in (folder / "reflection").iterdir()) == [
            f"{child}_attempt{n}.json" for child in ("c001", "c002") for n in range(1, attempts + 1)]
        down = FailingReflection(raises=(1, 2), reply=lambda current: "no block here")
        folder = tmp_path / f"down_{attempts}"
        record = g.OnlineLoop(FakePlayer(folder, better, down.log), down, folder, "verifier", (1, 2, 3, 4, 5, 6), block=1,
                              max_episodes=4, reflection_model=down.model_id, horizon=HORIZON,
                              **{**LOOP_OF, "proposal_attempts": attempts}).run()
        assert (record["proposals"], record["proposals_answered"]) == (2, 1 if attempts == 2 else 2)


def test_the_count_of_attempts_stands_in_the_record_and_a_resume_under_another_count_takes_up_no_loop(tmp_path, monkeypatch):
    """The attempts of a proposal change what a loop does, so they stand beside unanswered_proposals in run.json's gepa block, in
    the settings of the result and so in the identity that a resume compares: an unfinished loop saved under the registered count
    is taken up under that count, and once rft.json registers another count it is not, so no loop goes on under a count it did not
    start with."""
    a = g.build_parser().parse_args(LOOP_ARGS)
    g.with_registered_numbers(a, DEFAULT_ASSET)
    gepa = g.gepa_meta(a, DEFAULT_ASSET)
    assert gepa["proposal_attempts"] == CONFIG["gepa"]["proposal_attempts"] and gepa["unanswered_proposals"] == CONFIG["gepa"]["unanswered_proposals"]
    reflection = FakeReflection()
    loop = g.OnlineLoop(FakePlayer(tmp_path / "loop", better, reflection.log), reflection, tmp_path / "loop", "verifier", (1, 2), block=1,
                        max_episodes=2, reflection_model=reflection.model_id, horizon=HORIZON, **{**LOOP_OF, "proposal_attempts": 3})
    assert loop.record("running")["settings"]["proposal_attempts"] == 3
    meta = {"config_name": "static", "agent": "causal", "model": "m", "gepa": gepa, "code": {"identity": "abc"}}
    out = tmp_path / "out"
    write_json(out / "loop_a" / g.RESULT, {"status": "running", "identity": g.as_stored(g.loop_identity(meta))})
    assert g.folder_to_resume(out, g.as_stored(g.loop_identity(meta))) == out / "loop_a"
    registered = g.run_loop.numbers(DEFAULT_ASSET)
    monkeypatch.setattr(g.run_loop, "numbers", lambda asset_id: registered._replace(proposal_attempts=registered.proposal_attempts + 1))
    other = {**meta, "gepa": g.gepa_meta(a, DEFAULT_ASSET)}
    assert other["gepa"]["proposal_attempts"] == registered.proposal_attempts + 1
    assert g.folder_to_resume(out, g.as_stored(g.loop_identity(other))) is None
