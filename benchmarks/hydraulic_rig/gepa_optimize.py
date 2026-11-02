"""GEPA over the planner prompt of AssetOpsBench's Plan-Execute runner (docs/hydraulic_rig/benchmarks.md).

usage, from the fork root with the fork's virtual environment first on PATH:
  python benchmarks/hydraulic_rig/gepa_optimize.py --model NAME --base-url URL [--config static] [--train-seeds 8] [--val-seeds 4]
      [--first-seed 0] [--max-metric-calls 60] [--reflection-minibatch 3] [--no-diagnostics] [--no-grade] [--out benchmarks/hydraulic_rig/runs]

The candidate is the planner prompt of the runner, the module constant _PLAN_PROMPT of src/agent/plan_execute/planner.py,
which Planner.generate_plan reads at call time; replacing that constant for the duration of an episode (planner_prompt
below) is the override, so no copy of the runner exists. A candidate has to keep the placeholders {servers} and
{question} the runner fills; a candidate without them scores zero with that fact as its feedback.

The adapter follows GEPAAdapter of gepa.core.adapter: evaluate plays one episode per metric call through EpisodeRunner of
run_episodes.py (verified, judged when asked, written to the run folder like any episode) and returns the verifier's reward
as the score, zero when the verifier could not run; make_reflective_dataset hands the reflection model the question, the
plan, the closing answer, and as feedback the verdict's checks, the failure class, the end reason, the cost and the
messages of the diagnostics tools in the trajectory, the evaluation trace of design v2 section 5.7. The reflection model is
the same served endpoint as the agent, a limitation of the first runs: a prompt optimizer that reflects with the agent's
own model measures what that model can teach itself. The budget is --max-metric-calls, every episode counted, validation
rollouts included; the comparison with the untuned prompt at the same number of episodes is the GEPA table of aggregate.py.

The gepa package (gepa-ai/gepa) is imported in main() only and is not a dependency of the fork; the cluster setup script
installs it into the cluster venv with uv pip install gepa. The tests exercise the adapter with a fake episode player.
"""
import argparse
import contextlib
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_episodes as drivers  # noqa: E402
from common import CAUSAL_SERVER, HIDDEN, read_json, write_json  # noqa: E402

from agent.plan_execute import planner as planner_module  # noqa: E402

PROMPT_COMPONENT = "planner_prompt"
PLACEHOLDERS = ("{servers}", "{question}")
MISSING_PLACEHOLDERS = ("the prompt must keep the placeholders {servers} and {question}: the runner fills them with the tool table "
                        "of the servers and the question of the episode, and a plan cannot name tools it was not shown")


class EvaluationBatch:
    """The result shape GEPA reads back from evaluate; gepa's own dataclass of the same name is used when it imports."""

    def __init__(self, outputs, scores, trajectories=None):
        self.outputs, self.scores, self.trajectories = outputs, scores, trajectories


def evaluation_batch(outputs, scores, trajectories):
    try:
        from gepa.core.adapter import EvaluationBatch as GepaBatch
    except ImportError:
        GepaBatch = EvaluationBatch
    return GepaBatch(outputs=outputs, scores=scores, trajectories=trajectories)


def seed_candidate():
    return {PROMPT_COMPONENT: planner_module._PLAN_PROMPT}


@contextlib.contextmanager
def planner_prompt(text):
    """The override of the planner prompt for one episode, restored afterward whatever happens in between."""
    previous = planner_module._PLAN_PROMPT
    planner_module._PLAN_PROMPT = text
    try:
        yield
    finally:
        planner_module._PLAN_PROMPT = previous


def diagnostics_messages(trajectory):
    """The message field of every answer of the diagnostics server in a trajectory, the evaluation trace GEPA reads."""
    messages = []
    for step in trajectory or []:
        if step.get("server") != CAUSAL_SERVER or step.get("error"):
            continue
        try:
            answer = json.loads(step.get("response") or "")
        except (TypeError, ValueError):
            continue
        if isinstance(answer, dict) and answer.get("message"):
            messages.append(f"{step.get('tool')}: {answer['message']}")
    return messages


def feedback_text(ep_dir, row):
    """What the reflection model learns from one episode: every check of the verdict with its outcome, the failure class,
    the end reason, the cost against the oracle's, the reward, and the diagnostics tools' messages in order."""
    verdict = read_json(Path(ep_dir) / HIDDEN / "verdict.json", {})
    lines = []
    if row.get("error"):
        lines.append(f"error: {row['error']}")
    for name, check in (verdict.get("checks") or {}).items():
        state = "ok" if isinstance(check, dict) and check.get("ok") else "failed"
        detail = {k: v for k, v in check.items() if k != "ok"} if isinstance(check, dict) else check
        lines.append(f"check {name}: {state}" + (f" {json.dumps(detail, default=str)[:400]}" if detail else ""))
    lines.append(f"failure class: {verdict.get('failure_class', 'unknown')}; end reason: {row.get('end_reason')}; "
                 f"cost {row.get('cost')} against the oracle's {row.get('cost_oracle')}; reward {row.get('reward')}")
    messages = diagnostics_messages(read_json(Path(ep_dir) / "trajectory.json", []))
    if messages:
        lines.append("diagnostics server messages:")
        lines += [f"- {m}" for m in messages]
    return "\n".join(lines)


class PlannerPromptAdapter:
    """GEPAAdapter over the planner prompt: one episode per metric call through an EpisodeRunner, the reward as the
    score, the verdict as the feedback. `play(seed, prompt_text) -> (row, trace)` is injectable for the tests."""

    def __init__(self, runner, run_dir, meta, play=None):
        self.runner, self.run_dir, self.meta = runner, Path(run_dir), meta
        self.play = play or self._play_episode
        self.metric_calls = 0
        self.rows = []

    def _play_episode(self, seed, prompt_text):
        name = f"call{self.metric_calls:05d}_seed{seed:06d}"
        ep_dir = self.run_dir / name
        with planner_prompt(prompt_text):
            row = self.runner.play(ep_dir, name, seed, None, self.runner.config, self.meta)
        trace = {"question": _text(ep_dir / "question.txt"), "plan": _text(ep_dir / "plan.txt"), "answer": _text(ep_dir / "answer.txt"),
                 "feedback": feedback_text(ep_dir, row), "episode": name}
        return row, trace

    def evaluate(self, batch, candidate, capture_traces=False):
        text = candidate[PROMPT_COMPONENT]
        outputs, scores, trajectories = [], [], []
        for item in batch:
            self.metric_calls += 1
            if not all(p in text for p in PLACEHOLDERS):
                outputs.append("")
                scores.append(0.0)
                trajectories.append({"question": "", "plan": "", "answer": "", "feedback": MISSING_PLACEHOLDERS, "episode": None})
                continue
            row, trace = self.play(item["seed"], text)
            row = dict(row, gepa_call=self.metric_calls)
            self.rows.append(row)
            reward = row.get("reward")
            scores.append(float(reward) if reward not in ("", None) else 0.0)
            outputs.append(trace["answer"])
            trajectories.append(trace)
        return evaluation_batch(outputs, scores, trajectories if capture_traces else None)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        items = [{"Inputs": {"question": t["question"]}, "Generated Outputs": {"plan": t["plan"], "answer": t["answer"]}, "Feedback": t["feedback"]}
                 for t in eval_batch.trajectories or []]
        return {component: items for component in components_to_update}


def _text(path):
    return Path(path).read_text(encoding="utf-8") if Path(path).exists() else ""


def result_record(result, adapter, a):
    """gepa's result as a JSON record: the candidates, the validation scores of the seed and the best candidate, and the
    budget spent, read defensively since the field names belong to the gepa version the cluster installs."""
    candidates = getattr(result, "candidates", None) or []
    best_idx = getattr(result, "best_idx", None)
    subscores = getattr(result, "val_subscores", None) or []
    aggregate = getattr(result, "val_aggregate_scores", None) or []

    def val_scores(index):
        if index is None or index >= len(candidates):
            return []
        if index < len(subscores) and subscores[index]:
            return list(subscores[index].values()) if isinstance(subscores[index], dict) else list(subscores[index])
        return [aggregate[index]] if index < len(aggregate) else []

    return {
        "seed_candidate": candidates[0][PROMPT_COMPONENT] if candidates else None,
        "best_candidate": (getattr(result, "best_candidate", None) or {}).get(PROMPT_COMPONENT),
        "best_idx": best_idx, "candidates": [c.get(PROMPT_COMPONENT) for c in candidates], "val_aggregate_scores": list(aggregate),
        "seed_val_scores": val_scores(0), "best_val_scores": val_scores(best_idx),
        "total_metric_calls": getattr(result, "total_metric_calls", None), "episodes_played": adapter.metric_calls,
        "max_metric_calls": a.max_metric_calls, "train_seeds": a.train_seeds, "val_seeds": a.val_seeds, "first_seed": a.first_seed,
        "reflection_minibatch": a.reflection_minibatch, "reflection_model": a.model, "reflection_endpoint": a.base_url,
    }


def main(argv):
    ap = argparse.ArgumentParser(description="GEPA over the planner prompt of the Plan-Execute runner on hydraulic rig episodes")
    ap.add_argument("--config", default="static")
    ap.add_argument("--model", required=True, help="the agent's model name at the endpoint, also the reflection model")
    ap.add_argument("--base-url", default=drivers.ROUTER_BASE_URL)
    ap.add_argument("--train-seeds", type=int, default=8)
    ap.add_argument("--val-seeds", type=int, default=4)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--max-metric-calls", type=int, default=60, help="the budget: episodes played, validation rollouts included")
    ap.add_argument("--reflection-minibatch", type=int, default=3)
    ap.add_argument("--no-diagnostics", action="store_true")
    ap.add_argument("--no-grade", action="store_true")
    ap.add_argument("--judge", choices=("none", "local"), default="none")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-base-url", default=None)
    ap.add_argument("--judge-protocol", choices=("single", "paper"), default="single")
    ap.add_argument("--facts", default=None)
    ap.add_argument("--out", default="benchmarks/hydraulic_rig/runs")
    a = ap.parse_args(argv)
    import gepa

    settings = drivers.build_parser().parse_args(["--config", a.config, "--agent", "plan_execute", "--model", a.model, "--base-url", a.base_url,
                                                  "--judge", a.judge, "--judge-protocol", a.judge_protocol]
                                                 + (["--no-diagnostics"] if a.no_diagnostics else []) + (["--no-grade"] if a.no_grade else [])
                                                 + (["--judge-model", a.judge_model] if a.judge_model else [])
                                                 + (["--judge-base-url", a.judge_base_url] if a.judge_base_url else []))
    facts_path, cycles, blocks = drivers.load_world(a.facts)
    config_name, config, price_table = drivers.configure(settings)
    runner, judge_options = drivers.build_runner(settings, config_name, config, price_table, cycles, blocks, facts_path)
    runner.label = drivers.run_label(config_name, f"gepa_{a.model}")
    run_dir = drivers.run_folder(a.out, runner.label)
    settings.seeds_list = range(a.first_seed, a.first_seed + a.train_seeds + a.val_seeds)
    meta = drivers.run_meta(settings, runner, config, price_table, [], judge_options)
    meta.update({"mode": "gepa", "gepa": {"train_seeds": a.train_seeds, "val_seeds": a.val_seeds, "first_seed": a.first_seed,
                                          "max_metric_calls": a.max_metric_calls, "reflection_minibatch": a.reflection_minibatch,
                                          "component": PROMPT_COMPONENT, "reflection_model": a.model}})
    adapter = PlannerPromptAdapter(runner, run_dir, meta)
    reflection = drivers.router_backend(a.model, a.base_url)
    trainset = [{"seed": s} for s in range(a.first_seed, a.first_seed + a.train_seeds)]
    valset = [{"seed": s} for s in range(a.first_seed + a.train_seeds, a.first_seed + a.train_seeds + a.val_seeds)]
    result = gepa.optimize(seed_candidate=seed_candidate(), trainset=trainset, valset=valset, adapter=adapter,
                           reflection_lm=lambda prompt: reflection.generate(prompt), max_metric_calls=a.max_metric_calls,
                           reflection_minibatch_size=a.reflection_minibatch, seed=0, run_dir=str(run_dir / "gepa_state"))
    record = result_record(result, adapter, a)
    write_json(run_dir / "gepa_result.json", record)
    if record["best_candidate"]:
        (run_dir / "best_prompt.txt").write_text(record["best_candidate"], encoding="utf-8")
    meta["episodes_planned"] = adapter.metric_calls
    drivers.write_run(run_dir, meta, adapter.rows)
    print(f"wrote {run_dir}: {adapter.metric_calls} episodes played, {len(record['candidates'])} candidates, ended {datetime.now():%H:%M:%S}")
    return run_dir


if __name__ == "__main__":
    main(sys.argv[1:])
