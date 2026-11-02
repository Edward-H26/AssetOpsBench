"""The second scorer of an episode: the AssetOpsBench judge on the closing summary (docs/hydraulic_rig/benchmarks.md).

usage, from the fork root:
  python benchmarks/hydraulic_rig/judge.py --model MODEL [--base-url URL] [--protocol single|paper] [--agent-model ID]
      EPISODE_DIR [EPISODE_DIR ...]

The judge is the rubric scorer of src/evaluation/scorers/llm_judge.py, run through the fork's own Evaluator rather than a
copy: the scenario's question is the episode's task text, its characteristic answer is composed from the hidden oracle, the
trajectory record's answer is the closing summary and its trajectory is the released events, which hold nothing hidden. The
judge model is reached through an OpenAI compatible endpoint with the key in API_KEY, as the drivers reach the agent models.

Two protocols. "single" makes one call at temperature 0, the default of the first runs. "paper" makes five calls at
temperature 0.3 and reports their average, as AssetOpsBench section 5.2 does; the binary pass that the agreement study
compares with the recorded pass is the majority of the five. The result is written to hidden/judge.json beside the
verdict, under the folder no agent facing tool reads, since the reference answer names the hidden combination.
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from envs.hydraulic_rig import knowledge  # noqa: E402
from envs.hydraulic_rig.data import COMPONENTS, HEALTHY  # noqa: E402
from evaluation.evaluator import Evaluator  # noqa: E402
from evaluation.scorers import llm_judge  # noqa: E402
from llm import LLMBackend, make_backend  # noqa: E402
from servers.json_files import write_json_atomically  # noqa: E402

JUDGE_NAME = "judge.json"
JUDGE_FORMAT = 1
PROTOCOLS = {"single": {"calls": 1, "temperature": 0.0}, "paper": {"calls": 5, "temperature": 0.3}}
ROUTER_BASE_URL = "https://api.tokenrouter.com/v1"
SCENARIO_TYPE = "hydraulic_rig"
_JUDGE_ERRORS = ("judge backend error", "judge returned unparseable JSON", "scenario has neither")


class FixedTemperature(LLMBackend):
    """The judge scorer calls generate(prompt) without a temperature, so the protocol's temperature is fixed here."""

    def __init__(self, inner, temperature):
        self.inner = inner
        self.temperature = temperature

    def generate(self, prompt, temperature=0.0):
        return self.inner.generate(prompt, self.temperature)

    @property
    def model_id(self):
        return self.inner.model_id


def router_backend(model, base_url=ROUTER_BASE_URL):
    """The fork's OpenAI compatible backend pointed at the endpoint, with the key read from API_KEY."""
    key = os.environ.get("API_KEY")
    if not key:
        raise SystemExit("API_KEY is not set")
    os.environ["TOKENROUTER_BASE_URL"] = base_url
    os.environ["TOKENROUTER_API_KEY"] = key
    return make_backend(f"tokenrouter/{model}")


def reference_answer(oracle):
    """The characteristic answer of the judge, composed from the oracle: the level and the failure mode code of every
    component, the faulty set, one work order per faulty component closed inside its window, and the restored rig."""
    diagnosis = oracle["diagnosis"]
    levels = dict(zip(COMPONENTS, diagnosis)) if isinstance(diagnosis, list) else {c: diagnosis[c] for c in COMPONENTS}
    parts = []
    for component in COMPONENTS:
        level = levels[component]
        state = "healthy" if level == HEALTHY[component] else "faulty"
        parts.append(f"the {component} at level {level} ({state}, code {knowledge.level_code(component, level)})")
    faulty = [c for c in COMPONENTS if levels[c] != HEALTHY[c]]
    orders = [f"one work order on the {o['component']} closed by cycle {o['window_end']}" for o in oracle["work_orders"]]
    lines = [
        "The diagnosis names " + ", ".join(parts) + ".",
        ("The faulty components are " + " and ".join(faulty) + "." if faulty else "No component is faulty."),
        ("The repairs are " + " and ".join(orders) + ", and no work order on a healthy component." if orders
         else "No work order is needed."),
        "The rig is restored to the healthy level of every component before the episode is finished, within the budget and the horizon,",
        "and the answer states the diagnosis, the repairs made and the restored condition without claiming any repair the log does not show.",
    ]
    return " ".join(lines)


def closing_summary(released):
    finish = next((e for e in released["events"] if e["event"] == "finish"), None)
    return (finish.get("summary") if finish else None) or ""


def judge_once(scenario_id, task_text, reference, answer, events, backend, temperature, agent_model):
    """One judge call through the Evaluator, over a scenario file and a trajectory record written to a temporary folder,
    which is the only input shape the fork's evaluation pipeline reads."""
    llm_judge.install(FixedTemperature(backend, temperature))
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        scenarios = folder / "scenarios.json"
        scenarios.write_text(json.dumps([{"id": scenario_id, "text": task_text or "", "type": SCENARIO_TYPE,
                                          "characteristic_form": reference}]), encoding="utf-8")
        trajectories = folder / "trajectories"
        trajectories.mkdir()
        record = {"run_id": scenario_id, "scenario_id": scenario_id, "runner": SCENARIO_TYPE, "model": agent_model or "unknown",
                  "question": task_text or "", "answer": answer, "trajectory": {"events": events}}
        (trajectories / f"{scenario_id}.json").write_text(json.dumps(record, default=str), encoding="utf-8")
        report = Evaluator(default_scorer="llm_judge", judge_model=backend.model_id).evaluate(
            trajectories_path=trajectories, scenarios_paths=[scenarios])
    score = report.results[0].score
    # the scorer reports a backend error, an unparseable reply or a missing reference as a failed review; those are
    # failures of the judge, not of the agent, so they are kept apart from a parsed review that says fail
    error = score.rationale if not score.details or any(score.rationale.startswith(p) for p in _JUDGE_ERRORS) else None
    return {"passed": score.passed, "score": score.score, "rationale": score.rationale, "details": score.details, "error": error}


def judge_episode(episode_dir, backend, protocol="single", agent_model=None, base_url=None):
    """Judge one episode folder and write hidden/judge.json; returns the record."""
    episode_dir = Path(episode_dir)
    released = json.loads((episode_dir / "released.json").read_text(encoding="utf-8"))
    oracle = json.loads((episode_dir / "hidden" / "oracle.json").read_text(encoding="utf-8"))
    settings = PROTOCOLS[protocol]
    task_text, reference, answer = released.get("task_text"), reference_answer(oracle), closing_summary(released)
    calls = [judge_once(episode_dir.name, task_text, reference, answer, released["events"], backend, settings["temperature"], agent_model)
             for _ in range(settings["calls"])]
    reviewed = [c for c in calls if c["error"] is None]
    pass_rate = sum(1 for c in reviewed if c["passed"]) / len(reviewed) if reviewed else None
    record = {
        "format": JUDGE_FORMAT, "protocol": protocol, "temperature": settings["temperature"], "calls_planned": settings["calls"],
        "judge_model": backend.model_id, "base_url": base_url, "agent_model": agent_model,
        "task_text": task_text, "reference_answer": reference, "answer": answer, "calls": calls,
        "pass_rate": pass_rate, "score_mean": sum(c["score"] for c in reviewed) / len(reviewed) if reviewed else None,
        "pass": None if pass_rate is None else pass_rate > 0.5,  # the majority of the reviewed calls, one under "single"
    }
    (episode_dir / "hidden").mkdir(exist_ok=True)
    write_json_atomically(episode_dir / "hidden" / JUDGE_NAME, record, indent=1)
    return record


def agent_model_of(episode_dir):
    """The agent model of the run folder above the episode, so the Evaluator's rule against self judging applies."""
    run_file = Path(episode_dir).parent / "run.json"
    if run_file.exists():
        return json.loads(run_file.read_text(encoding="utf-8")).get("model")
    return None


def main(argv):
    ap = argparse.ArgumentParser(description="the AssetOpsBench judge on the closing summary of hydraulic rig episodes")
    ap.add_argument("episodes", nargs="+", help="episode folders holding released.json and hidden/oracle.json")
    ap.add_argument("--model", required=True, help="the judge model at the endpoint")
    ap.add_argument("--base-url", default=ROUTER_BASE_URL, help="OpenAI compatible endpoint (default: TokenRouter)")
    ap.add_argument("--protocol", choices=sorted(PROTOCOLS), default="single")
    ap.add_argument("--agent-model", default=None, help="the agent's model id (default: the run folder's run.json)")
    a = ap.parse_args(argv)
    backend = router_backend(a.model, a.base_url)
    for episode in a.episodes:
        record = judge_episode(episode, backend, a.protocol, a.agent_model or agent_model_of(episode), a.base_url)
        print(f"{episode}: judge pass {record['pass']} ({record['protocol']}, pass rate {record['pass_rate']})", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
