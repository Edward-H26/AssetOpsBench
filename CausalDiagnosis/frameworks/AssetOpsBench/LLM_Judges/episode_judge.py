"""The second scorer of an episode: the AssetOpsBench judge on the trajectory and the closing answer of the runner
(CausalDiagnosis/docs/experiments.md).

usage, from the fork root:
  python -m CausalDiagnosis.frameworks.AssetOpsBench.LLM_Judges.episode_judge --model MODEL [--base-url URL] [--protocol single|paper] [--agent-model ID]
      EPISODE_DIR [EPISODE_DIR ...]

The judge is the rubric scorer of src/evaluation/scorers/llm_judge.py, run through the fork's own Evaluator rather than a
copy, on the input of the judge of AssetOpsBench: the scenario's question is the episode's task text, its characteristic
answer is composed from the hidden oracle, the trajectory is trajectory.json, the list of the tool calls that the runner of
the agent keeps, one shape for every agent, and the final response is the closing answer of the runner, answer.txt (the
registered judge.record). An episode whose runner left no trajectory.json is not judged. The judge reads the whole trajectory
and the whole answer. Only a prompt that would pass the context of the served judge, less the reply tokens of the backend,
counted by the served model's own tokenizer and chat template (fitted), is cut, by the least that fits: records leave from
the middle first, a marker record counting each run of them, and never a record that the tool contracts class as a write; then
the texts of the kept records are headed to one length; then the answer. The record names what was cut. The judge model is
reached through an OpenAI compatible endpoint with the key in API_KEY, as the drivers reach the agent models.

Two protocols. "paper", the default of every judged run, makes five calls at
temperature 0.3 and reports their average, as AssetOpsBench section 5.2 does; the binary pass that the agreement study
compares with the recorded pass is the majority of the five planned calls, a failed call counted false (summary_of), the one
rule that the judge_pass cell, the A1 table, the leaderboard, the scoring pairs and the agents table share. "single" makes one call at
temperature 0, a development option that the tables of the judge do not read. The result is written to hidden/judge.json beside the
verdict, under the folder no agent facing tool reads, since the reference answer names the hidden combination.
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
from CausalDiagnosis.utils.paths import ROOT
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET  # noqa: E402
from evaluation.evaluator import Evaluator  # noqa: E402
from evaluation.scorers import llm_judge  # noqa: E402
from llm import LLMBackend  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.LLM_Judges import full_trajectory_scorer  # noqa: E402
from CausalDiagnosis.utils.llm_backends import NO_THINKING, endpoint_backend  # noqa: E402
from llm.routers import resolve_model  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import tool_contract as contract  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import aob_tool_contract as layer_contract  # noqa: E402
from CausalDiagnosis.utils.atomic_writes_and_locks import write_json_atomically  # noqa: E402
from CausalDiagnosis.evaluation.metrics.judge_records import JUDGE_NAME, reviewed, summary_of  # noqa: E402,F401  the record format of the core

TRAJECTORY_FILE = "trajectory.json"  # the runner's list of tool calls, the trajectory the judge reads
ANSWER_FILE = "answer.txt"  # the closing answer of the runner, and the one answer of the static agent (ablation A3)
TEXT_KEYS = ("tool_args", "response")  # the argument and the response text of a record of trajectory.json
ROUTER_PREFIXES = ("tokenrouter/", "litellm_proxy/")  # what a router puts before a served model name
JUDGE_FORMAT = 3  # 3: the record names the cut to the served context (cut) in place of the registered pair
PROTOCOLS = {"single": {"calls": 1, "temperature": 0.0}, "paper": {"calls": 5, "temperature": 0.3}}
ROUTER_BASE_URL = "https://api.tokenrouter.com/v1"
SCENARIO_TYPE = "hydraulic_rig"
_JUDGE_ERRORS = ("judge backend error", "judge returned unparseable JSON", "scenario has neither")
# The judge model is served with a reasoning parser. A review that thinks past the backend's 2048 token cap comes back as
# cut off thinking without the JSON, which the scorer counts as a failed call, so the judge answers without thinking.
JUDGE_REQUEST = NO_THINKING


class FixedTemperature(LLMBackend):
    """The judge scorer calls generate(prompt) without a temperature, so the protocol's temperature is fixed here. It also counts
    the replies that hold thinking, a think block in the reply text or the reasoning field of the reply (holds_thinking), which
    JUDGE_REQUEST must keep at zero."""

    def __init__(self, inner, temperature):
        self.inner = inner
        self.temperature = temperature
        self.thinking_replies = 0
        self.service_down = None  # the ServiceDown of a call, which the scorer would keep as a failed review (judge_once)

    def generate(self, prompt, temperature=0.0):
        from CausalDiagnosis.utils.model_request_retry import ServiceDown  # here and not at the top: the agent package loads its runners on import

        try:
            reply = self.inner.generate_with_usage(prompt, self.temperature)
        except ServiceDown as down:
            self.service_down = down
            raise
        self.thinking_replies += holds_thinking(reply.text, getattr(reply, "reasoning", None))
        return reply.text

    @property
    def model_id(self):
        return self.inner.model_id


def holds_thinking(text, reasoning=None):
    """Whether a reply holds thinking: a think block in its text, opened or closed, or a reasoning field, which a server with a
    reasoning parser fills with the thinking it moves out of the text and which a backend's reply carries as reasoning."""
    return bool(reasoning) or "<think>" in (text or "") or "</think>" in (text or "")


@lru_cache(maxsize=None)
def code_of_judge(root=ROOT):
    """(the commit of the clone, the SHA-256 of the code list that holds this file) of the tree at root, read once per process, so
    that a judge record names the code that wrote it. The commit is None outside a repository of its own."""
    from CausalDiagnosis.experiments.run_verification import run_ledger as ledger  # here and not at the top, as registered_truncation imports registered

    files = ledger.code_identity.tree_files(root)
    own = Path(__file__).resolve().relative_to(Path(root).resolve()).as_posix()
    name = next(name for name, listed in files.items() if own in listed)
    return ledger.code_identity.git_state([], root)[0], ledger.canonical_sha256(sorted(files[name].items()))


def reply_tokens():
    """The registered reply room of the judge, judge.reply_tokens: a value of the scoring, the same for every episode and every
    order of judging, which the input cut leaves free and every judge request sends as max_tokens."""
    from CausalDiagnosis.experiments.table_definitions import registered_analysis as registered  # here and not at the top, so that the driver that imports this module loads no file of the analysis list

    return registered.value("judge.reply_tokens")


def router_backend(model, base_url=ROUTER_BASE_URL):
    """The fork's OpenAI compatible backend pointed at the endpoint, with the key read from API_KEY. The reply room goes in the
    request body itself, since the backend of the fork sends no max_tokens of its own on the chat route."""
    return endpoint_backend(model, base_url, extra_body={**JUDGE_REQUEST, "max_tokens": reply_tokens()})


def reference_answer(oracle, profile=None):
    """The characteristic answer of the judge, composed from the oracle and the episode's asset profile (the default
    asset's when None): the level and the failure mode code of every component, the faulty set, one work order per faulty
    component closed inside its window, and the restored asset."""
    profile = profile or get_profile(DEFAULT_ASSET, engine=False)
    components, healthy = profile.components, profile.healthy
    diagnosis = oracle["diagnosis"]
    levels = dict(zip(components, diagnosis)) if isinstance(diagnosis, list) else {c: diagnosis[c] for c in components}
    parts = []
    for component in components:
        level = levels[component]
        state = "healthy" if level == healthy[component] else "faulty"
        parts.append(f"the {component} at level {level} ({state}, code {profile.level_code(component, level)})")
    faulty = [c for c in components if levels[c] != healthy[c]]
    orders = [f"one work order on the {o['component']} closed by cycle {o['window_end']}" for o in oracle["work_orders"]]
    lines = [
        "The diagnosis names " + ", ".join(parts) + ".",
        ("The faulty components are " + " and ".join(faulty) + "." if faulty else "No component is faulty."),
        ("The repairs are " + " and ".join(orders) + ", and no work order on a healthy component." if orders
         else "No work order is needed."),
        f"The {profile.text['short_noun']} is restored to the healthy level of every component before the episode is finished, within the budget "
        "and the horizon,",
        "and the answer states the diagnosis, the repairs made and the restored condition without claiming any repair the log does not show.",
    ]
    return " ".join(lines)


class NotJudged(ValueError):
    """An episode the judge does not judge, named with the reason."""


@lru_cache(maxsize=1)
def write_calls():
    """The (server, tool) pairs that the tool contracts class as writes, on the surface of the rig and on the servers of the
    rows of AssetOpsBench, so that the cut keeps every write and holds no tool name of its own."""
    rig = {(entry["server"], name) for name, entry in contract.load_contract()["tools"].items() if entry["class"] == "write"}
    aob = {(server, name) for server, tools in layer_contract.load_contract()["tools"].items() for name, entry in tools.items()
           if entry.get("class") == "write"}
    return frozenset(rig | aob)


def cut(text, limit):
    """A text kept to its head of limit characters with a marker that counts the characters cut, and whether it was cut; a text
    that the marker would not shorten stays whole, so the length of a cut text never falls as limit grows."""
    short = f"{text[:limit]} [{len(text) - limit} characters cut]"
    return (short, True) if len(text) > limit and len(short) < len(text) else (text, False)


def judge_prompt(task_text, reference, answer, trajectory):
    """The prompt of one judge call as the scorer builds it: llm_judge's template over the task text, the reference answer, the
    answer and the trajectory as the Evaluator prints it (json.dumps with indent 2)."""
    return llm_judge._PROMPT_TEMPLATE.format(question=task_text or "", characteristic=reference, answer=answer,
                                             trajectory=json.dumps(trajectory, indent=2, default=str))


class Budget:
    """The tokens a judge prompt may take at a served judge: its context less the reply tokens of the backend, since a request
    whose prompt and reply tokens pass the context is the request the server refuses. count(prompt) is the served model's count."""

    def __init__(self, context, reply_tokens, count):
        self.context, self.reply_tokens, self.count = context, reply_tokens, count

    def fits(self, prompt):
        return self.count(prompt) <= self.context - self.reply_tokens


class ContextUnknown(RuntimeError):
    """The endpoint of the judge gave no model listing, so the context it serves under is unknown and nothing is judged."""


def _request(method, base_url, route, body=None, model_reply=True):
    """The JSON answer of one request of the judge to its endpoint, sent under the one retry policy of the model clients
    (CausalDiagnosis.utils.model_request_retry): outside an episode each attempt has the read timeout READ_TIMEOUT_S, an attempt that it ends, no connection or
    a status of 429, 502 or 503 is a failed attempt sent again after the registered wait, and a dead endpoint ends the program
    through the health probe that it names (endpoint_probe). Any other status is raised. model_reply False marks a request that
    is no model reply, the model listing and the tokenize count, as CausalDiagnosis.utils.model_reply_room marks them."""
    import httpx

    from CausalDiagnosis.utils.model_request_retry import post_with_retries  # here and not at the top: the agent package loads its runners on import

    url = base_url.rstrip("/") + route
    headers = {"Authorization": f"Bearer {os.environ.get('API_KEY', '')}"}
    response = post_with_retries(lambda timeout: httpx.request(method, url, headers=headers, json=body, timeout=timeout),
                                 model_reply=model_reply)
    response.raise_for_status()
    return response.json()


def served_context(base_url, model):
    """The context length of the served judge, max_model_len of its entry in the /models answer of the endpoint (SGLang reports
    its --context-length there), or None when the endpoint answers without one, as a router does. An endpoint that gives no
    listing raises ContextUnknown with the reason, since a judge whose context is unknown would send prompts it cannot fit."""
    from CausalDiagnosis.utils.model_request_retry import ServiceDown

    try:
        models = _request("GET", base_url, "/models", model_reply=False).get("data") or []
    except ServiceDown:
        raise
    except Exception as exc:  # noqa: BLE001 - every other failure is raised again with its reason
        raise ContextUnknown(f"the judge endpoint {base_url} gave no model listing ({type(exc).__name__}: {exc}), so its context "
                             "is unknown and no episode is judged") from exc
    entry = next((m for m in models if isinstance(m, dict) and m.get("id") == model), {})
    return int(entry["max_model_len"]) if entry.get("max_model_len") else None


def prompt_tokens(base_url, model, prompt):
    """The tokens of a judge prompt as the served judge counts them: its /tokenize route (SGLang v0.5.19) over the one user message
    with the chat template switches of JUDGE_REQUEST, the input the chat route builds."""
    return int(_request("POST", base_url, "/tokenize", {"model": model, "messages": [{"role": "user", "content": prompt}], **JUDGE_REQUEST},
                        model_reply=False)["count"])


def endpoint_probe(base_url):
    """Name the health probe of the judge's endpoint for every request made inside (failures.endpoint_probe, which yields the
    probe): a failed attempt on an endpoint that no longer answers ends the program with ServiceDown instead of attempting again
    for ever."""
    from CausalDiagnosis.experiments.episode_runners import attempt_error_classes as failures

    return failures.endpoint_probe(base_url)


def served_budget(backend, base_url):
    """The Budget of the judge at base_url, or None when no endpoint is named or the endpoint answers without its context; an
    endpoint that gives no model listing raises ContextUnknown (served_context)."""
    if not base_url:
        return None
    model = served_name(backend)
    context = served_context(base_url, model)
    if context is None:
        return None
    return Budget(context, reply_tokens(), lambda prompt: prompt_tokens(base_url, model, prompt))


def _smallest(lo, hi, ok):
    """The smallest x in [lo, hi] with ok(x), for ok false then true along the range, or None when ok(hi) is false."""
    if not ok(hi):
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if ok(mid) else (mid + 1, hi)
    return lo


def _largest(lo, hi, ok):
    """The largest x in [lo, hi] with ok(x), for ok true then false along the range, or None when ok(lo) is false."""
    if not ok(lo):
        return None
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid - 1)
    return lo


def _without(records, leaving):
    """The records with the indexes of leaving gone, one marker record counting each run of records that left."""
    shown, run = [], 0
    for i, record in enumerate(records):
        if i in leaving:
            run += 1
            continue
        if run:
            shown.append({"records_left_out": run})
            run = 0
        shown.append(record)
    if run:
        shown.append({"records_left_out": run})
    return shown


def _text(value):
    return value if isinstance(value, str) else json.dumps(value, default=str)


def _headed(records, limit):
    """The records with every argument and response text kept to its head of limit characters and a marker, and the count of
    texts cut."""
    out, count = [], 0
    for record in records:
        record = dict(record)
        for key in TEXT_KEYS:
            if record.get(key) is not None:
                short, was_cut = cut(_text(record[key]), limit)
                if was_cut:
                    record[key], count = short, count + 1
        out.append(record)
    return out, count


def fitted(records, answer, prompt_of, budget, writes=None):
    """The trajectory and the answer that the judge reads, and what was cut so that prompt_of(trajectory, answer) fits budget.
    Nothing is cut without a budget or when the whole prompt fits. Otherwise, each step only while the prompt still does not fit and
    each by the least that fits: the records nearest the middle that are no write leave (writes, the (server, tool) pairs of the
    tool contracts by default); then the texts of the kept records are headed to the longest common length; then the answer is
    headed. A prompt that does not fit with all three at their end raises NotJudged."""
    counts = {"records_left_out": 0, "texts_headed": 0, "head_chars": None, "answer_headed": False}
    if budget is None or budget.fits(prompt_of(records, answer)):
        return records, answer, counts
    writes = write_calls() if writes is None else writes
    centre = (len(records) - 1) / 2
    order = sorted((i for i, r in enumerate(records) if (r.get("server"), r.get("tool")) not in writes), key=lambda i: (abs(i - centre), i))
    leaving = _smallest(1, len(order), lambda k: budget.fits(prompt_of(_without(records, set(order[:k])), answer))) if order else None
    counts["records_left_out"] = len(order) if leaving is None else leaving
    shown = _without(records, set(order[:counts["records_left_out"]]))
    if leaving is not None:
        return shown, answer, counts
    longest = max((len(_text(r[key])) for r in shown for key in TEXT_KEYS if r.get(key) is not None), default=0)
    head = _largest(0, longest, lambda n: budget.fits(prompt_of(_headed(shown, n)[0], answer)))
    counts["head_chars"] = 0 if head is None else head
    shown, counts["texts_headed"] = _headed(shown, counts["head_chars"])
    if head is not None:
        return shown, answer, counts
    keep = _largest(0, len(answer), lambda n: budget.fits(prompt_of(shown, cut(answer, n)[0])))
    if keep is None:
        raise NotJudged(f"the judge prompt passes the context {budget.context} less {budget.reply_tokens} reply tokens with every cut made")
    counts["answer_headed"] = keep < len(answer)
    return shown, cut(answer, keep)[0], counts


def served_name(backend):
    """The model as the requests of a judge backend name it, the served name at an endpoint that this project serves: the backend id without the
    prefix of the router that router_backend puts before it. A judge record names its judge by this name, so that the comparison
    with served_name of the judge block of the serving file of record reads one name on both sides."""
    return resolve_model(backend.model_id)


def bare_model(name):
    """A model name without the prefix of a router and without the adapter that SGLang takes after a colon."""
    name = (name or "").strip()
    for prefix in ROUTER_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.split(":", 1)[0].lower()


def same_model(judge_model, agent_model):
    """Whether the judge is the agent model, compared on the bare names, for every agent."""
    return bool(bare_model(judge_model)) and bare_model(judge_model) == bare_model(agent_model)


def judge_once(scenario_id, task_text, reference, answer, trajectory, backend, temperature, agent_model):
    """One judge call through the Evaluator, over a scenario file and a trajectory record written to a temporary folder,
    which is the only input shape the fork's evaluation pipeline reads."""
    scorer_backend = FixedTemperature(backend, temperature)
    full_trajectory_scorer.install(scorer_backend)
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        scenarios = folder / "scenarios.json"
        scenarios.write_text(json.dumps([{"id": scenario_id, "text": task_text or "", "type": SCENARIO_TYPE,
                                          "characteristic_form": reference}]), encoding="utf-8")
        trajectories = folder / "trajectories"
        trajectories.mkdir()
        record = {"run_id": scenario_id, "scenario_id": scenario_id, "runner": SCENARIO_TYPE, "model": agent_model or "unknown",
                  "question": task_text or "", "answer": answer, "trajectory": trajectory}
        (trajectories / f"{scenario_id}.json").write_text(json.dumps(record, default=str), encoding="utf-8")
        report = Evaluator(default_scorer="llm_judge", judge_model=backend.model_id).evaluate(
            trajectories_path=trajectories, scenarios_paths=[scenarios])
    if scorer_backend.service_down is not None:  # the endpoint no longer answers: the judging ends, no call is scored as failed
        raise scorer_backend.service_down
    score = report.results[0].score
    # the scorer reports a backend error, an unparseable reply or a missing reference as a failed review; those are
    # failures of the judge, not of the agent, so they are kept apart from a parsed review that says fail
    error = score.rationale if not score.details or any(score.rationale.startswith(p) for p in _JUDGE_ERRORS) else None
    return {"passed": score.passed, "score": score.score, "rationale": score.rationale, "details": score.details, "error": error,
            "thinking_replies": scorer_backend.thinking_replies}


def judge_episode(episode_dir, backend, protocol, agent_model=None, base_url=None, static=False, serving=None, budget=None):
    """Judge one episode folder and write hidden/judge.json; returns the record. The trajectory is the runner's trajectory.json
    and the final response is its answer.txt, both whole unless the prompt passes budget, the Budget of the served judge at
    base_url when not given (fitted). An episode without trajectory.json raises NotJudged, and a judge that is the agent model
    is refused. In the static mode the answer is the static agent's answer.txt, the judge sees no trajectory, and the record
    says mode static. The record names the cut and what it removed, the SHA-256 of the files it read, serving, the identity of
    the judge endpoint that the caller read, the commit of the clone that wrote the record and the hash of the code list that
    holds this file (code_of_judge), and the count of the judge's replies that hold thinking (thinking_replies). Every request to
    base_url, the model listing, the token counts and the calls of the backend, is made under the health probe of that endpoint
    (endpoint_probe), so a dead endpoint ends the judging with ServiceDown and a live one is asked again."""
    with endpoint_probe(base_url) if base_url else nullcontext():
        return _judged(episode_dir, backend, protocol, agent_model, base_url, static, serving, budget)


def _judged(episode_dir, backend, protocol, agent_model, base_url, static, serving, budget):
    """The record of judge_episode, made under the health probe of its endpoint."""
    episode_dir = Path(episode_dir)
    if same_model(backend.model_id, agent_model):
        raise ValueError(f"self-judging: the judge {backend.model_id} is the agent model {agent_model}")
    released = json.loads((episode_dir / "released.json").read_text(encoding="utf-8"))
    oracle = json.loads((episode_dir / "hidden" / "oracle.json").read_text(encoding="utf-8"))
    settings = PROTOCOLS[protocol]
    # released.json names the asset of its episode at the top level, and a record written before the key is one of the default asset
    profile = get_profile(released.get("asset_id") or DEFAULT_ASSET, engine=False)
    task_text, reference = released.get("task_text"), reference_answer(oracle, profile)
    trajectory_sha256, records = None, []
    if not static:
        path = episode_dir / TRAJECTORY_FILE
        if not path.exists():
            raise NotJudged(f"{episode_dir.name}: the runner left no {TRAJECTORY_FILE}, so the episode is not judged")
        raw = path.read_bytes()
        trajectory_sha256, records = hashlib.sha256(raw).hexdigest(), json.loads(raw)
    answer_path = episode_dir / ANSWER_FILE
    full_answer = answer_path.read_text(encoding="utf-8") if answer_path.exists() else ""
    budget = served_budget(backend, base_url) if budget is None else budget
    prompt_of = lambda trajectory, answer: judge_prompt(task_text, reference, answer, trajectory)  # noqa: E731
    trajectory, answer, counts = fitted(records, full_answer, prompt_of, budget)
    code_commit, judge_code_sha256 = code_of_judge()
    calls = [judge_once(episode_dir.name, task_text, reference, answer, trajectory, backend, settings["temperature"], agent_model)
             for _ in range(settings["calls"])]
    record = {
        "format": JUDGE_FORMAT, "protocol": protocol, "temperature": settings["temperature"], "calls_planned": settings["calls"],
        "judge_model": served_name(backend), "base_url": None, "judge_request": getattr(backend, "extra_body", None),
        "endpoint_identity": serving, "agent_model": agent_model, **({"mode": "static"} if static else {}),
        "code_commit": code_commit, "judge_code_sha256": judge_code_sha256,
        "thinking_replies": sum(call.get("thinking_replies", 0) for call in calls),
        "cut": {"context": budget and budget.context, "reply_tokens": budget and budget.reply_tokens,
                "prompt_tokens": budget and budget.count(prompt_of(records, full_answer)),
                "prompt_tokens_sent": budget and budget.count(prompt_of(trajectory, answer)), **counts},
        "read": {"trajectory_sha256": trajectory_sha256, "answer_sha256": hashlib.sha256(full_answer.encode("utf-8")).hexdigest()},
        "task_text": task_text, "reference_answer": reference, "answer": answer, "calls": calls, **summary_of(calls, settings["calls"]),
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


# short prompts of the thinking check of a judge job, whose replies must hold no thinking under JUDGE_REQUEST
THINKING_PROMPTS = ("Answer with the single word yes.", "Is 7 a prime number? Answer yes or no.", "Name the colour of a clear sky in one word.")


def thinking_replies(base_url, model, prompts=THINKING_PROMPTS):
    """How many replies of the served judge hold thinking under JUDGE_REQUEST, by the test of every judge record (holds_thinking): a
    reasoning field that the server's reasoning parser fills, or a think block in the text, opened or closed. The chat template of
    the model decides whether the switch works, so every judge job checks it on its judge before it judges (CausalDiagnosis/cluster_jobs/3_run/run_judge.sh,
    JUDGE_THINKING). Each request is a judge request, the registered reply room included."""
    count = 0
    for prompt in prompts:
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": reply_tokens(), "temperature": 0.0, **JUDGE_REQUEST}
        message = _request("POST", base_url, "/chat/completions", body)["choices"][0]["message"]
        count += holds_thinking(message.get("content"), message.get("reasoning_content") or message.get("reasoning"))
    return count


def main(argv):
    ap = argparse.ArgumentParser(description="the AssetOpsBench judge on the closing summary of hydraulic rig episodes")
    ap.add_argument("episodes", nargs="*", help="episode folders holding released.json and hidden/oracle.json")
    ap.add_argument("--thinking", action="store_true", help="print JUDGE_THINKING <n>, the replies of the judge that hold thinking")
    ap.add_argument("--model", required=True, help="the judge model at the endpoint")
    ap.add_argument("--base-url", default=ROUTER_BASE_URL, help="OpenAI compatible endpoint (default: TokenRouter)")
    ap.add_argument("--protocol", choices=sorted(PROTOCOLS), default="paper",
                    help="the paper's five calls at temperature 0.3, the default, or single, one call at temperature 0 for development")
    ap.add_argument("--agent-model", default=None, help="the agent's model id (default: the run folder's run.json)")
    a = ap.parse_args(argv)
    if not (a.thinking or a.episodes):
        ap.error("name the episode folders to judge")
    with endpoint_probe(a.base_url):
        if a.thinking:
            print(f"JUDGE_THINKING {thinking_replies(a.base_url, a.model)}", flush=True)
            return
        backend = router_backend(a.model, a.base_url)
        for episode in a.episodes:
            record = judge_episode(episode, backend, a.protocol, a.agent_model or agent_model_of(episode), a.base_url)
            print(f"{episode}: judge pass {record['pass']} ({record['protocol']}, pass rate {record['pass_rate']})", flush=True)


# The judge "local" of the declaration of AssetOpsBench (JUDGES): this module itself, whose router_backend and judge_episode the
# driver calls. A test that replaces one of them on the module reaches those callers too.
JUDGE = sys.modules[__name__]

if __name__ == "__main__":
    main(sys.argv[1:])
