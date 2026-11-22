"""The episode explorer (explorer/ beside step_through_page.py): the server on 127.0.0.1, the data builder and the exported page, on
scripted episodes played into pytest's temporary folder: bench_1 on a priced row, on the base row, on the base row of
AssetOpsBench's servers and on the token clock with a made-up count of generated tokens, the CausalDiagnosis agent on bench_1
with a planner without a model, demo_1 without the CausalDiagnosis MCP, and the rig when its data file is present. The tests assert relations between the pages and the run
folders, never a recorded value: no hidden value, launcher or lock file reaches the public side, the hidden side needs the
flag, no path leaves the runs folder, a missing file stays missing, the follow marks agree with the measures of
mcp_metrics.py, the frontend names no asset, the exported page needs no network, and no file of a run folder changes."""
import functools
import hashlib
import http.client
import importlib.util
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path
from CausalDiagnosis.utils.paths import ROOT, SRC

if importlib.util.find_spec("mcp") is None:
    pytest.skip("needs the mcp package", allow_module_level=True)

from llm.base import LLMResult  # noqa: E402

from CausalDiagnosis.agents.Causal_Diagnosis_Agent.prompts import ORIENTATION_HEADING, PLANNING_HEADING, SUMMARY_HEADING  # noqa: E402
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tests import test_causal_diagnosis_agent as causal_tests  # noqa: E402
from CausalDiagnosis.agents.utils import idle_hook as idle  # noqa: E402
from CausalDiagnosis.datasets.dataset_loader import dataset_packages as ports  # noqa: E402
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile  # noqa: E402
from CausalDiagnosis.datasets.tests.conftest import PORTS as LAYER_PORTS, mount_port, write_port_records, write_recordings  # noqa: E402
from CausalDiagnosis.episode_viewer.explorer import cli, data, server  # noqa: E402
from CausalDiagnosis.evaluation.metrics import mcp_metrics  # noqa: E402
from CausalDiagnosis.experiments.episode_runners import run_episodes  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import CAUSAL_SERVER, METERED_SERVERS  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.tests.test_port_rows import NO_PROFILE, OUR_PORTS, STILL_TO_READ, causal_reply, chat_endpoint  # noqa: E402
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import profile_view  # noqa: E402
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools import answer_levels  # noqa: E402
from CausalDiagnosis.mcp_servers.utils import transcript  # noqa: E402
from CausalDiagnosis.mcp_servers.utils.profile_levels import raw_of  # noqa: E402
from CausalDiagnosis.mcp_servers.utils.response_levels import DEFAULT_LEVEL  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents.scripted_wiring_check import ScriptedBackend  # noqa: E402

DRIVERS = Path(__file__).resolve().parents[1]
STATIC = DRIVERS / "explorer" / "static"
BENCH, DEMO = "bench_1", "demo_1"
# the rows each fixture plays: (name, extra arguments of run_episodes.py)
PLAYS = [("bench_priced", ["--config", "static", "--asset", BENCH]),
         ("bench_base", ["--config", "base", "--asset", BENCH]),
         ("bench_aob_base", ["--config", "base", "--asset", BENCH, "--servers", "aob"]),
         ("demo_priced", ["--config", "static", "--asset", DEMO, "--no-diagnostics"])]
RIG = Path(default_facts_path()).exists()
if RIG:
    PLAYS.append(("rig_priced", ["--config", "static"]))
# a row on the token clock, which charges the tokens the model generates: the scripted agent calls no model, so its backend
# reports a made-up count while this row plays (made_up_usage)
TOKEN_PLAY = ("bench_tokens", ["--config", "real_time_token_clock", "--asset", BENCH])
# the CausalDiagnosis agent on a priced row, its model a stand-in endpoint answered by bench_planner
CAUSAL_PLAY = ("bench_causal", ["--config", "static", "--asset", BENCH, "--agent", "causal", "--model", "vendor/stand-in-1.0"])
# the one URL a page may carry: the name of the SVG namespace, which no browser fetches
SVG_NAMESPACE = "http://www.w3.org/2000/svg"


def tree_md5(folder):
    return {p.relative_to(folder).as_posix(): hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(Path(folder).rglob("*")) if p.is_file()}


def made_up_usage(self, prompt, temperature=0.0):
    """The reply of the scripted backend with a made-up count of generated tokens, one per four characters of the reply: the
    scripted agent calls no model, so it reports no count and the token clock would charge nothing."""
    text = self.generate(prompt, temperature)
    return LLMResult(text=text, output_tokens=max(1, len(text) // 4))


def give_token_rate(asset_file):
    """The copy of bench_1 takes the token rate of the rig's asset file, since bench_1 states none and a row of the token clock
    needs one; every other clock leaves the rate unread (real_time.token_rate)."""
    spec = json.loads(asset_file.read_text())
    spec["defaults"]["clock"]["seconds_per_token"] = get_profile("hyd_1", engine=False).defaults["clock"]["seconds_per_token"]
    asset_file.write_text(json.dumps(spec, indent=1))


def bench_planner(patch):
    """The planner of the CausalDiagnosis agent without a model on bench_1: the opening and the orientation of the drivers'
    stand-in (test_port_rows.causal_reply), then the plans of the agent's own tests (act_now) over the components of bench_1, so
    that it reads the advised group while the board advises a read, else diagnoses the top candidate, orders its faulty parts,
    waits, reads once more and finishes."""
    profile = get_profile(BENCH)
    patch.setattr(causal_tests, "COMPONENTS", list(profile.components))
    patch.setattr(causal_tests, "HEALTHY", dict(profile.healthy))
    patch.setattr(causal_tests, "VERIFYING_READ", dict(causal_tests.VERIFYING_READ, args={"sensor_group": sorted(profile.observation_kinds)[0], "cycles": 1}))
    opening, plans = causal_reply(BENCH), itertools.count(1)

    def reply(prompt):
        if ORIENTATION_HEADING in prompt or SUMMARY_HEADING in prompt or NO_PROFILE in prompt or STILL_TO_READ.search(prompt):
            return opening(prompt)
        return causal_tests.act_now(prompt, next(plans))
    return reply


@pytest.fixture(scope="module")
def played(tmp_path_factory):
    """Every fixture episode played once by the scripted agent into one runs folder; returns (runs folder, {name: episode
    folder}, md5 of every file of the runs folder right after the plays)."""
    root = tmp_path_factory.mktemp("ports")
    runs = tmp_path_factory.mktemp("runs")
    for source in (LAYER_PORTS / DEMO, OUR_PORTS / BENCH):
        shutil.copytree(source, root / source.name, ignore=shutil.ignore_patterns("data", "__pycache__"))
    give_token_rate(root / BENCH / "asset.json")
    episodes = {}
    with mount_port(root), pytest.MonkeyPatch.context() as patch:
        patch.setenv(get_profile(DEMO).records["data_env"], str(write_port_records(root / "demo_records.jsonl")))
        patch.setenv(get_profile(BENCH).records["data_env"], str(write_recordings(root / "bench_cycles.jsonl", get_profile(BENCH))))
        for name, args in PLAYS:
            run = Path(run_episodes.main(args + ["--agent", "scripted", "--seeds", "1", "--store", "memory", "--out", str(runs)]))
            assert run.parent == runs and run.name not in {e.parent.name for e in episodes.values()}, run
            episodes[name] = run / "seed000000"
        with pytest.MonkeyPatch.context() as usage:
            usage.setattr(ScriptedBackend, "generate_with_usage", made_up_usage)
            run = Path(run_episodes.main(TOKEN_PLAY[1] + ["--agent", "scripted", "--seeds", "1", "--store", "memory", "--out", str(runs)]))
        assert run.parent == runs and run.name not in {e.parent.name for e in episodes.values()}, run
        episodes[TOKEN_PLAY[0]] = run / "seed000000"
        with pytest.MonkeyPatch.context() as model:
            model.setenv("API_KEY", "stand-in-key")
            model.setenv("NO_PROXY", "127.0.0.1,localhost")
            endpoint = chat_endpoint(bench_planner(model))
            try:
                run = Path(run_episodes.main(CAUSAL_PLAY[1] + ["--seeds", "1", "--store", "memory", "--out", str(runs),
                                                               "--base-url", f"http://127.0.0.1:{endpoint.server_address[1]}/v1/"]))
            finally:
                endpoint.shutdown()
        assert run.parent == runs and run.name not in {e.parent.name for e in episodes.values()}, run
        episodes[CAUSAL_PLAY[0]] = run / "seed000000"
        yield runs, episodes, tree_md5(runs)


def build(episode, out, hidden=False):
    data.build(episode, out, hidden=hidden)
    return json.loads((Path(out) / "data.json").read_text())


@pytest.fixture
def serve(tmp_path):
    """Start an explorer over a runs folder on a free port of 127.0.0.1; yields a function (runs, hidden, cache) -> (host, port)."""
    started = []

    def start(runs, hidden=False, cache=None):
        srv = server.make_server(runs, 0, hidden, cache=cache or tmp_path / f"cache{len(started)}")
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        started.append(srv)
        return srv.server_address

    yield start
    for srv in started:
        srv.shutdown()
        srv.server_close()


def get(address, path):
    """(status, body) of a GET with the path sent as written, so a dot segment reaches the server unnormalized."""
    conn = http.client.HTTPConnection(*address, timeout=30)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp.status, body


def page_path(episode):
    return f"/e/{episode.parent.name}/{episode.name}/"


def leaves(value):
    """Every string and number inside a JSON value."""
    if isinstance(value, dict):
        for v in value.values():
            yield from leaves(v)
    elif isinstance(value, list):
        for v in value:
            yield from leaves(v)
    elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
        yield value


def distinctive(value):
    """A hidden value that would identify itself if it leaked: a string of six characters or more, or a number with a
    fractional part; short strings and whole numbers occur in any page by chance."""
    if isinstance(value, str):
        return len(value) >= 6
    return isinstance(value, float) and value != int(value)


# a number of a text: digits with an optional fraction and exponent, never begun inside another number, with the minus before
# it unless that minus follows a digit or a dot
NUMBER = re.compile(r"(?<![\d.])(-?)(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


@functools.lru_cache(maxsize=8)
def numbers_in(text):
    """The value of every number of a text, read once for all the hidden values asked of it: the value of its digits, and the
    negative value too where a minus stands before them, since a positive number also stands whole after a minus."""
    found = set()
    for minus, digits in NUMBER.findall(text):
        found.add(float(digits))
        if minus:
            found.add(-float(digits))
    return frozenset(found)


def shows(value, text):
    """Whether a hidden value stands in a text: a string anywhere, a number wherever a number of the text has its value, so that a
    hidden 2.5 counts in a public 2.500, 2.5e0 or -2.5 and not in a public 2.5731, and a hidden -2.5 counts in -2.500 and not in
    2.5."""
    if isinstance(value, str):
        return value in text or json.dumps(value) in text
    return value in numbers_in(text)


def hidden_values(episode):
    """The distinctive values of the hidden side of an episode (hidden/*.json, the ledger outside its public keys, the
    score columns outside the public ones) that no public file of the folder and no public key of the ledger also holds."""
    run_dir = episode.parent
    found = set()
    for p in sorted((episode / "hidden").glob("*.json")):
        found.update(leaves(json.loads(p.read_text())))
    ledger = json.loads((run_dir / "ledger.json").read_text())
    for k, v in ledger.items():
        if k == "episodes":
            for e in v.values():
                found.update(leaves({key: x for key, x in e.items() if key not in data.LEDGER_EPISODE_KEYS}))
        elif k not in data.LEDGER_PUBLIC_KEYS:
            found.update(leaves(v))
    import csv
    for row in csv.DictReader(open(run_dir / "scores.csv")):
        found.update(v for k, v in row.items() if k not in data.PUBLIC_SCORE_COLUMNS and v)
    public = "".join(p.read_text(errors="replace") for p in sorted(episode.iterdir())
                     if p.is_file() and p.name in data.PUBLIC_EPISODE_FILES)
    public += "".join((run_dir / n).read_text() for n in data.PUBLIC_RUN_FILES if (run_dir / n).exists())
    # the public keys of the ledger reach the page by design, such as an event that names a generated table and its SHA-256
    public += json.dumps({k: v for k, v in ledger.items() if k in data.LEDGER_PUBLIC_KEYS})
    public += json.dumps([{k: e[k] for k in data.LEDGER_EPISODE_KEYS if k in e} for e in (ledger.get("episodes") or {}).values()])
    return {v for v in found if distinctive(v) and not shows(v, public)}


def public_texts(out):
    """Every text the public builder wrote for an episode."""
    return {p.relative_to(out).as_posix(): p.read_text(errors="replace") for p in sorted(Path(out).rglob("*")) if p.is_file()}


# ----- T1 -----
def test_public_payload_holds_no_hidden_value(played, tmp_path, serve):
    runs, episodes, _ = played
    address = serve(runs, hidden=False)
    for name, episode in episodes.items():
        secrets = hidden_values(episode)
        assert secrets, f"{name}: the hidden side holds no distinctive value, so the test would prove nothing"
        out = tmp_path / name
        build(episode, out)
        texts = public_texts(out)
        assert not (out / "hidden.json").exists() and not (out / "hidden_files").exists()
        shown = json.loads((out / "files" / "run" / "ledger.public.json").read_text())
        assert set(shown) <= {*data.LEDGER_PUBLIC_KEYS, "episodes"}, (name, sorted(set(shown) - set(data.LEDGER_PUBLIC_KEYS)))
        assert all(set(e) <= set(data.LEDGER_EPISODE_KEYS) for e in shown["episodes"].values()), name
        sent = {}
        for rel in texts:
            status, body = get(address, page_path(episode) + "data/" + rel)
            assert status == 200, (name, rel, status)
            sent[rel] = body.decode(errors="replace")
        for where, text in list(texts.items()) + [("sent " + k, v) for k, v in sent.items()]:
            leaked = [v for v in secrets if shows(v, text)]
            assert not leaked, (name, where, sorted(map(str, leaked))[:5])


def test_a_hidden_number_counts_by_its_value_wherever_a_number_stands():
    """shows() compares a hidden number with every number of a text by value: the same number with trailing zeros or in another
    notation counts, a longer number that begins or ends with its digits does not, a positive number counts after a minus too,
    and a negative number only with its minus. The numbers are made up."""
    assert not shows(2.5, '{"a": 2.5731}') and not shows(2.5, "12.5") and not shows(2.5, "25")
    assert shows(2.5, '{"a": 2.500}')
    for text in ("2.5", "2.5e0", "25E-1", "-2.5", "x2.5", "at 2.5.", "[1,2.5]"):
        assert shows(2.5, text), text
    assert shows(-2.5, "[-2.500]") and not shows(-2.5, "t: 2.5") and not shows(-2.5, "3-2.5")
    assert shows(2.5e-07, "0.00000025")
    assert shows("made-up text", "a made-up text here") and not shows("made-up text", "made-up")


# ----- T2 -----
BROWSER_SCRIPT = r'''
import json, sys
from playwright.sync_api import sync_playwright
pub, hid = sys.argv[1], sys.argv[2]
COUNT = "() => [...document.querySelectorAll('#app .hidden-side')].filter(e => e.getClientRects().length > 0).length"
# every recorded "healthy band" sits in a block that carries the line naming it the normal range
UNGLOSSED = """() => { const w = document.createTreeWalker(document.getElementById('app'), NodeFilter.SHOW_TEXT); let bad = 0;
  for (let n = w.nextNode(); n; n = w.nextNode()) { const e = n.parentElement;
    if (!e || e.closest('.band-gloss, svg') || !/healthy band/i.test(n.data)) continue;
    const inside = (x) => [...x.children].some(c => c.classList.contains('band-gloss'));
    const raw = e.closest('.rawrec'), item = e.closest('li'), block = e.closest('p, table, dl, pre');
    const after = block && (!item || item.contains(block));
    const ok = raw ? inside(raw) : after ? !!(block.nextElementSibling && block.nextElementSibling.classList.contains('band-gloss')) : inside(item || e);
    if (!ok) bad++; }
  return bad; }"""
out = {}
with sync_playwright() as p:
    try:
        b = p.chromium.launch()
    except Exception:  # the browser this playwright bundles is not installed: the Chrome of the machine
        b = p.chromium.launch(channel="chrome")
    for width in (1440, 390):
        for mode, url in (("public", pub), ("people", hid)):
            pg = b.new_page(viewport={"width": width, "height": 900})
            errors = []
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.goto(url); pg.wait_for_selector(".ovb-step"); pg.wait_for_timeout(400)
            row = {"off": pg.evaluate(COUNT), "locked": pg.locator("#hidden-toggle").is_disabled(), "errors": errors, "band_unglossed": pg.evaluate(UNGLOSSED)}
            # the overview draws one mark per step of the step bar, and a click on a mark selects that step
            row["overview_marks"] = pg.locator(".ovb-step").count()
            row["steps"] = int(__import__("re").search(r"of (\d+)", pg.inner_text(".step-title")).group(1))
            texts = [pg.inner_text("body")]
            for tab in ("#tab-decision", "#tab-graph", "#tab-readings", "#tab-raw"):
                pg.click(tab); pg.wait_for_timeout(250)
                row["off"] += pg.evaluate(COUNT)
                row["band_unglossed"] += pg.evaluate(UNGLOSSED)
                if tab == "#tab-readings":
                    row["sensor_rows"] = pg.locator(".rdA-row").count()
                texts.append(pg.inner_text("body"))
            row["bare_mcp"] = sorted({m for s in texts for m in __import__("re").findall(r".{0,30}(?:(?<!CausalDiagnosis )\bMCP\b|(?i:diagnostics server)).{0,10}", s)})
            if mode == "people":
                pg.click("#hidden-toggle"); pg.wait_for_timeout(600)
                row["on"] = pg.evaluate(COUNT)
            # last, since a picked step opens the record sheet over the tabs on a narrow screen
            pg.focus(".ovb-step[data-n='2']"); pg.keyboard.press("Enter"); pg.wait_for_timeout(300)
            row["picked"] = pg.evaluate("location.hash").rstrip("/").split("/")[-1]
            out[f"{mode}_{width}"] = row
            pg.close()
    b.close()
print(json.dumps(out))
'''


def browser_python():
    """An interpreter that imports playwright: this one, or the one EXPLORER_BROWSER_PYTHON names; None without either."""
    if importlib.util.find_spec("playwright") is not None:
        return sys.executable
    named = os.environ.get("EXPLORER_BROWSER_PYTHON")
    if named and subprocess.run([named, "-c", "import playwright.sync_api"], capture_output=True).returncode == 0:
        return named
    return None


def test_public_page_draws_zero_hidden_marks(played, serve, tmp_path):
    python = browser_python()
    if python is None:
        pytest.skip("needs playwright with a browser, in this interpreter or the one EXPLORER_BROWSER_PYTHON names")
    runs, episodes, _ = played
    episode = episodes["bench_priced"]
    pub = "http://%s:%d" % serve(runs, hidden=False) + page_path(episode)
    hid = "http://%s:%d" % serve(runs, hidden=True) + page_path(episode)
    script = tmp_path / "count_hidden.py"
    script.write_text(BROWSER_SCRIPT)
    done = subprocess.run([python, str(script), pub, hid], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    counts = json.loads(done.stdout.strip().splitlines()[-1])
    for width in (1440, 390):
        public, people = counts[f"public_{width}"], counts[f"people_{width}"]
        assert public["off"] == 0 and public["locked"] and not public["errors"], public
        assert people["off"] == 0 and not people["locked"] and not people["errors"], people
        assert not public["bare_mcp"] and not people["bare_mcp"], (public["bare_mcp"], people["bare_mcp"])
        assert public["band_unglossed"] == 0 and people["band_unglossed"] == 0, (public, people)
        assert people["on"] > 0, people
        for row in (public, people):
            assert row["overview_marks"] == row["steps"] and row["picked"] == "2", row
            assert row["sensor_rows"] > 0, row  # the selector is live: the same page draws hidden marks once the toggle is on


# ----- T3 -----
def test_launchers_and_locks_are_never_served(played, serve, tmp_path):
    runs, episodes, _ = played
    for hidden in (False, True):
        address = serve(runs, hidden=hidden)
        for name, episode in episodes.items():
            kept_out = [p.name for p in episode.iterdir() if p.name.endswith("_server.py") or p.name.endswith(".lock") or p.name == "launch_private.json"]
            assert "launch_private.json" in kept_out, f"{name}: no private launch file in the episode folder, so the test would prove nothing"
            payload = build(episode, tmp_path / f"{name}_{hidden}", hidden=hidden)
            listed = {f["name"] for f in payload["episode"]["files"]["ep"]}
            if hidden:
                listed |= {f["name"] for f in json.loads((tmp_path / f"{name}_{hidden}" / "hidden.json").read_text())["files"]["ep"]}
            for n in kept_out:
                assert n not in listed and not any(n in f for f in listed), (name, n)
                for path in (f"data/files/ep/{n}", f"data/hidden_files/ep/{n}", n):
                    assert get(address, page_path(episode) + path)[0] == 404, (name, hidden, path)
            # the private file of the job beside the run's files
            assert (episode.parent / "job.json").exists(), name
            run_listed = {f["name"] for f in payload["episode"]["files"]["run"]}
            assert "job.json" not in run_listed, name
            for path in ("data/files/run/job.json", "data/hidden_files/run/job.json"):
                assert get(address, page_path(episode) + path)[0] == 404, (name, hidden, path)


# ----- T4 -----
def test_hidden_routes_need_the_flag(played, serve):
    assert cli.parser().parse_args(["serve", "--runs", "x"]).hidden is False
    assert cli.parser().parse_args(["export", "x", "--out", "y"]).hidden is False
    runs, episodes, _ = played
    episode = episodes["bench_priced"]
    routes = ["data/hidden.json"] + [f"data/hidden_files/ep/hidden/{p.name}" for p in sorted((episode / "hidden").glob("*.json"))]
    public, people = serve(runs, hidden=False), serve(runs, hidden=True)
    assert json.loads(get(public, page_path(episode) + "hidden-status.json")[1]) == {"served": False}
    assert json.loads(get(people, page_path(episode) + "hidden-status.json")[1]) == {"served": True}
    for route in routes:
        assert get(public, page_path(episode) + route)[0] == 404, route
        assert get(people, page_path(episode) + route)[0] == 200, route


# ----- T5 -----
def test_paths_outside_the_runs_folder_are_refused(played, serve, tmp_path):
    runs, episodes, _ = played
    episode = episodes["bench_priced"]
    # a runs folder whose run is a symbolic link out of it, and whose real run holds a file that links out of its folder
    outside = tmp_path / "outside"
    shutil.copytree(episode.parent, outside / episode.parent.name)
    (outside / "secret.txt").write_text("outside the runs folder")
    linked = tmp_path / "linked_runs"
    shutil.copytree(episode.parent, linked / "real_run")
    (linked / "away_run").symlink_to(outside / episode.parent.name, target_is_directory=True)
    (linked / "real_run" / episode.name / "report.md").unlink()
    (linked / "real_run" / episode.name / "report.md").symlink_to(outside / "secret.txt")
    for runs_folder, hidden in ((runs, False), (runs, True), (linked, False)):
        address = serve(runs_folder, hidden=hidden)
        assert address[0] == "127.0.0.1"
        run, ep = episode.parent.name, episode.name
        for path in ("/e/../x/", f"/e/{run}/../{run}/{ep}/", "/e/%2e%2e/%2e%2e/", f"/e/{run}/{ep}/data/..%2F..%2Frun.json",
                     f"/e/{run}/{ep}/data/%2Fetc%2Fhosts", "/etc/hosts", f"/e/{run}/{ep}/data/files/..%2F..%2F..%2Fledger.json",
                     f"/e/{run}/{ep}/..%2F{ep}/data/data.json", "/e/.hidden/x/"):
            status, body = get(address, path)
            assert status == 404 and b"outside" not in body, (path, status)
    address = serve(linked, hidden=False)
    assert get(address, f"/e/away_run/{episode.name}/")[0] == 404
    assert get(address, f"/e/away_run/{episode.name}/data/data.json")[0] == 404
    status, body = get(address, f"/e/real_run/{episode.name}/data/data.json")
    assert status == 200 and "report.md" not in [f["name"] for f in json.loads(body)["episode"]["files"]["ep"]]
    assert get(address, f"/e/real_run/{episode.name}/data/files/ep/report.md")[0] == 404
    for verb in ("POST", "PUT", "DELETE"):
        conn = http.client.HTTPConnection(*address, timeout=30)
        conn.request(verb, f"/e/real_run/{episode.name}/")
        assert conn.getresponse().status == 405, verb
        conn.close()


def test_a_public_server_refuses_the_hidden_side_that_its_cache_already_holds(tmp_path, serve):
    """The route guard of the server answers 404 for every hidden part on a server without the hidden option, even where its cache
    folder already holds them, as after a build with the hidden option into the same cache. A server with the option serves the
    same files, so the routes resolve and the guard alone keeps them out. The runs folder and the files are made up."""
    runs, cache = tmp_path / "runs", tmp_path / "shared_cache"
    (runs / "run_a" / "seed000000").mkdir(parents=True)
    (runs / "run_a" / "seed000000" / "trajectory.json").write_text("[]")
    out = cache / "run_a" / "seed000000"
    hidden = {"hidden.json": '{"made_up": 1}', "hidden_files/ep/hidden/log.json": '[{"made_up": 2}]', "hidden_levels/1.json": '{"made_up": 3}'}
    for rel, text in {"data.json": "{}", **hidden}.items():  # data.json present, so the server builds nothing into this cache
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_text(text)
    public, people = serve(runs, hidden=False, cache=cache), serve(runs, hidden=True, cache=cache)
    for rel, text in hidden.items():
        assert get(people, "/e/run_a/seed000000/data/" + rel) == (200, text.encode()), rel
        assert get(public, "/e/run_a/seed000000/data/" + rel)[0] == 404, rel
    # every part the guard names has a route above, so a part added to it gets one too
    assert {rel.split("/")[0] for rel in hidden} == set(server.HIDDEN_PARTS)


def test_the_serve_command_prints_its_address_while_it_serves(tmp_path):
    """The documented command prints the address it serves on as soon as it listens, also when its output goes to a file, so the
    address is read from that file while the server runs."""
    (tmp_path / "runs").mkdir()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONUNBUFFERED"}  # unbuffered output would hide a missing flush
    env |= {"PYTHONPATH": os.pathsep.join((str(ROOT), str(SRC))), "PYTHONDONTWRITEBYTECODE": "1"}
    log = tmp_path / "serve.log"
    with log.open("w") as out:
        child = subprocess.Popen([sys.executable, "-m", "CausalDiagnosis", "episode_viewer", "explorer", "serve", "--runs", str(tmp_path / "runs"),
                                  "--port", "0"], cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 120
        while (found := re.search(r"http://127\.0\.0\.1:(\d+)/", log.read_text())) is None and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.2)
        assert found and child.poll() is None, log.read_text()[-2000:]
        assert get(("127.0.0.1", int(found.group(1))), "/") == (200, b"No played episode lies under the runs folder.")
    finally:
        child.kill()
        child.wait()


# ----- T6 -----
def test_a_missing_file_shows_as_not_recorded(played, tmp_path, serve):
    runs, episodes, _ = played
    episode = episodes["bench_priced"]
    copy = tmp_path / "runs" / episode.parent.name
    shutil.copytree(episode.parent, copy)
    gone = ("predictions.json", "plans.json", "plan.txt", "answer.txt", "llm_calls.json", "question.txt")
    for name in gone:
        (copy / episode.name / name).unlink(missing_ok=True)
    payload = build(copy / episode.name, tmp_path / "out")
    ep = payload["episode"]
    assert ep["predictions"] == [] and ep["plan"] == [] and ep["answer"] is None
    assert all(c["llm_index"] is None for c in ep["calls"])
    assert not {f["name"] for f in ep["files"]["ep"]} & set(gone)
    # an episode folder that holds no trajectory was not played: no page, and the run lists it nowhere
    unplayed = copy / "seed000001"
    unplayed.mkdir()
    (unplayed / "episode.json").write_text("{}")
    address = serve(tmp_path / "runs", hidden=False)
    assert get(address, f"/e/{copy.name}/seed000001/")[0] == 404
    assert "seed000001" not in build(copy / episode.name, tmp_path / "out2")["runs"][0]["played"]


# ----- T7 -----
def test_follow_marks_use_the_metrics_functions(played, tmp_path):
    runs, episodes, _ = played
    names = [n for n in episodes if n != "demo_priced"]
    assert any(n.startswith("bench") for n in names)
    for name in names + ["demo_priced"]:
        episode = episodes[name]
        calls = build(episode, tmp_path / name)["episode"]["calls"]
        verdict = json.loads((episode / "hidden" / "verdict.json").read_text())
        m = mcp_metrics.episode_measures(episode, None, None, verdict, with_verdicts=False)
        assert m and not m["unaligned"], name
        visit_marks = {}
        for c in calls:
            if c.get("visit"):
                visit_marks.setdefault(c["visit"], set()).add(c["follow"])
        assert all(len(marks) == 1 for marks in visit_marks.values()), (name, visit_marks)
        marks = [next(iter(v)) for _, v in sorted(visit_marks.items())]
        assert marks.count("followed") == m["follow_first"], (name, marks, m)
        assert marks.count("deviated") == m["deviate"], (name, marks, m)
        assert len(marks) - marks.count("no advice") == m["advised"], (name, marks, m)
        reads = [c for c in calls if c["kind"] == "reads" and "sensor_group" in c["args"]]
        assert sum(c["follow"] == "followed" for c in reads) == m["reads_advised"], name
        assert sum(c["follow"] in ("followed", "deviated") for c in reads) == m["reads_read_first"], name
        submits = [c for c in calls if c["tool"] == mcp_metrics.SUBMIT]
        word = {True: "followed", False: "deviated", None: "no advice"}
        assert [c["follow"] for c in submits] == ([word[m["diagnosis_follows"]]] if submits else []), name
        if name == "demo_priced":  # no ranking was asked, so nothing was advised
            assert {c["follow"] for c in calls} <= {"no advice", None}


def test_follow_marks_reach_followed_deviated_and_no_advice():
    """The played fixtures never follow the advice, so a call list made by hand takes every mark: two rankings, a read of the
    advised group and one of another group while the ranking says to read first, a run of two orders equal to the plan of
    the first branch, one order off the plan after a step, a diagnosis equal to the first candidate and one against it.
    Every mark equals what the functions of mcp_metrics.py give for the same call and answer."""
    order = sorted(mcp_metrics.order_tools())[0]
    comps = ["a", "b"]
    rank = {"candidates": [{"combination": {"a": 1, "b": 0}}], mcp_metrics.READ_FIRST: True,
            "separating_test": {"sensor_group": "g1"},
            "branches": [{"repair_plan": {"steps": [{"component": "a"}, {"component": "b"}]}}]}
    def call(n, tool, args, kind, stepped=True, price=1, answer=None):
        return {"n": n, "tool": tool, "args": args, "kind": kind, "stepped": stepped, "price": price, "answer": answer or {},
                "error": None, "refused": None}
    calls = [call(1, "read_sensors", {"sensor_group": "g1"}, "reads"),
             call(2, mcp_metrics.RANK, {}, "advice", stepped=False, price=0, answer=rank),
             call(3, "read_sensors", {"sensor_group": "g1"}, "reads"),
             call(4, "read_sensors", {"sensor_group": "g2"}, "reads"),
             call(5, "read_sensors", {"sensor_group": "g2"}, "reads", price=0),
             call(6, order, {"component": "a"}, "orders"),
             call(7, order, {"component": "b"}, "orders"),
             call(8, "wait_for_notification", {}, "time"),
             call(9, order, {"component": "a"}, "orders"),
             call(10, mcp_metrics.SUBMIT, {"a": 1, "b": 0}, "decide"),
             call(11, mcp_metrics.SUBMIT, {"a": 0, "b": 0}, "decide")]
    prices = {"read": {"g1": 1, "g2": 1}, "parts": {"a": 1, "b": 1}, "repair_charge": 1, "visit": 1,
              "fix_kind": {"a": "replace", "b": "replace"}, "misdiagnosis": 1}
    data.follow_marks(calls, {"components": comps, "prices": prices})
    marks = {c["n"]: c["follow"] for c in calls}
    assert marks == {1: "no advice", 2: None, 3: "followed", 4: "deviated", 5: "no advice", 6: "followed", 7: "followed",
                     8: "no advice", 9: "deviated", 10: "followed", 11: "deviated"}, marks
    assert calls[5]["visit"] == calls[6]["visit"] != calls[8]["visit"]
    assert mcp_metrics.read_followed("g1", rank) is True and mcp_metrics.read_followed("g2", rank) is False
    assert mcp_metrics.plan_followed({"a", "b"}, rank) is True and mcp_metrics.plan_followed({"a"}, rank) is False
    levels = lambda c: tuple(int(c[k]) for k in comps)  # noqa: E731
    assert mcp_metrics.diagnosis_followed((1, 0), rank, levels) is True and mcp_metrics.diagnosis_followed((0, 0), rank, levels) is False


def test_page_strings_name_the_mcp_in_full():
    """The CausalDiagnosis MCP has no short form in the paper, its figures or the docs pages, and the page's shots enter all
    three, so no string the page or its builder prints says MCP alone or calls it the diagnostics server."""
    bare = re.compile(r"(?<!CausalDiagnosis )\bMCP\b|(?i:diagnostics server)")
    strings = re.compile(r'"(?:[^"\\\n]|\\.)*"|`[^`]*`|\'(?:[^\'\\\n]|\\.)*\'')
    for path in (STATIC / "app.js", STATIC / "index.html", DRIVERS / "explorer" / "data.py", DRIVERS / "explorer" / "cli.py",
                 DRIVERS / "explorer" / "server.py", DRIVERS / "explorer" / "shots.py"):
        text = path.read_text()
        if path.suffix == ".html":
            hits = bare.findall(re.sub(r"<!--.*?-->", "", text, flags=re.S))
        else:
            hits = [s for s in strings.findall(text) if bare.search(s)]
        assert not hits, (path.name, hits[:5])


# ----- T8 -----
def asset_names():
    """Every name of an asset that a page could print: the ids, sites and asset types, the components, the channels and
    their public fields, of every asset of the tree and of the made-up ports of the tests."""
    found = set()
    profiles = [get_profile(a) for a in ports.asset_ids()]
    for folder in (LAYER_PORTS / DEMO, OUR_PORTS / BENCH):
        profiles.append(json.loads(next(folder.rglob("asset.json")).read_text()))
    for p in profiles:
        get = (lambda k: p[k]) if isinstance(p, dict) else (lambda k: getattr(p, k))
        found.add(get("asset_id"))
        found.update(get("components"))
        for channel, entry in get("channels").items():
            found.add(channel)
            found.add(entry["field"])
        fallback = get("text").get("fallback", {})
        found.update(v for k, v in fallback.items() if k in ("site", "asset_type"))
    return {n for n in found if n}


def test_frontend_names_no_asset():
    """A name the code holds is a string literal, so every name is searched inside quotes; a name with a digit or an
    underscore (an id, a channel, a field) is no English word and is searched as a whole word anywhere. A component named
    by a plain word (press, fan) may stand in the page's own sentences, as any word may."""
    names = asset_names()
    assert {"hyd_1", BENCH, DEMO} <= names
    for path in sorted(STATIC.iterdir()) + [DRIVERS / "explorer" / "data.py", DRIVERS / "explorer" / "server.py", DRIVERS / "explorer" / "cli.py"]:
        text = path.read_text()
        quoted = [n for n in names if re.search(r"[\"'`]" + re.escape(n) + r"[\"'`]", text)]
        coded = [n for n in names if re.search(r"[\d_]", n) and re.search(r"(?<![\w-])" + re.escape(n) + r"(?![\w-])", text)]
        assert not quoted and not coded, (path.name, sorted(quoted), sorted(coded))


# ----- T9 -----
def test_export_is_one_file_without_network(played, tmp_path):
    runs, episodes, _ = played
    episode = episodes["bench_priced"]
    page = cli.export(episode, tmp_path / "page.html")
    html = page.read_text()
    assert set(re.findall(r"https?://[^\s\"'<>)]*", html)) <= {SVG_NAMESPACE}
    assert not re.search(r"<link\b", html, re.I) and not re.search(r"\bsrc\s*=", html, re.I)
    # url() only ever names a fragment of the page itself (an SVG pattern or marker) or inline data
    assert not re.search(r"@import|url\(\s*['\"]?(?!data:|#)", html)
    secrets = hidden_values(episode)
    assert not [v for v in secrets if shows(v, html)]
    for refused in (ROOT / "explorer_page.html", episode / "page.html", episode.parent / "page.html"):
        with pytest.raises(SystemExit):
            cli.export(episode, refused)
        assert not refused.exists()
    assert "hidden.json" in cli.export(episode, tmp_path / "people.html", hidden=True).read_text()
    recorded = [c["n"] for c in build(episode, tmp_path / "levels_out")["episode"]["calls"] if c.get("levels") == "recorded"]
    assert recorded and all(f'"data/levels/{n}.json"' in html for n in recorded), recorded


# ----- T10 -----
def test_every_fixture_builds(played, tmp_path):
    runs, episodes, _ = played
    for name, episode in episodes.items():
        payload = build(episode, tmp_path / name, hidden=True)
        ep = payload["episode"]
        assert ep["calls"] and ep["id"] == episode.name and ep["run"] == episode.parent.name, name
        if name.endswith("_base"):  # no prices: the steps are the axis, the budget is absent
            assert ep["rules"].get("budget") is None and all(c["budget_before"] is None for c in ep["calls"]), name
            assert all(isinstance(c["steps_left_before"], int) for c in ep["calls"]), name
        else:  # priced: the cycles are the axis and every call meets the budget
            assert isinstance(ep["rules"].get("budget"), (int, float)), name
            assert all(isinstance(c["budget_before"], (int, float)) for c in ep["calls"]), name
            assert all(c["time_after"] >= c["time_before"] for c in ep["calls"]), name


def test_no_file_of_a_run_folder_changed(played):
    runs, _, before = played
    assert tree_md5(runs) == before


def plural(n, word):
    """A count with its word, as the page writes it (plural of app.js)."""
    return f"{n} {word}{'' if n == 1 else 's'}"


def readings_served(call):
    """The readings a call's answer served, as the page counts them for what was known before a step (obsBefore)."""
    answer = call.get("answer")
    result = answer.get("result") if isinstance(answer, dict) else None
    served = result.get("served") if isinstance(result, dict) else None
    return len(served) if isinstance(served, list) else 0


@pytest.fixture(scope="module")
def played_no_call(played, tmp_path_factory):
    """One more scripted episode of bench_1 on the priced row, played inside the mount and the settings of played into a runs
    folder of its own, in which the driver's no_call entry is sent once after the third tool call, from inside the reply that
    asks for the arguments of the fourth step and in a thread, as agent.idle sends it for an asynchronous design; returns the
    episode folder."""
    runs = tmp_path_factory.mktemp("runs_no_call")
    generate, prompts = ScriptedBackend.generate, []

    def generate_after_a_turn_without_a_tool(self, prompt, temperature=0.0):
        prompts.append(prompt)
        if len(prompts) == 5:  # the plan, then the arguments of steps 1 to 4: the fourth is asked once the third call answered
            sender = threading.Thread(target=idle.no_call)
            sender.start()
            sender.join()
        return generate(self, prompt, temperature)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ScriptedBackend, "generate", generate_after_a_turn_without_a_tool)
        run = Path(run_episodes.main(PLAYS[0][1] + ["--agent", "scripted", "--seeds", "1", "--store", "memory", "--out", str(runs)]))
    return run / "seed000000"


def played_in_order(e):
    """The calls and the turns without a tool of an episode's data in the order they were played."""
    return sorted([*e["calls"], *e["idle_steps"]], key=lambda x: x["n"] if "n" in x else x["after"] + 0.5)


def test_the_new_fields_equal_the_files_they_come_from(played, played_no_call, tmp_path):
    """The fields the page reads for the turns without a tool, the thinking clock, the serving facts, the cost unit and the
    causes a ranking leaves out equal what the driver and the servers wrote: episode.json (the clock record, and the thinking
    charged before each counted call in the order of the stepped lines of the transcript, with its tokens on the token clock),
    harness.json (the attempt that stands) and the answers of the CausalDiagnosis MCP."""
    _, episodes, _ = played
    for name, episode in [*episodes.items(), ("bench_no_call", played_no_call)]:
        e = build(episode, tmp_path / name)["episode"]
        state = json.loads((episode / "episode.json").read_text())
        standing = json.loads((episode / "harness.json").read_text())["standing"]
        clock_keys = ("credited_seconds", "uncredited_seconds", "failed_model_seconds", "failed_model_attempts")
        tokens = state.get("think_tokens")
        assert e["clock"] == {k: state[k] for k in clock_keys} | {"charged_seconds": sum(state["think_seconds"]),
                                                                  "charged_tokens": None if tokens is None else sum(tokens)}, name
        assert e["served"] == {"wall_limit": bool(standing["wall_limit"]), "cut_end": standing["cut_end"] or "",
                               "cut_probe": standing["cut_probe"], "cache_flush": standing["cache_flush"],
                               "episode_ended_at": state.get("episode_ended_at")}, name
        lines = [json.loads(line) for line in (episode / "transcript.jsonl").read_text().splitlines() if line.strip()]
        stepped = [line for line in lines if line.get("kind") == "call" and line.get("stepped")]
        assert len(stepped) == len(state["think_seconds"]), name
        thinking = [(x["think_seconds"], x["think_cycles"], x["think_tokens"]) for x in played_in_order(e) if x["think_seconds"] is not None]
        assert thinking == list(zip(state["think_seconds"], state["think_cycles"], tokens or [None] * len(state["think_seconds"]))), name
        # the token clock alone keeps tokens, and its row charged some
        assert (tokens is not None) == (name == TOKEN_PLAY[0]) and (tokens is None or sum(tokens) > 0), name
        assert all((c["think_seconds"] is not None) == bool(c["stepped"]) for c in e["calls"]), name
        assert len(e["idle_steps"]) == sum(1 for line in stepped if line["tool"] == "no_call"), name
        assert e["cost_unit"] == ("calls" if name.endswith("_base") else "money"), name
        if name != "demo_priced":  # the one fixture played without the CausalDiagnosis MCP asks for no ranking
            assert e["rankings"], name
        answers = {c["n"]: c["answer"] for c in e["calls"]}
        assert all(r["more_candidates"] == answers[r["n"]]["more_candidates"] for r in e["rankings"]), name


def test_a_turn_without_a_tool_is_set_apart_where_the_driver_wrote_it(played_no_call, tmp_path):
    """The no_call step the driver wrote after the third tool call: the data sets it apart after call 3 and nowhere else, every
    call of the agent keeps its line, the step takes one step of the environment, and the thinking entries of episode.json
    fall on the stepped lines in order, the no_call's included."""
    e = build(played_no_call, tmp_path / "page")["episode"]
    state = json.loads((played_no_call / "episode.json").read_text())
    trajectory = json.loads((played_no_call / "trajectory.json").read_text())
    assert [c["tool"] for c in state["tool_calls"]].index("no_call") == 3
    assert [x["after"] for x in e["idle_steps"]] == [3]
    assert [c["tool"] for c in e["calls"]] == [r["tool"] for r in trajectory]
    assert all(c["answer_sha256"] for c in e["calls"])
    assert e["idle_steps"][0]["env_step"] == next(c["env_step"] for c in e["calls"] if c["n"] == 3) + 1
    assert [x["think_seconds"] for x in played_in_order(e) if x["think_seconds"] is not None] == state["think_seconds"]


def test_the_thinking_follows_the_stepped_lines_whatever_the_server_is_called(played, tmp_path):
    """A counted call is a call whose line in the transcript stepped the environment, whatever server served it: on the rows
    of the AssetOpsBench servers the counted calls belong to servers other than the rig's. The thinking of each entry of
    episode.json goes to the call of its stepped line, so a copy whose action server carries another name keeps every value."""
    _, episodes, _ = played

    def renamed(row):
        return dict(row, server="another_server") if row.get("server") == run_episodes.RIG_SERVER else row

    for source in (episodes["bench_priced"], episodes[TOKEN_PLAY[0]]):
        run_copy = tmp_path / "runs" / source.parent.name
        shutil.copytree(source.parent, run_copy)
        folder = run_copy / source.name
        rules = (data.load(folder / "released.json", {}) or {}).get("rules", {})
        before = data.episode_calls(folder, rules)[0]
        rows = json.loads((folder / "trajectory.json").read_text())
        (folder / "trajectory.json").write_text(json.dumps([renamed(r) for r in rows]))
        lines = [json.loads(line) for line in (folder / "transcript.jsonl").read_text().splitlines() if line.strip()]
        (folder / "transcript.jsonl").write_text("".join(json.dumps(renamed(line)) + "\n" for line in lines))
        after = data.episode_calls(folder, rules)[0]
        assert any(c["think_seconds"] is not None for c in before), source
        thinking = [[(c["think_seconds"], c["think_cycles"], c["think_tokens"]) for c in calls] for calls in (before, after)]
        assert thinking[1] == thinking[0], source


def test_no_call_step_keeps_the_lines_of_later_calls(played, tmp_path):
    """A no_call step has a call line in the transcript and its thinking in the state, and no entry in the trajectory
    (episode.call); the data sets it apart after the call it follows, and every later call keeps its own line and its own
    thinking, so its stepped flag, answer hash and thinking do not move."""
    _, episodes, _ = played
    source = episodes["bench_priced"]
    run_copy = tmp_path / "runs" / source.parent.name
    shutil.copytree(source.parent, run_copy)
    folder = run_copy / source.name
    rules = (data.load(folder / "released.json", {}) or {}).get("rules", {})
    before = data.episode_calls(folder, rules)[0]
    lines = [json.loads(line) for line in (folder / "transcript.jsonl").read_text().splitlines() if line.strip()]
    calls_at = [i for i, line in enumerate(lines) if line.get("kind") == "call"]
    stepped_before = sum(1 for i in calls_at[:3] if lines[i].get("stepped"))
    base, top = lines[calls_at[2]], max(line["n"] for line in lines)
    no_call = {k: base[k] for k in base if k not in ("tool", "arguments", "answer_sha256", "n", "enter_n")}
    no_call.update(n=top + 2, tool="no_call", arguments={}, answer_sha256="made-up", stepped=True, enter_n=top + 1)
    lines[calls_at[2] + 1:calls_at[2] + 1] = [{"n": top + 1, "kind": "enter", "tool": "no_call"}, no_call, {"n": top + 3, "kind": "exit", "enter_n": top + 1}]
    (folder / "transcript.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    # the state takes the thinking of the step at the place of its stepped line, as episode.step appends it
    state = json.loads((folder / "episode.json").read_text())
    state["think_seconds"].insert(stepped_before, 7.5)
    state["think_cycles"].insert(stepped_before, 0)
    (folder / "episode.json").write_text(json.dumps(state))
    after, idle, _ = data.episode_calls(folder, rules)
    assert [c["answer_sha256"] for c in after] == [c["answer_sha256"] for c in before]
    assert [c["stepped"] for c in after] == [c["stepped"] for c in before]
    assert [(c["think_seconds"], c["think_cycles"]) for c in after] == [(c["think_seconds"], c["think_cycles"]) for c in before]
    assert len(idle) == 1 and idle[0]["after"] == 3 and idle[0]["env_step"] == (before[2]["env_step"] or 0) + 1
    assert (idle[0]["think_seconds"], idle[0]["think_cycles"]) == (7.5, 0)
    shifted = [(c["env_step"] or 0) - (b["env_step"] or 0) for b, c in zip(before, after)]
    assert shifted == [0] * 3 + [1 if b["stepped"] else 0 for b in before[3:]]


# ----- T11: the plain sentences of the page are true for every episode -----
TEXT_SCRIPT = r"""
import json, sys
from playwright.sync_api import sync_playwright
cases = json.loads(open(sys.argv[1]).read())
out = {}
with sync_playwright() as p:
    try:
        b = p.chromium.launch()
    except Exception:  # the browser this playwright bundles is not installed: the Chrome of the machine
        b = p.chromium.launch(channel="chrome")
    for name, case in cases.items():
        pg = b.new_page(viewport={"width": 1440, "height": 900})
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        if case["data"]:
            body = open(case["data"]).read()
            pg.route("**/data/data.json", lambda route, request, body=body: route.fulfill(body=body, content_type="application/json"))
        pg.goto(case["url"]); pg.wait_for_selector(".ovb-step"); pg.wait_for_timeout(300)
        row = pg.evaluate("() => ({ summary: plainSummary(ep()), steps: ep().calls.map((c) => callSentence(c)), tags: ep().calls.map((c) => stepWords(c)),"
                          " story: document.querySelector('.tl-top .rs-story').textContent, clock: document.querySelector('.tl-top .rs-clock').textContent,"
                          " chips: [...document.querySelectorAll('.tl-top .rs-facts span.fact')].map((x) => x.textContent),"
                          " idle: [...document.querySelectorAll('.ovb-step.k-idle text')].map((x) => x.textContent),"
                          " idlestory: (ep().idle_steps || []).length ? [...idleStory(ep().idle_steps[0]).querySelectorAll('li')].map((x) => x.textContent) : null,"
                          " think: ((el) => el ? [el.querySelector('summary').textContent, el.querySelector('.think-open').textContent] : null)(document.querySelector('.tl-top .f-think')),"
                          " thinklast: ((el) => el ? el === el.parentElement.lastElementChild : null)(document.querySelector('.tl-top .f-think')),"
                          " thinkrows: ep().calls.map((c) => { const r = thinkRow(c); return r ? r[1].textContent : null; }),"
                          " served: ((el) => el ? [...el.querySelectorAll('dd')].map((x) => x.textContent) : null)(servedBlock(ep().run)),"
                          " steps_axis: stepMode(ep()), lastband: [...timelineView({ fig: true }).querySelectorAll('text')].map((x) => x.textContent).filter((t) => t === 'Spending' || t === 'Steps taken'),"
                          " bandsub: [...timelineView({ fig: true }).querySelectorAll('text')].map((x) => x.textContent).filter((t) => /counts steps|has prices/.test(t)),"
                          " notlisted: ep().rankings.map((r) => { const el = notListed(r); return [r.n, el ? el.querySelector('summary').textContent : null, el ? el.textContent : null]; }) })")
        row["errors"] = errors
        out[name] = row
        pg.close()
    b.close()
print(json.dumps(out))
"""


def made_up(payload, how):
    """A copy of an episode's data changed one way; every case is made-up data, not a played episode."""
    d = json.loads(json.dumps(payload))
    e = d["episode"]
    calls = e["calls"]
    find = lambda tool: next(i for i, c in enumerate(calls) if c["tool"] == tool)
    end = [x for x in e["events"] if x["event"] == "end"]
    if how in ("matched", "not_matched"):
        # the repair is done before the check, and the check answers that the readings did or did not match
        chk = find("verify_outcome")
        closes = [n for c in calls for n in (c.get("notifications") or []) if n.get("event") == "work_order_closed"]
        for c in calls:
            c["notifications"] = [n for n in (c.get("notifications") or []) if n.get("event") != "work_order_closed"]
        for n in closes:
            n["t"] = calls[chk - 1]["time_before"]
        calls[chk - 1]["notifications"] += closes
        calls[chk]["answer"].update(status="verified", consistent=how == "matched")
    elif how == "declared":
        k = find("submit_diagnosis")
        calls[k].update(tool="declare_insufficient_evidence", args={"sensor_group": "full", "cycles": 1, "evidence_ids": ""}, follow=None)
        fin = dict(calls[-1], n=k + 2, notifications=[n for n in calls[-1]["notifications"] or [] if n.get("event") == "end"])
        e["calls"] = calls[:k + 1] + [fin]
    elif how == "no_repair":
        k = find("create_work_order")
        e["calls"] = [c for i, c in enumerate(calls[:k]) if c["tool"] != "verify_outcome"] + [dict(calls[-1], n=k + 1, notifications=[])]
        for i, c in enumerate(e["calls"]):
            c["n"] = i + 1
    elif how == "model_agent":
        run = next(r for r in d["runs"] if r["folder"] == e["run"])
        run.update(agent="causal", agent_label="causal", model="made-up-model")
    elif how == "all_listed":
        for r in e["rankings"]:
            r["more_candidates"] = {"count": 0, "mass": 0.0}
    elif how in ("generation_time", "instant"):
        e["rules"]["clock_mode"] = how
    elif how == "older_ranking":
        # rankings without more_candidates
        for r in e["rankings"]:
            r.pop("more_candidates", None)
    elif how in ("prices_no_budget", "free_no_budget"):
        # the budget switch off: on a priced row the prices stay, on the free row nothing has a price
        e["rules"]["budget"] = None
        if how == "free_no_budget":
            e["cost_unit"] = "calls"
    elif how == "priced_no_durations":
        # a priced row whose calls take no cycle, as the price layer on the base layer
        for c in calls:
            if c["kind"] != "time":
                c["cycles"] = 0
    elif how == "no_call":
        # one no_call step after the third call, in the shape data.episode_calls gives it
        e["idle_steps"] = [{"after": 3, "env_step": (calls[2].get("env_step") or 0) + 1, "time": calls[2]["time_after"],
                            "think_seconds": None, "think_cycles": None, "think_tokens": None}]
    elif how == "no_call_after_reads":
        # one no_call step after the first call that served readings, so its line says how many readings were known
        k = next(i for i, c in enumerate(calls) if readings_served(c))
        e["idle_steps"] = [{"after": k + 1, "env_step": (calls[k].get("env_step") or 0) + 1, "time": calls[k]["time_after"],
                            "think_seconds": None, "think_cycles": None, "think_tokens": None}]
    elif how == "thinking":
        e["rules"]["clock_mode"] = "generation_time"
        for i, c in enumerate(calls):
            c["think_seconds"], c["think_cycles"] = (round(4.0 + 3 * (i % 4), 1), (4 + 3 * (i % 4)) // 10) if c["server"] == "hydraulic_rig" else (None, None)
        e["clock"] = {"credited_seconds": {"starts": 20.0, "calls": 5.0}, "uncredited_seconds": {"starts": 1.0, "calls": 0.5},
                      "failed_model_seconds": 12.0, "failed_model_attempts": 2,
                      "charged_seconds": sum(c["think_seconds"] or 0 for c in calls)}
    elif how == "wall_cut":
        e["served"] = {"wall_limit": True, "cut_end": "service", "cut_probe": {"ok": True, "status": 200, "error": None, "seconds": 0.7, "cut_probes": 3},
                       "cache_flush": {"flush_seconds": 0.5, "status": 200, "first_request_cached_tokens": 0, "started_at": 1.0},
                       "episode_ended_at": 2.0}
    elif how in ("probe_failed", "cut_no_probe", "flush_no_cut"):
        # the other three states of the row of the check of the model: a check at the cut that got no token (the record of
        # wall_cut.cut_probe when the service refuses the connection), a cut with no check made, and a run that emptied the
        # cache and was not cut
        probe = {"ok": False, "status": None, "error": "URLError: <urlopen error [Errno 61] Connection refused>", "seconds": 0.004, "cut_probes": 0}
        e["served"] = {"wall_limit": how != "flush_no_cut", "cut_end": {"probe_failed": "service", "cut_no_probe": "wall"}.get(how, ""),
                       "cut_probe": probe if how == "probe_failed" else None,
                       "cache_flush": {"flush_seconds": 0.5, "status": 200, "first_request_cached_tokens": 0, "started_at": 1.0},
                       "episode_ended_at": 2.0}
    elif how in ("max_steps", "horizon", "budget"):
        k = find("verify_outcome")
        e["calls"] = calls[:k]
        e["events"] = [x for x in e["events"] if x["event"] not in ("finish", "work_order_closed", "end")] + [dict((end or [{}])[0], event="end", reason=how)]
    return d


def how_of(name):
    """The made-up variant a case name carries, or None for a played episode."""
    return name.split(": ", 1)[1] if name.startswith("made-up from ") else None


def facts(payload):
    """What the sentences must say, read from the data alone."""
    e = payload["episode"]
    calls, rules = e["calls"], e["rules"]
    bad = lambda comb: sorted(k for k in rules["components"] if k in comb and comb[k] != rules["healthy"][k])
    decide = next((c for c in calls if c["kind"] == "decide" and not c.get("refused") and c["tool"] in ("submit_diagnosis", "declare_insufficient_evidence")), None)
    ranks = [c for c in calls if c["kind"] == "advice" and isinstance((c.get("answer") or {}).get("candidates"), list) and (decide is None or c["n"] < decide["n"])]
    # an order of AssetOpsBench's servers names the child asset of its component
    orders = sorted({c["args"].get("component") or e["asset_children"][c["args"]["asset_num"]] for c in calls
                     if c["tool"] in mcp_metrics.order_tools() and not c.get("refused")})
    check = [c for c in calls if c["tool"] == "verify_outcome" and (c.get("answer") or {}).get("status")]
    end = [x for x in e.get("events") or [] if x["event"] == "end"]
    return {"top": bad(ranks[-1]["answer"]["candidates"][0]["combination"]) if ranks else None,
            "named": bad(decide["args"]) if decide and decide["tool"] == "submit_diagnosis" else None,
            "declared": bool(decide) and decide["tool"] == "declare_insufficient_evidence", "deviated": bool(decide) and decide.get("follow") == "deviated",
            "orders": orders, "check": check[-1]["answer"] if check else None, "end": end[-1].get("reason") if end else None,
            "clock": rules.get("clock_mode") or "declared", "durations": rules.get("durations") or {},
            "run": next(r for r in payload["runs"] if r["folder"] == e["run"])}


def test_plain_sentences_are_true_for_every_episode(played, serve, tmp_path):
    python = browser_python()
    if python is None:
        pytest.skip("needs playwright with a browser, in this interpreter or the one EXPLORER_BROWSER_PYTHON names")
    runs, episodes, _ = played
    address = serve(runs, hidden=False)
    cases, payloads = {}, {}
    for name, episode in episodes.items():
        url = "http://%s:%d" % address + page_path(episode)
        payloads[name] = json.loads(get(address, page_path(episode) + "data/data.json")[1])
        cases[name] = {"url": url, "data": None}
    for how in ("matched", "not_matched", "declared", "no_repair", "max_steps", "horizon", "budget", "generation_time", "instant", "model_agent", "all_listed",
                "no_call", "thinking", "wall_cut", "priced_no_durations", "prices_no_budget", "free_no_budget",
                "older_ranking", "probe_failed", "cut_no_probe", "flush_no_cut", "no_call_after_reads"):
        name = f"made-up from bench_priced: {how}"
        payloads[name] = made_up(payloads["bench_priced"], how)
        path = tmp_path / f"{how}.json"
        path.write_text(json.dumps(payloads[name]))
        cases[name] = {"url": cases["bench_priced"]["url"], "data": str(path)}
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    script = tmp_path / "texts.py"
    script.write_text(TEXT_SCRIPT)
    done = subprocess.run([python, str(script), str(tmp_path / "cases.json")], capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    pages = json.loads(done.stdout.strip().splitlines()[-1])
    assert len(pages) >= 8
    words = lambda parts: [f"the {k}" for k in parts]
    for name, page in pages.items():
        f, s, chips = facts(payloads[name]), page["summary"]["sentence"], [t for _, t in page["summary"]["chips"]]
        assert not page["errors"] and page["story"] == s and page["chips"] == chips, (name, page)
        # the ranking: said only when one was asked, naming every faulty part of the cause it put first
        assert ("The CausalDiagnosis MCP" in s) == (f["top"] is not None), (name, s)
        if f["top"]:
            assert all(w in s.split(". ")[0] for w in words(f["top"]) or ["no faulty part"]), (name, s)
        # the decision: the cause named and "instead" exactly when it went against the advice
        if f["named"] is not None:
            assert all(w in s for w in words(f["named"]) or ["no faulty part"]) and (" instead" in s) == f["deviated"], (name, s)
        assert ("declared that the evidence was not enough" in s) == f["declared"], (name, s)
        assert ("named no cause" in s) == (f["named"] is None and not f["declared"]), (name, s)
        # the repair: said exactly when a work order was placed, never "replaced", since a fix may be a service
        assert ("ordered a repair of" in s) == bool(f["orders"]) and all(w in s for w in words(f["orders"])), (name, s)
        assert ("work order" in " ".join(chips)) == bool(f["orders"]) and " replaced " not in s, (name, s, chips)
        # the check: "verified" only says the check ran; matched or not comes from consistent alone
        c = f["check"]
        if c is None:
            assert "to compare the readings" not in s, (name, s)
        elif c["status"] == "verified":
            assert ("readings matched the prediction" in chips) == (c.get("consistent") is True), (name, chips)
            assert ("readings did not match the prediction" in chips) == (c.get("consistent") is False), (name, chips)
        elif c["status"].startswith("not yet observable: the work order is open"):
            assert "compared too early, the repair was not done" in chips and "nothing to compare yet" in s, (name, s)
        # the agent's request is a comparison, never a check, so it is not read as the verifier
        for bad in ("held after the repair", "confirmed it", "the prediction held", "not yet observable", "candidate", "check"):
            assert bad not in s and all(bad not in t for t in chips) and all(bad not in t for t in page["steps"]), (name, bad)
        # the end: a limit is named exactly when the episode ended by one
        limits = {"max_steps": "stopped at the step limit", "horizon": "stopped at the time limit", "budget": "ran out of budget", "failure": "stopped by a failure"}
        assert [t for t in chips if t in limits.values()] == ([limits[f["end"]]] if f["end"] in limits else []), (name, chips)
        # the line for causes a ranking leaves out: the count and the summed chance of its own more_candidates, and none
        # where it leaves none out or does not say
        ranked = {r["n"]: r for r in payloads[name]["episode"]["rankings"]}
        for n, line, text in page["notlisted"]:
            more = ranked[n].get("more_candidates") or {}
            left = more.get("count") or 0
            want = None if left <= 0 else f"{left} more possible cause{'' if left == 1 else 's'} {'is' if left == 1 else 'are'} not listed"
            assert line == want, (name, n, line, want)
            if want:
                # the page rounds with toFixed(3), half up on the exact value of the double
                chance = "< 0.001" if 0 < more["mass"] < 0.001 else str(Decimal(more["mass"]).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
                assert text == f"{want}Their summed chance is {chance}.", (name, n, text)
        # one grey tag per no_call step, keyed by the call it follows, its count label, and its line
        e = payloads[name]["episode"]
        idle = e.get("idle_steps") or []
        assert page["idle"] == [f"{x['after']}{'bcdefghijk'[i - [y['after'] for y in idle].index(x['after'])]} no tool" for i, x in enumerate(idle)], (name, page["idle"])
        assert [t for t in chips if t.endswith("without a tool")] == ([f"{len(idle)} {'turn' if len(idle) == 1 else 'turns'} without a tool"] if idle else []), (name, chips)
        if idle:
            known = sum(readings_served(c) for c in e["calls"] if c["n"] <= idle[0]["after"])
            knew = f"Knew {known} reading{'' if known == 1 else 's'}" if known else "Knew no reading yet"
            assert page["idlestory"] == [knew, "Advice no advice at this step", "Did called no tool", "Got back nothing, one step used"], (name, page["idlestory"])
        else:
            assert page["idlestory"] is None, (name, page["idlestory"])
        # the thinking label, its sentence and the row of a step on both clocks that charge thinking, the label after the other
        # labels: the label gives the seconds charged in all, and the sentence on the wall clock the credited seconds, which the
        # clock left out, on the token clock the tokens charged and the wall-clock time it does not count; the row of a step
        # gives its tokens on the token clock
        js = lambda v: (lambda r: str(int(r)) if r == int(r) else str(r))(math.floor((v or 0) * 10 + 0.5) / 10) + " s"
        if f["clock"] in ("generation_time", "generated_tokens"):
            k = e.get("clock") or {}
            credited = k.get("credited_seconds") or {}
            total = k.get("charged_seconds") or 0
            on_tokens = f["clock"] == "generated_tokens"
            said = (f"The clock counted {js(total)} of thinking for the {plural(k.get('charged_tokens') or 0, 'token')} the model generated up to the last step. "
                    "It counts tokens, not wall-clock time, so the time the model took to generate them, server starts, tool service and failed model requests did not count, "
                    "and neither did tokens generated after the last step.") if on_tokens else \
                f"The clock counted {js(total)} of thinking. {js((credited.get('starts') or 0) + (credited.get('calls') or 0))} of server starts and tool service did not count, and {js(k.get('failed_model_seconds'))} of failed model requests did not count."
            assert page["think"] == [f"thinking {js(total)}", said] and page["thinklast"] is True, (name, page["think"], page["thinklast"])
            assert page["thinkrows"] == [None if c.get("think_seconds") is None else
                                         f"{js(c['think_seconds'])}{' for ' + plural(c.get('think_tokens') or 0, 'token') if on_tokens else ''}, {plural(c.get('think_cycles') or 0, 'cycle')}"
                                         for c in e["calls"]], (name, page["thinkrows"])
        else:
            assert page["think"] is None and not any(page["thinkrows"]), (name, page["think"])
        # the cut line only for a cut at the wall limit, with its end in words; the folded block only where the run flushed
        # the cache or checked the model
        sv = e.get("served") or {}
        cut_words = {"wall": "the agent's time ran out", "service": "the model service or a tool did not answer"}
        assert ("was cut at the time limit" in s) == bool(sv.get("wall_limit")), (name, s)
        if sv.get("wall_limit"):
            end_t = e["status"]["time"] if e.get("status") else (e["calls"][-1]["time_after"] if e["calls"] else 0)
            assert s.endswith(f"The episode was cut at the time limit at cycle {end_t} because {cut_words.get(sv.get('cut_end'), 'the harness recorded no reason')}."), (name, s)
        if sv.get("cache_flush") or sv.get("cut_probe"):
            pr = sv.get("cut_probe")
            if pr and pr["ok"]:
                probe = f"{pr['cut_probes']} token{'' if pr['cut_probes'] == 1 else 's'} generated, answered in {js(pr['seconds'])}"
            elif pr:
                probe = "no token generated" + (f" ({pr['error']})" if pr.get("error") else "")
            else:
                probe = "none was made" if sv.get("wall_limit") else "no cut"
            assert page["served"] is not None and page["served"][2] == probe, (name, page["served"])
        else:
            assert page["served"] is None, (name, page["served"])
        # the last band of the Timeline tab shows money exactly when the run has a budget, whatever its axis
        assert page["lastband"] == (["Steps taken"] if e["rules"].get("budget") is None else ["Spending"]), (name, page["steps_axis"], page["lastband"])
        assert e.get("cost_unit") in ("money", "calls"), (name, e.get("cost_unit"))
        if e["rules"].get("budget") is None:
            want = "this run counts steps, not money" if e["cost_unit"] == "calls" else "this run has prices but no budget"
            assert [t for t in page["bandsub"]] and all(want.startswith(t) for t in page["bandsub"]), (name, page["bandsub"], want)
        if how_of(name) == "priced_no_durations":
            assert page["steps_axis"] is True, (name, page["steps_axis"])
        # the first label names the player: a model agent by design and model, else the scripted example or a policy
        run = f["run"]
        who = (run.get("agent_label") or run.get("agent")).replace("_", " ")
        want = f"{who} agent, model {run['model']}" if run.get("model") else "scripted example, no model" if run.get("agent") == "scripted" else f"{who} policy, no model"
        assert chips[0] == want, (name, chips[0], want)
        # the clock line names what moves the clock under the run's own clock mode and durations
        d, line = f["durations"], page["clock"]
        if f["clock"] == "instant":
            assert line == "In this run every step takes one cycle.", (name, line)
        elif f["clock"] in ("generation_time", "generated_tokens"):
            assert line.startswith("In this run the agent's thinking time moves the clock too"), (name, line)
        else:
            assert line.startswith("In this run thinking takes no time"), (name, line)
            assert ("reads" in line) == (d.get("read_per_cycle", 0) > 0), (name, line)
            assert ("repairs" in line) == any(v > 0 for v in [*(d.get("lead_time") or {}).values(), *(d.get("repair_downtime") or {}).values()]), (name, line)
        # one word for one thing on the page
        for t in [s, *chips, *page["steps"], *page["tags"]]:
            assert not re.search(r"\b(asset|component|healthy band|deviated|call to the asset)\b", t), (name, t)


# ----- the record panel keeps every value cell wide enough for its words -----
# the plain words of every value cell of the record panel that break across lines, with every folded block open, and the count
# of records of a second level the panel drew
RECORD_MEASURE = r"""() => {
  const panel = document.querySelector('.inspector-panel');
  if (!panel) return null;
  panel.querySelectorAll('details').forEach((d) => { d.open = true; });
  const broken = [];
  for (const dd of panel.querySelectorAll('dd')) {
    const walker = document.createTreeWalker(dd, NodeFilter.SHOW_TEXT);
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      for (const m of n.data.matchAll(/(?<=^|\s)[A-Za-z]{2,16}(?=[,.;:)]?(\s|$))/g)) {
        const range = document.createRange(); range.setStart(n, m.index); range.setEnd(n, m.index + m[0].length);
        if (new Set([...range.getClientRects()].filter((q) => q.width > 0).map((q) => Math.round(q.top))).size > 1) broken.push(m[0]);
      }
    }
  }
  return { broken, nested: panel.querySelectorAll('dl.rd dl.rd').length };
}"""
RECORD_SCRIPT = r"""
import json, sys
from playwright.sync_api import sync_playwright
case = json.loads(open(sys.argv[1]).read())
urls, MEASURE = case["urls"], case["measure"]
out = {}
with sync_playwright() as p:
    try:
        b = p.chromium.launch()
    except Exception:  # the browser this playwright bundles is not installed: the Chrome of the machine
        b = p.chromium.launch(channel="chrome")
    pg = b.new_page(viewport={"width": 1440, "height": 900})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    for name, url in urls.items():
        pg.goto(url); pg.wait_for_selector(".step-title"); pg.wait_for_timeout(300)
        steps, rows = pg.evaluate("() => ep().calls.length"), {}
        for step in range(1, steps + 1):
            if step > 1:
                pg.keyboard.press("ArrowRight"); pg.wait_for_timeout(150)
            rows[step] = pg.evaluate(MEASURE)
        out[name] = rows
    b.close()
print(json.dumps({"pages": out, "errors": errors}))
"""


def test_record_panel_values_wrap_by_words(played, serve, tmp_path):
    """The readable face of a record nests one list of fields in a value cell of another. Every value cell of the record panel
    keeps a width that its words fit in, at every level, on every step of every played fixture: no plain word breaks across
    lines. A key column as wide as its longest key left the values of the second level a few letters wide."""
    python = browser_python()
    if python is None:
        pytest.skip("needs playwright with a browser, in this interpreter or the one EXPLORER_BROWSER_PYTHON names")
    runs, episodes, _ = played
    address = serve(runs, hidden=False)
    urls = {name: "http://%s:%d" % address + page_path(episode) + f"#/ep/{episode.parent.name}/{episode.name}/decision/1"
            for name, episode in episodes.items()}
    (tmp_path / "urls.json").write_text(json.dumps({"urls": urls, "measure": RECORD_MEASURE}))
    script = tmp_path / "record.py"
    script.write_text(RECORD_SCRIPT)
    done = subprocess.run([python, str(script), str(tmp_path / "urls.json")], capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert not result["errors"], result["errors"]
    pages = result["pages"]
    assert set(pages) == set(episodes) and all(rows for rows in pages.values()), sorted(pages)
    broken = {(name, step): row["broken"][:5] for name, rows in pages.items() for step, row in rows.items() if row and row["broken"]}
    assert not broken, broken
    # the case the defect hit: records of a second level drawn in the panel
    assert sum(row["nested"] for rows in pages.values() for row in rows.values() if row) > 0



# ----- the levels of response_format, the profile at a shorter level and the board of the CausalDiagnosis agent -----
def has_levels(c):
    """Whether a call answers at a level of response_format: every tool of the CausalDiagnosis MCP and the profile of the task
    server of AssetOpsBench's rows."""
    return c["server"] == CAUSAL_SERVER or (c["server"] in METERED_SERVERS and c["tool"] == "get_asset_profile")


def test_every_answer_with_levels_carries_the_level_it_was_written_at(played, tmp_path):
    """The level of an answer is the response_format its call named, the default where it named none, and no other tool has
    one. Every answer of the CausalDiagnosis MCP stands at the four levels: the level the agent received as the text it
    received, the raw level as the server kept it beside the call or served it, and the other two as the level code of the
    server writes them from that raw answer. A ranking that keeps n_candidates ranks every level combination, the count the page
    uses where a shorter level leaves n_candidates out. The controls: a copy whose first call of the CausalDiagnosis MCP names
    another level takes that level, and a copy without the raw answers keeps the level the agent received alone."""
    _, episodes, _ = played
    seen, raw_counts, written = set(), 0, 0
    for name, episode in episodes.items():
        out = tmp_path / name
        e = build(episode, out)["episode"]
        released = json.loads((episode / "released.json").read_text())
        view = profile_view(dict(released["rules"], asset_id=released["asset_id"]))
        for c in e["calls"]:
            asked = c["args"].get("response_format") if isinstance(c["args"], dict) else None
            assert c["level"] == ((asked or DEFAULT_LEVEL) if has_levels(c) else None), (name, c["n"], c["tool"], c["level"], asked)
            seen.add(c["level"])
            if c["server"] != CAUSAL_SERVER or "error" in c["answer"]:
                assert "levels" not in c, (name, c["n"])
                continue
            assert c["levels"] == "recorded", (name, c["n"], c.get("levels"))
            levels = json.loads((out / "levels" / f"{c['n']}.json").read_text())
            assert levels["seen"] == c["level"] and list(levels["texts"]) == ["headline", "concise", "detailed", "raw"], (name, c["n"])
            assert levels["texts"][c["level"]] == c["answer_text"], (name, c["n"])
            raw = json.loads(levels["texts"]["raw"])
            for level in ("headline", "concise", "detailed"):
                if level != c["level"]:
                    assert levels["texts"][level] == answer_levels.text_of(answer_levels.at_level(c["tool"], raw, view, level)), (name, c["n"], level)
            written += 1
        for r in e["rankings"]:
            answer = next(c for c in e["calls"] if c["n"] == r["n"])["answer"]
            assert ("n_candidates" in answer) == (r["n_candidates"] is not None), (name, r["n"])
            if r["n_candidates"] is not None:
                raw_counts += 1
                assert r["n_candidates"] == e["n_combinations"], (name, r["n"], r["n_candidates"], e["n_combinations"])
    assert {"raw", "concise", None} <= seen and raw_counts and written, (seen, raw_counts, written)
    source = episodes["bench_priced"]
    copy = tmp_path / "copy" / source.parent.name
    shutil.copytree(source.parent, copy)
    rows = json.loads((copy / source.name / "trajectory.json").read_text())
    k = next(i for i, r in enumerate(rows) if r["server"] == CAUSAL_SERVER)
    rows[k]["tool_args"] = {**(rows[k]["tool_args"] or {}), "response_format": "headline"}
    (copy / source.name / "trajectory.json").write_text(json.dumps(rows))
    (copy / source.name / transcript.RAW_NAME).unlink()
    calls = build(copy / source.name, tmp_path / "copy_out")["episode"]["calls"]
    assert calls[k]["level"] == "headline"
    assert {c.get("levels") for c in calls if c["server"] == CAUSAL_SERVER} == {"not recorded"} and not (tmp_path / "copy_out" / "levels").exists()


def test_a_run_that_kept_no_raw_answer_recomputes_it_under_the_hidden_option_alone(played, tmp_path, serve):
    """A run folder that holds no raw answers of the CausalDiagnosis MCP: the public side shows the level the agent received alone,
    and the hidden side replays every call of the CausalDiagnosis MCP over the released evidence as it stood at the call, with
    the fits that leave out the hidden start, and writes the four levels the recorded raw answers give. The controls: a call whose
    line names evidence that no prefix of the events gives, and a run of an asset whose fits the replay does not hold, are not
    recomputed."""
    _, episodes, _ = played
    if "rig_priced" not in episodes:
        pytest.skip("the replay uses the fitted models of the rig, which need its data file")
    source = episodes["rig_priced"]
    kept = build(source, tmp_path / "kept")["episode"]["calls"]
    copy = tmp_path / "old" / source.parent.name
    shutil.copytree(source.parent, copy)
    folder = copy / source.name
    (folder / transcript.RAW_NAME).unlink()
    public = build(folder, tmp_path / "public")
    assert {c.get("levels") for c in public["episode"]["calls"] if c["server"] == CAUSAL_SERVER} == {"not recorded"}
    assert not (tmp_path / "public" / "hidden_levels").exists() and not (tmp_path / "public" / "hidden.json").exists()
    build(folder, tmp_path / "hidden", hidden=True)
    recomputed = json.loads((tmp_path / "hidden" / "hidden.json").read_text())["levels"]
    asked = [c for c in kept if c.get("levels") == "recorded" and c["level"] != "raw"]
    assert asked and recomputed == {str(c["n"]): "recomputed" for c in asked}
    for c in asked:
        again = json.loads((tmp_path / "hidden" / "hidden_levels" / f"{c['n']}.json").read_text())
        assert again["source"] == "recomputed" and again["texts"] == json.loads((tmp_path / "kept" / "levels" / f"{c['n']}.json").read_text())["texts"], c["n"]
    public_old, people_old = serve(tmp_path / "old", hidden=False), serve(tmp_path / "old", hidden=True)
    for c in asked:
        route = page_path(folder) + f"data/hidden_levels/{c['n']}.json"
        assert get(public_old, route)[0] == 404 and get(people_old, route)[0] == 200, c["n"]
    lines = [json.loads(line) for line in (folder / "transcript.jsonl").read_text().splitlines()]
    first = next(line for line in lines if line.get("kind") == "call" and line.get("server") == CAUSAL_SERVER)
    first["released_sha256"] = "0" * 64
    (folder / "transcript.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    build(folder, tmp_path / "tampered", hidden=True)
    tampered = json.loads((tmp_path / "tampered" / "hidden.json").read_text())["levels"]
    assert tampered[str(asked[0]["n"])] == "not recomputed" and list(tampered.values()).count("recomputed") == len(asked) - 1
    bench = tmp_path / "bench" / episodes["bench_priced"].parent.name
    shutil.copytree(episodes["bench_priced"].parent, bench)
    (bench / "seed000000" / transcript.RAW_NAME).unlink()
    build(bench / "seed000000", tmp_path / "bench_out", hidden=True)
    assert set(json.loads((tmp_path / "bench_out" / "hidden.json").read_text())["levels"].values()) == {"not recomputed"}


def test_a_profile_read_at_a_shorter_level_carries_its_raw_form(played, tmp_path):
    """The scripted plan on AssetOpsBench's servers reads the profile at the default level concise, whose tool table and normal
    ranges are long tables. The call carries the raw form that profile_levels.raw_of rebuilds, whose normal ranges are those the
    episode released, one entry per field, and whose tools are the rows of the long table. The control: the rig's own profile
    has no levels and carries no raw form."""
    _, episodes, _ = played
    e = build(episodes["bench_aob_base"], tmp_path / "aob")["episode"]
    profile = next(c for c in e["calls"] if c["tool"] == "get_asset_profile")
    result = profile["answer"]["result"]
    assert profile["level"] == "concise" and set(result["tools"]) == set(result["normal_ranges"]) == {"columns", "rows"}
    rebuilt = profile["profile_raw"]
    assert rebuilt == raw_of(result)
    bands = lambda ranges: sorted((band["low"], band["high"]) for band in ranges.values())  # noqa: E731
    assert len(rebuilt["normal_ranges"]) == len(result["normal_ranges"]["rows"]) and bands(rebuilt["normal_ranges"]) == bands(e["rules"]["normal_ranges"])
    names = result["tools"]["columns"].index("name")
    assert [t["name"] for t in rebuilt["tools"]] == [row[names] for row in result["tools"]["rows"]]
    rig = build(episodes["bench_priced"], tmp_path / "rig")["episode"]
    assert all(c["level"] is None and "profile_raw" not in c for c in rig["calls"] if c["tool"] == "get_asset_profile")


def board_lines(sections):
    """The lines of a board as its sections give them back, a table without the rule under its header."""
    out = []
    for sec in sections:
        out.append("#" * sec["level"] + " " + sec["heading"])
        for block in sec["blocks"]:
            if "table" in block:
                out += ["| " + " | ".join(row) + " |" for row in [block["table"]["header"], *block["table"]["rows"]]]
            elif "items" in block:
                out += [f"- {item}" for item in block["items"]]
            else:
                out.append(block["text"])
    return out


def test_the_board_of_every_plan_is_the_board_its_planning_request_held(played, tmp_path):
    """Every plan the model of the CausalDiagnosis agent wrote has the board of its planning request: its sections give back every
    line of the board in order, every row of a table has the cells of its header, and every call names the board of the plan the
    walk places it on. The control: an episode of the scripted agent has no board."""
    _, episodes, _ = played
    episode = episodes["bench_causal"]
    e = build(episode, tmp_path / "causal")["episode"]
    llm = json.loads((episode / "llm_calls.json").read_text())
    plans = json.loads((episode / "plans.json").read_text())["plans"]
    assert e["boards"] and [b["plan"] for b in e["boards"] if not b["retry"]] == [p["index"] for p in plans]
    for b in e["boards"]:
        prompt = llm[b["llm_index"]]["prompt"]
        assert PLANNING_HEADING in prompt
        board = prompt[prompt.index("\n# ", prompt.index(PLANNING_HEADING)) + 1:prompt.rindex("\n\nObjective")]
        lines = [line for line in board.split("\n") if line.strip() and set(line.strip()) - set("|-: ")]
        assert board_lines(b["sections"]) == lines, b["plan"]
        tables = [block["table"] for sec in b["sections"] for block in sec["blocks"] if "table" in block]
        assert all(len(row) == len(table["header"]) for table in tables for row in table["rows"]), b["plan"]
        assert tables or b["plan"] == 1, b["plan"]  # the board before the profile is read holds no table
    assert all(c.get("board") is not None for c in e["calls"]), [c["n"] for c in e["calls"] if c.get("board") is None]
    scripted = build(episodes["bench_priced"], tmp_path / "scripted")["episode"]
    assert scripted["boards"] == [] and all("board" not in c for c in scripted["calls"])


LEVEL_SCRIPT = r'''
import json, sys
from playwright.sync_api import sync_playwright
case = json.loads(open(sys.argv[1]).read())
out, errors = {}, []
with sync_playwright() as p:
    try:
        b = p.chromium.launch()
    except Exception:  # the browser this playwright bundles is not installed: the Chrome of the machine
        b = p.chromium.launch(channel="chrome")
    for name, page in case.items():
        pg = b.new_page(viewport={"width": 1440, "height": 900})
        pg.on("pageerror", lambda e: errors.append(str(e)))
        if page["data"]:
            body = open(page["data"]).read()
            pg.route("**/data/data.json", lambda route, request, body=body: route.fulfill(body=body, content_type="application/json"))
        pg.goto(page["url"]); pg.wait_for_selector(".step-title"); pg.wait_for_timeout(300)
        if page.get("hidden"):
            pg.click("#hidden-toggle"); pg.wait_for_function("() => document.body.classList.contains('show-hidden')"); pg.wait_for_timeout(300)
        rows = {}
        for step in range(1, pg.evaluate("() => ep().calls.length") + 1):
            if step > 1:
                pg.keyboard.press("ArrowRight"); pg.wait_for_timeout(120)
            row = pg.evaluate("""() => { const dt = [...document.querySelectorAll('dt')].find((x) => x.textContent === 'level of the answer'); const dd = dt ? dt.nextElementSibling : null;
              return { level: dd ? dd.querySelector('span').textContent : null, buttons: dd ? [...dd.querySelectorAll('.lv-seg button')].map((x) => x.textContent) : [],
                       note: dd && dd.querySelector('.note') ? dd.querySelector('.note').textContent : null,
                       board: (document.querySelector('.board-line') || {}).textContent || null }; }""")
            row["opened"] = []
            for k in range(len(row["buttons"])):
                pg.locator(".lv-seg button").nth(k).click(); pg.wait_for_timeout(120)
                row["opened"].append(pg.evaluate("() => [document.querySelector('#raw-inspector .insp-title').firstChild.textContent, (document.querySelector('#raw-inspector .insp-note') || {}).textContent || null]"))
            if row["board"]:
                pg.click(".board-line"); pg.wait_for_timeout(200)
                row["sections"] = pg.evaluate("() => document.querySelectorAll('#raw-inspector .bd-sec').length")
                row["marked"] = pg.evaluate("() => [...document.querySelectorAll('#raw-inspector td.bd-out')].map((x) => x.textContent)")
            rows[step] = row
        out[name] = rows
        pg.close()
    b.close()
print(json.dumps({"pages": out, "errors": errors}))
'''


def test_the_page_shows_every_answer_at_its_four_levels_and_opens_the_board(played, serve, tmp_path):
    """On the page, the card of what came back names the level of every answer that has one, in the words of the four levels,
    and for every answer of the CausalDiagnosis MCP offers the four levels, the one the agent received marked as seen; each opens
    in the record panel with the note of where it came from. A step of the CausalDiagnosis agent opens the board of its plan in
    its sections, with a value that ends in low or high marked. Made-up cases add a ranking at headline and an answer at
    detailed, which no fixture plays, and a value outside its normal range on a board."""
    python = browser_python()
    if python is None:
        pytest.skip("needs playwright with a browser, in this interpreter or the one EXPLORER_BROWSER_PYTHON names")
    runs, episodes, _ = played
    address = serve(runs, hidden=False)
    url = lambda name: "http://%s:%d" % address + page_path(episodes[name])  # noqa: E731
    payloads = {name: json.loads(get(address, page_path(episodes[name]) + "data/data.json")[1]) for name in ("bench_priced", "bench_causal", "bench_aob_base")}
    made = json.loads(json.dumps(payloads["bench_priced"]))
    calls = made["episode"]["calls"]
    rank = next(c for c in calls if c["tool"] == mcp_metrics.RANK)
    a = rank["answer"]
    rank.update(level="headline", levels=None, args={**rank["args"], "response_format": "headline"},
                answer={"headline": a.get("message") or "made-up headline", mcp_metrics.READ_FIRST: a.get(mcp_metrics.READ_FIRST),
                        "separating_test": a.get("separating_test"), "belief_time": a.get("belief_time")})
    made["episode"]["rankings"] = [r for r in made["episode"]["rankings"] if r["n"] != rank["n"]]
    other = next(c for c in calls if c["server"] == CAUSAL_SERVER and c["tool"] != mcp_metrics.RANK)
    other.update(level="detailed", levels=None, args={**other["args"], "response_format": "detailed"})
    board = json.loads(json.dumps(payloads["bench_causal"]))
    board["episode"]["boards"][-1]["sections"][-1]["blocks"].append({"table": {"header": ["id", "value"], "rows": [["obs/9/made_up_mean", "31.05 high"]]}})
    cases = {name: {"url": url(name), "data": None} for name in payloads}
    for name, payload, source in (("made-up levels", made, "bench_priced"), ("made-up board", board, "bench_causal")):
        path = tmp_path / f"{name.replace(' ', '_')}.json"
        path.write_text(json.dumps(payload))
        cases[name] = {"url": url(source), "data": str(path)}
        payloads[name] = payload
    old = None
    if "rig_priced" in episodes:  # a run folder without the raw record: the page says why it shows one level, and recomputes under --hidden
        source = episodes["rig_priced"]
        shutil.copytree(source.parent, tmp_path / "old_runs" / source.parent.name)
        old = tmp_path / "old_runs" / source.parent.name / source.name
        (old / transcript.RAW_NAME).unlink()
        public_old, people_old = serve(tmp_path / "old_runs", hidden=False), serve(tmp_path / "old_runs", hidden=True)
        old_payload = json.loads(get(public_old, page_path(old) + "data/data.json")[1])
        recomputed = json.loads(get(people_old, page_path(old) + "data/hidden.json")[1])["levels"]
        cases["old run"] = {"url": "http://%s:%d" % public_old + page_path(old), "data": None}
        cases["old run, ground truth served"] = {"url": "http://%s:%d" % people_old + page_path(old), "data": None}
        cases["old run, ground truth shown"] = {"url": "http://%s:%d" % people_old + page_path(old), "data": None, "hidden": True}
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    script = tmp_path / "levels.py"
    script.write_text(LEVEL_SCRIPT)
    done = subprocess.run([python, str(script), str(tmp_path / "cases.json")], capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stderr[-2000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert not result["errors"], result["errors"]
    words = {"headline": "headline, one line", "concise": "concise, what an agent acts on", "detailed": "detailed, every value in compact form", "raw": "raw, the whole answer"}
    order = ["headline", "concise", "detailed", "raw"]
    shown, offered = set(), 0
    for name, rows in result["pages"].items():
        if name.startswith("old run"):
            continue
        e = payloads[name]["episode"]
        assert len(rows) == len(e["calls"]), name
        for c in e["calls"]:
            row = rows[str(c["n"])]
            if c["level"] is None:
                assert row["level"] is None and row["buttons"] == [], (name, c["n"], row)
                continue
            assert row["level"].startswith(words[c["level"]]), (name, c["n"], row["level"])
            assert row["level"].endswith(", the default") == ("response_format" not in c["args"]) or "unchanged since" in row["level"], (name, c["n"])
            shown.add(c["level"])
            if c.get("levels") == "recorded":
                assert row["buttons"] == [f"{lv}, seen" if lv == c["level"] else lv for lv in order], (name, c["n"], row["buttons"])
                assert [title for title, _note in row["opened"]] == [f"Step {c['n']}, the answer at the level {lv}" for lv in order], (name, c["n"])
                notes = [note for _title, note in row["opened"]]
                assert notes[order.index(c["level"])] == "The agent received the answer at this level.", (name, c["n"], notes)
                assert all(note.startswith("The page wrote this level from the raw answer") and note.endswith("The agent did not see it.")
                           for k, note in enumerate(notes) if order[k] != c["level"]), (name, c["n"], notes)
                offered += 1
            else:  # a call of the CausalDiagnosis MCP without levels says why; the profile of the task server opens its raw form instead
                assert row["buttons"] == [] and bool(row["note"]) == bool(c.get("levels")), (name, c["n"], row)
            if c.get("board") is not None:
                b = e["boards"][c["board"]]
                assert row["board"] and f"plan {b['plan']}" in row["board"] and row["sections"] == len(b["sections"]), (name, c["n"], row)
                cells = [cell for sec in b["sections"] for block in sec["blocks"] if "table" in block for row_cells in block["table"]["rows"] for cell in row_cells]
                assert row["marked"] == [cell for cell in cells if re.search(r"\s(low|high)$", cell)], (name, c["n"], row["marked"])
            else:
                assert row["board"] is None, (name, c["n"])
    assert shown == set(words) and offered, (shown, offered)
    assert any("31.05 high" in r.get("marked", []) for r in result["pages"]["made-up board"].values())
    if old is None:
        return
    notes_of = {"old run": "The run kept no raw answer of this call, so the page shows the level the agent received alone. A viewer started with the hidden option recomputes it.",
                "old run, ground truth served": "The run kept no raw answer of this call. With the ground truth shown, the page recomputes it from the hidden start."}
    missing = [c for c in old_payload["episode"]["calls"] if c.get("levels") == "not recorded"]
    assert missing and set(recomputed) == {str(c["n"]) for c in missing} and "recomputed" in recomputed.values(), recomputed
    for c in missing:
        for name, note in notes_of.items():
            row = result["pages"][name][str(c["n"])]
            assert row["buttons"] == [] and row["note"] == note, (name, c["n"], row)
        row = result["pages"]["old run, ground truth shown"][str(c["n"])]
        if recomputed[str(c["n"])] == "recomputed":
            assert row["buttons"] == [f"{lv}, seen" if lv == c["level"] else lv for lv in order] and row["note"] is None, (c["n"], row)
            notes = [note for _title, note in row["opened"]]
            assert notes[order.index(c["level"])] == "The agent received the answer at this level.", (c["n"], notes)
            assert all(note == "The page wrote this level from the raw answer that the page recomputed with the hidden option, since the run kept none, "
                       "with the level code of the CausalDiagnosis MCP. The agent did not see it." for k, note in enumerate(notes) if order[k] != c["level"]), (c["n"], notes)
        else:
            assert row["buttons"] == [] and row["note"].startswith("The run kept no raw answer of this call, and the replay with the hidden option"), (c["n"], row)


# ----- a run folder renamed away from the label of its run.json -----
def renamed_runs(episode, tmp_path):
    """A runs folder that holds the run folder of an episode and a copy of it under another name, which keeps the label of its
    run.json."""
    runs = tmp_path / "renamed_runs"
    shutil.copytree(episode.parent, runs / episode.parent.name)
    shutil.copytree(episode.parent, runs / "renamed_copy")
    return runs


def test_every_run_of_the_page_data_carries_its_folder(played, tmp_path):
    """Every run of data.json carries the name of its folder beside the label of its run.json, so the page tells a run folder
    from a copy of it under another name."""
    _, episodes, _ = played
    episode = episodes["bench_priced"]
    runs = renamed_runs(episode, tmp_path)
    label = json.loads((episode.parent / "run.json").read_text())["label"]
    for folder in (episode.parent.name, "renamed_copy"):
        payload = build(runs / folder / episode.name, tmp_path / folder)
        assert payload["episode"]["run"] == folder == payload["runs"][0]["folder"], folder
        assert sorted((r["folder"], r["label"]) for r in payload["runs"]) == sorted([(episode.parent.name, label), ("renamed_copy", label)]), folder


RENAMED_SCRIPT = r'''
import json, sys
from playwright.sync_api import sync_playwright
urls, episode = json.loads(open(sys.argv[1]).read()), sys.argv[2]
pages, errors = {}, []
with sync_playwright() as p:
    try:
        b = p.chromium.launch()
    except Exception:  # the browser this playwright bundles is not installed: the Chrome of the machine
        b = p.chromium.launch(channel="chrome")
    for folder, url in urls.items():
        pg = b.new_page(viewport={"width": 1440, "height": 900})
        pg.on("pageerror", lambda e: errors.append(str(e)))
        def route(to):
            pg.evaluate("(h) => { location.hash = h; }", to); pg.wait_for_function("(h) => location.hash === h", arg=to); pg.wait_for_timeout(300)
        pg.goto(url + f"#/ep/{folder}/{episode}/decision/1"); pg.wait_for_selector(".ovb-step"); pg.wait_for_timeout(300)
        row = {"facts": pg.inner_text(".ep-head .facts"), "player": pg.evaluate("() => playerWords(ep())"), "decision": pg.inner_text("#app")}
        route(f"#/run/{folder}")
        row["title"] = pg.evaluate("() => (document.querySelector('h1.page-title') || {}).textContent || null")
        row["links"] = pg.evaluate("() => [...document.querySelectorAll('#app a[href^=\"#/ep/\"]')].map((a) => decodeURIComponent(a.getAttribute('href')))")
        # the cell of the episode the page shows links its step count; it stands on the run page of its own folder alone
        row["here"] = {}
        for other in urls:
            route(f"#/run/{other}")
            row["here"][other] = pg.evaluate("() => document.querySelectorAll('.ov-grid a.c-sub').length")
        row["rows"] = []
        route("#/")
        for k in range(pg.locator("table.grid tbody tr").count()):
            route("#/")
            pg.locator("table.grid tbody tr").nth(k).click(); pg.wait_for_timeout(300)
            row["rows"].append(pg.evaluate("decodeURIComponent(location.hash)"))
        pages[folder] = row
        pg.close()
    b.close()
print(json.dumps({"pages": pages, "errors": errors}))
'''


def test_the_page_of_a_renamed_run_folder_keeps_the_facts_of_its_run(played, serve, tmp_path):
    """A run folder renamed away from the label of its run.json, beside the folder it was copied from: the episode page names the
    agent, the setting and the CausalDiagnosis MCP of its run, the run page links its episodes under its folder, only the run page
    of its own folder marks the episode the page shows, and each row of the run list opens its own folder."""
    python = browser_python()
    if python is None:
        pytest.skip("needs playwright with a browser, in this interpreter or the one EXPLORER_BROWSER_PYTHON names")
    _, episodes, _ = played
    episode = episodes["bench_priced"]
    run = json.loads((episode.parent / "run.json").read_text())
    assert run["diagnostics_server"], "the run has no CausalDiagnosis MCP, so the advice card would prove nothing"
    folders = [episode.parent.name, "renamed_copy"]
    address = serve(renamed_runs(episode, tmp_path), hidden=False)
    (tmp_path / "urls.json").write_text(json.dumps({f: "http://%s:%d" % address + f"/e/{f}/{episode.name}/" for f in folders}))
    script = tmp_path / "renamed.py"
    script.write_text(RENAMED_SCRIPT)
    done = subprocess.run([python, str(script), str(tmp_path / "urls.json"), episode.name], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert not result["errors"], result["errors"]
    for folder in folders:
        page = result["pages"][folder]
        assert f"agent {run.get('agent_label') or run['agent']}" in page["facts"] and f"setting {run['config_name']}" in page["facts"], (folder, page["facts"])
        assert "undefined" not in page["facts"] + page["player"] and "has no CausalDiagnosis MCP" not in page["decision"], (folder, page["facts"], page["player"])
        assert page["title"] == run["label"] and page["links"] and all(h.startswith(f"#/ep/{folder}/") for h in page["links"]), (folder, page["title"], page["links"])
        assert page["here"] == {f: int(f == folder) for f in folders}, (folder, page["here"])
        assert sorted(page["rows"]) == sorted(f"#/run/{f}" for f in folders), (folder, page["rows"])
