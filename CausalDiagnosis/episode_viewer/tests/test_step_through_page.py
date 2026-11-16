"""The step-through page of one episode: a scripted episode that the driver writes and episodes of the action server played
under a recorded clock, all under pytest's tmp_path, outside the repository. The page holds one step per tool call in order,
keeps every hidden value inside the containers that its toggle shows, which is off when the page opens, references nothing
on the network, closes every element it opens, places the thinking charge of a call by the transcript of the servers and is
never written inside the fork or the episode folder. The tests assert relations between the page and the episode files,
never a recorded value."""
import dataclasses
import importlib.util
import json
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

import pytest

from CausalDiagnosis.experiments.episode_runners.run_folder_layout import HIDDEN, STATE_FILE
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import ROOT
from CausalDiagnosis.utils.json_files import read_json
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS, HEALTHY, default_facts_path
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers import layer_2_real_time as real_time

if not Path(default_facts_path()).exists():
    pytest.skip("hydraulic rig facts file not present, set HYD_FACTS_PATH", allow_module_level=True)
if importlib.util.find_spec("mcp") is None:
    pytest.skip("needs the mcp package", allow_module_level=True)

from CausalDiagnosis.episode_viewer import step_through_page as viewer  # noqa: E402
from CausalDiagnosis.episode_viewer import walkthrough  # noqa: E402
from CausalDiagnosis.mcp_servers.utils import transcript  # noqa: E402
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode  # noqa: E402
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP.main import build_server  # noqa: E402
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP.tests.test_transcript import ask  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.tests.test_run_episodes import _run  # noqa: E402

VOID = {"meta", "input", "br", "img", "hr", "link"}
CHARGED = next(m for m in real_time.CLOCK_MODES if real_time.thinking(SimpleNamespace(clock_mode=m), 1))  # a mode that charges thinking
UNCHARGED = next(m for m in real_time.CLOCK_MODES if not real_time.thinking(SimpleNamespace(clock_mode=m), 1))
READ = ("read_sensors", {"sensor_group": "tier1", "cycles": 1})
CALLS = [READ, READ, READ, READ, ("submit_diagnosis", {**HEALTHY, "evidence_ids": ""}), ("finish", {"summary": "done"})]
CHARGE = re.compile(r"<dt>Thinking time</dt><dd>([\d.]+) s of this call plus ([\d.]+) s carried in, at ([\d.]+) s per cycle, make "
                    r"(\d+) cycles? with ([\d.]+) s carried on")
CELL = re.compile(r"Thinking charged before the call<b>([^<]*)</b>")


class Page(HTMLParser):
    """The text of a page split into what lies inside a hidden-side container and what lies outside, the tool named in
    the heading of every step, and every problem of nesting."""

    def __init__(self, text):
        super().__init__()
        self.stack, self.problems, self.inside, self.outside, self.step_tools, self.inputs = [], [], [], [], [], []
        self.hidden_depth = None
        self.feed(text)
        self.close()
        if self.stack:
            self.problems.append(f"unclosed {self.stack}")

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "input":
            self.inputs.append(attrs)
        if tag in VOID:
            return
        self.stack.append(tag)
        if self.hidden_depth is None and viewer.HIDDEN_CLASS in (attrs.get("class") or "").split():
            self.hidden_depth = len(self.stack)
        if tag == "section" and "step" in (attrs.get("class") or "").split():
            self.step_tools.append(None)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.problems.append(f"</{tag}> closes {self.stack[-1:]}")
            return
        if self.hidden_depth == len(self.stack):
            self.hidden_depth = None
        self.stack.pop()

    def handle_data(self, data):
        (self.inside if self.hidden_depth is not None else self.outside).append(data)
        if self.stack[-2:] == ["h2", "code"] and self.step_tools and self.step_tools[-1] is None:
            self.step_tools[-1] = data


def slides(text):
    """The HTML of every step of a page, in order."""
    return text.split('<section class="step"')[1:]


def scripted_episode(tmp_path):
    run_dir, rows, _meta = _run(["--config", "static", "--agent", "scripted", "--seeds", "1"], tmp_path / "scripted")
    assert rows[0]["error"] == "", rows[0]["error"]
    return run_dir, run_dir / "seed000000"


def served_episode(folder, gap, clock_mode=CHARGED):
    """An episode folder of the action server played through the handler that the MCP package calls for a client, under a
    recorded clock that moves gap seconds per call, in a clock mode of the engine."""
    folder.mkdir(parents=True)
    path = folder / STATE_FILE
    with episode.recorded_clock(100.0, 0.0):
        episode.start(path, 3, dataclasses.replace(EpisodeConfig(), clock_mode=clock_mode), default_facts_path())
        episode.hand_over(path)
    app = build_server(path)
    for k, (tool, arguments) in enumerate(CALLS, start=1):
        with episode.recorded_clock(100.0 + k * gap, k * gap):
            ask(app, tool, arguments)
    return folder


def test_a_scripted_episode_page_has_one_step_per_call_and_keeps_the_hidden_side_inside_its_toggle(tmp_path):
    run_dir, ep_dir = scripted_episode(tmp_path)
    out = tmp_path / "page.html"
    viewer.main([str(ep_dir), "--out", str(out)])
    text = out.read_text(encoding="utf-8")
    page = Page(text)
    assert text.startswith("<!DOCTYPE html>") and not page.problems, page.problems
    # one step per tool call, in the order of the trajectory
    tools = [t["tool"] for t in read_json(ep_dir / "trajectory.json")]
    assert page.step_tools == tools
    # self-contained: no network reference, the style and the script inline
    assert "http://" not in text and "https://" not in text and "<link" not in text and "src=" not in text
    # the hidden start sits inside the toggle's containers, and the page outside them reads the same without the hidden
    # folder, so no hidden value reaches the default view, while the toggle is off when the page opens
    start = walkthrough.long_levels(read_json(ep_dir / HIDDEN / "oracle.json")["diagnosis"], COMPONENTS)
    assert f"start combination {start}" in "".join(page.inside)
    bare = tmp_path / "bare" / run_dir.name / ep_dir.name
    shutil.copytree(ep_dir, bare, ignore=shutil.ignore_patterns(HIDDEN))
    for name in ("run.json", "scores.csv"):
        shutil.copy(run_dir / name, bare.parent / name)
    bare_page = Page(viewer.render(bare))
    assert not bare_page.problems and bare_page.outside == page.outside
    assert start not in "".join(bare_page.inside)
    toggle = next(i for i in page.inputs if i.get("id") == "show-hidden")
    assert "checked" not in toggle and "show-hidden" not in text.split("<body", 1)[1].split(">", 1)[0]
    assert f".{viewer.HIDDEN_CLASS} {{ display: none;" in text and viewer.PEOPLE_ONLY in "".join(page.outside)
    # the page is written where --out says and never into the episode folder, whose files the run ledger hashes
    assert not list(ep_dir.rglob("*.html"))


def test_the_page_says_no_thinking_charge_exactly_where_the_engine_charges_none():
    charge = {"seconds": 3.0, "carried_in": 0.0, "cycles": 0, "carried_out": 3.0, "recorded": 0}
    charging = set()
    for mode in real_time.CLOCK_MODES:
        config = dataclasses.replace(EpisodeConfig(), clock_mode=mode)
        charging.add(real_time.thinking(config, 1) != 0)
        assert viewer.thinking_text(charge, config).startswith("none") == (real_time.thinking(config, 1) == 0), mode
    assert charging == {True, False}  # the engine holds a mode of either kind, so both sides of the rule are read


def test_the_charges_of_an_episode_that_charges_thinking_time_add_up_slide_by_slide(tmp_path):
    folder = served_episode(tmp_path / "charged", gap=7.3)
    state = read_json(folder / STATE_FILE)
    text = viewer.render(folder)
    charges = [CHARGE.search(slide) for slide in slides(text)]
    assert all(charges) and len(charges) == len(CALLS) == len(state["think_cycles"])
    for match, recorded in zip(charges, state["think_cycles"]):
        seconds, carried_in, per_cycle, cycles, carried_on = (float(v) for v in match.groups())
        # every number is printed with two decimals, so the two sides agree within the rounding of three of them
        assert seconds + carried_in == pytest.approx(cycles * per_cycle + carried_on, abs=0.015) and cycles == recorded
    assert max(state["think_cycles"]) >= 1  # at least one slide charges a cycle
    assert [CELL.search(slide).group(1) for slide in slides(text)] == [walkthrough.plural(c, "cycle") for c in state["think_cycles"]]


def test_the_thinking_seconds_add_up_where_a_rounded_total_would_reach_the_seconds_per_cycle():
    config = SimpleNamespace(clock_mode=CHARGED, think_seconds_per_cycle=4)
    charges = viewer.charge_sequence(config, [0, 1], [3.996, 0.61])
    assert [viewer.thinking_text(c, config) for c in charges] == [
        "4.00 s of this call plus 0.00 s carried in, at 4 s per cycle, make 0 cycles with 4.00 s carried on",
        "0.61 s of this call plus 4.00 s carried in, at 4 s per cycle, make 1 cycle with 0.61 s carried on"]
    plain = SimpleNamespace(clock_mode=UNCHARGED, think_seconds_per_cycle=4)
    assert viewer.thinking_text(viewer.charge_sequence(plain, [0], [0.0])[0], plain) == (
        f"none, the {UNCHARGED} clock mode charges no thinking time")


def test_one_call_that_never_reached_a_server_leaves_the_other_slides_placed(tmp_path):
    run_dir, ep_dir = scripted_episode(tmp_path)
    trajectory = read_json(ep_dir / "trajectory.json")
    first = next(i for i, step in enumerate(trajectory) if step.get("server") == "hydraulic_rig")
    failed = {**trajectory[first], "task": "a call that failed in the client", "response": "", "error": "the connection closed"}
    (ep_dir / "trajectory.json").write_text(json.dumps(trajectory[:first + 1] + [failed] + trajectory[first + 1:]), encoding="utf-8")
    recorded = read_json(ep_dir / STATE_FILE)["think_cycles"]
    pages = slides(viewer.render(ep_dir))
    cells = [CELL.search(slide) for slide in pages]
    placed = [cell.group(1) for k, cell in enumerate(cells) if cell and k != first + 1]
    assert cells[first + 1].group(1) == "not placed" and viewer.NOT_PLACED in pages[first + 1]
    assert placed == [walkthrough.plural(c, "cycle") for c in recorded]
    # a copy without the transcript places by the count of the calls, which the failed call breaks for every slide
    copy = tmp_path / "copy" / run_dir.name / ep_dir.name
    shutil.copytree(ep_dir, copy, ignore=shutil.ignore_patterns(transcript.NAME))
    text = viewer.render(copy)
    assert viewer.BEFORE_TRANSCRIPT in text and {cell.group(1) for cell in (CELL.search(s) for s in slides(text)) if cell} == {"not placed"}


def test_the_page_is_refused_inside_the_fork_and_inside_the_episode_folder(tmp_path):
    folder = served_episode(tmp_path / "plain", gap=1.0, clock_mode=UNCHARGED)
    for out in (ROOT / "CausalDiagnosis" / "episode_viewer" / "page.html", folder / "page.html", folder / "deeper" / "page.html"):
        with pytest.raises(SystemExit, match="lies inside"):
            viewer.main([str(folder), "--out", str(out)])
        assert not out.exists()
    assert not (folder / "deeper").exists()
