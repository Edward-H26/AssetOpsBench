"""The texts of the planning call that the CausalDiagnosis agent shares with the general agent (agents/utils/core_texts.py) and
how this agent prints them: the shared file types no word of a kind and no name of an asset, every one of its constants is printed,
no slot stays open in a rendered prompt, every new reason has its sentence in the prompt, the planning turn names the steps that
made no call, the rewordings for a kind a row does not serve still find their text, the new texts on the tiny asset name its
sensor groups and no name of another asset, and the static part of the planning prompt stays the same from turn to turn of an
episode, so that the prefix cache of the serving engine holds."""
import ast
import json
import re
from pathlib import Path

import pytest

from CausalDiagnosis.agents.Causal_Diagnosis_Agent import prompts
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.prompts import (PLANNING_HEADING, SYSTEM_PROMPT, SYSTEM_PROMPT_WITHOUT_SERVER, WAIT_HEAD, WITHOUT_KIND,
                                                                  planning_turn, system_prompt)
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board import Board
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board.board import KIND_WORDS
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tests.test_causal_diagnosis_agent import ScriptedPlanner, _aob_run, _play_without_server, _synthetic_board, bands_plans
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tests.test_parallel_exploration import ASSETS_DIR, TINY_ASSET, asset_profile
from CausalDiagnosis.agents.utils import core_texts
from CausalDiagnosis.frameworks.AssetOpsBench.agents.causal_diagnosis_agent_surface import AobSurface
from CausalDiagnosis.utils.paths import ROOT

AGENT_FOLDER = ROOT / "CausalDiagnosis" / "agents" / "Causal_Diagnosis_Agent"
SHARED = {name: value for name, value in vars(core_texts).items() if name.isupper() and isinstance(value, str)}
SLOT = re.compile(r"\{[a-z_]+\}")
# the words of the kind recorded levels, its nouns, its identifier forms and the names of its step kinds, as whole words; a longer
# phrase stands before the word it holds, so that each match is counted once
KIND_WORD_PATTERNS = [
    ("sensor group", r"\bsensor[ _]groups?\b"), ("group", r"\bgroups?\b"), ("component", r"\bcomponents?\b"), ("cycle", r"\bcycles?\b"),
    ("horizon", r"\bhorizon\b"), ("work order", r"\bwork[ _]orders?\b"), ("order", r"\borders?\b"), ("channel", r"\bchannels?\b"),
    ("normal range", r"\bnormal ranges?\b"), ("level", r"\blevels?\b"), ("healthy", r"\bhealthy\b"), ("failure mode", r"\bfailure[ _]modes?\b"),
    ("profile", r"\bprofile\b"), ("fix history", r"\bfix[ _]history\b"), ("alarm", r"\balarms?\b"), ("visit", r"\bvisits?\b"),
    ("technician", r"\btechnicians?\b"), ("lead time", r"\blead times?\b"), ("downtime", r"\bdowntimes?\b"), ("repair charge", r"\brepair charge\b"),
    ("part", r"\bparts?\b"), ("budget", r"\bbudget\b"), ("step cap", r"\bstep cap\b"), ("price", r"\bprices?\b"), ("sensor", r"\bsensors?\b"),
    ("mean", r"\bmeans?\b"), ("statistic", r"\bstatistics?\b"), ("noun rig", r"\brigs?\b"), ("noun asset", r"\bassets?\b"), ("noun unit", r"\bunits?\b"),
    ("id fmea", r"fmea/"), ("id wo", r"\bwo/"), ("id alarm", r"alarm/"), ("id obs", r"\bobs/"), ("diagnose", r"\bdiagnos(?:e|es|is|ed)\b"),
    ("declare", r"\bdeclar(?:e|es|ation|ed)\b"), ("reading", r"\breadings?\b"), ("read", r"\breads?\b"), ("wait", r"\bwaits?\b"),
    ("finish", r"\bfinish\b"), ("repair", r"\brepairs?\b")]
# the two words that stand in sentences of the plan page taken word for word: "in this order" is the sequence of the parts of the
# board, and "the refuted repairs" names the write class repair of the general agent
PAGE_WORDS = {("BOARD", "order"), ("EARLIER_TURNS_KEEPS", "repair")}


def kind_words(text):
    """(label, matched) of every word of the kind in a text whose slots are taken out."""
    text, taken, found = SLOT.sub(" ", text), [], []
    for label, pattern in KIND_WORD_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            if not any(a < m.end() and m.start() < b for a, b in taken):
                taken.append((m.start(), m.end()))
                found.append(label)
    return found


def registered_names():
    """The names of every registered asset: its components, sensor groups, channels, asset id and short noun."""
    names = set()
    for path in sorted(ASSETS_DIR.glob("*/asset.json")):
        data, profile = asset_profile(path)
        names |= set(profile["components"]) | set(profile["sensor_groups"]) | set(data["channels"]) | {data["asset_id"]}
        names.add(data["text"]["short_noun"])
    return names


def test_the_shared_file_types_no_word_of_a_kind_and_no_name_of_a_registered_asset():
    hits = {(name, label) for name, text in SHARED.items() for label in kind_words(text)}
    assert hits == PAGE_WORDS, sorted(hits - PAGE_WORDS)
    names = registered_names()
    assert not [(n, w) for n, text in SHARED.items() for w in re.findall(r"[A-Za-z0-9_]+", SLOT.sub(" ", text)) if w in names]
    # the control: the reason of a step written with the words typed in is found by the same search
    typed = "leave this step out, since every recorded cycle of sensor group tier3 is read already"
    assert {"cycle", "sensor group"} <= set(kind_words(typed)) and "tier3" in names
    assert not kind_words(core_texts.STANDS_NOTHING_NEW)


def _shared_names_used(paths):
    """The names of the shared file that the files take: imported from it, or read as core_texts.<name>."""
    used = set()
    for path in paths:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("core_texts"):
                used |= {alias.name for alias in node.names}
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "core_texts":
                used.add(node.attr)
    return used


def test_every_constant_of_the_shared_file_is_printed_by_the_agent():
    sources = [p for p in AGENT_FOLDER.rglob("*.py") if "tests" not in p.parts]
    used = _shared_names_used(sources)
    assert set(SHARED) <= used, sorted(set(SHARED) - used)
    # the control: without the file that prints the prompts, the constants that only it prints are named
    assert {"OBJECTIVE", "LOOP_STANDING", "STOOD_LINE"} <= set(SHARED) - _shared_names_used([p for p in sources if p.name != "prompts.py"])
    rendered = system_prompt(None) + system_prompt(6, server=False) + planning_turn(_synthetic_board(), None, not_run="x", stood=[(1, "profile", "y")])
    for name in ("BOARD_HELD", "LOOP_STANDING", "OBJECTIVE", "STOOD_HEADING", "STOOD_CLOSING", "NOT_RUN_HEADING", "NOT_RUN_CLOSING"):
        assert SHARED[name] in rendered, name


def _prompts_of_every_row():
    """The system prompt of every combination of the switches and of the kinds a row may not serve, and the turns built on it."""
    surfaces = [None]
    for unserved in ((), ("declare",), ("wait", "time"), ("declare", "wait", "time")):
        surface = AobSurface()
        surface.unserved_kinds = lambda unserved=frozenset(unserved): unserved
        surfaces.append(surface)
    texts = []
    for surface in surfaces:
        for server in (True, False):
            for recall in (True, False):
                for plan_length in (None, 6):
                    texts.append(system_prompt(plan_length, surface=surface, server=server, recall=recall))
    board = _synthetic_board()
    texts += [planning_turn(board, 6, retry="r", not_run="n", stood=[(2, "extent", "reason")]), prompts.earlier_turns_turn(board, 6),
              prompts.summary_turn(board, 6), planning_turn(Board(), None)]
    return texts


def test_no_slot_and_no_mark_of_the_served_kinds_stays_open_in_a_rendered_prompt():
    texts = _prompts_of_every_row()
    for text in texts:
        assert not SLOT.findall(text), SLOT.findall(text)
        assert prompts.EVIDENCE_STEPS_MARK not in text and prompts.COMMITMENTS_MARK not in text
    assert all(core_texts.LOOP_STANDING in text for text in texts[:-4])  # every system prompt prints the shared texts
    # the control: a shared text printed with its slots left open is found
    assert SLOT.findall(texts[0] + prompts.RULE_SECOND_COMMITMENT) == ["{commitments}"]


def test_the_served_kinds_fill_the_kinds_that_carry_evidence_and_the_commitments_that_stand():
    full = system_prompt(6)
    assert 'A step of the kind diagnose, declare or order carries "evidence"' in full
    assert "no diagnosis or declaration that differs from the commitment that stands;" in full
    surface = AobSurface()
    surface.unserved_kinds = lambda: frozenset({"declare"})
    without = system_prompt(6, surface=surface)
    assert 'A step of the kind diagnose or order carries "evidence"' in without
    assert "no diagnosis that differs from the commitment that stands;" in without and "declar" not in without.split("The controller refuses", 1)[1]


def _missing_sentences(text):
    """The sentences of the new reasons that a system prompt lacks."""
    sentences = {"RULE_SECOND_COMMITMENT": prompts.RULE_SECOND_COMMITMENT.format(commitments="diagnosis or declaration"),
                 "READ_STANDS": prompts.READ_STANDS, "LOOP_STANDING": core_texts.LOOP_STANDING}
    return [name for name, sentence in sentences.items() if sentence not in text]


def test_every_new_reason_has_its_sentence_in_both_system_prompts():
    for server in (True, False):
        assert _missing_sentences(system_prompt(6, server=server)) == []
    # the control: a prompt with one sentence removed is named
    assert _missing_sentences(system_prompt(6).replace(prompts.READ_STANDS, "")) == ["READ_STANDS"]


def test_the_planning_turn_names_the_steps_that_made_no_call_after_the_steps_that_did_not_run():
    board = _synthetic_board()
    stood = [(1, "profile", "leave this step out, since a"), (3, "extent", "leave this step out, since b")]
    both = planning_turn(board, 6, not_run="Step 4 (read) did not run, for these reasons:\n- c", stood=stood)
    section = (f"{core_texts.STOOD_HEADING}\n- step 1 (profile): leave this step out, since a\n- step 3 (extent): leave this step out, since b\n"
               f"{core_texts.STOOD_CLOSING}")
    assert both.endswith(section) and both.index(core_texts.NOT_RUN_HEADING) < both.index(core_texts.STOOD_HEADING)
    alone = planning_turn(board, 6, stood=stood)
    assert alone.endswith(section) and core_texts.NOT_RUN_HEADING not in alone
    assert core_texts.STOOD_HEADING not in planning_turn(board, 6) and core_texts.OBJECTIVE in planning_turn(board, 6)


def test_every_rewording_for_a_kind_the_row_does_not_serve_still_finds_its_text():
    full = SYSTEM_PROMPT + SYSTEM_PROMPT_WITHOUT_SERVER
    for kind, pairs in WITHOUT_KIND.items():
        for old, _new in pairs:
            assert old in full, (kind, old)
    assert len(WAIT_HEAD.findall(SYSTEM_PROMPT)) == len(WAIT_HEAD.findall(SYSTEM_PROMPT_WITHOUT_SERVER)) == 1
    assert prompts.RECALL_CLAUSE in full and prompts.RECALL_LINE in full and prompts.RECALL_REFUSAL in full
    # the control: the two sentences that the shared texts replaced are gone, so their rewordings, left in, would be named above
    assert "a wait lets the record grow" not in full and "no second diagnosis or declaration;" not in full


def test_the_new_texts_on_the_tiny_asset_name_its_sensor_groups_and_no_name_of_another_asset():
    tiny, profile = asset_profile(TINY_ASSET)
    banned = registered_names() - set(profile["components"]) - set(profile["sensor_groups"]) - set(tiny["channels"]) - {tiny["asset_id"], tiny["text"]["short_noun"]}
    board = Board(clock_runs=False)
    board.set_profile(profile)
    first, last = 1, 4
    board.status = {"time": 0, "recorded_from": first, "recorded_until": last}
    board.note_answered("profile", {})
    board.note_answered("extent", {})
    group = next(iter(profile["sensor_groups"]))
    board.note_call("read", {"sensor_group": group, "cycles": 1})
    board.note_call("read", {"sensor_group": group, "cycles": 1})
    status = board._render_status()
    texts = [status, system_prompt(6), system_prompt(None, server=False),
             planning_turn(board, 6, stood=[(1, "read", core_texts.STANDS_NOTHING_NEW.format(name=group, **KIND_WORDS))])]
    words = {w for text in texts for w in re.findall(r"[A-Za-z0-9_]+", text)}
    assert not banned & words, sorted(banned & words)
    assert "tier3" in banned  # the control: a sensor group of the rig typed into one of these texts would be named
    assert all(core_texts.HELD_SERIES.format(name=g, first=first, last=last, held=0, recorded=last - first + 1, **KIND_WORDS) in status
               for g in profile["sensor_groups"])


def _static_prefixes(prompts_of_episode):
    """The system part of every planning request of an episode, up to the board: the system prompt with the tool texts, the
    heading of the turn and the question."""
    planning = [p for p in prompts_of_episode if PLANNING_HEADING in p]
    return [p[:p.index("\n\n", p.index("Question of the episode: ")) + 2] for p in planning], planning


def _changes(prefixes, planning):
    """The planning turns, counted from 0, at which the static part differs from the turn before."""
    return [i for i in range(1, len(prefixes)) if prefixes[i] != prefixes[i - 1]]


def test_the_static_part_of_the_planning_request_is_byte_identical_from_turn_to_turn(tmp_path, monkeypatch):
    _ep_dir, row, backend = _play_without_server(tmp_path / "rig", bands_plans())
    prefixes, planning = _static_prefixes(backend.inner.prompts)
    assert row["error"] == "" and len(planning) >= 3 and _changes(prefixes, planning) == []
    # the same on the AssetOpsBench rows, whose tool texts state the clock from the listing alone
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.recorded_history_env import RecordedRigEnv
    from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode
    from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS

    def truth():
        state = episode.load_state(tmp_path / "aob" / "seed000003" / "episode.json")
        return dict(zip(COMPONENTS, episode.replay({**state, "actions": [], "think_cycles": []}, env_class=RecordedRigEnv).hidden))
    plans = bands_plans(wait=0)
    _ep_dir, row, backend = _aob_run(tmp_path / "aob", "real_time", ScriptedPlanner(lambda prompt, call: plans(prompt, call, truth)), {}, seed=3,
                                     server=False)
    prefixes, planning = _static_prefixes(backend.inner.prompts)
    assert row["error"] == "" and len(planning) >= 3 and _changes(prefixes, planning) == []
    # the control: a system prompt that carries a text of the turn breaks it
    turns = iter(range(1000))
    printed = prompts.system_prompt
    monkeypatch.setattr(prompts, "system_prompt", lambda *a, **k: printed(*a, **k) + f" Turn {next(turns)}.")
    _ep_dir, row, backend = _play_without_server(tmp_path / "control", bands_plans())
    prefixes, planning = _static_prefixes(backend.inner.prompts)
    assert len(_changes(prefixes, planning)) == len(planning) - 1


def test_both_system_prompts_name_the_legend_of_the_normal_ranges_and_the_marks_of_a_value():
    for server in (True, False):
        text = system_prompt(6, server=server)
        assert "stands once in the legend above the evidence rows, a value outside it ends in low or high" in text
        assert "in the column in normal range" not in text and "marked NO" not in text


NOTES_LINE = "Controller notes: "


def _shared(first, second):
    """The count of the leading characters that two texts share."""
    return next((i for i, (a, b) in enumerate(zip(first, second)) if a != b), min(len(first), len(second)))


def _with_one_more_equal_note(board):
    """The planning request of the board, and the request once a later plan left a note equal to the board's note of a plan."""
    board.mark_turn(1)
    board.add_note("Your plan of turn 1 was refused: no profile step.", turn=1)
    before = planning_turn(board, 6)
    board.mark_turn(2)
    board.add_note("Your plan of turn 2 was refused: no profile step.", turn=2)
    return before, planning_turn(board, 6)


@pytest.mark.parametrize("facts", [False, True], ids=["before the profile", "after the profile"])
def test_one_more_equal_note_changes_no_byte_of_the_request_before_the_notes_line(facts, monkeypatch):
    """The notes line is the last line of the status, which is the last part of the board, so the count of an equal note moves
    nothing before it, and after the profile the evidence table stands wholly before it."""
    board = _synthetic_board(read_first=False) if facts else Board()
    before, after = _with_one_more_equal_note(board)
    assert before != after and core_texts.NOTE_REPEATED.format(count=1) in after
    assert _shared(before, after) >= before.index(NOTES_LINE)
    if facts:
        assert before.index("## Evidence") < before.index("## Status") < before.index(NOTES_LINE)
    # the control: a notes line printed ahead of the evidence moves bytes before it
    terms = Board._render_terms
    monkeypatch.setattr(Board, "_render_terms", lambda self: self._notes_line() + "\n" + terms(self))
    before, after = _with_one_more_equal_note(_synthetic_board(read_first=False))
    assert _shared(before, after) < before.index("## Evidence")


class _CandidatesBeforeTheProfile(ScriptedPlanner):
    """Two plans before the profile that list candidates of their own: the first reads the fix history alone and the second is the
    opening, so the request after the profile holds one note on candidates that were not kept, with the count of the earlier one."""

    def __init__(self, plans):
        super().__init__(plans)
        self.before = 0

    def start_episode(self):
        super().start_episode()
        self.before = 0

    def generate(self, prompt, temperature=0.0):
        if PLANNING_HEADING in prompt and "No profile read yet" in prompt and self.before < 2:
            self.before += 1
            self.prompts.append(prompt)
            steps = [{"kind": "fix_history"}] if self.before == 1 else [{"kind": "profile"}, {"kind": "failure_modes", "args": {"component": ""}},
                                                                         {"kind": "fix_history"}]
            return json.dumps({"why": "made up", "mode": "test_first", "steps": steps, "candidates": [{"levels": {"pump": 1}, "failure_mode_ids": []}]})
        return super().generate(prompt, temperature)


def test_an_episode_in_which_the_count_of_a_note_changes_keeps_the_static_part_byte_identical(tmp_path, monkeypatch):
    monkeypatch.setattr("CausalDiagnosis.agents.Causal_Diagnosis_Agent.tests.test_causal_diagnosis_agent.ScriptedPlanner", _CandidatesBeforeTheProfile)
    _ep_dir, row, backend = _play_without_server(tmp_path / "rig", bands_plans())
    prefixes, planning = _static_prefixes(backend.inner.prompts)
    kept = "The candidates of your plan of turn {} were not kept, since no profile was on the board; list them again in your next plan."
    after = [p for p in planning if "No profile read yet" not in p]
    assert row["error"] == "" and len(planning) >= 4 and _changes(prefixes, planning) == []
    assert after[0].count(kept.format(2) + core_texts.NOTE_REPEATED.format(count=1)) == 1 and kept.format(1) not in after[0]


def _three_reads(prompt, call, truth):
    """Three planning turns that each read one more sensor group, then the plan of bands_plans, which commits and finishes."""
    if call <= 3:
        group = ("tier1", "tier2", "tier3")[call - 1]
        return json.dumps({"why": "read", "mode": "test_first", "steps": [{"kind": "read", "args": {"sensor_group": group, "cycles": 1}}]})
    return bands_plans()(prompt, call, truth)


def _rows_end(prompt):
    """Where the evidence rows of a request end: the end of its last row, before the notifications and the next section."""
    section = prompt.index("## Evidence")
    last = prompt.rindex("\n| obs/", section, prompt.index("\n\n## ", section))
    return prompt.index("\n", last + 1)


def _first_change_and_rows_end(planning):
    """Per pair of consecutive planning requests after the profile whose earlier one holds evidence rows: where they first differ,
    and where the evidence rows of the earlier one end."""
    requests = [p for p in planning if "No profile read yet" not in p and SUMMARY_MARK not in p]
    return [(_shared(a, b), _rows_end(a)) for a, b in zip(requests, requests[1:]) if "| obs/" in a]


SUMMARY_MARK = "## Your summary of earlier turns"


def test_consecutive_requests_after_the_profile_share_every_evidence_row_of_the_earlier_one(tmp_path, monkeypatch):
    """A new reading adds its rows after the rows the board holds, and every other part that changes stands after them, so two
    consecutive requests share every byte up to the last row that the earlier one held."""
    _ep_dir, row, backend = _play_without_server(tmp_path / "rig", _three_reads)
    pairs = _first_change_and_rows_end([p for p in backend.inner.prompts if PLANNING_HEADING in p])
    assert row["error"] == "" and len(pairs) >= 2 and all(change >= end for change, end in pairs), pairs
    # the control: a legend that counts the rows changes a byte before them
    monkeypatch.setattr(Board, "_normal_range_legend", lambda self: f"{len(self.evidence)} rows")
    _ep_dir, _row, backend = _play_without_server(tmp_path / "control", _three_reads)
    pairs = _first_change_and_rows_end([p for p in backend.inner.prompts if PLANNING_HEADING in p])
    assert pairs and not all(change >= end for change, end in pairs), pairs
