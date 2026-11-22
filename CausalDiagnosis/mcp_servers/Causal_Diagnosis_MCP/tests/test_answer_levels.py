"""The levels of response_format (tools/answer_levels.py) against the raw answer: raw is the answer of CausalTools as FastMCP
writes it, byte for byte; a shorter level is a pure function of that answer and the public view, written as one line of JSON the
same way on every replay; detailed names every evidence identifier of the raw answer by its stated rule and keeps every other
value but those that only raw holds, the anomaly score of each value of the evidence and the channels of the attribution after the
first TOP_CHANNELS; concise is the documented part of detailed; a repeated answer opens with the key unchanged. Twin episodes play the same
steps, one by CausalTools itself and one served at each level, so that the predictions every call stores number alike in each.
No test states a value computed from the rig data."""
import asyncio
import json
from pathlib import Path

import pydantic_core
import pytest

from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import build_blocks, default_facts_path, load_cycles
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.main import build_server
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools import CausalTools, answer_levels, tool_texts
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests import answer_level_rules as rules
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.fixture_episode import FixtureEpisode, default_rules, episode_models
from CausalDiagnosis.utils.paths import ROOT

if not Path(default_facts_path()).exists():
    pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states", allow_module_level=True)

START = (100, 80, 0, 90)
FIRST = [("get_episode_evidence", {"sensor_group": "all"}), ("rank_root_causes", {}), ("rank_root_causes", {"top_k": 3, "branch_detail": "summary"}),
         ("estimate_intervention_effect", {"action": "replace:accumulator"})]
SECOND = [("get_episode_evidence", {"sensor_group": "tier1"}), ("get_episode_evidence", {"sensor_group": "PS1"}),
          ("rank_root_causes", {"top_k": 3, "branch_detail": "full"}), ("estimate_intervention_effect", {"action": "monitor"}),
          ("verify_outcome", {"prediction_id": "pred/1"})]
AFTER_REPAIR = [("rank_root_causes", {}), ("get_episode_evidence", {"sensor_group": "all"}), ("verify_outcome", {"prediction_id": "pred/1"}),
                ("verify_outcome", {"prediction_id": "pred/9"})]
CALLS = FIRST + SECOND + AFTER_REPAIR
SHORTER = {"get_episode_evidence": answer_levels.evidence, "rank_root_causes": answer_levels.ranking,
           "estimate_intervention_effect": answer_levels.effect,
           "verify_outcome": lambda full, view, level: answer_levels.verification(full, level)}


def direct(tools, tool, args):
    """The answer of CausalTools to a call that the server receives with these arguments."""
    if tool == "rank_root_causes":
        return tools.rank_root_causes(args.get("top_k", 5), args.get("branch_detail", "compact"))
    return getattr(tools, tool)(*args.values())


async def served(server, tool, args):
    from mcp.shared.memory import create_connected_server_and_client_session

    async with create_connected_server_and_client_session(server._mcp_server) as session:
        answer = await session.call_tool(tool, args)
    return answer.content[0].text, answer.structuredContent["result"]


def play(folder, serve):
    """The calls of CALLS on one twin episode, each answered by serve(tool, args, released_path)."""
    cycles = load_cycles(default_facts_path())
    blocks = build_blocks(cycles)
    episode = FixtureEpisode(folder, cycles, blocks, START, seed=0, rules=default_rules(cycles, blocks))
    answers = []
    episode.read("tier1")
    answers += [serve(tool, args, episode.released_path) for tool, args in FIRST]
    episode.read("tier3", cycles=2)
    answers += [serve(tool, args, episode.released_path) for tool, args in SECOND]
    episode.create_work_order("accumulator", prediction_id="pred/1")
    episode.advance(5)
    episode.read("tier3")
    answers += [serve(tool, args, episode.released_path) for tool, args in AFTER_REPAIR]
    return answers


@pytest.fixture(scope="module")
def twins(tmp_path_factory):
    """{"own": [(the answer of CausalTools, the view)], level: [(the text served, its structured content)]} for every call of
    CALLS, the raw twin and those of the two longer shorter levels each on an episode of its own."""
    by_tools = {}

    def by_causal_tools(tool, args, released):
        tools = by_tools.setdefault(released, CausalTools(episode_models(START), released))
        return direct(tools, tool, args), tools.released_view()

    def by_server(level):
        return lambda tool, args, released: asyncio.run(served(build_server(released, left_out=START), tool, {**args, "response_format": level}))

    out = {"own": play(tmp_path_factory.mktemp("own"), by_causal_tools)}
    for level in ("raw", "detailed", "concise"):
        out[level] = play(tmp_path_factory.mktemp(level), by_server(level))
    return out


def unmarked(answer, earlier):
    """The answer without the key unchanged, after checking that it opens the answer and names a call of earlier, the answers
    before it on the same episode, that returned the same answer."""
    if "unchanged" not in answer:
        return answer
    assert next(iter(answer)) == "unchanged"
    step = int(answer["unchanged"].removeprefix("unchanged since your call at step "))
    rest = {k: v for k, v in answer.items() if k != "unchanged"}
    assert earlier[step - 1] == rest
    return rest


def test_raw_is_the_answer_of_causal_tools_as_fastmcp_writes_it(twins):
    """The control: a server that writes any other form under raw, or opens a repeated raw answer with unchanged, fails."""
    for (tool, _args), (result, _view), (text, structured) in zip(CALLS, twins["own"], twins["raw"]):
        assert text == pydantic_core.to_json(result, fallback=str, indent=2).decode(), tool
        assert structured == result.model_dump(mode="json"), tool


@pytest.mark.parametrize("level", ["detailed", "concise"])
def test_a_shorter_level_is_its_function_of_the_raw_answer_written_alike_on_every_replay(twins, level):
    """The text served is the level's function of the raw answer of the twin of CausalTools, as one line of JSON byte for byte,
    the same on another episode with the same steps, and a repeated answer opens with unchanged and the call that first
    returned it. The control: an order of keys or a number format that varies between replays, an indented text or a marker
    that names the wrong call fails."""
    earlier = []
    for (tool, _args), (result, view), (text, structured) in zip(CALLS, twins["own"], twins[level]):
        full = result.model_dump(mode="json")
        if "error" in full:
            assert structured == full
            earlier.append(full)
            continue
        expected = SHORTER[tool](full, view, level)
        answer = json.loads(text)
        assert answer == structured and "\n" not in text
        assert answer_levels.text_of(unmarked(answer, earlier)) == answer_levels.text_of(expected)
        assert text == answer_levels.text_of(answer)
        earlier.append(expected)


def test_detailed_names_every_evidence_identifier_and_keeps_every_value_and_concise_is_its_documented_part(twins):
    """Every identifier of the raw answer follows from detailed, the rule its evidence_rule states and the public view of the
    asset (tests/answer_level_rules.py), for each candidate under branch_detail compact (the newest reading of each field) and
    the longer branch details, from the concise ranking and estimate too; every reading and every work order keeps a literal
    identifier at every level, as realized_read_elimination.forecasts reads them; every other value of the raw answer stands
    in detailed but those that only raw holds, and concise is detailed without what the documentation leaves to detailed. The control: a level that drops a
    reading from readings, names a reading by the wrong group, rounds a value, marks a value inside its range or leaves out a
    field that concise keeps fails, and so do those of the next test."""
    seen = set()
    for (tool, _args), (result, view) in zip(CALLS, twins["own"]):
        full = result.model_dump(mode="json")
        if "error" in full:
            continue
        detailed, concise = (SHORTER[tool](full, view, level) for level in ("detailed", "concise"))
        rules.assert_keeps_every_value(detailed, full, view)
        rules.assert_concise_is_the_documented_part(concise, detailed, tool, view)
        if tool == "rank_root_causes":
            seen.add(full.get("assumptions") == [tool_texts.A_COMPACT])
            for answer in (detailed, concise):
                every, candidates = rules.ranking_ids(answer, view)
                assert every == full["evidence_ids"] and candidates == [c["evidence_ids"] for c in full["candidates"]]
        elif tool == "get_episode_evidence":
            assert rules.table_ids(detailed) == full["evidence_ids"]
        elif tool == "estimate_intervention_effect":
            assert rules.effect_ids(detailed, view) == rules.effect_ids(concise, view) == full["evidence_ids"]
        else:
            assert rules.verification_ids(detailed) == full["evidence_ids"]
        for answer in (detailed, concise):
            assert rules.kept_in_order(answer["evidence_ids"], full["evidence_ids"])
            readings = {i.split("/")[1] for i in full["evidence_ids"] if i.startswith("obs/")}
            assert {i.split("/")[1] for i in answer["evidence_ids"] if i.startswith("obs/")} == readings
            assert [i for i in answer["evidence_ids"] if not i.startswith("obs/")] == [i for i in full["evidence_ids"] if not i.startswith("obs/")]
    assert seen == {True, False}  # both rules of the candidates were checked


def test_the_detailed_evidence_lacks_only_the_anomaly_score_of_a_raw_item(twins, monkeypatch):
    """The long table of the detailed evidence holds every value of each raw item but its anomaly score, which the levels line of
    every tool description and the documentation leave to raw. The control: a raw item with a value that the table does not
    hold, a level whose table drops the normal range of a field and one whose table carries the anomaly scores fail the rules."""
    answers = [(result.model_dump(mode="json"), view) for (tool, _args), (result, view) in zip(CALLS, twins["own"]) if tool == "get_episode_evidence"]
    full, view = next((full, view) for full, view in answers
                      if any(item["anomaly_score"] is not None and item["normal_high"] is not None for item in full.get("items", [])))
    rules.assert_keeps_every_value(answer_levels.evidence(full, view, "detailed"), full, view)
    extra = dict(full, items=[dict(full["items"][0], quality=1.0), *full["items"][1:]])
    with pytest.raises(AssertionError):
        rules.assert_keeps_every_value(answer_levels.evidence(extra, view, "detailed"), extra, view)
    long_table = answer_levels._long_table

    def range_dropped(items):
        columns, fields = long_table(items)
        del next(entry for entry in fields if "normal_high" in entry)["normal_high"]
        return columns, fields

    def scores_carried(items):
        columns, fields = long_table(items)
        scores = {(item["field"], item["n"]): item["anomaly_score"] for item in items}
        for entry in fields:
            entry["anomaly_scores"] = [scores.get((entry["field"], column["n"])) for column in columns]
        return columns, fields

    for planted in (range_dropped, scores_carried):
        monkeypatch.setattr(answer_levels, "_long_table", planted)
        with pytest.raises(AssertionError):
            rules.assert_keeps_every_value(answer_levels.evidence(full, view, "detailed"), full, view)


def test_the_headline_is_one_line_with_the_first_candidate_and_the_advice(twins):
    """The headline of a ranking names the first candidate with its mass and the advice and carries the flag, the separating test
    and the cycle of the belief, which evaluation/metrics/mcp_metrics.py reads of the last ranking before a decision; that of the
    evidence the count of the fields outside their range, and those of an estimate and a verification their message. The
    control: a headline of several lines, without the advice or without the flag fails."""
    for (tool, _args), (result, view) in zip(CALLS, twins["own"]):
        full = result.model_dump(mode="json")
        if "error" in full:
            continue
        headline = SHORTER[tool](full, view, "headline")
        flags = ("read_first", "separating_test", "belief_time") if tool == "rank_root_causes" else ()
        assert set(headline) == {"headline", *flags} and "\n" not in headline["headline"]
        assert all(headline[key] == full[key] for key in flags)  # what the measures of the decisions read
        if tool == "rank_root_causes":
            top = full["candidates"][0]
            assert "(" + ", ".join(f"{c} {v}" for c, v in top["combination"].items()) + ")" in headline["headline"]
            assert f"{top['probability']:.3f}" in headline["headline"] and full["advice"]["text"] in headline["headline"]
        elif tool == "get_episode_evidence":
            assert "fields outside their normal range" in headline["headline"]
        else:
            assert headline["headline"] == full["message"]


def test_a_repeated_call_returns_its_level_again_opened_by_the_call_that_first_returned_it(tmp_path):
    """Each server process reads the transcript of the episode, so a call whose answer an earlier call of the same tool returned
    opens with unchanged and the step of that call, one step per tool call of the episode; raw never does, and an answer that
    changed with the evidence does not. The control: a marker on raw, on a changed answer or with the step of the latest
    repeat fails."""
    cycles = load_cycles(default_facts_path())
    episode = FixtureEpisode(tmp_path, cycles, build_blocks(cycles), START, seed=0)
    episode.read("tier1")

    def call(args, level):
        return asyncio.run(served(build_server(episode.released_path, left_out=START), "get_episode_evidence", {**args, "response_format": level}))

    first, _ = call({"sensor_group": "all"}, "concise")
    raw_once, _ = call({"sensor_group": "all"}, "raw")
    again, structured = call({"sensor_group": "all"}, "concise")
    raw_twice, _ = call({"sensor_group": "all"}, "raw")
    third, _ = call({"sensor_group": "all"}, "concise")
    assert "unchanged" not in json.loads(first) and raw_once == raw_twice and not raw_twice.startswith('{"unchanged"')
    assert json.loads(again) == structured == {"unchanged": "unchanged since your call at step 1", **json.loads(first)}
    assert json.loads(third)["unchanged"] == "unchanged since your call at step 1"
    headline, _ = call({"sensor_group": "all"}, "headline")
    assert "unchanged" not in json.loads(headline)
    episode.read("tier3")
    changed, _ = call({"sensor_group": "all"}, "concise")
    assert "unchanged" not in json.loads(changed)


def test_every_assumption_text_has_a_key_or_tells_what_the_environment_refuses_and_the_docs_hold_every_fixed_sentence(twins):
    """A shorter level keys every assumption text of a formula, so only the notes on what the environment refuses and the
    declared error level stay as text, a fixed key stands for one sentence, and that sentence stands in
    CausalDiagnosis/docs/causal-diagnosis-mcp.md. The control: a new formula text without a key, or a fixed sentence the docs
    lack, fails."""
    texts = {text for result, _view in twins["own"] for text in result.model_dump(mode="json").get("assumptions", [])}
    kept = {text for text in texts if tool_texts.assumption_key(text) == text}
    assert all(text.startswith(("alpha = ", "the environment accepts a work order only after")) for text in kept), kept
    assert len(set(tool_texts.FIXED_KEYS.values())) == len(tool_texts.FIXED_KEYS)
    docs = (ROOT / "CausalDiagnosis" / "docs" / "causal-diagnosis-mcp.md").read_text(encoding="utf-8")
    assert [key for sentence, key in tool_texts.FIXED_KEYS.items() if sentence not in docs] == []


def test_a_level_of_another_name_is_an_error_answer_before_the_tool_runs(tmp_path):
    """An empty response_format is concise, and any other value, branch_detail's values among them, is an error answer before
    the tool runs, so a ranking or an estimate with a wrong level stores no prediction. The control: checking the level after
    the call stores one."""
    cycles = load_cycles(default_facts_path())
    episode = FixtureEpisode(tmp_path, cycles, build_blocks(cycles), START, seed=0)
    episode.read("tier1")
    server = build_server(episode.released_path, left_out=START)
    assert answer_levels.level_of("") == answer_levels.level_of(" concise ") == "concise" and answer_levels.level_of("raw") == "raw"
    assert all(answer_levels.level_of(value) is None for value in ("Raw", "compact", "summary", "full", None))
    for tool, args in (("rank_root_causes", {"top_k": 1}), ("estimate_intervention_effect", {"action": "replace:valve"})):
        text, structured = asyncio.run(served(server, tool, {**args, "response_format": "full"}))
        assert structured == {"error": "response_format must be one of headline, concise, detailed, raw"} and json.loads(text) == structured
    assert not (tmp_path / "predictions.json").exists()


def test_every_shorter_level_of_every_tool_is_one_json_object_for_a_client_and_in_process(tmp_path):
    """headline, concise and detailed of each of the four tools reach a client as one JSON object, and so does the answer that a
    caller in process reads off FastMCP's call_tool, as the test harness of the CausalDiagnosis agent reads it: the text of the
    content of the pair (content, structured content). A repeated call keeps one JSON object, opened by the key unchanged that
    names the step of the first call that returned the rest, whether the repeat comes through the protocol or in process. The
    control: handing the CallToolResult of a shorter level to a caller in process as it is gives a text that is no JSON."""
    cycles = load_cycles(default_facts_path())
    episode = FixtureEpisode(tmp_path, cycles, build_blocks(cycles), START, seed=0)
    episode.read("tier1")
    calls = [("get_episode_evidence", {"sensor_group": "all"}), ("rank_root_causes", {"branch_detail": "compact"}),
             ("estimate_intervention_effect", {"action": "replace:accumulator"}), ("verify_outcome", {"prediction_id": "pred/1"})]
    steps = 0  # the calls the transcript of the episode holds, one per call through the protocol

    def in_process(tool, args):
        result = asyncio.run(build_server(episode.released_path, left_out=START).call_tool(tool, args))
        content = result[0] if isinstance(result, tuple) else result  # as the test harness of the agent reads an answer
        text = "\n".join(getattr(block, "text", str(block)) for block in content)
        assert text.startswith('{"') and isinstance(json.loads(text), dict), (tool, text[:40])
        return text, result[1]["result"]  # the pair that every tool with an output schema gives a caller in process

    def parsed(text, structured, where):
        assert text.startswith('{"') and json.loads(text) == structured and isinstance(structured, dict), (where, text[:40])
        return structured

    for level in ("headline", "concise", "detailed"):
        for tool, args in calls:
            asked = {**args, "response_format": level}
            first = parsed(*asyncio.run(served(build_server(episode.released_path, left_out=START), tool, asked)), (level, tool, "client"))
            steps += 1
            if "unchanged" in first:  # an earlier level or tool cannot have returned this text
                raise AssertionError((level, tool, first["unchanged"]))
            repeats = [parsed(*in_process(tool, asked), (level, tool, "in process"))]
            repeats.append(parsed(*asyncio.run(served(build_server(episode.released_path, left_out=START), tool, asked)), (level, tool, "client")))
            steps += 1
            for again in repeats:
                if tool == "estimate_intervention_effect":  # every estimate stores a new prediction, so no answer repeats
                    assert "unchanged" not in again and again != first
                else:
                    assert next(iter(again)) == "unchanged" and again == {"unchanged": f"unchanged since your call at step {steps - 1}", **first}
