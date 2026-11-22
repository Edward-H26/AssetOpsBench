"""The raw level that the CausalDiagnosis MCP keeps beside every answer it serves at a shorter level (transcript.RAW_NAME): one
gzip record per such call, keyed by the line of the call in the transcript of the episode, from which the level code of the
server (answer_levels.at_level) writes the served text again, and whose text is the answer that the same call gives at the level
raw, byte for byte, for a tool that stores nothing. A call at raw and an error answer keep no record, and every record is a gzip
member without a file name or a time, so the bytes of the file follow from the records. The control: the record of another line
does not give back the served text. No test states a value computed from the rig data."""
import asyncio
import gzip
import json
from pathlib import Path

import pytest

from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import build_blocks, default_facts_path, load_cycles

if not Path(default_facts_path()).exists():
    pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states", allow_module_level=True)

from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import profile_view  # noqa: E402
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.main import build_server  # noqa: E402
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.fixture_episode import FixtureEpisode  # noqa: E402
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.test_answer_levels import START, served  # noqa: E402
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools import answer_levels  # noqa: E402
from CausalDiagnosis.mcp_servers.utils import transcript  # noqa: E402

# (tool, arguments, level) in the order served; the evidence stores nothing, so its concise call and the raw call after it read alike
CALLS = [("get_episode_evidence", {"sensor_group": "all"}, "concise"), ("get_episode_evidence", {"sensor_group": "all"}, "raw"),
         ("rank_root_causes", {}, "concise"), ("rank_root_causes", {"top_k": 3}, "detailed"),
         ("estimate_intervention_effect", {"action": "replace:accumulator"}, "headline"),
         ("verify_outcome", {"prediction_id": "pred/99"}, "concise"),  # an error answer: no stored prediction of that name
         ("get_episode_evidence", {"sensor_group": "all"}, "a level of no name")]


def members(data):
    """The gzip members of a file, each as (its header flags, its time, its records)."""
    out = []
    while data:
        assert data[:2] == b"\x1f\x8b", "a member opens with the gzip magic bytes"
        d = gzip.zlib.decompressobj(-15)
        body = d.decompress(data[10:])
        out.append((data[3], int.from_bytes(data[4:8], "little"), body.decode("utf-8")))
        data = d.unused_data[8:]  # the CRC and the size close the member
    return out


def test_every_shorter_answer_keeps_its_raw_level_beside_the_line_of_its_call(tmp_path):
    cycles = load_cycles(default_facts_path())
    episode = FixtureEpisode(tmp_path, cycles, build_blocks(cycles), START, seed=0)
    episode.read("tier1")
    texts = [asyncio.run(served(build_server(episode.released_path, left_out=START), tool, {**args, "response_format": level}))[0]
             for tool, args, level in CALLS]
    lines = [line for line in transcript.read(tmp_path) if line["kind"] == "call"]
    assert [(line["tool"], line["arguments"].get("response_format")) for line in lines] == [(tool, level) for tool, _args, level in CALLS]
    raws = transcript.raw_answers(tmp_path)
    # an error answer is a result of its own, which the server sends as it is at every level
    shorter = [line["n"] for (_tool, _args, level), line, text in zip(CALLS, lines, texts)
               if level in ("concise", "detailed", "headline") and "error" not in json.loads(text)]
    assert sorted(raws) == shorter and len(shorter) == 4
    released = json.loads(episode.released_path.read_text())
    view = profile_view(dict(released["rules"], asset_id=released["asset_id"]))
    for (tool, _args, level), line, text in zip(CALLS, lines, texts):
        if line["n"] not in raws:
            continue
        again = answer_levels.at_level(tool, json.loads(raws[line["n"]]), view, level)
        served_answer = {k: v for k, v in json.loads(text).items() if k != "unchanged"}
        assert answer_levels.text_of(again) == answer_levels.text_of(served_answer), (tool, level)
    # the raw call right after the concise one of the evidence, which stores nothing, gives the record byte for byte
    assert raws[lines[0]["n"]] == texts[1]
    # every record is one member without a file name (flag 8) and with the time 0, in the order the lines were written
    found = members((tmp_path / transcript.RAW_NAME).read_bytes())
    assert [(flags & 8, mtime) for flags, mtime, _body in found] == [(0, 0)] * len(shorter)
    assert [json.loads(body)["n"] for _flags, _mtime, body in found] == shorter
    # the control: the record of the concise ranking read for the line of the detailed one does not give back what that line served
    wrong = answer_levels.at_level("rank_root_causes", json.loads(raws[lines[2]["n"]]), view, "detailed")
    assert answer_levels.text_of(wrong) != answer_levels.text_of({k: v for k, v in json.loads(texts[3]).items() if k != "unchanged"})
