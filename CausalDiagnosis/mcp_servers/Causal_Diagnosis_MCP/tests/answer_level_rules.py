"""The rules by which a shorter level of response_format names the evidence identifiers of the raw answer and keeps its values
(tools/answer_levels.py), written out from the evidence_rule texts and the documented level contents alone and not from the code
of the levels, so that the tests and a replay of recorded answers derive the raw answer's identifiers from a shorter level and the
public view of the asset, and find the values of the raw answer in it, but those that the documentation leaves to raw: the anomaly
score of each value of the evidence and the channels of the attribution after the first TOP_CHANNELS."""
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import statistic_of
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools.tool_texts import assumption_key

TOP_CHANNELS = 3  # the channels per component that the documentation of the levels names
# what concise leaves out of a detailed ranking, as CausalDiagnosis/docs/causal-diagnosis-mcp.md lists it
RANKING_DETAILED_ONLY = {"n_candidates", "n_states", "resampled_fits", "level_probability", "stability", "expected_saving", "expected_loss_act",
                         "expected_loss_test", "monte_carlo", "e10_rule"}
BRANCH_REPEATS = {"current_combination"}
# the keys of an evidence item of the raw answer: those the long table of detailed accounts for, and the one it leaves to raw
TABLE_ITEM_KEYS = {"evidence_id", "field", "channel", "statistic", "value", "unit", "rate", "n", "t", "sensor_group", "normal_low",
                   "normal_high", "in_normal_range"}
RAW_ONLY_ITEM_KEYS = {"anomaly_score"}
ENTRY_KEYS = {"field", "unit", "rate", "normal_low", "normal_high", "values"}  # the documented keys of an entry of fields


def _parts(identifier):
    _, n, field = identifier.split("/", 2)
    return int(n), field


def _marked(cell):
    return cell is not None and cell.endswith((" low", " high"))


def reading_ids(readings, view):
    """obs/<n>/<field> for every field of the sensor group that readings names for reading n, or of the fields it lists, by
    reading and then by field name, as the raw answers list them."""
    return [f"obs/{n}/{field}" for n, named in readings.items()
            for field in sorted(view.group_fields(named) if isinstance(named, str) else named)]


def ranking_ids(answer, view):
    """(the identifiers of the raw ranking, the identifiers of each candidate): every reading and every work order; a candidate
    those whose channel indicates one of its faulty components, of the newest reading of each field under branch_detail
    compact, and every work order."""
    orders = [i for i in answer["evidence_ids"] if i.startswith("wo/")]
    every = reading_ids(answer["readings"], view)
    compact = "compact" in answer["assumptions"]

    def bearing(candidate):
        faulty = {fault["component"] for fault in candidate["faults"]}
        ids = [i for i in every if faulty & set(view.indicates(view.sensor_of(_parts(i)[1])))]
        if compact:
            newest = {}
            for i in ids:
                n, field = _parts(i)
                newest[field] = max(newest.get(field, n), n)
            ids = [i for i in ids if _parts(i)[0] == newest[_parts(i)[1]]]
        return ids + orders

    return every + orders, [bearing(c) for c in answer["candidates"]]


def table_ids(answer):
    """obs/<n>/<field> for the n of a reading and the field of an entry whose value for it is not null, then every closed work
    order."""
    return ([f"obs/{reading['n']}/{entry['field']}" for k, reading in enumerate(answer["readings"]) for entry in answer["fields"]
             if entry["values"][k] is not None]
            + [f"wo/{order['id']}" for order in answer["work_orders_closed"]])


def effect_ids(answer, view):
    """Every reading, then every work order and the prediction, which evidence_ids keeps."""
    return reading_ids(answer["readings"], view) + [i for i in answer["evidence_ids"] if not i.startswith("obs/")]


def verification_ids(answer):
    """The prediction, then obs/<observation>/<field> for the field of every channel."""
    observed = [] if answer["observation"] is None else [f"obs/{answer['observation']}/{c['field']}" for c in answer["channels"]]
    return [answer["prediction_id"]] + observed


def kept_in_order(kept, full):
    """Whether the kept identifiers are identifiers of the full list, in its order."""
    chosen = set(kept)
    return len(chosen) == len(kept) and kept == [i for i in full if i in chosen]


def assert_keeps_every_value(detailed, raw, view):
    """Every key of the raw answer stands in the detailed level with its value, except the forms that detailed shortens: the
    identifiers it keeps, a candidate's own among them, the keys of the assumptions, the first channels of the attribution,
    the key of the text of e10_rule and the long table in place of the items, which holds every value of an item but its anomaly
    score, read with the public view of the asset. detailed adds only the statement of its identifiers."""
    added = {"evidence_rule", "readings"} if "readings" in detailed else {"evidence_rule"}
    if "items" in raw:
        added |= {"fields"}
    assert set(detailed) == (set(raw) - {"items"}) | added
    for key, value in raw.items():
        if key == "evidence_ids":
            assert kept_in_order(detailed[key], value)
        elif key == "assumptions":
            assert detailed[key] == [assumption_key(text) for text in value]
        elif key == "candidates":
            assert len(detailed[key]) == len(value)
            kept = set(detailed["evidence_ids"])
            for short, long in zip(detailed[key], value):
                assert short["evidence_ids"] == [i for i in long["evidence_ids"] if i in kept]  # its identifiers that evidence_ids keeps
                assert {k: v for k, v in short.items() if k != "evidence_ids"} == {k: v for k, v in long.items() if k != "evidence_ids"}
        elif key == "attribution" and value is not None:
            assert [dict(a, channels=a["channels"][:TOP_CHANNELS]) for a in value] == detailed[key]
        elif key == "e10_rule":
            assert detailed[key] == dict(value, text=assumption_key(value["text"]))
        elif key == "items":
            _assert_long_table_holds(detailed, value, view)
        else:
            assert detailed[key] == value, key


def _assert_long_table_holds(answer, items, view):
    """Each item stands in the long table with every key it has but its anomaly score, which only raw holds: its identifier
    obs/<n>/<field>, its channel and statistic, which the public view reads off its field, its cycle and sensor group in its
    reading, its value in the cell of its field and reading, low or high after it where it lies outside its normal range, whether
    it lies inside that range, and the unit, the rate and the normal range of its field once in its entry. An item with a key
    the table does not account for fails, and so does an entry with a key the documentation does not name."""
    readings = {reading["n"]: (k, reading) for k, reading in enumerate(answer["readings"])}
    entries = {entry["field"]: entry for entry in answer["fields"]}
    assert all(set(entry) <= ENTRY_KEYS for entry in answer["fields"])
    assert sum(cell is not None for entry in answer["fields"] for cell in entry["values"]) == len(items)
    for item in items:
        assert set(item) == TABLE_ITEM_KEYS | RAW_ONLY_ITEM_KEYS, set(item) ^ (TABLE_ITEM_KEYS | RAW_ONLY_ITEM_KEYS)
        k, reading = readings[item["n"]]
        entry = entries[item["field"]]
        assert item["evidence_id"] == f"obs/{item['n']}/{item['field']}"
        assert (item["channel"], item["statistic"]) == (view.sensor_of(item["field"]), statistic_of(item["field"]))
        assert (reading["t"], reading["sensor_group"]) == (item["t"], item["sensor_group"])
        assert (entry["unit"], entry["rate"], entry.get("normal_low"), entry.get("normal_high")) == (item["unit"], item["rate"], item["normal_low"], item["normal_high"])
        value, *mark = entry["values"][k].split(" ")
        assert float(value) == item["value"]
        side = None if item["in_normal_range"] is not False else ("low" if item["value"] < item["normal_low"] else "high")
        assert mark == ([] if side is None else [side])
        assert item["in_normal_range"] == (None if "normal_low" not in entry else not mark)


def assert_concise_is_the_documented_part(concise, detailed, tool, view):
    """concise is detailed without what the documentation of the levels leaves to detailed: for a ranking the filter's
    bookkeeping, the level distributions, the stability, the expected losses, the Monte Carlo settings, e10_rule and the
    current_combination of the candidates and of the branches; for the evidence of all or of a sensor group the fields without a
    value outside their range, which not_listed counts, while a channel or a field keeps every field; for an estimate the
    predicted summaries and for a verification the channels. Every value it keeps is that of detailed."""
    if tool == "rank_root_causes":
        assert set(concise) == set(detailed) - RANKING_DETAILED_ONLY
        for key, value in concise.items():
            if key == "candidates":
                assert value == [{k: v for k, v in c.items() if k != "current_combination"} for c in detailed[key]]
            elif key == "branches":
                assert value == [{k: v for k, v in b.items() if k not in BRANCH_REPEATS} for b in detailed[key]]
            else:
                assert value == detailed[key], key
    elif tool == "get_episode_evidence":
        assert set(concise) == set(detailed) | {"not_listed"}
        narrowed = detailed["sensor_group"] == "all" or detailed["sensor_group"] in view.sensor_groups
        listed = [entry for entry in detailed["fields"] if not narrowed or any(_marked(cell) for cell in entry["values"])]
        assert concise["fields"] == listed and sum(concise["not_listed"].values()) == len(detailed["fields"]) - len(listed)
        assert {k: v for k, v in concise.items() if k not in ("fields", "not_listed", "evidence_rule")} == \
            {k: v for k, v in detailed.items() if k not in ("fields", "evidence_rule")}
    else:
        left = {"estimate_intervention_effect": "predicted", "verify_outcome": "channels"}[tool]
        assert {k: v for k, v in concise.items() if k != "evidence_rule"} == {k: v for k, v in detailed.items() if k not in (left, "evidence_rule")}
