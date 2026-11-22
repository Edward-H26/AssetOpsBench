"""The realized elimination of the paid reads of an episode, the measure of the scores table of the paper:
the filter of the CausalDiagnosis server replayed offline over the episode's released evidence, the same
yardstick for every agent and reference policy whether or not its run had the server. A paid read is a read event with a
charged price, or every read event on the free base row, where a call is
the unit of cost. Its realized elimination is N(b) of the belief predicted to the cycle of the read from every event
released before it, minus N(b) of that belief after the update with the read's own readings, N(b) being the effective
number of hypotheses of E21, so the replay runs to the cycle the read ends with and without those readings. Its cost is
the cost of E21 as the episode charged it: one call on the base row, and on the priced rows the price of the released
read event plus the cycles the read played times the expected production loss per cycle
(decision.expected_downtime_rate) of the belief the read met when the agent chose it, before its own update. The replay
runs the server's filter under the joint_block ranking model without the resampled fits of E6, which leave the masses of
the main fit unchanged, so the measure is a yardstick of the server's model of the asset and not of the agent's belief.

Beside it, for a run with the server, stands the expected elimination that the server reported for the read's sensor
group in the last rank_root_causes answer before the read, kept only when that answer saw every observation and closure
released before the read, the read served one full cycle of its group, and the effective number of hypotheses the
forecast started from equals the replay's before the read, so that the forecast was made on the belief the yardstick
measures.
"""
import hashlib
import sys
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent

from CausalDiagnosis.experiments.episode_runners import episode_launchers, run_configuration, run_folder_layout, servers_of_a_run
from CausalDiagnosis.utils import csv_rows
from CausalDiagnosis.utils import json_files  # noqa: E402,F401  puts src/ on the import path, as for every driver of this folder
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import cost_unit  # noqa: E402
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.fit_models import models_path
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools.fitted_models import load_models  # noqa: E402
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.diagnostics_model import EffectDiagnostics
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import profile_view  # noqa: E402
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.decision_rule import costs, expected_downtime_rate  # noqa: E402
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_filter import parse_released  # noqa: E402

TOLERANCE = 1e-6  # relative difference within which a forecast's effective number of hypotheses equals the replay's


class Yardstick:
    """The belief of the server's filter over a prefix of the released evidence, under the fits of one models.json that the
    server of the episode runs, which leave out the block of the combination the episode started in."""

    def __init__(self, path, left_out=None):
        models = load_models(path)  # raises FileNotFoundError with the fit command when models.json is absent
        if left_out is None:
            raise ValueError("the yardstick replays the fits of the episode's server, which leave out the block of its start, so it needs the start as left_out")
        self.models = models.without(left_out)
        self.backend = EffectDiagnostics(self.models, block=True, resampled=False)
        self.digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]

    def view(self, released):
        """The view of the asset that the server builds from the released rules (adapter.CausalTools._released)."""
        return profile_view(released["rules"])

    def effective(self, view, episode):
        """N(b) of the belief the filter replays over the episode's evidence."""
        return self.backend.replay(view, episode).posterior().effective_hypotheses()

    def rate(self, view, episode):
        """The expected production loss per cycle of the belief the filter replays, at the episode's cycle."""
        post = self.backend.replay(view, episode).posterior()
        return expected_downtime_rate(post.candidates, post.probabilities, costs(view, episode.time))

    def group_fields(self, group):
        return set(self.models.space.groups.get(group, ()))


def yardstick(path=None, left_out=None):
    """The yardstick of the models file at path, built once per file and start; without a path the models file of the default
    asset as fit_models holds it at the call, which the tests point at a copy."""
    return _yardstick(Path(models_path() if path is None else path), None if left_out is None else tuple(left_out))


@lru_cache(maxsize=None)
def _yardstick(path, left_out):
    return Yardstick(path, left_out)


def released_before(released, index, time, with_read=False):
    """The released evidence of the events before events[index], and of that event too when with_read, at the given cycle,
    in the form the filter reads: the readings those events served, their closures, their fix history and the rules of the
    row, which tell the filter whether time passes while the agent does not wait."""
    events = released.get("events", [])[: index + 1 if with_read else index]
    observations = [{"n": o["n"], "t": o["t"], "sensor_group": e["sensor_group"], "values": o["values"]}
                    for e in events if e.get("event") == "read" for o in e.get("served", [])]
    return parse_released({"events": events, "observations": observations, "status": {"time": int(time), "open_work_orders": []},
                           "rules": released["rules"]})


def is_paid(event, unit):
    """Whether a released read event is a paid read: every read on the free base row, where a call is the unit of cost, and
    on the priced rows a read with a charged price, so that a re-read of cycles paid before is no paid read there."""
    return event.get("event") == "read" and (unit == "calls" or float(event.get("cost") or 0) > 0)


def read_records(released, stick=None, left_out=None):
    """One record per paid read of the released evidence in event order: the event's index, sensor group and cycles, N(b)
    before and after the read, the realized elimination, the loss rate of the belief the read met and the cost of E21 as
    charged. Empty, without loading the models, when the evidence holds no read. left_out is the combination the episode
    started in, whose block the fits of the replay leave out, as the server's do."""
    events = released.get("events", [])
    if not any(e.get("event") == "read" for e in events):
        return []
    unit = cost_unit(released["rules"]["prices"])
    reads = [(i, e) for i, e in enumerate(events) if is_paid(e, unit)]
    if not reads:
        return []
    stick = stick or yardstick(left_out=left_out)
    view = stick.view(released)
    records = []
    for index, event in reads:
        start, end = int(event["t_start"]), int(event["t_end"])
        before = stick.effective(view, released_before(released, index, end))
        after = stick.effective(view, released_before(released, index, end, with_read=True))
        rate = stick.rate(view, released_before(released, index, start)) if unit != "calls" and end > start else 0.0
        cost = 1.0 if unit == "calls" else float(event.get("cost") or 0) + (end - start) * rate
        records.append({"index": index, "sensor_group": event.get("sensor_group"), "t_start": start, "t_end": end,
                        "effective_before": before, "effective_after": after, "eliminated": before - after, "rate": rate, "cost": cost})
    return records


def forecasts(answers):
    """What each rank_root_causes answer saw and forecast, in call order: the observation numbers and closure identifiers
    among its evidence identifiers and, per sensor group, the effective number of hypotheses before one cycle of the group
    and the expected elimination of that cycle. An answer without elimination rows, as from the servers before version 7,
    forecasts nothing."""
    out = []
    for answer in answers:
        rows = [r for r in answer.get("elimination") or [] if r.get("cycles") == 1 and r.get("expected_elimination") is not None
                and r.get("effective_before") is not None]
        if not rows:
            continue
        ids = [str(i) for i in answer.get("evidence_ids") or []]
        out.append({"observations": {int(i.split("/")[1]) for i in ids if i.startswith("obs/")},
                    "closures": {i[len("wo/"):] for i in ids if i.startswith("wo/")},
                    "groups": {r["sensor_group"]: (float(r["effective_before"]), float(r["expected_elimination"])) for r in rows}})
    return out


def calibrate(records, released, answers, stick=None, left_out=None):
    """Sets expected on every record whose read the server forecast on the belief the yardstick measures (module
    docstring), from the parsed rank_root_causes answers of the episode in call order. The other records stay without it."""
    found = forecasts(answers)
    if not records or not found:
        return records
    stick = stick or yardstick(left_out=left_out)
    events = released.get("events", [])
    for record in records:
        served = events[record["index"]].get("served", [])
        if len(served) != 1 or not stick.group_fields(record["sensor_group"]) <= set(served[0].get("values", {})):
            continue
        before = events[: record["index"]]
        seen = {o["n"] for e in before if e.get("event") == "read" for o in e.get("served", [])}
        closed = {str(e["id"]) for e in before if e.get("event") == "work_order_closed" and e.get("id")}
        fitting = [f for f in found if f["observations"] >= seen and served[0]["n"] not in f["observations"] and f["closures"] >= closed]
        if not fitting or record["sensor_group"] not in fitting[-1]["groups"]:
            continue
        effective_before, expected = fitting[-1]["groups"][record["sensor_group"]]
        if abs(effective_before - record["effective_before"]) <= TOLERANCE * max(1.0, abs(effective_before)):
            record["expected"] = expected
    return records


def episode_measures(records):
    """The per episode fields of the elimination tables: the paid reads, the realized elimination per read and per unit of
    cost, and over the reads with a forecast their number, the mean expected and realized elimination and their gap."""
    out = {"paid_reads": len(records), "realized_elimination_per_read": None, "realized_elimination_per_cost": None,
           "calibrated_reads": 0, "expected_elimination": None, "realized_elimination_calibrated": None, "calibration_gap": None}
    if not records:
        return out
    eliminated, cost = sum(r["eliminated"] for r in records), sum(r["cost"] for r in records)
    out["realized_elimination_per_read"] = eliminated / len(records)
    out["realized_elimination_per_cost"] = eliminated / cost if cost > 0 else None
    calibrated = [r for r in records if "expected" in r]
    if calibrated:
        expected = sum(r["expected"] for r in calibrated) / len(calibrated)
        realized = sum(r["eliminated"] for r in calibrated) / len(calibrated)
        out.update({"calibrated_reads": len(calibrated), "expected_elimination": expected,
                    "realized_elimination_calibrated": realized, "calibration_gap": realized - expected})
    return out
