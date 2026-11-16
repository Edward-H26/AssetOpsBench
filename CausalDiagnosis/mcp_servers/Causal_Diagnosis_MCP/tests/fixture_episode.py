"""A fixture that writes released.json format 3 as CausalDiagnosis/docs/environment-actions-mcp.md describes it, so that
the diagnostics server is tested without the action server. It plays reads, advances, a diagnosis, work orders and their
closures over the live segments of the recorded blocks at one seeded entry offset, charges the default prices, and releases no
hidden field. episode_models gives the fits of an episode of a start, the block the driver names at launch. Reading k cycles at time t serves the cycles t to t + k - 1, each stamped with the cycle it was recorded in,
with the mean and the standard deviation of every channel of the group, as the environment does; a work order closes at
its expected cycle, the end of the cycle before it, from which the component is healthy and the recording block
switches."""
import random
from functools import lru_cache
from pathlib import Path

from CausalDiagnosis.datasets.dataset_loader.evidence_source import BLOCK_CAP, SEGMENT, RecordingSource, segment_positions
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS, HEALTHY, LEVELS, TIER_SENSORS, tier_features
from CausalDiagnosis.datasets.hyd_1.former_field_names import ASSET_ID, HYD1_FIELDS, public_name
from CausalDiagnosis.utils.atomic_writes_and_locks import write_json_atomically

TASK_TEXT = "Hydraulic unit hyd_1 was flagged by the performance monitor. Find the condition of every component and restore the unit."
DEFAULT_RULES = {  # the default configuration of the fixture
    "prices": {"read": {"tier1": 1, "tier2": 3, "tier3": 10}, "parts": {"cooler": 20, "valve": 20, "pump": 30, "accumulator": 20}, "repair_charge": 20},
    "durations": {"read_per_cycle": 1, "advance_per_cycle": 1, "lead_time": {c: 3 for c in COMPONENTS},
                  "repair_downtime": {"cooler": 1, "valve": 1, "pump": 2, "accumulator": 1}},
    "downtime_rate": {"cooler": {"100": 0, "20": 1, "3": 2}, "valve": {"100": 0, "90": 1, "80": 2, "73": 3}, "pump": {"0": 0, "1": 1, "2": 2},
                      "accumulator": {"130": 0, "115": 1, "100": 2, "90": 3}, "stopped": 5},
    "budget": 300, "horizon": 60, "max_steps": 60, "fault_rule": {"min_faulty": 1, "max_faulty": 2}, "clock_mode": "declared",
}


@lru_cache(maxsize=None)
def episode_models(start):
    """The fits of an episode that started in the combination, whose block every fit leaves out, built once per start."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools.fitted_models import load_models

    return load_models().without(tuple(start))


def normal_ranges(cycles, blocks):
    """The public normal range of every channel mean, the environment's own: its recording source normal ranges the values it serves, which
    are rounded, so a normal range of the raw values would differ at its edges and could put a served healthy cycle outside it."""
    return RecordingSource(get_profile(ASSET_ID), {c.index: c for c in cycles}, blocks).normal_ranges


def default_rules(cycles, blocks, **overrides):
    rules = {"components": list(COMPONENTS), "levels": {c: list(LEVELS[c]) for c in COMPONENTS}, "healthy": dict(HEALTHY),
             "sensor_groups": {f"tier{t}": [HYD1_FIELDS[s] for s in sensors] for t, sensors in TIER_SENSORS.items()}, "statistics": ["mean", "std"],
             **{k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_RULES.items()},
             "normal_ranges": normal_ranges(cycles, blocks)}
    rules.update(overrides)
    return rules


def with_misdiagnosis(rules, price):
    """The rules with prices.misdiagnosis at price, or without the key when price is None, as the files
    released before the key existed."""
    prices = {k: v for k, v in rules["prices"].items() if k != "misdiagnosis"}
    return {**rules, "prices": prices if price is None else {**prices, "misdiagnosis": price}}


def newest_per_summary(ids):
    """The identifiers that a candidate of the compact reply keeps of its identifiers in summary: of the readings, the newest
    that released each summary, in the order of the list, and every identifier that names no reading, such as a closed order."""
    newest = {}
    for i in ids:
        if i.startswith("obs/"):
            _, n, field = i.split("/", 2)
            newest[field] = max(newest.get(field, 0), int(n))
    return [i for i in ids if not i.startswith("obs/") or int(i.split("/", 2)[1]) == newest[i.split("/", 2)[2]]]


def assert_compact_shortens(compact, summary):
    """The compact reply against the summary reply of the same evidence and options: compact leaves out e10_rule and the
    keys of adapter.COMPACT_LEFT_OUT, holds the one assumption text A_COMPACT and no edge cutting score, and gives each
    candidate newest_per_summary of its identifiers in summary. Every other field is that of summary."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools import A_COMPACT, COMPACT_LEFT_OUT

    short, long = compact.model_dump(), summary.model_dump()
    assert set(short) == set(long) - {"e10_rule", *COMPACT_LEFT_OUT}
    assert short.pop("assumptions") == [A_COMPACT] and A_COMPACT not in long.pop("assumptions")
    for kept, full in zip(short["candidates"], long["candidates"]):
        assert kept.pop("evidence_ids") == newest_per_summary(full.pop("evidence_ids"))
    for kept, full in zip(short["elimination"], long["elimination"]):
        assert (kept.pop("ec2"), kept.pop("ec2_per_cost")) == (None, None)
        assert full.pop("ec2") is not None and full.pop("ec2_per_cost") is not None
    assert short == {key: value for key, value in long.items() if key in short}


class FixtureEpisode:
    """One episode played by hand: the agent facing events and observations are written to released.json after every call."""

    def __init__(self, folder, cycles, blocks, start, seed=0, rules=None, task_text=TASK_TEXT, guidance_level=2):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.by_index = {c.index: c for c in cycles}
        self.blocks = blocks
        self.rules = rules or default_rules(cycles, blocks)
        self.combo = list(start)
        self.start = tuple(start)  # the block the launcher of the diagnostics server leaves out of its fits
        self.entry_offset = random.Random(seed).randrange(BLOCK_CAP)  # the window of the environment, which the server never learns
        self.time, self.cost, self.steps, self.n_orders = 0, 0, 0, 0
        self.events, self.observations, self.open = [], [], []
        self.task_text, self.guidance_level, self.over = task_text, guidance_level, False
        self._enter_block()
        self.write()

    @property
    def released_path(self):
        return self.folder / "released.json"

    def _enter_block(self):
        block = self.blocks[tuple(self.combo)]
        self.window, self.pos, self.seen = [block[p] for p in segment_positions(self.entry_offset + SEGMENT)], 0, 0

    def _tick(self):
        """One cycle passes: orders whose expected cycle has come close and switch the block."""
        self.time += 1
        for order in [o for o in self.open if o["expected_close_t"] <= self.time]:
            j = COMPONENTS.index(order["component"])
            was_faulty = self.combo[j] != HEALTHY[order["component"]]
            self.combo[j] = HEALTHY[order["component"]]
            if was_faulty:
                self._enter_block()
            self.open.remove(order)
            self.events.append({"event": "work_order_closed", "step": None, "t_start": self.time, "t_end": self.time, "class": "system", "cost": 0,
                                "id": order["id"], "component": order["component"], "source": order["source"]})

    def _event(self, kind, t_start, cls, cost, **fields):
        self.steps += 1
        self.events.append({"event": kind, "step": self.steps, "t_start": t_start, "t_end": self.time, "class": cls, "cost": cost, **fields})
        self.cost += cost
        self.write()

    def read(self, group, cycles=1):
        tier = int(group[len("tier"):])
        t_start = self.time
        served = []
        for _ in range(cycles):
            idx = self.window[(self.pos + self.seen) % len(self.window)]
            repeated = self.seen >= len(self.window)
            self.seen += 1
            values = {public_name(k): float(f"{v:.6g}") for k, v in tier_features(self.by_index[idx], tier).items()
                      if k.endswith(("_mean", "_std"))}
            n = len(self.observations) + 1
            self.observations.append({"n": n, "t": self.time, "sensor_group": group, "values": values})
            served.append({"n": n, "t": self.time, "repeated": repeated, "values": values})
            self._tick()
        self._event("read", t_start, "read", cycles * self.rules["prices"]["read"][group], sensor_group=group, cycles=cycles, served=served)
        return served

    def advance(self, cycles):
        t_start = self.time
        for _ in range(cycles):
            self._tick()
        self._event("advance", t_start, "read", 0, cycles=cycles)

    def submit_diagnosis(self, diagnosis, evidence_ids=()):
        self._event("submit_diagnosis", self.time, "write", 0, diagnosis=dict(diagnosis), evidence_ids=list(evidence_ids))

    def declare_insufficient_evidence(self, separating_test, evidence_ids=()):
        self._event("declare_insufficient_evidence", self.time, "write", 0, separating_test=dict(separating_test), evidence_ids=list(evidence_ids))

    def create_work_order(self, component, prediction_id="", evidence_ids=()):
        self.n_orders += 1
        durations = self.rules["durations"]
        order = {"id": f"wo-{self.n_orders}", "component": component, "created_t": self.time,
                 "expected_close_t": self.time + durations["lead_time"][component] + durations["repair_downtime"][component], "status": "open",
                 "source": "agent"}
        self.open.append(order)
        price = self.rules["prices"]["parts"][component] + self.rules["prices"]["repair_charge"]
        self._event("work_order_created", self.time, "write", price, id=order["id"], component=component, prediction_id=prediction_id,
                    evidence_ids=list(evidence_ids), expected_close_t=order["expected_close_t"])
        return order["id"]

    def refuse(self, action, reason):
        self._event("refused", self.time, "read", 0, action=dict(action), reason=reason)

    def finish(self, summary=""):
        self.over = True
        self._event("finish", self.time, "write", 0, summary=summary)

    def write(self):
        released = {
            "format": 3, "asset_id": ASSET_ID, "task_text": self.task_text, "guidance_level": self.guidance_level, "rules": self.rules,
            "status": {"time": self.time, "budget_left": self.rules["budget"] - self.cost, "steps_left": self.rules["max_steps"] - self.steps,
                       "open_work_orders": [{k: o[k] for k in ("id", "component", "created_t", "expected_close_t", "status")}
                                            for o in self.open if o["source"] == "agent"],
                       "episode_over": self.over},
            "events": self.events, "observations": self.observations,
        }
        write_json_atomically(self.released_path, released, indent=1)
