"""A fixture that writes released.json format 2 as the build plan of 2026-11-01 (sections 4 and 5) describes it, so that
the diagnostics server is tested without the action server. It plays reads, advances, a diagnosis, work orders and their
closures over the recorded blocks with a seeded window per block, charges the plan's placeholder prices, and releases no
hidden field. Reading k cycles at time t serves the cycles t + 1 to t + k with the mean and the standard deviation of every
channel of the group, as the action server does; a work order closes at its expected cycle, from which the component is
healthy and the recording block switches."""
import random
from pathlib import Path

import numpy as np

from envs.hydraulic_rig.data import COMPONENTS, HEALTHY, LEVELS, TIER_SENSORS, tier_features
from servers.causal_diagnosis.effect_model import HEALTHY_COMBO
from servers.hydraulic_rig.names import ASSET_ID, HYD1_FIELDS, public_name
from servers.json_files import write_json_atomically

BLOCK_CAP = 10
TASK_TEXT = "Hydraulic unit hyd_1 was flagged by the performance monitor. Find the condition of every component and restore the unit."
PLACEHOLDER_RULES = {  # the placeholder configuration of the build plan, section 2
    "prices": {"read": {"tier1": 1, "tier2": 3, "tier3": 10}, "parts": {"cooler": 20, "valve": 20, "pump": 30, "accumulator": 20}, "repair_charge": 20},
    "durations": {"read_per_cycle": 1, "advance_per_cycle": 1, "lead_time": {c: 3 for c in COMPONENTS},
                  "repair_downtime": {"cooler": 1, "valve": 1, "pump": 2, "accumulator": 1}},
    "downtime_rate": {"cooler": {"100": 0, "20": 1, "3": 2}, "valve": {"100": 0, "90": 1, "80": 2, "73": 3}, "pump": {"0": 0, "1": 1, "2": 2},
                      "accumulator": {"130": 0, "115": 1, "100": 2, "90": 3}, "stopped": 5},
    "budget": 300, "horizon": 60, "max_steps": 60, "fault_rule": {"min_faulty": 1, "max_faulty": 2}, "clock_mode": "declared",
}


def healthy_bands(cycles, blocks):
    """The public band of every channel mean: the 1st and 99th percentiles over the healthy combination's stable cycles."""
    by_index = {c.index: c for c in cycles}
    rows = [tier_features(by_index[i], max(TIER_SENSORS)) for i in blocks[HEALTHY_COMBO]]
    bands = {}
    for name in rows[0]:
        if name.endswith("_mean"):
            values = np.array([r[name] for r in rows])
            bands[public_name(name)] = {"low": float(np.percentile(values, 1)), "high": float(np.percentile(values, 99))}
    return bands


def default_rules(cycles, blocks, **overrides):
    rules = {"components": list(COMPONENTS), "levels": {c: list(LEVELS[c]) for c in COMPONENTS}, "healthy": dict(HEALTHY),
             "sensor_groups": {f"tier{t}": [HYD1_FIELDS[s] for s in sensors] for t, sensors in TIER_SENSORS.items()}, "statistics": ["mean", "std"],
             **{k: (dict(v) if isinstance(v, dict) else v) for k, v in PLACEHOLDER_RULES.items()},
             "healthy_bands": healthy_bands(cycles, blocks)}
    rules.update(overrides)
    return rules


class FixtureEpisode:
    """One episode played by hand: the agent facing events and observations are written to released.json after every call."""

    def __init__(self, folder, cycles, blocks, start, seed=0, rules=None, task_text=TASK_TEXT, guidance_level=2):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.by_index = {c.index: c for c in cycles}
        self.blocks = blocks
        self.rules = rules or default_rules(cycles, blocks)
        self.combo = list(start)
        self.rng = random.Random(seed)
        self.time, self.cost, self.steps, self.n_orders = 0, 0, 0, 0
        self.events, self.observations, self.open, self.pending_external = [], [], [], []
        self.task_text, self.guidance_level, self.over = task_text, guidance_level, False
        self._enter_block()
        self.write()

    @property
    def released_path(self):
        return self.folder / "released.json"

    def _enter_block(self):
        block = self.blocks[tuple(self.combo)]
        cap = min(BLOCK_CAP, len(block))
        first = self.rng.randrange(len(block) - cap + 1)
        self.window, self.pos, self.seen = block[first:first + cap], self.rng.randrange(cap), 0

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
            self._tick()
            idx = self.window[(self.pos + self.seen) % len(self.window)]
            repeated = self.seen >= len(self.window)
            self.seen += 1
            values = {public_name(k): float(f"{v:.6g}") for k, v in tier_features(self.by_index[idx], tier).items()
                      if k.endswith(("_mean", "_std"))}
            n = len(self.observations) + 1
            self.observations.append({"n": n, "t": self.time, "sensor_group": group, "values": values})
            served.append({"n": n, "t": self.time, "repeated": repeated, "values": values})
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

    def external_replacement(self, component, at_t):
        """A technician replaces the component at cycle at_t and closes a work order the agent never opened."""
        self.open.append({"id": "wo-ext-1", "component": component, "created_t": at_t, "expected_close_t": at_t, "status": "open", "source": "external"})

    def refuse(self, action, reason):
        self._event("refused", self.time, "read", 0, action=dict(action), reason=reason)

    def finish(self, summary=""):
        self.over = True
        self._event("finish", self.time, "write", 0, summary=summary)

    def write(self):
        released = {
            "format": 2, "asset_id": ASSET_ID, "task_text": self.task_text, "guidance_level": self.guidance_level, "rules": self.rules,
            "status": {"time": self.time, "budget_left": self.rules["budget"] - self.cost, "steps_left": self.rules["max_steps"] - self.steps,
                       "open_work_orders": [{k: o[k] for k in ("id", "component", "created_t", "expected_close_t", "status")}
                                            for o in self.open if o["source"] == "agent"],
                       "episode_over": self.over},
            "events": self.events, "observations": self.observations,
        }
        write_json_atomically(self.released_path, released, indent=1)
