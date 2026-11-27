"""The hydraulic rig environment: a hidden combination of component conditions, a clock with priced and timed tools,
work orders whose effect on the levels a declared rule gives, and a log for the verifier (CausalDiagnosis/docs/environment.md).

An episode starts at cycle 0 with a hidden combination h(t0) drawn from the seed. The agent reads sensors at a price and
a duration, reads the clock, waits for the next notification, summarizes the changes since a cycle, looks the asset, its
failure modes, the fix history of the rig and the repair manual up, submits one diagnosis or one declaration of
insufficient evidence, creates standard or emergency work orders that a technician executes in visits, and finishes. The
two time tools follow the System app of ARE (time_tools.py, notifications.py): get_current_time never
moves the clock and wait_for_notification lets it run until the next public event or a timeout, so no tool sets the
time, and every answer lists the public events since the previous call as notifications. The deterioration and the failure
card run on the clock (episode_clock.py, CausalDiagnosis/environment/Causal_Diagnosis_Environment/layers). A declared rule gives every change of the combination, and every
reading after it is a recorded cycle of the new combination, so no reading is simulated.

Every malformed or out-of-order action that is a dict is refused (logged, no charge, counts a step, the episode
continues) rather than raised, so a language model agent that emits a bad action loses a step, not the episode; an
action that is not a dict is a programming error and raises. An integer argument of a tool is read by its value, whatever
JSON type writes it, so "3", 3.0 and 3 are the same argument, and a value that writes no integer is refused. An action the budget cannot pay ends the episode. The budget
counter falls by prices alone, while the downtime of the hidden condition, the failure penalties and the misdiagnosis
price of a wrong diagnosis go to a hidden ledger
that the end entry of the log keeps for the verifier, since a counter that fell with the downtime would tell the agent the
hidden condition. The log
keeps deep copies of everything the agent hands in and the agent facing views are copies, so no in-process policy can
rewrite its own diagnosis. Hidden fields live only in `log`; `public_event` and `public_profile` are the agent facing
views, and the oracle of oracle.py reads the hidden state.
"""
import copy
import random
import re

from CausalDiagnosis.datasets.dataset_loader import evidence_source as evidence, evidence_kinds as kinds
from CausalDiagnosis.datasets.dataset_loader.evidence_source import BAND_QUANTILES, BLOCK_CAP, RecordingSource  # noqa: F401  (served under their former names)
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
from CausalDiagnosis.datasets.dataset_loader.profile_validation import is_int
from CausalDiagnosis.datasets.dataset_loader.profile_data import processes_block
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import tool_contract as contract
from CausalDiagnosis.utils.public_rules.visits import NO_CALL_ACTION, joins, keeps_run

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock.episode_clock import RigClock
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers import layer_5_deterioration_and_failure as deterioration, layer_1_priced as priced, layer_2_real_time as real_time, layer_4_repair_rules as repair_rules
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_5_deterioration_and_failure import draw_hazard, draw_schedule
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_4_repair_rules import draw_fix_history
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock.notifications import Message, MessageType, timeout_message
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock.time_tools import WaitForNotificationTimeout, wait_for_next_notification
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET, ORDER_MODES, EpisodeConfig
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.oracle import oracle
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import MISDIAGNOSIS_EVENT, misdiagnosis_charge

_DEFAULT = get_profile(DEFAULT_ASSET, engine=False)  # the asset of the module level helpers that read its recordings directly
HEALTHY_COMBO = _DEFAULT.healthy_state()
# repeated marks a served cycle played again after its window wrapped, which after an unchanged close would tell the agent
# that a repair changed nothing, so it stays in the log, and held names the levels that the emergency holds in force at the
# end replaced, which the verifier scores
HIDDEN_FIELDS = ("cycle", "combination", "combination_after", "unnecessary", "hidden", "schedule", "final", "seed", "config",
                 "played", "repeated", "entry_offset", "held")
# the schedule and the hazard are hidden, so a timed episode leaks nothing that a static one does not, and an ineffective
# repair or an expired emergency repair shows only in the readings, as on a real rig
HIDDEN_EVENTS = ("scheduled_drop", "hazard_drop", "order_ineffective", "emergency_expired", MISDIAGNOSIS_EVENT)
# the public system events that notify, the alarms only of the sensor groups the agent has paid for
NOTIFYING_EVENTS = ("work_order_executed", "work_order_closed", "failure", "alarm")
WRITE_LOCK_REASON = "a diagnosis or a declaration was already submitted"
ORDER_BEFORE_WRITE_REASON = "a diagnosis or a declaration must precede a work order"
# the reasons of a malformed argument, worded by the parameters the tools take, one table for both surfaces
DIAGNOSIS_COMPONENTS_REASON = "each component must be given one condition level"
DIAGNOSIS_LEVEL_REASON = "a component's level must be one of its condition levels"
SEPARATING_TEST_REASON = "sensor_group must name a sensor group and cycles must be a positive integer"
EVIDENCE_IDS_REASON = "evidence_ids must be evidence identifiers, separated by commas in a tool call"
UNLISTED_ACTION = "unlisted_tool"  # the action of a call to a tool the row does not list, which the metering layer sends
# the totals of the end entry that the downtime of the hidden condition drives, which no public view may carry
END_LEDGER_FIELDS = ("cost", "ledger", "downtime_total", "downtime_by_cycle")
_INDEXED = {}  # {recording index: record} of a records object, for the worlds that evidence.world_table keeps


def clock_block(config):
    """The clock as the manifest states it: the contract's modes, order of one cycle and notifying events,
    with the episode's mode, seconds_per_cycle, think_seconds_per_cycle and the declared duration of every action tool in
    cycles per unit."""
    block = copy.deepcopy(contract.clock(config.asset_id))
    block.update(mode=config.clock_mode, seconds_per_cycle=config.seconds_per_cycle, think_seconds_per_cycle=config.think_seconds_per_cycle,
                 durations={name: config.tool_duration(name) for name in contract.tools_of(contract.ACTION_SERVER)})
    if config.clock_mode == real_time.TOKEN_MODE:
        block["modes"][real_time.TOKEN_MODE] = contract.token_clock(config.asset_id)["mode"]
        block["seconds_per_token"] = config.seconds_per_token
    return block


def served_values(cycle, sensors):
    """What a reading of the default asset returns for one recorded cycle (evidence.served_values)."""
    return evidence.served_values(cycle, sensors, _DEFAULT)


def mean_field(sensor):
    return evidence.mean_field(sensor, _DEFAULT)


def unlisted_reason(action):
    """The refusal of a call that the metering layer passes as the action unlisted_tool: the server it was asked of has no such
    tool in this episode, which holds whether or not another server of the row lists it. None for any other action."""
    if action.get("type") != UNLISTED_ACTION or not isinstance(action.get("tool"), str):
        return None
    return f"the server {action.get('server')} has no tool {action['tool']} in this episode"


class HydraulicRigEnv:
    def __init__(self, cycles, blocks, config=None):
        """cycles and blocks come from data.load_cycles and data.build_blocks; config is an EpisodeConfig (default when
        None). The start candidates are the combinations within the fault rule, restricted to config.allowed."""
        self.cycles = evidence.world_table(_INDEXED, (cycles,), lambda: {c.index: c for c in cycles})
        self.blocks = blocks
        # the SHA-256 of the data file the recordings were read from, None for recordings that came from no file
        self.facts_sha256 = getattr(cycles, "facts_sha256", None)
        self.config = config if config is not None else EpisodeConfig()
        self.profile = get_profile(self.config.asset_id)
        self.source = kinds.kind_of(self.profile.evidence_kind).part("source")(self.profile, self.cycles, blocks)
        if self.config.start is not None and not self.source.starts(tuple(self.config.start["combination"])):
            raise ValueError(f"the start combination {list(self.config.start['combination'])} has no recording in the data of "
                             f"{self.config.asset_id}")
        if self.config.start is not None:
            combination, offset = tuple(self.config.start["combination"]), self.config.start["entry_offset"]
            offsets = sorted({entry for _window_start, entry in self.source.starts(combination)})
            if offset not in offsets:
                raise ValueError(f"the start entry offset {offset} is not among the {len(offsets)} entry offsets that the data of "
                                 f"{self.config.asset_id} serves for {list(combination)}: {offsets}")
        self.candidates = self.source.start_candidates(self.config)
        if not self.candidates:
            raise ValueError("no start combination satisfies the fault rule within the allowed set")
        self.normal_ranges = self.source.normal_ranges
        self._starts = {}  # seed -> (combination, window start, entry offset) of its episode
        self._handlers = {
            NO_CALL_ACTION: self._no_call,  # an iteration of the agent that called no tool of the rig
            "get_asset_profile": self._get_asset_profile, "read_sensors": self._read_sensors,
            "get_current_time": self._get_current_time, "wait_for_notification": self._wait_for_notification,
            "summarize_changes_since": self._summarize_changes_since, "get_failure_modes": self._get_failure_modes,
            "get_fix_history": self._get_fix_history, "lookup_fix": self._lookup_fix,
            "submit_diagnosis": self._submit_diagnosis, "declare_insufficient_evidence": self._declare_insufficient_evidence,
            "create_work_order": self._create_work_order, "finish": self._finish,
        }
        self.done = True

    # ----- episode control -----
    def reset(self, seed):
        """Draw the episode from the seed, in this order: the hidden combination (unless config.start names it), the
        served window and the degradation schedule, then the uniforms of the hazard and the fix history from generators of
        their own. A source that records the work orders of a start gives the fix history while its card is on. Returns the
        first envelope, whose result is the asset profile."""
        self._rng = random.Random(seed)
        hidden, offset = self._start_draw(self._rng)
        self.initial = hidden
        self.source.forget()
        self.clock = RigClock(self.config, self.source, self._rng, hidden, offset)
        self._starts[seed] = (hidden, self.clock.window_start, self.clock.entry_offset)
        self.clock.schedule = draw_schedule(self._rng, self.config, hidden)
        # streams apart from the episode's generator, so that neither changes the start nor any later draw
        self.clock.hazard = draw_hazard(random.Random(seed + 2_000_003), self.config)
        if hasattr(self.source, "fix_history") and "fix_history" in repair_rules.cards(self.config):
            self.fix_history = copy.deepcopy(self.source.fix_history(hidden))  # the recorded work orders of the start
        else:
            self.fix_history = draw_fix_history(random.Random(f"fix-history:{seed}"), self.config, hidden)
        self.cost = 0  # prices, downtime and penalties, the cost the verifier scores
        self.prices_paid = 0  # the prices alone, which the budget counter subtracts
        self.ledger = {"downtime": 0, "penalties": 0}  # hidden, so the counter never moves with the hidden condition
        self.steps = 0
        self.diagnosed = False
        self.declared = False
        self.done = False
        self._observations = 0
        self._strict_failure = False
        self._step_charges = 0
        self._step_messages = []
        self._run_visit = None  # the visit of the run of orders: the last step was an accepted standard order of it
        self._alarmed = set()
        self._paid_groups = set()  # the sensor groups of the reads that served a reading, which _entry keeps
        self._alarm_checked, self._alarm_paid = 0, ()  # the played cycles and the paid channels of the last alarm check
        self.log = [{"event": "reset", "step": None, "t_start": 0, "t_end": 0, "class": "system", "cost": 0, "seed": seed,
                     "hidden": list(hidden), "entry_offset": self.clock.entry_offset, "schedule": copy.deepcopy(self.clock.schedule),
                     "fix_history": copy.deepcopy(self.fix_history),
                     "config": self.config.to_json()}]
        self._notified = len(self.log)
        return self.envelope(self.public_profile())

    def start_of(self, seed):
        """(combination, window start, entry offset) of the episode of the seed, the state that reset(seed) reaches, learned
        without a reset where the source draws its start alone: the first draws of reset from a generator of the seed. The
        state is a function of the seed, the configuration and the world, which the environment holds for its whole life, so
        it is kept per seed, and reset keeps the state it reaches as well."""
        if seed not in self._starts:
            if hasattr(self.source, "draw_start"):
                rng = random.Random(seed)
                hidden, offset = self._start_draw(rng)
                self._starts[seed] = (hidden, *self.source.draw_start(hidden, rng, offset))
            else:
                self.reset(seed)
        return self._starts[seed]

    def _start_draw(self, rng):
        """The first draw of an episode from its generator: (hidden combination, entry offset), the fixed start of the
        configuration, or a choice among the candidates whose entry offset the source draws (None)."""
        if self.config.start is not None:
            return tuple(self.config.start["combination"]), self.config.start["entry_offset"]
        return rng.choice(self.candidates), None

    @property
    def time(self):
        return self.clock.time

    @property
    def hidden(self):
        """The current combination as a list (hidden from agents; the oracle and the tests read it)."""
        return list(self.clock.combination)

    @property
    def window_start(self):
        return self.clock.window_start

    @property
    def entry_offset(self):
        return self.clock.entry_offset

    def combination_at(self, t):
        return self.clock.combination_at(t)

    def cycle_at(self, t):
        """The recording index the rig played during cycle t, or None when it was stopped or has not run that cycle (hidden)."""
        return dict(self.clock.played).get(t)

    def oracle(self):
        return oracle(self)

    # ----- agent facing views -----
    def public_profile(self):
        """The answer of get_asset_profile: everything public about the asset and the episode's rules. Every row of the tool
        table carries the price and the declared duration of a call under this configuration, zero where the tool is free or
        takes no time, the work order row offers the emergency mode only while its card is on, and the clock
        block of the contract states the clock modes and the order of one cycle.
        The processes block holds the parameters of the rules, and the fault rule of the starts stays private."""
        c = self.config
        servers = [contract.ACTION_SERVER] + ([contract.DIAGNOSTICS_SERVER] if c.diagnostics_server else [])
        return {
            **({"environment_mode": c.environment_mode} if c.environment_mode is not None else {}),
            "asset_id": c.asset_id,
            "asset": self.profile.registry_entry(),
            "components": list(c.components),
            "levels": {k: list(v) for k, v in c.levels.items()},
            "healthy": dict(c.healthy),
            "sensor_groups": {g: [self.profile.field_name(s) for s in sensors] for g, sensors in c.sensor_groups.items()},
            "statistics": list(self.profile.statistics["served"]),
            "prices": copy.deepcopy(c.prices),
            "durations": copy.deepcopy(c.durations),
            "downtime_rate": copy.deepcopy(c.downtime_rate),
            "budget": c.budget,
            "horizon": c.horizon,
            "max_steps": c.max_steps,
            "processes": processes_block(c),
            "clock_mode": c.clock_mode,
            "seconds_per_cycle": c.seconds_per_cycle,
            "think_seconds_per_cycle": c.think_seconds_per_cycle,
            "clock": clock_block(c),
            "tools": [{**row, "price_value": c.tool_price(row["name"]), "duration_value": c.tool_duration(row["name"])}
                      for row in contract.public_table(servers, c.asset_id, () if c.emergency_on() else ("emergency",))],
            "normal_ranges": copy.deepcopy(self.normal_ranges),
        }

    def fix_manual(self, component):
        """The answer of lookup_fix, the repair manual of the repair rules layer."""
        return repair_rules.manual(self.config, component, self._fields_indicating(component))

    @staticmethod
    def public_event(entry):
        """The agent facing view of a log entry: without the recording indices, the combinations, the unnecessary flag,
        the hidden start, the drawn processes, the final combination, the seed and the configuration. A scheduled drop
        has no public view and gives None; a failure keeps its time but not its component, which the agent has to find."""
        if entry["event"] in HIDDEN_EVENTS:
            return None
        public = {k: copy.deepcopy(v) for k, v in entry.items() if k not in HIDDEN_FIELDS}
        if entry["event"] == "failure":
            public.pop("component", None)
        if entry["event"] == "end":
            for key in END_LEDGER_FIELDS:
                public.pop(key, None)
        if "served" in public:
            public["served"] = [{k: v for k, v in obs.items() if k not in HIDDEN_FIELDS} for obs in public["served"]]
        return public

    def public_events(self):
        """The public views of every log entry after reset, in order, without the entries the agent never sees."""
        return [p for p in (self.public_event(e) for e in self.log[1:]) if p is not None]

    def public_work_orders(self, closed_since=None):
        """The open work orders, or with closed_since the orders closed at or after that cycle."""
        if closed_since is None:
            return [o.public() for o in self.clock.open_orders()]
        return [o.public() for o in self.clock.orders if o.status == "closed" and o.closed_t >= closed_since]

    def envelope(self, result=None, refused=None, notifications=()):
        """The answer of a call: the clock, the counters, the open work orders, the notifications since the previous call
        and the result, or a refusal without result. Without a budget there is no budget counter to show."""
        envelope = {} if refused is None else {"refused": refused}
        envelope["time"] = self.time
        if self.config.anchors is not None:
            envelope["day"] = self.config.date_of(self.time)
        if self.config.budget is not None:
            envelope["budget_left"] = self.budget_left()
        envelope.update({"steps_left": self.config.max_steps - self.steps, "open_work_orders": self.public_work_orders(),
                         "episode_over": self.done, "notifications": list(notifications)})
        if refused is None:
            envelope["result"] = copy.deepcopy(result)
        return envelope

    def budget_left(self):
        """The budget less the prices paid, which never moves with the downtime or a penalty."""
        return self.config.budget - self.prices_paid

    def notifications(self):
        """The messages of the public events logged since the previous call, in log order, then the timeout of a wait that
        reached it and the stop of an episode that ended (the message types of ARE's notification system). Hidden events
        and the agent's own actions never notify."""
        messages = [self._notification(e) for e in self.log[self._notified:] if e["step"] is None and e["event"] in NOTIFYING_EVENTS]
        messages += self._step_messages
        if self.done:
            messages.append(Message(MessageType.ENVIRONMENT_STOP, f"The episode ended ({self.log[-1]['reason']}).", self.time)
                            .as_dict(event="end", reason=self.log[-1]["reason"]))
        return messages

    def _notification(self, entry):
        public = self.public_event(entry)
        fields = {k: v for k, v in public.items() if k not in ("event", "step", "t_start", "t_end", "class", "cost")}
        text = {
            "work_order_executed": lambda: (f"The technician began the repair of work order {public['id']}, and the "
                                            f"{self.profile.text['short_noun']} is stopped until it closes."),
            "work_order_closed": lambda: f"Work order {public['id']} on the {public['component']} closed.",
            "failure": lambda: "A component failed.",
            "alarm": lambda: f"The channel {public['channel']} left its normal range at cycle {entry['t_end']}.",
        }[entry["event"]]()
        return Message(MessageType.ENVIRONMENT_NOTIFICATION, text, entry["t_end"]).as_dict(event=entry["event"], **fields)

    def _notification_pending(self):
        return any(e["step"] is None and e["event"] in NOTIFYING_EVENTS for e in self.log[self._notified:])

    # ----- the step -----
    def step(self, action, think_cycles=0):
        """Apply one action. think_cycles is the agent's thinking time in cycles, charged before the action in the two
        thinking modes of the clock and ignored otherwise. Returns (envelope, cost, done, info), where cost is everything
        the step charged (price, downtime and penalties) and info holds `refused` after a refusal and the full log with
        its hidden state on the last step, which no tool server may forward."""
        if self.done:
            raise RuntimeError("episode is over; call reset")
        if not isinstance(action, dict):
            raise TypeError("action must be a dict")
        if not (is_int(think_cycles) and think_cycles >= 0):
            raise TypeError("think_cycles must be a non-negative integer")
        self.steps += 1
        self._step_charges = 0
        self._step_messages = []
        # the visit of the run of orders this step may join; any step but an accepted standard order ends the run, except a
        # no_call step, which calls no tool of the rig and leaves the run open (visits.py keeps_run)
        if not keeps_run(action.get("type")):
            self._joinable, self._run_visit = self._run_visit, None
            if action.get("type") != "create_work_order":
                self._settle_visit()
        think = real_time.thinking(self.config, think_cycles)
        self._advance(think)
        if self._cut_short() and action.get("type") == NO_CALL_ACTION:
            # a no_call is no tool call, so its thinking that ran into the end is its own event and no refusal
            self._log_no_call(self.time, think)
            result, refusal = {"no_call": True, "time": self.time}, None
        elif self._cut_short():
            # the thinking time alone ran into the horizon or a strict failure, so the action never runs
            self._log_refusal(action, "the episode ended before the action ran", self.time, think)
            refusal = "the episode ended before the action ran"
            result = None
        else:
            if think > 0:
                # an alarm of a thinking cycle is pending when the action runs, as a failure or a close of those cycles is
                self._record_alarms()
            kind = action.get("type")
            handler = self._handlers.get(kind, self._unknown) if isinstance(kind, str) else self._unknown
            result, refusal = handler(action, think)
        if not self.done:
            if self._strict_failure:
                self._end("failure")
            elif self.time >= self.config.horizon:
                self._end("horizon")
            elif self.steps >= self.config.max_steps:
                self._end("max_steps")
        if not self.done:
            self._record_alarms()
        info = {"refused": refusal} if refusal else {}
        if self.done:
            info.update(log=self.log, reason=self.log[-1]["reason"])
        notifications = self.notifications()
        if action.get("type") != NO_CALL_ACTION:
            # a no_call step answers no agent, so its notifications stay pending for the next counted call
            self._notified = len(self.log)
        return self.envelope(result, refusal, notifications), self._step_charges, self.done, info

    def _cut_short(self):
        return self._strict_failure or self.time >= self.config.horizon

    def _advance(self, cycles):
        """Move the clock by `cycles`, stopping at the horizon or at a failure under the strict rule. Returns the list of
        (t, played) of the advanced cycles, where played is (recording index, repeated) or None for a stopped cycle."""
        advanced = []
        for _ in range(min(cycles, self.config.horizon - self.time)):
            if self._strict_failure:
                break
            t = self.time
            events, played, charge = self.clock.advance_cycle()
            self._step_charges += charge
            self.cost += charge
            self.ledger["downtime"] += charge
            self._record(events)
            advanced.append((t, played))
        return advanced

    def _record(self, events):
        """Log the system events of the clock, a failure's penalty going to the hidden ledger."""
        for e in events:
            cost = 0
            if e["event"] == "failure":
                cost, strict = deterioration.failure_charge(self.config)
                self._step_charges += cost
                self.cost += cost
                self.ledger["penalties"] += cost
                self._strict_failure = self._strict_failure or strict
            fields = {k: v for k, v in e.items() if k not in ("event", "t")}
            self.log.append({"event": e["event"], "step": None, "t_start": e["t"], "t_end": e["t"], "class": "system", "cost": cost, **fields})

    def _charge_misdiagnosis(self, t, price):
        """Log the misdiagnosis price of a wrong diagnosis as a hidden system entry at the cycle of its write and add it to
        the hidden ledger beside the failure penalties, where no counter, reply, notification or release shows it."""
        self._step_charges += price
        self.cost += price
        self.ledger["penalties"] += price
        self.log.append({"event": MISDIAGNOSIS_EVENT, "step": None, "t_start": t, "t_end": t, "class": "system", "cost": price})

    def _record_alarms(self):
        """Log an alarm the first time a channel of a paid sensor group stood outside its normal range in a cycle the rig
        ran, once per channel and at the cycle the change summary names, so that a notification never tells
        of a group the agent did not pay for. It runs after every cycle of a wait and after every action that leaves the
        episode running. Under the paid channels of the last check only the cycles played since are checked, since a paid
        channel that stood outside its normal range in a checked cycle was alarmed then, while new paid channels check every cycle,
        and a source without outside() is asked over the whole history."""
        paid = self._paid_sensors()
        if paid == self._alarm_paid and hasattr(self.source, "outside"):
            alarms = self.source.alarms(self.clock.played[self._alarm_checked:], 0, paid)
        else:
            alarms = self._alarms(0)
        self._alarm_checked, self._alarm_paid = len(self.clock.played), paid
        for alarm in alarms:
            if alarm["channel"] not in self._alarmed:
                self._alarmed.add(alarm["channel"])
                self.log.append({"event": "alarm", "step": None, "t_start": alarm["first_t"], "t_end": alarm["first_t"], "class": "system",
                                 "cost": 0, "channel": alarm["channel"]})

    def _duration(self, cycles, per_cycle):
        """The clock advance of a timed action in the current mode."""
        return real_time.action_cycles(self.config, cycles, per_cycle)

    def _entry(self, event, t_start, think, price, **fields):
        """Append the log entry of an accepted action; the class comes from the contract. A read that served a reading adds
        its sensor group to the paid groups, which _paid_sensors reads."""
        entry = {"event": event, "step": self.steps, "t_start": t_start, "t_end": self.time, "class": contract.tool_class(_TOOL_OF_EVENT[event]),
                 "cost": price, "think_cycles": think, **fields}
        self.log.append(entry)
        if event == "read" and fields.get("served"):
            self._paid_groups.add(fields["sensor_group"])
        return entry

    def _log_refusal(self, action, reason, t_start, think):
        kind = action.get("type")
        if not (isinstance(kind, str) and kind in self._handlers):
            klass = "read"
        elif kind in contract.load_contract()["tools"]:
            klass = contract.tool_class(kind)
        else:  # a handler with no row in the contract (no_call) is no tool, so it has no class of one
            klass = "none"
        self.log.append({"event": "refused", "step": self.steps, "t_start": t_start, "t_end": self.time, "class": klass, "cost": 0,
                         "think_cycles": think, "action": copy.deepcopy(action), "reason": reason})

    def _refuse(self, action, reason, think):
        t_start = self.time
        self._advance(self._duration(0, 0))  # in instant mode every agent step takes a cycle, a wasted one included
        self._log_refusal(action, reason, t_start, think)
        return None, reason

    def _charge(self, action, price, think):
        """Charge a price, or refuse for lack of budget and end the episode. Returns whether the price was charged. The
        budget limits the prices paid alone, so the downtime and the penalties of the ledger never stop an action."""
        if not priced.affordable(self.config, self.prices_paid, price):
            self._log_refusal(action, "budget", self.time, think)
            self._end("budget")
            return False
        self.prices_paid += price
        self.cost += price
        self._step_charges += price
        return True

    def _settle_visit(self):
        """On a row without a clock whose repair card is on, the orders created since the last call that was not an order
        form one visit, which closes in priority order before the next call that is not an order runs and before the end of
        the episode, so a dependency created after its dependent still takes first."""
        if not self.config.has_clock() and self.config.repair_card_on():
            self._record(self.clock.settle_due_orders())

    def _settle_open_orders(self):
        """On a row with a clock, finish settles the open work orders. With no further agent call the
        clock runs to the last close, so the downtime and the events of that span count and the end entry holds the state
        after the last close, and an order of zero repair downtime due at the cycle reached closes there, as the order step
        of that cycle would. No alarm is logged in the span, since the agent's summary was written before it. The horizon
        or a failure under the strict rule stops the clock first and leaves the rest open, as the step cap does. A row
        without a clock keeps its own rule, the visit that _end settles."""
        if not self.config.has_clock():
            return
        self._advance(max((o.close_t for o in self.clock.open_orders()), default=self.time) - self.time)
        if not self._cut_short():
            self._record(self.clock.settle_due_orders())

    def _end(self, reason):
        """End the episode with the end entry of the log. A component that an emergency hold keeps at its healthy level at
        the end is listed under held with the level the hold replaced, which the verifier scores, and the key stands only
        while such a hold is in force, so the end entry of every other episode keeps its bytes."""
        self._settle_visit()
        self.done = True
        entry = {"event": "end", "step": None, "t_start": self.time, "t_end": self.time, "class": "system", "reason": reason,
                 "time": self.time, "final": self.hidden, "cost": self.cost, "ledger": dict(self.ledger),
                 "downtime_total": self.clock.downtime_total,
                 "downtime_by_cycle": copy.deepcopy(self.clock.downtime),
                 "played": [{"t": t, "cycle": index, "combination": list(self.combination_at(t))} for t, index in self.clock.played]}
        if self.clock.temporary:
            entry["held"] = {c: self.clock.temporary[c]["level"] for c in self.config.components if c in self.clock.temporary}
        self.log.append(entry)

    # ----- the twelve actions -----
    def _no_call(self, action, think):
        """The step of an agent's iteration that made no counted call: one step of the step cap, the clock advanced as by a
        step of no duration, no charge, the event no_call in the log (no tool of the contract, so no class of one)."""
        t_start = self.time
        self._advance(self._duration(0, 0))
        self._log_no_call(t_start, think)
        return {"no_call": True, "time": self.time}, None

    def _log_no_call(self, t_start, think):
        self.log.append({"event": "no_call", "step": self.steps, "t_start": t_start, "t_end": self.time, "class": "none", "cost": 0,
                         "think_cycles": think})

    def _unknown(self, action, think):
        return self._refuse(action, unlisted_reason(action) or "unknown action type", think)

    def _get_asset_profile(self, action, think):
        t_start = self.time
        self._advance(self._duration(0, 0))
        result = self.public_profile()
        self._entry("profile", t_start, think, 0)
        return result, None

    def _read_sensors(self, action, think):
        """Serve the sensor group's channels for the next cycles the rig plays. The rule of the clock rows
        is that a read of cycles already recorded returns at once, as AssetOpsBench's history does, while a read of cycles
        not yet recorded plays them on the rig and takes their declared duration, both charged per newly paid cycle. This
        engine records a cycle only by playing it, so every read here plays new cycles, and the metering layer of the
        AssetOpsBench rows applies the rule to its recorded history."""
        group, cycles = action.get("sensor_group"), _count(action.get("cycles"), 1)
        if not isinstance(group, str) or group not in self.config.sensor_groups:
            return self._refuse(action, "unknown sensor group", think)
        if cycles is None:
            return self._refuse(action, "cycles must be a positive integer", think)
        if self._duration(cycles, self.config.durations["read_per_cycle"]) < cycles and cycles > self.config.horizon:
            # the clock does not cover the cycles of such a read, which come at once, so one call never serves more
            # observations than the horizon holds cycles
            return self._refuse(action, "cycles must not exceed the horizon", think)
        price = self._read_price(group, cycles)
        if not self._charge(action, price, think):
            return None, "budget"
        t_start = self.time
        served, stopped = self._serve(self.config.sensor_groups[group], cycles)
        self._entry("read", t_start, think, price, sensor_group=group, cycles=cycles, stopped=stopped, served=served)
        return {"sensor_group": group, "served": [{k: v for k, v in o.items() if k not in HIDDEN_FIELDS} for o in served], "stopped": stopped}, None

    def _serve(self, sensors, cycles):
        """Play a read of the channels for `cycles` cycles: (the served observations, the cycles stopped for a repair). A
        cycle cut by the horizon or by a failure under the strict rule returns nothing, and so does a cycle for which the
        source holds no recording, which is no stopped cycle either."""
        served, stopped, remaining = [], 0, cycles
        # served cycle i is the recording the rig played during that cycle, so a change of the combination mid-read shows
        for t, played in self._advance(self._duration(cycles, self.config.durations["read_per_cycle"])):
            if remaining == 0:
                continue
            remaining -= 1
            if played is not None:
                served.append(self._observation(t, played, sensors))
            elif t in self.clock.stopped_cycles:
                stopped += 1
        # cycles the clock did not cover (instant mode, zero durations) come from the window at the current time
        while remaining > 0 and not self._cut_short():
            remaining -= 1
            if self.clock.stopped_at(self.time):
                stopped += 1
                continue
            played = self.clock.next_cycle()
            if played is not None:
                served.append(self._observation(self.time, played, sensors))
        return served, stopped

    def _read_price(self, group, cycles):
        """The price of a read, the group's price for every cycle that returns a reading, as the engine of the AssetOpsBench
        rows charges its served cycles. A free group costs nothing. Any other read plays first on a copy of the engine, which
        learns the cycles it serves, so that the budget decides on that price, the engine moves only once the price is paid,
        and a cycle for which the source holds no recording is charged nothing, a read that takes no time included."""
        if self.config.read_price(group, 1) == 0:
            return 0
        served, _stopped = self._dry_run()._serve(self.config.sensor_groups[group], cycles)
        return self.config.read_price(group, len(served))

    def _dry_run(self):
        """A copy of the engine on which a timed read plays its cycles to learn its price. It shares what a played cycle never
        changes: the recordings, the blocks, the profile, the configuration, the hazard table, the normal ranges and the
        stores of the source, and its log is a new list of the same entries, since a cycle only appends to it."""
        shared = (self.cycles, self.blocks, self.profile, self.config, self.clock.hazard, self.normal_ranges,
                  *(getattr(self.source, name, None) for name in ("_values", "_outside")))
        memo = {id(part): part for part in shared if part is not None}
        memo[id(self.log)] = list(self.log)
        return copy.deepcopy(self, memo=memo)

    def _observation(self, t, played, sensors):
        index, repeated = played
        self._observations += 1
        return {"n": self._observations, "t": t, "cycle": index, "combination": list(self.clock.combination_at(t)),
                "repeated": repeated, "values": self.source.served_values(index, sensors)}

    def _get_current_time(self, action, think):
        """The clock without moving it, in every clock mode, since no agent tool sets the time (ARE's System app)."""
        result = {"time": self.time, "horizon": self.config.horizon, "seconds_per_cycle": self.config.seconds_per_cycle,
                  "think_seconds_per_cycle": self.config.think_seconds_per_cycle}
        if self.config.clock_mode == real_time.TOKEN_MODE:
            result["seconds_per_token"] = self.config.seconds_per_token
        if self.config.budget is not None:
            result["budget_left"] = self.budget_left()
        if self.config.anchors is not None:
            result["day"] = self.config.date_of(self.time)
        self._entry("current_time", self.time, think, 0)
        return result, None

    def _wait_for_notification(self, action, think):
        """Let the rig run, one cycle at a time, until a notification is pending, the timeout is reached or the episode can
        run no further. A missing or empty timeout waits up to the horizon. In the instant clock mode the wait takes its one
        cycle like every other call."""
        if self.config.environment_mode == "static":
            return self._refuse(action, "static environment: time is frozen; no maintenance wait is required", think)
        timeout = action.get("timeout")
        if timeout is None or timeout == "":
            timeout = self.config.horizon - self.time
        elif _count(timeout, 1) is None:
            return self._refuse(action, "timeout must be a positive integer, or empty for the cycles left before the horizon", think)
        else:
            timeout = _count(timeout, 1)
        t_start = self.time
        wait = WaitForNotificationTimeout(time_created=t_start, timeout=real_time.wait_timeout(self.config, timeout))
        ended_by = wait_for_next_notification(wait, lambda: self.time, self._run_cycle, self._notification_pending)
        if ended_by == "timeout":
            self._step_messages.append(timeout_message(wait, self.time).as_dict(event="wait_timeout", timeout=wait.timeout))
        self._entry("wait", t_start, think, 0, timeout=timeout, ended_by=ended_by)
        return {"time": self.time, "waited": self.time - t_start, "ended_by": ended_by}, None

    def _run_cycle(self):
        """One cycle of a wait, or False when the clock can run no further (the horizon, or a failure under the strict rule).
        The alarms of the cycle are logged with it, so that a wait ends at an alarm of a paid group like at any notification."""
        if self._cut_short():
            return False
        self._advance(1)
        self._record_alarms()
        return True

    def _summarize_changes_since(self, action, think):
        since = _count(action.get("cycle"), 0)
        if since is None or since > self.time:
            return self._refuse(action, "cycle must be an integer between 0 and the current time", think)
        t_start = self.time
        self._advance(self._duration(0, 0))
        alarms = self._alarms(since)
        closed = self.public_work_orders(closed_since=since)
        self._entry("summary", t_start, think, 0, since=since, alarms=alarms, work_orders_closed=closed)
        return {"since": since, "time": self.time, "cycles_elapsed": self.time - since, "alarms": alarms, "work_orders_closed": closed}, None

    def _alarms(self, since):
        """The channel means of the paid sensor groups outside their public normal range in any cycle the rig ran since
        `since`, with the first such cycle; computed from the recording the rig played, never from a label. A free summary
        of a channel nobody paid for would replace its priced reading, so only the channels of a group that a charged read
        named in this episode can alarm."""
        return self.source.alarms(self.clock.played, since, self._paid_sensors())

    def _paid_sensors(self):
        """The channels of every sensor group a read that released a reading named so far in the episode, in the channel
        order of the profile. The rule is that of read_counts in the verifier: a read that served nothing, such as a
        history window past the record, costs nothing and must not free the alarms of its group."""
        return tuple(s for s in self.profile.channels if any(s in self.config.sensor_groups[g] for g in self._paid_groups))

    def _fields_indicating(self, component):
        """The public fields of the channels that indicate the component, in the channel order of the profile."""
        return [self.profile.field_name(s) for s in self.profile.channels if component in self.profile.knowledge["indicates"].get(s, ())]

    def _get_failure_modes(self, action, think):
        component = action.get("component")
        if not isinstance(component, str) or (component and component not in self.config.components):
            return self._refuse(action, "unknown component", think)
        t_start = self.time
        self._advance(self._duration(0, 0))
        wanted = [component] if component else list(self.config.components)
        modes = [{"component": e["component"], "level": e["level"], "code": e["code"], "severity": e["severity"], "action": e["action"],
                  "channels": self._fields_indicating(e["component"]),
                  "evidence_id": f"fmea/{self.profile.knowledge['fmea_source']}/{e['code']}"}
                 for e in self.profile.fault_modes() if e["component"] in wanted]
        self._entry("failure_modes", t_start, think, 0, component=component)
        return {"component": component or None, "association_source": self.profile.knowledge["association_source"], "failure_modes": modes}, None

    def _get_fix_history(self, action, think):
        t_start = self.time
        self._advance(self._duration(0, 0))
        history = copy.deepcopy(self.fix_history)
        self._entry("fix_history", t_start, think, 0, history=history)
        return {"history": history}, None

    def _lookup_fix(self, action, think):
        component = action.get("component")
        if not isinstance(component, str) or component not in self.config.components:
            return self._refuse(action, "unknown component", think)
        t_start = self.time
        self._advance(self._duration(0, 0))
        result = self.fix_manual(component)
        self._entry("fix_lookup", t_start, think, 0, component=component)
        return result, None

    def _submit_diagnosis(self, action, think):
        if self.diagnosed or self.declared:
            return self._refuse(action, WRITE_LOCK_REASON, think)
        diagnosis, evidence = action.get("diagnosis"), action.get("evidence_ids")
        if not isinstance(diagnosis, dict) or set(diagnosis) != set(self.config.components):
            return self._refuse(action, DIAGNOSIS_COMPONENTS_REASON, think)
        # a level is read by its value, so 100, 100.0 and "100" give the level 100, while True, 100.5 and "high" are refused
        levels = {c: _count(diagnosis[c], None) for c in self.config.components}
        if any(levels[c] is None or levels[c] not in self.config.levels[c] for c in self.config.components):
            return self._refuse(action, DIAGNOSIS_LEVEL_REASON, think)
        if not _is_evidence(evidence):
            return self._refuse(action, EVIDENCE_IDS_REASON, think)
        t_start = self.time
        self.diagnosed = True
        stored = levels
        # a diagnosis is scored against the state at the cycle it is submitted in every clock mode, so its
        # entry precedes the events of its own duration, which only the instant mode gives a cycle
        # the charge reads the log before the write, the entries that give the truth the diagnosis is scored against
        charge = misdiagnosis_charge(self.log, stored)
        entry = self._entry("submit_diagnosis", t_start, think, 0, diagnosis=stored, evidence_ids=list(evidence))
        if charge:
            self._charge_misdiagnosis(t_start, charge)
        self._advance(self._duration(0, 0))
        entry["t_end"] = self.time
        return {"diagnosis": stored, "evidence_ids": list(evidence)}, None

    def _declare_insufficient_evidence(self, action, think):
        if self.diagnosed or self.declared:
            return self._refuse(action, WRITE_LOCK_REASON, think)
        test, evidence = action.get("separating_test"), action.get("evidence_ids")
        # a group that is no string, such as a list, is refused like an unknown one and raises nothing
        if (not isinstance(test, dict) or not isinstance(test.get("sensor_group"), str)
                or test["sensor_group"] not in self.config.sensor_groups or _count(test.get("cycles"), 1) is None):
            return self._refuse(action, SEPARATING_TEST_REASON, think)
        if not _is_evidence(evidence):
            return self._refuse(action, EVIDENCE_IDS_REASON, think)
        t_start = self.time
        self._advance(self._duration(0, 0))
        self.declared = True
        stored = {"sensor_group": test["sensor_group"], "cycles": _count(test["cycles"], 1)}
        self._entry("declare_insufficient_evidence", t_start, think, 0, separating_test=stored, evidence_ids=list(evidence))
        return {"separating_test": stored, "evidence_ids": list(evidence)}, None

    def _create_work_order(self, action, think):
        if not (self.diagnosed or self.declared):
            return self._refuse(action, ORDER_BEFORE_WRITE_REASON, think)
        component, prediction, evidence = action.get("component"), action.get("prediction_id"), action.get("evidence_ids")
        if component not in self.config.components:
            return self._refuse(action, "unknown component", think)
        if not isinstance(prediction, str):
            return self._refuse(action, "prediction_id must be a string, empty when there is no prediction", think)
        if not _is_evidence(evidence):
            return self._refuse(action, EVIDENCE_IDS_REASON, think)
        mode = action.get("mode")
        mode = "standard" if mode in (None, "") else mode
        if mode not in ORDER_MODES:
            return self._refuse(action, "mode must be standard or emergency", think)
        if mode == "emergency" and not self.config.emergency_on():
            return self._refuse(action, "the emergency mode is off in this episode", think)
        if self.clock.order_open_on(component):
            close = next(o.close_t for o in self.clock.open_orders() if o.component == component)
            return self._refuse(action, f"a work order on the {component} is open; it closes at cycle {close}", think)
        visit = self._joinable if joins(self._joinable is not None, mode) else None
        visit_charge = 0 if visit is not None else self.config.prices["visit"]
        price = self.config.order_price(component, mode) + visit_charge
        if not self._charge(action, price, think):
            return None, "budget"
        t_start = self.time
        self._advance(self._duration(0, 0))
        order = self.clock.create_order(component, prediction, evidence, mode, visit)
        if mode == "standard":
            self._run_visit = order.visit
        self._entry("work_order_created", t_start, think, price, id=order.id, component=component, mode=mode, visit=order.visit,
                    visit_charge=visit_charge, prediction_id=prediction, evidence_ids=list(evidence), expected_close_t=order.close_t,
                    unnecessary=order.unnecessary)
        if not self.config.has_clock() and not self.config.repair_card_on():
            self._record(self.clock.settle_due_orders())  # no visit to wait for
        return {"work_order": order.public()}, None

    def _finish(self, action, think):
        summary = action.get("summary")
        if not isinstance(summary, str):
            return self._refuse(action, "summary must be a string", think)
        t_start = self.time
        self._advance(self._duration(0, 0))
        self._entry("finish", t_start, think, 0, summary=summary)
        self._settle_open_orders()
        self._end("finish")
        return {"summary": summary}, None


_TOOL_OF_EVENT = {"profile": "get_asset_profile", "read": "read_sensors", "current_time": "get_current_time", "wait": "wait_for_notification",
                  "summary": "summarize_changes_since",
                  "failure_modes": "get_failure_modes", "fix_history": "get_fix_history", "fix_lookup": "lookup_fix",
                  "submit_diagnosis": "submit_diagnosis",
                  "declare_insufficient_evidence": "declare_insufficient_evidence", "work_order_created": "create_work_order", "finish": "finish"}


# a decimal integer as a string writes it, with an optional sign and an optional fraction of zeros only
_INTEGER_TEXT = re.compile(r"[+-]?[0-9]+(\.0+)?")


def _as_integer(value):
    """The integer that an argument writes, or None: an int that is no bool, a float without a fraction, or a string that,
    without its surrounding blanks, is a decimal integer with an optional sign and an optional fraction of zeros only, such
    as "100", "100.0" or "-5". A bool, null, a list, "two", "[2]", 2.5, "1e2" and an empty or blank string write none."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str) and _INTEGER_TEXT.fullmatch(value.strip()):
        return int(value.strip().split(".")[0])
    return None


def _count(value, minimum):
    """The integer of a tool argument at or above the minimum, minimum None checking the value alone, or None. Every handler
    reads its integer arguments through it, so that a tool that gains one follows the same rule."""
    number = _as_integer(value)
    return number if number is not None and (minimum is None or number >= minimum) else None


def _is_evidence(value):
    return isinstance(value, list) and all(isinstance(v, str) for v in value)
