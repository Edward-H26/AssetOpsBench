import copy
import re
from dataclasses import dataclass, field
from CausalDiagnosis.utils.public_rules.repair_completion import Rules, completion, schedule, verifies, visit_moments, wait_calls
from CausalDiagnosis.utils.public_rules.visits import joins, keeps_run
from CausalDiagnosis.agents.utils import core_texts
from . import asset_rules, text_formatting, troubleshooting_plan_fields


# the clock modes in which the agent's thinking plays cycles before an action, so that the record grows during any call (the
# environment's real_time.THINKING_MODES, which a test ties to this tuple)
THINKING_CLOCKS = ("generation_time", "generated_tokens")
DECLARED_CLOCK = "declared"  # the clock mode in which time moves only inside tool calls, by the declared duration of each
SUMMARY_SECTION = "## Your summary of earlier turns"  # the board's section of step 3 of the compaction
NO_SUMMARY = "(no summary available)"  # an empty summary reply, or one the engine cut at the context (after Codex)
# the place of each kind on the board, which breaks ties of the planning turn in the order a summary replaces entries
SUMMARY_PLACE = {"notifications": 0, "alarm": 1, "closed": 1, "surprises": 2, "notes": 3}
RECALL_SECTION = "## Recalled readings"  # the rows a recall step brought back
# the words of the kind recorded levels in the slots of the shared texts (agents/utils/core_texts.py), which the prompts, the
# board and the reasons of the check fill from this one map until an asset declares the words of its kind
KIND_WORDS = {"time_unit": "cycle", "series": "sensor group", "target": "component", "repair": "work order"}
# the line of a stall: planning turns in a row that added no evidence and changed no state, two being the least that is a row
NO_PROGRESS = "No progress for {count} turns in a row: no turn added evidence or changed a state, and a restated plan is no progress."
NO_PROGRESS_TURNS = 2
# the lines of the status on the work orders where a clock runs (_order_lines): the open orders on schedule with their closes and how
# time passes on the row, the read that verifies a repair whose orders have closed, and every repair verified
ORDERS_ON_SCHEDULE = "Open work orders, on schedule and not failed: {orders}; the latest answer was at cycle {time}. {clock}."
ORDER_CLOSES = "{id} on the {component} closes at cycle {close}"
TIME_ON_ROW = "On this row (clock mode {mode}) {moves}"
TIME_MOVES = {"declared": "time passes only inside calls that take time", "generation_time": "time passes inside calls and with your response time",
              "generated_tokens": "time passes inside calls and with the tokens you generate",
              "instant": "every call and every turn without a call moves the clock one cycle"}
CLOSE_MEANS = {"wait": "a wait step lets the {noun} run to the close",
               "read": "a read of {group} whose window reaches past the record plays the cycles to the close"}
REPAIR_UNVERIFIED = "{orders} closed by cycle {close} and the repair is not verified yet: {next}, and that reading verifies it."
READ_NOW = "read {group} now"
WAIT_THEN_READ = "wait to cycle {first}, then read {group}"
FORWARD_READ = "read {group} for {cycles}, whose window plays past the close"
ALL_VERIFIED = "Every work order has closed and every repair is verified, the latest consistent: no wait or read is pending, and a finish step ends the episode."


def call_text(kind, request):
    """A call as the status of the board and the reasons of a step name it: the kind, then its keyword fields as the board prints
    the arguments of a step, where it has any."""
    return f"{kind} {request}" if request else kind


def commitment_text(kind, value):
    """A commitment as the status of the board states it, (kind, value): the levels that a diagnosis submitted, or the sensor group
    and the cycles of the separating test of a declaration."""
    if kind == "diagnose":
        return "diagnosis submitted: " + text_formatting.combination_text(value)
    group, cycles = value
    return f"insufficient evidence declared, separating test {group} for {text_formatting.n_cycles(cycles)}"


def value_text(row):
    """A value of an evidence row as the board prints it: the value, ending in low or high where it lies outside its normal
    range, as the CausalDiagnosis MCP writes a value of its evidence table."""
    value = text_formatting.fmt(row["value"])
    if row["inside_band"] is False:
        return f"{value} {'low' if row['value'] < row['band_low'] else 'high'}"
    return value


def names(text, identifier):
    """Whether the text holds the identifier whole, so that wo/1 is not found inside wo/12."""
    return re.search(rf"(?<![\w/.-]){re.escape(identifier)}(?![\w/-]|\.\w)", text or "") is not None


@dataclass
class Board:
    show_branches: bool = True
    single_hypothesis: bool = False
    profile: dict = None
    failure_modes: list = field(default_factory=list)
    status: dict = field(default_factory=dict)
    evidence: list = field(default_factory=list)
    observations: int = 0
    ranking: dict = None
    ranking_basis: tuple = None
    alarms: list = field(default_factory=list)
    closed_orders: list = field(default_factory=list)
    orders: dict = field(default_factory=dict)
    predictions: dict = field(default_factory=dict)
    verifications: dict = field(default_factory=dict)
    diagnosis: dict = None
    declaration: dict = None
    differential: list = field(default_factory=list)
    plan: dict = None
    surprises: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    end_reason: str = None
    fix_history: list = field(default_factory=list)
    summaries: list = field(default_factory=list)
    absorbed_ids: set = field(default_factory=set)
    refuted: list = field(default_factory=list)
    refuted_last: list = None
    notifications: list = field(default_factory=list)
    compactions: int = 0
    compaction_log: list = field(default_factory=list)
    summary_text: str = None  # the model's summary of earlier turns (step 3 of the compaction), shown after the failure modes
    summary_turn: int = None  # the planning turn of the latest summary
    turn_marks: list = field(default_factory=list)  # (turn, lengths of the replaceable lists at its start), the key that orders the entries a summary replaces
    timeline_hidden: list = field(default_factory=list)  # [kind, identifier] of each alarm and closed order a summary replaced
    pinned: list = field(default_factory=list)  # [kind, text] of each entry kept because a summary names it, out of the replaceable set from then on
    collapsed_rows: list = field(default_factory=list)  # the full rows of every collapsed observation, which a recall step brings back
    recalled: list = field(default_factory=list)  # the rows a recall step brought back, shown on the board of the next planning turn
    recall_turn: int = None  # the planning turn whose board shows the recalled rows
    recall_offered: bool = True  # the recall step is offered; False refuses it in the plan check and drops it from the prompts
    reads: list = field(default_factory=list)  # every reading a plan named, the elimination expected against the realized one
    server: bool = True  # the diagnostics server is registered; without it the board shows the model's own candidates
    question: str = ""  # the task message of the episode, which every prompt prints above the board (prompts.py)
    opening_done: list = field(default_factory=list)  # the kinds of the opening read so far
    required_opening: list = field(default_factory=list)  # the opening kinds the surface needs before a reading (tools/, opening_kinds)
    clock_runs: bool = True  # a cycle runs after a close on this row, so a reading can verify a repair (tools/, can_observe_after)
    model_candidates: list = field(default_factory=list)  # the candidates of the model's latest plan without the server
    plan_basis: int = 0  # the observations the model had seen at its latest plan, which compaction counts as absorbed without a ranking
    latest_cycles: dict = field(default_factory=dict)  # per sensor group, its latest cycle read with the channel means outside and within their normal range
    read_advice: str = "gain"  # the arm of Ag7 whose read the advice names (plan.advice)
    backtracking: bool = True  # Ag8: a refuted repair set is recorded, marked and the cheapest other plan advised
    trimmed: bool = False
    evidence_form: str = "obs/<n>/<field>"  # the citation form of a value on the surface the agent runs on (tools/)
    parallel: bool = True  # the schedule of the candidate plans and the batch of consecutive read steps (candidate_plan_schedule.py)
    batches: list = field(default_factory=list)  # per run of consecutive read steps: the plan, its first step, the reads made, the rankings made
    candidates_seen: int = 0  # the observations at which the model listed its latest candidates, after which a reading serves their tests
    noun: str = "asset"  # the noun of the asset in the texts of the board, which the surface gets from the asset's text (tools/)
    reads_take_time: bool = True  # a read plays the cycles it reads on this surface, so it takes their declared duration (tools/)
    summary_call: bool = True  # the change summary after a wait is a tool call on this surface, and answered without one elsewhere (tools/)
    unserved: frozenset = frozenset()  # the step kinds whose tool the row does not serve (tools/, Surface.unserved_kinds)
    state_writes: dict = field(default_factory=dict)  # step kind -> (state, value) of a write of a public state, the key sets of the general agent's table; no surface declares one
    states: dict = field(default_factory=dict)  # state -> the value the board knows it to have, which a state write is held against (plan.standing)
    manual_channels: dict = field(default_factory=dict)  # component -> the indicating channels an answer of the repair manual named
    run: object = None  # the visit of the open run of orders: the last call of the environment was an accepted standard order of it
    answered: list = field(default_factory=list)  # [kind, keyword fields] of every answered request of the opening kinds (note_answered)
    calls: list = field(default_factory=list)  # [kind, keyword fields, count] of every call a step of a plan bound to a tool (note_call)
    retired_notes: int = 0  # the notes that left since the state they spoke of passed (retire_notes)
    reply_schema: dict = None  # the digests of the reply schemas, the requests that carried one and whether the backend carried the field
    note_ties: dict = field(default_factory=dict)  # note text -> the states whose passing takes it off (add_note), beside notes
    note_repeats: dict = field(default_factory=dict)  # note text of a plan -> the count of the earlier notes with its reasons that it replaced
    progress: str = None  # what the episode had done at the start of the latest planning turn (_progress)
    turns_without_progress: int = 0  # the planning turns in a row before the current one that added no evidence and changed no state

    # ----- the clock and the counters -----
    @property
    def time(self):
        return self.status.get("time", 0)

    @property
    def statistic(self):
        """The plain statistic of the served values, the one the profile lists ahead of the others, which the normal ranges judge
        and the texts of the board name in the plural with an s appended; value before the profile names one."""
        served = (self.profile or {}).get("statistics") or ()
        return served[0] if served else "value"

    @property
    def budget_left(self):
        """The envelope's budget left, else the manifest's budget, unlimited when neither names one (the free base row)."""
        left = self.status.get("budget_left")
        return asset_rules.limit_of(left if left is not None else (self.profile or {}).get("budget"))

    @property
    def steps_left(self):
        left = self.status.get("steps_left")
        return asset_rules.limit_of(left if left is not None else (self.profile or {}).get("max_steps"))

    @property
    def horizon(self):
        return asset_rules.limit_of((self.profile or {}).get("horizon"))

    @property
    def budget_limited(self):
        return self.budget_left != asset_rules.UNLIMITED

    @property
    def cycles_left(self):
        return self.horizon - self.time

    @property
    def over(self):
        return bool(self.status.get("episode_over"))

    @property
    def committed(self):
        return self.diagnosis is not None or self.declaration is not None

    # ----- the rules of the asset -----
    def read_price(self, group, cycles=1):
        """The price of a read, the price per cycle times the cycles it pays for anew, since a re-read of a paid cycle is free.
        The cycles a read covers end at the envelope's recorded_until on a row that records ahead of its
        reads, and the board's evidence says which of them it has paid for; the hydraulic rig records a cycle only by
        playing it and never sets the field, so every cycle of a read there is new."""
        return self.new_cycles(group, cycles) * asset_rules.read_prices(self.profile).get(group, 0)

    def new_cycles(self, group, cycles):
        """The cycles a read of the group pays for: on a row that records ahead of its reads, its unpaid cycles
        (unpaid_cycles), at most cycles of them; every cycle elsewhere. Where the record grows during a call (record_grows) and no
        recorded cycle is unpaid, the read asks for the open window, which serves and charges the cycles that thinking recorded
        meanwhile, at most cycles of them: the board counts that most."""
        unpaid = self.unpaid_cycles(group, cycles)
        if unpaid is None or (not unpaid and (self.record_grows or self.forward_read)):
            return cycles
        return min(cycles, len(unpaid))

    @property
    def forward_read(self):
        """The clock one cycle of a forward read takes: on a row that serves no wait, whose clock runs and whose reads serve recorded
        cycles, a history window past the present plays its cycles at the declared read duration; 0 elsewhere."""
        if "wait" not in self.unserved or self.reads_take_time or not self.clock_runs:
            return 0
        return asset_rules.durations_of(self.profile)["read_per_cycle"]

    def reads_forward(self, group):
        """Whether a read of the group takes the window past the present, which plays its cycles: on a row with a forward read
        (forward_read), a read of a group that verifies the repair while an order is open or a repaired close lies past the record."""
        repaired = self.repaired()
        if not self.forward_read or not repaired or not self.verifies(group, repaired):
            return False
        until = self.status.get("recorded_until", -1)
        return bool(self.open_orders()) or any(closed_t > until for _key, closed_t in self.pending_verifications())

    @property
    def record_grows(self):
        """Whether the record grows during any call: a cycle runs after a close on this row and its clock charges thinking."""
        return self.clock_runs and (self.profile or {}).get("clock_mode") in THINKING_CLOCKS

    @property
    def record_holds_between_calls(self):
        """Whether the record stays as the latest answer shows it until a call takes time: no cycle runs on this row, or the clock
        runs by declaration, so that a cycle is recorded only inside a call with a declared duration, whose answer shows the new end
        of the record. Elsewhere thinking records cycles between calls, or every call plays one (the instant mode)."""
        return not self.clock_runs or (self.profile or {}).get("clock_mode") == DECLARED_CLOCK

    def unpaid_cycles(self, group, cycles=1):
        """On a row that records ahead of its reads, the recorded cycles a read of the group would pay for, the newest first:
        from recorded_from to recorded_until (the newest cycles alone when the start of the record is unknown), less the
        cycles the board has paid for and the cycles that hold no record of the group. None on a row that records a cycle only
        by playing it, where every cycle of a read is new."""
        recorded_until = self.status.get("recorded_until")
        if recorded_until is None:
            return None
        recorded_from = self.status.get("recorded_from", recorded_until - cycles + 1)
        paid = self.paid_cycles(group) | self.unrecorded(group)
        return [c for c in range(recorded_until, recorded_from - 1, -1) if c not in paid]

    def unrecorded(self, group):
        """The cycles that a read of the group asked for and that came back without a row, which hold no record of the group,
        as the envelope of the surface names them (tools/)."""
        return set((self.status.get("unrecorded") or {}).get(group) or ())

    def in_group(self, field_name, group):
        """Whether a served field is a value of a channel of the group: the channel itself or one of its summaries, the channel
        name followed by an underscore and the summary."""
        return any(field_name == name or field_name.startswith(f"{name}_") for name in (self.profile.get("sensor_groups") or {}).get(group) or ())

    def covers(self, group, other):
        """Whether a read of the group serves every field of the other group, whose fields are a subset of its own; a group
        covers itself."""
        groups = self.profile.get("sensor_groups") or {}
        return set(groups.get(other) or []) <= set(groups.get(group) or [])

    def paid_cycles(self, group):
        """The cycles a read of the group would not pay for again: those the board holds readings of, as rows or as
        collapsed summaries, by the group or by a group whose fields cover its fields."""
        covering = {g for g in self.profile.get("sensor_groups") or {} if self.covers(g, group)} | {group}
        cycles = {row["t"] for row in self.evidence if row.get("sensor_group") in covering}
        for summary in self.summaries:
            if summary.get("sensor_group") in covering:
                cycles.update(summary["cycles"])
        return cycles

    def read_duration(self, cycles=1):
        """The clock a read of cycles takes: the declared duration of every cycle read on a surface that plays the cycles it reads,
        as the hydraulic rig's action server does, and none on a surface that serves cycles already recorded, as the
        AssetOpsBench servers do."""
        return cycles * asset_rules.durations_of(self.profile)["read_per_cycle"] if self.reads_take_time else 0

    def order_price(self, component):
        return asset_rules.order_prices(self.profile)[component]

    def order_duration(self, component):
        durations = asset_rules.durations_of(self.profile)
        return durations["lead_time"][component] + durations["repair_downtime"][component]

    def wait_cycles(self, cycles):
        """The cycles a wait step of cycles lets run: the cycles times the declared duration of one waited cycle."""
        return cycles * asset_rules.durations_of(self.profile)["advance_per_cycle"]

    @property
    def visit_charge(self):
        """The charge of one technician visit, paid once by the orders created one after another; zero for a profile without it."""
        return (self.profile.get("prices") or {}).get("visit", 0) or 0

    def first_cycle_after(self, close):
        """The first cycle at which a read serves a cycle at or after a close: the close itself, or the cycle after it on a row
        that records ahead of its reads, whose read serves the newest recorded cycles, those before the clock (tools/)."""
        return close + (1 if "recorded_until" in self.status else 0)

    def plan_price(self, components):
        """The price of a repair plan whose orders follow one another: every order's part plus repair charge, plus one visit charge."""
        components = list(components)
        return sum(self.order_price(c) for c in components) + (self.visit_charge if components else 0)

    def verifies(self, group, components):
        """Whether a reading of the group can verify a repair of the components (repair_completion.verifies): its channels hold an
        indicating channel of every repaired component, as the failure modes or the repair manual name them; for a component whose
        indicating channels the board does not know, or which no group holds, only a group of every channel verifies."""
        groups = self.profile.get("sensor_groups") or {}
        return verifies(groups.get(group) or (), components, self.indicating_channels(), groups)

    def indicating_channels(self):
        """{component: the channels the board knows to indicate it}: those the failure modes name and those an answer of the
        repair manual names (add_manual)."""
        indicating = {c: set(channels) for c, channels in self.manual_channels.items()}
        for mode in self.failure_modes:
            indicating.setdefault(mode.get("component"), set()).update(mode.get("channels") or [])
        return indicating

    def add_manual(self, result):
        """The indicating channels an answer of the repair manual (lookup_fix) names for its component, when it carries them."""
        if result.get("component") and result.get("indicating_channels"):
            self.manual_channels.setdefault(result["component"], set()).update(result["indicating_channels"])

    def verification_group(self, components):
        """The cheapest sensor group that verifies a repair of the components (verifies), so that the one reading after the last
        close can refute the plan; the cheapest group when none does, which no row of a registered asset reaches (the group of
        every channel verifies)."""
        prices = asset_rules.read_prices(self.profile)
        return next((g for g in sorted(prices, key=lambda g: (prices[g], g)) if self.verifies(g, components)), self.cheapest_group())

    def repaired(self, new=()):
        """The components whose repair the next verifying read judges: those of the open orders, of the closed orders whose
        verification is pending, and the new ones."""
        keys = {key for key, _closed_t in self.pending_verifications()}
        return sorted({o["component"] for o in self.orders.values() if o["status"] != "closed" or o["prediction_id"] in keys} | set(new))

    def rules(self):
        """The public values of the rule of finishing a repair (servers/public_rules/repair_completion.py) on this board."""
        durations = asset_rules.durations_of(self.profile)
        return Rules(lead_time=durations["lead_time"], repair_downtime=durations["repair_downtime"],
                     priority=(self.profile.get("processes") or {}).get("priority") or {}, order_price=asset_rules.order_prices(self.profile),
                     visit_charge=self.visit_charge, read_duration=self.read_duration(1), first_offset=self.first_cycle_after(0),
                     summary_call=self.summary_call, clock_runs=self.clock_runs, wait_served="wait" not in self.unserved,
                     forward_read=self.forward_read)

    def begun_close(self, order):
        """The close of an open order whose repair has begun, the clock being past the cycle its technician started it, else None."""
        close = order.get("expected_close_t")
        if close is None or order["status"] == "closed":
            return None
        return close if order["status"] == "in_repair" or close - asset_rules.durations_of(self.profile)["repair_downtime"][order["component"]] < self.time else None

    def open_visits(self):
        """{visit: [(component, created cycle, close of a repair begun or None)]} of the open orders, in creation order."""
        visits = {}
        for order in self.open_orders():
            visits.setdefault(order.get("visit"), []).append((order["component"], order["created_t"], self.begun_close(order)))
        return visits

    def note_step(self, kind, data):
        """A call of the environment: an accepted standard order keeps the run of orders open, joining the visit of the order before
        it or opening the next one, and every other call ends it, a refused one included (servers/public_rules/visits.py). The
        calls of the diagnostics server, the thinking between calls and a no_call step are no calls of the environment."""
        if keeps_run(kind):
            return
        order = (data.get("result") or {}).get("work_order") if kind == "order" and "refused" not in data else None
        if order is not None and joins(True, order.get("mode") or "standard"):
            if self.run is None:
                self.run = 1 + max((o.get("visit") or 0 for o in self.orders.values()), default=0)
        else:
            self.run = None

    def completion(self, components=(), time=None, visits=None, pending=None, repaired=None, commit=None, run=None, after_step=False):
        """What finishing the repair needs (repair_completion.completion) from this board, or from a state of a plan walk that
        gives its own clock, open visits, open run, pending closes and repaired components: orders on the components, the
        diagnosis or declaration while none is made, the waits to the first cycle that verifies, the read of the verification
        group of the repaired components and finish. On the board's own state the orders join its open run of orders unless a
        call comes before them: the diagnosis or declaration still to be made, or another step (after_step)."""
        time = self.time if time is None else time
        commit = not self.committed if commit is None else commit
        if visits is None:
            visits, run = self.open_visits(), (None if after_step or commit else self.run)
        pending = [closed_t for _key, closed_t in self.pending_verifications()] if pending is None else pending
        repaired = self.repaired(components) if repaired is None else repaired
        group = self.verification_group(repaired)
        return completion(self.rules(), time, new=components, visits=visits, pending=pending, read_price=self.read_price(group),
                          commit=commit, run=run, verifiable=self.verifies(group, repaired))

    def verifying_tail(self):
        """What finishing the repair still needs from this board (completion), the waits, the verifying read and finish, when that
        tail holds a verifying read and fits the budget, the steps and the horizon left; None otherwise. The finish guard of the
        controller and the status line of a closed repair read this one tail."""
        tail = self.completion(commit=False)
        fits = tail.price <= self.budget_left and tail.calls <= self.steps_left and self.time + tail.cycles < self.horizon
        return tail if tail.reads and fits else None

    def wait_calls(self, start, end, visits):
        """The tool calls of a wait step from start to end over the visits (open_visits) (repair_completion)."""
        rules, moments = self.rules(), set()
        for members in visits.values():
            moments |= visit_moments(schedule(members, rules.lead_time, rules.repair_downtime, rules.priority))
        return wait_calls(start, end, moments, self.summary_call)

    def limits_line(self):
        """The clock, the budget and the steps of the board in the words of its status, each limit the manifest leaves out named
        as absent, and a horizon on a row where no clock runs said so, since the tool texts of a surface leave that to the board;
        the board and the closing summary print the same line."""
        clock = (f"Time {self.time} of {self.horizon} cycles ({self.cycles_left} left{'' if self.clock_runs else ', no clock runs'})."
                 if self.horizon != asset_rules.UNLIMITED else f"Time {self.time}, no horizon (no clock runs).")
        budget = (f"Budget left {text_formatting.fmt(self.budget_left)} of {text_formatting.fmt((self.profile or {}).get('budget'))}." if self.budget_limited
                  else "No budget limit (every tool is free).")
        steps = f"Steps left {self.steps_left} (refused calls count)." if self.steps_left != asset_rules.UNLIMITED else "No step cap."
        return f"{clock} {budget} {steps}"

    def cheapest_group(self):
        prices = asset_rules.read_prices(self.profile)
        return min(prices, key=lambda g: (prices[g], g))

    def healthy_set(self):
        return dict(self.profile["healthy"])

    def faulty_set(self, combination):
        return [c for c, v in combination.items() if v != self.profile["healthy"][c]]

    # ----- updates from tool answers -----
    def update_status(self, envelope):
        for key in ("time", "budget_left", "steps_left", "open_work_orders", "episode_over", "recorded_from", "recorded_until", "unrecorded"):
            if key in envelope:
                self.status[key] = copy.deepcopy(envelope[key])
        if "open_work_orders" not in self.status:
            return  # a surface that lists no open orders closes an order by its notification, or inside its call where no clock runs
        open_orders = {o["id"]: o for o in self.status["open_work_orders"]}
        for order_id, order in self.orders.items():
            if order_id in open_orders:
                # a later order of the same visit with a smaller priority number moves the close (CausalDiagnosis/docs/environment.md)
                order["expected_close_t"] = open_orders[order_id]["expected_close_t"]
            elif order["status"] != "closed":
                order["status"] = "closed"

    def set_profile(self, result):
        self.profile = copy.deepcopy(result)
        self.retire_notes("facts")

    def set_failure_modes(self, result):
        self.failure_modes = copy.deepcopy(result.get("failure_modes", []))

    def add_failure_modes(self, modes):
        """Failure modes read by a step of the opening, each mode once beside those the board holds."""
        known = {(m.get("component"), m.get("code")) for m in self.failure_modes}
        for mode in modes:
            if (mode.get("component"), mode.get("code")) not in known:
                self.failure_modes.append(copy.deepcopy(mode))
                known.add((mode.get("component"), mode.get("code")))

    def set_fix_history(self, result):
        self.fix_history = copy.deepcopy(result.get("history", []) or [])

    def normal_ranges(self):
        """The public normal range of every channel mean as the profile serves it, none before the profile is read."""
        return self.profile.get("normal_ranges", {}) if self.profile else {}

    def add_reading(self, result):
        """The served values as evidence rows with the public normal range of every channel mean; the std summaries have no normal range. A
        row cites its values by its observation number, or by the label the surface gives it (the timestamp of a
        telemetry row on the AssetOpsBench rows). Returns the cycles served, each [observation, cycle, channel values]."""
        bands = self.normal_ranges()
        cycles = []
        for served in result.get("served", []):
            label = served.get("label", served["n"])
            for field_name, value in sorted(served["values"].items()):
                band = bands.get(field_name)
                self.evidence.append({
                    "evidence_id": f"obs/{label}/{field_name}", "n": served["n"], "t": served["t"], "sensor_group": result.get("sensor_group"),
                    "field": field_name, "value": value, "band_low": None if band is None else band["low"],
                    "band_high": None if band is None else band["high"],
                    "inside_band": None if band is None else bool(band["low"] <= value <= band["high"])})
            # every channel of the group the board holds for the cycle, also those an earlier read of a smaller group served
            held = {r["field"]: r["value"] for r in self.evidence if r["t"] == served["t"] and self.in_group(r["field"], result.get("sensor_group"))}
            outside = asset_rules.channels_outside(bands, held)
            banded = sorted(name for name in held if name in bands)
            self.observations = max(self.observations, served["n"])
            self.latest_cycles[result.get("sensor_group")] = {"n": served["n"], "t": served["t"], "outside": outside, "banded": len(banded),
                                                              "inside": [name for name in banded if name not in {o for o, _side in outside}]}
            cycles.append([served["n"], served["t"], dict(served["values"])])
        return cycles

    def set_ranking(self, result):
        """The ranking, with the troubleshooting plans filtered by the ablation switches; the prediction identifiers of the
        shown branches become known predictions, so the plan check can accept them for a work order."""
        ranking = copy.deepcopy(result)
        branches = ranking.pop("branches", None)
        if self.show_branches and branches:
            if self.single_hypothesis:
                branches = branches[:1]
            ranking["branches"] = branches
            for branch in branches:
                recovery = (branch.get("predicted_outcome") or {}).get("recovery_if_acted")
                components = list(branch.get("faulty_set") or [s.get("component") for s in troubleshooting_plan_fields.repair_steps(branch)])
                # the identifiers the orders of this branch cite become known predictions; with a plan identifier the
                # steps' own predictions stay uncited and unknown, so the plan check steers the model to the plan's
                for component in components:
                    prediction_id = troubleshooting_plan_fields.branch_prediction_id(branch, component)
                    if prediction_id:
                        self.predictions.setdefault(prediction_id, {
                            "action": "replace:" + ("+".join(components) if prediction_id == troubleshooting_plan_fields.plan_prediction_id(branch) else component),
                            "component": None if prediction_id == troubleshooting_plan_fields.plan_prediction_id(branch) else component, "recovery": recovery,
                            "source": f"branch {branch.get('rank')}"})
        self.ranking = ranking
        # the belief moves with the clock between readings (version 7), so the basis carries the cycle as well
        self.ranking_basis = (self.observations, len(self.closed_orders), self.time)
        self._release_refuted()
        self.retire_notes("ranking")

    def ranking_is_current(self):
        return self.ranking is not None and self.ranking_basis == (self.observations, len(self.closed_orders), self.time)

    def record_read(self, entry):
        """A reading a plan named, with what the last ranking before it expected of its sensor group against what it realized,
        whether the advice named the group, and the channel means outside their normal ranges per cycle."""
        self.reads.append(copy.deepcopy(entry))

    def record_batch(self, plan, first_step, count, rankings):
        """A run of consecutive read steps of one plan: the last count entries of the read log carry the batch, [plan, first
        step, count], and the effective hypotheses after the ranking that followed the whole batch, and the batches of the
        board gain the run with the reads made and the rankings made."""
        entries = self.reads[len(self.reads) - count:] if count else []
        for entry in entries:
            entry["batch"] = [plan, first_step, count]
            entry["realized_after"] = self.effective_hypotheses() if self.server else None
        self.batches.append({"plan": plan, "first_step": first_step, "reads": [[e["sensor_group"], e["cycles"]] for e in entries],
                             "rankings": rankings})

    def record_refuted(self, components, reason):
        """A repair plan the verification refuted: its component set is kept out of the next plan until a ranking after a
        further reading names that set first again."""
        entry = {"components": sorted(components), "reason": reason, "cycle": self.time, "observations": self.observations}
        self.refuted.append(entry)
        self.refuted_last = entry["components"]

    def _release_refuted(self):
        if not self.refuted_last or not self.refuted:
            return
        top = self.top()
        if top is None or self.observations <= self.refuted[-1]["observations"]:
            return
        if sorted(self.faulty_set(top.get("current_combination") or top["combination"])) == self.refuted_last:
            self.refuted_last = None

    # ----- the version 7 fields of the ranking -----
    def effective_hypotheses(self):
        return (self.ranking or {}).get("effective_hypotheses")

    def elimination(self):
        return list((self.ranking or {}).get("elimination") or [])

    def elimination_of(self, group):
        return next((e for e in self.elimination() if e.get("sensor_group") == group), None)

    def earlier_rule(self):
        """The block e10_rule of a long form of the ranking, the earlier rule that a run under an arm asks for, or None."""
        return (self.ranking or {}).get("e10_rule")

    def server_advice(self):
        """The advice of the server, the step its rule advises before a diagnosis, or None."""
        return (self.ranking or {}).get("advice")

    def read_first(self):
        """The flag the lines of the rule read: read_first of the earlier rule where the ranking carries it, else the flag of the
        ranking."""
        earlier = self.earlier_rule()
        return bool(earlier.get("read_first")) if earlier is not None else bool((self.ranking or {}).get("read_first"))

    def read_batch(self):
        """The ordered reads whose choice no outcome can change, as (sensor group, cycles), those of the earlier rule where the
        ranking carries it; the server lists them as objects with a sensor_group or as bare group names."""
        batch = []
        for item in (self.earlier_rule() or self.ranking or {}).get("read_batch") or []:
            if isinstance(item, str):
                batch.append((item, 1))
            elif isinstance(item, dict) and item.get("sensor_group"):
                batch.append((item["sensor_group"], int(item.get("cycles") or 1)))
        return batch

    def add_estimate(self, result):
        self.predictions[result["prediction_id"]] = {"action": result["action"], "component": result.get("component"),
                                                    "recovery": result.get("recovery"), "source": "estimate"}

    def add_changes(self, result):
        """Alarms and closed orders of a change summary, merged into the timeline; a closed order of the agent is marked
        closed with its closing cycle."""
        known = {a["channel"] for a in self.alarms}
        for alarm in result.get("alarms", []):
            if alarm["channel"] not in known:
                self.alarms.append(dict(alarm))
                known.add(alarm["channel"])
        closed_ids = {o["id"] for o in self.closed_orders}
        for order in result.get("work_orders_closed", []):
            if order["id"] not in closed_ids:
                self.closed_orders.append(dict(order))
                closed_ids.add(order["id"])
            if order["id"] in self.orders:
                self.orders[order["id"]].update(status="closed", closed_t=order.get("closed_t"))

    def add_notifications(self, items):
        """The public events a tool response carries since the previous call: work orders executed and
        closed, failures and alarms of the paid groups. They are evidence rows of their own kind,
        with no reading, and the closes and alarms among them update the orders and the timeline as a change summary does."""
        for item in items or []:
            if not isinstance(item, dict):
                continue
            kind = item.get("event") or item.get("kind") or "event"
            t = item.get("t_end", item.get("t", item.get("first_t")))
            if kind == "work_order_closed":
                self.add_changes({"alarms": [], "work_orders_closed": [{"id": item.get("id"), "component": item.get("component"),
                                                                        "source": item.get("source", "agent"), "closed_t": t}]})
            elif kind == "alarm" and item.get("channel"):
                self.add_changes({"alarms": [{"channel": item["channel"], "first_t": item.get("first_t", t)}], "work_orders_closed": []})
            detail = item.get("message") or ", ".join(f"{k} {text_formatting.fmt(v)}" for k, v in item.items()
                                                       if k not in ("event", "kind", "t", "t_end", "t_start", "class", "step", "cost", "message_type"))
            self.notifications.append({"t": t, "event": kind, "detail": detail})

    def add_order(self, result, prediction_id):
        """An order of the agent, in the visit of the open run of orders (note_step), or in a visit of its own. On a surface whose
        status lists no open orders the board estimates the close of every open order of the visit again, since a later order
        of a visit moves the closes of the earlier ones (visit_closes)."""
        order = result["work_order"]
        visit = self.run if self.run is not None else 1 + max((o.get("visit") or 0 for o in self.orders.values()), default=0)
        self.orders[order["id"]] = {"id": order["id"], "component": order["component"], "created_t": order["created_t"],
                                    "expected_close_t": order.get("expected_close_t"), "status": order["status"], "prediction_id": prediction_id,
                                    "closed_t": order.get("closed_t"), "visit": visit}
        if "open_work_orders" not in self.status:
            members = [o for o in self.orders.values() if o["visit"] == visit and o["status"] != "closed"]
            for component, close in asset_rules.visit_closes(self.profile, [(o["component"], o["created_t"], self.begun_close(o)) for o in members]).items():
                next(o for o in members if o["component"] == component)["expected_close_t"] = close

    def add_verification(self, result):
        self.verifications[result["prediction_id"]] = {
            "status": result.get("status"), "consistent": result.get("consistent"), "recovery_predicted": result.get("recovery_predicted"),
            "recovery_posterior": result.get("recovery_posterior"), "recovery_status": result.get("recovery_status"),
            "predicted_combination": result.get("predicted_combination"), "predicted_mass": result.get("predicted_mass"),
            "message": result.get("message")}

    # ----- lookups -----
    def candidates(self, own=None):
        """The candidates a diagnosis names by rank: the ranking's, then the model's own of its latest plan that listed them, or own
        in their place, in the model's order with the ranks continued. Without the server the model's own are the only ones, and
        with it the ranking stays advice."""
        ranked = self.ranked()
        return ranked + [dict(c, rank=len(ranked) + c["rank"]) for c in (self.model_candidates if own is None else own)]

    def ranked(self):
        """The candidates of the ranking, none without the server or before a ranking."""
        return list((self.ranking or {}).get("candidates") or []) if self.server else []

    def top_set(self):
        """(faulty set, mass) of the set that holds the most mass in the latest ranking, the mass summed by the server over every
        combination of the set (the reply's set_masses)."""
        best = max(self.ranking["set_masses"], key=lambda r: r["mass"])
        return sorted(best["faulty_set"]), float(best["mass"])

    def ranked_top(self):
        """The top candidate of the ranking, the one an expectation on the top or its mass reads, or None."""
        ranked = self.ranked()
        return ranked[0] if ranked else None

    def as_candidates(self, raw):
        """The model's own candidates in the shape of the ranking's, the rank in its order and every component it leaves out at
        its healthy level, with the codes of the failure modes it cites and no mass, since none exists without the server."""
        healthy = self.healthy_set()
        cited_modes = {m.get("evidence_id"): m for m in self.failure_modes if m.get("evidence_id")}
        rows = []
        for rank, item in enumerate(raw, 1):
            combination = {c: asset_rules.profile_level(self.profile, c, item["levels"][c]) if c in item["levels"] else healthy[c]
                           for c in self.profile["components"]}
            cited = [cited_modes[i] for i in item.get("failure_mode_ids") or [] if i in cited_modes]
            faults = [{"component": c, "level": v, "code": next((m.get("code") for m in cited if m.get("component") == c), None)}
                      for c, v in combination.items() if v != healthy[c]]
            rows.append({"rank": rank, "combination": combination, "current_combination": dict(combination), "probability": None, "faults": faults,
                         "failure_mode_ids": list(item.get("failure_mode_ids") or []), "evidence_ids": [], "own": True})
        return rows

    def set_candidates(self, raw):
        """The candidates of the model's latest plan that listed them, which take off the note on candidates that were not kept; a
        refuted set is released once the model's top candidate names it again after a further reading, the rule a later ranking
        follows with the server."""
        self.model_candidates = self.as_candidates(raw)
        self.candidates_seen = self.observations
        self._release_refuted()
        self.retire_notes("candidates")

    def candidate(self, rank):
        return next((c for c in self.candidates() if c.get("rank") == rank), None)

    def top(self):
        candidates = self.candidates()
        return candidates[0] if candidates else None

    def branches(self):
        return list((self.ranking or {}).get("branches") or [])

    def branch(self, rank):
        """The previewed branch of the candidate of this rank: the server keys a branch by its faulty set and lists in ranks every
        candidate of the set, so candidates of one set share one branch and one plan."""
        return next((b for b in self.branches() if rank in (b.get("ranks") or [b.get("rank")])), None)

    def separating_test(self):
        return (self.earlier_rule() or self.ranking or {}).get("separating_test")

    def pending_verifications(self):
        """[(prediction identifier, cycle of the last close)] for the predictions the agent's orders cite whose every citing
        order has closed and whose verdict is not final: a plan's prediction once the last order of the plan closed, an
        estimate's once its order closed. A verdict that says not yet observable is retried after the next reading."""
        orders_of = {}
        for order in self.orders.values():
            if order["prediction_id"]:
                orders_of.setdefault(order["prediction_id"], []).append(order)
        pending = []
        for prediction_id, orders in orders_of.items():
            if any(o["status"] != "closed" for o in orders):
                continue
            verdict = self.verifications.get(prediction_id)
            if verdict is not None and not str(verdict.get("status", "")).startswith("not yet observable"):
                continue
            closes = [o["closed_t"] if o.get("closed_t") is not None else o["expected_close_t"] for o in orders]
            pending.append((prediction_id, max(closes)))
        return pending

    def open_orders(self):
        return [o for o in self.orders.values() if o["status"] != "closed"]

    def known_ids(self):
        """The evidence identifiers the verifier accepts for what this board has seen."""
        ids = {row["evidence_id"] for row in self.evidence} | set(self.absorbed_ids)
        ids |= {f"alarm/{a['first_t']}/{a['channel']}" for a in self.alarms}
        # the verifier matches wo identifiers against the orders the agent created
        ids |= {f"wo/{order_id}" for order_id in self.orders}
        ids |= {m["evidence_id"] for m in self.failure_modes if m.get("evidence_id")}
        ids |= set(self.predictions)
        # the identifiers the diagnostics server names are those of the released evidence, whatever form the surface cites
        ids |= set((self.ranking or {}).get("evidence_ids") or [])
        for candidate in self.candidates():
            ids |= set(candidate.get("evidence_ids") or [])
        return ids

    def to_json(self):
        """The board as the driver records it beside the plans: everything but the profile and the evidence rows, which the
        released evidence of the episode already holds."""
        return {"status": copy.deepcopy(self.status), "observations": self.observations, "ranking_basis": list(self.ranking_basis or ()),
                "diagnosis": self.diagnosis, "declaration": self.declaration, "orders": copy.deepcopy(self.orders),
                "predictions": copy.deepcopy(self.predictions), "verifications": copy.deepcopy(self.verifications), "alarms": list(self.alarms),
                "closed_orders": list(self.closed_orders), "differential": list(self.differential), "notes": list(self.notes),
                "end_reason": self.end_reason, "fix_history": list(self.fix_history), "refuted": copy.deepcopy(self.refuted),
                "refuted_last": self.refuted_last, "summaries": copy.deepcopy(self.summaries), "notifications": list(self.notifications),
                "compactions": self.compactions, "compaction_log": copy.deepcopy(self.compaction_log), "reads": copy.deepcopy(self.reads),
                "server": self.server, "opening_done": list(self.opening_done), "batches": copy.deepcopy(self.batches), "parallel": self.parallel,
                "answered": copy.deepcopy(self.answered), "calls": copy.deepcopy(self.calls), "retired_notes": self.retired_notes,
                "reply_schema": copy.deepcopy(self.reply_schema),
                **({"candidates": copy.deepcopy(self.model_candidates)} if self.model_candidates or not self.server else {})}

    # ----- idle compaction -----
    def compact(self, fits=None, after_call=None, summarize=None, fits_context=None):
        """Collapse what a ranking has absorbed, and more only while the next planning request passes the compaction trigger.
        fits(board) says whether the request is under the trigger, its prompt alone within
        90 percent of the served context; without it every board is under it. With the server, rows of observations the latest
        ranking has seen collapse to one summary row per sensor group and contiguous observations whatever the size (step 1);
        past the trigger the latest absorbed observation collapses too (step 2); past it still, when an entry that a summary
        replaces is on the board, summarize(board, entries) writes one summary of the earlier turns, which replaces the notes,
        the notifications, the surprises but the latest and the timeline older than the latest ranking (step 3). Step 3 runs only
        when the board without every such entry would be under the trigger, or when fits_context(board) says the request does not
        fit the whole context, so a fixed part past the trigger is not summarized again at every turn. A request that
        then fits the context is sent, and one that does not ends the run as context where it is sent. Every refuted plan keeps
        its line and every collapsed identifier stays citable. after_call is the index of the last tool call, so the compaction
        log says when each compaction happened. Returns the list of what this call changed, empty when nothing changed."""
        fits = fits or (lambda board: True)
        collapsed = []
        # without the server no ranking absorbs a reading, so the observations the model planned on count as absorbed
        absorbed = (self.ranking_basis[0] if self.ranking_basis else 0) if self.server else self.plan_basis
        latest = max((r["n"] for r in self.evidence), default=0)
        older = [r for r in self.evidence if r["n"] <= absorbed and r["n"] < latest]
        if older and (self.server or not fits(self)):  # without the server the rows stay in full while the request is under the trigger
            collapsed += self._collapse(older)
        if not fits(self):
            rest = [r for r in self.evidence if r["n"] <= absorbed]
            if rest:
                collapsed += self._collapse(rest)
        if not fits(self) and summarize is not None:
            entries = self.replaceable()
            # a summary replaces every such entry, so one on the board was written after the last summary
            if entries and (fits(self.without(entries)) or (fits_context is not None and not fits_context(self))):
                written = summarize(self, entries)
                if written is not None:
                    text, left_out = written
                    collapsed.append(self.apply_summary(text, entries, left_out))
        if collapsed:
            self.compactions += 1
            self.compaction_log.append({"after_call": after_call, "collapsed": collapsed, "rendered_characters": len(self.render())})
        return collapsed

    def mark_turn(self, turn):
        """The lengths of the lists a summary replaces at the start of a planning turn, which date every entry to the turn that wrote
        it."""
        self.turn_marks.append((turn, {"notifications": len(self.notifications), "alarm": len(self.alarms), "closed": len(self.closed_orders),
                                       "surprises": len(self.surprises), "notes": len(self.notes)}))
        done = self._progress()
        self.turns_without_progress = self.turns_without_progress + 1 if done == self.progress else 0
        self.progress = done

    def _progress(self):
        """What the episode has done, which a turn changes only by adding evidence or changing a state: the values, the observations
        and the notifications, the answers of the opening, the commitment, the work orders with their states, the verdicts, the
        refuted plans and the timeline. The plan, the notes, the surprises, the candidates, the differential, the ranking, the steps
        left and the clock are not among them, so a restated plan is no progress, and neither is a cycle that the clock runs alone."""
        return repr((len(self.evidence) + len(self.absorbed_ids), self.observations, len(self.notifications), self.answered, self.opening_done,
                     self.diagnosis, self.declaration, sorted((o["id"], o["status"]) for o in self.orders.values()),
                     sorted((key, verdict.get("status")) for key, verdict in self.verifications.items()), len(self.refuted), len(self.alarms),
                     len(self.closed_orders)))

    def _no_progress_line(self):
        """The line of a stall, from NO_PROGRESS_TURNS planning turns in a row without progress on; empty before."""
        return NO_PROGRESS.format(count=self.turns_without_progress) if self.turns_without_progress >= NO_PROGRESS_TURNS else ""

    def _turn_of(self, kind, index):
        turn = 0
        for mark_turn, lengths in self.turn_marks:
            if lengths[kind] <= index:
                turn = mark_turn
        return turn

    def replaceable(self):
        """The entries a summary of earlier turns replaces, oldest first by the planning turn that wrote them and then by their place
        on the board: the notifications, the timeline older than the latest ranking, the surprises but
        the latest, and the notes. The summary of an earlier event is never among them."""
        cut = self.ranking_basis[2] if self.ranking_basis else None
        entries = [{"kind": "notifications", "index": i} for i in range(len(self.notifications))]
        timeline = [("alarm", i, a["first_t"]) for i, a in enumerate(self.alarms)]
        timeline += [("closed", i, o.get("closed_t", o.get("t"))) for i, o in enumerate(self.closed_orders)]
        entries += [{"kind": kind, "index": i, "t": t} for kind, i, t in timeline if cut is not None and t is not None and t < cut]
        entries += [{"kind": "surprises", "index": i} for i in range(len(self.surprises) - 1)]
        entries += [{"kind": "notes", "index": i} for i in range(len(self.notes))]
        # a replaced event stays hidden, and an entry a summary named stays on the board for the rest of the episode
        settled = {tuple(p) for p in self.pinned} | {(k, i) for k, i in self.timeline_hidden}
        entries = [e for e in entries if (e["kind"], self._entry_text(e)) not in settled]
        for entry in entries:
            entry["turn"] = self._turn_of(entry["kind"], entry["index"])
        return sorted(entries, key=lambda e: (e["turn"], SUMMARY_PLACE[e["kind"]], e["index"]))

    def _remove(self, entries):
        """Take these entries off the board: the listed notes, notifications and surprises leave, and the replaced alarms and
        closed orders leave the timeline by their identifiers. Returns the count per kind."""
        removed = {}
        for kind in ("notes", "notifications", "surprises"):
            gone = {e["index"] for e in entries if e["kind"] == kind}
            if kind == "notes":  # the ties of a note are keyed by its text, so they leave with it
                for i in gone:
                    self.note_ties.pop(self.notes[i], None)
                    self.note_repeats.pop(self.notes[i], None)
            setattr(self, kind, [x for i, x in enumerate(getattr(self, kind)) if i not in gone])
            removed[kind] = len(gone)
        events = [[e["kind"], self._entry_text(e)] for e in entries if e["kind"] in ("alarm", "closed")]
        self.timeline_hidden = self.timeline_hidden + events
        removed["timeline"] = len(events)
        return removed

    def without(self, entries):
        """A copy of the board without these entries, which the summary request renders when the oldest must be left out of it to
        fit the context; the board itself is unchanged."""
        board = copy.deepcopy(self)
        board._remove(entries)
        return board

    def _entry_text(self, entry):
        """The text of a replaceable entry in which an identifier can stand: an alarm or a closed order by its identifier, a note,
        a notification or a surprise as it is held."""
        kind, index = entry["kind"], entry["index"]
        if kind == "alarm":
            alarm = self.alarms[index]
            return f"alarm/{alarm['first_t']}/{alarm['channel']}"
        if kind == "closed":
            return f"wo/{self.closed_orders[index]['id']}"
        item = getattr(self, kind)[index]
        return item if isinstance(item, str) else repr(item)

    def apply_summary(self, text, entries, left_out):
        """Step 3 done: the summary stands in a section of its own after the failure modes and the entries it replaces leave. An
        entry that holds an identifier the summary names, as a whole identifier, stays word for word and is no longer replaceable.
        The log counts what was left out of the summary request, lost and not
        summarized, and what the summary kept."""
        named = {i for i in self.known_ids() if i and names(text, i)}
        kept = [e for e in entries if any(names(self._entry_text(e), i) for i in named)]
        self.pinned = self.pinned + [[e["kind"], self._entry_text(e)] for e in kept]
        gone = [e for e in entries if e not in kept]
        replaced = self._remove(gone)
        self.summary_text = (text or "").strip() or NO_SUMMARY
        self.summary_turn = self.turn_marks[-1][0] if self.turn_marks else 0
        lost, kept_named = {}, {}
        for entry in entries[:left_out]:
            if entry in kept:
                continue
            kind = "timeline" if entry["kind"] in ("alarm", "closed") else entry["kind"]
            lost[kind] = lost.get(kind, 0) + 1
        for entry in kept:
            kind = "timeline" if entry["kind"] in ("alarm", "closed") else entry["kind"]
            kept_named[kind] = kept_named.get(kind, 0) + 1
        return {"kind": "summary", "replaced": replaced, "left_out_of_summary_request": lost, "kept_for_named_identifiers": kept_named,
                "summary_characters": len(self.summary_text)}

    def recallable(self, group, first, last):
        """The full rows of the group's observations first to last that a collapse took off the board, oldest first."""
        return [r for r in self.collapsed_rows if r["sensor_group"] == group and first <= r["n"] <= last]

    def recall(self, group, first, last):
        """A recall step: those rows come back on the board of the next planning turn, once; no tool is
        called and nothing is charged. Returns how many came back."""
        rows = self.recallable(group, first, last)
        turn = (self.turn_marks[-1][0] if self.turn_marks else 0) + 1
        self.recalled = (self.recalled if self.recall_turn == turn else []) + rows
        self.recall_turn = turn
        return len(rows)

    def drop_unfit_recall(self, fits_context):
        """At the start of the planning turn that shows the recalled rows, before its request: the rows leave, logged, when the
        request with them would not fit the whole context, so a recall never ends a run as context."""
        if self._render_recalled() and not fits_context(self):
            self.compactions += 1
            self.compaction_log.append({"after_call": None, "collapsed": [{"kind": "recall_dropped", "rows": len(self.recalled)}],
                                        "rendered_characters": len(self.render()) - len(self._render_recalled())})
            self.recalled = []

    def _collapse(self, rows):
        """Replace evidence rows by summary rows, one per sensor group and run of consecutive observations, merged into the
        last summary of the same group when the observations continue it."""
        collapsed = []
        taken = {id(r) for r in rows}
        self.collapsed_rows += [dict(r) for r in rows]  # kept whole for a recall step
        self.evidence = [r for r in self.evidence if id(r) not in taken]
        for row in sorted(rows, key=lambda r: (r["n"], r["evidence_id"])):
            last = self.summaries[-1] if self.summaries else None
            if last is not None and last["sensor_group"] == row["sensor_group"] and row["n"] in (last["n_last"], last["n_last"] + 1):
                summary = last
            else:
                summary = {"sensor_group": row["sensor_group"], "n_first": row["n"], "n_last": row["n"], "t_first": row["t"], "t_last": row["t"],
                           "cycles": [], "values": 0, "outside": [], "sides": {}}
                self.summaries.append(summary)
                collapsed.append(summary)
            summary["n_last"] = row["n"]
            if row["t"] not in summary["cycles"]:
                summary["cycles"] = sorted(summary["cycles"] + [row["t"]])
            summary["t_first"], summary["t_last"] = summary["cycles"][0], summary["cycles"][-1]
            summary["values"] += 1
            if row["inside_band"] is False:
                summary["outside"].append(row["evidence_id"])
                summary["sides"][row["evidence_id"]] = "below" if row["value"] < row["band_low"] else "above"
            self.absorbed_ids.add(row["evidence_id"])
        return [{"kind": "evidence", "sensor_group": s["sensor_group"], "observations": [s["n_first"], s["n_last"]], "values": s["values"]} for s in collapsed]

    # ----- the rendering the model reads -----
    def render(self):
        """The board in the order of how often a part changes: first what holds for the whole
        episode (the prices, the work order terms, the fix history and the failure modes), then the evidence, which only grows at its
        end, then the parts a turn may rewrite, and last the status, which changes every turn. Two requests of one episode then
        share their longest beginning, which the prefix cache of the serving engine computes once."""
        if self.profile is None:
            return self._render_opening()
        if not self.server:
            parts = [self._render_terms(), self._render_fix_history(), self._render_failure_modes(), self._render_summary(), self._render_evidence(),
                     self._render_outside(), self._render_candidates(), self._render_schedule(), self._render_refuted(), self._render_timeline(), self._render_orders(),
                     self._render_differential(),
                     self._render_surprises(), self._render_plan(), self._render_recalled(), self._render_status()]
            return "\n\n".join(p for p in parts if p)
        parts = [self._render_terms(), self._render_fix_history(), self._render_failure_modes(), self._render_summary(), self._render_evidence(),
                 self._render_outside(), self._render_ranking(), self._render_candidates(), self._render_advice(), self._render_schedule()]
        if self.branches():
            parts.append(self._render_branches())
        parts += [self._render_refuted(), self._render_timeline(), self._render_orders(), self._render_predictions(), self._render_differential(),
                  self._render_surprises(), self._render_plan(), self._render_recalled(), self._render_status()]
        return "\n\n".join(p for p in parts if p)

    def _render_opening(self):
        """The board before the profile is read: what a plan made now holds, in the words the plan check refuses a plan by, and the
        controller's notes, as the board after the profile shows them, so that the reasons of every plan refused before the profile stay in view."""
        from CausalDiagnosis.agents.Causal_Diagnosis_Agent.plan import plan_schema  # plan imports this module, so the rule is imported once a board renders
        rule = plan_schema.OPENING_RULE
        text = (f"# Situation board\nNo profile read yet, so the board holds nothing about the {self.noun}. {rule[0].upper()}{rule[1:]}: "
                f"{plan_schema.OPENING_ORDER}. It lists no candidates. Plan every other step {plan_schema.NEXT_REQUEST}.")
        return "\n".join(line for line in (text, self._no_progress_line(), self._notes_line()) if line)

    def _notes_line(self):
        """The controller's notes on one line, the same before and after the profile is read; empty while there is none."""
        return "Controller notes: " + " ".join(self.notes) if self.notes else ""

    def _render_terms(self):
        """The head of the board: the read prices, the terms of a work order and the clock tools of the row, which hold for the
        whole episode."""
        lines = []
        durations = asset_rules.durations_of(self.profile)
        lines.append("Read prices per cycle: " + ", ".join(f"{g} {text_formatting.fmt(p)}" for g, p in asset_rules.read_prices(self.profile).items())
                     + (f"; a read of n cycles takes {text_formatting.n_cycles(durations['read_per_cycle'])} per cycle read." if self.reads_take_time
                        else "; a read takes no time, since it serves cycles already recorded."))
        lines.append("Work orders: part plus repair charge " + ", ".join(f"{c} {text_formatting.fmt(self.order_price(c))}" for c in self.profile["components"])
                     + f", plus one visit charge of {text_formatting.fmt(self.visit_charge)} for the orders created one after another with no other call between them"
                     + "; an order closes after the lead time plus the repair downtime: "
                     + ", ".join(f"{c} {self.order_duration(c)} cycles" for c in self.profile["components"]) + ".")
        listed = {t.get("name") for t in self.profile.get("tools") or [] if isinstance(t, dict)}
        if "get_current_time" in listed:  # the two time tools, listed on the clock rows only
            lines.append("Clock tools on this row: get_current_time reads the cycle without moving the clock, and wait_for_notification(timeout) "
                         f"lets the {self.noun} run until the next notification or the timeout, so a wait step ends at the first notification.")
        return "# Situation board\n" + "\n".join(lines)

    def _render_status(self):
        """The last part of the board, which changes every turn: the time, the budget and the steps left, the commitment, the counts
        of the evidence, what the board holds (_held_lines), a stall of the planning turns, the reads the row still requires and the
        controller's notes."""
        commitment = "none"
        if self.diagnosis is not None:
            commitment = commitment_text("diagnose", self.diagnosis["diagnosis"])
        elif self.declaration is not None:
            test = self.declaration["separating_test"]
            commitment = commitment_text("declare", (test["sensor_group"], test["cycles"]))
        lines = [f"{self.limits_line()} Commitment: {commitment}."]
        if self.evidence or self.summaries:
            lines.append(f"Evidence so far: {len(self.evidence) + len(self.absorbed_ids)} values from {self.observations} observation(s).")
        lines += self._order_lines()
        lines += self._held_lines()
        no_progress = self._no_progress_line()
        if no_progress:
            lines.append(no_progress)
        missing = [kind for kind in self.required_opening if kind not in self.opening_done]
        if missing:
            lines.append("Still to read before a reading on this row, as steps of their own: " + ", ".join(missing) + ".")
        if self.notes:
            lines.append(self._notes_line())
        return "## Status\n" + "\n".join(lines)

    def _render_fix_history(self):
        if not self.fix_history:
            return "## Fix history\nNo recorded repair before this episode."
        rows = [", ".join(f"{k} {text_formatting.fmt(v)}" for k, v in entry.items()) if isinstance(entry, dict) else str(entry) for entry in self.fix_history]
        return "## Fix history (repairs recorded before this episode)\n" + "\n".join(f"- {r}" for r in rows)

    def _render_advice(self):
        """The advice of the diagnostics server: the expected elimination and the information gain of one cycle of
        every sensor group with the server's read batch, then the reads, the repair plan and, after a refuted plan, the other plan
        that plan.advice names, where the plan check used to bind. The controller reads, diagnoses and orders
        only what a plan names."""
        from CausalDiagnosis.agents.Causal_Diagnosis_Agent.plan import advice  # plan imports this module, so the advice is imported once a board renders
        rows, lines = self.elimination(), advice(self)
        if not rows and not lines:
            return ""
        parts = []
        unit = ((self.ranking or {}).get("costs") or {}).get("unit") or next((e.get("cost_unit") for e in rows if e.get("cost_unit")), None)
        if rows:
            # the server chooses the batch head by information gain per unit cost and keeps the expected elimination as a
            # reported number, so both appear when the rows carry the gain
            with_gain = any(e.get("information_gain") is not None for e in rows)
            with_ec2 = any(e.get("ec2") is not None for e in rows)  # one step edge cutting over repair plan classes, after the EC2 objective of Golovin, Krause and Ray
            headers = ["group", "price", "cost", "effective before", "predicted effective after", "predicted elimination", "per unit cost"]
            parts.append(text_formatting.table(headers + (["predicted information gain (nats)", "predicted gain per unit cost"] if with_gain else [])
                               + (["predicted edge cut", "predicted edge cut per unit cost"] if with_ec2 else []),
                               [(e.get("sensor_group"), text_formatting.fmt(e.get("price")), text_formatting.fmt(e.get("cost")), text_formatting.fmt(e.get("effective_before")), text_formatting.fmt(e.get("expected_effective_after")),
                                 text_formatting.fmt(e.get("expected_elimination")), text_formatting.fmt(e.get("elimination_per_cost")))
                                + ((text_formatting.fmt(e.get("information_gain")), text_formatting.fmt(e.get("ig_per_cost"))) if with_gain else ())
                                + ((text_formatting.fmt(e.get("ec2")), text_formatting.fmt(e.get("ec2_per_cost"))) if with_ec2 else ()) for e in rows]))
            batch = self.read_batch()
            parts.append("Read batch of the server, the reads whose choice no outcome of the reads before them can change"
                         + (", best first by information gain per unit cost" if with_gain else "") + ": "
                         + (", ".join(f"{g} x{c}" for g, c in batch) if batch else "none") + ".")
        parts += lines
        parts.append("This is advice: the controller reads, diagnoses and orders only what your plan names.")
        return f"## Advice of the CausalDiagnosis MCP (E21 and E22{f', costs in {unit}' if unit else ''})\n" + "\n".join(parts)

    def _render_schedule(self):
        """The schedule of the candidate plans (candidate_plan_schedule.py), when it says more than the board without it: the reads of one
        batch with the plans each serves, the reads and the plans that no longer fit, and after a refuted plan what each
        remaining plan still needs. It is advice like the rest of the board."""
        from CausalDiagnosis.agents.Causal_Diagnosis_Agent.plan import exploration  # plan imports this module, so the schedule is imported once a board renders
        from CausalDiagnosis.agents.Causal_Diagnosis_Agent.candidate_plan_schedule import lines
        scheduled = exploration(self)
        if scheduled is None or not scheduled.differs:
            return ""
        return "\n".join(lines(scheduled, self.server))

    def _render_failure_modes(self):
        """The failure modes as the server of the row returns them: the level, the severity, the action and the indicating channels
        where the row gives them. Without the diagnostics server the model forms its candidates from them, which the heading says."""
        if not self.failure_modes:
            return "## Failure modes\nNone read yet. A failure_modes step reads them."
        rows = [(m.get("component"), "" if m.get("level") is None else m.get("level"), m.get("code"), m.get("severity") or "", m.get("action") or "",
                 ", ".join(m.get("channels") or []), m.get("evidence_id")) for m in self.failure_modes]
        heading = ("## Failure modes (no server ranks the combinations on this row, so you form the candidates from these, the fix history and the readings)"
                   if not self.server else "## Failure modes")
        return heading + "\n" + text_formatting.table(["component", "level", "code", "severity", "recommended action", "indicating channels", "identifier"], rows)

    def _render_outside(self):
        """The channel means outside their public normal range in the latest cycle of every sensor group read, below or above it."""
        if not self.latest_cycles:
            return "## Channels outside their normal range\nNo reading yet."
        lines = [f"- {group}, observation {c['n']} at cycle {c['t']}: {len(c['outside'])} of {c['banded']} channel {self.statistic}s outside"
                 + (", " + ", ".join(f"{name} {side}" for name, side in c["outside"]) if c["outside"] else "")
                 for group, c in self.latest_cycles.items()]
        return "## Channels outside their normal range (the latest cycle of every sensor group read)\n" + "\n".join(lines)

    def _render_candidates(self):
        if not self.model_candidates:
            return ""
        own = [c for c in self.candidates() if c.get("own")]
        rows = [(c["rank"], text_formatting.combination_text(c["combination"]), ", ".join(f"{f['component']} {f['level']}" for f in c["faults"]) or "none",
                 ", ".join(c.get("failure_mode_ids") or []) or "none") for c in own]
        heading = ("## Your candidates (from your latest plan that listed them, in your order, and a diagnosis names a rank of this list)" if not self.server
                   else "## Your candidates (from your latest plan that listed them, in your order after the ranked candidates, and a diagnosis may name their rank)")
        return heading + "\n" + text_formatting.table(["rank", "combination", "faulty", "failure modes cited"], rows)

    def _render_refuted(self):
        if not self.refuted:
            return ""
        rows = [(r["cycle"], ", ".join(r["components"]), r["reason"]) for r in self.refuted]
        return ("## Refuted repair plans (a verification found the reading after the last close inconsistent with the plan; the last set "
                "is not planned again until a later ranking names it first)\n"
                + text_formatting.table(["cycle", "components", "reason"], rows))

    def _render_evidence(self):
        """The evidence: the normal range of every channel once, in a legend that the profile fixes, the collapsed rows, one row
        per value, and the notifications. A row holds no range of its own, so the section grows only at the end of its rows and
        two requests share every row that the earlier one held."""
        if not self.evidence and not self.summaries and not self.notifications:
            return "## Evidence\nNo reading yet."
        parts = [self._normal_range_legend()]
        if self.summaries:
            parts.append(("Absorbed by the ranking and collapsed" if self.server else "Planned on and collapsed")
                         + f" (cite a value as {self.evidence_form}):\n" + text_formatting.table(
                ["group", "observations", "cycles", "values", "outside their normal range"],
                [(s["sensor_group"], f"{s['n_first']} to {s['n_last']}" if s["n_first"] != s["n_last"] else str(s["n_first"]),
                  text_formatting.cycles_text(s["cycles"]), s["values"],
                  ", ".join(f"{i} {s['sides'][i]}" if i in s.get("sides", {}) else i for i in s["outside"]) if s["outside"] else "none")
                 for s in self.summaries]))
        if self.evidence:
            parts.append(text_formatting.table(["id", "t", "group", "value"],
                                               [(r["evidence_id"], r["t"], r["sensor_group"], value_text(r)) for r in self.evidence]))
        if self.notifications:
            parts.append("Notifications, the public events the tool answers carried (no reading):\n"
                         + text_formatting.table(["t", "event", "detail"], [(n["t"], n["event"], n["detail"]) for n in self.notifications]))
        # the counts of values and observations stand in the status at the end of the board, so that this heading holds
        return f"## Evidence (the normal range applies to channel {self.statistic}s only)\n" + "\n".join(part for part in parts if part)

    def _normal_range_legend(self):
        """The public normal range of every channel that the profile gives one, once and by field name, empty without any."""
        bands = self.normal_ranges()
        if not bands:
            return ""
        return ("Normal ranges (a value outside its range ends in low or high):\n"
                + text_formatting.table(["field", "normal range"], [(name, f"{text_formatting.fmt(band['low'])} to {text_formatting.fmt(band['high'])}")
                                                                    for name, band in sorted(bands.items())]))

    def _render_ranking(self):
        if self.ranking is None:
            return "## Ranking\nNo ranking yet."
        r = self.ranking
        current = "" if self.ranking_is_current() else " (STALE: evidence arrived after this ranking)"
        belief = ""
        if r.get("effective_hypotheses") is not None:
            belief = f", effective hypotheses {text_formatting.fmt(r['effective_hypotheses'])}"
            if r.get("belief_time") is not None:
                belief += f" at cycle {r['belief_time']}"
        # the version 7 server keeps no start combination, so one column shows the combination a candidate holds now
        rows = [(c["rank"], text_formatting.combination_text(c.get("current_combination") or c["combination"]),
                 ", ".join(f["code"] for f in c["faults"] if f.get("code")) or "none", text_formatting.fmt(c["probability"]))
                for c in r.get("candidates", [])]
        states = f" over {r['n_states']} states" if r.get("n_states") is not None else ""
        lines = [f"## Ranking ({r.get('ranking_model')}, {r.get('n_candidates')} candidate combinations{states}, top {len(rows)} shown{belief}){current}",
                 text_formatting.table(["rank", "combination", "failure modes", "mass"], rows),
                 "Failure mode identifiers have the form fmea/" + self._fmea_source() + "/<code>. A diagnosis submits the combination of the candidate it names."]
        test = self.separating_test()
        decision = (f"Decision fields ({'E10' if self.earlier_rule() is not None else 'the rule of the server'}): read_first "
                    f"{self.read_first()}; predicted expected loss of acting now "
                    f"{text_formatting.fmt(r.get('expected_loss_act'))}; predicted expected saving of the best test {text_formatting.fmt(r.get('expected_saving'))}.")
        if test:
            decision += (f" Separating test: {test['sensor_group']} for {text_formatting.n_cycles(test['cycles'])}, price {text_formatting.fmt(test['price'])}, duration "
                         f"{test['duration']}, cost {text_formatting.fmt(test['cost'])}, predicted expected loss after it {text_formatting.fmt(test['expected_loss'])}.")
        losses = r.get("expected_loss_test") or {}
        if losses:
            decision += " Predicted expected loss after one cycle of each group: " + ", ".join(f"{g} {text_formatting.fmt(v)}" for g, v in losses.items()) + "."
        lines.append(decision)
        if r.get("unseparated"):
            lines.append("Unseparated components: " + ", ".join(
                f"{c} (channels {', '.join((r.get('separating_channels') or {}).get(c, []))})" for c in r["unseparated"]) + ".")
        now = r.get("current_fault_probability") or {}
        if now:
            lines.append("Fault probability per component now: " + ", ".join(f"{c} {text_formatting.fmt(p)}" for c, p in now.items()) + ".")
        return "\n".join(lines)

    def _fmea_source(self):
        for mode in self.failure_modes:
            parts = (mode.get("evidence_id") or "").split("/")
            if len(parts) == 3:
                return parts[1]
        return "<source>"

    def _render_branches(self):
        rows = []
        for b in self.branches():
            refuted = " (REFUTED, not planned again until a later ranking names it first)" if sorted(b.get("faulty_set") or []) == self.refuted_last else ""
            test = b.get("confirm_test") or {}
            two = troubleshooting_plan_fields.two_step_test(b)
            confirm = (f"{test.get('sensor_group')} x{test.get('cycles')}: price {text_formatting.fmt(test.get('price'))}, predicted mass if true {text_formatting.fmt(test.get('mass_if_true'))}, "
                       f"if false {text_formatting.fmt(test.get('mass_if_false'))}, gap {text_formatting.fmt(test.get('gap'))}") if test else "none"
            if two:
                groups = two.get("sensor_groups") or two.get("groups") or []
                confirm += f"; two step {' then '.join(groups)}: price {text_formatting.fmt(two.get('price'))}, gap {text_formatting.fmt(two.get('gap'))}"
            by_group = test.get("gap_by_group") or {}
            if by_group:
                confirm += "; gap by group " + ", ".join(f"{g} {text_formatting.fmt(v)}" for g, v in by_group.items())
            plan_id = troubleshooting_plan_fields.plan_prediction_id(b)
            plan = "; ".join(f"{i['component']}" + ("" if plan_id else f": {i.get('prediction_id') or 'no prediction'}")
                             + f" (price {text_formatting.fmt(i.get('part_price', 0) + i.get('repair_charge', 0))}, closes after "
                             f"{i.get('lead_time', 0) + i.get('repair_downtime', 0)} cycles)" for i in troubleshooting_plan_fields.repair_steps(b)) or "none"
            if plan_id and troubleshooting_plan_fields.repair_steps(b):
                plan += f"; every order of the plan cites {plan_id}; one visit charge {text_formatting.fmt(self.visit_charge)}"
            outcome = b.get("predicted_outcome") or {}
            restored = b.get("cost_to_restored") or {}
            rows.append((b.get("rank"), (", ".join(b.get("faulty_set") or []) or "none") + refuted, text_formatting.fmt(b.get("probability")), confirm, plan,
                         troubleshooting_plan_fields.cycles_to_restored(b), text_formatting.fmt(outcome.get("recovery_if_acted")), text_formatting.fmt(b.get("expected_loss_if_acted")),
                         text_formatting.fmt(b.get("expected_loss_if_tested")), f"{text_formatting.fmt(restored.get('act_now'))} / {text_formatting.fmt(restored.get('test_first'))}"))
        header = ["rank", "faulty set", "mass", "confirm test", "repair plan (work orders, price, duration, the prediction id every order cites)", "cycles to restored",
                  "predicted recovery if acted", "predicted loss if acted", "predicted loss if tested", "predicted cost to restored act now / test first"]
        return ("## Troubleshooting plans (the server walked each candidate forward; every work order of a repair plan cites the plan's prediction id, "
                "which one reading after the last order closed verifies)\n"
                + text_formatting.table(header, rows))

    def _render_timeline(self):
        events = [(a["first_t"], f"alarm alarm/{a['first_t']}/{a['channel']}: the channel {self.statistic} left its normal range") for a in self.alarms]
        events += [(o.get("closed_t", o.get("t")), f"work order wo/{o['id']} on the {o['component']} closed, source {o.get('source')}")
                   for o in self.closed_orders]
        if self.timeline_hidden:  # the events a summary of earlier turns replaced, by their identifiers
            hidden = {i for _, i in self.timeline_hidden}
            ids = [f"alarm/{a['first_t']}/{a['channel']}" for a in self.alarms] + [f"wo/{o['id']}" for o in self.closed_orders]
            events = [e for e, i in zip(events, ids) if i not in hidden]
        if not events:
            return "## Timeline\nNo alarm and no closed work order so far."
        return "## Timeline\n" + text_formatting.table(["t", "event"], sorted(events, key=lambda e: (e[0] if e[0] is not None else -1, e[1])))

    def _render_orders(self):
        if not self.orders:
            return "## Work orders\nNone created."
        rows = [(o["id"], o["component"], o["created_t"], o["expected_close_t"], o["status"], o["prediction_id"] or "none", self._verdict(o))
                for o in self.orders.values()]
        return "## Work orders\n" + text_formatting.table(["id", "component", "created", "expected close", "status", "prediction" if self.server else "repair plan",
                                           "verification"], rows)

    def _verdict(self, order):
        verdict = self.verifications.get(order["prediction_id"]) or {}
        if self.server or not verdict:
            return verdict.get("status", "not verified")
        return f"{verdict['status']}, consistent {verdict['consistent']}, {verdict['message']}"

    def _render_predictions(self):
        if not self.predictions:
            return "## Predictions\nNone yet; an order with prediction_id \"estimate\" makes the controller call estimate_intervention_effect first."
        rows = []
        for pid, p in self.predictions.items():
            v = self.verifications.get(pid)
            verdict = "not verified" if v is None else (f"{v['status']}; consistent {v['consistent']}; recovery predicted {text_formatting.fmt(v['recovery_predicted'])}, "
                                                        f"posterior after the repair {text_formatting.fmt(v['recovery_posterior'])}{text_formatting.scored_combination_text(v)}")
            rows.append((pid, p["action"], text_formatting.fmt(p.get("recovery")), p["source"], verdict))
        return "## Predictions\n" + text_formatting.table(["id", "action", "predicted recovery", "source", "verification"], rows)

    def _render_differential(self):
        if not self.differential:
            return ""
        rows = [(d.get("component"), d.get("expected_symptoms"), ", ".join(d.get("failure_mode_ids") or [])) for d in self.differential]
        return "## Differential (your orientation)\n" + text_formatting.table(["component", "expected symptoms", "failure modes"], rows)

    def _render_summary(self):
        return f"{SUMMARY_SECTION}\n{self.summary_text}" if self.summary_text else ""

    def _render_surprises(self):
        if not self.surprises:
            return "## Surprises\nNone so far."
        rows = [(s["plan"], s["step"], s["kind"], s["reason"]) for s in self.surprises]
        return "## Surprises (an expectation that failed stops its plan)\n" + text_formatting.table(["plan", "step", "kind", "what happened"], rows)

    def _render_recalled(self):
        """The rows a recall step brought back, on the board of the next planning turn only."""
        if not self.recalled or not self.turn_marks or self.turn_marks[-1][0] != self.recall_turn:
            return ""
        return (f"{RECALL_SECTION} (cite a value by its id)\n"
                + text_formatting.table(["id", "t", "group", "value"], [(r["evidence_id"], r["t"], r["sensor_group"], value_text(r)) for r in self.recalled]))

    def _render_plan(self):
        if not self.plan:
            return ""
        rows = [(i + 1, s["kind"], s["args"], s.get("expect"), s.get("status", "pending")) for i, s in enumerate(self.plan["steps"])]
        return (f"## Last plan (branch {self.plan.get('branch')}, mode {self.plan.get('mode')}): {self.plan.get('why', '')}\n"
                + text_formatting.table(["step", "kind", "args", "expectation", "status"], rows))
