"""Result models of the diagnostics tools (CausalDiagnosis/docs/causal-diagnosis-mcp.md). Every tool result carries the
evidence identifiers it rests on (obs/{n}/{field}, wo/{id}, pred/{n}), its time window in cycles, its assumptions as plain
statements that trace every threshold to a price, a duration, a downtime rate or the declared error level, one
uncertainty figure and a one line message; a failed call returns an ErrorResult."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, PrivateAttr, model_serializer


class ErrorResult(BaseModel):
    error: str


class TimeWindow(BaseModel):
    start: int
    end: int
    unit: str


class Uncertainty(BaseModel):
    kind: str
    value: float
    detail: Optional[Dict[str, float]] = None


class ToolResult(BaseModel):
    evidence_ids: List[str]
    time_window: TimeWindow
    assumptions: List[str]
    uncertainty: Uncertainty
    message: str


class ClosedWorkOrder(BaseModel):
    id: str
    component: str
    source: str
    t: int


class EvidenceItem(BaseModel):
    evidence_id: str
    field: str
    channel: str
    statistic: str
    value: float
    unit: str
    rate: str
    n: int
    t: int
    sensor_group: str
    normal_low: Optional[float]
    normal_high: Optional[float]
    in_normal_range: Optional[bool]
    anomaly_score: Optional[float]


class EpisodeEvidence(ToolResult):
    sensor_group: str
    n_observations: int
    items: List[EvidenceItem]
    work_orders_closed: List[ClosedWorkOrder]


class Fault(BaseModel):
    component: str
    level: int
    code: Optional[str]


class Candidate(BaseModel):
    rank: int
    combination: Dict[str, int]  # the combination now, the hypothesis a diagnosis names
    current_combination: Dict[str, int]  # the same combination, under the name the readers of version 6 use
    faults: List[Fault]
    probability: float
    evidence_ids: List[str]


class ChannelShare(BaseModel):
    field: str
    anomaly_score: float
    share: float


class ComponentAttribution(BaseModel):
    component: str
    current_fault_probability: float
    explained_anomaly: float
    channels: List[ChannelShare]


class Stability(BaseModel):
    top_component: str
    top_marginal: float
    interval_low: float
    interval_high: float
    top_share: float
    resamples: int


class SeparatingTest(BaseModel):
    sensor_group: str
    cycles: int
    price: float
    duration: int
    cost: float
    expected_loss: float


class MonteCarlo(BaseModel):
    samples: int
    seed: int
    standard_error: Dict[str, float]


class PredictedSummary(BaseModel):
    field: str
    mean: float
    low: float
    high: float
    quantile: float


class ConfirmTest(BaseModel):
    sensor_group: str
    cycles: int
    price: float
    duration: int
    cost: float
    mass_if_true: float
    mass_if_false: float
    gap: float
    standard_error: float
    gap_by_group: Optional[Dict[str, float]]


class TwoStepTest(BaseModel):
    sensor_groups: List[str]
    cycles: int
    price: float
    duration: int
    cost: float
    mass_if_true: float
    mass_if_false: float
    gap: float
    standard_error: float


class RepairStep(BaseModel):
    component: str
    part_price: float
    repair_charge: float
    lead_time: int
    repair_downtime: int
    close_offset: int
    downtime_cost: float
    depends_on: List[str]
    dependencies_in_plan: List[str]
    prediction_id: str


class RepairPlan(BaseModel):
    steps: List[RepairStep]
    plan_prediction_id: Optional[str]
    visits: int
    visit_charge: float
    price: float
    lead_time: int
    waiting_cost: float
    downtime_cost: float
    cycles_to_restored: int


class PredictedOutcome(BaseModel):
    predicted: Optional[List[PredictedSummary]]
    alpha: float
    level_reachable: bool  # whether the calibration blocks can carry the declared level (Calibration.level_reachable)
    recovery_if_acted: float
    restored_if_true: bool


class CostToRestored(BaseModel):
    act_now: float
    test_first: float


class Elimination(BaseModel):
    sensor_group: str
    cycles: int
    price: float
    duration: int
    cost: float
    cost_unit: str
    effective_before: float
    expected_effective_after: float
    expected_elimination: float
    elimination_per_cost: float
    standard_error: float
    information_gain: float
    ig_per_cost: float
    ec2: Optional[float]  # None in the compact reply, which computes no edge cutting
    ec2_per_cost: Optional[float]


class BatchRead(BaseModel):
    sensor_group: str
    cycles: int
    price: float
    cost: float
    cost_unit: str
    elimination_per_cost: float
    information_gain: float
    ig_per_cost: float


class Branch(BaseModel):
    rank: int  # the rank of the most probable candidate of the faulty set
    ranks: List[int]  # the ranks of the returned candidates whose faulty set this is
    combination: Dict[str, int]
    current_combination: Dict[str, int]
    faults: List[Fault]
    probability: float
    faulty_set: List[str]
    confirm_test: ConfirmTest
    two_step: Optional[TwoStepTest]
    repair_plan: RepairPlan
    predicted_outcome: PredictedOutcome
    expected_loss_if_acted: float
    expected_loss_if_tested: float
    cost_to_restored: CostToRestored


class Advice(BaseModel):
    """The step that the rule of the server advises before a diagnosis. declare_above gives, for each measure of the verifier,
    the chance of a start that both separability labels mark inseparable above which a declaration would earn more than a
    diagnosis of the first candidate, None where no chance does."""
    action: str  # read, diagnose or declare
    sensor_group: Optional[str] = None  # the reading to take first, or under the fixed cut the reading a declaration names
    cycles: Optional[int] = None
    rank: Optional[int] = None  # the candidate to diagnose and whose faulty set to order
    declare_above: Dict[str, Optional[float]]
    text: str


class E10Rule(BaseModel):
    """The earlier rule, which the long forms of the ranking report beside the rule of the server: read_first is E10 with the
    margin of the decision threshold card, separating_test its test for one cycle, read_batch the batch of E22 with the cap and
    the reserve of the card, and text that rule in words."""
    read_first: bool
    margin: float
    separating_test: Optional[SeparatingTest]
    read_batch: List[BatchRead]
    batch_cap: Optional[int]  # none unless the decision threshold card sets one
    reserve_plans: int
    text: str


class MoreCandidates(BaseModel):
    """The candidates with positive mass that a ranking leaves out: how many and their summed mass. A caller asks again with a
    larger top_k to see them."""
    count: int
    mass: float


class SetMass(BaseModel):
    """The mass of one faulty set under the belief: the summed probability of every combination whose faulty components are
    this set, whether or not the ranking returns it, so it does not move with top_k."""
    faulty_set: List[str]
    mass: float


class RootCauses(ToolResult):
    ranking_model: str
    n_candidates: int
    n_states: int
    resampled_fits: int
    belief_time: int
    effective_hypotheses: float
    candidates: List[Candidate]
    more_candidates: MoreCandidates
    set_masses: List[SetMass]
    current_fault_probability: Dict[str, float]
    # this key, attribution, stability and monte_carlo are left out of the compact reply, so their defaults keep them out of the
    # required keys of the output schema that the MCP checks a reply against
    level_probability: Optional[Dict[str, Dict[str, float]]] = None
    attribution: Optional[List[ComponentAttribution]] = None
    stability: Optional[Stability] = None
    read_first: bool
    separating_test: Optional[SeparatingTest]
    expected_saving: float
    expected_loss_act: float
    expected_loss_test: Dict[str, float]
    unseparated: List[str]
    separating_channels: Dict[str, List[str]]
    elimination: List[Elimination]
    read_batch: List[BatchRead]
    left_to_read: Dict[str, int]  # per sensor group the cycles the row can still serve it, all zero when no reading adds evidence
    monte_carlo: Optional[MonteCarlo] = None
    costs: Dict[str, Any]
    branches: List[Branch]
    advice: Optional[Advice] = None  # the step of the rule of the server, before a diagnosis or a declaration
    e10_rule: Optional[E10Rule] = None  # the earlier rule, in the long forms of the ranking
    _left_out: tuple = PrivateAttr(default=())  # the keys the detail of the call leaves out, named by the adapter

    @model_serializer(mode="wrap")
    def _without_absent_blocks(self, handler):
        """A reply leaves out advice and e10_rule when they are absent, so a reply without them keeps the keys it had before
        the fields existed, and the keys that the detail of the call leaves out. A stability that is None in summary is still
        written as null."""
        data = handler(self)
        for key in ("advice", "e10_rule"):
            if data.get(key) is None:
                data.pop(key, None)
        for key in self._left_out:
            data.pop(key, None)
        return data


class InterventionEffect(ToolResult):
    prediction_id: str
    action: str
    component: Optional[str]
    addresses_fault: float
    recovery: float
    recovery_interval_low: Optional[float]
    recovery_interval_high: Optional[float]
    price: float
    duration: int
    downtime_during_action: float
    predicted: List[PredictedSummary]
    alpha: float
    level_reachable: bool
    risk_before: float
    risk_after: float
    direction: str
    dead_band: Optional[float]
    resampled_fits: int  # the fits the range of E6 is taken over; below conformal.fits_for_level(alpha) the direction is undetermined


class ChannelCheck(BaseModel):
    field: str
    predicted: float
    observed: float
    low: float
    high: float
    score: float


class OutcomeVerification(ToolResult):
    prediction_id: str
    action: str
    status: str
    observation: Optional[int]
    observed_t: Optional[int]
    channels: List[ChannelCheck]
    n_channels: int
    alpha: float
    level_reachable: bool
    combined_score: Optional[float]
    p_value: Optional[float]
    consistent: Optional[bool]
    recovery_predicted: float
    recovery_interval_low: Optional[float]
    recovery_interval_high: Optional[float]
    recovery_posterior: Optional[float]
    predicted_combination: Optional[Dict[str, int]]  # the most probable combination after the action, which the verdict scores
    predicted_mass: Optional[float]
    recovery_status: str
