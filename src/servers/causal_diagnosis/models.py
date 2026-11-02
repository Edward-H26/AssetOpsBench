"""Result models of the diagnostics tools (docs/hydraulic_rig/causal-diagnosis-server.md). Every tool result carries the
evidence identifiers it rests on (obs/{n}/{field}, wo/{id}, pred/{n}), its time window in cycles, its assumptions as plain
statements that trace every threshold to a price, a duration, a downtime rate or the declared error level, one
uncertainty figure and a one line message; a failed call returns an ErrorResult."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


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
    band_low: Optional[float]
    band_high: Optional[float]
    inside_band: Optional[bool]
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
    combination: Dict[str, int]
    current_combination: Dict[str, int]
    faults: List[Fault]
    probability: float
    evidence_ids: List[str]


class ChannelShare(BaseModel):
    field: str
    anomaly_score: float
    share: float


class ComponentAttribution(BaseModel):
    component: str
    fault_probability: float
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


class RootCauses(ToolResult):
    ranking_model: str
    n_candidates: int
    candidates: List[Candidate]
    component_fault_probability: Dict[str, float]
    current_fault_probability: Dict[str, float]
    level_probability: Dict[str, Dict[str, float]]
    attribution: List[ComponentAttribution]
    stability: Optional[Stability]
    insufficient_evidence: bool
    separating_test: Optional[SeparatingTest]
    expected_saving: float
    expected_loss_act: float
    expected_loss_test: Dict[str, float]
    unseparated: List[str]
    separating_channels: Dict[str, List[str]]
    monte_carlo: MonteCarlo
    costs: Dict[str, Any]


class PredictedSummary(BaseModel):
    field: str
    mean: float
    low: float
    high: float
    quantile: float


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
    risk_before: float
    risk_after: float
    direction: str
    dead_band: Optional[float]


class ChannelCheck(BaseModel):
    field: str
    predicted: float
    observed: float
    low: float
    high: float
    score: float
    p_value: float
    bonferroni: bool
    holm: bool


class OutcomeVerification(ToolResult):
    prediction_id: str
    action: str
    status: str
    observation: Optional[int]
    observed_t: Optional[int]
    channels: List[ChannelCheck]
    n_channels: int
    alpha: float
    consistent: Optional[bool]
    bonferroni_rejections: int
    holm_rejections: int
    recovery_predicted: float
    recovery_interval_low: Optional[float]
    recovery_interval_high: Optional[float]
    recovery_posterior: Optional[float]
    recovery_status: str
