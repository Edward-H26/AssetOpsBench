"""Result models of the causal-diagnosis tools (docs/hydraulic_rig/causal-diagnosis-server.md). Every tool result carries the
evidence identifiers it rests on (obs/{n}/{field}, fmea/{source}/{code}, assoc/{source}/{id}, pred/{n}), its time window in
accepted measurements of the episode (static for knowledge), its assumptions as plain statements, one uncertainty figure
and a one-line message; a failed call returns an ErrorResult."""
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


class ComponentContext(BaseModel):
    name: str
    levels: List[int]
    healthy: int
    n_fault_codes: int


class ChannelContext(BaseModel):
    channel: str
    field: str
    unit: str
    rate: str
    group: str
    statistics: List[str]
    indicates: List[str]


class AssetContext(ToolResult):
    asset_id: str
    asset_type: Optional[str]
    description: Optional[str]
    site: Optional[str]
    components: List[ComponentContext]
    channels: List[ChannelContext]
    n_failure_codes: int
    n_fault_codes: int


class SymptomReport(ToolResult):
    asset_id: str
    task_text: str
    report_time: int
    fault_count_min: int
    fault_count_max: int
    n_candidates: int
    budget: float
    step_cap: int
    repairs: int
    prices: Dict[str, Any]
    status: Dict[str, Any]


class EvidenceItem(BaseModel):
    evidence_id: str
    field: str
    channel: str
    statistic: str
    value: float
    unit: str
    rate: str
    release_time: int
    repeated: bool
    band_low: float
    band_high: float
    inside_band: bool
    band_cycles: int
    band_healthy: List[str]


class EpisodeEvidence(ToolResult):
    sensor_group: str
    n_measurements: int
    items: List[EvidenceItem]


class IndicatingChannel(BaseModel):
    channel: str
    field: str
    correlation: Optional[float]
    group: str


class FmeaPath(BaseModel):
    component: str
    level: Optional[int]
    code: str
    severity: str
    recommended_action: str
    channels: List[IndicatingChannel]
    held_out_accuracy: Dict[str, float]
    evidence_ids: List[str]


class FmeaPaths(ToolResult):
    asset_id: str
    node: str
    node_kind: str
    paths: List[FmeaPath]


class Fault(BaseModel):
    component: str
    level: int
    code: Optional[str]


class Candidate(BaseModel):
    rank: int
    combination: Dict[str, int]
    faults: List[Fault]
    probability: float
    evidence_ids: List[str]


class RootCauses(ToolResult):
    n_candidates: int
    candidates: List[Candidate]
    component_fault_probability: Dict[str, float]
    level_probability: Dict[str, Dict[str, float]]
    insufficient_evidence: bool
    unseparated: List[str]
    separating_channels: Dict[str, List[str]]


class PredictedSummary(BaseModel):
    field: str
    mean: float
    band_low: float
    band_high: float


class InterventionEffect(ToolResult):
    prediction_id: str
    action: str
    component: Optional[str]
    addresses_fault: float
    recovery: float
    price: float
    predicted: List[PredictedSummary]


class Residual(BaseModel):
    field: str
    predicted: float
    observed: float
    residual: float
    inside_band: bool


class OutcomeVerification(ToolResult):
    prediction_id: str
    action: str
    status: str
    observation: Optional[int]
    residuals: List[Residual]
    n_inside: int
    n_outside: int
    n_outside_tolerated: Optional[int]
    p_value_outside: Optional[float]
    recovery_predicted: float
    recovered: Optional[bool]
    recovery_status: str
    consistent: Optional[bool]
