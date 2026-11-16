"""A made-up port of the asset layer tests: one stamping press of one component and two levels, of an evidence kind of its
own that records a history before the start, plays no live record and replays no repair. Its names share nothing with
hyd_1 or the tiny asset. The tests mount it from its folder alone (the fixture mounted_port of tests/conftest.py)."""
from CausalDiagnosis.datasets.dataset_loader import evidence_kinds as kinds

SOURCE = "CausalDiagnosis.datasets.demo_1.source"
KIND = kinds.register_kind(kinds.EvidenceKind("demo_history", replays_repairs=False, record_keys=("level_key",),
                                              check=f"{SOURCE}:record_problems", loader=f"{SOURCE}:load", source=f"{SOURCE}:HistorySource",
                                              truth=f"{SOURCE}:MarkedTruth"))
LABELS = "CausalDiagnosis.datasets.demo_1.labels:label_of"
TABLES = ("CausalDiagnosis.datasets.demo_1.labels:TABLE_PATH",)
