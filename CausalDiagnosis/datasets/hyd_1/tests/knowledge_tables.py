"""Static knowledge of the hydraulic rig: the channel-to-component associations, the failure-mode codes and the asset
description (CausalDiagnosis/docs/environment.md). The action server serves it through get_failure_modes and
get_asset_profile, and the diagnostics server reads it for its codes and channels, both from the asset profile.

INDICATES: a channel indicates a component when it, or the virtual sensor served as CE, CP or SE, is among the component's
most correlated features in Helwig et al. 2015, Table II (p. 3), with SE taken as the table's system efficiency; the
temperatures follow the cooler (pp. 3 to 4). PS4 to PS6, FS2 and VS1 indicate no component. An association is diagnostic
evidence, not a causal edge.

FAILURE_CODES: the 16 distinct failure-mode entries of the IndustryAssetEQA hydraulic facts file, extracted once from its
records without their cycles (component, the condition level that carries the code, code, severity and
recommended action); ten of them are faults, four say a component is healthy, two describe the operating regime.

The tables live in the asset file CausalDiagnosis/datasets/hyd_1/asset.json, and this module serves them under their former names
to the tests of hyd_1. The lookups over them are the asset profile's own (CausalDiagnosis/datasets/dataset_loader/profile_data.py, CausalDiagnosis/datasets/dataset_loader/profile_registry.py and CausalDiagnosis/datasets/dataset_loader/profile_validation.py), which every reader calls.
"""
import copy

from CausalDiagnosis.datasets.hyd_1 import records as hyd_1
from CausalDiagnosis.datasets.dataset_loader.recordings import REPO
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import SENSORS
from CausalDiagnosis.datasets.hyd_1.former_field_names import HYD1_FIELDS

_PROFILE = hyd_1.tables()
ASSOCIATION_SOURCE = _PROFILE.knowledge["association_source"]  # Helwig, Pignanelli and Schuetze 2015, Table II, pp. 3 to 4
FMEA_SOURCE = _PROFILE.knowledge["fmea_source"]
STATISTICS = tuple(_PROFILE.statistics["recorded"])

INDICATES = {channel: tuple(components) for channel, components in _PROFILE.knowledge["indicates"].items()}
# Helwig et al. 2015, Table II: the correlation of each component's condition with its most correlated features, returned
# as an assumption because they belong to interval features of the fixed working cycle rather than to the served summaries.
HELWIG_CORRELATIONS = copy.deepcopy(_PROFILE.knowledge["correlations"])

FAILURE_CODES = tuple(dict(entry) for entry in _PROFILE.failure_modes)
CODES = {e["code"]: e for e in FAILURE_CODES}
CODE_OF_LEVEL = {(e["component"], e["level"]): e["code"] for e in FAILURE_CODES if e["level"] is not None}
FAULT_CODES = tuple(e for e in FAILURE_CODES if e["severity"] in ("critical", "degraded"))
PREFIX_TO_SENSOR = {HYD1_FIELDS[s].split("_", 1)[0]: s for s in SENSORS}  # P1 -> EPS1
ASSET_PROFILE = REPO / _PROFILE.text["registry_file"]
