"""The view of the diagnostics server comes from the asset profile of CausalDiagnosis/datasets: every table
of the hyd_1 view is the profile's, the defaults of the processes block follow the profile's rule cards, and released
tables that are not the profile's own are refused."""
import dataclasses

import pytest

from CausalDiagnosis.datasets.dataset_loader.profile_data import PROCESS_FIELDS, processes_block
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile, register_profile, unregister_profile
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import profile_view


def test_the_view_of_hyd_1_holds_the_tables_of_its_asset_profile():
    profile, view = get_profile("hyd_1"), profile_view()
    assert view.asset_id == profile.asset_id and view.components == tuple(profile.components)
    assert view.levels == {c: tuple(profile.levels[c]) for c in profile.components} and view.healthy == dict(profile.healthy)
    assert view.degradation == {c: tuple(profile.degradation[c]) for c in profile.components}
    assert view.combinations == profile.combinations()
    assert view.sensor_groups == {g: tuple(profile.field_name(s) for s in spec["channels"]) for g, spec in profile.observation_kinds.items()}
    assert list(view.processes) == list(PROCESS_FIELDS) and view.processes == processes_block(profile.default_config())
    assert set(view.processes_source.values()) == {"defaults"}


def test_the_default_processes_follow_the_rule_cards_of_the_profile():
    bundled = get_profile("hyd_1")
    interval = next(c for c in bundled.rules if c["kind"] == "schedule")["parameters"]["interval"] + 3
    cards = tuple({**c, "parameters": {**c["parameters"], "interval": interval}} if c["kind"] == "schedule" else c for c in bundled.rules)
    register_profile(dataclasses.replace(bundled, rules=cards))
    try:
        assert profile_view().processes["schedule"]["interval"] == interval
    finally:
        unregister_profile("hyd_1")
    assert profile_view().processes["schedule"]["interval"] == interval - 3


def test_released_tables_that_are_not_the_profiles_own_are_refused():
    profile = get_profile("hyd_1")
    first = profile.components[0]
    with pytest.raises(ValueError, match="levels"):
        profile_view({"levels": {**{c: list(v) for c, v in profile.levels.items()}, first: list(profile.levels[first][1:])}})
    with pytest.raises(ValueError, match="healthy"):
        profile_view({"healthy": {**dict(profile.healthy), first: profile.levels[first][0]}})


def test_every_function_of_the_asset_model_is_exported_or_called():
    """Every public function at the top of asset_view.py is named in its __all__, the list of what the module offers to
    other packages, or imported or called by name in a module of the server's package, so the module keeps no function that
    nothing uses."""
    import ast
    from pathlib import Path

    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP import asset_view as asset_model

    source = Path(asset_view.__file__)
    module = ast.parse(source.read_text(encoding="utf-8"))
    used = set()
    for top in module.body:
        used.update(node.func.id for node in ast.walk(top) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id != getattr(top, "name", None))
    for path in source.parent.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[-1] == "asset_model":
                used.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "asset_model":
                used.add(node.attr)
    functions = [node.name for node in module.body if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")]
    unused = [name for name in functions if name not in (*asset_view.__all__, *diagnostics_model.__all__) and name not in used]
    assert functions and not unused, f"asset_view.py defines public functions that nothing exports or uses: {unused}"
