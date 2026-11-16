"""The asset packages (CausalDiagnosis/docs/adding-a-dataset.md).

An asset is a folder under CausalDiagnosis/datasets that holds its asset file, asset.json, and its package, __init__.py. The engine
finds an asset by its folder, imports its package and then validates its asset file, so the package registers what the
asset brings beyond data, such as an evidence kind or a rule kind, before the file is read. A package may declare two names
that the engine reads: LABELS, the "module:function" reference of the label of a start, and TABLES, a tuple of
"module:ATTRIBUTE" references, one per path of a table that code of the asset computes on first use. A reference is
resolved when it is first used, so the package imports from CausalDiagnosis.datasets alone. The folders are found on the search path of
the package CausalDiagnosis.datasets, so a test mounts a made-up asset by adding its folder to that path. This module reads the standard
library alone, so that a test repository can hold a copy of it.
"""
import importlib
import os
from pathlib import Path

ASSET_FILE = "asset.json"
PACKAGE = "CausalDiagnosis.datasets"
_FOUND = {}  # the folders of the search path of the package -> its asset folders, searched once per search path


def _search_path():
    """The folders of the package CausalDiagnosis.datasets, in the order that the import system searches them."""
    return tuple(importlib.import_module(PACKAGE).__path__)


def asset_dirs():
    """{asset_id: folder} of every folder with an asset file under a folder of the package, in search order, the first
    folder of an id winning as the import of its package does. The search runs once per search path, so a folder that
    joins the path, as a mounted port does, is found by a new search."""
    search = _search_path()
    if search not in _FOUND:
        found = {}
        for entry in search:
            for folder in sorted(Path(entry).iterdir()):
                if folder.name not in found and (folder / ASSET_FILE).is_file():
                    found[folder.name] = folder
        _FOUND[search] = found
    return dict(_FOUND[search])


def asset_ids():
    return sorted(asset_dirs())


def package(asset_id):
    """The imported package CausalDiagnosis.datasets.<asset_id>, or None for an id without a folder."""
    if asset_id not in asset_dirs():
        return None
    return importlib.import_module(f"{PACKAGE}.{asset_id}")


def declared(asset_id, name, default=None):
    """An attribute of the package of the asset, or default when the asset has no folder or its package no such name."""
    found = package(asset_id)
    return default if found is None else getattr(found, name, default)


def resolve(reference, owner="a reference"):
    """The object that a "module:attribute" reference names, its module imported on first use. A reference that names no
    such object raises LookupError, whose message opens with owner, such as an asset and the name of its declaration."""
    module, _, attribute = reference.partition(":") if isinstance(reference, str) else ("", "", "")
    try:
        return getattr(importlib.import_module(module), attribute)
    except (ImportError, AttributeError, ValueError) as e:
        raise LookupError(f"{owner} names {reference!r}, which is no attribute of an importable module ({e})") from None


def start_label(env, rules=None):
    """The label of the start of the episode of env, from the function that LABELS of its asset names, called as
    f(env, rules), or None when the asset declares none, so that every start of it counts as separable. The label has the
    form the verifier reads: a boolean separable and, under e18, a second verdict with a boolean separable of its own."""
    asset_id = env.config.asset_id
    reference = declared(asset_id, "LABELS")
    if reference is None:
        return None
    return resolve(reference, f"{asset_id}: LABELS")(env, rules)


def generated_tables():
    """[(module, attribute)] of every reference under TABLES of every asset package, in the order of the asset ids. Each
    reference must name a path, which a test session then points at a copy."""
    found = []
    for asset_id in asset_ids():
        for reference in declared(asset_id, "TABLES", ()):
            if not isinstance(resolve(reference, f"{asset_id}: TABLES"), (str, os.PathLike)):
                raise LookupError(f"{asset_id}: TABLES names {reference!r}, which is no path")
            module, _, attribute = reference.partition(":")
            found.append((module, attribute))
    return found
