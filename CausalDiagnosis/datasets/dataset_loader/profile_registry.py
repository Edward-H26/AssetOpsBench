from CausalDiagnosis.utils.paths import DATASETS
import json
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from CausalDiagnosis.datasets.dataset_loader import dataset_packages as ports
from . import profile_data


ASSETS_DIR = DATASETS
# ----- loading and the registry -----
_REGISTERED = {}


@contextmanager
def _served(profile):
    """Serve a profile under its asset id for the length of a with block, in place of whatever the registry held, which
    returns after it. The engine configuration checks its sensor groups against the channels of the profile its asset id
    names, so the configuration a profile builds is checked against that profile before anyone registers it."""
    previous = _REGISTERED.get(profile.asset_id)
    _REGISTERED[profile.asset_id] = profile
    try:
        yield profile
    finally:
        if previous is None:
            _REGISTERED.pop(profile.asset_id, None)
        else:
            _REGISTERED[profile.asset_id] = previous


def load_profile(path, engine=True):
    """The validated profile of an asset file, without the check of the engine configuration when engine is False."""
    path = Path(path)
    return profile_data.AssetProfile.from_json(json.loads(path.read_text(encoding="utf-8")), str(path), engine)


@lru_cache(maxsize=None)
def _bundled(asset_id, engine):
    """The profile of the asset file of an asset folder, read after the import of the asset's package, which registers what
    the asset brings, such as its evidence kind."""
    folder = ports.asset_dirs().get(asset_id)
    if folder is None:
        raise KeyError(f"no asset profile {asset_id!r} is registered or bundled under {ASSETS_DIR}")
    ports.package(asset_id)
    return load_profile(folder / ports.ASSET_FILE, engine)


def register_profile(profile):
    """Serve a profile under its asset id, in place of a bundled asset file of the same id, as a test or a draft does."""
    _REGISTERED[profile.asset_id] = profile
    return profile


def unregister_profile(asset_id):
    _REGISTERED.pop(asset_id, None)


def get_profile(asset_id, engine=True):
    """The profile of an asset id: a registered one, or the asset file CausalDiagnosis/datasets/<asset_id>/asset.json, validated
    without the check of the engine configuration when engine is False."""
    if not isinstance(asset_id, str) or not profile_validation.ASSET_ID.fullmatch(asset_id):
        raise KeyError(f"asset id {asset_id!r} must be letters, digits and underscores")
    if asset_id in _REGISTERED:
        return _REGISTERED[asset_id]
    return _bundled(asset_id, engine)


# Imported after the definitions: this module uses them only when a function runs, and they may load
# this module first (the order of the package is in __init__.py).
from . import profile_validation
