"""Point AssetOpsBench's iot, fmsr and wo modules at the databases of one episode.

Each episode has its own five databases in the job's CouchDB or in the memory store, named by a prefix the driver derives
from the episode folder, because the iot server lists every site of its asset database and reads telemetry by asset id,
so episodes that shared databases would see each other's rows. The handles are set on the modules after import, since
iot opens its handles at import and caches sites and sensor lists per process.
"""
import hashlib
from contextlib import contextmanager
from dataclasses import astuple, dataclass
from pathlib import Path

import servers.fmsr.main as fmsr_server
import servers.iot.main as iot_server
import servers.wo.main as wo_server


@dataclass(frozen=True)
class DatabaseNames:
    asset: str
    iot: str
    failure_mode: str
    workorder: str
    failure_code: str

    def all(self):
        return astuple(self)


def episode_prefix(episode_folder):
    """A CouchDB database name must start with a lower case letter, so the prefix is e and twelve hex digits of the
    hash of the episode folder's absolute path."""
    return "e" + hashlib.sha256(str(Path(episode_folder).resolve()).encode()).hexdigest()[:12]


def database_names(prefix):
    return DatabaseNames(*(f"{prefix}_{collection}" for collection in ("asset", "iot", "failure_mode", "workorder", "failure_code")))


_BOUND = ((iot_server, "asset_db"), (iot_server, "iot_db"), (iot_server, "_registry_sites_cache"), (iot_server, "_sensor_list_cache"),
          (fmsr_server, "fm_db"), (wo_server, "_db"), (wo_server, "_fcc_db"))


def bind(server, names):
    """Set the handles of the three servers to the episode's databases and empty iot's per-process caches."""
    iot_server.asset_db = server.database(names.asset)
    iot_server.iot_db = server.database(names.iot)
    iot_server._registry_sites_cache = None
    iot_server._sensor_list_cache = {}
    fmsr_server.fm_db = server.database(names.failure_mode)
    wo_server._db = server.couch_client(names.workorder)
    wo_server._fcc_db = server.couch_client(names.failure_code)


@contextmanager
def bound(server, names):
    """bind for the length of a block, then the handles the modules had before, as tests and the probe need."""
    saved = [(module, attribute, getattr(module, attribute)) for module, attribute in _BOUND]
    bind(server, names)
    try:
        yield
    finally:
        for module, attribute, value in saved:
            setattr(module, attribute, value)
