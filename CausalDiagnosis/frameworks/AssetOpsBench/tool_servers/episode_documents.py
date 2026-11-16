"""The engine as the producer of one episode's AssetOpsBench documents.

At reset the producer creates the episode's five databases as AssetOpsBench's loader would, with its design document
and Mango indexes but explicit names, and writes the registry, the failure mode documents, the failure code catalog as
it ships, the prehistory records and, where the fix history card is on, the past orders of the engine's fix history,
which wo's own create and close code writes closed at their closing cycles under the wonum hist-<n>. After every step it writes the difference between the engine's public state and
what is written: one record for each cycle the engine recorded since, and the status changes of the work orders through
wo's own lifecycle code at the cycle's time (APPR with the schedule at creation, INPRG at execution, COMP at the
close). It reads its own documents back only after the store rejected an insert, to tell a record that a stopped call
wrote already from one that contradicts the engine, so the database follows the engine and never the other way. It never
writes the stock hydraulic records or the stock failure mode sample, and nothing hidden.

The documents come from a provider with registry(), failure_modes() and telemetry(cycle, values), which
CausalDiagnosis/frameworks/AssetOpsBench/tool_servers/profile_documents.py implements for any asset profile; the tests use a hyd_1 fixture.
"""
import copy
import json
from pathlib import Path

from couchdb import loader
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_tool_contract import WORK_TYPE_MODES, child_asset, history_wonum
from CausalDiagnosis.utils.atomic_writes_and_locks import write_json_atomically
from servers.wo import workorders

COLLECTIONS = ("asset", "iot", "failure_mode", "workorder", "failure_code")
COUCHDB_DIR = Path(loader.__file__).resolve().parent
FAILURE_CODES = COUCHDB_DIR / "scenarios_data" / "shared" / "failure_code" / "failure_code_sample.csv"
EXPORT_NAME = "aob_documents.json"
WORK_TYPES = {mode: work_type for work_type, mode in WORK_TYPE_MODES.items()}  # the engine's order mode -> wo's work type


def normalise(collection, doc):
    """The loader's own normalisation, so a document has the _id and the dataset field load_collection gives it."""
    return loader._normalise(dict(doc), collection, loader.collection_config(collection), loader._transform_for(collection))


class Producer:
    def __init__(self, server, names, documents, timeline, asset_id, site):
        self.server = server
        self.names = names
        self.documents = documents
        self.timeline = timeline
        self.asset_id = asset_id
        self.site = site
        self._written = set()
        self._events = 0

    def reset(self, engine):
        """Create the databases and write everything the agent may read before its first call."""
        documents = {"asset": self.documents.registry(), "iot": [self._record(engine, c) for c in engine.recorded_cycles()],
                     "failure_mode": self.documents.failure_modes(), "workorder": self._past_orders(engine),
                     "failure_code": loader.parse_csv(FAILURE_CODES, loader.collection_config("failure_code"))}
        for collection, name in zip(COLLECTIONS, self.names.all()):
            config = loader.collection_config(collection)
            self.server.create_database(name, drop=True)
            if config.get("design_doc"):
                self.server.install_design(name, json.loads((COUCHDB_DIR / config["design_doc"]).read_text()))
            self.server.bulk_insert(name, [normalise(collection, d) for d in documents[collection]])
            self.server.create_indexes(name, config.get("indexes"))
        self.resume(engine)

    def resume(self, engine):
        """Take up an episode whose documents earlier calls wrote, possibly in other processes: every cycle the replayed
        engine has recorded and every public event it has logged is written already, since each call syncs before it
        returns, so the counters follow the engine and the database is read back only when the store rejects an insert (sync)."""
        self._written = set(engine.recorded_cycles())
        self._events = len(engine.public_events())

    async def sync(self, engine):
        """Write the records of the cycles recorded since the last sync, then the order events in log order. An event of
        an agent order waits until the metering layer has created the order's document through wo. A call that stopped
        after its records reached the store and before the state file of the episode was written leaves the next call to
        replay the older state and write those records again, so an insert that the store rejects goes record by record
        (_insert_unless_stored)."""
        new = [c for c in engine.recorded_cycles() if c not in self._written]
        if new:
            records = [self._record(engine, c) for c in new]
            try:
                self.server.bulk_insert(self.names.iot, records)
            except ValueError:
                await self._insert_unless_stored(self.names.iot, records)
            self._written.update(new)
        events = engine.public_events()
        client = self.server.couch_client(self.names.workorder)
        while self._events < len(events) and await self._apply(client, events[self._events]):
            self._events += 1

    def export(self, folder):
        """Every document of the episode without _rev, by collection, for the determinism check and the walkthrough."""
        path = Path(folder) / EXPORT_NAME
        write_json_atomically(path, {c: self.server.all_docs(n) for c, n in zip(COLLECTIONS, self.names.all())}, indent=1)
        return path

    def drop(self):
        for name in self.names.all():
            self.server.drop_database(name)

    # ----- parts -----
    async def _insert_unless_stored(self, name, records):
        """After a rejected insert, read every record back by its identifier: one stored with an equal body is skipped, one
        that is missing is inserted, and one stored with another body is refused, since a write that repeats a stored
        document is no error and a write that contradicts one is."""
        client = self.server.couch_client(name)
        missing = []
        for record in records:
            stored = await client.get(record["_id"])
            if stored is None:
                missing.append(record)
            elif {k: v for k, v in stored.items() if k != "_rev"} != record:
                raise ValueError(f"the store holds {record['_id']} with another body than the record of the engine")
        if missing:
            self.server.bulk_insert(name, missing)

    def _record(self, engine, cycle):
        return normalise("iot", self.documents.telemetry(cycle, engine.record_values(cycle)))

    def _past_orders(self, engine):
        """The past orders of the engine's fix history as wo's own code writes them: created and closed at the closing
        cycle, the only time the history gives, on the component's child asset."""
        database = _Documents()
        for entry in engine.fix_history:
            wonum, closed = history_wonum(entry["id"]), self.timeline.utc(entry["closed_t"])
            _completed(workorders.create_workorder(database, description=f"repair of the {entry['component']}",
                                                   asset_num=child_asset(self.asset_id, entry["component"]), site_id=self.site,
                                                   work_type=WORK_TYPES[entry["mode"]], wonum=wonum, now=closed))
            _completed(workorders.close_workorder(database, wonum, self.site, now=closed))
        return list(database.docs.values())

    async def _apply(self, client, event):
        kind, t = event["event"], event["t_end"]
        now = self.timeline.utc(t)
        if kind not in ("work_order_created", "work_order_executed", "work_order_closed"):
            return True
        if await client.get(workorders._doc_id(self.site, event["id"])) is None:
            return False
        if kind == "work_order_created":
            # wo's own status change, the one approve_workorder makes, with the public schedule of the order
            schedule = {"schedstart": workorders._iso(now), "schedfinish": workorders._iso(self.timeline.utc(event["expected_close_t"]))}
            await workorders._change_status(client, event["id"], self.site, "APPR", extra=schedule, now=now)
        elif kind == "work_order_executed":
            await workorders._change_status(client, event["id"], self.site, "INPRG", extra={"actstart": workorders._iso(now)}, now=now)
        else:
            await workorders.close_workorder(client, event["id"], self.site, now=now)
        return True


class _Documents:
    """A work order database that keeps what wo writes, so that wo's own code builds the past orders at reset, where no
    event loop runs."""

    def __init__(self):
        self.docs = {}

    async def get(self, doc_id):
        return copy.deepcopy(self.docs.get(doc_id))

    async def put(self, doc):
        self.docs[doc["_id"]] = copy.deepcopy(doc)


def _completed(coroutine):
    """The result of a call on _Documents, whose coroutine finishes without ever suspending."""
    try:
        coroutine.send(None)
    except StopIteration as stop:
        return stop.value
    coroutine.close()
    raise RuntimeError("a wo call on the collected documents suspended")
