"""An in-memory CouchDB under AssetOpsBench's own iot, fmsr and wo tool code, the local backend beside CouchDB.

On the laptop the tools run over this store and on the cluster over CouchDB, and the conformance test requires
the same result texts and documents from both. The store therefore follows CouchDB wherever a result depends on it: a
selector on a missing field fails except `$exists: false`, an unsorted find returns the order of the Mango index that
CouchDB would choose or else document id order, a sort needs an index, an update needs the current revision, the work
order database refuses what its design document refuses, and a document comes back with `_id` and `_rev` first.
"""
import copy
import hashlib
import json
from pathlib import Path

from couchdb3.exceptions import BadRequestError, ConflictError, ForbiddenError, NotFoundError

from CausalDiagnosis.utils.atomic_writes_and_locks import write_json_atomically
from servers.wo.couch import CouchClient, CouchConflict, CouchError, DatabaseUnavailable

# CouchDB's response bodies, which couchdb3 raises as the exception text and fmsr reads for the missing database
_MISSING_DATABASE = '{"error":"not_found","reason":"Database does not exist."}\n'
_MISSING_DOCUMENT = '{"error":"not_found","reason":"missing"}\n'
_CONFLICT = '{"error":"conflict","reason":"Document update conflict."}\n'
_NO_SORT_INDEX = '{"error":"no_usable_index","reason":"No index exists for this sort, try indexing by the sort fields."}\n'
_RANGE_OPERATORS = ("$gt", "$gte", "$lt", "$lte")
# the validate_doc_update of src/couchdb/_design_workorders.json that _validate_workorder ports, by its SHA-256
_WORKORDER_VALIDATION_SHA256 = "07932379303e5ba001070aa5b8e2c1372e7bf95e44cc7c012ee1c012e18838cb"
WORKORDER_STATUSES = ("WAPPR", "APPR", "WMATL", "WSCH", "INPRG", "WPCOND", "COMP", "CLOSE", "CAN")
WORKORDER_TYPES = ("CM", "PM", "EM", "PdM", "CAL", "INSP", "GEN")
SOURCE_AGENTS = ("iot", "fmsr", "tsfm", "wo", "human")
SOURCE_TRIGGERS = ("anomaly_detection", "forecast", "failure_mode", "rule_monitoring", "inspection", "manual")


class _Conflict(Exception):
    pass


class _Forbidden(Exception):
    pass


class _Database:
    """One database: stored documents by id, its Mango indexes in creation order and its validation, if any."""

    def __init__(self):
        self.docs = {}
        self.indexes = []
        self.validate = None


class MemoryServer:
    """The databases of one store, with the administration the loader's helpers perform on CouchDB."""

    def __init__(self):
        self._databases = {}

    def database_names(self):
        return sorted(self._databases)

    def create_database(self, name, drop=False):
        if name in self._databases and not drop:
            return
        self._databases[name] = _Database()

    def drop_database(self, name):
        self._databases.pop(name, None)

    def install_design(self, name, design):
        """Store a design document's validation. Only the work order design is known, and a changed version of it
        must be ported again before the store accepts it."""
        source = design.get("validate_doc_update")
        if source is None:
            return
        if hashlib.sha256(source.encode()).hexdigest() != _WORKORDER_VALIDATION_SHA256:
            raise NotImplementedError("this validate_doc_update is not ported to the memory store")
        self._require(name).validate = _validate_workorder

    def create_indexes(self, name, indexes):
        for columns in indexes or []:
            if list(columns) not in self._require(name).indexes:
                self._require(name).indexes.append(list(columns))

    def bulk_insert(self, name, docs):
        """Insert new documents, failing on the first that CouchDB would reject."""
        database = self._require(name)
        for doc in docs:
            try:
                _write(database, doc)
            except (_Conflict, _Forbidden) as exc:
                raise ValueError(f"bulk insert rejected {doc.get('_id')}: {exc}") from exc

    def all_docs(self, name):
        """Every stored document in id order without `_rev`, which CouchDB computes, for export and comparison."""
        database = self._require(name)
        return [{k: v for k, v in copy.deepcopy(database.docs[i]).items() if k != "_rev"} for i in sorted(database.docs)]

    def dump(self, path):
        """Write every database, revisions, indexes and validation included, to a JSON file. A store on the laptop lives in
        one process, while Plan-Execute starts a server process per call and Agent-As-Tool one per server, so each call
        loads the store under the state file's lock and dumps it before the lock is released, as CouchDB would keep it."""
        write_json_atomically(path, {name: {"docs": list(db.docs.values()), "indexes": db.indexes, "validated": db.validate is not None}
                                     for name, db in self._databases.items()})

    @classmethod
    def load(cls, path):
        """The store that dump wrote."""
        server = cls()
        for name, data in json.loads(Path(path).read_text(encoding="utf-8")).items():
            database = server._databases[name] = _Database()
            database.docs = {doc["_id"]: doc for doc in data["docs"]}
            database.indexes = [list(columns) for columns in data["indexes"]]
            database.validate = _validate_workorder if data["validated"] else None
        return server

    def database(self, name):
        """The handle iot and fmsr call, in place of a couchdb3.Database."""
        return MemoryDatabase(self, name)

    def couch_client(self, name):
        """The handle wo calls, in place of its CouchClient."""
        return MemoryCouchClient(self, name)

    def _get(self, name):
        return self._databases.get(name)

    def _require(self, name):
        if name not in self._databases:
            raise KeyError(f"no database {name} in the memory store")
        return self._databases[name]


class MemoryDatabase:
    """The couchdb3.Database calls of the iot and fmsr servers: check, find, get and save."""

    def __init__(self, server, name):
        self._server = server
        self.name = name

    def check(self, resource=None):
        return self._server._get(self.name) is not None

    def find(self, selector, limit=25, skip=0, sort=None, fields=None, bookmark=None, **_options):
        database = self._server._get(self.name)
        if database is None:
            raise NotFoundError(_MISSING_DATABASE)
        start = skip + (_bookmark_offset(bookmark) if bookmark is not None else 0)
        try:
            docs = _find(database, selector, fields=fields, sort=sort, limit=limit, skip=start)
        except _NoSortIndex as exc:
            raise BadRequestError(_NO_SORT_INDEX) from exc
        return {"docs": docs, "bookmark": f"memory-{start + len(docs)}"}

    def get(self, docid, *, check=False, default_value=None, **_options):
        database = self._server._get(self.name)
        if database is not None and docid in database.docs:
            return copy.deepcopy(database.docs[docid])
        if check:
            raise NotFoundError(_MISSING_DATABASE if database is None else _MISSING_DOCUMENT)
        return default_value

    def save(self, doc, **_options):
        database = self._server._get(self.name)
        if database is None:
            raise NotFoundError(_MISSING_DATABASE)
        try:
            rev = _write(database, doc)
        except _Conflict as exc:
            raise ConflictError(_CONFLICT) from exc
        except _Forbidden as exc:
            raise ForbiddenError(json.dumps({"error": "forbidden", "reason": str(exc)}, separators=(",", ":")) + "\n") from exc
        return doc["_id"], True, rev


class MemoryCouchClient:
    """The CouchClient calls of the wo server, with the errors CouchClient raises for the same CouchDB answers."""

    next_wonum = CouchClient.next_wonum  # AssetOpsBench's own counter logic over this client's get and put

    def __init__(self, server, db):
        self._server = server
        self.db = db

    async def aclose(self):
        return None

    def _database(self):
        database = self._server._get(self.db)
        if database is None:
            raise DatabaseUnavailable()
        return database

    async def get(self, doc_id):
        doc = self._database().docs.get(doc_id)
        return copy.deepcopy(doc) if doc is not None else None

    async def put(self, doc):
        if "_id" not in doc:
            raise CouchError("document must have _id")
        try:
            rev = _write(self._database(), doc)
        except _Conflict as exc:
            raise CouchConflict(f"conflict updating {doc['_id']} (stale _rev)") from exc
        except _Forbidden as exc:
            raise CouchError("database request failed (HTTP 403)") from exc
        return {"ok": True, "id": doc["_id"], "rev": rev}

    async def delete(self, doc_id, rev):
        database = self._database()
        current = database.docs.get(doc_id)
        if current is None:
            raise CouchError("database request failed (HTTP 404)")
        if current["_rev"] != rev:
            raise CouchError("database request failed (HTTP 409)")
        del database.docs[doc_id]
        return {"ok": True, "id": doc_id, "rev": rev}

    async def find(self, selector, *, fields=None, sort=None, limit=200, skip=0):
        try:
            return _find(self._database(), selector, fields=fields or None, sort=sort or None, limit=limit, skip=skip)
        except _NoSortIndex as exc:
            raise CouchError("database request failed (HTTP 400)") from exc


# ----- documents and revisions -----

def _write(database, doc):
    """Store a document as a PUT does: a new document needs no revision, an existing one its current revision."""
    doc_id = doc["_id"]
    current = database.docs.get(doc_id)
    if doc.get("_deleted") is True:
        if current is None or doc.get("_rev") != current["_rev"]:
            raise _Conflict(doc_id)
        del database.docs[doc_id]
        return None
    if (current is None and doc.get("_rev") is not None) or (current is not None and doc.get("_rev") != current["_rev"]):
        raise _Conflict(doc_id)
    body = json.loads(json.dumps({k: v for k, v in doc.items() if k not in ("_id", "_rev")}))
    if database.validate is not None:
        database.validate({"_id": doc_id, **body})
    generation = int(current["_rev"].split("-", 1)[0]) + 1 if current is not None else 1
    rev = f"{generation}-{hashlib.md5(json.dumps(body, sort_keys=True).encode()).hexdigest()}"
    database.docs[doc_id] = {"_id": doc_id, "_rev": rev, **body}
    return rev


def _validate_workorder(doc):
    """validate_doc_update of src/couchdb/_design_workorders.json in Python, checked against its SHA-256 at install."""
    def refuse(field, message):
        raise _Forbidden(f"WO validation failed [{field}]: {message}")

    def present(value):
        return value is not None and value != ""

    if doc["_id"].startswith("_design/"):
        return
    if doc.get("type") != "workorder":
        refuse("type", "must equal 'workorder'")
    if not (isinstance(doc.get("wonum"), str) and doc["wonum"]):
        refuse("wonum", "is required and must be a non-empty string")
    if present(doc.get("description")) and len(str(doc["description"])) > 100:
        refuse("description", "max length is 100 chars")
    if present(doc.get("status")) and doc["status"] not in WORKORDER_STATUSES:
        refuse("status", f"invalid status '{doc['status']}'")
    if present(doc.get("worktype")) and doc["worktype"] not in WORKORDER_TYPES:
        refuse("worktype", f"invalid worktype '{doc['worktype']}'")
    if present(doc.get("wopriority")):
        priority = doc["wopriority"]
        if isinstance(priority, bool) or not isinstance(priority, (int, float)) or not 1 <= priority <= 5 or priority % 1 != 0:
            refuse("wopriority", "must be an integer 1..5")
    source = doc.get("aob_source")
    if isinstance(source, dict):
        if present(source.get("agent")) and source["agent"] not in SOURCE_AGENTS:
            refuse("aob_source.agent", "invalid agent")
        if present(source.get("trigger_type")) and source["trigger_type"] not in SOURCE_TRIGGERS:
            refuse("aob_source.trigger_type", "invalid trigger_type")


# ----- Mango queries -----

class _NoSortIndex(Exception):
    pass


def _find(database, selector, *, fields=None, sort=None, limit=25, skip=0):
    rows = [database.docs[i] for i in sorted(database.docs) if _matches(database.docs[i], selector)]
    columns, descending = _order(database, selector, sort)
    if columns:
        rows.sort(key=lambda doc: tuple(_collation_key(doc.get(c)) for c in columns))
    if descending:
        rows.reverse()
    return [_project(doc, fields) for doc in rows[skip:skip + limit]]


def _order(database, selector, sort):
    """The index columns that order the result, as CouchDB's planner picks the index, and whether the scan runs down.
    Without a sort, a JSON index is usable when the selector requires all its fields, the longest prefix of fields with
    an equality or a range wins, then the shorter index, then the older one, and with none usable the order is by id."""
    if sort:
        fields = [next(iter(s)) if isinstance(s, dict) else s for s in sort]
        directions = {next(iter(s.values())) if isinstance(s, dict) else "asc" for s in sort}
        if len(directions) != 1:
            raise _NoSortIndex(sort)
        for columns in database.indexes:
            if _serves_sort(columns, fields, selector):
                return columns, directions == {"desc"}
        raise _NoSortIndex(sort)
    required = {f for f, c in selector.items() if c != {"$exists": False}}
    usable = [columns for columns in database.indexes if set(columns) <= required]
    if not usable:
        return None, False
    return min(usable, key=lambda columns: (-_prefix(columns, selector), len(columns), database.indexes.index(columns))), False


def _serves_sort(columns, fields, selector):
    """An index serves a sort when the sort fields follow its leading equality fields in order and the selector
    requires every other column."""
    if fields[0] not in columns:
        return False
    start = columns.index(fields[0])
    if columns[start:start + len(fields)] != fields:
        return False
    if not all(_is_equality(selector.get(c, _ABSENT)) for c in columns[:start]):
        return False
    required = {f for f, c in selector.items() if c != {"$exists": False}}
    return set(columns) - set(fields) <= required


_ABSENT = object()


def _is_operator_dict(condition):
    return isinstance(condition, dict) and bool(condition) and all(k.startswith("$") for k in condition)


def _is_equality(condition):
    if condition is _ABSENT:
        return False
    return not _is_operator_dict(condition) or set(condition) == {"$eq"}


def _prefix(columns, selector):
    count = 0
    for column in columns:
        condition = selector.get(column, _ABSENT)
        if _is_equality(condition):
            count += 1
        elif _is_operator_dict(condition) and set(condition) <= set(_RANGE_OPERATORS):
            return count + 1
        else:
            return count
    return count


def _matches(doc, selector):
    for field, condition in selector.items():
        if field.startswith("$"):
            raise NotImplementedError(f"the memory store has no {field}")
        if field not in doc:
            if condition == {"$exists": False}:
                continue
            return False
        if not _satisfies(doc[field], condition):
            return False
    return True


def _satisfies(value, condition):
    if not _is_operator_dict(condition):
        return _collation_key(value) == _collation_key(condition)
    key = _collation_key(value)
    for operator, argument in condition.items():
        if operator == "$exists":
            ok = bool(argument)
        elif operator == "$eq":
            ok = key == _collation_key(argument)
        elif operator == "$ne":
            ok = key != _collation_key(argument)
        elif operator == "$gt":
            ok = key > _collation_key(argument)
        elif operator == "$gte":
            ok = key >= _collation_key(argument)
        elif operator == "$lt":
            ok = key < _collation_key(argument)
        elif operator == "$lte":
            ok = key <= _collation_key(argument)
        elif operator in ("$in", "$nin"):
            values = value if isinstance(value, list) else [value]
            found = any(_collation_key(v) == _collation_key(a) for a in argument for v in values)
            ok = found if operator == "$in" else not found
        elif operator == "$elemMatch":
            ok = isinstance(value, list) and any(_element_matches(element, argument) for element in value)
        else:
            raise NotImplementedError(f"the memory store has no {operator}")
        if not ok:
            return False
    return True


def _element_matches(element, selector):
    if _is_operator_dict(selector):
        return _satisfies(element, selector)
    return isinstance(element, dict) and _matches(element, selector)


def _collation_key(value):
    """CouchDB's order across types, null, false, true, numbers, strings, arrays, objects. Strings compare by code
    point here, while CouchDB uses ICU collation, which agrees on the ASCII ids, dates and codes these databases hold."""
    if value is None:
        return (0,)
    if value is False:
        return (1,)
    if value is True:
        return (2,)
    if isinstance(value, (int, float)):
        return (3, value)
    if isinstance(value, str):
        return (4, value)
    if isinstance(value, list):
        return (5, tuple(_collation_key(v) for v in value))
    return (6, tuple((k, _collation_key(v)) for k, v in value.items()))


def _project(doc, fields):
    if fields is None:
        return copy.deepcopy(doc)
    return {f: copy.deepcopy(doc[f]) for f in fields if f in doc}


def _bookmark_offset(bookmark):
    if not (isinstance(bookmark, str) and bookmark.startswith("memory-") and bookmark[7:].isdigit()):
        raise BadRequestError('{"error":"invalid_bookmark","reason":"Invalid bookmark value"}\n')
    return int(bookmark[7:])
