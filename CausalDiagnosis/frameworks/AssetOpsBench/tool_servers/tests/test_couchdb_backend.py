"""The CouchDB backend's own HTTP calls, against a stand in for requests, since CouchDB runs only on the cluster."""
from types import SimpleNamespace

import pytest

from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import couchdb_backend as couch_server
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.couchdb_backend import CouchServer


class Response:
    def __init__(self, status, body):
        self.status_code, self._body = status, body
        self.ok = status < 400
        self.text = str(body)

    def json(self):
        return self._body

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(self.status_code)


@pytest.fixture
def http(monkeypatch):
    """The answers to give in order and the calls received."""
    state = SimpleNamespace(answers=[], calls=[])

    def answer(method):
        def send(url, **kwargs):
            state.calls.append((method, url, kwargs))
            return state.answers.pop(0)
        return send

    for method in ("get", "put", "post"):
        monkeypatch.setattr(couch_server.requests, method, answer(method))
    return state


def test_bulk_insert_fails_on_a_rejected_document(http):
    http.answers = [Response(201, [{"id": "a", "rev": "1-x"}, {"id": "b", "error": "conflict"}])]
    with pytest.raises(ValueError, match="conflict"):
        CouchServer("http://127.0.0.1:5984/", "admin", "password").bulk_insert("d", [{"_id": "a"}, {"_id": "b"}])
    assert http.calls[0][1] == "http://127.0.0.1:5984/d/_bulk_docs"


def test_all_docs_leaves_out_revisions_and_the_design_document(http):
    rows = [{"id": "_design/workorders", "doc": {"_id": "_design/workorders", "_rev": "1-a"}},
            {"id": "wo:MAIN:wo-1", "doc": {"_id": "wo:MAIN:wo-1", "_rev": "2-b", "status": "COMP"}}]
    http.answers = [Response(200, {"rows": rows})]
    assert CouchServer("http://127.0.0.1:5984", "admin", "password").all_docs("w") == [{"_id": "wo:MAIN:wo-1", "status": "COMP"}]
    assert http.calls[0][2]["params"] == {"include_docs": "true"}


def test_a_design_document_is_put_over_the_existing_revision(http):
    http.answers = [Response(200, {"_rev": "3-c"}), Response(201, {"ok": True})]
    CouchServer("http://127.0.0.1:5984", "admin", "password").install_design("w", {"_id": "_design/workorders", "views": {}})
    assert http.calls[1][0] == "put" and http.calls[1][2]["json"]["_rev"] == "3-c"
