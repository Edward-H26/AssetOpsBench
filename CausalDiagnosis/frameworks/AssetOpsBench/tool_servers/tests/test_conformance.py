"""The conformance test of the two backends. On the laptop it shows that the script repeats itself on the
memory store; on the cluster, with COUCHDB_URL naming the job's CouchDB, it requires CouchDB and the memory store to
give the same texts, tool_calls entries and documents, and writes the comparison to AOB_BASE_REPORT."""
import json
import os
from pathlib import Path

import pytest

from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.couchdb_backend import CouchServer
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.in_memory_couchdb import MemoryServer

from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tests import conformance


async def test_the_script_repeats_itself_on_the_memory_store():
    first, second = await conformance.run(MemoryServer()), await conformance.run(MemoryServer())
    assert conformance.compare(first, second) == []
    errors = [text for text, is_error in first["replies"] if is_error]
    assert len(errors) == 1 and "HTTP 403" in errors[0]  # the order the work order database refuses
    assert any("unknown sensors" in text for text, _ in first["replies"])
    assert [d["status"] for d in first["documents"]["workorder"]] == ["COMP", "COMP"]


@pytest.mark.skipif(not os.environ.get("COUCHDB_URL"), reason="CouchDB runs on the cluster only")
async def test_couchdb_gives_what_the_memory_store_gives():
    couch = CouchServer(os.environ["COUCHDB_URL"], os.environ.get("COUCHDB_USERNAME", "admin"), os.environ.get("COUCHDB_PASSWORD", "password"))
    try:
        on_couchdb = await conformance.run(couch)
    finally:
        await couch.aclose()
    in_memory = await conformance.run(MemoryServer())
    differences = conformance.compare(in_memory, on_couchdb)
    if os.environ.get("AOB_BASE_REPORT"):
        Path(os.environ["AOB_BASE_REPORT"]).write_text(json.dumps({"calls": len(in_memory["replies"]), "differences": differences}, indent=1))
    assert differences == []
