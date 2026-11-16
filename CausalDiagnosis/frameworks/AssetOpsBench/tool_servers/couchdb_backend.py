"""CouchDB as the second backend, with the same calls as the memory store: the administration the producer makes
through AssetOpsBench's loader helpers, and couchdb3 and wo's CouchClient as the handles the tools use. CouchDB runs only
on the cluster, one per job, so this module is exercised by the conformance test there (CausalDiagnosis/frameworks/AssetOpsBench/cluster_jobs/check_metering_layer.sh).
"""
import couchdb3
import requests

from couchdb import loader
from servers.wo.couch import CouchClient


class CouchServer:
    def __init__(self, url, username, password):
        self.url = url.rstrip("/")
        self.auth = (username, password)
        # the loader's helpers read the server and the credentials from these module globals
        loader.COUCHDB_URL, loader._AUTH = self.url, self.auth
        self._databases = {}
        self._clients = {}

    def database_names(self):
        return loader.list_databases()

    def create_database(self, name, drop=False):
        loader._ensure_db(name, drop)

    def drop_database(self, name):
        loader.drop_database(name)

    def install_design(self, name, design):
        """The design document as loader._install_design puts it, from the parsed document rather than its file."""
        url = f"{self.url}/{name}/{design['_id']}"
        existing = requests.get(url, auth=self.auth, timeout=10)
        body = dict(design, _rev=existing.json()["_rev"]) if existing.status_code == 200 else design
        response = requests.put(url, json=body, auth=self.auth, timeout=10)
        if not response.ok:
            raise RuntimeError(f"design document install failed for {name} ({response.status_code}): {response.text}")

    def create_indexes(self, name, indexes):
        loader._create_indexes(name, indexes)

    def bulk_insert(self, name, docs):
        """_bulk_docs as the loader posts it, failing on any rejected document where the loader only logs."""
        for start in range(0, len(docs), 500):
            response = requests.post(f"{self.url}/{name}/_bulk_docs", json={"docs": docs[start:start + 500]}, auth=self.auth, timeout=60)
            response.raise_for_status()
            rejected = [row for row in response.json() if row.get("error")]
            if rejected:
                raise ValueError(f"bulk insert rejected {rejected[:3]}")

    def all_docs(self, name):
        """Every stored document in id order without _rev, the design document left out, as the memory store exports."""
        response = requests.get(f"{self.url}/{name}/_all_docs", params={"include_docs": "true"}, auth=self.auth, timeout=60)
        response.raise_for_status()
        return [{k: v for k, v in row["doc"].items() if k != "_rev"} for row in response.json()["rows"] if not row["id"].startswith("_design/")]

    def database(self, name):
        if name not in self._databases:
            self._databases[name] = couchdb3.Database(name, url=self.url, user=self.auth[0], password=self.auth[1])
        return self._databases[name]

    def couch_client(self, name):
        if name not in self._clients:
            self._clients[name] = CouchClient(self.url, name, username=self.auth[0], password=self.auth[1])
        return self._clients[name]

    async def aclose(self):
        for client in self._clients.values():
            await client.aclose()
        self._clients = {}
