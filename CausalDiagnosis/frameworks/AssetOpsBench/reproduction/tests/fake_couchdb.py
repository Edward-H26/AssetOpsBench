"""A CouchDB in one process for the layer B tests, with what the loaders and the servers of main-0.x call: databases, bulk
documents and Mango find with equality and ranges. It compares strings by code point, where CouchDB collates them."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def matches(doc, selector):
    for field, condition in selector.items():
        value = doc.get(field)
        tests = condition if isinstance(condition, dict) else {"$eq": condition}
        for op, target in tests.items():
            if op == "$exists":
                if (field in doc) != target:
                    return False
                continue
            passed = value is not None and {"$eq": lambda: value == target, "$gte": lambda: value >= target, "$gt": lambda: value > target,
                                            "$lt": lambda: value < target, "$lte": lambda: value <= target}[op]()
            if not passed:
                return False
    return True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, body=None):
        data = b"" if body is None else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def parts(self):
        return [p for p in self.path.split("?")[0].split("/") if p]

    def body(self):
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")

    def do_HEAD(self):
        parts = self.parts()
        self.reply(200 if not parts or parts[0] in self.server.dbs else 404)

    def do_GET(self):
        parts = self.parts()
        if not parts:
            return self.reply(200, {"couchdb": "Welcome", "version": "3.5.0"})
        if parts[0] not in self.server.dbs:
            return self.reply(404, {"error": "not_found"})
        return self.reply(200, {"db_name": parts[0], "doc_count": len(self.server.dbs[parts[0]])})

    def do_PUT(self):
        self.server.dbs[self.parts()[0]] = []
        self.reply(201, {"ok": True})

    def do_DELETE(self):
        self.server.dbs.pop(self.parts()[0], None)
        self.reply(200, {"ok": True})

    def do_POST(self):
        parts = self.parts()
        body = self.body()
        if parts == ["_session"]:  # the login of couchdb3
            return self.reply(200, {"ok": True, "name": body.get("name"), "roles": ["_admin"]})
        db, action = parts[:2]
        if action == "_bulk_docs":
            self.server.dbs[db] += body["docs"]
            return self.reply(201, [{"ok": True} for _ in body["docs"]])
        if action == "_index":
            return self.reply(200, {"result": "created"})
        if action == "_find":
            found = [d for d in self.server.dbs[db] if matches(d, body.get("selector", {}))]
            for rule in reversed(body.get("sort") or []):
                (field, direction), = rule.items() if isinstance(rule, dict) else ((rule, "asc"),)
                found.sort(key=lambda d: str(d.get(field)), reverse=direction == "desc")
            return self.reply(200, {"docs": found[: body.get("limit", 25)]})
        return self.reply(404, {"error": "not_found"})


def start():
    """(the server, its URL), with an empty set of databases in server.dbs."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.dbs = {}
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"
