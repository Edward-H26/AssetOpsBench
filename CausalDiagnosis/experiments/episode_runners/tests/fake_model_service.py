"""A model service on a loopback port for the tests of the cut probe and the cache flush: /models always answers, and the
chat route, /flush_cache and /metrics answer as the test sets them. No model runs."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeService:
    def __init__(self, generates=True, flush_status=200, metrics="sglang:num_running_reqs 0.0\nsglang:num_queue_reqs 0.0\n"):
        self.generates, self.flush_status, self.metrics = generates, flush_status, metrics
        self.paths = []
        service = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, body, kind="application/json"):
                data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                service.paths.append(("GET", self.path))
                if self.path.endswith("/models"):
                    self._send(200, {"data": [{"id": "stand-in"}]})
                elif self.path == "/metrics":
                    self._send(200, service.metrics, "text/plain")
                else:
                    self._send(404, {})

            def do_POST(self):
                service.paths.append(("POST", self.path))
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                if self.path.endswith("/chat/completions"):
                    if service.generates:
                        self._send(200, {"choices": [{"message": {"content": "o"}}], "usage": {"completion_tokens": 1}})
                    else:  # a front end that takes the request and generates nothing
                        self._send(503, {"error": "engine not ready"})
                elif self.path.startswith("/flush_cache"):
                    self._send(service.flush_status, "Cache flushed.\n" if service.flush_status == 200 else "Timed out waiting for idle state.",
                               "text/plain")
                else:
                    self._send(404, {})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
