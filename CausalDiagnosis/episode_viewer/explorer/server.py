"""The server of the episode explorer: the page of every episode of the run folders under one folder, on this machine only.

It binds 127.0.0.1 and nothing else, answers GET and HEAD alone, and serves:
  /                                   a redirect to the first played episode, or a sentence when there is none
  /e/<run>/<episode>/                 the page (static/index.html), with app.js and app.css beside it
  /e/<run>/<episode>/hidden-status.json   whether this server sends the hidden side
  /e/<run>/<episode>/data/...         what data.build writes for that episode, built on the first request into a cache
                                      folder outside the runs folder
The hidden side (data/hidden.json and data/hidden_files/) is sent only when the server was started with hidden=True; without
it those routes answer 404, and the builder does not even write them. A run or episode name must be a folder directly under
the runs folder, an episode must hold trajectory.json, and every path is resolved and refused when it leaves its root.
"""
import http.server
import tempfile
import threading
import urllib.parse
from pathlib import Path

STATIC = Path(__file__).resolve().parent / "static"
STATIC_FILES = {"app.js": "text/javascript; charset=utf-8", "app.css": "text/css; charset=utf-8",
                "index.html": "text/html; charset=utf-8"}
TEXT_TYPES = {".json": "application/json", ".jsonl": "text/plain; charset=utf-8", ".md": "text/plain; charset=utf-8",
              ".csv": "text/plain; charset=utf-8", ".txt": "text/plain; charset=utf-8"}
HIDDEN_PARTS = ("hidden.json", "hidden_files", "hidden_levels")
HOST = "127.0.0.1"


def _name(part):
    """A run or episode name: one path component that is no dot name."""
    return bool(part) and part not in (".", "..") and not part.startswith(".") and "/" not in part and "\\" not in part


class Explorer:
    def __init__(self, runs, hidden=False, cache=None):
        self.runs = Path(runs).resolve()
        if not self.runs.is_dir():
            raise SystemExit(f"{runs} is no folder")
        self.hidden = bool(hidden)
        self.cache = Path(cache or tempfile.mkdtemp(prefix="explorer_")).resolve()
        if self.cache.is_relative_to(self.runs):
            raise SystemExit(f"{self.cache} lies inside the runs folder, whose files the run ledgers hash")
        self._lock = threading.Lock()

    def episodes(self):
        return [(r.name, e.name) for r in sorted(self.runs.iterdir()) if r.is_dir() and _name(r.name)
                for e in sorted(r.iterdir()) if e.is_dir() and _name(e.name) and (e / "trajectory.json").is_file()]

    def episode_dir(self, run, episode):
        if not (_name(run) and _name(episode)):
            return None
        folder = (self.runs / run / episode).resolve()
        if folder.parent.parent != self.runs or not (folder / "trajectory.json").is_file():
            return None
        return folder

    def built(self, run, episode):
        from . import data  # here and not at module level, so the module also loads by its path alone, as a script does
        out = self.cache / run / episode
        with self._lock:
            if not (out / "data.json").exists():
                data.build(self.runs / run / episode, out, hidden=self.hidden)
        return out

    def resolve(self, path):
        """(status, file path or redirect target or text, content type) for a request path."""
        parts = [urllib.parse.unquote(p) for p in path.split("?")[0].split("#")[0].split("/") if p]
        if not parts:
            found = self.episodes()
            if not found:
                return 200, "No played episode lies under the runs folder.", "text/plain; charset=utf-8"
            return 302, "/e/{}/{}/".format(*map(urllib.parse.quote, found[0])), None
        if parts[0] != "e" or len(parts) < 3:
            return 404, None, None
        run, episode, rest = parts[1], parts[2], parts[3:]
        if self.episode_dir(run, episode) is None:
            return 404, None, None
        if not rest:
            return 200, STATIC / "index.html", STATIC_FILES["index.html"]
        if len(rest) == 1 and rest[0] in STATIC_FILES:
            return 200, STATIC / rest[0], STATIC_FILES[rest[0]]
        if rest == ["hidden-status.json"]:
            return 200, '{"served": %s}' % ("true" if self.hidden else "false"), "application/json"
        if rest[0] != "data" or len(rest) < 2 or any(not _name(p) for p in rest[1:]):
            return 404, None, None
        if rest[1] in HIDDEN_PARTS and not self.hidden:
            return 404, None, None
        out = self.built(run, episode)
        target = out.joinpath(*rest[1:]).resolve()
        if not target.is_relative_to(out) or not target.is_file():
            return 404, None, None
        return 200, target, TEXT_TYPES.get(target.suffix, "application/octet-stream")


def handler_for(explorer):
    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, head_only):
            status, body, kind = explorer.resolve(self.path)
            if status == 302:
                self.send_response(302)
                self.send_header("Location", body)
                self.send_header("Content-Length", "0")
                self._common()
                return
            if status != 200:
                self.send_error(status)
                return
            payload = body.read_bytes() if isinstance(body, Path) else body.encode()
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(payload)))
            self._common()
            if not head_only:
                self.wfile.write(payload)

        def _common(self):
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:")
            self.end_headers()

        def do_GET(self):
            self._send(False)

        def do_HEAD(self):
            self._send(True)

        def _refuse(self):
            self.send_error(405, "the explorer is read-only")

        do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _refuse

        def log_message(self, *args):
            pass

    return Handler


def make_server(runs, port=8765, hidden=False, cache=None):
    """The server object, bound to 127.0.0.1 and port (0 picks a free one), not yet serving."""
    http.server.ThreadingHTTPServer.request_queue_size = 64  # the page fetches every allowed file at once
    return http.server.ThreadingHTTPServer((HOST, port), handler_for(Explorer(runs, hidden, cache)))


def serve(runs, port=8765, hidden=False):
    server = make_server(runs, port, hidden)
    print(f"episode explorer at http://{HOST}:{server.server_address[1]}/ for {Path(runs).resolve().name}, "
          f"ground truth {'sent' if hidden else 'not sent'}. Stop it with Ctrl-C.", flush=True)  # a file or a pipe would hold it back
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
