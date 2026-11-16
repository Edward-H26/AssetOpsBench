"""A scripted stand-in for the three routes of SGLang that the Agent-As-Tool port calls: /v1/completions, /v1/tokenize and
/v1/models, on 127.0.0.1 and without a model. It also answers /flush_cache as an idle SGLang does, since the driver flushes the
cache of the model server before an episode of a preset with the wall clock.

The script is the list of the model's continuations in call order. A completion request takes the next one and cuts it at
the earliest stop string of the request, keeping that string when the request sets no_stop_trim, as SGLang does. A
continuation without a stop string ends as if the model wrote its end token. Token counts are whitespace words, so a test
can predict them. Every request body but that of the flush is kept in ``requests`` as (path, body).

With fail_first the first attempt of every distinct POST (path and body) is answered 503 and takes no continuation, so the
retry of the client sends it again. ``stats`` counts the requests open at once (open, most_open, counted from the arrival of
the headers to the start of the answer), the POSTs in arrival order (arrivals, one key per path and body) and the attempts
failed (failed), so that a test sees a second request opened while one is in flight."""
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REVIEW_OK = '{"status": "Accomplished", "reasoning": "The agent answered the question.", "suggestions": "None."}\n(END OF RESPONSE)'
END_TOKEN = 0


def cut(text, stop, keep):
    """(reply, finish_reason, matched_stop) of a continuation under the stop strings of a request."""
    hits = [(text.find(s), s) for s in stop or [] if s and text.find(s) >= 0]
    if not hits:
        return text, "stop", END_TOKEN
    index, matched = min(hits)
    return text[: index + len(matched)] if keep else text[:index], "stop", matched


class ScriptedEndpoint:
    def __init__(self, script, model="granite-stand-in", context_length=131072, cards=None, tokenizer_max_len=None, fail_first=False):
        self.script, self.model, self.context_length, self.fail_first = list(script), model, context_length, fail_first
        # SGLang lists a card per adapter beside the base model's, and /v1/tokenize reports the tokenizer's own maximum
        self.cards = cards if cards is not None else [{"id": model, "object": "model", "max_model_len": context_length}]
        self.tokenizer_max_len = context_length if tokenizer_max_len is None else tokenizer_max_len
        self.requests = []
        self.stats = {"failed": 0, "open": 0, "most_open": 0, "arrivals": []}
        self._failed_keys = set()
        self._lock, self._stats_lock = threading.Lock(), threading.Lock()
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, body):
                data = json.dumps(body).encode("utf-8")
                # closed before the answer leaves, so that the client's next request can never be counted beside this one
                self._close()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _close(self):
                if getattr(self, "_open", False):
                    self._open = False
                    endpoint.closed()

            def do_GET(self):
                with endpoint.opened(self):
                    self._get()

            def do_POST(self):
                with endpoint.opened(self):
                    self._post()

            def _get(self):
                if self.path.rstrip("/").endswith("/v1/models"):
                    self._send(200, {"object": "list", "data": endpoint.cards})
                else:
                    self._send(404, {"error": self.path})

            def _post(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path.split("?")[0].rstrip("/").endswith("/flush_cache"):
                    # no model request: it takes no continuation, enters no record and is never failed first
                    self._send(200, {"message": "Cache flushed."})
                    return
                with endpoint._lock:
                    endpoint.requests.append((self.path, body))
                    if endpoint.fails(self.path, body):
                        self._send(503, {"error": {"message": "the first attempt of this request fails"}})
                        return
                    if self.path.endswith("/v1/tokenize"):
                        count = len(str(body.get("prompt", "")).split())
                        self._send(200, {"tokens": list(range(count)), "count": count, "max_model_len": endpoint.tokenizer_max_len})
                        return
                    if not self.path.endswith("/v1/completions"):
                        self._send(404, {"error": self.path})
                        return
                    if not endpoint.script:
                        self._send(500, {"error": {"message": "the script is exhausted"}})
                        return
                    entry = endpoint.script.pop(0)
                    text = entry(body) if callable(entry) else entry
                reply, reason, matched = cut(text, body.get("stop"), body.get("no_stop_trim", False))
                self._send(200, {"id": "cmpl-scripted", "object": "text_completion", "model": body.get("model"),
                                 "choices": [{"index": 0, "text": reply, "finish_reason": reason, "matched_stop": matched}],
                                 "usage": {"prompt_tokens": len(body.get("prompt", "").split()), "completion_tokens": len(reply.split()),
                                           "total_tokens": len(body.get("prompt", "").split()) + len(reply.split())}})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    @contextmanager
    def opened(self, handler):
        """One request open from the arrival of its headers until its answer starts or its handler ends."""
        with self._stats_lock:
            self.stats["open"] += 1
            self.stats["most_open"] = max(self.stats["most_open"], self.stats["open"])
        handler._open = True
        try:
            yield
        finally:
            handler._close()

    def closed(self):
        with self._stats_lock:
            self.stats["open"] -= 1

    def fails(self, path, body):
        """Records the arrival of a POST and tells whether it is the first attempt of its path and body under fail_first."""
        key = json.dumps([path, body], sort_keys=True)
        with self._stats_lock:
            self.stats["arrivals"].append(key)
            if not self.fail_first or key in self._failed_keys:
                return False
            self._failed_keys.add(key)
            self.stats["failed"] += 1
            return True

    def completions(self):
        return [body for path, body in self.requests if path.endswith("/v1/completions")]

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def step(thought, action, action_input):
    """The two continuations of one ReAct step: the thought, which stops at the action line, and the action with its input,
    which stops at the observation line."""
    return [f" {thought}\nAction", f" {action}\nAction Input: {action_input}\nObservation"]


def finish(answer, thought="I now know the final answer"):
    return step(thought, "Finish", answer)
