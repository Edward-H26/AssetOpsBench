"""The one retry policy of every fetch from outside the machine (CausalDiagnosis/utils/download_retry.py), against a local server
that follows a plan per request: it breaks a stream off, cuts a body short, stalls, refuses, fails, answers a range or ignores
it. No network."""
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from CausalDiagnosis.utils import download_retry

BODY = bytes(range(256)) * 64  # 16 KiB


class _Server(ThreadingHTTPServer):
    block_on_close = False  # a stalled answer does not hold the end of a test


class PlannedServer:
    """A server on the loopback that answers request i by plan[i] and every later request by "serve". ("break", n) sends the
    first n bytes of the file as a chunked body and closes, as the stream of the UCI archive broke off on the cluster;
    ("short", n) sends n bytes under the Content-Length of the whole file and closes; ("stall", n, s) sends n bytes as a chunked
    body and waits s seconds before it closes; ("cut", n) answers a range as "serve" does but closes after n bytes of it;
    ("status", code) answers code alone; "ignore" sends the whole file whatever range was asked; "serve" answers a range with
    its bytes, a range past the end with 416, and a request without one with the whole file. requests holds the Range header
    of every request, or None."""

    def __init__(self, body, plan=()):
        self.body, self.plan, self.requests = body, list(plan), []
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):  # noqa: N802 - the http.server hook name
                asked = self.headers.get("Range")
                server.requests.append(asked)
                step = server.plan[len(server.requests) - 1] if len(server.requests) <= len(server.plan) else "serve"
                kind = step if isinstance(step, str) else step[0]
                self.close_connection = True
                if kind == "status":
                    self.send_response(step[1])
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                elif kind in ("break", "stall"):
                    self.send_response(200)
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    part = server.body[:step[1]]
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(part), part))
                    self.wfile.flush()
                    if kind == "stall":
                        time.sleep(step[2])
                elif kind == "short":
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(server.body)))
                    self.end_headers()
                    self.wfile.write(server.body[:step[1]])
                else:
                    start = int(asked[len("bytes="):].split("-")[0]) if asked and kind != "ignore" else 0
                    if asked and kind != "ignore" and start >= len(server.body):
                        self.send_response(416)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    self.send_response(206 if start else 200)
                    if start:
                        self.send_header("Content-Range", f"bytes {start}-{len(server.body) - 1}/{len(server.body)}")
                    self.send_header("Content-Length", str(len(server.body) - start))
                    self.end_headers()
                    rest = server.body[start:]
                    self.wfile.write(rest[:step[1]] if kind == "cut" else rest)

            def log_message(self, *args):
                pass

        self.http = _Server(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.http.server_port}/file"
        threading.Thread(target=self.http.serve_forever, daemon=True).start()

    def close(self):
        self.http.shutdown()
        self.http.server_close()


@pytest.fixture
def served():
    """served(body, plan) starts a planned server; every server stops at the end of the test."""
    servers = []

    def start(body=BODY, plan=()):
        servers.append(PlannedServer(body, plan))
        return servers[-1]

    yield start
    for server in servers:
        server.close()


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    """No pause between attempts, so that the tests wait no second of their own."""
    monkeypatch.setenv("FETCH_PAUSE", "0")


def test_a_download_that_broke_off_asks_for_the_rest_and_ends_whole(served, tmp_path, capsys):
    server = served(plan=[("break", 5000)])
    path = download_retry.download(server.url, tmp_path / "file.part")
    assert path.read_bytes() == BODY and server.requests == [None, "bytes=5000-"]
    assert capsys.readouterr().err.count("FETCH_RETRY 1 of 5 ") == 1


def test_a_body_cut_short_under_its_length_is_a_failed_attempt_and_not_a_finished_file(served, tmp_path):
    server = served(plan=[("short", 3000)])
    assert download_retry.download(server.url, tmp_path / "file.part").read_bytes() == BODY
    assert server.requests == [None, "bytes=3000-"]


def test_a_server_that_takes_no_range_sends_the_whole_file_again(served, tmp_path):
    server = served(plan=[("break", 5000), "ignore"])
    assert download_retry.download(server.url, tmp_path / "file.part").read_bytes() == BODY and server.requests == [None, "bytes=5000-"]


def test_a_stalled_attempt_ends_at_the_timeout_and_the_next_one_goes_on(served, tmp_path, monkeypatch):
    monkeypatch.setenv("FETCH_TIMEOUT", "1")
    server = served(plan=[("stall", 4000, 20)])
    started = time.monotonic()
    assert download_retry.download(server.url, tmp_path / "file.part").read_bytes() == BODY
    assert server.requests == [None, "bytes=4000-"] and time.monotonic() - started < 15


def test_a_refusal_of_the_server_is_not_asked_again_and_a_failure_of_the_server_is(served, tmp_path):
    server = served(plan=[("status", 404)])
    with pytest.raises(urllib.error.HTTPError, match="404"):
        download_retry.download(server.url, tmp_path / "file.part")
    assert server.requests == [None]
    server = served(plan=[("status", 503), ("status", 429), ("status", 408)])
    assert download_retry.fetched(server.url) == BODY and len(server.requests) == 4


def test_the_count_of_retries_ends_a_fetch_that_brings_nothing_and_starts_again_whenever_bytes_arrive(served, tmp_path, monkeypatch):
    monkeypatch.setenv("FETCH_RETRIES", "2")
    server = served(plan=[("status", 503)] * 5)
    with pytest.raises(download_retry.FetchFailed, match="FETCH_FAILED .* after 2 retries that brought nothing"):
        download_retry.fetched(server.url)
    assert len(server.requests) == 3
    server = served(plan=[("break", 3000)] * 5)
    with pytest.raises(download_retry.FetchFailed):
        download_retry.download(server.url, tmp_path / "same_bytes.part")
    assert len(server.requests) == 3
    # eight attempts in a row break off, each after 2000 bytes more, so with two retries allowed the download still ends whole
    server = served(plan=[("cut", 2000)] * 8)
    assert download_retry.download(server.url, tmp_path / "moving.part").read_bytes() == BODY and len(server.requests) == 9


def test_kept_bytes_past_the_end_of_the_file_start_the_download_again(served, tmp_path):
    path = tmp_path / "file.part"
    path.write_bytes(BODY + b"left by another file")
    server = served()
    assert download_retry.download(server.url, path).read_bytes() == BODY
    assert server.requests == [f"bytes={len(BODY) + 20}-", None]


def test_the_settings_come_from_the_environment_and_default_to_the_values_of_the_module(monkeypatch):
    for name in ("FETCH_TIMEOUT", "FETCH_RETRIES", "FETCH_PAUSE"):
        monkeypatch.delenv(name, raising=False)
    assert download_retry.settings() == (download_retry.TIMEOUT_S, download_retry.RETRIES, download_retry.PAUSE_S) == (300, 5, 1)
    monkeypatch.setenv("FETCH_TIMEOUT", "12")
    monkeypatch.setenv("FETCH_RETRIES", "3")
    monkeypatch.setenv("FETCH_PAUSE", "0.5")
    assert download_retry.settings() == (12.0, 3, 0.5)
