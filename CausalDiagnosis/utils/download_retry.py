"""The one retry policy of every fetch from outside the machine: the data of record, the scenario files and the trees that
the reproduction fetches, and through retry_fetch of CausalDiagnosis/cluster_jobs/functions.sh every tool that the cluster
jobs fetch with. A compute node fetches over a link that breaks off and stalls, so an attempt waits at most FETCH_TIMEOUT
seconds for its next bytes, and a failed attempt is made again after FETCH_PAUSE seconds. The count of retries starts again
whenever an attempt brought bytes, so a download that moves on is taken up for as long as it moves, and only FETCH_RETRIES
retries in a row that bring nothing end it with FetchFailed. A download into a file keeps what arrived and asks the server
for the rest with a range, which a server that takes no range answers with the whole file, so the next attempt, or the next
run, goes on where the last one stopped. A server that refuses a request, with a status from 400 to 499 other than 408 and
429, is not asked again, and the caller checks what arrived against its pin.

The rule is the one of huggingface_hub, which takes up a download that broke off five times one second apart and starts its
count again whenever bytes arrived. The timeout is the one under which the training environment passed on a compute node of
the cluster, where the 30 seconds of uv's default timed out three times in a row. The cluster jobs set the three values in
the environment, and the defaults below are those of functions.sh, which a test compares."""
import http.client
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

TIMEOUT_S = 300  # FETCH_TIMEOUT: the seconds an attempt waits for its next bytes
RETRIES = 5  # FETCH_RETRIES: the retries in a row that bring nothing before a fetch fails
PAUSE_S = 1  # FETCH_PAUSE: the seconds before a retry
LATER = (408, 429)  # the statuses from 400 to 499 that ask for a later attempt
CHUNK = 1 << 20
# the failures of an attempt that a later attempt can mend: the link (connection, timeout, a stream cut short), a status of
# the server, and a fetching program that ended with an error
TRANSIENT = (OSError, http.client.HTTPException, subprocess.CalledProcessError)


class FetchFailed(OSError):
    """A fetch whose retries in a row brought nothing; its cause is the error of the last attempt."""


def settings():
    """(timeout, retries, pause): FETCH_TIMEOUT, FETCH_RETRIES and FETCH_PAUSE of the environment, else the defaults."""
    def value(name, default, kind):
        text = os.environ.get(name, "").strip()
        return kind(text) if text else default

    return value("FETCH_TIMEOUT", TIMEOUT_S, float), value("FETCH_RETRIES", RETRIES, int), value("FETCH_PAUSE", PAUSE_S, float)


def refused(error):
    """Whether a failed attempt is the server's refusal of the request, which no later attempt changes."""
    return isinstance(error, urllib.error.HTTPError) and 400 <= error.code < 500 and error.code not in LATER


def retried(attempt, what, progress=lambda: 0):
    """The result of attempt(timeout), made again after each failure that a later attempt can mend. progress() tells how far
    the fetch has come, the bytes it keeps, so that an attempt that moved it starts the count of retries again."""
    timeout, retries, pause = settings()
    left = retries
    while True:
        before = progress()
        try:
            return attempt(timeout)
        except TRANSIENT as error:
            if refused(error):
                raise
            if progress() > before:
                left = retries
            if left <= 0:
                raise FetchFailed(f"FETCH_FAILED {what} after {retries} retries that brought nothing: {error!r}") from error
            left -= 1
            print(f"FETCH_RETRY {retries - left} of {retries} in {pause:g} s, {what}: {error!r}", file=sys.stderr, flush=True)
            time.sleep(pause)


def fetched(url, headers=None, opener=None):
    """The bytes at url, for a file small enough to hold in memory, which the caller checks against its pin before it writes
    them. opener takes the place of urllib.request.urlopen, as a test hands in its own."""
    def attempt(timeout):
        with (opener or urllib.request.urlopen)(urllib.request.Request(url, headers=dict(headers or {})), timeout=timeout) as answer:
            return answer.read()

    return retried(attempt, url)


def download(url, path, headers=None, opener=None):
    """The file at url written into path, which keeps the bytes of an attempt that broke off, so that the next attempt, or the
    next run that finds the file, asks the server for the rest. A server that sends the whole file in place of the rest writes
    it from the start, and a server that holds no byte past those kept starts it again. The caller checks the finished file
    against its pin and gives it its final name. opener takes the place of urllib.request.urlopen."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def kept():
        return path.stat().st_size if path.exists() else 0

    def answer(have, timeout):
        ranged = {"Range": f"bytes={have}-"} if have else {}
        return (opener or urllib.request.urlopen)(urllib.request.Request(url, headers={**dict(headers or {}), **ranged}), timeout=timeout)

    def attempt(timeout):
        have = kept()
        try:
            response = answer(have, timeout)
        except urllib.error.HTTPError as error:
            if error.code != 416 or not have:
                raise
            path.unlink()  # no byte past those kept: they are not the start of the file at url
            have, response = 0, answer(0, timeout)
        with response:
            status = getattr(response, "status", 200)
            resumed = status == 206 and have and (response.headers.get("Content-Range") or "").startswith(f"bytes {have}-")
            if status == 206 and not resumed:
                path.unlink(missing_ok=True)
                raise http.client.HTTPException(f"the server answered a range of {url} that does not start at byte {have}")
            length = response.headers.get("Content-Length") if getattr(response, "headers", None) is not None else None
            # read1 hands over the bytes as they arrive, so a stream that breaks off or stalls keeps all that came before
            read, written = getattr(response, "read1", response.read), 0
            with open(path, "ab" if resumed else "wb") as out:
                for chunk in iter(lambda: read(CHUNK), b""):
                    out.write(chunk)
                    written += len(chunk)
            # http.client ends a body cut short without an error when it read with a size, so the length is checked here
            if length is not None and written < int(length):
                raise http.client.IncompleteRead(b"", int(length) - written)
        return path

    return retried(attempt, url, kept)
