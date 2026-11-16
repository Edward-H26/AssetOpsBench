"""The chess clock of the thinking time (CausalDiagnosis/docs/environment.md, the section on the clock of the thinking time).

The clock of an episode runs only while the environment waits for the agent. Its origin is the handover H, which the driver
stamps just before it hands the task to the agent, and its cursor u starts there. A counted call k, a call that reaches
env.step, enters its service at e_k and ends it at f_k, both read on the monotonic clock of the host by the thread that serves
the call. The call is charged |[u, e_k] \\ B| when e_k > u, and nothing otherwise, where B is the union of the credited
segments of the first start of every tool server and of every failed attempt of a model request. The cursor then moves on by
the credit of the service alone, u = max(u, min(f_k, max(e_k, u) + g_k)), where g_k is the thread time of the serving thread
between e_k and f_k and never more than f_k - e_k, so the part of a service beyond its credit is charged at the next call. A
start [s, r] of a server is credited from s for the main thread's thread time between s and r and never more than r - s, and a
failed attempt from its start to the start of the next attempt. The module holds the arithmetic alone; the episode module
keeps the cursor in the state file and reads the transcript and the ledger of failed attempts."""


def credit(length, thread_seconds):
    """The credited part of an interval: its thread time, never more than its length and never below zero."""
    return max(0.0, min(float(length), float(thread_seconds)))


def start_segment(line):
    """The credited segment [s, s + c] of the first start of a server from its ready line, None for a line without the
    launcher's stamp, as for a server main started without a launcher, whose start is credited nothing."""
    if line.get("s_mono") is None or line.get("s_thread_cpu") is None:
        return None
    c = credit(line["r_mono"] - line["s_mono"], line["r_thread_cpu"] - line["s_thread_cpu"])
    return (float(line["s_mono"]), float(line["s_mono"]) + c) if c > 0 else None


def merged(segments):
    """The union of segments as disjoint segments in rising order."""
    out = []
    for a, b in sorted((float(a), float(b)) for a, b in segments if b > a):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def uncovered(a, b, segments):
    """|[a, b] \\ union of segments|, zero when b <= a."""
    if b <= a:
        return 0.0
    covered = sum(max(0.0, min(b, y) - max(a, x)) for x, y in merged(segments))
    return max(0.0, (b - a) - covered)


def charge(u, e, segments):
    """theta_k: the seconds charged to a counted call entering at e with the cursor at u."""
    return uncovered(u, e, segments) if e > u else 0.0


def advance(u, e, f, g):
    """The cursor after a counted call served from e to f with the credit g."""
    return max(u, min(f, max(e, u) + g))


def after(segments, u):
    """The segments of B that still matter once the cursor stands at u, those that end after it, merged."""
    return [[a, b] for a, b in merged(segments) if b > u]
