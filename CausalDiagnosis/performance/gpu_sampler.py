"""The rows of the GPU sampler of a serving job and the figures of one episode from them (CausalDiagnosis/performance/README.md).

The serving job runs nvidia-smi for its whole life, one row per period into gpu-<job id>.csv with the fields that the driver of
its node lists (cluster_jobs/2_serve_model/serve_model.sbatch), and job_record.sh copies the file into job_logs of the grid
folder after the job. A row carries the local time of the node without its zone, so it is placed on the epoch clock of the
episode stamps by the UTC offset of the started line of the endpoint file, which the same job wrote on the same node. One row
stands for one period, which is coarse against one model call, so the figures are per episode: the rows whose time lies inside
the episode's span, the mean of utilization.gpu over them as the busy share, and the sum of power.draw over them times the
period as the energy. The period is the median gap between two rows of the file, the period the job ran the sampler at."""
import statistics
from datetime import datetime, timezone
from pathlib import Path

from CausalDiagnosis.experiments.run_verification import run_ledger as ledger
from . import run_files

TIME, BUSY, POWER = "timestamp", "utilization.gpu", "power.draw"  # the query names of the three fields that the figures read
TIME_FORMATS = ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S")  # the timestamp of nvidia-smi --format=csv, in local time


def zone_of(started):
    """The UTC offset of an ISO time that carries one, as the started line of an endpoint file does, as a timezone; None for no
    time or a time without an offset."""
    try:
        offset = datetime.fromisoformat(started).utcoffset() if started else None
    except ValueError:
        return None
    return timezone(offset) if offset is not None else None


def _number(text):
    """A value of a sampler row without its unit, None for a value that the driver does not report, such as [N/A]."""
    try:
        return float(text.split(" ")[0])
    except ValueError:
        return None


def _epoch(text, zone):
    """The epoch seconds of the local timestamp of a row, None for a text that is no timestamp."""
    for form in TIME_FORMATS:
        try:
            return datetime.strptime(text, form).replace(tzinfo=zone).timestamp()
        except ValueError:
            continue
    return None


def read_rows(path, zone):
    """(rows, unread): the rows of a sampler file in file order, each {"t": epoch seconds, "busy": percent, "power": watts} with None
    for a field that the row lacks, its local time placed by zone; and the count of lines that are neither a header line nor a
    row, such as a line of an error that the sampler printed."""
    names, rows, unread = None, [], 0
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if cells[0] == TIME:
            names = [cell.split(" [")[0] for cell in cells]
            continue
        values = dict(zip(names, cells)) if names is not None and len(cells) == len(names) else {}
        t = _epoch(values[TIME], zone) if values else None
        if t is None:
            unread += 1
            continue
        rows.append({"t": t, "busy": _number(values[BUSY]) if BUSY in values else None,
                     "power": _number(values[POWER]) if POWER in values else None})
    return rows, unread


def period(rows):
    """The period of the sampler, the median of the gaps between consecutive rows, None with fewer than two rows."""
    gaps = [b["t"] - a["t"] for a, b in zip(rows, rows[1:]) if b["t"] > a["t"]]
    return statistics.median(gaps) if gaps else None


def episode_figures(rows, start, end, step):
    """The GPU figures of one episode from the rows whose time lies in [start, end]: their count, the mean of utilization.gpu as a
    share of one and the sum of power.draw times the period step in joules, each None without a row that holds its field."""
    inside = [row for row in rows if start <= row["t"] <= end]
    busy = [row["busy"] for row in inside if row["busy"] is not None]
    power = [row["power"] for row in inside if row["power"] is not None]
    return {"sampler_rows": len(inside), "gpu_busy_share": statistics.mean(busy) / 100 if busy else None,
            "gpu_energy_j": sum(power) * step if power and step is not None else None, "gpu_period_s": step}


def run_sampler(run_dir, meta):
    """(rows, period, note) of the sampler of the serving job of a run: the copy in job_logs of its grid folder, read only when it
    begins with the bytes whose SHA-256 and length run.json recorded at the end of the run and when the ledger holds the started
    line of the endpoint file that places its local times; otherwise no rows and the note that says which of these is missing."""
    grid, job = run_files.grid_of(run_dir)
    recorded = ((meta.get("serving") or {}).get("agent") or {}).get("sampler") or {}
    if grid is None:
        return [], None, "the run lies in no grid folder of a cluster job"
    if not recorded.get("sha256"):
        return [], None, "run.json records no sampler of a model endpoint"
    path = run_files.sampler_copy(grid, job)
    if not path.is_file():
        return [], None, f"job_logs holds no {path.name}; job_record.sh copies it after the job"
    if ledger.prefix_hash(path, recorded.get("bytes"))[0] != recorded["sha256"]:
        return [], None, f"{path.name} does not begin with the bytes whose SHA-256 run.json records"
    zone = zone_of(run_files.endpoint_started(run_dir))
    if zone is None:
        return [], None, "the ledger holds no started line of the endpoint file, which would place the local times of the sampler"
    rows, unread = read_rows(path, zone)
    return rows, period(rows), f"{unread} lines of {path.name} are no row" if unread else ""
