"""A stand-in of the client module of the tracking export for its tests: the calls that tracking_client.send makes, login, init,
Table and the methods and the configuration of a tracked run, each recorded with every value it is handed, and no network. Every
value here is made up."""


class StandInTable:
    def __init__(self, columns=None, data=None, allow_mixed_types=False):
        self.columns, self.data, self.allow_mixed_types = columns, data, allow_mixed_types


class StandInSummary:
    def __init__(self, record):
        self.record = record

    def update(self, values):
        self.record["summary"].append(dict(values))


class StandInConfig:
    def __init__(self, record):
        self.record = record

    def update(self, values, allow_val_change=None):
        self.record["config"].append(dict(values))


class StandInRun:
    """A tracked run as init returns it: its id, project and team or user, which the client may resolve otherwise than it was asked,
    and step, the step of its next history row, which a tracked run taken up starts above the steps it holds."""

    def __init__(self, record, step, placed):
        self.record, self.step, self.summary, self.config = record, step, StandInSummary(record), StandInConfig(record)
        self.id, self.project, self.entity = placed["id"], placed["project"], placed["entity"]

    def log(self, values, step=None):
        self.record["log"].append({"step": step, "values": dict(values)})

    def finish(self, exit_code=None):
        self.record["finished"], self.record["exit_code"] = True, exit_code


class StandInClient:
    """The client module as tracking_client.send calls it. key says whether login finds a key, held gives per id the step of the
    next row of a tracked run that exists already, and placed the id, project or team or user under which init opens every tracked
    run instead of the asked ones, as a client that a variable of its own turns would."""
    Table = StandInTable

    def __init__(self, key=True, held=None, placed=None):
        self.key, self.held, self.placed, self.logins, self.runs = key, dict(held or {}), dict(placed or {}), [], []

    def login(self, **options):
        self.logins.append(options)
        return self.key

    def init(self, **arguments):
        record = {"init": arguments, "config": [], "summary": [], "log": [], "finished": False}
        self.runs.append(record)
        asked = {name: arguments.get(name) for name in ("id", "project", "entity")}
        return StandInRun(record, self.held.get(arguments.get("id"), 0), {**asked, **self.placed})


def leaves(value):
    """Every key and every leaf of a nested value, a table's columns and cells among them, so that a test reads all of it."""
    if isinstance(value, StandInTable):
        return leaves([value.columns, value.data])
    if isinstance(value, dict):
        return [found for key, item in value.items() for found in (*leaves(key), *leaves(item))]
    if isinstance(value, (list, tuple)):
        return [found for item in value for found in leaves(item)]
    return [value]


def handed(client, local_folder=True):
    """Every key and leaf of the values that the stand-in was handed: the arguments of init, the configuration, the summaries with
    their tables and the rows of the history with their steps. local_folder False leaves out the dir argument of init, the client's
    own folder of this machine, where it writes its files."""
    found = []
    for run in client.runs:
        found += leaves([value for name, value in run["init"].items() if local_folder or name != "dir"])
        found += leaves(run["config"]) + leaves(run["summary"]) + leaves([[row["step"], row["values"]] for row in run["log"]])
    return found
