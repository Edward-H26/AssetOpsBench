# /// script
# requires-python = ">=3.12"
# dependencies = ["wandb==0.31.0"]
# ///
"""The client side of the tracking export (CausalDiagnosis/performance/README.md): it hands the payload that tracking_export.py
built to Weights and Biases, one tracked run per entry, and reads nothing else, no run folder and no other file of the tree.

uv runs this file with the one version of the client that the header above pins, released on 2026-10-16, in an environment of its
own that uv builds from that header, since the client asks for other versions of packages that the locked environments of the
tree pin, and the locks stay as they are:
    uv run --python PYTHON --script CausalDiagnosis/performance/tracking_client.py PAYLOAD

The client goes to the address that the payload declares and to no address of its own. It reads its key the way its own login
leaves it, from WANDB_API_KEY or from the file that the login of its command line tool writes under the host of that address, and
never asks for one, so that a shell without a key sends nothing. SETTINGS turns off what the client would otherwise send by
itself. A tracked run that exists under the id of an entry is taken up and updated, so a folder exported again gives one tracked
run, and a history row whose step the tracked run already holds is not sent again. A tracked run that the client opens under
another id, project or team or user than its entry names stops the export before its configuration, summary, table or history is
sent."""
import json
import re
import sys
import tempfile
import tomllib
from pathlib import Path

# what the client would send by itself and must not, each a setting of the pinned client (wandb.sdk.wandb_settings.Settings)
SETTINGS = {
    "x_disable_meta": True,          # the record of the machine: its host, the user, the working folder, the command line, the program
    "x_disable_machine_info": True,  # the machine information, which also reads the repository and builds the job of the program
    "x_disable_stats": True,         # the metrics of the machine while a tracked run is open
    "host": "",                      # the host name in the record of a tracked run, which the client fills in when this is unset
    "disable_git": True,             # the repository: its remote, its commit and its changes
    "disable_code": True,            # the code of the program
    "save_code": False,
    "x_save_requirements": False,    # the list of the installed packages
    "console": "off",                # the output of this process
    "disable_job_creation": True,    # the job of the program, which records the program and its packages
}
NOTHING_SENT = 3  # the exit status when the client finds no key, or the payload declares no address, and sends nothing
WRONG_PLACE = 4  # the exit status when the client opens a tracked run elsewhere than its entry names, which stops the export
COMMAND = "wandb"  # the command line tool of the pinned client, whose login keeps the key
# the inline header of a script that uv reads (the format of PEP 723)
HEADER = re.compile(r"(?m)^# /// script$\s(?P<content>(^#(| .*)$\s)+)^# ///$")


def pinned(path=__file__):
    """The dependencies that the inline header of the script at path pins, as uv reads them."""
    block = HEADER.search(Path(path).read_text(encoding="utf-8")).group("content")
    return tomllib.loads("".join(line[2:] + "\n" for line in block.splitlines()))["dependencies"]


def placed_elsewhere(run, entity, entry):
    """Whether the client opened the tracked run under another id than the entry's, or in another project or for another team or
    user, the names compared without case as the service resolves them."""
    return (run.id != entry["id"] or str(run.project).casefold() != entry["project"].casefold()
            or str(run.entity).casefold() != entity.casefold())


def send(payload, client):
    """Hands every entry of the payload to client, the module of the pinned client or a stand-in of it, and returns the exit
    status: NOTHING_SENT when the payload declares no address or client finds no key, WRONG_PLACE when client opens a tracked run
    elsewhere than its entry names, else 0 once every entry is sent. An entry gives one tracked run at the payload's address in the
    project that the entry names, opened with its id, name, group, job type and tags, and once the tracked run is shown to stand
    where the entry names, its configuration, its summary, its table as one summary value, and its history from the first step
    that the tracked run does not hold yet; an entry that names no project is kept back, since the client would otherwise send it
    to a project of its own choice."""
    address = payload.get("address")
    if not address:
        print("the payload declares no address of the service, so nothing is sent", file=sys.stderr)
        return NOTHING_SENT
    if not client.login(prompt=False, verify=False, host=address):
        print("the client finds no key, neither WANDB_API_KEY nor the file of its login, so nothing is sent", file=sys.stderr)
        return NOTHING_SENT
    with tempfile.TemporaryDirectory(prefix="tracking_client_") as local:  # the client's own files of a run, removed once sent
        for entry in payload["runs"]:
            if not entry.get("project"):
                print(f"KEPT_BACK {entry['name']}: the entry names no project", file=sys.stderr)
                continue
            run = client.init(entity=payload["entity"], project=entry["project"], id=entry["id"], name=entry["name"],
                              group=entry["group"], job_type=entry["job_type"], tags=entry["tags"], resume="allow", mode="online",
                              dir=local, settings={**SETTINGS, "base_url": address})
            if placed_elsewhere(run, payload["entity"], entry):
                print(f"WRONG_PLACE {entry['name']}: the client opened {run.entity}/{run.project} {run.id}, not "
                      f"{payload['entity']}/{entry['project']} {entry['id']}, so nothing more is sent", file=sys.stderr)
                run.finish(exit_code=1)
                return WRONG_PLACE
            run.config.update(entry["config"], allow_val_change=True)
            run.summary.update(entry["summary"])
            if entry["table"] is not None:
                table = entry["table"]
                run.summary.update({table["key"]: client.Table(columns=table["columns"], data=table["rows"], allow_mixed_types=True)})
            first = run.step  # the step of the next row, above every step that the tracked run already holds
            for row in entry["history"]:
                if row["step"] >= first:
                    run.log(row["values"], step=row["step"])
            run.finish()
            print(f"SENT {entry['name']} as {entry['id']}")
    return 0


def main(argv):
    import wandb  # the pinned client, present only in the environment that uv builds from the header

    return send(json.loads(Path(argv[0]).read_text(encoding="utf-8")), wandb)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
