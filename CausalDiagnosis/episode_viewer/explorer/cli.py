"""The two commands of the episode explorer:

    python -m CausalDiagnosis episode_viewer explorer serve --runs <folder> [--port N] [--hidden]
    python -m CausalDiagnosis episode_viewer explorer export <episode folder> --out <file.html> [--hidden]

serve starts the explorer for every run folder under --runs on http://127.0.0.1:N/. export writes the page of one episode as
one HTML file with its style, script and data inline and no network reference, never into the run folder and never inside
the fork. --hidden, off by default, adds the hidden side (the true combination, the verdict, the hidden log and losses) for a
person who reads the page on their own machine; the page opens with that side switched off all the same.
"""
import argparse
import json
import tempfile
from pathlib import Path

from CausalDiagnosis.experiments.episode_runners.run_folder_layout import refuse_page_output

STATIC = Path(__file__).resolve().parent / "static"


def _files(out, hidden):
    """The texts of an exported page by the URL the page asks for them."""
    found = {"data/data.json": (out / "data.json").read_text(),
             "hidden-status.json": json.dumps({"served": bool(hidden)})}
    for part in ("files", "levels", "hidden_files", "hidden_levels") if hidden else ("files", "levels"):
        for p in sorted((out / part).rglob("*")):
            if p.is_file():
                found["data/" + p.relative_to(out).as_posix()] = p.read_text(errors="replace")
    if hidden:
        found["data/hidden.json"] = (out / "hidden.json").read_text()
    return found


def export(episode, out_file, hidden=False):
    folder, target = Path(episode).resolve(), Path(out_file).resolve()
    refuse_page_output(target)
    if target.is_relative_to(folder.parent):
        raise SystemExit(f"{target} lies inside the run folder, whose files the run ledger hashes")
    from . import data  # here and not at module level, so the module also loads by its path alone, as a script does
    with tempfile.TemporaryDirectory(prefix="explorer_export_") as tmp:
        data.build(folder, Path(tmp), hidden=hidden)
        files = _files(Path(tmp), hidden)
    # the JSON element may hold any text but its own closing tag
    payload = json.dumps(files).replace("</", "<\\/")
    html = (STATIC / "index.html").read_text()
    html = html.replace('<link rel="stylesheet" href="app.css">', "<style>\n" + (STATIC / "app.css").read_text() + "\n</style>")
    html = html.replace('<script src="app.js" defer></script>\n', "")
    script = (STATIC / "app.js").read_text().replace("</script", "<\\/script")
    html = html.replace("</body>", f'<script type="application/json" id="explorer-data">{payload}</script>\n<script>\n{script}\n</script>\n</body>')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")
    return target


def parser():
    ap = argparse.ArgumentParser(prog="python -m CausalDiagnosis episode_viewer explorer", description="The episode explorer: served on this machine, or one episode as one file.")
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("serve", help="serve every run folder under --runs on 127.0.0.1")
    s.add_argument("--runs", required=True, help="the folder that holds the run folders")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--hidden", action="store_true", default=False, help="also send the ground truth, for a person on this computer")
    e = sub.add_parser("export", help="write the page of one episode as one HTML file")
    e.add_argument("episode", help="an episode folder of a run, such as <run folder>/seed000015")
    e.add_argument("--out", required=True, help="the HTML file to write, outside the fork and the run folder")
    e.add_argument("--hidden", action="store_true", default=False, help="also put the ground truth into the file")
    return ap


def main(argv):
    args = parser().parse_args(argv)
    if args.command == "serve":
        from . import server  # here and not at module level, as data in export
        server.serve(args.runs, args.port, args.hidden)
    else:
        print(export(args.episode, args.out, args.hidden))
