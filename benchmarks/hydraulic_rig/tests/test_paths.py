"""Every repository path that the two READMEs of the hydraulic rig, the fork's sections of docs/mcp-servers.md and the
pages under docs/hydraulic_rig/ name exists, so a reader who follows them finds the file."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
READMES = ("benchmarks/hydraulic_rig/README.md", "src/envs/hydraulic_rig/README.md")
MCP_SERVERS = "docs/mcp-servers.md"
PATH = re.compile(r"\b(?:src|benchmarks|docs)/[A-Za-z0-9_.\-/]*[A-Za-z0-9_/]")
DOC_NAME = re.compile(r"`((?:hydraulic-rig|causal-diagnosis)-[a-z\-]+\.md)`")  # documents named without docs/
# generated and git-ignored: the run folders of the drivers, the data folder that fetch_uci.py and build_blocks.py fill
# (all but CHECKSUMS.txt, which the first download writes for committing), the models.json that fit.py writes and the
# grades.json that grade.py writes beside it
GENERATED = ("benchmarks/hydraulic_rig/runs/", "benchmarks/hydraulic_rig/results/", "src/envs/hydraulic_rig/data/",
             "src/servers/causal_diagnosis/models.json", "src/servers/causal_diagnosis/grades.json")


def named_paths(text):
    paths = {m.group(0) for m in PATH.finditer(text)} | {f"docs/{m.group(1)}" for m in DOC_NAME.finditer(text)}
    return sorted(p for p in paths if not p.startswith(GENERATED))


def fork_sections(text):
    """The two sections the fork adds to docs/mcp-servers.md, each from its heading to the next. The upstream sections
    stay out: one of them names src/couchdb/init_wo.py, a file upstream has since removed."""
    sections = [s for s in re.split(r"(?m)^(?=## )", text) if re.match(r"## (?:hydraulic_rig|causal_diagnosis)\b", s)]
    assert len(sections) == 2, f"{MCP_SERVERS} has lost its hydraulic_rig or causal_diagnosis section"
    return "".join(sections)


def documents():
    """(name, text) of every document checked; the pages under docs/hydraulic_rig/ are found when the test runs."""
    docs = [(name, (ROOT / name).read_text(encoding="utf-8")) for name in READMES]
    docs.append((MCP_SERVERS, fork_sections((ROOT / MCP_SERVERS).read_text(encoding="utf-8"))))
    pages = sorted((ROOT / "docs" / "hydraulic_rig").rglob("*.md"))
    return docs + [(str(p.relative_to(ROOT)), p.read_text(encoding="utf-8")) for p in pages]


def test_named_paths_exist():
    missing = [(name, path) for name, text in documents() for path in named_paths(text) if not (ROOT / path).exists()]
    assert missing == []
