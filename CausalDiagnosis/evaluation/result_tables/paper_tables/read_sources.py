from CausalDiagnosis.utils.paths import RESULTS
import csv
from pathlib import Path


PAGE, LATEX, OFFLINE = "MAIN_TABLES.md", "main_tables.tex", "offline_mcp.csv"
OFFLINE_PATH = RESULTS / OFFLINE  # written by CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/offline_table.py, read wherever the pages go
SOURCES = ("AGENTS", "AGENTS_pairs", "JUDGE", "EFFICIENCY", "THINKING", "STRATA", "POLICIES", "LAYERS_pairs", "MISDIAGNOSIS_PRICE", "TRAINING", "TRAINING_pairs",
           "TRAINING_RECORD", "H2PRIME_pairs", "H0", "MCP", "RESTORED", "FULL_ENV_pairs", "POLICY_LAYERS", "PARALLEL", "PARALLEL_pairs", "RUNS")


# ----- the files -----
def read_csv(path):
    """The rows of a CSV file, each with its number in the file, as the results index numbers them, and none for an absent file."""
    if not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [{**row, "row_number": position} for position, row in enumerate(csv.DictReader(f), 1)]


def read_tables(aggregate_dir):
    """{table: its rows} of the aggregate tables and of OFFLINE_PATH, the offline table of the MCP in the results folder, which the
    cluster writes after the grid (CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/offline_table.py) and which holds its header alone until then. It is an
    input of the pages, so it is read there and never in the folder the pages go to."""
    return {**{name: read_csv(Path(aggregate_dir) / f"{name}.csv") for name in SOURCES}, "OFFLINE": read_csv(OFFLINE_PATH)}
