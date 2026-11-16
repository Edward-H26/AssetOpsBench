"""The result tables of the experiments (CausalDiagnosis/docs/experiments.md) in the order and
layout of the result tables of the paper, each under its label there. The body holds one table per experiment with the metrics of
the AssetOpsBench paper first and the verified outcome after them: each design without and with the CausalDiagnosis MCP on
layer 0 and in the CausalDiagnosis environment, the other measures of the same lines, the hypotheses H1 and H2, the cumulative layer
study, and the trained agents. The appendix holds the execution
efficiency of the six rows, the thinking time charged and where its seconds went, the effect of each big layer, the reference policies on the ladder, the training hypotheses decided on
every test episode under two-fold cross-fitting with the single fault and two fault starts beside them and each fold apart, what
each agent trained on per fold and the training episodes that reached a test combination, the strata of the two separability labels,
the CausalDiagnosis MCP offline, which plays no episode and comes from OFFLINE_PATH, offline_mcp.csv of the results folder, and what the MCP adds
and risks in the played episodes, the measures of metrics.md. The rows, the presets,
the hypotheses and the training variants come from experiment_tables.py. The reproduction of AssetOpsBench answers the scenarios of that paper
and stands apart from these tables. The appendix also holds exact match against its majority-class baseline, the judge against four
constructs of the verifier, the full environment against the base layer, the layer effects on the reference policies and the schedule of
candidate plans on against off.

tables_from_runs/ calls write() once its tables are on disk, and this module computes no statistic: every cell is a cell of an aggregate
table of SOURCES or of offline_mcp.csv, and the section main_tables of INDEX.json names the row behind every line. MAIN_TABLES.md is
the page and main_tables.tex the LaTeX form, which compiles on its own. They differ where a column of the LaTeX form has no room for
an interval or a count. A row without a run says so in every cell, as does the whole page before any run.
"""

from . import read_sources
from . import table_cells
from . import result_sheet
from . import appendix_layer_tables
from . import thinking_tables
from . import markdown_page
from . import latex_page
from . import body_tables
from . import appendix_judge_tables
from . import appendix_schedule_tables
from . import appendix_agent_tables
from . import appendix_training_tables
from . import appendix_mcp_tables
from . import write_pages

from .read_sources import LATEX, OFFLINE, OFFLINE_PATH, PAGE, SOURCES, read_tables
from .table_cells import Column, DAGGER, DASH, NO_VALUE, UNJUDGED_RULE, aggregate, holm_cell, in_words
from .result_sheet import AGENTS_PRESET, Sheet, row_state
from .appendix_layer_tables import ARM_PRESETS, MCP
from .thinking_tables import TRAINING_PRESET, charges_thinking, plays_generation_time, thinking_groups
from .body_tables import AGENT_MEASURES, PASS_K_SHOWN
from .appendix_schedule_tables import SWITCH_NAMES, parallel_table
from .appendix_agent_tables import STRATA_SHOWN, h2_prime_table
from .appendix_training_tables import fold_groups
from .appendix_mcp_tables import MCP_CELLS, OFFLINE_PREHISTORY, OFFLINE_READINGS, mcp_table
from .write_pages import built_layout, write
