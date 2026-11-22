"""The run sheet, README.md of this folder: every variable, option, preset and file that it names exists in the tree, the flags of
the engine, the files of its export and the words of sbatch named apart; its command is in the command table, every column it
names is a column of the profile, the lines of every job set its export folder and make it before they submit the job, every job
plays on the card type that the card bullet gives it, and the prefix cache of the engine is off in the job that says so alone.
The controls plant a made-up name of each kind, and a wrong line of each kind, which the check must report."""
import fnmatch
import re
from pathlib import Path

import pytest

from CausalDiagnosis import __main__ as table
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path
from CausalDiagnosis.datasets.splits.development_split import development_starts
from CausalDiagnosis.experiments.cluster_runs.run_health_checks import JOB_RECORD
from CausalDiagnosis.experiments.episode_runners.run_configuration import load_world
from CausalDiagnosis.experiments.episode_runners.run_episodes import command_line, planned_episodes, play_one_episode
from CausalDiagnosis.experiments.episode_runners.run_episodes.command_line import build_parser
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import STATE_FILE
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger
from CausalDiagnosis.experiments.episode_runners.episode_launchers import LAUNCH_PRIVATE
from CausalDiagnosis.experiments.run_verification.check_grid_complete import GRID, IN_FLIGHT, LLM_CALLS, SAMPLER
from CausalDiagnosis.experiments.run_verification.public_record import PRIVATE_FILE
from CausalDiagnosis.experiments.table_definitions import experiment_tables
from CausalDiagnosis.mcp_servers.utils import transcript
from CausalDiagnosis.performance import engine_export, jobs, profile_table, run_files, stage_profile, tracking_export
from CausalDiagnosis.utils.paths import CLUSTER_JOBS, PACKAGE, ROOT, preset_files

SHEET = (PACKAGE / "performance" / "README.md").read_text(encoding="utf-8")
# flags of SGLang's launch that SGLANG_EXTRA carries to the serving job and that the tree does not define
ENGINE_FLAGS = ("--enable-metrics", "--enable-request-time-stats-logging", "--export-metrics-to-file", "--export-metrics-to-file-dir",
                "--disable-radix-cache")
# the options of sbatch on a submit line, which submit.sh passes on and the cluster scripts or their page name
SLURM_OPTIONS = ("--gres", "--export")
END_OF_OPTIONS = "--"  # the end of the options of submit.sh, after which the command of a CPU job follows
# the files that a run, a grid job, the profile or a training writes, by the constants of their writers
WRITTEN = {STATE_FILE, "scores.csv", run_files.RUN_FILE, ledger.LEDGER, LLM_CALLS, run_files.TRAJECTORY_FILE, transcript.NAME,
           JOB_RECORD, IN_FLIGHT, profile_table.EPISODES_FILE, profile_table.CALLS_FILE, profile_table.CELLS_FILE, LAUNCH_PRIVATE, PRIVATE_FILE,
           tracking_export.TRAINER_STATE, tracking_export.SOURCE, tracking_export.ADAPTER_MANIFEST}
SUFFIXES = (".sh", ".sbatch", ".py", ".json", ".jsonl", ".txt", ".csv", ".tsv", ".md")
SETTINGS = ("env.sh", "env.example.sh")  # the settings file of a clone, which each person makes from its tracked template
PLACEHOLDER = re.compile(r"<[^>]*>")
COMMAND = re.compile(r"python -m CausalDiagnosis (\S+) (\S+)")
JOB = re.compile(r"^(J\d(?: and J\d)*), ", re.M)  # the first words of the paragraph of a measurement job
SERVING_SUBMIT = re.compile(r'SGLANG_EXTRA="\$F (\$D/[^" ]+)[^"]*" bash \$H/submit\.sh serve_model\.sbatch')
CARD_LINE = re.compile(r"^C=<the card type of (J\d)>$", re.M)  # the line of a job that plays on the card type of an earlier one
CACHE_OFF = "--disable-radix-cache"  # the flag of SGLang's launch that turns the prefix cache of the engine off


def scripts_text():
    """The text of the tree's cluster scripts, its settings template and its cluster page, where a variable of a job is named."""
    paths = [*CLUSTER_JOBS.rglob("*.sh"), *CLUSTER_JOBS.rglob("*.sbatch"), CLUSTER_JOBS / "README.md"]
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


SCRIPTS = scripts_text()
IGNORED_GRIDS = [line.strip() for line in (CLUSTER_JOBS / ".gitignore").read_text(encoding="utf-8").splitlines() if line.startswith("grids/")]


def code_and_spans(text):
    """The fenced code blocks and the inline code spans of a page, each as one text."""
    blocks = re.findall(r"```[a-z]*\n(.*?)```", text, re.S)
    rest = re.sub(r"```[a-z]*\n.*?```", "", text, flags=re.S)
    return blocks + re.findall(r"`([^`\n]+)`", rest)


def atoms(text):
    """The words of a code text, a placeholder <...> kept as one word, split at blanks, commas, equal signs and quotes."""
    text = PLACEHOLDER.sub(lambda match: match.group(0).replace(" ", "_"), text)
    return [atom.strip(":;.()[]") for atom in re.split(r"[\s,=\"']+", text) if atom.strip(":;.()[]")]


def file_exists(atom):
    """Whether a file that the page names exists in the tree, is written by a run, a grid job, the profile or the engine's export,
    or is a local grid file that git ignores; a path under a placeholder folder names no file of the tree."""
    if atom.startswith(engine_export.FILE_PREFIX):
        return True
    if PLACEHOLDER.search(atom):
        filled = PLACEHOLDER.sub("1", atom)
        if atom.startswith("<"):
            return True
        return (PLACEHOLDER.sub("{job}", atom) == SAMPLER or bool(GRID.match(filled))
                or any(fnmatch.fnmatch(filled, rule) for rule in IGNORED_GRIDS))
    if atom.startswith("$H/"):
        return (CLUSTER_JOBS / atom[3:]).exists()
    if atom == SETTINGS[0]:
        return (CLUSTER_JOBS / SETTINGS[1]).is_file()
    local_grid = any(fnmatch.fnmatch(name, rule) for name in (atom, f"grids/{atom}") for rule in IGNORED_GRIDS)
    if local_grid or atom in WRITTEN or (ROOT / atom).exists():
        return True
    return "/" not in atom and bool([*CLUSTER_JOBS.glob(atom), *CLUSTER_JOBS.glob(f"[0-9]_*/{atom}")])


def defined_names(text):
    """The shell variables of the page's own: those that a code text assigns at its start and that the page then uses as $NAME.
    Every other name before an equal sign is a variable of a job, which the scripts must name."""
    code = "\n".join(code_and_spans(text))
    return set(re.findall(r"^([A-Z][A-Z0-9_]*)=", code, re.M)) & set(re.findall(r"\$([A-Z][A-Z0-9_]*)", code))


def unknown_names(text):
    """The names of a page that neither the tree, the flags of the engine, the files of its export nor the words of sbatch hold."""
    options = set().union(*(parser()._option_string_actions for parser in (build_parser, profile_table.parser, tracking_export.parser)))
    presets, unknown = set(preset_files()), []
    defined = defined_names(text)
    for piece in code_and_spans(text):
        found = atoms(piece)
        for before, atom in zip(["", *found], found):
            if before == "--config" and atom not in presets:
                unknown.append(atom)
            elif atom == END_OF_OPTIONS:
                if f"cpu_job.sbatch {END_OF_OPTIONS} " not in SCRIPTS:
                    unknown.append(atom)
            elif atom.startswith("--"):
                sbatch = atom in SLURM_OPTIONS and atom in SCRIPTS
                if atom not in options | set(ENGINE_FLAGS) and not sbatch:
                    unknown.append(atom)
            elif re.fullmatch(r"\$?[A-Z][A-Z0-9_]*", atom):
                name = atom.lstrip("$")
                if name not in defined and not re.search(rf"\b{name}\b", SCRIPTS):
                    unknown.append(atom)
            elif atom.startswith("$PWD") or (re.match(r"\$([A-Z][A-Z0-9_]*)/", atom) and atom[1:].split("/")[0] in defined):
                continue  # a path of the shell at the fork root, or under a folder that the page's own variable names
            elif ("/" in atom or atom.endswith(SUFFIXES)) and not file_exists(atom):
                unknown.append(atom)
    unknown += [name for name in re.findall(r"preset `([a-z0-9_]+)`", text) if name not in presets]
    unknown += [f"{group} {command}" for group, command in COMMAND.findall(text) if command not in table.COMMANDS.get(group, {})]
    return unknown


def test_every_name_of_the_sheet_exists_in_the_tree():
    assert unknown_names(SHEET) == []


@pytest.mark.parametrize("planted", ["`--made-up-option`", "`MADE_UP_VARIABLE`", "`--config made_up_preset_name`",
                                     "the preset `made_up_preset_name`", "`CausalDiagnosis/no_such_made_up_file.py`",
                                     "`$H/3_run/no_such_made_up_script.sh`", "`python -m CausalDiagnosis performance made-up-command`",
                                     "`MADE_UP_VARIABLE=1`", "```bash\nMADE_UP_VARIABLE=1 bash $H/submit.sh serve_model.sbatch\n```"])
def test_a_made_up_name_of_each_kind_is_reported(planted):
    assert unknown_names(SHEET + f"\n{planted}\n") != []


def test_a_variable_is_the_page_own_only_when_the_page_assigns_it_and_uses_it():
    assert defined_names(SHEET) == {"C", "D", "E", "F", "T"}
    assert "MADE_UP_VARIABLE" in defined_names(SHEET + "\n`MADE_UP_VARIABLE=1` and `$MADE_UP_VARIABLE`\n")  # the control
    assert unknown_names(SHEET + "\n```bash\nMADE_UP_VARIABLE=1 bash $H/submit.sh serve_model.sbatch\n```\n") == ["MADE_UP_VARIABLE"]


def test_the_start_cap_of_j1_is_the_smallest_that_the_driver_accepts_on_its_line():
    """J1 plays the development starts that cover every faulty level of the asset: the driver accepts its cap on J1's line, the
    smallest cap that it accepts, and refuses one less with the count that the coverage needs; no development start lies within
    the fault rule of J1's preset, so none can be named alone."""
    if not Path(default_facts_path()).exists():
        pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH")
    (cap,) = {int(found) for found in re.findall(r"--start-cap=(\d+)", SHEET)}
    (line,) = [text for text in SHEET.splitlines() if text.startswith("# grids/local_probe.txt:")]
    a = build_parser().parse_args([*line.split(":", 1)[1].split(), "--starts", "development", "--seeds", "1", "--start-cap", str(cap)])
    _name, config, _prices = command_line.configure(a)
    _facts, _cycles, blocks = load_world(a.facts, config.asset_id)
    assert planned_episodes.played_config(a, config, blocks).fault_rule is not None
    profile = play_one_episode.get_profile(config.asset_id)
    assert not [start for start in development_starts(profile, set(blocks)) if config.within_fault_rule(list(start))]  # no named start
    a.start_cap = cap - 1
    with pytest.raises(SystemExit, match=f"below the {cap} starts that cover every faulty level"):
        planned_episodes.played_config(a, config, blocks)


def test_the_sizing_of_j3_and_j4_holds_the_episodes_of_their_runs_and_the_job_limit_of_the_tree():
    """The arithmetic of the sheet: the episodes of a run of J3's line, the share of the job limit that the pilot's rule projects a run
    against, and the seconds per planned episode that it leaves, all three from the tree; the time limit of the serving job that the
    sheet names is the job limit that the tree declares."""
    if not Path(default_facts_path()).exists():
        pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH")
    (line,) = [text for text in SHEET.splitlines() if text.startswith("# grids/local_sampling_1.txt:")]
    a = build_parser().parse_args([*line.split(":", 1)[1].split(), "--starts", "training", "--training-split", "development"])
    _name, config, _prices = command_line.configure(a)
    _facts, cycles, blocks = load_world(a.facts, config.asset_id)
    played = planned_episodes.played_config(a, config, blocks)
    episodes = len(planned_episodes.chosen_seeds(a, played, cycles, blocks))
    bound = experiment_tables.PROJECTION_SHARE * experiment_tables.JOB_LIMIT_S
    assert f"plays the {episodes} training starts of its sample" in SHEET
    assert f"3/4 x {experiment_tables.JOB_LIMIT_S:,} = {int(bound):,} seconds" in SHEET and bound.denominator == 1
    assert f"{int(bound):,} / {episodes} = {int(bound / episodes)} seconds per planned episode" in SHEET and (bound / episodes).denominator == 1
    sbatch = (CLUSTER_JOBS / "2_serve_model" / "serve_model.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH -t 12:00:00" in sbatch and experiment_tables.JOB_LIMIT_S == 12 * 3600 and "`-t 12:00:00`" in SHEET


def test_the_engine_flags_are_named_as_flags_of_the_engine_that_the_serving_job_passes_on():
    named = {flag for piece in code_and_spans(SHEET) for flag in atoms(piece) if flag in ENGINE_FLAGS}
    assert named == set(ENGINE_FLAGS)
    assert "`SGLANG_EXTRA` carries flags of SGLang's launch, not names of this tree" in SHEET
    assert "SGLANG_EXTRA" in (CLUSTER_JOBS / "2_serve_model" / "serve_model.sbatch").read_text(encoding="utf-8")


def export_folder_gaps(text):
    """The jobs of a page whose lines do not submit each job with an export folder of its own made first: the code blocks between
    the paragraph of a job and the next one hold one serving submit per job that the paragraph names, each after a line of its
    block that sets D anew, so that no job writes into the folder of the job before it, through twins, whose folders the function
    makes, or through a submit line whose export folder a mkdir of the same block makes before it."""
    heads = list(JOB.finditer(text))
    gaps = []
    for head, end in zip(heads, [match.start() for match in heads[1:]] + [len(text)]):
        submits = 0
        for block in re.findall(r"```[a-z]*\n(.*?)```", text[head.start():end], re.S):
            made, named = set(), False
            for line in block.splitlines():
                named = named or line.startswith("D=")
                made.update(re.findall(r'^mkdir -p "(\$D/[^"]+)"$', line))
                folders = SERVING_SUBMIT.findall(line)
                if (line.startswith("twins ") or folders) and not named:
                    gaps.append(f"{head.group(1)} submits before its lines set D")
                submits += line.startswith("twins ")
                for folder in folders:
                    submits += 1
                    if folder not in made:
                        gaps.append(f"{head.group(1)} submits with {folder}, which no mkdir before it makes")
        if submits != len(re.findall(r"J\d", head.group(1))):
            gaps.append(f"{head.group(1)} has {submits} serving submits")
    return gaps


def test_every_job_of_the_sheet_submits_with_an_export_folder_that_its_lines_make_first():
    assert export_folder_gaps(SHEET) == []
    (twins,) = [line for line in SHEET.splitlines() if line.startswith("twins() {")]
    assert twins.index('mkdir -p "$D/$card"') < twins.index("submit.sh")
    planted = '\nJ9, made up:\n\n```bash\nD=<ROOT>/x\nRUN_KIND=development SGLANG_EXTRA="$F $D/$C" bash $H/submit.sh serve_model.sbatch --export=$E\n```\n'
    assert export_folder_gaps(SHEET + planted) == ["J9 submits with $D/$C, which no mkdir before it makes"]  # the control
    unset = '\nJ9, made up:\n\n```bash\nmkdir -p "$D/$C"\nRUN_KIND=development SGLANG_EXTRA="$F $D/$C" bash $H/submit.sh serve_model.sbatch --export=$E\n```\n'
    assert export_folder_gaps(SHEET + unset) == ["J9 submits before its lines set D"]  # the control of D
    assert export_folder_gaps(SHEET + "\nJ9, made up, without lines.\n") == ["J9 has 0 serving submits"]  # the control of a job without lines


def test_every_column_that_the_sheet_names_is_a_column_of_the_profile():
    rows = [line for line in SHEET.splitlines() if line.startswith("| ") and line.count("`") >= 2]
    named = {match for line in rows for match in re.findall(r"^\| [^|]+ \| `([a-z0-9_]+)` \|", line)}
    known = set(profile_table.COLUMNS) | set(jobs.PARTS) | set(engine_export.CALL_COLUMNS)
    assert named and named <= known, sorted(named - known)
    assert set(stage_profile.CHECKS) <= named and set(jobs.PARTS) <= named


def serving_blocks(text):
    """(job, block) of every code block of a page that submits a serving job, each given to the job whose paragraph it follows,
    the blocks of a paragraph of two jobs to the two in order."""
    heads = list(JOB.finditer(text))
    found = []
    for head, end in zip(heads, [match.start() for match in heads[1:]] + [len(text)]):
        blocks = [block for block in re.findall(r"```[a-z]*\n(.*?)```", text[head.start():end], re.S)
                  if re.search(r"^twins ", block, re.M) or SERVING_SUBMIT.search(block)]
        found += list(zip(re.findall(r"J\d", head.group(1)), blocks))
    return found


def card_gaps(text):
    """The jobs of a page whose lines do not play on the card type that its card bullet gives them: a job that the bullet says
    follows an earlier one sets C to the card type of that one and submits on the card $C alone, and every other job is submitted
    as twins."""
    (bullet,) = [line for line in text.splitlines() if line.startswith("- The card.")]
    followed = {job: earlier for jobs, earlier in re.findall(r"((?:J\d)(?: and J\d)*) follows? (J\d)", bullet)
                for job in re.findall(r"J\d", jobs)}
    gaps = []
    for job, block in serving_blocks(text):
        twins = bool(re.search(r"^twins ", block, re.M))
        submits = [line for line in block.splitlines() if SERVING_SUBMIT.search(line)]
        named = CARD_LINE.findall(block)
        on_card = bool(submits) and all("--gres=gpu:$C:1 " in line for line in submits)
        found = "twins" if twins and not submits and not named else named[0] if len(named) == 1 and on_card and not twins else None
        if found != followed.get(job, "twins"):
            gaps.append(f"{job} plays on {found or 'no card type of a job'}, not on {followed.get(job, 'twins')}")
    return gaps + [f"{job} follows {earlier} and has no lines" for job, earlier in followed.items() if job not in dict(serving_blocks(text))]


def test_every_job_plays_on_the_card_type_that_the_card_bullet_gives_it():
    """The card bullet says which job follows which, and the lines of each job follow it: J3 on the card type of J1, J4 and J5 on
    that of J3, and J1 and J2 as twins. The controls plant the wrong lines of each kind."""
    assert card_gaps(SHEET) == [] and [job for job, _block in serving_blocks(SHEET)] == ["J1", "J2", "J3", "J4", "J5"]
    j3 = SHEET.index("\nJ3 and J4,")
    j3_submit = 'RUN_KIND=development SGLANG_EXTRA="$F $D/$C" bash $H/submit.sh serve_model.sbatch --gres=gpu:$C:1 --export=$E\n'
    forms = {"J3 without its C= line": SHEET.replace("C=<the card type of J1>\n", "", 1),
             "J3 back to twins": SHEET[:j3] + SHEET[j3:].replace('mkdir -p "$D/$C"\n' + j3_submit, "twins $E\n", 1),
             "J3 on the card type of J2": SHEET.replace("C=<the card type of J1>", "C=<the card type of J2>", 1),
             "J3 on a card named in its submit": SHEET[:j3] + SHEET[j3:].replace("--gres=gpu:$C:1", "--gres=gpu:l40s:1", 1),
             "J3 following J2 by the bullet": SHEET.replace("as J3 follows J1", "as J3 follows J2", 1),
             "J1 on the card type of J2": SHEET.replace("GRID_EXTRA=--start-cap=5\ntwins $E", "GRID_EXTRA=--start-cap=5\nC=<the card type of J2>\n"
                                                        'mkdir -p "$D/$C"\n' + j3_submit, 1)}
    expected = {"J3 without its C= line": ["J3 plays on no card type of a job, not on J1"],
                "J3 back to twins": ["J3 plays on no card type of a job, not on J1"],
                "J3 on the card type of J2": ["J3 plays on J2, not on J1"],
                "J3 on a card named in its submit": ["J3 plays on no card type of a job, not on J1"],
                "J3 following J2 by the bullet": ["J3 plays on J1, not on J2"],
                "J1 on the card type of J2": ["J1 plays on J2, not on twins"]}
    assert all(form != SHEET for form in forms.values())
    assert {name: card_gaps(form) for name, form in forms.items()} == expected  # the controls


def cache_off(text):
    """(submitted, named): the jobs of a page whose serving submits carry the flag that turns the prefix cache off, after the
    export folder, the twins through the line of the function twins, and the jobs whose paragraph names the flag before its
    lines."""
    (twins,) = [line for line in text.splitlines() if line.startswith("twins() {")]
    submitted = set()
    for job, block in serving_blocks(text):
        extras = [match.group(1).split()[2:] for match in re.finditer(r'SGLANG_EXTRA="([^"]*)"', block)]
        if CACHE_OFF in [flag for extra in extras for flag in extra] or (re.search(r"^twins ", block, re.M) and CACHE_OFF in twins):
            submitted.add(job)
    heads = list(JOB.finditer(text))
    named = {job for head, end in zip(heads, [match.start() for match in heads[1:]] + [len(text)])
             if CACHE_OFF in text[head.start():end].split("```")[0] for job in re.findall(r"J\d", head.group(1))}
    return submitted, named


def test_the_prefix_cache_is_off_in_the_job_that_says_so_alone():
    """J5 turns the prefix cache of the engine off, which its paragraph says, and its submit carries the flag after the export
    folder; no other job carries it, or the comparison of J5 with J3 measures nothing. The controls plant the wrong lines."""
    assert cache_off(SHEET) == ({"J5"}, {"J5"})
    j5 = SHEET.index("\nJ5, optional")
    j3 = SHEET.index("\nJ3 and J4,")
    forms = {"J5 without the flag": SHEET.replace(f' {CACHE_OFF}" bash', '" bash', 1),
             "J3 with the flag": SHEET[:j3] + SHEET[j3:].replace('SGLANG_EXTRA="$F $D/$C" bash', f'SGLANG_EXTRA="$F $D/$C {CACHE_OFF}" bash', 1),
             "the twins with the flag": SHEET.replace('SGLANG_EXTRA="$F $D/$card"', f'SGLANG_EXTRA="$F $D/$card {CACHE_OFF}"', 1),
             "J5 with the flag before the folder": SHEET[:j5] + SHEET[j5:].replace(f'SGLANG_EXTRA="$F $D/$C {CACHE_OFF}"',
                                                                                  f'SGLANG_EXTRA="{CACHE_OFF} $F $D/$C"', 1)}
    assert all(form != SHEET for form in forms.values())
    assert {name: cache_off(form) for name, form in forms.items()} == {  # the controls
        "J5 without the flag": (set(), {"J5"}), "J3 with the flag": ({"J3", "J5"}, {"J5"}),
        "the twins with the flag": ({"J1", "J2", "J5"}, {"J5"}), "J5 with the flag before the folder": (set(), {"J5"})}
