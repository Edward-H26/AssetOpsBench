from CausalDiagnosis.utils.paths import RESULTS
from pathlib import Path
from CausalDiagnosis.utils.csv_rows import write_rows  # noqa: E402
from CausalDiagnosis.experiments.table_definitions import registered_analysis as registered  # noqa: E402
from CausalDiagnosis.evaluation.result_tables import index_file, pooling_rules  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.experiment_tables import AS_RELEASED, TRAINING_MODE  # noqa: E402
from . import episode_files, cell_statistics, family_correction, episode_selection, episode_fields, training_tables, table_declarations, replayed_fields, results_pages


# the first three criteria of the judge prompt (src/evaluation/scorers/llm_judge.py), the three scores of Figure 5 of AssetOpsBench
TABLE_LINE_END = "\n"  # a carriage return at the end of every line of a generated table is what git diff --check flags


def plan_judge_of_record(episodes, kind):
    """In a root of runs of record every played gym episode and every static answer (JUDGED_MODES) of a model agent is to be judged
    by the judge of record (judge_of_record), in the judge job after the grid, so an episode without its record is unjudged under
    counting.unjudged, in the row status and on every judge outcome, whatever its run.json names: a registered grid names no judge in
    run.json, and judge_finished_runs.py judges a static run there in the judge's static mode. Every other root keeps the judge that its runs
    name."""
    if kind != "record":
        return
    served = None
    for e in episodes:
        if not e.get("run_judge_model") and e.get("mode", "gym") in cell_statistics.JUDGED_MODES and e.get("model"):
            served = (episode_fields.judge_of_record().get("served_name") or "") if served is None else served
            e["run_judge_model"] = served


@pooling_rules.reads_once  # the pooling rule reads every run.json once per root for the whole call
def aggregate(runs_root, out_dir, resamples=None, alpha=None, seed=None, readme_path=None, published_root=None, cluster=None,
              index_path=None, pages=True, registered_path=None, departure=False, commit=None, *, fraction, only=None):
    """Every table of tables() as a CSV file of out_dir, the tables of record, with the results index, and with pages the two pages
    of tables_from_runs.py and the main tables of paper_tables/, which is imported only here, so that no module of the list rendering of
    experiment_tables.py can move a table of record. The level of the tests, the resamples and the seed of the bootstrap default to those of
    the registered file. A registered root needs every number of the file that the tables read (registered.check), and its tables are
    refused under other resamples or another seed unless departure names the choice, which INDEX.json then states in words. commit
    names the commit whose runs of record a root of more than one group reads (results_index.select_commit). fraction names the
    fraction of the test set the tables read (read_of), a keyword with no default: the command line holds the default."""
    pooling_rules.select_commit(commit)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    readme_path = Path(readme_path) if readme_path else out_dir.parent / "README.md"
    published_root = Path(published_root) if published_root else out_dir.parent / "published_runs"
    index_path = Path(index_path) if index_path else out_dir.parent / "INDEX.json"
    kind = pooling_rules.refuse_results_folder(runs_root, out_dir, readme_path, index_path)
    registry = table_declarations.tables() if only is None else [table_declarations.table_named(only)]
    episodes = episodes_with_fields(runs_root, {step for table in registry for step in table.needs})
    judges_left_out = episode_fields.foreign_judges(episodes, kind)
    plan_judge_of_record(episodes, kind)
    plan = registered.load(registered_path)  # the registered analysis, whose level, sizes, floor and cluster minimum the tables read
    alpha = plan["level"] if alpha is None else alpha
    resamples = plan["intervals"]["resamples"] if resamples is None else resamples
    seed = plan["intervals"]["seed"] if seed is None else seed
    departed = departure_words(plan, resamples, seed)
    if kind == "record":
        problems = registered.check(plan, kind)
        if problems:
            raise SystemExit("the tables of registered runs read every number of the registered file: " + "; ".join(problems))
        if departed and not departure:
            raise SystemExit(f"{departed}, and the tables of registered runs use the registered values: drop the options or name the departure (--departure)")
    read = episode_selection.read_of(fraction, plan)
    episodes, read_left_out = episode_selection.at_read(episodes, read)
    ctx = {"runs": str(runs_root), "resamples": resamples, "alpha": alpha, "seed": seed, "episodes": len(episodes), "registered": plan,
           "read": read}
    sets = episode_sets(episodes)
    tables_rows, traces = [], []
    for table in registry:
        rows = build_rows(table, sets[table.episodes], ctx)
        write_table(out_dir, table, rows, ctx)
        tables_rows.append((table, rows))
        traces.append([ctx["trace"].get(id(row)) for row in rows])
    ctx.pop("trace")
    if only is not None:
        return tables_rows
    parameters = {"resamples": resamples, "alpha": alpha, "seed": seed, "cluster": cluster, "registered_sha256": registered.sha256(registered_path),
                  "departure": departed, "commit": commit, "fraction": read["text"], "read_left_out": dict(sorted(read_left_out.items()))}
    index = index_file.build(runs_root, tables_rows, traces, parameters, judges_left_out)
    index_file.write(index, index_path)
    if not pages:
        return tables_rows
    opening = "\n".join(pooling_rules.head_lines(kind, index["record"]))
    foot = f"\nThe registered analysis file of these tables has the SHA-256 {parameters['registered_sha256']}.\n"
    (out_dir / "aggregate.md").write_text((opening + "\n" if opening else "") + results_pages.markdown(tables_rows, ctx) + foot, encoding="utf-8")
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text((opening + "\n" if opening else "") + results_pages.results_readme(tables_rows, ctx, Path(runs_root), published_root, cluster), encoding="utf-8")
    from CausalDiagnosis.evaluation.result_tables import paper_tables as main_tables  # here, since it imports leaderboard.py, which imports this module

    main_tables.write(out_dir, readme_path.parent, index_path)
    return tables_rows


@pooling_rules.reads_once
def departure_words(plan, resamples, seed):
    """The sentence that a table set built with other resamples or another seed than the registered file carries in INDEX.json, or
    empty when it uses the registered values."""
    registered_values = plan["intervals"]
    if (resamples, seed) == (registered_values["resamples"], registered_values["seed"]):
        return ""
    return (f"these tables depart from the registered analysis: {resamples} resamples where it registers {registered_values['resamples']}, "
            f"seed {seed} where it registers {registered_values['seed']}")


def episodes_with_fields(runs_root, needs):
    """The episodes under the root with the derived fields, and the realized elimination and the measures of the MCP only when a
    table to build needs them."""
    episodes = episode_fields.add_derived_fields(episode_files.load_episodes(runs_root))
    if table_declarations.ELIMINATION in needs:
        episodes = replayed_fields.add_elimination_fields(episodes)
    if table_declarations.MCP_MEASURES in needs:
        episodes = replayed_fields.add_mcp_fields(episodes)
    return episodes


def episode_sets(episodes):
    """{episode set: its episodes}. The runs of the online loop enter only the tables that read ALL_EPISODES, and the sampling runs
    of fine tuning no table."""
    played = [e for e in episodes if e.get("mode") not in (training_tables.GEPA_ONLINE, TRAINING_MODE)]
    return {table_declarations.ALL_EPISODES: episodes, table_declarations.PLAYED: played, table_declarations.UNTRAINED: [e for e in played if e["training_variant"] == AS_RELEASED]}


def build_rows(table, episodes, ctx):
    """The rows of one table from its episodes, with a fresh trace of the episodes behind each row in ctx; below the fraction of
    record they hold no registered test (untested)."""
    ctx["trace"] = {}
    rows = table.build(episodes, ctx) if episodes else []
    return [family_correction.untested(row) for row in rows] if ctx["read"]["partial"] else rows


def write_table(out_dir, table, rows, ctx):
    """The CSV file of a table and, for a table with a companion, its second file from the rows its builder put in ctx."""
    write_rows(Path(out_dir) / f"{table.id}.csv", rows, table.columns, lineterminator=TABLE_LINE_END)
    if table.companion:
        stem, key, columns = table.companion
        write_rows(Path(out_dir) / f"{stem}.csv", ctx.get(key, []), columns, lineterminator=TABLE_LINE_END)


def build_table(table_id, runs_root, out_dir, resamples=None, alpha=None, seed=None, registered_path=None, departure=False, commit=None,
                *, fraction):
    """One table of record alone, as the same CSV file (and companion) that aggregate() writes for it: aggregate() with only that
    table and the shared steps it needs, which writes no index and no page. So it refuses the results folder of the fork, whose
    index lists every table."""
    out_dir = Path(out_dir)
    results = RESULTS
    if out_dir.resolve() == results or results in out_dir.resolve().parents:
        raise SystemExit(f"{out_dir} lies in {results}, whose index lists every table: build one table into another folder")
    [(_table, rows)] = aggregate(runs_root, out_dir, resamples, alpha, seed, pages=False, registered_path=registered_path, departure=departure,
                                 commit=commit, fraction=fraction, only=table_id)
    return rows
