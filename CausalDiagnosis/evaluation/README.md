# evaluation

The metrics and the programs that turn finished run folders into the result tables.

## What lives here

| Folder | What it holds |
| --- | --- |
| `metrics/` | the statistics behind the tables (confidence intervals, tests and their correction), the measures of the CausalDiagnosis MCP, the realized elimination of hypotheses per paid reading, and `judge_records.py`, which is the record format of the judge and not a metric: the file name `judge.json`, the rule of a reviewed call and the summary of a record over its planned calls, which the tables, the rescoring and the judge of a framework share |
| `result_tables/` | the builder of every table, the tables of the paper as Markdown and LaTeX, the index that says which runs each row comes from, the leaderboard, the difficulty table, the statistics of the datasheet, and the publisher of the readable part of the runs |

The judge of AssetOpsBench, which scores an episode a second time from its trajectory, lives in `frameworks/AssetOpsBench/`.

## How to use it

```bash
python -m CausalDiagnosis tables list
python -m CausalDiagnosis tables <table name> --runs <runs folder> --out <folder outside results/>
python -m CausalDiagnosis tables all
python -m CausalDiagnosis evaluation leaderboard --help
```

`tables all` writes every table, the index and the pages under `results/`. A single table is written alone into another folder, and its file is the same as the one `tables all` writes.
`CausalDiagnosis/docs/metrics.md` defines every metric, and `CausalDiagnosis/docs/experiments.md` describes the tables.

## How to add to it

A metric is defined in `CausalDiagnosis/docs/metrics.md` and computed in `metrics/`. A table is one builder in `result_tables/`, registered under its name, and `python -m CausalDiagnosis tables list` shows it once it is registered. Its runs come from `experiments/table_definitions/`.
