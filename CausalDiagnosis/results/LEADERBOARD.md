# Leaderboard of the hydraulic rig gym

This page is a status board of the runs and no page of results, and the tables of the paper, in their order and layout, are those of `MAIN_TABLES.md` beside it.

No run has been played on the current code, so the tables below hold no number. They stay empty until the test set is played on Columbia's cluster, and every number then traces to the run folders its row names through `CausalDiagnosis/evaluation/result_tables/tables_from_runs.py`, whose table `RUNS.csv` holds one row per run folder, or per line of the test set over the runs of its replicates, and which `CausalDiagnosis/evaluation/result_tables/leaderboard.py` turns into this page. The run folders stay out of the repository, since they hold sensor readings of single cycles.

- Agent models: none yet
- Scoring: the verifier of the environment on the outcome in the hidden state, and the AssetOpsBench judge on the closing summary as a second scorer (judge models: none yet)
- Runs counted: every run, by the start time of the run
- Written on 2026-11-22 from 0 counted run folders

Pass is the verifier's pass with its 95 percent interval from a bootstrap that resamples start combinations, which on the lines of En4 also needs the read gate of the verifier scope all. The judge success rate is the share of episodes the judge passes, shown once every episode of the run has a verdict. The cost ratio is the mean over the episodes of the cost, prices plus downtime, over the cost of the oracle policy on the same seed. The delay of an episode is the larger of its detection and repair delay. The combinations set plays every start combination under its smallest seed, and the evaluation set plays each at five start states, one run per replicate, the first under a seed drawn with the master seed, so a line counts the episodes of every replicate played.

The main table plays layer 0, the base preset, with the six rows of the agents and the CausalDiagnosis MCP, and the environment table plays the three agents with the MCP on every layer, each row adding one big layer to the row above it.

## Main comparison, no model yet

| Agent | Model | Preset | Episode set | Episodes | Pass rate (95% interval clustered by start) | Judge success rate | Cost / oracle policy cost | Median delay in cycles (episodes with a delay) | Run folder |
|---|---|---|---|---|---|---|---|---|---|
| Plan-and-Execute |  | base | combinations | not run |  |  |  |  |  |
| Plan-and-Execute + CausalDiagnosis MCP |  | base | combinations | not run |  |  |  |  |  |
| Agents-as-Tools |  | base | combinations | not run |  |  |  |  |  |
| Agents-as-Tools + CausalDiagnosis MCP |  | base | combinations | not run |  |  |  |  |  |
| CausalDiagnosis agent |  | base | combinations | not run |  |  |  |  |  |
| CausalDiagnosis agent + CausalDiagnosis MCP |  | base | combinations | not run |  |  |  |  |  |

The oracle policy, which reads the hidden state, is the best of the five reference policies and not a row of this table. On the same set it has not run.

The declaring policy, which reads the cheapest sensor set once, cites that reading and declares insufficient evidence on every start, shows the most a constant declaration can earn and is not a row of this table. On the same set it has not run.

## Environment table, no model yet

| Row of the environment table | Agent | Model | Preset | Episode set | Episodes | Pass rate (95% interval clustered by start) | Judge success rate | Cost / oracle policy cost | Median delay in cycles (episodes with a delay) | Run folder |
|---|---|---|---|---|---|---|---|---|---|---|
| Base | the rows with the CausalDiagnosis MCP of the main table above |  | base | combinations |  |  |  |  |  |  |
| + priced reads and writes | Plan-and-Execute + CausalDiagnosis MCP |  | priced | combinations | not run |  |  |  |  |  |
| + priced reads and writes | Agents-as-Tools + CausalDiagnosis MCP |  | priced | combinations | not run |  |  |  |  |  |
| + priced reads and writes | CausalDiagnosis agent + CausalDiagnosis MCP |  | priced | combinations | not run |  |  |  |  |  |
| + real-time clock | Plan-and-Execute + CausalDiagnosis MCP |  | real_time | combinations | not run |  |  |  |  |  |
| + real-time clock | Agents-as-Tools + CausalDiagnosis MCP |  | real_time | combinations | not run |  |  |  |  |  |
| + real-time clock | CausalDiagnosis agent + CausalDiagnosis MCP |  | real_time | combinations | not run |  |  |  |  |  |
| + advanced real-time clock | Plan-and-Execute + CausalDiagnosis MCP |  | advanced_real_time | combinations | not run |  |  |  |  |  |
| + advanced real-time clock | Agents-as-Tools + CausalDiagnosis MCP |  | advanced_real_time | combinations | not run |  |  |  |  |  |
| + advanced real-time clock | CausalDiagnosis agent + CausalDiagnosis MCP |  | advanced_real_time | combinations | not run |  |  |  |  |  |
| + repair rules | Plan-and-Execute + CausalDiagnosis MCP |  | repair_rules | combinations | not run |  |  |  |  |  |
| + repair rules | Agents-as-Tools + CausalDiagnosis MCP |  | repair_rules | combinations | not run |  |  |  |  |  |
| + repair rules | CausalDiagnosis agent + CausalDiagnosis MCP |  | repair_rules | combinations | not run |  |  |  |  |  |
| + deterioration and failure | Plan-and-Execute + CausalDiagnosis MCP |  | deterioration | combinations | not run |  |  |  |  |  |
| + deterioration and failure | Agents-as-Tools + CausalDiagnosis MCP |  | deterioration | combinations | not run |  |  |  |  |  |
| + deterioration and failure | CausalDiagnosis agent + CausalDiagnosis MCP |  | deterioration | combinations | not run |  |  |  |  |  |

## Ablations, no model yet

| Id | Family | Ablation | Arm | Agent | Model | Preset | Episode set | Episodes | Pass rate (95% interval clustered by start) | Judge success rate | Cost / oracle policy cost | Median delay in cycles (episodes with a delay) | Run folder |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Sc3 | Scoring | the static answering |  |  |  |  | combinations | not run |  |  |  |  |  |
| En4 | Environment | the verifier scope |  |  |  |  | combinations | not run |  |  |  |  |  |
| En5 | Environment | the abstention rule |  |  |  |  | combinations | not run |  |  |  |  |  |
| En6 | Environment | the price table |  |  |  |  | combinations | not run |  |  |  |  |  |
| En7 | Environment | the guidance level |  |  |  |  | combinations | not run |  |  |  |  |  |
| En9 | Environment | the ranking model |  |  |  |  | combinations | not run |  |  |  |  |  |
| L1 | Environment | the misdiagnosis price of layer 1 |  |  |  |  | combinations | not run |  |  |  |  |  |
| L2 | Environment | the thinking time of layer 2 |  |  |  |  | combinations | not run |  |  |  |  |  |
| L3 | Environment | the action durations of layer 3 |  |  |  |  | combinations | not run |  |  |  |  |  |
| L5 | Environment | the deterioration and failure of layer 5 |  |  |  |  | combinations | not run |  |  |  |  |  |
| Ag1 | Agent | the candidate troubleshooting plans |  |  |  |  | combinations | not run |  |  |  |  |  |
| Ag2 | Agent | the number of candidate troubleshooting plans |  |  |  |  | combinations | not run |  |  |  |  |  |
| Ag3 | Agent | the plan length |  |  |  |  | combinations | not run |  |  |  |  |  |
| Ag7 | Agent | the read the board advises |  |  |  |  | combinations | not run |  |  |  |  |  |
| Ag9 | Agent | the schedule of candidate plans |  |  |  |  | combinations | not run |  |  |  |  |  |
| Ag10 | Agent | the structured output of the replies |  |  |  |  | combinations | not run |  |  |  |  |  |

