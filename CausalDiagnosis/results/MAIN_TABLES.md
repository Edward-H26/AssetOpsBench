# Main tables of the hydraulic rig gym

No run has been played on the current code, so the tables below hold no number. They stay empty until the test set is played on Columbia's cluster, and every number then traces to a cell of `tables/AGENTS.csv`, `tables/AGENTS_pairs.csv`, `tables/JUDGE.csv`, `tables/EFFICIENCY.csv`, `tables/THINKING.csv`, `tables/STRATA.csv`, `tables/POLICIES.csv`, `tables/LAYERS_pairs.csv`, `tables/MISDIAGNOSIS_PRICE.csv`, `tables/TRAINING.csv`, `tables/TRAINING_pairs.csv`, `tables/TRAINING_RECORD.csv`, `tables/H2PRIME_pairs.csv`, `tables/H0.csv`, `tables/MCP.csv`, `tables/RESTORED.csv`, `tables/FULL_ENV_pairs.csv`, `tables/POLICY_LAYERS.csv`, `tables/PARALLEL.csv`, `tables/PARALLEL_pairs.csv` or of `offline_mcp.csv`, and `INDEX.json` names its row. The reproduction of AssetOpsBench answers the scenarios of that paper and stands apart. `main_tables.tex` holds the same tables in LaTeX.

## No model yet

### Tables of the body of the paper, one per experiment, the metrics of the AssetOpsBench paper first and the verified outcome after them

#### Agents, the CausalDiagnosis MCP and the CausalDiagnosis environment (`tab:agents`)

Each design without and with the CausalDiagnosis MCP, on the base layer and in the CausalDiagnosis environment with all six big layers on. The first three columns are the metrics of Figure 5 of AssetOpsBench, task completion (TC), data retrieval accuracy (DRA) and result verification (RV), each per episode the mean of five judge calls at temperature 0.3. An episode without a judge record counts as false on every criterion of the judge. The verified outcome follows. Top-1 and Pass decide the hypotheses H1 and H2 on the base layer, and Pass needs the right diagnosis and the restored asset. Pass^5 is the probability that five episodes of one test combination from different initial states all pass, by the estimator of tau-bench, averaged over the test combinations. Every cell is a percentage, the mean over the episodes of the line and, after the sign ±, its standard error from a bootstrap that resamples test combinations. The other measures of the same lines follow in the next table.

| Agent | TC (%) | DRA (%) | RV (%) | Top-1 (%) | Pass (%) | Pass^5 (%) |
|---|---|---|---|---|---|---|
| Plan-and-Execute | no run | no run | no run | no run | no run | no run |
| + CausalDiagnosis MCP | no run | no run | no run | no run | no run | no run |
| + CausalDiagnosis environment | no run | no run | no run | no run | no run | no run |
| + CausalDiagnosis MCP and environment | no run | no run | no run | no run | no run | no run |
| Agents-as-Tools | no run | no run | no run | no run | no run | no run |
| + CausalDiagnosis MCP | no run | no run | no run | no run | no run | no run |
| + CausalDiagnosis environment | no run | no run | no run | no run | no run | no run |
| + CausalDiagnosis MCP and environment | no run | no run | no run | no run | no run | no run |
| CausalDiagnosis agent | no run | no run | no run | no run | no run | no run |
| + CausalDiagnosis MCP | no run | no run | no run | no run | no run | no run |
| + CausalDiagnosis environment | no run | no run | no run | no run | no run | no run |
| + CausalDiagnosis MCP and environment | no run | no run | no run | no run | no run | no run |

#### The measures of the agents (`tab:agentmeasures`)

The lines of the agents table. Exact match is the share of episodes whose diagnosis names exactly the faulty components. Success is the share of episodes that pass or declare insufficient evidence where the declaration earns its credit. Wrong is the share of episodes whose submitted diagnosis is wrong, so an episode that declared or wrote no diagnosis is not wrong. Declared is the share of episodes that declared insufficient evidence. Model calls is the number of calls of the model per episode. Cost / oracle is the cost of the episode over the cost of the oracle policy on the same seed, in steps on the base layer and in money with the layers on. Cost is the cost of the episode in money, the prices paid, the misdiagnosis price, the production loss and the failure penalties. Production loss is the part of that cost that the asset loses in the cycles it runs degraded or stands stopped for a repair. Time to repair is the largest number of cycles from the onset of a fault to the close of the work order that restored its component. Exact match stands against the majority-class baseline that the appendix table of exact match prints per row. Exact match, Success, Wrong and Declared are percentages. Cost, Production loss and Time to repair exist only with the layers on, so a line on the base layer prints − there. Every cell is the mean over the episodes of the line and, after the sign ±, its standard error from a bootstrap that resamples test combinations.

| Agent | Exact match (%) | Success (%) | Wrong (%) | Declared (%) | Model calls | Cost / oracle | Cost | Production loss | Time to repair |
|---|---|---|---|---|---|---|---|---|---|
| Plan-and-Execute | no run | no run | no run | no run | no run | no run | no run | no run | no run |
| + CausalDiagnosis MCP | no run | no run | no run | no run | no run | no run | no run | no run | no run |
| + CausalDiagnosis environment | no run | no run | no run | no run | no run | no run | no run | no run | no run |
| + CausalDiagnosis MCP and environment | no run | no run | no run | no run | no run | no run | no run | no run | no run |
| Agents-as-Tools | no run | no run | no run | no run | no run | no run | no run | no run | no run |
| + CausalDiagnosis MCP | no run | no run | no run | no run | no run | no run | no run | no run | no run |
| + CausalDiagnosis environment | no run | no run | no run | no run | no run | no run | no run | no run | no run |
| + CausalDiagnosis MCP and environment | no run | no run | no run | no run | no run | no run | no run | no run | no run |
| CausalDiagnosis agent | no run | no run | no run | no run | no run | no run | no run | no run | no run |
| + CausalDiagnosis MCP | no run | no run | no run | no run | no run | no run | no run | no run | no run |
| + CausalDiagnosis environment | no run | no run | no run | no run | no run | no run | no run | no run | no run |
| + CausalDiagnosis MCP and environment | no run | no run | no run | no run | no run | no run | no run | no run | no run |

#### The hypotheses H1 and H2 (`tab:hyp`)

The hypotheses H1 on the agent and H2 on the CausalDiagnosis MCP as paired differences of the agents table by test combination and entry offset, each with its 95 percent interval from a bootstrap that resamples test combinations. H1, the CausalDiagnosis agent does better than each AssetOpsBench agent. H2, the CausalDiagnosis MCP helps every agent. A comparison holds when both differences lie above zero and both Holm decisions reject, Holm over the 14 tests of this table. A baseline whose verified pass rate on the test runs lies below 5% is at floor, and H1 against it is uninformative, its tests kept in the family.

|  | Comparison | ΔTop-1 | Holm p | ΔPass | Holm p |
|---|---|---|---|---|---|
| H1a | CausalDiagnosis agent against Plan-and-Execute | no run | no run | no run | no run |
| H1b | CausalDiagnosis agent against Agents-as-Tools | no run | no run | no run | no run |
| H1c | CausalDiagnosis agent + CausalDiagnosis MCP against Plan-and-Execute + CausalDiagnosis MCP | no run | no run | no run | no run |
| H1d | CausalDiagnosis agent + CausalDiagnosis MCP against Agents-as-Tools + CausalDiagnosis MCP | no run | no run | no run | no run |
| H2a | Plan-and-Execute + CausalDiagnosis MCP against Plan-and-Execute | no run | no run | no run | no run |
| H2b | Agents-as-Tools + CausalDiagnosis MCP against Agents-as-Tools | no run | no run | no run | no run |
| H2c | CausalDiagnosis agent + CausalDiagnosis MCP against CausalDiagnosis agent | no run | no run | no run | no run |

#### The cumulative layer study (`tab:ladder`)

Each row adds one big layer to the row above it within one agent with the CausalDiagnosis MCP. The first three columns are the metrics of Table 16 of AssetOpsBench, TC, DRA and RV, and the verified outcome follows, where cost over the cost of the oracle counts calls on Base and prices from the next row on. Each cell is the mean over the episodes of the line, a percentage in every column but the cost over the oracle, and, with the sign ±, its standard error from a bootstrap that resamples test combinations. An episode without a judge record counts as false on every criterion of the judge. The Base rows are the runs of the agents table.

| Agent | Setting | TC (%) | DRA (%) | RV (%) | Top-1 (%) | Pass (%) | Cost / oracle |
|---|---|---|---|---|---|---|---|
| Plan-and-Execute + CausalDiagnosis MCP | Base | no run | no run | no run | no run | no run | no run |
| Plan-and-Execute + CausalDiagnosis MCP | + priced reads and writes | no run | no run | no run | no run | no run | no run |
| Plan-and-Execute + CausalDiagnosis MCP | + real-time clock | no run | no run | no run | no run | no run | no run |
| Plan-and-Execute + CausalDiagnosis MCP | + advanced real-time clock | no run | no run | no run | no run | no run | no run |
| Plan-and-Execute + CausalDiagnosis MCP | + repair rules | no run | no run | no run | no run | no run | no run |
| Plan-and-Execute + CausalDiagnosis MCP | + deterioration and failure | no run | no run | no run | no run | no run | no run |
| Agents-as-Tools + CausalDiagnosis MCP | Base | no run | no run | no run | no run | no run | no run |
| Agents-as-Tools + CausalDiagnosis MCP | + priced reads and writes | no run | no run | no run | no run | no run | no run |
| Agents-as-Tools + CausalDiagnosis MCP | + real-time clock | no run | no run | no run | no run | no run | no run |
| Agents-as-Tools + CausalDiagnosis MCP | + advanced real-time clock | no run | no run | no run | no run | no run | no run |
| Agents-as-Tools + CausalDiagnosis MCP | + repair rules | no run | no run | no run | no run | no run | no run |
| Agents-as-Tools + CausalDiagnosis MCP | + deterioration and failure | no run | no run | no run | no run | no run | no run |
| CausalDiagnosis agent + CausalDiagnosis MCP | Base | no run | no run | no run | no run | no run | no run |
| CausalDiagnosis agent + CausalDiagnosis MCP | + priced reads and writes | no run | no run | no run | no run | no run | no run |
| CausalDiagnosis agent + CausalDiagnosis MCP | + real-time clock | no run | no run | no run | no run | no run | no run |
| CausalDiagnosis agent + CausalDiagnosis MCP | + advanced real-time clock | no run | no run | no run | no run | no run | no run |
| CausalDiagnosis agent + CausalDiagnosis MCP | + repair rules | no run | no run | no run | no run | no run | no run |
| CausalDiagnosis agent + CausalDiagnosis MCP | + deterioration and failure | no run | no run | no run | no run | no run | no run |

#### Training in the environment (`tab:training`)

All six big layers on, the metrics of AssetOpsBench over every test episode and the verified outcome over every episode and on the test combinations with a single fault and with two faults, under cross-fitting, where every test episode is played by the model that holds out the fold of its fault combination. GEPA on the score alone runs the loop of GEPA on the base weights with feedback that carries the score alone. Fine-tuning is LoRA rejection sampling fine-tuning of the model with the filter on the verified pass, and the control, fine-tuning on unfiltered trajectories, trains on as many episodes. A dagger marks a cell whose episodes include a fold on which the filter kept fewer episodes than the model has training initial states, after one more round of sampling, so that no adapter was trained and the variant reads the untrained runs of that fold, with paired differences of zero there. The first row of each agent plays it without the CausalDiagnosis MCP and without training, its reference line on all six big layers on the same episodes, the second row with the CausalDiagnosis MCP without training on the server that holds the LoRA adapters of its trained variants, and the rows under it its trained variants. An episode without a judge record counts as false on every criterion of the judge. Judge kappa and Unmatched read every test episode of the row. Each cell is the mean over the episodes, a percentage in the columns of the judge, of Top-1 and of Pass, and, with the sign ±, its standard error from a bootstrap that resamples test combinations.

| Agent | TC (%) | DRA (%) | RV (%) | Judge κ | Top-1 all (%) | Top-1 single-fault (%) | Top-1 two-fault (%) | Pass all (%) | Pass single-fault (%) | Pass two-fault (%) | Unmatched |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Plan-and-Execute | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run |
| Plan-and-Execute + CausalDiagnosis MCP | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run |
| + GEPA | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run |
| + GEPA on the score alone | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run |
| + fine-tuning | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run |
| + fine-tuning + GEPA | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run |
| + fine-tuning on unfiltered trajectories | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run |
| Agents-as-Tools | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run |
| Agents-as-Tools + CausalDiagnosis MCP | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run |
| + GEPA | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run |
| + GEPA on the score alone | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run |
| + fine-tuning | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run |
| + fine-tuning + GEPA | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run |
| + fine-tuning on unfiltered trajectories | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run |
| CausalDiagnosis agent | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run |
| CausalDiagnosis agent + CausalDiagnosis MCP | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run |
| + GEPA | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run |
| + GEPA on the score alone | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run |
| + fine-tuning | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run |
| + fine-tuning + GEPA | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run |
| + fine-tuning on unfiltered trajectories | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run |

### Tables of the appendix of the paper

#### Execution efficiency (`tab:efficiency`)

Agents and the CausalDiagnosis MCP on layer 0 in the metrics of Table 3 of AssetOpsBench, execution efficiency as mean ± standard deviation over episodes. Steps count every attempted tool call, one rule for every agent, of the action or metered servers and of the CausalDiagnosis MCP, failed and refused calls included, and the test combinations with a single fault and with two faults take the place of its single-agent and multi-agent tasks.

| Agent | Single fault steps | Single fault runtime (s) | Two faults steps | Two faults runtime (s) |
|---|---|---|---|---|
| Plan-and-Execute | no run | no run | no run | no run |
| Plan-and-Execute + CausalDiagnosis MCP | no run | no run | no run | no run |
| Agents-as-Tools | no run | no run | no run | no run |
| Agents-as-Tools + CausalDiagnosis MCP | no run | no run | no run | no run |
| CausalDiagnosis agent | no run | no run | no run | no run |
| CausalDiagnosis agent + CausalDiagnosis MCP | no run | no run | no run | no run |

#### Exact match and its majority-class baseline (`tab:exactmatch`)

Layer 0, the rows of the agents table. Exact match is the share of the episodes of a row whose diagnosis names exactly the faulty components, the mean and, with the sign ±, its standard error. The majority-class baseline is the share of the episodes of the row that fall on the most frequent faulty set of the test combinations it played, the rate of an agent that always names that set. The floor of a row is the larger of that baseline and the floor of Pass, and a row at or below its floor is marked.

| Agent | Exact match | Majority-class baseline | Floor | At floor |
|---|---|---|---|---|
| Plan-and-Execute | no run | no run | no run | no run |
| Plan-and-Execute + CausalDiagnosis MCP | no run | no run | no run | no run |
| Agents-as-Tools | no run | no run | no run | no run |
| Agents-as-Tools + CausalDiagnosis MCP | no run | no run | no run | no run |
| CausalDiagnosis agent | no run | no run | no run | no run |
| CausalDiagnosis agent + CausalDiagnosis MCP | no run | no run | no run | no run |

#### Thinking time charged and where its seconds went (`tab:thinking`)

Every row played in the mode that charges the wall seconds between tool calls as thinking time, one line per agent and setting: the cycles charged per episode, the seconds charged behind them, and of these the seconds of the model's generation, the sleep of the machine or a step of the wall clock, measured as the wall gap minus the monotonic gap, which cannot see a stopped process since both clocks run while a process is stopped, and the rest, serving, tools and the harness, each a mean over the episodes. A dagger marks a line of a trained variant whose fold trained no adapter and reads the runs of no training there. The setting of a line with the CausalDiagnosis MCP is its training variant on the preset of training, and a line of the last group plays an arm of the layer that charges thinking time.

| Agent | Setting | Cycles charged | Seconds charged | Generation (s) | Wall minus monotonic (s) | Rest (s) |
|---|---|---|---|---|---|---|
| Plan-and-Execute + CausalDiagnosis MCP | + real-time clock | no run | no run | no run | no run | no run |
| Plan-and-Execute + CausalDiagnosis MCP | + advanced real-time clock | no run | no run | no run | no run | no run |
| Plan-and-Execute + CausalDiagnosis MCP | + repair rules | no run | no run | no run | no run | no run |
| Plan-and-Execute + CausalDiagnosis MCP | + deterioration and failure | no run | no run | no run | no run | no run |
| Agents-as-Tools + CausalDiagnosis MCP | + real-time clock | no run | no run | no run | no run | no run |
| Agents-as-Tools + CausalDiagnosis MCP | + advanced real-time clock | no run | no run | no run | no run | no run |
| Agents-as-Tools + CausalDiagnosis MCP | + repair rules | no run | no run | no run | no run | no run |
| Agents-as-Tools + CausalDiagnosis MCP | + deterioration and failure | no run | no run | no run | no run | no run |
| CausalDiagnosis agent + CausalDiagnosis MCP | + real-time clock | no run | no run | no run | no run | no run |
| CausalDiagnosis agent + CausalDiagnosis MCP | + advanced real-time clock | no run | no run | no run | no run | no run |
| CausalDiagnosis agent + CausalDiagnosis MCP | + repair rules | no run | no run | no run | no run | no run |
| CausalDiagnosis agent + CausalDiagnosis MCP | + deterioration and failure | no run | no run | no run | no run | no run |
| Plan-and-Execute + CausalDiagnosis MCP | no training | no run | no run | no run | no run | no run |
| Plan-and-Execute + CausalDiagnosis MCP | GEPA | no run | no run | no run | no run | no run |
| Plan-and-Execute + CausalDiagnosis MCP | GEPA on the score alone | no run | no run | no run | no run | no run |
| Plan-and-Execute + CausalDiagnosis MCP | fine-tuning | no run | no run | no run | no run | no run |
| Plan-and-Execute + CausalDiagnosis MCP | fine-tuning + GEPA | no run | no run | no run | no run | no run |
| Plan-and-Execute + CausalDiagnosis MCP | fine-tuning on unfiltered trajectories | no run | no run | no run | no run | no run |
| Agents-as-Tools + CausalDiagnosis MCP | no training | no run | no run | no run | no run | no run |
| Agents-as-Tools + CausalDiagnosis MCP | GEPA | no run | no run | no run | no run | no run |
| Agents-as-Tools + CausalDiagnosis MCP | GEPA on the score alone | no run | no run | no run | no run | no run |
| Agents-as-Tools + CausalDiagnosis MCP | fine-tuning | no run | no run | no run | no run | no run |
| Agents-as-Tools + CausalDiagnosis MCP | fine-tuning + GEPA | no run | no run | no run | no run | no run |
| Agents-as-Tools + CausalDiagnosis MCP | fine-tuning on unfiltered trajectories | no run | no run | no run | no run | no run |
| CausalDiagnosis agent + CausalDiagnosis MCP | no training | no run | no run | no run | no run | no run |
| CausalDiagnosis agent + CausalDiagnosis MCP | GEPA | no run | no run | no run | no run | no run |
| CausalDiagnosis agent + CausalDiagnosis MCP | GEPA on the score alone | no run | no run | no run | no run | no run |
| CausalDiagnosis agent + CausalDiagnosis MCP | fine-tuning | no run | no run | no run | no run | no run |
| CausalDiagnosis agent + CausalDiagnosis MCP | fine-tuning + GEPA | no run | no run | no run | no run | no run |
| CausalDiagnosis agent + CausalDiagnosis MCP | fine-tuning on unfiltered trajectories | no run | no run | no run | no run | no run |
| Plan-and-Execute | the thinking time at twice its price | no run | no run | no run | no run | no run |
| Plan-and-Execute | the thinking time at half its price | no run | no run | no run | no run | no run |
| Agents-as-Tools | the thinking time at twice its price | no run | no run | no run | no run | no run |
| Agents-as-Tools | the thinking time at half its price | no run | no run | no run | no run | no run |
| CausalDiagnosis agent | the thinking time at twice its price | no run | no run | no run | no run | no run |
| CausalDiagnosis agent | the thinking time at half its price | no run | no run | no run | no run | no run |

#### What each big layer changes (`tab:layers`)

The paired difference of Pass between a layer and the layer below it for the three agents with the CausalDiagnosis MCP, by test combination and entry offset for the pooled difference, with its 95 percent interval. A layer has an effect when the difference pooled over the three agents differs from zero, Holm over the five layers. A test whose layer below has a pooled Pass that lies below the floor of Pass is uninformative.

| Layer added | ΔPass Plan-and-Execute + CausalDiagnosis MCP | ΔPass Agents-as-Tools + CausalDiagnosis MCP | ΔPass CausalDiagnosis agent + CausalDiagnosis MCP | pooled ΔPass | Holm p |
|---|---|---|---|---|---|
| + priced reads and writes | no run | no run | no run | no run | no run |
| + real-time clock | no run | no run | no run | no run | no run |
| + advanced real-time clock | no run | no run | no run | no run | no run |
| + repair rules | no run | no run | no run | no run | no run |
| + deterioration and failure | no run | no run | no run | no run | no run |

#### The misdiagnosis price at the budget (`tab:misdiagnosis`)

Each design without the CausalDiagnosis MCP on all six big layers, at the misdiagnosis price of the asset file and at a misdiagnosis price equal to the budget of the asset. Top-1 is the share of episodes whose diagnosis is right for every component. Pass is the share of episodes that pass. Wrong is the share of episodes whose submitted diagnosis is wrong. Declared is the share of episodes that declared insufficient evidence. Cycles read is the number of cycles the reads served before the first diagnosis. Cost without the charge is the cost of the episode without the misdiagnosis charge, so that both prices stand on one scale. Charge is the misdiagnosis charge of the episode. Top-1, Pass, Wrong and Declared are percentages. Every cell is the mean over the episodes of the line and, after the sign ±, its standard error from a bootstrap that resamples test combinations. The table describes the ablation and runs no test.

| Agent | Misdiagnosis price | Top-1 (%) | Pass (%) | Wrong (%) | Declared (%) | Cycles read | Cost without the charge | Charge |
|---|---|---|---|---|---|---|---|---|
| Plan-and-Execute | Asset file | no run | no run | no run | no run | no run | no run | no run |
| Plan-and-Execute | Budget | no run | no run | no run | no run | no run | no run | no run |
| Agents-as-Tools | Asset file | no run | no run | no run | no run | no run | no run | no run |
| Agents-as-Tools | Budget | no run | no run | no run | no run | no run | no run | no run |
| CausalDiagnosis agent | Asset file | no run | no run | no run | no run | no run | no run | no run |
| CausalDiagnosis agent | Budget | no run | no run | no run | no run | no run | no run | no run |

#### The full environment against the base layer (`tab:fullenv`)

For each agent with the CausalDiagnosis MCP, the paired difference of Pass between its line without training in the training table, all six big layers on, and its line of the agents table on layer 0, on the same test combinations, entry offsets and seeds, with its 95 percent interval, the exact two-sided sign-flip p value over test combinations, Holm over the tests of this table and the reading, which gives the sign of a difference that holds. A test whose base line lies below the floor of Pass is uninformative.

|  | Agent | ΔPass | p | Holm p | Reading |
|---|---|---|---|---|---|
| F_a | Plan-and-Execute + CausalDiagnosis MCP | no run | no run | no run | no run |
| F_b | Agents-as-Tools + CausalDiagnosis MCP | no run | no run | no run | no run |
| F_c | CausalDiagnosis agent + CausalDiagnosis MCP | no run | no run | no run | no run |

#### The reference policies on the layers (`tab:policies`)

The policies need no model. The oracle knows the hidden state and must remain the best of the five reference policies, and doing nothing must score below every policy that passes, on every initial state of every row. Both rules cover the five reference policies on the rig's action server. The MCP follower plays AssetOpsBench's servers and submits the root cause that the CausalDiagnosis MCP ranks first, and its two columns are its pass and its reward. Both columns of pass are percentages.

| Setting | Oracle pass (%) | Oracle reward | Greedy reference reward | Random reward | Do nothing reward | Always declare reward | MCP follower pass (%) | MCP follower reward |
|---|---|---|---|---|---|---|---|---|
| Base | no run | no run | no run | no run | no run | no run | no run | no run |
| + priced reads and writes | no run | no run | no run | no run | no run | no run | no run | no run |
| + real-time clock | no run | no run | no run | no run | no run | no run | no run | no run |
| + advanced real-time clock | no run | no run | no run | no run | no run | no run | no run | no run |
| + repair rules | no run | no run | no run | no run | no run | no run | no run | no run |
| + deterioration and failure | no run | no run | no run | no run | no run | no run | no run | no run |

#### What each big layer changes for the reference policies (`tab:policylayers`)

The paired difference of Pass, success and reward between a layer and the layer below it for each reference policy, matched on seed and every factor but the preset and the clock mode, with its 95 percent interval. The policies need no model, so the effect of each single layer reads without the noise of an agent. The table tests nothing.

| Policy | Layer added | ΔPass | ΔSuccess | ΔReward |
|---|---|---|---|---|
| Oracle | + priced reads and writes | no run | no run | no run |
| Oracle | + real-time clock | no run | no run | no run |
| Oracle | + advanced real-time clock | no run | no run | no run |
| Oracle | + repair rules | no run | no run | no run |
| Oracle | + deterioration and failure | no run | no run | no run |
| Greedy reference | + priced reads and writes | no run | no run | no run |
| Greedy reference | + real-time clock | no run | no run | no run |
| Greedy reference | + advanced real-time clock | no run | no run | no run |
| Greedy reference | + repair rules | no run | no run | no run |
| Greedy reference | + deterioration and failure | no run | no run | no run |
| Random | + priced reads and writes | no run | no run | no run |
| Random | + real-time clock | no run | no run | no run |
| Random | + advanced real-time clock | no run | no run | no run |
| Random | + repair rules | no run | no run | no run |
| Random | + deterioration and failure | no run | no run | no run |
| Do nothing | + priced reads and writes | no run | no run | no run |
| Do nothing | + real-time clock | no run | no run | no run |
| Do nothing | + advanced real-time clock | no run | no run | no run |
| Do nothing | + repair rules | no run | no run | no run |
| Do nothing | + deterioration and failure | no run | no run | no run |
| Always declare | + priced reads and writes | no run | no run | no run |
| Always declare | + real-time clock | no run | no run | no run |
| Always declare | + advanced real-time clock | no run | no run | no run |
| Always declare | + repair rules | no run | no run | no run |
| Always declare | + deterioration and failure | no run | no run | no run |

#### The training hypotheses (`tab:trainhyp`)

Paired by test combination and entry offset over every test episode, each played by the model that holds out the fold of its fault combination, with Holm over the 30 tests of this table, and the paired differences on the test combinations with a single fault and with two faults beside each test, without a test. The second comparison of each agent asks whether a gain of GEPA comes from the feedback of the verifier or only from the reflection model, and the last whether a gain of fine-tuning comes from the signal of the environment or only from more exposure to the task.

| Agent | Comparison | ΔTop-1, all episodes | Holm p | ΔPass, all episodes | Holm p | ΔTop-1 single-fault | ΔTop-1 two-fault | ΔPass single-fault | ΔPass two-fault |
|---|---|---|---|---|---|---|---|---|---|
| Plan-and-Execute + CausalDiagnosis MCP | GEPA against no training | no run | no run | no run | no run | no run | no run | no run | no run |
| Plan-and-Execute + CausalDiagnosis MCP | GEPA against GEPA on the score alone | no run | no run | no run | no run | no run | no run | no run | no run |
| Plan-and-Execute + CausalDiagnosis MCP | fine-tuning against no training | no run | no run | no run | no run | no run | no run | no run | no run |
| Plan-and-Execute + CausalDiagnosis MCP | fine-tuning + GEPA against no training | no run | no run | no run | no run | no run | no run | no run | no run |
| Plan-and-Execute + CausalDiagnosis MCP | fine-tuning against fine-tuning on unfiltered trajectories | no run | no run | no run | no run | no run | no run | no run | no run |
| Agents-as-Tools + CausalDiagnosis MCP | GEPA against no training | no run | no run | no run | no run | no run | no run | no run | no run |
| Agents-as-Tools + CausalDiagnosis MCP | GEPA against GEPA on the score alone | no run | no run | no run | no run | no run | no run | no run | no run |
| Agents-as-Tools + CausalDiagnosis MCP | fine-tuning against no training | no run | no run | no run | no run | no run | no run | no run | no run |
| Agents-as-Tools + CausalDiagnosis MCP | fine-tuning + GEPA against no training | no run | no run | no run | no run | no run | no run | no run | no run |
| Agents-as-Tools + CausalDiagnosis MCP | fine-tuning against fine-tuning on unfiltered trajectories | no run | no run | no run | no run | no run | no run | no run | no run |
| CausalDiagnosis agent + CausalDiagnosis MCP | GEPA against no training | no run | no run | no run | no run | no run | no run | no run | no run |
| CausalDiagnosis agent + CausalDiagnosis MCP | GEPA against GEPA on the score alone | no run | no run | no run | no run | no run | no run | no run | no run |
| CausalDiagnosis agent + CausalDiagnosis MCP | fine-tuning against no training | no run | no run | no run | no run | no run | no run | no run | no run |
| CausalDiagnosis agent + CausalDiagnosis MCP | fine-tuning + GEPA against no training | no run | no run | no run | no run | no run | no run | no run | no run |
| CausalDiagnosis agent + CausalDiagnosis MCP | fine-tuning against fine-tuning on unfiltered trajectories | no run | no run | no run | no run | no run | no run | no run | no run |

#### The training hypotheses by fold (`tab:trainfolds`)

The paired differences of the training hypotheses on each fold of the cross-fitting, without a test. Fold k holds the test episodes whose fault combination lies in fold k, each played by the model that holds fold k out.

| Agent | Comparison | ΔTop-1 fold 1 | ΔTop-1 fold 2 | ΔPass fold 1 | ΔPass fold 2 |
|---|---|---|---|---|---|
| Plan-and-Execute + CausalDiagnosis MCP | GEPA against no training | no run | no run | no run | no run |
| Plan-and-Execute + CausalDiagnosis MCP | GEPA against GEPA on the score alone | no run | no run | no run | no run |
| Plan-and-Execute + CausalDiagnosis MCP | fine-tuning against no training | no run | no run | no run | no run |
| Plan-and-Execute + CausalDiagnosis MCP | fine-tuning + GEPA against no training | no run | no run | no run | no run |
| Plan-and-Execute + CausalDiagnosis MCP | fine-tuning against fine-tuning on unfiltered trajectories | no run | no run | no run | no run |
| Agents-as-Tools + CausalDiagnosis MCP | GEPA against no training | no run | no run | no run | no run |
| Agents-as-Tools + CausalDiagnosis MCP | GEPA against GEPA on the score alone | no run | no run | no run | no run |
| Agents-as-Tools + CausalDiagnosis MCP | fine-tuning against no training | no run | no run | no run | no run |
| Agents-as-Tools + CausalDiagnosis MCP | fine-tuning + GEPA against no training | no run | no run | no run | no run |
| Agents-as-Tools + CausalDiagnosis MCP | fine-tuning against fine-tuning on unfiltered trajectories | no run | no run | no run | no run |
| CausalDiagnosis agent + CausalDiagnosis MCP | GEPA against no training | no run | no run | no run | no run |
| CausalDiagnosis agent + CausalDiagnosis MCP | GEPA against GEPA on the score alone | no run | no run | no run | no run |
| CausalDiagnosis agent + CausalDiagnosis MCP | fine-tuning against no training | no run | no run | no run | no run |
| CausalDiagnosis agent + CausalDiagnosis MCP | fine-tuning + GEPA against no training | no run | no run | no run | no run |
| CausalDiagnosis agent + CausalDiagnosis MCP | fine-tuning against fine-tuning on unfiltered trajectories | no run | no run | no run | no run |

#### What each agent trained on (`tab:budget`)

The same budget for all three, one line per fold of the cross-fitting, whose model holds that fold out and trained on the initial states of the combinations whose faults all lie outside it, at entry offsets that no test episode plays, and a loop of GEPA plays those initial states alone, past the seeds of the sampling. The rounds sampled, the episodes sampled and kept by the filter and the distinct training initial states among the kept come from the gate of fine-tuning, which samples a second round when the first kept fewer episodes than the model has training initial states and trains no adapter when both rounds together keep fewer.

| Agent | Fold | rounds sampled | episodes sampled | kept by the filter | distinct training initial states | model calls as examples | GEPA candidates |
|---|---|---|---|---|---|---|---|
| Plan-and-Execute + CausalDiagnosis MCP | 1 | no record | no record | no record | no record | no record | no record |
| Plan-and-Execute + CausalDiagnosis MCP | 2 | no record | no record | no record | no record | no record | no record |
| Agents-as-Tools + CausalDiagnosis MCP | 1 | no record | no record | no record | no record | no record | no record |
| Agents-as-Tools + CausalDiagnosis MCP | 2 | no record | no record | no record | no record | no record | no record |
| CausalDiagnosis agent + CausalDiagnosis MCP | 1 | no record | no record | no record | no record | no record | no record |
| CausalDiagnosis agent + CausalDiagnosis MCP | 2 | no record | no record | no record | no record | no record | no record |

#### Training episodes through a test combination (`tab:hazard`)

The training episodes of each trained variant whose hidden log passes through a combination that holds a single fault of the fold its model holds out, a path that only the random hazard of deterioration opens, counted from the hidden records and left out of the training data before the filter, the control and every loop of GEPA, one line per agent and fold: the sampled episodes for fine-tuning and its control, fine-tuning on unfiltered trajectories, and the stream of its loop for each variant of GEPA.

| Agent | Fold | GEPA | GEPA on the score alone | fine-tuning | fine-tuning + GEPA | fine-tuning on unfiltered trajectories |
|---|---|---|---|---|---|---|
| Plan-and-Execute + CausalDiagnosis MCP | 1 | no record | no record | no record | no record | no record |
| Plan-and-Execute + CausalDiagnosis MCP | 2 | no record | no record | no record | no record | no record |
| Agents-as-Tools + CausalDiagnosis MCP | 1 | no record | no record | no record | no record | no record |
| Agents-as-Tools + CausalDiagnosis MCP | 2 | no record | no record | no record | no record | no record |
| CausalDiagnosis agent + CausalDiagnosis MCP | 1 | no record | no record | no record | no record | no record |
| CausalDiagnosis agent + CausalDiagnosis MCP | 2 | no record | no record | no record | no record | no record |

#### The hypotheses H2' on the top layer (`tab:mcptop`)

All six big layers on, each agent with the CausalDiagnosis MCP as released against itself without it, as paired differences of pass and of the verifier's reward by test combination and entry offset. A comparison holds when every difference favors the CausalDiagnosis MCP and every Holm decision rejects, with Holm run once over the registered size of the group that joins these tests to those of H1 on the top layer.

|  | Comparison | ΔPass | Holm p | ΔReward | Holm p |
|---|---|---|---|---|---|
| H2'a | Plan-and-Execute + CausalDiagnosis MCP against Plan-and-Execute | no run | no run | no run | no run |
| H2'b | Agents-as-Tools + CausalDiagnosis MCP against Agents-as-Tools | no run | no run | no run | no run |
| H2'c | CausalDiagnosis agent + CausalDiagnosis MCP against CausalDiagnosis agent | no run | no run | no run | no run |

#### The judge against the verified pass, per row (`tab:judgerows`)

Layer 0, the judge's majority under the protocol of AssetOpsBench against the verified pass of the same episodes. An episode without a judge record counts as false on every criterion of the judge. The difference is the share the judge passes minus the share the verifier passes, Holm over the row and comparison tests of the judge together. A test of a row that misses a planned episode, with an episode judged under another protocol or without any judge record, or with too few initial states on which the two disagree counts with p equal to one. Kappa, the discordance and the false pass and false fail rates of the judge decide nothing.

| Agent | ΔJudge - verified | Holm p | Judge κ | Discordance | False pass | False fail |
|---|---|---|---|---|---|---|
| Plan-and-Execute | no run | no run | no run | no run | no run | no run |
| Plan-and-Execute + CausalDiagnosis MCP | no run | no run | no run | no run | no run | no run |
| Agents-as-Tools | no run | no run | no run | no run | no run | no run |
| Agents-as-Tools + CausalDiagnosis MCP | no run | no run | no run | no run | no run | no run |
| CausalDiagnosis agent | no run | no run | no run | no run | no run | no run |
| CausalDiagnosis agent + CausalDiagnosis MCP | no run | no run | no run | no run | no run | no run |

#### The judge against the verified pass, per comparison (`tab:judgecmp`)

Layer 0, the comparisons of H1 and H2 scored twice on the same matched episodes. An episode without a judge record counts as false on every criterion of the judge. The difference is the judge's paired difference minus the verified one, in the same Holm family as the row tests, and the judge reverses a comparison when the two paired differences have opposite signs.

|  | Comparison | ΔJudge - verified | Holm p | ΔPass judge | ΔPass verified | Reversed |
|---|---|---|---|---|---|---|
| H1a | CausalDiagnosis agent against Plan-and-Execute | no run | no run | no run | no run | no run |
| H1b | CausalDiagnosis agent against Agents-as-Tools | no run | no run | no run | no run | no run |
| H1c | CausalDiagnosis agent + CausalDiagnosis MCP against Plan-and-Execute + CausalDiagnosis MCP | no run | no run | no run | no run | no run |
| H1d | CausalDiagnosis agent + CausalDiagnosis MCP against Agents-as-Tools + CausalDiagnosis MCP | no run | no run | no run | no run | no run |
| H2a | Plan-and-Execute + CausalDiagnosis MCP against Plan-and-Execute | no run | no run | no run | no run | no run |
| H2b | Agents-as-Tools + CausalDiagnosis MCP against Agents-as-Tools | no run | no run | no run | no run | no run |
| H2c | CausalDiagnosis agent + CausalDiagnosis MCP against CausalDiagnosis agent | no run | no run | no run | no run | no run |

#### The judge against four constructs of the verifier (`tab:judgeagree`)

Layer 0, the rows of the agents table, the majority of the judge under the protocol of AssetOpsBench against four constructs of the verifier: Cohen's kappa against Pass, Top-1, exact match and restored with no unnecessary order, each over the judged episodes and, with the sign ±, its standard error, none when either rule puts every episode on one side. Cut is the share of the judged episodes whose judge input was cut to the context of the served judge, and Passed unrecovered the share of the judged episodes that the judge passes while a component ends off its healthy level. Unmatched is the share of episodes that cited an evidence identifier the released evidence does not hold, and Restored the share of episodes that end with every component at its healthy level and no order on a healthy component. A row that two judge models scored prints one line per judge.

| Agent | κ Pass | κ Top-1 | κ Exact match | κ Restored | Cut | Passed unrecovered | Unmatched | Restored |
|---|---|---|---|---|---|---|---|---|
| Plan-and-Execute | no run | no run | no run | no run | no run | no run | no run | no run |
| Plan-and-Execute + CausalDiagnosis MCP | no run | no run | no run | no run | no run | no run | no run | no run |
| Agents-as-Tools | no run | no run | no run | no run | no run | no run | no run | no run |
| Agents-as-Tools + CausalDiagnosis MCP | no run | no run | no run | no run | no run | no run | no run | no run |
| CausalDiagnosis agent | no run | no run | no run | no run | no run | no run | no run | no run |
| CausalDiagnosis agent + CausalDiagnosis MCP | no run | no run | no run | no run | no run | no run | no run | no run |

#### Success and declarations by stratum (`tab:strata`)

Layer 0, a secondary table. The label is the separability of the reference policy, E18 the first commitment majority of the CausalDiagnosis MCP, and a declaration earns its credit only where both mark the initial state inseparable.

| Agent | Label separable success | Label separable declared | E18 separable success | E18 separable declared | Both inseparable success | Both inseparable declared |
|---|---|---|---|---|---|---|
| Plan-and-Execute | no run | no run | no run | no run | no run | no run |
| Plan-and-Execute + CausalDiagnosis MCP | no run | no run | no run | no run | no run | no run |
| Agents-as-Tools | no run | no run | no run | no run | no run | no run |
| Agents-as-Tools + CausalDiagnosis MCP | no run | no run | no run | no run | no run | no run |
| CausalDiagnosis agent | no run | no run | no run | no run | no run | no run |
| CausalDiagnosis agent + CausalDiagnosis MCP | no run | no run | no run | no run | no run | no run |

#### The CausalDiagnosis MCP offline (`tab:mcpoffline`)

No model and no agent, on the episodes of the test set with the five recordings that precede the initial state served before the live readings. Every initial state is scored by fits that leave out its block, the predicted top-1 probability is the mean probability that the CausalDiagnosis MCP gives its first candidate, printed beside the top-1 rate, exact wrong counts the episodes whose first candidate is not the true combination, the fifth column is the share of the released summaries that lie inside the E14 prediction interval of the true combination, and the last column is the share of the wrong repairs whose first reading after the repair E17 finds inconsistent.

| Live readings | Predicted top-1 probability | Top-1 rate | Exact wrong | E14 prediction interval covers the truth | E17 rejects a wrong repair |
|---|---|---|---|---|---|
| 0 | no run | no run | no run | no run | no run |
| 1 | no run | no run | no run | no run | no run |
| 2 | no run | no run | no run | no run | no run |
| 3 | no run | no run | no run | no run | no run |
| 5 | no run | no run | no run | no run | no run |
| 10 | no run | no run | no run | no run | no run |

#### What the CausalDiagnosis MCP adds and risks (`tab:mcpvalue`)

Descriptive measures of the CausalDiagnosis MCP for every agent that has it, on the preset with all six big layers and on layer 0. Each cell pools the visits or the plans of the episodes of its row, with intervals from a bootstrap that resamples the test combinations. Uptake is the share of the visits after a ranking of the CausalDiagnosis MCP that order the candidate plan of the first candidate of the last ranking before them. Harm is the share of those visits whose plan differs from the faulty set of the hidden combination at the visit. The Brier skill of the recovery rate prediction is taken against a predictor at the base rate and scored at the reading that verifies the plan. The error of the cost prediction is the predicted cost of acting now minus the cost of the episode from the first order of the plan until the asset is restored, divided by the cost of the oracle on the same seed. Its skill is taken against a prediction of the price and the downtime of the plan alone. The last column is the paired difference of the production loss in the cycles charged to thinking, the agent with the CausalDiagnosis MCP minus the same agent without it on the same episodes. Layer 0 prices nothing and charges no thinking time, so its cost columns and its last column print none.

| Setting | Agent | Uptake | Harm | Recovery rate prediction Brier skill | Cost prediction bias | Cost prediction absolute error | Cost prediction skill over price | ΔProduction loss while thinking |
|---|---|---|---|---|---|---|---|---|
| All six big layers | Plan-and-Execute + CausalDiagnosis MCP | no run | no run | no run | no run | no run | no run | no run |
| All six big layers | Agents-as-Tools + CausalDiagnosis MCP | no run | no run | no run | no run | no run | no run | no run |
| All six big layers | CausalDiagnosis agent + CausalDiagnosis MCP | no run | no run | no run | no run | no run | no run | no run |
| Layer 0 | Plan-and-Execute + CausalDiagnosis MCP | no run | no run | no run | no run | no run | no run | no run |
| Layer 0 | Agents-as-Tools + CausalDiagnosis MCP | no run | no run | no run | no run | no run | no run | no run |
| Layer 0 | CausalDiagnosis agent + CausalDiagnosis MCP | no run | no run | no run | no run | no run | no run | no run |

#### The schedule of candidate plans (`tab:parallel`)

The CausalDiagnosis agent with the CausalDiagnosis MCP and all six big layers on, with the schedule of candidate plans on, the agent as released, and off, and under them the other designs with the CausalDiagnosis MCP on the same layer, each over its registered episodes. With the schedule on, the agent carries out its candidate plans as one schedule, in which a read that several plans name runs once and consecutive reads run as one batch with one ranking after the last, and with it off every read step runs alone with its own ranking. The columns are the time to repair in declared cycles, which leaves out the cycles charged for thinking, the time to repair with them, the work orders, the downtime, Pass as a percentage, the cost over the cost of the oracle, the model calls and the calls of the ranking tool of the CausalDiagnosis MCP, each the mean over the episodes and, with the sign ±, its standard error. Batches kept counts the batches of reads that the board advised to a plan the model wrote and that the plan kept unchanged, out of all such batches. Gap explained is the share of the gap between a design and the agent with the schedule on that the switch explains, computed on the measure that the next table tests as the difference off less on over the difference of the design less on, and printed where the design is behind. Neither is a test, and the next table tests the pair.

|  | Repair, declared | Repair with thinking | Work orders | Downtime | Pass (%) | Cost / oracle | Model calls | Ranking calls | Batches kept | Gap explained |
|---|---|---|---|---|---|---|---|---|---|---|
| Schedule of candidate plans on | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run |
| Schedule of candidate plans off | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run |
| Plan-and-Execute + CausalDiagnosis MCP | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run |
| Agents-as-Tools + CausalDiagnosis MCP | no run | no run | no run | no run | no run | no run | no run | no run | no run | no run |

#### The schedule of candidate plans, on against off (`tab:parallelpair`)

The schedule of candidate plans on less off on the same seeds: the paired difference of the time to repair in declared cycles with its 95 percent interval, its exact p value and its Holm decision over the registered size of its family, the paired differences of Pass and of the cost over the oracle with their 95 percent intervals beside their registered margins, the pairs left out of the second for an episode without a cost, and whether the comparison holds. The last two lines describe and test nothing: the mean paired difference of the time to repair over the pairs in which an episode met a refuted repair, the only place the mechanism acts, and over the pairs in which neither did.

|  | On less off |
|---|---|
| Pairs | no run |
| ΔRepair, declared cycles | no run |
| Exact p | no run |
| Holm p | no run |
| Reading | no run |
| ΔPass | no run |
| Margin on Pass | no run |
| ΔCost / oracle | no run |
| Margin on cost | no run |
| Pairs without a cost | no run |
| Holds | no run |
| With a refuted repair | no run |
| Without a refuted repair | no run |
