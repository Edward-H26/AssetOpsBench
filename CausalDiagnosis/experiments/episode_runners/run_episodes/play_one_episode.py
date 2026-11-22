import asyncio
import json
import os
import shutil
import time
from pathlib import Path
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import BOOKKEEPING_COLUMNS, COLUMNS, HIDDEN, RELEASED_FILE, SCORE_COLUMNS, STATE_FILE  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import RIG_SERVER, contract_tools, surface_of, tool_surface  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.episode_launchers import DEFAULT_RANKING_MODEL, write_launchers  # noqa: E402
from CausalDiagnosis.utils.csv_rows import flat  # noqa: E402
from CausalDiagnosis.utils.json_files import read_json, write_json  # noqa: E402
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock import token_clock  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_2_real_time import THINKING_MODES, TOKEN_MODE  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock.wall_clock import charges_wall_seconds  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.experiment_tables import EPISODE_WALL_LIMIT_S, HEALTH_TIMEOUT_S, INFRASTRUCTURE_REPLAYS  # noqa: E402
from CausalDiagnosis.experiments.episode_runners import attempt_error_classes as failures  # noqa: E402
from CausalDiagnosis.experiments.episode_runners import cache_flush  # noqa: E402
from CausalDiagnosis.experiments.episode_runners import wall_cut  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.training_variants import with_task_block  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.audit_reports import episode_report  # noqa: E402
from CausalDiagnosis.mcp_servers.utils import transcript  # noqa: E402
from CausalDiagnosis.utils.atomic_writes_and_locks import locked  # noqa: E402
from CausalDiagnosis.utils.public_rules.visits import is_no_call  # noqa: E402
from CausalDiagnosis.agents.utils import idle_hook as idle  # noqa: E402
from CausalDiagnosis.utils import model_request_retry as retry  # noqa: E402
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode  # noqa: E402
from CausalDiagnosis.utils import frameworks  # noqa: E402
from . import agent_kinds, reference_policy_episodes, episode_scoring, static_agent


# ----- one run -----
class EpisodeRunner:
    """The settings of one run and the machinery that plays, verifies, judges and records one episode. main() builds it
    from the command line through build_runner, and so does gepa_prompt_optimization/, whose loop plays each episode with the block of
    a candidate in place."""

    def __init__(self, config, config_name, agent, servers, cycles, blocks, facts_path, label, options, backend=None, scripted=None,
                 classifiers=None, model=None, ranking_model=DEFAULT_RANKING_MODEL, abstention=None, prices_name="default", causal_options=None,
                 base_url=None, store=None,
                 task_block="", served_model=None, temperature=None, adapter=None, task_block_record=None, models_path=None):
        self.config, self.config_name, self.agent, self.servers = config, config_name, agent, servers
        # the training variant: the block appended to the task message, the model name with the adapter that the
        # endpoint serves, the sampling temperature of a training run, and the records of the adapter and the block for run.json
        self.task_block, self.served_model, self.temperature = task_block, served_model or model, temperature
        self.adapter, self.task_block_record = adapter, task_block_record
        self.base_url = base_url  # the Agent-As-Tool runner reaches the model at it
        self.store = store  # the store of the AssetOpsBench servers, None with the rig's action server
        # the object of the framework's tool surface the run registers, which serves its engine, start, store and finish
        self.surface = tool_surface(surface_of(servers[0])) if store else None
        if store:  # the surface's own checks of the asset's profile, before the first episode (frameworks.surface_profile_problems)
            problems = frameworks.surface_profile_problems(get_profile(config.asset_id), surface_of(servers[0]))
            if problems:
                raise SystemExit(f"the {surface_of(servers[0])} surface refuses the profile of {config.asset_id}: {'; '.join(problems)}")
        self.env_class = self.surface.env_class if store else HydraulicRigEnv  # the rows of a framework replay their own engine
        self.cycles, self.blocks, self.facts_path, self.label = cycles, blocks, facts_path, label
        self.options, self.backend, self.scripted, self.classifiers = options, backend, scripted, classifiers
        self.model, self.ranking_model, self.abstention, self.prices_name = model, ranking_model, abstention, prices_name
        self.models_path = models_path  # the models file of the diagnostics server for the asset of the run, None without the server
        self.causal_options = causal_options
        self.agent_label = agent_kinds.agent_label_of(agent, model, causal_options)
        # the example start of the two paper-era ports, computed once per run (CausalDiagnosis.datasets.splits.development_split.example_start)
        self.diagnosis_example = frameworks.code("AGENTS", agent).example_start(config, blocks) if agent in agent_kinds.PAPER_ERA_AGENTS else None
        self.checked_servers = False
        self.server_threads = {}  # server -> the threads its process ran at its start in check_servers_once, the baseline of its kind
        self.load_averages = None  # the 1, 5 and 15 minute load averages at check_servers_once
        self.tool_listings = {}  # {server: hash of the tool listing its process sent}, filled when the servers are listed
        # the reasons of the start check, which main() and the loops of GEPA set from registration() before run.json, and which
        # run_meta records as the stamp
        self.unregistered = None
        self.served = None  # run.json served, what the model endpoint serves, which build_runner reads (runner_setup.served_facts)

    def start(self, ep_dir, seed, episode_config):
        """The episode folder with its state file, released evidence, launchers, store and question, and in the generated_tokens
        mode the empty ledger of generated tokens; returns the launchers, the question, the entry offset and the start
        combination of the episode. The tools are listed before the first episode of a run (check_servers_once), so no episode
        pays for it."""
        ep_dir.mkdir(parents=True, exist_ok=True)
        if episode_config.clock_mode == TOKEN_MODE:
            token_clock.open_ledger(ep_dir)
        if self.store:
            state = self.surface.start(ep_dir / STATE_FILE, seed, episode_config, self.facts_path)  # the engine of the rows and their briefing
        else:
            state = episode.start(ep_dir / STATE_FILE, seed, episode_config, self.facts_path)
        replayed = episode.replay(state, env_class=self.env_class)
        entry_offset = replayed.entry_offset
        launchers = write_launchers(ep_dir, self.servers, ranking_model=self.ranking_model, abstention=self.abstention, store=self.store,
                                    left_out=tuple(replayed.initial), models_path=self.models_path,
                                    data_file=(get_profile(self.config.asset_id).records["data_env"], self.facts_path))
        if self.store:
            self.episode_store(ep_dir).create()
        question = read_json(ep_dir / RELEASED_FILE)["task_text"]  # the briefing the server composed at the start
        (ep_dir / "question.txt").write_text(question, encoding="utf-8")
        return launchers, question, entry_offset, list(replayed.initial)

    def close_store(self, ep_dir):
        """The episode's databases exported into the episode folder and dropped, once the episode is scored, on a run with a
        store; nothing without one."""
        if self.store and (ep_dir / STATE_FILE).exists():
            store = self.episode_store(ep_dir)
            store.export(ep_dir)
            store.drop()

    def episode_store(self, ep_dir):
        """The episode's databases on the store of the run: created before the first call, exported and dropped after
        the verifier read the episode. The surface imports its store when it is asked, since the store loads its servers."""
        return self.surface.store(ep_dir / STATE_FILE, self.store)

    def check_servers_once(self, run_dir, seed, episode_config):
        """Before the first episode of a run: the tools every server lists, on a scratch folder beside the episode folders with a
        state file of the first planned episode, its launchers and its store, compared with the contract of the row. The store is
        dropped and the folder removed, so no episode pays for the start of the servers, and a server that does not start still
        stops the run before any episode."""
        if self.checked_servers or self.agent == "static":
            return
        scratch = Path(run_dir) / harness_rule.SERVER_CHECK_FOLDER
        try:
            launchers = self.start(scratch, seed, episode_config)[0]
            check_servers(launchers, self.config if self.store else None)
            self.tool_listings.update(listed_tools(scratch))
            self.server_threads = started_threads(scratch)
            self.load_averages = list(os.getloadavg())  # the load the starts and the calls of the run are measured under
        finally:
            if self.store and (scratch / STATE_FILE).exists():
                self.episode_store(scratch).drop()
            shutil.rmtree(scratch, ignore_errors=True)
        self.checked_servers = True

    def run_agent(self, launchers, task, name, ep_dir, seed=None, episode_config=None):
        """The one place that hands the task message to the runner of the run's agent: (result, error, extra, calls, tool calls),
        extra the cells of the row from the runner's record, calls its model calls and tool calls the records of its tool calls.
        A reference policy plays the seed and the configuration of the episode, which it reads no task message for."""
        if self.agent in agent_kinds.POLICY_AGENTS:
            try:
                matches = reference_policy_episodes.play_policy(self.agent, ep_dir / STATE_FILE, launchers[RIG_SERVER], self.cycles, self.blocks, episode_config, seed,
                                      self.classifiers)
                return None, None, {"replay_matches": matches}, [], []
            except Exception as exc:  # noqa: BLE001 - a failed episode is a result of the benchmark
                return None, f"{type(exc).__name__}: {exc}", {}, [], []
        if self.agent in agent_kinds.PAPER_ERA_AGENTS:
            return frameworks.code("AGENTS", self.agent).play(self, launchers, task, f"{self.label}-{name}", ep_dir)
        self.backend.calls = []
        if self.scripted:
            self.scripted.start_episode()
        if self.agent == "causal":
            result, error, _ = agent_runs.run_causal(self.backend, launchers, task, self.causal_options, agent_runs.causal_surface(self.config.asset_id, self.store))
            return result, error, agent_runs.record_causal(ep_dir, result, self.backend.calls), self.backend.calls, result.trajectory if result is not None else []
        return frameworks.code("AGENTS", self.agent).play(self, launchers, task, f"{self.label}-{name}", ep_dir)

    def attempt(self, ep_dir, name, seed, episode_config):
        """One play of a planned episode: the start, the runner, its record files, the evidence of a failure with the class of the
        attempt (failures.attempt_class), and the harness finish of an episode the runner left open, which an attempt of the
        infrastructure does not get, since it is set aside and played again. A step of the harness that raises (the start, the
        store, the finish) is evidence of kind harness and ends the attempt. The runners catch what their agents raise and return
        it as their error, so what leaves run_agent is no failure of an attempt."""
        record = {"entry_offset": None, "combination": None, "result": None, "error": None, "extra": {}, "calls": [], "tools": [],
                  "harness_error": None, "finish_by": "", "finish": None, "cache_flush": None, "cut_probe": None}
        CUT_FACTS.pop(str(ep_dir), None)
        started = time.perf_counter()
        try:
            launchers, question, entry_offset, combination = self.start(ep_dir, seed, episode_config)
            record.update(entry_offset=entry_offset, combination=combination)
            record["cache_flush"] = self.flush_cache(ep_dir, episode_config)
            if self.task_block:
                (ep_dir / "task_block.txt").write_text(self.task_block, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - a step of the harness that raised is evidence against the infrastructure
            record["harness_error"] = f"{type(exc).__name__}: {exc}"
        ended = None
        if record["harness_error"] is None:
            write_server_baseline(ep_dir, launchers, self.server_threads)
            episode.hand_over(ep_dir / STATE_FILE)  # H, the origin of the thinking clock and of the wall limit, under the state lock
            wall = WallLimit(ep_dir / STATE_FILE)
            with token_clock.episode(ep_dir), retry.episode_deadline(wall.deadline), idle.bound(self.no_call_entry(ep_dir)):
                result, error, extra, calls, tools = self.run_agent(launchers, with_task_block(question, self.task_block), name, ep_dir, seed,
                                                                    episode_config)
            ended = time.time()
            record["ended_mono"] = time.monotonic()
            record.update(result=result, error=error, extra=extra, calls=calls, tools=tools, ended_wall=ended)
            if record["cache_flush"] is not None:
                record["cache_flush"]["first_request_cached_tokens"] = retry.first_cached_tokens()
            update_state(ep_dir / STATE_FILE, episode_ended_at=ended,
                         **({} if record["cache_flush"] is None else {"cache_flush": record["cache_flush"]}))
        record["cut"] = failures.error_type(record["error"] or "") == harness_rule.WALL_LIMIT
        facts = CUT_FACTS.pop(str(ep_dir), {})
        record["cut_at"] = facts.get("at_wall") if record["cut"] else None
        record["cut_end"], record["cut_probe"] = self.cut_end(facts) if record["cut"] else ("", None)
        record["runner_end"] = harness_rule.runner_end_of(self.agent, record["result"], record["error"])
        # the error of a tool answer (a refusal, an argument the schema refuses) is an answer, so a tool call is evidence only
        # when it got no answer at all
        tool_failures = [e for e in map(harness_rule.tool_error, record["tools"]) if failures.kind_of(e or "") == failures.TOOL]
        errors = [e for e in (record["error"], *(c.get("error") for c in record["calls"]), *tool_failures) if e]
        healthy = None
        if self.agent in agent_kinds.MODEL_AGENTS and self.base_url:
            def healthy():
                return failures.endpoint_health(self.base_url, os.environ.get("API_KEY", ""), HEALTH_TIMEOUT_S)
        servers, model_agent = failures.server_evidence(ep_dir), self.agent in agent_kinds.MODEL_AGENTS
        record["class"], record["kinds"], record["health"] = failures.attempt_class(errors, record["harness_error"], record["cut"], healthy,
                                                                                    servers, model_agent, record["cut_end"] or None)
        if record["class"] != failures.INFRASTRUCTURE:
            stamp, source = harness_rule.finish_stamp(record["calls"], record["cut_at"] or ended, record["cut"])
            try:
                record["finish_by"], record["finish"] = self.finish_episode(ep_dir, getattr(record["result"], "answer", None) or "", stamp, source)
            except Exception as exc:  # noqa: BLE001 - a finish that raised is evidence against the infrastructure
                record["harness_error"] = f"{type(exc).__name__}: {exc}"
                record["class"], record["kinds"], _health = failures.attempt_class(errors, record["harness_error"], record["cut"],
                                                                                   servers=servers, model_agent=model_agent,
                                                                                   cut_end=record["cut_end"] or None)
        record["wall"] = time.perf_counter() - started
        return record

    def flush_cache(self, ep_dir, episode_config):
        """The flush of the model server's cache before an episode of a preset with the wall clock, inside the creation of the
        episode before its handover, written into the state as cache_flush with the epoch second of its request: None, and nothing written, on another
        clock or for an agent without a model endpoint. CacheNotFlushed propagates, so the episode does not start and the attempt
        is the harness's."""
        if not charges_wall_seconds(episode_config) or self.agent not in agent_kinds.MODEL_AGENTS or not self.base_url:
            return None
        started_at = time.time()
        flushed = {**cache_flush.flush_before_episode(self.base_url, os.environ.get("API_KEY") or None), "started_at": started_at}
        update_state(ep_dir / STATE_FILE, cache_flush=flushed)
        return flushed

    def cut_end(self, facts):
        """(end, probe record) of an attempt cut at the wall limit, by wall_cut.cut_end from the facts run_limited took at the cut
        (a fact it did not take counts as open). The probe asks the model service for one token, only for a model agent with an
        endpoint; without it an open model request ends the cut as the service's."""
        probe = None
        if self.agent in agent_kinds.MODEL_AGENTS and self.base_url:
            def probe():
                return wall_cut.cut_probe(self.base_url, os.environ.get("API_KEY", ""), self.served_model)
        ended = wall_cut.cut_end(facts.get("failed_span_open"), facts.get("tool_call_open"), facts.get("model_request_open"), probe)
        return ended["end"], ended["probe"]

    def no_call_entry(self, ep_dir):
        """The episode's no_call entry for CausalDiagnosis.agents.utils.idle_hook, None where the row has no in-process entry: on the rig's own rows one
        no_call step through episode.call, a counted call stamped and charged by the clock itself, which returns whether the
        environment ended the episode. The rows of AssetOpsBench step through the meter of their task process
        (CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.metering_layer.Meter.no_call) under the meter and state locks with the store
        sync. The hook calls the entry outside the event loop (CausalDiagnosis.agents.utils.idle_hook runs it in a thread for an async design), so the
        meter's coroutine gets a loop of its own."""
        if self.store:
            return lambda: asyncio.run(self.episode_store(ep_dir).meter().no_call())
        state_path = ep_dir / STATE_FILE

        def entry():
            episode.call(state_path, "no_call", {}, {"type": "no_call"})
            return episode.replay(episode.load_state(state_path), env_class=self.env_class).done
        return entry

    def no_call_at(self, ep_dir, wall, monotonic):
        """One no_call step of the episode stamped with the given readings on the surface of the row, the step that the entry
        of CausalDiagnosis.agents.utils.idle_hook sends unstamped (no_call_entry): on the rig's own rows through episode.call, on the rows of AssetOpsBench
        through the meter of the task process, built and awaited inside one loop."""
        if self.store:
            async def step():
                return await self.episode_store(ep_dir).meter(wall_clock=lambda: wall, monotonic_clock=lambda: monotonic).no_call()
            asyncio.run(step())
        else:
            episode.call(ep_dir / STATE_FILE, "no_call", {}, {"type": "no_call"}, now=wall, monotonic=monotonic)

    def set_aside(self, ep_dir):
        """An attempt the infrastructure broke: its databases exported and dropped, its files moved to attempt1 inside the
        episode folder, so the next attempt starts on a fresh state file under the same seed."""
        if self.store and (ep_dir / STATE_FILE).exists():
            store = self.episode_store(ep_dir)
            store.export(ep_dir)
            store.drop()
        aside = ep_dir / harness_rule.ATTEMPT_FOLDER
        aside.mkdir(parents=True, exist_ok=True)
        for path in list(ep_dir.iterdir()):
            if path.name != harness_rule.ATTEMPT_FOLDER:
                shutil.move(str(path), str(aside / path.name))
        return harness_rule.ATTEMPT_FOLDER

    def finish_episode(self, ep_dir, answer, stamp, source):
        """The end of an episode its runner left open: one finish action whose summary is the runner's own answer, through the
        code that serves the task tool finish on the surface of the row, in this process and stamped with the time of the last
        model call (finish_stamp). It is one step and has no price. Returns (finish_by, the record of harness.json): agent when
        the log holds the runner's finish, harness when the driver applied it, empty when the environment ended the episode
        without one. An episode that is still open after it raises, which is evidence against the infrastructure. On a row with
        a thinking clock the thinking accrued up to the stamp is played first through the episode's no_call entry, stamped at
        that moment (no_call_at), at a cut at the wall limit and after the runner ended on its own alike; an end the engine
        reaches with it is the record, so no finish follows and the log holds no finish the engine refused."""
        state_path = ep_dir / STATE_FILE
        env = episode.replay(episode.load_state(state_path), env_class=self.env_class)
        if env.done:
            return ("agent" if any(e["event"] == "finish" for e in env.log) else ""), None
        monotonic = time.monotonic() - max(0.0, time.time() - stamp)
        if env.config.clock_mode in THINKING_MODES:
            self.no_call_at(ep_dir, stamp, monotonic)
            if episode.replay(episode.load_state(state_path), env_class=self.env_class).done:
                return "", None
        if self.store:
            async def finish():
                # the meter holds a lock of the loop it runs on, so it is built and awaited inside one loop
                meter = self.episode_store(ep_dir).meter(wall_clock=lambda: stamp, monotonic_clock=lambda: monotonic)
                return await meter.call(self.surface.harness_server, "finish", {"summary": answer})
            reply = asyncio.run(finish())
            refusal = reply.text if reply.is_error else None
        else:
            text = episode.call(state_path, "finish", {"summary": answer}, {"type": "finish", "summary": answer}, now=stamp, monotonic=monotonic)
            refusal = json.loads(text).get("refused")
        env = episode.replay(episode.load_state(state_path), env_class=self.env_class)
        if not env.done:
            raise RuntimeError(f"the harness finish left the episode open: {refusal}")
        charged = next((e.get("think_cycles") for e in reversed(env.log) if e["event"] == "finish"), None)
        return "harness", {"stamp": stamp, "source": source, "think_cycles": charged, "refusal": refusal}

    def row_of(self, seed, scenario, combination, entry_offset, wall, error):
        """The cells of a row that every episode has, a start combination of None for an episode that never started."""
        row = {k: None for k in COLUMNS}
        row.update({
            "run": self.label, "seed": seed, "scenario_id": scenario["id"] if scenario else "", "date": scenario["date"] if scenario else "",
            "config": self.config_name, "agent": self.agent_label, "model": self.model if self.agent in agent_kinds.MODEL_AGENTS else "",
            "clock_mode": self.config.clock_mode, "guidance_level": self.config.guidance_level, "diagnostics_server": self.config.diagnostics_server,
            "prices": self.prices_name, "verifier_scope": self.config.verifier_scope, "ranking_model": self.ranking_model,
            "block": combination, "entry_offset": entry_offset,
            "n_faulty": None if combination is None else sum(1 for c, v in zip(self.config.components, combination) if v != self.config.healthy[c]),
            "wall_s": round(wall, 1), "error": error or "",
        })
        return row

    def play(self, ep_dir, name, seed, scenario, episode_config, meta):
        """One episode under the harness rule: the tools listed once before the first episode, an attempt, one more attempt under
        the same seed when the infrastructure broke the first, then the verifier (or the judge and the parse of the static
        answer), the hidden files, harness.json and the report. Two attempts that the infrastructure broke give a row of the class
        harness_error with empty scores and the start combination of the episode. Returns the row of scores.csv."""
        if self.agent == "static":
            return self.play_static_row(ep_dir, seed, scenario, episode_config, meta)
        self.check_servers_once(ep_dir.parent, seed, episode_config)
        attempts = [self.attempt(ep_dir, name, seed, episode_config)]
        aside = None
        if attempts[0]["class"] == failures.INFRASTRUCTURE and INFRASTRUCTURE_REPLAYS:
            aside = self.set_aside(ep_dir)
            attempts.append(self.attempt(ep_dir, name, seed, episode_config))
        standing = attempts[-1]
        error, error_class = standing["harness_error"] or standing["error"], ""
        scores = {k: None for k in SCORE_COLUMNS}
        if standing["class"] == failures.INFRASTRUCTURE:
            error_class = failures.HARNESS_ERROR
        else:
            try:
                scores, _combination = episode_scoring.verify_episode(ep_dir, episode_config, self.cycles, self.blocks, seed, self.options, self.env_class)
            except Exception as exc:  # noqa: BLE001 - a verifier that cannot run leaves the scores empty and the error in the row
                error = (error + "; " if error else "") + f"verifier: {type(exc).__name__}: {exc}"
                error_class = failures.VERIFIER
            if not error_class and standing["class"] == failures.AGENT:
                error_class = failures.AGENT
        self.close_store(ep_dir)
        # every planned episode keeps its start combination, and so its cluster, scored or not
        combination = standing["combination"] or list(planned_episodes.start_combination(episode_config, seed, self.cycles, self.blocks))
        row = self.row_of(seed, scenario, combination, standing["entry_offset"], standing["wall"], error)
        row.update(scores)
        if (ep_dir / STATE_FILE).exists():
            state = episode.load_state(ep_dir / STATE_FILE)
            row.update(episode_scoring.tool_counts(state))
            row["diagnosis_refusals"] = harness_rule.diagnosis_refusals(episode.replay(state, env_class=self.env_class).log)
        if scores.get("steps") is not None:
            row["refusals"] = read_json(ep_dir / HIDDEN / "verdict.json", {}).get("refusals", row["refusals"])
        row.update({k: v for k, v in standing["extra"].items() if k in BOOKKEEPING_COLUMNS})
        row.update({"finish_by": standing["finish_by"], "runner_end": standing["runner_end"], "wall_limit": int(standing["cut"]),
                    "error_class": error_class, "replays": len(attempts) - 1, "failed_calls": harness_rule.failed_calls(standing["calls"], standing["tools"]),
                    "cut_replies": harness_rule.cut_replies(standing["calls"])})
        row.update(clock_cells(ep_dir / STATE_FILE, standing))
        if standing.get("cache_flush") is not None:
            row["server_cache_flushed"] = int(standing["cache_flush"].get("status") == 200)
        write_json(ep_dir / harness_rule.HARNESS_FILE, harness_rule.harness_record(attempts, aside))
        (ep_dir / "report.md").write_text(episode_report(ep_dir, meta), encoding="utf-8")
        return {k: flat(v) for k, v in row.items()}

    def play_static_row(self, ep_dir, seed, scenario, episode_config, meta):
        """The row of an episode of the static agent, which plays no episode, so no step of the harness applies to it. Its model
        request is made outside an episode, so it names the health probe of the model's endpoint (failures.endpoint_probe): a failed
        attempt is sent again while the endpoint answers, and an endpoint that no longer answers ends the run with ServiceDown
        before the row is written, since a failure of the service is no answer of the agent."""
        _launchers, question, entry_offset, _combination = self.start(ep_dir, seed, episode_config)
        started = time.perf_counter()
        with failures.endpoint_probe(self.base_url) as probe:
            extra, error = self.play_static(ep_dir, question, episode_config, seed)
        if probe.down:
            raise retry.ServiceDown(f"the model endpoint {self.base_url} no longer answers its health probe, so the static "
                                    f"row of {ep_dir.name} is not written")
        self.close_store(ep_dir)
        scores, combination = extra.pop("scores"), extra.pop("combination")
        row = self.row_of(seed, scenario, combination, entry_offset, time.perf_counter() - started, error)
        row.update(scores)
        row.update(episode_scoring.tool_counts(episode.load_state(ep_dir / STATE_FILE)))
        row.update({k: v for k, v in extra.items() if k in BOOKKEEPING_COLUMNS})
        row.update({"finish_by": "", "runner_end": "", "wall_limit": 0, "error_class": failures.AGENT if error else "", "replays": 0,
                    "failed_calls": harness_rule.failed_calls(self.backend.calls, []), "cut_replies": harness_rule.cut_replies(self.backend.calls)})
        (ep_dir / "report.md").write_text(episode_report(ep_dir, meta), encoding="utf-8")
        return {k: flat(v) for k, v in row.items()}

    def play_static(self, ep_dir, question, episode_config, seed):
        """The static answer of ablation A3 and its two scorings: the judge against the oracle's reference answer, and
        the parsed faulty and repaired sets against the recorded start and the oracle's work orders."""
        env = episode.replay(episode.load_state(ep_dir / STATE_FILE), env_class=self.env_class)
        oracle = env.oracle()
        context = static_agent.static_context(self.cycles, self.blocks, episode_config, seed, self.env_class)
        from CausalDiagnosis.environment.Causal_Diagnosis_Environment.briefing import call_rule_sentences
        prompt = static_agent.static_prompt(question, context, episode_config.components, call_rule_sentences(env.public_profile(), episode_config))
        (ep_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        self.backend.calls = []
        error, answer = None, ""
        try:
            answer = self.backend.generate(prompt) or ""
        except Exception as exc:  # noqa: BLE001 - a failed call is a result of the benchmark
            error = f"{type(exc).__name__}: {exc}"
        (ep_dir / "answer.txt").write_text(answer, encoding="utf-8")
        write_json(ep_dir / "llm_calls.json", self.backend.calls)
        components, healthy = episode_config.components, episode_config.healthy
        parsed = static_agent.parse_static_answer(answer, components)
        truth_faulty = [c for c in components if oracle["diagnosis"][c] != healthy[c]]
        hidden = ep_dir / HIDDEN
        hidden.mkdir(exist_ok=True)
        write_json(hidden / "oracle.json", oracle)
        write_json(hidden / "log.json", env.log)
        judge = episode_scoring.run_judge(ep_dir, self.options["judge"], static=True) if self.options["judge"] and answer else None
        static = {"mode": "static", "cycle_index": context["cycle_index"], "features_shown": len(context["features"]),
                  "failure_modes_shown": len(context["failure_modes"]), "answer": answer, "parsed": parsed, "truth_faulty": truth_faulty,
                  "oracle_work_orders": [o["component"] for o in oracle["work_orders"]],
                  "faulty_set_correct": set(parsed["faulty"]) == set(truth_faulty),
                  "per_component_correct": {c: (c in parsed["faulty"]) == (c in truth_faulty) for c in components},
                  "repairs_match_oracle": set(parsed["repaired"]) == {o["component"] for o in oracle["work_orders"]}}
        write_json(hidden / "static.json", static)
        scores = {k: None for k in SCORE_COLUMNS}
        scores.update({"end_reason": "static", "steps": 0, "static_faulty_set_correct": static["faulty_set_correct"],
                       "static_repairs_match_oracle": static["repairs_match_oracle"], "judge_pass": None if judge is None else judge.get("pass")})
        write_json(hidden / "score.json", {**scores, "judge": judge})
        extra = {"llm_calls": len(self.backend.calls), "input_tokens": sum(c.get("input_tokens", 0) for c in self.backend.calls),
                 "output_tokens": sum(c.get("output_tokens", 0) for c in self.backend.calls), "scores": scores,
                 "combination": [oracle["diagnosis"][c] for c in components]}
        return extra, error


def listed_tools(ep_dir):
    """{server: the hash of the tool listing its process sent} from the transcript of an episode folder, which the servers
    write, the last listing of each server."""
    return {line["server"]: line["listing_sha256"] for line in transcript.read(ep_dir) if line["kind"] == "list"}


def check_servers(launchers, config=None):
    """List the tools of every registered server once before the first episode: a server that fails to start would
    otherwise only show up as unavailable inside the planner prompt. config is the configuration of an AssetOpsBench run,
    whose big layers set the tools the task server lists."""
    async def listed():
        async with reference_policy_episodes.open_sessions(dict(launchers)) as sessions:
            return {server: {tool.name for tool in tools} for server, tools in sessions.tools.items()}

    try:
        offered = asyncio.run(listed())
    except Exception as exc:  # noqa: BLE001 - a server that does not start stops the run before any episode
        raise SystemExit(f"the servers did not start: {type(exc).__name__}: {exc}") from exc
    for server, names in offered.items():
        if names != set(contract_tools(server, config=config)):
            raise SystemExit(f"server {server} offers {sorted(names)}, expected {sorted(contract_tools(server, config=config))}")


# {episode folder: the facts run_limited took at a cut}, read once by the attempt of that folder (EpisodeRunner.cut_end)
CUT_FACTS = {}
# the fields that the driver writes into an episode's state file outside any tool call (update_state), which no transcript line records
DRIVER_STATE_FIELDS = ("episode_ended_at", "cache_flush")


def update_state(state_path, **fields):
    """Write fields of the driver into an episode's state file under the state lock, beside the fields of the servers."""
    with locked(state_path):
        state = episode.load_state(state_path)
        state.update(fields)
        episode.save_state(state_path, state)


def note_cut(deadline):
    """Take the facts of a cut at the deadline (a monotonic instant just passed) for the episode the driver plays: the model
    request and the wait after a failed attempt open at it (CausalDiagnosis.utils.model_request_retry.open_at) and whether a counted tool call is open in the
    transcript. Taken before the runner is cancelled, since the cancellation closes them."""
    folder = token_clock.episode_folder()
    if folder is None:
        return
    CUT_FACTS[str(folder)] = {**retry.open_at(deadline), "tool_call_open": bool(open_tool_calls(folder)),
                              "at_wall": time.time() - (time.monotonic() - deadline)}  # the deadline in wall time


def open_tool_calls(folder):
    """The counted tool calls of an episode folder whose entry line has no exit line in the transcript, open at this moment. A
    no_call step is no tool call of the agent, so a cut inside the driver's own no_call is classed by the other facts."""
    lines, _offset = transcript.read_since(folder, 0)
    entered = {line["n"] for line in lines if line.get("kind") == "enter" and not is_no_call(line)}
    return entered - {line.get("enter_n") for line in lines if line.get("kind") == "exit"}


def clock_cells(state_path, attempt):
    """The clock cells of an episode row from episode.clock_report and the instants the agent's part ended:
    empty where the state holds no handover, as for an episode that never started."""
    cells = {"cut_end": attempt.get("cut_end", "")}
    state = episode.load_state(state_path) if Path(state_path).exists() else {}
    if state.get("handover_mono") is None or attempt.get("ended_mono") is None:
        return cells
    clock = episode.clock_report(state_path)
    agent_mono = attempt["ended_mono"] - state["handover_mono"]
    agent_wall = attempt["ended_wall"] - state["handover_wall"]
    cells.update(credited_s=clock["credited_seconds"], uncredited_s=clock["uncredited_seconds"],
                 failed_model_s=round(clock["failed_model_seconds"], 3), failed_model_attempts=clock["failed_model_attempts"],
                 uncharged_after_last_s=round(max(0.0, attempt["ended_mono"] - clock["charged_until_mono"]), 3),
                 clock_skew=int(abs(agent_wall - agent_mono) > 1.0))
    return cells


def started_threads(folder):
    """{server: the threads its process ran at its start}, from the ready lines in the transcript of a folder."""
    lines, _offset = transcript.read_since(folder, 0)
    return {line["server"]: line.get("threads") for line in lines if line.get("kind") == "ready" and line.get("server")}


def write_server_baseline(ep_dir, launchers, threads):
    """hidden/server_baseline.json of an episode: per server the SHA-256 of the launcher the driver wrote and, where
    check_servers_once measured it, the thread count of its kind; the ready-line check of the servers refuses a process that
    differs (CausalDiagnosis.mcp_servers.Environment_Actions_MCP.episode_state._baseline)."""
    baseline = {}
    for server, path in launchers.items():
        record = {"launcher_sha256": transcript.file_sha256(path)}
        if threads.get(server) is not None:
            record["threads"] = threads[server]
        baseline[server] = record
    write_json(Path(ep_dir) / episode.BASELINE_FILE, baseline)


class WallLimit:
    """The deadline of an episode's agent part: the handover H, read from the state file (handover_mono, written
    by episode.hand_over) or taken now where the state holds none, plus the wall limit, plus the seconds the state credits to the
    starts of the servers and the service of the counted calls (credited_seconds), so a credit moves the deadline and a failed model
    attempt does not. deadline() is an absolute time of time.monotonic()."""

    def __init__(self, state_path, limit_s=EPISODE_WALL_LIMIT_S):
        self.state_path, self.limit_s = Path(state_path), limit_s
        state = self._state()
        self.handover = state.get("handover_mono") if state.get("handover_mono") is not None else time.monotonic()

    def _state(self):
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def deadline(self):
        return self.handover + self.limit_s + sum(self._credited().values())

    def _credited(self):
        """The credited seconds of the starts and the calls, caught up with the transcript (episode.clock_report reads the
        state and writes nothing), or the stored ones where the state file is no episode state."""
        try:
            return episode.clock_report(self.state_path)["credited_seconds"]
        except (OSError, ValueError, KeyError):
            return self._state().get("credited_seconds") or {}


# Imported after the definitions: this module uses them only when a function runs, and they may load
# this module first (the order of the package is in __init__.py).
from . import harness_rule, agent_runs, planned_episodes
