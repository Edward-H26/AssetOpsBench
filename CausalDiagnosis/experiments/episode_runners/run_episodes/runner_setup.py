import importlib
import os
from pathlib import Path
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import CAUSAL_SERVER, registered_servers, tool_surface  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.run_configuration import ROW_PRESETS  # noqa: E402
from CausalDiagnosis.experiments.episode_runners import cache_flush  # noqa: E402
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.training_variants import adapter_of, block_of, is_marker, numbers  # noqa: E402
from CausalDiagnosis.utils import frameworks  # noqa: E402
from CausalDiagnosis.utils.model_reply_room import card_context_length  # noqa: E402
from . import agent_kinds, model_backends, reference_policy_episodes, run_record, play_one_episode


class ContextNotCounted(SystemExit):
    """A served model whose backend cannot count the context of its endpoint: the context check, the compaction trigger and
    the recall drop of the causal agent would take every request as fitting, so the run stops before its first episode."""


def require_context_count(backend):
    """Stops the run of a row with a served model when its backend does not answer the readers that the causal runner and
    its compaction read (request_fits, request_under_trigger, longest_reply). The driver calls it for a served row only, so a
    row of the scripted planner, which serves no model, is never checked."""
    missing = [name for name in ("request_fits", "request_under_trigger") if not callable(getattr(backend, name, None))]
    missing += [] if hasattr(backend, "longest_reply") else ["longest_reply"]
    if missing:
        raise ContextNotCounted(f"the model backend {type(backend).__name__} cannot count the context of its endpoint (it has no "
                                f"{' and no '.join(missing)}), so no request of the run could be checked against that context")


def served_facts(base_url, model):
    """run.json served for a run of a model agent, read once before the run folder: context_length, the context the endpoint
    serves the model with, max_model_len of its card in the endpoint's model listing as the completion route reads it
    (card_context_length), None when the endpoint names none or does not answer; cache_report, the server's rule for the cached
    prompt tokens of a reply (cache_flush.cache_report), None where the endpoint does not state it."""
    cards = ledger.endpoint_models(base_url) if base_url else None
    return {"context_length": card_context_length(cards, model) if cards else None,
            "cache_report": cache_flush.cache_report(base_url, os.environ.get("API_KEY")) if base_url else None}


def store_settings(a):
    """The store of the AssetOpsBench servers for the launchers and run.json, or None with the rig's action server."""
    if a.servers not in frameworks.surfaces():
        return None
    return tool_surface(a.servers).store_settings(a.store)


def judge_settings(a, agent_label):
    """The judge options of a run, or None; a judge equal to the agent model is refused here, since the fork's evaluator
    refuses self judging on every episode."""
    if a.judge == "none":
        return None
    from CausalDiagnosis.utils import frameworks

    judge = frameworks.code("JUDGES", a.judge)
    judge_model, judge_base_url = a.judge_model or a.model, a.judge_base_url or a.base_url
    if a.agent in agent_kinds.MODEL_AGENTS and judge_model == a.model:
        raise SystemExit(f"--judge {a.judge} needs a judge model other than the agent model, since the fork's evaluator refuses self "
                         "judging; pass --judge-model (and --judge-base-url for a second endpoint)")
    backend = judge.router_backend(judge_model, judge_base_url)
    # the served context of the judge, read once before any episode plays: an endpoint that gives no model listing stops the run
    # here with the reason (judge.ContextUnknown), and a judge that names no context (a router) judges every prompt whole
    with judge.endpoint_probe(judge_base_url):
        budget = judge.served_budget(backend, judge_base_url)
    return {"judge": a.judge, "backend": backend, "protocol": a.judge_protocol, "base_url": judge_base_url, "budget": budget,
            "model": judge_model, "agent_model": a.model if a.agent in agent_kinds.MODEL_AGENTS else agent_label,
            "serving": ledger.serving_identity(ledger.served(judge_model, judge_base_url, "HYD_JUDGE_ENDPOINT_FILE"))}


def default_servers(agent, config_name):
    """The tool surface of a run without --servers: a model agent on a row of the environment table or on an arm of its rows plays
    AssetOpsBench's servers, since every row runs on them, and a reference policy, the scripted plan, the static
    agent, which plays no tool, and a development preset keep the rig's action server."""
    surfaces = frameworks.surfaces()
    on_rows = [name for name in agent_kinds.MODEL_AGENTS if any(surfaces.get(surface, {}).get("default_on_row_presets")
                                                    for surface in agent_kinds.DECLARED_AGENTS.get(name, {}).get("surfaces", ()))]
    row_surface = next(name for name, entry in surfaces.items() if entry.get("default_on_row_presets"))
    return row_surface if agent in (*on_rows, "causal") and config_name in ROW_PRESETS else "rig"


def build_runner(a, config_name, config, price_table, cycles, blocks, facts_path, registered=False):
    """An EpisodeRunner for the command line settings; it fits the diagnostics models when a server or the grade needs them, except
    in a registered run, which opens the generated tables as they are and computes none."""
    # the modules of the agent of the run, imported once here, where the run takes its agent, since the driver imports no file
    # of an agent list for every run: an agent that cannot be imported stops the run before its run folder, not in an episode
    for module in agent_kinds.AGENT_ENTRY_MODULES.get(a.agent, ()):
        importlib.import_module(module)
    a.servers = a.servers or default_servers(a.agent, config_name)
    if a.agent in agent_kinds.PAPER_ERA_AGENTS and a.servers not in agent_kinds.DECLARED_AGENTS[a.agent]["surfaces"]:
        raise SystemExit(frameworks.code("AGENTS", a.agent).servers_refusal)
    servers = registered_servers(a.servers, config.diagnostics_server)
    if a.servers in frameworks.surfaces() and a.agent in agent_kinds.POLICY_AGENTS:
        raise SystemExit(f"the reference policies play through the rig's action server; drop --servers {a.servers}")
    # the models of the diagnostics server for the asset of the run, fitted before any server process starts and reads them, when
    # the server serves the run or the grade reads them, except in a registered run, which opens the generated tables as they are
    models_path = diagnostics_models(config.asset_id, servers, blocks, facts_path,
                                     fit_models=(CAUSAL_SERVER in servers or not a.no_grade) and not registered)
    backend = scripted = classifiers = None
    adapter = adapter_of(a.agent, a.training_variant, a.training_fold, config.asset_id, served_started=run_record.serving_started(a.base_url),
                         split=a.training_split)
    block, block_record = block_of(a.agent, a.training_variant, a.training_fold, config.asset_id, split=a.training_split)
    # SGLang selects an adapter by the model <base>:<adapter>; a fold without an adapter plays nothing
    served = f"{a.model}:{adapter['name']}" if adapter and not is_marker(adapter) else a.model
    temperature = numbers(config.asset_id).sampling_temperature if a.starts == "training" else None
    if a.agent == "greedy":
        classifiers = reference_policy_episodes.served_classifiers(cycles, blocks)
    elif a.agent in agent_kinds.PAPER_ERA_AGENTS:
        if not os.environ.get("API_KEY"):
            raise SystemExit(frameworks.code("AGENTS", a.agent).api_key_refusal)
    elif a.agent not in agent_kinds.POLICY_AGENTS:
        scripted = (frameworks.code("AGENTS", a.agent).backend(servers, play_one_episode.get_profile(config.asset_id)) if a.agent in agent_kinds.SCRIPTED_AGENTS
                    else None)
        backend = model_backends.RecordingBackend(scripted or model_backends.router_backend(served, a.base_url), temperature=temperature)
        if scripted is None:  # a row with a served model
            require_context_count(backend.inner)
    causal = causal_options(a) if a.agent == "causal" else None
    if causal and CAUSAL_SERVER not in servers:
        # without the server the agent plans on its own candidates and the normal ranges, where these switches have nothing to change
        needs_server = [flag for flag, on in (("--causal-no-branches", a.causal_no_branches), ("--causal-single-hypothesis", a.causal_single_hypothesis),
                                              ("--causal-e10-reads", a.causal_e10_reads),
                                              ("--causal-ec2-reads", a.causal_ec2_reads)) if on]
        if needs_server:
            raise SystemExit(f"{', '.join(needs_server)} change what the CausalDiagnosis MCP advises and need it. Drop them or drop --no-diagnostics")
    served_record = served_facts(a.base_url, served) if a.agent in agent_kinds.MODEL_AGENTS else None
    if a.agent in agent_kinds.PAPER_ERA_AGENTS and (refusal := frameworks.code("AGENTS", a.agent).served_refusal(servers, served_record)):
        raise SystemExit(refusal)
    agent_label = agent_kinds.agent_label_of(a.agent, a.model, causal)
    judge_options = judge_settings(a, agent_label)
    options = {"judge": judge_options, "grade": not a.no_grade, "ranking_model": a.ranking_model}
    abstention = (a.abstention_rule, a.abstention_cut) if a.abstention_rule != "decision" else None
    runner = play_one_episode.EpisodeRunner(config, config_name, a.agent, servers, cycles, blocks, facts_path, run_record.run_label(config_name, agent_label), options,
                           backend=backend, scripted=scripted, classifiers=classifiers, model=a.model if a.agent in agent_kinds.MODEL_AGENTS else None,
                           ranking_model=a.ranking_model, abstention=abstention, prices_name=price_table["name"] if price_table else "default",
                           causal_options=causal, base_url=a.base_url, store=store_settings(a),
                           task_block=block, served_model=served if a.agent in agent_kinds.MODEL_AGENTS else None,
                           temperature=temperature, adapter=adapter, task_block_record=block_record, models_path=models_path
)
    runner.served = served_record
    return runner, judge_options


def causal_options(a):
    """The ablation switches of the causal agent from the command line, as CausalLoopOptions takes them and run.json records them."""
    if a.causal_plan_length is not None and a.causal_plan_length < 1:
        raise SystemExit("--causal-plan-length must be at least 1")
    if not 0 <= a.causal_expectation_tolerance <= 1:
        raise SystemExit("--causal-expectation-tolerance must lie between 0 and 1")
    if a.causal_e10_reads and a.causal_ec2_reads:
        raise SystemExit("--causal-e10-reads and --causal-ec2-reads are two arms of one ablation; pass one of them")
    options = {"branches": not a.causal_no_branches, "single_hypothesis": a.causal_single_hypothesis, "plan_length": a.causal_plan_length,
               "expectations": not a.causal_no_expectations,
               "reads": "separating" if a.causal_e10_reads else "ec2" if a.causal_ec2_reads else "gain", "backtracking": not a.causal_no_backtrack,
               "parallel": not a.causal_sequential,
               "expectation_tolerance": a.causal_expectation_tolerance}
    if a.causal_no_structured_output:  # recorded only where the run sets it, so the run.json of a run without the flag keeps its keys
        options["structured_output"] = False
    return options


def diagnostics_models(asset_id, servers, blocks, facts_path, fit_models):
    """The models file of the CausalDiagnosis server for the asset of a run, which the launcher of the server names, or None for a
    run without the server. This is the one place where the driver decides how a row with the server gets its models. A run with
    the server on an asset whose models the model code of the server cannot fit (CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.fit_models.unfit_reasons)
    stops here by name, before any server starts and before the run folder exists, and the message names the row without the
    server, so that no row reads the models of another asset. With fit_models the models of an asset that the server can fit are
    fitted on the data file of the run when they are missing or of another format (fit.ensure_models)."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline import fit_models as fit

    unfit = fit.unfit_reasons(asset_id, set(blocks))
    if CAUSAL_SERVER in servers and unfit:
        raise SystemExit(f"a row with the CausalDiagnosis MCP plays with the models fitted for its asset, and the model code of the server "
                         f"fits none for the asset {asset_id}: {'; '.join(unfit)}. Play the row without the MCP (--no-diagnostics)")
    models_path = fit.models_path(asset_id)
    if fit_models and not unfit:
        Path(models_path).parent.mkdir(parents=True, exist_ok=True)  # the data folder of a port, which a fresh clone lacks
        fit.ensure_models(models_path, asset_id=asset_id, facts_path=facts_path)
    return models_path if CAUSAL_SERVER in servers else None
