import fnmatch
from CausalDiagnosis.utils import frameworks


# ----- the file lists of a run, which run.json hashes and the group key of the tables reads (results_index.group_key) -----
# Every file of the fork under these roots lies in one list, or in agent lists only. A path rule is a folder prefix that ends in
# "/", a pattern over the whole path when it holds *, ? or [, where * also crosses "/", or an exact path, each relative to the fork
# root. lists_of tries CODE_LISTS in their order and the first list that matches wins, then the agent lists, then upstream.
CODE_ROOTS = ("CausalDiagnosis/agents/", "CausalDiagnosis/cluster_jobs/", "CausalDiagnosis/configs/",
              "CausalDiagnosis/datasets/", "CausalDiagnosis/environment/", "CausalDiagnosis/episode_viewer/",
              "CausalDiagnosis/evaluation/", "CausalDiagnosis/experiments/", "CausalDiagnosis/frameworks/",
              "CausalDiagnosis/mcp_servers/", "CausalDiagnosis/results/", "CausalDiagnosis/training/",
              "CausalDiagnosis/utils/", "src/", "uv.lock", "CausalDiagnosis/__init__.py",
              "CausalDiagnosis/examples/__init__.py")
# hashed apart under their own keys of run.json data or not at all: the generated tables and the outputs of runs and tools
EXCLUDED = ("CausalDiagnosis/results/", "CausalDiagnosis/results/runs/", "CausalDiagnosis/cluster_jobs/runs_synced/",
            "CausalDiagnosis/cluster_jobs/uploads/", "CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/models.json",
            "CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/grades.json",
            "CausalDiagnosis/datasets/*/data/labels.json", "*.json.lock")
# every runner of an agent imports src/agent/__init__.py, which imports the four agents of AssetOpsBench and the released
# Plan-Execute, and the driver and every runner open their tool sessions through CausalDiagnosis/agents/utils/mcp_sessions.py. The driver also
# imports the tools of the CausalDiagnosis agent, which with the Agent-As-Tool port read the receipt of the metering layer through
# CausalDiagnosis/agents/utils/tool_answers.py. So those files are shared by every run, as the test of the loaded modules shows
_SHARED_AGENT = ("src/agent/__init__.py", "src/agent/runner.py", "src/agent/models.py", "src/agent/_prompts.py",
                 "src/agent/plan_execute/", "CausalDiagnosis/agents/utils/mcp_sessions.py",
                 "CausalDiagnosis/agents/utils/tool_answers.py", "CausalDiagnosis/agents/utils/idle_hook.py",
                 "CausalDiagnosis/utils/model_request_retry.py", "CausalDiagnosis/utils/model_reply_room.py",
                 "src/agent/claude_agent/__init__.py", "src/agent/claude_agent/runner.py", "src/agent/deep_agent/__init__.py",
                 "src/agent/deep_agent/runner.py", "src/agent/direct_llm_agent/__init__.py",
                 "src/agent/direct_llm_agent/runner.py", "src/agent/openai_agent/__init__.py",
                 "src/agent/openai_agent/runner.py")
CODE_LISTS = {
    "checks": ("*/tests/*", "conftest.py", "*/conftest.py", "test_*.py", "*/test_*.py",
              "CausalDiagnosis/datasets/*/verify_data_facts.py",
              "CausalDiagnosis/experiments/run_verification/check_grid_complete.py",
              "CausalDiagnosis/experiments/run_verification/publish_check.py"),
    "documents": ("*.md", "*LICENSE"),
    "scoring": ("CausalDiagnosis/evaluation/metrics/judge_records.py",
               "CausalDiagnosis/experiments/run_verification/recompute_verdicts.py",
               "CausalDiagnosis/experiments/run_verification/rescore_under_verifier_scope.py", "src/evaluation/",
               "CausalDiagnosis/environment/reference_policies/diagnosability_label.py",
               "CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/diagnosability_grade.py"),
    "analysis": ("CausalDiagnosis/evaluation/result_tables/tables_from_runs/",
                "CausalDiagnosis/evaluation/metrics/table_statistics.py",
                "CausalDiagnosis/evaluation/result_tables/pooling_rules.py", "CausalDiagnosis/evaluation/result_tables/index_file.py",
                "CausalDiagnosis/evaluation/metrics/realized_read_elimination.py",
                "CausalDiagnosis/experiments/cluster_runs/run_health_checks.py", "CausalDiagnosis/experiments/cluster_runs/development_checks.py", "CausalDiagnosis/experiments/cluster_runs/cluster_run_checks.py",
                "CausalDiagnosis/experiments/run_verification/verify_results_folder.py",
                "CausalDiagnosis/experiments/run_verification/invariant_digests.py",
                "CausalDiagnosis/experiments/run_verification/invariant_digests.json",
                "CausalDiagnosis/evaluation/metrics/mcp_metrics.py",
                "CausalDiagnosis/experiments/table_definitions/registered_analysis.json",
                "CausalDiagnosis/experiments/table_definitions/registered_analysis.py",
                "CausalDiagnosis/experiments/run_verification/replay_check/",
                "CausalDiagnosis/experiments/table_definitions/serving_of_record.json"),
    "rendering": ("CausalDiagnosis/episode_viewer/walkthrough/", "CausalDiagnosis/evaluation/result_tables/paper_tables/",
                 "CausalDiagnosis/evaluation/result_tables/leaderboard.py",
                 "CausalDiagnosis/episode_viewer/run_review_pack.py",
                 "CausalDiagnosis/evaluation/result_tables/publish_runs.py",
                 "CausalDiagnosis/episode_viewer/fixed_starts.json", "CausalDiagnosis/episode_viewer/step_through_page.py",
                 "CausalDiagnosis/datasets/new_dataset_scaffold.py",
                 "CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/offline_table.py",
                 "CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/calibration_report.py",
                 "CausalDiagnosis/evaluation/result_tables/difficulty_table.py",
                 "CausalDiagnosis/evaluation/result_tables/datasheet_statistics.py"),
    "training": ("CausalDiagnosis/training/rejection_sampling_examples.py", "CausalDiagnosis/training/train_lora.py",
                "CausalDiagnosis/training/gepa_prompt_optimization/", "CausalDiagnosis/training/gepa_prompt_optimization/feedback_card.py",
                "CausalDiagnosis/experiments/table_definitions/write_training_split.py",
                "CausalDiagnosis/cluster_jobs/3_run/train_lora.sbatch",
                "CausalDiagnosis/cluster_jobs/0_setup/setup_train_env.sbatch",
                "CausalDiagnosis/cluster_jobs/3_run/run_gepa_online.sh",
                "CausalDiagnosis/cluster_jobs/0_setup/requirements-train.txt", "CausalDiagnosis/configs/training/"),
    "harness": ("CausalDiagnosis/experiments/episode_runners/run_episodes/",
               "CausalDiagnosis/experiments/episode_runners/audit_reports.py",
               "CausalDiagnosis/experiments/episode_runners/servers_of_a_run.py", "CausalDiagnosis/utils/json_files.py", "CausalDiagnosis/experiments/episode_runners/run_folder_layout.py", "CausalDiagnosis/experiments/episode_runners/episode_launchers.py", "CausalDiagnosis/experiments/episode_runners/run_configuration.py", "CausalDiagnosis/utils/csv_rows.py",
               "CausalDiagnosis/experiments/table_definitions/experiment_tables.py", "CausalDiagnosis/experiments/run_verification/run_identity_file_lists.py",
               "CausalDiagnosis/experiments/table_definitions/training_variants/",
               "CausalDiagnosis/experiments/run_verification/run_ledger/",
               "CausalDiagnosis/experiments/episode_runners/attempt_error_classes.py",
               "CausalDiagnosis/experiments/episode_runners/wall_cut.py",
               "CausalDiagnosis/experiments/episode_runners/cache_flush.py",
               "CausalDiagnosis/experiments/cluster_runs/pending_grid_lines.py",
               "CausalDiagnosis/experiments/cluster_runs/run_lines.py", "CausalDiagnosis/experiments/cluster_runs/grid_files.py", "CausalDiagnosis/experiments/cluster_runs/grid_job_lists.py",
               "CausalDiagnosis/experiments/cluster_runs/precompute_grid_tables.py",
               "CausalDiagnosis/experiments/run_verification/upload_manifest.py",
               "CausalDiagnosis/experiments/run_verification/check_synced_runs.py",
               "CausalDiagnosis/experiments/run_verification/public_record.py",
               "CausalDiagnosis/experiments/cluster_runs/engine_record.py",
               "CausalDiagnosis/experiments/episode_runners/run_reference_policies.py",
               "CausalDiagnosis/datasets/*/build_features_and_blocks.py", "CausalDiagnosis/datasets/*/download_and_build.py",
               "CausalDiagnosis/utils/llm_backends.py", "src/llm/", "src/observability/", *_SHARED_AGENT,
               "CausalDiagnosis/agents/utils/model_adapters.py", "CausalDiagnosis/agents/utils/declaration.py",
               "CausalDiagnosis/agents/utils/answer.py", "CausalDiagnosis/agents/utils/answer_forms.py", "CausalDiagnosis/agents/utils/core_texts.py", "uv.lock", "CausalDiagnosis/__init__.py",
               "CausalDiagnosis/agents/__init__.py", "CausalDiagnosis/agents/utils/__init__.py",
               "CausalDiagnosis/cluster_jobs/__init__.py", "CausalDiagnosis/datasets/dataset_loader/__init__.py",
               "CausalDiagnosis/datasets/splits/__init__.py",
               "CausalDiagnosis/environment/Causal_Diagnosis_Environment/clock/__init__.py",
               "CausalDiagnosis/environment/__init__.py", "CausalDiagnosis/environment/reference_policies/__init__.py",
               "CausalDiagnosis/episode_viewer/__init__.py", "CausalDiagnosis/evaluation/__init__.py",
               "CausalDiagnosis/evaluation/metrics/__init__.py", "CausalDiagnosis/evaluation/result_tables/__init__.py",
               "CausalDiagnosis/examples/__init__.py", "CausalDiagnosis/experiments/__init__.py",
               "CausalDiagnosis/experiments/cluster_runs/__init__.py",
               "CausalDiagnosis/experiments/episode_runners/__init__.py",
               "CausalDiagnosis/experiments/run_verification/__init__.py",
               "CausalDiagnosis/experiments/table_definitions/__init__.py",
               "CausalDiagnosis/frameworks/__init__.py",
               "CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/advice/__init__.py",
               "CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/belief/__init__.py",
               "CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/__init__.py",
               "CausalDiagnosis/mcp_servers/__init__.py", "CausalDiagnosis/mcp_servers/utils/__init__.py",
               "CausalDiagnosis/training/__init__.py", "CausalDiagnosis/utils/__init__.py", "CausalDiagnosis/utils/paths.py",
               "CausalDiagnosis/utils/frameworks.py"),
    "episode": ("src/couchdb/", "CausalDiagnosis/configs/", "CausalDiagnosis/environment/",
               "CausalDiagnosis/mcp_servers/", "CausalDiagnosis/utils/", "src/servers/", "CausalDiagnosis/datasets/"),
    "reproduction": (),
    "cluster": ("CausalDiagnosis/cluster_jobs/",),
}
UPSTREAM = ("src/",)  # every other file under src: the agents and tools of AssetOpsBench that no run loads
# the files of each agent of --agent beyond the shared ones, by what its runner loads alone; static and the reference policies run
# no agent code. causal_next is the package of the general agent (CausalDiagnosis/docs/general-agent-plan.md), which the driver
# plays once its entry lands, so a file of it never enters the identity of a run of the present agent. The agents of a framework
# and their files come from its declaration (_declared_lists)
AGENT_LISTS = {"causal": ("CausalDiagnosis/agents/Causal_Diagnosis_Agent/",),
               "causal_next": ("CausalDiagnosis/agents/Causal_Diagnosis_Agent_next/",)}
# identity: a change can change an episode or its score, so the list enters code.identity and the start check. bound: no run loads
# the file, yet a registered number or record depends on it, so the start check compares it. free: recorded and never compared
LIST_ROLES = {"checks": "free", "documents": "free", "scoring": "identity", "analysis": "bound", "rendering": "free", "training": "free",
              "harness": "identity", "episode": "identity", "reproduction": "bound", "cluster": "bound",
              "upstream": "free",
              "framework": "bound"}
AGENT_LIST = "agent:"  # the prefix of the name of an agent list
FRAMEWORK_ROOT = "CausalDiagnosis/frameworks/"
FRAMEWORK_LIST = "framework"  # the list of a file of a framework folder that its declaration puts in no list


def _declared_lists():
    """(lists, agent lists) that the declarations of the frameworks give (utils/frameworks.py, field LISTS), each path made
    relative to the fork root: [(list name, rules)] in the order of the frameworks and of each declaration, and {agent: rules}
    with every declared agent, an agent without a list of its own holding none. A declaration names a list of CODE_LISTS or
    the agent list of an agent it declares itself, and any other name stops the registry, since a framework would otherwise
    move files of the core or of another framework into its lists."""
    lists, agents = [], {}
    for name in frameworks.frameworks():
        own = tuple(frameworks.declaration(name).get("AGENTS", {}))
        clash = sorted(set(own) & set(AGENT_LISTS))
        if clash:
            raise frameworks.FrameworkError(f"frameworks/{name}: AGENTS declares {', '.join(clash)}, an agent of the core")
        agents.update({agent: () for agent in own})
        for listed, paths in frameworks.lists(name).items():
            rules = tuple(f"{FRAMEWORK_ROOT}{name}/{path}" for path in paths)
            if listed.startswith(AGENT_LIST) and listed[len(AGENT_LIST):] in own:
                agents[listed[len(AGENT_LIST):]] += rules
            elif listed in CODE_LISTS:
                lists.append((listed, rules))
            else:
                raise frameworks.FrameworkError(f"frameworks/{name}: LISTS names {listed}, neither a list of the core nor the "
                                                f"agent list of an agent the declaration names")
    return tuple(lists), agents


FRAMEWORK_LISTS, _DECLARED_AGENTS = _declared_lists()
AGENT_LISTS.update(_DECLARED_AGENTS)


def matches(path, rule):
    """Whether a path relative to the fork root falls under a path rule of CODE_LISTS, AGENT_LISTS or EXCLUDED."""
    if rule.endswith("/"):
        return path.startswith(rule)
    if any(sign in rule for sign in "*?["):
        return fnmatch.fnmatchcase(path, rule)
    return path == rule


def in_roots(path):
    return any(path == root or path.startswith(root) for root in CODE_ROOTS)


def excluded(path):
    return any(matches(path, rule) for rule in EXCLUDED)


def lists_of(path):
    """The names of the lists a path relative to the fork root lies in: one list of CODE_LISTS, then of the declarations of the
    frameworks, or the agent lists that hold it (agent:<agent>), or framework for any other file of a framework folder, or
    upstream for any other file under src, and none for a path outside the roots or excluded."""
    if not in_roots(path) or excluded(path):
        return ()
    for name, rules in (*CODE_LISTS.items(), *FRAMEWORK_LISTS):
        if any(matches(path, rule) for rule in rules):
            return (name,)
    agents = tuple(f"{AGENT_LIST}{agent}" for agent, rules in AGENT_LISTS.items() if any(matches(path, rule) for rule in rules))
    if agents:
        return agents
    if path.startswith(FRAMEWORK_ROOT):
        return (FRAMEWORK_LIST,)
    return ("upstream",) if any(matches(path, rule) for rule in UPSTREAM) else ()


def role_of(name, agent):
    """The role of a list for a run of an agent: an agent list is identity for a run of its own agent and bound for any other."""
    if name.startswith(AGENT_LIST):
        return "identity" if name == f"{AGENT_LIST}{agent}" else "bound"
    return LIST_ROLES[name]
