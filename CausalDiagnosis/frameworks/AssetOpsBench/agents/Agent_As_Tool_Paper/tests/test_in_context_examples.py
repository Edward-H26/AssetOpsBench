"""The in-context examples of the two ports, read without a model, since the paper's appendix prints them. Every example written for the Agent-As-Tool port
on the rig surface shows each answer of a tool as JSON in keys that the answer has, with made-up values of a made-up world, and its
Finish line relays a value that its observation returned. Its Action Input names the arguments of the tool's listed schema, and the
call it shows is accepted by the servers of a rig row. The one diagnosis example is written at run time from the tool contract and
the example start, and the levels of that start are the only levels of a registered asset in any text of either port. The
CausalDiagnosis MCP examples are read against the result model each tool's function returns in CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/main.py and
called on the server over the released file of an episode, the rig examples of the IoT, FMSR and WO agents against those of
servers/iot, servers/fmsr and servers/wo, and the task tool examples against the answers of the task server of the AssetOpsBench rows
on the repair rules row, which lists every task tool. On a run with the CausalDiagnosis MCP the examples written for the port show the
site, a field and the profile of the asset of the run, the supervisor's diagnosis names the levels of the example start, the
repair manual, the failure modes and the examples of the CausalDiagnosis MCP are written for the asset (their readings and
probabilities illustrative), and every check above holds for them too."""
from CausalDiagnosis.utils.paths import PACKAGE
import ast
import asyncio
import dataclasses
import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent

from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.tests.engine_forms import engine_split  # noqa: E402

from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper import in_context_examples as examples, react_tool_adapter, sub_agent_registry  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Plan_Execute_Paper import workflow  # noqa: E402
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile  # noqa: E402
from CausalDiagnosis.datasets.splits.development_split import development_starts, example_start, starts_of_test_set  # noqa: E402
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import HEALTHY  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.tests.conftest import world  # noqa: E402,F401 - the recorded cycles, skipped in a clone that states that it holds no data
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import aob_briefing as briefing  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.bind_episode_databases import bound  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_tool_contract import TASK_SERVER  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tests import harness  # noqa: E402
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP import main as causal_main  # noqa: E402
from servers.fmsr import main as fmsr_main  # noqa: E402
from servers.iot import main as iot_main  # noqa: E402
from servers.wo import main as wo_main  # noqa: E402

from CausalDiagnosis.utils.paths import ROOT, SRC
SERVER_NAME = {shown: tool for tool, shown in react_tool_adapter.RESERVED.items()}  # the agent calls the task tool finish end_episode
RIG = {"iot": (iot_main, examples.IOT_RIG), "fmsr": (fmsr_main, examples.FMSR_RIG), "wo": (wo_main, examples.WO_RIG)}
SUPERVISOR = {"TASK_SUPERVISOR": examples.TASK_SUPERVISOR, "CAUSAL_SUPERVISOR": examples.CAUSAL_SUPERVISOR}
# the made-up names of the examples and the names of a rig row that stand in for them in a call (for the replay of the examples)
EXAMPLE_ASSET, EXAMPLE_PART = "CHL-A100", "compressor"
RIG_ASSET = "hyd_1"  # the asset of the rig, which the examples of a run with the CausalDiagnosis MCP show
WITH_MCP, WITHOUT_MCP = ("iot", "fmsr", "wo", "task", "causal_diagnosis"), ("iot", "fmsr", "wo", "task")
ROW_SERVERS = {"without the MCP": WITHOUT_MCP, "with the MCP": WITH_MCP}


def steps(example):
    """(step, action, Action Input, observation) of every numbered step of an example."""
    names = dict(re.findall(r"^Action (\d+): (.+)$", example, re.M))
    inputs = dict(re.findall(r"^Action [Ii]nput (\d+): (.*)$", example, re.M))
    observations = dict(re.findall(r"^Observation (\d+): (.*)$", example, re.M))
    return [(k, names[k].strip(), inputs.get(k, ""), observations.get(k)) for k in sorted(names, key=int)]


def question(example):
    return re.search(r"^Question: (.*)$", example, re.M).group(1)


def finish_line(example):
    return [given for _, action, given, _ in steps(example) if action == "Finish"][-1]


def leaves(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from leaves(item)
    else:
        yield value


def token_in(value, text):
    shown = json.dumps(value) if isinstance(value, bool) else str(value)
    return re.search(r"(?<![\w.])" + re.escape(shown) + r"(?![\w])", text) is not None


def tool_observations(example):
    """(tool, answer) of every step of an example whose observation is a JSON object."""
    return [(action, json.loads(observation)) for _, action, _, observation in steps(example) if observation and observation.startswith("{")]


def _result_models(main, tool):
    """The models besides ErrorResult in the return annotation of the tool's function in a server's main.py."""
    tree = ast.parse(Path(main.__file__).read_text(encoding="utf-8"))
    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == tool]
    assert len(functions) == 1, (main.__name__, tool)
    return [getattr(main, n.id) for n in ast.walk(functions[0].returns) if isinstance(n, ast.Name) and n.id not in ("Union", "Optional", "ErrorResult")]


def result_model(main, tool):
    """The one model besides ErrorResult in the return annotation of the tool's function in a server's main.py."""
    models = _result_models(main, tool)
    assert len(models) == 1, (tool, models)
    return models[0]


def result_fields(main, tool):
    """The fields of every model besides ErrorResult in the return annotation of the tool's function, since a tool of the
    CausalDiagnosis MCP answers in the model of the level the call asks for."""
    models = _result_models(main, tool)
    assert models, tool
    return {field for model in models for field in model.model_fields}


@pytest.fixture(scope="module")
def rig_start(world):
    """{component: level} of the rig's example start, the development start the registered seed draws (example_start)."""
    _cycles, blocks = world
    profile = get_profile("hyd_1")
    return dict(zip(profile.components, example_start(profile, set(blocks))))


def port_examples(start, row=None):
    """The examples written for the rig surface by table, with the diagnosis example written for the start, on a run whose McpRow
    is row (None without the CausalDiagnosis MCP), where every table but the IoT agent's shows the asset of the run."""
    parameters = [*start, "evidence_ids"]
    task = [examples.diagnosis_example(parameters, start) if name == examples.DIAGNOSIS_TOOL else text for name, text in examples.TASK_TOOLS.items()]
    shown = (lambda text: text) if row is None else (lambda text: examples.on_mcp_row(text, row))
    orders = (lambda text: text) if row is None else (lambda text: examples.orders_on_mcp_row(text, row))
    return {"IOT_RIG": examples.IOT_RIG[3:], "FMSR_RIG": [shown(t) for t in examples.FMSR_RIG], "WO_RIG": [orders(t) for t in examples.WO_RIG],
            "TASK_TOOLS": [shown(t) for t in task], "CAUSAL_TOOLS": [shown(t) for t in examples.CAUSAL_TOOLS.values()]}


def supervisor_texts(row=None):
    """The supervisor examples written for this port on a run whose McpRow is row."""
    specs = {s.name: s for s in sub_agent_registry.SURFACES["rig"]}
    return {"TASK_SUPERVISOR": specs["TaskAgent"].supervisor_examples_for(row), "CAUSAL_SUPERVISOR": specs["CausalDiagnosisAgent"].supervisor_examples_for(row)}


def returned_values(example):
    """Every leaf value of the example's observations that neither its question nor its Action Inputs before Finish hold."""
    given = question(example) + " " + " ".join(i for _, a, i, _ in steps(example) if a != "Finish")
    return [value for _, answer in tool_observations(example) for value in leaves(answer) if not token_in(value, given)]


def relays_a_returned_value(example):
    returned = returned_values(example)
    return not returned or any(token_in(value, finish_line(example)) for value in returned)


@pytest.mark.parametrize("servers", ROW_SERVERS.values(), ids=ROW_SERVERS.keys())
def test_every_finish_line_written_for_the_port_carries_a_value_that_its_observation_returned(rig_start, servers):
    with_values = 0
    for table, texts in port_examples(rig_start, sub_agent_registry.mcp_row(servers, RIG_ASSET, rig_start)).items():
        for example in texts:
            found = tool_observations(example)  # every observation that opens with a brace is JSON as it stands
            assert found, (table, question(example))
            if returned_values(example):
                with_values += 1
            assert relays_a_returned_value(example), (table, question(example))
    assert with_values >= 19  # the examples whose Finish lines carried no value before, so the loop cannot pass empty
    # the predicate refuses the new profile observation under the old Finish line
    profile = examples.TASK_TOOLS["get_asset_profile"]
    planted = profile.replace(finish_line(profile), "The asset profile lists the asset, its components, its sensor groups, its budget and its horizon.")
    assert not relays_a_returned_value(planted)


def test_every_mcp_example_shows_only_fields_of_its_tools_result_model():
    assert set(examples.CAUSAL_TOOLS) == set(causal_main.TOOLS)
    for name, example in examples.CAUSAL_TOOLS.items():
        [(tool, answer)] = tool_observations(example)
        assert tool == name
        unknown = set(answer) - result_fields(causal_main, tool)
        assert not unknown, (tool, sorted(unknown))


@pytest.mark.parametrize("server", sorted(RIG))
def test_every_rig_example_of_the_iot_fmsr_and_wo_agents_shows_only_fields_of_its_tools_result_model(server):
    main, table = RIG[server]
    for example in table:
        for tool, answer in tool_observations(example):
            unknown = set(answer) - set(result_model(main, tool).model_fields)
            assert not unknown, (server, tool, sorted(unknown))


async def test_every_task_tool_example_shows_only_keys_of_the_task_servers_answer(world, rig_start):
    meter, engine, *_ = harness.memory_engine_episode(world, "repair_rules")
    calls = [("get_asset_profile", {}), ("get_current_time", {}), ("wait_for_notification", {"timeout": 1}),
             ("lookup_fix", {"component": engine.config.components[0]}), ("submit_diagnosis", {**HEALTHY, "evidence_ids": ""}),
             ("finish", {"summary": "done"})]
    answers = {}
    for tool, arguments in calls:
        reply = await meter.call(TASK_SERVER, tool, arguments)
        assert not reply.is_error, (tool, reply.text)
        answers[tool] = harness.body(reply)
    meter, engine, *_ = harness.memory_engine_episode(world, "repair_rules")  # a declaration needs an episode without a diagnosis
    group = next(iter(engine.config.sensor_groups))
    reply = await meter.call(TASK_SERVER, "declare_insufficient_evidence", {"sensor_group": group, "cycles": 1, "evidence_ids": ""})
    assert not reply.is_error, reply.text
    answers["declare_insufficient_evidence"] = harness.body(reply)
    task = dict(zip(examples.TASK_TOOLS, port_examples(rig_start)["TASK_TOOLS"]))  # the diagnosis example as written for the rig's example start
    assert {SERVER_NAME.get(name, name) for name in task} == set(answers)
    for name, example in task.items():
        [(tool, answer)] = tool_observations(example)
        assert tool == name
        unknown = set(answer) - set(answers[SERVER_NAME.get(name, name)])
        assert not unknown, (name, sorted(unknown))


def test_the_effect_of_an_intervention_is_named_a_prediction_in_the_causal_diagnosis_examples():
    """A value the CausalDiagnosis MCP predicts is named a prediction wherever a reader sees it, and the paper prints these
    examples. A ranking keeps its probabilities, which are the MCP's belief over causes."""
    texts = [examples.CAUSAL_SUPERVISOR, *examples.CAUSAL_TOOLS.values()]
    for text in texts:
        for wording in ("estimated effect", "is estimated", "the estimate."):
            assert wording not in text, wording
    effect = examples.CAUSAL_TOOLS["estimate_intervention_effect"]
    assert "predicted effect" in effect and "predicted effect" in examples.CAUSAL_SUPERVISOR


def sent_by_wrapper(definition, arguments):
    """The arguments that the wrapper of hub.mcp_tool, built from the listed schema as Runtime.tools_of builds it, hands to the MCP
    call for the arguments that the engine split from an Action Input."""
    got = {}
    react_tool_adapter.mcp_tool(definition.name, definition.description, definition.inputSchema or {}, lambda sent: got.update(sent=sent) or "CALL").run(arguments)
    return got["sent"]


def first_call(example):
    """(action, Action Input) of the first step of an example whose action is a tool."""
    return [(action, given) for _, action, given, _ in steps(example) if action not in ("Finish", "Self-Ask")][0]


async def test_every_call_that_a_rig_example_shows_is_accepted_by_the_servers_of_a_rig_row(world):
    """The replay: every tool call that an example written for the rig surface shows goes through hub.mcp_tool, built from the
    listed schema as Runtime.tools_of builds it, with the Action Input split as the engine splits it, to an episode of the repair
    rules row, in an order the environment allows. The names of the made-up world are replaced by the episode's own through one
    table, so a wrong form of a value still fails. Every name of an Action Input is a property of the listed schema, every
    required property is named, the wrapper raises nothing and the answer is no error body."""
    _cycles, blocks = world
    meter, engine, store, _ = harness.memory_engine_episode(world, "repair_rules")
    with bound(store, harness.NAMES):
        listed = {}
        for server in ("iot", "fmsr", "wo", TASK_SERVER):
            for definition in await meter.listed_tools(server):
                listed[react_tool_adapter.agent_name(definition.name)] = (server, definition)  # every listed tool passes agent_name
        profile = harness.body(await meter.call(TASK_SERVER, "get_asset_profile", {}))
        for key in json.loads(steps(examples.TASK_TOOLS["get_asset_profile"])[0][3]):
            assert key in profile, key
        components = profile["components"]
        start = dict(zip(components, example_start(get_profile(profile["asset_id"]), set(blocks))))
        diagnosis = examples.diagnosis_example([*listed[examples.DIAGNOSIS_TOOL][1].inputSchema["properties"]], start)
        part = components[0]
        swaps = [(f"{EXAMPLE_ASSET}_{EXAMPLE_PART}", profile["asset"]["children"][part]), (EXAMPLE_ASSET, profile["asset_id"]),
                 ("Chiller 4", profile["asset_id"]), ("SiteX", "MAIN"), (EXAMPLE_PART, part), ("chiller", part),
                 ("Supply Temperature", "Temperature")]
        made = {}

        async def call(shown, given):
            server, definition = listed[shown]
            schema = definition.inputSchema or {}
            for old, new in swaps:
                given = given.replace(old, new)
            if "wonum" in made:
                given = given.replace("WO-12345", made["wonum"])
            arguments = engine_split(given)
            if isinstance(arguments, dict):
                assert set(arguments) <= set(schema.get("properties", {})), (shown, sorted(arguments))
                assert set(schema.get("required", [])) <= set(arguments), (shown, sorted(arguments))
            reply = await meter.call(server, definition.name, sent_by_wrapper(definition, arguments))
            text = reply.text.rsplit("\n", 1)[0] if "\n{" in reply.text else reply.text
            answer = json.loads(text) if text.strip().startswith("{") else None
            assert not reply.is_error and not (isinstance(answer, dict) and set(answer) == {"error"}), (shown, reply.text[:200])
            if shown == "generate_work_order":
                made["wonum"] = answer["wonum"]
            return answer

        for example in (*examples.IOT_RIG[3:], *examples.FMSR_RIG):
            await call(*first_call(example))
        for name in ("get_asset_profile", "get_current_time", "wait_for_notification", "lookup_fix"):
            await call(*first_call(examples.TASK_TOOLS[name]))
        await call(*first_call(diagnosis))
        await call(*first_call(examples.WO_RIG[3]))  # the order, then the reads that name it
        for index in (0, 1, 2, 4):
            await call(*first_call(examples.WO_RIG[index]))
        await call(*first_call(examples.TASK_TOOLS[react_tool_adapter.agent_name("finish")]))
    # the declaration needs an episode without a diagnosis
    meter, engine, store, _ = harness.memory_engine_episode(world, "repair_rules")
    with bound(store, harness.NAMES):
        listed = {react_tool_adapter.agent_name(d.name): (TASK_SERVER, d) for d in await meter.listed_tools(TASK_SERVER)}
        shown, given = first_call(examples.TASK_TOOLS["declare_insufficient_evidence"])
        server, definition = listed[shown]
        arguments = engine_split(given.replace("Supply Temperature", "Temperature"))
        assert set(arguments) <= set(definition.inputSchema["properties"]) and set(definition.inputSchema["required"]) <= set(arguments)
        reply = await meter.call(server, definition.name, sent_by_wrapper(definition, arguments))
        assert not reply.is_error and '"error"' not in reply.text[:20], reply.text[:200]


async def test_an_optional_filter_left_at_none_reaches_the_server_as_no_filter(world):
    """The rule of the wrapper on text arguments for an optional one: the engine's None reaches the server as no value, so the
    server's default holds. The work orders of a site listed with status=None are those listed with no status at all."""
    meter, _engine, store, _ = harness.memory_engine_episode(world, "repair_rules")
    with bound(store, harness.NAMES):
        definition = next(d for d in await meter.listed_tools("wo") if d.name == "list_workorders")
        assert "status" not in definition.inputSchema.get("required", [])
        sent = sent_by_wrapper(definition, engine_split("site_id=MAIN, status=None"))
        assert sent["status"] is None
        with_none = harness.body(await meter.call("wo", definition.name, sent))
        without = harness.body(await meter.call("wo", definition.name, {"site_id": "MAIN"}))
        assert with_none == without and "error" not in with_none


def test_every_call_of_a_causal_diagnosis_example_is_accepted_by_the_server(world, tmp_path):
    """The four tools of the CausalDiagnosis MCP over stdio, on the released file of an episode with the fixture of its own
    tests: the evidence after a read, the ranking, the prediction of a repair, and its verification after the order of that
    repair closed and a reading followed. Every name of an Action Input is a parameter of the listed schema, every required one
    is named, the wrapper raises nothing and the answer is no error body."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.effect_model import combo_key
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.fit_models import ensure_models
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.fixture_episode import FixtureEpisode

    ensure_models()
    cycles, blocks = world
    start = (100, 80, 0, 90)
    episode = FixtureEpisode(tmp_path, cycles, blocks, start, seed=0)
    episode.read("tier1")
    shutil.copy(PACKAGE / "mcp_servers" / "Causal_Diagnosis_MCP" / "launcher.py", tmp_path / "causal_server.py")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (str(ROOT), str(SRC), os.environ.get("PYTHONPATH", "")) if p)}
    part = "valve"  # the component that the start holds off its healthy level
    order = ("get_episode_evidence", "rank_root_causes", "estimate_intervention_effect", "verify_outcome")
    assert set(order) == set(examples.CAUSAL_TOOLS)

    async def replay():
        params = StdioServerParameters(command=sys.executable, args=[str(tmp_path / "causal_server.py"), f"left_out={combo_key(start)}"], env=env)
        answers = {}
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = {t.name: t for t in (await session.list_tools()).tools}
                for name in order:
                    if name == "verify_outcome":  # the order of the repair that the prediction names closes, and a reading follows
                        episode.create_work_order(part, prediction_id="pred/1")
                        episode.advance(5)
                        episode.read("tier1")
                    _shown, given = first_call(examples.CAUSAL_TOOLS[name])
                    arguments = engine_split(given.replace(EXAMPLE_PART, part))
                    schema = listed[name].inputSchema
                    assert set(arguments) <= set(schema["properties"]) and set(schema.get("required", [])) <= set(arguments), (name, arguments)
                    reply = await session.call_tool(name, sent_by_wrapper(listed[name], arguments))
                    answers[name] = (reply.isError, json.loads(reply.content[0].text))
        return answers

    replayed = asyncio.run(replay())
    for name, (is_error, answer) in replayed.items():
        assert not is_error and "error" not in answer, (name, answer)
        # the example shows the answer of the default level, at which the ports call the tool, in keys that answer has
        [(_tool, shown)] = tool_observations(examples.CAUSAL_TOOLS[name])
        assert set(shown) <= set(answer), (name, sorted(set(shown) - set(answer)))
    assert "items" not in replayed["get_episode_evidence"][1]  # the key of the raw level, which an example of the default level lacks


def test_the_diagnosis_example_is_written_from_the_tool_contract_and_the_example_start():
    text = examples.diagnosis_example(["alpha", "beta", "evidence_ids"], {"beta": 7, "alpha": 4})
    action, given = first_call(text)
    assert action == examples.DIAGNOSIS_TOOL and given.startswith("alpha=4, beta=7, evidence_ids=")  # the order of the contract
    assert "alpha 4, beta 7" in question(text) and "alpha 4, beta 7" in finish_line(text)
    observation = steps(text)[0][3]
    assert json.loads(observation)["diagnosis"] == {"alpha": 4, "beta": 7}
    for start, word in (({"alpha": 4, "beta": 7, "gamma": 1}, "gamma"), ({"alpha": 4}, "beta"), (None, "example start")):
        with pytest.raises(ValueError, match=word):
            examples.diagnosis_example(["alpha", "beta", "evidence_ids"], start)


def _texts_agent(src):
    """The test of the texts an agent reads (CausalDiagnosis/agents/tests/test_agent_texts.py), loaded by its path, for registered_assets and finder."""
    spec = importlib.util.spec_from_file_location("agent_texts", ROOT / "CausalDiagnosis" / "agents" / "tests" / "test_agent_texts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_the_only_levels_of_a_registered_asset_in_the_texts_of_both_ports_are_those_of_the_example_start(world):
    """Every text that either port shows, with and without the CausalDiagnosis MCP, the latter showing the site, a field and the
    profile of the asset of the run, and in the diagnosis of TASK_SUPERVISOR the levels of the example start, which are checked
    as the diagnosis example's. The examples written for the asset of a run with the MCP (the ranking, the repair manual) also
    state its healthy levels, which every profile answer states, and name each component at the example start's level or at the
    healthy level of that component alone."""
    _cycles, blocks = world
    texts = _texts_agent(SRC)
    meter, _engine, store, _ = harness.memory_engine_episode(world, "repair_rules")
    with bound(store, harness.NAMES):
        definitions = {server: await meter.listed_tools(server) for server in ("iot", "fmsr", "wo", TASK_SERVER)}
    definitions["causal_diagnosis"] = []
    checked = []
    for asset in texts.registered_assets(ROOT):
        profile = get_profile(asset["asset_id"])
        if not set(profile.combinations()) >= set(blocks):
            continue
        checked.append(asset["asset_id"])
        combination = example_start(profile, set(blocks))
        healthy = tuple(profile.healthy[c] for c in profile.components)
        assert combination in development_starts(profile, set(blocks)) and combination not in starts_of_test_set(profile, set(blocks))
        assert combination != healthy
        start = dict(zip(profile.components, combination))
        written, fixed, written_on_row, fixed_on_row = [], [], [], []
        for servers in (WITHOUT_MCP, WITH_MCP):
            specs = sub_agent_registry.registration(servers, "rig")
            row = sub_agent_registry.mcp_row(servers, asset["asset_id"], start)
            fixed.append(sub_agent_registry.merged_examples(specs, row))
            fixed_on_row.append(row is not None)
            for spec in (s for s in specs if s.kind == "react"):
                tools = [react_tool_adapter.mcp_tool(d.name, d.description, d.inputSchema or {}, lambda sent: "CALL")
                         for server in spec.servers for d in definitions.get(TASK_SERVER if server == "task" else server, [])]
                own = spec.own_examples(tools, start, row)
                written.append(own)
                fixed.append("\n".join(workflow.task_examples(own)))
                written_on_row.append(row is not None)
                fixed_on_row.append(row is not None)
        module = [text for text in module_texts() if text != examples.DIAGNOSIS_TEMPLATE]  # every fixed text of the examples
        fixed += module
        fixed_on_row += [False] * len(module)
        find = texts.finder(asset)
        diagnosis = examples.diagnosis_example([*start, "evidence_ids"], start)
        mcp = sub_agent_registry.mcp_row(WITH_MCP, asset["asset_id"], start)
        diagnoses = (diagnosis, examples.on_mcp_row(diagnosis, mcp), examples.diagnosis_named(mcp))
        levels_of = {c: {start[c], profile.healthy[c]} for c in profile.components}
        wrong = []
        for text, on_row in zip(fixed + written, fixed_on_row + written_on_row, strict=True):
            outside = text
            for shown in diagnoses:
                outside = outside.replace(shown, "")
                for line in workflow.task_examples(shown):  # the planner lists the question of the diagnosis example
                    outside = outside.replace(line, "")
            if on_row:  # the examples written for the asset of the run name each component at its start's or its healthy level alone
                for _, matched in find(outside):
                    stated = stated_levels(texts, asset, matched)
                    if not stated or any(level not in levels_of[c] for c, level in stated):
                        wrong.append((asset["asset_id"], matched, stated, text.split("\n", 1)[0]))
            else:
                assert not find(outside), find(outside)[:3]
        assert not wrong, "\n".join(map(repr, wrong))  # every case, in a message string, which pytest does not shorten
        # control: a level that the start or the healthy combination holds for another component only, where there is one, is stated
        # for its own component, so the check above refuses it
        c, level = min(((c, level) for c in asset["components"] for level in asset["levels"][c] if level not in levels_of[c]),
                       key=lambda pair: pair[1] not in set(start.values()) | set(profile.healthy.values()))
        assert find(f"{c} {level}") and stated_levels(texts, asset, f"{c} {level}") == [(c, level)]
        for shown in diagnoses:
            stated = [stated_levels(texts, asset, matched) for _, matched in find(shown)]  # the levels of the example start, and no other number
            assert stated and all(pairs and all(level == start[c] for c, level in pairs) for pairs in stated), stated
            assert any(shown in text for text in written)
    assert checked, "no registered asset whose recordings are the data of this run was checked"


def stated_levels(texts, asset, matched):
    """The (component, level) pairs that a text found by texts.finder(asset) states: a run of numbers states one level of each
    component in the order of the components, and a named form or a failure mode code states the pair whose own pattern matches it."""
    if texts.NUMBER_RUN.fullmatch(matched):
        return list(zip(asset["components"], map(int, re.split(r"\s*,\s*", matched))))
    return [(component, level) for component in asset["components"] for level in asset["levels"][component]
            if texts.named_level(_at_one_level(asset, component, level)).fullmatch(matched)]


def _at_one_level(asset, component, level):
    """The asset with one component at one level and the failure mode codes of that level alone, whose named forms state that pair."""
    return {**asset, "components": [component], "levels": {component: [level]},
            "failure_modes": [m for m in asset.get("failure_modes") or [] if (m.get("component"), m.get("level")) == (component, level)]}


def module_texts():
    """Every text of in_context_examples.py: its string constants and the strings of its lists and tables."""
    found = []
    for value in vars(examples).values():
        if isinstance(value, str):
            found.append(value)
        elif isinstance(value, (list, tuple)):
            found += [v for v in value if isinstance(v, str)]
        elif isinstance(value, dict):
            found += [v for v in value.values() if isinstance(v, str)]
    return found


def test_no_example_names_a_channel_of_a_registered_asset():
    texts = _texts_agent(SRC)
    assets = texts.registered_assets(ROOT)
    assert assets, "no registered asset was found"
    for asset in assets:
        names = [*asset.get("channels", {}), *(c["field"] for c in asset.get("channels", {}).values() if "field" in c)]
        assert names
        for text in module_texts():
            for name in names:
                assert not re.search(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])", text), (asset["asset_id"], name)


# the two observations that confirm a write as its request states it, and so hold nothing that the request does not
CONFIRMS_A_WRITE = ("submit the diagnosis", "end the task")


def test_every_reply_in_a_supervisor_example_written_for_the_port_has_content():
    for label, text in SUPERVISOR.items():
        for example in (e for e in text.split("\n\n") if e.strip()):
            for _, action, given, observation in steps(example):
                if any(given.startswith(f"request={write}") for write in CONFIRMS_A_WRITE):
                    continue
                words = set(re.findall(r"[A-Za-z0-9./:_-]+", observation)) - set(re.findall(r"[A-Za-z0-9./:_-]+", question(example) + " " + given))
                assert [w for w in words if re.search(r"\d", w) or w[:1].isupper()], (label, observation)


def identifiers(example):
    """The evidence and prediction identifiers that an example cites or shows."""
    found = []
    for _, action, given, observation in steps(example):
        if action != "Finish" and "=" in given:
            arguments = engine_split(given)
            if isinstance(arguments, dict):
                if "evidence_ids" in arguments:
                    found += [item.strip() for item in str(arguments["evidence_ids"]).split(",") if item.strip()]
                if "prediction_id" in arguments:
                    found.append(arguments["prediction_id"])

        def walk(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == "evidence_ids":
                        found.extend(item)
                    elif key in ("evidence_id", "prediction_id"):
                        found.append(item)
                    else:
                        walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        if observation and observation.startswith("{"):
            walk(json.loads(observation))
    # an identifier that a sentence cites ends at a space, a comma, a line end or the period that ends the sentence
    found += [i.rstrip(".") for i in re.findall(r"(?:evidence|citing) ((?:obs|wo|fmea|alarm|pred|ev)[^ ,\n]*(?: [A-Z][a-z]+)*)", example)]
    return found


@pytest.mark.parametrize("servers", ROW_SERVERS.values(), ids=ROW_SERVERS.keys())
def test_every_evidence_identifier_of_an_example_written_for_the_port_has_a_form_of_the_surface(rig_start, servers):
    forms = {}
    for form in (*briefing.EVIDENCE_FORMS, briefing.ALARM_FORM, briefing.PREDICTION_FORM):
        head = form.split(" ")[0]
        forms[head.split("/")[0]] = head.count("/") + 1
    row = sub_agent_registry.mcp_row(servers, RIG_ASSET, rig_start)
    checked = [i for texts in (*port_examples(rig_start, row).values(), list(supervisor_texts(row).values())) for example in texts for i in identifiers(example)]
    assert len(checked) >= 5
    for identifier in checked:
        head = identifier.split("/")[0]
        assert head in forms and len(identifier.split("/", forms[head] - 1)) == forms[head], identifier



# ----- the examples of a run with the CausalDiagnosis MCP -----

def _diagnosis_tool(start):
    """A stand-in of the listed diagnosis tool, whose parameters are the start's components and evidence_ids."""
    return SimpleNamespace(name=examples.DIAGNOSIS_TOOL, args_schema=SimpleNamespace(model_fields=dict.fromkeys([*start, "evidence_ids"])))


@pytest.mark.parametrize("servers", [WITH_MCP, WITHOUT_MCP], ids=["with the MCP", "control: without it"])
def test_on_a_run_with_the_mcp_every_example_written_for_the_port_shows_the_assets_site_and_field(rig_start, servers):
    """Where the servers list the CausalDiagnosis MCP, the examples written for this port show the asset's site in place of SiteX and
    SiteY, the site list of TASK_SUPERVISOR as the asset's one site, and cite the asset's field in every identifier of a row, and the
    WO agent's examples name the asset, the first faulty component of the example start and its child asset in place of the made-up
    chiller; the examples of AssetOpsBench and the IoT agent's, adapted from them, stay as they are. Without the MCP every text is
    the module's."""
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_tool_contract import child_asset

    row = sub_agent_registry.mcp_row(servers, RIG_ASSET, rig_start)
    profile = get_profile(RIG_ASSET)
    specs = {s.name: s for s in sub_agent_registry.registration(servers, "rig")}
    task_tools = [*(SimpleNamespace(name=n) for n in examples.TASK_TOOLS if n != examples.DIAGNOSIS_TOOL), _diagnosis_tool(rig_start)]
    own = {name: specs[name].own_examples(tools, rig_start, row) for name, tools in (
        ("TaskAgent", task_tools), ("WOAgent", []), ("FMSRAgent", []), ("IoTAgent", []))}
    task_supervisor = specs["TaskAgent"].supervisor_examples_for(row)
    merged = sub_agent_registry.merged_examples(list(specs.values()), row)
    assert own["IoTAgent"] == "\n\n".join(examples.IOT_RIG)
    assert own["FMSRAgent"] == ("\n\n".join(examples.FMSR_RIG) if row is None else examples.fmsr_example(row))  # with the MCP the asset's or a component's class
    assert all(examples.PAPER_SUPERVISOR[name] in merged for name in ("IoTAgent", "FMSRAgent", "WOAgent"))  # AssetOpsBench's, SiteX and all
    cited = re.findall(r"obs/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/(Supply Temperature|\w+)", own["TaskAgent"] + task_supervisor)
    if row is None:
        assert own["WOAgent"] == "\n\n".join(examples.WO_RIG) and task_supervisor == examples.TASK_SUPERVISOR
        assert set(cited) == {examples.MADE_UP_FIELD}
        return
    site, first_group = profile.text["fallback"]["site"], next(iter(profile.observation_kinds.values()))
    component = next(c for c in profile.components if rig_start[c] != profile.healthy[c])
    assert (row.site, row.field, row.asset, row.component, row.child) == (site, profile.field_name(first_group["channels"][0]), RIG_ASSET,
                                                                          component, child_asset(RIG_ASSET, component))
    for text in (own["WOAgent"], own["TaskAgent"], task_supervisor):
        assert not re.search(r"Site[XYZ]", text)
    assert own["WOAgent"].count(f"site_id={site}") == 4 and f'{{"sites": ["{site}"]' in task_supervisor
    assert len(cited) >= 8 and set(cited) == {row.field}
    assert examples.MADE_UP_ASSET not in own["WOAgent"] and examples.MADE_UP_PART not in own["WOAgent"]
    assert own["WOAgent"].count(f"asset_num={row.child}") == 1 and own["WOAgent"].count(f'"assetnum": "{row.child}"') == 3


@pytest.mark.parametrize("servers", [WITH_MCP, WITHOUT_MCP], ids=["with the MCP", "control: without it"])
async def test_on_a_run_with_the_mcp_the_profile_examples_and_the_supervisors_diagnosis_describe_the_asset_of_the_run(world, rig_start, servers):
    """Where the servers list the CausalDiagnosis MCP, the TaskAgent's example of get_asset_profile shows in every key it shows what
    the task server answers for the asset of the run (the raw profile of an episode of its rows, whose preset keeps the asset's
    default budget and horizon), its Finish line and TASK_SUPERVISOR's reply restate it, and the diagnosis of TASK_SUPERVISOR names
    the levels that the diagnosis example of the task tools shows. Without the MCP every text is the module's, which holds every
    text written from the made-up chiller (examples.ROW_TEXTS), so a replacement cannot silently miss."""
    row = sub_agent_registry.mcp_row(servers, RIG_ASSET, rig_start)
    specs = {s.name: s for s in sub_agent_registry.registration(servers, "rig")}
    task_tools = [*(SimpleNamespace(name=n) for n in examples.TASK_TOOLS if n != examples.DIAGNOSIS_TOOL), _diagnosis_tool(rig_start)]
    profile_example = next(e for e in specs["TaskAgent"].own_examples(task_tools, rig_start, row).split("\n\n")
                           if first_call(e)[0] == "get_asset_profile")
    supervisor = specs["TaskAgent"].supervisor_examples_for(row)
    if row is None:
        assert profile_example == examples.TASK_TOOLS["get_asset_profile"] and supervisor == examples.TASK_SUPERVISOR
        assert all(written(examples.MADE_UP_ROW) in profile_example + supervisor for written in examples.ROW_TEXTS)
        return
    meter, _engine, store, _ = harness.memory_engine_episode(world, "repair_rules")
    with bound(store, harness.NAMES):
        real = harness.body(await meter.call(TASK_SERVER, "get_asset_profile", {"response_format": "raw"}))
    [(_tool, shown)] = tool_observations(profile_example)
    assert {key: real[key] for key in shown if key != "asset"} == {key: value for key, value in shown.items() if key != "asset"}
    assert {key: real["asset"][key] for key in shown["asset"]} == shown["asset"]
    assert finish_line(profile_example) == examples.profile_finish(row) and examples.profile_summary(row) in supervisor
    named = examples.diagnosis_named(row)
    assert named in examples.diagnosis_example([*rig_start, "evidence_ids"], rig_start) and supervisor.count(named) == 3
    for text in (profile_example, supervisor):
        assert examples.MADE_UP_ASSET not in text and "CHILLER" not in text and "compressor" not in text and not re.search(r"Site[XYZ]", text)


@pytest.mark.parametrize("servers", [WITH_MCP, WITHOUT_MCP], ids=["with the MCP", "control: without it"])
async def test_on_a_run_with_the_mcp_the_repair_manual_the_failure_modes_and_the_causal_examples_are_the_assets(world, rig_start, servers):
    """Where the servers list the CausalDiagnosis MCP, the TaskAgent's lookup_fix example shows in every key it shows what the task
    server answers for the first faulty component of the example start, the FMSR example what fmsr answers for the asset's own
    class where it files codes under it, else for the first class with codes among those of that component and the other
    components, and the examples of the CausalDiagnosis MCP and CAUSAL_SUPERVISOR name the asset, its components and its served
    fields, show only fields of each tool's result model, rank only combinations of the start's and the healthy levels, and price
    the replacement as the MCP does (part, repair charge and visit charge; lead time and repair downtime). Their readings and
    probabilities are the illustrative values of the module. Without the MCP every text is the module's, which holds every made-up
    example that ROW_EXAMPLES replaces."""
    row = sub_agent_registry.mcp_row(servers, RIG_ASSET, rig_start)
    specs = {s.name: s for s in sub_agent_registry.SURFACES["rig"]}  # the CausalDiagnosisAgent too, whose texts the control reads
    task_tools = [*(SimpleNamespace(name=n) for n in examples.TASK_TOOLS if n != examples.DIAGNOSIS_TOOL), _diagnosis_tool(rig_start)]
    task = {first_call(e)[0]: e for e in specs["TaskAgent"].own_examples(task_tools, rig_start, row).split("\n\n")}
    causal_tools = [SimpleNamespace(name=n) for n in examples.CAUSAL_TOOLS]
    causal = {first_call(e)[0]: e for e in specs["CausalDiagnosisAgent"].own_examples(causal_tools, rig_start, row).split("\n\n")}
    fmsr = specs["FMSRAgent"].own_examples([], rig_start, row)
    supervisor = specs["CausalDiagnosisAgent"].supervisor_examples_for(row)
    if row is None:
        assert task["lookup_fix"] == examples.TASK_TOOLS["lookup_fix"] and fmsr == "\n\n".join(examples.FMSR_RIG)
        assert causal == examples.CAUSAL_TOOLS and supervisor == examples.CAUSAL_SUPERVISOR
        assert all(made_up in "\n\n".join([*task.values(), *causal.values(), fmsr, supervisor]) for made_up, _ in examples.ROW_EXAMPLES)
        return
    profile = get_profile(RIG_ASSET)
    meter, _engine, store, _ = harness.memory_engine_episode(world, "repair_rules")
    [(_tool, manual)] = tool_observations(task["lookup_fix"])
    [(_tool, modes)] = tool_observations(fmsr)
    with bound(store, harness.NAMES):
        real_manual = harness.body(await meter.call(TASK_SERVER, "lookup_fix", {"component": row.component}))
        real_modes = harness.body(await meter.call("fmsr", "get_failure_modes", {"asset_class": row.asset_class}))
    assert manual["component"] == row.component and {key: real_manual[key] for key in manual} == manual
    assert {key: real_modes[key] for key in modes} == modes and modes["failure_modes"]
    start, healthy = dict(row.levels), dict(row.healthy)
    for name, example in causal.items():
        [(tool, answer)] = tool_observations(example)
        assert tool == name and not set(answer) - result_fields(causal_main, tool), (tool, sorted(set(answer) - result_fields(causal_main, tool)))
    ranked = tool_observations(causal["rank_root_causes"])[0][1]
    for candidate in ranked["candidates"]:
        assert set(candidate["combination"]) == set(profile.components)
        assert all(level in (start[c], healthy[c]) and level in profile.levels[c] for c, level in candidate["combination"].items())
    served = {f"{profile.field_name(s)}_{profile.plain_statistic()}" for s in profile.channels}
    assert set(ranked["evidence_ids"]) <= {f"obs/1/{field}" for field in served}
    effect = tool_observations(causal["estimate_intervention_effect"])[0][1]
    config = profile.default_config()
    assert effect["action"] == f"replace:{row.component}" and effect["duration"] == config.lead_time(row.component) + config.durations["repair_downtime"][row.component]
    assert effect["price"] == config.prices["parts"][row.component] + config.prices["repair_charge"] + config.prices["visit"]
    for text in (*causal.values(), supervisor, fmsr, task["lookup_fix"]):
        assert not any(word in text for word in (examples.MADE_UP_ASSET, "compressor", "condenser", "evaporator", "chiller", "Supply Temperature"))


@pytest.mark.parametrize("servers", [WITH_MCP, WITHOUT_MCP], ids=["with the MCP", "control: without it"])
def test_the_wo_examples_order_on_a_child_asset_at_the_site_of_the_episode_only_on_a_run_with_the_mcp(world, rig_start, servers):
    """Every site_id that an Action Input of the WO agent's examples names, split as the engine splits it, against the site of the
    episode, and the order of the example through the adapter and the checks with which the metering layer refuses an order before
    the engine sees it (metering_layer._validated, _order_problem) and maps it to a component (MeterConfig.component_of). The control
    in the test: the order with the site of the asset alone and the made-up chiller's child asset maps to no component."""
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import metering_layer as layer

    meter, *_ = harness.memory_engine_episode(world, "repair_rules")
    row = sub_agent_registry.mcp_row(servers, RIG_ASSET, rig_start)
    wo = next(s for s in sub_agent_registry.SURFACES["rig"] if s.name == "WOAgent")
    shown = wo.own_examples([], rig_start, row)
    named = [engine_split(given)["site_id"] for example in shown.split("\n\n")
             for _, action, given, _ in steps(example) if action != "Finish" and "site_id=" in given]
    assert len(named) == 4 and all((site.upper() == meter.config.site.upper()) == (row is not None) for site in named)
    schema = layer.SERVERS["wo"].mcp._tool_manager.get_tool("generate_work_order").parameters
    definition = SimpleNamespace(name="generate_work_order", description="", inputSchema=schema)

    def component_of(text):
        order = next(example for example in text.split("\n\n") if first_call(example)[0] == "generate_work_order")
        parsed = layer._validated("wo", "generate_work_order", sent_by_wrapper(definition, engine_split(first_call(order)[1])))
        assert parsed is not None and layer._order_problem(parsed) is None
        return meter.config.component_of(parsed["asset_num"], parsed["site_id"])

    assert component_of(shown) == (row.component if row is not None else None)
    if row is not None:
        assert component_of(examples.on_mcp_row("\n\n".join(examples.WO_RIG), row)) is None  # the site alone does not map the order


@pytest.mark.parametrize("asset, component", [(None, None), ("compressor_7", "motor")],
                         ids=["the rig's asset", "an asset whose name holds a made-up name"])
def test_the_wo_examples_name_the_child_asset_and_the_asset_whole_also_where_the_asset_name_holds_a_made_up_name(rig_start, asset, component):
    """orders_on_mcp_row replaces the made-up chiller's names in one pass, the longest first, so that no name written for the asset is
    read again: every asset_num and assetnum names the child asset of the row's component, and every made-up asset becomes the asset,
    also where the asset's name holds the made-up part (compressor_7 with its component motor). The rig's asset is the control."""
    row = sub_agent_registry.mcp_row(WITH_MCP, RIG_ASSET, rig_start)
    if asset is not None:
        row = dataclasses.replace(row, asset=asset, component=component, child=f"{asset}_{component}")
    made_up = "\n\n".join(examples.WO_RIG)
    shown = "\n\n".join(examples.orders_on_mcp_row(text, row) for text in examples.WO_RIG)
    assert shown.count(f"asset_num={row.child}") == 1 and shown.count(f'"assetnum": "{row.child}"') == 3
    assert shown.count(row.asset) == made_up.count(examples.MADE_UP_ASSET) and examples.MADE_UP_ASSET not in shown


async def test_an_identifier_copied_from_an_example_of_a_run_with_the_mcp_maps_to_no_released_evidence(world, rig_start):
    """The diagnosis and the declaration example of a run with the CausalDiagnosis MCP cite the asset's field at a moment of the
    made-up world (metering_layer.rewrite_evidence), so a copy of them maps to nothing and the provenance check flags it. The control
    in the test: the same field at the timestamp of a row that the episode served maps to the released evidence."""
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import metering_layer as layer

    meter, engine, store, _ = harness.memory_engine_episode(world, "repair_rules")
    row = sub_agent_registry.mcp_row(WITH_MCP, RIG_ASSET, rig_start)
    with bound(store, harness.NAMES):
        served = harness.body(await meter.call("iot", "history", {"site_name": meter.config.site, "asset_id": RIG_ASSET, "limit": 1}))["observations"][0]
    texts = [examples.on_mcp_row(examples.diagnosis_example([*rig_start, "evidence_ids"], rig_start), row),
             examples.on_mcp_row(examples.TASK_TOOLS["declare_insufficient_evidence"], row)]
    own = sorted({i for text in texts for i in identifiers(text)})
    assert own and all(i.endswith("/" + row.field) for i in own)
    assert layer.rewrite_evidence(own, layer._observations(engine), meter.config) == own
    copied = layer.rewrite_evidence([f"obs/{served['timestamp']}/{row.field}"], layer._observations(engine), meter.config)
    assert re.fullmatch(r"obs/\d+/" + re.escape(meter.config.engine_field(row.field)), copied[0]), copied  # in the engine's naming


async def test_the_declaration_and_diagnosis_calls_of_a_run_with_the_mcp_are_accepted_by_the_task_server(world, rig_start):
    """The calls as the examples of a run with the CausalDiagnosis MCP show them, with no name of the made-up world to replace:
    through hub.mcp_tool built from the listed schema, with the Action Input split as the engine splits it."""
    row = sub_agent_registry.mcp_row(WITH_MCP, RIG_ASSET, rig_start)
    for name in ("declare_insufficient_evidence", examples.DIAGNOSIS_TOOL):
        meter, engine, store, _ = harness.memory_engine_episode(world, "repair_rules")  # each write on an episode of its own
        with bound(store, harness.NAMES):
            listed = {react_tool_adapter.agent_name(d.name): d for d in await meter.listed_tools(TASK_SERVER)}
            start = dict(zip(engine.config.components, rig_start.values()))
            text = examples.TASK_TOOLS[name] if name != examples.DIAGNOSIS_TOOL else examples.diagnosis_example(
                [*listed[name].inputSchema["properties"]], start)
            shown, given = first_call(examples.on_mcp_row(text, row))
            reply = await meter.call(TASK_SERVER, listed[shown].name, sent_by_wrapper(listed[shown], engine_split(given)))
            assert not reply.is_error and '"error"' not in reply.text[:20], (name, reply.text[:200])


@pytest.mark.parametrize("servers", [WITH_MCP, WITHOUT_MCP], ids=["with the MCP", "control: without it"])
def test_a_run_with_the_mcp_that_names_no_asset_is_refused_before_any_model_call(rig_start, servers):
    """Runtime.prepare returns the refusal, so that a driver that does not pass the asset of the run stops it with the reason."""
    from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper import sub_agent_runtime

    specs = sub_agent_registry.registration(servers, "rig")
    runtime = sub_agent_runtime.Runtime(None, None, dict.fromkeys(servers, []), None, None, example_start=rig_start)
    refusal = runtime.prepare(specs)
    assert (isinstance(refusal, ValueError) and "asset of the run" in str(refusal)) == (servers == WITH_MCP)
    named = sub_agent_runtime.Runtime(None, None, dict.fromkeys(servers, []), None, None, example_start=rig_start, example_asset=RIG_ASSET)
    assert named.prepare(specs) is None and (named.mcp_row() is None) == (servers == WITHOUT_MCP)
