"""The in-context examples of the two ports, read without a model, since the paper's appendix prints them. Every example written for the Agent-As-Tool port
on the rig surface shows each answer of a tool as JSON in keys that the answer has, with made-up values of a made-up world, and its
Finish line relays a value that its observation returned. Its Action Input names the arguments of the tool's listed schema, and the
call it shows is accepted by the servers of a rig row. The one diagnosis example is written at run time from the tool contract and
the example start, and the levels of that start are the only levels of a registered asset in any text of either port. The
CausalDiagnosis MCP examples are read against the result model each tool's function returns in CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/main.py and
called on the server over the released file of an episode, the rig examples of the IoT, FMSR and WO agents against those of
servers/iot, servers/fmsr and servers/wo, and the task tool examples against the answers of the task server of the AssetOpsBench rows
on the repair rules row, which lists every task tool."""
from CausalDiagnosis.utils.paths import PACKAGE
import ast
import asyncio
import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path

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


def result_model(main, tool):
    """The model besides ErrorResult in the return annotation of the tool's function in a server's main.py."""
    tree = ast.parse(Path(main.__file__).read_text(encoding="utf-8"))
    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == tool]
    assert len(functions) == 1, (main.__name__, tool)
    names = [n.id for n in ast.walk(functions[0].returns) if isinstance(n, ast.Name) and n.id not in ("Union", "Optional", "ErrorResult")]
    assert len(names) == 1, (tool, names)
    return getattr(main, names[0])


@pytest.fixture(scope="module")
def rig_start(world):
    """{component: level} of the rig's example start, the development start the registered seed draws (example_start)."""
    _cycles, blocks = world
    profile = get_profile("hyd_1")
    return dict(zip(profile.components, example_start(profile, set(blocks))))


def ours(start):
    """The examples written for the rig surface by table, with the diagnosis example written for the start."""
    parameters = [*start, "evidence_ids"]
    task = [examples.diagnosis_example(parameters, start) if name == examples.DIAGNOSIS_TOOL else text for name, text in examples.TASK_TOOLS.items()]
    return {"IOT_RIG": examples.IOT_RIG[3:], "FMSR_RIG": examples.FMSR_RIG, "WO_RIG": examples.WO_RIG, "TASK_TOOLS": task,
            "CAUSAL_TOOLS": list(examples.CAUSAL_TOOLS.values())}


def returned_values(example):
    """Every leaf value of the example's observations that neither its question nor its Action Inputs before Finish hold."""
    given = question(example) + " " + " ".join(i for _, a, i, _ in steps(example) if a != "Finish")
    return [value for _, answer in tool_observations(example) for value in leaves(answer) if not token_in(value, given)]


def relays_a_returned_value(example):
    returned = returned_values(example)
    return not returned or any(token_in(value, finish_line(example)) for value in returned)


def test_every_finish_line_of_ours_carries_a_value_that_its_observation_returned(rig_start):
    with_values = 0
    for table, texts in ours(rig_start).items():
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
        unknown = set(answer) - set(result_model(causal_main, tool).model_fields)
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
    task = dict(zip(examples.TASK_TOOLS, ours(rig_start)["TASK_TOOLS"]))  # the diagnosis example as written for the rig's example start
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

    for name, (is_error, answer) in asyncio.run(replay()).items():
        assert not is_error and "error" not in answer, (name, answer)


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
        written, fixed = [], []
        for servers in (("iot", "fmsr", "wo", "task"), ("iot", "fmsr", "wo", "task", "causal_diagnosis")):
            specs = sub_agent_registry.registration(servers, "rig")
            fixed.append(sub_agent_registry.merged_examples(specs))
            for spec in (s for s in specs if s.kind == "react"):
                tools = [react_tool_adapter.mcp_tool(d.name, d.description, d.inputSchema or {}, lambda sent: "CALL")
                         for server in spec.servers for d in definitions.get(TASK_SERVER if server == "task" else server, [])]
                own = spec.own_examples(tools, start)
                written.append(own)
                fixed.append("\n".join(workflow.task_examples(own)))
        fixed += [text for text in module_texts() if text != examples.DIAGNOSIS_TEMPLATE]  # every fixed text of the examples
        find = texts.finder(asset)
        diagnosis = examples.diagnosis_example([*start, "evidence_ids"], start)
        for text in fixed + written:
            outside = text.replace(diagnosis, "")
            for line in workflow.task_examples(diagnosis):  # the planner lists the question of the diagnosis example
                outside = outside.replace(line, "")
            assert not find(outside), find(outside)[:3]
        hits = find(diagnosis)  # the levels of the example start, and no other number
        assert hits and all(set(map(int, re.findall(r"\d+", matched))) <= set(start.values()) for _, matched in hits)
        assert any(diagnosis in text for text in written)


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


def test_every_reply_in_a_supervisor_example_of_ours_has_content():
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
    found += re.findall(r"(?:evidence|citing) ((?:obs|wo|fmea|alarm|pred|ev)[^ ,]*(?: [A-Z][a-z]+)*)", example)
    return found


def test_every_evidence_identifier_of_an_example_of_ours_has_a_form_of_the_surface(rig_start):
    forms = {}
    for form in (*briefing.EVIDENCE_FORMS, briefing.ALARM_FORM, briefing.PREDICTION_FORM):
        head = form.split(" ")[0]
        forms[head.split("/")[0]] = head.count("/") + 1
    checked = [i for texts in (*ours(rig_start).values(), list(SUPERVISOR.values())) for example in texts for i in identifiers(example)]
    assert len(checked) >= 5
    for identifier in checked:
        head = identifier.split("/")[0]
        assert head in forms and len(identifier.split("/", forms[head] - 1)) == forms[head], identifier
