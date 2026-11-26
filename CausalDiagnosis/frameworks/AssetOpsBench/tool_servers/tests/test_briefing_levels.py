"""The question of an AssetOpsBench row at both tool levels: on every preset of the five grid files, at guidance 1 to
3, the question names no tool of this project that the row does not serve, states the declaration and its scoring at both
levels, and tells every repair rule that plays, the emergency order
and the fix history included, whether or not the row serves the manual."""
import dataclasses

import pytest

from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.recorded_history_env import RecordedRigEnv
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import tool_contract as contract
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import aob_tool_contract as layer_contract
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_briefing import BRIEFING_VERSION, briefing, fix_history_sentence
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_tool_contract import AOB_SERVERS, TASK_SERVER
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.briefing import (DECLARATION_RULE, DECLARE_STEP, REPAIR_SENTENCES, WALL_END_SENTENCE, idle_turn_sentence,
                                            repair_cards, repair_step)

from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tests.harness import preset_config
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tests.test_tool_levels import PRESETS

OURS = tuple(contract.tools_of(contract.ACTION_SERVER))


def question(world, preset, ours, level):
    cycles, blocks = world
    config = dataclasses.replace(preset_config(preset, diagnostics_server=ours), guidance_level=level)
    env = RecordedRigEnv(cycles, blocks, config)
    env.reset(7)
    return briefing(env.public_profile(), config), config


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("ours", (False, True))
@pytest.mark.parametrize("level", (1, 2, 3))
def test_the_question_names_only_served_tools_and_tells_every_rule_that_plays(world, preset, ours, level):
    text, config = question(world, preset, ours, level)
    served = {name for server in (*AOB_SERVERS, TASK_SERVER) for name in layer_contract.tools_for(server, config)}  # fmsr shares a name
    assert not [name for name in OURS if name in text and name not in served]
    assert "declaration of insufficient evidence" in text  # the three answer tools at both levels
    if level >= 2:  # the lines of the task tools and their rules start at guidance 2
        assert "declare_insufficient_evidence" in text and DECLARATION_RULE in text
    if layer_contract.tools_for(TASK_SERVER, dataclasses.replace(config, diagnostics_server=True)).count("lookup_fix"):
        cards = repair_cards(config)
        wanted = [fix_history_sentence("rig") if card == "fix_history" else REPAIR_SENTENCES[card] for card in cards]
        assert all(sentence in text for sentence in wanted)
        if level >= 2 and "emergency" in cards:
            assert "With work_type EM" in text


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("ours", (False, True))
def test_the_question_tells_the_turn_without_a_call_and_takes_the_procedure_of_the_rig(world, preset, ours):
    """Levels 2 and 3 state the step of a turn without a call, with the rule that a call of the diagnostics server is no call of
    the rig where the row registers that server, and the scored end at the wall limit in the rig's words, level 1 neither;
    level 3 declares when the evidence cannot separate the remaining candidates and creates the orders one after another as
    one visit, as the rig's procedure does, with the manual and the wait only where the task server lists them."""
    texts = {}
    for level in (1, 2, 3):
        texts[level], config = question(world, preset, ours, level)
    assert [texts[level].count(idle_turn_sentence("rig", ours)) for level in (1, 2, 3)] == [0, 1, 1]
    assert ("diagnostics server is no call of the rig" in texts[2]) == ours
    assert [texts[level].count(WALL_END_SENTENCE) for level in (1, 2, 3)] == [0, 1, 1]
    listed = layer_contract.tools_for(TASK_SERVER, config)
    step = repair_step(config, " on the child asset of its component", manual="lookup_fix" in listed, waits="wait_for_notification" in listed)
    procedure = texts[3][texts[3].index("Procedure: "):]
    assert procedure.count(DECLARE_STEP) == 1 and procedure.count(step) == 1 and procedure.index(DECLARE_STEP) < procedure.index(step)
    # the visit, its wait and its dependencies only where the row makes them, the engine's own condition
    later = config.has_clock() or config.repair_card_on()
    assert ("one visit repairs" in step) == later and ("after the components it depends on" in step) == ("dependencies" in repair_cards(config))
    assert ("wait for the notifications of their closes" in step) == (later and "wait_for_notification" in listed)
    if not later:  # base, healthy_control and priced: an order closes inside its call, so no visit and no wait
        assert step == "create one work order per diagnosed fault on the child asset of its component, one after another, with no other call between them"
    assert BRIEFING_VERSION == "aob-v1.20"
