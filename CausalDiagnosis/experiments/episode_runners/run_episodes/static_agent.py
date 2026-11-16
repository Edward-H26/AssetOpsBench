import re
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import tool_surface  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv  # noqa: E402
from CausalDiagnosis.utils import frameworks  # noqa: E402
from . import model_backends


# ----- the static agent (ablation A3) -----
FAULT_WORDS = re.compile(r"\b(fault\w*|degrad\w*|fail\w*|leak\w*|worn|abnormal\w*|critical\w*|severe\w*|reduced|low|lag\w*|"
                         r"below|off|defect\w*|malfunction\w*|insufficient|poor)\b", re.IGNORECASE)
HEALTHY_WORDS = re.compile(r"\b(healthy|normal\w*|nominal|ok|fine|intact|within (?:its |the )?(?:range|band)|no (?:fault|issue|problem))\b", re.IGNORECASE)
REPAIR_WORDS = re.compile(r"\b(replac\w*|repair\w*|work order\w*|servic\w*|overhaul\w*|fix\w*|swap\w*)\b", re.IGNORECASE)
NEGATION = re.compile(r"\b(not|no|never|without|cannot|unable|isn't|aren't|wasn't|doesn't|don't)\b", re.IGNORECASE)
SENTENCES = re.compile(r"[.!?;\n]+")  # a semicolon ends a clause too, since answers list one component per clause


def parse_static_answer(answer, components):
    """A conservative reading of a free text answer: a component is called faulty when a clause names it with a fault
    word and no healthy word, and called repaired when a clause names it with a repair word; a clause with a negation
    counts for neither. The clauses behind every call are kept, so a reader can check the rule on the report."""
    faulty, repaired, evidence = [], [], {c: [] for c in components}
    for sentence in (s.strip() for s in SENTENCES.split(answer or "") if s.strip()):
        named = [c for c in components if re.search(rf"\b{c}\b", sentence, re.IGNORECASE)]
        if not named or NEGATION.search(sentence):
            continue
        is_faulty = bool(FAULT_WORDS.search(sentence)) and not HEALTHY_WORDS.search(sentence)
        is_repaired = bool(REPAIR_WORDS.search(sentence))
        for c in named:
            if is_faulty or is_repaired:
                evidence[c].append(sentence)
            if is_faulty and c not in faulty:
                faulty.append(c)
            if is_repaired and c not in repaired:
                repaired.append(c)
    return {"faulty": faulty, "repaired": repaired, "sentences": evidence}


def full_sensor_group(config):
    """The full sensor group of a configuration, the one with the most channels, so that the first served cycle is the one the
    static answer sees."""
    return model_backends.full_group_of(config.sensor_groups)


def first_read(env, group):
    """The first read of the full sensor group on the surface of an environment: the read the declared surface whose engine the
    environment is gives (on the AssetOpsBench rows the latest reading of every served field), on the rig's action server a read
    of one cycle."""
    surface = next((tool_surface(name) for name in frameworks.surfaces() if isinstance(env, tool_surface(name).env_class)), None)
    if surface is not None:
        return surface.first_read(env, group)
    return {"type": "read_sensors", "sensor_group": group, "cycles": 1}


def static_context(cycles, blocks, config, seed, env_class=HydraulicRigEnv):
    """What the static answer sees besides the question: the features of the recorded cycle the gym would have served
    first (the features of that cycle as the data file holds them, never its labels) and the failure modes
    the gym's get_failure_modes returns. Both are read from a throwaway environment of env_class, the class of the surface of the
    run, so that the first cycle is the one the first read of that surface serves, and the episode's own state file records no
    action."""
    env = env_class(cycles, blocks, config)
    env.reset(seed)
    env.step(first_read(env, full_sensor_group(config)))
    served = next(e for e in reversed(env.log) if e["event"] == "read")["served"][0]
    cycle = next(c for c in cycles if c.index == served["cycle"])
    envelope, _cost, _done, _info = env.step({"type": "get_failure_modes", "component": ""})
    return {"cycle_index": served["cycle"], "features": dict(sorted(cycle.features.items())), "failure_modes": envelope["result"]["failure_modes"]}


def static_prompt(question, context, components, left_out=()):
    # the sentences of the question on steps and wall time hold for an agent that calls tools, and this prompt allows none
    for sentence in left_out:
        question = question.replace(f"{sentence} ", "")
    facts = "\n".join(f"- {name}: {value}" for name, value in context["features"].items())
    modes = "\n".join(f"- {m['component']} at level {m['level']}: {m['code']} ({m['severity']}), recommended action {m['action']}, "
                      f"indicated by {', '.join(m['channels'])}" for m in context["failure_modes"])
    return (f"{question}\n\nYou cannot call any tool. Answer from the facts below.\n\n"
            f"Facts of the current operating cycle of the asset (feature: value):\n{facts}\n\n"
            f"Failure modes of the asset:\n{modes}\n\n"
            f"Answer in prose: give the condition level of each component ({', '.join(components)}), name the components "
            "you judge faulty, and name the work orders (component replacements) you would create.")


STATIC_KIND = "recorded_levels"  # the evidence kind whose records hold the fields that static_context shows
