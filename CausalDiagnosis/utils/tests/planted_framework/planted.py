"""The code that the planted framework's declaration names (Planted_Check/__init__.py): objects the core resolves by name, each of
which only says that it was reached."""
from types import SimpleNamespace

AGENT = SimpleNamespace(name="planted agent")
SURFACE = SimpleNamespace(name="planted surface", harness_server="planted_task")
JUDGE = SimpleNamespace(name="planted judge")
RUN_KIND = SimpleNamespace(name="planted_run", after_the_fact=False, not_replayed="not replayed, a planted run")
FAMILY = SimpleNamespace(scripted=lambda facts: {"scripted.planted": "reached"},
                         texts=lambda world, name, config, facts: {f"texts.planted.{name}": "reached"},
                         agent_texts=lambda question, profile, envelope, facts, detail: {"texts.agent.planted": "reached"})


def profile_problems(profile):
    return ["the planted surface's profile check ran"]
