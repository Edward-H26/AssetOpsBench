"""The token twins of the clock presets and the harness that feeds the clock of generated tokens: every twin is its
wall-clock preset with the switch clock_mode at generated_tokens and nothing else changed, and every harness of an agent adds
the tokens its model generated in a call to the ledger of the episode before the reply goes back, and adds nothing for a call
that failed or outside an episode of that mode."""
import json

import httpx
import pytest

from CausalDiagnosis.utils.paths import preset_path  # noqa: E402
from CausalDiagnosis.experiments.table_definitions import experiment_tables as design
from CausalDiagnosis.experiments.episode_runners import run_episodes
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper import text_completion_route as completion
from CausalDiagnosis.experiments.episode_runners.run_configuration import CONFIGS, load_config
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock import token_clock, wall_clock
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers import layer_2_real_time as real_time
from llm import LLMBackend
from llm.base import LLMResult


def _file(name):
    return json.loads(preset_path(name).read_text(encoding="utf-8"))


def test_the_token_presets_differ_from_their_wall_presets_in_the_clock_mode_alone():
    """Each rung of the ladder whose clock charges the wall seconds has its twin, the same file but for the name, the description
    and the switch clock_mode at generated_tokens, and the composed configurations differ in the clock mode and the rate of the
    asset file alone."""
    clocked = {preset for preset in design.LADDER if wall_clock.charges_wall_seconds(load_config(preset)[1])}
    assert set(design.TOKEN_PRESETS.values()) == clocked and clocked
    for twin, preset in design.TOKEN_PRESETS.items():
        wall_file, twin_file = _file(preset), _file(twin)
        assert twin_file["name"] == twin == f"{preset}{design.TOKEN_SUFFIX}"
        rest = {key for key in wall_file if key not in ("name", "description", "switches")}
        assert {key: twin_file[key] for key in rest} == {key: wall_file[key] for key in rest} and set(twin_file) - {"switches"} == set(wall_file) - {"switches"}
        assert twin_file["switches"] == {**wall_file.get("switches", {}), "clock_mode": real_time.TOKEN_MODE}
        wall, token = load_config(preset)[1].to_json(), load_config(twin)[1].to_json()
        assert token["clock_mode"] == real_time.TOKEN_MODE and token["seconds_per_token"] > 0 and "seconds_per_token" not in wall
        assert {k: v for k, v in token.items() if k not in ("clock_mode", "seconds_per_token")} == {k: v for k, v in wall.items() if k != "clock_mode"}


class _Inner(LLMBackend):
    """A model that answers with a fixed count of generated tokens, or fails."""

    def __init__(self, tokens, fail=False):
        self.tokens, self.fail = tokens, fail

    def generate(self, prompt, temperature=0.0):
        return self.generate_with_usage(prompt, temperature).text

    def generate_with_usage(self, prompt, temperature=0.0):
        if self.fail:
            raise RuntimeError("no reply")
        return LLMResult(text="reply", output_tokens=self.tokens)


def test_the_recording_backend_and_the_completion_route_append_the_generated_tokens_of_each_call_before_returning(tmp_path):
    """The recording backend of the driver and the completion route of the two paper-era ports add the tokens of a call to the
    ledger before the reply returns, a failed call adds nothing, and outside an episode or without a ledger nothing is written."""
    token_clock.open_ledger(tmp_path)
    backend = run_episodes.RecordingBackend(_Inner(7))
    with token_clock.episode(tmp_path):
        assert backend.generate_with_usage("prompt").output_tokens == 7 and token_clock.total(tmp_path) == 7
        with pytest.raises(RuntimeError):
            run_episodes.RecordingBackend(_Inner(5, fail=True)).generate_with_usage("prompt")
        assert token_clock.total(tmp_path) == 7

        def answer(request):
            return httpx.Response(200, json={"choices": [{"text": "Action", "finish_reason": "stop", "matched_stop": "\nObservation"}],
                                             "usage": {"prompt_tokens": 3, "completion_tokens": 11}})

        route = completion.CompletionRoute("http://endpoint.example/v1", "local", "stand-in", client=httpx.Client(transport=httpx.MockTransport(answer)))
        assert route.generate("prompt", stop=["\nObservation"])["generated_token_count"] == 11 and token_clock.total(tmp_path) == 18
    backend.generate_with_usage("prompt")
    assert token_clock.total(tmp_path) == 18  # outside the episode the harness writes nothing
    other = tmp_path / "other_mode"
    other.mkdir()
    with token_clock.episode(other):
        backend.generate_with_usage("prompt")
    assert token_clock.total(other) is None and not list(other.iterdir())
