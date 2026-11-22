import os
import time
from datetime import datetime
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock import token_clock  # noqa: E402
from CausalDiagnosis.utils.llm_backends import LLMBackend  # noqa: E402
from CausalDiagnosis.utils.llm_backends import EndpointChatBackend  # noqa: E402
from . import agent_kinds


CONTEXT_READERS = ("request_fits", "request_under_trigger", "longest_reply", "room")


class RecordingBackend(LLMBackend):
    """Keeps every prompt and reply of an episode with its token counts and duration. With a temperature, the one of a sampling
    run, every call is made at it in place of the temperature the agent asks for, so that no agent changes for the sampling."""

    def __init__(self, inner, temperature=None):
        self.inner = inner
        self.temperature = temperature
        self.calls = []

    def generate(self, prompt, temperature=0.0):
        return self.generate_with_usage(prompt, temperature).text

    def generate_with_usage(self, prompt, temperature=0.0, response_format=None):
        # the day of the call, which the served model's chat template prints into its system message, for rejection_sampling_examples.py
        # started_at and ended_at in wall time place the call between the tool calls of the episode
        call = {"prompt": prompt, "date": datetime.now().date().isoformat(), "started_at": time.time()}
        self.calls.append(call)
        started = time.perf_counter()
        try:
            result = self.inner.generate_with_usage(prompt, temperature if self.temperature is None else self.temperature)
        except Exception as exc:
            call["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            call["seconds"] = round(time.perf_counter() - started, 2)
            call["ended_at"] = time.time()
        call.update(reply=result.text, input_tokens=result.input_tokens, output_tokens=result.output_tokens,
                    finish_reason=getattr(result, "finish_reason", None))  # a backend with the released LLMResult reports none
        token_clock.add_generated(result.output_tokens)
        return result

    @property
    def model_id(self):
        return self.inner.model_id

    @property
    def takes_response_format(self):
        """Whether the inner route can bind a reply to a JSON schema, so that the causal runner sends one only where it is carried."""
        return bool(getattr(self.inner, "takes_response_format", False))

    def request_settings(self, temperature=0.0):
        """The blocks decoding and transport of the inner route, at the temperature of a sampling run when it has one."""
        return self.inner.request_settings(temperature if self.temperature is None else self.temperature)

    def __getattr__(self, name):
        """The context readers of the inner backend (request_fits, request_under_trigger, longest_reply), present here exactly
        when the inner backend has them: the causal runner reads them by getattr, and a wrapper without them would make its
        context check, its compaction trigger and its recall drop take every request as fitting."""
        if name in CONTEXT_READERS:
            return getattr(self.inner, name)
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

    def __setattr__(self, name, value):
        # the summary turn of the compaction puts longest_reply back after its request, so that the summary reply stays out
        # of the reply term; the inner backend reads its own longest_reply in request_fits, so the write must reach it
        if name == "longest_reply" and hasattr(self.__dict__.get("inner"), name):
            setattr(self.inner, name, value)
        else:
            super().__setattr__(name, value)


def full_group_of(groups):
    """The sensor group with the most channels, the first of them on a tie: the full sensor group of an asset."""
    return max(groups, key=lambda group: len(groups[group]))


def router_backend(model, base_url=agent_kinds.ROUTER_BASE_URL):
    key = os.environ.get("API_KEY")
    if not key:
        raise SystemExit("API_KEY is not set; configure a key or run with --agent scripted")
    # the endpoint is bound to the backend, so the agent's and the judge's backends never share one address
    return EndpointChatBackend(f"tokenrouter/{model}", base_url=base_url, api_key=key)
