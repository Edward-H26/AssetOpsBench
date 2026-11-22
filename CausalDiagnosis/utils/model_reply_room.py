"""The room a reply has on the served model (SGLang v0.5.19).

No client of this project caps a reply. The chat route sends no max_tokens: the engine then gives the reply the rest of the context by
its own count (the context less the prompt less 2), which is the exact room, while a count made by the client could be one token short
and make every request a refusal. A reply that ends by length is checked against its usage, so an engine that cut earlier is
caught. The completion route defaults an absent max_tokens to 16 tokens, so every request there carries the context less the
tokenize count of the raw prompt. The context is max_model_len of the endpoint's model listing, and a prompt that does not fit
is refused with HTTP 400, which is the end of the context and never a failed attempt to retry.
"""

from __future__ import annotations

from typing import Callable

# the engine's texts of a prompt past the context (TokenizerManager._validate_one_request in managers/tokenizer_manager.py and validate_input_length in managers/utils.py of SGLang v0.5.19)
CONTEXT_REFUSALS = ("is longer than the model's context length", "exceeds the model's maximum context length",
                    "exceeds the maximum allowed length")
ENGINE_MARGIN = 2  # the engine generates at most the context less the prompt less 2 (Scheduler.init_req_max_new_tokens in managers/scheduler.py of SGLang v0.5.19)
# the trigger of compaction by size, a share of the served context that the prompt alone may take (Codex's trigger of 90
# percent); no reply term enters it
COMPACT_TRIGGER = (9, 10)


class NoReplyRoom(RuntimeError):
    """A backend with no endpoint to ask cannot tell the context it serves under."""


class ContextEnd(RuntimeError):
    """The prompt does not fit the context of the served model: the episode ends with end "context", nothing is retried."""


class EarlyCut(RuntimeError):
    """A reply ended by length before the room the engine gives, so something cut it short."""


class ReplyRoom:
    """The context length of the served model and a prompt token count, both read from the endpoint."""

    def __init__(self, context_length: int, count_tokens: Callable[[str], int]) -> None:
        if int(context_length) < 1:
            raise ValueError(f"a context length is a positive count of tokens, not {context_length!r}")
        self.context_length = int(context_length)
        self.count_tokens = count_tokens

    def max_tokens(self, prompt: str) -> int:
        """The completion route's max_tokens: the context less the prompt's tokens; a prompt that leaves none is the end of the context."""
        left = self.context_length - self.count_tokens(prompt)
        if left < 1:
            raise ContextEnd(f"the prompt takes {self.context_length - left} tokens of a context of {self.context_length}")
        return left

    def request_fits(self, prompt: str, longest_reply: int = 0) -> bool:
        """Whether the prompt leaves room for a reply as long as the longest of the episode, and for one token before any reply."""
        return self.count_tokens(prompt) + max(1, int(longest_reply or 0)) <= self.context_length

    def request_under_trigger(self, prompt: str) -> bool:
        """Whether the prompt's tokens alone stay at or below the trigger of compaction, floor(context * 9 / 10), counted as request_fits counts them."""
        numerator, denominator = COMPACT_TRIGGER
        return self.count_tokens(prompt) <= self.context_length * numerator // denominator

    def check_length_finish(self, prompt_tokens: int, completion_tokens: int) -> None:
        """A reply that ended by length must have used the room the engine gives (the context less the prompt less 2)."""
        room = self.context_length - int(prompt_tokens) - ENGINE_MARGIN
        if int(completion_tokens) < room:
            raise EarlyCut(f"the reply ended by length at {completion_tokens} tokens, before the room of {room} tokens "
                           f"(context {self.context_length}, prompt {prompt_tokens})")


def card_context_length(cards: list, model: str) -> int | None:
    """max_model_len of the card of the model in an endpoint's model listing, else of the card of its base model, since SGLang lists
    the adapter of a name base:adapter as a card of its own without a context length and splits the name at its first colon,
    else of the first card; None when that card names none."""
    names = (model, model.split(":", 1)[0])
    card = next((c for name in names for c in cards if c.get("id") == name), cards[0] if cards else {})
    return int(card["max_model_len"]) if card.get("max_model_len") else None


def served_context(base_url: str, api_key: str, model: str, get=None) -> int:
    """max_model_len of the model in the endpoint's listing (GET {base_url}/models), the context the engine checks."""
    get = get or _http("get")
    listing = get(f"{base_url.rstrip('/')}/models", api_key)
    rows = listing.get("data") or []
    row = next((r for r in rows if r.get("id") == model), rows[0] if len(rows) == 1 else None)
    if row is None or not row.get("max_model_len"):
        raise NoReplyRoom(f"the model listing of {base_url} names no max_model_len for {model!r}")
    return int(row["max_model_len"])


def tokenize_counter(base_url: str, api_key: str, model: str, chat: bool = True, extra: dict | None = None, post=None):
    """count(prompt): the token count of the tokenize route (POST {base_url}/tokenize) for the prompt as the route that is called
    renders it: on the chat route the same messages and template arguments, on the completion route the raw prompt."""
    post = post or _http("post")
    url = f"{base_url.rstrip('/')}/tokenize"

    def count(prompt: str) -> int:
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], **(extra or {})} if chat else {"model": model, "prompt": prompt}
        return int(post(url, api_key, body)["count"])

    return count


def endpoint_room(base_url: str, api_key: str, model: str, chat: bool = True, extra: dict | None = None, get=None, post=None) -> ReplyRoom:
    return ReplyRoom(served_context(base_url, api_key, model, get), tokenize_counter(base_url, api_key, model, chat, extra, post))


def is_context_refusal(exc: BaseException) -> bool:
    """Whether an error is the engine's HTTP 400 for a prompt past the context."""
    if getattr(exc, "status_code", None) != 400:
        return False
    text = str(exc) + str(getattr(exc, "body", "") or "")
    return any(phrase in text for phrase in CONTEXT_REFUSALS)


def _http(method):
    def call(url, api_key, body=None):
        import httpx

        from CausalDiagnosis.utils.model_request_retry import post_with_retries

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

        # under the one retry policy of the model clients (CausalDiagnosis.utils.model_request_retry): a lost connection or a status of 429, 502 or 503 is a
        # failed attempt, sent again inside an episode, under a deadline or while the named health probe answers, so one blip of
        # the endpoint never ends a run at the check of the reply room; each attempt has the read bound of the policy and the
        # connection 5 s as the clients' own
        def send(timeout):
            return httpx.request(method.upper(), url, headers=headers, json=body, timeout=httpx.Timeout(timeout, connect=5.0))

        response = post_with_retries(send, model_reply=False)
        response.raise_for_status()
        return response.json()

    return call
