# Replaces src/reactxen/utils/model_inference.py of IBM/ReActXen (https://github.com/IBM/ReActXen) at commit
# da40e4ba3b93c8c9dde100717323a0c6cab11edc, licensed under the Apache License, Version 2.0, a copy of which is LICENSE in the folder
# CausalDiagnosis/frameworks/AssetOpsBench/agents/ReActXen of this repository.
# Changed by the CausalDiagnosis fork of AssetOpsBench: the file keeps the three names the vendored agents import,
# watsonx_llm, count_tokens and get_context_length, with their signatures, and routes them to the SGLang raw completion
# route of CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.text_completion_route instead of watsonx.ai. The watsonx, OpenAI and Azure clients are not copied.
"""The model calls of the vendored ReActXen agents, through the completion route of the running Agent-As-Tool episode.

ReActXen's watsonx_llm sent the prompt text to watsonx.ai text generation with greedy decoding, temperature 0, at most
500 new tokens, at least 1, and the stop sequences of the call, and returned the first result, whose generated_text
ends with the matched stop sequence. CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.text_completion_route sends the same request to SGLang's /v1/completions and
returns a dict with the same keys. count_tokens and get_context_length use the route's tokenizer endpoint and the
served context length.
"""
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.text_completion_route import active_route


def watsonx_llm(prompt, model_id=8, decoding_method="greedy", temperature=0.0, max_tokens=500, n=1, stop=None, seed=None,
                is_system_prompt=False):
    if isinstance(stop, str):
        stop = [stop]
    return active_route().generate(prompt, temperature=temperature, max_tokens=max_tokens, stop=stop, seed=seed)


def count_tokens(input_text, model_id=None, upper_limit=10000000, skip_token_counting=False):
    if skip_token_counting:
        return 1
    return active_route().count_tokens(input_text)


def get_context_length(model_id):
    return active_route().context_length()
