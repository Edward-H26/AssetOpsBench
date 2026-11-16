"""Parts of IBM's ReActXen, https://github.com/IBM/ReActXen at commit da40e4ba3b93c8c9dde100717323a0c6cab11edc, under the
Apache License, Version 2.0 in the LICENSE file of this folder.

The paper-era Agent-As-Tool of CausalDiagnosis/frameworks/AssetOpsBench/agents/Agent_As_Tool_Paper runs on ReActXen's ReactReflectAgent with its reviewer and on its
JSONReader, whose import closure is copied here, every file with a header that names its source and what changed.
utils/model_inference.py is the fork's own, which sends the model calls to SGLang instead of watsonx.ai.
THIRD_PARTY_NOTICES.md at the root of the repository lists what was adapted.
"""
