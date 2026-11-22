"""The core texts of the prompt as named constants (CausalDiagnosis/docs/general-agent-plan.md, section The prompt): the blocks
ROLE, the head of TOOLS, EXPECTATIONS, EVIDENCE, CANDIDATES, BOARD, PLAN with RECALL and BATCH, the head of RULES and LOOP, and the
texts of the four turns, each a constant whose slots are nouns of the vocabulary, names of tools, names of limits or a clause
that the assembler renders. The texts that the present agent prints as well are imported from agents/utils/core_texts.py once
that file is in the tree. A stub of the freeze: the prompt writes the constants."""
