"""The general CausalDiagnosis agent, which plays any dataset from what the dataset declares
(CausalDiagnosis/docs/general-agent-plan.md). The driver plays it as the agent causal_next once its entry lands, until the last
change of the plan gives it the name of the present agent. Modules of this package, one per line:

agent_options
episode_result
schedule
controller
plan
working_memory
prompt
advice
tests
"""
