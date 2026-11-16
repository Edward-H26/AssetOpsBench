"""The workflow of IBM's paper-era Plan-Execute, AgentHive on branch archive/legacy-tmp of
IBM/AssetOpsBench at commit 27552e9b (src/tmp/agent_hive), as tests/multi_agent/run.py ran it:
PlanningReviewWorkflow with summarization on, SequentialWorkflow with ContextType.SELECTED, PlanReviewerAgent and
SummarizationAgent, each step on its agent with the outputs of its dependencies as context.

The planner writes a plan of #Task, #Agent, #Dependency and #ExpectedOutput lines and the reviewer judges it. An
invalid plan goes back to the planner with the reviewer's reason and suggestion, at most five plans, and the last plan
runs when none is valid. The plan's steps run in order on their agents and a summarization step answers the question
from every step's output. The model is called through watsonx_llm of the vendored ReActXen, which reaches the
completion route of the running episode. One change from the source: tests/multi_agent/run.py retried the whole
workflow up to seven times when it raised, and an episode of the environment cannot be played again, so an error ends
the run instead."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from CausalDiagnosis.frameworks.AssetOpsBench.agents.Plan_Execute_Paper import prompts
from CausalDiagnosis.frameworks.AssetOpsBench.agents.ReActXen.utils.model_inference import watsonx_llm

PLANS = 5  # PlanningReviewWorkflow.max_retries
REVIEW_ATTEMPTS = 3  # PlanReviewerAgent(max_retries=3)
SUMMARY_ATTEMPTS = 3  # SummarizationAgent(max_retries=3)
END_OF_RESPONSE = ["\n(END OF RESPONSE)"]
NO_ANSWER = "not enough information available"
PLANNER, REVIEWER, SUMMARIZER = "planner", "reviewer", "summarizer"  # the names the route records these calls under
SUMMARY_AGENT = "SummarizationAgent"  # the agent name of the last history entry, as the archived runner logs it


@dataclass
class PlanAgent:
    """An agent the planner can name: AgentHive's name, description and task examples, and the call that runs a step."""
    name: str
    description: str
    task_examples: list
    execute: Callable[[str], str]


@dataclass
class Step:
    description: str
    expected_output: str
    agent: PlanAgent
    context: list = field(default_factory=list)  # the indexes of the steps whose outputs this step reads


def task_examples(few_shots):
    """ReactReflectAgent's task examples when none are given: the Question lines of its few-shots."""
    return [example.strip() for example in re.findall(r"^Question:(.*)$", few_shots, re.MULTILINE)]


def agent_descriptions(agents):
    text = ""
    for ii, agent in enumerate(agents):
        text += f"\n({ii + 1}) Agent name: {agent.name}"
        text += f"\nAgent description: {agent.description}"
        if agent.task_examples:
            text += "\nTasks that agent can solve:"
            for idx, example in enumerate(agent.task_examples, start=1):
                text += f"\n{idx}. {example}"
        text += "\n"
    return text


def planner_prompt(question, descriptions, prev_plan, prev_review):
    invalid = ""
    if prev_plan:
        invalid = prompts.INVALID_PLAN.format(prev_plan=prev_plan, invalid_reason=prev_review["reasoning"],
                                              invalid_suggestions=prev_review["suggestions"])
    return prompts.PLANNER.format(agent_descriptions=descriptions, task_description=question, invalid_plan_description=invalid)


def parse_review(response):
    """PlanReviewerAgent.extract_and_parse_json."""
    try:
        match = re.search(r"\{.*\}", response.strip(), re.DOTALL)
        block = match.group(0).strip() if match else response.strip()
        if not block:
            raise ValueError("Extracted JSON block is empty.")
        return json.loads(block)
    except json.JSONDecodeError as ex:
        return {"status": "Error", "reasoning": f"The extracted JSON block could not be parsed. {ex}",
                "suggestions": "Ensure the LLM outputs valid JSON inside the ```json``` block."}
    except ValueError as ex:
        return {"status": "Error", "reasoning": str(ex),
                "suggestions": "Check if the extracted JSON block is empty or improperly formatted."}


def parse_review_by_hand(response):
    """PlanReviewerAgent.extract_and_parse_json_using_manual_parser."""
    cleaned = response.strip().replace("\n", " ").replace("\\n", " ").replace("\\", "")
    status = re.search(r'"status":\s*"([^"]+)"', cleaned)
    reasoning = re.search(r'"reasoning":\s*"([^"]+)"', cleaned)
    suggestions = re.search(r'"suggestions":\s*"([^"]+)"', cleaned)
    if status and reasoning and suggestions:
        return {"status": status.group(1), "reasoning": reasoning.group(1), "suggestions": suggestions.group(1)}
    return {"status": "Error", "reasoning": "The extracted JSON block could not be parsed.",
            "suggestions": "Ensure the LLM outputs valid JSON inside the ```json``` block."}


def review(question, descriptions, plan, model):
    """PlanReviewerAgent.execute_task: up to three calls, each read as JSON and then by hand."""
    prompt = prompts.REVIEWER.format(question=question, agent_expertise=descriptions, plan=plan)
    for _ in range(REVIEW_ATTEMPTS):
        text = watsonx_llm(prompt, model_id=model, stop=END_OF_RESPONSE)["generated_text"]
        parsed = parse_review(text)
        if parsed.get("status") != "Error":
            return parsed
        parsed = parse_review_by_hand(text)
        if parsed.get("status") != "Error":
            return parsed
    return {"status": "Error", "reasoning": f"Failed to produce valid JSON after {REVIEW_ATTEMPTS} attempts.",
            "suggestions": "Review the prompt and refine the LLM response strategy."}


def plan_steps(plan, agents):
    """The steps of a plan as generate_steps reads them, an unknown agent name falling back to the first agent."""
    tasks = re.findall(r"#Task\d+: (.+)", plan)
    names = re.findall(r"#Agent\d+: (.+)", plan)
    dependencies = re.findall(r"#Dependency\d+: (.+)", plan)
    outputs = re.findall(r"#ExpectedOutput\d+: (.+)", plan)
    steps = []
    for i in range(len(tasks)):
        if i == len(names):
            break
        dependency = dependencies[i] if i < len(dependencies) else "None"
        agent = next((a for a in agents if a.name == names[i]), agents[0])
        # #S0 reads the last step and a later step raises, as the source's planned_tasks[i - 1] does
        context = [range(len(steps))[int(n) - 1] for n in re.findall(r"#S(\d+)", dependency)] if dependency != "None" else []
        steps.append(Step(tasks[i], outputs[i] if i < len(outputs) else "", agent, context))
    return steps


def summarize(user_input, model):
    """SummarizationAgent.execute_task: up to three calls at temperature 0, 0.1 and 0.2 until an answer comes."""
    prompt = prompts.SUMMARIZER.format(user_input=user_input)
    for attempt in range(SUMMARY_ATTEMPTS):
        result = watsonx_llm(prompt, model_id=model, temperature=attempt / 10, stop=END_OF_RESPONSE)["generated_text"]
        if NO_ANSWER not in result.lower():
            return result.strip()
    return ""


class Workflow:
    """One run: the plans and reviews, the steps and the history, filled as the run goes so that a run that stops early
    leaves them. ``reached`` counts the plan steps the run started, the step in which it ends included: its step count."""

    def __init__(self, question, agents, model, route, round_ended=None):
        self.question, self.agents, self.model, self.route = question, list(agents), model, route
        # called after each planning round, which calls no tool of the rig: the driver's no_call entry
        self.round_ended = round_ended
        self.plans: list[dict] = []
        self.plan = ""
        self.steps: list[Step] = []
        self.history: list[dict] = []
        self.reached = 0

    def make_plan(self):
        """PlanningReviewWorkflow.generate_steps of the archive: up to PLANS rounds of a plan and its review, ending on a valid one,
        else the plan of the last round. As in the archive, the plan and the review of a round become prev_plan and prev_review
        before the status is read, so a review without a status keeps that plan for the next round and as the fallback, while a
        round whose planner or reviewer raises loses its plan; either failed round counts as a retry."""
        descriptions = agent_descriptions(self.agents)
        final, prev_plan, prev_review = "", "", ""
        for _ in range(PLANS):
            try:
                with self.route.agent(PLANNER):
                    plan = watsonx_llm(planner_prompt(self.question, descriptions, prev_plan, prev_review), model_id=self.model)["generated_text"]
                with self.route.agent(REVIEWER):
                    verdict = review(self.question, descriptions, plan, self.model)
                prev_review, prev_plan = verdict, plan
                self.plans.append({"plan": plan, "review": verdict})
                if verdict["status"].lower() == "valid":
                    final = plan
            except Exception as exc:  # noqa: BLE001 - the source counts a failed round as a retry and goes on
                self.plans.append({"plan": None, "review": None, "error": f"{type(exc).__name__}: {exc}"})
            if self.round_ended is not None:
                self.round_ended()
            if final:
                break
        self.plan = final or prev_plan
        self.steps = plan_steps(self.plan, self.agents)
        return self.steps

    def _record(self, number, description, agent_name, response):
        self.history.append({"task_number": number, "task_description": description, "agent_name": agent_name, "response": response})

    def execute(self):
        """SequentialWorkflow.run with ContextType.SELECTED, then the summarization step over every step's output."""
        outputs = []
        for number, step in enumerate(self.steps, start=1):
            self.reached = number
            if step.context:
                context = "".join(outputs[index] + "\n" for index in step.context)
                user_input = f"{step.description}\n\nContext:\n{context}"
            else:
                user_input = f"{step.description}\n"
            response = step.agent.execute(user_input).split("Final Answer:")[0].strip()
            outputs.append(response)
            self._record(number, step.description, step.agent.name, response)
        if outputs:
            user_input = f"{self.question}\n\nContext:\n" + "".join(o + "\n" for o in outputs)
        else:
            user_input = f"{self.question}\n"
        with self.route.agent(SUMMARIZER):
            response = summarize(user_input, self.model).split("Final Answer:")[0].strip()
        self._record(len(self.steps) + 1, self.question, SUMMARY_AGENT, response)
        return self.history

    def answer(self):
        """The final answer as tests/evaluate.py reads it: the last response, or the one before it when that is empty."""
        if not self.history:
            return ""
        if self.history[-1]["response"].strip() == "" and len(self.history) > 1:
            return self.history[-2]["response"]
        return self.history[-1]["response"]
