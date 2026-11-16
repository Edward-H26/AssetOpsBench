import contextlib
import random
import shutil
from datetime import datetime
from pathlib import Path
from CausalDiagnosis.experiments.run_verification.public_record import public_value  # noqa: E402
from CausalDiagnosis.experiments.episode_runners import attempt_error_classes as failures  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import COLUMNS, HIDDEN
from CausalDiagnosis.utils.json_files import read_json, write_json
from CausalDiagnosis.utils.csv_rows import write_rows  # noqa: E402
from CausalDiagnosis.training.gepa_prompt_optimization.feedback_card import episode_card, feedback_string  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.training_variants import AS_RELEASED, REGISTERED, VISITS, visits_test_combination  # noqa: E402
from . import run_folders, candidate_proposals, stream_scores


FORMAT = 2  # one component, the tunable block
FEEDBACK = ("verifier", "score_only")
TEMPERATURE = 0.0
INTERRUPTED = "interrupted"  # the folder of a run folder that holds the episodes of the rounds a resume cut short


class LoopFailed(Exception):
    """The reflection endpoint did not answer the registered number of proposals in a row."""


class OnlineLoop:
    """The online loop over a player, play(candidate, seed, call) returning the row and the folder of one scored episode, and a
    reflection model, an LLMBackend. identity is what a resume compares."""

    def __init__(self, play, reflection, run_dir, feedback, stream, block=3, window=2, children=1, max_episodes=240,
                 gepa_seed=0, reflection_model=None, reflection_endpoint=None, *, horizon, unanswered_proposals, proposal_attempts,
                 asset_id, training_fold, split=REGISTERED, agent="causal", training_variant=AS_RELEASED, identity=None):
        if feedback not in FEEDBACK:
            raise ValueError(f"the feedback card is one of {FEEDBACK}, not {feedback!r}")
        self.play, self.reflection, self.run_dir, self.feedback = play, reflection, Path(run_dir), feedback
        self.horizon = horizon  # of the run's configuration, which the reward of a stream episode reads
        self.unanswered_limit, self.attempts = unanswered_proposals, proposal_attempts  # both of the asset of the run's configuration
        self.stream = list(stream)
        self.block, self.window, self.children, self.max_episodes = block, window, children, max_episodes
        self.gepa_seed, self.reflection_model, self.reflection_endpoint = gepa_seed, reflection_model, reflection_endpoint
        # the agent and the weights of the loop, for the record, and its asset, fold and split, which say what episode is left out
        self.agent, self.training_variant, self.asset_id, self.training_fold = agent, training_variant, asset_id, training_fold
        self.split = split
        self.identity = identity
        self.rng = random.Random(gepa_seed)
        self.candidates, self.episodes, self.rounds, self.rows, self.files = {}, [], [], [], {}
        self.calls = self.played = self.reflection_calls = 0
        self.start_id, self.front, self.incumbent, self.first_chosen = None, [], None, {}
        # what a resume needs beyond the record: the children that wait for the next round, whether the proposals after the
        # last round are still due, with the strata each member of the front led in it, and the rounds that were cut short
        self.pending, self.proposals_due, self.leads, self.interrupted = [], False, {}, []
        self.proposals = self.answered = self.unanswered_in_a_row = self.without_example = 0

    # ----- the loop -----
    def run(self):
        """Plays the loop to the end of its budget, from its start or from the state a resume restored. A reflection endpoint
        that does not answer ends it with the status failed."""
        if self.start_id is None:
            start = self._candidate(None, None, 0)
            start.update(texts=candidate_proposals.seed_texts(), status="start")
            self._write_texts(start)
            self.start_id, self.front, self.incumbent, self.first_chosen = start["id"], [start["id"]], start["id"], {start["id"]: 0}
        try:
            while self.proposals_due or self._affordable(len(self.front) + len(self.pending)):
                if not self.proposals_due:
                    self.leads = self._round(len(self.rounds) + 1, [self.candidates[cid] for cid in self.pending])
                    self.pending, self.proposals_due = [], True
                    self.save("running")
                if self._affordable(len(self.front) + self.children):
                    proposals = [self._propose(self._parent(self.leads)) for _ in range(self.children)]
                    self.pending = [child["id"] for child in proposals if child is not None]
                self.proposals_due = False
                self.save("running")
        except LoopFailed:
            self.save("failed")
            raise
        self.save("finished")
        return self.record("finished")

    def _affordable(self, members):
        """Whether the stream holds the seeds of the next round and the budget its episodes."""
        return (len(self.rounds) + 1) * self.block <= len(self.stream) and self.played + members * self.block <= self.max_episodes

    def _round(self, round_no, pending):
        """Plays the block of a round with the front and the pending children, then records the point of the curve, admits
        the children that improve on their parents, prunes the front and chooses the incumbent. Returns the strata that each
        member of the front leads."""
        seeds = self.stream[(round_no - 1) * self.block: round_no * self.block]
        members = [*self.front, *(child["id"] for child in pending)]
        rollouts = self.played
        for seed in seeds:
            for cid in members:
                self._play(cid, seed, round_no)
        means = {cid: stream_scores.mean(self._scores(cid, round_no)) for cid in members}
        played = [e for e in self.episodes if e["round"] == round_no]
        point = {"round": round_no, "rollouts": rollouts, "incumbent": self.incumbent, "mean_score": means[self.incumbent],
                 "n": len(stream_scores.scored(self._scores(self.incumbent, round_no))),
                 "unscored": sum(1 for e in played if e["score"] is None and not e["left_out"]),
                 "left_out": sum(1 for e in played if e["left_out"])}
        admitted = [child["id"] for child in pending if stream_scores.improves(means[child["id"]], means[child["parent"]])]
        for child in pending:
            child.update(status="admitted" if child["id"] in admitted else "rejected", played_round=round_no)
        self.front, leads = self._prune([*self.front, *admitted], round_no)
        self.incumbent = self._choose(round_no)
        self.first_chosen.setdefault(self.incumbent, self.played)
        self.rounds.append({**point, "seeds": seeds, "members": members, "means": means, "admitted": admitted,
                            "rejected": [child["id"] for child in pending if child["id"] not in admitted], "front": self.front,
                            "chosen": self.incumbent})
        return leads

    def _play(self, cid, seed, round_no):
        self.calls += 1
        row, ep_dir = self.play(self.candidates[cid], seed, self.calls)
        self.rows.append(row)
        # the rows so far, so that a run cut short by the end of an allocation keeps every finished episode
        write_rows(self.run_dir / "scores.csv", self.rows, COLUMNS)
        self.played += 1
        self.files[self.calls] = Path(ep_dir)
        # an episode that met a single fault of the fold the loop's model holds out scores nothing and no reflection call reads it
        left_out = visits_test_combination(read_json(Path(ep_dir) / HIDDEN / "log.json", []) or [], self.training_fold, self.asset_id,
                                           self.split)
        self.episodes.append({"call": self.calls, "episode": Path(ep_dir).name, "phase": "stream", "round": round_no, "candidate": cid,
                              "seed": seed, "n_faulty": row.get("n_faulty"), "reward": row.get("reward"), "left_out": left_out,
                              "score": None if left_out else stream_scores.score_of(row, ep_dir, self.horizon), "strata": stream_scores.strata_of(row, ep_dir)})

    def _scores(self, cid, round_no):
        return [e["score"] for e in self.episodes if e["candidate"] == cid and e["round"] == round_no]

    def _window_scores(self, cid, round_no):
        """The mean score per stratum of a candidate over its stream episodes of the last window rounds."""
        by_stratum = {}
        for e in self.episodes:
            if e["candidate"] == cid and e["round"] > round_no - self.window and e["score"] is not None:
                for stratum in e["strata"]:
                    by_stratum.setdefault(stratum, []).append(e["score"])
        return {stratum: stream_scores.mean(scores) for stratum, scores in by_stratum.items()}

    def _prune(self, members, round_no):
        """GEPA's Pareto selection over strata, which keeps the members that lead some stratum less those that another
        strictly dominates. Returns the front in candidate order and the strata each member leads."""
        scores = {cid: self._window_scores(cid, round_no) for cid in members}
        strata = sorted({s for by in scores.values() for s in by})
        if not strata:  # no member has a scored episode in the window, so nothing ranks them
            return sorted(members), {cid: 1 for cid in members}
        leaders = {s: {cid for cid in members if scores[cid].get(s) == max(by[s] for by in scores.values() if s in by)} for s in strata}
        kept = [cid for cid in members if any(cid in leaders[s] for s in strata)]

        def dominates(other, cid):
            mine, theirs = scores[cid], scores[other]
            return (set(mine) <= set(theirs) and all(theirs[s] >= mine[s] for s in mine) and any(theirs[s] > mine[s] for s in mine))

        front = sorted(cid for cid in kept if not any(dominates(other, cid) for other in kept if other != cid))
        return front, {cid: sum(1 for s in strata if cid in leaders[s]) for cid in front}

    def _choose(self, round_no):
        """The member of the front with the best mean over the window, the oldest of a tie, and the incumbent while no
        member of the front has a scored episode in the window."""
        means = {cid: self._window_scores(cid, round_no).get(stream_scores.WHOLE_BLOCK) for cid in self.front}
        means = {cid: m for cid, m in means.items() if m is not None}
        if not means:
            return self.incumbent if self.incumbent in self.front else self.front[0]
        best = max(means.values())
        return min(cid for cid in means if means[cid] == best)

    def _parent(self, leads):
        """A parent drawn from the front with weights equal to the strata it leads, as GEPA samples its candidates."""
        front = sorted(leads)
        return self.rng.choices(front, weights=[leads[cid] for cid in front])[0]

    # ----- reflection -----
    def _propose(self, parent_id):
        """A child of the parent from the reflection model over the parent's episodes of the last round, or None: for a
        proposal refused at each of its registered attempts, which stays in the record with its problems, and for a parent whose
        episodes of the round were all left out, which asks nothing. A proposal whose every attempt ended on an error of the
        endpoint is unanswered, and the registered number of them in a row fails the loop."""
        parent = self.candidates[parent_id]
        calls = [e["call"] for e in self.episodes if e["candidate"] == parent_id and e["round"] == len(self.rounds) and not e["left_out"]]
        if not calls:
            self.without_example += 1
            return None
        component = candidate_proposals.MUTABLE[parent["proposals"] % len(candidate_proposals.MUTABLE)]
        parent["proposals"] += 1
        child = self._candidate(parent_id, component, len(self.rounds))
        prompt = self._meta_prompt(parent, component, calls)
        self.proposals += 1
        problems, errors = [], 0
        for attempt in range(1, self.attempts + 1):
            asked = prompt if attempt == 1 else f"{prompt}\n\n{candidate_proposals.RETRY_HEADING}\n" + "\n".join(f"- {p}" for p in problems)
            result, error = self._ask(asked)
            errors += error is not None
            texts, problems = candidate_proposals.proposal(parent, component, result.text) if error is None else (None, [error])
            if texts is None and error is None and getattr(result, "finish_reason", None) == "length":
                problems = [candidate_proposals.CUT_REPLY]  # the reply reached the model's context, so the retry asks for a shorter text
            self._log(child, attempt, asked, result, error, problems)
            if not problems:
                child.update(texts=texts, status="proposed")
                self._write_texts(child)
                break
        else:
            child.update(status="refused", problems=problems)
        if errors == self.attempts:
            self.unanswered_in_a_row += 1
            if self.unanswered_in_a_row >= self.unanswered_limit:
                raise LoopFailed(f"the reflection model at {self.reflection_endpoint} answered none of the last "
                                 f"{self.unanswered_in_a_row} proposals: {problems[-1]}")
        else:
            self.answered += 1
            self.unanswered_in_a_row = 0
        return child if child["status"] == "proposed" else None

    def _meta_prompt(self, parent, component, calls):
        examples = "\n\n".join(self._example(n, call) for n, call in enumerate(calls, 1))
        return candidate_proposals.PROMPT_META.format(current=parent["texts"][component], examples=examples)

    def _example(self, n, call):
        """One example of the reflective dataset in GEPA's layout, with the question, the agent's outputs and the feedback,
        which is the verifier's card or the score alone, the score being the one the selection used for the episode."""
        ep_dir = self.files[call]
        score = next(e["score"] for e in self.episodes if e["call"] == call)
        feedback = feedback_string(episode_card(ep_dir, score)) if self.feedback == "verifier" else f"reward {score}"
        return (f"# Example {n}\n## Inputs\n### question\n{candidate_proposals._text(ep_dir / 'question.txt')}\n"
                f"## Generated Outputs\n{candidate_proposals.agent_outputs(ep_dir)}\n## Feedback\n{feedback}")

    def _ask(self, prompt):
        """(result, None) for a reply of the reflection model, whatever it holds, and (None, the error) when the endpoint raised,
        which is no answer, under the endpoint's health probe (failures.endpoint_probe), whose ServiceDown ends the program."""
        self.reflection_calls += 1
        try:
            with failures.endpoint_probe(self.reflection_endpoint) if self.reflection_endpoint else contextlib.nullcontext() as probe:
                return self.reflection.generate_with_usage(prompt, TEMPERATURE), None
        except Exception as exc:  # noqa: BLE001  whatever the backend raises is an error of the endpoint, counted by _propose
            if probe is not None and probe.down:  # the endpoint no longer answers its probe: the ServiceDown ends the program
                raise
            return None, f"the reflection model could not answer ({type(exc).__name__}: {exc})"

    def _log(self, child, attempt, prompt, result, error, problems):
        write_json(self.run_dir / "reflection" / f"{child['id']}_attempt{attempt}.json", {
            "child": child["id"], "parent": child["parent"], "component": child["component"], "after_round": child["proposed_round"],
            "attempt": attempt, "model": self.reflection_model, "temperature": TEMPERATURE, "prompt": prompt,
            "reply": result.text if result else None, "finish_reason": getattr(result, "finish_reason", None),
            "input_tokens": result.input_tokens if result else None, "output_tokens": result.output_tokens if result else None,
            "error": error, "problems": problems})

    # ----- candidates and the record -----
    def _candidate(self, parent, component, round_no):
        cid = f"c{len(self.candidates):03d}"
        self.candidates[cid] = {"id": cid, "parent": parent, "component": component, "proposed_round": round_no, "status": None,
                                "played_round": None, "problems": [], "texts": None, "proposals": 0}
        return self.candidates[cid]

    def _write_texts(self, candidate):
        folder = self.run_dir / "candidates"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{candidate['id']}_{candidate_proposals.BLOCK}.txt").write_text(candidate["texts"][candidate_proposals.BLOCK], encoding="utf-8")

    def _public(self, candidate):
        record = {k: candidate[k] for k in ("id", "parent", "component", "proposed_round", "status", "played_round", "problems")}
        if candidate["texts"] is not None:
            record["files"] = {candidate_proposals.BLOCK: f"candidates/{candidate['id']}_{candidate_proposals.BLOCK}.txt"}
            record["characters"] = len(candidate["texts"][candidate_proposals.BLOCK])
        return record

    def record(self, status):
        """The result file of the run, gepa_online.json, without the state of a resume."""
        return {"format": FORMAT, "status": status, "card": self.feedback, "reflection_model": self.reflection_model,
                "reflection_endpoint": self.reflection_endpoint,
                "settings": {"agent": self.agent, "training_variant": self.training_variant, "training_fold": self.training_fold,
                             "block": self.block, "window": self.window,
                             "children": self.children, "max_episodes": self.max_episodes, "gepa_seed": self.gepa_seed, "temperature": TEMPERATURE,
                             "components": list(candidate_proposals.MUTABLE), "unanswered_proposals": self.unanswered_limit, "proposal_attempts": self.attempts},
                "identity": self.identity,
                "start_candidate": self.start_id, "final_candidate": self.incumbent, "episodes_played": self.played,
                "rollouts_to_best": self.first_chosen.get(self.incumbent), VISITS: sum(1 for e in self.episodes if e["left_out"]),
                "reflection_calls": self.reflection_calls, "proposals": self.proposals, "proposals_answered": self.answered,
                "proposals_without_an_example": self.without_example, "stream_seeds": self.stream[:len(self.rounds) * self.block],
                "rounds": self.rounds, "interrupted": self.interrupted, "candidates": [self._public(c) for c in self.candidates.values()],
                "episodes": self.episodes}

    def state(self):
        """What a resume needs beyond the record, as JSON: every candidate with its texts, the rows of scores.csv, the counters,
        the children that wait, whether the proposals after the last round are due, and the state of the generator of the parent
        draws."""
        version, internal, gauss = self.rng.getstate()
        return {"candidates": list(self.candidates.values()), "rows": self.rows, "calls": self.calls, "played": self.played,
                "reflection_calls": self.reflection_calls, "front": self.front, "first_chosen": self.first_chosen,
                "pending": self.pending, "proposals_due": self.proposals_due, "leads": self.leads,
                "proposals": self.proposals, "answered": self.answered, "unanswered_in_a_row": self.unanswered_in_a_row,
                "without_example": self.without_example, "rng": [version, list(internal), gauss]}

    def restore(self, saved):
        """Takes up the loop that a result file saved: the finished rounds with their episodes, and the state. Every episode
        folder that the file does not know belongs to a round that was cut short. It moves to interrupted/ inside the run folder,
        the record counts it, and its seed plays again."""
        state = saved["state"]
        self.candidates = {c["id"]: c for c in state["candidates"]}
        self.episodes, self.rounds, self.rows = saved["episodes"], saved["rounds"], state["rows"]
        self.calls, self.played, self.reflection_calls = state["calls"], state["played"], state["reflection_calls"]
        self.files = {e["call"]: self.run_dir / e["episode"] for e in self.episodes}
        self.start_id, self.incumbent, self.front = saved["start_candidate"], saved["final_candidate"], state["front"]
        self.first_chosen, self.pending = state["first_chosen"], state["pending"]
        self.proposals_due, self.leads = state["proposals_due"], state["leads"]
        self.proposals, self.answered = state["proposals"], state["answered"]
        self.unanswered_in_a_row, self.without_example = state["unanswered_in_a_row"], state["without_example"]
        version, internal, gauss = state["rng"]
        self.rng.setstate((version, tuple(internal), gauss))
        self.interrupted = saved.get("interrupted") or []
        known = {e["episode"] for e in self.episodes}
        cut = sorted(p.name for p in self.run_dir.glob("call*_seed*") if p.is_dir() and p.name not in known)
        folder = f"{INTERRUPTED}/{len(self.interrupted) + 1}"
        for name in cut:
            (self.run_dir / folder).mkdir(parents=True, exist_ok=True)
            shutil.move(str(self.run_dir / name), str(self.run_dir / folder / name))
        self.interrupted.append({"after_round": len(self.rounds), "episodes": cut, "folder": folder if cut else None,
                                 "status": saved.get("status"), "resumed": datetime.now().isoformat(timespec="seconds")})
        write_rows(self.run_dir / "scores.csv", self.rows, COLUMNS)

    def save(self, status):
        write_json(self.run_dir / run_folders.RESULT, public_value({**self.record(status), "state": self.state()}))
