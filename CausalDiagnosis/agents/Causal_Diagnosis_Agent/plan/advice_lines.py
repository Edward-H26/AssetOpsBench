from CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board import n_cycles
from . import repair_budget, read_advice


def advice(board):
    """The lines of advice the board shows, where the evidence guard and the
    rule's backtracking used to bind: while nothing is committed and the ranking says read first, the reads to take
    before a diagnosis, as the arm of Ag7 proposes them, or a declaration when none fits; before the commitment, the repair plan to
    act on, the rank the server advises or rank 1 without advice; after a refuted plan, with
    backtracking on (Ag8), the cheapest other plan; after a commitment with no candidate plan left (repair_options), the last
    plan refuted or none offered, the components that the ranking still holds as faulty outside the last refuted set, when the
    budget left pays their repair. None of it binds the model, the plan check holds none of it."""
    lines = []
    if board.ranking is None:
        return lines
    server = board.server_advice() or {}
    follow_read, follow_repair = read_advice.follows_advice(board, "read"), read_advice.follows_advice(board, "repair")
    if follow_read and server.get("action") == "read":  # under the main settings the advice of the agent is the server's
        lines.append(f"Advised reads, the read of the server: {server['sensor_group']} x{server['cycles']}. While the ranking says read "
                     "first, read them, the separating test or the confirm test of the plan you follow before a diagnosis.")
    elif not follow_read and not board.committed and board.read_first():
        source, _ = read_advice.read_candidates(board, board.read_advice)
        reads = read_advice.batch_reads(board, board.read_advice)
        if reads:
            lines.append(f"Advised reads, {read_advice.ARMS[source]}: " + ", ".join(f"{g} x{c}" for g, c in reads)
                         + ". While the ranking says read first, read them, the separating test or the confirm test of the plan you follow before a diagnosis.")
        elif board.separating_test():
            test = board.separating_test()
            lines.append(f"No separating read fits the budget after the reserve, so the advice is to declare insufficient evidence with the separating "
                         f"test, {test['sensor_group']} for {n_cycles(int(test.get('cycles') or 1))}, after a reading.")
    if follow_repair and server.get("action") == "declare":
        lines.append(f"The advice of the server is to declare insufficient evidence with the separating test, {server['sensor_group']} for "
                     f"{n_cycles(int(server.get('cycles') or 1))}, after a reading.")
    elif follow_repair and server.get("action") == "diagnose":
        lines.append(f"Advised repair plan, the advice of the server: act on rank {server['rank']}" + (", the top candidate." if server["rank"] == 1 else "."))
    elif follow_repair:
        pass  # the server advises a read first, and its repair plan follows the read
    elif not board.committed and board.candidates():
        lines.append("Advised repair plan, with no troubleshooting plan shown: act on rank 1, the top candidate.")
    elif board.committed or (board.backtracking and board.refuted_last):
        refuted = bool(board.backtracking and board.refuted_last)
        options = repair_budget.repair_options(board)
        if refuted and options:
            lines.append(f"The last repair plan was refuted, so the advice is to repair rank {options[0][1]}, the cheapest other "
                         "plan, and not the refuted set until a later ranking names it first.")
        elif not options:
            # what the latest ranking still holds as faulty is its faulty set with the most mass, which the server sums over every
            # combination of the set, so the line holds for any asset and does not move with top_k; a component whose work order is
            # open is left out, since the environment refuses a second order on it, and so is a component of the last refuted set,
            # since the board and the prompt say that the advice keeps that set out until a later ranking names it first; a repair
            # whose completion the budget left cannot pay, priced as repair_options prices a plan, is not advised, since the plan
            # check would refuse it
            left_out = {o["component"] for o in board.open_orders()} | set(board.refuted_last if refuted else ())
            still_faulty = [c for c in board.top_set()[0] if c not in left_out]
            if still_faulty and board.completion(still_faulty).price <= board.budget_left:
                lines.append(("The last repair plan was refuted and no other plan is left" if refuted else "No candidate plan is left")
                             + ", so the advice is to order the repair of the components that the ranking still holds as faulty, those "
                             "of the faulty set with the most mass that hold no open work order"
                             + (" and lie outside the refuted set" if refuted else "") + f": {', '.join(still_faulty)}.")
    if server:
        lines.append(f"Advice of the server: {server['text']}")
    return lines
