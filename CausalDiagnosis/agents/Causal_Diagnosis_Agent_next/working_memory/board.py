"""The board of the general agent (CausalDiagnosis/docs/general-agent-plan.md, section The loop): what holds for the episode, the
observations with their identifiers, the candidates, the commitment, the sections of the kind and of the modules, and the status,
with the compaction and the recall of the present board under their present names. Stubs of the freeze; the working memory
builds them."""


class Board:
    """The board of one episode, over its table, the ledger of the episode, the task message question and the Options."""

    # the schemas of the replies: {"plan": digest, "differential": digest, "bound": the count of the requests that carried a
    # schema, "carried": whether the backend could carry the field}, written by to_json; None before the first request
    reply_schema: dict | None = None

    def __init__(self, table, ledger, question, options):
        raise NotImplementedError

    @property
    def ledger(self):
        """The ledger of the episode."""
        raise NotImplementedError

    @property
    def facts(self):
        """The public facts, a dict, or None until the facts tool is answered."""
        raise NotImplementedError

    @property
    def knowledge(self):
        """What the knowledge tools returned: tool name -> its calls in their order, each {"args", "result"}."""
        raise NotImplementedError

    @property
    def observations(self):
        """The observation rows of the episode, each with its identifier, as the forms of the answers gave them."""
        raise NotImplementedError

    @property
    def commitment(self):
        """The commit write made, {"tool", "args", "answer"}, or None."""
        raise NotImplementedError

    @property
    def status(self):
        """The status of the ledger, Ledger.status(). The rendered status adds the answers that the board holds for the whole
        episode and the calls made more than once with the same arguments (holds_answer, repeated)."""
        raise NotImplementedError

    def record_call(self, step, call, answer, observations):
        """A call answered: step the Step that named it, call (server, tool, args) of the call made, answer in the shape of
        agents/utils/answer.py, observations its rows."""
        raise NotImplementedError

    def holds_answer(self, tool, args):
        """Whether an answered call of the tool with these arguments stands among the calls that record_call recorded; a refused
        call is none."""
        raise NotImplementedError

    def repeated(self):
        """(tool, args, count) of every call of a step that was made more than once with the same arguments."""
        raise NotImplementedError

    def add_note(self, text, tie=None, turn=None):
        """A note on the board, tied to the state it speaks of, or to none; turn is the turn of a note of a plan, under which an
        earlier note equal to it but for the number of its turn gives way to it, and the later note states their count."""
        raise NotImplementedError

    def retire_notes(self, passed):
        """Takes off the board the notes whose tie has passed; the count of the notes that left."""
        raise NotImplementedError

    def candidates(self):
        """The candidates shown, those of the modules first, then the model's."""
        raise NotImplementedError

    def known_ids(self):
        """The identifiers that a step may cite."""
        raise NotImplementedError

    def render(self, modules):
        """The board as the model reads it, with the sections of the modules."""
        raise NotImplementedError

    def to_json(self, modules):
        """The board as the driver records it beside the plans, with the sections of the ledger and of the modules under their
        names."""
        raise NotImplementedError

    def compact(self, fits=None, after_call=None, summarize=None, fits_context=None):
        """The compaction of the present board: what changed, a list."""
        raise NotImplementedError

    def mark_turn(self, turn):
        raise NotImplementedError

    def replaceable(self):
        """The entries that a summary of earlier turns replaces, oldest first."""
        raise NotImplementedError

    def without(self, entries):
        """A copy of the board without these entries; the board itself is unchanged."""
        raise NotImplementedError

    def apply_summary(self, text, entries, left_out):
        raise NotImplementedError

    def recall(self, series, first, last):
        """A recall step: the collapsed rows of the series from observation first to last come back on the next board, once;
        the count of the rows."""
        raise NotImplementedError

    def recallable(self, series, first, last):
        raise NotImplementedError

    def drop_unfit_recall(self, fits_context):
        raise NotImplementedError
