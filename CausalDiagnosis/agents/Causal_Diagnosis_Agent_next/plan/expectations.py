"""Facts-free expectation readers shared by parsing, checking and schemas."""

from dataclasses import dataclass
from functools import partial

from . import readers


@dataclass(frozen=True)
class Form:
    reader: object
    schema: dict

    def __call__(self, value):
        return self.reader(value)


def text_list(value, names=None):
    values, reason = readers.read_list(value, "expectation")
    if reason:
        return None, reason
    result = []
    for item in values:
        text, reason = readers.read_text(item, "expectation") if names is None else readers.read_name(item, names, "expectation")
        if reason:
            return None, reason
        result.append(text)
    return result, None


def expectation_forms(table, modules):
    forms = {}
    for row in table.tools.values():
        own = {}
        if row.answer == "records":
            own["outside"] = Form(text_list, {"type": "array", "items": {"type": "string", "minLength": 1}})
        if row.answer == "text" and row.words:
            own["word"] = Form(partial(readers.read_name, names=row.words, name="expect.word"), {"type": "string", "enum": list(row.words)})
        if table.declaration.events:
            own["no_event"] = Form(partial(text_list, names=table.declaration.events),
                                    {"type": "array", "items": {"type": "string", "enum": list(table.declaration.events)}})
        for module in modules:
            for name, form in module.forms(row).items():
                if name in own:
                    raise ValueError(f"{row.name}: duplicate expectation {name}")
                own[name] = form
        forms[row.name] = own
    return forms
