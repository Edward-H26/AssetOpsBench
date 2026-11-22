"""The schema of a reply, which gives the form of a reply to the serving engine as a JSON schema
(CausalDiagnosis/docs/general-agent-plan.md, section The plan, The schema of a reply). The reader stays the one way in for every
reply, with a schema or without one. Stubs of the freeze; the reader of a reply builds them."""


def plan_schema_of(table, forms, options):
    """The schema of a plan, generated from the table, the forms of the expectations and the options and from nothing of the facts,
    so that one table has one schema."""
    raise NotImplementedError


def differential_schema():
    """The schema of the reply of the orientation turn."""
    raise NotImplementedError


def response_format(name, schema):
    """The request field of a schema: {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": true}}."""
    raise NotImplementedError


def digest(schema):
    """The first 16 hexadecimal digits of the SHA-256 of the canonical JSON of a schema."""
    raise NotImplementedError
