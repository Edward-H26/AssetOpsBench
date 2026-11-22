#!/usr/bin/env python3
"""Block commits that put real credentials into a committed .env template.

The repo ships .env.public as a blank template. Every key in it must stay
empty, except the handful of non-secret defaults listed in ALLOWED_DEFAULTS.
A shell template such as env.example.sh, which scripts source, gives its keys
defaults: there the secret keys must ship empty and no word may look like a
credential.
This hook reads the *staged* blob rather than the working tree, so it sees
exactly what is about to be committed.

Usage:
    check_env_templates.py [FILE ...]

With no arguments it inspects every staged file itself. pre-commit passes
the matching filenames as arguments; the .githooks fallback does not.

Exit code 0 = clean, 1 = a value was found, 2 = the hook itself failed.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------

# Filenames that are meant to live in the repo as blank templates.
TEMPLATE_NAMES = {
    ".env.public",
    ".env.example",
    ".env.sample",
    ".env.template",
    "env.example",
}

# Shell templates: sourced by scripts, they give their keys defaults, so instead
# of the key rules of check_template a secret key must ship empty and every word
# of the text, comments included, is held against SECRET_VALUE_SHAPES.
SHELL_TEMPLATE_NAMES = {name + ".sh" for name in TEMPLATE_NAMES}

# A variable name (after $, ${ or export, or before = at the start of a word)
# is no value, and a long one looks like a token, so it is matched apart from
# the words, which are runs of the characters of the credential shapes. A name
# after export or before = that is itself shaped like a credential (see
# looks_like_a_value) is still checked: a padded base64 token ends in =.
SHELL_TOKEN_RE = re.compile(
    r"(?<=\$)\{?[A-Za-z_][A-Za-z0-9_]*"
    r"|(?<=export )(?P<exported>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?<![^\s;&|({])(?P<assigned>[A-Za-z_][A-Za-z0-9_]*)(?==)"
    r"|(?P<word>[A-Za-z0-9_.-]+)"
)

# An assignment NAME=WORD, NAME+=WORD or NAME[i]=WORD wherever a name can
# start: also after a quote, inside $(...) or `...`, and inside the value of
# another assignment, which is why the value is read in a lookahead. In
# NAME=${NAME:-default} (or -, :=, =) the default is the value the template ships.
SHELL_ASSIGNMENT_RE = re.compile(
    r"(?<![^\s;&|({\"'`])(?<!\$\{)([A-Za-z_][A-Za-z0-9_]*)(?:\[[^\]]*\])?\+?=(?=([^\s;&|)`]*))"
)
SHELL_DEFAULT_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):?[-=](.*)\}")
# Wherever it stands, ${NAME:-word}, ${NAME-word}, ${NAME:=word} or ${NAME=word}
# gives NAME the value word (the = forms assign it, the - forms ship it as the
# fallback); :? and :+ give it none.
SHELL_EXPANSION_DEFAULT_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):?[-=]([^}]*)\}")

# Anything else matching this pattern is a private env file and must never
# be committed at all.
ENV_FILE_RE = re.compile(r"(^|/)\.?env(\.|$)")

# A Python module named env.py is source code, not an env file.
CODE_SUFFIXES = (".py",)

# Keys allowed to carry a value in a committed template, and the exact shape
# that value may take. Add a key here only when the value is a public,
# non-secret default such as a documented service endpoint.
ALLOWED_DEFAULTS = {
    # Public service endpoints.
    "WATSONX_URL": re.compile(r"^https://[a-z0-9.-]+\.cloud\.ibm\.com/?$"),
    "TOKENROUTER_BASE_URL": re.compile(r"^https://api\.tokenrouter\.com/v1/?$"),
    # Local-only CouchDB defaults. Loopback hosts only, never a remote host.
    "COUCHDB_URL": re.compile(r"^https?://(localhost|127\.0\.0\.1)(:[0-9]{2,5})?/?$"),
    "COUCHDB_USERNAME": re.compile(r"^admin$"),
    # Non-secret identifiers: database names and the runner image tag.
    "IOT_DBNAME": re.compile(r"^[a-z0-9_-]{1,40}$"),
    "WO_DBNAME": re.compile(r"^[a-z0-9_-]{1,40}$"),
    "FAILURE_CODE_DBNAME": re.compile(r"^[a-z0-9_-]{1,40}$"),
    "STIRRUP_CODE_IMAGE": re.compile(r"^[a-z0-9._/-]{1,60}(:[a-zA-Z0-9._-]{1,40})?$"),
}

# Key names that must ALWAYS be empty, even if someone adds them to
# ALLOWED_DEFAULTS by mistake. Matching is on the final underscore-separated
# component so that TOKENROUTER_BASE_URL is not mistaken for a token.
SECRET_TERMINALS = {
    "KEY", "APIKEY", "KEYS", "SECRET", "SECRETS", "TOKEN", "PASSWORD",
    "PASSWD", "PASS", "CREDENTIAL", "CREDENTIALS", "PAT", "AUTH",
}

# A trailing _ID is a secret only after one of these.
SECRET_ID_PREFIXES = {
    "PROJECT", "SPACE", "ACCOUNT", "CLIENT", "TENANT", "ORG", "DEPLOYMENT",
    "SUBSCRIPTION", "INSTANCE",
}

LONG_OPAQUE = "a long opaque token"

# Value shapes that look like a credential wherever they appear.
SECRET_VALUE_SHAPES = [
    (re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
                r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"), "a UUID (project / space id)"),
    (re.compile(r"^sk-[A-Za-z0-9_\-]{16,}$"), "an OpenAI-style secret key"),
    (re.compile(r"^(ghp|gho|ghs|ghu|github_pat)_[A-Za-z0-9_]{16,}$"), "a GitHub token"),
    (re.compile(r"^xox[abprs]-[A-Za-z0-9-]{10,}$"), "a Slack token"),
    (re.compile(r"^AKIA[0-9A-Z]{16}$"), "an AWS access key id"),
    (re.compile(r"^ey[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\."), "a JWT"),
    (re.compile(r"^[A-Za-z0-9_\-]{24,}$"), LONG_OPAQUE),
]

MAX_BYTES = 512 * 1024


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def git(*args: str) -> str:
    result = subprocess.run(
        ("git",) + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or " ".join(args))
    return result.stdout


def staged_paths() -> list[str]:
    out = git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    return [p for p in out.split("\0") if p]


def staged_blob(path: str) -> str | None:
    """Return the staged content of path, or None if it is not staged."""
    try:
        raw = subprocess.run(
            ["git", "show", f":{path}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        ).stdout
    except subprocess.CalledProcessError:
        return None
    if len(raw) > MAX_BYTES:
        return None
    return raw.decode("utf-8", errors="replace")


def mask(value: str) -> str:
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}{'*' * 12}{value[-4:]}"


def strip_inline_comment(value: str) -> str:
    """Drop a trailing ' # comment' but keep '#' inside a quoted value."""
    if value[:1] in {'"', "'"}:
        quote = value[0]
        end = value.find(quote, 1)
        if end != -1:
            return value[1:end]
    return re.split(r"\s+#", value, maxsplit=1)[0].strip()


def is_secret_key(key: str) -> bool:
    parts = key.upper().split("_")
    last = parts[-1]
    if last in SECRET_TERMINALS:
        return True
    if last == "ID" and len(parts) >= 2 and parts[-2] in SECRET_ID_PREFIXES:
        return True
    return False


def classify(value: str) -> str | None:
    for pattern, label in SECRET_VALUE_SHAPES:
        if pattern.match(value):
            return label
    return None


def looks_like_a_value(name: str) -> bool:
    """A name in the place of a shell variable is a value when it has a
    credential shape of its own, or is a long opaque run that mixes upper and
    lower case, as a base64 token does and a shell variable name does not."""
    shape = classify(name)
    if shape is None:
        return False
    return shape != LONG_OPAQUE or (name.lower() != name and name.upper() != name)


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def check_template(path: str, content: str) -> list[str]:
    problems: list[str] = []

    for lineno, raw in enumerate(content.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue

        key, _, rest = line.partition("=")
        key = key.strip()
        value = strip_inline_comment(rest.strip()).strip().strip("\"'")

        if not value:
            continue

        shape = classify(value)

        if is_secret_key(key):
            problems.append(
                f"  {path}:{lineno}  {key} has a value ({mask(value)}). "
                f"Secret keys must ship empty."
            )
            continue

        allowed = ALLOWED_DEFAULTS.get(key)
        if allowed and allowed.match(value):
            if shape:
                problems.append(
                    f"  {path}:{lineno}  {key} is an allowed default but its value "
                    f"looks like {shape}: {mask(value)}"
                )
            continue

        if allowed:
            problems.append(
                f"  {path}:{lineno}  {key} carries a value that does not match its "
                f"approved default shape: {mask(value)}"
            )
            continue

        detail = f" and looks like {shape}" if shape else ""
        problems.append(
            f"  {path}:{lineno}  {key} has a value ({mask(value)}){detail}. "
            f"Leave it empty, or add it to ALLOWED_DEFAULTS if it is a public default."
        )

    return problems


def check_shell_template(path: str, content: str) -> list[str]:
    problems: list[str] = []

    for lineno, raw in enumerate(content.splitlines(), start=1):
        # one report per key and line, so NAME=${NAME:-x} is not counted twice
        values: dict[str, str] = {}
        for key, word in SHELL_ASSIGNMENT_RE.findall(raw):
            word = word.strip("\"'")
            default = SHELL_DEFAULT_RE.fullmatch(word)
            value = default.group(2) if default and default.group(1) == key else word
            if value.strip("\"'"):
                values.setdefault(key, value)
        for key, word in SHELL_EXPANSION_DEFAULT_RE.findall(raw):
            if word.strip("\"'"):
                values.setdefault(key, word)
        for key, value in values.items():
            if is_secret_key(key):
                problems.append(
                    f"  {path}:{lineno}  {key} has a value ({mask(value)}). "
                    f"Secret keys must ship empty."
                )

        for match in SHELL_TOKEN_RE.finditer(raw):
            name = match.group("exported") or match.group("assigned") or ""
            # "-" and "." at the ends belong to the shell (":-", "--flag") or
            # the prose of a comment, not to the value.
            word = (match.group("word") or (name if looks_like_a_value(name) else "")).strip("-.")
            shape = classify(word)
            if shape:
                problems.append(
                    f"  {path}:{lineno}  a value looks like {shape}: {mask(word)}"
                )

    return problems


def check_private_env(path: str) -> list[str]:
    return [
        f"  {path}  is a private env file. Add it to .gitignore and keep secrets "
        f"out of the repo."
    ]


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    try:
        candidates = argv[1:] or staged_paths()
    except RuntimeError as exc:
        print(f"check_env_templates: git failed: {exc}", file=sys.stderr)
        return 2

    problems: list[str] = []

    for path in candidates:
        base = os.path.basename(path)
        if not ENV_FILE_RE.search("/" + path) or path.endswith(CODE_SUFFIXES):
            continue

        content = staged_blob(path)
        if content is None:
            continue

        if base in TEMPLATE_NAMES:
            problems.extend(check_template(path, content))
        elif base in SHELL_TEMPLATE_NAMES:
            problems.extend(check_shell_template(path, content))
        else:
            problems.extend(check_private_env(path))

    if not problems:
        return 0

    print("", file=sys.stderr)
    print("BLOCKED: credentials found in a committed env file", file=sys.stderr)
    print("", file=sys.stderr)
    for problem in problems:
        print(problem, file=sys.stderr)
    print("", file=sys.stderr)
    print("Fix it:", file=sys.stderr)
    print("  1. Rotate any key that was pasted in. Assume it is burned.", file=sys.stderr)
    print("  2. Blank the values:  git restore --staged .env.public && "
          "git checkout -- .env.public", file=sys.stderr)
    print("  3. Put your real values in .env (gitignored), not in the template.",
          file=sys.stderr)
    print("", file=sys.stderr)
    print("If a value is genuinely public, add its key to ALLOWED_DEFAULTS in "
          "scripts/check_env_templates.py.", file=sys.stderr)
    print("", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))