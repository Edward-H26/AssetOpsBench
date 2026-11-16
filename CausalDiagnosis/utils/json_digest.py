"""The canonical digest of a JSON value, shared by the byte identity checks of the asset layer
(CausalDiagnosis/experiments/run_verification/invariant_digests.py and the tests of CausalDiagnosis/datasets): the SHA-256 of compact ASCII JSON whose keys
keep the order the code wrote them, so that a reordered key counts as a change."""
import hashlib
import json


def canonical(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def digest(value):
    return hashlib.sha256(canonical(value).encode("ascii")).hexdigest()
