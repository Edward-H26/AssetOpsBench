"""The recordings of an asset of the recorded condition levels kind (CausalDiagnosis/docs/adding-a-dataset.md, section The data file).

A data file holds one JSON record per recorded cycle with cycle_index, labels and features. The labels hold one column
per component, named by records.label_keys of the profile, and the stable flag. The features are named
<Quantity>_<Channel>_<statistic> with every statistic of statistics.recorded for every channel. The stable cycles of one
combination form one contiguous block, which the environment serves through the segments of an entry offset.

The data file of an asset is its release file, built by the command that records.fetch of its asset file names from the
source that records.page names, and the variable of records.data_env, when set and not empty, names another file for
development. The release file is read only when its SHA-256 equals its line of CHECKSUMS.txt in the release folder. A
missing data file, a release file with another hash and a release file without a line are refused by DataFileError, whose
text names the page and the command of the asset, and no reader falls back to another file. Every file built from the
recordings names the SHA-256 of the data file it was built from under the key facts_sha256, and every reader refuses one
built from another data file, with the command that builds it again.
"""
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]  # CausalDiagnosis/datasets -> the fork root, which the release folders of the profiles start from
CHECKSUMS_NAME = "CHECKSUMS.txt"  # the pinned checksums of a release folder, the one file of that folder in the repository
WITHOUT_DATA = "ASSETS_WITHOUT_DATA"  # set to 1 by a clone that holds no data, so that the tests that need recordings skip


class DataFileError(RuntimeError):
    """A data file, or a file built from one, that the code refuses to read. missing is True when the data file does not exist."""

    def __init__(self, message, missing=False):
        super().__init__(message)
        self.missing = missing


class Recordings(list):
    """The cycles of a data file in recording order, with facts_sha256, the SHA-256 of the file they were read from."""

    def __init__(self, cycles=(), facts_sha256=None):
        super().__init__(cycles)
        self.facts_sha256 = facts_sha256


@dataclass(frozen=True)
class Cycle:
    index: int
    combo: tuple  # one level per component, in the component order of the profile
    stable: bool
    features: dict  # feature name -> value


def combo_of(labels, profile):
    return tuple(labels[profile.records["label_keys"][c]] for c in profile.components)


def release_path(profile):
    """The data file that the record builder of the asset writes into its release folder."""
    spec = profile.records
    return REPO / spec["release_dir"] / spec["release_file"]


def default_facts_path(profile):
    """The data file of the asset: the file that the environment variable of its profile names when the variable is set and
    not empty, else its release file."""
    return Path(os.environ.get(profile.records["data_env"]) or release_path(profile))


def public_path(path):
    """A path as a public record of a run holds it (CausalDiagnosis/experiments/run_verification/public_record.py): a relative path as it is, an
    absolute path inside the fork relative to the fork root in posix form, any other absolute path as its basename, so that no
    public file names a home, a scratch folder or a node of the machine that wrote it."""
    if path is None:
        return None
    text = str(path)
    if not text.startswith(("/", "~")):
        return text
    absolute = Path(text).expanduser().resolve()
    try:
        return absolute.relative_to(REPO.resolve()).as_posix() or "."
    except ValueError:
        return absolute.name


def resolve_recorded_path(recorded, profile=None):
    """The file that a path of a public record names on this machine: an absolute path as it is, a path relative to the fork
    root under REPO, and a basename that names the data file of the asset (default_facts_path, which reads the variable of the
    profile) as that file; None when none of them is a file."""
    if recorded is None:
        return None
    path = Path(str(recorded)).expanduser()
    if path.is_absolute():
        return path
    if (REPO / path).is_file():
        return REPO / path
    if profile is not None and len(path.parts) == 1 and default_facts_path(profile).name == path.name and default_facts_path(profile).is_file():
        return default_facts_path(profile)
    return None


def read_checksums(path):
    """{name: SHA-256} of a file of sha256sum lines: a hash, two spaces and a name relative to the folder of the file."""
    pins = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or not separator or not name:
            raise DataFileError(f"{path}: {line!r} is no sha256sum line")
        pins[name] = digest
    return pins


def checksums(profile):
    """The pinned checksums of the asset, empty when its release folder holds no CHECKSUMS.txt."""
    path = release_path(profile).parent / CHECKSUMS_NAME
    return read_checksums(path) if path.exists() else {}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _how_to_get(profile):
    """The sentences that name the official page and the one command of an asset that states them."""
    spec = profile.records
    if "fetch" not in spec:
        return ""
    return (f" The official page of the data is {spec['page']}. Run `{spec['fetch']}` from the root of the repository: it fetches "
            "the source, checks its SHA-256 and builds the file.")


def _require_file(path, profile):
    if Path(path).exists():
        return
    variable = profile.records["data_env"]
    if os.environ.get(variable) and Path(os.environ[variable]) == Path(path):
        raise DataFileError(f"{variable} names {path}, which does not exist. Unset it to read the data file of {profile.asset_id}, or "
                            "name a file that exists.", missing=True)
    raise DataFileError(f"The data file of {profile.asset_id} is not built: {path} does not exist." + _how_to_get(profile), missing=True)


def _require_pin(path, digest, profile):
    """Refuse the release file of the asset when its SHA-256 is not the pinned one. A file at another path has no pin."""
    release = release_path(profile)
    if Path(path).resolve() != release.resolve():
        return
    where = release.parent / CHECKSUMS_NAME
    pinned = checksums(profile).get(release.name)
    if pinned is None:
        raise DataFileError(f"{where} pins no checksum for {release.name}, so {release} cannot be read as the data file of {profile.asset_id}.")
    if digest != pinned:
        raise DataFileError(f"{release} has the SHA-256 {digest}, and {where} pins {pinned}, so it is not the data file of "
                            f"{profile.asset_id}." + _how_to_get(profile))


def check_data(profile):
    """The SHA-256 of the data file of the asset, after the checks that load_cycles makes: the file exists, and the release
    file equals its pin."""
    path = default_facts_path(profile)
    _require_file(path, profile)
    digest = file_sha256(path)
    _require_pin(path, digest, profile)
    return digest


def load_cycles(path, profile):
    """Every record of the data file as a Cycle, in recording order, with the SHA-256 of the file. A missing file is refused,
    and so is the release file of the asset when its SHA-256 differs from its pin."""
    _require_file(path, profile)
    stable_flag = profile.records["stable_flag_stable"]
    cycles, digest = [], hashlib.sha256()
    with open(path, "rb") as f:
        for line in f:
            digest.update(line)
            rec = json.loads(line)
            feats = {x["name"]: x["value"] for x in rec["features"]}
            cycles.append(Cycle(rec["cycle_index"], combo_of(rec["labels"], profile), rec["labels"]["stable_flag"] == stable_flag, feats))
    _require_pin(path, digest.hexdigest(), profile)
    _require_features(path, cycles, profile)
    cycles.sort(key=lambda c: c.index)
    return Recordings(cycles, digest.hexdigest())


def _require_features(path, cycles, profile):
    """Refuse the first record that lacks a summary the asset file declares, a recorded statistic of a channel under the
    declared quantity of the channel, since a channel whose quantity the asset file misspells would else be served without
    it and nothing would say so."""
    wanted = [(channel, statistic, feature_name(channel, statistic, profile)) for channel in profile.channels
              for statistic in profile.statistics["recorded"]]
    for cycle in cycles:
        for channel, statistic, name in wanted:
            if name not in cycle.features:
                raise DataFileError(f"{path}: the record of cycle {cycle.index} holds no feature {name}, the {statistic} of the channel "
                                    f"{channel} that the asset file of {profile.asset_id} declares")


def build_blocks(cycles):
    """{combo: [cycle indices]} over the stable cycles only, with each block checked to be contiguous."""
    blocks = {}
    for c in cycles:
        if c.stable:
            blocks.setdefault(c.combo, []).append(c.index)
    for combo, idx in blocks.items():
        if idx[-1] - idx[0] + 1 != len(idx):
            raise ValueError(f"stable cycles of {combo} are not contiguous")
    return blocks


def load_recordings(path, profile):
    """The loader of the recorded levels kind (CausalDiagnosis.datasets.dataset_loader.evidence_kinds): the records of the data file and their stable blocks."""
    cycles = load_cycles(path, profile)
    return cycles, build_blocks(cycles)


def record_problems(data):
    """The check of the recorded levels kind on the records table of a raw asset file: a distinct label column for every
    component, and the integer value of the stable flag."""
    from CausalDiagnosis.datasets.dataset_loader.profile_validation import _names, is_int  # imported here, since profile.py imports this module

    spec, components = data["records"], data.get("components")
    problems = []
    labels = spec.get("label_keys")
    if not isinstance(labels, dict) or not _names(list(labels.values())) \
            or (_names(components) and components and set(labels) != set(components)):
        problems.append("records.label_keys must give each component a distinct label column")
    if "stable_flag_stable" in spec and not is_int(spec["stable_flag_stable"]):
        problems.append("records.stable_flag_stable must be an integer")
    return problems


def summary_suffixes(profile):
    """The feature name endings of the recorded statistics, _mean, _std and so on."""
    return tuple(f"_{s}" for s in profile.statistics["recorded"])


def feature_name(channel, statistic, profile):
    """The name of a recorded summary among the features of a record, <Quantity>_<Channel>_<statistic>, with the quantity
    that the profile declares for the channel."""
    return f"{profile.channels[channel]['quantity']}_{channel}_{statistic}"


def group_features(cycle, sensors, profile, suffixes=None):
    """The recorded summaries of the given channels for the statistics that the suffixes name, _mean for the mean, picked
    from the cycle's features by their exact names, in the order of the record. Without suffixes, the statistics are the
    recorded ones of the profile."""
    suffixes = summary_suffixes(profile) if suffixes is None else suffixes
    wanted = {feature_name(s, suffix.removeprefix("_"), profile) for s in sensors for suffix in suffixes}
    return {name: value for name, value in cycle.features.items() if name in wanted}


def percentile(values, q):
    """The q-th quantile (0 <= q <= 1) of the values by linear interpolation between order statistics, the default rule of
    numpy, so that a normal range computed here equals one computed with numpy on the same cycles."""
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def public_name(feature_name, profile):
    """<Quantity>_<Channel>_<statistic> -> the channel's public field with the statistic appended."""
    _quantity, channel, statistic = feature_name.rsplit("_", 2)
    return f"{profile.field_name(channel)}_{statistic}"


def require_same_data(file, stored_sha256, facts_sha256, rebuild):
    """Refuse a file built from the recordings when the data file it names is not the one in use. stored_sha256 is the hash
    the file names, None when it names none, facts_sha256 the hash of the data file in use, and rebuild the sentence that
    says how the file is built again."""
    if facts_sha256 is None:
        raise DataFileError(f"{file} names the data file it was built from, so it is read and written only for recordings that were "
                            "loaded from a data file.")
    if stored_sha256 != facts_sha256:
        named = f"the data file with SHA-256 {stored_sha256}" if stored_sha256 else "a data file that it does not name"
        raise DataFileError(f"{file} was built from {named}, and the data file in use has SHA-256 {facts_sha256}. {rebuild}")


def _declared():
    """The profiles of the assets under CausalDiagnosis/datasets that state the official page of their data. An asset file that
    does not load is left to the tests of its profile, so that a port in progress stops no session."""
    from CausalDiagnosis.datasets.dataset_loader import dataset_packages as ports
    from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile  # imported here, since profile.py imports this module

    profiles = []
    for asset_id in ports.asset_ids():
        try:
            profile = get_profile(asset_id, engine=False)
        # a folder name that is no asset id, a package that does not import, a file that is no JSON, a profile that fails validation
        except (ImportError, LookupError, ValueError):
            continue
        if "page" in profile.records:
            profiles.append(profile)
    return profiles


def data_refusal():
    """What stops a test session before its first test, or None: the refusal of check_data for every bundled asset. A
    missing file is no refusal in a clone that states that it holds no data."""
    stated = os.environ.get(WITHOUT_DATA) == "1"
    refusals = []
    for profile in _declared():
        try:
            check_data(profile)
        except DataFileError as e:
            if not (e.missing and stated):
                refusals.append(str(e))
    if not refusals:
        return None
    return "\n".join(refusals) + (f"\nA clone that holds no data states it with {WITHOUT_DATA}=1, and the tests that need recordings "
                                  "are then skipped.")


def data_header():
    """One line per bundled asset for the head of a test log: its data file with the SHA-256, or that the clone holds none."""
    lines = []
    for profile in _declared():
        path = default_facts_path(profile)
        state = f"{path}, SHA-256 {file_sha256(path)}" if path.exists() else "none"
        lines.append(f"data file of {profile.asset_id}: {state}")
    return lines
