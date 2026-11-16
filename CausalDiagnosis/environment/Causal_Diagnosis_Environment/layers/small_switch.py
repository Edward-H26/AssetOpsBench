"""A small switch of a big layer: the configuration paths it sets, their values when it is off and when it
is on for an asset profile, and the other settings it takes. A path is a field of EpisodeConfig, or a key of `prices` or
`durations`, the two tables whose keys belong to different layers. A setting is "on", "off", one of the switch's modes, a
number for a switch that sets one number, or a mapping whose entries replace parameters of the on value, so a preset can
set any switch on its own (CausalDiagnosis/docs/environment.md, section Layers)."""
import copy

from CausalDiagnosis.datasets.dataset_loader.profile_validation import is_number


def path_value(data, path):
    """The value at a path of a configuration's JSON."""
    field, _, key = path.partition(".")
    return data[field][key] if key else data[field]


def set_path(data, path, value):
    """Set a path of a configuration's JSON in place, so that a key keeps its place in its table and the JSON keeps its
    byte order."""
    field, _, key = path.partition(".")
    if key:
        data[field][key] = copy.deepcopy(value)
    else:
        data[field] = copy.deepcopy(value)


def asset_value(field):
    """The values of a switch that sets one field to the asset's value, the on value of most switches."""
    return lambda profile: {field: profile.config_fields()[field]}


class Switch:
    """off(profile) and on(profile) give {path: value} for every path of the switch. modes maps a mode name to such a
    function, and parameters(mapping, profile) resolves a mapping setting for a switch whose mapping is not the parameters of
    one table or card. needs names the small switches of lower big layers without which the switch has no meaning, each
    with the reason, so that the loader refuses a composition that turns the switch on without them."""

    def __init__(self, name, paths, off, on, modes=None, parameters=None, needs=()):
        self.name, self.paths = name, tuple(paths)
        self._off, self._on = off, on
        self.modes = dict(modes or {})
        self._parameters = parameters
        self.needs = tuple(needs)

    def off(self, profile):
        return self._off(profile)

    def on(self, profile):
        return self._on(profile)

    def is_on(self, data, profile):
        """Whether a configuration's JSON holds the switch away from its off value at any of its paths."""
        return any(path_value(data, path) != value for path, value in self.off(profile).items())

    def values(self, setting, profile):
        """{path: value} of the switch under a setting, or ValueError naming the switch and what it takes."""
        if setting == "on":
            return self.on(profile)
        if setting == "off":
            return self.off(profile)
        if isinstance(setting, str) and setting in self.modes:
            return self.modes[setting](profile)
        if isinstance(setting, dict) and self._parameters is not None:
            return self._parameters(setting, profile)
        if len(self.paths) == 1 and not isinstance(setting, str):
            (path, current), = self.on(profile).items()
            if isinstance(setting, dict) and isinstance(current, dict) and setting and set(setting) <= set(current):
                return {path: {**current, **copy.deepcopy(setting)}}
            if is_number(setting) and (current is None or is_number(current)):
                return {path: setting}
        raise ValueError(f"the small switch {self.name} takes {self.accepted(profile)}, not {setting!r}")

    def accepted(self, profile):
        """The settings the switch takes, in words, for the message of a refused setting."""
        kinds = ["on", "off", *self.modes]
        if self._parameters is not None:
            kinds.append("a mapping of its parameters")
        elif len(self.paths) == 1:
            (current,) = self.on(profile).values()
            if isinstance(current, dict):
                kinds.append("a mapping of the entries of its value")
            elif current is None or is_number(current):
                kinds.append("a number")
        return ", ".join(kinds[:-1]) + f" or {kinds[-1]}"
