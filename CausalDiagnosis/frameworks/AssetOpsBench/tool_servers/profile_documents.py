"""The AssetOpsBench documents of an asset profile (CausalDiagnosis/docs/adding-a-dataset.md).

The producer of CausalDiagnosis/frameworks/AssetOpsBench/tool_servers writes these documents into the databases of an episode for both backends, the memory store and CouchDB,
so that AssetOpsBench's own iot, fmsr and wo servers serve the asset as they serve their sample assets. The registry
holds the asset with one child per component, fmsr holds one failure mode document per component class and one for the
codes of the whole asset, and a telemetry record carries the plain statistic of a channel, the served statistic listed
first, under the channel's field and every other served statistic under <field>_<statistic>, so every installed name is
a measured name. Every value comes
from the public tables of the profile, the registry file it names and the timeline of the episode, never from a hidden
field, a recording index or a stock date. The module names no asset, and the producer normalises the documents as
AssetOpsBench's loader does.
"""
import json
import re

from CausalDiagnosis.datasets.dataset_loader.recordings import REPO  # the fork root, against which a profile names its files

STATUS = "OPERATING"


def class_key(asset_class):
    """The key under which fmsr files the failure modes of an asset class, as src/servers/fmsr/main.py computes it:
    digits removed, underscores and hyphens as spaces, lower case."""
    key = re.sub(r"\d+", "", asset_class or "")
    key = re.sub(r"[_\-]+", " ", key)
    return re.sub(r"\s+", " ", key).strip().lower()


def stock_entry(profile):
    """The entry of the asset in the registry file the profile names, or None."""
    name = profile.text.get("registry_file")
    if not name:
        return None
    try:
        entries = json.loads((REPO / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return next((e for e in entries if isinstance(e, dict) and e.get("assetnum") == profile.asset_id), None)


def class_key_problems(components, asset_type):
    """The problems of a table whose components, or a component and the asset type, share an fmsr class key, since the
    fmsr server would file their failure modes under one document: one problem for every holder of a key after its first."""
    owners, problems = {}, []
    for owner, key in [*((c, class_key(c.upper())) for c in components), ("the asset type", class_key(asset_type))]:
        if key in owners:
            problems.append(f"{owners[key]} and {owner} share the fmsr class {key!r}, under which the fmsr server files their failure modes")
        else:
            owners[key] = owner
    return problems


def surface_profile_problems(profile):
    """The problems of an asset profile that the tool surface of these servers finds when a run of it starts and in the port
    contract test (utils/frameworks.py surface_profile_problems): components, or a component and the asset type, that share an
    fmsr class key, for a profile whose evidence kind plays episodes. profile is a loaded profile or the raw mapping of an
    asset.json; a raw profile whose components or asset type the core's own check refuses is left to that check."""
    from CausalDiagnosis.datasets.dataset_loader import profile_validation, evidence_kinds as kinds

    if isinstance(profile, dict):
        kind, components, asset_type = kinds.kind_of(profile.get("evidence_kind")), profile.get("components"), profile_validation._asset_type(profile)
    else:
        kind, components, asset_type = kinds.kind_of(profile.evidence_kind), list(profile.components), profile.text["fallback"]["asset_type"]
    if kind is None or not kind.plays_episodes or asset_type is None:
        return []
    if not isinstance(components, list) or not all(isinstance(c, str) and c for c in components) or len(set(components)) != len(components):
        return []
    return class_key_problems(components, asset_type)


def class_keys(profile):
    """The fmsr key of every component and, under None, of the whole asset. Two that collide would share one failure
    mode document, so the profile is refused (class_key_problems)."""
    asset_type = profile.text["fallback"]["asset_type"]
    problems = class_key_problems(profile.components, asset_type)
    if problems:
        raise ValueError(f"{profile.asset_id}: {problems[0]}")
    keys = {component: class_key(component.upper()) for component in profile.components}
    keys[None] = class_key(asset_type)
    return keys


class ProfileDocuments:
    """The document provider of the producer for one profile and the timeline of one episode, which gives the
    timestamp of a cycle."""

    def __init__(self, profile, timeline):
        self.profile = profile
        self.timeline = timeline
        self.classes = class_keys(profile)

    def registry(self):
        """The asset, then one child per component in profile order. The asset keeps the fields of its stock entry in
        their order, with the description, type and site of the profile and the channel fields as its sensors, since
        an agent asks for history by the installed names."""
        p = self.profile
        fallback = p.text["fallback"]
        asset = {"assetnum": p.asset_id, **{k: v for k, v in (stock_entry(p) or {}).items() if k != "assetnum"}}
        asset.update({"description": fallback["description"], "assettype": fallback["asset_type"], "siteid": fallback["site"]})
        asset.setdefault("status", STATUS)
        asset.setdefault("parent", None)
        asset["sensors"] = [p.field_name(channel) for channel in p.channels]
        children = [{"assetnum": f"{p.asset_id}_{component}", "description": f"The {component} of {p.asset_id}",
                     "siteid": fallback["site"], "orgid": asset.get("orgid"), "status": STATUS, "assettype": component.upper(),
                     "parent": p.asset_id, "sensors": []} for component in p.components]
        return [asset, *children]

    def failure_modes(self):
        """One document per component class with the codes of that component in table order, then one under the
        asset's class with the codes of no component, each exhaustive and citing the profile's FMEA source."""
        p = self.profile
        owner = {e["code"]: (e["component"] if e["component"] in p.components else None) for e in p.failure_modes}
        return [{"asset_class": self.classes[key], "failure_modes": [code for code, o in owner.items() if o == key],
                 "exhaustive": True, "source": p.knowledge["fmea_source"]} for key in (*p.components, None)]

    def telemetry(self, cycle, values):
        """The record of one cycle from the served values the engine names <field>_<statistic>: the asset, the cycle's
        timestamp, then every channel in profile order with its statistics in served order, the plain statistic under
        the bare field. A value under any other name is refused, so no field outside the served ones reaches a record."""
        p = self.profile
        record = {"asset_id": p.asset_id, "timestamp": self.timeline.timestamp(cycle)}
        unknown, plain = set(values), p.plain_statistic()
        for channel in p.channels:
            field = p.field_name(channel)
            for statistic in p.statistics["served"]:
                name = f"{field}_{statistic}"
                if name in values:
                    record[field if statistic == plain else name] = values[name]
                    unknown.discard(name)
        if unknown:
            raise ValueError(f"{p.asset_id}: {sorted(unknown)} are no served fields of the profile")
        return record
