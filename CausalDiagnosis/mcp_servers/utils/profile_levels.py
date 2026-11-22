"""The levels of the asset profile that the task server of the AssetOpsBench rows answers, the parameter response_format of
get_asset_profile, which its metering layer adds to the rig's signature. The module stands among the utilities of the servers,
outside the frameworks, since the episode viewer reads a recorded profile at every level through raw_of. raw is the profile as the task server always wrote it. The sub-agents
of the two reference designs never see the task text and read the rules from the profile alone, so concise and detailed keep
every rule and every value and shorten only what the profile repeats: the asset named once, the normal ranges as a long table
with one entry per field, and the tool table as its columns once and one row per tool, as JSON without indentation. headline is
one line."""

# the line on the levels in the description of get_asset_profile, whose shorter levels keep every rule
LEVELS_LINE = (" response_format is headline (one line), concise (the default) or detailed, both every rule and value once with the normal "
               "ranges and the tool table as long tables in one line of JSON, or raw (the profile with one entry per field and per tool).")


def _table(rows):
    """{"columns", "rows"}: the keys of the rows in the order they first appear, once, and each row as its values in that order,
    None where a row lacks a key."""
    columns = list(dict.fromkeys(key for row in rows for key in row))
    return {"columns": columns, "rows": [[row.get(column) for column in columns] for row in rows]}


def profile(view, level):
    """The profile as the task server sends it (MeterConfig.profile_view) at a shorter level."""
    if level == "headline":
        asset = view.get("asset") or {}
        budget = "" if view.get("budget") is None else f", budget {view['budget']}"
        return {"headline": f"{view['asset_id']} ({asset.get('asset_type')}) at site {asset.get('site')}: {len(view['components'])} components, "
                            f"horizon {view['horizon']} cycles, step cap {view['max_steps']}{budget}"}
    out = {}
    for key, value in view.items():
        if key == "asset" and value.get("asset_id") == view.get("asset_id"):
            out[key] = {k: v for k, v in value.items() if k != "asset_id"}
        elif key == "normal_ranges":
            out[key] = _table([{"field": field, **band} for field, band in value.items()])
        elif key == "tools":
            out[key] = _table(value)
        else:
            out[key] = value
    return out


def raw_of(shorter):
    """The raw profile rebuilt from a concise or detailed one, for a reader of a record that renders the profile an agent read,
    such as the episode walkthrough: the asset with the profile's asset_id and every long table back to one entry per field or
    per tool."""
    def entries(table):
        return [{column: value for column, value in zip(table["columns"], row) if value is not None} for row in table["rows"]]

    view = {k: v for k, v in shorter.items() if k != "unchanged"}
    view["asset"] = {"asset_id": shorter["asset_id"], **shorter["asset"]}
    view["normal_ranges"] = {entry.pop("field"): entry for entry in entries(shorter["normal_ranges"])}
    view["tools"] = entries(shorter["tools"])
    return {key: view[key] for key in shorter if key != "unchanged"}
