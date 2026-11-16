"""Big layer 0, the base, always on: the hidden fault combination of the start, the recordings of the asset
served through AssetOpsBench's iot, fmsr and wo servers or through the rig's action server, the task tools and the verifier.
It has no small switch. Its fields are the asset with its tables, the rule and the list of the starts, the horizon and the
step cap, the length in seconds of one recorded cycle, which spaces the timestamps of the documents, the guidance level and
the question of the briefing, the anchor date, the verifier's scope, whether the diagnostics server is listed, and the two
keys of the layered tables that no layer switches, the kind of the standard fix of each component and the one cycle that a
waited cycle lasts."""
NAME = "base"
FIELDS = ("asset_id", "components", "levels", "healthy", "sensor_groups", "fault_rule", "allowed", "start", "horizon", "max_steps",
          "seconds_per_cycle", "guidance_level", "question", "anchors", "verifier_scope", "diagnostics_server", "prices.fix_kind",
          "durations.advance_per_cycle")
SWITCHES = {}
