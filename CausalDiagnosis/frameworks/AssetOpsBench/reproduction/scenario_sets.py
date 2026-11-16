# Families by id range, since the pinned rows give the FMSR rows no type and the multi-agent rows an empty one. The counts
# are the rows of the pinned file. The paper-era runner held 22 FMSR and 40 end-to-end scenarios instead of 20 and 42.
FAMILIES = (("IoT", 1, 99), ("FMSR", 100, 199), ("TSFM", 200, 299), ("Workorder", 400, 499), ("Multi-agent", 500, 699))
PINNED_COUNTS = {"IoT": 20, "FMSR": 20, "TSFM": 23, "Workorder": 36, "Multi-agent": 42}
# The scenarios whose data and tools IBM's public branches hold (README.md, section "Runnable subset"): 120 of the 141.
RUNNABLE = {
    "IoT": (1, 2, 3, 4, 5, 6, 9, 12, 47),
    "FMSR": tuple(range(101, 121)),
    "TSFM": tuple(range(201, 224)),
    "Workorder": tuple(range(400, 436)),
    "Multi-agent": (502, 506, 507, 510, 511, 514, 515, 517, 518, 519, *range(601, 623)),
}
# Ten of them form the pilot subset, one or two per family, the new tools and data among them. Id 12 is the context stress
# case, since it returns the 2,896 records of the June 2020 telemetry inline.
PILOT = (5, 12, 101, 112, 217, 222, 400, 412, 502, 609)
# What each further scenario needs from the optional rows file of fetch_public_servers_and_data.py: (asset, day), day None for the asset alone.
_APRIL_WEEK = (("Chiller 6", "2020-04-27"),)
RECOVERED_UNLOCKS = {
    **{i: _APRIL_WEEK for i in (11, 501, 503, 504, 505, 508, 509, 512, 513, 516, 520)},
    7: (("Chiller 3", None),),
    8: (("Chiller 6", "2020-12-25"),),
    10: (("Chiller 6", "2020-06-01"), ("Chiller 9", "2020-06-01")),
    41: (("CQPA AHU 1", "2015-09-03"),),
    42: (("CQPA AHU 2B", "2020-09-19"),),
    44: (("CQPA AHU 2B", None),),
    46: (("Chiller 3", None),),
    48: (("CQPA AHU 2B", "2020-09-19"),),
}
SETS = {"runnable": tuple(i for ids in RUNNABLE.values() for i in ids), "pilot": PILOT}
SETS["recovered"] = (*SETS["runnable"], *sorted(RECOVERED_UNLOCKS))  # layer_b.recovered_ids keeps those the file covers
# what the acceptance check of a port takes as a decidable set (acceptance_check.py): a run of one of these sets with this many
# runs per scenario, the paper's own count
ACCEPTANCE_SETS = ("runnable", "recovered")
ACCEPTANCE_REPEATS = 1


def family_of(scenario_id):
    for name, low, high in FAMILIES:
        if low <= int(scenario_id) <= high:
            return name
    raise ValueError(f"scenario id {scenario_id} lies in no family of the paper")


def task_kind(scenario_id):
    """The paper's Table 3 split: multi-agent tasks against single-agent tasks (the four domain families)."""
    return "multi" if family_of(scenario_id) == "Multi-agent" else "single"
