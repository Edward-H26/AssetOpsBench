from CausalDiagnosis.datasets.dataset_loader.ground_truth import misdiagnosed
from CausalDiagnosis.utils.public_rules.visits import is_no_call
from . import episode_log


MISDIAGNOSIS_EVENT = "misdiagnosis"  # the hidden system entry of the misdiagnosis charge


# ----- the scores -----
def _costs(log, end, cost_oracle, unit="money"):
    """In money, the prices paid plus the hidden ledger of penalties and downtime: agent entries carry their price, a
    failure event carries its penalty and a misdiagnosis entry the misdiagnosis price, and the downtime of the episode
    is listed by cycle in the end event and summed
    there, so the three are added here. On the free base row the unit is calls, so the cost is the number
    of tool calls, refused ones included as the step cap counts them, a no_call step not (visits.is_no_call), and the oracle
    policy's cost is its own number of calls, which keeps the normalized cost defined where every price is zero."""
    prices = sum(e.get("cost") or 0 for e in log if e.get("step") is not None)
    penalties = sum(e.get("cost") or 0 for e in log if e.get("step") is None and e["event"] != "end")
    downtime = end.get("downtime_total", 0)
    cost = prices + penalties + downtime if unit == "money" else len({e["step"] for e in log if e.get("step") is not None and not is_no_call(e)})
    normalized = cost / cost_oracle if cost_oracle else None
    return {"cost": cost, "cost_unit": unit, "cost_prices": prices, "penalties": penalties, "downtime_total": downtime,
            "cost_oracle": cost_oracle, "cost_normalized": normalized}


def misdiagnosis_charge(log, diagnosis):
    """The misdiagnosis price that a diagnosis written after the entries of log is charged: prices.misdiagnosis of the
    configuration of the reset entry when the diagnosis misses the truth at its write in the level of any component, and
    0 when it does not, when there is no diagnosis or when the rules price no misdiagnosis. The engine calls it at the
    write on its own log, and the verdict reads the charge that the log then holds, so the charge and
    diagnosis_all_correct apply one predicate to one truth."""
    price = (episode_log._reset_config(log).get("prices") or {}).get("misdiagnosis") or 0
    if price <= 0 or diagnosis is None:
        return 0
    asset = episode_log._asset_truth(log)
    truth = episode_log._combination_after(log, asset.levels(log[0]["hidden"]), asset)
    return price if misdiagnosed(asset, asset.levels(diagnosis), truth) else 0


def _misdiagnosis_field(log):
    """The misdiagnosis charge of the log beside the penalties, present only when the log holds one, so that the verdict of
    every episode without a charged diagnosis keeps its keys."""
    charges = [e.get("cost") or 0 for e in log if e["event"] == MISDIAGNOSIS_EVENT]
    return {"misdiagnosis_charge": sum(charges)} if charges else {}


def misdiagnosis_charge_of(verdict):
    """The misdiagnosis charge that a verdict names, 0 for a verdict without the field, as every verdict of an episode
    without a charged diagnosis is: the reader of the field for the tables, so that the cost without the charge,
    verdict["cost"] - misdiagnosis_charge_of(verdict), puts a charged and an uncharged episode on one scale."""
    return verdict.get("misdiagnosis_charge", 0)
