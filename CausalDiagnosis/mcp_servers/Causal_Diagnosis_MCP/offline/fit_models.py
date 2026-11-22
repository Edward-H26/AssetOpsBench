"""Fit the models file of the diagnostics server once and write it to models.json beside this file, a file generated from
the public rig data and never committed; ensure_models fits it on first use in the drivers and the tests
(CausalDiagnosis/docs/causal-diagnosis-mcp.md, section The models file).

Format 9 holds the space of the asset, the stable cycles of every recorded combination, the folds of the stratified cross fit
of the environment's classifiers, the development blocks, which are the recorded combinations outside the test set of the
asset, and for every recorded combination the form of its model: the interaction pairs and the covariance estimator, each with
its log and the blocks its choice left out, the grouping of its covariances (offline/covariance_grouping.py) with the gain
that chose it, and the report of the block effects. The entry form records the rule, where each
part of the form was chosen and a hash of the chosen forms. The fits of an episode leave out the block of its start, so the
server builds them at launch from these cycles and the form of the start, in one fit per hypothesis that also leaves out the
hypothesis's own block or the second block. The same file serves every episode. The fit reads the fault rule of the starts
once, through asset_model.development_blocks, to tell the test set from the development blocks. No tool reads it.

usage: python -m CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.fit_models [--resamples N] [--seed S] [--out PATH]
       python -m CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.fit_models --asset-id ID --out PATH [options] fits another registered asset on the data
       file of its profile into a models file of its own.
"""
from CausalDiagnosis.utils.paths import PACKAGE
import argparse
import datetime
import hashlib
import json
from pathlib import Path

from CausalDiagnosis.utils.atomic_writes_and_locks import files_digest, refuse_writing, write_json_atomically
from CausalDiagnosis.datasets.dataset_loader.dataset_packages import table_path

from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import LIVE_WINDOW, asset_folder, asset_view, default_facts_path, development_blocks, folds_of, is_default_asset, recorded_blocks, require_same_data, served_segments
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.effect_model import BLOCK_EFFECTS, COVARIANCE_ESTIMATORS, FOLD_SEED, FOLDS, EffectSpace, RecordedBlocks, combo_key, model_forms
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.covariance_grouping import GRID, candidates, select_groupings
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_at_one_cycle import DEFAULT_RANKING_MODEL, RANKING_MODELS
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import STATISTICS, file_sha256

MODELS_PATH = table_path(Path(__file__).resolve().with_name("models.json"))
RECORDED_LEVELS = "recorded_levels"  # the evidence kind whose stable cycles the model code fits
MODELS_FORMAT = 9  # 9 since the form of every start holds the grouping of its covariances
RESAMPLES, RESAMPLE_SEED = 200, 0
RULE = ("one fit per hypothesis on the stable cycles of the recorded combinations other than the episode's start and the hypothesis, "
        "or, for a hypothesis without a block of its own among them, other than the start and one second block that a hash of the "
        "start's levels picks among the recorded combinations one component away from the start, or among all of them when none "
        "is recorded, so that every fit leaves out two blocks, for its mean, its within covariance by the covariance estimator "
        "of the start's form and its block covariance of leave one block out misfits, each mixed with the centred sample covariance of the "
        "blocks at the hypothesis's level of the component that the grouping of the start's form names, without the same left out "
        "block, at one minus its weight; the form of every start chosen before any episode as the entry form states; the conformal "
        "calibration once per start on the blocks without the start's")
FORM_RULE = ("the development blocks are the recorded combinations outside the test set; a part of the form that the development "
             "blocks identify is chosen on them, for a start among them on the others; a part that they do not identify is chosen once "
             "per start on every recorded block but the two that the fit of the start's own hypothesis leaves out, the start's and its "
             "second block; the covariance estimator of one cycle is the registered one of the largest held out within block density of "
             "window means on the development blocks, ties to the earlier registered one, and the first registered one when the held "
             "out windows give fewer degrees of freedom than there are summary fields or when no estimator can be evaluated; the "
             "interaction pairs are chosen by forward selection on the held out log likelihood of the cycles, on the development blocks "
             "when every pool of them that the rule uses identifies every pair in every training fold, and per start otherwise; the "
             "block effect is the registered ranking model, and its held out density at window level is reported for both registered "
             "values and chooses nothing; the grouping of the covariances is the pooled form or a component and a weight, the candidate of "
             "the largest held out log score of the true start over the development starts other than the start, pooled unless it wins by "
             "more than one standard error of the paired total, the nested gain reported; the entry parts says where each part was chosen")
FORM_WINDOW = LIVE_WINDOW  # the window of the window means, the width of the live window that the environment serves
REFIT = "Remove it and fit again with `python -m CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.fit_models`."


def fit(facts_path, resamples=RESAMPLES, seed=RESAMPLE_SEED, asset_id=None):
    """The content of models.json for the asset that asset_id names, the default asset when None."""
    view = asset_view(asset_id)
    space = EffectSpace.from_view(view)
    blocks, records = recorded_blocks(view, facts_path)
    fold_of = folds_of(view, FOLDS, FOLD_SEED)
    recorded = [h for h in space.combinations if h in blocks]
    development = development_blocks(view, blocks)
    forms, parts = model_forms(space, blocks, development, fold_of, FORM_WINDOW)
    parts["block_effect"]["value"] = DEFAULT_RANKING_MODEL
    chosen, report, scores = select_groupings(RecordedBlocks(space, blocks, window=FORM_WINDOW), forms, development, view, served_segments())
    for start, form in forms.items():
        form["grouping"] = grouping_json(space, chosen[start])
    parts["grouping"] = {"chosen": "per_start", "nested_gain": report}
    out = {
        "format": MODELS_FORMAT,
        "generated": datetime.date.today().isoformat(),
        "facts_sha256": facts_sha256(facts_path),
        "records": records,
        "fit": {"rule": RULE, "fitted": len(recorded), "cycles": int(sum(len(blocks[h]) for h in recorded)), "folds": FOLDS,
                "fold_seed": FOLD_SEED, "resamples": resamples, "resample_seed": seed},
        "space": space.to_json(),
        "ranking_models": list(RANKING_MODELS), "default_ranking_model": DEFAULT_RANKING_MODEL,
        "folds": {combo_key(h): int(fold_of[h]) for h in recorded},
        "blocks": {combo_key(h): blocks[h].tolist() for h in recorded},
        "form": {"rule": FORM_RULE, "window": FORM_WINDOW, "covariances": list(COVARIANCE_ESTIMATORS), "block_effects": list(BLOCK_EFFECTS),
                 "development": [combo_key(h) for h in development], "parts": parts,
                 "grouping": {"weights": list(GRID), "candidates": [None if c is None else [space.components[c[0]], c[1]] for c in candidates(space)],
                              "scores": {combo_key(s): v for s, v in scores.items()}}},
        "selections": {combo_key(start): selection_json(space, form) for start, form in forms.items()},
    }
    out["form"]["sha256"] = form_sha256(out)
    return out


def grouping_json(space, choice):
    """The grouping of one start as models.json holds it: the component by name and the weight, both null for the pooled form,
    with the gain over the pooled form and its standard error that chose it."""
    candidate = choice["candidate"]
    return {"component": None if candidate is None else space.components[candidate[0]], "weight": None if candidate is None else candidate[1],
            "gain": choice.get("gain", 0.0), "se": choice.get("se", 0.0)}


def selection_json(space, form):
    """The form of one start as models.json holds it: the pairs by component name and the blocks by combination key."""
    def named(pairs):
        return [[space.components[a], space.components[b]] for a, b in pairs]
    return {**form, "pairs": named(form["pairs"]), "report_pairs": named(form["report_pairs"]),
            "left_out": [combo_key(h) for h in form["left_out"]], "covariance_left_out": [combo_key(h) for h in form["covariance_left_out"]]}


def form_sha256(models):
    """The hash of the chosen forms of a models file: the development blocks, where each part was chosen, and the pairs and the
    covariance estimator of every start. A freeze record can name it, so that a refit that chose another form is seen."""
    chosen = {"development": models["form"]["development"], "parts": models["form"]["parts"],
              "selections": {key: {**{k: s[k] for k in ("pairs", "covariance")}, "grouping": [s["grouping"]["component"], s["grouping"]["weight"]]}
                             for key, s in models["selections"].items()}}
    return hashlib.sha256(json.dumps(chosen, sort_keys=True).encode()).hexdigest()


def facts_sha256(facts_path):
    return file_sha256(facts_path)


def code_digest():
    """The digest of the code that writes the models file and the grade tables: every .py file of this package outside its tests
    folders, which both record, so that a change of the server's code fits and grades again as a change of the engine labels
    again."""
    return files_digest(str(PACKAGE / "mcp_servers" / "Causal_Diagnosis_MCP"), "")


def models_problems(path, asset_id=None, facts_path=None):
    """What a check of the tables finds wrong with the models file: missing, of another format or fitted by other code. A file of
    another data file is refused as ensure_models refuses it. Reads the file and writes nothing."""
    facts_path = default_facts_path(asset_id) if facts_path is None else facts_path
    if path is None or not Path(path).exists():
        return [f"no models file at {path}"]
    stored = json.loads(Path(path).read_text(encoding="utf-8"))
    if stored.get("format") != MODELS_FORMAT:
        return [f"{Path(path).name} has format {stored.get('format')}, not {MODELS_FORMAT}"]
    require_same_data(path, stored.get("facts_sha256"), facts_sha256(facts_path), REFIT)
    return [] if stored.get("code") == code_digest() else [f"{Path(path).name} was fitted by other code"]


def write_models(path=None, asset_id=None, **options):
    out = {**fit(default_facts_path(asset_id), asset_id=asset_id, **options), "code": code_digest()}
    write_json_atomically(MODELS_PATH if path is None else path, out)
    return out


def ensure_models(path=None, asset_id=None, facts_path=None):
    """Fit the models of the asset that asset_id names, the default asset when None, when the file is missing or in another
    format or fitted by other code. A file fitted on another data file than the one in use is refused and never refitted in place, so
that no stale
    file is served and none is replaced unseen. facts_path is the data file to fit on and to compare
    with, the data file of that asset when None, so the file of another asset is fitted for that asset when it is named.
    Without a path the file is MODELS_PATH as the module holds it at the call, which the tests point at a copy."""
    path = MODELS_PATH if path is None else path
    facts_path = default_facts_path(asset_id) if facts_path is None else facts_path
    stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if stored.get("format") == MODELS_FORMAT:
        require_same_data(path, stored.get("facts_sha256"), facts_sha256(facts_path), REFIT)
    if stored.get("format") != MODELS_FORMAT or stored.get("code") != code_digest():
        refuse_writing(f"the models file {path}")
        write_json_atomically(path, {**fit(facts_path, asset_id=asset_id), "code": code_digest()})
    return path


def models_path(asset_id=None):
    """The models file of an asset: MODELS_PATH beside this file for the default asset, which the server serves when it is given
    no path, models.json in the data folder of the folder of any other asset, where a port keeps what it generates, and None for
    an asset registered without a folder."""
    if is_default_asset(asset_id):
        return MODELS_PATH
    folder = asset_folder(asset_id)
    return None if folder is None else folder / "data" / "models.json"


def unfit_reasons(asset_id=None, recorded=()):
    """Why the model code of the server cannot fit an asset on its recorded combinations, an empty list when it can: the
    recordings must be of the recorded levels kind, the asset must serve the summaries the model reads (STATISTICS), and the
    recorded combinations must admit the development split on which the form of the models is chosen."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import DEFAULT_ASSET, development_starts, get_profile

    profile = get_profile(asset_id or DEFAULT_ASSET)
    reasons = [] if models_path(asset_id) is not None else ["it has no folder on the search path of CausalDiagnosis.datasets to keep its models in"]
    if profile.evidence_kind != RECORDED_LEVELS:
        reasons.append(f"its evidence kind is {profile.evidence_kind}, and the model code fits recordings of the kind {RECORDED_LEVELS}")
    served = tuple(profile.statistics["served"])
    if served != tuple(STATISTICS):
        reasons.append(f"it serves the statistics {', '.join(served)}, and the model code reads {', '.join(STATISTICS)}")
    try:
        development_starts(profile, {tuple(c) for c in recorded})
    except ValueError as e:  # development_split.SplitRefused
        reasons.append(f"its recordings admit no development split, on which the form of the models is chosen ({e})")
    return reasons


def refuse_default_files(parser, args, **defaults):
    """Refuse a command line of another asset than the default whose options keep a default file, since models.json and
    grades.json beside this file and the results folder of the rig serve the default asset alone."""
    kept = [f"--{option}" for option, default in defaults.items() if getattr(args, option) == default]
    if not is_default_asset(args.asset_id) and kept:
        parser.error(f"--asset-id {args.asset_id} needs its own {' and '.join(kept)}, since the default files serve the default asset")


def main(argv=None):
    parser = argparse.ArgumentParser(description="fit models.json of the diagnostics server from the recordings of an asset, the rig by default")
    parser.add_argument("--asset-id", help="a registered asset profile, the default asset when omitted")
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument("--seed", type=int, default=RESAMPLE_SEED)
    parser.add_argument("--out", type=Path, default=MODELS_PATH)
    args = parser.parse_args(argv)
    refuse_default_files(parser, args, out=MODELS_PATH)
    out = write_models(args.out, asset_id=args.asset_id, resamples=args.resamples, seed=args.seed)
    distinct = sorted({tuple(map(tuple, s["pairs"])) for s in out["selections"].values()})
    estimators = sorted({s["covariance"] for s in out["selections"].values()})
    print(f"wrote {args.out}: format {out['format']}, {out['fit']['fitted']} combinations, {len(out['form']['development'])} development blocks, "
          f"pairs chosen {out['form']['parts']['pairs']['chosen']}, {len(distinct)} distinct pair set(s), covariance {', '.join(estimators)}, "
          f"grouping {sorted({(s['grouping']['component'], s['grouping']['weight']) for s in out['selections'].values()}, key=str)}, "
          f"form {out['form']['sha256'][:12]}, {args.out.stat().st_size} bytes")


if __name__ == "__main__":
    main()
