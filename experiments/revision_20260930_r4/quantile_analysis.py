"""Fixed q=.90 scale sensitivity; separate outputs preserve all prior R4 CSVs."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze as r4

ROOT = r4.ROOT
HERE = Path(__file__).resolve().parent
Q = 0.90
CAL = "quantile90"
POLICY6 = "replicate_excluded_six_both_modes"
POLICY8 = "eight_original_stable"


def quantile(values: np.ndarray) -> float:
    r4.positive(values, "quantile90 raw maxima")
    if len(values) not in (4, 6, 8):
        raise ValueError("quantile90 membership must have four, six or eight streams")
    return float(np.quantile(values, Q, method="linear"))


def held(panel: str) -> pd.DataFrame:
    h = r4.read(ROOT / f"experiments/stable_holdout/panel_{panel.lower()}/holdout_streams.csv")
    r4.positive(h.max_statistic, panel + " additional maxima")
    return h.assign(panel=panel, stable=True, condition="stable_" + h["mode"],
                    GLR=[None] * len(h), cut_search=[None] * len(h))


def fit(stable: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if not stable.stable.all():
        raise ValueError("Only original stable streams may enter scale fitting")
    divisors, members, assignments, levels = [], [], [], {}
    for panel, f in stable.groupby("panel", sort=True):
        for stat in ("GLR", "cut_search"):
            full = {}
            for seed, group in f.groupby("checkpoint_seed", sort=True):
                for rep in (-1, 0, 1, 2, 3):
                    subset = group if rep == -1 else group[group.replicate != rep]
                    values = np.array([p.max() for p in subset[stat]])
                    n = 8 if rep == -1 else 6
                    assert len(values) == n
                    d = quantile(values)
                    identity = f"{panel}:{stat}:{CAL}:{seed}:exclude_{rep}"
                    bound = 10 / 3 if n == 8 else 2
                    assert values.max() / d <= bound * (1 + 1e-14)
                    divisors.append(dict(panel=panel, statistic=stat, calibration=CAL,
                        checkpoint_seed=seed, excluded_replicate=rep, divisor_id=identity,
                        divisor=d, estimation_streams=n, quantile_probability=Q,
                        quantile_method="linear_on_raw_maxima", maximum=float(values.max()),
                        maximum_to_divisor=float(values.max() / d), structural_bound=bound))
                    members.extend(dict(divisor_id=identity, stream_id=r.stream_id,
                                        stable_maximum=float(p.max()))
                                   for r, p in zip(subset.itertuples(), subset[stat], strict=True))
                    if rep == -1:
                        full[seed] = d
            for seed in full:
                foreign = f[f.checkpoint_seed != seed]
                assert len(foreign) == 152
                rows = [dict(panel=panel, statistic=stat, calibration=CAL,
                    target_checkpoint=seed, foreign_checkpoint=r.checkpoint_seed,
                    foreign_stream_id=r.stream_id, maximum=float(getattr(r, stat).max()),
                    foreign_divisor=full[r.checkpoint_seed],
                    normalized_maximum=float(getattr(r, stat).max()) / full[r.checkpoint_seed])
                    for r in foreign.itertuples()]
                levels[panel, stat, seed] = max(x["normalized_maximum"] for x in rows)
                assignments.extend(rows)
    return pd.DataFrame(divisors), pd.DataFrame(members), pd.DataFrame(assignments), levels


def build_points(f: pd.DataFrame, d: pd.DataFrame, levels: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = d.set_index("divisor_id").divisor
    original = r4.read(HERE / "results/operating_points.csv")
    matched = r4.read(HERE / "results/matched_operating_points.csv")
    points, stops, checkpoint = [], [], []
    for panel, original_caps in (("D", (4, 2)), ("E", (4, 1))):
        frame = f[f.panel == panel]
        additional = held(panel)
        for family in ("matched_target_budget", "additional_stable"):
            is_six = family == "matched_target_budget"
            data = frame if is_six else pd.concat([frame[~frame.stable], additional], ignore_index=True)
            for stat in (("GLR", "cut_search") if is_six else ("GLR",)):
                ids = [f"{panel}:{stat}:{CAL}:{r.checkpoint_seed}:exclude_{r.replicate if is_six else -1}"
                       for r in data.itertuples()]
                base = np.array([levels[panel, stat, r.checkpoint_seed] * index.loc[i]
                                 for r, i in zip(data.itertuples(), ids, strict=True)])
                mask = data.stable.to_numpy()
                maxima = np.array([float(p.max()) if p is not None else float(x)
                    for p, x in zip(data[stat], data.get("max_statistic", pd.Series(np.nan, index=data.index)), strict=True)])
                r4.positive(maxima, panel + stat + " evaluation maxima")
                configurations = (
                    [(name, cap) for name, cap in zip(r4.CAPS,
                        (original_caps, tuple(c // 2 for c in original_caps), (0, 0)), strict=True)]
                    if is_six else [(f"total_{cap}", cap) for cap in r4.old.CAPS])
                for budget, cap in configurations:
                    candidates = matched if is_six else original
                    match = candidates[(candidates.panel == panel) & (candidates.statistic == stat) &
                                       (candidates.family == family) & (candidates.budget == budget)]
                    anchor = match[match.calibration == "arithmetic"].iloc[0].point_id
                    selection = r4.old.choose(maxima[mask], base[mask], data.loc[mask, "mode"].to_numpy(), cap)
                    pid = (f"{panel}:{stat}:{CAL}:{budget}:matched6" if is_six
                           else f"{panel}:{stat}:{CAL}:additional_{cap}")
                    meta = dict(panel=panel, statistic=stat, calibration=CAL, family=family,
                        budget=budget, point_id=pid, source_point_id=anchor,
                        source_point_role="comparison_anchor",
                        comparator_point_ids=json.dumps(sorted(match.point_id)),
                        target_budget_policy=POLICY6 if is_six else POLICY8,
                        divisor_identity="panel:statistic:quantile90:checkpoint:exclude_replicate",
                        quantile_probability=Q, quantile_method="linear_on_raw_maxima",
                        selection="stable_original_only" if is_six else "stable_additional_only",
                        evaluation_role="retrospective_quantile_sensitivity",
                        frozen_cap=cap[0] if is_six else None,
                        updating_cap=cap[1] if is_six else None,
                        total_cap=None if is_six else cap, **selection)
                    result, rows, cps = r4.old.evaluate(data, stat, base, meta)
                    for row, identity in zip(rows, ids, strict=True):
                        row.update(source_point_id=anchor, source_point_role="comparison_anchor",
                                   target_budget_policy=meta["target_budget_policy"], divisor_id=identity,
                                   target_estimation_streams=6 if is_six else 8,
                                   foreign_calibration_streams=152)
                    for cp in cps:
                        cp.update(source_point_id=anchor, source_point_role="comparison_anchor",
                                  target_budget_policy=meta["target_budget_policy"],
                                  divisor_identity=meta["divisor_identity"])
                    points.append(result)
                    stops.extend(rows)
                    checkpoint.extend(cps)
    assert len(points) == 24 and len(stops) == 11520 and len(checkpoint) == 480
    return pd.DataFrame(points), pd.DataFrame(stops), pd.DataFrame(checkpoint)


def mechanism(points: pd.DataFrame, stops: pd.DataFrame, divisors: pd.DataFrame,
              assignments: pd.DataFrame, levels: dict, out: Path) -> None:
    old_div = r4.read(HERE / "results/divisors.csv").set_index("divisor_id").divisor
    qdiv = divisors.set_index("divisor_id").divisor
    old_levels = r4.read(HERE / "results/thresholds.csv").groupby(
        ["panel", "statistic", "calibration", "checkpoint_seed"]).foreign_level.first()
    old_points = pd.concat([r4.read(HERE / "results/operating_points.csv"),
                            r4.read(HERE / "results/matched_operating_points.csv")]).set_index("point_id")
    old_stops = pd.concat([r4.read(HERE / "results/stopping_times.csv"),
                          r4.read(HERE / "results/matched_stopping_times.csv")])
    old_groups = {pid: g.set_index("stream_id") for pid, g in old_stops.groupby("point_id")}
    donors, allocation, reconcile, contrasts = [], [], [], []
    for keys, g in assignments.groupby(["panel", "statistic", "target_checkpoint"]):
        tied = g[g.normalized_maximum == levels[keys]]
        donors.extend(dict(row, donor_tie_count=len(tied), foreign_calibration_records=len(g))
                      for row in tied.to_dict("records"))
    for p in points.itertuples():
        right = stops[stops.point_id == p.point_id].set_index("stream_id")
        is_six = p.family == "matched_target_budget"
        for other_id in json.loads(p.comparator_point_ids):
            other = old_points.loc[other_id]
            left = old_groups[other_id]
            assert set(left.index) == set(right.index)
            meta = dict(panel=p.panel, statistic=p.statistic, family=p.family, budget=p.budget,
                        quantile_point_id=p.point_id, comparator_point_id=other_id,
                        comparator=other.calibration, source_point_id=p.source_point_id,
                        source_point_role="comparison_anchor", target_budget_policy=p.target_budget_policy)
            for seed in sorted(right.checkpoint_seed.unique()):
                for rep in (range(4) if is_six else (-1,)):
                    qi = f"{p.panel}:{p.statistic}:{CAL}:{seed}:exclude_{rep}"
                    oi = f"{p.panel}:{p.statistic}:{other.calibration}:{seed}:exclude_{rep}"
                    qd, od = float(qdiv.loc[qi]), float(old_div.loc[oi])
                    qb = levels[p.panel, p.statistic, seed]
                    ob = float(old_levels.loc[(p.panel, p.statistic, other.calibration, seed)])
                    qh, oh = (qb * qd) * p.multiplier, (ob * od) * other.multiplier
                    allocation.append(dict(meta, checkpoint_seed=seed, excluded_replicate=rep,
                        quantile_divisor_id=qi, comparator_divisor_id=oi,
                        quantile_divisor=qd, comparator_divisor=od,
                        quantile_foreign_extreme=qb, comparator_foreign_extreme=ob,
                        quantile_multiplier=p.multiplier, comparator_multiplier=other.multiplier,
                        quantile_threshold=qh, comparator_threshold=oh,
                        threshold_ratio=qh / oh,
                        decomposed_ratio=(p.multiplier / other.multiplier) * (qd / od) * (qb / ob)))
            for sid, q in right.iterrows():
                o = left.loc[sid]
                assert q.maximum == o.maximum
                between = min(q.threshold, o.threshold) < q.maximum <= max(q.threshold, o.threshold)
                assert between == (q.fired != o.fired)
                available = bool(q.stopping_time_available and o.stopping_time_available)
                reconcile.append(dict(meta, stream_id=sid, checkpoint_seed=q.checkpoint_seed,
                    replicate=q.replicate, condition=q.condition, mode=q["mode"], stable=q.stable,
                    quantile_divisor_id=q.divisor_id, maximum=q.maximum,
                    quantile_threshold=q.threshold, comparator_threshold=o.threshold,
                    quantile_fired=q.fired, comparator_fired=o.fired,
                    quantile_gained=bool(q.fired and not o.fired),
                    quantile_lost=bool(o.fired and not q.fired),
                    maximum_between_thresholds=bool(between), stopping_time_available=available,
                    quantile_delay=q.restricted_delay, comparator_delay=o.restricted_delay,
                    delay_difference=q.restricted_delay - o.restricted_delay if available else None))
            for seed in sorted(right.checkpoint_seed.unique()):
                q = right[(right.checkpoint_seed == seed) & (right.condition == r4.PRIMARY)]
                o = left.loc[q.index]
                contrasts.append(dict(meta, checkpoint_seed=seed, primary_denominator=len(q),
                    quantile_detections=int(q.fired.sum()), comparator_detections=int(o.fired.sum()),
                    detection_difference=int(q.fired.sum() - o.fired.sum()),
                    quantile_delay=float(q.restricted_delay.mean()),
                    comparator_delay=float(o.restricted_delay.mean()),
                    delay_difference=float(q.restricted_delay.mean() - o.restricted_delay.mean())))
    for name, rows in [("foreign_extreme_donors", donors), ("threshold_allocation", allocation),
                       ("detection_delay_reconciliation", reconcile), ("checkpoint_contrasts", contrasts)]:
        r4.write(rows, out / (name + ".csv"))


def dispersion(f: pd.DataFrame, out: Path) -> None:
    rows, pairs, members, deletions, ranges = [], [], [], [], []
    for panel in ("D", "E"):
        stable = f[(f.panel == panel) & f.stable]
        configs = [(stat, ",".join(map(str, rep)), stable[stable.replicate.isin(rep)],
                    stable[~stable.replicate.isin(rep)], 4)
                   for stat in ("GLR", "cut_search") for rep in itertools.combinations(range(4), 2)]
        configs.append(("GLR", "all_original_to_additional", stable, held(panel), 8))
        for stat, allocation, est, test, n in configs:
            assert set(est.stream_id).isdisjoint(test.stream_id)
            scale = {}
            for seed, group in est.groupby("checkpoint_seed"):
                identity = f"{panel}:{stat}:{CAL}:{seed}:allocation_{allocation}"
                values = np.array([p.max() for p in group[stat]])
                d = quantile(values)
                scale[seed] = d
                members.extend(dict(divisor_id=identity, panel=panel, statistic=stat,
                    allocation=allocation, checkpoint_seed=seed, stream_id=r.stream_id,
                    stable_maximum=float(p.max()), divisor=d, estimation_streams=n)
                    for r, p in zip(group.itertuples(), group[stat], strict=True))
            test = test.copy()
            test["logmax"] = [np.log(p.max()) if p is not None else np.log(x)
                for p, x in zip(test[stat], test.get("max_statistic", pd.Series(np.nan, index=test.index)), strict=True)]
            g = test.groupby("checkpoint_seed").logmax
            for name, target in (("mean_log_maximum", g.mean()), ("q90_log_maximum", g.quantile(Q)),
                                 ("max_log_maximum", g.max())):
                rid = f"{panel}:{stat}:{allocation}:{name}:{CAL}"
                d = pd.Series(scale)
                normalized = target - np.log(d)
                raw, adjusted = float(target.std(ddof=1)), float(normalized.std(ddof=1))
                meta = dict(dispersion_id=rid, panel=panel, statistic=stat, calibration=CAL,
                    allocation=allocation, summary=name, estimation_streams_per_checkpoint=n,
                    evaluation_streams_per_checkpoint=n, quantile_probability=Q,
                    quantile_method="linear_on_raw_maxima")
                rows.append(dict(meta, checkpoints=20, raw_sd=raw, normalized_sd=adjusted,
                    change=adjusted / raw - 1, uncertainty="descriptive; no population interval"))
                changes = []
                for seed in target.index:
                    pairs.append(dict(meta, checkpoint_seed=seed, raw_summary=target.loc[seed],
                        divisor=d.loc[seed], log_divisor=np.log(d.loc[seed]),
                        normalized_summary=normalized.loc[seed],
                        divisor_id=f"{panel}:{stat}:{CAL}:{seed}:allocation_{allocation}"))
                    raw_del = float(target.drop(seed).std(ddof=1))
                    adjusted_del = float(normalized.drop(seed).std(ddof=1))
                    change = adjusted_del / raw_del - 1
                    changes.append(change)
                    deletions.append(dict(meta, omitted_checkpoint=seed, checkpoints=19,
                        raw_sd=raw_del, normalized_sd=adjusted_del, change=change,
                        uncertainty="finite_panel_deletion_sensitivity; paired summary/divisor retained"))
                ranges.append(dict(meta, full_change=adjusted / raw - 1,
                                   minimum=min(changes), maximum=max(changes)))
    assert len(rows) == 78 and len(deletions) == 1560
    for name, values in [("dispersion", rows), ("dispersion_pairs", pairs),
                         ("dispersion_membership", members), ("dispersion_leave_one_checkpoint_out", deletions),
                         ("dispersion_sensitivity_ranges", ranges)]:
        r4.write(values, out / (name + ".csv"))


def run(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    f, _, groups = r4.load()
    print("Fitting fixed q=.90 stable-only divisors", flush=True)
    d, members, assignments, levels = fit(f[f.stable])
    points, stops, checkpoints = build_points(f, d, levels)
    for name, rows in [("divisors", d), ("divisor_membership", members),
                       ("calibration_assignments", assignments), ("operating_points", points),
                       ("stopping_times", stops), ("checkpoint_results", checkpoints)]:
        r4.write(rows, out / (name + ".csv"))
    mechanism(points, stops, d, assignments, levels, out)
    print("Reconstructing six quantile cut-search diagnostics", flush=True)
    r4.cuts(points, stops, groups, out)
    # The historical helper derives source IDs from names; new source IDs denote comparison anchors.
    anchors = points.set_index("point_id").source_point_id.to_dict()
    for name in ("cut_level_diagnostics", "cut_alarm_diagnostics", "monitoring_only_diagnostics", "cut_summary"):
        frame = r4.read(out / (name + ".csv"))
        frame["source_point_id"] = frame.point_id.map(anchors)
        frame["source_point_role"] = "comparison_anchor"
        frame["target_budget_policy"] = POLICY6
        r4.write(frame, out / (name + ".csv"))
    print("Generating 78 dispersion rows and 1560 paired deletions", flush=True)
    dispersion(f, out)
    inputs = [Path(__file__), HERE / "QUANTILE_PROTOCOL.md", HERE / "analyze.py",
              r4.R3 / "analyze.py", ROOT / "src/rl_hanbat/socis/sequential.py"]
    inputs += sorted((HERE / "results").glob("*.csv"))
    inputs += sorted((r4.R3 / "results").glob("*.csv"))
    inputs += [ROOT / f"experiments/stable_holdout/panel_{p}/holdout_streams.csv" for p in ("d", "e")]
    manifest = dict(revision=r4.ROUND, status="retrospective_fixed_quantile_sensitivity",
        quantile_probability=Q, quantile_method="linear_on_raw_maxima",
        operating_points=24, matched_points=12, additional_points=12, dispersion_rows=78,
        inputs={str(p.relative_to(ROOT)): r4.old.sha(p) for p in sorted(set(inputs))},
        outputs={p.name: r4.old.sha(p) for p in sorted(out.glob("*.csv"))})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(points[["point_id", "frozen_alarms", "updating_alarms", "primary_detections", "primary_mean_delay"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output.resolve())
