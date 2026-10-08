"""Independent quantile audit. No import of production analysis or sequential code."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PRIMARY = "gradual_updating_down"


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(a, b, message: str) -> None:
    require(np.allclose(a, b, rtol=2e-12, atol=2e-12, equal_nan=True), message)


def percentile(values: list[float]) -> float:
    require(bool(values), "nonempty quantile")
    require(all(math.isfinite(x) for x in values), "finite quantile input")
    s = sorted(values)
    h = (len(s) - 1) * .90
    lower = math.floor(h)
    return s[lower] + (h - lower) * (s[min(lower + 1, len(s) - 1)] - s[lower])


def stopping(path, threshold: float) -> tuple[bool, int]:
    for t, value in enumerate(path, 1):
        if value > threshold:
            return True, t
    return False, 81


def summarize(point, rows: pd.DataFrame) -> None:
    primary = rows[rows.condition == PRIMARY]
    changed = rows[~rows.stable]
    require(point.primary_denominator == len(primary), "primary denominator")
    require(point.primary_detections == primary.fired.sum(), "primary detections")
    require(point.changed_denominator == len(changed), "changed denominator")
    require(point.changed_detections == changed.fired.sum(), "changed detections")
    require(point.primary_delay_sum == primary.restricted_delay.sum(), "primary delay sum")
    close(point.primary_mean_delay, primary.restricted_delay.mean(), "primary mean delay")
    for mode in ("frozen", "updating"):
        g = rows[rows.stable & rows["mode"].eq(mode)]
        require(getattr(point, mode + "_denominator") == len(g), "stable denominator")
        require(getattr(point, mode + "_alarms") == g.fired.sum(), "stable alarm counts")
    require(point.actual_total_alarms == rows.loc[rows.stable, "fired"].sum(), "total alarms")


def audit(results: Path, skip_cuts: bool = False) -> dict:
    windows = read(HERE / "results/windows.csv")
    streams = read(HERE / "results/streams.csv")
    paths = {(sid, stat): g.loc[g.window > 40, stat].to_numpy()
             for sid, g in windows.groupby("stream_id", sort=False)
             for stat in ("GLR", "cut_search")}
    stable = streams[streams.stable & streams.stream_id.isin(windows.stream_id)]
    meta = streams.set_index("stream_id")
    maxima = {(sid, stat): float(max(path)) for (sid, stat), path in paths.items()}
    require(all(math.isfinite(x) and x > 0 for x in maxima.values()), "positive finite maxima")
    divisors = read(results / "divisors.csv").set_index("divisor_id")
    members = read(results / "divisor_membership.csv")
    require(len(divisors) == 400 and len(members) == 2560, "divisor coverage")
    for identity, row in divisors.iterrows():
        group = stable[(stable.panel == row.panel) & (stable.checkpoint_seed == row.checkpoint_seed)]
        if row.excluded_replicate != -1:
            group = group[group.replicate != row.excluded_replicate]
        exported = members[members.divisor_id == identity]
        n = 8 if row.excluded_replicate == -1 else 6
        require(len(group) == len(exported) == row.estimation_streams == n, "exact target budget")
        require(set(exported.stream_id) == set(group.stream_id), "exact target membership")
        vals = [maxima[sid, row.statistic] for sid in group.stream_id]
        close(percentile(vals), row.divisor, "independent interpolated divisor")
        for r in exported.itertuples():
            require(r.stable_maximum == maxima[r.stream_id, row.statistic], "member maximum")
        close(max(vals) / row.divisor, row.maximum_to_divisor, "within-estimation ratio")
        require(row.maximum_to_divisor <= (10 / 3 if n == 8 else 2) * (1 + 1e-14), "quantile structural bound")
        require(row.quantile_probability == .9 and row.quantile_method == "linear_on_raw_maxima", "fixed quantile definition")
    assignments = read(results / "calibration_assignments.csv")
    require(len(assignments) == 12160, "foreign assignment coverage")
    levels, expected_donors = {}, set()
    for key, g in assignments.groupby(["panel", "statistic", "target_checkpoint"]):
        panel, stat, seed = key
        expected = stable[(stable.panel == panel) & (stable.checkpoint_seed != seed)]
        require(len(g) == 152 and set(g.foreign_stream_id) == set(expected.stream_id), "all 152 foreign records")
        for r in g.itertuples():
            require(r.foreign_checkpoint == meta.loc[r.foreign_stream_id, "checkpoint_seed"], "foreign checkpoint")
            d = divisors.loc[f"{panel}:{stat}:quantile90:{r.foreign_checkpoint}:exclude_-1", "divisor"]
            require(r.maximum == maxima[r.foreign_stream_id, stat] and r.foreign_divisor == d, "foreign input identity")
            require(r.normalized_maximum == r.maximum / d, "foreign normalization")
        levels[key] = g.normalized_maximum.max()
        expected_donors.update((*key, sid) for sid in g.loc[g.normalized_maximum == levels[key], "foreign_stream_id"])
    donors = read(results / "foreign_extreme_donors.csv")
    require({(r.panel, r.statistic, r.target_checkpoint, r.foreign_stream_id) for r in donors.itertuples()} == expected_donors,
            "all foreign extreme donors including ties")
    for _, g in donors.groupby(["panel", "statistic", "target_checkpoint"]):
        require(g.donor_tie_count.eq(len(g)).all(), "donor tie multiplicity")
    points = read(results / "operating_points.csv")
    stops = read(results / "stopping_times.csv")
    checkpoint = read(results / "checkpoint_results.csv")
    require(len(points) == 24 and points.point_id.is_unique and len(stops) == 11520, "24 complete points")
    require(points.family.value_counts().to_dict() == {"matched_target_budget": 12, "additional_stable": 12}, "point families")
    require(not stops.duplicated(["point_id", "stream_id"]).any(), "unique stream decisions")
    old_points = pd.concat([read(HERE / "results/operating_points.csv"), read(HERE / "results/matched_operating_points.csv")]).set_index("point_id")
    old_stops = pd.concat([read(HERE / "results/stopping_times.csv"), read(HERE / "results/matched_stopping_times.csv")])
    additional = {}
    for panel in ("D", "E"):
        h = read(ROOT / f"experiments/stable_holdout/panel_{panel.lower()}/holdout_streams.csv")
        additional[panel] = h.set_index("stream_id")
    for p in points.itertuples():
        g = stops[stops.point_id == p.point_id]
        is_six = p.family == "matched_target_budget"
        require(len(g) == 480 and p.source_point_role == "comparison_anchor", "point interface")
        require(p.calibration == "quantile90" and p.quantile_probability == .9 and p.quantile_method == "linear_on_raw_maxima", "point quantile specification")
        require(g.source_point_id.eq(p.source_point_id).all() and g.source_point_role.eq("comparison_anchor").all(), "decision anchor links")
        require(g.foreign_calibration_streams.eq(152).all(), "decision foreign counts")
        require(p.source_point_id in old_points.index, "existing comparison anchor")
        expected_policy = "replicate_excluded_six_both_modes" if is_six else "eight_original_stable"
        require(p.target_budget_policy == expected_policy and g.target_budget_policy.eq(expected_policy).all(), "target budget policy")
        expected_comparators = old_points[(old_points.panel == p.panel) & (old_points.statistic == p.statistic) &
                                          (old_points.family == p.family) & (old_points.budget == p.budget)]
        require(set(json.loads(p.comparator_point_ids)) == set(expected_comparators.index), "all specified comparators")
        expected_streams = set(streams[(streams.panel == p.panel) & streams.stream_id.isin(windows.stream_id)].stream_id)
        if not is_six:
            expected_streams = {sid for sid in expected_streams if not bool(meta.loc[sid, "stable"])} | set(additional[p.panel].index)
        require(set(g.stream_id) == expected_streams, "exact evaluation membership")
        for r in g.itertuples():
            rep = r.replicate if is_six else -1
            identity = f"{p.panel}:{p.statistic}:quantile90:{r.checkpoint_seed}:exclude_{rep}"
            require(r.divisor_id == identity and r.target_estimation_streams == (6 if is_six else 8), "decision divisor identity")
            base = levels[p.panel, p.statistic, r.checkpoint_seed] * divisors.loc[identity, "divisor"]
            require(r.base_threshold == base and r.threshold == base * p.multiplier, "exact threshold products")
            if (r.stream_id, p.statistic) in paths:
                fired, delay = stopping(paths[r.stream_id, p.statistic], r.threshold)
                require(r.stopping_time_available and r.maximum == maxima[r.stream_id, p.statistic], "path maximum identity")
                require(r.fired == fired and r.restricted_delay == delay, "strict original first crossing")
                require(r.first_crossing == delay if fired else pd.isna(r.first_crossing), "first crossing field")
            else:
                maximum = additional[p.panel].loc[r.stream_id, "max_statistic"]
                require(r.maximum == maximum and r.fired == (maximum > r.threshold), "additional maximum decision")
                require(not r.stopping_time_available and pd.isna(r.restricted_delay) and pd.isna(r.first_crossing), "no invented additional delay")
        tuning = g[g.stable]
        ratios = tuning.maximum.to_numpy() / tuning.base_threshold.to_numpy()
        if is_six:
            caps = (int(p.frozen_cap), int(p.updating_cap))
            candidate = max(sorted(ratios[tuning["mode"].eq(mode)], reverse=True)[cap]
                            for mode, cap in zip(("frozen", "updating"), caps, strict=True))
        else:
            candidate = sorted(ratios, reverse=True)[int(p.total_cap)]
        require(candidate == p.ratio_candidate, "independent stable ratio candidate")
        multiplier, increments = candidate, 0
        while True:
            fired = tuning.maximum.to_numpy() > tuning.base_threshold.to_numpy() * multiplier
            counts = [int(fired[tuning["mode"].eq(mode)].sum()) for mode in ("frozen", "updating")]
            feasible = all(x <= y for x, y in zip(counts, caps, strict=True)) if is_six else sum(counts) <= int(p.total_cap)
            if feasible:
                break
            multiplier = np.nextafter(multiplier, np.inf)
            increments += 1
            require(increments <= 32, "strict threshold boundary limit")
        require(p.multiplier == multiplier and p.boundary_ulp_increments == increments, "stable-only multiplier selection")
        summarize(p, g)
        for seed, cg in g.groupby("checkpoint_seed"):
            cp = checkpoint[(checkpoint.point_id == p.point_id) & (checkpoint.checkpoint_seed == seed)]
            require(len(cp) == 1, "checkpoint coverage")
            summarize(next(cp.itertuples()), cg)
    all_decisions = pd.concat([old_stops, stops]).set_index(["point_id", "stream_id"])
    allocation = read(results / "threshold_allocation.csv")
    old_d = read(HERE / "results/divisors.csv").set_index("divisor_id").divisor
    old_levels = read(HERE / "results/thresholds.csv").groupby(
        ["panel", "statistic", "calibration", "checkpoint_seed"]).foreign_level.first()
    qpoint = points.set_index("point_id")
    for r in allocation.itertuples():
        q, o = qpoint.loc[r.quantile_point_id], old_points.loc[r.comparator_point_id]
        qd, od = divisors.loc[r.quantile_divisor_id, "divisor"], old_d.loc[r.comparator_divisor_id]
        qb, ob = levels[r.panel, r.statistic, r.checkpoint_seed], old_levels.loc[(r.panel, r.statistic, r.comparator, r.checkpoint_seed)]
        close([r.quantile_divisor, r.comparator_divisor, r.quantile_foreign_extreme, r.comparator_foreign_extreme],
              [qd, od, qb, ob], "decomposition inputs")
        require(r.quantile_multiplier == q.multiplier and r.comparator_multiplier == o.multiplier, "decomposition multipliers")
        close([r.quantile_threshold, r.comparator_threshold], [(qb * qd) * q.multiplier, (ob * od) * o.multiplier], "allocation thresholds")
        ratio = r.quantile_threshold / r.comparator_threshold
        close([r.threshold_ratio, r.decomposed_ratio], [ratio, (q.multiplier / o.multiplier) * (qd / od) * (qb / ob)], "threshold ratio identity")
    require(len(allocation) == 3360, "all threshold comparison configurations")
    rec = read(results / "detection_delay_reconciliation.csv")
    require(len(rec) == 28800, "all 60 paired decision comparisons")
    for r in rec.itertuples():
        q = all_decisions.loc[r.quantile_point_id, r.stream_id]
        o = all_decisions.loc[r.comparator_point_id, r.stream_id]
        require(r.quantile_threshold == q.threshold and r.comparator_threshold == o.threshold, "reconciliation thresholds")
        require(r.maximum == q.maximum == o.maximum, "reconciliation maximum")
        require(r.quantile_fired == q.fired and r.comparator_fired == o.fired, "reconciliation decisions")
        require(r.quantile_divisor_id == q.divisor_id, "reconciliation divisor")
        close([r.quantile_delay, r.comparator_delay], [q.restricted_delay, o.restricted_delay], "reconciliation exported delays")
        require(r.quantile_gained == bool(q.fired and not o.fired) and r.quantile_lost == bool(o.fired and not q.fired), "gains and losses")
        require(r.maximum_between_thresholds == (min(q.threshold, o.threshold) < q.maximum <= max(q.threshold, o.threshold)), "interval reconciliation")
        if r.stopping_time_available:
            require(r.delay_difference == q.restricted_delay - o.restricted_delay, "delay reconciliation")
        else:
            require(pd.isna(r.delay_difference), "missing path remains missing")
    contrasts = read(results / "checkpoint_contrasts.csv")
    require(len(contrasts) == 1200, "all checkpoint pair contrasts")
    for r in contrasts.itertuples():
        g = rec[(rec.quantile_point_id == r.quantile_point_id) &
                (rec.comparator_point_id == r.comparator_point_id) &
                (rec.checkpoint_seed == r.checkpoint_seed) & (rec.condition == PRIMARY)]
        require(len(g) == r.primary_denominator == 4, "contrast denominator")
        require(r.detection_difference == g.quantile_gained.sum() - g.quantile_lost.sum(), "checkpoint detection contrast")
        close(r.delay_difference, g.delay_difference.mean(), "checkpoint delay contrast")
    report = dict(divisors=len(divisors), foreign_assignments=len(assignments), extreme_donors=len(donors),
                  points=len(points), decisions=len(stops), checkpoint_results=len(checkpoint),
                  threshold_decompositions=len(allocation), paired_decisions=len(rec), checkpoint_contrasts=len(contrasts))
    audit_dispersion(results, stable, maxima, additional, report)
    if not skip_cuts:
        audit_cuts(results, windows, all_decisions, report)
    report["status"] = "PASS"
    return report


def audit_dispersion(results: Path, stable: pd.DataFrame, maxima: dict, additional: dict, report: dict) -> None:
    dispersion = read(results / "dispersion.csv")
    pairs = read(results / "dispersion_pairs.csv")
    members = read(results / "dispersion_membership.csv")
    deletion = read(results / "dispersion_leave_one_checkpoint_out.csv")
    ranges = read(results / "dispersion_sensitivity_ranges.csv").set_index("dispersion_id")
    require(len(dispersion) == 78 and len(deletion) == len(pairs) == 1560, "dispersion and deletion coverage")
    for r in dispersion.itertuples():
        panel = stable[stable.panel == r.panel]
        n = r.estimation_streams_per_checkpoint
        reps = None if n == 8 else list(map(int, r.allocation.split(",")))
        targets, normalized, seeds = [], [], sorted(panel.checkpoint_seed.unique())
        for seed in seeds:
            group = panel[panel.checkpoint_seed == seed]
            est = group if n == 8 else group[group.replicate.isin(reps)]
            eval_sids = set(additional[r.panel][additional[r.panel].checkpoint_seed == seed].index) if n == 8 else set(group[~group.replicate.isin(reps)].stream_id)
            require(set(est.stream_id).isdisjoint(eval_sids), "disjoint dispersion membership")
            vals = [maxima[sid, r.statistic] for sid in est.stream_id]
            d = percentile(vals)
            raw_values = [float(additional[r.panel].loc[sid, "max_statistic"]) if n == 8 else maxima[sid, r.statistic] for sid in eval_sids]
            logvalues = [math.log(x) for x in raw_values]
            target = (math.fsum(logvalues) / len(logvalues) if r.summary == "mean_log_maximum" else
                      percentile(logvalues) if r.summary == "q90_log_maximum" else max(logvalues))
            identity = f"{r.panel}:{r.statistic}:quantile90:{seed}:allocation_{r.allocation}"
            mm = members[members.divisor_id == identity]
            require(len(mm) == n and set(mm.stream_id) == set(est.stream_id), "dispersion scale members")
            close(mm.divisor, d, "dispersion raw quantile")
            for m in mm.itertuples():
                require(m.stable_maximum == maxima[m.stream_id, r.statistic], "dispersion scale value")
            pair = pairs[(pairs.dispersion_id == r.dispersion_id) & (pairs.checkpoint_seed == seed)]
            require(len(pair) == 1 and pair.divisor_id.iloc[0] == identity, "paired divisor identity")
            close([pair.raw_summary.iloc[0], pair.divisor.iloc[0], pair.normalized_summary.iloc[0]],
                  [target, d, target - math.log(d)], "paired summary")
            targets.append(target)
            normalized.append(target - math.log(d))
        targets, normalized = np.array(targets), np.array(normalized)
        close([r.raw_sd, r.normalized_sd, r.change], [targets.std(ddof=1), normalized.std(ddof=1),
              normalized.std(ddof=1) / targets.std(ddof=1) - 1], "independent dispersion")
        changes = []
        for seed in seeds:
            keep = np.array(seeds) != seed
            change = normalized[keep].std(ddof=1) / targets[keep].std(ddof=1) - 1
            row = deletion[(deletion.dispersion_id == r.dispersion_id) & (deletion.omitted_checkpoint == seed)]
            require(len(row) == 1 and row.checkpoints.iloc[0] == 19, "one paired deletion")
            close(row.change.iloc[0], change, "paired deletion without refitting")
            changes.append(change)
        rr = ranges.loc[r.dispersion_id]
        close([rr.full_change, rr.minimum, rr.maximum], [r.change, min(changes), max(changes)], "deletion range")
    report.update(dispersion_rows=len(dispersion), paired_deletion_rows=len(deletion))


def audit_cuts(results: Path, windows: pd.DataFrame, all_decisions: pd.DataFrame, report: dict) -> None:
    # Independent eligible-index mapping and compensated left/right means.
    normal=NormalDist();eligible={};monitor={};max_cut_error=0.
    for sid,w in windows.groupby('stream_id',sort=False):
        e=w[w.eligible]
        eligible[sid]=(e.window.to_numpy(dtype=int),np.array([normal.cdf(float(x)) for x in e.z]))
        observed=[];path=[]
        for r in w[w.window>40].itertuples():
            if r.eligible:observed.append(normal.cdf(float(r.z)))
            if not r.eligible:
                path.append(path[-1] if path else 0.);continue
            n=len(observed);prefix=np.cumsum([0.]+observed);evidence=0.
            for k in range(5,n-4):
                difference=prefix[k]/k-(prefix[n]-prefix[k])/(n-k)
                value=abs(difference)/math.sqrt(.5*math.log(40)*(1/k+1/(n-k)))
                evidence=max(evidence,value)
            path.append(evidence)
        monitor[sid]=np.array(path)
    cuts=read(results/'cut_level_diagnostics.csv')
    alarm=read(results/'cut_alarm_diagnostics.csv').set_index(['point_id','stream_id'])
    cut_moments={}
    for r in cuts.itertuples():
        key=(r.stream_id,r.alarm_original_window,r.split_eligible_index)
        if key not in cut_moments:
            loc,x=eligible[r.stream_id];mask=loc<=r.alarm_original_window;loc,x=loc[mask],x[mask]
            k=r.split_eligible_index;n=len(x)
            require(5<=k<=n-5,'five-observation eligibility')
            left=math.fsum(x[:k])/k;right=math.fsum(x[k:])/(n-k)
            cut_moments[key]=(int(loc[k-1]),int(loc[k]),n,left,right)
        last,next_window,n,left,right=cut_moments[key]
        require(r.left_last_window==last and r.right_first_window==next_window and r.eligible_count==n,'cut original windows')
        require(r.boundary_position==('before' if last<40 else ('at' if last==40 else 'after')),'warmup boundary class')
        close([r.left_mean,r.right_mean,r.signed_difference],[left,right,right-left],'cut signed means')
        evidence=abs(right-left)/math.sqrt(.5*math.log(40)*(1/r.split_eligible_index+1/(n-r.split_eligible_index)))
        max_cut_error=max(max_cut_error,abs(evidence-r.evidence));close(evidence,r.evidence,'cut evidence')
        require(r.exceeds_threshold==(r.evidence>r.threshold),'strict cut exceedance')
        require(r.alarm_original_window==r.first_crossing+40,'alarm original index')
    for (pid,sid),g in cuts.groupby(['point_id','stream_id'],sort=False):
        a=alarm.loc[(pid,sid)];x=all_decisions.loc[(pid,sid)]
        peak=g.evidence.max();w=g.evidence.eq(peak);above=g[g.exceeds_threshold]
        require(g.winning_cut.equals(w) and g.winning_cut_ties.eq(int(w.sum())).all(),'winning cut ties')
        require(len(g)==g.eligible_count.iloc[0]-9 and g.split_eligible_index.tolist()==list(range(5,int(g.eligible_count.iloc[0])-4)),'all eligible cuts')
        require(a.first_crossing==x.first_crossing and bool(x.fired),'accepted cut first alarm')
        require(a.exceeding_cuts==len(above),'all exceeding cuts')
        inc=(above.signed_difference>0).any();dec=(above.signed_difference<0).any()
        require(a.supported_direction==('both' if inc and dec else ('increase' if inc else 'decrease')),'alarm direction')
        for position in ('before','at','after'):
            require(a['exceeding_'+position]==above.boundary_position.eq(position).sum(),'boundary alarm count')
    monitor_diag=read(results/'monitoring_only_diagnostics.csv');boundary=[]
    for r in monitor_diag.itertuples():
        fired,delay=stopping(monitor[r.stream_id],r.threshold)
        if fired!=r.monitoring_only_fired or delay!=r.monitoring_only_delay:
            gap=float(np.min(np.abs(monitor[r.stream_id]-r.threshold)))
            require(gap<=2e-12,'monitoring-only independent mismatch beyond rounding')
            boundary.append(dict(point_id=r.point_id,stream_id=r.stream_id,minimum_gap=gap))
    summaries=read(results/'cut_summary.csv')
    for r in summaries.itertuples():
        d=monitor_diag[monitor_diag.point_id==r.point_id]
        primary=d[d.condition==PRIMARY]
        require(r.primary_full_history==primary.full_history_fired.sum(),'cut summary full count')
        require(r.primary_monitoring_only==primary.monitoring_only_fired.sum(),'cut summary diagnostic count')
        close([r.full_delay,r.monitoring_only_delay],[primary.full_history_delay.mean(),primary.monitoring_only_delay.mean()],'cut summary delays')
        alarms=alarm.loc[r.point_id];prim=alarms[alarms.condition==PRIMARY]
        for direction in ('increase','decrease','both'):
            require(getattr(r,direction)==prim.supported_direction.eq(direction).sum(),'cut direction summary')
        for mode in ('frozen','updating'):
            require(getattr(r,'monitoring_only_'+mode+'_alarms')==d.loc[d.stable & d['mode'].eq(mode),'monitoring_only_fired'].sum(),'cut diagnostic stable count')
    report.update(cut_records=len(cuts),cut_alarm_records=len(alarm),maximum_independent_cut_error=max_cut_error,
                  monitoring_only_records=len(monitor_diag),monitoring_only_rounding_boundary_cases=boundary)
    require(len(monitor_diag)==2880 and len(summaries)==6,'six complete quantile cut points')
    require(not boundary,'monitoring-only quantile crossing parity')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "fresh audit output required")
    result = audit(args.results)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
