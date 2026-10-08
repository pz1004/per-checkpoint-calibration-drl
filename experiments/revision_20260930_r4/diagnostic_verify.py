"""Independent audit of the table interface and every emitted detector/stream.

Does not import the diagnostic implementation or historical analysis functions.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision='round_trip', keep_default_na=False, low_memory=False)


def check(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def audit(manifest: Path, output: Path) -> dict:
    spec = json.loads(manifest.read_text())
    inputs = {k: read(manifest.parent / v) for k, v in spec['tables'].items()}
    si = inputs['streams'].set_index('stream_key')
    di = inputs['divisors'].set_index('divisor_id')
    ps = inputs['points'].set_index('source_point_id')
    members = inputs['memberships'].groupby('divisor_id').stream_key.agg(list)
    for key, d in di.iterrows():
        values = si.loc[members[key], 'maximum'].to_numpy()
        check(len(values) == d.membership_count, 'membership count')
        check(all(si.loc[members[key], 'role'] == 'original_stable'), 'divisor role')
        expected = {'pooled': lambda: 1., 'arithmetic': lambda: float(sum(values) / len(values)),
                    'geometric': lambda: float(np.exp(sum(np.log(values)) / len(values))),
                    'quantile90': lambda: float(np.interp(.9 * (len(values) - 1), np.arange(len(values)), sorted(values)))}[d.calibration]()
        check(np.isclose(expected, d.divisor, rtol=3e-15, atol=0), 'divisor recomputation')
    path_table = read(manifest.parent / spec['indexed_paths'])
    paths = {k: pd.to_numeric(g.sort_values('index').value, errors='coerce').to_numpy()
             for k, g in path_table.groupby('stream_key', sort=False)}
    points = read(output / 'operating_points.csv').set_index('point_id', drop=False)
    condition = {k: g for k, g in read(output / 'condition_summary.csv').groupby('point_id', sort=False)}
    constraints = {k: g for k, g in read(output / 'tuning_constraints.csv').groupby('point_id', sort=False)}
    tuning = {k: g for k, g in read(output / 'tuning_streams.csv').groupby('point_id', sort=False)}
    donors = {k: g for k, g in read(output / 'donor_records.csv').groupby('point_id', sort=False)}
    parameters = {k: g for k, g in read(output / 'checkpoint_parameters.csv').groupby('point_id', sort=False)}
    calibration = {k: g for k, g in read(output / 'calibration_records.csv').groupby('point_id', sort=False)}
    target = {k: g.set_index('stream_key') for k, g in inputs['targets'].groupby('source_point_id', sort=False)}
    check(len(points) == 1100, '1100 detector evaluations')
    check(points.result_kind.value_counts().to_dict() == {'deletion': 880, 'existing': 136, 'control': 84}, 'point accounting')
    check(points[points.result_kind == 'deletion'].scenario_id.nunique() == 440, '440 deletion scenarios')
    check(int(((points.result_kind == 'existing') & (points.selection_type == 'stable_only_retuned')).sum()) == 120, '120 existing retuned')
    full, totals = {}, dict(detectors=0, decisions=0, ranks=0, donor_rows=0, condition_rows=0,
                           coordinate_decision_differences=0, deletion_scenarios=440, preserved_existing_points=136)
    with (output / 'stream_decisions.csv').open(newline='') as handle:
        for pid, rows in itertools.groupby(csv.DictReader(handle), key=lambda r: r['point_id']):
            rows = list(rows)
            p, frame = points.loc[pid], pd.DataFrame(rows)
            source = ps.loc[p.source_point_id]
            omitted = None if p.omitted_checkpoint == '' else int(p.omitted_checkpoint)
            ids = frame.stream_key.tolist()
            actual_sources = target[p.source_point_id]
            expected_ids = [k for k in actual_sources.index if si.loc[k, 'checkpoint_seed'] != omitted]
            check(set(ids) == set(expected_ids) and len(ids) == len(set(ids)), 'evaluation coverage ' + pid)
            rec = si.loc[ids]
            mask = (rec.role == source.tuning_role).to_numpy()
            cal_input = inputs['calibration']
            cal_input = cal_input[(cal_input.panel == p.panel) & (cal_input.statistic == p.statistic) & (cal_input.calibration == p.calibration)]
            cal_ids = [k for k in cal_input.stream_key if si.loc[k, 'checkpoint_seed'] != omitted]
            check(set(calibration[pid].stream_key) == set(cal_ids), 'calibration coverage')
            c = si.loc[cal_ids]
            cal_div = cal_input.set_index('stream_key').loc[cal_ids, 'divisor_id']
            norm = c.maximum.to_numpy() / di.loc[cal_div, 'divisor'].to_numpy()
            B = float(max(norm))
            check(B == p.global_level, 'global B')
            expected_donors = set()
            for seed in rec.checkpoint_seed.unique():
                foreign = c.checkpoint_seed.to_numpy() != seed
                Bminus = float(max(norm[foreign]))
                par = parameters[pid].set_index('checkpoint_seed').loc[seed]
                check(par.global_level == B and par.foreign_level == Bminus and par.loco_correction == Bminus / B, 'B/b parameters')
                check(par.checkpoint_ratio == max(norm[c.checkpoint_seed == seed]), 'checkpoint R')
                check(par.foreign_calibration_records == (144 if omitted is not None else 152), 'foreign count')
                check(par.global_donor_streams == int((norm == B).sum()) and par.global_donor_checkpoints == c.loc[norm == B, 'checkpoint_seed'].nunique(), 'global tie count')
                for dtype, idx in (('global', norm == B), ('foreign', foreign & (norm == Bminus))):
                    for key in c.index[idx]:
                        expected_donors.add((seed, dtype, key))
            actual_donors = set(zip(donors[pid].target_checkpoint, donors[pid].donor_type, donors[pid].stream_key))
            check(expected_donors == actual_donors and len(actual_donors) == len(donors[pid]), 'all donor ties')
            for (_, _), dg in donors[pid].groupby(['target_checkpoint', 'donor_type']):
                check(all(dg.tie_stream_count == len(dg)) and all(dg.tie_checkpoint_count == dg.checkpoint_seed.nunique()), 'donor tie metadata')
            div_ids = actual_sources.loc[ids, 'divisor_id']
            check(frame.divisor_id.tolist() == div_ids.tolist(), 'target divisor IDs')
            d = di.loc[div_ids, 'divisor'].to_numpy()
            foreign = np.array([max(norm[c.checkpoint_seed.to_numpy() != seed]) for seed in rec.checkpoint_seed])
            base = d * foreign if p.control_type == 'loco' else d
            M, mode = rec.maximum.to_numpy(), rec['mode'].to_numpy()
            caps = json.loads(source.caps_json)
            if p.selection_type == 'fixed_multiplier':
                check(p.multiplier == source.multiplier == 1, 'native fixed multiplier')
            else:
                ratios = M[mask] / base[mask]
                cmode = mode[mask]
                cuts = {m: sorted(ratios if m == 'total' else ratios[cmode == m], reverse=True)[cap] for m, cap in caps.items()}
                candidate, steps = max(cuts.values()), 0
                k = candidate
                while any(int(((M[mask] > base[mask] * k) & (True if m == 'total' else cmode == m)).sum()) > cap for m, cap in caps.items()):
                    k = np.nextafter(k, np.inf)
                    steps += 1
                    check(steps <= 32, 'selector termination')
                check(k == p.multiplier and candidate == float(p.ratio_candidate) and steps == p.boundary_ulp_increments, 'exact selector')
                tr = tuning[pid].set_index('stream_key').loc[rec.index[mask]]
                check(np.array_equal(tr.stable_ratio.to_numpy(), ratios), 'stable ratios')
                for i, (_, row) in enumerate(tr.iterrows()):
                    local = ratios[cmode == cmode[i]]
                    check(row.rank_total_min == 1 + int((ratios > ratios[i]).sum()), 'total min rank')
                    check(row.rank_total_max == int((ratios >= ratios[i]).sum()), 'total max rank')
                    check(row.rank_mode_min == 1 + int((local > ratios[i]).sum()), 'mode min rank')
                    check(row.rank_mode_max == int((local >= ratios[i]).sum()), 'mode max rank')
                    check(row.selected_boundary == (ratios[i] == candidate), 'boundary tie')
                    cap = caps.get(cmode[i], caps.get('total'))
                    cutoff = sorted(local, reverse=True)[cap]
                    check(row.mode_candidate_cutoff == cutoff and row.mode_boundary == (ratios[i] == cutoff), 'per-mode candidate')
                for con in constraints[pid].itertuples():
                    check(con.cap == caps[con.constraint] and con.candidate_cutoff == cuts[con.constraint], 'constraint cutoff')
                    check(con.binding == (con.candidate_cutoff == candidate), 'binding constraint')
                    cmask = np.ones(len(ratios), dtype=bool) if con.constraint == 'total' else cmode == con.constraint
                    check(con.denominator == int(cmask.sum()), 'constraint denominator')
                    check(con.initial_alarms == int(((M[mask] > base[mask] * candidate) & cmask).sum()), 'candidate alarms')
                    check(con.achieved_alarms == int(((M[mask] > base[mask] * k) & cmask).sum()), 'achieved caps')
                    check(con.achieved_alarms <= con.cap, 'cap feasibility')
                    check(con.boundary_ties == int(((ratios == con.candidate_cutoff) & cmask).sum()), 'constraint ties')
                totals['ranks'] += len(ratios)
            h = base * p.multiplier
            check(np.array_equal(frame.threshold.astype(float).to_numpy(), h), 'exact thresholds')
            theta = p.multiplier * B if p.control_type == 'loco' else p.multiplier
            coordinate = theta * d * (foreign / B if p.control_type == 'loco' else 1.)
            check(theta == p.theta, 'theta')
            check(np.allclose(h, coordinate, rtol=8e-16, atol=0), 'reparameterization')
            fired = M > h
            for mode_name in ('frozen', 'updating'):
                mode_mask = mask & (mode == mode_name)
                check(p[mode_name + '_alarms'] == int(fired[mode_mask].sum()) and p[mode_name + '_denominator'] == int(mode_mask.sum()), 'operating point alarms/denominator')
            delays = []
            for key, threshold in zip(ids, h, strict=True):
                if key not in paths:
                    delays.append(np.nan)
                else:
                    crossings = [j + 1 for j, value in enumerate(paths[key]) if value > threshold]
                    delays.append(float(crossings[0] if crossings else spec['horizon'] + 1))
            delays = np.array(delays)
            check(np.array_equal(frame.fired.eq('True').to_numpy(), fired), 'detections')
            check(np.array_equal(pd.to_numeric(frame.restricted_delay, errors='coerce').to_numpy(), delays, equal_nan=True), 'restricted delay')
            check(np.array_equal(frame.stopping_time_available.eq('True').to_numpy(), np.isfinite(delays)), 'delay availability')
            first_crossing = pd.to_numeric(frame.first_crossing, errors='coerce').to_numpy()
            check(np.array_equal(first_crossing, np.where(fired, delays, np.nan), equal_nan=True), 'first crossings')
            coord_diff = (M > coordinate) != fired
            check(np.array_equal(frame.coordinate_decision_differs.eq('True'), coord_diff), 'floating coordinate decisions')
            totals['coordinate_decision_differences'] += int(coord_diff.sum())
            key = p.source_point_id, p.control_type
            if omitted is None:
                reference = (h, fired, delays) if p.control_type == 'loco' else full[p.source_point_id, 'loco'][1:]
            else:
                full_ids, fh, ff, fd = full[key]
                lookup = {k: i for i, k in enumerate(full_ids)}
                idx = [lookup[k] for k in ids]
                reference = fh[idx], ff[idx], fd[idx]
                check(int(mask.sum()) == 152 and all(int(((mode == m) & mask).sum()) == 76 for m in ('frozen', 'updating')), 'deletion stable denominator')
                check(all(int((rec.condition == cnd).sum()) == 76 for cnd in rec[rec.role == 'changed'].condition.unique()), 'deletion changed denominator')
            rh, rf, rd = reference
            check(np.array_equal(frame.reference_threshold.astype(float), rh), 'restricted reference threshold')
            check(np.array_equal(frame.reference_fired.eq('True'), rf), 'restricted reference fired')
            for row in condition[pid].itertuples():
                cm = rec.condition.to_numpy() == row.condition
                check(row.denominator == int(cm.sum()) and row.detections == int(fired[cm].sum()), 'condition count')
                check(row.detected_checkpoints == len(set(rec.checkpoint_seed.to_numpy()[cm & fired])), 'checkpoint coverage')
                check(row.gained == int((cm & fired & ~rf).sum()) and row.lost == int((cm & ~fired & rf).sum()), 'gains/losses')
                check(json.loads(row.gained_stream_ids) == sorted(rec.stream_id.to_numpy()[cm & fired & ~rf]), 'gained identities')
                check(json.loads(row.lost_stream_ids) == sorted(rec.stream_id.to_numpy()[cm & ~fired & rf]), 'lost identities')
                if np.isfinite(delays[cm]).any():
                    check(float(row.restricted_delay) == float(np.mean(delays[cm])), 'condition delay')
                else:
                    check(row.restricted_delay == '', 'missing stable delay stays missing')
            if omitted is None:
                full[key] = ids, h, fired, delays
            if p.result_kind == 'existing':
                hist = actual_sources.loc[ids]
                check(np.array_equal(h, hist.historical_threshold), 'preserved historical thresholds')
                check(np.array_equal(fired, hist.historical_fired), 'preserved historical decisions')
            totals['detectors'] += 1
            totals['decisions'] += len(rows)
            totals['donor_rows'] += len(donors[pid])
            totals['condition_rows'] += len(condition[pid])
    # Independent pair summaries for both unchanged and control families.
    for row in read(output / 'calibration_comparisons.csv').itertuples():
        a = full[row.source_point_id, row.control_type]
        reference_pid = row.reference_point_id.removesuffix(':scaling_only')
        b = full[reference_pid, row.control_type]
        lookup = {k: i for i, k in enumerate(b[0])}
        idx = [lookup[k] for k in a[0]]
        f, g = a[2], b[2][idx]
        rec = si.loc[a[0]]
        mask = rec.condition.to_numpy() == row.condition
        check(row.gained == int((mask & f & ~g).sum()) and row.lost == int((mask & ~f & g).sum()), 'calibration gains/losses')
    totals['status'] = 'PASS'
    return totals


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.manifest, args.results)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
