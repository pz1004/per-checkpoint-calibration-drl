"""Independent numerical and semantic checks; does not import the analysis code.

Reference paths use forward prefix sums and scipy's normal CDF. Exact decisions
are checked on the exported float paths; near-boundary differences from the
independent floating-point implementation are counted explicitly.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[2]
FILES = {
    'streams', 'windows', 'maxima', 'divisors', 'divisor_membership',
    'calibration_assignments', 'thresholds', 'operating_points',
    'stopping_times', 'checkpoint_results', 'dispersion', 'baseline_regression',
}
MODES = ('frozen', 'updating')
PRIMARY = 'gradual_updating_down'
CALIBRATIONS = ('pooled', 'arithmetic', 'geometric')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(actual: object, expected: object, message: str) -> None:
    require(bool(np.allclose(actual, expected, rtol=2e-12, atol=2e-12)), message)


def load(folder: Path) -> dict[str, pd.DataFrame]:
    require({p.stem for p in folder.glob('*.csv')} == FILES, 'exact analysis CSV membership')
    return {name: pd.read_csv(folder / f'{name}.csv', float_precision='round_trip') for name in FILES}


def reference_paths(z: list[float | None]) -> dict[str, np.ndarray]:
    """Independent direct prefix-sum formulations of the two stated statistics."""
    glr, cut, observed, monitoring = [], [], [], []
    previous_glr = previous_cut = 0.0
    for wall, value in enumerate(z):
        if value is not None:
            observed.append(float(ndtr(value)))
            n = len(observed)
            if n >= 10:
                prefix = np.cumsum(observed)
                k = np.arange(5, n - 4)
                differences = prefix[k - 1] / k - (prefix[-1] - prefix[k - 1]) / (n - k)
                bounds = np.sqrt(.5 * np.log(40.) * (1. / k + 1. / (n - k)))
                previous_cut = float(np.max(np.abs(differences) / bounds))
            if wall >= 40:
                monitoring.append(value)
                prefix = np.concatenate(([0.], np.cumsum(monitoring)))
                sums = prefix[-1] - prefix[:-1]
                lengths = np.arange(len(monitoring), 0, -1)
                previous_glr = float(np.max(np.maximum(sums, 0.) ** 2 / (2 * lengths)))
        if wall >= 40:
            glr.append(previous_glr)
            cut.append(previous_cut)
    return {'GLR': np.asarray(glr), 'cut_search': np.asarray(cut)}


def check_paths(data: dict[str, pd.DataFrame]) -> tuple[dict, dict]:
    streams, windows, maxima = (data[n] for n in ('streams', 'windows', 'maxima'))
    require(not streams.duplicated(['panel', 'stream_id']).any(), 'unique stream identities')
    require(not windows.duplicated(['panel', 'stream_id', 'window']).any(), 'unique window identities')
    require(not maxima.duplicated(['panel', 'statistic', 'stream_id']).any(), 'unique maximum identities')
    maximum = maxima.set_index(['panel', 'statistic', 'stream_id'])
    grouped = dict(tuple(windows.groupby(['panel', 'stream_id'], sort=False)))
    originals = streams[streams.source.str.endswith('.json')]
    require(len(originals) == 960 and len(grouped) == 960, '960 original paths')
    cache, paths, references = {}, {}, {}
    errors = {'GLR': 0., 'cut_search': 0.}
    for stream in originals.itertuples():
        if stream.source not in cache:
            cache[stream.source] = {r['stream_id']: r for r in json.loads((ROOT / stream.source).read_text())}
        source = cache[stream.source][stream.stream_id]
        row = grouped[stream.panel, stream.stream_id].sort_values('window')
        require(row.window.tolist() == list(range(1, 121)), 'full wall-clock coverage')
        scores = source['scores']
        eligible = np.array([x is not None for x in scores])
        require(np.array_equal(row.eligible, eligible), 'source eligibility and abstentions')
        close(row.loc[eligible, 'raw_score'], np.array(scores, dtype=float)[eligible], 'cached raw score identity')
        for field in ('checkpoint_seed', 'replicate', 'condition', 'mode'):
            require(getattr(stream, field) == source[field], 'stream metadata: ' + field)
        warm = np.array([x for x in scores[:40] if x is not None])
        center = float(np.median(warm))
        scale = float(1.4826 * np.median(np.abs(warm - center)))
        require(len(warm) >= 32 and scale >= 1e-12, 'warm-up eligibility')
        close([stream.center, stream.scale], [center, scale], 'warm-up normalizer')
        z = [None if x is None else (x - center) / scale for x in scores]
        close(row.loc[eligible, 'z'], np.array(z, dtype=float)[eligible], 'local standardized scores')
        ref = reference_paths(z)
        for statistic in ('GLR', 'cut_search'):
            key = stream.panel, stream.stream_id, statistic
            actual = row.iloc[40:][statistic].to_numpy()
            close(actual, ref[statistic], 'independent path: ' + str(key))
            errors[statistic] = max(errors[statistic], float(np.max(np.abs(actual - ref[statistic]))))
            require(actual.max() == maximum.loc[(stream.panel, statistic, stream.stream_id), 'maximum'], 'path maximum identity')
            paths[key], references[key] = actual, ref[statistic]
    return (paths, references), {'streams': 960, 'windows': len(windows), 'maximum_absolute_path_difference': errors,
                                  'path_tolerance': 'rtol=2e-12, atol=2e-12'}


def check_calibration(data: dict[str, pd.DataFrame]) -> dict:
    streams = data['streams'].set_index(['panel', 'stream_id'])
    original = data['streams'][data['streams'].source.str.endswith('.json') & data['streams'].stable]
    maxima = data['maxima'].set_index(['panel', 'statistic', 'stream_id']).maximum
    div = data['divisors']
    require(div.divisor_id.is_unique and len(div) == 1200, 'divisor coverage')
    members = data['divisor_membership']
    require(not members.duplicated(['divisor_id', 'stream_id']).any(), 'duplicate divisor member')
    groups = dict(tuple(members.groupby('divisor_id')))
    calculated = {}
    for r in div.itertuples():
        subset = original[(original.panel == r.panel) & (original.checkpoint_seed == r.checkpoint_seed)]
        if r.excluded_replicate != -1:
            subset = subset[subset.replicate != r.excluded_replicate]
        if r.calibration == 'pooled':
            expected = 1.
            require(r.estimation_streams == 0 and r.divisor_id not in groups, 'pooled divisor has no estimated members')
        else:
            m = groups[r.divisor_id]
            require(set(m.stream_id) == set(subset.stream_id), 'exact target divisor membership')
            require(r.estimation_streams == len(m) == (8 if r.excluded_replicate == -1 else 6), 'target divisor budget')
            values = np.array([maxima.loc[r.panel, r.statistic, sid] for sid in m.stream_id])
            close(m.stable_maximum, values, 'divisor membership maxima')
            expected = float(np.mean(values) if r.calibration == 'arithmetic' else np.exp(np.mean(np.log(values))))
        close(r.divisor, expected, 'independently recomputed divisor')
        calculated[r.panel, r.statistic, r.calibration, r.checkpoint_seed, r.excluded_replicate] = expected
    assignments = data['calibration_assignments']
    require(not assignments.duplicated(['panel', 'statistic', 'calibration', 'target_checkpoint', 'foreign_stream_id']).any(), 'duplicate foreign record')
    levels = {}
    for key, group in assignments.groupby(['panel', 'statistic', 'calibration', 'target_checkpoint']):
        panel, statistic, calibration, target = key
        foreign = original[(original.panel == panel) & (original.checkpoint_seed != target)]
        require(set(group.foreign_stream_id) == set(foreign.stream_id) and len(group) == 152, 'exact foreign calibration membership')
        ratios = []
        for a in group.itertuples():
            source = streams.loc[panel, a.foreign_stream_id]
            require(source.stable and source.checkpoint_seed == a.foreign_checkpoint, 'foreign checkpoint identity')
            value = maxima.loc[panel, statistic, a.foreign_stream_id]
            divisor = calculated[panel, statistic, calibration, source.checkpoint_seed, -1]
            close([a.maximum, a.foreign_divisor, a.normalized_maximum], [value, divisor, value / divisor], 'foreign normalized value')
            ratios.append(value / divisor)
        levels[key] = max(ratios)
    thresholds = data['thresholds']
    require(len(thresholds) == 5760 and not thresholds.duplicated(['panel', 'statistic', 'calibration', 'stream_id']).any(), 'threshold coverage')
    for t in thresholds.itertuples():
        source = streams.loc[t.panel, t.stream_id]
        level = levels[t.panel, t.statistic, t.calibration, t.checkpoint_seed]
        values = {rep: level * calculated[t.panel, t.statistic, t.calibration, t.checkpoint_seed, rep] for rep in (-1, 0, 1, 2, 3)}
        rep = int(source.replicate) if source.stable else -1
        close([t.foreign_level, t.full_divisor, t.changed, t.base_threshold],
              [level, calculated[t.panel, t.statistic, t.calibration, t.checkpoint_seed, -1], values[-1], values[rep]], 'base threshold reconstruction')
        for rep in range(4):
            close(getattr(t, f'stable_{rep}'), values[rep], 'replicate-excluded threshold')
        require(t.target_estimation_streams == (0 if t.calibration == 'pooled' else 6 if source.stable else 8), 'target estimation budget metadata')
    return {'divisors': len(div), 'foreign_assignments': len(assignments), 'base_thresholds': len(thresholds)}


def check_role(point: object) -> None:
    if point.family == 'additional_stable':
        expected = 'native_held_out_evaluation' if point.calibration == 'arithmetic' else 'retrospective_unscaled_reuse'
    else:
        expected = 'historical_frozen_rule_reconstruction' if point.statistic == 'GLR' and point.calibration in ('pooled', 'arithmetic') else 'retrospective_unscaled_rule'
    require(point.multiplier == 1. and point.selection == expected, 'evaluation role and selection chronology')


def check_decisions(data: dict[str, pd.DataFrame], paths: dict, references: dict) -> dict:
    points, stops, checkpoints = (data[n] for n in ('operating_points', 'stopping_times', 'checkpoint_results'))
    require(points.point_id.is_unique and len(points) == 76, 'operating point coverage')
    require(not stops.duplicated(['point_id', 'stream_id']).any(), 'unique point/stream decisions')
    mapping = data['streams'].set_index(['panel', 'stream_id'])
    thresholds = data['thresholds'].set_index(['panel', 'statistic', 'calibration', 'stream_id'])
    unscaled = data['thresholds'].groupby(['panel', 'statistic', 'calibration', 'checkpoint_seed']).changed.first()
    expected_ids = {f'{p}:{s}:{c}:{b}' for p in ('D','E') for s in ('GLR','cut_search') for c in CALIBRATIONS for b in ('native','original_caps','floor_halved','zero')}
    expected_ids |= {f'{p}:GLR:{c}:additional_{cap}' for p in ('D','E') for c in ('arithmetic','geometric') for cap in (None,0,1,2,3,5,8)}
    require(set(points.point_id) == expected_ids, 'exact operating point identities')
    ambiguous = disagreements = 0
    boundaries = []
    for p in points.itertuples():
        selected = stops[stops.point_id == p.point_id]
        require(len(selected) == 480, '480 decisions per point')
        stable = selected[selected.stable]
        observed_counts = []
        for mode in MODES:
            subset = stable[stable['mode'] == mode]
            require(len(subset) == getattr(p, f'{mode}_denominator') == 80, 'stable denominator')
            observed_counts.append(int(subset.fired.sum()))
            require(observed_counts[-1] == getattr(p, f'{mode}_alarms'), 'aggregate stable count')
        require(sum(observed_counts) == p.actual_total_alarms, 'total stable count')
        if p.budget == 'native':
            check_role(p)
        else:
            ratio = stable.maximum.to_numpy() / stable.base_threshold.to_numpy()
            if p.family == 'additional_stable':
                candidate = float(np.sort(ratio)[::-1][int(p.total_cap)])
                feasible = sum(observed_counts) <= p.total_cap
            else:
                candidates = [np.sort(ratio[stable['mode'].eq(mode)])[::-1][int(getattr(p, f'{mode}_cap'))] for mode in MODES]
                candidate = float(max(candidates))
                feasible = all(count <= getattr(p, f'{mode}_cap') for mode, count in zip(MODES, observed_counts, strict=True))
            require(feasible and candidate == p.ratio_candidate, 'stable-only order-statistic candidate and feasibility')
            require(0 <= p.boundary_ulp_increments <= 32, 'bounded ULP adjustment')
            for _ in range(int(p.boundary_ulp_increments)):
                candidate = np.nextafter(candidate, np.inf)
            require(candidate == p.multiplier, 'declared upward adjustment')
        for s in selected.itertuples():
            original = mapping.loc[s.panel, s.stream_id]
            for field in ('checkpoint_seed', 'condition', 'mode', 'replicate', 'stable'):
                require(original[field] == getattr(s, field), 'decision source identity')
            key = s.panel, s.stream_id, p.statistic
            expected_base = unscaled.loc[s.panel, p.statistic, p.calibration, s.checkpoint_seed] if p.family == 'additional_stable' else thresholds.loc[(s.panel, p.statistic, p.calibration, s.stream_id), 'base_threshold']
            require(s.base_threshold == expected_base and s.threshold == s.base_threshold * p.multiplier, 'threshold product and base identity')
            require(s.fired == (s.maximum > s.threshold), 'strict maximum decision')
            if key in paths:
                path = paths[key]
                require(s.stopping_time_available and s.maximum == path.max(), 'decision path maximum')
                hit = np.flatnonzero(path > s.threshold)
                delay = int(hit[0]) + 1 if len(hit) else 81
                require(s.restricted_delay == delay and (s.first_crossing == delay if len(hit) else pd.isna(s.first_crossing)), 'exact first-crossing decision')
                ref = references[key]
                tolerance = 2e-12 + 2e-12 * np.abs(path)
                near = bool(np.any(np.abs(path - s.threshold) <= tolerance))
                ambiguous += int(near)
                rhit = np.flatnonzero(ref > s.threshold)
                rdelay = int(rhit[0]) + 1 if len(rhit) else 81
                if rdelay != delay:
                    require(near, 'independent first-crossing mismatch away from a rounding boundary')
                    disagreements += 1
                if near:
                    boundaries.append({'point_id': p.point_id, 'stream_id': s.stream_id, 'condition': s.condition, 'stable': bool(s.stable), 'threshold': float(s.threshold), 'exact_exported_delay': delay, 'reference_delay': rdelay, 'minimum_distance_to_threshold': float(np.min(np.abs(path - s.threshold)))})
            else:
                require(not s.stopping_time_available and pd.isna(s.restricted_delay) and pd.isna(s.first_crossing), 'maxima-only missing stopping time')
                require(s.maximum == original.max_statistic, 'additional-stable source maximum')
        primary = selected[selected.condition == PRIMARY]
        require(len(primary) == p.primary_denominator == 80 and int(primary.fired.sum()) == p.primary_detections, 'primary detection aggregate')
        require(primary.restricted_delay.sum() == p.primary_delay_sum and primary.restricted_delay.mean() == p.primary_mean_delay, 'primary delay aggregate')
        c = checkpoints[checkpoints.point_id == p.point_id]
        require(len(c) == 20 and c.checkpoint_seed.is_unique, 'checkpoint coverage')
        for t in c.itertuples():
            group = selected[selected.checkpoint_seed == t.checkpoint_seed]
            primary = group[group.condition == PRIMARY]
            require(t.primary_detections == primary.fired.sum() and t.primary_delay_sum == primary.restricted_delay.sum(), 'checkpoint primary counts/delays')
            for mode in MODES:
                g = group[group.stable & group['mode'].eq(mode)]
                require(len(g) == getattr(t, f'{mode}_denominator') == 4 and g.fired.sum() == getattr(t, f'{mode}_alarms'), 'checkpoint stable counts/denominators')
    return {'points': len(points), 'decisions': len(stops), 'reference_near_boundary_decisions': ambiguous,
            'reference_rounding_boundary_delay_disagreements': disagreements,
            'changed_stream_reference_disagreements': sum(not r['stable'] and r['exact_exported_delay'] != r['reference_delay'] for r in boundaries),
            'boundary_records': boundaries,
            'boundary_scope': 'Exact exported float paths checked for every stopping time; independent prefix-sum paths checked numerically, with near-boundary disagreements separately disclosed.'}


def check_dispersion(data: dict[str, pd.DataFrame]) -> dict:
    streams = data['streams']
    original = streams[streams.source.str.endswith('.json') & streams.stable]
    additional = streams[~streams.source.str.endswith('.json')]
    maxima = data['maxima'].set_index(['panel', 'statistic', 'stream_id']).maximum
    frame = data['dispersion']
    keys = ['panel','statistic','calibration','allocation','summary']
    require(len(frame) == 156 and not frame.duplicated(keys).any(), 'dispersion coverage')
    balanced = {','.join(map(str, ids)) for ids in itertools.combinations(range(4), 2)}
    require(set(frame[frame.estimation_streams_per_checkpoint == 4].allocation) == balanced, 'all balanced allocations')
    for r in frame.itertuples():
        stable = original[original.panel == r.panel]
        if r.allocation == 'all_original_to_additional':
            require(r.statistic == 'GLR', 'no invented additional cut-search cache')
            est, test, n = stable, additional[additional.panel == r.panel], 8
        else:
            chosen = set(map(int, r.allocation.split(',')))
            est, test, n = stable[stable.replicate.isin(chosen)], stable[~stable.replicate.isin(chosen)], 4
        require(not set(est.stream_id) & set(test.stream_id), 'disjoint dispersion stream membership')
        require(r.estimation_streams_per_checkpoint == r.evaluation_streams_per_checkpoint == n, 'dispersion budget')
        raw, normalized = [], []
        for seed in sorted(stable.checkpoint_seed.unique()):
            e, t = est[est.checkpoint_seed == seed], test[test.checkpoint_seed == seed]
            require(len(e) == len(t) == n, 'per-checkpoint dispersion sample budget')
            evalues = np.array([maxima.loc[r.panel, r.statistic, sid] for sid in e.stream_id])
            tvalues = t.max_statistic.to_numpy() if n == 8 else np.array([maxima.loc[r.panel, r.statistic, sid] for sid in t.stream_id])
            logged = np.log(tvalues)
            summary = float(np.mean(logged) if r.summary == 'mean_log_maximum' else np.quantile(logged, .9) if r.summary == 'q90_log_maximum' else np.max(logged))
            divisor = float(np.mean(evalues) if r.calibration == 'arithmetic' else np.exp(np.mean(np.log(evalues))))
            raw.append(summary)
            normalized.append(summary - np.log(divisor))
        raw_sd, normalized_sd = np.std(raw, ddof=1), np.std(normalized, ddof=1)
        close([r.raw_sd, r.normalized_sd, r.change], [raw_sd, normalized_sd, normalized_sd / raw_sd - 1], 'independent dispersion value')
    return {'rows': len(frame), 'balanced_rows': 144, 'additional_GLR_rows': 12}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'use a fresh verification output')
    data = load(args.results)
    (paths, references), path_report = check_paths(data)
    report = {'status': 'PASS', 'independence': 'No import of analyze.py or the production sequential implementation.',
              'paths': path_report, 'calibration': check_calibration(data),
              'decisions': check_decisions(data, paths, references), 'dispersion': check_dispersion(data)}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
