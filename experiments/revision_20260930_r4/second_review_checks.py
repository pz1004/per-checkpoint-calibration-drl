"""Independent exact-arithmetic and stored-path checks for the second review."""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def exact_checks() -> list[str]:
    passed = []
    for name, groups in (
        ('unique maximum', [[2, 5], [1, 3], [1, 2]]),
        ('within-checkpoint ties', [[5, 5], [1, 3], [1, 2]]),
        ('cross-checkpoint ties', [[2, 5], [5, 3], [1, 2]]),
    ):
        groups = [[F(v) for v in group] for group in groups]
        maxima = [max(g) for g in groups]
        m1 = max(maxima)
        foreign = [max(m for j, m in enumerate(maxima) if j != i) for i in range(3)]
        k = max(v / foreign[i] for i, group in enumerate(groups) for v in group)
        h = [k * x for x in foreign]
        assert all(v <= h[i] for i, group in enumerate(groups) for v in group)
        assert max(v for g in groups for v in g) == m1  # scaling-only zero-cap selector
        if name == 'cross-checkpoint ties':
            assert k == 1 and h == [m1] * 3 and foreign == [m1] * 3
        else:
            star = maxima.index(m1)
            m2 = max(m for j, m in enumerate(maxima) if j != star)
            assert k == m1 / m2
            assert h[star] == m1
            assert all(x == m1 * m1 / m2 for j, x in enumerate(h) if j != star)
        # Any smaller positive multiplier violates the binding stable observation.
        binding = [(i, v) for i, g in enumerate(groups) for v in g if v / foreign[i] == k]
        assert binding and all(v > foreign[i] * (k - k / 1000) for i, v in binding)
        passed.append(name)
    low, high = F(5), F(25, 3)
    assert not (low > low) and not (high > high)
    assert high > low and low < (low + high) / 2 <= high
    assert [(m > low) and not (m > high) for m in (low, (low + high) / 2, high)] == [False, True, True]
    passed.append('strict equality: gain interval is (lower, upper]')
    return passed


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision='round_trip', keep_default_na=False, low_memory=False)


def run(results: Path, inputs: Path) -> dict:
    checks = exact_checks()
    hopper = 'D:cut_search:pooled:zero:matched6'
    eids = [f'E:GLR:{cal}:additional_2' for cal in ('arithmetic', 'quantile90')]
    ids = {hopper, hopper + ':scaling_only', *eids, *[s + ':loco:omit_808' for s in eids]}
    points = read(results / 'operating_points.csv').set_index('point_id')
    summary = read(results / 'condition_summary.csv')
    parts = pd.read_csv(results / 'stream_decisions.csv', float_precision='round_trip',
                        keep_default_na=False, chunksize=20000, low_memory=False)
    decisions = pd.concat([g[g.point_id.isin(ids)] for g in parts], ignore_index=True)
    primary = decisions[decisions.condition == 'gradual_updating_down']
    a = primary[primary.point_id == hopper].set_index('stream_key')
    b = primary[primary.point_id == hopper + ':scaling_only'].set_index('stream_key').loc[a.index]
    assert len(a) == len(b) == 80 and a.index.is_unique and b.index.is_unique
    assert a.fired.sum() == 17 and b.fired.sum() == 55
    assert a.loc[a.fired, 'checkpoint_seed'].nunique() == 11
    assert b.loc[b.fired, 'checkpoint_seed'].nunique() == 20
    gained = b.fired & ~a.fired
    assert gained.sum() == 38 and not (a.fired & ~b.fired).any()
    assert ((b.maximum > b.threshold) & (b.maximum <= a.threshold)).equals(gained)
    assert set(a.loc[gained, 'checkpoint_seed']) <= set(range(700, 720)) - {703}
    cal = read(results / 'calibration_records.csv')
    cal = cal[cal.point_id == hopper]
    stable = decisions[(decisions.point_id == hopper) & decisions.tuning]
    assert set(cal.stream_key) == set(stable.stream_key) and len(cal) == 160
    assert (cal.divisor == 1).all() and (stable.divisor == 1).all()
    maximum = cal.groupby('checkpoint_seed').maximum.max()
    assert maximum.idxmax() == 703 and (maximum == maximum.max()).sum() == 1
    m1, m2 = float(maximum.loc[703]), float(maximum.drop(703).max())
    assert len(cal[cal.maximum == m1]) == 1
    hl = float(a.loc[a.checkpoint_seed != 703, 'threshold'].iloc[0])
    hs = float(b.threshold.iloc[0])
    assert (a.loc[a.checkpoint_seed != 703, 'threshold'] == hl).all()
    assert (b.threshold == hs).all() and hs == m1
    assert (a.loc[a.checkpoint_seed == 703, 'threshold'] == m1).all()
    assert points.loc[hopper, 'multiplier'] == m1 / m2
    assert hl == m1 * (m1 / m2)  # retained base * multiplier, not m1**2/m2
    for pid in (hopper, hopper + ':scaling_only'):
        assert points.loc[pid, 'frozen_alarms'] == points.loc[pid, 'updating_alarms'] == 0
        assert points.loc[pid, 'boundary_ulp_increments'] == 0
    selected_keys = set(primary.stream_key)
    paths = read(inputs / 'paths.csv')
    paths = {key: g.sort_values('index') for key, g in paths[paths.stream_key.isin(selected_keys)].groupby('stream_key')}
    checked_paths = 0
    for row in primary.itertuples():
        path = paths[row.stream_key]
        assert path['index'].tolist() == list(range(1, 81))
        values = pd.to_numeric(path.value, errors='coerce').to_numpy()
        assert float(np.nanmax(values)) == row.maximum
        crossings = np.flatnonzero(values > row.threshold)
        delay = int(crossings[0] + 1) if len(crossings) else 81
        assert bool(len(crossings)) == row.fired and delay == float(row.restricted_delay)
        checked_paths += 1
    gain_records = [dict(stream_id=r.stream_id, checkpoint=int(r.checkpoint_seed),
                         maximum=r.maximum, lower_threshold=r.threshold,
                         upper_threshold=float(a.loc[key, 'threshold']),
                         first_crossing=int(float(r.restricted_delay))) for key, r in b[gained].iterrows()]
    retained_keys = None
    detected_keys = None
    omission = {}
    for source, count in zip(eids, (5, 7), strict=True):
        ref = primary[(primary.point_id == source) & (primary.checkpoint_seed != 808)].set_index('stream_key')
        fitted = primary[primary.point_id == source + ':loco:omit_808'].set_index('stream_key')
        assert set(ref.index) == set(fitted.index) and len(ref) == len(fitted) == 76
        fitted = fitted.loc[ref.index]
        assert np.array_equal(fitted.reference_threshold, ref.threshold)
        assert np.array_equal(fitted.reference_fired, ref.fired)
        assert np.array_equal(fitted.reference_delay, ref.restricted_delay)
        assert ref.fired.sum() == count and fitted.fired.sum() == 7
        if retained_keys is None:
            retained_keys, detected_keys = set(fitted.index), set(fitted.index[fitted.fired])
        else:
            assert set(fitted.index) == retained_keys and set(fitted.index[fitted.fired]) == detected_keys
        p = points.loc[source + ':loco:omit_808']
        assert p.frozen_alarms == p.updating_alarms == 1
        s = summary[(summary.point_id == source + ':loco:omit_808') & (summary.condition == 'gradual_updating_down')].iloc[0]
        assert float(s.restricted_delay) == fitted.restricted_delay.astype(float).mean()
        omission[source] = dict(restricted_reference_detections=int(ref.fired.sum()),
                                recalibrated_detections=int(fitted.fired.sum()), denominator=76,
                                stable_alarms_frozen=1, stable_alarms_updating=1,
                                restricted_reference_delay=ref.restricted_delay.astype(float).mean(),
                                recalibrated_delay=fitted.restricted_delay.astype(float).mean(),
                                changed_delay_streams=int((fitted.restricted_delay != ref.restricted_delay).sum()))
    assert omission[eids[0]]['recalibrated_delay'] != omission[eids[1]]['recalibrated_delay']
    assert omission[eids[1]]['changed_delay_streams'] > 0
    sources = [results / (name + '.csv') for name in ('operating_points', 'condition_summary', 'stream_decisions', 'calibration_records')]
    sources += [inputs / 'paths.csv']
    return dict(status='PASS', exact_arithmetic_checks=checks,
                hopper=dict(unique_donor_checkpoint=703, unique_donor_stream=cal.loc[cal.maximum == m1, 'stream_id'].iloc[0],
                            m1=m1, m2=m2, nonexceptional_loco_threshold=hl, scaling_only_threshold=hs,
                            reduction_percent=100 * (1 - hs / hl), gains=38, losses=0,
                            detections=[17, 55], checkpoints_with_detection=[11, 20],
                            gain_interval='lower < maximum <= upper', gain_records=gain_records),
                checkpoint_808=omission, retained_primary_stream_ids=sorted(retained_keys),
                common_detected_stream_ids=sorted(detected_keys), checked_stored_primary_paths=checked_paths,
                source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                scope='exact rational fixtures separate from binary64 stored-threshold and indexed-path checks')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.results, args.inputs)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('hopper', 'retained_primary_stream_ids', 'source_sha256')}, indent=2))
