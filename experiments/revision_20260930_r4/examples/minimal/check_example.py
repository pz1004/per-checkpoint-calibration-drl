"""Independent standard-library checker; never imports the diagnostic engine."""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import json
import math
from pathlib import Path


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def check(output: Path) -> dict:
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['schema_version'] == 1 and 'indexed_paths' not in manifest
    tables = {key: read(root / value) for key, value in manifest['tables'].items()}
    streams = {r['stream_key']: r for r in tables['streams']}
    divisors = {r['divisor_id']: r for r in tables['divisors']}
    targets = {r['stream_key']: r for r in tables['targets']}
    assert len(streams) == 36 and len(divisors) == 15 and len(targets) == 36
    seeds = sorted({int(r['checkpoint_seed']) for r in streams.values()})
    assert seeds == [11, 31, 71]
    declared = {did: {r['stream_key'] for r in tables['memberships'] if r['divisor_id'] == did}
                for did in divisors}
    for did, row in divisors.items():
        members = {key for key, s in streams.items() if s['checkpoint_seed'] == row['checkpoint_seed']
                   and s['role'] == 'original_stable' and s['replicate'] != row['excluded_replicate']}
        assert declared[did] == members
        assert len(members) == int(row['membership_count']) == (8 if row['excluded_replicate'] == '-1' else 6)
        mean = sum(Fraction(streams[k]['maximum']) for k in members) / len(members)
        assert float(row['divisor']) == float(mean)
    assert {r['stream_key'] for r in tables['calibration']} == {
        key for key, s in streams.items() if s['role'] == 'original_stable'}
    for row in tables['calibration']:
        s, d = streams[row['stream_key']], divisors[row['divisor_id']]
        assert d['excluded_replicate'] == '-1' and d['checkpoint_seed'] == s['checkpoint_seed']
    for key, row in targets.items():
        d, s = divisors[row['divisor_id']], streams[key]
        assert d['excluded_replicate'] == s['replicate'] and d['checkpoint_seed'] == s['checkpoint_seed']
        assert (row['tuning'] == 'True') == (s['role'] == 'original_stable')
    point = tables['points'][0]
    assert len(tables['points']) == 1 and json.loads(point['caps_json']) == {'frozen': 0, 'updating': 0}
    assert point['include_control'] == point['include_influence'] == 'True'
    actual_points = read(output / 'operating_points.csv')
    actual_decisions = read(output / 'stream_decisions.csv')
    actual_summary = read(output / 'condition_summary.csv')
    assert len(actual_points) == 8 and len(actual_decisions) == 216
    assert {(p['control_type'], p['omitted_checkpoint']) for p in actual_points} == {
        (rule, omission) for rule in ('loco', 'scaling_only') for omission in ('', '11', '31', '71')}
    expected_summary = []
    references: dict[str, dict[str, bool]] = {}
    for p in actual_points:
        omitted = p['omitted_checkpoint']
        retained = {k: s for k, s in streams.items() if s['checkpoint_seed'] != omitted}
        stable = {k: s for k, s in retained.items() if s['role'] == 'original_stable'}
        ratios = {seed: max(float(s['maximum']) / float(divisors[f'd{seed}_exclude_-1']['divisor'])
                           for s in stable.values() if int(s['checkpoint_seed']) == seed)
                  for seed in seeds if str(seed) != omitted}
        base = {}
        for k, s in retained.items():
            d = float(divisors[targets[k]['divisor_id']]['divisor'])
            foreign = max(v for seed, v in ratios.items() if seed != int(s['checkpoint_seed']))
            base[k] = foreign * d if p['control_type'] == 'loco' else d
        multiplier = max(float(s['maximum']) / base[k] for k, s in stable.items())
        while any(float(s['maximum']) > base[k] * multiplier for k, s in stable.items()):
            multiplier = math.nextafter(multiplier, math.inf)
        assert float(p['multiplier']) == multiplier
        assert int(p['frozen_alarms']) == int(p['updating_alarms']) == 0
        observed = {r['stream_key']: r for r in actual_decisions if r['point_id'] == p['point_id']}
        assert set(observed) == set(retained)
        fires = {}
        for k, s in retained.items():
            threshold = base[k] * multiplier
            fires[k] = float(s['maximum']) > threshold
            r = observed[k]
            assert float(r['threshold']) == threshold and (r['fired'] == 'True') == fires[k]
            assert r['stopping_time_available'] == 'False'
            assert r['restricted_delay'] == r['reference_delay'] == r['first_crossing'] == ''
            assert int(r['target_membership_count']) == 6
        rule = p['control_type']
        if not omitted:
            references[rule] = fires
        reference = references[rule] if omitted else (references['loco'] if rule == 'scaling_only' else fires)
        for k, r in observed.items():
            assert (r['reference_fired'] == 'True') == reference[k]
            assert (r['gained'] == 'True') == (fires[k] and not reference[k])
            assert (r['lost'] == 'True') == (reference[k] and not fires[k])
        changed = [k for k, s in retained.items() if s['role'] == 'changed']
        summary = next(r for r in actual_summary if r['point_id'] == p['point_id'] and r['condition'] == 'changed_updating')
        assert int(summary['detections']) == sum(fires[k] for k in changed)
        assert int(summary['denominator']) == len(changed)
        assert int(summary['delay_available_streams']) == 0 and summary['restricted_delay'] == ''
        expected_summary.append(dict(rule=rule, omitted_checkpoint=int(omitted) if omitted else None,
                                     detections=sum(fires[k] for k in changed), denominator=len(changed),
                                     stable_alarms=0, delay='unavailable'))
    expected = json.loads((root / 'expected_summary.json').read_text())
    assert expected_summary == expected['operating_points']
    assert [r['detections'] for r in expected_summary[:2]] == [5, 7]
    report = dict(status='PASS', operating_points=8, checked_stream_decisions=216,
                  calibration_members_per_checkpoint=8, target_members_per_checkpoint=6,
                  formulas='arithmetic means independently recomputed from declared members',
                  delay='unavailable for every stream; no indexed paths supplied',
                  operating_point_summary=expected_summary)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = check(args.results)
    text = json.dumps(report, indent=2) + '\n'
    if args.report:
        with args.report.open('x') as handle:
            handle.write(text)
    print(text)
