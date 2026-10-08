"""Export the immutable R4 study caches into the reusable diagnostic interface."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

import analyze as r4

HERE = Path(__file__).resolve().parent


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    f, _, _ = r4.load()
    streams, paths = [], []
    for r in f.itertuples():
        for stat in ('GLR', 'cut_search'):
            key = f'{r.panel}:{stat}:{r.stream_id}'
            values = getattr(r, stat)
            streams.append(dict(stream_key=key, stream_id=r.stream_id, panel=r.panel, statistic=stat,
                                checkpoint_seed=r.checkpoint_seed, replicate=r.replicate, mode=r.mode,
                                condition=r.condition, role='original_stable' if r.stable else 'changed',
                                maximum=float(values.max())))
            paths.extend(dict(stream_key=key, index=i, value=x) for i, x in enumerate(values, 1))
    for panel in ('D', 'E'):
        held = r4.read(r4.ROOT / f'experiments/stable_holdout/panel_{panel.lower()}/holdout_streams.csv')
        for r in held.itertuples():
            streams.append(dict(stream_key=f'{panel}:GLR:{r.stream_id}', stream_id=r.stream_id,
                                panel=panel, statistic='GLR', checkpoint_seed=r.checkpoint_seed,
                                replicate=r.replicate, mode=r.mode, condition='stable_' + r.mode,
                                role='additional_stable', maximum=r.max_statistic))
    streams = pd.DataFrame(streams)
    divisors = pd.concat([r4.read(HERE / 'results/divisors.csv'), r4.read(HERE / 'quantile_results/divisors.csv')])
    divisors = divisors[['divisor_id', 'panel', 'statistic', 'calibration', 'checkpoint_seed', 'excluded_replicate', 'divisor']].copy()
    members, calibration = [], []
    for d in divisors.itertuples():
        s = streams[(streams.panel == d.panel) & (streams.statistic == d.statistic) &
                    (streams.checkpoint_seed == d.checkpoint_seed) & (streams.role == 'original_stable')]
        if d.excluded_replicate != -1:
            s = s[s.replicate != d.excluded_replicate]
        members.extend(dict(divisor_id=d.divisor_id, stream_key=key) for key in s.stream_key)
        if d.excluded_replicate == -1:
            calibration.extend(dict(panel=d.panel, statistic=d.statistic, calibration=d.calibration,
                                    stream_key=key, divisor_id=d.divisor_id) for key in s.stream_key)
    divisors['membership_count'] = [8 if rep == -1 else 6 for rep in divisors.excluded_replicate]
    points, targets = [], []
    for folder, prefix in (('results', ''), ('results', 'matched_'), ('quantile_results', '')):
        ps = r4.read(HERE / folder / (prefix + 'operating_points.csv'))
        stops = r4.read(HERE / folder / (prefix + 'stopping_times.csv'))
        groups = {key: g for key, g in stops.groupby('point_id', sort=False)}
        for p in ps.itertuples():
            matched, additional = p.family == 'matched_target_budget', p.family == 'additional_stable'
            policy = 'matched_six' if matched else ('additional_eight' if additional else 'historical_six_stable_eight_changed')
            caps = {} if p.budget == 'native' else ({'total': int(p.total_cap)} if additional else {'frozen': int(p.frozen_cap), 'updating': int(p.updating_cap)})
            control = matched or (additional and p.budget != 'native')
            influence = (matched and p.budget == 'original_caps') or (additional and p.budget == 'total_2')
            points.append(dict(source_point_id=p.point_id, panel=p.panel, statistic=p.statistic,
                               calibration=p.calibration, family=p.family, budget=p.budget,
                               target_budget_policy=policy, tuning_role='additional_stable' if additional else 'original_stable',
                               caps_json=json.dumps(caps, sort_keys=True), multiplier=p.multiplier,
                               include_control=control, include_influence=influence))
            for s in groups[p.point_id].itertuples():
                rep = s.replicate if matched or (s.stable and not additional) else -1
                targets.append(dict(source_point_id=p.point_id, stream_key=f'{p.panel}:{p.statistic}:{s.stream_id}',
                                    divisor_id=f'{p.panel}:{p.statistic}:{p.calibration}:{s.checkpoint_seed}:exclude_{rep}',
                                    tuning=s.stable, historical_threshold=s.threshold, historical_fired=s.fired,
                                    historical_delay=s.restricted_delay))
    points = pd.DataFrame(points)
    assert len(points) == 136 and int(points.include_control.sum()) == 84 and int(points.include_influence.sum()) == 22
    assert int((points.budget != 'native').sum()) == 120
    tables = dict(streams=streams, divisors=divisors, memberships=pd.DataFrame(members),
                  calibration=pd.DataFrame(calibration), targets=pd.DataFrame(targets), points=points,
                  paths=pd.DataFrame(paths))
    for name, frame in tables.items():
        r4.write(frame, output / (name + '.csv'))
    source_files = [p for folder in ('results', 'quantile_results') for p in sorted((HERE / folder).glob('*.csv'))]
    manifest = dict(schema_version=1, horizon=80, miss_delay=81, indexed_paths='paths.csv',
                    tables={name: name + '.csv' for name in tables if name != 'paths'},
                    source_csv_sha256={str(p.relative_to(r4.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({name: len(frame) for name, frame in tables.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args().output)
