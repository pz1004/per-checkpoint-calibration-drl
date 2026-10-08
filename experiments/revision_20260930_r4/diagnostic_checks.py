"""Deterministic software fixtures and changed-outcome noninterference checks."""
from __future__ import annotations

import argparse
import copy
from fractions import Fraction as F
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

import diagnostic as engine


def portable_fixture() -> dict[str, pd.DataFrame]:
    """Three checkpoints, four declared stable members each, supplied scales."""
    streams, divisors, members, calibration, targets = [], [], [], [], []
    for i, seed in enumerate((11, 31, 71)):
        did = f'd{seed}'
        divisors.append(dict(divisor_id=did, panel='fixture', statistic='S', calibration='arithmetic',
                             checkpoint_seed=seed, excluded_replicate=-1, divisor=float(i + 2), membership_count=4))
        for rep in (0, 1):
            for mode in ('frozen', 'updating'):
                key = f's{seed}_{rep}_{mode}'
                streams.append(dict(stream_key=key, stream_id=key, panel='fixture', statistic='S',
                                    checkpoint_seed=seed, replicate=rep, mode=mode, condition='stable_' + mode,
                                    role='original_stable', maximum=float(10 + i + rep)))
                members.append(dict(divisor_id=did, stream_key=key))
                calibration.append(dict(panel='fixture', statistic='S', calibration='arithmetic', stream_key=key, divisor_id=did))
                targets.append(dict(source_point_id='fixture', stream_key=key, divisor_id=did, tuning=True))
    point = dict(source_point_id='fixture', panel='fixture', statistic='S', calibration='arithmetic', family='fixture',
                 budget='retuned', target_budget_policy='additional_eight', tuning_role='original_stable',
                 caps_json='{"frozen": 0, "updating": 0}', include_control=True, include_influence=True)
    return {name: pd.DataFrame(rows) for name, rows in dict(streams=streams, divisors=divisors, memberships=members,
                                                          calibration=calibration, targets=targets, points=[point]).items()}


def coverage_checks() -> list[str]:
    """Exercise public manifest loading and ensure invalid coverage creates no output."""
    passed = []
    fixture = portable_fixture()
    with tempfile.TemporaryDirectory(prefix='diagnostic-coverage-') as tmp:
        root = Path(tmp)

        def manifest_for(name: str, frames: dict[str, pd.DataFrame]) -> Path:
            directory = root / name
            directory.mkdir()
            for key, frame in frames.items():
                frame.to_csv(directory / (key + '.csv'), index=False)
            manifest = directory / 'manifest.json'
            manifest.write_text(json.dumps(dict(schema_version=1, horizon=4,
                                                tables={key: key + '.csv' for key in frames})))
            return manifest

        valid = manifest_for('valid', fixture)
        report = engine.run(valid, root / 'valid-output')
        assert report['schema_version'] == 1 and report['counts']['operating_points'] == 8
        params = engine.read(root / 'valid-output/checkpoint_parameters.csv')
        assert set(params.foreign_calibration_records) == {4, 8}
        assert np.isfinite(params.checkpoint_ratio).all() and (params.checkpoint_ratio > 0).all()
        # These supplied values intentionally differ from arithmetic means. Formula
        # provenance belongs to the study adapter/verifier, not generic membership validation.
        assert fixture['divisors'].divisor.tolist() == [2., 3., 4.]
        passed.append('valid non-study tables: three checkpoints, four members, supplied divisors, both rules and deletions')

        for case, expected in (
            ('missing_stream', 'missing streams='),
            ('missing_checkpoint', 'missing checkpoints=[11]'),
            ('missing_group', 'missing calibration group'),
            ('inconsistent_divisor', 'inconsistent full-membership divisor identity at checkpoint 11'),
        ):
            frames = {key: frame.copy() for key, frame in fixture.items()}
            cal = frames['calibration']
            if case == 'missing_stream':
                frames['calibration'] = cal.iloc[1:]
            elif case == 'missing_checkpoint':
                frames['calibration'] = cal[~cal.stream_key.str.startswith('s11_')]
            elif case == 'missing_group':
                frames['calibration'] = cal.iloc[:0]
            else:
                alias = frames['divisors'].iloc[[0]].assign(divisor_id='alias')
                frames['divisors'] = pd.concat([frames['divisors'], alias], ignore_index=True)
                alias_members = frames['memberships'].query('divisor_id == "d11"').assign(divisor_id='alias')
                frames['memberships'] = pd.concat([frames['memberships'], alias_members], ignore_index=True)
                frames['calibration'].loc[0, 'divisor_id'] = 'alias'
            output = root / (case + '-output')
            manifest = manifest_for(case, frames)
            try:
                engine.run(manifest, output)
            except ValueError as error:
                message = str(error)
                assert "calibration group ('fixture', 'S', 'arithmetic')" in message and expected in message, message
                assert not output.exists(), 'invalid group created output'
                passed.append(case + ': identified group and rejected before output creation')
            else:
                raise AssertionError('invalid calibration accepted: ' + case)

        # Bypass loading to probe the last guard immediately before ratio export.
        obj = engine.Tables(valid)
        group = ('fixture', 'S', 'arithmetic')
        obj.calibration_groups[group] = obj.calibration_groups[group].iloc[4:]
        try:
            engine.evaluate(obj, obj.points.iloc[0].to_dict(), 'loco', None)
        except ValueError as error:
            assert 'retained checkpoint 11 ratio' in str(error)
            passed.append('undefined retained checkpoint ratio rejected before export')
        else:
            raise AssertionError('undefined checkpoint ratio accepted')
    return passed


def run(manifest: Path) -> dict:
    passed = coverage_checks()

    def rejected(label, fn):
        try:
            fn()
        except (ValueError, AssertionError, KeyError):
            passed.append(label)
        else:
            raise AssertionError('not rejected: ' + label)

    modes = np.array(['frozen', 'frozen', 'updating', 'updating'])
    selected, con, ranks = engine.select(np.array([4., 4., 2., 1.]), np.ones(4), modes, {'total': 1})
    assert selected['multiplier'] == 4 and selected['ratio_ties_at_candidate'] == 2
    assert ranks.rank_total_min.tolist() == [1, 1, 3, 4] and ranks.rank_total_max.tolist() == [2, 2, 3, 4]
    assert con[0]['achieved_alarms'] == 0
    passed.append('ratio ties and strict equality need not attain cap')
    assert engine.first(np.array([np.nan, 2., 3., 5.]), 3., 5., 4) == (True, 4.)
    assert engine.first(np.array([1., 3.]), 3., 3., 2) == (False, 3.)
    assert np.isnan(engine.first(None, 2., 3., 2)[1])
    passed.append('abstentions preserve indices; equality misses; missing paths explicit')
    rejected('infeasible cap', lambda: engine.select(np.ones(4), np.ones(4), modes, {'total': 4}))
    for bad in (0., -1., np.inf, np.nan):
        rejected('invalid maximum ' + str(bad), lambda bad=bad: engine.select(np.array([bad, 1.]), np.ones(2), modes[:2], {'total': 0}))
        rejected('invalid divisor ' + str(bad), lambda bad=bad: engine.select(np.ones(2), np.array([bad, 1.]), modes[:2], {'total': 0}))
    # Exact fractions prove common-foreign-factor and uniform-divisor invariance separately.
    maxima, d, foreign = [F(3), F(8), F(5)], [F(2), F(3), F(4)], [F(7), F(7), F(5)]
    choose = lambda bases: sorted((m / b for m, b in zip(maxima, bases)), reverse=True)[1]
    base = [x * y for x, y in zip(d, foreign)]
    k = choose(base)
    h = [k * b for b in base]
    scale = F(17, 3)
    scaled_base = [b * scale for b in base]
    assert choose(scaled_base) == k / scale
    assert [choose(scaled_base) * b for b in scaled_base] == h
    scaled_divisor_base = [x * scale * (y / scale) for x, y in zip(d, foreign)]
    assert scaled_divisor_base == base
    passed.append('exact rational common-factor and divisor-coordinate invariance')
    # Tie scenarios are evaluated through the same public table interface calculations.
    def tie_fixture(values):
        stream = pd.DataFrame([dict(stream_key=f's{i}', stream_id=f's{i}', panel='fixture', statistic='S',
                                   checkpoint_seed=i // 2, replicate=i % 2, mode='frozen' if i % 2 == 0 else 'updating',
                                   condition='stable', role='original_stable', maximum=float(v)) for i, v in enumerate(values)])
        div = pd.DataFrame([dict(divisor_id=f'd{i}', divisor=1., membership_count=2) for i in range(3)])
        targets = pd.DataFrame([dict(source_point_id='fixture', stream_key=r.stream_key, divisor_id=f'd{r.checkpoint_seed}', tuning=True) for r in stream.itertuples()])
        cal = targets[['stream_key', 'divisor_id']]
        obj = SimpleNamespace(stream_index=stream.set_index('stream_key', drop=False), divisor_index=div.set_index('divisor_id', drop=False),
                              target_groups={'fixture': targets}, calibration_groups={('fixture', 'S', 'pooled'): cal},
                              paths={}, manifest={'horizon': 4})
        p = dict(source_point_id='fixture', panel='fixture', statistic='S', calibration='pooled', family='fixture', budget='retuned',
                 target_budget_policy='fixture', caps_json='{"total": 0}')
        return engine.evaluate(obj, p, 'loco', None)
    _, _, extra = tie_fixture([5, 5, 2, 3, 1, 2])
    par = extra['checkpoint_parameters']
    assert par.loco_correction.tolist() == [.6, 1., 1.]
    assert set(extra['donor_records'].query('donor_type == "global"').stream_id) == {'s0', 's1'}
    _, _, extra = tie_fixture([5, 4, 5, 3, 1, 2])
    assert extra['checkpoint_parameters'].loco_correction.tolist() == [1., 1., 1.]
    passed.append('all global and foreign donor ties; within versus across checkpoint ties')
    t = engine.Tables(manifest)
    obj = copy.copy(t)
    omitted_keys = set(t.streams.query("panel == 'D' and statistic == 'GLR' and checkpoint_seed == 700").stream_key)
    assert omitted_keys
    cal = t.calibration
    obj.calibration = cal[~((cal.panel == 'D') & (cal.statistic == 'GLR') & (cal.calibration == 'pooled') & cal.stream_key.isin(omitted_keys))]
    try:
        obj.validate()
    except ValueError as error:
        assert "calibration group ('D', 'GLR', 'pooled')" in str(error) and 'missing checkpoints=[700]' in str(error)
        passed.append('Hopper pooled GLR checkpoint 700 regression rejected with affected group')
    else:
        raise AssertionError('checkpoint 700 regression accepted')
    for attr, label, mutate in (
        ('streams', 'duplicate stream identities', lambda f: pd.concat([f, f.iloc[[0]]])),
        ('targets', 'duplicate target identities', lambda f: pd.concat([f, f.iloc[[0]]])),
        ('memberships', 'duplicate divisor members', lambda f: pd.concat([f, f.iloc[[0]]])),
        ('memberships', 'missing divisor member', lambda f: f.iloc[1:]),
        ('divisors', 'nonpositive input divisor', lambda f: f.assign(divisor=0.)),
        ('streams', 'nonfinite input maximum', lambda f: f.assign(maximum=np.inf)),
    ):
        obj = copy.copy(t)
        setattr(obj, attr, mutate(getattr(t, attr).copy()))
        rejected(label, obj.validate)
    obj = copy.copy(t)
    obj.memberships = t.memberships.copy()
    obj.memberships.loc[0, 'stream_key'] = t.streams[t.streams.role == 'changed'].stream_key.iloc[0]
    rejected('changed observation in divisor membership', obj.validate)
    obj = copy.copy(t)
    obj.targets = t.targets.copy()
    changed_key = t.streams[t.streams.role == 'changed'].stream_key.iloc[0]
    obj.targets.loc[obj.targets.stream_key == changed_key, 'tuning'] = True
    rejected('changed observation in tuning', obj.validate)
    # Perturb every changed outcome; stable-only fitting quantities must be identical.
    altered = copy.copy(t)
    altered.stream_index = t.stream_index.copy()
    changed = altered.stream_index.role == 'changed'
    altered.stream_index.loc[changed, 'maximum'] *= 3.
    altered.paths = {k: v * (3. if altered.stream_index.loc[k, 'role'] == 'changed' else 1.) for k, v in t.paths.items()}
    altered.target_groups = {k: v.copy() for k, v in t.target_groups.items()}
    for frame in altered.target_groups.values():
        frame['historical_fired'] = altered.stream_index.loc[frame.stream_key, 'maximum'].to_numpy() > frame.historical_threshold.to_numpy()
    compared = 0
    for p in t.points[t.points.include_influence].to_dict('records'):
        for rule in ('loco', 'scaling_only'):
            for omitted in (None, int(t.stream_index[t.stream_index.panel == p['panel']].checkpoint_seed.min())):
                a, ad, ax = engine.evaluate(t, p, rule, omitted)
                b, bd, bx = engine.evaluate(altered, p, rule, omitted)
                assert a == b
                for name in ('checkpoint_parameters', 'donor_records', 'calibration_records', 'tuning_constraints', 'tuning_streams'):
                    pd.testing.assert_frame_equal(ax[name], bx[name])
                assert np.array_equal(ad.threshold, bd.threshold)
                compared += 1
    passed.append(f'changed-outcome perturbation: {compared} complete fits unchanged')
    # Find a deterministic product boundary requiring an upward ULP (fixed finite search).
    rng = np.random.default_rng(19)
    for _ in range(1000):
        m, b = float(rng.uniform(.01, 100)), float(rng.uniform(.01, 100))
        if b * (m / b) < m:
            chosen, _, _ = engine.select(np.array([m]), np.array([b]), np.array(['frozen']), {'total': 0})
            assert chosen['boundary_ulp_increments'] > 0 and b * chosen['multiplier'] >= m
            passed.append('floating-point product boundary separately audited')
            break
    else:
        raise AssertionError('boundary fixture unavailable')
    return dict(status='PASS', checks=passed, check_count=len(passed), changed_outcome_fits=compared,
                scope='deterministic software fixtures; no simulation or collection')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.manifest)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
