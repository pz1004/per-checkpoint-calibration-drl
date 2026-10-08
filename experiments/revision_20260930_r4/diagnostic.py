"""Table-driven calibration diagnostics, independent of the RL study loader.

CLI: python diagnostic.py --manifest inputs/manifest.json --output NEW_DIRECTORY
Required tables and numerical conventions are documented in DIAGNOSTIC_INTERFACE.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def positive(values, identity: str) -> None:
    x = np.asarray(values, dtype=float)
    require(bool(np.isfinite(x).all() and (x > 0).all()), 'positive finite domain: ' + identity)


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision='round_trip', keep_default_na=False, low_memory=False)


def select(maxima: np.ndarray, base: np.ndarray, modes: np.ndarray,
           caps: dict[str, int]) -> tuple[dict, list[dict], pd.DataFrame]:
    """Historical order statistic followed by upward ULPs, strict M > base*k."""
    positive(maxima, 'tuning maxima')
    positive(base, 'tuning base')
    require(len(maxima) == len(base) == len(modes), 'tuning lengths')
    require(bool(caps), 'no selection constraint')
    ratio = maxima / base
    positive(ratio, 'stable ratios')
    constraints = []
    for mode, cap in caps.items():
        mask = np.ones(len(ratio), dtype=bool) if mode == 'total' else modes == mode
        require(isinstance(cap, int) and 0 <= cap < int(mask.sum()), 'infeasible cap: ' + mode)
        cutoff = float(np.sort(ratio[mask])[::-1][cap])
        constraints.append(dict(constraint=mode, cap=cap, denominator=int(mask.sum()),
                                candidate_cutoff=cutoff, descending_order_position=cap + 1))
    candidate = max(x['candidate_cutoff'] for x in constraints)

    def feasible(k: float) -> bool:
        fired = maxima > base * k
        return all(int(fired.sum() if mode == 'total' else fired[modes == mode].sum()) <= cap
                   for mode, cap in caps.items())

    final, steps = candidate, 0
    while not feasible(final):
        final = float(np.nextafter(final, np.inf))
        steps += 1
        require(steps <= 32 and np.isfinite(final), 'infeasible floating-point selection')
    initial, achieved = maxima > base * candidate, maxima > base * final
    for c in constraints:
        mask = np.ones(len(ratio), dtype=bool) if c['constraint'] == 'total' else modes == c['constraint']
        c.update(binding=c['candidate_cutoff'] == candidate,
                 initial_alarms=int(initial[mask].sum()), achieved_alarms=int(achieved[mask].sum()),
                 boundary_ties=int((ratio[mask] == c['candidate_cutoff']).sum()))
    ranks = pd.DataFrame(dict(stable_ratio=ratio, selected_boundary=ratio == candidate,
                              rank_total_min=pd.Series(ratio).rank(method='min', ascending=False).astype(int),
                              rank_total_max=pd.Series(ratio).rank(method='max', ascending=False).astype(int)))
    ranks['rank_mode_min'] = 0
    ranks['rank_mode_max'] = 0
    ranks['mode_candidate_cutoff'] = np.nan
    ranks['mode_boundary'] = False
    for mode in sorted(set(modes)):
        mask = modes == mode
        for method in ('min', 'max'):
            ranks.loc[mask, 'rank_mode_' + method] = pd.Series(ratio[mask]).rank(method=method, ascending=False).astype(int).to_numpy()
        own = next((c for c in constraints if c['constraint'] == mode), None)
        # Under a total cap, also expose each mode's cap-index candidate; it is not a constraint.
        cap = caps.get(mode, caps.get('total'))
        cutoff = float(np.sort(ratio[mask])[::-1][cap]) if cap < int(mask.sum()) else np.nan
        ranks.loc[mask, 'mode_candidate_cutoff'] = cutoff
        ranks.loc[mask, 'mode_boundary'] = ratio[mask] == cutoff
        if own:
            require(cutoff == own['candidate_cutoff'], 'mode cutoff mismatch')
    result = dict(multiplier=final, ratio_candidate=candidate, boundary_ulp_increments=steps,
                  ratio_ties_at_candidate=int((ratio == candidate).sum()),
                  actual_boundary_equalities=int((maxima == base * final).sum()))
    return result, constraints, ranks


def first(path: np.ndarray | None, threshold: float, maximum: float, horizon: int) -> tuple[bool, float]:
    if path is None:
        return bool(maximum > threshold), np.nan
    # Indexed NaN entries represent abstentions and preserve their original positions.
    crossed = np.flatnonzero(path > threshold)
    return bool(len(crossed)), float(crossed[0] + 1 if len(crossed) else horizon + 1)


class Tables:
    """Validated manifest input. No changed record is used to fit any quantity."""

    def __init__(self, manifest: Path):
        self.manifest = json.loads(manifest.read_text())
        require(self.manifest['schema_version'] == 1, 'manifest version')
        require(isinstance(self.manifest['horizon'], int) and self.manifest['horizon'] > 0, 'invalid horizon')
        require(self.manifest.get('miss_delay', self.manifest['horizon'] + 1) == self.manifest['horizon'] + 1, 'invalid miss convention')
        root = manifest.parent
        for name in ('streams', 'divisors', 'memberships', 'calibration', 'targets', 'points'):
            setattr(self, name, read(root / self.manifest['tables'][name]))
        self.paths = {}
        if self.manifest.get('indexed_paths'):
            paths = read(root / self.manifest['indexed_paths'])
            require(not paths.duplicated(['stream_key', 'index']).any(), 'duplicate path index')
            for key, group in paths.groupby('stream_key', sort=False):
                group = group.sort_values('index')
                require(group['index'].tolist() == list(range(1, self.manifest['horizon'] + 1)), 'path index coverage')
                values = pd.to_numeric(group.value, errors='coerce').to_numpy()
                # Empty entries are explicit abstentions; reject arbitrary nonnumeric values/infinities.
                require(all(v == '' or np.isfinite(float(v)) for v in group.value), 'invalid path value')
                self.paths[key] = values
        self.validate()
        self.stream_index = self.streams.set_index('stream_key', drop=False)
        self.divisor_index = self.divisors.set_index('divisor_id', drop=False)
        self.target_groups = {k: g for k, g in self.targets.groupby('source_point_id', sort=False)}
        self.calibration_groups = {k: g for k, g in self.calibration.groupby(['panel', 'statistic', 'calibration'], sort=False)}

    def validate(self) -> None:
        s, d = self.streams, self.divisors
        require(not s.stream_key.duplicated().any(), 'duplicate stream identity')
        require(not s.duplicated(['panel', 'statistic', 'stream_id']).any(), 'duplicate stream tuple')
        require(not s.duplicated(['panel', 'statistic', 'checkpoint_seed', 'replicate', 'condition', 'role']).any(), 'duplicate semantic stream identity')
        require(not d.divisor_id.duplicated().any(), 'duplicate divisor identity')
        require(not self.points.source_point_id.duplicated().any(), 'duplicate operating-point identity')
        require(not self.targets.duplicated(['source_point_id', 'stream_key']).any(), 'duplicate target identity')
        require(not self.memberships.duplicated(['divisor_id', 'stream_key']).any(), 'duplicate membership')
        require(not self.calibration.duplicated(['panel', 'statistic', 'calibration', 'stream_key']).any(), 'duplicate calibration identity')
        positive(s.maximum, 'maxima')
        positive(d.divisor, 'divisors')
        require(set(s.role) <= {'original_stable', 'additional_stable', 'changed'}, 'invalid stream role')
        require(set(s['mode']) <= {'frozen', 'updating'}, 'invalid mode')
        si, di = s.set_index('stream_key'), d.set_index('divisor_id')
        for table in (self.memberships, self.calibration, self.targets):
            require(set(table.stream_key) <= set(si.index), 'unknown stream member')
            require(set(table.divisor_id) <= set(di.index), 'unknown divisor member')
        require(set(self.targets.source_point_id) == set(self.points.source_point_id), 'target point coverage')
        for row in self.memberships.itertuples():
            stream, divisor = si.loc[row.stream_key], di.loc[row.divisor_id]
            require(stream.role == 'original_stable', 'changed/additional divisor member')
            require(all(stream[k] == divisor[k] for k in ('panel', 'statistic', 'checkpoint_seed')), 'foreign divisor member')
        for row in d.itertuples():
            members = self.memberships[self.memberships.divisor_id == row.divisor_id]
            eligible = s[(s.panel == row.panel) & (s.statistic == row.statistic) &
                         (s.checkpoint_seed == row.checkpoint_seed) & (s.role == 'original_stable')]
            if row.excluded_replicate != -1:
                eligible = eligible[eligible.replicate != row.excluded_replicate]
            require(set(members.stream_key) == set(eligible.stream_key) and len(members) > 0, 'invalid divisor membership')
            require(len(members) == row.membership_count, 'divisor membership count')
            if row.calibration == 'pooled':
                require(row.divisor == 1, 'pooled divisor')
        for row in self.calibration.itertuples():
            stream, divisor = si.loc[row.stream_key], di.loc[row.divisor_id]
            require(stream.role == 'original_stable' and divisor.excluded_replicate == -1, 'calibration role/membership')
            require(all(getattr(row, k) == stream[k] == divisor[k] for k in ('panel', 'statistic')), 'calibration group')
            require(row.calibration == divisor.calibration and stream.checkpoint_seed == divisor.checkpoint_seed, 'calibration divisor')
        calibration_groups = {k: g for k, g in self.calibration.groupby(['panel', 'statistic', 'calibration'], sort=False)}
        # Validate every referenced group before Writer can create any output.
        for group in self.points[['panel', 'statistic', 'calibration']].drop_duplicates().itertuples(index=False, name=None):
            identity = f'calibration group {group!r}'
            require(group in calibration_groups, identity + ': missing calibration group')
            cal = calibration_groups[group]
            expected = s[(s.panel == group[0]) & (s.statistic == group[1]) & (s.role == 'original_stable')]
            records = si.loc[cal.stream_key]
            missing = sorted(set(expected.stream_key) - set(cal.stream_key))
            extra = sorted(set(cal.stream_key) - set(expected.stream_key))
            missing_checkpoints = sorted(set(expected.checkpoint_seed) - set(records.checkpoint_seed))
            require(not missing and not extra and len(expected) > 0,
                    f'{identity}: incomplete calibration coverage; missing checkpoints={missing_checkpoints}; '
                    f'missing streams={missing}; extra streams={extra}')
            checkpoint_ids = pd.DataFrame({'checkpoint_seed': records.checkpoint_seed.to_numpy(),
                                           'divisor_id': cal.divisor_id.to_numpy()})
            for seed, checkpoint in checkpoint_ids.groupby('checkpoint_seed', sort=False):
                require(checkpoint.divisor_id.nunique() == 1,
                        f'{identity}: inconsistent full-membership divisor identity at checkpoint {seed}')
            positive(records.maximum.to_numpy() / di.loc[cal.divisor_id, 'divisor'].to_numpy(),
                     identity + ': calibration ratios')
        for p in self.points.itertuples():
            require(not p.include_influence or p.include_control, 'influence requires both full-panel rules')
            require(p.budget != 'native' or (not p.include_control and not p.include_influence), 'native is fixed only')
            if p.budget == 'native':
                positive([p.multiplier], 'native multiplier')
            target = self.targets[self.targets.source_point_id == p.source_point_id]
            records = si.loc[target.stream_key]
            require(all(records.panel == p.panel) and all(records.statistic == p.statistic), 'point stream group')
            require(set(records.checkpoint_seed) == set(s[(s.panel == p.panel) & (s.statistic == p.statistic) & (s.role == 'original_stable')].checkpoint_seed), 'checkpoint coverage')
            for t, r in zip(target.itertuples(), records.itertuples(), strict=True):
                div = di.loc[t.divisor_id]
                require(div.calibration == p.calibration and div.checkpoint_seed == r.checkpoint_seed and div.statistic == p.statistic and div.panel == p.panel, 'target divisor identity')
                rep = r.replicate if p.target_budget_policy == 'matched_six' or (p.target_budget_policy == 'historical_six_stable_eight_changed' and r.role == 'original_stable') else -1
                require(div.excluded_replicate == rep, 'target budget membership')
                require(t.tuning == (r.role == p.tuning_role), 'invalid tuning role')
            require(p.target_budget_policy in {'matched_six', 'additional_eight', 'historical_six_stable_eight_changed'}, 'target budget policy')
            require(p.tuning_role in {'original_stable', 'additional_stable'}, 'changed tuning role')
            if p.budget != 'native':
                require(bool(json.loads(p.caps_json)), 'missing caps')
        for key, path in self.paths.items():
            require(key in si.index, 'unknown indexed path')
            require(np.isfinite(path).any() and float(np.nanmax(path)) == float(si.loc[key, 'maximum']), 'path maximum mismatch')


class Writer:
    def __init__(self, output: Path):
        output.mkdir(parents=True, exist_ok=False)
        self.output, self.counts, self.columns = output, {}, {}

    def put(self, name: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        first_write = name not in self.counts
        if first_write:
            self.columns[name] = list(frame.columns)
        require(list(frame.columns) == self.columns[name], 'inconsistent output schema ' + name)
        frame.to_csv(self.output / (name + '.csv'), index=False, header=first_write,
                     mode='w' if first_write else 'a', float_format='%.17g', lineterminator='\n')
        self.counts[name] = self.counts.get(name, 0) + len(frame)


def evaluate(t: Tables, p: dict, control: str, omitted: int | None) -> tuple[dict, pd.DataFrame, dict[str, pd.DataFrame]]:
    source = p['source_point_id']
    kind = 'existing' if omitted is None and control == 'loco' else ('control' if omitted is None else 'deletion')
    pid = source if kind == 'existing' else source + ':' + control + ('' if omitted is None else ':omit_' + str(omitted))
    meta = {k: p[k] for k in ('source_point_id', 'panel', 'statistic', 'calibration', 'family', 'budget', 'target_budget_policy')}
    meta.update(point_id=pid, control_type=control, omitted_checkpoint='' if omitted is None else omitted,
                scenario_id=source + (':full' if omitted is None else ':omit_' + str(omitted)), result_kind=kind,
                divisor_identity='explicit_divisor_id_and_membership_table')
    targets = t.target_groups[source].copy()
    records = t.stream_index.loc[targets.stream_key].reset_index(drop=True)
    targets = targets.reset_index(drop=True)
    require(omitted is None or omitted in set(records.checkpoint_seed), 'unknown omitted checkpoint')
    keep = records.checkpoint_seed != omitted
    records, targets = records[keep].reset_index(drop=True), targets[keep].reset_index(drop=True)
    cal = t.calibration_groups[p['panel'], p['statistic'], p['calibration']].copy()
    cr = t.stream_index.loc[cal.stream_key].reset_index(drop=True)
    cal = cal.reset_index(drop=True)
    keep = cr.checkpoint_seed != omitted
    cr, cal = cr[keep].reset_index(drop=True), cal[keep].reset_index(drop=True)
    require(len(cr.checkpoint_seed.unique()) >= 2, 'infeasible foreign calibration')
    cr['divisor_id'] = cal.divisor_id.to_numpy()
    cr['divisor'] = t.divisor_index.loc[cal.divisor_id, 'divisor'].to_numpy()
    cr['normalized_maximum'] = cr.maximum.to_numpy() / cr.divisor.to_numpy()
    positive(cr.normalized_maximum, 'calibration ratios')
    global_level = float(cr.normalized_maximum.max())
    global_donors = cr[cr.normalized_maximum == global_level]
    levels, params, donors = {}, [], []
    for seed in sorted(records.checkpoint_seed.unique()):
        checkpoint_ratio = float(cr[cr.checkpoint_seed == seed].normalized_maximum.max())
        positive([checkpoint_ratio], f'calibration group {(p["panel"], p["statistic"], p["calibration"])!r}: retained checkpoint {seed} ratio')
        foreign = cr[cr.checkpoint_seed != seed]
        peak = float(foreign.normalized_maximum.max())
        levels[seed] = peak
        params.append(dict(checkpoint_seed=seed, checkpoint_ratio=checkpoint_ratio,
                           global_level=global_level, foreign_level=peak, loco_correction=peak / global_level,
                           foreign_calibration_records=len(foreign), global_donor_streams=len(global_donors),
                           global_donor_checkpoints=global_donors.checkpoint_seed.nunique()))
        for donor_type, group in (('global', global_donors), ('foreign', foreign[foreign.normalized_maximum == peak])):
            for donor in group.itertuples():
                donors.append(dict(target_checkpoint=seed, donor_type=donor_type, stream_id=donor.stream_id,
                                   stream_key=donor.stream_key, checkpoint_seed=donor.checkpoint_seed,
                                   divisor_id=donor.divisor_id, normalized_maximum=donor.normalized_maximum,
                                   tie_stream_count=len(group), tie_checkpoint_count=group.checkpoint_seed.nunique()))
    d = t.divisor_index.loc[targets.divisor_id, 'divisor'].to_numpy(dtype=float)
    foreign = np.array([levels[s] for s in records.checkpoint_seed])
    base = foreign * d if control == 'loco' else d.copy()
    positive(base, 'threshold base')
    mask = targets.tuning.to_numpy(dtype=bool)
    maxima, modes = records.maximum.to_numpy(dtype=float), records['mode'].to_numpy()
    fixed = p['budget'] == 'native'
    if fixed:
        require(control == 'loco' and omitted is None, 'native is fixed, no native control/deletion')
        selected = dict(multiplier=float(p['multiplier']), ratio_candidate=np.nan, boundary_ulp_increments=0,
                        ratio_ties_at_candidate=0, actual_boundary_equalities=int((maxima[mask] == base[mask] * float(p['multiplier'])).sum()))
        constraints, ranks = [], pd.DataFrame()
    else:
        selected, constraints, ranks = select(maxima[mask], base[mask], modes[mask], json.loads(p['caps_json']))
        if kind == 'existing' and p.get('multiplier', '') != '':
            require(selected['multiplier'] == float(p['multiplier']), 'historical tuning changed: ' + source)
    k = selected['multiplier']
    theta = k * global_level if control == 'loco' else k
    threshold = base * k  # Accepted historical multiplication order, deliberately not theta*d*b.
    coordinate = (theta * d) * (foreign / global_level if control == 'loco' else 1.)
    positive(threshold, 'effective thresholds')
    fired_delay = [first(t.paths.get(key), h, m, t.manifest['horizon']) for key, h, m in zip(records.stream_key, threshold, maxima, strict=True)]
    decision = records[['stream_key', 'stream_id', 'checkpoint_seed', 'replicate', 'condition', 'mode', 'role', 'maximum']].copy()
    decision['tuning'] = mask
    decision['divisor_id'] = targets.divisor_id.to_numpy()
    decision['target_membership_count'] = t.divisor_index.loc[targets.divisor_id, 'membership_count'].to_numpy()
    decision['divisor'] = d
    decision['global_level'] = global_level
    decision['foreign_level'] = foreign
    decision['loco_correction'] = foreign / global_level
    decision['applied_correction'] = foreign / global_level if control == 'loco' else 1.
    decision['theta'] = theta
    decision['multiplier'] = k
    decision['base_threshold'] = base
    decision['threshold'] = threshold
    decision['reparameterized_threshold'] = coordinate
    decision['coordinate_absolute_difference'] = coordinate - threshold
    decision['coordinate_decision_differs'] = (maxima > coordinate) != (maxima > threshold)
    decision['fired'] = [x[0] for x in fired_delay]
    decision['restricted_delay'] = [x[1] for x in fired_delay]
    decision['stopping_time_available'] = [key in t.paths for key in records.stream_key]
    decision['first_crossing'] = np.where(decision.fired, decision.restricted_delay, np.nan)
    if kind == 'existing' and 'historical_threshold' in targets:
        require(np.array_equal(threshold, targets.historical_threshold.to_numpy(dtype=float)), 'historical threshold changed: ' + source)
        require(np.array_equal(decision.fired, targets.historical_fired), 'historical decisions changed: ' + source)
    if len(ranks):
        ranks = pd.concat([decision.loc[mask, ['stream_key', 'stream_id', 'checkpoint_seed', 'divisor_id', 'mode', 'maximum', 'base_threshold']].reset_index(drop=True), ranks], axis=1)
        ranks['theta_ratio'] = maxima[mask] / (d[mask] * (foreign[mask] / global_level if control == 'loco' else 1.))
        ranks['final_multiplier'] = k
        ranks['final_theta'] = theta
        ranks['final_fired'] = decision.loc[mask, 'fired'].to_numpy()
    point = dict(meta, selection_type='fixed_multiplier' if fixed else 'stable_only_retuned', **selected,
                 global_level=global_level, theta=theta, stable_denominator=int(mask.sum()),
                 frozen_alarms=int(decision.loc[mask & (modes == 'frozen'), 'fired'].sum()),
                 updating_alarms=int(decision.loc[mask & (modes == 'updating'), 'fired'].sum()),
                 frozen_denominator=int((mask & (modes == 'frozen')).sum()),
                 updating_denominator=int((mask & (modes == 'updating')).sum()),
                 coordinate_decision_differences=int(decision.coordinate_decision_differs.sum()),
                 maximum_coordinate_absolute_difference=float(np.abs(coordinate - threshold).max()))
    extras = dict(checkpoint_parameters=pd.DataFrame(params), donor_records=pd.DataFrame(donors),
                  tuning_constraints=pd.DataFrame(constraints), tuning_streams=ranks,
                  calibration_records=cr[['stream_key', 'stream_id', 'checkpoint_seed', 'divisor_id', 'maximum', 'divisor', 'normalized_maximum']])
    for frame in [decision, *extras.values()]:
        for key, value in reversed(list(meta.items())):
            frame.insert(0, key, value)
    return point, decision, extras


def summarize(decisions: pd.DataFrame, reference: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = reference.set_index('stream_key').loc[decisions.stream_key].reset_index(drop=True)
    d = decisions.copy()
    d['reference_threshold'] = reference.threshold.to_numpy()
    d['reference_fired'] = reference.fired.to_numpy()
    d['reference_delay'] = reference.restricted_delay.to_numpy()
    d['gained'] = d.fired & ~d.reference_fired
    d['lost'] = ~d.fired & d.reference_fired
    d['delay_difference'] = d.restricted_delay - d.reference_delay
    meta = {k: d.iloc[0][k] for k in ('point_id', 'source_point_id', 'scenario_id', 'panel', 'statistic', 'calibration', 'family', 'budget', 'target_budget_policy', 'control_type', 'omitted_checkpoint', 'result_kind', 'divisor_identity')}
    rows = []
    for condition, group in d.groupby('condition', sort=True):
        rows.append(dict(meta, condition=condition, detections=int(group.fired.sum()), denominator=len(group),
                         detected_checkpoints=group.loc[group.fired, 'checkpoint_seed'].nunique(),
                         checkpoint_denominator=group.checkpoint_seed.nunique(),
                         detected_checkpoint_ids=json.dumps(sorted(map(int, group.loc[group.fired, 'checkpoint_seed'].unique()))),
                         restricted_delay=float(group.restricted_delay.mean()),
                         delay_available_streams=int(group.stopping_time_available.sum()),
                         reference_detections=int(group.reference_fired.sum()),
                         reference_detected_checkpoints=group.loc[group.reference_fired, 'checkpoint_seed'].nunique(),
                         reference_delay=float(group.reference_delay.mean()),
                         gained=int(group.gained.sum()), lost=int(group.lost.sum()),
                         gained_stream_ids=json.dumps(sorted(group.loc[group.gained, 'stream_id'])),
                         lost_stream_ids=json.dumps(sorted(group.loc[group.lost, 'stream_id'])),
                         reference_point_id=reference.point_id.iloc[0]))
    return pd.DataFrame(rows), d


def run(manifest: Path, output: Path) -> dict:
    t, writer = Tables(manifest), Writer(output)
    full = {}
    points = t.points.to_dict('records')

    def emit(p: dict, control: str, omitted: int | None) -> None:
        point, decisions, extras = evaluate(t, p, control, omitted)
        key = p['source_point_id'], control
        reference = (decisions if control == 'loco' else full[p['source_point_id'], 'loco']) if omitted is None else full[key]
        summary, paired = summarize(decisions, reference)
        writer.put('operating_points', pd.DataFrame([point]))
        writer.put('condition_summary', summary)
        writer.put('stream_decisions', paired)
        for name, frame in extras.items():
            writer.put(name, frame)
        if omitted is None:
            full[key] = decisions

    for p in points:
        emit(p, 'loco', None)
        if p['include_control']:
            emit(p, 'scaling_only', None)
    for p in points:
        if p['include_influence']:
            seeds = sorted(full[p['source_point_id'], 'loco'].checkpoint_seed.unique())
            for seed in seeds:
                for control in ('loco', 'scaling_only'):
                    emit(p, control, int(seed))
            print('Completed influence: ' + p['source_point_id'], flush=True)
    # Direct calibration-family comparisons at identical settings and control type.
    for p in points:
        if p['family'] not in ('matched_target_budget', 'additional_stable'):
            continue
        anchors = [a for a in points if all(a[k] == p[k] for k in ('panel', 'statistic', 'family', 'budget')) and a['calibration'] == 'arithmetic']
        if len(anchors) != 1:
            continue
        for control in ('loco', 'scaling_only'):
            key, anchor = (p['source_point_id'], control), (anchors[0]['source_point_id'], control)
            if key in full and anchor in full:
                summary, paired = summarize(full[key], full[anchor])
                writer.put('calibration_comparisons', summary)
                writer.put('calibration_gain_loss_streams', paired[paired.gained | paired.lost])
    report = dict(schema_version=1, counts=writer.counts,
                  input_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  interpretation='retrospective finite-panel influence; unchanged integer caps; no population interval',
                  outputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob('*.csv'))})
    (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.output), indent=2))
