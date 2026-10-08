"""Effect sizes and frozen finite-input sensitivity from processed R4 tables.

No numerical engine or historical output is modified. Every destination is new.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction as F
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from diagnostic import Tables, require

PRIMARY = 'gradual_updating_down'


def rows(path: Path) -> list[dict]:
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def js(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def sha(path: Path) -> str:
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def pid(source: str, control: str, omit: int | None) -> str:
    if control == 'loco' and omit is None:
        return source
    return source + ':' + control + ('' if omit is None else ':omit_' + str(omit))


class Writer:
    """Deterministic compressed CSVs, with one schema per new result group."""

    def __init__(self, root: Path):
        self.root = root
        self.handles: dict = {}
        self.writers: dict = {}
        self.counts: Counter = Counter()

    def put(self, name: str, row: dict) -> None:
        if name not in self.writers:
            path = self.root / name
            if name.endswith('.gz'):
                handle = io.TextIOWrapper(gzip.GzipFile(filename=str(path), mode='wb',
                                                      mtime=0, compresslevel=1), newline='')
            else:
                handle = path.open('w', newline='')
            self.handles[name] = handle
            self.writers[name] = csv.DictWriter(handle, fieldnames=list(row))
            self.writers[name].writeheader()
        self.writers[name].writerow(row)
        self.counts[name] += 1

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()


def effect_sizes(inputs: Path, reference: Path, out: Writer) -> dict:
    """Exact consequences of all 84 existing matched correction controls."""
    points = [p for p in rows(inputs / 'points.csv') if p['include_control'] == 'True']
    require(len(points) == 84, '84 control points required')
    rules = {r['point_id']: r for r in json.loads((reference / 'rules.json').read_text())}
    streams: dict = defaultdict(dict)
    for row in rows(reference / 'stream_reference.csv'):
        streams[row['point_id']][row['stream_key']] = row
    donors: dict = defaultdict(list)
    for row in rows(reference / 'donors.csv'):
        donors[row['point_id']].append(row)
    summaries = []
    for point in points:
        source = point['source_point_id']
        left, right = rules[source], rules[source + ':scaling_only']
        B = max(F(r['exact_level']) for r in donors[source])
        foreign = {r['target_checkpoint']: F(r['exact_level']) for r in donors[source]
                   if r['donor_type'] == 'foreign'}
        beta = min(foreign.values()) / B
        tau = F(right['exact_theta']) / F(left['exact_theta'])
        require(beta <= tau <= 1, 'exact multiplier bounds')
        exceptional = [c for c, v in foreign.items() if v < B]
        require(len(exceptional) <= 1, 'at most one exceptional checkpoint')
        counts: Counter = Counter()
        gain_ids, loss_ids = [], []
        for key, a in streams[source].items():
            b = streams[source + ':scaling_only'][key]
            H, S = F(a['exact_threshold']), F(b['exact_threshold'])
            M = F(float(a['maximum']))
            inside = a['checkpoint_seed'] in exceptional
            ratio = tau / beta if inside else tau
            require(S / H == ratio, 'exact threshold ratio')
            gain_interval = not inside and S < M <= H
            loss_interval = inside and H < M <= S
            gain, loss = M > S and M <= H, M > H and M <= S
            require((gain, loss) == (gain_interval, loss_interval), 'strict intervals')
            primary = a['condition'] == PRIMARY
            changed = a['role'] == 'changed'
            for scope, active in [('primary', primary), ('changed', changed), ('all', True)]:
                if active:
                    for name, value in [('gains', gain), ('losses', loss),
                                        ('loco_detections', M > H), ('scaling_detections', M > S),
                                        ('denominator', True)]:
                        counts[scope + '_' + name] += int(value)
            if primary and gain:
                gain_ids.append(key)
            if primary and loss:
                loss_ids.append(key)
            out.put('effect_intervals.csv.gz', dict(source_point_id=source, stream_key=key,
                    condition=a['condition'], role=a['role'], checkpoint_seed=a['checkpoint_seed'],
                    inside=inside, maximum=a['maximum'], exact_loco_threshold=str(H),
                    exact_scaling_threshold=str(S), exact_ratio=str(ratio),
                    exact_movement=str(S-H), gain_interval=gain_interval, loss_interval=loss_interval,
                    loco_fired=M > H, scaling_fired=M > S,
                    loco_delay=a['exact_delay'], scaling_delay=b['exact_delay']))
        summary = dict(source_point_id=source, panel=point['panel'], statistic=point['statistic'],
                       calibration=point['calibration'], budget=point['budget'],
                       maximizing_checkpoint=','.join(exceptional), exact_beta=str(beta), exact_tau=str(tau),
                       beta=float(beta), tau=float(tau), outside_ratio=float(tau),
                       inside_ratio=float(tau/beta), outside_fractional_reduction=float(1-tau),
                       inside_fractional_increase=float(tau/beta-1),
                       max_outside_reduction=float(1-beta), max_inside_ratio=float(1/beta),
                       **counts, primary_gained_ids=js(gain_ids), primary_lost_ids=js(loss_ids))
        out.put('effects.csv', summary)
        summaries.append(summary)
    return dict(controls=84, unchanged_primary_sets=sum(x['primary_gains'] == x['primary_losses'] == 0
                                                      for x in summaries),
                simultaneous_primary_gains_losses=sum(x['primary_gains'] > 0 and x['primary_losses'] > 0
                                                      for x in summaries))


class Study:
    def __init__(self, inputs: Path, protocol: dict):
        self.tables = Tables(inputs / 'manifest.json')
        self.protocol = protocol
        self.streams = self.tables.streams.to_dict('records')
        self.keys = [s['stream_key'] for s in self.streams]
        self.index = {k: i for i, k in enumerate(self.keys)}
        self.maximum = np.array([s['maximum'] for s in self.streams], dtype=float)
        self.seeds = np.array([int(s['checkpoint_seed']) for s in self.streams])
        self.modes = np.array([s['mode'] for s in self.streams])
        self.stable = np.array([s['role'] != 'changed' for s in self.streams])
        self.primary = np.array([s['condition'] == PRIMARY for s in self.streams])
        self.horizon = self.tables.manifest['horizon']
        self.path = np.full((len(self.keys), self.horizon), np.nan)
        self.has_path = np.zeros(len(self.keys), dtype=bool)
        for key, path in self.tables.paths.items():
            self.path[self.index[key]] = path
            self.has_path[self.index[key]] = True
        self.points = {p['source_point_id']: p for p in self.tables.points.to_dict('records')}
        require(all(c in self.points for c in protocol['cases']), 'unknown sensitivity case')
        self.relevant_groups = {(self.points[c]['panel'], self.points[c]['statistic']) for c in protocol['cases']}
        self.universe = sorted(s['stream_key'] for s in self.streams
                               if (s['panel'], s['statistic']) in self.relevant_groups
                               and (s['role'] != 'changed' or s['condition'] == PRIMARY))
        self.members = {k: [self.index[s] for s in g.stream_key]
                        for k, g in self.tables.memberships.groupby('divisor_id', sort=False)}
        selected = {(self.points[c]['panel'], self.points[c]['statistic'], self.points[c]['calibration'])
                    for c in protocol['cases']}
        self.divisors = [d for d in self.tables.divisors.to_dict('records')
                        if (d['panel'], d['statistic'], d['calibration']) in selected]
        self.configs = []
        for source in protocol['cases']:
            point = self.points[source]
            cal = self.tables.calibration_groups[point['panel'], point['statistic'], point['calibration']]
            target = self.tables.target_groups[source]
            for omit in protocol['omissions'][point['panel']]:
                c = cal[[self.seeds[self.index[k]] != omit for k in cal.stream_key]]
                t = target[[self.seeds[self.index[k]] != omit for k in target.stream_key]]
                ix = np.array([self.index[k] for k in t.stream_key])
                ci = np.array([self.index[k] for k in c.stream_key])
                for control in protocol['controls']:
                    self.configs.append(dict(point_id=pid(source, control, omit), source=source,
                        control=control, omit=omit, caps=json.loads(point['caps_json']),
                        ix=ix, ci=ci, td=list(t.divisor_id), cd=list(c.divisor_id),
                        tuning=np.array(t.tuning, dtype=bool),
                        seeds=sorted(set(self.seeds[ix].tolist()))))

    def perturb(self, factor: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        require(factor.shape == self.maximum.shape and bool(np.isfinite(factor).all())
                and bool((factor > 0).all()), 'positive finite factors and matching stream count')
        path = self.path * factor[:, None]
        maximum = self.maximum * factor
        maximum[self.has_path] = np.nanmax(path[self.has_path], axis=1)
        require(bool(np.isfinite(maximum).all() and (maximum > 0).all()), 'perturbed maxima domain')
        div = {}
        for d in self.divisors:
            v = maximum[self.members[d['divisor_id']]]
            require(len(v) == d['membership_count'] and len(v) in (6, 8), 'six/eight membership')
            family = d['calibration']
            if family == 'pooled':
                value = 1.
            elif family == 'arithmetic':
                value = float(np.mean(v))
            elif family == 'geometric':
                value = float(np.exp(np.mean(np.log(v))))
            elif family == 'quantile90':
                value = float(np.quantile(v, .9, method='linear'))
            else:
                raise ValueError('Unknown divisor family: ' + family)
            require(np.isfinite(value) and value > 0, 'perturbed divisor domain')
            div[d['divisor_id']] = value
        return maximum, path, div

    def fit(self, config: dict, maximum: np.ndarray, path: np.ndarray, div: dict) -> dict:
        ix, ci = config['ix'], config['ci']
        norm = maximum[ci] / np.array([div[k] for k in config['cd']])
        require(bool(np.isfinite(norm).all() and (norm > 0).all()), 'calibration domain')
        B = float(norm.max())
        foreign, donors = {}, {}
        for seed in config['seeds']:
            eligible = self.seeds[ci] != seed
            require(bool(eligible.any()), 'foreign calibration empty')
            foreign[seed] = float(norm[eligible].max())
            donors[str(seed)] = sorted(self.keys[j] for j in ci[eligible & (norm == foreign[seed])])
        d = np.array([div[k] for k in config['td']])
        base = np.array([foreign[s] for s in self.seeds[ix]]) * d if config['control'] == 'loco' else d
        tuning = config['tuning']
        ratio = maximum[ix[tuning]] / base[tuning]
        require(bool(np.isfinite(ratio).all() and (ratio > 0).all()), 'tuning domain')
        ti = ix[tuning]
        boundaries = []
        for mode, cap in config['caps'].items():
            mask = np.ones(len(ti), dtype=bool) if mode == 'total' else self.modes[ti] == mode
            require(type(cap) is int and 0 <= cap < int(mask.sum()), 'invalid remaining cap index')
            cutoff = float(np.sort(ratio[mask])[::-1][cap])
            boundaries.append(dict(constraint=mode, cap=cap, candidate=cutoff,
                ties=sorted(self.keys[j] for j in ti[mask & (ratio == cutoff)])))
        candidate = max(b['candidate'] for b in boundaries)
        k, steps = candidate, 0
        while True:
            fired = maximum[ti] > base[tuning] * k
            if all(int(fired.sum() if m == 'total' else fired[self.modes[ti] == m].sum()) <= cap
                   for m, cap in config['caps'].items()):
                break
            k = float(np.nextafter(k, np.inf))
            steps += 1
            require(steps <= 32 and np.isfinite(k), 'multiplier infeasible')
        for b in boundaries:
            b['binding'] = b['candidate'] == candidate
        threshold = base * k
        require(bool(np.isfinite(threshold).all() and (threshold > 0).all()), 'threshold domain')
        crossed = path[ix] > threshold[:, None]
        delay = np.where(crossed.any(axis=1), crossed.argmax(axis=1)+1, self.horizon+1)
        delay[~self.has_path[ix]] = -1
        fired = maximum[ix] > threshold
        primary = self.primary[ix]
        summary = dict(point_id=config['point_id'], source_point_id=config['source'], control=config['control'],
            omitted_checkpoint='' if config['omit'] is None else config['omit'], multiplier=k,
            ratio_candidate=candidate, ulp_increments=steps, global_level=B,
            theta=k*B if config['control'] == 'loco' else k, beta=min(foreign.values())/B,
            global_donors=js(sorted(self.keys[j] for j in ci[norm == B])), foreign_levels=js(foreign),
            foreign_donors=js(donors), boundaries=js(boundaries),
            primary_ids=js(sorted(self.keys[j] for j in ix[primary & fired])),
            primary_detections=int(fired[primary].sum()), primary_denominator=int(primary.sum()),
            primary_delay_sum=int(delay[primary].sum()),
            stable_ids=js(sorted(self.keys[j] for j in ix[tuning & fired])),
            frozen_alarms=int(fired[tuning & (self.modes[ix] == 'frozen')].sum()),
            updating_alarms=int(fired[tuning & (self.modes[ix] == 'updating')].sum()),
            stable_denominator=int(tuning.sum()))
        return dict(summary=summary, threshold=threshold, fired=fired, delay=delay, base=base, divisor=d,
                    boundaries=boundaries, foreign=foreign, donors=donors)

    def baseline(self, reference: Path) -> tuple[dict, list[str], dict]:
        maximum, path, div = self.perturb(np.ones(len(self.keys)))
        require(np.array_equal(maximum, self.maximum), 'zero maxima reproduction')
        for d in self.divisors:
            require(div[d['divisor_id']] == d['divisor'], 'zero divisor reproduction: ' + d['divisor_id'])
        rules = {r['point_id']: r for r in json.loads((reference/'rules.json').read_text())}
        old = defaultdict(dict)
        selected = {c['point_id'] for c in self.configs}
        for r in rows(reference/'stream_reference.csv'):
            if r['point_id'] in selected:
                old[r['point_id']][r['stream_key']] = r
        old_donors = defaultdict(list)
        for r in rows(reference/'donors.csv'):
            old_donors[r['point_id']].append(r)
        old_boundaries = defaultdict(list)
        for r in rows(reference/'boundaries.csv'):
            old_boundaries[r['point_id']].append(r)
        fits, singles = {}, set()
        decisions = 0
        for c in self.configs:
            fit = self.fit(c, maximum, path, div)
            fits[c['point_id']] = fit
            s, r = fit['summary'], rules[c['point_id']]
            require(s['multiplier'] == float(r['binary64_multiplier']) and
                    s['theta'] == float(r['binary64_theta']) and
                    s['ulp_increments'] == r['binary64_ulp_increments'], 'zero selector reproduction')
            for b, recorded in zip(fit['boundaries'], old_boundaries[c['point_id']], strict=True):
                require(b['candidate'] == float(recorded['binary64_candidate']) and
                        b['ties'] == json.loads(recorded['binary64_ties']) and
                        b['binding'] == (recorded['binary64_binding'] == 'True'), 'zero boundary reproduction')
                if b['binding']:
                    singles.update(b['ties'])
            for recorded in old_donors[c['point_id']]:
                seed = int(recorded['target_checkpoint'])
                level = s['global_level'] if recorded['donor_type'] == 'global' else fit['foreign'][seed]
                ties = json.loads(s['global_donors']) if recorded['donor_type'] == 'global' else fit['donors'][str(seed)]
                require(level == float(recorded['binary64_level']) and ties == json.loads(recorded['binary64_ties']),
                        'zero donor reproduction')
                singles.update(ties)
            require(len(old[c['point_id']]) == len(c['ix']), 'zero target coverage')
            for j, i in enumerate(c['ix']):
                r = old[c['point_id']][self.keys[i]]
                expected_delay = int(r['binary64_delay']) if r['binary64_delay'] else -1
                require(fit['threshold'][j] == float(r['binary64_threshold']) and
                        fit['fired'][j] == (r['binary64_fired'] == 'True') and fit['delay'][j] == expected_delay,
                        'zero decision/delay reproduction: ' + self.keys[i])
                decisions += 1
                if c['tuning'][j] and self.seeds[i] == 808:
                    singles.add(self.keys[i])
        require(singles <= set(self.universe), 'single stream universe')
        return fits, sorted(singles), dict(status='PASS', rules=len(fits), decisions=decisions,
                    recomputed_divisors=len(div), single_streams=sorted(singles),
                    checks='exact maxima/divisors/selectors/ULPs/donors/boundaries/thresholds/decisions/available delays')


def paired(study: Study, fits: dict, scenario: str, out: Writer) -> None:
    configs = {c['point_id']: c for c in study.configs}
    pairs = []
    for c in study.configs:
        if c['control'] == 'scaling_only':
            pairs.append(('correction_removal', pid(c['source'], 'loco', c['omit']), c['point_id']))
        if c['omit'] is not None:
            pairs.append(('omit_808', pid(c['source'], c['control'], None), c['point_id']))
    for kind, aid, bid in pairs:
        a, b = fits[aid], fits[bid]
        ca, cb = configs[aid], configs[bid]
        amap = {int(i): j for j, i in enumerate(ca['ix'])}
        for scope in ('primary', 'tuning'):
            mask = study.primary[cb['ix']] if scope == 'primary' else cb['tuning']
            jb = np.flatnonzero(mask)
            ia = np.array([amap[int(cb['ix'][j])] for j in jb])
            left, right = a['fired'][ia], b['fired'][jb]
            gain = sorted(study.keys[i] for i in cb['ix'][jb][right & ~left])
            loss = sorted(study.keys[i] for i in cb['ix'][jb][left & ~right])
            changed_delays = sorted(study.keys[i] for i in cb['ix'][jb][a['delay'][ia] != b['delay'][jb]])
            out.put('paired.csv', dict(scenario=scenario, comparison=kind, scope=scope, left=aid, right=bid,
                denominator=len(jb), left_detections=int(left.sum()), right_detections=int(right.sum()),
                gains=len(gain), losses=len(loss), gained_ids=js(gain), lost_ids=js(loss),
                changed_delay_ids=js(changed_delays), available_delays=int((b['delay'][jb] >= 0).sum())))


def run(inputs: Path, reference: Path, protocol_path: Path, output: Path) -> dict:
    require(not output.exists(), 'output directory must be fresh')
    p = json.loads(protocol_path.read_text())
    require(p['schema_version'] == 1 and p['magnitudes'] == [0, 1e-6, 1e-4, 1e-3, 1e-2, .05], 'protocol magnitudes')
    require(p['seed'] == 20261006 and p['joint_vectors'] == 100 and p['bit_generator'] == 'PCG64', 'protocol RNG')
    require(p['scopes'] == ['stable_only', 'stable_plus_primary'] and p['controls'] == ['loco', 'scaling_only'], 'protocol scope')
    study = Study(inputs, p)
    baseline, singles, gate = study.baseline(reference)
    output.mkdir(parents=True, exist_ok=False)
    out = Writer(output)
    effect = effect_sizes(inputs, reference, out)
    (output/'zero_gate.json').write_text(json.dumps(gate, indent=2)+'\n')
    (output/'protocol.json').write_bytes(protocol_path.read_bytes())
    universe = study.universe
    ui = np.array([study.index[k] for k in universe])
    rng = np.random.Generator(np.random.PCG64(p['seed']))
    joint = rng.uniform(-1, 1, (p['joint_vectors'], len(universe)))
    vectors = []
    for key in singles:
        for sign in (-1, 1):
            vector = np.zeros(len(universe))
            vector[universe.index(key)] = sign
            vectors.append(('single', f's{len(vectors):03d}', key, sign, vector))
    vectors += [('joint', f'j{n:03d}', '', 0, vector) for n, vector in enumerate(joint)]
    for kind, name, key, sign, vector in vectors:
        for stream, u in zip(universe, vector, strict=True):
            out.put('vectors.csv.gz', dict(vector=name, kind=kind, selected_stream=key, sign=sign, stream_key=stream, u=float(u)))
    scenarios = 0
    persistence: dict = defaultdict(Counter)
    for magnitude_index, epsilon in enumerate(p['magnitudes']):
        for scope in p['scopes']:
            for kind, name, key, sign, vector in vectors:
                scenario = f'e{magnitude_index}:{scope}:{name}'
                factors = np.ones(len(study.keys))
                factors[ui] = 1 + epsilon*vector
                if scope == 'stable_only':
                    factors[~study.stable] = 1
                maximum, path, div = study.perturb(factors)
                for i in ui:
                    out.put('factors.csv.gz', dict(scenario=scenario, stream_key=study.keys[i], factor=float(factors[i])))
                for identity, value in div.items():
                    out.put('divisors.csv.gz', dict(scenario=scenario, divisor_id=identity, divisor=value))
                fits = {}
                for c in study.configs:
                    fit = study.fit(c, maximum, path, div)
                    fits[c['point_id']] = fit
                    s, base = fit['summary'], baseline[c['point_id']]['summary']
                    changes = dict(donor_identity_changed=s['global_donors'] != base['global_donors'] or s['foreign_donors'] != base['foreign_donors'],
                        boundary_identity_changed=[(b['ties'], b['binding']) for b in fit['boundaries']] !=
                                                  [(b['ties'], b['binding']) for b in baseline[c['point_id']]['boundaries']],
                        primary_set_changed=s['primary_ids'] != base['primary_ids'],
                        stable_set_changed=s['stable_ids'] != base['stable_ids'])
                    out.put('fits.csv', dict(scenario=scenario, magnitude=epsilon, input_scope=scope, kind=kind,
                                             vector=name, **s, **changes))
                    counter = persistence[epsilon, scope, kind, c['point_id']]
                    counter['scenarios'] += 1
                    for k, value in changes.items():
                        counter[k] += value
                    for j, i in enumerate(c['ix']):
                        if not (study.primary[i] or c['tuning'][j]):
                            continue
                        H, M = float(fit['threshold'][j]), float(maximum[i])
                        delay = int(fit['delay'][j])
                        out.put('decisions.csv.gz', dict(scenario=scenario, point_id=c['point_id'], stream_key=study.keys[i],
                            scope='primary' if study.primary[i] else 'tuning', maximum=M, divisor=float(fit['divisor'][j]),
                            base=float(fit['base'][j]), threshold=H, fired=bool(fit['fired'][j]),
                            delay='' if delay < 0 else delay, signed_margin=M-H, absolute_margin=abs(M-H),
                            relative_margin=(M-H)/H))
                paired(study, fits, scenario, out)
                scenarios += 1
            print(f'Completed epsilon={epsilon:g}, scope={scope}: {scenarios} scenarios', flush=True)
    for (epsilon, scope, kind, point), counts in persistence.items():
        out.put('persistence.csv', dict(magnitude=epsilon, input_scope=scope, kind=kind, point_id=point, **counts))
    out.close()
    summary = dict(status='PASS', effect_sizes=effect, zero_gate=gate, scenarios=scenarios,
        vectors=len(vectors), single_vectors=2*len(singles), joint_vectors=100, universe_streams=len(universe),
        fitted_rules=scenarios*len(study.configs), rows=dict(out.counts),
        interpretation=p['interpretation'], protocol_sha256=sha(protocol_path))
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    sources = {f'inputs/{f.name}': sha(f) for f in sorted(inputs.iterdir()) if f.is_file()}
    sources.update({f'reference/{f.name}': sha(f) for f in sorted(reference.iterdir()) if f.is_file()})
    sources['sensitivity_protocol.json'] = sha(protocol_path)
    (output/'input_hashes.json').write_text(json.dumps(sources, indent=2)+'\n')
    (output/'manifest.json').write_text(json.dumps({f.name:sha(f) for f in sorted(output.iterdir()) if f.is_file()},indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.inputs, args.reference, args.protocol, args.output), indent=2))
