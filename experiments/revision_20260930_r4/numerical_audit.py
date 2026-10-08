"""Independent exact-rational propagation audit of 175 existing fitted rules.

Only the Python standard library is used. Recorded binary64 inputs, including
supplied divisors, are interpreted as exact rationals; divisors are not refitted.
Historical binary64 outputs are verified independently and never overwritten.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path


def rows(path: Path) -> list[dict]:
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def rational(value: str | float) -> F:
    return F.from_float(float(value))


def truth(value: str) -> bool:
    assert value in ('True', 'False'), value
    return value == 'True'


def dump(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def select(ratios: dict, streams: dict, caps: dict) -> tuple:
    constraints = []
    for mode, cap in caps.items():
        group = {key: x for key, x in ratios.items() if mode == 'total' or streams[key]['mode'] == mode}
        assert isinstance(cap, int) and 0 <= cap < len(group)
        cutoff = sorted(group.values(), reverse=True)[cap]
        ties = sorted(key for key, x in group.items() if x == cutoff)
        constraints.append((mode, cap, cutoff, ties))
    return max(c[2] for c in constraints), constraints


def fixtures() -> list[str]:
    passed = []
    for name, groups in [('unique', [[5, 4], [3, 1], [2, 1]]),
                         ('within_ties', [[5, 5], [3, 1], [2, 1]]),
                         ('across_ties', [[5, 4], [5, 1], [2, 1]])]:
        groups = [list(map(F, group)) for group in groups]
        m = [max(group) for group in groups]
        B = max(m)
        b = [max(m[j] for j in range(3) if j != i) / B for i in range(3)]
        beta = min(b)
        records = {str(i): {'mode': ['frozen', 'updating'][i % 2]} for i in range(6)}
        S = {str(2*c+j): x for c, group in enumerate(groups) for j, x in enumerate(group)}
        L = {key: x / b[int(key)//2] for key, x in S.items()}
        cap_list = [{'total': i} for i in range(6)]
        cap_list += [{'frozen': i, 'updating': j} for i in range(3) for j in range(3)]
        for caps in cap_list:
            tl, _ = select(L, records, caps)
            ts, _ = select(S, records, caps)
            assert beta * tl <= ts <= tl
            for c in range(3):
                if b[c] < 1:
                    assert ts >= tl * b[c]
                else:
                    assert ts <= tl
            for mode, cap in caps.items():
                for ratio, level in ((L, tl), (S, ts)):
                    assert sum(x > level for k, x in ratio.items() if mode == 'total' or records[k]['mode'] == mode) <= cap
        if name == 'across_ties':
            assert b == [1, 1, 1] and L == S
        else:
            m1, m2 = B, max(m[1:])
            assert all(x/m2 <= m1/m2 for x in groups[0])
            assert all(x/m1 <= m2/m1 for group in groups[1:] for x in group)
            assert groups[0][1]/m2 > m2/m1  # catches the old incorrect within-checkpoint bound
            pooled = {str(2*c+j): x/(B*b[c]) for c, g in enumerate(groups) for j, x in enumerate(g)}
            k, _ = select(pooled, records, {'total': 0})
            assert k == m1/m2 and k*B*b[0] == m1
            assert k*B == m1*m1/m2
        passed.append(name + ': all 6 total caps and all 9 per-mode cap pairs')
    lo, hi = F(5), F(25, 3)
    assert [(x > lo) and not (x > hi) for x in (lo, (lo+hi)/2, hi)] == [False, True, True]
    passed.append('strict equality and (lower, upper] crossing-gain interval')
    for caps in ({'total': -1}, {'total': 6}, {'frozen': 3}):
        try:
            select(S, records, caps)
        except AssertionError:
            continue
        raise AssertionError('invalid cap accepted')
    passed.append('invalid cap indices rejected')
    return passed


class Output:
    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=False)
        self.root = root
        self.writers = {}
        self.handles = []
        self.counts = defaultdict(int)

    def put(self, name: str, record: dict) -> None:
        if name not in self.writers:
            handle = (self.root / (name + '.csv')).open('x', newline='')
            self.handles.append(handle)
            writer = csv.DictWriter(handle, list(record), lineterminator='\n')
            writer.writeheader()
            self.writers[name] = writer
        self.writers[name].writerow(record)
        self.counts[name] += 1

    def close(self) -> None:
        for h in self.handles:
            h.close()


def run(inputs: Path, historical: Path, output: Path) -> dict:
    checked_fixtures = fixtures()
    spec = json.loads((inputs/'manifest.json').read_text())
    assert spec['schema_version'] == 1
    horizon = spec['horizon']
    streams = {x['stream_key']: x for x in rows(inputs/'streams.csv')}
    divisors = {x['divisor_id']: x for x in rows(inputs/'divisors.csv')}
    maximum = {key: float(s['maximum']) for key, s in streams.items()}
    exact_max = {key: rational(x) for key, x in maximum.items()}
    d64 = {key: float(d['divisor']) for key, d in divisors.items()}
    dex = {key: rational(x) for key, x in d64.items()}
    for x in [*maximum.values(), *d64.values()]:
        assert math.isfinite(x) and x > 0
    target_groups, cal_groups = defaultdict(list), defaultdict(list)
    for t in rows(inputs/'targets.csv'):
        target_groups[t['source_point_id']].append(t)
    for c in rows(inputs/'calibration.csv'):
        cal_groups[(c['panel'], c['statistic'], c['calibration'])].append(c)
    paths, qpaths = defaultdict(list), defaultdict(list)
    for p in rows(inputs/'paths.csv'):
        key = p['stream_key']
        assert int(p['index']) == len(paths[key])+1
        x = float(p['value']) if p['value'] else None
        paths[key].append(x)
        qpaths[key].append(rational(x) if x is not None else None)
    for key, path in paths.items():
        assert len(path) == horizon and max(x for x in path if x is not None) == maximum[key]
    points = {p['source_point_id']: p for p in rows(inputs/'points.csv')}
    selected = []
    for source, p in points.items():
        if truth(p['include_control']):
            selected += [(source, 'loco', None), (source, 'scaling_only', None)]
    assert len(selected) == 168
    for family in ('arithmetic', 'geometric', 'quantile90'):
        selected += [(f'E:GLR:{family}:additional_2', control, 808) for control in ('loco', 'scaling_only')]
    selected.append(('D:GLR:arithmetic:original_caps', 'loco', None))

    def pid(source: str, control: str, omit: int | None) -> str:
        return source if control == 'loco' and omit is None else source+':'+control+('' if omit is None else ':omit_'+str(omit))

    wanted = {pid(*x) for x in selected}
    assert len(wanted) == 175
    old_points = {p['point_id']: p for p in rows(historical/'operating_points.csv') if p['point_id'] in wanted}
    old_streams = defaultdict(dict)
    with (historical/'stream_decisions.csv').open(newline='') as handle:
        for s in csv.DictReader(handle):
            if s['point_id'] in wanted:
                old_streams[s['point_id']][s['stream_key']] = s
    out = Output(output)
    reports, calculated = [], {}
    for source, control, omit in selected:
        point_id = pid(source, control, omit)
        p, old = points[source], old_points[point_id]
        caps = json.loads(p['caps_json'])
        ts = {t['stream_key']: t for t in target_groups[source] if int(streams[t['stream_key']]['checkpoint_seed']) != omit}
        cr = [c for c in cal_groups[p['panel'], p['statistic'], p['calibration']] if int(streams[c['stream_key']]['checkpoint_seed']) != omit]
        assert set(ts) == set(old_streams[point_id])
        normalized = {c['stream_key']: maximum[c['stream_key']]/d64[c['divisor_id']] for c in cr}
        normalized_q = {c['stream_key']: exact_max[c['stream_key']]/dex[c['divisor_id']] for c in cr}
        B, Bq = max(normalized.values()), max(normalized_q.values())
        seeds = sorted({streams[k]['checkpoint_seed'] for k in ts}, key=int)
        foreign, foreign_q = {}, {}
        donor_differences = 0
        for seed in seeds:
            keys = [k for k in normalized if streams[k]['checkpoint_seed'] != seed]
            foreign[seed], foreign_q[seed] = max(normalized[k] for k in keys), max(normalized_q[k] for k in keys)
            for kind, members, v, vq in [('global', list(normalized), B, Bq), ('foreign', keys, foreign[seed], foreign_q[seed])]:
                ties = sorted(k for k in members if normalized[k] == v)
                ties_q = sorted(k for k in members if normalized_q[k] == vq)
                donor_differences += ties != ties_q
                out.put('donors', dict(point_id=point_id, target_checkpoint=seed, donor_type=kind,
                                      binary64_level=repr(v), exact_level=str(vq), exact_level_rounded=repr(float(vq)),
                                      binary64_ties=dump(ties), exact_ties=dump(ties_q), identities_differ=ties != ties_q))
        base, baseq = {}, {}
        for key, t in ts.items():
            seed = streams[key]['checkpoint_seed']
            base[key] = d64[t['divisor_id']] * foreign[seed] if control == 'loco' else d64[t['divisor_id']]
            baseq[key] = dex[t['divisor_id']] * foreign_q[seed] if control == 'loco' else dex[t['divisor_id']]
        tuning = [key for key in ts if truth(ts[key]['tuning'])]
        ratio = {key: maximum[key]/base[key] for key in tuning}
        ratioq = {key: exact_max[key]/baseq[key] for key in tuning}
        k, constraints = select(ratio, streams, caps)
        kq, constraintsq = select(ratioq, streams, caps)
        candidate, steps = k, 0
        while any(sum(maximum[key] > base[key]*k for key in tuning if mode == 'total' or streams[key]['mode'] == mode) > cap for mode, cap in caps.items()):
            k = math.nextafter(k, math.inf)
            steps += 1
            assert steps <= 32
        theta, thetaq = (k*B, kq*Bq) if control == 'loco' else (k, kq)
        assert k == float(old['multiplier']) and candidate == float(old['ratio_candidate'])
        assert steps == int(old['boundary_ulp_increments']) and B == float(old['global_level']) and theta == float(old['theta'])
        boundary_differences = 0
        for c, cq in zip(constraints, constraintsq, strict=True):
            mode, cap, value, ties = c
            qvalue, qties = cq[2:]
            boundary_differences += ties != qties or (value == candidate) != (qvalue == kq)
            out.put('boundaries', dict(point_id=point_id, constraint=mode, cap=cap, descending_position=cap+1,
                                      binary64_candidate=repr(value), exact_candidate=str(qvalue), exact_candidate_rounded=repr(float(qvalue)),
                                      binary64_ties=dump(ties), exact_ties=dump(qties), binary64_binding=value == candidate, exact_binding=qvalue == kq))
        outcomes = {}
        stable_diff, changed_diff, delays_diff = [], [], []
        stable_old, stable_q = defaultdict(list), defaultdict(list)
        threshold_count = 0
        for key, target in ts.items():
            s, recorded = streams[key], old_streams[point_id][key]
            threshold, thresholdq = base[key]*k, baseq[key]*kq
            fired, firedq = maximum[key] > threshold, exact_max[key] > thresholdq
            delay = next((i+1 for i, x in enumerate(paths[key]) if x is not None and x > threshold), horizon+1) if key in paths else None
            delayq = next((i+1 for i, x in enumerate(qpaths[key]) if x is not None and x > thresholdq), horizon+1) if key in qpaths else None
            assert threshold == float(recorded['threshold']) and fired == truth(recorded['fired'])
            assert delay == (int(float(recorded['restricted_delay'])) if recorded['restricted_delay'] else None)
            assert key not in paths or (delay <= horizon) == fired
            assert key not in paths or (delayq <= horizon) == firedq
            is_tuning = truth(target['tuning'])
            if is_tuning:
                if fired: stable_old[s['mode']].append(key)
                if firedq: stable_q[s['mode']].append(key)
            if fired != firedq:
                (stable_diff if s['role'] != 'changed' else changed_diff).append(key)
            if delay != delayq:
                delays_diff.append(key)
            threshold_count += rational(threshold) != thresholdq
            # Two counterfactual arithmetic stages localize every decision discrepancy:
            # exact threshold product at the recorded k, then fully exact selection.
            product_at_old_k = baseq[key]*rational(k)
            product_decision = exact_max[key] > product_at_old_k
            causes = []
            if fired != product_decision:
                causes.append('calibration/base/product rounding at recorded multiplier')
            if product_decision != firedq:
                causes.append('exact order-statistic selection versus recorded multiplier')
            decision = dict(point_id=point_id, stream_key=key, stream_id=s['stream_id'], checkpoint_seed=s['checkpoint_seed'],
                            condition=s['condition'], mode=s['mode'], role=s['role'], tuning=is_tuning,
                            divisor_id=target['divisor_id'], maximum=repr(maximum[key]), divisor=repr(d64[target['divisor_id']]),
                            binary64_base=repr(base[key]), exact_base=str(baseq[key]), binary64_threshold=repr(threshold),
                            exact_threshold=str(thresholdq), exact_threshold_rounded=repr(float(thresholdq)),
                            threshold_difference=str(thresholdq-rational(threshold)), binary64_fired=fired, exact_fired=firedq,
                            binary64_delay='' if delay is None else delay, exact_delay='' if delayq is None else delayq,
                            delay_difference='' if delay is None else delayq-delay, binary64_margin=repr(maximum[key]-threshold),
                            exact_margin=str(exact_max[key]-thresholdq), exact_at_recorded_multiplier_fired=product_decision,
                            discrepancy_explanation='; '.join(causes))
            out.put('stream_reference', decision)
            if fired != firedq or delay != delayq:
                assert causes or delay != delayq
                out.put('discrepancies', decision)
            outcomes[key] = dict(decision, threshold=threshold, threshold_q=thresholdq, delay=delay, delay_q=delayq,
                                 fired=fired, fired_q=firedq, ratio=ratio.get(key), ratio_q=ratioq.get(key))
        for mode in ('frozen', 'updating'):
            assert len(stable_old[mode]) == int(old[mode+'_alarms'])
        for mode, cap in caps.items():
            count = sum(len(x) for x in stable_q.values()) if mode == 'total' else len(stable_q[mode])
            assert count <= cap
        report = dict(point_id=point_id, source_point_id=source, control_type=control, omitted_checkpoint=omit,
                      caps=caps, binary64_multiplier=repr(k), exact_multiplier=str(kq), exact_multiplier_rounded=repr(float(kq)),
                      multiplier_difference=str(kq-rational(k)), multiplier_difference_ulps=float((kq-rational(k))/rational(math.ulp(k))),
                      binary64_ulp_increments=steps, binary64_theta=repr(theta), exact_theta=str(thetaq),
                      exact_theta_rounded=repr(float(thetaq)), donor_identity_differences=donor_differences,
                      boundary_identity_differences=boundary_differences, unequal_exact_thresholds=threshold_count,
                      binary64_stable_ids=dict(stable_old), exact_stable_ids=dict(stable_q),
                      binary64_stable_counts={m: len(stable_old[m]) for m in ('frozen', 'updating')},
                      exact_stable_counts={m: len(stable_q[m]) for m in ('frozen', 'updating')},
                      stable_decision_difference_ids=stable_diff, changed_decision_difference_ids=changed_diff,
                      crossing_time_difference_ids=delays_diff,
                      primary_binary64_ids=sorted(k for k,x in outcomes.items() if x['condition']=='gradual_updating_down' and x['fired']),
                      primary_exact_ids=sorted(k for k,x in outcomes.items() if x['condition']=='gradual_updating_down' and x['fired_q']))
        out.put('rules', {key: dump(value) if isinstance(value, (dict, list)) else value for key, value in report.items()})
        reports.append(report)
        calculated[point_id] = dict(outcomes=outcomes, report=report, constraints=constraints, constraintsq=constraintsq,
                                    B=B, Bq=Bq, foreign=foreign, foreignq=foreign_q, normalized=normalized, normalizedq=normalized_q)
    # Verify exact inequalities on every prescribed paired control, including omissions.
    exact_pairs = 0
    for source, control, omit in selected:
        if control != 'scaling_only':
            continue
        a, b = calculated[pid(source,'loco',omit)], calculated[pid(source,control,omit)]
        beta = min(a['foreignq'].values())/a['Bq']
        tl, ts = F(a['report']['exact_theta']), F(b['report']['exact_theta'])
        assert beta*tl <= ts <= tl
        for key, x in a['outcomes'].items():
            seed = streams[key]['checkpoint_seed']
            y = b['outcomes'][key]
            if a['foreignq'][seed] == a['Bq']:
                assert y['threshold_q'] <= x['threshold_q']
            else:
                assert y['threshold_q'] >= x['threshold_q']
        exact_pairs += 1
    omission = omission_trace(calculated, streams, out)
    out.close()
    summary = dict(status='PASS', convention='Each recorded finite binary64 input is an exact rational. Divisors stay supplied; exact ratios, extrema, order statistics, products and strict crossings have no intermediate rounding.',
                   fitted_rules=len(reports), control_pairs=84, omission_rules=6, unequal_membership_rules=1,
                   exact_control_inequalities=exact_pairs, exact_fixtures=checked_fixtures,
                   independently_reproduced_binary64_stream_decisions=out.counts['stream_reference'],
                   rules_with_multiplier_difference=sum(F(r['exact_multiplier']) != rational(r['binary64_multiplier']) for r in reports),
                   maximum_absolute_multiplier_difference_ulps=max(abs(r['multiplier_difference_ulps']) for r in reports),
                   donor_identity_difference_rows=sum(r['donor_identity_differences'] for r in reports),
                   boundary_identity_difference_constraints=sum(r['boundary_identity_differences'] for r in reports),
                   rules_with_stable_decision_differences=sum(bool(r['stable_decision_difference_ids']) for r in reports),
                   stable_decision_differences=sum(len(r['stable_decision_difference_ids']) for r in reports),
                   changed_decision_differences=sum(len(r['changed_decision_difference_ids']) for r in reports),
                   crossing_time_differences=sum(len(r['crossing_time_difference_ids']) for r in reports),
                   output_rows=dict(out.counts), checkpoint_808=omission,
                   scope='Numerical propagation from recorded inputs only; no raw-score or divisor-formula precision audit. Unavailable paths retain unavailable delays.')
    sources = sorted(inputs.glob('*')) + [historical/'operating_points.csv', historical/'stream_decisions.csv']
    hashes = {(('inputs/' if p.parent == inputs else 'historical/')+p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources if p.is_file()}
    (output/'input_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'rules.json').write_text(json.dumps(reports,indent=2)+'\n')
    return summary


def omission_trace(calculated: dict, streams: dict, out: Output) -> dict:
    report = {}
    detected = []
    for family in ('arithmetic', 'geometric', 'quantile90'):
        source = f'E:GLR:{family}:additional_2'
        a, b = calculated[source], calculated[source+':loco:omit_808']
        old, new = a['outcomes'], b['outcomes']
        retained = {k for k in old if streams[k]['checkpoint_seed'] != '808'}
        assert retained == set(new)
        assert a['B'] == b['B'] and a['Bq'] == b['Bq']
        assert all(a['foreign'][c] == b['foreign'][c] and a['foreignq'][c] == b['foreignq'][c] for c in b['foreign'])
        assert all(old[k]['divisor_id'] == new[k]['divisor_id'] and old[k]['divisor'] == new[k]['divisor'] for k in retained)
        # All donor ties, not only donor values, must remain identical.
        for seed in b['foreign']:
            for keyset, level in ((a['normalized'],a['foreign'][seed]), (a['normalizedq'],a['foreignq'][seed])):
                first = sorted(k for k,v in keyset.items() if streams[k]['checkpoint_seed'] != seed and v == level)
                second = sorted(k for k in first if streams[k]['checkpoint_seed'] != '808')
                assert first == second
        removed = [x for k, x in old.items() if k not in retained and x['tuning']]
        assert len(removed) == 8
        binding = a['constraints'][0][2]
        influential = [x for x in removed if x['ratio'] >= binding]
        assert len(influential) == 1 and influential[0]['stream_key'] == 'E:GLR:seed_808_rep_6_stable_frozen'
        primary = sorted(k for k in new if new[k]['condition']=='gradual_updating_down')
        assert len(primary) == 76
        gains = [k for k in primary if new[k]['fired'] and not old[k]['fired']]
        losses = [k for k in primary if old[k]['fired'] and not new[k]['fired']]
        delays = [k for k in primary if old[k]['delay'] != new[k]['delay']]
        exact_gains = [k for k in primary if new[k]['fired_q'] and not old[k]['fired_q']]
        assert gains == exact_gains
        for key in primary:
            x,y = old[key],new[key]
            out.put('omission_streams',dict(family=family, stream_key=key, stream_id=streams[key]['stream_id'],
                       old_threshold=repr(x['threshold']), new_threshold=repr(y['threshold']), maximum=x['maximum'],
                       old_margin=x['binary64_margin'], new_margin=y['binary64_margin'], old_fired=x['fired'], new_fired=y['fired'],
                       old_delay=x['delay'], new_delay=y['delay'], exact_old_delay=x['delay_q'], exact_new_delay=y['delay_q']))
        for name, fit in [('full', a), ('omit_808', b)]:
            ranked = sorted((x for x in fit['outcomes'].values() if x['tuning']), key=lambda x:(-x['ratio'],x['stream_key']))
            rankedq = sorted(ranked, key=lambda x:(-x['ratio_q'],x['stream_key']))
            for rank,x in enumerate(ranked,1):
                out.put('omission_tuning',dict(family=family, fit=name, rank=rank, exact_rank=1+next(i for i,y in enumerate(rankedq) if y['stream_key']==x['stream_key']),
                     stream_key=x['stream_key'], checkpoint_seed=x['checkpoint_seed'], mode=x['mode'],
                     removed_by_808=x['checkpoint_seed']=='808', binary64_ratio=repr(x['ratio']), exact_ratio=str(x['ratio_q']),
                     selected_boundary=x['stream_key'] in fit['constraints'][0][3],
                     binary64_fired=x['fired'], exact_fired=x['fired_q'], binary64_theta=fit['report']['binary64_theta'],
                     exact_theta=fit['report']['exact_theta']))
        family_report=dict(identical_retained_ids=len(retained), retained_primary=76, unchanged_donors=True,
            old_boundary=a['constraints'][0][3],new_boundary=b['constraints'][0][3],
            old_multiplier=a['report']['binary64_multiplier'],new_multiplier=b['report']['binary64_multiplier'],
            old_theta=a['report']['binary64_theta'],new_theta=b['report']['binary64_theta'],
            original_detections=sum(old[k]['fired'] for k in primary),recalibrated_detections=sum(new[k]['fired'] for k in primary),
            gained_ids=gains,lost_ids=losses,changed_delay_ids=delays,
            old_delay_sum=sum(old[k]['delay'] for k in primary),new_delay_sum=sum(new[k]['delay'] for k in primary))
        report[family]=family_report
        if family in ('arithmetic','quantile90'):
            detected.append({k for k in primary if new[k]['fired']})
    assert report['arithmetic']['original_detections']==5 and len(report['arithmetic']['gained_ids'])==2
    assert report['arithmetic']['recalibrated_detections']==7 and not report['arithmetic']['lost_ids']
    assert report['quantile90']['original_detections']==report['quantile90']['recalibrated_detections']==7
    assert not report['quantile90']['gained_ids'] and not report['quantile90']['lost_ids']
    assert len(report['quantile90']['changed_delay_ids'])==3 and detected[0]==detected[1]
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--historical',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.inputs,args.historical,args.output),indent=2))
