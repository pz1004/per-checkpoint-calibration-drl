"""Independent replay of R4 review outputs; no analysis engine used for replay.

Uses scalar order statistics and the declared NumPy reduction primitives.
The producer is imported only in explicitly identified adverse-input tests.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction as F
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import tempfile

import numpy as np


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: Path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', newline='') as handle:
        yield from csv.DictReader(handle)


def sha(path: Path) -> str:
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def selector(values: dict, records: dict, caps: dict) -> tuple[float, list]:
    constraints = []
    for mode, cap in caps.items():
        keys = [k for k in values if mode == 'total' or records[k]['mode'] == mode]
        require(type(cap) is int and 0 <= cap < len(keys), 'invalid remaining cap index')
        v = sorted((values[k] for k in keys), reverse=True)[cap]
        constraints.append(dict(constraint=mode, cap=cap, candidate=v,
                                ties=sorted(k for k in keys if values[k] == v)))
    selected = max(c['candidate'] for c in constraints)
    for c in constraints:
        c['binding'] = c['candidate'] == selected
    return selected, constraints


def fixtures() -> list[str]:
    for groups in ([[5, 4], [3, 1], [2, 1]], [[5, 5], [3, 1], [2, 1]], [[5, 4], [5, 1], [2, 1]]):
        R = list(map(max, groups))
        B = max(R)
        b = [F(max(R[j] for j in range(3) if j != i), B) for i in range(3)]
        S = {str(2*c+j): F(x) for c,g in enumerate(groups) for j,x in enumerate(g)}
        L = {k:x/b[int(k)//2] for k,x in S.items()}
        records = {k:dict(mode=('frozen' if int(k)%2 else 'updating')) for k in S}
        for caps in ([{'total':k} for k in range(6)] +
                     [{'frozen':a,'updating':b} for a in range(3) for b in range(3)]):
            tl, _ = selector(L, records, caps)
            ts, _ = selector(S, records, caps)
            beta, tau = min(b), ts/tl
            require(beta <= tau <= 1, 'fixture: tau bounds')
            for c in range(3):
                require((tau/b[c] >= 1 if b[c] < 1 else tau/b[c] <= 1), 'fixture: directions')
        if R.count(B) > 1:
            require(b == [1,1,1] and L == S, 'across-checkpoint equality')
    # Strict endpoint inclusion, simultaneous gains/losses and unchanged sets.
    H, S = F(10), F(8)
    require([S < x <= H for x in (S,F(9),H)] == [False,True,True], 'gain endpoints')
    H, S = F(6), F(8)
    require([H < x <= S for x in (H,F(7),S)] == [False,True,True], 'loss endpoints')
    maxima = [F(9), F(7), F(5), F(11)]
    L, S = [F(10),F(6),F(10),F(6)], [F(8)]*4
    gains = [m > s and m <= l for m,l,s in zip(maxima,L,S)]
    losses = [m > l and m <= s for m,l,s in zip(maxima,L,S)]
    require(sum(gains) == sum(losses) == 1, 'simultaneous gain and loss')
    require(all((m > l) == (m > s) for m,l,s in zip(maxima[2:],L[2:],S[2:])), 'unchanged set')
    # Fixed total cap, unchanged ratios and positive bases: deletion cannot raise cutoff.
    values = [F(9),F(8),F(8),F(3),F(1)]
    checked = 0
    for n in range(1,len(values)+1):
        for subset in itertools.combinations(range(len(values)), n):
            for cap in range(n):
                old = sorted(values,reverse=True)[cap]
                new = sorted([values[i] for i in subset],reverse=True)[cap]
                require(new <= old, 'deletion monotonicity')
                checked += 1
    for cap in (-1, 1, True, 1.5):
        try:
            selector({'a':1.}, {'a':dict(mode='frozen')}, {'total':cap})
        except ValueError:
            pass
        else:
            raise AssertionError('invalid remaining index accepted')
    return ['within/across checkpoint ties, 45 cap fixtures', 'strict lower/upper equality',
            'simultaneous gains/losses', 'unchanged detected set',
            f'{checked} fixed-cap deletion checks', 'invalid remaining indices rejected']


def effects(inputs: Path, reference: Path, output: Path) -> int:
    rules = {r['point_id']:r for r in json.loads((reference/'rules.json').read_text())}
    old = defaultdict(dict)
    for r in read(reference/'stream_reference.csv'):
        old[r['point_id']][r['stream_key']] = r
    donors = defaultdict(list)
    for r in read(reference/'donors.csv'):
        donors[r['point_id']].append(r)
    expected = {r['source_point_id'] for r in read(inputs/'points.csv') if r['include_control'] == 'True'}
    summaries = {r['source_point_id']:r for r in read(output/'effects.csv')}
    require(set(summaries) == expected and len(expected) == 84, 'all 84 effects')
    grouped = itertools.groupby(read(output/'effect_intervals.csv.gz'),lambda x:x['source_point_id'])
    total = 0
    for source, lines in grouped:
        summary = summaries[source]
        B = max(F(d['exact_level']) for d in donors[source])
        f = {r['target_checkpoint']:F(r['exact_level']) for r in donors[source] if r['donor_type']=='foreign'}
        beta = min(f.values())/B
        tau = F(rules[source+':scaling_only']['exact_theta'])/F(rules[source]['exact_theta'])
        require(F(summary['exact_beta']) == beta and F(summary['exact_tau']) == tau and beta <= tau <= 1, 'effect beta/tau')
        counts = Counter()
        seen = set()
        for r in lines:
            key = r['stream_key'];seen.add(key)
            a,b = old[source][key],old[source+':scaling_only'][key]
            M,H,S = F(float(a['maximum'])),F(a['exact_threshold']),F(b['exact_threshold'])
            inside = f[a['checkpoint_seed']] < B
            require(F(r['exact_loco_threshold']) == H and F(r['exact_scaling_threshold']) == S, 'effect threshold reference')
            require(F(r['exact_ratio']) == S/H == (tau/beta if inside else tau), 'effect ratio identity')
            require(F(r['exact_movement']) == S-H and float(r['maximum'])==float(a['maximum']), 'effect movement/maximum')
            gain,loss = S < M <= H, H < M <= S
            require((r['gain_interval']=='True') == gain and (r['loss_interval']=='True') == loss, 'effect intervals')
            require((r['loco_fired']=='True') == (M > H) and (r['scaling_fired']=='True') == (M > S), 'effect decisions')
            require(r['loco_delay']==a['exact_delay'] and r['scaling_delay']==b['exact_delay'], 'effect delays')
            for scope,active in [('primary',a['condition']=='gradual_updating_down'),('changed',a['role']=='changed'),('all',True)]:
                if active:
                    for field,value in [('gains',gain),('losses',loss),('loco_detections',M>H),('scaling_detections',M>S),('denominator',True)]:
                        counts[scope+'_'+field] += int(value)
            total += 1
        require(seen == set(old[source]), 'effect interval coverage')
        for field,count in counts.items():
            require(int(summary[field]) == count, 'effect aggregate: '+field)
    return total


def check(inputs: Path, reference: Path, protocol: Path, output: Path, report: Path) -> dict:
    require(not report.exists(), 'check report must be fresh')
    p = json.loads(protocol.read_text())
    require((output/'protocol.json').read_bytes() == protocol.read_bytes(), 'frozen protocol bytes')
    manifest = json.loads((output/'manifest.json').read_text())
    require(set(manifest) == {f.name for f in output.iterdir() if f.is_file()}-{'manifest.json'}, 'result inventory')
    for name,digest in manifest.items():
        require(sha(output/name)==digest, 'result hash: '+name)
    inputs_hash = {f'inputs/{f.name}':sha(f) for f in inputs.iterdir() if f.is_file()}
    inputs_hash.update({f'reference/{f.name}':sha(f) for f in reference.iterdir() if f.is_file()})
    inputs_hash['sensitivity_protocol.json']=sha(protocol)
    require(inputs_hash==json.loads((output/'input_hashes.json').read_text()), 'input identities')
    fixture_checks = fixtures()
    n_effects = effects(inputs,reference,output)
    streams = {r['stream_key']:r for r in read(inputs/'streams.csv')}
    paths = defaultdict(list)
    for r in read(inputs/'paths.csv'):
        paths[r['stream_key']].append(float(r['value']) if r['value'] else np.nan)
    points = {r['source_point_id']:r for r in read(inputs/'points.csv')}
    members = defaultdict(list)
    for r in read(inputs/'memberships.csv'):
        members[r['divisor_id']].append(r['stream_key'])
    divisors = {r['divisor_id']:r for r in read(inputs/'divisors.csv')}
    calibration = list(read(inputs/'calibration.csv'))
    targets = defaultdict(list)
    for r in read(inputs/'targets.csv'):
        targets[r['source_point_id']].append(r)
    groups={(points[c]['panel'],points[c]['statistic']) for c in p['cases']}
    universe=sorted(k for k,s in streams.items() if (s['panel'],s['statistic']) in groups and
                    (s['role']!='changed' or s['condition']==p['primary_condition']))
    gate=json.loads((output/'zero_gate.json').read_text())
    # Independently construct the deterministic single-stream union from reference ties.
    rule_specs={}
    for source in p['cases']:
        for omit in p['omissions'][points[source]['panel']]:
            for control in p['controls']:
                key=source if omit is None and control=='loco' else source+':'+control+('' if omit is None else ':omit_'+str(omit))
                rule_specs[key]=(source,control,omit)
    singles=set()
    for r in read(reference/'donors.csv'):
        if r['point_id'] in rule_specs:singles.update(json.loads(r['binary64_ties']))
    for r in read(reference/'boundaries.csv'):
        if r['point_id'] in rule_specs and r['binary64_binding']=='True':singles.update(json.loads(r['binary64_ties']))
    for source in p['cases']:
        singles.update(t['stream_key'] for t in targets[source] if t['tuning']=='True' and streams[t['stream_key']]['checkpoint_seed']=='808')
    require(sorted(singles)==gate['single_streams'], 'single-stream selection union')
    vectors={}
    for key in sorted(singles):
        for sign in (-1,1):
            v=np.zeros(len(universe));v[universe.index(key)]=sign;vectors[f's{len(vectors):03d}']=v
    joint=np.random.Generator(np.random.PCG64(20261006)).uniform(-1,1,(100,len(universe)))
    for i,v in enumerate(joint):vectors[f'j{i:03d}']=v
    recorded_vectors=itertools.groupby(read(output/'vectors.csv.gz'),lambda r:r['vector'])
    vector_names=[]
    for name,records in recorded_vectors:
        rows=list(records);vector_names.append(name)
        require([r['stream_key'] for r in rows]==universe and
                np.array_equal([float(r['u']) for r in rows],vectors[name]), 'shared deterministic vector')
    require(vector_names==list(vectors), 'vector coverage')
    factor_groups=iter(itertools.groupby(read(output/'factors.csv.gz'),lambda r:r['scenario']))
    divisor_groups=iter(itertools.groupby(read(output/'divisors.csv.gz'),lambda r:r['scenario']))
    fit_groups=iter(itertools.groupby(read(output/'fits.csv'),lambda r:r['scenario']))
    decision_groups=iter(itertools.groupby(read(output/'decisions.csv.gz'),lambda r:r['scenario']))
    pair_groups=iter(itertools.groupby(read(output/'paired.csv'),lambda r:r['scenario']))
    baseline={}; old_reference=defaultdict(dict)
    for r in read(reference/'stream_reference.csv'):
        if r['point_id'] in rule_specs:old_reference[r['point_id']][r['stream_key']]=r
    counts=Counter();persistence=defaultdict(Counter)
    for mi,epsilon in enumerate(p['magnitudes']):
        for scope in p['scopes']:
            for name,v in vectors.items():
                scenario=f'e{mi}:{scope}:{name}'
                fg,fl=next(factor_groups);dg,dl=next(divisor_groups);sg,sl=next(fit_groups);og,ol=next(decision_groups);pg,pl=next(pair_groups)
                require(fg==dg==sg==og==pg==scenario, 'scenario coverage/order')
                factor_rows=list(fl)
                require([r['stream_key'] for r in factor_rows]==universe, 'factor stream coverage')
                factors={}
                for i,r in enumerate(factor_rows):
                    k=r['stream_key'];factor=1.+epsilon*v[i]
                    if scope=='stable_only' and streams[k]['role']=='changed':factor=1.
                    require(float(r['factor'])==factor and factor>0, 'shared factor/magnitude/scope')
                    factors[k]=factor
                perturbed_paths={k:np.asarray(paths[k])*factors.get(k,1.) for k in universe if k in paths}
                maximum={k:float(np.nanmax(perturbed_paths[k])) if k in perturbed_paths else float(streams[k]['maximum'])*factors[k] for k in universe}
                dr={r['divisor_id']:float(r['divisor']) for r in dl}
                expected_divs={k for k,d in divisors.items() if any((d['panel'],d['statistic'],d['calibration'])==
                    (points[c]['panel'],points[c]['statistic'],points[c]['calibration']) for c in p['cases'])}
                require(set(dr)==expected_divs, 'divisor coverage')
                for identity,recorded in dr.items():
                    d=divisors[identity];values=np.asarray([maximum[k] for k in members[identity]])
                    require(all(streams[k]['role']=='original_stable' for k in members[identity]), 'stable-only divisor fitting')
                    expected={'pooled':lambda:1.,'arithmetic':lambda:float(values.mean()),
                              'geometric':lambda:float(np.exp(np.log(values).mean())),
                              'quantile90':lambda:float(np.quantile(values,.9,method='linear'))}[d['calibration']]()
                    require(recorded==expected, 'independent formula: '+identity)
                result={}; summaries={};fit_rows=list(sl)
                require([r['point_id'] for r in fit_rows]==list(rule_specs), 'fit coverage')
                for r in fit_rows:
                    point=r['point_id'];source,control,omit=rule_specs[point];spec=points[source]
                    ts=[t for t in targets[source] if int(streams[t['stream_key']]['checkpoint_seed'])!=omit]
                    cr=[t for t in calibration if (t['panel'],t['statistic'],t['calibration'])==(spec['panel'],spec['statistic'],spec['calibration'])
                        and int(streams[t['stream_key']]['checkpoint_seed'])!=omit]
                    norm={t['stream_key']:maximum[t['stream_key']]/dr[t['divisor_id']] for t in cr}
                    B=max(norm.values());seeds=sorted({streams[t['stream_key']]['checkpoint_seed'] for t in ts},key=int)
                    foreign={seed:max(x for k,x in norm.items() if streams[k]['checkpoint_seed']!=seed) for seed in seeds}
                    donors={seed:sorted(k for k,x in norm.items() if streams[k]['checkpoint_seed']!=seed and x==foreign[seed]) for seed in seeds}
                    td={t['stream_key']:dr[t['divisor_id']] for t in ts}
                    base={k:foreign[streams[k]['checkpoint_seed']]*d if control=='loco' else d for k,d in td.items()}
                    tune=[t['stream_key'] for t in ts if t['tuning']=='True']
                    require(all(streams[k]['role']!='changed' for k in tune), 'stable-only selection')
                    ratio={k:maximum[k]/base[k] for k in tune};caps=json.loads(spec['caps_json'])
                    candidate,boundaries=selector(ratio,streams,caps);multiplier=candidate;steps=0
                    while any(sum(maximum[k]>base[k]*multiplier for k in tune if mode=='total' or streams[k]['mode']==mode)>cap for mode,cap in caps.items()):
                        multiplier=math.nextafter(multiplier,math.inf);steps+=1;require(steps<=32,'ULP bound')
                    require(float(r['multiplier'])==multiplier and float(r['ratio_candidate'])==candidate and int(r['ulp_increments'])==steps,'scalar selector replay')
                    require(float(r['global_level'])==B and json.loads(r['foreign_levels'])==foreign and
                            json.loads(r['global_donors'])==sorted(k for k,x in norm.items() if x==B) and json.loads(r['foreign_donors'])==donors, 'donor replay')
                    require(json.loads(r['boundaries'])==boundaries, 'boundary replay')
                    require(float(r['theta'])==(multiplier*B if control=='loco' else multiplier) and float(r['beta'])==min(foreign.values())/B, 'global coordinates')
                    rr={}
                    for t in ts:
                        k=t['stream_key'];s=streams[k]
                        if not (t['tuning']=='True' or s['condition']==p['primary_condition']):continue
                        H=base[k]*multiplier;M=maximum[k]
                        delay=next((i for i,x in enumerate(perturbed_paths[k],1) if x>H),81) if k in perturbed_paths else None
                        rr[k]=dict(maximum=M,divisor=td[k],base=base[k],threshold=H,fired=M>H,delay=delay,
                                   signed_margin=M-H,absolute_margin=abs(M-H),relative_margin=(M-H)/H,
                                   scope='tuning' if t['tuning']=='True' else 'primary')
                        if epsilon==0:
                            old=old_reference[point][k]
                            require(float(old['binary64_threshold'])==H and (old['binary64_fired']=='True')==(M>H) and
                                    (int(old['binary64_delay']) if old['binary64_delay'] else None)==delay,'independent historical zero gate')
                    primary=sorted(k for k,x in rr.items() if x['scope']=='primary' and x['fired'])
                    stable=sorted(k for k,x in rr.items() if x['scope']=='tuning' and x['fired'])
                    require(json.loads(r['primary_ids'])==primary and json.loads(r['stable_ids'])==stable,'fitted decision sets')
                    require(int(r['primary_detections'])==len(primary) and int(r['primary_delay_sum'])==sum(x['delay'] for x in rr.values() if x['scope']=='primary'),'primary totals/delays')
                    for mode in ('frozen','updating'):
                        require(int(r[mode+'_alarms'])==sum(streams[k]['mode']==mode for k in stable),'stable totals')
                    if point not in baseline:baseline[point]=r
                    b=baseline[point]
                    changes=dict(donor_identity_changed=(json.loads(r['global_donors'])!=json.loads(b['global_donors']) or json.loads(r['foreign_donors'])!=json.loads(b['foreign_donors'])),
                        boundary_identity_changed=[(x['ties'],x['binding']) for x in boundaries]!=[(x['ties'],x['binding']) for x in json.loads(b['boundaries'])],
                        primary_set_changed=primary!=json.loads(b['primary_ids']),stable_set_changed=stable!=json.loads(b['stable_ids']))
                    counter=persistence[epsilon,scope,r['kind'],point];counter['scenarios']+=1
                    for key,value in changes.items():
                        require((r[key]=='True')==value,'change summary');counter[key]+=value
                    result[point]=rr;summaries[point]=r;counts['fits']+=1
                seen=set()
                for r in ol:
                    point,k=r['point_id'],r['stream_key'];identity=(point,k)
                    require(identity not in seen,'duplicate decision');seen.add(identity);expected=result[point][k]
                    for field,value in expected.items():
                        observed=(r[field]=='True') if field=='fired' else ((int(r[field]) if r[field] else None) if field=='delay' else (r[field] if field=='scope' else float(r[field])))
                        require(observed==value,'decision replay: '+field)
                    counts['decisions']+=1
                require(len(seen)==sum(len(r) for r in result.values()),'decision coverage')
                pairs=list(pl);require(len(pairs)==28,'paired comparison coverage')
                for r in pairs:
                    left,right=result[r['left']],result[r['right']]
                    keys=sorted(k for k in right if right[k]['scope']==r['scope'])
                    gains=[k for k in keys if right[k]['fired'] and not left[k]['fired']]
                    losses=[k for k in keys if left[k]['fired'] and not right[k]['fired']]
                    delays=[k for k in keys if left[k]['delay']!=right[k]['delay']]
                    require(json.loads(r['gained_ids'])==gains and json.loads(r['lost_ids'])==losses and json.loads(r['changed_delay_ids'])==delays,'paired identities')
                    for field,value in dict(denominator=len(keys),gains=len(gains),losses=len(losses),left_detections=sum(left[k]['fired'] for k in keys),right_detections=sum(right[k]['fired'] for k in keys),available_delays=sum(right[k]['delay'] is not None for k in keys)).items():
                        require(int(r[field])==value,'paired totals')
                counts['scenarios']+=1
            print(f'Independently checked epsilon={epsilon:g}, scope={scope}; {counts["decisions"]} decisions',flush=True)
    for iterator in (factor_groups,divisor_groups,fit_groups,decision_groups,pair_groups):
        require(next(iterator,None) is None,'unexpected extra scenarios')
    for r in read(output/'persistence.csv'):
        c=persistence.pop((float(r['magnitude']),r['input_scope'],r['kind'],r['point_id']))
        require(all(int(r[k])==v for k,v in c.items()),'persistence counts')
    require(not persistence,'persistence coverage')
    summary=json.loads((output/'summary.json').read_text())
    require(summary['scenarios']==counts['scenarios'] and summary['fitted_rules']==counts['fits'],'summary counts')
    # Producer-specific rejection tests are separate from the independent replay above.
    import review_analysis as producer
    study=producer.Study(inputs,p)
    bad_checks=0
    for value in (0.,-1.,np.nan,np.inf):
        factors=np.ones(len(study.keys));factors[0]=value
        try:study.perturb(factors)
        except ValueError:bad_checks+=1
        else:raise AssertionError('invalid factor accepted')
    maximum,path,div=study.perturb(np.ones(len(study.keys)))
    config=dict(study.configs[-1],caps={'total':int(study.configs[-1]['tuning'].sum())})
    try:study.fit(config,maximum,path,div)
    except ValueError:bad_checks+=1
    else:raise AssertionError('invalid remaining cap accepted')
    # Changed-path perturbation must not affect fitting, even when the changed maximum is huge.
    base_fit=study.fit(study.configs[0],maximum,path,div)
    f=np.ones(len(study.keys));f[~study.stable]=100.
    m2,p2,d2=study.perturb(f);new_fit=study.fit(study.configs[0],m2,p2,d2)
    require(div==d2 and np.array_equal(base_fit['threshold'],new_fit['threshold']),'changed inputs cannot fit thresholds')
    with tempfile.TemporaryDirectory(prefix='r4-review-adverse-') as tmp:
        existing=Path(tmp)
        try:producer.run(inputs,reference,protocol,existing)
        except ValueError:bad_checks+=1
        else:raise AssertionError('existing destination accepted')
        require(not list(existing.iterdir()),'existing destination mutated')
    receipt=dict(status='PASS',independent_effect_intervals=n_effects,controls=84,**counts,
        fixtures=fixture_checks,producer_rejection_checks=bad_checks,stable_only_fitting='PASS',
        factor_formula_membership_missing_path_checks='PASS',protocol_sha256=sha(protocol),
        scope='Independent complete replay of every sensitivity scenario and all exact effect intervals; deterministic rerun is a separate byte-comparison check.')
    report.parent.mkdir(parents=True,exist_ok=True);report.write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','reference','protocol','results','report'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    print(json.dumps(check(a.inputs,a.reference,a.protocol,a.results,a.report),indent=2))
