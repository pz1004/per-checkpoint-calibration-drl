"""R4 retrospective sensitivity from accepted R3 paths; no raw scoring or fitting.

Historical CSVs retain their bytes and interfaces. New rows carry source IDs and
explicit target-budget/divisor identities. Every output directory must be fresh.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
R3 = ROOT / 'experiments/revision_20260929_r3'
ROUND = 'revision_20260930_r4'
PRIMARY = 'gradual_updating_down'
CAPS = ('original_caps', 'floor_halved', 'zero')


def module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


old = module(R3 / 'analyze.py', 'r3_analysis')


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision='round_trip')


def write(rows, path: Path) -> None:
    old.write(rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows), path)


def positive(values, identity: str) -> None:
    a = np.asarray(values, dtype=float)
    if not np.isfinite(a).all() or not (a > 0).all():
        raise ValueError('positive finite domain violation: ' + identity)


def first(path: np.ndarray, threshold: float) -> tuple[bool, int]:
    crossed = np.flatnonzero(path > threshold)
    return bool(len(crossed)), int(crossed[0]) + 1 if len(crossed) else 81


def load() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    streams = read(R3 / 'results/streams.csv')
    windows = read(R3 / 'results/windows.csv')
    groups = {sid: g.sort_values('window') for sid, g in windows.groupby('stream_id', sort=False)}
    f = streams[streams.stream_id.isin(groups)].copy()
    for stat in ('GLR', 'cut_search'):
        f[stat] = [groups[sid].loc[lambda g: g.window > 40, stat].to_numpy() for sid in f.stream_id]
        positive([p.max() for p in f[stat]], stat + ' monitoring maxima')
    assert len(f) == 960 and len(windows) == 115200
    return f, windows, groups


def summary(g: pd.DataFrame) -> dict:
    primary, changed = g[g.condition == PRIMARY], g[~g.stable]
    ans = dict(primary_detections=int(primary.fired.sum()), primary_denominator=len(primary),
               primary_mean_delay=float(primary.restricted_delay.mean()),
               primary_delay_sum=float(primary.restricted_delay.sum()),
               changed_detections=int(changed.fired.sum()), changed_denominator=len(changed))
    for mode in ('frozen', 'updating'):
        sub = g[g.stable & g['mode'].eq(mode)]
        ans[mode + '_alarms'] = int(sub.fired.sum())
        ans[mode + '_denominator'] = len(sub)
    ans['actual_total_alarms'] = ans['frozen_alarms'] + ans['updating_alarms']
    return ans


def matched(f: pd.DataFrame, points: pd.DataFrame, decisions: pd.DataFrame, out: Path):
    results, stops, checkpoints, deltas, contrasts = [], [], [], [], []
    div = read(R3 / 'results/divisors.csv').set_index('divisor_id')
    levels = read(R3 / 'results/thresholds.csv').groupby(['panel', 'statistic', 'calibration', 'checkpoint_seed']).foreign_level.first()
    for p in points[points.budget.isin(CAPS)].to_dict('records'):
        source = p['point_id']
        new = dict(p, point_id=source + ':matched6', source_point_id=source,
                   family='matched_target_budget', target_budget_policy='replicate_excluded_six_both_modes',
                   divisor_identity='panel:statistic:calibration:checkpoint:exclude_replicate')
        original = decisions[decisions.point_id == source].set_index('stream_id')
        panel = f[f.panel == p['panel']]
        rows = []
        for r in panel.itertuples():
            identity = f'{r.panel}:{p["statistic"]}:{p["calibration"]}:{r.checkpoint_seed}:exclude_{r.replicate}'
            divisor = float(div.loc[identity, 'divisor'])
            base = float(levels.loc[(r.panel, p['statistic'], p['calibration'], r.checkpoint_seed)]) * divisor
            threshold = base * p['multiplier']
            fired, delay = first(getattr(r, p['statistic']), threshold)
            row = dict(original.loc[r.stream_id], point_id=new['point_id'], source_point_id=source,
                       stream_id=r.stream_id, threshold=threshold, base_threshold=base,
                       fired=fired, first_crossing=delay if fired else None, restricted_delay=delay,
                       target_budget_policy=new['target_budget_policy'], divisor_id=identity,
                       target_estimation_streams=0 if p['calibration'] == 'pooled' else 6,
                       target_membership_streams=6, foreign_calibration_streams=152)
            rows.append(row)
            before = original.loc[r.stream_id]
            if r.stable or p['calibration'] == 'pooled':
                assert threshold == before.threshold and fired == before.fired and delay == before.restricted_delay
            if not r.stable:
                deltas.append(dict(point_id=new['point_id'], source_point_id=source,
                                   target_budget_policy=new['target_budget_policy'], divisor_id=identity,
                                   panel=r.panel, stream_id=r.stream_id, checkpoint_seed=r.checkpoint_seed,
                                   condition=r.condition, replicate=r.replicate,
                                   original_threshold=before.threshold, matched_threshold=threshold,
                                   original_fired=before.fired, matched_fired=fired,
                                   original_delay=before.restricted_delay, matched_delay=delay,
                                   decision_changed=fired != before.fired, delay_difference=delay-before.restricted_delay))
        frame = pd.DataFrame(rows)
        result = dict(new, **summary(frame))
        for field in ('frozen_alarms','updating_alarms','multiplier','frozen_denominator','updating_denominator'):
            assert result[field] == p[field]
        results.append(result)
        stops += rows
        for seed, g in frame.groupby('checkpoint_seed'):
            checkpoints.append(dict(point_id=new['point_id'], source_point_id=source, panel=p['panel'],
                                    checkpoint_seed=seed, target_budget_policy=new['target_budget_policy'],
                                    divisor_identity='exclude target replicate in both stable modes', **summary(g)))
    m, s, c = map(pd.DataFrame, (results, stops, checkpoints))
    assert len(m) == 36
    for (panel, stat, budget), g in m.groupby(['panel','statistic','budget']):
        ids = g.set_index('calibration').point_id
        for seed in c[c.panel == panel].checkpoint_seed.unique():
            pooled = c[(c.point_id == ids['pooled']) & (c.checkpoint_seed == seed)].iloc[0]
            for cal in ('arithmetic','geometric'):
                row = c[(c.point_id == ids[cal]) & (c.checkpoint_seed == seed)].iloc[0]
                contrasts.append(dict(panel=panel, statistic=stat, budget=budget, calibration=cal,
                                      point_id=ids[cal], source_point_id=ids[cal].removesuffix(':matched6'),
                                      checkpoint_seed=seed, target_budget_policy='replicate_excluded_six_both_modes',
                                      divisor_identity='exclude target replicate in both stable modes',
                                      detection_difference_from_pooled=row.primary_detections-pooled.primary_detections,
                                      delay_improvement_over_pooled=pooled.primary_mean_delay-row.primary_mean_delay))
    for name, frame in [('matched_operating_points',m),('matched_stopping_times',s),('matched_checkpoint_results',c),
                        ('changed_budget_differences',deltas),('matched_checkpoint_contrasts',contrasts)]:
        write(frame,out/(name+'.csv'))
    return m,s,c


def mechanism(points: pd.DataFrame, decisions: pd.DataFrame, f: pd.DataFrame, out: Path) -> None:
    divisors = read(R3/'results/divisors.csv').set_index('divisor_id')
    members = read(R3/'results/divisor_membership.csv')
    assignments = read(R3/'results/calibration_assignments.csv')
    donors, within = [], []
    extremes = {}
    for key, group in assignments.groupby(['panel','statistic','calibration','target_checkpoint']):
        peak = group.normalized_maximum.max()
        extremes[key] = peak
        tied = group[group.normalized_maximum == peak]
        for row in tied.to_dict('records'):
            donors.append(dict(row, donor_tie_count=len(tied),foreign_calibration_records=len(group)))
    for identity, group in members.groupby('divisor_id'):
        d = divisors.loc[identity]
        ratio = group.stable_maximum.max()/d.divisor
        if d.calibration == 'arithmetic':
            assert ratio <= len(group)*(1+1e-14)
        within.append(dict(divisor_id=identity,panel=d.panel,statistic=d.statistic,calibration=d.calibration,
                           checkpoint_seed=d.checkpoint_seed,excluded_replicate=d.excluded_replicate,
                           membership_streams=len(group),maximum=group.stable_maximum.max(),divisor=d.divisor,
                           maximum_to_divisor=ratio))
    paths = {(r.stream_id,stat):getattr(r,stat) for r in f.itertuples() for stat in ('GLR','cut_search')}
    allocation, reconciled = [], []
    for key, group in points[points.calibration.isin(['arithmetic','geometric'])].groupby(['panel','statistic','family','budget']):
        panel, stat, family, budget = key
        p = group.set_index('calibration')
        a,g = p.loc['arithmetic'],p.loc['geometric']
        reps = [-1] if family == 'additional_stable' else (list(range(4)) if family == 'matched_target_budget' else [-1,0,1,2,3])
        for seed in sorted(f[f.panel==panel].checkpoint_seed.unique()):
            qa,qg = (extremes[panel,stat,cal,seed] for cal in ('arithmetic','geometric'))
            for rep in reps:
                ia,ig = [f'{panel}:{stat}:{cal}:{seed}:exclude_{rep}' for cal in ('arithmetic','geometric')]
                da,dg = divisors.loc[ia,'divisor'],divisors.loc[ig,'divisor']
                ha,hg = (qa*da)*a.multiplier,(qg*dg)*g.multiplier
                lhs,rhs = hg/ha,(g.multiplier/a.multiplier)*(dg/da)*(qg/qa)
                assert np.isclose(lhs,rhs,rtol=2e-15,atol=0)
                allocation.append(dict(panel=panel,statistic=stat,family=family,budget=budget,checkpoint_seed=seed,
                                       excluded_replicate=rep, arithmetic_point_id=a.point_id, geometric_point_id=g.point_id,
                                       arithmetic_source_point_id=a.point_id.removesuffix(':matched6'),geometric_source_point_id=g.point_id.removesuffix(':matched6'),
                                       target_budget_policy='eight' if rep==-1 else 'replicate_excluded_six_both_modes',
                                       arithmetic_divisor_id=ia,geometric_divisor_id=ig,arithmetic_divisor=da,geometric_divisor=dg,
                                       arithmetic_over_geometric=da/dg, geometric_over_arithmetic=dg/da,
                                       arithmetic_foreign_extreme=qa,geometric_foreign_extreme=qg,
                                       arithmetic_multiplier=a.multiplier,geometric_multiplier=g.multiplier,
                                       arithmetic_threshold=ha,geometric_threshold=hg,threshold_ratio=lhs,decomposed_ratio=rhs))
        left = decisions[decisions.point_id==a.point_id].set_index('stream_id')
        right = decisions[decisions.point_id==g.point_id].set_index('stream_id')
        assert set(left.index)==set(right.index)
        for sid,x in left.iterrows():
            y=right.loc[sid]
            assert x.maximum==y.maximum
            low,high=min(x.threshold,y.threshold),max(x.threshold,y.threshold)
            between=bool(low < x.maximum <= high)
            assert between==bool(x.fired!=y.fired)
            available=bool(x.stopping_time_available)
            if available:
                for record in (x,y):
                    fired,delay=first(paths[sid,stat],record.threshold)
                    assert fired==record.fired and delay==record.restricted_delay
            reconciled.append(dict(panel=panel,statistic=stat,family=family,budget=budget,stream_id=sid,
                                   checkpoint_seed=x.checkpoint_seed,condition=x.condition,stable=x.stable,
                                   arithmetic_point_id=a.point_id,geometric_point_id=g.point_id,
                                   source_point_id=a.point_id.removesuffix(':matched6'),
                                   target_budget_policy='matched_six' if family=='matched_target_budget' else ('additional_eight' if family=='additional_stable' else 'historical_six_stable_eight_changed'),
                                   divisor_identity=f'{panel}:{stat}:{{calibration}}:{x.checkpoint_seed}:exclude_{int(x.replicate) if family=="matched_target_budget" or (x.stable and family=="original_cache") else -1}',
                                   maximum=x.maximum,arithmetic_threshold=x.threshold,geometric_threshold=y.threshold,
                                   arithmetic_fired=x.fired,geometric_fired=y.fired,
                                   geometric_gained=bool(y.fired and not x.fired),geometric_lost=bool(x.fired and not y.fired),
                                   maximum_between_thresholds=between,arithmetic_delay=x.restricted_delay,
                                   geometric_delay=y.restricted_delay,delay_difference=y.restricted_delay-x.restricted_delay if available else None,
                                   stopping_time_available=available))
    for name, rows in [('foreign_extreme_donors',donors),('within_checkpoint_ratios',within),('threshold_allocation',allocation),('detection_delay_reconciliation',reconciled)]:
        write(rows,out/(name+'.csv'))


def cuts(points: pd.DataFrame, decisions: pd.DataFrame, groups: dict, out: Path) -> None:
    point_map=points.set_index('point_id')
    rows,alarms,diagnostics=[],[],[]
    normal=NormalDist()
    cut_cache,monitor={},{ }
    for sid,w in groups.items():
        z=[None if not r.eligible else float(r.z) for r in w.itertuples()]
        monitor[sid]=old.adwin_evidence(z[40:],delta=.1,minimum_subwindow=5)
    selected=decisions[decisions.point_id.isin(points[points.statistic=='cut_search'].point_id)]
    for r in selected.itertuples():
        p=point_map.loc[r.point_id]
        source=r.point_id.removesuffix(':matched6')
        meta=dict(point_id=r.point_id,source_point_id=source,panel=r.panel,stream_id=r.stream_id,
                  checkpoint_seed=r.checkpoint_seed,condition=r.condition,stable=r.stable,mode=r.mode,
                  target_budget_policy='matched_six' if p.family=='matched_target_budget' else 'historical_six_stable_eight_changed',
                  divisor_id=f'{r.panel}:cut_search:{p.calibration}:{r.checkpoint_seed}:exclude_{int(r.replicate) if r.stable or p.family=="matched_target_budget" else -1}')
        fired,delay=first(monitor[r.stream_id],r.threshold)
        diagnostics.append(dict(meta,threshold=r.threshold,full_history_fired=r.fired,full_history_delay=r.restricted_delay,
                                monitoring_only_fired=fired,monitoring_only_delay=delay,
                                diagnostic='fixed_threshold_reference_history_dependence',threshold_retuned=False))
        if not r.fired:
            continue
        end=int(r.first_crossing)+40
        cache_key=(r.stream_id,end)
        if cache_key not in cut_cache:
            w=groups[r.stream_id]
            eligible=w[w.eligible & (w.window<=end)]
            x=np.array([normal.cdf(float(v)) for v in eligible.z])
            loc=eligible.window.to_numpy(dtype=int)
            details=[]
            for k in range(5,len(x)-5+1):
                left,right=float(np.mean(x[:k])),float(np.mean(x[k:]))
                bound=float(np.sqrt(.5*np.log(40)*(1/k+1/(len(x)-k))))
                # Split location means last eligible window on the left, not the next eligible index.
                boundary=int(loc[k-1])
                details.append(dict(split_eligible_index=k,eligible_count=len(x),left_last_window=boundary,
                                    right_first_window=int(loc[k]),alarm_original_window=end,
                                    left_mean=left,right_mean=right,signed_difference=right-left,
                                    evidence=abs(left-right)/bound,bound=bound,
                                    boundary_position='before' if boundary<40 else ('at' if boundary==40 else 'after')))
            peak=max(d['evidence'] for d in details)
            assert peak==float(w.loc[w.window==end,'cut_search'].iloc[0])
            cut_cache[cache_key]=details,peak
        details,peak=cut_cache[cache_key]
        assert peak>r.threshold
        above=[d for d in details if d['evidence']>r.threshold]
        inc=any(d['signed_difference']>0 for d in above)
        dec=any(d['signed_difference']<0 for d in above)
        direction='both' if inc and dec else ('increase' if inc else 'decrease')
        winner=[d for d in details if d['evidence']==peak]
        alarms.append(dict(meta,threshold=r.threshold,first_crossing=int(r.first_crossing),alarm_original_window=end,
                           eligible_cuts=len(details),winning_cut_ties=len(winner),exceeding_cuts=len(above),
                           supported_direction=direction,
                           exceeding_before=sum(d['boundary_position']=='before' for d in above),
                           exceeding_at=sum(d['boundary_position']=='at' for d in above),
                           exceeding_after=sum(d['boundary_position']=='after' for d in above)))
        rows.extend(dict(meta,threshold=r.threshold,first_crossing=int(r.first_crossing),**d,
                         winning_cut=d['evidence']==peak,winning_cut_ties=len(winner),exceeds_threshold=d['evidence']>r.threshold)
                    for d in details)
    write(rows,out/'cut_level_diagnostics.csv')
    write(alarms,out/'cut_alarm_diagnostics.csv')
    write(diagnostics,out/'monitoring_only_diagnostics.csv')
    report=[]
    alarms=pd.DataFrame(alarms);diagnostics=pd.DataFrame(diagnostics)
    for pid,group in diagnostics.groupby('point_id',sort=False):
        p=point_map.loc[pid]
        a=alarms[(alarms.point_id==pid)&(alarms.condition==PRIMARY)]
        main=group[group.condition==PRIMARY]
        assert len(a)==int(p.primary_detections)
        report.append(dict(point_id=pid,source_point_id=pid.removesuffix(':matched6'),panel=p.panel,statistic=p.statistic,
                           calibration=p.calibration,budget=p.budget,family=p.family,target_budget_policy=group.target_budget_policy.iloc[0],
                           divisor_identity='per-stream divisor_id in monitoring_only_diagnostics.csv',
                           primary_full_history=int(main.full_history_fired.sum()),primary_monitoring_only=int(main.monitoring_only_fired.sum()),
                           full_delay=float(main.full_history_delay.mean()),monitoring_only_delay=float(main.monitoring_only_delay.mean()),
                           increase=int(a.supported_direction.eq('increase').sum()),decrease=int(a.supported_direction.eq('decrease').sum()),
                           both=int(a.supported_direction.eq('both').sum()),
                           monitoring_only_frozen_alarms=int(group.loc[group.stable & group['mode'].eq('frozen'),'monitoring_only_fired'].sum()),
                           monitoring_only_updating_alarms=int(group.loc[group.stable & group['mode'].eq('updating'),'monitoring_only_fired'].sum())))
    write(report,out/'cut_summary.csv')


def dispersion_sensitivity(f: pd.DataFrame, out: Path) -> None:
    original=read(R3/'results/dispersion.csv')
    cells,summary_rows=[] ,[]
    for row_id,r in enumerate(original.itertuples()):
        stable=f[(f.panel==r.panel)&f.stable]
        if r.estimation_streams_per_checkpoint==8:
            est=stable
            held=read(ROOT/f'experiments/stable_holdout/panel_{r.panel.lower()}/holdout_streams.csv')
            test=held.assign(logmax=np.log(held.max_statistic))
        else:
            reps=list(map(int,r.allocation.split(',')))
            est=stable[stable.replicate.isin(reps)]
            test=stable[~stable.replicate.isin(reps)].copy()
            test['logmax']=[np.log(p.max()) for p in test[r.statistic]]
        e=est.assign(maximum=[p.max() for p in est[r.statistic]])
        d=e.groupby('checkpoint_seed').maximum.apply(lambda a: float(a.mean()) if r.calibration=='arithmetic' else float(np.exp(np.log(a).mean())))
        grouped=test.groupby('checkpoint_seed').logmax
        target=grouped.mean() if r.summary=='mean_log_maximum' else (grouped.quantile(.9) if r.summary=='q90_log_maximum' else grouped.max())
        normalized=target-np.log(d)
        assert np.isclose(normalized.std(ddof=1)/target.std(ddof=1)-1,r.change,rtol=1e-12,atol=1e-12)
        changes=[]
        meta=dict(dispersion_row_id=row_id,panel=r.panel,statistic=r.statistic,calibration=r.calibration,
                  allocation=r.allocation,summary=r.summary,estimation_streams_per_checkpoint=r.estimation_streams_per_checkpoint,
                  evaluation_streams_per_checkpoint=r.evaluation_streams_per_checkpoint)
        for seed in target.index:
            raw=target.drop(seed).std(ddof=1);adjusted=normalized.drop(seed).std(ddof=1)
            change=adjusted/raw-1;changes.append(change)
            cells.append(dict(meta,omitted_checkpoint=seed,checkpoints=19,raw_sd=raw,normalized_sd=adjusted,change=change,
                              uncertainty='finite_panel_deletion_sensitivity; paired summaries and divisors retained'))
        summary_rows.append(dict(meta,full_change=r.change,minimum=min(changes),maximum=max(changes)))
    write(cells,out/'dispersion_leave_one_checkpoint_out.csv')
    write(summary_rows,out/'dispersion_sensitivity_ranges.csv')


def domain(f: pd.DataFrame, windows: pd.DataFrame, out: Path) -> None:
    rows=[]
    for panel,g in f.groupby('panel'):
        for phase in ('warmup','monitoring'):
            w=windows[(windows.panel==panel)&windows.phase.eq(phase)]
            raw=w.loc[w.eligible,'raw_score'].to_numpy()
            if not np.isfinite(raw).all():
                raise ValueError('nonfinite eligible raw score')
            inactive=int(((g.warmup_eligible<32)|~np.isfinite(g.scale)|(g.scale<1e-12)).sum())
            for stat in ('GLR','cut_search'):
                maxima=np.array([p.max() for p in g[stat]]) if phase=='monitoring' else np.array([])
                positive(maxima, panel+stat)
                rows.append(dict(panel=panel,corpus='original',phase=phase,statistic=stat,streams=len(g),inactive_streams=inactive,
                                  windows=len(w),unsupported_windows=int((~w.eligible).sum()),
                                  nonpositive_maxima=int((maxima<=0).sum()),nonfinite_maxima=int((~np.isfinite(maxima)).sum()),
                                  maximum_domain_applicable=phase=='monitoring'))
        h=read(ROOT/f'experiments/stable_holdout/panel_{panel.lower()}/holdout_streams.csv')
        positive(h.max_statistic,panel+' additional GLR')
        rows.append(dict(panel=panel,corpus='additional_stable',phase='monitoring_maxima_only',statistic='GLR',streams=len(h),
                          inactive_streams=int((~h.warmup_active).sum()),windows=None,unsupported_windows=None,
                          nonpositive_maxima=0,nonfinite_maxima=0,maximum_domain_applicable=True))
    write(rows,out/'domain_audit.csv')


def run(out: Path) -> None:
    out.mkdir(parents=True,exist_ok=False)
    for source in sorted((R3/'results').glob('*.csv')):
        (out/source.name).write_bytes(source.read_bytes())
    f,windows,groups=load()
    p,s=read(out/'operating_points.csv'),read(out/'stopping_times.csv')
    print('Loaded 960 accepted paths; generating 36 matched target-budget points',flush=True)
    m,ms,_=matched(f,p,s,out)
    all_p=pd.concat([p,m],ignore_index=True)
    all_s=pd.concat([s,ms],ignore_index=True)
    print('Threshold decomposition and detection/delay accounting',flush=True)
    mechanism(all_p,all_s,f,out)
    print('Cut-level diagnostics at accepted first alarms',flush=True)
    cuts(all_p,all_s,groups,out)
    print('Checkpoint deletion sensitivity and domain audit',flush=True)
    dispersion_sensitivity(f,out)
    domain(f,windows,out)
    inputs=list((R3/'results').glob('*.csv'))+[Path(__file__),R3/'analyze.py',ROOT/'src/rl_hanbat/socis/sequential.py']
    inputs += [ROOT/f'experiments/stable_holdout/panel_{x}/holdout_streams.csv' for x in ('d','e')]
    manifest=dict(revision=ROUND,status='retrospective_cache_analysis',historical_points=76,matched_points=36,
                  inputs={str(x.relative_to(ROOT)):old.sha(x) for x in sorted(inputs)},
                  outputs={x.name:old.sha(x) for x in sorted(out.glob('*.csv'))},
                  exclusions=['training','collection','detector fitting','raw rescoring','simulation reruns','Panel B recovery',
                              'additional stable cut paths and stopping times','population confidence intervals or significance tests'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),csvs=len(manifest['outputs']),matched_points=len(m))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output.resolve())
