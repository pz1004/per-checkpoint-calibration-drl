"""Independent R4 arithmetic, membership, stopping, cut and deletion audit.

Does not import the R4 analysis or production sequential implementation.
Accepted R3 float paths determine stopping times; separately reconstructed cut
means use compensated sums and record their numerical error.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
R3=ROOT/'experiments/revision_20260929_r3/results'
PRIMARY='gradual_updating_down'


def read(p):
    return pd.read_csv(p,float_precision='round_trip')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a,b,what):
    require(np.allclose(a,b,rtol=2e-12,atol=2e-12,equal_nan=True),what)


def positive(x):
    require(np.isfinite(x).all() and (np.asarray(x)>0).all(),'positive finite domain')


def stopping(path,h):
    for index,value in enumerate(path,1):
        if value>h:
            return True,index
    return False,81


def audit(results: Path) -> dict:
    report={}
    for source in sorted(R3.glob('*.csv')):
        require(source.read_bytes()==(results/source.name).read_bytes(),'historical CSV bytes '+source.name)
    report['historical_csvs_byte_identical']=12
    points=read(results/'operating_points.csv');matched=read(results/'matched_operating_points.csv')
    require(len(points)==76 and len(matched)==36,'76 plus 36 point coverage')
    for panel,expected in [('D',[50,68]),('E',[0,4])]:
        p=points[(points.panel==panel)&(points.statistic=='GLR')&(points.budget=='native')&(points.family=='original_cache')].set_index('calibration')
        require([int(p.loc[c,'primary_detections']) for c in ('pooled','arithmetic')]==expected,'historical endpoint')
    windows=read(results/'windows.csv');streams=read(results/'streams.csv')
    paths={(sid,stat):g.loc[g.window>40,stat].to_numpy() for sid,g in windows.groupby('stream_id',sort=False) for stat in ('GLR','cut_search')}
    meta=streams.set_index('stream_id')
    stable=streams[streams.stable & streams.stream_id.isin(windows.stream_id)]
    maxima={(r.stream_id,stat):float(paths[r.stream_id,stat].max()) for r in stable.itertuples() for stat in ('GLR','cut_search')}
    positive(list(maxima.values()))
    divisors={};members={}
    for panel,g in stable.groupby('panel'):
        for seed,h in g.groupby('checkpoint_seed'):
            for excluded in [-1,0,1,2,3]:
                subset=h if excluded==-1 else h[h.replicate!=excluded]
                require(len(subset)==(8 if excluded==-1 else 6),'declared membership count')
                for stat in ('GLR','cut_search'):
                    vals=[maxima[sid,stat] for sid in subset.stream_id]
                    for cal in ('pooled','arithmetic','geometric'):
                        value=1. if cal=='pooled' else (math.fsum(vals)/len(vals) if cal=='arithmetic' else math.exp(math.fsum(map(math.log,vals))/len(vals)))
                        identity=f'{panel}:{stat}:{cal}:{seed}:exclude_{excluded}'
                        divisors[identity]=value;members[identity]=set(subset.stream_id)
    exported=read(results/'divisors.csv').set_index('divisor_id')
    for identity,value in divisors.items():close(value,exported.loc[identity,'divisor'],'independent divisor')
    for identity,g in read(results/'divisor_membership.csv').groupby('divisor_id'):
        require(set(g.stream_id)==members[identity],'exact divisor membership')
    assignments=read(results/'calibration_assignments.csv')
    extremes={};donor_expected=set()
    for key,g in assignments.groupby(['panel','statistic','calibration','target_checkpoint']):
        panel,stat,cal,target=key
        expected=stable[(stable.panel==panel)&(stable.checkpoint_seed!=target)]
        require(len(g)==152 and set(g.foreign_stream_id)==set(expected.stream_id),'152 foreign membership')
        vals=[]
        for r in g.itertuples():
            require(r.foreign_checkpoint==meta.loc[r.foreign_stream_id,'checkpoint_seed'],'foreign checkpoint identity')
            d=divisors[f'{panel}:{stat}:{cal}:{r.foreign_checkpoint}:exclude_-1']
            value=maxima[r.foreign_stream_id,stat]/d
            close(value,r.normalized_maximum,'foreign ratio')
            vals.append(value)
        extremes[key]=max(vals)
        # Exact accepted arithmetic defines float ties; independent values agree above.
        for r in g[g.normalized_maximum==g.normalized_maximum.max()].itertuples():
            donor_expected.add((*key,r.foreign_stream_id))
    donors=read(results/'foreign_extreme_donors.csv')
    actual={(r.panel,r.statistic,r.calibration,r.target_checkpoint,r.foreign_stream_id) for r in donors.itertuples()}
    require(actual==donor_expected,'all extreme donors including ties')
    for key,g in donors.groupby(['panel','statistic','calibration','target_checkpoint']):
        require(g.donor_tie_count.eq(len(g)).all(),'donor tie count')
    stops=read(results/'matched_stopping_times.csv')
    old=read(results/'stopping_times.csv').set_index(['point_id','stream_id'])
    source_points=points.set_index('point_id');pm=matched.set_index('point_id')
    for p in matched.itertuples():
        original=source_points.loc[p.source_point_id]
        for field in ['multiplier','frozen_alarms','updating_alarms','frozen_denominator','updating_denominator']:
            require(getattr(p,field)==original[field],'stable selection unchanged '+field)
    for r in stops.itertuples():
        p=pm.loc[r.point_id]
        expected_id=f'{r.panel}:{p.statistic}:{p.calibration}:{r.checkpoint_seed}:exclude_{r.replicate}'
        require(r.divisor_id==expected_id and r.target_membership_streams==6,'matched six divisor identity')
        d=divisors[expected_id];q=extremes[r.panel,p.statistic,p.calibration,r.checkpoint_seed]
        close(q*d*p.multiplier,r.threshold,'matched threshold')
        fired,delay=stopping(paths[r.stream_id,p.statistic],r.threshold)
        require(fired==r.fired and delay==r.restricted_delay,'strict crossing')
        before=old.loc[(r.source_point_id,r.stream_id)]
        if r.stable or p.calibration=='pooled':
            require(r.threshold==before.threshold and r.fired==before.fired and r.restricted_delay==before.restricted_delay,'unchanged stable or pooled')
    checkpoints=read(results/'matched_checkpoint_results.csv')
    for p in matched.itertuples():
        all_rows=stops[stops.point_id==p.point_id]
        for seed,g in all_rows.groupby('checkpoint_seed'):
            c=checkpoints[(checkpoints.point_id==p.point_id)&(checkpoints.checkpoint_seed==seed)].iloc[0]
            prim=g[g.condition==PRIMARY]
            require(c.primary_detections==prim.fired.sum() and c.primary_delay_sum==prim.restricted_delay.sum(),'checkpoint counts/delay')
        primary=all_rows[all_rows.condition==PRIMARY]
        require(p.primary_detections==primary.fired.sum() and p.primary_delay_sum==primary.restricted_delay.sum(),'aggregate primary')
        require(p.primary_mean_delay==primary.restricted_delay.mean(),'aggregate delay')
    differences=read(results/'changed_budget_differences.csv')
    matched_index=stops.set_index(['point_id','stream_id'])
    require(len(differences)==11520,'all changed budget differences')
    for r in differences.itertuples():
        a=old.loc[(r.source_point_id,r.stream_id)];b=matched_index.loc[(r.point_id,r.stream_id)]
        require(r.original_threshold==a.threshold and r.matched_threshold==b.threshold,'budget threshold links')
        require(r.original_fired==a.fired and r.matched_fired==b.fired,'budget decision links')
        require(r.original_delay==a.restricted_delay and r.matched_delay==b.restricted_delay,'budget delay links')
        require(r.decision_changed==bool(a.fired!=b.fired) and r.delay_difference==b.restricted_delay-a.restricted_delay,'budget difference identities')
    report['changed_budget_difference_rows']=len(differences)
    report.update(divisors=len(divisors),foreign_assignments=len(assignments),extreme_donors=len(donors),matched_decisions=len(stops))
    allocation=read(results/'threshold_allocation.csv')
    for r in allocation.itertuples():
        da,dg=divisors[r.arithmetic_divisor_id],divisors[r.geometric_divisor_id]
        qa,qg=(extremes[r.panel,r.statistic,cal,r.checkpoint_seed] for cal in ('arithmetic','geometric'))
        close(r.arithmetic_divisor,da,'allocation arithmetic divisor');close(r.geometric_divisor,dg,'allocation geometric divisor')
        close(r.arithmetic_threshold,qa*da*r.arithmetic_multiplier,'arithmetic effective threshold')
        close(r.geometric_threshold,qg*dg*r.geometric_multiplier,'geometric effective threshold')
        ratio=r.geometric_threshold/r.arithmetic_threshold
        close(ratio,(r.geometric_multiplier/r.arithmetic_multiplier)*(dg/da)*(qg/qa),'threshold ratio identity')
        close(ratio,r.threshold_ratio,'exported threshold ratio');close(ratio,r.decomposed_ratio,'exported decomposition')
    within=read(results/'within_checkpoint_ratios.csv')
    for r in within.itertuples():
        vals=[maxima[sid,r.statistic] for sid in members[r.divisor_id]]
        close(r.maximum_to_divisor,max(vals)/divisors[r.divisor_id],'within ratio')
        if r.calibration=='arithmetic':require(r.maximum_to_divisor<=len(vals)*(1+1e-14),'arithmetic structural bound')
    rec=read(results/'detection_delay_reconciliation.csv')
    all_decisions=pd.concat([old.reset_index(),stops],ignore_index=True).set_index(['point_id','stream_id'])
    for r in rec.itertuples():
        a=all_decisions.loc[(r.arithmetic_point_id,r.stream_id)];g=all_decisions.loc[(r.geometric_point_id,r.stream_id)]
        require(r.geometric_gained==bool(g.fired and not a.fired) and r.geometric_lost==bool(a.fired and not g.fired),'gained lost identity')
        require(r.maximum_between_thresholds==(min(a.threshold,g.threshold)<a.maximum<=max(a.threshold,g.threshold)),'between-threshold identity')
        if r.stopping_time_available:
            for x in (a,g):
                fired,delay=stopping(paths[r.stream_id,r.statistic],x.threshold)
                require(fired==x.fired and delay==x.restricted_delay,'reconciled accepted crossing')
            require(r.delay_difference==g.restricted_delay-a.restricted_delay,'reconciled delay difference')
    report.update(threshold_decompositions=len(allocation),detection_delay_reconciliations=len(rec))
    # Independent eligible-index mapping and compensated left/right means.
    normal=NormalDist();eligible={};monitor={};max_cut_error=0.
    for sid,w in windows.groupby('stream_id',sort=False):
        e=w[w.eligible]
        eligible[sid]=(e.window.to_numpy(dtype=int),np.array([normal.cdf(float(x)) for x in e.z]))
        observed=[];path=[]
        for r in w[w.window>40].itertuples():
            if r.eligible:observed.append(normal.cdf(float(r.z)))
            if not r.eligible:
                path.append(path[-1] if path else 0.);continue
            n=len(observed);prefix=np.cumsum([0.]+observed);evidence=0.
            for k in range(5,n-4):
                difference=prefix[k]/k-(prefix[n]-prefix[k])/(n-k)
                value=abs(difference)/math.sqrt(.5*math.log(40)*(1/k+1/(n-k)))
                evidence=max(evidence,value)
            path.append(evidence)
        monitor[sid]=np.array(path)
    cuts=read(results/'cut_level_diagnostics.csv')
    alarm=read(results/'cut_alarm_diagnostics.csv').set_index(['point_id','stream_id'])
    cut_moments={}
    for r in cuts.itertuples():
        key=(r.stream_id,r.alarm_original_window,r.split_eligible_index)
        if key not in cut_moments:
            loc,x=eligible[r.stream_id];mask=loc<=r.alarm_original_window;loc,x=loc[mask],x[mask]
            k=r.split_eligible_index;n=len(x)
            require(5<=k<=n-5,'five-observation eligibility')
            left=math.fsum(x[:k])/k;right=math.fsum(x[k:])/(n-k)
            cut_moments[key]=(int(loc[k-1]),int(loc[k]),n,left,right)
        last,next_window,n,left,right=cut_moments[key]
        require(r.left_last_window==last and r.right_first_window==next_window and r.eligible_count==n,'cut original windows')
        require(r.boundary_position==('before' if last<40 else ('at' if last==40 else 'after')),'warmup boundary class')
        close([r.left_mean,r.right_mean,r.signed_difference],[left,right,right-left],'cut signed means')
        evidence=abs(right-left)/math.sqrt(.5*math.log(40)*(1/r.split_eligible_index+1/(n-r.split_eligible_index)))
        max_cut_error=max(max_cut_error,abs(evidence-r.evidence));close(evidence,r.evidence,'cut evidence')
        require(r.exceeds_threshold==(r.evidence>r.threshold),'strict cut exceedance')
        require(r.alarm_original_window==r.first_crossing+40,'alarm original index')
    for (pid,sid),g in cuts.groupby(['point_id','stream_id'],sort=False):
        a=alarm.loc[(pid,sid)];x=all_decisions.loc[(pid,sid)]
        peak=g.evidence.max();w=g.evidence.eq(peak);above=g[g.exceeds_threshold]
        require(g.winning_cut.equals(w) and g.winning_cut_ties.eq(int(w.sum())).all(),'winning cut ties')
        require(len(g)==g.eligible_count.iloc[0]-9 and g.split_eligible_index.tolist()==list(range(5,int(g.eligible_count.iloc[0])-4)),'all eligible cuts')
        require(a.first_crossing==x.first_crossing and bool(x.fired),'accepted cut first alarm')
        require(a.exceeding_cuts==len(above),'all exceeding cuts')
        inc=(above.signed_difference>0).any();dec=(above.signed_difference<0).any()
        require(a.supported_direction==('both' if inc and dec else ('increase' if inc else 'decrease')),'alarm direction')
        for position in ('before','at','after'):
            require(a['exceeding_'+position]==above.boundary_position.eq(position).sum(),'boundary alarm count')
    monitor_diag=read(results/'monitoring_only_diagnostics.csv');boundary=[]
    for r in monitor_diag.itertuples():
        fired,delay=stopping(monitor[r.stream_id],r.threshold)
        if fired!=r.monitoring_only_fired or delay!=r.monitoring_only_delay:
            gap=float(np.min(np.abs(monitor[r.stream_id]-r.threshold)))
            require(gap<=2e-12,'monitoring-only independent mismatch beyond rounding')
            boundary.append(dict(point_id=r.point_id,stream_id=r.stream_id,minimum_gap=gap))
    summaries=read(results/'cut_summary.csv')
    for r in summaries.itertuples():
        d=monitor_diag[monitor_diag.point_id==r.point_id]
        primary=d[d.condition==PRIMARY]
        require(r.primary_full_history==primary.full_history_fired.sum(),'cut summary full count')
        require(r.primary_monitoring_only==primary.monitoring_only_fired.sum(),'cut summary diagnostic count')
        close([r.full_delay,r.monitoring_only_delay],[primary.full_history_delay.mean(),primary.monitoring_only_delay.mean()],'cut summary delays')
        alarms=alarm.loc[r.point_id];prim=alarms[alarms.condition==PRIMARY]
        for direction in ('increase','decrease','both'):
            require(getattr(r,direction)==prim.supported_direction.eq(direction).sum(),'cut direction summary')
        for mode in ('frozen','updating'):
            require(getattr(r,'monitoring_only_'+mode+'_alarms')==d.loc[d.stable & d['mode'].eq(mode),'monitoring_only_fired'].sum(),'cut diagnostic stable count')
    report.update(cut_records=len(cuts),cut_alarm_records=len(alarm),maximum_independent_cut_error=max_cut_error,
                  monitoring_only_records=len(monitor_diag),monitoring_only_rounding_boundary_cases=boundary)
    dispersion=read(results/'dispersion.csv');deletion=read(results/'dispersion_leave_one_checkpoint_out.csv')
    ranges=read(results/'dispersion_sensitivity_ranges.csv')
    for index,r in enumerate(dispersion.itertuples()):
        panel=stable[stable.panel==r.panel]
        seeds=sorted(panel.checkpoint_seed.unique());targets=[];scales=[]
        if r.estimation_streams_per_checkpoint==8:
            test=read(ROOT/f'experiments/stable_holdout/panel_{r.panel.lower()}/holdout_streams.csv')
            estimation=panel
        else:
            reps=list(map(int,r.allocation.split(',')))
            estimation=panel[panel.replicate.isin(reps)]
            test=panel[~panel.replicate.isin(reps)].copy()
            test['max_statistic']=[maxima[sid,r.statistic] for sid in test.stream_id]
        for seed in seeds:
            est=[maxima[sid,r.statistic] for sid in estimation[estimation.checkpoint_seed==seed].stream_id]
            positive(est)
            scale=math.fsum(est)/len(est) if r.calibration=='arithmetic' else math.exp(math.fsum(map(math.log,est))/len(est))
            logvalues=np.log(test[test.checkpoint_seed==seed].max_statistic.to_numpy())
            target=math.fsum(logvalues)/len(logvalues) if r.summary=='mean_log_maximum' else (float(np.quantile(logvalues,.9)) if r.summary=='q90_log_maximum' else float(max(logvalues)))
            targets.append(target);scales.append(math.log(scale))
        targets=np.array(targets);scaled=targets-np.array(scales);values=[]
        for seed in seeds:
            keep=np.array(seeds)!=seed
            value=scaled[keep].std(ddof=1)/targets[keep].std(ddof=1)-1
            x=deletion[(deletion.dispersion_row_id==index)&(deletion.omitted_checkpoint==seed)]
            require(len(x)==1,'each deletion present');close(x.change.iloc[0],value,'paired deletion SD')
            values.append(value)
        x=ranges[ranges.dispersion_row_id==index].iloc[0]
        close([x.minimum,x.maximum],[min(values),max(values)],'deletion range')
    require(len(deletion)==3120 and len(ranges)==156,'dispersion coverage')
    report.update(dispersion_rows=156,paired_deletion_rows=3120)
    domain=read(results/'domain_audit.csv')
    require(domain.nonpositive_maxima.eq(0).all() and domain.nonfinite_maxima.eq(0).all(),'domain counts')
    require(domain.inactive_streams.eq(0).all(),'inactive streams')
    for panel,expected in [('D',[0,0]),('E',[9,28])]:
        for phase,count in zip(['warmup','monitoring'],expected):
            w=windows[(windows.panel==panel)&(windows.phase==phase)]
            require(int((~w.eligible).sum())==count,'abstention counts')
            require(domain[(domain.panel==panel)&(domain.phase==phase)].unsupported_windows.eq(count).all(),'exported abstentions')
    report['status']='PASS'
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),'fresh audit output required')
    report=audit(a.results)
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
