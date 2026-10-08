"""Changed-outcome invariance and meaningful corruption/positive-domain probes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
import analyze
import verify


def run(results:Path,output:Path):
    assert not output.exists()
    probes=[]
    f,_,_=analyze.load();p=analyze.read(results/'operating_points.csv');s=analyze.read(results/'stopping_times.csv')
    # The synthetic historical-decision fixture must reflect the same perturbed
    # paths, so the production pooled-outcome regression guard remains active.
    s.loc[~s.stable,'fired']=True
    s.loc[~s.stable,'restricted_delay']=1.
    s.loc[~s.stable,'first_crossing']=1.
    s.loc[~s.stable,'maximum']=1e12
    altered=f.copy(deep=True)
    for stat in ('GLR','cut_search'):
        altered[stat]=[path if stable else np.full_like(path,1e12) for path,stable in zip(altered[stat],altered.stable)]
    with tempfile.TemporaryDirectory(prefix='r4-changed-perturbation-') as tmp:
        tmp=Path(tmp)
        m,stops,_=analyze.matched(altered,p,s,tmp)
        original=analyze.read(results/'matched_operating_points.csv').set_index('point_id')
        for row in m.itertuples():
            for field in ('multiplier','frozen_alarms','updating_alarms'):
                verify.require(getattr(row,field)==original.loc[row.point_id,field],'changed perturbation altered selection')
        before=analyze.read(results/'matched_stopping_times.csv').set_index(['point_id','stream_id'])
        after=stops.set_index(['point_id','stream_id'])
        verify.require(before.index.equals(after.index),'same stream/point interface')
        for field in ('threshold','base_threshold','divisor_id'):
            verify.require(before[field].equals(after[field]),'changed perturbation altered divisor/threshold')
        verify.require((m.primary_detections==80).all(),'changed paths were actually perturbed')
        # Independent re-fit interface accepts stable data only; changed outcomes cannot reach it.
        for panel in ('D','E'):
            a=f[(f.panel==panel)&f.stable];b=altered[(altered.panel==panel)&altered.stable]
            for stat in ('GLR','cut_search'):
                for cal in ('pooled','arithmetic','geometric'):
                    x=analyze.old.fit(a,stat,cal);y=analyze.old.fit(b,stat,cal)
                    verify.require(x==y,'changed outcome changed fitted divisor or membership')
        probes.append({'case':'changed outcomes replaced by 1e12','result':'PASS','points':36,'thresholds':17280,'changed_decisions':'all primary detect','stable_selection':'identical'})
    # Mutate actual cache paths, not just an isolated numeric helper.
    for value,label in [(0.,'zero'),(-1.,'negative'),(float('nan'),'nonfinite')]:
        with tempfile.TemporaryDirectory(prefix='r4-domain-probe-') as tmp:
            tmp=Path(tmp);(tmp/'results').mkdir()
            shutil.copy2(analyze.R3/'results/streams.csv',tmp/'results/streams.csv')
            windows=analyze.read(analyze.R3/'results/windows.csv')
            sid=windows.stream_id.iloc[0]
            windows.loc[(windows.stream_id==sid)&(windows.window>40),'GLR']=value
            analyze.write(windows,tmp/'results/windows.csv')
            original=analyze.R3;analyze.R3=tmp
            try:
                try:analyze.load()
                except ValueError as error:
                    verify.require('positive finite domain' in str(error),'explicit domain error')
                    probes.append({'case':label+' monitoring maximum','result':'REJECTED','error':str(error)})
                else:raise ValueError('domain corruption accepted')
            finally:analyze.R3=original
    # Full verifier rejects linked-table corruptions. Existing CSVs stay intact.
    cases=[('matched_stopping_times.csv','divisor_id','wrong_divisor'),
           ('foreign_extreme_donors.csv','foreign_stream_id','wrong_donor'),
           ('threshold_allocation.csv','decomposed_ratio',-99.)]
    for name,column,value in cases:
        with tempfile.TemporaryDirectory(prefix='r4-audit-corruption-') as tmp:
            tmp=Path(tmp)
            for path in results.glob('*.csv'):(tmp/path.name).symlink_to(path.resolve())
            mutated=analyze.read(results/name);mutated.loc[0,column]=value
            (tmp/name).unlink();analyze.write(mutated,tmp/name)
            try:verify.audit(tmp)
            except ValueError as error:probes.append({'case':name+':'+column,'result':'REJECTED','error':str(error)})
            else:raise ValueError('corrupted export accepted: '+name)
    output.write_text(json.dumps({'status':'PASS','probes':probes},indent=2)+'\n')
    print(output.read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.results,a.output)
