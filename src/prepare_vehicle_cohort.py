"""Build an additive labelled vehicle cohort; never modifies original trips or raw files."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,shapely
from shapely.geometry import Point
import research_experiment as r
from prepare_trips import read_geolife_plt,RAW_ROOT

def main():
    config=json.loads((r.ROOT/'configs/extension_protocol.json').read_text())
    out=r.ROOT/'data/processed/vehicle_cohort'
    if out.exists(): raise FileExistsError('Cohort exists; preserve it and choose a new version')
    out.mkdir(); roads=r.legacy.load_roads(); bounds=roads.to_crs(4326).total_bounds
    labels=sorted(RAW_ROOT.glob('*/labels.txt')); rng=np.random.default_rng(42); order=rng.permutation(len(labels))
    records=[]; log=[]; accepted_users=0
    for ix in order:
        labpath=labels[ix]; user=labpath.parent.name
        lab=pd.read_csv(labpath,sep='\t'); lab.columns=lab.columns.str.strip()
        lab=lab[lab['Transportation Mode'].str.strip().isin(['car','taxi'])].copy()
        if lab.empty: continue
        lab['start']=pd.to_datetime(lab['Start Time']); lab['end']=pd.to_datetime(lab['End Time']); lab=lab.sort_values('start')
        selected=[]; used_intervals=set()
        for path in sorted((labpath.parent/'Trajectory').glob('*.plt')):
            fileday=pd.to_datetime(path.stem,format='%Y%m%d%H%M%S',errors='coerce')
            if pd.isna(fileday): continue
            candidates=lab[(lab.start.dt.date<=fileday.date())&(lab.end.dt.date>=fileday.date())]
            if candidates.empty: continue
            try: d=read_geolife_plt(path)
            except (ValueError,OSError,pd.errors.ParserError) as exc:
                log.append(dict(user=user,file=path.name,reason=str(exc))); continue
            for idx,row in candidates.iterrows():
                if idx in used_intervals: continue
                part=d[d.timestamp.between(row.start,row.end)].copy()
                if len(part)<config['min_points']: continue
                # Split time gaps rather than joining discontinuous fragments.
                blocks=(part.timestamp.diff().dt.total_seconds().fillna(0)>config['max_gap_s']).cumsum()
                for _,block in part.groupby(blocks):
                    reason=None
                    if len(block)<config['min_points']: continue
                    if (block.timestamp.iloc[-1]-block.timestamp.iloc[0]).total_seconds()<config['min_duration_s']: continue
                    inside=block.longitude.between(bounds[0],bounds[2])&block.latitude.between(bounds[1],bounds[3])
                    if not inside.all(): reason='outside existing network bounding box'
                    if reason:
                        log.append(dict(user=user,file=path.name,reason=reason)); continue
                    sample=block.iloc[np.linspace(0,len(block)-1,min(100,len(block)),dtype=int)]
                    xy=np.column_stack(r.TRANSFORM.transform(sample.longitude,sample.latitude)); pts=shapely.points(xy)
                    targets=np.array([r.nearest(p,roads) for p in pts]); ds=shapely.distance(roads.geometry.to_numpy()[targets],pts)
                    if np.median(ds)>50 or np.quantile(ds,.95)>100:
                        log.append(dict(user=user,file=path.name,reason='road proximity quality threshold')); continue
                    selected.append((block,dict(source_user_id=user,source_file=str(path.relative_to(r.ROOT)),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),labels_sha256=hashlib.sha256(labpath.read_bytes()).hexdigest(),mode=row['Transportation Mode'],label_start=str(row.start),label_end=str(row.end),start=str(block.timestamp.iloc[0]),end=str(block.timestamp.iloc[-1]),points=len(block),screen_median_m=np.median(ds),screen_p95_m=np.quantile(ds,.95))))
                    used_intervals.add(idx); break
                if len(selected)>=config['trips_per_user']: break
            if len(selected)>=config['trips_per_user']: break
        if len(selected)<config['trips_per_user']:
            log.append(dict(user=user,file='',reason=f'fewer than {config["trips_per_user"]} eligible intervals')); continue
        accepted_users+=1
        for block,rec in selected:
            tid=f'trip_{len(records)+1:03d}'; block.to_csv(out/(tid+'.csv'),index=False); records.append(dict(trip_id=tid,**rec))
        print('Accepted',user,'users',accepted_users,'trips',len(records),flush=True)
        if accepted_users>=config['max_users']: break
    pd.DataFrame(records).to_csv(out/'trip_summary.csv',index=False); pd.DataFrame(log).to_csv(out/'screening_log.csv',index=False)
    (out/'selection.json').write_text(json.dumps(dict(config=config,users=accepted_users,trips=len(records),label_files_available=len(labels),network_bounds=bounds.tolist()),indent=2))
    if accepted_users<8: raise RuntimeError('Insufficient eligible users for planned extension')
if __name__=='__main__': main()
