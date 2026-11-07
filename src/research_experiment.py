"""Reproducible proxy map-matching evaluation; legacy implementation is preserved."""
from pathlib import Path
import argparse, hashlib, json, platform, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import Point
from pyproj import Transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import sklearn, joblib
import run_experiment_multitrip as legacy
ROOT=Path(__file__).resolve().parents[1]
legacy.ROADS_FILE=ROOT/'data/processed/osm_roads.geojson'
legacy.TRIPS_DIR=ROOT/'data/processed/trips'
TRANSFORM=Transformer.from_crs(4326,32650,always_xy=True)

def read_trip(path):
    d=pd.read_csv(path)
    if d.empty: raise ValueError(f'Empty trajectory: {path}')
    if not {'latitude','longitude','timestamp'} <= set(d): raise ValueError(f'Missing columns: {path}')
    xy=d[['longitude','latitude']].apply(pd.to_numeric,errors='coerce').to_numpy()
    if not np.isfinite(xy).all() or (abs(xy[:,0])>180).any() or (abs(xy[:,1])>90).any(): raise ValueError(f'Invalid coordinates: {path}')
    t=pd.to_datetime(d.timestamp,errors='coerce')
    if t.isna().any() or (t.diff().dt.total_seconds().dropna()<0).any(): raise ValueError(f'Invalid timestamps: {path}')
    return d,t,np.column_stack(TRANSFORM.transform(xy[:,0],xy[:,1]))

def split_ids(ids):
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=42).split(ids,groups=ids))
    return set(np.asarray(ids)[a]),set(np.asarray(ids)[b])

def nearest(point,roads):
    ids=roads.sindex.nearest(point,return_all=True)[1]
    ds=shapely.distance(roads.geometry.to_numpy()[ids],point)
    return int(ids[np.lexsort((ids,ds))[0]])

def candidates(point,roads,cap=8):
    for radius in (100,250,500,1000):
        ids=roads.sindex.query(point.buffer(radius))
        ds=shapely.distance(roads.geometry.to_numpy()[ids],point)
        keep=ds<=radius
        ids,ds=ids[keep],ds[keep]
        if len(ids)>=2: break
    if not len(ids):
        ids=np.array([nearest(point,roads)]); ds=shapely.distance(roads.geometry.to_numpy()[ids],point)
    order=np.lexsort((ids,ds))[:cap]
    return ids[order],ds[order],radius

def metrics(d,method):
    hit=d[method].to_numpy()==d.target.to_numpy(); present=d.present.to_numpy(dtype=bool)
    return dict(points=len(d),candidate_recall=present.mean(),segment_e2e=hit.mean(),segment_conditional=hit[present].mean() if present.any() else np.nan,
        class_accuracy=accuracy_score(d.target_class,d[method+'_class']),class_macro_f1=f1_score(d.target_class,d[method+'_class'],average='macro',zero_division=0))

def run(args):
    start=time.perf_counter(); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    if (out/'metadata.json').exists(): raise FileExistsError('Use a new output directory to preserve results')
    roads=legacy.load_roads(); assert roads.geometry.is_valid.all() and not roads.geometry.is_empty.any()
    # Preserve directed edges for comparability; also report orientation-invariant geometry agreement.
    canonical=shapely.to_wkb(shapely.normalize(roads.geometry.to_numpy()))
    geometry_ids=pd.factorize(canonical)[0]
    headings=np.array([legacy.line_heading(g) for g in roads.geometry]); classes=roads.road_class.to_numpy()
    trips={}; quality=[]
    for path in sorted((ROOT/'data/processed/trips').glob('trip_[0-9][0-9][0-9].csv')):
        d,t,xy=read_trip(path); delta=np.linalg.norm(np.diff(xy,axis=0),axis=1); dt=t.diff().dt.total_seconds().to_numpy()[1:]
        keep=np.linspace(0,len(d)-1,min(args.max_points,len(d)),dtype=int); sample=xy[keep]
        target=np.array([nearest(Point(p),roads) for p in sample]); dist=shapely.distance(roads.geometry.to_numpy()[target],shapely.points(sample))
        quality.append(dict(trip_id=path.stem,points=len(d),evaluated_points=len(sample),length_m=delta.sum(),start=str(t.min()),end=str(t.max()),missing_values=int(d.isna().sum().sum()),duplicate_coordinates=int(d.duplicated(['latitude','longitude']).sum()),duplicate_timestamps=int(t.duplicated().sum()),gap_median_s=float(np.median(dt)),gap_max_s=float(max(dt)),jumps_over_60_mps=int((delta/np.maximum(dt,1)>60).sum()),lon_min=d.longitude.min(),lon_max=d.longitude.max(),lat_min=d.latitude.min(),lat_max=d.latitude.max(),nearest_median_m=np.median(dist),nearest_p95_m=np.quantile(dist,.95),nearest_max_m=max(dist),nearest_ties=int(sum(len(roads.sindex.nearest(Point(p))[1])>1 for p in sample))))
        trips[path.stem]=(sample,target)
    pd.DataFrame(quality).to_csv(out/'data_quality.csv',index=False)
    train,test=split_ids(sorted(trips)); print('Split',sorted(train),sorted(test),flush=True)
    train_frames=[]; test_batches=[]; generation=[]
    for ti,(tid,(xy,targets)) in enumerate(trips.items(),1):
        print(tid,flush=True)
        for seed in ([args.seeds[0]] if tid in train else args.seeds):
            for di,kind in enumerate(legacy.DEGRADATIONS):
                for level in args.levels:
                    noisy=legacy.perturb(xy,kind,level,seed+10000*ti+100*di+int(level))
                    rows=[]; pointrows=[]; tick=time.perf_counter()
                    for pi,p in enumerate(noisy):
                        ids,ds,radius=candidates(Point(p),roads,16)
                        target=targets[pi]
                        pointrows.append(dict(trip_id=tid,seed=seed,degradation=kind,noise_m=level,point_id=pi,target=int(target),target_class=classes[target],present=target in ids[:8],present16=target in ids,candidate_count=min(8,len(ids)),candidate_count16=len(ids),radius_m=radius))
                        for rank,(cid,dist) in enumerate(zip(ids,ds)):
                            rows.append(dict(point_id=pi,candidate_id=int(cid),distance_m=dist,road_heading=headings[cid],rank=rank,road_class=classes[cid],is_target=int(cid==target)))
                    elapsed=time.perf_counter()-tick
                    c=pd.DataFrame(rows); pts=pd.DataFrame(pointrows)
                    if tid in train:
                        c=c[(c['rank']<8)&c.point_id.isin(pts.loc[pts.present,'point_id'])]; train_frames.append(c)
                    else:
                        test_batches.append((c,pts)); generation.append(dict(trip_id=tid,seed=seed,degradation=kind,noise_m=level,method='candidate_generation_16',seconds=elapsed,points=len(pts)))
    training=pd.concat(train_frames,ignore_index=True); X=legacy.feature_frame(training); columns=list(X)
    tick=time.perf_counter(); model=RandomForestClassifier(n_estimators=150,max_depth=14,class_weight='balanced',random_state=42,n_jobs=2).fit(X,training.is_target); fit_seconds=time.perf_counter()-tick
    joblib.dump(dict(model=model,feature_columns=columns,train_ids=sorted(train),test_ids=sorted(test)),out/'model.joblib')
    results=[]
    for c,pts in test_batches:
        for cap in (8,16):
            subset=c[c['rank']<cap].copy(); tick=time.perf_counter()
            subset['score']=model.predict_proba(legacy.feature_frame(subset).reindex(columns=columns,fill_value=0))[:,list(model.classes_).index(1)]
            chosen=subset.loc[subset.groupby('point_id').score.idxmax()].set_index('point_id')
            seconds=time.perf_counter()-tick
            method='rf' if cap==8 else 'rf16'; pts[method]=pts.point_id.map(chosen.candidate_id); pts[method+'_class']=pts.point_id.map(chosen.road_class)
            generation.append(dict(**pts.iloc[0][['trip_id','seed','degradation','noise_m']].to_dict(),method=method,seconds=seconds,points=len(pts)))
        subset=c[c['rank']<8]
        tick=time.perf_counter(); chosen=subset.loc[subset.groupby('point_id').distance_m.idxmin()].set_index('point_id')
        pts['nearest']=pts.point_id.map(chosen.candidate_id); pts['nearest_class']=pts.point_id.map(chosen.road_class)
        generation.append(dict(**pts.iloc[0][['trip_id','seed','degradation','noise_m']].to_dict(),method='nearest',seconds=time.perf_counter()-tick,points=len(pts)))
        # Interpretable fixed rank prior is deliberately redundant with nearest and therefore omitted.
        # Distance plus causal noisy heading is a simple alternative; no labels or future observations.
        xy=trips[pts.trip_id.iloc[0]][0]; ti=list(trips).index(pts.trip_id.iloc[0])+1; di=legacy.DEGRADATIONS.index(pts.degradation.iloc[0]); level=pts.noise_m.iloc[0]
        noisy=legacy.perturb(xy,pts.degradation.iloc[0],level,int(pts.seed.iloc[0])+10000*ti+100*di+int(level))
        disp=np.diff(noisy,axis=0,prepend=noisy[:1]); heading=np.arctan2(disp[:,1],disp[:,0]); tick=time.perf_counter(); subset=subset.copy()
        angle=np.abs(np.sin(subset.road_heading.to_numpy()-heading[subset.point_id]))
        angle[np.linalg.norm(disp[subset.point_id],axis=1)<10]=0
        subset['cost']=subset.distance_m+10*angle
        chosen=subset.loc[subset.groupby('point_id').cost.idxmin()].set_index('point_id'); pts['heading']=pts.point_id.map(chosen.candidate_id); pts['heading_class']=pts.point_id.map(chosen.road_class)
        generation.append(dict(**pts.iloc[0][['trip_id','seed','degradation','noise_m']].to_dict(),method='heading',seconds=time.perf_counter()-tick,points=len(pts)))
        for m in ('nearest','rf','rf16','heading'): pts[m+'_geometry_correct']=geometry_ids[pts[m].astype(int)]==geometry_ids[pts.target]
        results.append(pts)
    data=pd.concat(results,ignore_index=True); data.to_csv(out/'predictions.csv',index=False)
    summary=[]; per=[]; classrows=[]; conf=[]
    for keys,g in data.groupby(['seed','degradation','noise_m']):
        base=dict(zip(['seed','degradation','noise_m'],keys))
        for method in ('nearest','rf','heading','rf16'):
            z=g.copy()
            if method=='rf16': z['present']=z.present16
            summary.append(dict(**base,method=method,**metrics(z,method),geometry_accuracy=z[method+'_geometry_correct'].mean()))
            for tid,h in z.groupby('trip_id'): per.append(dict(**base,method=method,trip_id=tid,**metrics(h,method)))
            rep=classification_report(z.target_class,z[method+'_class'],output_dict=True,zero_division=0)
            for label,v in rep.items():
                if isinstance(v,dict): classrows.append(dict(**base,method=method,road_class=label,**v))
            for (a,b),n in z.groupby(['target_class',method+'_class']).size().items(): conf.append(dict(**base,method=method,true_class=a,predicted_class=b,count=n))
    pd.DataFrame(summary).to_csv(out/'metrics.csv',index=False); per=pd.DataFrame(per); per.to_csv(out/'per_trajectory.csv',index=False)
    pd.DataFrame(classrows).to_csv(out/'class_metrics.csv',index=False); pd.DataFrame(conf).to_csv(out/'confusion.csv',index=False)
    pd.DataFrame(generation).to_csv(out/'runtime.csv',index=False)
    data.groupby(['seed','degradation','noise_m']).agg(recall8=('present','mean'),recall16=('present16','mean'),count_mean=('candidate_count','mean'),count_min=('candidate_count','min'),count16_mean=('candidate_count16','mean'),radius_mean_m=('radius_m','mean'),radius_max_m=('radius_m','max')).to_csv(out/'candidate_diagnostics.csv')
    uncertainty=[]
    for keys,g in per.groupby(['degradation','noise_m']):
        pivot=g.pivot_table(index='trip_id',columns='method',values='segment_e2e'); diff=(pivot.rf-pivot.nearest).to_numpy(); rng=np.random.default_rng(42); bs=rng.choice(diff,(10000,len(diff)),replace=True).mean(axis=1)
        uncertainty.append(dict(degradation=keys[0],noise_m=keys[1],mean_difference=diff.mean(),median_difference=np.median(diff),ci_low=np.quantile(bs,.025),ci_high=np.quantile(bs,.975),trajectories=len(diff)))
    pd.DataFrame(uncertainty).to_csv(out/'uncertainty.csv',index=False)
    files=[ROOT/'data/processed/osm_roads.geojson',*sorted((ROOT/'data/processed/trips').glob('*.csv'))]
    meta=dict(config=vars(args),created_utc=datetime.now(timezone.utc).isoformat(),train_ids=sorted(train),test_ids=sorted(test),roads=len(roads),unique_normalized_geometries=len(set(canonical)),fit_seconds=fit_seconds,total_seconds=time.perf_counter()-start,versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,geopandas=gpd.__version__,shapely=shapely.__version__,sklearn=sklearn.__version__),sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},model=model.get_params(),label='nearest clean GPS OSM directed edge; ties ordered by distance then row ID; proxy only',runtime='perf_counter wall time; shared 16-candidate search; batch feature encoding, prediction and selection timed; excludes IO and projection; not streaming latency',training='first degradation seed only; fixed historical hyperparameters, no tuning; test seeds never used in fitting')
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)); print('Completed',meta['total_seconds'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',default='results/final'); p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45,46]); p.add_argument('--levels',nargs='+',type=int,default=[5,10,20]); p.add_argument('--max-points',type=int,default=300); p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    if a.smoke: a.max_points=20; a.seeds=[42]; a.levels=[10]
    if a.max_points<2 or any(x<0 for x in a.levels): p.error('Invalid point count or severity')
    run(a)
