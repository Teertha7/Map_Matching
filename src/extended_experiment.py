"""User-disjoint vehicle experiment with canonical road identities and corruption shift.

All model settings are fixed in advance. No test-driven selection is performed.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import platform
import time

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import shapely
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import classification_report, roc_curve, roc_auc_score

import research_experiment as core

ROOT = core.ROOT
PATTERNS = core.legacy.DEGRADATIONS
METHODS = ['nearest', 'heading', 'rf_original', 'rf_context', 'rf_gaussian']


def canonical_roads(roads):
    """Collapse only exact normalized geometry AND class duplicates, preserving provenance."""
    roads = roads.copy()
    roads['original_row'] = np.arange(len(roads))
    normalized = shapely.normalize(roads.geometry.to_numpy())
    roads['identity'] = [hashlib.sha256(wkb + str(cls).encode()).hexdigest()
                         for wkb, cls in zip(shapely.to_wkb(normalized), roads.road_class)]
    roads.geometry = normalized
    mapping = roads[['original_row', 'identity']].copy()
    roads = roads.drop_duplicates('identity').sort_values('identity').reset_index(drop=True)
    lookup = dict(zip(roads.identity, roads.index))
    mapping['canonical_id'] = mapping.identity.map(lookup)
    return roads, mapping


def user_split(summary):
    groups = summary.source_user_id.astype(str).to_numpy()
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=42)
                .split(summary, groups=groups))
    train = set(summary.trip_id.iloc[a]); test = set(summary.trip_id.iloc[b])
    assert not set(groups[a]) & set(groups[b])
    return train, test


def feature_frame(rows, contextual=False):
    columns = ['distance_m', 'road_heading', 'rank', 'road_class']
    if contextual:
        # Undirected local orientation, causal displacement and candidate competition.
        columns = ['distance_m', 'rank', 'heading_mismatch', 'log_length',
                   'candidate_count', 'distance_gap_m', 'road_class']
    return pd.get_dummies(rows[columns], columns=['road_class'], dtype=float)


def make_batch(xy, timestamps, target, roads, kind, level, seed, trip_number):
    noisy = core.legacy.perturb(xy, kind, level, seed + 10000*trip_number + 100*PATTERNS.index(kind) + level)
    displacements = np.diff(noisy, axis=0, prepend=noisy[:1])
    intervals = np.diff(timestamps, prepend=timestamps[:1])
    bearings = np.arctan2(displacements[:, 1], displacements[:, 0])
    rows = []; points = []
    geoms = roads.geometry.to_numpy(); classes = roads.road_class.to_numpy()
    headings = roads.endpoint_heading.to_numpy()
    for i, coordinate in enumerate(noisy):
        point = shapely.Point(coordinate)
        ids, distances, radius = core.candidates(point, roads, 16)
        lines = geoms[ids]
        projected = shapely.line_locate_point(lines, point)
        lower = shapely.line_interpolate_point(lines, np.maximum(projected - 5, 0))
        upper = shapely.line_interpolate_point(lines, np.minimum(projected + 5, shapely.length(lines)))
        tangent = np.arctan2(shapely.get_y(upper)-shapely.get_y(lower),
                             shapely.get_x(upper)-shapely.get_x(lower))
        mismatch = abs(np.sin(tangent - bearings[i]))
        reliable = i > 0 and np.linalg.norm(displacements[i]) >= 10 and 0 < intervals[i] <= 60
        if not reliable:
            mismatch[:] = 0
        points.append(dict(point_id=i, target=int(target[i]), target_class=classes[target[i]],
                           present=target[i] in ids[:8], present16=target[i] in ids,
                           candidate_count=min(len(ids), 8), radius_m=radius,
                           clean_distance_m=float(shapely.distance(geoms[target[i]], shapely.Point(xy[i]))),
                           realized_error_m=float(np.linalg.norm(noisy[i]-xy[i])),
                           heading_available=reliable))
        for rank, (cid, distance) in enumerate(zip(ids, distances)):
            rows.append(dict(point_id=i, candidate_id=int(cid), distance_m=distance,
                             road_heading=headings[cid], rank=rank, road_class=classes[cid],
                             heading_mismatch=mismatch[rank], log_length=np.log1p(geoms[cid].length),
                             candidate_count=min(len(ids),8), distance_gap_m=distance-distances[0],
                             is_target=int(cid==target[i])))
    return pd.DataFrame(rows), pd.DataFrame(points)


def summarize(predictions, out):
    metrics = []; per_user = []; per_trip = []; classes = []; confusion = []
    for key, group in predictions.groupby(['seed', 'degradation', 'noise_m']):
        base = dict(zip(['seed', 'degradation', 'noise_m'], key))
        for method in METHODS:
            metrics.append(dict(**base, method=method, **core.metrics(group, method)))
            for user, subset in group.groupby('user_id'):
                per_user.append(dict(**base, method=method, user_id=user, **core.metrics(subset, method)))
            for trip, subset in group.groupby('trip_id'):
                per_trip.append(dict(**base, method=method, trip_id=trip, **core.metrics(subset, method)))
            report = classification_report(group.target_class, group[method+'_class'], output_dict=True, zero_division=0)
            for label, values in report.items():
                if isinstance(values, dict):
                    classes.append(dict(**base, method=method, road_class=label, **values))
            for (truth, prediction), count in group.groupby(['target_class', method+'_class']).size().items():
                confusion.append(dict(**base, method=method, true_class=truth, predicted_class=prediction, count=count))
    pd.DataFrame(metrics).to_csv(out/'metrics.csv', index=False)
    pd.DataFrame(per_trip).to_csv(out/'per_trajectory.csv', index=False)
    users = pd.DataFrame(per_user); users.to_csv(out/'per_user.csv', index=False)
    pd.DataFrame(classes).to_csv(out/'class_metrics.csv', index=False)
    pd.DataFrame(confusion).to_csv(out/'confusion.csv', index=False)
    uncertainty = []
    for key, group in users.groupby(['degradation', 'noise_m']):
        pivot = group.pivot_table(index='user_id', columns='method', values='segment_e2e')
        for method in METHODS[1:]:
            diff = (pivot[method]-pivot.nearest).to_numpy()
            bootstrap = np.random.default_rng(42).choice(diff, (10000,len(diff)), replace=True).mean(axis=1)
            uncertainty.append(dict(degradation=key[0], noise_m=key[1], method=method,
                                    users=len(diff), mean_difference=diff.mean(), median_difference=np.median(diff),
                                    ci_low=np.quantile(bootstrap,.025), ci_high=np.quantile(bootstrap,.975)))
    pd.DataFrame(uncertainty).to_csv(out/'uncertainty.csv', index=False)
    predictions.groupby(['seed','degradation','noise_m']).agg(
        recall8=('present','mean'), recall16=('present16','mean'), mean_count=('candidate_count','mean'),
        mean_radius_m=('radius_m','mean'), max_radius_m=('radius_m','max'),
        mean_realized_error_m=('realized_error_m','mean'), heading_availability=('heading_available','mean')
    ).to_csv(out/'candidate_diagnostics.csv')


def run(args):
    started = time.perf_counter()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError('Use a new directory to preserve experiment outputs')
    out.mkdir(parents=True)
    cohort = ROOT/'data/processed/vehicle_cohort'
    summary = pd.read_csv(cohort/'trip_summary.csv', dtype={'source_user_id':str})
    train, test = user_split(summary)
    roads, mapping = canonical_roads(core.legacy.load_roads())
    if not roads.geom_type.eq('LineString').all():
        raise ValueError('Local-tangent features require LineString roads')
    roads['endpoint_heading'] = [core.legacy.line_heading(g) for g in roads.geometry]
    mapping.to_csv(out/'road_identity_mapping.csv', index=False)
    roads[['identity','road_class']].to_csv(out/'road_catalog.csv', index_label='canonical_id')
    print('Train users',sorted(summary.loc[summary.trip_id.isin(train),'source_user_id'].unique()),flush=True)
    print('Test users',sorted(summary.loc[summary.trip_id.isin(test),'source_user_id'].unique()),flush=True)
    train_rows = []; batches = []; runtime = []; quality = []
    for ti, item in enumerate(summary.itertuples(),1):
        frame, times, xy = core.read_trip(cohort/(item.trip_id+'.csv'))
        keep = np.linspace(0,len(frame)-1,min(args.max_points,len(frame)),dtype=int)
        xy = xy[keep]; seconds = times.to_numpy(dtype='datetime64[ns]').astype(np.int64)[keep]/1e9
        target = np.array([core.nearest(shapely.Point(p), roads) for p in xy])
        ds = shapely.distance(roads.geometry.to_numpy()[target], shapely.points(xy))
        quality.append(dict(trip_id=item.trip_id,user_id=item.source_user_id,mode=item.mode,points=len(frame),
                            sampled_points=len(xy),sampled_median_gap_s=np.median(np.diff(seconds)),
                            sampled_max_gap_s=np.max(np.diff(seconds)),median_distance_m=np.median(ds),
                            p95_distance_m=np.quantile(ds,.95),max_distance_m=max(ds),
                            duplicate_coordinates=int(frame.duplicated(['latitude','longitude']).sum())))
        print(item.trip_id,item.source_user_id,flush=True)
        # Three mixed patterns at seed 42 vs three Gaussian seeds: equal point-condition budget.
        jobs = ([(k,s) for k in PATTERNS for s in args.seeds] if item.trip_id in test else
                [(k,42) for k in PATTERNS]+[('gaussian',43),('gaussian',44)])
        for kind, seed in jobs:
            for level in args.levels:
                tick = time.perf_counter()
                candidates, points = make_batch(xy,seconds,target,roads,kind,level,seed,ti)
                shared = dict(trip_id=item.trip_id,user_id=item.source_user_id,seed=seed,degradation=kind,noise_m=level)
                for name,value in shared.items():
                    points[name] = value; candidates[name] = value
                elapsed = time.perf_counter()-tick
                if item.trip_id in train:
                    valid = candidates['rank'].lt(8) & candidates.point_id.isin(points.loc[points.present,'point_id'])
                    train_rows.append(candidates[valid])
                else:
                    batches.append((candidates,points))
                    runtime.append(dict(**shared,method='shared_candidates_and_features',seconds=elapsed,points=len(points)))
    training = pd.concat(train_rows,ignore_index=True)
    training.to_csv(out/'training_candidates.csv.gz',index=False)
    models = {}; fit_times = {}; model_specs = {}
    for method,contextual,subset in [
        ('rf_original',False,training[training.seed==42]),
        ('rf_context',True,training[training.seed==42]),
        ('rf_gaussian',True,training[training.degradation=='gaussian'])]:
        x = feature_frame(subset,contextual)
        model = RandomForestClassifier(n_estimators=150,max_depth=14,class_weight='balanced',random_state=42,n_jobs=2)
        tick = time.perf_counter(); model.fit(x,subset.is_target); fit_times[method]=time.perf_counter()-tick
        models[method] = (model,list(x),contextual)
        model_specs[method] = dict(parameters=model.get_params(),candidate_rows=len(subset),feature_columns=list(x))
    joblib.dump(models,out/'models.joblib')
    outputs = []; scored = []
    for candidates, points in batches:
        selected = candidates[candidates['rank']<8].copy()
        tick=time.perf_counter()
        nearest = selected.loc[selected.groupby('point_id').distance_m.idxmin()].set_index('point_id')
        points['nearest'] = points.point_id.map(nearest.candidate_id)
        points['nearest_class'] = points.point_id.map(nearest.road_class)
        shared=points.iloc[0][['trip_id','user_id','seed','degradation','noise_m']].to_dict()
        runtime.append(dict(**shared,method='nearest',seconds=time.perf_counter()-tick,points=len(points)))
        tick=time.perf_counter(); selected['cost']=selected.distance_m+10*selected.heading_mismatch
        chosen=selected.loc[selected.groupby('point_id').cost.idxmin()].set_index('point_id')
        points['heading']=points.point_id.map(chosen.candidate_id); points['heading_class']=points.point_id.map(chosen.road_class)
        runtime.append(dict(**shared,method='heading',seconds=time.perf_counter()-tick,points=len(points)))
        for method,(model,columns,contextual) in models.items():
            tick=time.perf_counter()
            selected[method+'_score']=model.predict_proba(feature_frame(selected,contextual).reindex(columns=columns,fill_value=0))[:,list(model.classes_).index(1)]
            chosen=selected.loc[selected.groupby('point_id')[method+'_score'].idxmax()].set_index('point_id')
            points[method]=points.point_id.map(chosen.candidate_id); points[method+'_class']=points.point_id.map(chosen.road_class)
            runtime.append(dict(**shared,method=method,seconds=time.perf_counter()-tick,points=len(points)))
        outputs.append(points)
        scored.append(selected[['trip_id','user_id','seed','degradation','noise_m','point_id','candidate_id','is_target']+[m+'_score' for m in models]])
    predictions=pd.concat(outputs,ignore_index=True); predictions.to_csv(out/'predictions.csv',index=False)
    scored=pd.concat(scored,ignore_index=True); scored.to_csv(out/'candidate_scores.csv.gz',index=False)
    summarize(predictions,out)
    pd.DataFrame(runtime).to_csv(out/'runtime.csv',index=False)
    pd.DataFrame(quality).to_csv(out/'data_quality.csv',index=False)
    roc=[]; auc=[]
    for method in models:
        fpr,tpr,thresholds=roc_curve(scored.is_target,scored[method+'_score'])
        roc.extend(dict(method=method,fpr=a,tpr=b,threshold=c) for a,b,c in zip(fpr,tpr,thresholds))
        auc.append(dict(method=method,candidate_auc=roc_auc_score(scored.is_target,scored[method+'_score'])))
    pd.DataFrame(roc).to_csv(out/'candidate_roc.csv',index=False); pd.DataFrame(auc).to_csv(out/'candidate_auc.csv',index=False)
    files=[cohort/'trip_summary.csv',cohort/'selection.json',ROOT/'data/processed/osm_roads.geojson',ROOT/'configs/extension_protocol.json',*sorted(cohort.glob('trip_[0-9][0-9][0-9].csv'))]
    meta=dict(config=vars(args),protocol=json.loads((ROOT/'configs/extension_protocol.json').read_text()),created_utc=datetime.now(timezone.utc).isoformat(),
              train_trips=sorted(train),test_trips=sorted(test),train_users=sorted(summary.loc[summary.trip_id.isin(train),'source_user_id'].unique()),
              test_users=sorted(summary.loc[summary.trip_id.isin(test),'source_user_id'].unique()),roads=len(roads),fit_seconds=fit_times,
              total_seconds=time.perf_counter()-started,model_specs=model_specs,
              versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,geopandas=gpd.__version__,shapely=shapely.__version__,sklearn=sklearn.__version__),
              hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              label='Nearest clean-GPS normalized geometry and class; undirected proxy, no verified road truth',
              timing='perf_counter batch wall time; shared search/features separately; fit separately; excludes IO from inference',
              training='Mixed: seed42 all patterns; Gaussian-only: seeds42,43,44 Gaussian. Equal point-condition budget; retained candidate counts may differ.')
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)); print('Completed',meta['total_seconds'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='results/extension')
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45,46])
    p.add_argument('--levels',nargs='+',type=int,default=[5,10,20])
    p.add_argument('--max-points',type=int,default=300)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.smoke: args.seeds=[42]; args.levels=[10]; args.max_points=20
    if args.max_points<2 or any(s<0 for s in args.seeds) or any(x<0 for x in args.levels): p.error('Invalid configuration')
    run(args)
