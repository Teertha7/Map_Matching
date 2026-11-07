"""Fixed-model grouped training-only diagnostic; never selects using held-out test data."""
import time,json
from pathlib import Path
import numpy as np,pandas as pd
from shapely.geometry import Point
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
import research_experiment as r
root=Path('results/final'); meta=json.loads((root/'metadata.json').read_text()); roads=r.legacy.load_roads(); frames=[]; headings=np.array([r.legacy.line_heading(g) for g in roads.geometry]); classes=roads.road_class.to_numpy()
for ti,path in enumerate(sorted((r.ROOT/'data/processed/trips').glob('trip_[0-9][0-9][0-9].csv')),1):
    if path.stem not in meta['train_ids']: continue
    _,_,xy=r.read_trip(path); xy=xy[np.linspace(0,len(xy)-1,min(meta['config']['max_points'],len(xy)),dtype=int)]; targets=[r.nearest(Point(p),roads) for p in xy]
    for di,kind in enumerate(r.legacy.DEGRADATIONS):
        for level in meta['config']['levels']:
            noisy=r.legacy.perturb(xy,kind,level,meta['config']['seeds'][0]+10000*ti+100*di+level)
            for pi,p in enumerate(noisy):
                ids,ds,_=r.candidates(Point(p),roads,8)
                if targets[pi] not in ids: continue
                for rank,(cid,dist) in enumerate(zip(ids,ds)):
                    frames.append(dict(trip_id=path.stem,set_id=f'{path.stem}_{kind}_{level}_{pi}',candidate_id=cid,distance_m=dist,road_heading=headings[cid],rank=rank,road_class=classes[cid],is_target=int(cid==targets[pi])))
d=pd.DataFrame(frames); results=[]
for fold,(a,b) in enumerate(GroupKFold(n_splits=3).split(d,groups=d.trip_id),1):
    train=d.iloc[a]; val=d.iloc[b].copy(); assert not set(train.trip_id)&set(val.trip_id)
    x=r.legacy.feature_frame(train); model=RandomForestClassifier(n_estimators=150,max_depth=14,class_weight='balanced',random_state=42,n_jobs=2).fit(x,train.is_target)
    val['score']=model.predict_proba(r.legacy.feature_frame(val).reindex(columns=x.columns,fill_value=0))[:,1]
    rf=val.loc[val.groupby('set_id').score.idxmax()].is_target.mean(); near=val.loc[val.groupby('set_id')['rank'].idxmin()].is_target.mean()
    results.append(dict(fold=fold,train_ids=' '.join(sorted(train.trip_id.unique())),validation_ids=' '.join(sorted(val.trip_id.unique())),rf_conditional=rf,nearest_conditional=near,sets=val.set_id.nunique()))
    print(results[-1],flush=True)
pd.DataFrame(results).to_csv(root/'training_cv.csv',index=False)
