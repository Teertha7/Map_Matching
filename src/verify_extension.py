"""Independent saved-output checks for the user-disjoint extension."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,f1_score
import extended_experiment as e
root=Path('results/extension'); meta=json.loads((root/'metadata.json').read_text())
assert not set(meta['train_users'])&set(meta['test_users'])
assert not set(meta['train_trips'])&set(meta['test_trips'])
d=pd.read_csv(root/'predictions.csv',dtype={'user_id':str}); m=pd.read_csv(root/'metrics.csv')
assert set(d.user_id)==set(meta['test_users'])
assert not d.duplicated(['trip_id','point_id','seed','degradation','noise_m']).any()
q=pd.read_csv(root/'data_quality.csv'); expected=int(q[q.trip_id.isin(meta['test_trips'])].sampled_points.sum())*len(meta['config']['seeds'])*3*len(meta['config']['levels'])
assert len(d)==expected
for row in m.itertuples():
    g=d[(d.seed==row.seed)&(d.degradation==row.degradation)&(d.noise_m==row.noise_m)]
    hit=g[row.method].to_numpy()==g.target.to_numpy()
    assert np.isclose(hit.mean(),row.segment_e2e)
    assert np.isclose(hit[g.present].mean(),row.segment_conditional)
    assert np.isclose(g.present.mean(),row.candidate_recall)
    assert not hit[~g.present].any()
    assert np.isclose(accuracy_score(g.target_class,g[row.method+'_class']),row.class_accuracy)
    assert np.isclose(f1_score(g.target_class,g[row.method+'_class'],average='macro',zero_division=0),row.class_macro_f1)
for path,digest in meta['hashes'].items():
    assert hashlib.sha256((e.ROOT/path).read_bytes()).hexdigest()==digest
summary=pd.read_csv(e.ROOT/'data/processed/vehicle_cohort/trip_summary.csv',dtype={'source_user_id':str})
for row in summary.itertuples():
    f,t,xy=e.core.read_trip(e.ROOT/'data/processed/vehicle_cohort'/(row.trip_id+'.csv'))
    assert t.between(pd.Timestamp(row.label_start),pd.Timestamp(row.label_end)).all()
    assert row.mode in ['car','taxi'] and t.diff().dt.total_seconds().max()<=60
    assert hashlib.sha256((e.ROOT/row.source_file).read_bytes()).hexdigest()==row.source_sha256
    label=e.ROOT/'data/raw/Geolife Trajectories 1.3/Data'/row.source_user_id/'labels.txt'
    assert hashlib.sha256(label.read_bytes()).hexdigest()==row.labels_sha256
scores=pd.read_csv(root/'candidate_scores.csv.gz')
for method in ['rf_original','rf_context','rf_gaussian']:
    assert scores[method+'_score'].between(0,1).all()
    winners=scores.loc[scores.groupby(['trip_id','point_id','seed','degradation','noise_m'])[method+'_score'].idxmax()]
    check=d.merge(winners[['trip_id','point_id','seed','degradation','noise_m','candidate_id']],on=['trip_id','point_id','seed','degradation','noise_m'],validate='one_to_one')
    assert (check[method]==check.candidate_id).all()
train=pd.read_csv(root/'training_candidates.csv.gz',dtype={'user_id':str})
assert not set(train.user_id)&set(d.user_id)
assert set(train.user_id)==set(meta['train_users'])
shared=float(d.target.isin(set(train.loc[train.is_target==1,'candidate_id'])).mean())
report=dict(status='PASS',test_records=len(d),test_users=len(meta['test_users']),train_users=len(meta['train_users']),tests='All metric rows independently recomputed; candidate winners verified; disjoint users; input and raw-source hashes; labels interval; timestamps; omission failures',test_reference_overlap_with_training=shared)
(root/'verification.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
