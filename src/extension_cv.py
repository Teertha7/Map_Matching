"""Three-fold user-grouped diagnostics, fixed settings, using saved training rows only."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from extended_experiment import feature_frame
root=Path('results/extension'); d=pd.read_csv(root/'training_candidates.csv.gz',dtype={'user_id':str}); d=d[d.seed==42].reset_index(drop=True)
keys=['trip_id','point_id','degradation','noise_m']; results=[]
for fold,(a,b) in enumerate(GroupKFold(3).split(d,groups=d.user_id),1):
    train=d.iloc[a]; val=d.iloc[b].copy(); assert not set(train.user_id)&set(val.user_id)
    near=val.loc[val.groupby(keys).distance_m.idxmin()].is_target.mean()
    for method,context in [('rf_original',False),('rf_context',True)]:
        x=feature_frame(train,context); model=RandomForestClassifier(n_estimators=150,max_depth=14,class_weight='balanced',random_state=42,n_jobs=2).fit(x,train.is_target)
        val['score']=model.predict_proba(feature_frame(val,context).reindex(columns=x.columns,fill_value=0))[:,1]
        correct=val.loc[val.groupby(keys).score.idxmax()].is_target.mean()
        results.append(dict(fold=fold,method=method,conditional_accuracy=correct,nearest_conditional=near,train_users=' '.join(sorted(train.user_id.unique())),validation_users=' '.join(sorted(val.user_id.unique()))))
        print(results[-1],flush=True)
pd.DataFrame(results).to_csv(root/'training_cv.csv',index=False)
