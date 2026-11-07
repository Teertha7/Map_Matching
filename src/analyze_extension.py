"""Rebuild extension tables, plots and evidence summaries solely from saved results."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

NAMES={'nearest':'Nearest road','heading':'Local heading','rf_original':'RF original','rf_context':'RF contextual','rf_gaussian':'RF Gaussian only'}
COLORS={'nearest':'#3A6EA5','heading':'#8C6BB1','rf_original':'#D98C36','rf_context':'#27856A','rf_gaussian':'#B94E48'}

def build(path):
    root=Path(path); figs=root/'figures'; figs.mkdir(exist_ok=True)
    m=pd.read_csv(root/'metrics.csv'); d=pd.read_csv(root/'predictions.csv',dtype={'user_id':str}); u=pd.read_csv(root/'per_user.csv',dtype={'user_id':str})
    plt.rcParams.update({'font.size':10,'figure.dpi':150,'savefig.dpi':240})
    captions=[]
    def save(name,title,caption):
        plt.title(title); plt.tight_layout(); plt.savefig(figs/(name+'.png')); plt.savefig(figs/(name+'.pdf')); plt.close()
        captions.append(dict(file=name+'.png',title=title,caption=caption))
    for kind in ['gaussian','persistent_bias','correlated_drift']:
        plt.figure(figsize=(7.3,4.5))
        for method in ['nearest','rf_original','rf_context','rf_gaussian']:
            g=m[(m.method==method)&(m.degradation==kind)].groupby('noise_m').segment_e2e.agg(['mean','std'])
            plt.errorbar(g.index,g['mean'],yerr=g['std'],label=NAMES[method],color=COLORS[method],marker='o',capsize=3)
        plt.xlabel('Error parameter (m)'); plt.ylabel('End-to-end canonical-road proxy agreement'); plt.ylim(0,1); plt.xticks([5,10,20]); plt.legend(fontsize=8); plt.grid(alpha=.2)
        save('segment_'+kind,kind.replace('_',' ').title()+' on held-out users',
             'Each point is the mean across five corruption seeds; bars show sample SD across seeds, not confidence intervals. Gaussian and drift levels are per-axis SD; bias levels are radial magnitude. All methods use identical observations and proxy labels.')
    plt.figure(figsize=(7.3,4.5))
    for i,(kind,g) in enumerate(d.groupby('degradation')):
        a=g.groupby('noise_m')[['present','present16']].mean(); color=plt.get_cmap('tab10')(i)
        plt.plot(a.index,a.present,marker='o',color=color,label=kind.replace('_',' ')+' K=8')
        plt.plot(a.index,a.present16,linestyle='--',color=color,label=kind.replace('_',' ')+' K=16')
    plt.ylabel('Candidate recall'); plt.xlabel('Error parameter (m)'); plt.xticks([5,10,20]); plt.ylim(max(0,float(d.groupby(['degradation','noise_m']).present.mean().min())-.01),1.002); plt.legend(fontsize=8)
    save('candidate_recall','Candidate retrieval under degradation','Solid lines use eight candidates and dashed lines sixteen. This evaluates retrieval only; sixteen-candidate model inference is not included in the extension. The expanded vertical scale exposes small omission rates.')
    plt.figure(figsize=(7.3,4.5)); a=m.groupby('method')[['segment_e2e','segment_conditional']].mean().rename(index=NAMES,columns={'segment_e2e':'End-to-end','segment_conditional':'Conditional'})
    a.plot.bar(ax=plt.gca(),rot=20); plt.ylim(0,1); plt.ylabel('Canonical-road proxy agreement'); plt.xlabel('')
    save('conditional_vs_e2e','Candidate omissions and ranking performance','Equal-weight mean over conditions and seeds. Conditional metrics omit retrieval failures; end-to-end metrics count them as failures. The gap isolates the effect of candidate omissions.')
    c=pd.read_csv(root/'confusion.csv'); c=c[c.method=='rf_context'].pivot_table(index='true_class',columns='predicted_class',values='count',aggfunc='sum',fill_value=0)
    labels=sorted(set(c.index)|set(c.columns)); c=c.reindex(index=labels,columns=labels,fill_value=0)
    plt.figure(figsize=(8,6)); plt.imshow(c.to_numpy()/np.maximum(c.sum(axis=1).to_numpy()[:,None],1),vmin=0,vmax=1,cmap='Blues'); plt.xticks(range(len(labels)),labels,rotation=65,ha='right'); plt.yticks(range(len(labels)),labels); plt.colorbar(label='Fraction of true-class observations'); plt.xlabel('Predicted class'); plt.ylabel('Proxy class')
    save('class_confusion','Contextual RF class confusion','Row-normalized pooled confusion. White rows with no true-class support must not be interpreted as zero recall. Exact class support, precision, recall and F1 are saved separately. OSM class is broader than road identity.')
    a=u.pivot_table(index='user_id',columns='method',values='segment_e2e')[['nearest','rf_original','rf_context','rf_gaussian']]; plt.figure(figsize=(8,4.5)); a.rename(columns=NAMES).plot.bar(ax=plt.gca(),rot=0); plt.ylim(0,1); plt.ylabel('Mean end-to-end proxy agreement'); plt.xlabel('Held-out user')
    save('per_user','User-level performance','Each user contributes three trips. Bars average condition and seed scores within users; all journeys belonging to these users are excluded from training.')
    a=pd.read_csv(root/'per_trajectory.csv').pivot_table(index='trip_id',columns='method',values='segment_e2e')[['nearest','rf_context']]; plt.figure(figsize=(8,4.5)); a.rename(columns=NAMES).plot.bar(ax=plt.gca(),rot=65); plt.ylim(0,1); plt.ylabel('Mean end-to-end proxy agreement'); plt.xlabel('Held-out journey')
    save('per_trajectory','Journey-level performance','Means across nine conditions and five seeds. The comparison exposes trip heterogeneity that pooled scores hide; uncertainty resampling uses users rather than these correlated journeys.')
    rt=pd.read_csv(root/'runtime.csv').groupby('method')[['seconds','points']].sum(); rt['ms_per_point']=1000*rt.seconds/rt.points; rt.to_csv(root/'runtime_summary.csv')
    plt.figure(figsize=(8,4.5)); rt.ms_per_point.rename(index=NAMES).plot.bar(ax=plt.gca(),rot=25); plt.yscale('log'); plt.ylabel('Amortized milliseconds per point'); plt.xlabel('')
    save('runtime','Batch inference cost','Shared candidate generation and contextual feature extraction are timed separately. Model bars include feature encoding and batched selection. These are wall times on an Apple M4 host with two RF workers, not streaming latency or a hardware-matched comparison to published systems.')
    roc=pd.read_csv(root/'candidate_roc.csv'); auc=pd.read_csv(root/'candidate_auc.csv').set_index('method').candidate_auc
    plt.figure(figsize=(6.5,4.5))
    for method,g in roc.groupby('method'): plt.plot(g.fpr,g.tpr,label=f'{NAMES[method]} (AUC {auc[method]:.3f})',color=COLORS[method])
    plt.plot([0,1],[0,1],'k--',alpha=.4); plt.xlabel('False-positive rate among candidate rows'); plt.ylabel('True-positive rate among candidate rows'); plt.legend(fontsize=8)
    save('candidate_roc','Candidate-level ROC diagnostics','ROC assesses the binary candidate classifier across all test candidate rows, including sets with no positive candidate. It does not measure segment selection, is not user-independent evidence, and is not used for threshold tuning.')
    pivot=m.pivot_table(index=['seed','degradation','noise_m'],columns='method',values='segment_e2e'); diff=100*(pivot.rf_context-pivot.rf_gaussian)
    x=diff.groupby(['degradation','noise_m']).mean().unstack(); plt.figure(figsize=(7,4.3)); x.T.plot.bar(ax=plt.gca(),rot=0); plt.axhline(0,color='black',linewidth=.6); plt.ylabel('Mixed minus Gaussian-only training (pp)'); plt.xlabel('Error parameter (m)')
    save('training_shift','Does mixed-error training transfer better?','Positive values favor mixed-error training; negative values favor Gaussian-only training. Models use the same contextual features and equal point-condition training budgets. Bias and drift are absent from Gaussian-only training; no model is selected using this result.')
    uncertainty=pd.read_csv(root/'uncertainty.csv'); g=uncertainty[uncertainty.method=='rf_context']; plt.figure(figsize=(8,4.5)); labels=[f'{r.degradation.replace("_"," ")} {r.noise_m} m' for r in g.itertuples()]
    plt.errorbar(100*g.mean_difference,np.arange(len(g)),xerr=[100*(g.mean_difference-g.ci_low),100*(g.ci_high-g.mean_difference)],fmt='o',capsize=3); plt.axvline(0,color='black',linewidth=.7); plt.yticks(np.arange(len(g)),labels); plt.xlabel('Contextual RF minus nearest (percentage points)')
    save('user_uncertainty','Paired user-cluster uncertainty','Percentile 95% intervals from 10,000 user resamples, after seed averaging within users. Only five test users are available. These are descriptive conditional intervals; no significance or population-generalization claim is made.')
    plt.figure(figsize=(9,3.4)); plt.axis('off')
    labels=['Raw GeoLife\n+ mode labels','Screened vehicle\njourneys','Split by user\n13 train / 5 test','Project and assign\ncanonical proxy','Simulate errors\nand retrieve','RF / geometric\ncomparisons','Saved metrics\nand uncertainty']
    for i,label in enumerate(labels):
        row=i//4; col=i%4; x=.02+col*.25; y=.75-row*.48
        plt.text(x+.105,y,label,ha='center',va='center',fontsize=9,bbox=dict(boxstyle='round,pad=.5',fc='#EAF0F5',ec='#5D7182'),transform=plt.gca().transAxes)
        if col<3 and i<6: plt.annotate('',xy=(x+.245,y),xytext=(x+.22,y),xycoords='axes fraction',arrowprops=dict(arrowstyle='->'))
    save('workflow','Reproducible evaluation workflow','Training fits use only training-user candidates. Clean reference roads are used for supervised labels and evaluation, never inserted into candidate lists. All methods share held-out observations.')
    # Pool classes separately from averaging macro-F1 across conditions.
    from sklearn.metrics import classification_report
    rows=[]
    for method in NAMES:
        rep=classification_report(d.target_class,d[method+'_class'],output_dict=True,zero_division=0)
        for label,v in rep.items():
            if isinstance(v,dict): rows.append(dict(method=method,road_class=label,**v))
    pd.DataFrame(rows).to_csv(root/'pooled_class_metrics.csv',index=False)
    overall=m.groupby(['degradation','noise_m','method']).agg(e2e=('segment_e2e','mean'),seed_sd=('segment_e2e','std'),conditional=('segment_conditional','mean'),candidate_recall=('candidate_recall','mean'),class_accuracy=('class_accuracy','mean'),macro_f1=('class_macro_f1','mean')).reset_index()
    overall.to_csv(root/'overall_metrics.csv',index=False)
    importance=[]
    for method,(model,columns,_) in joblib.load(root/'models.joblib').items():
        importance.extend(dict(method=method,feature=f,impurity_importance=v) for f,v in zip(columns,model.feature_importances_))
    pd.DataFrame(importance).to_csv(root/'feature_importance.csv',index=False)
    (figs/'captions.json').write_text(json.dumps(captions,indent=2))
    print('Built',len(captions),'figures')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--input',default='results/extension'); build(p.parse_args().input)
