import argparse, hashlib, json, platform, time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from features import load_checked, make_pipeline, NUM, CAT

SEED=42
SCORING={'roc_auc':'roc_auc','ap':'average_precision','f1':'f1','recall':'recall','precision':'precision'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',required=True,help='Exact base.csv from PR2 (82 rows)')
    parser.add_argument('--out',default='results_pr3')
    args=parser.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    base=Path(args.base); df,bad=load_checked(base,training=True)
    if len(df)!=82 or len(bad) or df.cardio.value_counts().to_dict()!={0:39,1:43}:
        raise ValueError('Expected PR2 base: 82 valid rows, cardio 0=39, 1=43. Do not use full dataset.')
    train,test=train_test_split(df,test_size=.2,random_state=SEED,stratify=df.cardio)
    train.to_csv(out/'train.csv',index=False); test.to_csv(out/'test.csv',index=False)
    X=train[NUM+CAT]; y=train.cardio.astype(int)
    cv=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(X,y))
    records=[]; candidates={}; specs={}
    def evaluate(name,model,extended=True,scale=True,eligible=True):
        p=make_pipeline(model,extended,scale)
        v=cross_validate(p,X,y,cv=cv,scoring=SCORING,return_train_score=True,error_score='raise')
        rec={'model':name,'extended':extended,'params':json.dumps(model.get_params(),default=str,ensure_ascii=False),'seconds':float(v['fit_time'].mean()),'train_auc':float(v['train_roc_auc'].mean())}
        for k in SCORING:
            rec[k]=float(v['test_'+k].mean()); rec[k+'_std']=float(v['test_'+k].std()); rec[k+'_folds']=json.dumps(v['test_'+k].tolist())
        records.append(rec)
        if eligible: candidates[name]=p; specs[name]=rec
        return rec
    evaluate('Dummy',DummyClassifier(strategy='most_frequent'),eligible=False)
    reg=[]
    for penalty in ['l1','l2']:
        for c in [.001,.01,.1,1,10,100]:
            name=f'LogReg {penalty.upper()} C={c:g}'
            model=LogisticRegression(penalty=penalty,C=c,solver='liblinear',max_iter=3000,random_state=SEED)
            r=evaluate(name,model)
            p=make_pipeline(clone(model)).fit(X,y)
            reg.append({'penalty':penalty,'C':c,'auc':r['roc_auc'],'nonzero':int(np.count_nonzero(np.abs(p['model'].coef_)>1e-8))})
    evaluate('KNN k=5',KNeighborsClassifier(n_neighbors=5))
    depth=[]
    for d in [1,2,3,4,5,8,None]:
        r=evaluate(f'Tree depth={d}',DecisionTreeClassifier(max_depth=d,random_state=SEED),scale=False)
        depth.append({'depth':str(d),'train':r['train_auc'],'cv':r['roc_auc']})
    evaluate('Random forest',RandomForestClassifier(n_estimators=200,max_depth=4,min_samples_leaf=3,random_state=SEED,n_jobs=1),scale=False)
    evaluate('Gradient boosting',GradientBoostingClassifier(n_estimators=100,max_depth=2,random_state=SEED),scale=False)
    # Controlled feature ablation: same algorithms and folds.
    evaluate('LogReg L2 raw',LogisticRegression(C=1,solver='liblinear',max_iter=3000,random_state=SEED),extended=False)
    evaluate('Gradient boosting raw',GradientBoostingClassifier(n_estimators=100,max_depth=2,random_state=SEED),extended=False,scale=False)
    # Select two distinct top families for grid search.
    family=lambda n: n.split(' C=')[0] if n.startswith('LogReg') else ('Tree' if n.startswith('Tree') else n.replace(' tuned','').split(' raw')[0].split(' k=')[0])
    # group L1/L2 and raw together
    family=lambda n: 'LogReg' if n.startswith('LogReg') else ('Tree' if n.startswith('Tree') else n.replace(' tuned','').split(' raw')[0].split(' k=')[0])
    grids={'LogReg':{'model__C':[.01,.1,1,10,100],'model__penalty':['l1','l2']},'Tree':{'model__max_depth':[1,2,3,5,None],'model__min_samples_leaf':[1,3,5]},'KNN':{'model__n_neighbors':[3,5,7,11],'model__weights':['uniform','distance']},'Random forest':{'model__max_depth':[2,4,None],'model__min_samples_leaf':[1,3,5]},'Gradient boosting':{'model__n_estimators':[50,100,150],'model__learning_rate':[.03,.1],'model__max_depth':[1,2]}}
    tuned=[]; seen=set()
    for name in sorted(candidates,key=lambda n:specs[n]['roc_auc'],reverse=True):
        fam=family(name)
        if fam in seen: continue
        seen.add(fam); p=candidates[name]
        gs=GridSearchCV(p,grids[fam],scoring='roc_auc',cv=cv,n_jobs=1,error_score='raise').fit(X,y)
        pd.DataFrame(gs.cv_results_).to_csv(out/f'grid_{fam.replace(" ","_")}.csv',index=False)
        n=name+' tuned'; evaluate(n,clone(gs.best_estimator_['model']),extended=specs[name]['extended'],scale='scale' in p['prep'].transformers[0][1].named_steps)
        tuned.append({'family':fam,'grid':grids[fam],'best':gs.best_params_,'auc':gs.best_score_})
        if len(seen)==2: break
    # Prefer simple model if within 0.005 AUC; then recall, then name.
    best_auc=max(specs[n]['roc_auc'] for n in candidates)
    complexity={'LogReg':0,'Tree':1,'KNN':2,'Random forest':3,'Gradient boosting':4}
    winner=min((n for n in candidates if best_auc-specs[n]['roc_auc']<.005),key=lambda n:(complexity[family(n)],-specs[n]['recall'],-specs[n]['roc_auc'],n))
    final=clone(candidates[winner]).fit(X,y)
    matrix1=final[:-1].transform(X); matrix2=final[:-1].transform(X)
    assert np.array_equal(matrix1,matrix2)
    joblib.dump(final,out/'model_v1.joblib'); assert np.allclose(final.predict_proba(X),joblib.load(out/'model_v1.joblib').predict_proba(X))
    pd.DataFrame(records).to_csv(out/'experiments.csv',index=False)
    pd.DataFrame(reg).to_csv(out/'regularization.csv',index=False); pd.DataFrame(depth).to_csv(out/'tree_depth.csv',index=False)
    lp=make_pipeline(LogisticRegression(C=1,solver='liblinear',max_iter=3000,random_state=SEED)).fit(X,y)
    coefs=pd.DataFrame({'feature':lp['prep'].get_feature_names_out(),'coef':lp['model'].coef_[0]}).sort_values('coef',key=abs,ascending=False)
    coefs.to_csv(out/'coefficients.csv',index=False)
    def plot(filename,draw,xlabel,ylabel):
        plt.figure(figsize=(8,4.5)); draw(); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(out/filename,dpi=160); plt.close()
    def regplot(key):
        for pen in ['l1','l2']:
            a=[r for r in reg if r['penalty']==pen]; plt.semilogx([r['C'] for r in a],[r[key] for r in a],marker='o',label=pen.upper())
        plt.legend()
    plot('regularization_auc.png',lambda:regplot('auc'),'C','ROC-AUC (CV)')
    plot('regularization_nonzero.png',lambda:regplot('nonzero'),'C','Ненулевые коэффициенты')
    def treeplot():
        for k,l in [('train','Train'),('cv','CV')]: plt.plot([r['depth'] for r in depth],[r[k] for r in depth],marker='o',label=l)
        plt.legend()
    plot('tree_depth.png',treeplot,'Глубина дерева','ROC-AUC')
    top=pd.DataFrame(records).sort_values('roc_auc').tail(10)
    plot('comparison.png',lambda:plt.barh(top.model,top.roc_auc,xerr=top.roc_auc_std),'ROC-AUC (CV)','Модель')
    info={'version':'v1','date':datetime.now(timezone.utc).isoformat(),'source_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),'code_sha256':{f:hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest() for f in ['train.py','features.py']},'train_rows':len(train),'test_rows':len(test),'seed':SEED,'python':platform.python_version(),'sklearn':sklearn.__version__,'pandas':pd.__version__,'winner':winner,'metrics':specs[winner],'threshold':.5,'matrix_shape':list(matrix1.shape),'repeatability':True,'tuning':tuned,'test_not_evaluated':True}
    (out/'model_v1.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Winner:',winner,'CV ROC-AUC:',specs[winner]['roc_auc']); print('Saved:',out)

if __name__=='__main__': main()
