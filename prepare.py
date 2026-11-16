import argparse
from pathlib import Path
import joblib
import pandas as pd
from features import load_checked, NUM, CAT

p=argparse.ArgumentParser(); p.add_argument('file'); p.add_argument('--model',default='results_pr3/model_v1.joblib'); p.add_argument('--out',default='prepared'); a=p.parse_args()
out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
df,bad=load_checked(a.file); bad.to_csv(out/'quarantine.csv',index=False)
if df.empty: raise ValueError('No valid records')
model=joblib.load(a.model); X=df[NUM+CAT]; matrix=model[:-1].transform(X)
pd.DataFrame(matrix,columns=model['prep'].get_feature_names_out()).to_csv(out/'features.csv',index=False)
prob=model.predict_proba(X)[:,1]; pd.DataFrame({'id':df.id,'probability':prob,'prediction':(prob>=.5).astype(int)}).to_csv(out/'predictions.csv',index=False)
