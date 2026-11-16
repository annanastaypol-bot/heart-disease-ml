"""Reconstruct the base part with the split protocol in PR2."""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
p=argparse.ArgumentParser(); p.add_argument('source'); p.add_argument('--out',default='base.csv'); a=p.parse_args()
d=pd.read_excel(a.source)
assert d.shape==(138,13) and d.cardio.value_counts().to_dict()=={0:66,1:72}
base,remaining=train_test_split(d,train_size=.6,random_state=42,stratify=d.cardio)
assert base.cardio.value_counts().to_dict()=={0:39,1:43}
base.to_csv(a.out,index=False)
print('Base restored:',len(base),'Remaining:',len(remaining))
