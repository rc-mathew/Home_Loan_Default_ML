from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.validation import temporal_split

p=argparse.ArgumentParser(description='Run a true chronological split when an application/decision date exists.')
p.add_argument('--application-file',default='data/application_train.csv'); p.add_argument('--date-column',required=True); p.add_argument('--output',default='artifacts/temporal_split_report.json'); a=p.parse_args()
df=pd.read_csv(a.application_file,usecols=['SK_ID_CURR','TARGET',a.date_column])
tr,va,te=temporal_split(df,a.date_column)
report={'date_column':a.date_column,'train_rows':len(tr),'validation_rows':len(va),'test_rows':len(te),'train_end':str(pd.to_datetime(df.loc[tr,a.date_column]).max()),'validation_end':str(pd.to_datetime(df.loc[va,a.date_column]).max()),'test_start':str(pd.to_datetime(df.loc[te,a.date_column]).min()),'note':'This is a chronological validation design; do not substitute SK_ID_CURR or relative feature dates for a true application timestamp.'}
Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
