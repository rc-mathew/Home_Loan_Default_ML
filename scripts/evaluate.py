from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

def summarize(model_dir='models'):
    p=Path(model_dir)
    val=pd.read_csv(p/'validation_metrics.csv') if (p/'validation_metrics.csv').exists() else pd.DataFrame()
    test=pd.read_csv(p/'holdout_test_metrics.csv') if (p/'holdout_test_metrics.csv').exists() else pd.DataFrame()
    print('Validation metrics:'); print(val.to_string(index=False) if not val.empty else 'not found')
    print('\nUntouched holdout test:'); print(test.to_string(index=False) if not test.empty else 'not found')
if __name__=='__main__': summarize()
