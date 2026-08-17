import numpy as np
from src.metrics import evaluate_binary, find_f2_threshold

def test_metrics_keys():
    y=np.array([0,0,1,1]); p=np.array([.1,.2,.7,.9])
    m=evaluate_binary(y,p,.5)
    assert 0 <= m["ROC_AUC"] <= 1
    assert 0 <= m["PR_AUC"] <= 1

def test_threshold_is_valid():
    t,f2=find_f2_threshold([0,0,1,1],[.1,.2,.7,.9])
    assert .05 <= t <= .95
    assert f2 >= 0
