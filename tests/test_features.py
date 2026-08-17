import pandas as pd
from src.features import make_model_matrix, SELECTED_FEATURES

def test_feature_matrix_shape_and_dtype():
    df=pd.DataFrame({c:[1.0] for c in SELECTED_FEATURES})
    X=make_model_matrix(df)
    assert list(X.columns)==SELECTED_FEATURES
    assert str(X.dtypes.iloc[0])=='float32'
