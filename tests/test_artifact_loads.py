import joblib
from src.data_prep import load_data
from src.train import num_cols, cate_cols

def test_artifact_loading():
    pipeline = joblib.load("artifacts/model.pkl")

    #Loading reference data
    df = load_data('data/raw/customer_churn.csv')
    X_sample = df[num_cols + cate_cols].head(5)

    preds = pipeline.predict(X_sample)
    assert len(preds) == 5
    assert set(preds).issubset({0, 1})
