EXPECTED_DTYPES = {
    'CreditScore': int,
    'Gender': 'category',
    'Age': int,
    'Tenure': int,
    'Balance': float,
    'NumOfProducts': int,
    'HasCrCard': int,
    'IsActiveMember': int,
    'EstimatedSalary': float,
    'Complain': int,
    'Satisfaction Score': int,
    'Point Earned': int,
    'Exited' : int
}

target_col = "Exited"

def validate_schema(df):
    for col, dtype in EXPECTED_DTYPES.items():
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the dataset.")
        if df[col].dtype != dtype:
            raise TypeError(f"Column '{col}' has dtype {df[col].dtype} but expected {dtype}.")
