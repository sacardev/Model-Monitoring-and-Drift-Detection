import pandas as pd
from src.data_prep import preprocessor

def test_pipeline_handles_misssing_value():
    df = pd.DataFrame({
        'CreditScore':[650, 700, None],
        'Gender':['Male', 'Female', 'Male'],
        'Geography':['France', 'Spain', 'Germany'],
        'Card Type':['Visa', 'Mastercard', 'American Express'],
        'Age': [30, 40, 50],
        'Tenure': [2, 3, 4],
        'Balance': [100000, 200000, 300000],
        'NumOfProducts': [1, 2, 3],
        'HasCrCard': [1, 0, 1],
        'IsActiveMember': [1, 0, 1],
        'EstimatedSalary': [50000, 60000, 70000],
        'Complain': [0, 1, 0],
        'Satisfaction Score': [4, 5, 3],
        'Point Earned': [10, 20, 30],
        'Exited': [0, 1, 0]
    })

    pre = preprocessor()
    transformed = pre.fit_transform(df)
    assert transformed.shape[1] == 19


