import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

#Loading the dataset into the Dataframe
def load_data(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

#Preprocessing steps for the dataset
def preprocessor() -> ColumnTransformer:

    target_col = 'Exited'

    cate_cols = [
        'Geography', 
        'Gender', 
        'Card Type'
    ]

    num_cols = [
    'CreditScore', 
    'Age', 
    'Tenure', 
    'Balance', 
    'NumOfProducts', 
    'HasCrCard', 
    'IsActiveMember', 
    'EstimatedSalary', 
    'Complain', 
    'Satisfaction Score', 
    'Point Earned'
    ]

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cate_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    return ColumnTransformer([
        ('nums', num_pipeline, num_cols),
        ('cate', cate_pipeline, cate_cols)
    ])