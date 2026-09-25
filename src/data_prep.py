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

# Feature and target column definitions
TARGET_COL = 'Exited'

CATEGORICAL_COLS = [
    'Geography', 
    'Gender', 
    'Card Type'
]

NUMERICAL_COLS = [
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

# Aliases for backwards compatibility and lowercase naming conventions
target_col = TARGET_COL
cate_cols = CATEGORICAL_COLS
num_cols = NUMERICAL_COLS

# Preprocessing steps for the dataset
def preprocessor() -> ColumnTransformer:
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cate_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    return ColumnTransformer([
        ('nums', num_pipeline, NUMERICAL_COLS),
        ('cate', cate_pipeline, CATEGORICAL_COLS)
    ])