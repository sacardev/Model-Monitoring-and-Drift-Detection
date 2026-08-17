import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.data_prep import load_data, preprocessor

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

def main():
    df = load_data('data/raw/customer_churn.csv')

    X = df[num_cols + cate_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)

    model = Pipeline([
        ('preprocessor', preprocessor()),
        ('classifier', RandomForestClassifier(n_estimators = 200, max_depth=10, random_state=42))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')

    joblib.dump(model, 'artifacts/model.pkl')
    X_train.assign(**{target_col: y_train}).to_csv("reference_data.csv", index=False)

    with open("artifacts/baseline_metrics.txt", "w") as f:
        f.write(f"accuracy={accuracy:.4f}\nf1={f1:.4f}\n")
    

    print("Model trained and baseline created")

if __name__ == "__main__":
    main()
