# 📊 Model Monitoring and Drift Detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/pytest-passing-brightgreen.svg)](https://pytest.org/)

An end-to-end Machine Learning pipeline and drift monitoring framework built for **Customer Churn Prediction**. This repository implements automated data preparation, robust preprocessing pipelines, model training with baseline tracking, schema validation, and unit testing.

---

## 🚀 Features

- **Robust Data Preprocessing:** Automated numerical imputation & scaling combined with categorical imputation & one-hot encoding via `scikit-learn` `ColumnTransformer` and `Pipeline`.
- **Reproducible Model Training:** Random Forest classifier pipeline with automated evaluation (`Accuracy`, `F1-Score`).
- **Baseline & Artifact Generation:** Exports trained pipelines (`model.pkl`), reference datasets (`reference_data.csv`), and baseline performance metrics.
- **Data Validation & Schema Checks:** Enforces column data types and structural integrity before ingestion.
- **Automated Testing Suite:** Comprehensive unit tests using `pytest` to validate preprocessing pipelines, missing value handling, and artifact loading.

---

## 📂 Project Structure

```text
drift-monitor/
├── artifacts/                  # Exported models, pipelines, and baseline metrics
│   ├── baseline_metrics.txt    # Baseline accuracy & F1 score
│   └── model.pkl               # Serialized end-to-end ML pipeline
├── data/
│   └── raw/
│       └── customer_churn.csv  # Raw dataset
├── src/
│   ├── drift/                  # Drift detection modules & statistical tests
│   │   ├── detector.py
│   │   ├── evidently_report.py
│   │   └── stats.py
│   ├── config.py               # Central configuration settings
│   ├── data_prep.py            # Data loading & preprocessing pipelines
│   ├── schema.py               # Schema validation & type assertions
│   └── train.py                # Model training, evaluation & artifact export
├── tests/
│   ├── test_artifact_loads.py  # Tests verifying saved model inference
│   └── test_data_prep.py       # Tests validating preprocessing transformations
├── .gitignore
├── reference_data.csv          # Baseline reference dataset for drift comparison
├── requirements.txt
└── README.md
```

---

## 🛠️ Getting Started

### 1. Prerequisites
- Python 3.10+
- `pip` / `virtualenv`

### 2. Clone the Repository
```bash
git clone https://github.com/sacardev/Model-Monitoring-and-Drift-Detection.git
cd Model-Monitoring-and-Drift-Detection/drift-monitor
```

### 3. Set Up Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Train Model & Create Baselines
Runs the full training workflow, evaluates against test split, and outputs `artifacts/model.pkl` along with baseline metrics:
```bash
python -m src.train
```

### Run Tests
Execute the unit tests to ensure preprocessing stability and model artifact compatibility:
```bash
python -m pytest tests/ -v
```

---

## 📈 Monitoring & Drift Workflow

1. **Reference Data**: `reference_data.csv` serves as the training baseline distribution.
2. **Current Inference Data**: New batch data is validated against `schema.py`.
3. **Statistical Drift Detection**: Compares incoming distributions against baseline using Kolmogorov-Smirnov (KS) tests, PSI (Population Stability Index), and Evidently AI reports.

---

## 👤 Author

- **sacardev** - [GitHub Profile](https://github.com/sacardev)
