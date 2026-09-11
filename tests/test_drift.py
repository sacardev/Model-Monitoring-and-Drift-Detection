import numpy as np
import pandas as pd
from src.drift.detector import DriftDetector, DriftReport
from src.drift.evidently_report import generate_drift_report
from src.drift.stats import chi_squared_test, ks_test, population_stability_index


def test_psi_identical_distributions():
    np.random.seed(42)
    s1 = pd.Series(np.random.normal(0, 1, 1000).tolist())
    s2 = s1.copy()
    psi = population_stability_index(s1, s2)
    assert round(psi, 4) == 0.0


def test_psi_drifted_distributions():
    np.random.seed(42)
    s1 = pd.Series(np.random.normal(0, 1, 1000).tolist())
    s2 = pd.Series(np.random.normal(5, 1, 1000).tolist())
    psi = population_stability_index(s1, s2)
    assert psi > 0.2


def test_ks_test():
    np.random.seed(42)
    s1 = pd.Series(np.random.normal(0, 1, 500).tolist())
    s2 = pd.Series(np.random.normal(5, 1, 500).tolist())
    stat, p_val = ks_test(s1, s2)
    assert p_val < 0.05
    assert stat > 0.5


def test_chi_squared_test():
    s1 = pd.Series(["A"] * 50 + ["B"] * 50)
    s2 = pd.Series(["A"] * 5 + ["B"] * 95)
    stat, p_val = chi_squared_test(s1, s2)
    assert p_val < 0.05


def test_drift_detector_run():
    np.random.seed(42)
    ref = pd.DataFrame({
        "num1": np.random.normal(0, 1, 200),
        "cat1": ["A"] * 100 + ["B"] * 100,
    })
    curr = pd.DataFrame({
        "num1": np.random.normal(10, 1, 200),
        "cat1": ["A"] * 10 + ["B"] * 190,
    })

    detector = DriftDetector(numeric_cols=["num1"], categorical_cols=["cat1"])
    report: DriftReport = detector.run(ref, curr)

    assert report.dataset_drifted is True
    assert report.drift_share == 1.0

    report_dict = report.to_dict()
    assert report_dict["dataset_drifted"] is True
    assert report_dict["drift_share"] == 1.0
    assert len(report_dict["feature_result"]) == 2


def test_evidently_drift_report(tmp_path):
    ref = pd.DataFrame({
        "CreditScore": [600, 650, 700, 750],
        "Age": [25, 35, 45, 55],
    })
    curr = pd.DataFrame({
        "CreditScore": [610, 640, 710, 760],
        "Age": [26, 36, 44, 56],
    })
    html_path = tmp_path / "report.html"
    report = generate_drift_report(ref, curr, output_html_path=html_path)
    assert report is not None
    assert html_path.exists()
