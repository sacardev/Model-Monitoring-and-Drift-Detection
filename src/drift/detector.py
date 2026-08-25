from dataclasses import dataclass, field
import pandas as pd
from .stats import population_stability_index, ks_test, chi_squared_test

PSI_THRESHOLD = 0.2
P_VALUE_THRESHOLD = 0.05

@dataclass
class DriftResult:
    feature: str
    feature_type: str
    metric: str
    value: float
    drifted: bool

@dataclass
class DriftReport:
    feature_result: list = field(default_factory=list)

    @property
    def dataset_drifted(self) -> bool:
        return any(r.drifted for r in self.feature_result)
    
    @property
    def drift_share(self) -> float:
        return sum(r.drifted for r in self.feature_result) / len(self.feature_result) if self.feature_result else 0.0

    def to_dict(self) -> dict:
        return {
            "dataset_drifted": self.dataset_drifted(),
            "drift_share": self.drift_share(),
            "feature_result": [r.__dict__ for r in self.feature_result]
        }

class DriftDetector:
    def __init__(self, numeric_cols:list[str], categorical_cols:list[str]):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols

    def run(self, reference: pd.DataFrame, current: pd.DataFrame) -> DriftReport:
        result = []

        for col in self.numeric_cols:
            psi = population_stability_index(reference[col], current[col])
            _, p_value = ks_test(reference[col], current[col])
            drifted = psi > PSI_THRESHOLD or p_value < P_VALUE_THRESHOLD
            result.append(DriftResult(col,"numeric", "psi", round(psi,4), drifted))

        for col in self.categorical_cols:
            _, p_value = chi_squared_test(reference[col], current[col])
            drifted = p_value < P_VALUE_THRESHOLD
            result.append(DriftResult(col, "categorical", "chi2", round(p_value, 4), drifted))
        
        return DriftReport(feature_result=result)