import numpy as np
import pandas as pd
from scipy import stats


def population_stability_index(
    reference: pd.Series | np.ndarray,
    current: pd.Series | np.ndarray,
    bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) between reference and current continuous data."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)

    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]

    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    quantiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(ref, quantiles)
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    ref_pct = ref_counts / len(ref)
    cur_pct = cur_counts / len(cur)

    # Handle zero frequencies with epsilon smoothing
    ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
    cur_pct = np.where(cur_pct == 0, epsilon, cur_pct)

    # Re-normalize
    ref_pct = ref_pct / np.sum(ref_pct)
    cur_pct = cur_pct / np.sum(cur_pct)

    psi_value = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi_value)


def ks_test(
    reference: pd.Series | np.ndarray,
    current: pd.Series | np.ndarray,
) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test for numeric data distributions.
    Returns (ks_statistic, p_value).
    """
    ref = pd.Series(reference).dropna()
    cur = pd.Series(current).dropna()

    if len(ref) == 0 or len(cur) == 0:
        return 0.0, 1.0

    res = stats.ks_2samp(ref, cur)
    return float(res.statistic), float(res.pvalue)


def chi_squared_test(
    reference: pd.Series | np.ndarray,
    current: pd.Series | np.ndarray,
) -> tuple[float, float]:
    """Chi-Square test of independence for categorical distributions.
    Returns (chi2_statistic, p_value).
    """
    ref = pd.Series(reference).dropna().astype(str)
    cur = pd.Series(current).dropna().astype(str)

    if len(ref) == 0 or len(cur) == 0:
        return 0.0, 1.0

    categories = sorted(list(set(ref.unique()).union(set(cur.unique()))))
    if len(categories) < 2:
        return 0.0, 1.0

    ref_counts = ref.value_counts().reindex(categories, fill_value=0).values
    cur_counts = cur.value_counts().reindex(categories, fill_value=0).values

    contingency_table = np.array([ref_counts, cur_counts])

    if contingency_table.sum() == 0:
        return 0.0, 1.0

    res = stats.chi2_contingency(contingency_table)
    return float(res.statistic), float(res.pvalue)