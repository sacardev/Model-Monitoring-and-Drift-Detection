"""
simulate_drift.py
Script to simulate synthetic covariate shift (numerical & categorical drift)
for testing drift detection pipelines (e.g. PSI, KS-test, Chi-squared, Evidently).
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


def inject_drift(
    df: pd.DataFrame,
    numerical_col: str = "MonthlyCharges",
    categorical_col: str = "Contract",
    source_category: str = "Two year",
    target_category: str = "Month-to-month",
    num_scale: float = 1.4,
    num_shift: float = 15.0,
    cat_flip_prob: float = 0.5,
    seed: Optional[int] = 42,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Injects synthetic numerical and categorical drift into a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        Input reference dataset.
    numerical_col : str
        Name of numerical feature to perturb.
    categorical_col : str
        Name of categorical feature to perturb.
    source_category : str
        The categorical value to sample and convert.
    target_category : str
        The target category to flip into.
    num_scale : float
        Multiplicative factor for numerical shift (mean and variance expansion).
    num_shift : float
        Additive offset for numerical shift.
    cat_flip_prob : float
        Probability of flipping each matching row to target_category.
    seed : int, optional
        Random seed for reproducibility using modern NumPy Generator.
    inplace : bool
        If True, modifies DataFrame in place to conserve memory.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing injected drift.
    """
    rng = np.random.default_rng(seed)
    drifted = df if inplace else df.copy()

    # 1. Numerical Covariate Shift (Vectorized scaling + shift)
    if numerical_col in drifted.columns:
        drifted[numerical_col] = drifted[numerical_col] * num_scale + num_shift

    # 2. Categorical Covariate Shift (Optimized: generate RNG only for target rows)
    if categorical_col in drifted.columns and categorical_col:
        # If column is of categorical dtype, ensure target_category exists in categories
        if isinstance(drifted[categorical_col].dtype, pd.CategoricalDtype):
            if target_category not in drifted[categorical_col].cat.categories:
                drifted[categorical_col] = drifted[categorical_col].cat.add_categories([target_category])

        target_indices = drifted.index[drifted[categorical_col] == source_category]
        if len(target_indices) > 0:
            # Generate random flags only for the subset matching the condition
            flip_mask = rng.random(len(target_indices)) < cat_flip_prob
            drifted.loc[target_indices[flip_mask], categorical_col] = target_category

    return drifted


def main():
    # Resolve robust paths relative to this script's directory
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent

    # Check reference data location (project root or current dir)
    ref_path = repo_dir / "reference_data.csv"
    if not ref_path.exists():
        ref_path = Path("reference_data.csv")

    if not ref_path.exists():
        raise FileNotFoundError(
            f"Could not locate reference_data.csv at {ref_path}. Please check the path."
        )

    print(f"Loading reference data from: {ref_path}")
    ref_df = pd.read_csv(ref_path)

    # Detect dataset schema (Telco vs Bank Churn)
    if "MonthlyCharges" in ref_df.columns:
        # Telco Churn schema
        drifted_df = inject_drift(
            ref_df,
            numerical_col="MonthlyCharges",
            categorical_col="Contract",
            source_category="Two year",
            target_category="Month-to-month",
        )
    elif "EstimatedSalary" in ref_df.columns and "Geography" in ref_df.columns:
        # Bank Churn schema
        drifted_df = inject_drift(
            ref_df,
            numerical_col="EstimatedSalary",
            categorical_col="Geography",
            source_category="France",
            target_category="Germany",
        )
    elif "Balance" in ref_df.columns and "Card Type" in ref_df.columns:
        # Bank Churn schema variant
        drifted_df = inject_drift(
            ref_df,
            numerical_col="Balance",
            categorical_col="Card Type",
            source_category="DIAMOND",
            target_category="SILVER",
        )
    else:
        # Robust fallback: identify available numerical and categorical columns
        num_cols = ref_df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = ref_df.select_dtypes(exclude=[np.number]).columns.tolist()
        num_col = num_cols[0] if num_cols else ""
        cat_col = cat_cols[0] if cat_cols else ""
        src_cat, tgt_cat = "", ""

        if cat_col:
            val_counts = ref_df[cat_col].value_counts()
            if len(val_counts) >= 2:
                src_cat = val_counts.index[0]
                tgt_cat = val_counts.index[1]
            elif len(val_counts) == 1:
                src_cat = val_counts.index[0]
                tgt_cat = f"{src_cat}_drifted"

        drifted_df = inject_drift(
            ref_df,
            numerical_col=num_col,
            categorical_col=cat_col,
            source_category=src_cat,
            target_category=tgt_cat,
        )

    # Ensure output directory exists
    output_path = repo_dir / "data" / "drifted_sample.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    drifted_df.to_csv(output_path, index=False)
    print(f"Wrote drifted dataset to: {output_path} ({len(drifted_df)} rows)")


if __name__ == "__main__":
    main()