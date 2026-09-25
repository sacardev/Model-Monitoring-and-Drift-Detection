from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

# NumPy 2.0 compatibility for older Evidently versions
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64

try:
    from evidently.report import Report
except ImportError:
    from evidently import Report

try:
    from evidently.metric_preset import DataDriftPreset
except ImportError:
    from evidently.presets import DataDriftPreset


def generate_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_html_path: Optional[Union[str, Path]] = None,
    output_json_path: Optional[Union[str, Path]] = None,
    output_html: Optional[Union[str, Path]] = None,
    output_json: Optional[Union[str, Path]] = None,
):
    """Generate Evidently data drift report and save to HTML/JSON files."""
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference, current_data=current)
    target = snapshot if snapshot is not None else report

    html_dest = output_html_path if output_html_path is not None else output_html
    json_dest = output_json_path if output_json_path is not None else output_json

    if html_dest is not None:
        target.save_html(str(html_dest))

    if json_dest is not None:
        target.save_json(str(json_dest))

    return target


# Alias for backwards compatibility
generate_evidently_report = generate_drift_report

