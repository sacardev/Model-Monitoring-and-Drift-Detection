from pathlib import Path
from typing import Optional, Union

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


def generate_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_html_path: Optional[Union[str, Path]] = None,
    output_json_path: Optional[Union[str, Path]] = None,
    output_html: Optional[Union[str, Path]] = None,
    output_json: Optional[Union[str, Path]] = None,
):
    """Generate Evidently data drift report and save to HTML/JSON files.

    In Evidently 0.6+, report.run() returns a Snapshot object which provides
    the save_html() and save_json() methods.
    """
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference, current_data=current)

    html_dest = output_html_path if output_html_path is not None else output_html
    json_dest = output_json_path if output_json_path is not None else output_json

    if html_dest is not None:
        snapshot.save_html(str(html_dest))

    if json_dest is not None:
        snapshot.save_json(str(json_dest))

    return snapshot


# Alias for backwards compatibility
generate_evidently_report = generate_drift_report

