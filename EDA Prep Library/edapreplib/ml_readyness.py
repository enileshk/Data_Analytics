"""
ml_readyness.py

Assess a Pandas DataFrame for common machine-learning readiness issues
and export the findings to a formatted Excel workbook.

Usage:
    from edapreplib.ml_readyness import assess_ml_readiness

    results = assess_ml_readiness(
        df,
        target="Churn",
        task="classification",
        output_file="ML_Readiness_Report.xlsx"
    )

Dependencies:
    pandas, numpy, openpyxl

Notes:
    - This module reports issues; it does not modify the input DataFrame.
    - Heuristics such as identifier/leakage detection require human review.
    - Classification/regression task should preferably be supplied explicitly.
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter


def _safe_pct(numerator, denominator):
    """Return a percentage, or NaN when the denominator is zero."""
    if denominator == 0:
        return np.nan
    return round((numerator / denominator) * 100, 2)


def _is_datetime(series):
    """Detect datetime columns, including strings that pandas can parse reliably."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    if not (pd.api.types.is_object_dtype(series) or
            pd.api.types.is_string_dtype(series)):
        return False

    non_null = series.dropna()
    if non_null.empty:
        return False

    # Avoid treating ordinary categorical strings as dates.
    sample = non_null.astype(str).head(100)
    date_like = sample.str.contains(
        r"[-/]\d{1,4}[-/]|^\d{4}-\d{2}|:\d{2}",
        regex=True
    ).mean()

    if date_like < 0.7:
        return False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() >= 0.8


def _column_kind(series):
    """Classify a feature for readiness reporting."""
    if _is_datetime(series):
        return "Datetime"
    if pd.api.types.is_bool_dtype(series):
        return "Boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "Numeric"
    return "Categorical"


def _iqr_outlier_stats(series):
    """Calculate IQR outlier count and percentage for a numeric series."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 0, 0.0

    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1

    if pd.isna(iqr) or iqr == 0:
        return 0, 0.0

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = int(((numeric < lower) | (numeric > upper)).sum())
    return count, _safe_pct(count, len(numeric))


def _target_task(target_series, task):
    """Validate/infer task. Inference is deliberately conservative."""
    allowed = {"classification", "regression"}
    if task is not None:
        task = task.lower().strip()
        if task not in allowed:
            raise ValueError("task must be 'classification', 'regression', or None")
        return task, "User specified"

    non_null = target_series.dropna()
    if non_null.empty:
        return "unknown", "Cannot infer: target has no non-missing values"

    # A categorical dtype or text/bool target is generally a classification target.
    if (pd.api.types.is_bool_dtype(non_null) or
            pd.api.types.is_object_dtype(non_null) or
            pd.api.types.is_string_dtype(non_null) or
            isinstance(non_null.dtype, pd.CategoricalDtype)):
        return "classification", "Inferred from target data type"

    unique_count = non_null.nunique()
    if unique_count <= 10:
        return "classification", "Possible classification target; confirm task"
    return "regression", "Possible regression target; confirm task"


def _add_check(summary_rows, check, status, finding, recommendation):
    # Do not show recommendations for PASS cases
    if status == "PASS":
        recommendation = "NA"

    summary_rows.append({
        "Check": check,
        "Status": status,
        "Finding": finding,
        "Recommendation": recommendation
    })


def _format_workbook(path):
    """Apply readable formatting to all worksheets."""
    wb = load_workbook(path)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    thin_gray = Side(style="thin", color="D9E2F3")

    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(
                horizontal="left", vertical="center", wrap_text=True
            )
            cell.border = Border(bottom=thin_gray)

        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(
                    horizontal="left", vertical="top", wrap_text=True
                )

        # Set sensible widths without allowing very long text to make sheets huge.
        for col_cells in ws.columns:
            col_letter = get_column_letter(col_cells[0].column)
            max_len = 0
            for cell in col_cells:
                value = "" if cell.value is None else str(cell.value)
                max_len = max(max_len, min(len(value), 55))
            ws.column_dimensions[col_letter].width = min(max(max_len + 2, 12), 42)

        ws.row_dimensions[1].height = 30

    wb.save(path)


def assess_ml_readiness(
    df,
    target=None,
    task=None,
    output_file="ML_Readiness_Report.xlsx",
    missing_review_pct=5.0,
    missing_investigate_pct=30.0,
    max_class_imbalance_ratio=10.0,
    identifier_unique_pct=98.0,
    write_excel=True
):
    """
    Assess common ML-readiness concerns and optionally export an Excel report.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset. It is not modified.
    target : str or None
        Target column name. If None, target-specific checks are skipped.
    task : {'classification', 'regression'} or None
        ML task. Prefer specifying this explicitly.
    output_file : str or Path
        Excel output path.
    missing_review_pct : float
        Missing percentage at/above which a feature is flagged for review.
    missing_investigate_pct : float
        Missing percentage at/above which stronger investigation is suggested.
    max_class_imbalance_ratio : float
        Minority/majority imbalance threshold. For example, 10 means 10:1.
    identifier_unique_pct : float
        Unique-value percentage used to flag possible identifier columns.
    write_excel : bool
        If True, write the report to Excel.

    Returns
    -------
    dict
        Contains summary, feature_readiness, target_analysis and recommendations
        DataFrames, plus the output path when Excel export is enabled.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if df.empty:
        raise ValueError("df is empty")
    if df.columns.duplicated().any():
        raise ValueError("DataFrame contains duplicate column names")

    if target is not None and target not in df.columns:
        raise ValueError(f"Target column {target!r} not found in DataFrame")

    n_rows, n_cols = df.shape
    summary_rows = []
    feature_rows = []
    recommendation_rows = []
    target_rows = []

    # Dataset-level checks
    duplicate_count = int(df.duplicated().sum())
    duplicate_pct = _safe_pct(duplicate_count, n_rows)

    _add_check(
        summary_rows,
        "Dataset dimensions",
        "PASS" if n_rows > 0 and n_cols > 0 else "FAIL",
        f"{n_rows:,} rows and {n_cols:,} columns",
        "Confirm that the dataset contains enough representative observations for the intended task."
    )

    _add_check(
        summary_rows,
        "Duplicate rows",
        "REVIEW" if duplicate_count else "PASS",
        f"{duplicate_count:,} exact duplicate rows ({duplicate_pct}%)",
        "Investigate duplicates. Remove them only if they are erroneous for the business problem."
    )

    all_missing_cols = [str(c) for c in df.columns if df[c].isna().all()]
    _add_check(
        summary_rows,
        "Entirely missing columns",
        "REVIEW" if all_missing_cols else "PASS",
        ", ".join(all_missing_cols) if all_missing_cols else "No entirely missing columns",
        "Drop or recover entirely missing columns after checking their meaning."
    )

    constant_cols = [
        str(c) for c in df.columns
        if df[c].nunique(dropna=True) <= 1
    ]
    _add_check(
        summary_rows,
        "Constant columns",
        "REVIEW" if constant_cols else "PASS",
        ", ".join(constant_cols) if constant_cols else "No constant columns detected",
        "Constant features usually add no predictive information; investigate and consider removing them."
    )

    # Feature-level assessment
    for col in df.columns:
        series = df[col]
        kind = _column_kind(series)
        missing_count = int(series.isna().sum())
        missing_pct = _safe_pct(missing_count, n_rows)
        unique_count = int(series.nunique(dropna=True))
        unique_pct = _safe_pct(unique_count, n_rows)
        outlier_count, outlier_pct = (np.nan, np.nan)

        if kind == "Numeric":
            outlier_count, outlier_pct = _iqr_outlier_stats(series)

        flags = []
        actions = []

        if missing_pct > 0:
            if missing_pct >= missing_investigate_pct:
                flags.append("High missingness")
                actions.append("Investigate usefulness and missingness pattern; consider suitable imputation.")
            elif missing_pct >= missing_review_pct:
                flags.append("Missing values")
                actions.append("Choose an imputation strategy appropriate to the feature and model.")
            else:
                flags.append("Some missing values")
                actions.append("Handle missing values before fitting models.")

        if unique_count <= 1:
            flags.append("Constant or all-missing")
            actions.append("Investigate and usually exclude from model features.")

        if kind == "Categorical" and unique_count > 50:
            flags.append("High-cardinality category")
            actions.append("Review category frequency; consider grouping or a suitable encoding strategy.")

        if kind == "Datetime":
            flags.append("Datetime feature")
            actions.append("Extract meaningful date/time components; check availability at prediction time.")

        if kind == "Categorical" and unique_count > 0:
            actions.append("Encode categories for algorithms that require numeric input.")

        if kind == "Numeric" and pd.api.types.is_integer_dtype(series.dropna()) and unique_count > 20:
            actions.append("Check whether this is a continuous measurement, count, or encoded category.")

        if pd.api.types.is_numeric_dtype(series):
            if outlier_pct is not np.nan and pd.notna(outlier_pct) and outlier_pct > 0:
                flags.append("Potential IQR outliers")
                actions.append("Investigate extreme values; do not remove valid observations automatically.")

        # Heuristic identifier detection: high uniqueness plus ID-like column name.
        col_lower = str(col).lower()
        id_name = bool(re.search(r"(^id$|_id$|id$|^id_|customer.?number|account.?number|record.?number)", col_lower))
        if n_rows > 1 and unique_pct >= identifier_unique_pct and id_name and col != target:
            flags.append("Possible identifier")
            actions.append("Check whether this identifier should be excluded; verify it does not encode useful grouping/time information.")

        # Suspiciously target-like names. This is only a heuristic.
        leakage_name = bool(re.search(
            r"(churn|target|label|outcome|cancel|cancell|termination|default|fraud).*(date|status|reason|result)|"
            r"(date|status|reason|result).*(churn|cancel|termination|default|fraud)",
            col_lower
        ))
        if leakage_name and col != target:
            flags.append("Possible target leakage")
            actions.append("Verify that this feature is legitimately available at prediction time and does not reveal the target.")

        if not flags:
            flags.append("No obvious issue detected")
        if not actions:
            actions.append("No immediate preprocessing action identified; validate with the chosen model.")

        feature_rows.append({
            "Feature": str(col),
            "Data Type": str(series.dtype),
            "Feature Kind": kind,
            "Missing Count": missing_count,
            "Missing %": missing_pct,
            "Unique Count": unique_count,
            "Unique %": unique_pct,
            "Outlier Count (IQR)": outlier_count,
            "Outlier % (IQR)": outlier_pct,
            "Flags": "; ".join(dict.fromkeys(flags)),
            "Suggested Action": "; ".join(dict.fromkeys(actions))
        })

        for action in dict.fromkeys(actions):
            recommendation_rows.append({
                "Feature": str(col),
                "Issue / Observation": "; ".join(dict.fromkeys(flags)),
                "Recommendation": action
            })

    feature_df = pd.DataFrame(feature_rows)

    # Target-specific assessment
    if target is None:
        _add_check(
            summary_rows,
            "Target variable",
            "NOT CHECKED",
            "No target column supplied",
            "Supply target='your_target_column' to enable target-specific checks."
        )
        resolved_task = "unknown"
        task_note = "No target supplied"
    else:
        target_series = df[target]
        target_missing = int(target_series.isna().sum())
        target_non_null = target_series.dropna()
        target_unique = int(target_non_null.nunique())
        resolved_task, task_note = _target_task(target_series, task)

        _add_check(
            summary_rows,
            "Target missing values",
            "FAIL" if target_missing == n_rows else ("REVIEW" if target_missing else "PASS"),
            f"{target_missing:,} missing target values ({_safe_pct(target_missing, n_rows)}%)",
            "Rows without a target generally cannot be used for supervised training; investigate why values are missing."
        )

        if target_unique <= 1:
            target_status = "FAIL"
            target_finding = f"Only {target_unique} distinct non-missing target value(s)"
            target_action = "A supervised model cannot learn class/target variation from a constant target."
        else:
            target_status = "PASS"
            target_finding = f"{target_unique:,} distinct non-missing target values; task={resolved_task} ({task_note})"
            target_action = "Confirm the target and task definition with the business problem."
        _add_check(summary_rows, "Target variation and task", target_status, target_finding, target_action)

        if resolved_task == "classification" and not target_non_null.empty:
            counts = target_non_null.value_counts(dropna=False)
            total = int(counts.sum())
            majority = int(counts.max())
            minority = int(counts.min())
            imbalance_ratio = (majority / minority) if minority else np.inf

            # Calculate class percentages
            class_percentages = (counts / total) * 100

            majority_label = counts.idxmax()
            minority_label = counts.idxmin()

            majority_pct = float(class_percentages.max())
            minority_pct = float(class_percentages.min())

            # Assess overall class balance
            if majority_pct >= 95:
                balance_status = "WARNING"
            elif majority_pct >= 80:
                balance_status = "REVIEW"
            else:
                balance_status = "PASS"

            # Add each class to Target Analysis
            for label, count in counts.items():

                if label == majority_label:
                    class_role = "Majority class"
                elif label == minority_label:
                    class_role = "Minority class"
                else:
                    class_role = "Other class"

                target_rows.append({
                    "Class / Target Value": str(label),
                    "Count": int(count),
                    "Percentage": _safe_pct(int(count), total),
                    "Class Role": class_role,
                    "Overall Balance Status": balance_status,
                    "Majority Class %": round(majority_pct, 2),
                    "Minority Class %": round(minority_pct, 2)
                })

            if len(counts) < 2:
                _add_check(
                    summary_rows, "Target class distribution", "FAIL",
                    f"Only {len(counts)} class(es) present",
                    "Check target definition and data selection; classification normally requires at least two classes."
                )
            else:
                imbalanced = imbalance_ratio >= max_class_imbalance_ratio
                _add_check(
                    summary_rows,
                    "Target class distribution",
                    "REVIEW" if imbalanced else "PASS",
                    f"{len(counts)} classes; majority/minority ratio {imbalance_ratio:.2f}:1",
                    "If imbalance affects the task, use suitable metrics (e.g. precision, recall, F1, PR-AUC) and evaluate resampling/class weighting only within training data."
                )

        elif resolved_task == "regression" and not target_non_null.empty:
            numeric_target = pd.to_numeric(target_non_null, errors="coerce")
            nonnumeric_count = int(numeric_target.isna().sum())
            _add_check(
                summary_rows,
                "Regression target format",
                "REVIEW" if nonnumeric_count else "PASS",
                f"{nonnumeric_count} non-numeric target values after numeric conversion",
                "A standard regression target should be numeric; review parsing and whether regression is the correct task."
            )

        else:
            _add_check(
                summary_rows,
                "Target-specific checks",
                "NOT CHECKED",
                f"Task could not be confidently assessed: {task_note}",
                "Set task='classification' or task='regression' explicitly."
            )

        # Basic leakage-like exact duplicate of target feature
        for col in df.columns:
            if col == target:
                continue
            try:
                both_present = df[col].notna() & df[target].notna()
                if both_present.sum() > 0:
                    exact_match = (
                        df.loc[both_present, col].astype(str).values ==
                        df.loc[both_present, target].astype(str).values
                    ).mean()
                    if exact_match >= 0.99:
                        _add_check(
                            summary_rows,
                            f"Possible target duplicate: {col}",
                            "REVIEW",
                            f"{exact_match * 100:.1f}% of jointly non-missing values match target values",
                            "Inspect this feature for target duplication or leakage before modelling."
                        )
            except Exception:
                pass

    # General recommendations
    if df.select_dtypes(include="object").shape[1] > 0:
        _add_check(
            summary_rows, "Categorical encoding", "REVIEW",
            f"{df.select_dtypes(include='object').shape[1]} object column(s) detected",
            "Inspect text/object columns and encode suitable features inside a train-fitted preprocessing pipeline."
        )

    if n_rows < 100:
        _add_check(
            summary_rows, "Dataset size", "REVIEW",
            f"Only {n_rows} rows",
            "Small datasets can be difficult to evaluate reliably; use an appropriate validation strategy and interpret metrics cautiously."
        )
    else:
        _add_check(
            summary_rows, "Dataset size", "REVIEW",
            f"{n_rows:,} rows; row count alone cannot establish adequacy",
            "Assess sample size relative to task complexity, class counts, feature count, and validation needs."
        )

    _add_check(
        summary_rows, "Train/test leakage prevention", "REVIEW",
        "Not fully verifiable from a DataFrame-only check",
        "Split data before fitting imputers, encoders, scalers, feature selectors, or resampling methods. Use pipelines and group/time-aware splitting when appropriate."
    )

    summary_df = pd.DataFrame(summary_rows)
    target_df = pd.DataFrame(target_rows, columns=[
        "Class / Target Value",
        "Count",
        "Percentage",
        "Class Role",
        "Overall Balance Status",
        "Majority Class %",
        "Minority Class %"
    ])
    recommendations_df = pd.DataFrame(recommendation_rows)

    output_path = None
    if write_excel:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            summary_df.to_excel(
                writer,
                sheet_name="ML_Readiness_Summary",
                index=False,
                startrow=2
            )

            feature_df.to_excel(
                writer, sheet_name="Feature_Readiness", index=False
            )
            target_df.to_excel(
                writer, sheet_name="Target_Analysis", index=False
            )
            recommendations_df.to_excel(
                writer, sheet_name="Recommendations", index=False
            )

            # Add a highlighted warning in one merged row
            ws = writer.sheets["ML_Readiness_Summary"]

            ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=4)

            warning_cell = ws.cell(row=1, column=1)
            warning_cell.value = (
                "IMPORTANT: This is an automated, preliminary ML readiness assessment. "
                "Independently review and validate all findings before taking action. "
                "PASS does not guarantee ML suitability."
            )

            warning_cell.font = Font(
                bold=True,
                color="9C0006",
                size=11
            )

            warning_cell.fill = PatternFill(
                fill_type="solid",
                fgColor="FFC7CE"
            )

            warning_cell.alignment = Alignment(
                horizontal="left",
                vertical="center",
                wrap_text=True
            )

            # Highlight the complete merged warning row
            for cell in ws[1]:
                cell.fill = PatternFill(
                    fill_type="solid",
                    fgColor="FFC7CE"
                )

            ws.row_dimensions[1].height = 45
            ws.row_dimensions[2].height = 8
    
            feature_df.to_excel(writer, sheet_name="Feature_Readiness", index=False)
            target_df.to_excel(writer, sheet_name="Target_Analysis", index=False)
            recommendations_df.to_excel(writer, sheet_name="Recommendations", index=False)

        '''with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="ML_Readiness_Summary", index=False)
            feature_df.to_excel(writer, sheet_name="Feature_Readiness", index=False)
            target_df.to_excel(writer, sheet_name="Target_Analysis", index=False)
            recommendations_df.to_excel(writer, sheet_name="Recommendations", index=False)'''

        _format_workbook(output_path)

    return {
        "summary": summary_df,
        "feature_readiness": feature_df,
        "target_analysis": target_df,
        "recommendations": recommendations_df,
        "task": resolved_task,
        "output_file": str(output_path) if output_path else None
    }

if __name__ == "__main__":
    # Small smoke test; replace with your own CSV and target column.
    demo = pd.DataFrame({
        "CustomerID": [1001, 1002, 1003, 1004, 1005, 1006],
        "Age": [25, 42, 38, np.nan, 67, 31],
        "Contract": ["Month-to-month", "Two year", "One year",
                     "Month-to-month", "Two year", "One year"],
        "MonthlyCharges": [70.5, 45.0, 82.1, 99.0, 40.0, 71.0],
        "Churn": ["Yes", "No", "No", "Yes", "No", "No"]
    })

    result = assess_ml_readiness(
        demo,
        target="Churn",
        task="classification",
        output_file="ML_Readiness_Demo.xlsx"
    )
    print(result["summary"].to_string(index=False))
    print(f"\nExcel report created: {result['output_file']}")

