"""
STELLAR Validation Framework - Complete Integration for Streamlit App
Enhanced version with comprehensive data quality checks
"""

import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import logging
from scipy import stats
from scipy.optimize import curve_fit
import streamlit as st
import json
from pathlib import Path

try:
    from commodity_validators import (
        agricultural_seasonality_validator,
        metal_supply_constraint_validator,
        carbon_credit_validator
    )
    COMMODITY_VALIDATORS_AVAILABLE = True
except ImportError:
    COMMODITY_VALIDATORS_AVAILABLE = False
# Keep all existing imports from ground_truth_validators
from ground_truth_validators import (
    validate_units_and_scale,
    check_year_anomalies,
    cost_parity_threshold,
    adoption_saturation_feasibility,
    oil_displacement_check,
    derived_oil_displacement_validator,
    global_oil_sanity_validator,
    unit_conversion_validator,
    multi_source_consistency_validator,
    capacity_factor_validator,
    market_context_validator,
    data_source_integrity_validator,
    metric_validity_validator,
    regional_definition_validator,
    run_all_domain_expert_validators,
    format_expert_validation_summary,
)

logger = logging.getLogger(__name__)

# ================================================================================
# CORE VALIDATION CLASSES
# ================================================================================

class ValidationScore:
    """Score container for validation results"""

    def __init__(self):
        self.overall_score = 0.0
        self.grade = "F"
        self.reliability = 0.0
        self.total_records = 0
        self.passed_records = 0
        self.flagged_records = 0
        self.dimension_scores = {}

# ================================================================================
# NEW DATA QUALITY VALIDATORS
# ================================================================================

def detect_duplicate_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect duplicate records based on primary key fields.
    For time series data: (product_name, region, metric, year) should be unique.
    Returns a DataFrame with duplicate record details.
    """
    if df.empty:
        return pd.DataFrame()

    # Define primary key columns for time series data
    key_columns = ['product_name', 'region', 'metric', 'year']
    available_keys = [col for col in key_columns if col in df.columns]

    if not available_keys:
        return pd.DataFrame()

    # Find all duplicated rows (including first occurrence)
    duplicates_mask = df.duplicated(subset=available_keys, keep=False)

    if duplicates_mask.any():
        duplicates = df[duplicates_mask].copy()

        # Group duplicates for better reporting
        duplicate_summary = duplicates.groupby(available_keys).agg({
            'value': 'count'  # Count occurrences
        }).rename(columns={'value': 'occurrences'}).reset_index()

        duplicate_summary = duplicate_summary[duplicate_summary['occurrences'] > 1]
        return duplicate_summary

    return pd.DataFrame()

def check_data_type_consistency(df: pd.DataFrame) -> Dict[str, List[int]]:
    """
    Check for data type mismatches in numeric columns.
    Returns dict of column names with list of row indices that have type issues.
    """
    issues = {}

    # Define columns that should be numeric
    numeric_columns = ['year', 'value', 'price', 'capacity', 'cumulative_capacity',
                      'quality_score', 'source_count']

    for col in numeric_columns:
        if col not in df.columns:
            continue

        # Try to convert to numeric and find failures
        original = df[col]
        converted = pd.to_numeric(df[col], errors='coerce')

        # Find indices where conversion failed (resulted in NaN but original wasn't NaN)
        failed_mask = pd.isna(converted) & pd.notna(original)
        failed_indices = df.index[failed_mask].tolist()

        if failed_indices:
            issues[col] = failed_indices
            logger.warning(f"Type consistency issue in column '{col}': {len(failed_indices)} non-numeric values")

    return issues

def detect_statistical_outliers(df: pd.DataFrame, method: str = "iqr", threshold: float = 1.5) -> Dict[str, List[Dict]]:
    """
    Detect statistical outliers using IQR or Z-score method.
    Groups by product/region for context-aware detection.
    Returns dict with outlier details per series.
    """
    outliers = {}

    if df.empty or 'value' not in df.columns:
        return outliers

    # Group by product/region for context-aware outlier detection
    grouping_cols = []
    if 'product_name' in df.columns:
        grouping_cols.append('product_name')
    if 'region' in df.columns:
        grouping_cols.append('region')

    if not grouping_cols:
        grouping_cols = ['product_name']  # Default fallback

    for group_key, group in df.groupby(grouping_cols, dropna=False):
        # Convert values to numeric, dropping non-numeric
        values = pd.to_numeric(group['value'], errors='coerce').dropna()

        if len(values) < 3:  # Need at least 3 points for outlier detection
            continue

        outlier_indices = []

        if method == "iqr":
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1

            if IQR > 0:  # Avoid division by zero
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                outlier_mask = (values < lower_bound) | (values > upper_bound)
                outlier_indices = values[outlier_mask].index.tolist()

        elif method == "zscore":
            mean = values.mean()
            std = values.std()

            if std > 0:  # Avoid division by zero
                z_scores = np.abs((values - mean) / std)
                outlier_mask = z_scores > threshold
                outlier_indices = values[outlier_mask].index.tolist()

        if outlier_indices:
            # Format group key for reporting
            if isinstance(group_key, tuple):
                key = "_".join(str(k) for k in group_key)
            else:
                key = str(group_key)

            outliers[key] = []

            for idx in outlier_indices:
                if idx in group.index:
                    row = group.loc[idx]
                    outliers[key].append({
                        'index': int(idx),
                        'year': int(row.get('year', 0)) if pd.notna(row.get('year')) else None,
                        'value': float(row.get('value', 0)),
                        'type': 'high' if row.get('value', 0) > values.median() else 'low',
                        'unit': row.get('unit', '')
                    })

    return outliers

def check_logical_consistency(df: pd.DataFrame) -> Dict[str, List[int]]:
    """
    Check for logical inconsistencies in the data.
    Includes: negative values where inappropriate, year ordering issues, etc.
    """
    issues = {}

    # Check for negative values in fields that should be non-negative
    non_negative_columns = ['value', 'price', 'capacity', 'cumulative_capacity',
                           'quality_score', 'source_count']

    for col in non_negative_columns:
        if col not in df.columns:
            continue

        # Convert to numeric and check for negatives
        numeric_col = pd.to_numeric(df[col], errors='coerce')
        negative_mask = numeric_col < 0

        if negative_mask.any():
            negative_indices = df.index[negative_mask].tolist()
            issues[f'negative_{col}'] = negative_indices
            logger.warning(f"Logical consistency issue: {len(negative_indices)} negative values in '{col}'")

    # Check year ordering if start/end years exist
    if 'start_year' in df.columns and 'end_year' in df.columns:
        start_years = pd.to_numeric(df['start_year'], errors='coerce')
        end_years = pd.to_numeric(df['end_year'], errors='coerce')

        invalid_order_mask = (start_years >= end_years) & pd.notna(start_years) & pd.notna(end_years)

        if invalid_order_mask.any():
            invalid_indices = df.index[invalid_order_mask].tolist()
            issues['invalid_year_order'] = invalid_indices
            logger.warning(f"Logical consistency issue: {len(invalid_indices)} records with end_year <= start_year")

    # Check for unrealistic year values
    if 'year' in df.columns:
        years = pd.to_numeric(df['year'], errors='coerce')
        unrealistic_mask = (years < 1900) | (years > 2100)

        if unrealistic_mask.any():
            unrealistic_indices = df.index[unrealistic_mask].tolist()
            issues['unrealistic_years'] = unrealistic_indices
            logger.warning(f"Logical consistency issue: {len(unrealistic_indices)} unrealistic year values")

    return issues

def calculate_detailed_completeness(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate detailed completeness metrics per column and overall.
    Distinguishes between critical and non-critical fields.
    """
    if df.empty:
        return {
            'overall_completeness': 0.0,
            'column_stats': {},
            'critical_fields_complete': False,
            'missing_critical_count': 0
        }

    # Calculate overall completeness
    total_cells = len(df) * len(df.columns)
    total_missing = df.isnull().sum().sum()
    overall_completeness = 1 - (total_missing / total_cells) if total_cells > 0 else 0

    # Define critical columns
    critical_columns = ['product_name', 'year', 'value']

    # Calculate per-column statistics
    column_stats = {}
    missing_critical_count = 0

    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100 if len(df) > 0 else 0

        is_critical = col in critical_columns

        column_stats[col] = {
            'missing_count': int(missing_count),
            'missing_percentage': round(missing_pct, 2),
            'is_critical': is_critical,
            'completeness': round(100 - missing_pct, 2)
        }

        if is_critical and missing_count > 0:
            missing_critical_count += missing_count

    # Check if all critical fields are complete
    critical_fields_complete = all(
        column_stats.get(col, {}).get('missing_count', 1) == 0
        for col in critical_columns if col in column_stats
    )

    return {
        'overall_completeness': round(overall_completeness, 3),
        'column_stats': column_stats,
        'critical_fields_complete': critical_fields_complete,
        'missing_critical_count': missing_critical_count,
        'total_records': len(df),
        'total_columns': len(df.columns)
    }

def check_format_consistency(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Check for format and unit consistency within each series.
    """
    issues = {}

    if df.empty:
        return issues

    # Check unit consistency within each product/region/metric combination
    if all(col in df.columns for col in ['product_name', 'region', 'metric', 'unit']):
        for (product, region, metric), group in df.groupby(['product_name', 'region', 'metric']):
            unique_units = group['unit'].dropna().unique()

            if len(unique_units) > 1:
                key = f"{product}_{region}_{metric}"
                issues[key] = list(unique_units)
                logger.warning(f"Unit inconsistency for {key}: {unique_units}")

    return issues

def validate_enhanced_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Master function to run all enhanced data quality checks.
    Returns comprehensive validation results.
    """
    validation_results = {}

    logger.info("Running enhanced data quality validation...")

    # 1. Completeness check
    validation_results['completeness'] = calculate_detailed_completeness(df)
    logger.info(f"Completeness: {validation_results['completeness']['overall_completeness']*100:.1f}%")

    # 2. Duplicate detection
    validation_results['duplicates'] = detect_duplicate_records(df)
    logger.info(f"Duplicates found: {len(validation_results['duplicates'])}")

    # 3. Type consistency
    validation_results['type_issues'] = check_data_type_consistency(df)
    logger.info(f"Type issues in {len(validation_results['type_issues'])} columns")

    # 4. Outlier detection
    validation_results['outliers'] = detect_statistical_outliers(df, method='iqr')
    outlier_count = sum(len(v) for v in validation_results['outliers'].values())
    logger.info(f"Outliers detected: {outlier_count}")

    # 5. Logical consistency
    validation_results['logic_issues'] = check_logical_consistency(df)
    logic_issue_count = sum(len(v) for v in validation_results['logic_issues'].values())
    logger.info(f"Logical issues found: {logic_issue_count}")

    # 6. Format consistency
    validation_results['format_issues'] = check_format_consistency(df)
    logger.info(f"Format inconsistencies: {len(validation_results['format_issues'])}")

    return validation_results

# ================================================================================
# STEP 1: DATA CONVERSION AND PREPROCESSING - ENHANCED
# ================================================================================

def convert_results_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert collected entity results to DataFrame format for STELLAR validation
    """
    if not results:
        return pd.DataFrame()

    validation_data = []

    for result in results:
        entity_name = result.get('Entity_Name', 'Unknown')
        region = result.get('Region', 'Unknown')
        unit = result.get('Unit', '')
        metric = result.get('Metric', '')

        X_data = result.get('X', [])  # Years
        Y_data = result.get('Y', [])  # Values

        for year, value in zip(X_data, Y_data):
            if value is None or not isinstance(value, (int, float)) or value <= 0:
                continue

            validation_data.append({
                'product_name': entity_name,
                'year': year,
                'value': float(value),
                'price': float(value) if '$' in unit else np.nan,
                'capacity': float(value) if 'GW' in unit or 'MW' in unit else np.nan,
                'region': region,
                'unit': unit,
                'metric': metric,
                'entity_type': result.get('Entity_Type', 'Technology'),
                'curve_type': result.get('Curve_Type', 'Adoption'),
                'is_cost_metric': '$' in unit or 'cost' in metric.lower(),
                'source_count': len(result.get('DataSource_URLs', [])),
                'quality_score': result.get('Quality_Score', 0),
            })

    df = pd.DataFrame(validation_data)

    if not df.empty:
        numeric_columns = ['year', 'value', 'price', 'capacity', 'quality_score', 'source_count']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

# ================================================================================
# STEP 2: DATA QUALITY FILTERS
# ================================================================================

def filter_unreasonable_data(df: pd.DataFrame, max_yoy_change: float = 3.0) -> Tuple[pd.DataFrame, List[str]]:
    """
    Filter out data points with unrealistic year-over-year changes
    """
    if df.empty:
        return df, []

    issues_found = []
    pieces = []

    if not {"product_name", "region", "year", "value"}.issubset(df.columns):
        return df.copy(), issues_found

    for (product, region), g in df.groupby(["product_name", "region"], dropna=False):
        g = g.sort_values("year").copy()
        vals = pd.to_numeric(g["value"], errors="coerce").values
        yrs = pd.to_numeric(g["year"], errors="coerce").values

        if len(g) <= 1:
            pieces.append(g)
            continue

        keep_mask = [True]
        for i in range(1, len(vals)):
            prev, curr = vals[i - 1], vals[i]
            if np.isfinite(prev) and np.isfinite(curr) and prev > 0:
                yoy = abs(curr - prev) / prev
                if yoy <= max_yoy_change:
                    keep_mask.append(True)
                else:
                    issues_found.append(
                        f"{product} - {region}: {yoy * 100:.0f}% jump from {int(yrs[i - 1])} to {int(yrs[i])}")
                    keep_mask.append(False)
            else:
                keep_mask.append(True)

        pieces.append(g.iloc[np.array(keep_mask, dtype=bool)])

    cleaned_df = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    return cleaned_df, issues_found

# ================================================================================
# STEP 3: WRIGHT'S LAW ANALYSIS
# ================================================================================
def analyze_wrights_law(df: pd.DataFrame, product: str) -> Dict[str, Any]:
    """
    Wright's Law analysis - ONLY for cost/price data, NOT generation/adoption
    Tony Seba Framework: Costs decline predictably with cumulative production
    """
    if product and "product_name" in df.columns:
        sub = df[df["product_name"] == product].copy()
    else:
        sub = df.copy()

    # ENHANCED: Check multiple fields for cost data
    is_cost_data = False

    # Check unit field
    if 'unit' in sub.columns and len(sub) > 0:
        unit_str = str(sub['unit'].iloc[0]).lower()
        # Extended list of cost indicators
        cost_indicators = ['$', 'usd', 'cost', 'price', '/mwh', '/kwh', 'lcoe',
                           'dollar', 'eur', '€', '£', 'gbp', 'cent', 'levelized']
        is_cost_data = any(indicator in unit_str for indicator in cost_indicators)

    # Check metric field
    if 'metric' in sub.columns and len(sub) > 0 and not is_cost_data:
        metric_str = str(sub['metric'].iloc[0]).lower()
        is_cost_data = any(term in metric_str for term in ['cost', 'price', 'lcoe', 'levelized', 'capex', 'opex'])

    # Check curve_type from database
    if 'curve_type' in sub.columns and len(sub) > 0 and not is_cost_data:
        curve_type_str = str(sub['curve_type'].iloc[0]).lower()
        is_cost_data = 'cost' in curve_type_str or 'wright' in curve_type_str
    # Skip Wright's Law for non-cost data - THIS IS KEY
    if not is_cost_data:
        return {
            "error": "Not applicable - Wright's Law only applies to cost data",
            "skipped": True,
            "reason": "Generation/adoption data should increase over time",
            "data_type": "generation/adoption"
        }

    if len(sub) < 5:
        return {"error": "Insufficient data points"}

    try:
        x = sub["year"].values
        y = sub["value"].values

        # For cost data, we expect DECLINING values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, np.log(y))
        r2 = r_value ** 2
        learning_rate = 1 - (2 ** slope) if slope < 0 else 0.0

        # Wright's Law compliance: negative slope (declining costs) with good fit
        # Technology-specific learning rates
        entity_lower = product.lower() if product else ''
        if 'solar' in entity_lower:
            min_rate, max_rate = 0.15, 0.30
        elif 'battery' in entity_lower or 'lithium' in entity_lower:
            min_rate, max_rate = 0.10, 0.25
        elif 'wind' in entity_lower:
            min_rate, max_rate = 0.05, 0.15
        else:
            min_rate, max_rate = 0.05, 0.40

        compliant = (slope < 0) and (r2 >= 0.60) and (min_rate <= learning_rate <= max_rate)

        return {
            "learning_rate": float(learning_rate),
            "r_squared": float(r2),
            "slope": float(slope),
            "compliant": bool(compliant),
            "data_points": int(len(sub)),
            "data_type": "cost"
        }
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}


def analyze_scurve_adoption(df: pd.DataFrame, product: str = None) -> Dict[str, Any]:
    """
    S-curve adoption analysis - for generation/adoption data
    Tony Seba Framework: Technology adoption follows S-curve pattern
    """
    if product and "product_name" in df.columns:
        plot_data = df[df["product_name"] == product].copy()
    else:
        plot_data = df.copy()

    if len(plot_data) < 4:
        return {"error": "Insufficient data points"}

    # Check data type
    is_generation_data = False
    if 'unit' in plot_data.columns and len(plot_data) > 0:
        unit_str = str(plot_data['unit'].iloc[0]).lower()
        is_generation_data = any(indicator in unit_str for indicator in ['gwh', 'twh', 'mwh', 'gw', 'mw', 'capacity'])

    if 'metric' in plot_data.columns and len(plot_data) > 0:
        metric_str = str(plot_data['metric'].iloc[0]).lower()
        is_generation_data = is_generation_data or any(
            indicator in metric_str for indicator in ['generation', 'capacity', 'adoption', 'sales'])

    try:
        x_data = plot_data['year'].values
        y_data = plot_data['value'].values

        # Normalize data for fitting
        x_norm = x_data - x_data.min()
        y_norm = (y_data - y_data.min()) / (y_data.max() - y_data.min() + 1e-10)

        def logistic(x, L, k, x0):
            return L / (1 + np.exp(-k * (x - x0)))

        # Initial parameters
        L_init = 1.0  # Normalized max
        k_init = 0.5
        x0_init = x_norm.mean()

        popt, _ = curve_fit(
            logistic, x_norm, y_norm,
            p0=[L_init, k_init, x0_init],
            bounds=([0.8, 0.01, 0], [1.2, 5, x_norm.max()]),
            maxfev=5000
        )

        y_pred = logistic(x_norm, *popt)
        ss_res = np.sum((y_norm - y_pred) ** 2)
        ss_tot = np.sum((y_norm - np.mean(y_norm)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # For generation data, good S-curve fit is positive
        compliant = r_squared > 0.7 and is_generation_data

        return {
            "r_squared": float(r_squared),
            "compliant": compliant,
            "data_points": len(plot_data),
            "data_type": "generation" if is_generation_data else "other",
            "growth_rate": float(popt[1]),
            "inflection_point": float(popt[2] + x_data.min())
        }
    except Exception as e:
        return {"error": f"S-curve fitting failed: {str(e)}"}

# ================================================================================
# STEP 4: S-CURVE ANALYSIS
# ================================================================================

def analyze_scurve_adoption(df: pd.DataFrame, product: str = None) -> Dict[str, Any]:
    """
    S-curve adoption analysis
    """
    if product and "product_name" in df.columns:
        plot_data = df[df["product_name"] == product].copy()
    else:
        plot_data = df.copy()

    if len(plot_data) < 4:
        return {"error": "Insufficient data points"}

    try:
        x_data = plot_data['year'].values
        y_data = plot_data['value'].values

        def logistic(x, L, k, x0):
            return L / (1 + np.exp(-k * (x - x0)))

        popt, _ = curve_fit(logistic, x_data, y_data, maxfev=2000)

        y_pred = logistic(x_data, *popt)
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        compliant = r_squared > 0.7

        return {
            "r_squared": float(r_squared),
            "compliant": compliant,
            "data_points": len(plot_data)
        }
    except:
        return {"error": "S-curve fitting failed"}

# ================================================================================
# STEP 5: BASIC VALIDATION FUNCTIONS (for compatibility)
# ================================================================================

def validate_data_completeness(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Original completeness validation - kept for compatibility
    """
    total_records = len(df)

    if total_records == 0:
        return {
            "completeness_score": 0.0,
            "missing_data_summary": "No data available",
            "total_records": 0
        }

    missing_summary = {}
    critical_columns = ['product_name', 'year', 'value']

    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / total_records) * 100
        missing_summary[col] = {
            "missing_count": missing_count,
            "missing_percentage": missing_pct,
            "is_critical": col in critical_columns
        }

    total_weight = 0
    weighted_missing = 0

    for col, info in missing_summary.items():
        weight = 2.0 if info["is_critical"] else 1.0
        total_weight += weight
        weighted_missing += info["missing_percentage"] * weight

    completeness_score = 1 - (weighted_missing / (100 * total_weight))

    return {
        "completeness_score": max(0, completeness_score),
        "missing_data_summary": missing_summary,
        "total_records": total_records
    }

def validate_domain_specific_rules(df: pd.DataFrame, rulepack: str = "energy_disruption") -> Dict[str, List[int]]:
    """
    Original domain-specific validation - kept for compatibility
    """
    validation_errors = {}

    if rulepack == "energy_disruption":
        if "value" in df.columns and "product_name" in df.columns:
            flagged_indices = []

            for idx, row in df.iterrows():
                product = str(row.get("product_name", "")).lower()
                value = row.get("value", None)
                unit = str(row.get("unit", "")).lower()

                if pd.notna(value) and value > 0:
                    if "solar" in product:
                        if "gwh" in unit and (value < 0 or value > 1000):
                            flagged_indices.append(idx)
                        elif "gw" in unit and (value < 0 or value > 500):
                            flagged_indices.append(idx)
                    elif "wind" in product:
                        if "gwh" in unit and (value < 0 or value > 2000):
                            flagged_indices.append(idx)

                    if value < 0:
                        flagged_indices.append(idx)

            if flagged_indices:
                validation_errors["value_range_violation"] = list(set(flagged_indices))

    return validation_errors

# ================================================================================
# STEP 6: CURVE ANALYSIS FUNCTIONS
# ================================================================================

def analyze_curve_missing_values(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze missing values for each curve
    """
    missing_analysis = {}

    for i, result in enumerate(results):
        entity_name = result.get('Entity_Name', f'Entity_{i}')
        region = result.get('Region', 'Unknown')
        curve_id = f"{entity_name}_{region}"

        X_data = result.get('X', [])
        Y_data = result.get('Y', [])

        if not X_data or not Y_data:
            continue

        total_points = len(X_data)
        missing_points = sum(1 for y in Y_data if y is None or (isinstance(y, (int, float)) and y == 0))

        missing_analysis[curve_id] = {
            'entity_name': entity_name,
            'region': region,
            'total_points': total_points,
            'missing_points': missing_points,
            'missing_percentage': (missing_points / total_points * 100) if total_points > 0 else 0
        }

    return missing_analysis

def detect_curve_outliers(results: List[Dict[str, Any]], method: str = "iqr") -> Dict[str, Any]:
    """
    Detect outliers for each curve
    """
    outlier_analysis = {}

    for i, result in enumerate(results):
        entity_name = result.get('Entity_Name', f'Entity_{i}')
        region = result.get('Region', 'Unknown')
        curve_id = f"{entity_name}_{region}"

        Y_data = result.get('Y', [])
        valid_data = [y for y in Y_data if y is not None and isinstance(y, (int, float)) and y > 0]

        if len(valid_data) < 3:
            outlier_analysis[curve_id] = {
                'entity_name': entity_name,
                'region': region,
                'outliers': [],
                'outlier_count': 0
            }
            continue

        values = pd.Series(valid_data)
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1

        outlier_count = 0
        if IQR > 0:
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_count = sum(1 for v in valid_data if v < lower_bound or v > upper_bound)

        outlier_analysis[curve_id] = {
            'entity_name': entity_name,
            'region': region,
            'outlier_count': outlier_count
        }

    return outlier_analysis

# ================================================================================
# ENHANCED VALIDATION PIPELINE
# ================================================================================

def run_validation_pipeline(results: List[Dict[str, Any]],
                           enable_seba: bool = True,
                           enable_llm: bool = False) -> Dict[str, Any]:
    """
    ENHANCED: Includes comprehensive data quality checks
    """

    # Convert to DataFrame
    df = convert_results_to_dataframe(results)

    if df.empty:
        return {
            'error': 'No valid data points found for validation',
            'clean_df': pd.DataFrame(),
            'flagged_df': pd.DataFrame(),
            'score': ValidationScore(),
            'report': 'No data available for validation',
            'seba_results': {},
            'validation_logs': ['No data points to validate'],
            'data_quality_issues': [],
            'enhanced_validation': {}
        }

    validation_logs = []
    validation_logs.append(f"✅ Converted {len(results)} entities to {len(df)} data points")

    try:
        # NEW: Run enhanced data quality validation
        validation_logs.append("🔍 Running comprehensive data quality checks...")
        enhanced_validation = validate_enhanced_data_quality(df)

        # Log summary of findings
        completeness = enhanced_validation['completeness']
        validation_logs.append(f"   - Completeness: {completeness['overall_completeness']*100:.1f}%")

        if not completeness['critical_fields_complete']:
            validation_logs.append(f"     ⚠️ Critical fields missing: {completeness['missing_critical_count']} records")

        duplicates = enhanced_validation['duplicates']
        if not duplicates.empty:
            validation_logs.append(f"   - Duplicates: {len(duplicates)} duplicate groups found")
        else:
            validation_logs.append("   - Duplicates: None found ✓")

        type_issues = enhanced_validation['type_issues']
        if type_issues:
            validation_logs.append(f"   - Type issues: {sum(len(v) for v in type_issues.values())} invalid values")
        else:
            validation_logs.append("   - Type consistency: All valid ✓")

        outliers = enhanced_validation['outliers']
        outlier_count = sum(len(v) for v in outliers.values())
        if outlier_count > 0:
            validation_logs.append(f"   - Outliers: {outlier_count} statistical outliers detected")
        else:
            validation_logs.append("   - Outliers: None detected ✓")

        logic_issues = enhanced_validation['logic_issues']
        if logic_issues:
            validation_logs.append(f"   - Logic issues: {sum(len(v) for v in logic_issues.values())} inconsistencies")
        else:
            validation_logs.append("   - Logical consistency: All valid ✓")

        # Filter unreasonable data
        validation_logs.append("🔍 Filtering unreasonable data jumps...")
        cleaned_df, data_quality_issues = filter_unreasonable_data(df, max_yoy_change=3.0)

        if data_quality_issues:
            validation_logs.append(f"   - Found {len(data_quality_issues)} data quality issues")
            for issue in data_quality_issues[:5]:
                validation_logs.append(f"     ⚠️ {issue}")
        else:
            validation_logs.append("   - No unreasonable data jumps detected")

        # Use cleaned data for validation
        df = cleaned_df

        # Step 1: Basic Validation
        validation_logs.append("📋 Step 1: Running basic validation...")
        completeness_results = validate_data_completeness(df)
        domain_violations = validate_domain_specific_rules(df, "energy_disruption")

        validation_logs.append(f"   - Completeness score: {completeness_results['completeness_score']:.1%}")
        validation_logs.append(f"   - Domain violations: {sum(len(v) for v in domain_violations.values())}")

        # Separate clean and flagged data
        flagged_indices = set()

        # Add indices from enhanced validation
        if not completeness['critical_fields_complete']:
            for col in ['product_name', 'year', 'value']:
                if col in df.columns:
                    flagged_indices.update(df[df[col].isnull()].index.tolist())

        # Add duplicate indices
        if not duplicates.empty:
            dup_keys = ['product_name', 'region', 'metric', 'year']
            available_keys = [col for col in dup_keys if col in df.columns]
            if available_keys:
                dup_mask = df.duplicated(subset=available_keys, keep=False)
                flagged_indices.update(df[dup_mask].index.tolist())

        # Add type issue indices
        for col, indices in type_issues.items():
            flagged_indices.update(indices)

        # Add logic issue indices
        for issue_type, indices in logic_issues.items():
            flagged_indices.update(indices)

        # Add existing domain violations
        for violation_list in domain_violations.values():
            flagged_indices.update(violation_list)

        clean_df = df[~df.index.isin(flagged_indices)].copy()
        flagged_df = df[df.index.isin(flagged_indices)].copy()

        validation_logs.append(f"   - Clean records: {len(clean_df)}")
        validation_logs.append(f"   - Flagged records: {len(flagged_df)}")

        # Step 2: Tony Seba Analysis
        seba_results = {}
        if enable_seba and not clean_df.empty:
            validation_logs.append("📈 Step 2: Running Tony Seba analysis...")

            products = clean_df['product_name'].unique() if 'product_name' in clean_df.columns else []

            for product in products[:5]:  # Analyze up to 5 products
                product_df = clean_df[clean_df['product_name'] == product]

                if len(product_df) >= 3:
                    # Wright's Law Analysis
                    wright_result = analyze_wrights_law(product_df, product)
                    if 'error' not in wright_result:
                        seba_results[f'{product}_wrights_law'] = wright_result
                        validation_logs.append(f"   - {product} Wright's Law: R²={wright_result.get('r_squared', 0):.3f}, Compliant={wright_result.get('compliant', False)}")

                    # S-Curve Analysis
                    scurve_result = analyze_scurve_adoption(product_df, product)
                    if 'error' not in scurve_result:
                        seba_results[f'{product}_scurve'] = scurve_result
                        validation_logs.append(f"   - {product} S-Curve: R²={scurve_result.get('r_squared', 0):.3f}, Compliant={scurve_result.get('compliant', False)}")

        # Step 3: Scoring and Grading
        validation_logs.append("🎯 Step 3: Calculating validation scores...")
        score = calculate_enhanced_validation_score(
            clean_df, flagged_df, seba_results,
            completeness_results, enhanced_validation, data_quality_issues
        )

        validation_logs.append(f"   - Overall Grade: {score.grade}")
        validation_logs.append(f"   - Overall Score: {score.overall_score:.1%}")

        # Step 4: Curve Analysis
        validation_logs.append("📊 Step 4: Analyzing individual curves...")
        missing_analysis = analyze_curve_missing_values(results)
        outlier_analysis = detect_curve_outliers(results, method="iqr")

        validation_logs.append(f"   - Analyzed {len(missing_analysis)} curves for missing values")
        validation_logs.append(f"   - Analyzed {len(outlier_analysis)} curves for outliers")

        # Step 5: Domain Expert Validation
        validation_logs.append("🔎 Step 5: Running domain expert validations...")

        expert_config = {
            'regional_params': {
                'usa': {'km_per_vehicle_per_year': 18000, 'liters_per_100km': 10.0},
                'china': {'km_per_vehicle_per_year': 12000, 'liters_per_100km': 7.0},
                'europe': {'km_per_vehicle_per_year': 13000, 'liters_per_100km': 6.5},
                'global': {'km_per_vehicle_per_year': 15000, 'liters_per_100km': 8.0},
            },
            'performance_thresholds': {
                'ev_range_km': 100,
                'charge_time_80pct_min': 40,
                'battery_cycles': 1000,
            },
            'transport_band': (55, 60),
            'total_band': (90, 110),
            'crisis_windows': [(2020, 2021)],
            'parity_level': 70,
            'cutoff_year': 2022,
            'material_pct': 0.10,
            'high_adoption': 0.70,
            'decline_years': 3,
            'consistency_threshold': 0.10,
            'conversion_tolerance': 0.05,
        }

        expert_validations = run_all_domain_expert_validators(
            results, clean_df, expert_config
        )

        # Log summary of expert validations
        expert_summary = format_expert_validation_summary(expert_validations)
        for line in expert_summary.split('\n'):
            validation_logs.append(f"   {line}")
            # Add commodity-specific validations if available
            if COMMODITY_VALIDATORS_AVAILABLE:
                validation_logs.append("🌾 Step 6: Running commodity-specific validations...")

                commodity_validations = {}

                try:
                    commodity_validations['agricultural'] = agricultural_seasonality_validator(clean_df)
                except Exception as e:
                    logger.error(f"Agricultural validator failed: {e}")

                try:
                    commodity_validations['metals'] = metal_supply_constraint_validator(clean_df)
                except Exception as e:
                    logger.error(f"Metal validator failed: {e}")

                try:
                    commodity_validations['carbon'] = carbon_credit_validator(clean_df)
                except Exception as e:
                    logger.error(f"Carbon validator failed: {e}")

                # Add to expert validations
                for key, vals in commodity_validations.items():
                    if vals:
                        expert_validations[f'commodity_{key}'] = vals
        # Step 6: Generate Report
        validation_logs.append("📄 Step 6: Generating validation report...")
        report = generate_enhanced_validation_report(
            score, seba_results, enhanced_validation,
            {'enabled': enable_llm}, data_quality_issues
        )

        validation_logs.append("✅ Validation pipeline completed successfully!")

        return {
            'clean_df': clean_df,
            'flagged_df': flagged_df,
            'score': score,
            'report': report,
            'seba_results': seba_results,
            'missing_analysis': missing_analysis,
            'outlier_analysis': outlier_analysis,
            'validation_logs': validation_logs,
            'completeness_results': completeness_results,
            'domain_violations': domain_violations,
            'entity_count': len(results),
            'data_points_count': len(df),
            'original_results': results,
            'data_quality_issues': data_quality_issues,
            'enhanced_validation': enhanced_validation,
            'expert_validations': expert_validations,
            'expert_summary': expert_summary,
        }

    except Exception as e:
        error_msg = f"Validation pipeline failed: {str(e)}"
        validation_logs.append(f"❌ {error_msg}")
        logger.error(error_msg, exc_info=True)

        return {
            'clean_df': pd.DataFrame(),
            'flagged_df': pd.DataFrame(),
            'score': ValidationScore(),
            'report': error_msg,
            'seba_results': {},
            'validation_logs': validation_logs,
            'error': error_msg,
            'enhanced_validation': {}
        }

def calculate_enhanced_validation_score(clean_df: pd.DataFrame, flagged_df: pd.DataFrame,
                                       seba_results: Dict[str, Any],
                                       validation_results: Dict[str, Any],
                                       enhanced_validation: Dict[str, Any],
                                       data_quality_issues: List[str] = None) -> ValidationScore:
    """
    ENHANCED: Incorporates all data quality dimensions in scoring
    """
    score = ValidationScore()

    total_records = len(clean_df) + len(flagged_df)
    score.total_records = total_records
    score.passed_records = len(clean_df)
    score.flagged_records = len(flagged_df)

    if total_records == 0:
        return score

    # Calculate dimension scores

    # 1. Completeness (from enhanced validation)
    enhanced_completeness = enhanced_validation.get('completeness', {})
    completeness = enhanced_completeness.get('overall_completeness', 0.0)
    score.dimension_scores["completeness"] = completeness

    # 2. Accuracy (penalize for flagged records and issues)
    accuracy_penalty = len(flagged_df) / total_records

    # Add penalties for various issues
    if enhanced_validation.get('duplicates') is not None and not enhanced_validation['duplicates'].empty:
        accuracy_penalty += 0.1  # 10% penalty for duplicates

    type_issue_count = sum(len(v) for v in enhanced_validation.get('type_issues', {}).values())
    if type_issue_count > 0:
        accuracy_penalty += min(0.1, type_issue_count / total_records)

    if data_quality_issues:
        accuracy_penalty += len(data_quality_issues) * 0.02  # 2% penalty per issue

    accuracy = max(0, 1 - accuracy_penalty)
    score.dimension_scores["accuracy"] = accuracy

    # 3. Consistency
    format_issues = enhanced_validation.get('format_issues', {})
    consistency = 1.0 if not format_issues else max(0.5, 1 - len(format_issues) * 0.05)
    score.dimension_scores["consistency"] = consistency

    # 4. Validity
    logic_issues = enhanced_validation.get('logic_issues', {})
    logic_issue_count = sum(len(v) for v in logic_issues.values())
    validity = 1.0 if logic_issue_count == 0 else max(0.5, 1 - logic_issue_count / total_records)
    score.dimension_scores["validity"] = validity

    # 5. Tony Seba dimensions - handle skipped analyses properly
    wright_applicable = sum(1 for k, v in seba_results.items()
                            if 'wrights_law' in k and not v.get("skipped", False))
    wright_compliant = sum(1 for k, v in seba_results.items()
                           if 'wrights_law' in k and v.get("compliant", False))

    if wright_applicable > 0:
        score.dimension_scores["wrights_law"] = wright_compliant / wright_applicable
    else:
        # No applicable Wright's Law data - neutral score
        score.dimension_scores["wrights_law"] = 0.7  # Don't penalize

    scurve_compliant = sum(1 for k, v in seba_results.items()
                           if 'scurve' in k and v.get("compliant", False))
    scurve_total = sum(1 for k in seba_results.keys() if 'scurve' in k)

    if scurve_total > 0:
        score.dimension_scores["scurve_adoption"] = scurve_compliant / scurve_total
    else:
        score.dimension_scores["scurve_adoption"] = 0.7  # Don't penalize if not applicable
    # Calculate weighted overall score
    weights = {
        "completeness": 0.25,
        "accuracy": 0.25,
        "consistency": 0.15,
        "validity": 0.15,
        "wrights_law": 0.10,
        "scurve_adoption": 0.10
        }

    # Only weight dimensions that were actually validated
    active_dims = {dim: weight for dim, weight in weights.items()
                    if dim in score.dimension_scores and score.dimension_scores[dim] is not None}

    if active_dims:
        total_weight = sum(active_dims.values())
        weighted_sum = sum(score.dimension_scores[dim] * weight
                               for dim, weight in active_dims.items())
    # Normalize by actual total weight
        score.overall_score = max(0, min(1, weighted_sum / total_weight if total_weight > 0 else 0))
    else:
        score.overall_score = 0.0

    # Grade assignment
    if score.overall_score >= 0.95:
        score.grade = "A+"
    elif score.overall_score >= 0.90:
        score.grade = "A"
    elif score.overall_score >= 0.85:
        score.grade = "A-"
    elif score.overall_score >= 0.80:
        score.grade = "B+"
    elif score.overall_score >= 0.75:
        score.grade = "B"
    elif score.overall_score >= 0.70:
        score.grade = "B-"
    elif score.overall_score >= 0.60:
        score.grade = "C"
    elif score.overall_score >= 0.50:
        score.grade = "D"
    else:
        score.grade = "F"

    score.reliability = score.overall_score

    return score


def generate_enhanced_validation_report(score: ValidationScore, seba_results: Dict[str, Any],
                                        enhanced_validation: Dict[str, Any],
                                        ai_results: Dict[str, Any],
                                        data_quality_issues: List[str] = None) -> str:
    """
    Generate comprehensive text validation report for STELLAR Framework
    """
    report_lines = [
        "=" * 80,
        "STELLAR FRAMEWORK - COMPREHENSIVE DATA VALIDATION REPORT",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "OVERALL ASSESSMENT",
        "-" * 40,
        f"Quality Grade: {score.grade}",
        f"Overall Score: {score.overall_score:.1%}",
        f"Data Reliability: {score.reliability:.1%}",
        "",
        "RECORD SUMMARY",
        "-" * 40,
        f"Total Records: {score.total_records:,}",
        f"Clean Records: {score.passed_records:,}",
        f"Flagged Records: {score.flagged_records:,}",
        f"Pass Rate: {(score.passed_records / score.total_records * 100 if score.total_records > 0 else 0):.1f}%",
        "",
        "DATA QUALITY VALIDATION RESULTS",
        "-" * 40,
    ]

    # 1. Completeness Results
    completeness = enhanced_validation.get('completeness', {})
    overall_completeness = completeness.get('overall_completeness', 0)
    critical_complete = completeness.get('critical_fields_complete', False)
    missing_critical = completeness.get('missing_critical_count', 0)

    report_lines.extend([
        "",
        "1. COMPLETENESS CHECK:",
        f"   Overall Completeness: {overall_completeness * 100:.1f}%",
        f"   Critical Fields Complete: {'✅ Yes' if critical_complete else '❌ No'}",
        f"   Missing Critical Count: {missing_critical}",
        f"   Status: {'PASS' if overall_completeness > 0.95 else 'WARNING' if overall_completeness > 0.8 else 'FAIL'}",
    ])

    # Add column-level completeness details
    column_stats = completeness.get('column_stats', {})
    if column_stats:
        report_lines.append("   Column Completeness:")
        for col, stats in list(column_stats.items())[:5]:
            if stats.get('is_critical'):
                report_lines.append(f"     - {col}: {stats.get('completeness', 0):.1f}% (CRITICAL)")

    # 2. Duplicate Records
    duplicates = enhanced_validation.get('duplicates', pd.DataFrame())
    has_duplicates = not duplicates.empty if hasattr(duplicates, 'empty') else False

    report_lines.extend([
        "",
        "2. DUPLICATE RECORDS:",
        f"   Status: {'❌ FAIL' if has_duplicates else '✅ PASS'}",
        f"   Duplicate Groups Found: {len(duplicates) if has_duplicates else 0}",
    ])

    if has_duplicates and hasattr(duplicates, 'head'):
        report_lines.append("   Sample Duplicates:")
        for idx, row in duplicates.head(3).iterrows():
            product = row.get('product_name', 'Unknown')
            region = row.get('region', 'Unknown')
            year = row.get('year', 'Unknown')
            occurrences = row.get('occurrences', 2)
            report_lines.append(f"     - {product} ({region}, Year {year}): {occurrences} duplicates")

    # 3. Data Type Consistency
    type_issues = enhanced_validation.get('type_issues', {})
    type_issue_count = sum(len(v) for v in type_issues.values())

    report_lines.extend([
        "",
        "3. DATA TYPE CONSISTENCY:",
        f"   Status: {'❌ FAIL' if type_issues else '✅ PASS'}",
        f"   Columns with Issues: {len(type_issues)}",
        f"   Total Invalid Entries: {type_issue_count}",
    ])

    if type_issues:
        for col, indices in list(type_issues.items())[:3]:
            report_lines.append(f"     - {col}: {len(indices)} non-numeric values")

    # 4. Statistical Outliers
    outliers = enhanced_validation.get('outliers', {})
    outlier_count = sum(len(v) for v in outliers.values())

    report_lines.extend([
        "",
        "4. STATISTICAL OUTLIERS:",
        f"   Status: {'⚠️ WARNING' if outlier_count > 0 else '✅ PASS'}",
        f"   Total Outliers: {outlier_count}",
        f"   Series with Outliers: {len(outliers)}",
    ])

    if outliers:
        for series, outlier_list in list(outliers.items())[:3]:
            report_lines.append(f"     - {series}: {len(outlier_list)} outliers detected")

    # 5. Logical Consistency
    logic_issues = enhanced_validation.get('logic_issues', {})
    logic_count = sum(len(v) for v in logic_issues.values())

    report_lines.extend([
        "",
        "5. LOGICAL CONSISTENCY:",
        f"   Status: {'❌ FAIL' if logic_issues else '✅ PASS'}",
        f"   Total Issues: {logic_count}",
    ])

    if logic_issues:
        for issue_type, indices in list(logic_issues.items())[:3]:
            issue_name = issue_type.replace('_', ' ').title()
            report_lines.append(f"     - {issue_name}: {len(indices)} issues")

    # 6. Format & Unit Consistency
    format_issues = enhanced_validation.get('format_issues', {})

    report_lines.extend([
        "",
        "6. FORMAT & UNIT CONSISTENCY:",
        f"   Status: {'❌ FAIL' if format_issues else '✅ PASS'}",
        f"   Inconsistent Series: {len(format_issues)}",
    ])

    if format_issues:
        for series, units in list(format_issues.items())[:3]:
            report_lines.append(f"     - {series}: Mixed units ({', '.join(units[:2])})")

    # Data Quality Issues Section
    if data_quality_issues:
        report_lines.extend([
            "",
            "DATA QUALITY ISSUES DETECTED",
            "-" * 40,
            f"Total Issues: {len(data_quality_issues)}"
        ])
        for issue in data_quality_issues[:10]:
            report_lines.append(f"  ⚠️ {issue}")
        if len(data_quality_issues) > 10:
            report_lines.append(f"  ... and {len(data_quality_issues) - 10} more issues")

    # DIMENSION SCORES SECTION
    report_lines.extend([
        "",
        "DIMENSION SCORES",
        "-" * 40
    ])

    # Ensure we have all dimension scores
    dimensions = {
        'completeness': score.dimension_scores.get('completeness', overall_completeness),
        'accuracy': score.dimension_scores.get('accuracy', 0),
        'consistency': score.dimension_scores.get('consistency', 0),
        'validity': score.dimension_scores.get('validity', 0),
        'wrights_law': score.dimension_scores.get('wrights_law', 0),
        'scurve_adoption': score.dimension_scores.get('scurve_adoption', 0)
    }

    for dimension, score_val in dimensions.items():
        status = "✅" if score_val >= 0.8 else "⚠️" if score_val >= 0.6 else "❌"
        dim_name = dimension.replace('_', ' ').title()
        report_lines.append(f"{status} {dim_name}: {score_val:.1%}")

    # TONY SEBA FRAMEWORK ANALYSIS
    report_lines.extend([
        "",
        "TONY SEBA FRAMEWORK ANALYSIS",
        "-" * 40
    ])

    # Wright's Law Results
    wright_products = []
    scurve_products = []

    for key, result in seba_results.items():
        if 'wrights_law' in key and 'error' not in result and not result.get('skipped'):
            product = key.replace('_wrights_law', '')
            wright_products.append((product, result))
        elif 'scurve' in key and 'error' not in result:
            product = key.replace('_scurve', '')
            scurve_products.append((product, result))

    if wright_products:
        report_lines.append("\nWright's Law Analysis (Cost Curves):")
        for product, result in wright_products:
            report_lines.append(f"\n  {product}:")
            report_lines.append(f"    Learning Rate: {result.get('learning_rate', 0):.1%}")
            report_lines.append(f"    R-squared: {result.get('r_squared', 0):.3f}")
            report_lines.append(f"    Data Points: {result.get('data_points', 0)}")
            report_lines.append(f"    Compliant: {'✅ Yes' if result.get('compliant', False) else '❌ No'}")
            if result.get('compliant'):
                report_lines.append(f"    Assessment: Cost declining as expected for technology")
    else:
        report_lines.append("\nWright's Law Analysis: No applicable cost curves found")

    if scurve_products:
        report_lines.append("\nS-Curve Analysis (Adoption Patterns):")
        for product, result in scurve_products:
            report_lines.append(f"\n  {product}:")
            report_lines.append(f"    R-squared: {result.get('r_squared', 0):.3f}")
            report_lines.append(f"    Growth Rate: {result.get('growth_rate', 0):.2f}")
            report_lines.append(f"    Data Points: {result.get('data_points', 0)}")
            report_lines.append(f"    Compliant: {'✅ Yes' if result.get('compliant', False) else '❌ No'}")
            if result.get('compliant'):
                report_lines.append(f"    Assessment: Following expected S-curve adoption pattern")
    else:
        report_lines.append("\nS-Curve Analysis: No adoption curves analyzed")

    # INVESTMENT GRADE ASSESSMENT
    report_lines.extend([
        "",
        "INVESTMENT GRADE ASSESSMENT",
        "-" * 40
    ])

    if score.overall_score >= 0.80:
        report_lines.extend([
            "✅ INVESTMENT-GRADE QUALITY ACHIEVED",
            "",
            "Data meets requirements for investment-grade analysis.",
            "The dataset has passed critical quality thresholds and can be used",
            "for strategic decision-making and financial modeling.",
            "",
            "Key Strengths:",
            f"  • High completeness rate ({overall_completeness * 100:.1f}%)",
            f"  • {score.passed_records} clean records validated",
            f"  • Consistent units and formats across series"
        ])
    elif score.overall_score >= 0.60:
        report_lines.extend([
            "⚠️ NEAR INVESTMENT-GRADE QUALITY",
            "",
            "Data quality approaching investment-grade but requires improvements.",
            "",
            "Required Actions:"
        ])
        if missing_critical > 0:
            report_lines.append(f"  • Fill {missing_critical} missing critical values")
        if has_duplicates:
            report_lines.append(f"  • Remove {len(duplicates)} duplicate records")
        if type_issue_count > 0:
            report_lines.append(f"  • Fix {type_issue_count} data type errors")
        if outlier_count > 10:
            report_lines.append(f"  • Review {outlier_count} statistical outliers")
        if data_quality_issues:
            report_lines.append(f"  • Address {len(data_quality_issues)} data quality issues")
    else:
        report_lines.extend([
            "❌ BELOW INVESTMENT-GRADE STANDARDS",
            "",
            "Significant data quality issues detected.",
            "Dataset requires substantial cleaning before analysis.",
            "",
            "Critical Issues to Address:",
            f"  • Overall quality score: {score.overall_score:.1%} (minimum 80% required)",
            f"  • Flagged records: {score.flagged_records} ({score.flagged_records / score.total_records * 100 if score.total_records > 0 else 0:.1f}%)",
            f"  • Data completeness: {overall_completeness * 100:.1f}% (minimum 95% required)"
        ])

    # RECOMMENDATIONS
    report_lines.extend([
        "",
        "RECOMMENDATIONS",
        "-" * 40
    ])

    recommendations = []

    if overall_completeness < 0.95:
        recommendations.append("1. Improve data completeness by filling missing values or finding additional sources")
    if has_duplicates:
        recommendations.append("2. Remove duplicate records to ensure data integrity")
    if type_issue_count > 0:
        recommendations.append("3. Standardize data types across all numeric columns")
    if outlier_count > 10:
        recommendations.append("4. Investigate and validate statistical outliers")
    if len(wright_products) < 3:
        recommendations.append("5. Add more cost curve data for Wright's Law analysis")
    if len(scurve_products) < 3:
        recommendations.append("6. Include more adoption curves for S-curve pattern validation")

    if recommendations:
        report_lines.extend(recommendations)
    else:
        report_lines.append("No critical recommendations - data quality meets standards")


    return "\n".join(report_lines)

def export_enhanced_validation_results(validation_results: Dict[str, Any],
                                       filename_base: str = "validation_results") -> Dict[str, str]:
    """
    Export enhanced validation results to multiple formats
    """
    exports = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    try:
        # 1. Export validation report (text)
        report = validation_results.get('report', 'No report available')
        report_filename = f"{filename_base}_report_{timestamp}.txt"

        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
            f.write('\n\n' + '=' * 80)
            f.write('\nVALIDATION LOGS:\n')
            f.write('=' * 80 + '\n')

            logs = validation_results.get('validation_logs', [])
            for log in logs:
                f.write(f"{log}\n")

        exports['report'] = report_filename

        # 2. Export validation summary (CSV)
        enhanced_validation = validation_results.get('enhanced_validation', {})

        # Create summary DataFrame
        summary_data = []

        # Completeness
        completeness = enhanced_validation.get('completeness', {})
        summary_data.append({
            'Validation Check': 'Missing Values',
            'Status': 'PASS' if completeness.get('critical_fields_complete', False) else 'FAIL',
            'Issues': completeness.get('missing_critical_count', 0),
            'Details': f"{completeness.get('overall_completeness', 0) * 100:.1f}% complete"
        })

        # Duplicates
        duplicates = enhanced_validation.get('duplicates', pd.DataFrame())
        has_duplicates = not duplicates.empty if hasattr(duplicates, 'empty') else False
        summary_data.append({
            'Validation Check': 'Duplicate Records',
            'Status': 'FAIL' if has_duplicates else 'PASS',
            'Issues': len(duplicates) if has_duplicates else 0,
            'Details': f"{len(duplicates)} duplicate groups" if has_duplicates else "No duplicates"
        })

        # Type consistency
        type_issues = enhanced_validation.get('type_issues', {})
        summary_data.append({
            'Validation Check': 'Data Type Consistency',
            'Status': 'FAIL' if type_issues else 'PASS',
            'Issues': sum(len(v) for v in type_issues.values()),
            'Details': f"{len(type_issues)} columns with issues" if type_issues else "All types valid"
        })

        # Outliers
        outliers = enhanced_validation.get('outliers', {})
        outlier_count = sum(len(v) for v in outliers.values())
        summary_data.append({
            'Validation Check': 'Statistical Outliers',
            'Status': 'WARNING' if outlier_count > 0 else 'PASS',
            'Issues': outlier_count,
            'Details': f"{outlier_count} outliers in {len(outliers)} series" if outlier_count else "No outliers"
        })

        # Logical consistency
        logic_issues = enhanced_validation.get('logic_issues', {})
        summary_data.append({
            'Validation Check': 'Logical Consistency',
            'Status': 'FAIL' if logic_issues else 'PASS',
            'Issues': sum(len(v) for v in logic_issues.values()),
            'Details': f"{len(logic_issues)} issue types" if logic_issues else "All logical"
        })

        # Format consistency
        format_issues = enhanced_validation.get('format_issues', {})
        summary_data.append({
            'Validation Check': 'Format & Unit Consistency',
            'Status': 'FAIL' if format_issues else 'PASS',
            'Issues': len(format_issues),
            'Details': f"{len(format_issues)} inconsistent series" if format_issues else "All consistent"
        })

        summary_df = pd.DataFrame(summary_data)
        summary_filename = f"{filename_base}_summary_{timestamp}.csv"
        summary_df.to_csv(summary_filename, index=False)
        exports['summary'] = summary_filename

        # 3. Export clean data
        clean_df = validation_results.get('clean_df', pd.DataFrame())
        if not clean_df.empty:
            clean_filename = f"{filename_base}_clean_data_{timestamp}.csv"
            clean_df.to_csv(clean_filename, index=False)
            exports['clean_data'] = clean_filename

        # 4. Export flagged data
        flagged_df = validation_results.get('flagged_df', pd.DataFrame())
        if not flagged_df.empty:
            flagged_filename = f"{filename_base}_flagged_data_{timestamp}.csv"
            flagged_df.to_csv(flagged_filename, index=False)
            exports['flagged_data'] = flagged_filename

        # 5. Export detailed validation results (JSON)
        detailed_results = {
            'timestamp': timestamp,
            'score': {
                'overall': validation_results.get('score').overall_score if validation_results.get('score') else 0,
                'grade': validation_results.get('score').grade if validation_results.get('score') else 'F',
                'dimensions': validation_results.get('score').dimension_scores if validation_results.get(
                    'score') else {}
            },
            'enhanced_validation': {
                'completeness': completeness,
                'duplicates_count': len(duplicates) if has_duplicates else 0,
                'type_issues_count': sum(len(v) for v in type_issues.values()),
                'outliers_count': outlier_count,
                'logic_issues_count': sum(len(v) for v in logic_issues.values()),
                'format_issues_count': len(format_issues)
            }
        }

        json_filename = f"{filename_base}_detailed_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, default=str)

        exports['detailed'] = json_filename

    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        exports['error'] = str(e)

    return exports

# ================================================================================
# HELPER FUNCTIONS FOR STREAMLIT INTEGRATION
# ================================================================================

def create_validation_charts(validation_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create validation charts for Streamlit display
    """
    charts = {}

    # Placeholder for chart creation
    # You can expand this based on your visualization needs

    return charts

def get_validation_summary(validation_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get validation summary for display
    """
    if not validation_results:
        return {}

    score = validation_results.get('score', ValidationScore())

    return {
        'total_records': score.total_records,
        'clean_records': score.passed_records,
        'flagged_records': score.flagged_records,
        'overall_score': score.overall_score,
        'grade': score.grade,
        'dimension_scores': score.dimension_scores
    }

def format_validation_logs_for_display(validation_logs: List[str]) -> str:
    """
    Format validation logs for display in Streamlit
    """
    if not validation_logs:
        return "No validation logs available"

    return "\n".join(validation_logs)

def export_validation_results(validation_results: Dict[str, Any],
                             filename_base: str = "validation_results") -> Dict[str, str]:
    """
    Wrapper function for export - calls the enhanced version
    """
    return export_enhanced_validation_results(validation_results, filename_base)