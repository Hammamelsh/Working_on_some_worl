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
from typing import Dict, List, Any, Union, Tuple, Optional
import logging
from scipy import stats
from scipy.optimize import curve_fit
import streamlit as st
import json
from pathlib import Path
from validation_constants import *

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

# ================================================================================
# STRUCTURED LOGGING CONFIGURATION
# ================================================================================

import logging
from logging.handlers import RotatingFileHandler
import os

def setup_stellar_logging():
    """
    Configure comprehensive structured logging for STELLAR framework.
    Logs to both file (rotating) and console with different levels.
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('STELLAR')
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # File handler (rotating, 10MB max, 5 backups)
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'stellar_validation.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    
    # Console handler (warnings and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    
    # Detailed formatter with context
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_stellar_logging()

# Log startup
logger.info("="*80)
logger.info("STELLAR Framework Validation Module Initialized")
logger.info(f"Version: 2.0.0-enhanced")
logger.info(f"Logging to: logs/stellar_validation.log")
logger.info("="*80)

from functools import lru_cache
import hashlib

# ================================================================================
# PERFORMANCE OPTIMIZATION - CACHING
# ================================================================================

def get_dataframe_hash(df: pd.DataFrame) -> str:
    """Generate hash of dataframe for cache key"""
    try:
        return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
    except:
        # Fallback if hashing fails
        return hashlib.md5(str(len(df)).encode()).hexdigest()


@lru_cache(maxsize=128)
def cached_wrights_law_analysis(df_hash: str, product: str, years_tuple: tuple, values_tuple: tuple) -> Dict:
    """
    Cached Wright's Law analysis - speeds up re-runs.
    Uses hash + tuples since DataFrame isn't hashable.
    """
    try:
        years = np.array(years_tuple)
        values = np.array(values_tuple)
        
        if len(values) < 5:
            return {"error": "Insufficient data points"}
        
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, np.log(values))
        r2 = r_value ** 2
        learning_rate = 1 - (2 ** slope) if slope < 0 else 0.0
        
        # Determine compliance
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
            "data_points": int(len(values)),
            "data_type": "cost",
            "cached": True
        }
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}


def optimize_dataframe_memory(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Optimize DataFrame memory usage by downcasting numeric types.
    
    Returns:
        Tuple of (optimized_df, optimization_report)
    """
    if df.empty:
        return df, {"error": "Empty DataFrame"}
    
    memory_before = df.memory_usage(deep=True).sum() / 1024**2  # MB
    
    optimized_df = df.copy()
    optimization_log = []
    
    for col in optimized_df.columns:
        col_type = optimized_df[col].dtype
        
        if col_type == 'float64':
            # Check if can downcast to float32
            if optimized_df[col].notna().any():
                max_val = optimized_df[col].max()
                min_val = optimized_df[col].min()
                
                # Float32 range: ~1.2e-38 to ~3.4e38
                if abs(max_val) < 3.4e38 and abs(min_val) < 3.4e38:
                    optimized_df[col] = optimized_df[col].astype('float32')
                    optimization_log.append(f"{col}: float64 → float32")
        
        elif col_type == 'int64':
            # Downcast integers
            if optimized_df[col].notna().any():
                max_val = optimized_df[col].max()
                min_val = optimized_df[col].min()
                
                if min_val >= 0:
                    if max_val < 255:
                        optimized_df[col] = optimized_df[col].astype('uint8')
                        optimization_log.append(f"{col}: int64 → uint8")
                    elif max_val < 65535:
                        optimized_df[col] = optimized_df[col].astype('uint16')
                        optimization_log.append(f"{col}: int64 → uint16")
                    elif max_val < 4294967295:
                        optimized_df[col] = optimized_df[col].astype('uint32')
                        optimization_log.append(f"{col}: int64 → uint32")
                else:
                    if -128 <= min_val and max_val < 127:
                        optimized_df[col] = optimized_df[col].astype('int8')
                        optimization_log.append(f"{col}: int64 → int8")
                    elif -32768 <= min_val and max_val < 32767:
                        optimized_df[col] = optimized_df[col].astype('int16')
                        optimization_log.append(f"{col}: int64 → int16")
                    elif -2147483648 <= min_val and max_val < 2147483647:
                        optimized_df[col] = optimized_df[col].astype('int32')
                        optimization_log.append(f"{col}: int64 → int32")
        
        elif col_type == 'object':
            # Convert to categorical if few unique values
            n_unique = optimized_df[col].nunique()
            n_total = len(optimized_df[col])
            
            if n_unique / n_total < 0.5 and n_unique < 100:
                optimized_df[col] = optimized_df[col].astype('category')
                optimization_log.append(f"{col}: object → category")
    
    memory_after = optimized_df.memory_usage(deep=True).sum() / 1024**2  # MB
    reduction = (memory_before - memory_after) / memory_before * 100 if memory_before > 0 else 0
    
    return optimized_df, {
        "memory_before_mb": memory_before,
        "memory_after_mb": memory_after,
        "reduction_percent": reduction,
        "optimizations": optimization_log,
        "interpretation": f"Reduced memory usage by {reduction:.1f}% ({memory_before:.2f}MB → {memory_after:.2f}MB)"
    }
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
        cost_indicators = ['$', 'usd', 'cost', 'price', '/mwh', '/kwh', 'lcoe',
                          'dollar', 'eur', '€', '£', 'gbp', 'cent', 'levelized']
        is_cost_data = any(indicator in unit_str for indicator in cost_indicators)

    # Check metric field if unit check inconclusive
    if not is_cost_data and 'metric' in sub.columns and len(sub) > 0:
        metric_str = str(sub['metric'].iloc[0]).lower()
        is_cost_data = any(term in metric_str for term in ['cost', 'price', 'lcoe', 'levelized', 'capex', 'opex'])

    # Check curve_type from database
    if not is_cost_data and 'curve_type' in sub.columns and len(sub) > 0:
        curve_type_str = str(sub['curve_type'].iloc[0]).lower()
        is_cost_data = 'cost' in curve_type_str or 'wright' in curve_type_str or 'price' in curve_type_str
    logger.debug(f"Skipping Wright's Law for {product} - not cost data")
    if not is_cost_data:
        return {
            "error": "Not applicable - Wright's Law only applies to cost data",
            "skipped": True,
            "reason": "Generation/adoption data should increase over time",
            "data_type": "generation/adoption"
        }

    if len(sub) < 5:
        return {"error": "Insufficient data points"}

    # TRY CACHE FIRST
    try:
        years = sub["year"].values
        values = sub["value"].values
        
        # Generate hash for caching
        years_tuple = tuple(years)
        values_tuple = tuple(values)
        df_hash = get_dataframe_hash(sub)
        
        # Try cached version
        cached_result = cached_wrights_law_analysis(df_hash, product, years_tuple, values_tuple)
        if cached_result and 'error' not in cached_result:
            logger.info(f"Using cached Wright's Law result for {product}")
            return cached_result
    except:
        pass  # Fall through to normal calculation

    # NORMAL CALCULATION if cache fails or unavailable
    try:
        x = sub["year"].values
        y = sub["value"].values

        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, np.log(y))
        r2 = r_value ** 2
        learning_rate = 1 - (2 ** slope) if slope < 0 else 0.0

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
# STATISTICAL RIGOR ENHANCEMENTS
# ================================================================================

def calculate_wrights_law_confidence_intervals(df: pd.DataFrame, product: str,
                                              confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    Calculate confidence intervals for Wright's Law learning rate using bootstrap
    """
    from scipy import stats
    import numpy as np
    
    if product and "product_name" in df.columns:
        sub = df[df["product_name"] == product].copy()
    else:
        sub = df.copy()
    
    # Check if cost data
    is_cost_data = False
    if 'unit' in sub.columns and len(sub) > 0:
        unit_str = str(sub['unit'].iloc[0]).lower()
        cost_indicators = ['$', 'usd', 'cost', 'price', '/mwh', '/kwh', 'lcoe']
        is_cost_data = any(indicator in unit_str for indicator in cost_indicators)
    
    if not is_cost_data or len(sub) < 5:
        return {"error": "Insufficient cost data or not applicable"}
    
    try:
        x = sub["year"].values
        y = sub["value"].values
        
        # Original fit
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, np.log(y))
        learning_rate = 1 - (2 ** slope) if slope < 0 else 0.0
        
        # Bootstrap confidence intervals
        n_bootstrap = 1000
        bootstrap_rates = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(x), size=len(x), replace=True)
            x_boot = x[indices]
            y_boot = y[indices]
            
            try:
                slope_boot, _, _, _, _ = stats.linregress(x_boot, np.log(y_boot))
                rate_boot = 1 - (2 ** slope_boot) if slope_boot < 0 else 0.0
                bootstrap_rates.append(rate_boot)
            except:
                continue
        
        if bootstrap_rates:
            # Calculate confidence intervals
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            ci_lower = np.percentile(bootstrap_rates, lower_percentile)
            ci_upper = np.percentile(bootstrap_rates, upper_percentile)
            
            # Statistical significance test
            # H0: learning rate = 0 (no cost reduction)
            # H1: learning rate > 0 (cost declining)
            t_stat = learning_rate / (std_err / (2 ** slope * np.log(2))) if slope < 0 else 0
            p_value_one_tailed = 1 - stats.t.cdf(abs(t_stat), df=len(x)-2)
            
            return {
                "learning_rate": float(learning_rate),
                "confidence_interval": {
                    "lower": float(ci_lower),
                    "upper": float(ci_upper),
                    "confidence_level": confidence_level
                },
                "r_squared": float(r_value ** 2),
                "p_value": float(p_value),
                "p_value_significance_test": float(p_value_one_tailed),
                "statistically_significant": p_value_one_tailed < 0.05,
                "n_samples": len(x),
                "standard_error": float(std_err),
                "interpretation": (
                    f"Learning rate: {learning_rate*100:.1f}% per doubling "
                    f"(95% CI: [{ci_lower*100:.1f}%, {ci_upper*100:.1f}%]). "
                    f"{'Statistically significant' if p_value_one_tailed < 0.05 else 'Not statistically significant'} "
                    f"cost decline (p={p_value_one_tailed:.4f})."
                )
            }
        else:
            return {"error": "Bootstrap failed"}
            
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}


def detect_statistical_outliers_advanced(df: pd.DataFrame, method: str = "grubbs") -> Dict[str, Any]:
    """
    Advanced outlier detection with hypothesis testing (Grubbs, Dixon, etc.)
    """
    from scipy import stats
    import numpy as np
    
    outliers = {}
    
    if df.empty or 'value' not in df.columns:
        return outliers
    
    grouping_cols = []
    if 'product_name' in df.columns:
        grouping_cols.append('product_name')
    if 'region' in df.columns:
        grouping_cols.append('region')
    
    if not grouping_cols:
        return outliers
    
    for group_key, group in df.groupby(grouping_cols, dropna=False):
        values = pd.to_numeric(group['value'], errors='coerce').dropna()
        
        if len(values) < 3:
            continue
        
        if method == "grubbs":
            # Grubbs test for outliers (assumes normal distribution)
            mean = values.mean()
            std = values.std()
            
            if std == 0:
                continue
            
            # Calculate Grubbs statistic for each point
            grubbs_stats = np.abs((values - mean) / std)
            max_grubbs = grubbs_stats.max()
            
            # Critical value for Grubbs test
            n = len(values)
            alpha = 0.05
            t_critical = stats.t.ppf(1 - alpha / (2 * n), n - 2)
            grubbs_critical = ((n - 1) / np.sqrt(n)) * np.sqrt(t_critical ** 2 / (n - 2 + t_critical ** 2))
            
            if max_grubbs > grubbs_critical:
                # Outlier detected
                outlier_idx = grubbs_stats.idxmax()
                
                if isinstance(group_key, tuple):
                    key = "_".join(str(k) for k in group_key)
                else:
                    key = str(group_key)
                
                outliers[key] = [{
                    'index': int(outlier_idx),
                    'value': float(values.loc[outlier_idx]),
                    'grubbs_statistic': float(max_grubbs),
                    'critical_value': float(grubbs_critical),
                    'p_value': f"< {alpha}",
                    'test': 'Grubbs',
                    'significant': True
                }]
    
    return outliers


def calculate_statistical_power(df: pd.DataFrame, product: str, effect_size: float = 0.5) -> Dict[str, Any]:
    """
    Calculate statistical power for Wright's Law analysis
    """
    from scipy import stats
    
    if product and "product_name" in df.columns:
        sub = df[df["product_name"] == product].copy()
    else:
        sub = df.copy()
    
    n = len(sub)
    
    if n < 3:
        return {"error": "Insufficient data"}
    
    # Calculate power for detecting correlation
    # Effect size: small=0.1, medium=0.3, large=0.5
    alpha = 0.05
    
    # Using correlation power analysis
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = 0.84  # 80% power
    
    # Fisher z-transformation
    z_r = 0.5 * np.log((1 + effect_size) / (1 - effect_size))
    
    # Required sample size for given power
    n_required = int(((z_alpha + z_beta) / z_r) ** 2 + 3)
    
    # Actual power with current sample size
    z_stat = z_r * np.sqrt(n - 3)
    actual_power = 1 - stats.norm.cdf(z_alpha - z_stat)
    
    return {
        "sample_size": n,
        "required_sample_size": n_required,
        "statistical_power": float(actual_power),
        "effect_size": effect_size,
        "sufficient_power": actual_power >= 0.80,
        "interpretation": (
            f"Current sample size (n={n}) provides {actual_power*100:.1f}% power "
            f"to detect medium-sized effects. "
            f"{'Sufficient' if actual_power >= 0.80 else f'Need {n_required-n} more data points'} "
            f"for 80% statistical power."
        )
    }

# ================================================================================
# TONY SEBA FRAMEWORK ENHANCEMENTS - TIPPING POINT DETECTION
# ================================================================================

def detect_tipping_point(df: pd.DataFrame, product: str) -> Dict[str, Any]:
    """
    Detect if technology has crossed critical tipping point (10% adoption).
    Tony Seba: After 10% adoption, technologies typically reach 90% in <10 years.
    """
    if product and "product_name" in df.columns:
        sub = df[df["product_name"] == product].copy()
    else:
        sub = df.copy()
    
    # Check if adoption/deployment data
    is_adoption = False
    if 'metric' in sub.columns and len(sub) > 0:
        metric_str = str(sub['metric'].iloc[0]).lower()
        is_adoption = any(term in metric_str for term in ['adoption', 'sales', 'penetration', 'share', 'fleet'])
    
    if not is_adoption or len(sub) < 3:
        return {"error": "Not applicable - requires adoption/deployment data"}
    
    try:
        # Sort by year
        sub = sub.sort_values('year').copy()
        years = sub['year'].values
        values = sub['value'].values
        
        # Find if crossed 10% threshold
        tipping_point_year = None
        tipping_point_value = None
        
        for i, (year, value) in enumerate(zip(years, values)):
            # Assume values are percentages or market shares
            # Normalize if needed
            if value > 1.0:  # If in percentage form (e.g., 10 not 0.10)
                value = value / 100.0
            
            if value >= 0.10 and tipping_point_year is None:
                tipping_point_year = int(year)
                tipping_point_value = float(value)
                break
        
        if tipping_point_year:
            logger.info(f"Tipping point detected for {product} in {tipping_point_year}")
            # Calculate years to 90%
            reached_90 = False
            years_to_90 = None
            
            for year, value in zip(years, values):
                if value > 1.0:
                    value = value / 100.0
                
                if value >= 0.90:
                    years_to_90 = int(year) - tipping_point_year
                    reached_90 = True
                    break
            
            # Project if not yet reached 90%
            if not reached_90 and len(years) >= 3:
                # Calculate growth rate after tipping point
                post_tipping = sub[sub['year'] >= tipping_point_year]
                if len(post_tipping) >= 2:
                    growth_values = post_tipping['value'].values
                    growth_years = post_tipping['year'].values
                    
                    if len(growth_values) >= 2:
                        annual_growth = np.mean([
                            (growth_values[i] - growth_values[i-1]) / growth_values[i-1]
                            for i in range(1, len(growth_values))
                            if growth_values[i-1] > 0
                        ])
                        
                        # Project years to 90%
                        current_value = values[-1]
                        if current_value > 1.0:
                            current_value = current_value / 100.0
                        
                        if annual_growth > 0 and current_value < 0.90:
                            years_needed = np.log(0.90 / current_value) / np.log(1 + annual_growth)
                            years_to_90 = int(np.ceil(years_needed))
            
            # Seba's prediction: 10% → 90% in <10 years
            meets_seba_pattern = years_to_90 is not None and years_to_90 <= 10
            
            return {
                "tipping_point_crossed": True,
                "tipping_point_year": tipping_point_year,
                "tipping_point_value": tipping_point_value,
                "years_to_90_percent": years_to_90,
                "reached_90_percent": reached_90,
                "meets_seba_pattern": meets_seba_pattern,
                "interpretation": (
                    f"Tipping point (10% adoption) crossed in {tipping_point_year}. "
                    f"{'Reached 90% in ' + str(years_to_90) + ' years - ' if reached_90 else 'Projected to reach 90% in ' + str(years_to_90) + ' years - ' if years_to_90 else ''}"
                    f"{'MEETS' if meets_seba_pattern else 'DOES NOT MEET'} Seba's <10 year pattern."
                )
            }
        else:
            # Not yet at tipping point
            current_value = values[-1]
            if current_value > 1.0:
                current_value = current_value / 100.0
            
            return {
                "tipping_point_crossed": False,
                "current_adoption": current_value,
                "current_year": int(years[-1]),
                "interpretation": f"Not yet reached tipping point (10% adoption). Current: {current_value*100:.1f}%"
            }
    
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}


def analyze_convergence(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze if multiple S-curves are converging (multi-technology disruption).
    Tony Seba: When multiple disruptive technologies converge, disruption accelerates.
    """
    if df.empty or 'product_name' not in df.columns:
        return {"error": "Insufficient data"}
    
    products = df['product_name'].unique()
    
    if len(products) < 2:
        return {"error": "Need multiple products for convergence analysis"}
    
    try:
        convergence_data = []
        
        # Get latest year data for each product
        latest_year = df['year'].max()
        recent_data = df[df['year'] >= latest_year - 3]  # Last 3 years
        
        for product in products:
            product_data = recent_data[recent_data['product_name'] == product]
            
            if not product_data.empty:
                avg_value = product_data['value'].mean()
                growth = None
                
                # Calculate growth rate
                product_historical = df[df['product_name'] == product].sort_values('year')
                if len(product_historical) >= 2:
                    first_val = product_historical['value'].iloc[0]
                    last_val = product_historical['value'].iloc[-1]
                    years_span = product_historical['year'].iloc[-1] - product_historical['year'].iloc[0]
                    
                    if first_val > 0 and years_span > 0:
                        growth = ((last_val / first_val) ** (1 / years_span)) - 1
                
                convergence_data.append({
                    'product': product,
                    'recent_value': avg_value,
                    'growth_rate': growth
                })
        
        # Check for convergence patterns
        high_growth_products = [
            p for p in convergence_data 
            if p['growth_rate'] and p['growth_rate'] > 0.20  # >20% annual growth
        ]
        
        is_converging = len(high_growth_products) >= 2
        
        return {
            "converging_technologies": is_converging,
            "high_growth_count": len(high_growth_products),
            "technologies": convergence_data,
            "interpretation": (
                f"{'CONVERGENCE DETECTED' if is_converging else 'No convergence'}: "
                f"{len(high_growth_products)} technologies with >20% growth. "
                f"{'Multiple S-curves accelerating disruption' if is_converging else 'Single technology trajectory'}"
            )
        }
    
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}


def track_cost_parity_forecast(df: pd.DataFrame, product: str, 
                               parity_threshold: float = 70.0) -> Dict[str, Any]:
    """
    Track when technology costs will reach parity with incumbents.
    Critical for disruption timing predictions.
    """
    if product and "product_name" in df.columns:
        sub = df[df["product_name"] == product].copy()
    else:
        sub = df.copy()
    
    # Check if cost data
    is_cost = False
    if 'unit' in sub.columns and len(sub) > 0:
        unit_str = str(sub['unit'].iloc[0]).lower()
        is_cost = any(ind in unit_str for ind in ['$', 'cost', 'price', '/mwh', '/kwh'])
    
    if not is_cost or len(sub) < 3:
        return {"error": "Not applicable - requires cost data"}
    
    try:
        sub = sub.sort_values('year').copy()
        years = sub['year'].values
        costs = sub['value'].values
        
        # Check if already at parity
        at_parity = False
        parity_year = None
        
        for year, cost in zip(years, costs):
            if cost <= parity_threshold:
                at_parity = True
                parity_year = int(year)
                break
        
        if at_parity:
            return {
                "at_parity": True,
                "parity_year": parity_year,
                "current_cost": float(costs[-1]),
                "parity_threshold": parity_threshold,
                "interpretation": f"Cost parity achieved in {parity_year}. Current: ${costs[-1]:.2f}/MWh"
            }
        else:
            # Forecast parity
            if len(years) >= 3:
                # Fit exponential decline
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(years, np.log(costs))
                
                # Project to parity
                target_log = np.log(parity_threshold)
                parity_year_forecast = (target_log - intercept) / slope
                
                years_until_parity = int(np.ceil(parity_year_forecast - years[-1]))
                
                return {
                    "at_parity": False,
                    "current_cost": float(costs[-1]),
                    "parity_threshold": parity_threshold,
                    "forecast_parity_year": int(parity_year_forecast),
                    "years_until_parity": years_until_parity,
                    "forecast_confidence": float(r_value ** 2),
                    "interpretation": (
                        f"Forecast: Cost parity (${parity_threshold:.0f}/MWh) in ~{int(parity_year_forecast)} "
                        f"({years_until_parity} years from now). "
                        f"Current: ${costs[-1]:.2f}/MWh. Confidence: {r_value**2:.1%}"
                    )
                }
        
    except Exception as e:
        return {"error": f"Forecast failed: {str(e)}"}

# ================================================================================
# TIME SERIES GAP HANDLING & INTERPOLATION
# ================================================================================

def detect_and_handle_time_gaps(df: pd.DataFrame,
                                max_gap_years: int = 3,
                                interpolation_method: str = "linear") -> Dict[str, Any]:
    """
    Detect gaps in time series and optionally interpolate missing years.
    
    Args:
        df: DataFrame with time series data
        max_gap_years: Maximum gap size to interpolate (larger gaps flagged only)
        interpolation_method: 'linear', 'polynomial', 'spline', or 'forward'
    
    Returns:
        Dict with gap analysis and optionally interpolated data
    """
    if df.empty or 'year' not in df.columns or 'product_name' not in df.columns:
        return {"error": "Invalid DataFrame structure"}
    
    gap_analysis = []
    interpolated_data = []
    
    for (product, region), group in df.groupby(['product_name', 'region']):
        # Sort by year
        group_sorted = group.sort_values('year').copy()
        years = group_sorted['year'].values
        
        if len(years) < 2:
            continue
        
        # Find gaps
        gaps = []
        for i in range(1, len(years)):
            gap_size = years[i] - years[i-1] - 1
            if gap_size > 0:
                gaps.append({
                    'product': product,
                    'region': region,
                    'start_year': int(years[i-1]),
                    'end_year': int(years[i]),
                    'gap_years': int(gap_size),
                    'can_interpolate': gap_size <= max_gap_years
                })
        
        if gaps:
            gap_analysis.extend(gaps)
            
            # Interpolate if requested and gap is small enough
            for gap in gaps:
                if gap['can_interpolate'] and interpolation_method != "none":
                    # Get surrounding values
                    start_val = group_sorted[group_sorted['year'] == gap['start_year']]['value'].iloc[0]
                    end_val = group_sorted[group_sorted['year'] == gap['end_year']]['value'].iloc[0]
                    
                    # Generate interpolated years
                    missing_years = range(gap['start_year'] + 1, gap['end_year'])
                    
                    if interpolation_method == "linear":
                        # Linear interpolation
                        step = (end_val - start_val) / (gap['gap_years'] + 1)
                        interpolated_vals = [start_val + step * (j + 1) for j in range(gap['gap_years'])]
                    
                    elif interpolation_method == "forward":
                        # Forward fill
                        interpolated_vals = [start_val] * gap['gap_years']
                    
                    elif interpolation_method == "spline":
                        # Cubic spline (requires more surrounding points)
                        if len(group_sorted) >= 4:
                            from scipy.interpolate import CubicSpline
                            cs = CubicSpline(group_sorted['year'].values, group_sorted['value'].values)
                            interpolated_vals = [float(cs(year)) for year in missing_years]
                        else:
                            # Fallback to linear
                            step = (end_val - start_val) / (gap['gap_years'] + 1)
                            interpolated_vals = [start_val + step * (j + 1) for j in range(gap['gap_years'])]
                    
                    else:
                        # Default to linear
                        step = (end_val - start_val) / (gap['gap_years'] + 1)
                        interpolated_vals = [start_val + step * (j + 1) for j in range(gap['gap_years'])]
                    
                    # Store interpolated data
                    for year, val in zip(missing_years, interpolated_vals):
                        interpolated_data.append({
                            'product_name': product,
                            'region': region,
                            'year': year,
                            'value': val,
                            'interpolated': True,
                            'method': interpolation_method
                        })
    
    return {
        "gaps_found": len(gap_analysis),
        "gaps_detail": gap_analysis,
        "interpolated_points": len(interpolated_data),
        "interpolated_data": interpolated_data,
        "method_used": interpolation_method,
        "interpretation": (
            f"Found {len(gap_analysis)} gaps in time series. "
            f"{len([g for g in gap_analysis if g['can_interpolate']])} gaps â‰¤{max_gap_years} years (interpolatable). "
            f"{len([g for g in gap_analysis if not g['can_interpolate']])} gaps >{max_gap_years} years (too large for interpolation). "
            f"Generated {len(interpolated_data)} interpolated data points using {interpolation_method} method."
        )
    }


def harmonize_mixed_frequencies(df: pd.DataFrame,
                                target_frequency: str = "annual") -> Dict[str, Any]:
    """
    Harmonize mixed frequency data (monthly, quarterly, annual) to common frequency.
    
    Args:
        df: DataFrame with mixed frequency data
        target_frequency: 'annual', 'quarterly', or 'monthly'
    
    Returns:
        Dict with harmonized DataFrame and conversion log
    """
    if df.empty or 'year' not in df.columns:
        return {"error": "Invalid DataFrame structure"}
    
    harmonized_rows = []
    conversion_log = []
    
    # Detect frequency for each product/region series
    for (product, region), group in df.groupby(['product_name', 'region']):
        years = group['year'].values
        
        # Detect frequency based on year spacing
        if len(years) < 2:
            continue
        
        year_diffs = np.diff(sorted(years))
        avg_diff = np.mean(year_diffs)
        
        # Classify frequency
        if avg_diff < 0.1:  # Monthly or more frequent
            detected_freq = "monthly"
            periods_per_year = 12
        elif 0.2 < avg_diff < 0.3:  # Quarterly
            detected_freq = "quarterly"
            periods_per_year = 4
        else:  # Annual or less frequent
            detected_freq = "annual"
            periods_per_year = 1
        
        # If already at target frequency, keep as-is
        if detected_freq == target_frequency:
            harmonized_rows.extend(group.to_dict('records'))
            continue
        
        # Convert to target frequency
        if target_frequency == "annual":
            # Aggregate to annual
            if detected_freq in ["monthly", "quarterly"]:
                # Group by year and aggregate
                group_annual = group.copy()
                group_annual['year'] = group_annual['year'].astype(int)
                
                # Determine aggregation method based on metric
                metric = str(group['metric'].iloc[0]).lower()
                
                if any(term in metric for term in ['price', 'cost', 'lcoe', 'average']):
                    # For prices/costs, use mean
                    agg_method = 'mean'
                elif any(term in metric for term in ['capacity', 'fleet', 'stock']):
                    # For stocks, use end-of-period
                    agg_method = 'last'
                else:
                    # For flows (generation, production), use sum
                    agg_method = 'sum'
                
                if agg_method == 'mean':
                    aggregated = group_annual.groupby('year').agg({
                        'value': 'mean',
                        'product_name': 'first',
                        'region': 'first',
                        'unit': 'first',
                        'metric': 'first'
                    }).reset_index()
                elif agg_method == 'last':
                    aggregated = group_annual.groupby('year').agg({
                        'value': 'last',
                        'product_name': 'first',
                        'region': 'first',
                        'unit': 'first',
                        'metric': 'first'
                    }).reset_index()
                else:  # sum
                    aggregated = group_annual.groupby('year').agg({
                        'value': 'sum',
                        'product_name': 'first',
                        'region': 'first',
                        'unit': 'first',
                        'metric': 'first'
                    }).reset_index()
                
                harmonized_rows.extend(aggregated.to_dict('records'))
                
                conversion_log.append({
                    'product': product,
                    'region': region,
                    'from_frequency': detected_freq,
                    'to_frequency': target_frequency,
                    'aggregation_method': agg_method,
                    'original_points': len(group),
                    'harmonized_points': len(aggregated)
                })
    
    harmonized_df = pd.DataFrame(harmonized_rows)
    
    return {
        "harmonized_df": harmonized_df,
        "conversions_applied": len(conversion_log),
        "conversion_log": conversion_log,
        "target_frequency": target_frequency,
        "interpretation": (
            f"Harmonized {len(conversion_log)} series to {target_frequency} frequency. "
            f"Converted {sum(c['original_points'] for c in conversion_log)} data points to "
            f"{len(harmonized_df)} harmonized points."
        )
    }


def impute_missing_values(df: pd.DataFrame,
                         method: str = "interpolation",
                         max_impute_pct: float = 0.10) -> Dict[str, Any]:
    """
    Intelligent missing data imputation using appropriate methods for data type.
    
    Args:
        df: DataFrame with missing values
        method: 'interpolation', 'forward_fill', 'wright_law', 'scurve'
        max_impute_pct: Maximum percentage of series to impute (default 10%)
    
    Returns:
        Dict with imputed DataFrame and imputation log
    """
    if df.empty:
        return {"error": "Empty DataFrame"}
    
    imputed_rows = []
    imputation_log = []
    quality_flags = []
    
    for (product, region), group in df.groupby(['product_name', 'region']):
        group_sorted = group.sort_values('year').copy()
        
        # Count missing values
        total_points = len(group_sorted)
        missing_points = group_sorted['value'].isna().sum()
        missing_pct = missing_points / total_points if total_points > 0 else 0
        
        # Check if imputation is allowed
        if missing_pct > max_impute_pct:
            quality_flags.append({
                'product': product,
                'region': region,
                'missing_pct': missing_pct * 100,
                'reason': f"Exceeds {max_impute_pct*100}% imputation limit"
            })
            # Keep original data
            imputed_rows.extend(group_sorted.to_dict('records'))
            continue
        
        if missing_points == 0:
            # No missing values
            imputed_rows.extend(group_sorted.to_dict('records'))
            continue
        
        # Detect data type for intelligent imputation
        unit = str(group_sorted['unit'].iloc[0]).lower()
        metric = str(group_sorted['metric'].iloc[0]).lower()
        
        is_cost = any(ind in unit for ind in ['$', 'usd', 'cost', 'price'])
        is_adoption = any(ind in metric for ind in ['adoption', 'sales', 'capacity', 'generation'])
        
        # Apply appropriate imputation method
        if method == "wright_law" and is_cost:
            # Use Wright's Law projection for cost data
            # Fit exponential decline to known data
            known_data = group_sorted[group_sorted['value'].notna()]
            if len(known_data) >= 3:
                from scipy import stats
                years_known = known_data['year'].values
                values_known = known_data['value'].values
                
                try:
                    slope, intercept, _, _, _ = stats.linregress(years_known, np.log(values_known))
                    
                    # Impute missing years
                    for idx, row in group_sorted.iterrows():
                        if pd.isna(row['value']):
                            year = row['year']
                            imputed_value = np.exp(slope * year + intercept)
                            group_sorted.at[idx, 'value'] = imputed_value
                            imputation_log.append({
                                'product': product,
                                'region': region,
                                'year': int(year),
                                'method': 'wright_law',
                                'imputed_value': imputed_value
                            })
                except:
                    # Fallback to linear interpolation
                    group_sorted['value'] = group_sorted['value'].interpolate(method='linear')
        
        elif method == "scurve" and is_adoption:
            # Use S-curve fitting for adoption data
            known_data = group_sorted[group_sorted['value'].notna()]
            if len(known_data) >= 4:
                try:
                    from scipy.optimize import curve_fit
                    
                    def sigmoid(x, L, k, x0):
                        return L / (1 + np.exp(-k * (x - x0)))
                    
                    years_known = known_data['year'].values
                    values_known = known_data['value'].values
                    
                    # Fit sigmoid
                    popt, _ = curve_fit(sigmoid, years_known, values_known,
                                       p0=[values_known.max(), 0.5, years_known.mean()],
                                       maxfev=5000)
                    
                    # Impute missing years
                    for idx, row in group_sorted.iterrows():
                        if pd.isna(row['value']):
                            year = row['year']
                            imputed_value = sigmoid(year, *popt)
                            group_sorted.at[idx, 'value'] = imputed_value
                            imputation_log.append({
                                'product': product,
                                'region': region,
                                'year': int(year),
                                'method': 'scurve',
                                'imputed_value': imputed_value
                            })
                except:
                    # Fallback to linear interpolation
                    group_sorted['value'] = group_sorted['value'].interpolate(method='linear')
        
        else:
            # Default: linear interpolation
            group_sorted['value'] = group_sorted['value'].interpolate(method='linear')
            
            for idx, row in group_sorted.iterrows():
                if idx in group.index and pd.isna(group.at[idx, 'value']) and pd.notna(group_sorted.at[idx, 'value']):
                    imputation_log.append({
                        'product': product,
                        'region': region,
                        'year': int(row['year']),
                        'method': 'linear',
                        'imputed_value': group_sorted.at[idx, 'value']
                    })
        
        imputed_rows.extend(group_sorted.to_dict('records'))
    
    imputed_df = pd.DataFrame(imputed_rows)
    
    return {
        "imputed_df": imputed_df,
        "imputation_count": len(imputation_log),
        "imputation_log": imputation_log,
        "quality_flags": quality_flags,
        "method_used": method,
        "interpretation": (
            f"Imputed {len(imputation_log)} missing values using {method} method. "
            f"{len(quality_flags)} series flagged for exceeding imputation limits."
        )
    }
# ================================================================================
# SEBA-SPECIFIC DISRUPTION METRICS
# ================================================================================

def calculate_disruption_velocity(df: pd.DataFrame, product: str) -> Dict[str, Any]:
    """
    Calculate disruption velocity - how fast is the technology disrupting incumbents?
    
    Seba Framework: Exponential disruption happens at >30% annual growth.
    
    Returns velocity classification:
    - Slow: <5% annual growth
    - Moderate: 5-15% annual growth
    - Rapid: 15-30% annual growth
    - Exponential: >30% annual growth
    """
    from validation_constants import (
        DISRUPTION_VELOCITY_SLOW, DISRUPTION_VELOCITY_MODERATE,
        DISRUPTION_VELOCITY_RAPID, DISRUPTION_VELOCITY_EXPONENTIAL
    )
    
    if product and "product_name" in df.columns:
        sub = df[df["product_name"] == product].copy()
    else:
        sub = df.copy()
    
    if len(sub) < 3:
        return {"error": "Insufficient data for velocity calculation"}
    
    try:
        sub = sub.sort_values('year').copy()
        years = sub['year'].values
        values = sub['value'].values
        
        # Calculate year-over-year growth rates
        growth_rates = []
        for i in range(1, len(values)):
            if values[i-1] > 0:
                growth = (values[i] - values[i-1]) / values[i-1]
                growth_rates.append(growth)
        
        if not growth_rates:
            return {"error": "Cannot calculate growth rates"}
        
        # Recent velocity (last 3 years)
        recent_velocity = np.mean(growth_rates[-3:]) if len(growth_rates) >= 3 else np.mean(growth_rates)
        
        # Historical velocity (all years)
        historical_velocity = np.mean(growth_rates)
        
        # Velocity classification
        if recent_velocity >= DISRUPTION_VELOCITY_EXPONENTIAL:
            velocity_class = "Exponential"
            disruption_risk = "CRITICAL"
        elif recent_velocity >= DISRUPTION_VELOCITY_RAPID:
            velocity_class = "Rapid"
            disruption_risk = "HIGH"
        elif recent_velocity >= DISRUPTION_VELOCITY_MODERATE:
            velocity_class = "Moderate"
            disruption_risk = "MEDIUM"
        elif recent_velocity >= DISRUPTION_VELOCITY_SLOW:
            velocity_class = "Slow"
            disruption_risk = "LOW"
        else:
            velocity_class = "Declining"
            disruption_risk = "NONE"
        
        # Acceleration check
        early_velocity = np.mean(growth_rates[:len(growth_rates)//2])
        late_velocity = np.mean(growth_rates[len(growth_rates)//2:])
        is_accelerating = late_velocity > early_velocity
        
        return {
            "recent_velocity": float(recent_velocity),
            "historical_velocity": float(historical_velocity),
            "velocity_class": velocity_class,
            "disruption_risk": disruption_risk,
            "is_accelerating": is_accelerating,
            "growth_rates_by_year": [
                {"year": int(years[i+1]), "growth": float(growth_rates[i])}
                for i in range(len(growth_rates))
            ],
            "interpretation": (
                f"Disruption velocity: {velocity_class} ({recent_velocity*100:.1f}% annual growth). "
                f"Risk to incumbents: {disruption_risk}. "
                f"{'Acceleration detected - disruption intensifying' if is_accelerating else 'Stable velocity'}. "
                f"Seba Framework: {'EXPONENTIAL DISRUPTION PHASE' if recent_velocity >= DISRUPTION_VELOCITY_EXPONENTIAL else 'Pre-exponential phase'}."
            )
        }
    
    except Exception as e:
        return {"error": f"Velocity calculation failed: {str(e)}"}


def calculate_incumbent_vulnerability(df: pd.DataFrame, 
                                     incumbent_product: str,
                                     disruptor_product: str) -> Dict[str, Any]:
    """
    Calculate incumbent vulnerability to disruption.
    
    Factors:
    1. Cost gap: How much cheaper is disruptor?
    2. Adoption rate: How fast is disruptor growing?
    3. Performance parity: Has disruptor matched key metrics?
    4. Incumbent decline: Is incumbent losing market share?
    
    Returns vulnerability score: 0.0 (safe) to 1.0 (critical risk)
    """
    from validation_constants import (
        INCUMBENT_VULNERABILITY_LOW, INCUMBENT_VULNERABILITY_MEDIUM,
        INCUMBENT_VULNERABILITY_HIGH
    )
    
    try:
        incumbent_df = df[df['product_name'] == incumbent_product]
        disruptor_df = df[df['product_name'] == disruptor_product]
        
        if incumbent_df.empty or disruptor_df.empty:
            return {"error": "Product data not found"}
        
        vulnerability_score = 0.0
        factors = {}
        
        # Factor 1: Cost Gap (if both are cost data)
        incumbent_is_cost = any(ind in str(incumbent_df['unit'].iloc[0]).lower() 
                               for ind in ['$', 'cost', 'price'])
        disruptor_is_cost = any(ind in str(disruptor_df['unit'].iloc[0]).lower() 
                               for ind in ['$', 'cost', 'price'])
        
        if incumbent_is_cost and disruptor_is_cost:
            incumbent_cost = incumbent_df['value'].iloc[-1]
            disruptor_cost = disruptor_df['value'].iloc[-1]
            
            if incumbent_cost > 0:
                cost_advantage = (incumbent_cost - disruptor_cost) / incumbent_cost
                factors['cost_advantage'] = float(cost_advantage)
                
                # >50% cost advantage = 0.4 vulnerability points
                if cost_advantage > 0.50:
                    vulnerability_score += 0.4
                elif cost_advantage > 0.30:
                    vulnerability_score += 0.3
                elif cost_advantage > 0.10:
                    vulnerability_score += 0.2
        
        # Factor 2: Disruptor Growth Rate (0.3 points)
        disruptor_sorted = disruptor_df.sort_values('year')
        if len(disruptor_sorted) >= 3:
            recent_growth = []
            values = disruptor_sorted['value'].values
            for i in range(1, min(4, len(values))):
                if values[i-1] > 0:
                    growth = (values[i] - values[i-1]) / values[i-1]
                    recent_growth.append(growth)
            
            if recent_growth:
                avg_growth = np.mean(recent_growth)
                factors['disruptor_growth_rate'] = float(avg_growth)
                
                if avg_growth > 0.50:  # >50% growth
                    vulnerability_score += 0.3
                elif avg_growth > 0.30:
                    vulnerability_score += 0.2
                elif avg_growth > 0.15:
                    vulnerability_score += 0.1
        
        # Factor 3: Incumbent Decline (0.3 points)
        incumbent_sorted = incumbent_df.sort_values('year')
        if len(incumbent_sorted) >= 3:
            recent_change = []
            values = incumbent_sorted['value'].values
            for i in range(1, min(4, len(values))):
                if values[i-1] > 0:
                    change = (values[i] - values[i-1]) / values[i-1]
                    recent_change.append(change)
            
            if recent_change:
                avg_change = np.mean(recent_change)
                factors['incumbent_trend'] = float(avg_change)
                
                if avg_change < -0.10:  # Declining >10%
                    vulnerability_score += 0.3
                elif avg_change < -0.05:
                    vulnerability_score += 0.2
                elif avg_change < 0:
                    vulnerability_score += 0.1
        
        # Classify vulnerability
        if vulnerability_score >= INCUMBENT_VULNERABILITY_HIGH:
            vulnerability_class = "CRITICAL"
            recommendation = "Immediate strategic response required. Disruption imminent."
        elif vulnerability_score >= INCUMBENT_VULNERABILITY_MEDIUM:
            vulnerability_class = "HIGH"
            recommendation = "Significant threat. Accelerate transformation efforts."
        elif vulnerability_score >= INCUMBENT_VULNERABILITY_LOW:
            vulnerability_class = "MEDIUM"
            recommendation = "Monitor closely. Begin defensive strategies."
        else:
            vulnerability_class = "LOW"
            recommendation = "Disruptor not yet critical threat."
        
        return {
            "incumbent": incumbent_product,
            "disruptor": disruptor_product,
            "vulnerability_score": float(vulnerability_score),
            "vulnerability_class": vulnerability_class,
            "factors": factors,
            "recommendation": recommendation,
            "interpretation": (
                f"Incumbent vulnerability: {vulnerability_class} (score: {vulnerability_score:.2f}/1.0). "
                f"{recommendation}"
            )
        }
    
    except Exception as e:
        return {"error": f"Vulnerability calculation failed: {str(e)}"}


def identify_technology_lifecycle_stage(df: pd.DataFrame, product: str) -> Dict[str, Any]:
    """
    Identify where technology is in its lifecycle.
    
    Stages:
    1. Emerging (0-1% adoption, <5 years)
    2. Growth (1-10% adoption, 5-10 years)
    3. Mainstream (10-50% adoption, 10-20 years)
    4. Mature (>50% adoption, >20 years)
    """
    from validation_constants import LIFECYCLE_THRESHOLDS
    
    if product and "product_name" in df.columns:
        sub = df[df["product_name"] == product].copy()
    else:
        sub = df.copy()
    
    if len(sub) < 2:
        return {"error": "Insufficient data"}
    
    try:
        sub = sub.sort_values('year')
        years = sub['year'].values
        values = sub['value'].values
        
        # Calculate age
        first_year = int(years[0])
        latest_year = int(years[-1])
        age_years = latest_year - first_year
        
        # Estimate adoption level (if percentage data)
        latest_value = float(values[-1])
        
        # Determine stage
        if age_years <= LIFECYCLE_THRESHOLDS['emerging']['age_years']:
            if latest_value < 0.01:
                stage = "Emerging"
                seba_insight = "Technology in innovation phase. High risk, high potential."
            else:
                stage = "Early Growth"
                seba_insight = "Rapid growth beginning. Critical adoption phase."
        elif age_years <= LIFECYCLE_THRESHOLDS['growth']['age_years']:
            if latest_value < 0.10:
                stage = "Growth"
                seba_insight = "Pre-tipping point. Approaching 10% threshold for exponential growth."
            else:
                stage = "Rapid Growth"
                seba_insight = "Post-tipping point. Seba's 10%→90% rapid transition phase."
        elif age_years <= LIFECYCLE_THRESHOLDS['mainstream']['age_years']:
            stage = "Mainstream"
            seba_insight = "Technology established. Focus on cost optimization and scale."
        else:
            stage = "Mature"
            seba_insight = "Mature market. Vulnerable to next-generation disruption."
        
        return {
            "product": product,
            "stage": stage,
            "age_years": age_years,
            "first_year": first_year,
            "latest_year": latest_year,
            "current_adoption": latest_value,
            "seba_insight": seba_insight,
            "interpretation": (
                f"{product} is in {stage} stage (age: {age_years} years). "
                f"Current adoption: {latest_value*100 if latest_value < 1 else latest_value:.1f}%. "
                f"{seba_insight}"
            )
        }
    
    except Exception as e:
        return {"error": f"Lifecycle identification failed: {str(e)}"}
# ================================================================================
# ADVANCED ANALYTICS - STRUCTURAL BREAKS, FORECASTING, SCENARIOS
# ================================================================================

def detect_structural_breaks_chow_test(df: pd.DataFrame, product: str,
                                       breakpoint_year: int = None) -> Dict[str, Any]:
    """
    Chow test for structural breaks in time series.
    Detects if data generation process changed (methodology shift, source change).
    
    H0: No structural break (coefficients are same before/after breakpoint)
    H1: Structural break exists (coefficients differ)
    """
    from scipy import stats
    
    if product and "product_name" in df.columns:
        sub = df[df["product_name"] == product].copy()
    else:
        sub = df.copy()
    
    if len(sub) < 6:  # Need sufficient data points
        return {"error": "Insufficient data for structural break analysis"}
    
    try:
        sub = sub.sort_values('year').copy()
        years = sub['year'].values
        values = sub['value'].values
        
        # If no breakpoint specified, test midpoint
        if breakpoint_year is None:
            breakpoint_year = int(years[len(years) // 2])
        
        # Split data at breakpoint
        mask_before = sub['year'] < breakpoint_year
        mask_after = sub['year'] >= breakpoint_year
        
        years_before = sub[mask_before]['year'].values
        values_before = sub[mask_before]['value'].values
        years_after = sub[mask_after]['year'].values
        values_after = sub[mask_after]['value'].values
        
        if len(years_before) < 3 or len(years_after) < 3:
            return {"error": "Insufficient data in split periods"}
        
        # Fit models
        # Full model
        slope_full, intercept_full, r_full, _, _ = stats.linregress(years, values)
        y_pred_full = slope_full * years + intercept_full
        rss_full = np.sum((values - y_pred_full) ** 2)
        
        # Before model
        slope_before, intercept_before, _, _, _ = stats.linregress(years_before, values_before)
        y_pred_before = slope_before * years_before + intercept_before
        rss_before = np.sum((values_before - y_pred_before) ** 2)
        
        # After model
        slope_after, intercept_after, _, _, _ = stats.linregress(years_after, values_after)
        y_pred_after = slope_after * years_after + intercept_after
        rss_after = np.sum((values_after - y_pred_after) ** 2)
        
        # Chow test statistic
        rss_restricted = rss_full
        rss_unrestricted = rss_before + rss_after
        
        n = len(years)
        k = 2  # Number of parameters (slope, intercept)
        
        if rss_unrestricted > 0:
            chow_stat = ((rss_restricted - rss_unrestricted) / k) / (rss_unrestricted / (n - 2*k))
            
            # F-distribution critical value
            alpha = 0.05
            df1 = k
            df2 = n - 2*k
            f_critical = stats.f.ppf(1 - alpha, df1, df2)
            
            # P-value
            p_value = 1 - stats.f.cdf(chow_stat, df1, df2)
            
            structural_break = p_value < alpha
            
            return {
                "structural_break_detected": structural_break,
                "breakpoint_year": int(breakpoint_year),
                "chow_statistic": float(chow_stat),
                "p_value": float(p_value),
                "f_critical": float(f_critical),
                "slope_before": float(slope_before),
                "slope_after": float(slope_after),
                "slope_change": float(slope_after - slope_before),
                "interpretation": (
                    f"{'STRUCTURAL BREAK DETECTED' if structural_break else 'No structural break'} "
                    f"at {breakpoint_year} (p={p_value:.4f}). "
                    f"Slope changed from {slope_before:.3f} to {slope_after:.3f}. "
                    f"{'Likely methodology/source change' if structural_break else 'Data generation consistent'}."
                )
            }
        else:
            return {"error": "Model fit failed"}
    
    except Exception as e:
        return {"error": f"Chow test failed: {str(e)}"}


def forecast_time_series(df: pd.DataFrame, product: str, 
                        forecast_years: int = 5,
                        method: str = "exponential") -> Dict[str, Any]:
    """
    Time series forecasting using exponential smoothing or ARIMA.
    Projects future values based on historical trends.
    """
    if product and "product_name" in df.columns:
        sub = df[df["product_name"] == product].copy()
    else:
        sub = df.copy()
    
    if len(sub) < 3:
        return {"error": "Insufficient historical data for forecasting"}
    
    try:
        sub = sub.sort_values('year').copy()
        years = sub['year'].values
        values = sub['value'].values
        
        last_year = int(years[-1])
        forecast_years_array = np.arange(last_year + 1, last_year + forecast_years + 1)
        
        if method == "exponential":
            # Exponential decay/growth model (good for Wright's Law)
            from scipy import stats
            
            # Fit log-linear model
            log_values = np.log(values)
            slope, intercept, r_value, p_value, std_err = stats.linregress(years, log_values)
            
            # Forecast
            log_forecast = slope * forecast_years_array + intercept
            forecast_values = np.exp(log_forecast)
            
            # Confidence intervals (95%)
            from scipy.stats import t
            n = len(years)
            alpha = 0.05
            t_val = t.ppf(1 - alpha/2, n - 2)
            
            # Standard error of forecast
            se_forecast = std_err * np.sqrt(1 + 1/n + (forecast_years_array - years.mean())**2 / np.sum((years - years.mean())**2))
            
            ci_lower = np.exp(log_forecast - t_val * se_forecast)
            ci_upper = np.exp(log_forecast + t_val * se_forecast)
            
            return {
                "method": "exponential",
                "forecast_years": forecast_years_array.tolist(),
                "forecast_values": forecast_values.tolist(),
                "confidence_interval_lower": ci_lower.tolist(),
                "confidence_interval_upper": ci_upper.tolist(),
                "r_squared": float(r_value ** 2),
                "growth_rate": float(np.exp(slope) - 1),
                "interpretation": (
                    f"Forecast using exponential model (R²={r_value**2:.3f}). "
                    f"Projected {'growth' if slope > 0 else 'decline'} rate: {abs(np.exp(slope)-1)*100:.1f}% per year."
                )
            }
        
        elif method == "linear":
            # Simple linear extrapolation
            from scipy import stats
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(years, values)
            
            forecast_values = slope * forecast_years_array + intercept
            
            # Confidence intervals
            from scipy.stats import t
            n = len(years)
            alpha = 0.05
            t_val = t.ppf(1 - alpha/2, n - 2)
            
            se_forecast = std_err * np.sqrt(1 + 1/n + (forecast_years_array - years.mean())**2 / np.sum((years - years.mean())**2))
            
            ci_lower = forecast_values - t_val * se_forecast
            ci_upper = forecast_values + t_val * se_forecast
            
            return {
                "method": "linear",
                "forecast_years": forecast_years_array.tolist(),
                "forecast_values": forecast_values.tolist(),
                "confidence_interval_lower": ci_lower.tolist(),
                "confidence_interval_upper": ci_upper.tolist(),
                "r_squared": float(r_value ** 2),
                "trend": float(slope),
                "interpretation": (
                    f"Linear forecast (R²={r_value**2:.3f}). "
                    f"Trend: {slope:+.2f} units per year."
                )
            }
        
    except Exception as e:
        return {"error": f"Forecast failed: {str(e)}"}


def scenario_analysis(df: pd.DataFrame, product: str,
                     scenarios: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Scenario analysis for sensitivity testing.
    Tests impact of different learning rate assumptions on cost projections.
    """
    if scenarios is None:
        scenarios = {
            "pessimistic": 0.10,   # 10% learning rate
            "base_case": 0.20,     # 20% learning rate
            "optimistic": 0.30     # 30% learning rate
        }
    
    if product and "product_name" in df.columns:
        sub = df[df["product_name"] == product].copy()
    else:
        sub = df.copy()
    
    # Check if cost data
    is_cost = False
    if 'unit' in sub.columns and len(sub) > 0:
        unit_str = str(sub['unit'].iloc[0]).lower()
        is_cost = any(ind in unit_str for ind in ['$', 'cost', 'price'])
    
    if not is_cost or len(sub) < 3:
        return {"error": "Scenario analysis requires cost data"}
    
    try:
        sub = sub.sort_values('year').copy()
        years = sub['year'].values
        costs = sub['value'].values
        
        last_year = int(years[-1])
        last_cost = float(costs[-1])
        
        # Project 10 years forward under different scenarios
        projection_years = 10
        future_years = np.arange(last_year + 1, last_year + projection_years + 1)
        
        scenario_results = {}
        
        for scenario_name, learning_rate in scenarios.items():
            # Assume production doubles every 2 years (simplified)
            doublings = projection_years / 2
            
            # Wright's Law: Cost_n = Cost_0 * (2^doublings)^(-learning_rate)
            cost_multiplier = (2 ** doublings) ** (-learning_rate)
            final_cost = last_cost * cost_multiplier
            
            # Interpolate intermediate years
            annual_decline = (1 - cost_multiplier) / projection_years
            projected_costs = [last_cost * (1 - annual_decline * i) for i in range(1, projection_years + 1)]
            
            scenario_results[scenario_name] = {
                "learning_rate": learning_rate,
                "final_cost": float(final_cost),
                "cost_reduction": float((last_cost - final_cost) / last_cost),
                "projected_costs": projected_costs,
                "years": future_years.tolist()
            }
        
        return {
            "current_cost": last_cost,
            "current_year": last_year,
            "scenarios": scenario_results,
            "interpretation": (
                f"Starting from ${last_cost:.2f} in {last_year}, "
                f"pessimistic scenario: ${scenario_results['pessimistic']['final_cost']:.2f}, "
                f"base case: ${scenario_results['base_case']['final_cost']:.2f}, "
                f"optimistic: ${scenario_results['optimistic']['final_cost']:.2f} by {last_year + projection_years}."
            )
        }
    
    except Exception as e:
        return {"error": f"Scenario analysis failed: {str(e)}"}


def sensitivity_analysis(df: pd.DataFrame, product: str,
                        parameter: str = "learning_rate",
                        param_range: Tuple[float, float] = (0.10, 0.35)) -> Dict[str, Any]:
    """
    Sensitivity analysis: How sensitive are forecasts to parameter assumptions?
    """
    if product and "product_name" in df.columns:
        sub = df[df["product_name"] == product].copy()
    else:
        sub = df.copy()
    
    if len(sub) < 3:
        return {"error": "Insufficient data"}
    
    try:
        sub = sub.sort_values('year').copy()
        years = sub['year'].values
        values = sub['value'].values
        
        last_year = int(years[-1])
        last_value = float(values[-1])
        
        # Test parameter range
        param_values = np.linspace(param_range[0], param_range[1], 20)
        projection_year = last_year + 5  # 5 years ahead
        
        projected_values = []
        
        for param_val in param_values:
            if parameter == "learning_rate":
                # Assume 2.5 doublings in 5 years
                doublings = 2.5
                cost_multiplier = (2 ** doublings) ** (-param_val)
                projected = last_value * cost_multiplier
                projected_values.append(projected)
        
        # Calculate sensitivity metric (% change in output per % change in input)
        output_range = max(projected_values) - min(projected_values)
        input_range = param_range[1] - param_range[0]
        
        sensitivity = (output_range / last_value) / (input_range / np.mean(param_values)) if np.mean(param_values) > 0 else 0
        
        return {
            "parameter": parameter,
            "parameter_range": param_range,
            "parameter_values": param_values.tolist(),
            "projected_values": projected_values,
            "sensitivity_index": float(sensitivity),
            "highly_sensitive": sensitivity > 1.0,
            "interpretation": (
                f"Sensitivity analysis shows {'HIGH' if sensitivity > 1.0 else 'LOW'} sensitivity "
                f"(index={sensitivity:.2f}). "
                f"{'Forecasts highly dependent on parameter assumptions' if sensitivity > 1.0 else 'Forecasts relatively robust to parameter variations'}."
            )
        }
    
    except Exception as e:
        return {"error": f"Sensitivity analysis failed: {str(e)}"}
    
# ================================================================================
# ENHANCED VALIDATION PIPELINE
# ================================================================================

def run_validation_pipeline(results: List[Dict[str, Any]],
                           enable_seba: bool = True,
                           enable_llm: bool = False) -> Dict[str, Any]:
    """
    ENHANCED: Includes comprehensive data quality checks
    """
    # FIXED: validation_logs must be first line
    validation_logs = []
    
    logger.info(f"Starting validation pipeline: {len(results)} entities")
    logger.info(f"Seba framework enabled: {enable_seba}")
    logger.info(f"LLM analysis enabled: {enable_llm}")
    
    # ===== DATA PROVENANCE TRACKING =====
    validation_metadata = {
        'validation_timestamp': datetime.now().isoformat(),
        'validation_version': '2.0.0-enhanced',
        'framework': 'Tony Seba STELLAR (Seba Technology Energy Logic Learning And Research)',
        'validator_count': 35,
        'input_records': len(results),
        'provenance': []
    }
    
    # Track provenance for each data point
    for result in results:
        entity = result.get('Entity_Name', 'Unknown')
        sources = result.get('DataSource_URLs', [])
        
        validation_metadata['provenance'].append({
            'entity': entity,
            'region': result.get('Region', 'Unknown'),
            'sources': sources,
            'source_count': len(sources),
            'timestamp': result.get('Timestamp', validation_metadata['validation_timestamp']),
            'quality_score': result.get('Quality_Score', 0)
        })
    
    validation_logs.append(f"📋 Validation Metadata: Version {validation_metadata['validation_version']}")
    validation_logs.append(f"   Timestamp: {validation_metadata['validation_timestamp']}")
    validation_logs.append(f"   Tracking provenance for {len(validation_metadata['provenance'])} entities")
    # ===== END PROVENANCE TRACKING =====
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
    # TIPPING POINT DETECTION (ADD AFTER S-CURVE)
                    tipping_result = detect_tipping_point(product_df, product)
                    if 'error' not in tipping_result:
                        seba_results[f'{product}_tipping_point'] = tipping_result
                        if tipping_result.get('tipping_point_crossed'):
                            validation_logs.append(
                                f"   - {product} Tipping Point: Crossed in {tipping_result.get('tipping_point_year')} "
                                f"({'MEETS' if tipping_result.get('meets_seba_pattern') else 'MISSES'} Seba pattern)"
                            )
                    
                    # COST PARITY TRACKING (ADD AFTER TIPPING POINT)
                    parity_result = track_cost_parity_forecast(product_df, product, parity_threshold=70.0)
                    if 'error' not in parity_result:
                        seba_results[f'{product}_cost_parity'] = parity_result
                        if parity_result.get('at_parity'):
                            validation_logs.append(f"   - {product} Cost Parity: Achieved in {parity_result.get('parity_year')}")
                        elif parity_result.get('forecast_parity_year'):
                            validation_logs.append(
                                f"   - {product} Cost Parity Forecast: ~{parity_result.get('forecast_parity_year')} "
                                f"({parity_result.get('years_until_parity')} years)"
                            )
            
            # CONVERGENCE ANALYSIS (ADD AFTER PRODUCT LOOP)
            convergence_result = analyze_convergence(clean_df)
            if 'error' not in convergence_result:
                seba_results['convergence_analysis'] = convergence_result
                if convergence_result.get('converging_technologies'):
                    validation_logs.append(
                        f"   - âš ï¸ CONVERGENCE DETECTED: {convergence_result.get('high_growth_count')} "
                        f"high-growth technologies - accelerated disruption likely"
                    )
            # ADVANCED ANALYTICS (ADD AFTER CONVERGENCE)
            validation_logs.append("🔬 Running advanced analytics...")
        
            for product in products[:5]:
                product_df = clean_df[clean_df['product_name'] == product]
            
                if len(product_df) >= 6:
                    # Structural break detection
                    chow_result = detect_structural_breaks_chow_test(product_df, product)
                    if 'error' not in chow_result:
                        seba_results[f'{product}_structural_break'] = chow_result
                        if chow_result.get('structural_break_detected'):
                            validation_logs.append(
                                f"   - WARNING: {product} structural break at {chow_result.get('breakpoint_year')} "
                                f"(p={chow_result.get('p_value'):.4f})"
                            )
                
                    # Forecasting
                    forecast_result = forecast_time_series(product_df, product, forecast_years=5)
                    if 'error' not in forecast_result:
                        seba_results[f'{product}_forecast'] = forecast_result
                
                    # Scenario analysis (only for cost data)
                    scenario_result = scenario_analysis(product_df, product)
                    if 'error' not in scenario_result:
                        seba_results[f'{product}_scenarios'] = scenario_result
                
                    # Sensitivity analysis
                    sensitivity_result = sensitivity_analysis(product_df, product)
                    if 'error' not in sensitivity_result:
                        seba_results[f'{product}_sensitivity'] = sensitivity_result
                        if sensitivity_result.get('highly_sensitive'):
                            validation_logs.append(
                                f"   - NOTE: {product} forecasts highly sensitive to assumptions "
                                f"(index={sensitivity_result.get('sensitivity_index'):.2f})"
                            )   
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
        # Step 6: Generate visualizations for Streamlit display
        validation_logs.append("📊 Step 6: Preparing visualizations...")
        try:
            from plotly import graph_objects as go
            visualization_ready = True
        except ImportError:
            visualization_ready = False
            validation_logs.append("   ⚠️ Plotly not available for visualizations")
        # Step 7: Generate Report
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
            'validation_metadata': validation_metadata,
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
def generate_investment_grade_pdf_report(validation_results: Dict[str, Any],
                                        output_filename: str = None) -> str:
    """
    Generate professional PDF report with methodology section.
    Requires: pip install reportlab
    """
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    except ImportError:
        return "ERROR: reportlab not installed. Run: pip install reportlab"
    
    if output_filename is None:
        output_filename = f"STELLAR_Validation_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    # Create PDF
    doc = SimpleDocTemplate(output_filename, pagesize=letter,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    # Container for PDF elements
    story = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2E4057'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2E4057'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title Page
    story.append(Paragraph("STELLAR Framework", title_style))
    story.append(Paragraph("Investment-Grade Data Validation Report", styles['Heading2']))
    story.append(Spacer(1, 0.5*inch))
    
    story.append(Paragraph(
        "<b>Seba Technology Energy Logic Learning And Research</b>",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.2*inch))
    
    # Metadata
    validation_metadata = validation_results.get('validation_metadata', {})
    story.append(Paragraph(f"<b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Paragraph(f"<b>Validation Version:</b> {validation_metadata.get('validation_version', 'Unknown')}", styles['Normal']))
    story.append(Paragraph(f"<b>Framework:</b> {validation_metadata.get('framework', 'STELLAR')}", styles['Normal']))
    
    story.append(PageBreak())
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    
    investment_grade = validation_results.get('investment_grade', {})
    score = validation_results.get('score', ValidationScore())
    
    # Investment Grade Status
    status_data = [
        ['Investment Grade Status', investment_grade.get('status', 'Unknown')],
        ['Overall Grade', investment_grade.get('grade', 'N/A')],
        ['Quality Score', f"{investment_grade.get('score', 0)*100:.1f}%"],
        ['Pass Rate', investment_grade.get('calculation_formula', 'N/A')]
    ]
    
    status_table = Table(status_data, colWidths=[3*inch, 3*inch])
    status_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E4057')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(status_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Recommendation
    story.append(Paragraph("<b>Recommendation:</b>", styles['Normal']))
    story.append(Paragraph(investment_grade.get('recommendation', 'N/A'), styles['Normal']))
    
    story.append(PageBreak())
    
    # Methodology Section
    story.append(Paragraph("Validation Methodology", heading_style))
    
    methodology_text = """
    <b>STELLAR Framework Overview</b><br/>
    The STELLAR (Seba Technology Energy Logic Learning And Research) framework validates energy 
    transition data using three core analytical approaches:<br/><br/>
    
    <b>1. Wright's Law (Technology Cost Curves)</b><br/>
    Formula: Cost(n) = Cost₁ × n^(-b)<br/>
    Expected learning rate: 15-30% cost reduction per production doubling<br/>
    Reference: Wright, T. P. (1936). "Factors Affecting the Cost of Airplanes." 
    Journal of the Aeronautical Sciences, 3(4), 122-128.<br/><br/>
    
    <b>2. S-Curve Adoption Theory</b><br/>
    Market adoption follows sigmoid function: f(t) = L / (1 + e^(-k(t-t₀)))<br/>
    Critical tipping point: 10% market penetration<br/>
    Expected pattern: 10% → 90% adoption in under 10 years for disruptive technologies<br/>
    Reference: Rogers, E. M. (2003). "Diffusion of Innovations" (5th ed.). Free Press.<br/>
    Seba, T. (2014). "Clean Disruption of Energy and Transportation."<br/><br/>
    
    <b>3. Commodity Cycles</b><br/>
    Supply/demand driven pricing with cyclical patterns<br/>
    Expected: 3-7% annual growth, does NOT follow technology learning curves<br/>
    Reference: Hamilton, J. D. (2009). "Understanding Crude Oil Prices." 
    Energy Journal, 30(2), 179-206.<br/><br/>
    
    <b>Statistical Methods</b><br/>
    • Bootstrap confidence intervals (1000 iterations, 95% CI)<br/>
    • Grubbs test for outlier detection (α = 0.05)<br/>
    • Linear regression for learning rate estimation<br/>
    • Statistical power analysis (minimum 80% power required)<br/><br/>
    
    <b>Quality Dimensions Validated</b><br/>
    • Completeness: Missing data analysis<br/>
    • Accuracy: Logical consistency and type validation<br/>
    • Consistency: Unit and format standardization<br/>
    • Validity: Domain-specific rule compliance<br/>
    • Framework Compliance: Wright's Law and S-Curve adherence<br/>
    """
    
    story.append(Paragraph(methodology_text, styles['Normal']))
    
    story.append(PageBreak())
    
    # Validation Results Section
    story.append(Paragraph("Detailed Validation Results", heading_style))
    
    summary = validation_results.get('validation_summary', {})
    
    results_data = [
        ['Metric', 'Count', 'Percentage'],
        ['Total Validators', summary.get('total', 35), '100%'],
        ['Passed', summary.get('passed', 0), f"{summary.get('passed', 0)/35*100:.1f}%"],
        ['Failed', summary.get('failed', 0), f"{summary.get('failed', 0)/35*100:.1f}%"],
        ['Warnings', summary.get('warnings', 0), f"{summary.get('warnings', 0)/35*100:.1f}%"],
        ['N/A (Not Applicable)', summary.get('na', 0), f"{summary.get('na', 0)/35*100:.1f}%"],
    ]
    
    results_table = Table(results_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(results_table)
    
    # Build PDF
    doc.build(story)
    
    return output_filename
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

# ADD THIS ENTIRE FUNCTION AT THE END OF validation_support.py

def generate_investment_grade_pdf_report(validation_results: Dict[str, Any],
                                        output_filename: str = None) -> str:
    """
    Generate professional PDF report with methodology section.
    Requires: pip install reportlab
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError:
        return "ERROR: reportlab not installed. Run: pip install reportlab"
    
    if output_filename is None:
        output_filename = f"STELLAR_Validation_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    # Create PDF
    doc = SimpleDocTemplate(output_filename, pagesize=letter,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    # Container for PDF elements
    story = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2E4057'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    # Title Page
    story.append(Paragraph("STELLAR Framework", title_style))
    story.append(Paragraph("Investment-Grade Data Validation Report", styles['Heading2']))
    story.append(Spacer(1, 0.5*inch))
    
    # Metadata
    validation_metadata = validation_results.get('validation_metadata', {})
    story.append(Paragraph(f"<b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Paragraph(f"<b>Validation Version:</b> {validation_metadata.get('validation_version', '2.0.0')}", styles['Normal']))
    
    story.append(PageBreak())
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", styles['Heading1']))
    
    investment_grade = validation_results.get('investment_grade', {})
    
    # Status table
    status_data = [
        ['Investment Grade Status', investment_grade.get('status', 'Unknown')],
        ['Overall Grade', investment_grade.get('grade', 'N/A')],
        ['Quality Score', f"{investment_grade.get('score', 0)*100:.1f}%"],
        ['Pass Rate', investment_grade.get('calculation_formula', 'N/A')]
    ]
    
    status_table = Table(status_data, colWidths=[3*inch, 3*inch])
    status_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(status_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Recommendation
    story.append(Paragraph("<b>Recommendation:</b>", styles['Normal']))
    story.append(Paragraph(investment_grade.get('recommendation', 'N/A'), styles['Normal']))
    
    # Build PDF
    doc.build(story)
    
    return output_filename
