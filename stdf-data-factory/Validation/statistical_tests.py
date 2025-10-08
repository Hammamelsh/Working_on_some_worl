"""
Advanced Statistical Tests for STELLAR Framework
Bonferroni correction, ADF, KPSS, Heteroskedasticity tests
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from scipy import stats
from validation_constants import *

try:
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.stats.diagnostic import het_breuschpagan, het_white
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


# ================================================================================
# BONFERRONI CORRECTION FOR MULTIPLE COMPARISONS
# ================================================================================

def apply_bonferroni_correction(p_values: List[float], 
                               alpha: float = BONFERRONI_ALPHA) -> Dict[str, Any]:
    """
    Apply Bonferroni correction for multiple hypothesis testing.
    
    Critical for avoiding false positives when running 35+ validators.
    Corrected alpha = alpha / number_of_tests
    
    Args:
        p_values: List of p-values from multiple tests
        alpha: Family-wise error rate (default 0.05)
    
    Returns:
        Dict with corrected results and interpretation
    """
    n_tests = len(p_values)
    
    if n_tests == 0:
        return {"error": "No p-values provided"}
    
    # Bonferroni correction
    corrected_alpha = alpha / n_tests
    
    # Determine which tests are significant after correction
    significant_before = sum(1 for p in p_values if p < alpha)
    significant_after = sum(1 for p in p_values if p < corrected_alpha)
    
    # Calculate adjusted p-values (multiply by number of tests, cap at 1.0)
    adjusted_p_values = [min(p * n_tests, 1.0) for p in p_values]
    
    return {
        "n_tests": n_tests,
        "original_alpha": alpha,
        "corrected_alpha": corrected_alpha,
        "significant_before_correction": significant_before,
        "significant_after_correction": significant_after,
        "false_positive_reduction": significant_before - significant_after,
        "adjusted_p_values": adjusted_p_values,
        "interpretation": (
            f"Bonferroni correction: {n_tests} tests, α={alpha} → corrected α={corrected_alpha:.4f}. "
            f"{significant_before} tests significant before correction, "
            f"{significant_after} after correction. "
            f"Reduced false positives by {significant_before - significant_after}."
        )
    }


def apply_benjamini_hochberg(p_values: List[float],
                             fdr: float = FALSE_DISCOVERY_RATE) -> Dict[str, Any]:
    """
    Apply Benjamini-Hochberg procedure (less conservative than Bonferroni).
    Controls False Discovery Rate instead of Family-Wise Error Rate.
    
    Better for exploratory analysis where Bonferroni is too strict.
    """
    n_tests = len(p_values)
    
    if n_tests == 0:
        return {"error": "No p-values provided"}
    
    # Sort p-values and track original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    
    # BH critical values
    critical_values = [(i + 1) / n_tests * fdr for i in range(n_tests)]
    
    # Find largest i where p[i] <= (i+1)/n * FDR
    significant_indices = []
    for i in range(n_tests - 1, -1, -1):
        if sorted_p[i] <= critical_values[i]:
            # All tests up to this point are significant
            significant_indices = sorted_indices[:i+1].tolist()
            break
    
    n_significant = len(significant_indices)
    
    return {
        "n_tests": n_tests,
        "fdr": fdr,
        "n_significant": n_significant,
        "significant_indices": significant_indices,
        "critical_values": critical_values,
        "interpretation": (
            f"Benjamini-Hochberg: {n_tests} tests, FDR={fdr}. "
            f"{n_significant} tests significant. "
            f"Expected false discoveries: {n_significant * fdr:.1f}"
        )
    }


# ================================================================================
# TIME SERIES STATIONARITY TESTS
# ================================================================================

def augmented_dickey_fuller_test(series: pd.Series, 
                                 maxlag: int = ADF_MAX_LAGS) -> Dict[str, Any]:
    """
    Augmented Dickey-Fuller test for stationarity.
    
    H0: Series has unit root (non-stationary)
    H1: Series is stationary
    
    Critical for time series forecasting - non-stationary series need differencing.
    """
    if not STATSMODELS_AVAILABLE:
        return {"error": "statsmodels not installed. Run: pip install statsmodels"}
    
    # Remove NaN values
    clean_series = series.dropna()
    
    if len(clean_series) < 10:
        return {"error": "Insufficient data for ADF test (need ≥10 observations)"}
    
    try:
        # Run ADF test
        result = adfuller(clean_series, maxlag=maxlag, autolag='AIC')
        
        adf_statistic = result[0]
        p_value = result[1]
        n_lags = result[2]
        n_obs = result[3]
        critical_values = result[4]
        
        # Determine stationarity
        is_stationary = p_value < STATIONARITY_P_VALUE
        
        # Determine level of evidence
        if p_value < 0.01:
            evidence = "strong"
        elif p_value < 0.05:
            evidence = "moderate"
        elif p_value < 0.10:
            evidence = "weak"
        else:
            evidence = "insufficient"
        
        return {
            "test": "Augmented Dickey-Fuller",
            "adf_statistic": float(adf_statistic),
            "p_value": float(p_value),
            "n_lags_used": int(n_lags),
            "n_observations": int(n_obs),
            "critical_values": {
                "1%": float(critical_values['1%']),
                "5%": float(critical_values['5%']),
                "10%": float(critical_values['10%'])
            },
            "is_stationary": is_stationary,
            "evidence_strength": evidence,
            "interpretation": (
                f"ADF test: {'Stationary' if is_stationary else 'Non-stationary'} "
                f"(p={p_value:.4f}, {evidence} evidence). "
                f"ADF statistic ({adf_statistic:.3f}) {'<' if is_stationary else '>='} "
                f"critical value ({critical_values['5%']:.3f} at 5%). "
                f"{'Series suitable for forecasting' if is_stationary else 'Consider differencing before forecasting'}."
            ),
            "recommendation": (
                "Series is stationary - proceed with forecasting" if is_stationary else
                "Series is non-stationary - apply first-order differencing and retest"
            )
        }
    
    except Exception as e:
        return {"error": f"ADF test failed: {str(e)}"}


def kpss_stationarity_test(series: pd.Series,
                           regression: str = KPSS_REGRESSION) -> Dict[str, Any]:
    """
    KPSS test for stationarity (complementary to ADF).
    
    H0: Series is stationary
    H1: Series is non-stationary
    
    Note: Opposite null hypothesis from ADF!
    Use both tests for robust stationarity assessment.
    """
    if not STATSMODELS_AVAILABLE:
        return {"error": "statsmodels not installed. Run: pip install statsmodels"}
    
    clean_series = series.dropna()
    
    if len(clean_series) < 10:
        return {"error": "Insufficient data for KPSS test (need ≥10 observations)"}
    
    try:
        # Run KPSS test
        result = kpss(clean_series, regression=regression, nlags='auto')
        
        kpss_statistic = result[0]
        p_value = result[1]
        n_lags = result[2]
        critical_values = result[3]
        
        # Note: For KPSS, we WANT to fail to reject H0 (stationary)
        is_stationary = p_value > STATIONARITY_P_VALUE
        
        return {
            "test": "KPSS",
            "kpss_statistic": float(kpss_statistic),
            "p_value": float(p_value),
            "n_lags_used": int(n_lags),
            "critical_values": {
                "10%": float(critical_values['10%']),
                "5%": float(critical_values['5%']),
                "2.5%": float(critical_values['2.5%']),
                "1%": float(critical_values['1%'])
            },
            "is_stationary": is_stationary,
            "interpretation": (
                f"KPSS test: {'Stationary' if is_stationary else 'Non-stationary'} "
                f"(p={p_value:.4f}). "
                f"KPSS statistic ({kpss_statistic:.3f}) {'<' if is_stationary else '>='} "
                f"critical value ({critical_values['5%']:.3f} at 5%). "
                f"{'Confirms stationarity' if is_stationary else 'Suggests trend-stationarity'}."
            ),
            "recommendation": (
                "Series is stationary" if is_stationary else
                "Series may have deterministic trend - consider detrending"
            )
        }
    
    except Exception as e:
        return {"error": f"KPSS test failed: {str(e)}"}


def combined_stationarity_assessment(series: pd.Series) -> Dict[str, Any]:
    """
    Combined ADF + KPSS stationarity assessment.
    
    Decision Matrix:
    - ADF: Stationary, KPSS: Stationary → Stationary
    - ADF: Non-stationary, KPSS: Non-stationary → Non-stationary (difference)
    - ADF: Stationary, KPSS: Non-stationary → Trend-stationary (detrend)
    - ADF: Non-stationary, KPSS: Stationary → Difference-stationary (difference)
    """
    adf_result = augmented_dickey_fuller_test(series)
    kpss_result = kpss_stationarity_test(series)
    
    if 'error' in adf_result or 'error' in kpss_result:
        return {
            "error": "Stationarity tests failed",
            "adf_result": adf_result,
            "kpss_result": kpss_result
        }
    
    adf_stationary = adf_result['is_stationary']
    kpss_stationary = kpss_result['is_stationary']
    
    # Decision matrix
    if adf_stationary and kpss_stationary:
        conclusion = "stationary"
        action = "No transformation needed - proceed with forecasting"
    elif not adf_stationary and not kpss_stationary:
        conclusion = "non-stationary"
        action = "Apply first-order differencing"
    elif adf_stationary and not kpss_stationary:
        conclusion = "trend-stationary"
        action = "Remove deterministic trend (detrend)"
    else:  # not adf_stationary and kpss_stationary
        conclusion = "difference-stationary"
        action = "Apply first-order differencing"
    
    return {
        "adf_test": adf_result,
        "kpss_test": kpss_result,
        "combined_conclusion": conclusion,
        "recommended_action": action,
        "interpretation": (
            f"Combined assessment: Series is {conclusion}. "
            f"ADF: {'stationary' if adf_stationary else 'non-stationary'} "
            f"(p={adf_result['p_value']:.4f}), "
            f"KPSS: {'stationary' if kpss_stationary else 'non-stationary'} "
            f"(p={kpss_result['p_value']:.4f}). "
            f"{action}."
        )
    }


# ================================================================================
# HETEROSKEDASTICITY TESTS
# ================================================================================

def breusch_pagan_test(residuals: np.ndarray, 
                       exog: np.ndarray) -> Dict[str, Any]:
    """
    Breusch-Pagan test for heteroskedasticity.
    
    H0: Homoskedasticity (constant variance)
    H1: Heteroskedasticity (non-constant variance)
    
    Critical for regression validity - heteroskedasticity violates OLS assumptions.
    """
    if not STATSMODELS_AVAILABLE:
        return {"error": "statsmodels not installed"}
    
    try:
        # Run Breusch-Pagan test
        lm_statistic, lm_pvalue, f_statistic, f_pvalue = het_breuschpagan(residuals, exog)
        
        has_heteroskedasticity = lm_pvalue < 0.05
        
        return {
            "test": "Breusch-Pagan",
            "lm_statistic": float(lm_statistic),
            "lm_pvalue": float(lm_pvalue),
            "f_statistic": float(f_statistic),
            "f_pvalue": float(f_pvalue),
            "has_heteroskedasticity": has_heteroskedasticity,
            "interpretation": (
                f"Breusch-Pagan test: {'Heteroskedasticity detected' if has_heteroskedasticity else 'Homoskedasticity'} "
                f"(p={lm_pvalue:.4f}). "
                f"{'Variance is non-constant - consider robust standard errors or transformation' if has_heteroskedasticity else 'Constant variance assumption satisfied'}."
            ),
            "recommendation": (
                "Use robust standard errors (White's correction) or log-transform dependent variable" 
                if has_heteroskedasticity else
                "OLS assumptions satisfied - standard errors are valid"
            )
        }
    
    except Exception as e:
        return {"error": f"Breusch-Pagan test failed: {str(e)}"}


def white_test(residuals: np.ndarray,
               exog: np.ndarray) -> Dict[str, Any]:
    """
    White's test for heteroskedasticity (more general than Breusch-Pagan).
    
    Tests for any form of heteroskedasticity, not just linear.
    """
    if not STATSMODELS_AVAILABLE:
        return {"error": "statsmodels not installed"}
    
    try:
        # Run White's test
        lm_statistic, lm_pvalue, f_statistic, f_pvalue = het_white(residuals, exog)
        
        has_heteroskedasticity = lm_pvalue < 0.05
        
        return {
            "test": "White",
            "lm_statistic": float(lm_statistic),
            "lm_pvalue": float(lm_pvalue),
            "f_statistic": float(f_statistic),
            "f_pvalue": float(f_pvalue),
            "has_heteroskedasticity": has_heteroskedasticity,
            "interpretation": (
                f"White's test: {'Heteroskedasticity detected' if has_heteroskedasticity else 'Homoskedasticity'} "
                f"(p={lm_pvalue:.4f}). "
                f"More general than Breusch-Pagan. "
                f"{'Use heteroskedasticity-consistent standard errors' if has_heteroskedasticity else 'Standard errors are valid'}."
            )
        }
    
    except Exception as e:
        return {"error": f"White's test failed: {str(e)}"}


def test_wrights_law_heteroskedasticity(df: pd.DataFrame, product: str) -> Dict[str, Any]:
    """
    Test Wright's Law regression for heteroskedasticity.
    Ensures learning rate estimates have valid standard errors.
    """
    if product and "product_name" in df.columns:
        sub = df[df["product_name"] == product].copy()
    else:
        sub = df.copy()
    
    if len(sub) < 5:
        return {"error": "Insufficient data"}
    
    try:
        years = sub['year'].values
        costs = sub['value'].values
        
        # Fit Wright's Law (log-linear)
        X = years.reshape(-1, 1)
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        y = np.log(costs)
        
        # OLS regression
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, y)
        
        # Calculate residuals
        y_pred = slope * years + intercept
        residuals = y - y_pred
        
        # Breusch-Pagan test
        bp_result = breusch_pagan_test(residuals, X_with_intercept)
        
        # White test
        white_result = white_test(residuals, X_with_intercept)
        
        return {
            "product": product,
            "breusch_pagan": bp_result,
            "white": white_result,
            "has_heteroskedasticity": (
                bp_result.get('has_heteroskedasticity', False) or 
                white_result.get('has_heteroskedasticity', False)
            ),
            "interpretation": (
                f"Wright's Law regression for {product}: "
                f"{'Heteroskedasticity detected - standard errors may be unreliable' if bp_result.get('has_heteroskedasticity') else 'Homoskedastic - standard errors valid'}. "
                f"Consider robust standard errors for confidence intervals."
            )
        }
    
    except Exception as e:
        return {"error": f"Heteroskedasticity test failed: {str(e)}"}


# ================================================================================
# WRAPPER FUNCTION FOR ALL STATISTICAL TESTS
# ================================================================================

def run_all_statistical_tests(df: pd.DataFrame, 
                              product: str = None,
                              p_values: List[float] = None) -> Dict[str, Any]:
    """
    Run comprehensive statistical test suite.
    """
    results = {}
    
    # Multiple comparison correction
    if p_values and len(p_values) > 1:
        results['bonferroni'] = apply_bonferroni_correction(p_values)
        results['benjamini_hochberg'] = apply_benjamini_hochberg(p_values)
    
    # Time series tests
    if product and 'product_name' in df.columns:
        product_df = df[df['product_name'] == product]
        if 'value' in product_df.columns and len(product_df) >= 10:
            series = product_df['value']
            results['stationarity'] = combined_stationarity_assessment(series)
    
    # Heteroskedasticity tests (for Wright's Law)
    if product:
        results['heteroskedasticity'] = test_wrights_law_heteroskedasticity(df, product)
    
    return results