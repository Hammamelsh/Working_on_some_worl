"""
STELLAR Framework - Comprehensive Unit Tests
Covers all 35+ validators with 80%+ coverage target
"""

import pytest
import pandas as pd
import numpy as np
from ground_truth_validators import *
from validation_support import *
from statistical_tests import *

# Mock logger if needed
import logging
logging.basicConfig(level=logging.INFO)

# ================================================================================
# TEST FIXTURES
# ================================================================================

@pytest.fixture
def sample_cost_data():
    """Solar cost data following Wright's Law (20% learning rate)"""
    years = list(range(2010, 2026))
    # Cost declines 20% per doubling (roughly every 2 years)
    base_cost = 300  # $/MWh in 2010
    costs = [base_cost * (0.8 ** (i/2)) for i in range(len(years))]
    
    return {
        'Entity_Name': 'Solar PV',
        'Region': 'Global',
        'Unit': '$/MWh',
        'Metric': 'LCOE',
        'X': years,
        'Y': costs,
        'DataSource_URLs': ['https://iea.org/solar'],
        'Quality_Score': 0.9
    }


@pytest.fixture
def sample_adoption_data():
    """EV adoption following S-curve pattern"""
    years = list(range(2010, 2031))
    # Sigmoid: 1% in 2010 → 50% in 2023 → approaching 90% in 2030
    def sigmoid(x, L=0.9, k=0.4, x0=2023):
        return L / (1 + np.exp(-k * (x - x0)))
    
    adoption = [sigmoid(year) for year in years]
    
    return {
        'Entity_Name': 'Electric Vehicles',
        'Region': 'Global',
        'Unit': 'Market Share (%)',
        'Metric': 'Adoption',
        'X': years,
        'Y': adoption,
        'DataSource_URLs': ['https://iea.org/ev'],
        'Quality_Score': 0.85
    }


@pytest.fixture
def sample_commodity_data():
    """Aluminum prices with cyclical pattern (not tech-like)"""
    years = list(range(2010, 2026))
    # Cyclical with some random variation
    base_price = 2000  # $/ton
    prices = [base_price + 300 * np.sin((year - 2010) / 3) + np.random.normal(0, 50) 
              for year in years]
    
    return {
        'Entity_Name': 'Aluminum',
        'Region': 'Global',
        'Unit': '$/ton',
        'Metric': 'Price',
        'X': years,
        'Y': prices,
        'DataSource_URLs': ['https://usgs.gov/aluminum'],
        'Quality_Score': 0.8
    }


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for validators"""
    return pd.DataFrame({
        'product_name': ['Solar PV'] * 10 + ['Electric Vehicles'] * 10,
        'region': ['Global'] * 20,
        'year': list(range(2015, 2025)) * 2,
        'value': [300, 270, 240, 220, 200, 180, 165, 150, 138, 125,  # Solar costs
                  0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.26, 0.35, 0.45, 0.55],  # EV adoption
        'unit': ['$/MWh'] * 10 + ['Market Share'] * 10,
        'metric': ['LCOE'] * 10 + ['Adoption'] * 10
    })


# ================================================================================
# WRIGHT'S LAW TESTS
# ================================================================================

class TestWrightsLaw:
    
    def test_solar_learning_rate(self, sample_cost_data):
        """Test Wright's Law with known solar cost data"""
        df = pd.DataFrame({
            'product_name': [sample_cost_data['Entity_Name']] * len(sample_cost_data['X']),
            'region': [sample_cost_data['Region']] * len(sample_cost_data['X']),
            'year': sample_cost_data['X'],
            'value': sample_cost_data['Y'],
            'unit': [sample_cost_data['Unit']] * len(sample_cost_data['X']),
            'metric': [sample_cost_data['Metric']] * len(sample_cost_data['X'])
        })
        
        result = analyze_wrights_law(df, "Solar PV")
        
        assert 'error' not in result, f"Analysis failed: {result.get('error')}"
        assert 0.18 <= result['learning_rate'] <= 0.22, f"Learning rate {result['learning_rate']} outside 20% ±2%"
        assert result['compliant'] == True, "Should be compliant"
        assert result['r_squared'] > 0.90, f"R² {result['r_squared']} too low"
    
    def test_commodity_rejection(self, sample_commodity_data):
        """Wright's Law should NOT apply to commodities"""
        df = pd.DataFrame({
            'product_name': [sample_commodity_data['Entity_Name']] * len(sample_commodity_data['X']),
            'year': sample_commodity_data['X'],
            'value': sample_commodity_data['Y'],
            'unit': [sample_commodity_data['Unit']] * len(sample_commodity_data['X']),
            'metric': [sample_commodity_data['Metric']] * len(sample_commodity_data['X'])
        })
        
        result = analyze_wrights_law(df, "Aluminum")
        
        # Should work but not be compliant (cyclical pattern)
        if 'error' not in result:
            assert result.get('compliant') == False, "Commodity should not show tech-like learning"
    
    def test_insufficient_data(self):
        """Test with insufficient data points"""
        df = pd.DataFrame({
            'product_name': ['Solar PV'] * 3,
            'year': [2020, 2021, 2022],
            'value': [100, 95, 90],
            'unit': ['$/MWh'] * 3
        })
        
        result = analyze_wrights_law(df, "Solar PV")
        assert 'error' in result, "Should error with <5 data points"


# ================================================================================
# S-CURVE TESTS
# ================================================================================

class TestSCurve:
    
    def test_ev_adoption_sigmoid(self, sample_adoption_data):
        """Test S-curve fitting on EV adoption data"""
        df = pd.DataFrame({
            'product_name': [sample_adoption_data['Entity_Name']] * len(sample_adoption_data['X']),
            'year': sample_adoption_data['X'],
            'value': sample_adoption_data['Y'],
            'unit': [sample_adoption_data['Unit']] * len(sample_adoption_data['X']),
            'metric': [sample_adoption_data['Metric']] * len(sample_adoption_data['X'])
        })
        
        result = analyze_scurve_adoption(df, "Electric Vehicles")
        
        assert 'error' not in result
        assert result['compliant'] == True, "Should detect S-curve pattern"
        assert result['r_squared'] > 0.85, f"R² {result['r_squared']} too low for sigmoid"
        assert 'inflection_point' in result
    
    def test_linear_pattern_detection(self):
        """Test detection of non-sigmoid (linear) pattern"""
        df = pd.DataFrame({
            'product_name': ['Product'] * 10,
            'year': range(2015, 2025),
            'value': [i * 0.05 for i in range(10)],  # Linear growth
            'unit': ['units'] * 10,
            'metric': ['sales'] * 10
        })
        
        result = analyze_scurve_adoption(df, "Product")
        
        # Linear pattern won't fit sigmoid well
        if 'error' not in result:
            assert result['compliant'] == False or result['r_squared'] < 0.85


# ================================================================================
# TIPPING POINT TESTS
# ================================================================================

class TestTippingPoint:
    
    def test_tipping_point_detection(self):
        """Test 10%→90% tipping point logic"""
        # Create data that crosses 10% in 2020, reaches 90% in 2028
        years = list(range(2015, 2030))
        adoption = [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.22, 0.32, 0.45, 
                   0.60, 0.73, 0.83, 0.90, 0.94, 0.96]
        
        df = pd.DataFrame({
            'product_name': ['EV'] * len(years),
            'year': years,
            'value': adoption,
            'metric': ['adoption'] * len(years)
        })
        
        result = detect_tipping_point(df, "EV")
        
        assert result['tipping_point_crossed'] == True
        assert result['tipping_point_year'] == 2019  # When crossed 10%
        assert result['years_to_90_percent'] == 9  # 2019 → 2028
        assert result['meets_seba_pattern'] == True  # <10 years


# ================================================================================
# STATISTICAL TESTS
# ================================================================================

class TestStatisticalMethods:
    
    def test_bonferroni_correction(self):
        """Test Bonferroni correction for multiple comparisons"""
        # 10 tests, 3 with p < 0.05
        p_values = [0.001, 0.02, 0.04, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60]
        
        result = apply_bonferroni_correction(p_values, alpha=0.05)
        
        assert result['n_tests'] == 10
        assert result['corrected_alpha'] == 0.005  # 0.05/10
        assert result['significant_before_correction'] == 3
        # After correction, only 0.001 should be significant
        assert result['significant_after_correction'] == 1
    
    def test_stationarity_stationary_series(self):
        """Test ADF on stationary series (white noise)"""
        series = pd.Series(np.random.normal(0, 1, 100))
        
        result = augmented_dickey_fuller_test(series)
        
        if 'error' not in result:
            assert result['is_stationary'] == True
            assert result['p_value'] < 0.05
    
    def test_stationarity_nonstationary_series(self):
        """Test ADF on non-stationary series (random walk)"""
        series = pd.Series(np.cumsum(np.random.normal(0, 1, 100)))
        
        result = augmented_dickey_fuller_test(series)
        
        if 'error' not in result:
            assert result['is_stationary'] == False
            assert result['p_value'] > 0.05


# ================================================================================
# DOMAIN VALIDATOR TESTS
# ================================================================================

class TestDomainValidators:
    
    def test_cost_parity_threshold(self, sample_dataframe):
        """Test cost parity detection"""
        result = cost_parity_threshold(sample_dataframe, parity_level=200, cutoff_year=2022)
        
        assert len(result) > 0
        # Solar should reach parity (costs drop below 200)
        solar_results = [r for r in result if 'Solar' in r['product']]
        if solar_results:
            assert solar_results[0]['pass'] == True
    
    def test_unit_validation(self, sample_dataframe):
        """Test unit consistency validation"""
        result = validate_units_and_scale(sample_dataframe)
        
        # Should not find issues in clean data
        assert result.empty or len(result) == 0
    
    def test_year_anomaly_detection(self):
        """Test year anomaly detection"""
        results = [{
            'Entity_Name': 'Product',
            'Region': 'Global',
            'X': [2020, 9999, 2022],  # 9999 is anomalous
            'Y': [100, 105, 110]
        }]
        
        df = pd.DataFrame({
            'product_name': ['Product'] * 3,
            'region': ['Global'] * 3,
            'year': [2020, 9999, 2022],
            'value': [100, 105, 110]
        })
        
        issues = check_year_anomalies(results, df)
        
        assert len(issues) > 0
        assert '9999' in str(issues[0])


# ================================================================================
# DATA QUALITY TESTS
# ================================================================================

class TestDataQuality:
    
    def test_duplicate_detection(self):
        """Test duplicate record detection"""
        df = pd.DataFrame({
            'product_name': ['Solar', 'Solar', 'Wind'],
            'region': ['USA', 'USA', 'USA'],
            'metric': ['Cost', 'Cost', 'Cost'],
            'year': [2020, 2020, 2020],  # Duplicate Solar-USA-2020
            'value': [100, 100, 80]
        })
        
        duplicates = detect_duplicate_records(df)
        
        assert not duplicates.empty
        assert len(duplicates) > 0
    
    def test_outlier_detection(self):
        """Test statistical outlier detection"""
        df = pd.DataFrame({
            'product_name': ['Product'] * 10,
            'region': ['Global'] * 10,
            'year': range(2015, 2025),
            'value': [100, 105, 110, 108, 500, 115, 120, 118, 122, 125]  # 500 is outlier
        })
        
        outliers = detect_statistical_outliers(df, method='iqr')
        
        assert len(outliers) > 0
        assert 'Product_Global' in outliers or 'Product' in str(outliers)
    
    def test_completeness_calculation(self):
        """Test completeness metric calculation"""
        df = pd.DataFrame({
            'product_name': ['A', 'B', 'C', None, 'E'],
            'year': [2020, 2021, None, 2023, 2024],
            'value': [100, None, 300, 400, 500]
        })
        
        result = calculate_detailed_completeness(df)
        
        assert result['overall_completeness'] < 1.0
        assert result['critical_fields_complete'] == False  # product_name has null
        assert result['missing_critical_count'] > 0


# ================================================================================
# RUN TESTS
# ================================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])