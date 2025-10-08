"""
STELLAR Framework - Configuration Constants
All hardcoded values centralized for easy maintenance
"""

# ================================================================================
# VALIDATION THRESHOLDS
# ================================================================================

# Investment Grade
INVESTMENT_GRADE_THRESHOLD = 0.95  # 95% pass rate required
NEAR_INVESTMENT_GRADE_THRESHOLD = 0.90
GOOD_QUALITY_THRESHOLD = 0.80
ACCEPTABLE_THRESHOLD = 0.70
MINIMUM_THRESHOLD = 0.60

# Wright's Law
WRIGHT_LAW_LEARNING_RATE_MIN = 0.15  # 15% per doubling
WRIGHT_LAW_LEARNING_RATE_MAX = 0.30  # 30% per doubling
WRIGHT_LAW_R_SQUARED_MIN = 0.60  # Minimum model fit
WRIGHT_LAW_P_VALUE_MAX = 0.05  # Statistical significance

# Technology-Specific Learning Rates
LEARNING_RATES = {
    'solar': {'min': 0.15, 'max': 0.30, 'expected': 0.20},
    'battery': {'min': 0.10, 'max': 0.25, 'expected': 0.18},
    'lithium_ion': {'min': 0.10, 'max': 0.25, 'expected': 0.18},
    'wind': {'min': 0.05, 'max': 0.15, 'expected': 0.10},
    'ev': {'min': 0.08, 'max': 0.20, 'expected': 0.12}
}

# S-Curve
SCURVE_R_SQUARED_MIN = 0.70
SCURVE_TIPPING_POINT = 0.10  # 10% adoption
SCURVE_SATURATION = 0.90  # 90% adoption
SCURVE_RAPID_TRANSITION_YEARS = 10  # Seba's <10 years prediction

# Commodity Cycles
COMMODITY_ANNUAL_GROWTH_MIN = 0.03  # 3%
COMMODITY_ANNUAL_GROWTH_MAX = 0.07  # 7%

# ================================================================================
# DATA QUALITY THRESHOLDS
# ================================================================================

COMPLETENESS_THRESHOLD = 0.95  # 95% data completeness required
CRITICAL_FIELD_COMPLETENESS = 1.00  # 100% for critical fields
DUPLICATE_PENALTY = 0.10  # 10% score penalty
TYPE_ISSUE_PENALTY_PER_ERROR = 0.02  # 2% per type error

# Outlier Detection
IQR_MULTIPLIER = 1.5  # Standard IQR outlier threshold
GRUBBS_ALPHA = 0.05  # 5% significance level
OUTLIER_WARNING_THRESHOLD = 10  # Warn if >10 outliers

# Year-over-Year Changes
MAX_YOY_CHANGE_DEFAULT = 3.0  # 300% maximum change
MAX_YOY_CHANGE_EMERGING_TECH = 5.0  # 500% for emerging tech
MAX_YOY_CHANGE_MATURE = 1.0  # 100% for mature tech

# ================================================================================
# STATISTICAL PARAMETERS
# ================================================================================

BOOTSTRAP_ITERATIONS = 1000
CONFIDENCE_LEVEL = 0.95
STATISTICAL_POWER_REQUIRED = 0.80  # 80% power
EFFECT_SIZE_MEDIUM = 0.5  # Cohen's d

# Multiple Comparisons
BONFERRONI_ALPHA = 0.05
FALSE_DISCOVERY_RATE = 0.05  # For Benjamini-Hochberg

# Time Series
ADF_MAX_LAGS = 12
KPSS_REGRESSION = 'c'  # constant
STATIONARITY_P_VALUE = 0.05

# Structural Breaks
CHOW_TEST_ALPHA = 0.05
MINIMUM_SEGMENT_SIZE = 3  # Minimum data points per segment

# ================================================================================
# REGIONAL PARAMETERS
# ================================================================================

REGIONAL_DRIVING_PATTERNS = {
    'usa': {'km_per_vehicle_per_year': 18000, 'liters_per_100km': 10.0},
    'china': {'km_per_vehicle_per_year': 12000, 'liters_per_100km': 7.0},
    'europe': {'km_per_vehicle_per_year': 13000, 'liters_per_100km': 6.5},
    'india': {'km_per_vehicle_per_year': 10000, 'liters_per_100km': 6.0},
    'japan': {'km_per_vehicle_per_year': 9000, 'liters_per_100km': 5.5},
    'global': {'km_per_vehicle_per_year': 15000, 'liters_per_100km': 8.0}
}

# Regional Market Share Expectations
REGIONAL_MARKET_SHARES = {
    'two_wheeler_ev': {
        'india': {'min': 0.15, 'max': 0.40},
        'china': {'min': 0.30, 'max': 0.60}
    },
    'passenger_vehicle_ev': {
        'china': {'min': 0.30, 'max': 0.60},
        'europe': {'min': 0.15, 'max': 0.35},
        'usa': {'min': 0.10, 'max': 0.25}
    }
}

# ================================================================================
# COST PARITY & OIL DISPLACEMENT
# ================================================================================

COST_PARITY_THRESHOLD = 70.0  # $/MWh
TRANSPORT_OIL_BAND = (55, 60)  # Million barrels/day
TOTAL_OIL_BAND = (90, 110)  # Million barrels/day
LITERS_PER_BARREL = 159.0

# ================================================================================
# CRISIS PERIODS
# ================================================================================

CRISIS_WINDOWS = {
    2008: {"name": "Global Financial Crisis", "expected_impact": 0.15},
    2009: {"name": "GFC Recovery", "expected_impact": 0.10},
    2020: {"name": "COVID-19 Pandemic", "expected_impact": 0.08},
    2021: {"name": "COVID Recovery", "expected_impact": 0.12},
    2022: {"name": "Russia-Ukraine Conflict", "expected_impact": 0.20}
}

# ================================================================================
# DATA SOURCE TIERS
# ================================================================================

TIER1_SOURCES = [
    'iea.org', 'eia.gov', 'bp.com/statistical', 'irena.org',
    'usgs.gov', 'worldbank.org', 'imf.org', 'un.org', 'opec.org'
]

TIER2_SOURCES = [
    'bnef.com', 'statista.com', 'ourworldindata.org', 'ember-climate.org',
    'rystadenergy.com', 'woodmac.com', 'mckinsey.com', 'ieefa.org'
]

TIER3_SOURCES = [
    'wikipedia.org', 'tradingeconomics.com', 'indexmundi.com'
]

# ================================================================================
# SEBA DISRUPTION METRICS
# ================================================================================

# Disruption Velocity Thresholds
DISRUPTION_VELOCITY_SLOW = 0.05  # 5% annual change
DISRUPTION_VELOCITY_MODERATE = 0.15  # 15% annual change
DISRUPTION_VELOCITY_RAPID = 0.30  # 30% annual change
DISRUPTION_VELOCITY_EXPONENTIAL = 0.50  # 50% annual change

# Incumbent Vulnerability
INCUMBENT_VULNERABILITY_LOW = 0.3
INCUMBENT_VULNERABILITY_MEDIUM = 0.6
INCUMBENT_VULNERABILITY_HIGH = 0.8

# Technology Lifecycle Stages
LIFECYCLE_THRESHOLDS = {
    'emerging': {'age_years': 5, 'adoption': 0.01},
    'growth': {'age_years': 10, 'adoption': 0.10},
    'mainstream': {'age_years': 20, 'adoption': 0.50},
    'mature': {'age_years': 30, 'adoption': 0.90}
}

# ================================================================================
# FORECASTING PARAMETERS
# ================================================================================

DEFAULT_FORECAST_YEARS = 5
SCENARIO_PESSIMISTIC_LR = 0.10
SCENARIO_BASE_LR = 0.20
SCENARIO_OPTIMISTIC_LR = 0.30

# ================================================================================
# VALIDATOR CONFIGURATION
# ================================================================================

TOTAL_VALIDATORS = 35
VALIDATOR_TIMEOUT_SECONDS = 30
MAX_PRODUCTS_TO_ANALYZE = 5

# ================================================================================
# EXPORT & REPORTING
# ================================================================================

PDF_PAGE_SIZE = 'letter'
PDF_MARGIN_INCHES = 1.0
MAX_EXPORT_ROWS = 1000000

# ================================================================================
# PERFORMANCE & CACHING
# ================================================================================

CACHE_MAX_SIZE = 128
CACHE_TTL_SECONDS = 3600  # 1 hour
MAX_WORKERS = 4  # For parallel processing