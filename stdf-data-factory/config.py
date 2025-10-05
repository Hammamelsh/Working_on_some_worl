#!/usr/bin/env python3
"""
Configuration Module
Centralized configuration management for the Smart Data Analyzer
"""

from dataclasses import dataclass
from typing import Dict, Tuple

# Application Constants
DEFAULT_MAX_URLS = 5
DEFAULT_ENABLE_CRAWL = True
DEFAULT_TIME_RANGE = (2010, 2025)
DEFAULT_MIN_TABLE_ROWS = 3
DEFAULT_MIN_TABLE_COLS = 2

# Web Scraping Configuration
WEB_REQUEST_TIMEOUT = 20
WEB_CRAWL_TIMEOUT = 30000
MAX_TEXT_LENGTH = 500

# LLM Configuration
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS_VALIDATION = 100
LLM_MAX_TOKENS_GENERATION = 2000
LLM_MAX_TOKENS_ANALYSIS = 400

LLM_MODELS = [
    "llama-3.1-8b-instant",
    "llama3-groq-70b-8192-tool-use-preview",
    "gemma2-9b-it"
]

# Economic Indicators (GDP per capita ratios, USA = 1.0 baseline)
ECONOMIC_INDICATORS = {
    'usa': 1.0, 'china': 0.28, 'germany': 0.85, 'japan': 0.75,
    'india': 0.05, 'uk': 0.78, 'france': 0.74, 'canada': 0.82,
    'australia': 0.89, 'brazil': 0.18, 'russia': 0.24, 'global': 0.45,
    # PPP adjustments
    'ppp_china': 0.6, 'ppp_india': 0.25, 'ppp_russia': 0.4, 'ppp_brazil': 0.35
}

# Regional Aliases
REGION_ALIASES = {
    'united_states': 'usa', 'america': 'usa', 'us': 'usa',
    'prc': 'china', 'chinese': 'china',
    'deutschland': 'germany', 'german': 'germany',
    'world': 'global', 'worldwide': 'global'
}

# Market Maturity Factors
MATURITY_FACTORS = {
    'usa': 1.0, 'germany': 1.1, 'japan': 1.05, 'uk': 1.0, 'france': 1.0,
    'china': 0.7, 'india': 0.4, 'brazil': 0.6, 'russia': 0.65, 'global': 0.8
}

# Technology Adoption Factors
TECH_FACTORS = {
    'china': 1.5, 'usa': 1.0, 'germany': 0.8, 'japan': 0.7, 'india': 1.2
}

# Crisis Years and Impact
CRISIS_YEARS = {2008: -0.18, 2009: -0.15, 2020: -0.09, 2022: 0.05}

# Unit Type Mappings
UNIT_MAPPINGS = {
    'monetary': (['usd', 'dollar', 'eur', 'price', 'cost'], (10, 100000), 0.08),
    'percentage': (['percent', '%', 'rate', 'ratio'], (0, 100), 0.05),
    'power': (['gw', 'mw', 'kw', 'watt', 'capacity'], (1, 1000), 0.12),
    'mass': (['ton', 'kg', 'lb', 'gram', 'weight', 'mass'], (100, 100000), 0.15),
    'quantity': (['million', 'billion', 'thousand', 'unit'], (0.1, 1000), 0.18),
    'energy': (['kwh', 'energy', 'joule', 'btu'], (0.01, 10), 0.08),
    'volume': (['barrel', 'liter', 'gallon', 'volume'], (10, 10000), 0.12),
    'index': (['index', 'score', 'rating'], (50, 200), 0.06)
}

# Trusted Data Sources
TRUSTED_DOMAINS = [
    'ourworldindata.org', 'worldbank.org', 'data.gov', 'eia.gov',
    'iea.org', 'irena.org', 'statista.com', 'tradingeconomics.com',
    'census.gov', 'bls.gov', 'fred.stlouisfed.org', 'imf.org',
    'oecd.org', 'who.int', 'fao.org', 'un.org', 'europa.eu'
]

# International Organizations and Confidence Scores
INTL_ORGS_CONFIDENCE = {
    'worldbank': 0.35, 'oecd': 0.35, 'imf': 0.3, 'iea': 0.35, 'irena': 0.35,
    'who': 0.3, 'fao': 0.3, 'wto': 0.25, 'un.org': 0.3, 'europa.eu': 0.25
}

# Data Platforms Confidence Scores
DATA_PLATFORMS_CONFIDENCE = {
    'ourworldindata': 0.3, 'statista': 0.25, 'tradingeconomics': 0.2,
    'knoema': 0.15, 'ceicdata': 0.15, 'fred.stlouisfed': 0.3
}

# Skip Domains for Web Scraping
SKIP_DOMAINS = [
    'facebook.com', 'twitter.com', 'youtube.com', 'ads.',
    'pinterest.com', 'instagram.com', 'linkedin.com', 'reddit.com', 'quora.com'
]

# Disruptor Keywords
DISRUPTOR_KEYWORDS = [
    'electric vehicle', 'ev', 'solar', 'renewable', 'wind energy', 'battery',
    'disrupt', 'revolutionary', 'breakthrough', 'exponential', 'rapid adoption',
    'clean energy', 'sustainable', 'innovation', 'transformation'
]

# Streamlit Styling
CUSTOM_CSS = """
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .workflow-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 12px;
        border: 2px solid #dee2e6;
        margin: 1rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .workflow-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 20px rgba(102,126,234,0.15);
        transform: translateY(-2px);
    }
    .market-intelligence-badge {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    .web-extracted-badge {
        background: linear-gradient(135deg, #007bff 0%, #17a2b8 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
</style>
"""


@dataclass(frozen=True)
class AppConfig:
    """Application configuration container"""
    max_urls: int = DEFAULT_MAX_URLS
    enable_crawl: bool = DEFAULT_ENABLE_CRAWL
    time_range: Tuple[int, int] = DEFAULT_TIME_RANGE
    min_table_rows: int = DEFAULT_MIN_TABLE_ROWS
    min_table_cols: int = DEFAULT_MIN_TABLE_COLS
    
    @classmethod
    def create(cls, **kwargs) -> 'AppConfig':
        """Factory method for creating configuration"""
        return cls(**{k: v for k, v in kwargs.items() if k in cls.__dataclass_fields__})