#!/usr/bin/env python3
"""
Unified Public API
Re-exports all major functions for external use
"""

# Core extraction functions
from extraction_core import (
    extract_data_single_output,
    extract_data_batch_output,
    get_availability_status
)

# Query utilities
from query_utils import (
    validate_query,
    validate_api_key,
    create_query,
    extract_entities_from_text,
    extract_time_range,
    build_search_terms
)

# Market knowledge
from market_knowledge import (
    find_market_knowledge,
    get_regional_multiplier,
    MARKET_KNOWLEDGE_BASE,
    ENTITY_MAPPING,
    METRIC_MAPPING
)

# Calibration
from calibration import calibrate_any_data, UniversalCalibrator

# Web extraction
from web_extraction import (
    search_web_sources,
    extract_from_url,
    calculate_table_relevance,
    CRAWL4AI_AVAILABLE
)

# Table utilities
from table_utils import (
    create_realistic_table_data_with_calibration,
    standardize_web_table_with_calibration,
    export_to_csv,
    export_to_json
)

# LLM integration
from llm_integration import (
    generate_llm_data_with_calibration,
    generate_analysis_with_calibration,
    GROQ_AVAILABLE
)

# Data models
from models import (
    DataQuery,
    SingleTableResult,
    BatchTableResult,
    TableData,
    ValidationPoint,
    CalibrationResult,
    MarketKnowledge
)

# Version info
__version__ = "1.0.0"
__author__ = "Smart Data Analyzer Team"

# Public API
__all__ = [
    # Main extraction
    'extract_data_single_output',
    'extract_data_batch_output',
    
    # Validation
    'validate_query',
    'validate_api_key',
    
    # Query processing
    'create_query',
    'extract_entities_from_text',
    'extract_time_range',
    'build_search_terms',
    
    # Market intelligence
    'find_market_knowledge',
    'get_regional_multiplier',
    'MARKET_KNOWLEDGE_BASE',
    
    # Calibration
    'calibrate_any_data',
    'UniversalCalibrator',
    
    # Web extraction
    'search_web_sources',
    'extract_from_url',
    'calculate_table_relevance',
    
    # Table utilities
    'create_realistic_table_data_with_calibration',
    'standardize_web_table_with_calibration',
    'export_to_csv',
    'export_to_json',
    
    # LLM
    'generate_llm_data_with_calibration',
    'generate_analysis_with_calibration',
    
    # Models
    'DataQuery',
    'SingleTableResult',
    'BatchTableResult',
    'TableData',
    
    # Status
    'get_availability_status',
    'CRAWL4AI_AVAILABLE',
    'GROQ_AVAILABLE',
]


# Convenience functions
def quick_extract(query: str, api_key: str, max_urls: int = 5):
    """
    Quick extraction helper for simple use cases
    
    Example:
        result = quick_extract("solar capacity China 2010-2025", "your_api_key")
    """
    import asyncio
    return asyncio.run(
        extract_data_single_output(query, api_key, max_urls)
    )


def batch_extract(queries: list, api_key: str, max_urls: int = 5):
    """
    Batch extraction helper
    
    Example:
        queries = [
            {"Entity_Name": "Solar", "Region": "China"},
            {"Entity_Name": "Wind", "Region": "USA"}
        ]
        result = batch_extract(queries, "your_api_key")
    """
    import asyncio
    return asyncio.run(
        extract_data_batch_output(queries, api_key, max_urls)
    )