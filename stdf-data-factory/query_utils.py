#!/usr/bin/env python3
"""
Query Utilities Module
Functions for query validation, parsing, and entity extraction
"""

import re
from typing import Dict, Tuple, Union, List
from models import DataQuery, QueryInput
from config import DISRUPTOR_KEYWORDS, DEFAULT_TIME_RANGE


def validate_api_key(api_key: str) -> bool:
    """Validate API key format"""
    return bool(api_key and len(api_key) > 20)


def validate_query(query: Union[str, Dict, List, None]) -> Tuple[bool, str]:
    """
    Enhanced query validation with detailed feedback
    
    Pure function that validates query structure
    """
    if query is None:
        return False, "No query provided"
    
    if isinstance(query, str):
        is_valid = bool(query.strip())
        return is_valid, "" if is_valid else "Query cannot be empty"
    
    if isinstance(query, dict):
        if not query.get('Entity_Name', '').strip():
            return False, "Entity_Name is required"
        return True, ""
    
    if isinstance(query, list):
        if not query:
            return False, "Query array cannot be empty"
        
        valid_count = sum(
            1 for q in query 
            if isinstance(q, dict) and q.get('Entity_Name', '').strip()
        )
        
        if valid_count == 0:
            return False, "No valid queries found"
        elif valid_count < len(query):
            return True, f"{valid_count}/{len(query)} queries are valid"
        else:
            return True, f"All {len(query)} queries are valid"
    
    return False, "Invalid query format"


def create_query(query_input: QueryInput) -> DataQuery:
    """
    Create DataQuery from input with intelligent parsing
    
    Pure function that transforms input to structured query
    """
    if isinstance(query_input, dict):
        return _create_query_from_dict(query_input)
    elif isinstance(query_input, str):
        return _create_query_from_string(query_input)
    else:
        return DataQuery(entity_name=str(query_input))


def _create_query_from_dict(query_dict: Dict) -> DataQuery:
    """Create query from dictionary"""
    time_range = extract_time_range_from_entity(
        query_dict.get('Entity_Name', '')
    )
    
    return DataQuery(
        entity_name=query_dict.get('Entity_Name', ''),
        entity_type=query_dict.get('Entity_Type', ''),
        region=query_dict.get('Region', ''),
        metric=query_dict.get('Metric', ''),
        unit=query_dict.get('Unit', ''),
        time_range=time_range,
        is_disruptor=query_dict.get('Is_Disruptor', False)
    )


def _create_query_from_string(query_str: str) -> DataQuery:
    """Create query from string with entity extraction"""
    entities = extract_entities_from_text(query_str)
    time_range = extract_time_range(query_str)
    
    return DataQuery(
        entity_name=query_str,
        region=entities.get('region', ''),
        metric=entities.get('metric', ''),
        unit=entities.get('unit', ''),
        entity_type=entities.get('entity', ''),
        time_range=time_range,
        is_disruptor=detect_disruptor_patterns(query_str)
    )


def extract_entities_from_text(text: str) -> Dict[str, str]:
    """
    Enhanced entity extraction with comprehensive region and unit mapping
    
    Pure function that extracts structured entities from text
    """
    entities = {}
    text_lower = text.lower()
    
    # Region extraction
    entities['region'] = _extract_region(text_lower)
    
    # Entity and metric extraction
    entities['entity'] = _extract_entity_type(text_lower)
    entities['metric'] = _extract_metric(text_lower)
    
    # Unit extraction
    entities['unit'] = _extract_unit(text_lower)
    
    return entities


def _extract_region(text_lower: str) -> str:
    """Extract region from text"""
    region_patterns = {
        'usa': r'\b(usa|us|united states|america|american)\b',
        'china': r'\b(china|chinese|prc|peoples republic)\b',
        'india': r'\b(india|indian|bharat)\b',
        'germany': r'\b(germany|german|deutschland)\b',
        'japan': r'\b(japan|japanese|nippon)\b',
        'global': r'\b(global|world|worldwide|international)\b'
    }
    
    for region, pattern in region_patterns.items():
        if re.search(pattern, text_lower):
            return region
    
    return ''


def _extract_entity_type(text_lower: str) -> str:
    """Extract entity type from text"""
    from market_knowledge import ENTITY_MAPPING
    
    for entity_key in ENTITY_MAPPING.keys():
        if entity_key in text_lower:
            return entity_key
    
    return ''


def _extract_metric(text_lower: str) -> str:
    """Extract metric from text"""
    from market_knowledge import METRIC_MAPPING
    
    for metric_key in METRIC_MAPPING.keys():
        if metric_key in text_lower:
            return metric_key
    
    return ''


def _extract_unit(text_lower: str) -> str:
    """Extract unit from text using patterns"""
    unit_patterns = [
        (r'\busd\s+per\s+kwh\b', 'USD per kWh'),
        (r'\bgigawatts?\b|gw\b', 'GW'),
        (r'\bmillion\s+units?\b', 'Million Units'),
        (r'\bpercent\b|%', 'Percent'),
        (r'\btons?\b', 'Tons'),
        (r'\bbarrels?\b', 'Barrels')
    ]
    
    for pattern, unit in unit_patterns:
        if re.search(pattern, text_lower):
            return unit
    
    return ''


def extract_time_range(text: str) -> Tuple[int, int]:
    """
    Extract time range with flexible patterns
    
    Pure function that parses time ranges from text
    """
    time_patterns = [
        r'(\d{4})\s*(?:to|until|-|through)\s*(\d{4})',
        r'from\s+(\d{4})\s+to\s+(\d{4})',
        r'between\s+(\d{4})\s+and\s+(\d{4})'
    ]
    
    for pattern in time_patterns:
        match = re.search(pattern, text.lower())
        if match:
            start_year = max(2000, int(match.group(1)))
            end_year = min(2030, int(match.group(2)))
            return (start_year, end_year)
    
    return DEFAULT_TIME_RANGE


def extract_time_range_from_entity(entity_name: str) -> Tuple[int, int]:
    """Extract time range from entity name"""
    if not entity_name:
        return DEFAULT_TIME_RANGE
    
    match = re.search(
        r'(\d{4})\s*(?:to|until|-|through)\s*(\d{4})', 
        entity_name.lower()
    )
    
    if match:
        start_year = max(2000, int(match.group(1)))
        end_year = min(2030, int(match.group(2)))
        return (start_year, end_year)
    
    return DEFAULT_TIME_RANGE


def detect_disruptor_patterns(text: str) -> bool:
    """
    Detect disruptive technology patterns
    
    Pure function that identifies disruptor keywords
    """
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in DISRUPTOR_KEYWORDS)


def build_search_terms(query: DataQuery) -> str:
    """
    Build intelligent search terms with data-specific keywords
    
    Pure function that creates optimized search query
    """
    terms = []
    
    if query.entity_name:
        terms.append(query.entity_name)
    if query.region and query.region.lower() != 'global':
        terms.append(query.region)
    if query.metric:
        terms.append(query.metric)
    
    terms.extend(["data", "statistics", "historical", "annual"])
    
    start_year, end_year = query.time_range
    if end_year - start_year <= 5:
        terms.append(f"{start_year}-{end_year}")
    else:
        terms.extend([str(start_year), str(end_year)])
    
    if query.unit and len(query.unit) <= 20:
        terms.append(f'"{query.unit}"')
    
    return ' '.join(terms)