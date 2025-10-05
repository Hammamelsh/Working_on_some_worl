#!/usr/bin/env python3
"""
Market Knowledge Module
Comprehensive market intelligence database with validated datasets
"""

from typing import Dict, Optional, List
from models import MarketKnowledge, DataQuery
from config import REGION_ALIASES


# Comprehensive market database with validated datasets
MARKET_KNOWLEDGE_BASE: Dict[str, MarketKnowledge] = {
    'electricity_cost_residential': MarketKnowledge(
        base_value=0.13, unit='USD per kWh', typical_range=(0.04, 0.45),
        growth_pattern='linear', annual_growth_rate=0.025, volatility=0.08,
        regional_multipliers={
            'usa': 1.0, 'china': 0.32, 'india': 0.28, 'germany': 2.7, 'japan': 1.9,
            'uk': 1.5, 'france': 1.4, 'canada': 0.78, 'australia': 1.7, 'norway': 0.85,
            'brazil': 0.65, 'russia': 0.45, 'global': 0.85
        }
    ),
    
    'solar_capacity': MarketKnowledge(
        base_value=28, unit='GW', typical_range=(1, 1500),
        growth_pattern='exponential', annual_growth_rate=0.32, volatility=0.18,
        regional_multipliers={
            'usa': 1.0, 'china': 8.2, 'india': 1.8, 'germany': 2.1, 'japan': 2.5,
            'australia': 0.9, 'spain': 0.8, 'italy': 0.7, 'global': 1.0
        },
        disruption_threshold_year=2010
    ),
    
    'electric_vehicle_sales': MarketKnowledge(
        base_value=0.15, unit='Million Units', typical_range=(0.005, 22),
        growth_pattern='exponential', annual_growth_rate=0.78, volatility=0.32,
        regional_multipliers={
            'usa': 1.0, 'china': 6.8, 'norway': 12.5, 'germany': 1.8, 'uk': 1.2,
            'netherlands': 2.5, 'sweden': 1.8, 'india': 0.08, 'japan': 0.6, 'global': 1.0
        },
        disruption_threshold_year=2015, seasonal_pattern=True
    ),
    
    'steel_production': MarketKnowledge(
        base_value=85, unit='Million Tons', typical_range=(20, 520),
        growth_pattern='cyclical', annual_growth_rate=0.018, volatility=0.25,
        regional_multipliers={
            'usa': 1.0, 'china': 24.5, 'india': 2.2, 'japan': 1.3, 'germany': 0.52,
            'russia': 0.85, 'south_korea': 0.82, 'brazil': 0.42, 'global': 1.0
        }
    )
}


# Entity and metric mapping for intelligent lookup
ENTITY_MAPPING: Dict[str, List[str]] = {
    'electricity': ['electricity_cost_residential', 'electricity_cost_industrial'],
    'solar': ['solar_capacity'],
    'electric vehicle': ['electric_vehicle_sales'],
    'steel': ['steel_production', 'steel_price'],
}


METRIC_MAPPING: Dict[str, List[str]] = {
    'cost': ['_cost_', '_price'],
    'price': ['_price', '_cost_'],
    'production': ['_production'],
    'capacity': ['_capacity'],
    'sales': ['_sales']
}


def find_market_knowledge(query: DataQuery) -> Optional[MarketKnowledge]:
    """
    Advanced market knowledge lookup with fuzzy matching
    
    Pure function that searches market database for matching knowledge
    """
    entity_lower = query.entity_name.lower()
    metric_lower = query.metric.lower()
    unit_lower = query.unit.lower()
    
    # Primary search: entity mapping
    for entity_key, market_keys in ENTITY_MAPPING.items():
        if entity_key not in entity_lower:
            continue
            
        for market_key in market_keys:
            if market_key not in MARKET_KNOWLEDGE_BASE:
                continue
                
            market_data = MARKET_KNOWLEDGE_BASE[market_key]
            
            # Validate metric compatibility
            metric_match = _is_metric_compatible(metric_lower, market_key)
            
            # Validate unit compatibility
            unit_match = _is_unit_compatible(unit_lower, market_data.unit)
            
            if metric_match and unit_match:
                return market_data
        
        # Return first match if specific validation fails
        if market_keys and market_keys[0] in MARKET_KNOWLEDGE_BASE:
            return MARKET_KNOWLEDGE_BASE[market_keys[0]]
    
    return None


def _is_metric_compatible(metric_lower: str, market_key: str) -> bool:
    """Check if metric is compatible with market key"""
    if not metric_lower:
        return True
    
    metric_patterns = METRIC_MAPPING.get(metric_lower, [])
    return any(pattern.strip('_') in market_key for pattern in metric_patterns)


def _is_unit_compatible(unit_lower: str, market_unit: str) -> bool:
    """Check if unit is compatible with market unit"""
    if not unit_lower:
        return True
    
    market_unit_lower = market_unit.lower()
    return (unit_lower in market_unit_lower or 
            any(word in market_unit_lower for word in unit_lower.split()))


def get_regional_multiplier(market_data: MarketKnowledge, region: str) -> float:
    """
    Get regional adjustment multiplier with alias support
    
    Pure function that resolves region to multiplier
    """
    return market_data.get_regional_multiplier(region, REGION_ALIASES)


def get_market_knowledge_stats() -> Dict[str, int]:
    """Get statistics about market knowledge base"""
    return {
        'total_entities': len(MARKET_KNOWLEDGE_BASE),
        'entity_mappings': len(ENTITY_MAPPING),
        'metric_mappings': len(METRIC_MAPPING),
        'unique_regions': len(set(
            region 
            for mk in MARKET_KNOWLEDGE_BASE.values() 
            for region in mk.regional_multipliers.keys()
        ))
    }