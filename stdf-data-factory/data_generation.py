#!/usr/bin/env python3
"""
Data Generation Module
Functions for generating realistic time series data with market intelligence
"""

import random
import math
from typing import List, Tuple, Optional
from models import DataQuery, TimeSeriesData, MarketKnowledge, CalibrationResult
from market_knowledge import find_market_knowledge, get_regional_multiplier
from config import CRISIS_YEARS


async def generate_realistic_time_series_with_calibration(
    query: DataQuery,
    api_key: Optional[str] = None
) -> Tuple[TimeSeriesData, Optional[CalibrationResult]]:
    """
    Generate highly realistic time series data with universal calibration
    
    Functional pipeline: query -> market data -> series -> calibration -> result
    """
    market_data = find_market_knowledge(query)
    start_year, end_year = query.time_range
    
    # Generate base series
    base_series = (
        generate_market_based_series(query, market_data, start_year, end_year)
        if market_data
        else generate_intelligent_fallback_series(query, start_year, end_year)
    )
    
    # Apply calibration if API key available
    if api_key:
        calibrated_series, calibration_result = await _apply_calibration(
            base_series, query, api_key
        )
        if calibration_result and calibration_result.confidence_score > 0.7:
            return calibrated_series, calibration_result
    
    return base_series, None


async def _apply_calibration(
    series: TimeSeriesData,
    query: DataQuery,
    api_key: str
) -> Tuple[TimeSeriesData, Optional[CalibrationResult]]:
    """Apply calibration to time series"""
    from calibration import UniversalCalibrator
    
    try:
        values = [value for year, value in series]
        
        calibrator = UniversalCalibrator(api_key)
        calibration_result = await calibrator.calibrate_values(
            values=values,
            query_description=f"{query.entity_name} {query.region}",
            unit=query.unit,
            region=query.region
        )
        
        if calibration_result.confidence_score > 0.7:
            calibrated_series = [
                (year, calibration_result.calibrated_values[i])
                for i, (year, _) in enumerate(series)
                if i < len(calibration_result.calibrated_values)
            ]
            return calibrated_series, calibration_result
        
    except Exception:
        pass
    
    return series, None


def generate_market_based_series(
    query: DataQuery,
    market_data: MarketKnowledge,
    start_year: int,
    end_year: int
) -> TimeSeriesData:
    """
    Generate time series using market knowledge
    
    Pure function that creates realistic market data
    """
    regional_multiplier = get_regional_multiplier(market_data, query.region)
    base_value = market_data.base_value * regional_multiplier
    
    growth_rate = _calculate_growth_rate(query, market_data)
    years = list(range(start_year, end_year + 1))
    
    return [
        (year, _calculate_year_value(
            year, i, base_value, growth_rate, market_data, query
        ))
        for i, year in enumerate(years)
    ]


def _calculate_growth_rate(query: DataQuery, market_data: MarketKnowledge) -> float:
    """Calculate adjusted growth rate"""
    growth_rate = market_data.annual_growth_rate
    
    if query.is_disruptor and market_data.growth_pattern == 'exponential':
        growth_rate *= 1.5
        
        # Regional acceleration
        if query.region.lower() in ['usa', 'china', 'germany', 'norway', 'netherlands']:
            growth_rate *= 1.2
    
    return growth_rate


def _calculate_year_value(
    year: int,
    index: int,
    base_value: float,
    growth_rate: float,
    market_data: MarketKnowledge,
    query: DataQuery
) -> float:
    """Calculate value for specific year"""
    # Apply growth pattern
    growth_factor = _apply_growth_pattern(
        index, year, growth_rate, market_data, query
    )
    
    value = base_value * growth_factor
    
    # Apply economic crisis effects
    value *= _get_crisis_factor(year, query.entity_name)
    
    # Add year-over-year noise
    if index > 0:
        noise = random.gauss(0, market_data.volatility * 0.4)
        value *= (1 + noise)
    
    # Apply seasonal patterns
    if market_data.seasonal_pattern and 'sales' in query.metric.lower():
        seasonal_boost = 0.12 if (year % 4 == 0) else 0
        value *= (1 + seasonal_boost)
    
    # Apply bounds
    min_val, max_val = market_data.get_adjusted_range(
        get_regional_multiplier(market_data, query.region)
    )
    value = max(min_val, min(max_val, value))
    
    # Round to appropriate precision
    return _round_to_precision(value)


def _apply_growth_pattern(
    index: int,
    year: int,
    growth_rate: float,
    market_data: MarketKnowledge,
    query: DataQuery
) -> float:
    """Apply growth pattern based on market characteristics"""
    pattern = market_data.growth_pattern
    
    if pattern == 'exponential':
        return _exponential_growth(
            index, year, growth_rate, market_data, query
        )
    elif pattern == 'linear':
        return 1 + (growth_rate * index)
    elif pattern == 'cyclical':
        return _cyclical_growth(index, growth_rate)
    elif pattern == 'volatile':
        return _volatile_growth(index, growth_rate, market_data.volatility)
    else:
        return 1 + (growth_rate * index)


def _exponential_growth(
    index: int,
    year: int,
    growth_rate: float,
    market_data: MarketKnowledge,
    query: DataQuery
) -> float:
    """Calculate exponential growth with saturation"""
    if not (query.is_disruptor and year >= market_data.disruption_threshold_year):
        return (1 + growth_rate) ** index
    
    years_since = year - market_data.disruption_threshold_year
    
    if years_since <= 10:
        return (1 + growth_rate) ** years_since
    else:
        # Saturation effect
        saturation = 1 / (1 + math.exp(-0.3 * (years_since - 12)))
        base_growth = (1 + growth_rate) ** 10
        extended_growth = (1 + growth_rate * 0.2) ** (years_since - 10)
        return base_growth * extended_growth * saturation


def _cyclical_growth(index: int, growth_rate: float) -> float:
    """Calculate cyclical growth pattern"""
    cycle = math.sin(2 * math.pi * index / 8.5) * 0.15
    trend = 1 + (growth_rate * index)
    return trend * (1 + cycle)


def _volatile_growth(index: int, growth_rate: float, volatility: float) -> float:
    """Calculate volatile growth pattern"""
    base_trend = 1 + (growth_rate * index)
    short_cycle = math.sin(2 * math.pi * index / 3) * 0.1
    medium_cycle = math.sin(2 * math.pi * index / 7) * 0.2
    random_shock = random.uniform(-0.15, 0.15) if random.random() < 0.3 else 0
    volatility_factor = 1 + (short_cycle + medium_cycle + random_shock) * volatility
    return base_trend * volatility_factor


def _get_crisis_factor(year: int, entity_name: str) -> float:
    """Get economic crisis impact factor"""
    if year not in CRISIS_YEARS:
        return 1.0
    
    crisis_impact = CRISIS_YEARS[year]
    entity_lower = entity_name.lower()
    
    # Adjust impact by sector
    if any(term in entity_lower for term in ['vehicle', 'construction']):
        crisis_impact *= 1.5
    elif any(term in entity_lower for term in ['food', 'oil']):
        crisis_impact *= 0.5
    elif any(term in entity_lower for term in ['renewable', 'electric']):
        crisis_impact *= 0.3
    
    return 1 + crisis_impact


def _round_to_precision(value: float) -> float:
    """Round value to appropriate precision"""
    if value >= 10000:
        return round(value, 0)
    elif value >= 1000:
        return round(value, 1)
    elif value >= 100:
        return round(value, 2)
    elif value >= 1:
        return round(value, 3)
    else:
        return round(value, 4)


def generate_intelligent_fallback_series(
    query: DataQuery,
    start_year: int,
    end_year: int
) -> TimeSeriesData:
    """
    Generate intelligent fallback series when no market data available
    
    Pure function with heuristic-based generation
    """
    base_value = _estimate_base_value(query)
    growth_rate, pattern = _estimate_growth_pattern(query)
    regional_factor = _estimate_regional_factor(query.region)
    
    base_value *= regional_factor
    years = list(range(start_year, end_year + 1))
    
    return [
        (year, _calculate_fallback_value(
            year, i, base_value, growth_rate, pattern
        ))
        for i, year in enumerate(years)
    ]


def _estimate_base_value(query: DataQuery) -> float:
    """Estimate base value from query characteristics"""
    entity_lower = query.entity_name.lower()
    unit_lower = query.unit.lower()
    
    if 'cost' in entity_lower or 'price' in entity_lower:
        if 'kwh' in unit_lower:
            return 0.12
        elif 'ton' in unit_lower:
            return 750
        elif 'barrel' in unit_lower:
            return 70
        else:
            return 50
    elif 'production' in entity_lower or 'capacity' in entity_lower:
        if 'gw' in unit_lower:
            return 25
        elif 'million' in unit_lower:
            return 10
        else:
            return 100
    elif 'sales' in entity_lower:
        return 2 if 'million' in unit_lower else 150
    else:
        return 100


def _estimate_growth_pattern(query: DataQuery) -> Tuple[float, str]:
    """Estimate growth rate and pattern"""
    entity_lower = query.entity_name.lower()
    
    if query.is_disruptor or any(term in entity_lower for term in ['solar', 'electric']):
        return 0.28, 'exponential'
    elif any(term in entity_lower for term in ['price', 'cost']):
        return 0.03, 'linear'
    else:
        return 0.05, 'linear'


def _estimate_regional_factor(region: str) -> float:
    """Estimate regional multiplier"""
    regional_factors = {
        'china': 2.5, 'usa': 1.0, 'india': 0.6,
        'germany': 1.3, 'japan': 1.2, 'global': 0.9
    }
    return regional_factors.get(region.lower(), 1.0)


def _calculate_fallback_value(
    year: int,
    index: int,
    base_value: float,
    growth_rate: float,
    pattern: str
) -> float:
    """Calculate fallback value for year"""
    if pattern == 'exponential':
        if index <= 8:
            value = base_value * (1 + growth_rate) ** index
        else:
            base_growth = base_value * (1 + growth_rate) ** 8
            value = base_growth * (1 + growth_rate * 0.3) ** (index - 8)
    else:
        value = base_value * (1 + growth_rate * index)
        value *= (1 + random.uniform(-0.08, 0.08))
    
    # Economic crisis effects
    if year in [2008, 2009, 2020]:
        crisis_factor = 0.85 if year in [2008, 2009] else 0.92
        value *= crisis_factor
    
    value = max(0.01, value)
    return _round_to_precision(value)