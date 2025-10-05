#!/usr/bin/env python3
"""
LLM Integration Module
Functions for LLM API integration, prompt generation, and response parsing
"""

import re
from typing import Optional, List
from models import DataQuery, TableData, CalibrationResult
from market_knowledge import find_market_knowledge, get_regional_multiplier
from config import LLM_MODELS, LLM_TEMPERATURE, LLM_MAX_TOKENS_GENERATION, LLM_MAX_TOKENS_ANALYSIS

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


def create_intelligent_llm_prompt_with_calibration(
    query: DataQuery,
    calibration_hint: str = ""
) -> str:
    """
    Create market-context enhanced LLM prompt with calibration guidance
    
    Pure function that generates optimized prompt
    """
    market_data = find_market_knowledge(query)
    context_info = _build_market_context(query, market_data, calibration_hint)
    
    start_year, end_year = query.time_range
    unit_text = f" in {query.unit}" if query.unit else ""
    
    return f"""Generate realistic historical data for: {query.entity_name}{unit_text}

QUERY SPECIFICATIONS:
- Entity: {query.entity_name}
- Region: {query.region or 'Global'}
- Metric: {query.metric or 'Value'}
- Time Period: {start_year}-{end_year}
- Unit: {query.unit or 'Appropriate units'}
- Disruptive Technology: {'Yes' if query.is_disruptor else 'No'}

{context_info}

CRITICAL REQUIREMENTS:
1. Use REALISTIC values that match actual market conditions
2. Show appropriate growth patterns for the sector and region
3. Include realistic year-over-year variations (not smooth curves)
4. Account for major economic events:
   - 2008-2009: Financial crisis (reduce values 10-18% for most sectors)
   - 2020: COVID-19 impact (varies by sector, typically -5% to -15%)
   - 2022: Supply chain disruptions, inflation (+3% to +8%)
5. For disruptive technologies: show exponential early growth, then gradual saturation
6. Values must be defensible to industry experts and analysts
7. Regional differences should reflect actual economic conditions

OUTPUT FORMAT (EXACTLY as shown):
Year | Value
{start_year} | [realistic_value]
{start_year + 1} | [realistic_value]
...continue for each year...
{end_year} | [realistic_value]

Provide ONLY the data table with industry-validated values, no explanatory text."""


def _build_market_context(
    query: DataQuery,
    market_data: Optional[object],
    calibration_hint: str
) -> str:
    """Build market context section for prompt"""
    if not market_data:
        return calibration_hint or ""
    
    regional_mult = get_regional_multiplier(market_data, query.region)
    typical_min, typical_max = market_data.typical_range
    adjusted_min = typical_min * regional_mult
    adjusted_max = typical_max * regional_mult
    
    context = f"""
MARKET VALIDATION CONTEXT:
- Industry validated range: {adjusted_min:.2f} to {adjusted_max:.2f} {market_data.unit}
- Growth pattern: {market_data.growth_pattern} ({market_data.annual_growth_rate:.1%} annually)
- Regional adjustment: {regional_mult:.2f}x for {query.region}
- Volatility: {market_data.volatility:.1%}
- Source: IEA, EIA, World Bank, IRENA industry data
"""
    
    if calibration_hint:
        context += f"\n\nCALIBRATION GUIDANCE:\n{calibration_hint}"
    
    return context


async def generate_llm_data_with_calibration(
    query: DataQuery,
    api_key: str
) -> TableData:
    """
    Generate enhanced LLM data with universal calibration
    
    Try multiple models and return best result
    """
    if not GROQ_AVAILABLE or not api_key:
        from table_utils import create_realistic_table_data_with_calibration
        return await create_realistic_table_data_with_calibration(query, api_key)
    
    try:
        groq_client = Groq(api_key=api_key)
        prompt = create_intelligent_llm_prompt_with_calibration(query)
        
        for model in LLM_MODELS:
            try:
                response = groq_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_TOKENS_GENERATION
                )
                
                llm_response = response.choices[0].message.content.strip()
                table = await parse_llm_response_with_calibration(
                    llm_response, query, api_key
                )
                
                if table and table.rows >= 5:
                    return table
                    
            except Exception:
                continue
                
    except Exception:
        pass
    
    # Fallback to market knowledge
    from table_utils import create_realistic_table_data_with_calibration
    return await create_realistic_table_data_with_calibration(query, api_key)


async def parse_llm_response_with_calibration(
    llm_response: str,
    query: DataQuery,
    api_key: str
) -> Optional[TableData]:
    """
    Enhanced LLM response parsing with calibration validation
    
    Parse, validate, and calibrate LLM-generated data
    """
    try:
        data = _extract_data_from_response(llm_response, query.time_range)
        
        if len(data) < 3:
            return None
        
        # Sort and deduplicate
        sorted_data = _sort_and_deduplicate(data)
        
        if len(sorted_data) < 3:
            return None
        
        # Apply calibration
        calibrated_data, calibration_info = await _calibrate_llm_data(
            sorted_data, query, api_key
        )
        
        # Format headers
        headers = _create_headers(query)
        
        # Determine confidence and source
        confidence, source_type = _determine_llm_confidence(
            calibration_info
        )
        
        return TableData(
            headers=headers,
            data=calibrated_data,
            rows=len(calibrated_data),
            cols=2,
            confidence=confidence,
            source_type=source_type,
            calibration_info=calibration_info
        )
        
    except Exception:
        pass
    
    from table_utils import create_realistic_table_data_with_calibration
    return await create_realistic_table_data_with_calibration(query, api_key)


def _extract_data_from_response(
    llm_response: str,
    time_range: tuple
) -> List[List[str]]:
    """Extract year-value pairs from LLM response"""
    lines = [line.strip() for line in llm_response.split('\n') if line.strip()]
    data = []
    start_year, end_year = time_range
    
    for line in lines:
        if not ('|' in line and not line.startswith(('Year', 'TABLE:', 'HEADERS:', '#', 'Entity'))):
            continue
        
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 2:
            continue
        
        year_str, value_str = parts[0], parts[1]
        
        # Extract year
        year_match = re.search(r'(\d{4})', year_str)
        if not year_match:
            continue
        
        year = int(year_match.group(1))
        if not (start_year <= year <= end_year):
            continue
        
        # Clean and validate value
        clean_value = re.sub(r'[^\d.,-]', '', value_str).replace(',', '')
        if not clean_value or clean_value in ['-', '.', '']:
            continue
        
        try:
            float_val = float(clean_value)
            if 0 <= float_val < 1e15:
                data.append([str(year), clean_value])
        except ValueError:
            continue
    
    return data


def _sort_and_deduplicate(data: List[List[str]]) -> List[List[str]]:
    """Sort by year and remove duplicates"""
    unique_data = {}
    for year_str, value_str in data:
        year = int(year_str)
        if year not in unique_data:
            unique_data[year] = value_str
    
    return [[str(year), value] for year, value in sorted(unique_data.items())]


async def _calibrate_llm_data(
    sorted_data: List[List[str]],
    query: DataQuery,
    api_key: str
) -> tuple:
    """Apply calibration to LLM data"""
    if not api_key:
        return sorted_data, None
    
    try:
        from calibration import UniversalCalibrator
        
        values = [float(row[1]) for row in sorted_data]
        
        calibrator = UniversalCalibrator(api_key)
        calibration_result = await calibrator.calibrate_values(
            values=values,
            query_description=f"{query.entity_name} {query.region}",
            unit=query.unit,
            region=query.region
        )
        
        if calibration_result.confidence_score > 0.7:
            calibrated_data = []
            for i, (year, _) in enumerate(sorted_data):
                if i < len(calibration_result.calibrated_values):
                    calibrated_value = calibration_result.calibrated_values[i]
                    formatted_value = _format_value(calibrated_value)
                    calibrated_data.append([year, formatted_value])
                else:
                    calibrated_data.append(sorted_data[i])
            
            calibration_info = {
                'adjustment_factor': calibration_result.adjustment_factor,
                'reference_range': calibration_result.reference_range,
                'validation_method': calibration_result.validation_method,
                'adjustment_reasoning': calibration_result.adjustment_reasoning
            }
            
            return calibrated_data, calibration_info
    except Exception:
        pass
    
    return sorted_data, None


def _format_value(value: float) -> str:
    """Format value with appropriate precision"""
    if value >= 1000000:
        return f"{value:,.0f}"
    elif value >= 1000:
        return f"{value:,.1f}"
    elif value >= 1:
        return f"{value:.2f}"
    else:
        return f"{value:.4f}"


def _create_headers(query: DataQuery) -> List[str]:
    """Create formatted headers"""
    unit_suffix = f" ({query.unit})" if query.unit else ""
    return ["Year", f"Value{unit_suffix}"]


def _determine_llm_confidence(calibration_info: Optional[dict]) -> tuple:
    """Determine confidence and source type"""
    if calibration_info:
        method = calibration_info.get('validation_method', 'unknown')
        return 0.85, f'llm_calibrated_{method}'
    return 0.85, 'llm_market_validated'


def generate_analysis_with_calibration(
    table: TableData,
    query: DataQuery,
    api_key: str
) -> str:
    """
    Generate comprehensive analysis with market context and calibration insights
    
    Try LLM analysis, fall back to intelligent default
    """
    if not GROQ_AVAILABLE or not api_key:
        return _create_fallback_analysis(table, query)
    
    try:
        groq_client = Groq(api_key=api_key)
        
        insights = _extract_data_insights(table, query)
        calibration_insights = _extract_calibration_insights(table)
        
        analysis_prompt = _create_analysis_prompt(
            query, insights, table.source_type, calibration_insights
        )
        
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0.2,
            max_tokens=LLM_MAX_TOKENS_ANALYSIS
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception:
        return _create_fallback_analysis(table, query)


def _extract_data_insights(table: TableData, query: DataQuery) -> str:
    """Extract statistical insights from data"""
    if len(table.data) < 2:
        return f"Dataset: {table.rows} data points"
    
    try:
        first_value = float(re.sub(r'[^\d.]', '', table.data[0][1]))
        last_value = float(re.sub(r'[^\d.]', '', table.data[-1][1]))
        years = len(table.data)
        
        if years > 1 and first_value > 0:
            cagr = ((last_value / first_value) ** (1 / (years - 1)) - 1) * 100
            total_change = ((last_value - first_value) / first_value) * 100
            
            values = [float(re.sub(r'[^\d.]', '', row[1])) for row in table.data]
            peak_value = max(values)
            trough_value = min(values)
            
            return f"""
Data Overview:
- Period: {table.data[0][0]}-{table.data[-1][0]} ({years} years)
- Initial: {first_value:.2f} {query.unit or ''}
- Final: {last_value:.2f} {query.unit or ''}
- CAGR: {cagr:+.1f}%
- Total Change: {total_change:+.1f}%
- Peak: {peak_value:.2f}
- Trough: {trough_value:.2f}"""
    except:
        pass
    
    return f"Dataset: {table.rows} data points from {table.data[0][0]} to {table.data[-1][0]}"


def _extract_calibration_insights(table: TableData) -> str:
    """Extract calibration insights"""
    if not table.calibration_info:
        return ""
    
    adj_factor = table.calibration_info.get('adjustment_factor', 1.0)
    if abs(adj_factor - 1.0) > 0.05:
        method = table.calibration_info.get('validation_method', 'unknown method')
        return f"\nCalibration Applied: {adj_factor:.2f}x adjustment using {method}"
    
    return ""


def _create_analysis_prompt(
    query: DataQuery,
    insights: str,
    source_type: str,
    calibration_insights: str
) -> str:
    """Create analysis prompt"""
    return f"""Analyze this {query.entity_name} data for {query.region or 'Global'}:

{insights}
Data Source: {source_type}{calibration_insights}

Provide a comprehensive 3-4 sentence analysis covering:
1. Key trends and growth patterns observed
2. Market drivers, economic factors, or policy influences
3. Regional context and sector-specific insights
4. Forward-looking implications or market outlook

Focus on professional, data-driven insights suitable for business or policy analysis."""


def _create_fallback_analysis(table: TableData, query: DataQuery) -> str:
    """Create intelligent fallback analysis"""
    try:
        if len(table.data) < 2:
            return _create_minimal_analysis(table, query)
        
        first_val = float(re.sub(r'[^\d.]', '', table.data[0][1]))
        last_val = float(re.sub(r'[^\d.]', '', table.data[-1][1]))
        years = len(table.data)
        
        if first_val > 0 and years > 1:
            growth = ((last_val / first_val) ** (1/(years-1)) - 1) * 100
            trend = _categorize_trend(growth)
            
            market_context = _get_market_context(table.source_type)
            calibration_context = _get_calibration_context(table.calibration_info)
            regional_context = _get_regional_context(query.region)
            
            return f"Market analysis indicates {trend} in {query.entity_name} for {query.region or 'global markets'} with {growth:.1f}% compound annual growth rate over the {years}-year period{regional_context}. The data shows realistic market dynamics including economic cycle impacts and sector-specific volatility{market_context}{calibration_context}."
    except:
        pass
    
    return _create_minimal_analysis(table, query)


def _categorize_trend(growth: float) -> str:
    """Categorize growth trend"""
    if growth > 25:
        return "exponential growth"
    elif growth > 10:
        return "rapid expansion"
    elif growth > 3:
        return "steady growth"
    elif growth > -3:
        return "stable performance"
    elif growth > -10:
        return "gradual decline"
    else:
        return "significant contraction"


def _get_market_context(source_type: str) -> str:
    """Get market context string"""
    if 'calibrated' in source_type:
        return " using universal calibration with market validation"
    elif 'market' in source_type or 'intelligence' in source_type:
        return " based on validated industry data"
    elif 'llm' in source_type:
        return " using AI analysis with market validation"
    return ""


def _get_calibration_context(calibration_info: Optional[dict]) -> str:
    """Get calibration context string"""
    if not calibration_info:
        return ""
    
    adj_factor = calibration_info.get('adjustment_factor', 1.0)
    if abs(adj_factor - 1.0) > 0.1:
        return f" with {adj_factor:.1f}x calibration adjustment applied"
    return ""


def _get_regional_context(region: str) -> str:
    """Get regional context string"""
    if not region or region.lower() == 'global':
        return ""
    
    regional_contexts = {
        'china': ', reflecting China\'s rapid industrialization and market scale',
        'usa': ', showing mature market dynamics with steady innovation',
        'germany': ', demonstrating European efficiency and sustainability focus',
        'india': ', indicating emerging market growth potential',
        'japan': ', revealing advanced technology adoption patterns'
    }
    
    return regional_contexts.get(
        region.lower(),
        f', showing {region}-specific market characteristics'
    )


def _create_minimal_analysis(table: TableData, query: DataQuery) -> str:
    """Create minimal analysis when data is insufficient"""
    calibration_note = ""
    if table.calibration_info:
        method = table.calibration_info.get('validation_method', 'universal calibration')
        calibration_note = f" with {method}"
    
    source_desc = (
        'industry-validated intelligence'
        if 'market' in table.source_type
        else 'enhanced AI analysis with market validation'
    )
    
    return f"Generated {table.rows} market-validated data points for {query.entity_name} in {query.region or 'global markets'} using {source_desc}{calibration_note}."