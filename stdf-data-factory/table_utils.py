#!/usr/bin/env python3
"""
Table Utilities Module
Functions for table creation, formatting, and standardization
"""

import re
from typing import Optional, List, Dict, Any
from models import DataQuery, TableData, CalibrationResult
from data_generation import generate_realistic_time_series_with_calibration


async def create_realistic_table_data_with_calibration(
    query: DataQuery,
    api_key: Optional[str] = None
) -> TableData:
    """
    Create market-validated table data with universal calibration
    
    Main entry point for generating table data
    """
    time_series, calibration_result = await generate_realistic_time_series_with_calibration(
        query, api_key
    )
    
    # Format headers
    headers = create_headers(query)
    
    # Format data
    data = format_time_series(time_series)
    
    # Determine confidence and source type
    from market_knowledge import find_market_knowledge
    market_data = find_market_knowledge(query)
    
    confidence, source_type, calibration_info = determine_source_metrics(
        calibration_result, market_data
    )
    
    return TableData(
        headers=headers,
        data=data,
        rows=len(data),
        cols=2,
        confidence=confidence,
        source_type=source_type,
        calibration_info=calibration_info
    )


def create_headers(query: DataQuery) -> List[str]:
    """
    Create formatted table headers
    
    Pure function that generates headers from query
    """
    unit_suffix = f" ({query.unit})" if query.unit else ""
    return ["Year", f"Value{unit_suffix}"]


def format_time_series(time_series: List[tuple]) -> List[List[str]]:
    """
    Format time series data for display
    
    Pure function that formats numeric data to strings
    """
    return [
        [str(year), format_value(value)]
        for year, value in time_series
    ]


def format_value(value: float) -> str:
    """
    Format single value with appropriate precision
    
    Pure function for value formatting
    """
    if value >= 1000000:
        return f"{value:,.0f}"
    elif value >= 1000:
        return f"{value:,.1f}"
    elif value >= 1:
        return f"{value:.2f}"
    else:
        return f"{value:.4f}"


def determine_source_metrics(
    calibration_result: Optional[CalibrationResult],
    market_data: Optional[object]
) -> tuple:
    """
    Determine confidence score and source type
    
    Pure function that computes metrics from inputs
    """
    if calibration_result:
        confidence = calibration_result.confidence_score
        source_type = f'calibrated_{calibration_result.validation_method}'
        calibration_info = {
            'adjustment_factor': calibration_result.adjustment_factor,
            'reference_range': calibration_result.reference_range,
            'validation_method': calibration_result.validation_method,
            'adjustment_reasoning': calibration_result.adjustment_reasoning
        }
        return confidence, source_type, calibration_info
    else:
        confidence = 0.88 if market_data else 0.72
        source_type = 'market_intelligence' if market_data else 'intelligent_fallback'
        return confidence, source_type, None


async def standardize_web_table_with_calibration(
    table_data: Dict[str, Any],
    query: DataQuery,
    api_key: Optional[str] = None
) -> Optional[TableData]:
    """
    Advanced web table standardization with intelligent column detection and calibration
    
    Transform web-extracted table to standard format
    """
    try:
        headers = table_data['headers']
        data = table_data['data']
        
        if not headers or not data:
            return None
        
        # Find year and value columns
        year_col_idx = find_year_column(headers, data)
        value_col_idx = find_value_column(headers, data, year_col_idx)
        
        if year_col_idx is None or value_col_idx is None:
            return None
        
        # Extract and clean data
        cleaned_data, raw_values = extract_and_clean_data(
            data, year_col_idx, value_col_idx, query.time_range
        )
        
        if len(cleaned_data) < 3:
            return None
        
        # Apply calibration
        final_data, calibration_info = await apply_table_calibration(
            cleaned_data, raw_values, query, api_key
        )
        
        # Create headers
        new_headers = create_value_headers(query)
        
        # Determine confidence
        confidence, source_type = determine_web_table_confidence(
            table_data, calibration_info
        )
        
        return TableData(
            headers=new_headers,
            data=final_data,
            rows=len(final_data),
            cols=2,
            confidence=confidence,
            source_type=source_type,
            calibration_info=calibration_info
        )
        
    except Exception:
        pass
    
    return None


def find_year_column(headers: List[str], data: List[List[str]]) -> Optional[int]:
    """
    Find year column index
    
    Pure function that identifies year column
    """
    year_keywords = ['year', 'date', 'time', 'period', 'yr']
    
    # Strategy 1: Check headers
    for i, header in enumerate(headers):
        if any(keyword in header.lower() for keyword in year_keywords):
            return i
    
    # Strategy 2: Check data for year patterns
    if not data:
        return None
    
    for i in range(len(headers)):
        if i >= len(data[0]):
            continue
        
        sample_values = [row[i] for row in data[:5] if i < len(row)]
        year_count = sum(
            1 for val in sample_values 
            if re.search(r'(19|20)\d{2}', str(val))
        )
        
        if year_count >= 2:
            return i
    
    return None


def find_value_column(
    headers: List[str],
    data: List[List[str]],
    year_col_idx: Optional[int]
) -> Optional[int]:
    """
    Find value column index
    
    Pure function that identifies value column
    """
    if not data:
        return None
    
    for i, header in enumerate(headers):
        if i == year_col_idx:
            continue
        
        # Check if this column has numeric data
        sample_values = [row[i] for row in data[:5] if i < len(row)]
        numeric_count = sum(
            1 for val in sample_values 
            if re.search(r'\d+\.?\d*', str(val)) and str(val).strip() not in ['', '-']
        )
        
        if numeric_count >= 3:
            return i
    
    return None


def extract_and_clean_data(
    data: List[List[str]],
    year_col_idx: int,
    value_col_idx: int,
    time_range: tuple
) -> tuple:
    """
    Extract and clean data from table
    
    Pure function that processes raw table data
    """
    cleaned_data = []
    raw_values = []
    start_year, end_year = time_range
    
    for row in data:
        if len(row) <= max(year_col_idx, value_col_idx):
            continue
        
        year_str = str(row[year_col_idx]).strip()
        value_str = str(row[value_col_idx]).strip()
        
        # Extract year
        year_match = re.search(r'(19|20)(\d{2})', year_str)
        if not year_match:
            continue
        
        year = int(year_match.group(0))
        
        if not (start_year <= year <= end_year):
            continue
        
        # Clean value
        if not value_str or value_str in ['-', '', 'N/A', 'n/a']:
            continue
        
        clean_value = re.sub(r'[^\d.,-]', '', value_str).replace(',', '')
        
        # Handle negative values
        is_negative = '-' in value_str and clean_value.count('-') == 1
        clean_value = clean_value.replace('-', '')
        
        if not clean_value or clean_value == '.' or '.' in clean_value[:-1] if len(clean_value) > 1 else True:
            continue
        
        try:
            float_val = float(clean_value)
            if is_negative:
                float_val = -float_val
            
            if abs(float_val) < 1e12:
                cleaned_data.append([str(year), float_val])
                raw_values.append(float_val)
        except ValueError:
            continue
    
    return cleaned_data, raw_values


async def apply_table_calibration(
    cleaned_data: List[List],
    raw_values: List[float],
    query: DataQuery,
    api_key: Optional[str]
) -> tuple:
    """
    Apply calibration to table data
    
    Returns calibrated data and calibration info
    """
    if not api_key or len(raw_values) < 3:
        # Sort and format without calibration
        sorted_data = sort_and_format_data(cleaned_data)
        return sorted_data, None
    
    try:
        from calibration import UniversalCalibrator
        
        # Get unique sorted values
        unique_data = {}
        unique_values = []
        for year_str, value in cleaned_data:
            year = int(year_str)
            if year not in unique_data:
                unique_data[year] = value
                unique_values.append(value)
        
        # Apply calibration
        calibrator = UniversalCalibrator(api_key)
        calibration_result = await calibrator.calibrate_values(
            values=unique_values,
            query_description=f"{query.entity_name} {query.region}",
            unit=query.unit,
            region=query.region
        )
        
        if calibration_result.confidence_score > 0.7:
            # Create calibrated data
            final_data = []
            value_index = 0
            
            for year in sorted(unique_data.keys()):
                if value_index < len(calibration_result.calibrated_values):
                    calibrated_value = calibration_result.calibrated_values[value_index]
                    formatted_value = format_value(calibrated_value)
                    final_data.append([str(year), formatted_value])
                    value_index += 1
                else:
                    original_value = unique_data[year]
                    formatted_value = str(int(original_value)) if original_value == int(original_value) else str(original_value)
                    final_data.append([str(year), formatted_value])
            
            calibration_info = {
                'adjustment_factor': calibration_result.adjustment_factor,
                'reference_range': calibration_result.reference_range,
                'validation_method': calibration_result.validation_method,
                'adjustment_reasoning': calibration_result.adjustment_reasoning
            }
            
            return final_data, calibration_info
        
    except Exception:
        pass
    
    # Fallback: sort and format without calibration
    sorted_data = sort_and_format_data(cleaned_data)
    return sorted_data, None


def sort_and_format_data(cleaned_data: List[List]) -> List[List[str]]:
    """Sort data by year and format values"""
    unique_data = {}
    for year_str, value in cleaned_data:
        year = int(year_str)
        if year not in unique_data:
            unique_data[year] = value
    
    return [
        [str(year), format_value(value)]
        for year, value in sorted(unique_data.items())
    ]


def create_value_headers(query: DataQuery) -> List[str]:
    """Create headers for value column"""
    unit_suffix = f" ({query.unit})" if query.unit else ""
    value_header = f"Value{unit_suffix}"
    
    if query.metric:
        value_header = f"{query.metric.title()}{unit_suffix}"
    
    return ["Year", value_header]


def determine_web_table_confidence(
    table_data: Dict[str, Any],
    calibration_info: Optional[dict]
) -> tuple:
    """Determine confidence and source type for web table"""
    confidence = table_data.get('confidence_score', 0.7)
    source_type = 'web_standardized'
    
    if calibration_info:
        confidence = max(confidence, 0.8)
        method = calibration_info.get('validation_method', 'unknown')
        source_type = f'web_calibrated_{method}'
    
    return confidence, source_type


def export_to_csv(table: TableData) -> str:
    """
    Export table to CSV format
    
    Pure function that converts table to CSV string
    """
    import io
    import csv
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write headers
    writer.writerow(table.headers)
    
    # Write data
    for row in table.data:
        writer.writerow(row)
    
    return output.getvalue()


def export_to_json(
    table: TableData,
    query: DataQuery,
    include_metadata: bool = True
) -> Dict[str, Any]:
    """
    Export table to JSON format with optional metadata
    
    Pure function that converts table to JSON structure
    """
    from datetime import datetime
    
    json_data = {
        'table_data': {
            'headers': table.headers,
            'data': table.data,
            'dimensions': {
                'rows': table.rows,
                'columns': table.cols
            }
        }
    }
    
    if include_metadata:
        json_data['extraction_metadata'] = {
            'entity': query.entity_name,
            'region': query.region,
            'extraction_date': datetime.now().isoformat(),
            'source_type': table.source_type,
            'confidence_score': table.confidence,
            'unit': query.unit,
            'time_range': f"{query.time_range[0]}-{query.time_range[1]}"
        }
        
        if table.calibration_info:
            json_data['calibration'] = table.calibration_info
    
    return json_data