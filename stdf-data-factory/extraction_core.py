#!/usr/bin/env python3
"""
Extraction Core Module
Main extraction orchestration with web and AI fallback strategies
"""

from typing import Union, Dict, Any, List
from models import (
    DataQuery, SingleTableResult, BatchTableResult, TableData,
    QueryInput, QueryBatch
)
from query_utils import create_query
from web_extraction import search_web_sources, extract_from_url, calculate_table_relevance
from table_utils import (
    create_realistic_table_data_with_calibration,
    standardize_web_table_with_calibration
)
from llm_integration import generate_analysis_with_calibration


async def extract_data_single_output(
    query: QueryInput,
    groq_api_key: str,
    max_urls: int = 5,
    enable_deep_crawl: bool = True
) -> SingleTableResult:
    """
    Main single query extraction with comprehensive fallback strategy
    
    Extraction pipeline:
    1. Parse and validate query
    2. Try web extraction if max_urls > 0
    3. Fall back to AI generation with market intelligence
    4. Apply calibration and analysis
    """
    try:
        structured_query = create_query(query)
        
        # AI-only mode (max_urls = 0)
        if max_urls == 0:
            return await _extract_ai_only(query, structured_query, groq_api_key)
        
        # Hybrid mode: Try web extraction first
        web_result = await _try_web_extraction(
            query, structured_query, groq_api_key, max_urls
        )
        
        if web_result:
            return web_result
        
        # Fallback to AI generation
        return await _extract_ai_only(query, structured_query, groq_api_key)
        
    except Exception as e:
        return SingleTableResult(
            success=False,
            query=query,
            headers=[],
            data=[],
            table_rows=0,
            data_source='error',
            confidence=0.0,
            analysis="",
            error=f"Extraction failed: {str(e)}"
        )


async def _extract_ai_only(
    original_query: QueryInput,
    structured_query: DataQuery,
    api_key: str
) -> SingleTableResult:
    """AI-only extraction using market intelligence"""
    table = await create_realistic_table_data_with_calibration(
        structured_query, api_key
    )
    
    analysis = generate_analysis_with_calibration(
        table, structured_query, api_key
    )
    
    return SingleTableResult(
        success=True,
        query=original_query,
        headers=table.headers,
        data=table.data,
        table_rows=table.rows,
        data_source=table.source_type,
        confidence=table.confidence,
        analysis=analysis,
        calibration_info=table.calibration_info
    )


async def _try_web_extraction(
    original_query: QueryInput,
    structured_query: DataQuery,
    api_key: str,
    max_urls: int
) -> Union[SingleTableResult, None]:
    """
    Try web extraction with intelligent source selection
    
    Returns SingleTableResult if successful, None otherwise
    """
    discovered_urls = search_web_sources(structured_query, max_urls)
    
    if not discovered_urls:
        return None
    
    best_table = None
    best_score = 0
    extraction_attempts = 0
    
    for url_info in discovered_urls:
        try:
            extraction_attempts += 1
            tables = await extract_from_url(url_info['url'], url_info)
            
            for table_data in tables:
                relevance_score = calculate_table_relevance(
                    table_data, structured_query
                )
                quality_score = table_data['confidence_score']
                combined_score = relevance_score * 0.6 + quality_score * 0.4
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_table = table_data
                    
        except Exception:
            continue
    
    # Use web data if good quality
    if best_table and best_score >= 0.5:
        standardized = await standardize_web_table_with_calibration(
            best_table, structured_query, api_key
        )
        
        if standardized and standardized.rows >= 5:
            analysis = generate_analysis_with_calibration(
                standardized, structured_query, api_key
            )
            
            return SingleTableResult(
                success=True,
                query=original_query,
                headers=standardized.headers,
                data=standardized.data,
                table_rows=standardized.rows,
                data_source='web_extracted',
                confidence=best_score,
                analysis=f"Web extracted from {extraction_attempts} sources: {analysis}",
                source_url=best_table.get('source_url'),
                calibration_info=standardized.calibration_info
            )
    
    return None


async def extract_data_batch_output(
    queries: QueryBatch,
    groq_api_key: str,
    max_urls: int = 5,
    enable_deep_crawl: bool = True
) -> BatchTableResult:
    """
    Enhanced batch processing with parallel execution and detailed reporting
    
    Process multiple queries and aggregate results
    """
    successful_results = []
    failed_results = []
    
    for i, query_dict in enumerate(queries, 1):
        try:
            result = await extract_data_single_output(
                query_dict, groq_api_key, max_urls, enable_deep_crawl
            )
            
            if result.success:
                successful_results.append(result)
            else:
                failed_results.append({
                    'query_index': i,
                    'query': query_dict,
                    'error': result.error or 'Unknown error'
                })
                
        except Exception as e:
            failed_results.append({
                'query_index': i,
                'query': query_dict,
                'error': str(e)
            })
    
    # Calculate summary statistics
    summary = _calculate_batch_summary(successful_results, len(queries))
    
    return BatchTableResult(
        success=len(successful_results) > 0,
        total_queries=len(queries),
        successful_results=successful_results,
        failed_results=failed_results,
        summary=summary
    )


def _calculate_batch_summary(
    successful_results: List[SingleTableResult],
    total_queries: int
) -> Dict[str, Any]:
    """Calculate comprehensive batch processing summary"""
    if not successful_results:
        return {
            'total_queries': total_queries,
            'successful_queries': 0,
            'failed_queries': total_queries,
            'success_rate': 0,
            'total_rows': 0,
            'average_confidence': 0
        }
    
    total_rows = sum(r.table_rows for r in successful_results)
    avg_confidence = sum(r.confidence for r in successful_results) / len(successful_results)
    
    # Count data sources
    data_sources = {}
    calibrated_count = 0
    
    for result in successful_results:
        source = result.data_source
        data_sources[source] = data_sources.get(source, 0) + 1
        if result.calibration_info:
            calibrated_count += 1
    
    return {
        'total_queries': total_queries,
        'successful_queries': len(successful_results),
        'failed_queries': total_queries - len(successful_results),
        'success_rate': len(successful_results) / total_queries,
        'total_rows': total_rows,
        'average_confidence': avg_confidence,
        'data_sources': data_sources,
        'avg_rows_per_query': total_rows / len(successful_results),
        'calibrated_queries': calibrated_count,
        'calibration_rate': calibrated_count / len(successful_results)
    }


def get_availability_status() -> Dict[str, bool]:
    """
    Comprehensive system availability status
    
    Check which features are available
    """
    from web_extraction import CRAWL4AI_AVAILABLE
    from llm_integration import GROQ_AVAILABLE
    
    return {
        'crawl4ai': CRAWL4AI_AVAILABLE,
        'groq': GROQ_AVAILABLE,
        'market_knowledge': True,
        'web_extraction': True,
        'realistic_data_generation': True,
        'universal_calibration': True,
        'batch_processing': True,
        'multi_format_export': True,
        'comprehensive_analysis': True
    }