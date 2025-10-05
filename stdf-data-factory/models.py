#!/usr/bin/env python3
"""
Data Models Module
Core data structures and types for the data extraction system
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple, NamedTuple


@dataclass(frozen=True)
class DataQuery:
    """Immutable query structure for data extraction requests"""
    entity_name: str = ""
    entity_type: str = ""
    region: str = ""
    metric: str = ""
    unit: str = ""
    time_range: Tuple[int, int] = (2010, 2025)
    is_disruptor: bool = False
    
    def to_search_query(self) -> str:
        """Convert query to search string"""
        terms = filter(None, [self.entity_name, self.metric, self.region])
        return ' '.join(terms)
    
    def is_valid(self) -> bool:
        """Check if query has minimum required fields"""
        return bool(self.entity_name.strip())


class SingleTableResult(NamedTuple):
    """Result structure for single data extraction"""
    success: bool
    query: Union[str, Dict[str, Any]]
    headers: List[str]
    data: List[List[str]]
    table_rows: int
    data_source: str
    confidence: float
    analysis: str
    error: Optional[str] = None
    source_url: Optional[str] = None
    calibration_info: Optional[Dict[str, Any]] = None


class BatchTableResult(NamedTuple):
    """Result structure for batch processing"""
    success: bool
    total_queries: int
    successful_results: List[SingleTableResult]
    failed_results: List[Dict[str, Any]]
    summary: Dict[str, Any]


class TableData(NamedTuple):
    """Internal table data structure"""
    headers: List[str]
    data: List[List[str]]
    rows: int
    cols: int
    confidence: float = 0.0
    source_type: str = ""
    calibration_info: Optional[Dict[str, Any]] = None


@dataclass
class ValidationPoint:
    """Universal validation point from any source"""
    value: float
    year: Optional[int]
    source_type: str
    confidence: float
    context: str
    
    def is_reliable(self, threshold: float = 0.7) -> bool:
        """Check if validation point meets confidence threshold"""
        return self.confidence >= threshold


@dataclass
class CalibrationResult:
    """Result of universal calibration process"""
    original_values: List[float]
    calibrated_values: List[float]
    adjustment_factor: float
    confidence_score: float
    validation_method: str
    reference_range: Tuple[float, float]
    adjustment_reasoning: List[str] = field(default_factory=list)
    
    def has_significant_adjustment(self, threshold: float = 0.05) -> bool:
        """Check if adjustment is significant"""
        return abs(self.adjustment_factor - 1.0) > threshold
    
    def is_high_confidence(self, threshold: float = 0.75) -> bool:
        """Check if calibration has high confidence"""
        return self.confidence_score >= threshold


@dataclass(frozen=True)
class MarketKnowledge:
    """Market knowledge with realistic ranges validated against industry sources"""
    base_value: float
    unit: str
    typical_range: Tuple[float, float]
    growth_pattern: str
    annual_growth_rate: float
    volatility: float
    regional_multipliers: Dict[str, float]
    seasonal_pattern: bool = False
    disruption_threshold_year: int = 2015
    
    def get_regional_multiplier(self, region: str, aliases: Dict[str, str]) -> float:
        """Get regional multiplier with alias support"""
        if not region:
            return 1.0
        
        region_key = region.lower().replace(' ', '_')
        
        if region_key in self.regional_multipliers:
            return self.regional_multipliers[region_key]
        
        canonical_region = aliases.get(region_key, region_key)
        return self.regional_multipliers.get(canonical_region, 1.0)
    
    def get_adjusted_range(self, regional_multiplier: float) -> Tuple[float, float]:
        """Get range adjusted for region"""
        return (
            self.typical_range[0] * regional_multiplier,
            self.typical_range[1] * regional_multiplier
        )


@dataclass
class URLInfo:
    """URL information container"""
    url: str
    title: str
    confidence: float
    found_via: str = 'search'
    
    def is_trusted(self, trusted_domains: List[str]) -> bool:
        """Check if URL is from trusted domain"""
        url_lower = self.url.lower()
        return any(domain in url_lower for domain in trusted_domains)


@dataclass
class ExtractionSettings:
    """Settings for data extraction"""
    mode: str = 'hybrid'
    max_urls: int = 5
    enable_crawl: bool = True
    use_market_intelligence: bool = True
    query_type: str = 'simple'
    filter_by_relevance: bool = True
    min_relevance: float = 0.3
    
    def is_ai_only(self) -> bool:
        """Check if AI-only mode"""
        return self.mode == "AI-Only (Fastest)" or self.max_urls == 0
    
    def should_use_web(self) -> bool:
        """Check if web extraction should be used"""
        return not self.is_ai_only() and self.max_urls > 0


@dataclass
class ScaleInfo:
    """Unit scale intelligence information"""
    magnitude_order: int
    unit_type: str
    expected_range: Tuple[float, float]
    volatility_factor: float
    growth_expectation: float
    
    @staticmethod
    def from_unit(unit: str, values: List[float]) -> 'ScaleInfo':
        """Create scale info from unit analysis"""
        from math import log10, floor
        
        avg_value = sum(values) / len(values) if values else 100
        magnitude = floor(log10(abs(avg_value))) if avg_value > 0 else 2
        
        return ScaleInfo(
            magnitude_order=magnitude,
            unit_type='unknown',
            expected_range=(1, 1000),
            volatility_factor=0.1,
            growth_expectation=0.05
        )


@dataclass
class WebTableInfo:
    """Extracted web table information"""
    id: str
    headers: List[str]
    data: List[List[str]]
    rows: int
    cols: int
    confidence_score: float
    source_url: str
    source_title: str
    extraction_method: str
    relevance_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'headers': self.headers,
            'data': self.data,
            'rows': self.rows,
            'cols': self.cols,
            'confidence_score': self.confidence_score,
            'source_url': self.source_url,
            'source_title': self.source_title,
            'extraction_method': self.extraction_method,
            'relevance_score': self.relevance_score
        }


# Type Aliases for better code readability
QueryInput = Union[str, Dict[str, Any]]
QueryBatch = List[Dict[str, Any]]
TimeSeriesData = List[Tuple[int, float]]
HeadersAndData = Tuple[List[str], List[List[str]]]