#!/usr/bin/env python3
"""
Web Extraction Module
Functions for web scraping, URL validation, and table extraction
"""

import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, quote_plus, parse_qs, unquote
from typing import List, Dict, Optional, Tuple, Any
from models import DataQuery, URLInfo, WebTableInfo, HeadersAndData
from config import (
    TRUSTED_DOMAINS, INTL_ORGS_CONFIDENCE, DATA_PLATFORMS_CONFIDENCE,
    SKIP_DOMAINS, WEB_REQUEST_TIMEOUT, MAX_TEXT_LENGTH
)

try:
    from crawl4ai import AsyncWebCrawler
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False


# ============================================================================
# Web Session Management
# ============================================================================

def create_web_session() -> requests.Session:
    """
    Create configured web session with proper headers
    
    Pure function that returns configured session
    """
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    })
    session.timeout = WEB_REQUEST_TIMEOUT
    return session


# ============================================================================
# URL Utilities
# ============================================================================

def fix_url_protocol(url: str) -> str:
    """
    Fix URL protocol issues from search results
    
    Pure function that normalizes URLs
    """
    if not url:
        return ""
    
    # Handle DuckDuckGo redirect URLs
    if 'duckduckgo.com/l/' in url:
        extracted = _extract_from_redirect(url)
        if extracted:
            return extracted
    
    # Fix protocol issues
    if url.startswith('//'):
        return f'https:{url}'
    elif not url.startswith(('http://', 'https://')):
        return f'https://{url}' if url.startswith('www.') else ""
    
    return url


def _extract_from_redirect(url: str) -> Optional[str]:
    """Extract actual URL from redirect"""
    try:
        parsed = urlparse(url)
        if parsed.query:
            params = parse_qs(parsed.query)
            if 'uddg' in params:
                return unquote(params['uddg'][0])
    except:
        pass
    return None


def is_valid_data_source(url: str) -> bool:
    """
    Enhanced data source validation with quality scoring
    
    Pure function that validates URL quality
    """
    if not url:
        return False
    
    url_lower = url.lower()
    
    # Skip social media and low-quality domains
    if any(domain in url_lower for domain in SKIP_DOMAINS):
        return False
    
    # Check for quality indicators
    quality_indicators = [
        '.gov', '.edu', '.org',
        'worldbank', 'oecd', 'eia.gov', 'iea.org', 'irena.org',
        'ourworldindata', 'statista', 'tradingeconomics',
        'data.', 'statistics', 'stats.', 'census', 'bureau'
    ]
    
    return any(indicator in url_lower for indicator in quality_indicators)


def calculate_source_confidence(url: str, title: str) -> float:
    """
    Advanced source confidence calculation
    
    Pure function that scores URL confidence
    """
    score = 0.4
    url_lower = url.lower()
    title_lower = title.lower()
    
    # Government and academic sources
    if '.gov' in url_lower:
        score += 0.4
    elif '.edu' in url_lower:
        score += 0.3
    elif '.org' in url_lower:
        score += 0.2
    
    # International organizations
    for org, bonus in INTL_ORGS_CONFIDENCE.items():
        if org in url_lower:
            score += bonus
            break
    
    # Data platforms
    for platform, bonus in DATA_PLATFORMS_CONFIDENCE.items():
        if platform in url_lower:
            score += bonus
            break
    
    # Data-related keywords
    data_keywords = ['data', 'statistics', 'database', 'dataset', 'report', 'analysis']
    keyword_matches = sum(
        1 for keyword in data_keywords 
        if keyword in url_lower or keyword in title_lower
    )
    score += min(keyword_matches * 0.05, 0.2)
    
    # Penalize low-quality indicators
    low_quality = ['blog', 'forum', 'wiki', 'personal']
    for indicator in low_quality:
        if indicator in url_lower or indicator in title_lower:
            score -= 0.1
    
    return max(0.1, min(1.0, score))


# ============================================================================
# Web Search
# ============================================================================

def search_web_sources(query: DataQuery, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Enhanced web source search with multiple search engines
    
    Functional approach to web search
    """
    from query_utils import build_search_terms
    
    search_terms = build_search_terms(query)
    
    try:
        session = create_web_session()
        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(search_terms)}"
        
        response = session.get(search_url, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        sources = _extract_search_results(soup, max_results)
        
        # Sort by confidence and return top results
        return sorted(sources, key=lambda s: s['confidence'], reverse=True)[:max_results]
        
    except Exception:
        return []


def _extract_search_results(soup: BeautifulSoup, max_results: int) -> List[Dict[str, Any]]:
    """Extract and validate search results"""
    sources = []
    
    for result in soup.find_all('a', class_='result__a')[:max_results * 3]:
        title = result.get_text().strip()
        url = fix_url_protocol(result.get('href', ''))
        
        if url and is_valid_data_source(url):
            confidence = calculate_source_confidence(url, title)
            sources.append({
                'url': url,
                'title': title,
                'confidence': confidence
            })
    
    return sources


# ============================================================================
# HTML Table Extraction
# ============================================================================

async def extract_from_url(url: str, url_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Advanced URL extraction with crawl4ai priority and comprehensive fallbacks
    
    Async function that extracts tables from URL
    """
    tables = []
    
    try:
        # Method 1: Try crawl4ai first if available
        if CRAWL4AI_AVAILABLE:
            tables = await _extract_with_crawl4ai(url, url_info)
            if tables:
                return tables
        
        # Method 2: Fallback to requests
        tables = _extract_with_requests(url, url_info)
        
    except Exception:
        pass
    
    return tables


async def _extract_with_crawl4ai(url: str, url_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract using crawl4ai library"""
    try:
        async with AsyncWebCrawler(
            headless=True,
            page_timeout=30000,
            browser_type="chromium"
        ) as crawler:
            result = await crawler.arun(
                url=url,
                wait_for="table, .table, [data-table]",
                css_selector="table, .table, .data-table",
                timeout=25000
            )
            
            if result.success and hasattr(result, 'html') and result.html:
                return parse_html_tables(result.html, url_info)
    except Exception:
        pass
    
    return []


def _extract_with_requests(url: str, url_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract using requests library"""
    session = create_web_session()
    response = session.get(url, timeout=WEB_REQUEST_TIMEOUT)
    response.raise_for_status()
    
    content_type = response.headers.get('content-type', '').lower()
    if 'html' in content_type or not content_type:
        return parse_html_tables(response.text, url_info)
    
    return []


def parse_html_tables(html: str, url_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Comprehensive HTML table parsing with advanced data extraction
    
    Pure function that parses HTML to table data
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove unwanted elements
    for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
        tag.decompose()
    
    # Find all tables
    unique_tables = _find_unique_tables(soup)
    
    # Parse each table
    tables = []
    for i, table_elem in enumerate(unique_tables, 1):
        try:
            headers, data = extract_table_data(table_elem)
            
            if _is_valid_table(headers, data):
                quality_score = calculate_table_quality(headers, data)
                
                if quality_score >= 0.3:
                    table_data = _create_table_dict(
                        i, headers, data, url_info, quality_score
                    )
                    tables.append(table_data)
        except Exception:
            continue
    
    return tables


def _find_unique_tables(soup: BeautifulSoup) -> List:
    """Find unique tables in HTML"""
    table_selectors = [
        'table', '.table', '.data-table', '[role="table"]'
    ]
    
    found_tables = []
    for selector in table_selectors:
        found_tables.extend(soup.select(selector))
    
    # Remove duplicates
    unique_tables = []
    seen = set()
    for table in found_tables:
        table_id = id(table)
        if table_id not in seen:
            unique_tables.append(table)
            seen.add(table_id)
    
    return unique_tables


def _is_valid_table(headers: List[str], data: List[List[str]]) -> bool:
    """Check if table meets minimum requirements"""
    return bool(headers and data and len(data) >= 3 and len(headers) >= 2)


def _create_table_dict(
    index: int,
    headers: List[str],
    data: List[List[str]],
    url_info: Dict[str, Any],
    quality_score: float
) -> Dict[str, Any]:
    """Create table dictionary"""
    return {
        'id': f"table_{index}",
        'headers': headers,
        'data': data,
        'rows': len(data),
        'cols': len(headers),
        'confidence_score': url_info['confidence'] * quality_score,
        'source_url': url_info.get('url', ''),
        'source_title': url_info.get('title', ''),
        'extraction_method': 'crawl4ai' if CRAWL4AI_AVAILABLE else 'requests'
    }


def extract_table_data(table_elem) -> HeadersAndData:
    """
    Enhanced table data extraction with improved header detection
    
    Pure function that extracts headers and data from table element
    """
    rows = table_elem.find_all('tr')
    if len(rows) < 2:
        return [], []
    
    # Try multiple strategies to find headers
    headers, data_start = _extract_headers(table_elem, rows)
    
    if len(headers) < 2:
        return [], []
    
    # Extract data rows
    data = _extract_data_rows(rows, data_start, len(headers))
    
    return headers, data


def _extract_headers(table_elem, rows: List) -> Tuple[List[str], int]:
    """Extract headers using multiple strategies"""
    # Strategy 1: Look for thead section
    thead = table_elem.find('thead')
    if thead:
        header_row = thead.find('tr')
        if header_row:
            th_cells = header_row.find_all(['th', 'td'])
            headers = [clean_text(cell.get_text()) for cell in th_cells]
            return headers, 0
    
    # Strategy 2: Look for th tags in first row
    first_row = rows[0]
    th_cells = first_row.find_all('th')
    
    if th_cells and len(th_cells) >= 2:
        headers = [clean_text(th.get_text()) for th in th_cells]
        return headers, 1
    
    # Strategy 3: Use first row as headers if mostly non-numeric
    cells = first_row.find_all(['td', 'th'])
    if cells and len(cells) >= 2:
        cell_texts = [clean_text(cell.get_text()) for cell in cells]
        non_numeric_count = sum(
            1 for text in cell_texts 
            if not re.search(r'^\d+\.?\d*', text.strip())
        )
        
        if non_numeric_count >= len(cell_texts) * 0.7:
            return cell_texts, 1
    
    return [], 0


def _extract_data_rows(
    rows: List, 
    data_start: int, 
    header_length: int
) -> List[List[str]]:
    """Extract data rows from table"""
    data = []
    
    for row in rows[data_start:]:
        cells = row.find_all(['td', 'th'])
        if not cells:
            continue
        
        row_data = [clean_text(cell.get_text()) for cell in cells]
        
        # Skip empty rows
        if not any(cell.strip() for cell in row_data):
            continue
        
        # Normalize row length to match headers
        while len(row_data) < header_length:
            row_data.append('')
        row_data = row_data[:header_length]
        
        # Check if this is a data row (has at least one numeric value)
        if _has_numeric_data(row_data):
            data.append(row_data)
    
    return data


def _has_numeric_data(row_data: List[str]) -> bool:
    """Check if row contains numeric data"""
    return any(re.search(r'\d', cell) for cell in row_data)


def clean_text(text: str) -> str:
    """
    Professional text cleaning with comprehensive character handling
    
    Pure function that normalizes text
    """
    if not text:
        return ''
    
    cleaned = re.sub(r'\s+', ' ', text.strip())
    cleaned = cleaned.replace('\u00a0', ' ')
    cleaned = cleaned.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
    
    # Remove HTML entities
    html_entities = {
        '&nbsp;': ' ', '&amp;': '&', '&lt;': '<', '&gt;': '>',
        '&quot;': '"', '&#x27;': "'", '&#39;': "'", '&hellip;': '...'
    }
    for entity, replacement in html_entities.items():
        cleaned = cleaned.replace(entity, replacement)
    
    # Keep meaningful symbols
    cleaned = re.sub(r'[^\w\s\-\.\,\%\$\(\)\/\:]', '', cleaned)
    
    return cleaned[:MAX_TEXT_LENGTH]


def calculate_table_quality(headers: List[str], data: List[List[str]]) -> float:
    """
    Enhanced table quality assessment with comprehensive scoring
    
    Pure function that scores table quality
    """
    score = 0.3
    
    # Size scoring
    score += _calculate_size_score(len(data), len(headers))
    
    # Data completeness
    score += _calculate_completeness_score(headers, data)
    
    # Header quality
    score += _calculate_header_quality(headers)
    
    # Numeric data analysis
    score += _calculate_numeric_quality(data, headers)
    
    # Penalize very wide tables
    if len(headers) > 15:
        score -= 0.1
    
    return max(0.1, min(1.0, score))


def _calculate_size_score(num_rows: int, num_cols: int) -> float:
    """Calculate score based on table size"""
    row_score = min(0.35, 0.15 + (num_rows - 5) * 0.02) if num_rows >= 5 else 0
    col_score = min(0.2, 0.1 + (num_cols - 2) * 0.025) if num_cols >= 3 else 0
    return row_score + col_score


def _calculate_completeness_score(
    headers: List[str], 
    data: List[List[str]]
) -> float:
    """Calculate completeness score"""
    if not data:
        return 0
    
    total_cells = len(headers) * len(data)
    non_empty_cells = sum(
        1 for row in data for cell in row if str(cell).strip()
    )
    completeness = non_empty_cells / total_cells if total_cells > 0 else 0
    return completeness * 0.25


def _calculate_header_quality(headers: List[str]) -> float:
    """Calculate header quality score"""
    header_text = ' '.join(headers).lower()
    score = 0
    
    # Time indicators
    time_indicators = ['year', 'date', 'time', 'period', 'quarter']
    if any(indicator in header_text for indicator in time_indicators):
        score += 0.15
    
    # Value indicators
    value_indicators = ['value', 'amount', 'total', 'price', 'production', 'sales']
    if any(indicator in header_text for indicator in value_indicators):
        score += 0.1
    
    return score


def _calculate_numeric_quality(data: List[List[str]], headers: List[str]) -> float:
    """Calculate numeric data quality score"""
    numeric_cells = 0
    sample_rows = min(20, len(data))
    
    for row in data[:sample_rows]:
        for cell in row:
            cell_str = str(cell).strip()
            if re.search(r'\d+\.?\d*', cell_str):
                numeric_cells += 1
    
    sample_size = sample_rows * len(headers)
    if sample_size > 0:
        numeric_ratio = numeric_cells / sample_size
        return numeric_ratio * 0.2
    
    return 0


def calculate_table_relevance(table_data: Dict[str, Any], query: DataQuery) -> float:
    """
    Advanced table relevance calculation with multiple factors
    
    Pure function that scores table relevance to query
    """
    score = 0.5
    
    content = ' '.join(table_data['headers']).lower()
    if table_data['data']:
        sample_data = ' '.join(
            ' '.join(row[:5]) for row in table_data['data'][:10]
        ).lower()
        content += f" {sample_data}"
    
    # Entity matching
    score += _calculate_entity_match(content, query.entity_name)
    
    # Region, metric, and unit matching
    if query.region and query.region.lower() in content:
        score += 0.2
    if query.metric and query.metric.lower() in content:
        score += 0.15
    if query.unit:
        score += _calculate_unit_match(content, query.unit)
    
    # Time-based indicators
    score += _calculate_time_match(content)
    
    # Numeric data quality
    score += _calculate_table_numeric_quality(table_data)
    
    return max(0.2, min(1.0, score))


def _calculate_entity_match(content: str, entity_name: str) -> float:
    """Calculate entity match score"""
    entity_words = [w for w in entity_name.lower().split() if len(w) > 2]
    matches = sum(1 for word in entity_words if word in content)
    return min(matches * 0.15, 0.4) if matches > 0 else 0


def _calculate_unit_match(content: str, unit: str) -> float:
    """Calculate unit match score"""
    unit_words = [w for w in unit.lower().split() if len(w) > 1]
    matches = sum(1 for word in unit_words if word in content)
    return min(matches * 0.1, 0.15) if matches > 0 else 0


def _calculate_time_match(content: str) -> float:
    """Calculate time-based match score"""
    time_indicators = ['year', 'date', 'time', 'period', 'quarter', '201', '202']
    matches = sum(1 for indicator in time_indicators if indicator in content)
    return min(matches * 0.05, 0.15) if matches > 0 else 0


def _calculate_table_numeric_quality(table_data: Dict[str, Any]) -> float:
    """Calculate numeric quality of table"""
    try:
        numeric_cells = 0
        sample_size = min(50, len(table_data.get('data', [])) * len(table_data.get('headers', [])))
        
        for row in table_data.get('data', [])[:20]:
            for cell in row[:5]:
                if re.search(r'\d+\.?\d*', str(cell).strip()):
                    numeric_cells += 1
        
        if sample_size > 0:
            numeric_ratio = numeric_cells / min(sample_size, 100)
            return numeric_ratio * 0.1
    except:
        pass
    
    return 0