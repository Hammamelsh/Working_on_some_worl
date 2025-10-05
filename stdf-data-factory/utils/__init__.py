"""
Utils module for PDF Time Series Analyzer.

This module provides utility functions for:
- PDF processing and image extraction
- Image manipulation and conversion
- API client interactions with Grok and Groq
"""

from .pdf_processor import (
    extract_images_from_pdf,
    check_existing_analysis,
    process_pdf_time_series
)

from .image_processor import (
    extract_page_images,
    save_image,
    convert_image_to_base64,
    get_image_size_mb,
    resize_image_if_needed
)

from .api_clients import (
    extract_time_series_from_images_using_grok,
    extract_structured_data_from_markdown
)

__all__ = [
    # PDF processing
    'extract_images_from_pdf',
    'check_existing_analysis',
    'process_pdf_time_series',
    
    # Image processing
    'extract_page_images',
    'save_image',
    'convert_image_to_base64',
    'get_image_size_mb',
    'resize_image_if_needed',
    
    # API clients
    'extract_time_series_from_images_using_grok',
    'extract_structured_data_from_markdown'
]
