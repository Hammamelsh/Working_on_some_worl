import fitz
import os
from typing import List, Optional, Tuple
from .image_processor import extract_page_images


def extract_images_from_pdf(pdf_path: str, output_base_dir: str) -> List[str]:
    """
    Extract images from PDF pages by rendering each page as an image.
    
    Args:
        pdf_path: Path to the PDF file
        output_base_dir: Base directory for output
        
    Returns:
        List of paths to extracted images
    """
    extracted_images = []
    
    try:
        # Create output directory based on PDF name
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_dir = os.path.join(output_base_dir, pdf_name, "images")
        
        pdf_document = fitz.open(pdf_path)
        os.makedirs(output_dir, exist_ok=True)
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            images = extract_page_images(page, page_num, output_dir)
            extracted_images.extend(images)
        
        pdf_document.close()
        
    except Exception as e:
        raise Exception(f"Error extracting images from PDF: {e}")
    
    return extracted_images


def check_existing_analysis(pdf_path: str, output_base_dir: str) -> Tuple[bool, str]:
    """
    Check if time series analysis already exists for the given PDF.
    
    Args:
        pdf_path: Path to the PDF file
        output_base_dir: Base output directory
        
    Returns:
        Tuple of (exists, markdown_path)
    """
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    markdown_path = os.path.join(output_base_dir, pdf_name, "time_series_analysis.md")
    return os.path.exists(markdown_path), markdown_path


def process_pdf_time_series(pdf_path: str, output_base_dir: str = "extracted_data") -> dict:
    """
    Complete pipeline to process PDF for time series analysis.
    
    Args:
        pdf_path: Path to the PDF file
        output_base_dir: Base directory for outputs (default: "extracted_data")
        
    Returns:
        Dictionary containing processing results
    """
    from .api_clients import extract_time_series_from_images_using_grok, extract_structured_data_from_markdown
    
    results = {
        "pdf_path": pdf_path,
        "pdf_name": os.path.splitext(os.path.basename(pdf_path))[0],
        "extracted_images": [],
        "processed_images": [],
        "structured_data": {},
        "skipped": False
    }
    
    # Check if analysis already exists
    analysis_exists, markdown_path = check_existing_analysis(pdf_path, output_base_dir)
    
    if analysis_exists:
        results["skipped"] = True
        
        # Still extract structured data from existing markdown
        structured_data = extract_structured_data_from_markdown(markdown_path)
        results["structured_data"] = structured_data
        
        return results
    
    # Extract images from PDF
    extracted_images = extract_images_from_pdf(pdf_path, output_base_dir)
    results["extracted_images"] = extracted_images
    
    if not extracted_images:
        return results
    
    # Get output directory for this PDF
    output_dir = os.path.join(output_base_dir, results['pdf_name'])
    
    # Process images with Grok API for time series analysis
    processed_images = extract_time_series_from_images_using_grok(extracted_images, output_dir)
    results["processed_images"] = processed_images
    
    # Extract structured data from markdown
    if os.path.exists(markdown_path):
        structured_data = extract_structured_data_from_markdown(markdown_path)
        results["structured_data"] = structured_data
    
    return results
