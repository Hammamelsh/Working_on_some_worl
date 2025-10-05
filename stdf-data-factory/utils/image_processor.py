import fitz
import os
from PIL import Image
from typing import List, Optional
import io
import base64


def extract_page_images(page: fitz.Page, page_num: int, output_dir: str) -> List[str]:
    """
    Extract images from a PDF page by rendering it as an image.
    
    Args:
        page: PyMuPDF page object
        page_num: Page number (0-indexed)
        output_dir: Directory to save images
        
    Returns:
        List of paths to extracted images
    """
    extracted = []
    
    try:        
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        image_bytes = pix.tobytes("png")
        image_filename = f"page_{page_num + 1}_render.png"
        image_path = os.path.join(output_dir, image_filename)
        
        saved_path = save_image(image_bytes, image_path)
        if saved_path:
            extracted.append(saved_path)
                
    except Exception as e:
        raise Exception(f"Error extracting images from page {page_num + 1}: {e}")
    
    return extracted


def save_image(image_data: bytes, filepath: str) -> Optional[str]:
    """
    Save image data to file.
    
    Args:
        image_data: Raw image bytes
        filepath: Path to save the image
        
    Returns:
        Path to saved image or None if failed
    """
    try:
        if filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
            with open(filepath, 'wb') as f:
                f.write(image_data)
        else:
            img = Image.open(io.BytesIO(image_data))
            
            png_filepath = filepath.rsplit('.', 1)[0] + '.png'
            img.save(png_filepath, 'PNG')
            filepath = png_filepath
        
        return filepath
    except Exception as e:
        raise Exception(f"Error saving image: {e}")


def convert_image_to_base64(image_path: str) -> Optional[str]:
    """
    Convert image file to base64 string.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Base64 encoded string or None if failed
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        raise Exception(f"Error converting image to base64: {e}")


def get_image_size_mb(image_path: str) -> float:
    """
    Get image file size in megabytes.
    
    Args:
        image_path: Path to image file
        
    Returns:
        File size in MB
    """
    try:
        size_bytes = os.path.getsize(image_path)
        return size_bytes / (1024 * 1024)
    except Exception as e:
        raise Exception(f"Error getting image size: {e}")


def resize_image_if_needed(image_path: str, max_size_mb: float = 4) -> str:
    """
    Resize image if it exceeds the maximum size limit.
    
    Args:
        image_path: Path to image file
        max_size_mb: Maximum size in MB (default: 4)
        
    Returns:
        Path to resized image (or original if no resize needed)
    """
    try:
        size_mb = get_image_size_mb(image_path)
        
        if size_mb <= max_size_mb:
            return image_path
        
        img = Image.open(image_path)
        
        scale_factor = (max_size_mb / size_mb) ** 0.5
        new_width = int(img.width * scale_factor)
        new_height = int(img.height * scale_factor)
        
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        resized_path = image_path.rsplit('.', 1)[0] + '_resized.png'
        img_resized.save(resized_path, 'PNG', optimize=True)
        
        return resized_path
    except Exception as e:
        raise Exception(f"Error resizing image: {e}")
