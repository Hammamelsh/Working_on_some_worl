import fitz
import os
from PIL import Image
from typing import List, Optional, Dict, Any, Tuple
import io
import base64
import json
from openai import OpenAI
from groq import Groq

def extract_images_from_pdf(pdf_path: str, output_base_dir: str) -> List[str]:
    extracted_images = []
    
    try:
        # Create output directory based on PDF name
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_dir = os.path.join(output_base_dir, pdf_name, "images")
        
        pdf_document = fitz.open(pdf_path)
        os.makedirs(output_dir, exist_ok=True)
        
        image_counter = 0
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            images = extract_page_images(page, page_num, output_dir)
            extracted_images.extend(images)
        
        pdf_document.close()
        
    except Exception as e:
        print(f"Error extracting images from PDF: {e}")
    
    return extracted_images

def extract_page_images(page: fitz.Page, page_num: int, output_dir: str) -> List[str]:
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
        print(f"Error extracting images from page {page_num + 1}: {e}")
    
    return extracted

def save_image(image_data: bytes, filepath: str) -> Optional[str]:
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
        print(f"Error saving image: {e}")
        return None

def convert_image_to_base64(image_path: str) -> Optional[str]:
    try:
        import base64
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

def get_image_size_mb(image_path: str) -> float:
    try:
        size_bytes = os.path.getsize(image_path)
        return size_bytes / (1024 * 1024)
    except Exception as e:
        print(f"Error getting image size: {e}")
        return 0

def resize_image_if_needed(image_path: str, max_size_mb: float = 4) -> str:
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
        print(f"Error resizing image: {e}")
        return image_path


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


def extract_time_series_from_images_using_grok(image_paths: List[str], output_dir: str) -> List[str]:
    """
    Process images using Grok API and save results to markdown file.
    
    Args:
        image_paths: List of paths to image files
        output_dir: Directory to save the markdown output
        
    Returns:
        List of processed image paths
    """
    # Get API key from environment
    XAI_API_KEY = os.getenv("XAI_API_KEY")
    if not XAI_API_KEY:
        print("Error: XAI_API_KEY environment variable not set")
        return []
    
    # Initialize OpenAI client for Grok
    client = OpenAI(
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai/v1",
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Markdown output file path
    markdown_file = os.path.join(output_dir, "time_series_analysis.md")
    
    processed_images = []
    
    # Process each image
    for i, image_path in enumerate(image_paths):
        try:
            print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            # Convert image to base64
            image_base64 = convert_image_to_base64(image_path)
            if not image_base64:
                print(f"Failed to convert image to base64: {image_path}")
                continue
            
            # Create data URL for the image
            image_url = f"data:image/png;base64,{image_base64}"
            
            # Prepare messages for Grok API
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "detail": "high",
                            },
                        },
                        {
                            "type": "text",
                            "text": "Analyze this image for any time series data, charts, graphs, or visualizations. Extract any numerical data, trends, or patterns you can identify. If there are no time series elements, just say `no time series elements`. Focus on extracting the data points and chart information. DO NOT include any text or commentary in your response.",
                        },
                    ],
                },
            ]
            
            # Call Grok API
            completion = client.chat.completions.create(
                model="grok-4",
                messages=messages,
            )
            
            # Get response content
            response_content = completion.choices[0].message.content
            
            # Append to markdown file
            with open(markdown_file, 'a', encoding='utf-8') as f:
                if i == 0:  # First image, add header
                    f.write("# Time Series Analysis Results\n\n")
                
                f.write(f"## Image {i+1}: {os.path.basename(image_path)}\n\n")
                f.write(f"**Image Path:** `{image_path}`\n\n")
                f.write("**Analysis:**\n\n")
                f.write(response_content)
                f.write("\n\n---\n\n")
            
            processed_images.append(image_path)
            print(f"Successfully processed: {image_path}")
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue
    
    print(f"Time series analysis complete. Results saved to: {markdown_file}")
    return processed_images


def extract_structured_data_from_markdown(markdown_path: str) -> Dict[str, Any]:
    """
    Extract structured chart/graph data from time series analysis markdown using Groq.
    
    Args:
        markdown_path: Path to the markdown file containing time series analysis
        
    Returns:
        Dictionary containing structured chart data for all graphs found
    """
    try:
        # Read the markdown file
        with open(markdown_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Initialize Groq client
        client = Groq()
        
        # Create prompt for structured data extraction
        prompt = f"""
            Analyze the following time series analysis markdown content and extract structured data for any charts or graphs mentioned.

            For each chart/graph found, extract the information in this exact JSON format:
            {{
                "charts": [
                    {{
                        "X axis label": "",
                        "Y axis label": "",
                        "X values": [],
                        "Y values": [],
                        "Chart name": ""
                    }}
                ]
            }}

            Instructions:
            1. Look for any charts, graphs, or visualizations described in the text
            2. Extract the X-axis label (time dimension, categories, etc.)
            3. Extract the Y-axis label (units, metrics, etc.)
            4. Extract all X values (years, categories, time points, etc.)
            5. Extract all Y values (numerical data points)
            6. Extract the chart/graph title or name
            7. If multiple charts are described, include them all in the "charts" array
            8. If no numerical data is available, use empty arrays for X values and Y values
            9. Return ONLY the JSON object, no additional text or explanation

            Markdown content to analyze:

            {markdown_content}
            """

        # Call Groq API
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1  # Low temperature for more consistent structured output
        )
        
        # Get response content
        response_content = completion.choices[0].message.content.strip()
        
        # Try to parse JSON response
        try:
            # Remove any markdown code block formatting if present
            if response_content.startswith('```json'):
                response_content = response_content[7:]
            if response_content.endswith('```'):
                response_content = response_content[:-3]
            if response_content.startswith('```'):
                response_content = response_content[3:]
            
            structured_data = json.loads(response_content.strip())
            
            # Save structured data to JSON file
            output_dir = os.path.dirname(markdown_path)
            json_output_path = os.path.join(output_dir, "structured_chart_data.json")
            
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False)
            
            print(f"Structured data extracted and saved to: {json_output_path}")
            return structured_data
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Raw response: {response_content}")
            return {"charts": [], "error": "Failed to parse JSON response"}
            
    except FileNotFoundError:
        print(f"Markdown file not found: {markdown_path}")
        return {"charts": [], "error": "Markdown file not found"}
        
    except Exception as e:
        print(f"Error extracting structured data: {e}")
        return {"charts": [], "error": str(e)}


def process_pdf_time_series(pdf_path: str, output_base_dir: str = "extracted_data") -> Dict[str, Any]:
    """
    Complete pipeline to process PDF for time series analysis.
    
    Args:
        pdf_path: Path to the PDF file
        output_base_dir: Base directory for outputs (default: "extracted_data")
        
    Returns:
        Dictionary containing processing results
    """
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
        print(f"Time series analysis already exists for {results['pdf_name']}, skipping image processing...")
        results["skipped"] = True
        
        # Still extract structured data from existing markdown
        structured_data = extract_structured_data_from_markdown(markdown_path)
        results["structured_data"] = structured_data
        print(f"Extracted structured data for {len(structured_data.get('charts', []))} charts")
        
        return results
    
    # Extract images from PDF
    print(f"Processing PDF: {results['pdf_name']}")
    extracted_images = extract_images_from_pdf(pdf_path, output_base_dir)
    results["extracted_images"] = extracted_images
    print(f"Extracted {len(extracted_images)} images")
    
    if not extracted_images:
        print("No images extracted, skipping further processing")
        return results
    
    # Get output directory for this PDF
    output_dir = os.path.join(output_base_dir, results['pdf_name'])
    
    # Process images with Grok API for time series analysis
    processed_images = extract_time_series_from_images_using_grok(extracted_images, output_dir)
    results["processed_images"] = processed_images
    print(f"Processed {len(processed_images)} images for time series analysis")
    
    # Extract structured data from markdown
    if os.path.exists(markdown_path):
        structured_data = extract_structured_data_from_markdown(markdown_path)
        results["structured_data"] = structured_data
        print(f"Extracted structured data for {len(structured_data.get('charts', []))} charts")
    
    return results


if __name__ == "__main__":
    pdf_path = "Visualization Engine v3.0.pdf"
    output_base_dir = "extracted_data"
    
    # Process PDF using the complete pipeline
    results = process_pdf_time_series(pdf_path, output_base_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"PDF: {results['pdf_name']}")
    print(f"Skipped (already processed): {results['skipped']}")
    if not results['skipped']:
        print(f"Images extracted: {len(results['extracted_images'])}")
        print(f"Images processed: {len(results['processed_images'])}")
    print(f"Charts found: {len(results['structured_data'].get('charts', []))}")
    print(f"Output directory: extracted_data/{results['pdf_name']}/")
    print("="*50)