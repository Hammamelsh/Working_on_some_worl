import os
import json
from typing import List, Dict, Any
from openai import OpenAI
from groq import Groq
from .image_processor import convert_image_to_base64


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
        raise Exception("XAI_API_KEY environment variable not set")
    
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
            # Convert image to base64
            image_base64 = convert_image_to_base64(image_path)
            if not image_base64:
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
                            "text": "Analyze this image for any time series data, charts, graphs, or visualizations. Pay special attention to MULTIPLE data series or regions shown in the same chart. Extract any numerical data, trends, or patterns you can identify. If there are different regions, countries, or categories shown (like China, European Union, North America, etc.), identify each separately with their specific data points. If there are legends, different colors, or multiple lines/bars, describe each series individually. Focus on extracting the data points and chart information for each series. If there are no time series elements, just say `no time series elements`. DO NOT include any text or commentary in your response.",
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
            
        except Exception as e:
            raise Exception(f"Error processing image {image_path}: {e}")
    
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

            CRITICAL: If the analysis describes charts with MULTIPLE data series (different regions, countries, categories), create SEPARATE chart entries for each series.

            For each chart/graph found, extract the information in this exact JSON format:
            {{
                "charts": [
                    {{
                        "X axis label": "",
                        "Y axis label": "",
                        "X values": [],
                        "Y values": [],
                        "Chart name": "",
                        "Region": ""
                    }}
                ]
            }}

            CRITICAL Instructions for Multi-Series Charts:
            1. Look for mentions of multiple regions/countries/categories in the same chart (e.g., "China", "European Union", "North America", "Japan", "Korea", etc.)
            2. If the analysis mentions data for different regions/series, create a SEPARATE chart entry for each region
            3. For each region/series, extract:
               - The specific region/country name exactly as mentioned
               - The X values (years, time periods, categories)
               - The Y values (data points specific to that region)
            4. DO NOT combine different regions into one "Global" entry unless the data is explicitly described as global/worldwide
            5. Look for phrases like "China shows", "European Union data", "North America production", etc.
            6. If data is broken down by region/country, each should be a separate chart entry
            7. Extract the exact region names as mentioned in the analysis
            8. Chart name should describe what the data represents
            9. If multiple charts are described, include them all in the "charts" array
            10. If no numerical data is available, use empty arrays for X values and Y values
            11. Return ONLY the JSON object, no additional text or explanation

            Examples:
            - If analysis mentions "China production increased from 3.5M in 2021 to 11.5M in 2024" → Create entry with Region: "China"
            - If analysis mentions "European Union shows 2.3M in 2021, 3M in 2022" → Create entry with Region: "European Union"  
            - If analysis describes 5 different countries' data → Create 5 separate chart entries

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
            
            return structured_data
            
        except json.JSONDecodeError as e:
            return {"charts": [], "error": f"Failed to parse JSON response: {e}"}
            
    except FileNotFoundError:
        return {"charts": [], "error": "Markdown file not found"}
        
    except Exception as e:
        return {"charts": [], "error": str(e)}
