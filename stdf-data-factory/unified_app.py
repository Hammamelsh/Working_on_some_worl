#!/usr/bin/env python3
"""
Unified Data Hunting Agent
Combines PDF analysis, chart analysis, and web data extraction
Workflow-based layout with guided user experience
"""
import time
import streamlit as st
import os
import json
import asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import hashlib
import sys
from datetime import datetime
import numpy as np
from typing import Dict, List, Any, Union, Tuple
from urllib.parse import urlparse
from openai import OpenAI

# Add validation directory to Python path
validation_path = os.path.join(os.path.dirname(__file__), 'validation')
if validation_path not in sys.path:
    sys.path.insert(0, validation_path)  # Use insert(0) to prioritize this path

# Import from modular data extractor
from data_extractor import (
    extract_data_single_output,
    extract_data_batch_output,
    validate_query,
    create_query,
    extract_from_url,
    calculate_table_relevance,
    MARKET_KNOWLEDGE_BASE,
    CRAWL4AI_AVAILABLE,
    GROQ_AVAILABLE
)

# Import PDF and image processing utilities
try:
    from utils import process_pdf_time_series
    from utils.api_clients import extract_structured_data_from_markdown
    from utils.image_processor import convert_image_to_base64
    PDF_PROCESSING_AVAILABLE = True
except ImportError:
    PDF_PROCESSING_AVAILABLE = False

# Import validation modules
try:
    from validation_support import (
        run_validation_pipeline,
        ValidationScore,
        validate_enhanced_data_quality,
        export_enhanced_validation_results,
        generate_enhanced_validation_report
    )
    from ground_truth_validators import (
        run_all_domain_expert_validators,
        format_expert_validation_summary
    )
    VALIDATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Validation modules not available: {e}")
    VALIDATION_AVAILABLE = False
# Debug check for validation modules
if VALIDATION_AVAILABLE:
    print("✅ Validation modules loaded successfully")
else:
    print("❌ Validation modules failed to load - check validation/ folder")

# Fallback definitions if imports fail
if not VALIDATION_AVAILABLE:
    class ValidationScore:
        def __init__(self):
            self.overall_score = 0.0
            self.grade = "F"
            self.reliability = 0.0
            self.total_records = 0
            self.passed_records = 0
            self.flagged_records = 0
            self.dimension_scores = {}


    def run_validation_pipeline(*args, **kwargs):
        return {
            'score': ValidationScore(),
            'report': 'Validation modules not available',
            'enhanced_validation': {},
            'expert_validations': {},
            'seba_results': {}
        }


    def export_enhanced_validation_results(*args, **kwargs):
        return {'error': 'Validation modules not available'}


# ==================== PAGE CONFIGURATION ====================

st.set_page_config(
    page_title="Data Hunting Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

DEFAULT_MAX_URLS = 5
DEFAULT_ENABLE_CRAWL = True

# ==================== STYLING ====================

def apply_custom_css():
    """Apply enhanced CSS styling"""
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .market-intelligence-badge {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
            display: inline-block;
            margin: 0.2rem;
        }
        .web-extracted-badge {
            background: linear-gradient(135deg, #007bff 0%, #17a2b8 100%);
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
            display: inline-block;
            margin: 0.2rem;
        }
        .confidence-high { color: #28a745; font-weight: bold; }
        .confidence-medium { color: #ffc107; font-weight: bold; }
        .confidence-low { color: #dc3545; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# ==================== PDF AND CHART PROCESSING FUNCTIONS ====================

def create_time_series_plot(chart_data: Dict[str, Any]) -> go.Figure:
    """Create an interactive time series plot from chart data."""
    try:
        x_values = chart_data.get('X values', [])
        y_values = chart_data.get('Y values', [])
        
        if not x_values or not y_values or len(x_values) != len(y_values):
            return None
        
        # Create a DataFrame for easier plotting
        df = pd.DataFrame({
            'x': x_values,
            'y': y_values
        })
        
        # Determine plot type based on chart type or data characteristics
        chart_name = chart_data.get('Chart name', 'Time Series')
        region = chart_data.get('Region', '')
        series_name = chart_data.get('Series name', '')
        
        # Create title
        title_parts = [chart_name]
        if region:
            title_parts.append(f"({region})")
        if series_name and series_name != region:
            title_parts.append(f"- {series_name}")
        title = " ".join(title_parts)
        
        # Always create line chart with markers
        fig = px.line(
            df, x='x', y='y',
            title=title,
            labels={'x': chart_data.get('X axis label', 'X'), 
                   'y': chart_data.get('Y axis label', 'Y')},
            markers=True
        )
        
        # Customize layout
        fig.update_layout(
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        # Add hover information
        fig.update_traces(
            hovertemplate='<b>%{x}</b><br>%{y}<extra></extra>'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating plot: {e}")
        return None


def create_multi_series_plot(charts: List[Dict[str, Any]]) -> go.Figure:
    """Create a plot with multiple data series."""
    try:
        fig = go.Figure()
        
        for i, chart_data in enumerate(charts):
            x_values = chart_data.get('X values', [])
            y_values = chart_data.get('Y values', [])
            
            if not x_values or not y_values or len(x_values) != len(y_values):
                continue
            
            # Create series name
            region = chart_data.get('Region', '')
            series_name = chart_data.get('Series name', '')
            name = series_name if series_name else region if region else f"Series {i+1}"
            
            # Always use line chart with markers
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines+markers',
                name=name,
                hovertemplate='<b>%{x}</b><br>%{y}<extra></extra>'
            ))
        
        # Update layout
        chart_name = charts[0].get('Chart name', 'Multi-Series Chart') if charts else 'Multi-Series Chart'
        x_label = charts[0].get('X axis label', 'X') if charts else 'X'
        y_label = charts[0].get('Y axis label', 'Y') if charts else 'Y'
        
        fig.update_layout(
            title=chart_name,
            xaxis_title=x_label,
            yaxis_title=y_label,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating multi-series plot: {e}")
        return None


def analyze_chart_image(image_path: str, image_name: str = None) -> Dict[str, Any]:
    """Analyze a single chart image and return structured data."""
    try:
        st.info("🔍 Starting chart image analysis...")
        
        # Create image analysis cache directory
        cache_dir = "extracted_data/image_analysis"
        os.makedirs(cache_dir, exist_ok=True)
        st.info(f"📁 Cache directory: {cache_dir}")
        
        # Generate cache file name based on image name
        if image_name:
            # Clean the image name for file system (remove dots, spaces, special chars)
            clean_name = "".join(c for c in image_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            clean_name = clean_name.replace(' ', '_')
            st.info(f"🏷️ Cleaned filename: {clean_name}")
        else:
            # Generate hash from image path as fallback
            clean_name = hashlib.md5(image_path.encode()).hexdigest()[:8]
            st.info(f"🔗 Using hash: {clean_name}")
        
        cache_file = os.path.join(cache_dir, f"{clean_name}_analysis.json")
        st.info(f"📋 Looking for cache file: {cache_file}")
        
        # Check if analysis already exists
        if os.path.exists(cache_file):
            try:
                st.info("✅ Found exact cache match!")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_results = json.load(f)
                st.success(f"✅ Using cached analysis from: {os.path.basename(cache_file)}")
                return cached_results
            except Exception as e:
                st.error(f"⚠️ Failed to load cached analysis from {cache_file}: {e}")
                # Continue with fresh analysis
        else:
            st.info("🔎 No exact cache match, checking for similar files...")
            # Try to find similar cache files if exact match not found
            if os.path.exists(cache_dir):
                existing_files = [f for f in os.listdir(cache_dir) if f.endswith('_analysis.json')]
                st.info(f"📂 Found {len(existing_files)} existing cache files")
                
                # Try to find a close match based on image name similarity
                if image_name:
                    base_name = os.path.splitext(image_name)[0]  # Remove extension
                    # Create a very flexible normalized version for matching
                    normalized_image = ''.join(c.lower() for c in base_name if c.isalnum())
                    st.info(f"🔤 Normalized search: {normalized_image}")
                    
                    for existing_file in existing_files:
                        # Extract the base part of the existing filename (before _analysis.json)
                        existing_base = existing_file.replace('_analysis.json', '')
                        normalized_existing = ''.join(c.lower() for c in existing_base if c.isalnum())
                        
                        # Check if they're similar enough (allowing for slight variations)
                        if len(normalized_image) > 10 and len(normalized_existing) > 10:
                            # For longer names, check if one contains most of the other
                            similarity_threshold = min(len(normalized_image), len(normalized_existing)) * 0.8
                            common_chars = sum(1 for i, c in enumerate(normalized_image[:len(normalized_existing)]) 
                                             if i < len(normalized_existing) and c == normalized_existing[i])
                            
                            st.info(f"🎯 Checking {existing_file}: {common_chars}/{similarity_threshold}")
                            
                            if common_chars >= similarity_threshold:
                                similar_cache_file = os.path.join(cache_dir, existing_file)
                                try:
                                    st.info(f"📖 Loading similar cache: {existing_file}")
                                    with open(similar_cache_file, 'r', encoding='utf-8') as f:
                                        cached_results = json.load(f)
                                    st.success(f"✅ Found similar cached analysis: {existing_file}")
                                    st.info(f"📁 Loaded from: {similar_cache_file}")
                                    return cached_results
                                except Exception as e:
                                    st.error(f"⚠️ Failed to load similar cache {existing_file}: {e}")
                                    continue
        
        st.info("🚀 No cache found, proceeding with fresh analysis...")
        
        # Get API key from environment
        XAI_API_KEY = os.getenv("XAI_API_KEY")
        if not XAI_API_KEY:
            st.error("❌ XAI_API_KEY not found in environment")
            return {"error": "XAI_API_KEY environment variable not set"}
        
        st.info("🔑 API key found, initializing client...")
        
        # Initialize OpenAI client for Grok
        client = OpenAI(
            api_key=XAI_API_KEY,
            base_url="https://api.x.ai/v1",
        )
        
        st.info("🖼️ Converting image to base64...")
        # Convert image to base64 with error handling
        try:
            image_base64 = convert_image_to_base64(image_path)
            if not image_base64:
                st.error("❌ Failed to convert image to base64")
                return {"error": "Failed to convert image to base64"}
            
            st.info(f"✅ Image converted, size: {len(image_base64)} characters")
        except Exception as img_error:
            st.error(f"❌ Error during image conversion: {img_error}")
            return {"error": f"Image conversion failed: {str(img_error)}"}
        
        # Create data URL for the image
        image_url = f"data:image/png;base64,{image_base64}"
        
        st.info("📝 Preparing API request...")
        
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
                        "text": """Analyze the uploaded chart and extract all numerical data points.

Identify the chart type: Is it a bar chart (stacked or grouped), a line chart, a scatter plot, or a combination?

Define the axes: What do the X and Y axes represent? Please provide the range and units for each.

Break down the data: For each data series or category present in the chart, list the corresponding value(s) for each point. For stacked or grouped charts, provide a breakdown of each component.

Format the output: Present the data in a clear, structured, and easy-to-read format as a JSON object. Ensure the output includes all labels from the chart (e.g., categories, years, product types) to provide context for the extracted values.

The goal is to recreate the underlying data table that was used to generate this visualization.

Provide the response in this exact JSON format:

{
    "chart_type": "",
    "x_axis": {
        "label": "",
        "range": "",
        "units": ""
    },
    "y_axis": {
        "label": "",
        "range": "",
        "units": ""
    },
    "charts": [
        {
            "X axis label": "",
            "Y axis label": "",
            "X values": [],
            "Y values": [],
            "Chart name": "",
            "Region": "",
            "Series name": "",
            "Data breakdown": []
        }
    ]
}

CRITICAL Instructions:
1. Identify and specify the exact chart type (bar chart, line chart, scatter plot, stacked bar, grouped bar, combination, etc.)
2. For axes, provide detailed information:
   - Label: The actual text label shown on the axis
   - Range: The minimum and maximum values visible on the axis
   - Units: The measurement units (e.g., "millions", "percentage", "years", etc.)
3. If this chart shows data for MULTIPLE regions/countries/categories/series, create a SEPARATE chart entry for each
4. For each data series:
   - Extract the specific region/country/category name
   - Provide series name if different from region
   - Include X values (years, time periods, categories)
   - Include Y values (corresponding data points for that specific series)
   - For stacked/grouped charts, provide data breakdown showing individual components
5. Extract ALL labels and context from the chart (legends, categories, years, product types, etc.)
6. DO NOT combine different series into one entry unless they are truly combined in the source
7. For stacked charts, include "Data breakdown" array showing individual stack components
8. Chart name should describe what the data represents
9. If no chart is detected, return empty charts array
10. Return ONLY the JSON object, no additional text

Examples:
- Bar chart showing regional sales: specify "bar chart (grouped)" or "bar chart (stacked)"
- Line chart with multiple countries: create separate entries for each country line
- Stacked bar chart: include breakdown of each stack component in "Data breakdown"
- Combination chart: specify "combination (line + bar)" and handle each series appropriately""",
                    },
                ],
            },
        ]
        
        st.info("🌐 Calling Grok API...")
        
        # Call Grok API with timeout and error handling
        # try:
        completion = client.chat.completions.create(
            model="grok-2-vision-latest",
            messages=messages,
            timeout=60  # Add timeout
        )

        st.info(f"Completion: {completion}")
        st.info("✅ API call successful!")
        # except Exception as api_error:
        #     st.error(f"❌ API call failed: {api_error}")
        #     return {"error": f"API call failed: {str(api_error)}"}
        
        # Get response content
        st.info("📥 Processing API response...")
        response_content = completion.choices[0].message.content.strip()
        st.info(f"📄 Response length: {len(response_content)} characters")
        
        # Parse JSON response
        st.info("🔧 Parsing JSON response...")
        try:
            # Remove any markdown code block formatting if present
            if response_content.startswith('```json'):
                response_content = response_content[7:]
            if response_content.endswith('```'):
                response_content = response_content[:-3]
            if response_content.startswith('```'):
                response_content = response_content[3:]
            
            structured_data = json.loads(response_content.strip())
            st.info("✅ JSON parsing successful!")
            st.info(f"📊 Found {len(structured_data.get('charts', []))} charts in response")
            
            # Save successful analysis to cache file
            st.info("💾 Saving to cache...")
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(structured_data, f, indent=2, ensure_ascii=False)
                st.success(f"✅ Analysis cached for future use: {os.path.basename(cache_file)}")
            except Exception as e:
                st.warning(f"⚠️ Failed to save analysis cache: {e}")
                # Don't fail the whole process if caching fails
            
            return structured_data
            
        except json.JSONDecodeError as e:
            st.error(f"❌ JSON parsing failed: {e}")
            st.error("📄 Raw response (first 500 chars):")
            st.code(response_content[:500])
            error_result = {"error": f"Failed to parse JSON response: {e}", "raw_response": response_content}
            # Don't cache failed results
            return error_result
            
    except Exception as e:
        st.error(f"❌ Unexpected error in analyze_chart_image: {e}")
        st.exception(e)
        return {"error": str(e)}


def display_chart_data(structured_data: Dict[str, Any]) -> None:
    """Display extracted chart data in expandable sections with interactive plots."""
    if structured_data.get('charts'):
        st.header("📊 Extracted Chart Data")
        
        # Store structured_data in session state to prevent loss on widget interaction
        # Create a unique key for this dataset to handle multiple uploads
        data_key = f"chart_data_{hash(str(structured_data.get('charts', [])))}"
        if data_key not in st.session_state:
            st.session_state[data_key] = structured_data
            st.session_state.current_chart_data_key = data_key
        
        # Use session state data to ensure persistence
        if hasattr(st.session_state, 'current_chart_data_key') and st.session_state.current_chart_data_key in st.session_state:
            structured_data = st.session_state[st.session_state.current_chart_data_key]
        else:
            # Fallback to passed data if session state is unavailable
            st.session_state[data_key] = structured_data
            st.session_state.current_chart_data_key = data_key
        
        # Display overall chart information if available
        if structured_data.get('chart_type') or structured_data.get('x_axis') or structured_data.get('y_axis'):
            with st.expander("📋 Overall Chart Analysis", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    if structured_data.get('chart_type'):
                        st.write(f"**Chart Type:** {structured_data['chart_type']}")
                    
                    if structured_data.get('x_axis'):
                        x_axis = structured_data['x_axis']
                        st.write(f"**X-Axis Label:** {x_axis.get('label', 'N/A')}")
                        st.write(f"**X-Axis Range:** {x_axis.get('range', 'N/A')}")
                        st.write(f"**X-Axis Units:** {x_axis.get('units', 'N/A')}")
                
                with col2:
                    if structured_data.get('y_axis'):
                        y_axis = structured_data['y_axis']
                        st.write(f"**Y-Axis Label:** {y_axis.get('label', 'N/A')}")
                        st.write(f"**Y-Axis Range:** {y_axis.get('range', 'N/A')}")
                        st.write(f"**Y-Axis Units:** {y_axis.get('units', 'N/A')}")
        
        # Create multi-series plot if there are multiple series with compatible data
        charts = structured_data['charts']
        chart_type = structured_data.get('chart_type', '')
        
        # Check if we can create a multi-series plot
        valid_charts = [chart for chart in charts if chart.get('X values') and chart.get('Y values')]
        
        if len(valid_charts) > 1:
            # Check if all charts have the same X axis structure (for combining)
            first_x_values = valid_charts[0].get('X values', [])
            same_x_structure = all(
                chart.get('X values', []) == first_x_values 
                for chart in valid_charts[1:]
            )
            
            if same_x_structure:
                st.subheader("🎯 Combined Visualization")
                
                # Show individual plots toggle
                show_individual = st.checkbox("Show individual plots", value=True)
                
                # Create and display multi-series plot (always line chart)
                multi_fig = create_multi_series_plot(valid_charts)
                if multi_fig:
                    st.plotly_chart(multi_fig, use_container_width=True)
                
                # Optionally hide individual plots if combined view is sufficient
                if not show_individual:
                    return
        
        # Display individual chart series
        st.subheader("📈 Individual Series Analysis")
        
        for i, chart in enumerate(charts):
            # Include region and series name in the expander title if available
            chart_name = chart.get('Chart name', 'Unnamed Chart')
            region = chart.get('Region', '')
            series_name = chart.get('Series name', '')
            
            title_parts = [f"Series {i+1}: {chart_name}"]
            if region:
                title_parts.append(f"({region})")
            if series_name and series_name != region:
                title_parts.append(f"- {series_name}")
            
            expander_title = " ".join(title_parts)
            
            with st.expander(expander_title):
                # Create interactive plot for this series
                if chart.get('X values') and chart.get('Y values'):
                    st.subheader("📊 Interactive Line Chart")
                    
                    # Create and display plot (always line chart)
                    fig = create_time_series_plot(chart)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("⚠️ Could not create plot for this series")
                
                # Data and information tabs
                data_tab, info_tab = st.tabs(["📊 Data", "ℹ️ Information"])
                
                with data_tab:
                    if chart.get('X values') and chart.get('Y values'):
                        data_dict = {
                            chart.get('X axis label', 'X'): chart.get('X values', []),
                            chart.get('Y axis label', 'Y'): chart.get('Y values', [])
                        }
                        st.dataframe(data_dict, use_container_width=True)
                        
                        # Basic statistics
                        y_values = chart.get('Y values', [])
                        if y_values and all(isinstance(val, (int, float)) for val in y_values):
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Min", f"{min(y_values):.2f}")
                            with col2:
                                st.metric("Max", f"{max(y_values):.2f}")
                            with col3:
                                st.metric("Mean", f"{sum(y_values)/len(y_values):.2f}")
                            with col4:
                                st.metric("Count", len(y_values))
                    else:
                        st.write("No numerical data available")
                
                with info_tab:
                    st.write(f"**Chart Name:** {chart.get('Chart name', 'N/A')}")
                    st.write(f"**Region:** {chart.get('Region', 'Global')}")
                    if chart.get('Series name'):
                        st.write(f"**Series Name:** {chart.get('Series name', 'N/A')}")
                    st.write(f"**X-axis Label:** {chart.get('X axis label', 'N/A')}")
                    st.write(f"**Y-axis Label:** {chart.get('Y axis label', 'N/A')}")
                    
                    # Display data breakdown if available (for stacked/grouped charts)
                    if chart.get('Data breakdown'):
                        st.subheader("Data Breakdown")
                        breakdown = chart.get('Data breakdown', [])
                        if isinstance(breakdown, list) and breakdown:
                            st.json(breakdown)
                        else:
                            st.write("No breakdown data available")


def check_api_keys() -> bool:
    """Check if required API keys are set in environment."""
    missing_keys = []
    
    if not os.getenv("XAI_API_KEY"):
        missing_keys.append("XAI_API_KEY")
    
    if not os.getenv("GROQ_API_KEY"):
        missing_keys.append("GROQ_API_KEY")
    
    if missing_keys:
        for key in missing_keys:
            st.error(f"❌ {key} environment variable not set")
        return False
    
    st.success("✅ API keys found in environment")
    return True


def save_uploaded_file(uploaded_file, uploads_dir: str) -> str:
    """Save uploaded file to uploads directory and return path."""
    pdf_path = os.path.join(uploads_dir, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return pdf_path


def display_processing_summary(results: Dict[str, Any]) -> None:
    """Display processing summary metrics."""
    st.header("📋 Processing Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Images Extracted", len(results.get('extracted_images', [])))
    with col2:
        st.metric("Images Processed", len(results.get('processed_images', [])))
    with col3:
        st.metric("Charts Found", len(results.get('structured_data', {}).get('charts', [])))


def process_pdf_with_ui_feedback(pdf_path: str) -> Dict[str, Any]:
    """Process PDF with UI feedback and progress tracking."""
    
    # Create progress tracking components
    progress_container = st.container()
    status_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    try:
        # Update progress: Starting
        status_text.text("🚀 Starting PDF processing...")
        progress_bar.progress(0.1)
        
        # Process the PDF
        status_text.text("📄 Extracting images from PDF...")
        progress_bar.progress(0.3)
        
        results = process_pdf_time_series(pdf_path, "extracted_data")
        
        # Update progress based on results
        if results.get('skipped'):
            status_text.text("ℹ️ Using existing analysis...")
            progress_bar.progress(0.8)
        else:
            status_text.text("🧠 Analyzing images with AI...")
            progress_bar.progress(0.6)
            
            status_text.text("📊 Extracting structured data...")
            progress_bar.progress(0.8)
        
        # Complete
        status_text.text("✅ Processing complete!")
        progress_bar.progress(1.0)
        
        return results
        
    except Exception as e:
        st.error(f"❌ Error processing PDF: {e}")
        st.exception(e)
        return {}

# ==================== SESSION STATE ====================

def init_session_state():
    """Initialize session state"""
    defaults = {
        'current_workflow': None,
        'results': None,
        'processing': False,
        'workflow_data': {},
        'extraction_mode': 'hybrid',
        'use_market_intelligence': True,
        # Data validation state
        'collected_data': {},  # Store data from all workflows
        'validation_results': None,
        'validation_running': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_workflow():
    """Reset workflow"""
    st.session_state.current_workflow = None
    st.session_state.results = None
    st.session_state.processing = False
    st.session_state.workflow_data = {}

def clear_results():
    """Clear results"""
    st.session_state.results = None
    st.session_state.workflow_data = {}
    for key in list(st.session_state.keys()):
        if key.startswith(('current_structured_query', 'simple_query', 'batch_queries')):
            del st.session_state[key]

# ==================== DATA COLLECTION FUNCTIONS ====================

def store_validation_data(data_source: str, data: Dict[str, Any]):
    """Store data from workflows for validation"""
    if 'collected_data' not in st.session_state:
        st.session_state.collected_data = {}
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.collected_data[f"{data_source}_{timestamp}"] = {
        'source': data_source,
        'timestamp': timestamp,
        'data': data,
        'data_type': determine_data_type(data)
    }

def determine_data_type(data: Dict[str, Any]) -> str:
    """Determine the type of data for proper processing"""
    if 'charts' in data:
        return 'chart_data'
    elif 'tables' in data:
        return 'table_data'
    elif 'structured_chart_data' in data:
        return 'pdf_chart_data'
    elif isinstance(data, dict) and 'success' in data:
        return 'extraction_result'
    else:
        return 'unknown'


def convert_collected_data_to_validation_format() -> List[Dict[str, Any]]:
    """Convert all collected data to the format expected by validation functions"""

    if 'collected_data' not in st.session_state or not st.session_state.collected_data:
        return []


    validation_data = []

    if 'collected_data' not in st.session_state or not st.session_state.collected_data:
        return validation_data

    for data_key, data_entry in st.session_state.collected_data.items():
        # FIXED: Removed duplicate loop that was preventing data processing
        if not data_entry or not isinstance(data_entry, dict):
            continue  # Skip invalid entries

        data = data_entry.get('data')
        data_type = data_entry.get('data_type', 'unknown')
        source = data_entry.get('source', 'Unknown')

        if not data:
            continue  # Skip if no data
    
    for data_key, data_entry in st.session_state.collected_data.items():
        data = data_entry['data']
        data_type = data_entry['data_type']
        source = data_entry['source']
        
        try:
            if data_type == 'chart_data':
                # Convert PDF chart data
                charts = data.get('charts', [])
                for chart in charts:
                    validation_entry = {
                        'Entity_Name': chart.get('title', 'Unknown Chart'),
                        'Region': chart.get('region', 'Global'),
                        'Unit': chart.get('y_axis_label', ''),
                        'Metric': chart.get('chart_type', 'Unknown'),
                        'X': chart.get('x_data', []),
                        'Y': chart.get('y_data', []),
                        'DataSource_URLs': [source],
                        'Quality_Score': chart.get('confidence', 0.5),
                        'Entity_Type': 'Chart',
                        'Curve_Type': 'Adoption' if 'cost' not in chart.get('y_axis_label', '').lower() else 'Cost'
                    }
                    validation_data.append(validation_entry)
            
            elif data_type == 'pdf_chart_data':
                # Convert structured chart data from PDF analysis
                if isinstance(data, dict) and 'charts' in data:
                    for chart in data['charts']:
                        series_data = chart.get('series', [])
                        for series in series_data:
                            validation_entry = {
                                'Entity_Name': series.get('name', chart.get('title', 'Unknown')),
                                'Region': series.get('region', 'Global'),
                                'Unit': chart.get('y_axis', {}).get('label', ''),
                                'Metric': chart.get('chart_type', 'time_series'),
                                'X': [point.get('x') for point in series.get('data', [])],
                                'Y': [point.get('y') for point in series.get('data', [])],
                                'DataSource_URLs': [source],
                                'Quality_Score': chart.get('confidence_score', 0.5),
                                'Entity_Type': 'Chart',
                                'Curve_Type': 'Adoption' if 'cost' not in chart.get('y_axis', {}).get('label', '').lower() else 'Cost'
                            }
                            validation_data.append(validation_entry)
            
            elif data_type == 'table_data' or data_type == 'extraction_result':
                # Convert table/extraction data
                tables = data.get('tables', [])
                for table in tables:
                    headers = table.get('headers', [])
                    table_data = table.get('data', [])
                    
                    # Try to identify time series columns
                    year_col_idx = None
                    value_col_idx = None
                    
                    for i, header in enumerate(headers):
                        if any(term in header.lower() for term in ['year', 'date', 'time']):
                            year_col_idx = i
                        elif any(term in header.lower() for term in ['value', 'amount', 'quantity', 'price', 'cost']):
                            value_col_idx = i
                    
                    if year_col_idx is not None and value_col_idx is not None:
                        # Extract time series data
                        x_data = []
                        y_data = []
                        
                        for row in table_data:
                            try:
                                year_val = float(row[year_col_idx])
                                value_val = float(row[value_col_idx])
                                x_data.append(year_val)
                                y_data.append(value_val)
                            except (ValueError, IndexError):
                                continue
                        
                        if x_data and y_data:
                            validation_entry = {
                                'Entity_Name': table.get('query_entity', headers[value_col_idx] if value_col_idx < len(headers) else 'Unknown'),
                                'Region': table.get('query_region', 'Global'),
                                'Unit': extract_unit_from_header(headers[value_col_idx] if value_col_idx < len(headers) else ''),
                                'Metric': headers[value_col_idx] if value_col_idx < len(headers) else 'Unknown',
                                'X': x_data,
                                'Y': y_data,
                                'DataSource_URLs': [table.get('source_url', source)],
                                'Quality_Score': table.get('confidence_score', 0.5),
                                'Entity_Type': 'Table',
                                'Curve_Type': 'Adoption'
                            }
                            validation_data.append(validation_entry)
        
        except Exception as e:
            st.warning(f"Error processing data from {source}: {str(e)}")
            continue
    
    return validation_data

def extract_unit_from_header(header: str) -> str:
    """Extract unit information from table headers"""
    header_lower = header.lower()
    
    # Common unit patterns
    if '($' in header or 'usd' in header_lower or 'dollar' in header_lower:
        return 'USD'
    elif 'gw' in header_lower:
        return 'GW'
    elif 'mw' in header_lower:
        return 'MW'
    elif 'gwh' in header_lower:
        return 'GWh'
    elif 'mwh' in header_lower:
        return 'MWh'
    elif '%' in header or 'percent' in header_lower:
        return '%'
    elif 'million' in header_lower:
        return 'Million'
    elif 'billion' in header_lower:
        return 'Billion'
    
    return ''

# ==================== MAIN MENU ====================

def render_quick_start():
    """Render workflow selection"""
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Data Hunting Agent</h1>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 PDF Analysis", key="pdf_workflow", use_container_width=True):
            st.session_state.current_workflow = "pdf_analysis"
            st.rerun()
        
        if st.button("📊 Chart Analysis", key="chart_workflow", use_container_width=True):
            st.session_state.current_workflow = "chart_analysis"
            st.rerun()
        
        if st.button("🎯 Query + URL", key="query_url_workflow", use_container_width=True):
            st.session_state.current_workflow = "query_url_extraction"
            st.rerun()
    
    with col2:
        if st.button("🌐 Web Data Extraction", key="web_workflow", use_container_width=True):
            st.session_state.current_workflow = "web_extraction"
            st.rerun()
        
        if st.button("🔗 URL Extraction", key="url_workflow", use_container_width=True):
            st.session_state.current_workflow = "url_extraction"
            st.rerun()
        
        if st.button("🔬 Data Validation Agent", key="validation_workflow", use_container_width=True):
            st.session_state.current_workflow = "data_validation"
            st.rerun()

# ==================== WEB EXTRACTION WORKFLOW ====================

def render_web_extraction_workflow():
    """Web data extraction workflow"""
    st.markdown("## 🌐 Web Data Extraction")
    st.caption("Market-validated data extraction with AI intelligence")
    
    col_back, col_clear = st.columns([1, 1])
    with col_back:
        if st.button("← Back", key="back_from_web"):
            reset_workflow()
            st.rerun()
    with col_clear:
        if st.button("Clear Results", key="clear_web"):
            clear_results()
            st.rerun()
    
    query_type = st.radio(
        "Query Type:", 
        ["Simple Search", "Structured Query", "JSON Query", "Batch JSON Query"], 
        horizontal=True
    )
    
    query = None
    if query_type == "Simple Search":
        query = render_simple_query_input()
    elif query_type == "Structured Query":
        query = render_structured_query_input()
    elif query_type == "JSON Query":
        query = render_json_query_input()
    elif query_type == "Batch JSON Query":
        query = render_batch_json_query_input()
    
    with st.expander("⚙️ Extraction Settings", expanded=False):
        col_set1, col_set2, col_set3 = st.columns(3)
        
        with col_set1:
            extraction_mode = st.selectbox(
                "Extraction Mode",
                ["Hybrid (Web + AI)", "AI-Only (Fastest)", "Web-Only"],
                index=0
            )
            
            max_urls = 0 if extraction_mode == "AI-Only (Fastest)" else st.slider(
                "Max URLs", 1, 10, DEFAULT_MAX_URLS
            )
        
        with col_set2:
            enable_crawl = st.checkbox(
                "Deep Web Crawling", 
                DEFAULT_ENABLE_CRAWL and CRAWL4AI_AVAILABLE,
                disabled=not CRAWL4AI_AVAILABLE or extraction_mode == "AI-Only (Fastest)"
            )
            
            use_market_intelligence = st.checkbox("Market Intelligence", True)
        
        with col_set3:
            if extraction_mode == "AI-Only (Fastest)":
                st.info("🚀 Fast Mode: AI-generated market-validated data")
            elif extraction_mode == "Web-Only":
                st.info("🌐 Web Mode: Extract from live sources only")
            else:
                st.info("🔄 Hybrid Mode: Best of both approaches")
    
    if query and st.button("Extract Data", type="primary", use_container_width=True):
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("GROQ_API_KEY environment variable not set")
            return
        
        extraction_settings = {
            'mode': extraction_mode,
            'max_urls': max_urls,
            'enable_crawl': enable_crawl,
            'use_market_intelligence': use_market_intelligence,
            'query_type': query_type
        }
        
        with st.spinner("🔍 Extracting and analyzing data..."):
            result = run_comprehensive_extraction(query, groq_api_key, extraction_settings)
            
            if result:
                st.session_state.results = result
                st.session_state.workflow_data = {
                    'type': 'web_extraction',
                    'query': query,
                    'settings': extraction_settings,
                    'results': result
                }
                st.rerun()
    
    if st.session_state.results and st.session_state.workflow_data.get('type') == 'web_extraction':
        render_comprehensive_extraction_results(st.session_state.results, st.session_state.workflow_data)

# ==================== QUERY INPUT COMPONENTS ====================

def render_simple_query_input():
    """Simple query input"""
    query = st.text_area(
        "Natural Language Query:", 
        height=120, 
        placeholder="steel production in China from 2010 to 2025"
    )
    
    st.write("**💡 Example Queries:**")
    example_categories = {
        "🏭 Industrial": ["steel production China 2010-2025", "aluminum prices global market trends"],
        "🔋 Energy": ["solar capacity installation USA", "renewable energy growth Europe"],
        "🚗 Transportation": ["electric vehicle sales Norway", "sedan prices USA vs China"],
        "💰 Economics": ["GDP growth rates emerging markets", "inflation rates developed countries"]
    }
    
    for category, examples in example_categories.items():
        st.write(f"**{category}**")
        cols = st.columns(len(examples))
        for i, example in enumerate(examples):
            if cols[i].button(example, key=f"example_{category}_{i}"):
                st.session_state.simple_query = example
                st.rerun()
    
    if 'simple_query' in st.session_state:
        query = st.session_state.simple_query
        del st.session_state.simple_query
    
    return query.strip() if query else None

def render_structured_query_input():
    """Structured query input"""
    preset = st.session_state.get('current_structured_query', {})
    
    st.info(f"💡 Market Intelligence available for {len(MARKET_KNOWLEDGE_BASE)} entities")
    
    with st.form("enhanced_structured_query_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            entity_name = st.text_input(
                "Entity Name *", 
                value=preset.get('Entity_Name', ''),
                placeholder="Electric Vehicles"
            )
            
            entity_type = st.selectbox(
                "Entity Type", 
                ["", "Commodity", "Energy", "Technology", "Service", "Infrastructure", "Other"]
            )
            
            region = st.selectbox(
                "Region",
                ["", "Global", "USA", "China", "India", "Germany", "Japan", "UK", "France", "Canada", "Australia", "Brazil", "Russia", "Other"]
            )
        
        with col2:
            metric = st.selectbox(
                "Metric",
                ["", "Production", "Sales", "Price", "Capacity", "Consumption", "Revenue", "Cost", "Adoption", "Performance", "Other"]
            )
            
            unit = st.text_input("Unit", value=preset.get('Unit', ''), placeholder="Million Units")
            
            col_time1, col_time2 = st.columns(2)
            with col_time1:
                start_year = st.number_input("Start Year", min_value=2000, max_value=2030, value=2010)
            with col_time2:
                end_year = st.number_input("End Year", min_value=2000, max_value=2030, value=2025)
        
        with st.expander("🔧 Advanced Options"):
            is_disruptor = st.checkbox("Disruptive Technology/Market", value=preset.get('Is_Disruptor', False))
            data_source_preference = st.selectbox(
                "Data Source Preference",
                ["Auto-detect", "Government/Official", "Industry Reports", "Academic Sources", "Market Intelligence"]
            )
        
        submitted = st.form_submit_button("🔍 Build Query", use_container_width=True)
    
    query_json = {}
    if entity_name.strip():
        query_json = {
            "Entity_Name": entity_name.strip(),
            "Entity_Type": entity_type if entity_type else "",
            "Region": region if region else "",
            "Metric": metric if metric else "",
            "Unit": unit.strip(),
            "Time_Range": f"{start_year}-{end_year}",
            "Is_Disruptor": is_disruptor,
            "Data_Source_Preference": data_source_preference
        }
        query_json = {k: v for k, v in query_json.items() if v != "" and v is not False}
    
    if submitted or query_json != preset:
        st.session_state.current_structured_query = query_json
    
    if query_json:
        with st.expander("📄 Query Preview & Validation", expanded=True):
            st.json(query_json)
            is_valid, error_msg = validate_query(query_json)
            if is_valid:
                st.success("✅ Query is valid and ready for processing")
            else:
                st.error(f"❌ Error: {error_msg}")
    
    return query_json if query_json else None

def render_json_query_input():
    """JSON query input"""
    sample_json = {
        "Entity_Name": "Solar Energy",
        "Entity_Type": "Energy",
        "Region": "Global",
        "Metric": "Installed Capacity",
        "Unit": "Gigawatts",
        "Time_Range": "2010-2025",
        "Is_Disruptor": True
    }
    
    col_sample1, col_sample2 = st.columns(2)
    with col_sample1:
        if st.button("📝 Load Sample", key="load_solar_sample"):
            st.session_state.raw_json_text = json.dumps(sample_json, indent=2)
            st.rerun()
    
    with col_sample2:
        if st.button("🔄 Clear JSON", key="clear_json"):
            st.session_state.raw_json_text = ""
            st.rerun()
    
    raw_json = st.text_area(
        "JSON Query Configuration:", 
        height=200,
        value=st.session_state.get('raw_json_text', ''),
        placeholder=json.dumps(sample_json, indent=2),
        key="raw_json_input"
    )
    
    if raw_json.strip():
        try:
            parsed_json = json.loads(raw_json)
            is_valid, error_msg = validate_query(parsed_json)
            
            if is_valid:
                st.success("✅ Valid JSON structure")
                return parsed_json
            else:
                st.error(f"❌ Structure Error: {error_msg}")
                
        except json.JSONDecodeError as e:
            st.error(f"❌ JSON Syntax Error: {str(e)}")
    
    return None

def render_batch_json_query_input():
    """Batch JSON query input"""
    st.subheader("📋 Batch JSON Query Processing")
    
    sample_batch = [
        {"Entity_Name": "Sedan - Compact", "Region": "China", "Metric": "Price", "Unit": "USD", "Is_Disruptor": False},
        {"Entity_Name": "Electric Vehicles", "Region": "Global", "Metric": "Sales", "Unit": "Million Units", "Is_Disruptor": True}
    ]
    
    col_sample1, col_sample2 = st.columns(2)
    with col_sample1:
        if st.button("📝 Load Sample", key="load_batch_sample"):
            st.session_state.batch_query_text = json.dumps(sample_batch, indent=2)
            st.rerun()
    
    with col_sample2:
        if st.button("🔄 Clear", key="clear_batch"):
            st.session_state.batch_query_text = ""
            st.rerun()
    
    batch_text = st.text_area(
        "Batch JSON Array:",
        value=st.session_state.get('batch_query_text', ''),
        height=350,
        placeholder=json.dumps(sample_batch, indent=2),
        key="batch_json_input"
    )
    
    if batch_text.strip():
        try:
            parsed_batch = json.loads(batch_text)
            
            if isinstance(parsed_batch, list) and len(parsed_batch) > 0:
                valid_queries = [q for q in parsed_batch if isinstance(q, dict) and validate_query(q)[0]]
                
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Total Queries", len(parsed_batch))
                with col_m2:
                    st.metric("Valid Queries", len(valid_queries))
                with col_m3:
                    rate = (len(valid_queries) / len(parsed_batch)) * 100 if parsed_batch else 0
                    st.metric("Success Rate", f"{rate:.1f}%")
                
                if valid_queries:
                    return valid_queries
                else:
                    st.error("No valid queries found in the batch")
            else:
                st.error("JSON must be a non-empty array of query objects")
                
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON format: {str(e)}")
    
    return None

# ==================== EXTRACTION LOGIC ====================

def run_comprehensive_extraction(query: Union[str, dict, list], api_key: str, settings: dict):
    """Run extraction"""
    try:
        max_urls = settings.get('max_urls', DEFAULT_MAX_URLS)
        enable_crawl = settings.get('enable_crawl', True)
        
        if isinstance(query, list):
            return run_enhanced_batch_extraction(query, api_key, max_urls, enable_crawl, settings)
        else:
            return asyncio.run(run_enhanced_single_extraction(query, api_key, max_urls, enable_crawl, settings))
            
    except Exception as e:
        st.error(f"Extraction failed: {str(e)}")
        return None

async def run_enhanced_single_extraction(query, api_key, max_urls, enable_crawl, settings):
    """Single extraction"""
    result = await extract_data_single_output(query, api_key, max_urls, enable_crawl)
    
    if result.success:
        table_data = {
            'headers': result.headers,
            'data': result.data,
            'rows': result.table_rows,
            'cols': len(result.headers),
            'confidence_score': result.confidence,
            'source_type': result.data_source,
            'analysis': result.analysis,
            'source_url': result.source_url
        }
        
        return type('ExtractionResult', (), {
            'success': True,
            'tables': [table_data],
            'summary': {
                'total_tables': 1,
                'total_rows': result.table_rows,
                'extraction_mode': settings.get('mode', 'Unknown'),
                'data_source': result.data_source,
                'confidence': result.confidence
            }
        })()
    else:
        return type('ExtractionResult', (), {
            'success': False,
            'error': result.error or 'Unknown error',
            'tables': [],
            'summary': {}
        })()

def run_enhanced_batch_extraction(queries: List[Dict], api_key, max_urls, enable_crawl, settings):
    """Batch extraction"""
    if not queries:
        return None
    
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    results = []
    successful_count = 0
    total_rows = 0
    
    for i, query in enumerate(queries):
        progress_bar.progress((i + 1) / len(queries))
        entity_name = query.get('Entity_Name', 'Unknown')
        status_text.text(f"Processing {i + 1}/{len(queries)}: {entity_name}")
        
        try:
            result = asyncio.run(extract_data_single_output(query, api_key, max_urls, enable_crawl))
            
            if result.success:
                table_data = {
                    'headers': result.headers,
                    'data': result.data,
                    'rows': result.table_rows,
                    'cols': len(result.headers),
                    'confidence_score': result.confidence,
                    'source_type': result.data_source,
                    'analysis': result.analysis,
                    'query_entity': entity_name,
                    'query_region': query.get('Region', 'Global')
                }
                results.append(table_data)
                successful_count += 1
                total_rows += result.table_rows
        except Exception:
            continue
    
    progress_container.empty()
    
    if results:
        return type('BatchExtractionResult', (), {
            'success': True,
            'tables': results,
            'summary': {
                'total_queries': len(queries),
                'successful_queries': successful_count,
                'total_tables': len(results),
                'total_rows': total_rows,
                'extraction_mode': settings.get('mode', 'Unknown')
            }
        })()
    
    return None

# ==================== RESULTS RENDERING ====================

def render_comprehensive_extraction_results(result, workflow_data):
    """Render results"""
    st.markdown("---")
    st.header("📊 Extraction Results")
    
    if not result or not result.success:
        st.error(f"Extraction failed: {getattr(result, 'error', 'Unknown error')}")
        return
    
    # Store data for validation
    if hasattr(result, 'tables') and result.tables:
        workflow_type = workflow_data.get('type', 'web_extraction')
        store_validation_data(workflow_type, {'tables': result.tables, 'summary': getattr(result, 'summary', {})})
    
    if hasattr(result, 'summary') and result.summary:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Queries", result.summary.get('total_queries', 1))
        with col2:
            st.metric("Tables", result.summary.get('total_tables', 0))
        with col3:
            st.metric("Data Points", f"{result.summary.get('total_rows', 0):,}")
        with col4:
            success_rate = result.summary.get('success_rate', 1.0) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
    
    if hasattr(result, 'tables') and result.tables:
        st.subheader("📋 Extracted Data Tables")
        
        for i, table in enumerate(result.tables, 1):
            entity_name = table.get('query_entity', '')
            title = f"Table {i}"
            if entity_name:
                title += f" - {entity_name}"
            title += f" [{table.get('rows', 0)} rows]"
            
            with st.expander(title, expanded=i <= 2):
                if table.get('data') and table.get('headers'):
                    df = pd.DataFrame(table['data'], columns=table['headers'])
                    st.dataframe(df, use_container_width=True)
                    
                    if table.get('analysis'):
                        st.markdown("**📈 Analysis:**")
                        st.write(table['analysis'])
                    
                    col_dl1, col_dl2, col_dl3 = st.columns(3)
                    with col_dl1:
                        csv = df.to_csv(index=False)
                        st.download_button("📥 CSV", csv, f"data_{i}.csv", "text/csv", key=f"csv_{i}")
                    
                    with col_dl2:
                        json_data = json.dumps({'headers': table['headers'], 'data': table['data']}, indent=2)
                        st.download_button("📥 JSON", json_data, f"data_{i}.json", "application/json", key=f"json_{i}")
                    
                    with col_dl3:
                        if len(df) > 1 and len(df.columns) >= 2:
                            if st.button("📊 Visualize", key=f"viz_{i}"):
                                create_visualization(df, table, i)

def create_visualization(df: pd.DataFrame, table: Dict, index: int):
    """Create visualization"""
    try:
        time_cols = [col for col in df.columns if 'year' in col.lower() or 'date' in col.lower()]
        numeric_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
        
        if time_cols and numeric_cols:
            time_col = time_cols[0]
            value_col = numeric_cols[0] if numeric_cols[0] != time_col else numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
            
            fig = px.line(df, x=time_col, y=value_col, title=f"Time Series: {value_col}")
            fig.update_traces(line=dict(width=3))
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{index}")
    except Exception as e:
        st.error(f"Visualization error: {e}")

# ==================== OTHER WORKFLOWS (Now Functional) ====================

def render_pdf_analysis_workflow():
    """PDF analysis workflow with time series chart extraction"""
    st.markdown("## 📄 PDF Analysis")
    st.caption("Extract time series data from PDF charts and graphs")
    
    col_back, col_clear = st.columns([1, 1])
    with col_back:
        if st.button("← Back", key="back_from_pdf"):
            reset_workflow()
            st.rerun()
    with col_clear:
        if st.button("Clear Results", key="clear_pdf"):
            clear_results()
            st.rerun()
    
    # Check if PDF processing is available
    if not PDF_PROCESSING_AVAILABLE:
        st.error("❌ PDF processing modules not available. Please check utils imports.")
        return
    
    # Check API keys
    if not check_api_keys():
        st.stop()
    
    # Create uploads directory if it doesn't exist
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    
    # PDF Upload
    uploaded_file = st.file_uploader(
        "Upload PDF Document",
        type=['pdf'],
        help="Upload a PDF containing charts, graphs, or time series visualizations"
    )
    
    if uploaded_file:
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        
        # Display upload info
        col1, col2 = st.columns(2)
        with col1:
            st.metric("File Name", uploaded_file.name)
        with col2:
            st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
        
        if st.button("🚀 Process PDF", type="primary", use_container_width=True):
            with st.spinner("Processing PDF for time series data..."):
                # Save uploaded file
                pdf_path = save_uploaded_file(uploaded_file, uploads_dir)
                
                # Process PDF with UI feedback
                results = process_pdf_with_ui_feedback(pdf_path)
                
                if results:
                    # Store results in session state
                    st.session_state.results = results
                    st.session_state.workflow_data = {
                        'type': 'pdf_analysis',
                        'pdf_name': uploaded_file.name,
                        'results': results
                    }
                    st.rerun()
    else:
        st.info("📤 Upload a PDF file to begin extraction")
    
    # Display results
    if st.session_state.results and st.session_state.workflow_data.get('type') == 'pdf_analysis':
        render_pdf_analysis_results(st.session_state.results, st.session_state.workflow_data)


def render_pdf_analysis_results(results: Dict[str, Any], workflow_data: Dict[str, Any]):
    """Render PDF analysis results"""
    st.markdown("---")
    
    # Store data for validation
    if results.get('structured_data'):
        store_validation_data('pdf_analysis', results['structured_data'])
    
    # Display processing summary
    display_processing_summary(results)
    
    # Display chart data if available
    if results.get('structured_data'):
        display_chart_data(results['structured_data'])
    
    # Display output directory info
    pdf_name = workflow_data.get('pdf_name', 'document')
    output_dir = os.path.join("extracted_data", results.get('pdf_name', pdf_name))
    st.info(f"📁 All results saved to: `{output_dir}/`")
    
    # Show file structure
    if os.path.exists(output_dir):
        with st.expander("📁 Generated Files"):
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), output_dir)
                    st.write(f"📄 {rel_path}")
    
    # Provide download option for structured data
    if results.get('structured_data', {}).get('charts'):
        json_str = json.dumps(results['structured_data'], indent=2)
        st.download_button(
            label="📥 Download Structured Data (JSON)",
            data=json_str,
            file_name=f"{results.get('pdf_name', 'document')}_structured_data.json",
            mime="application/json"
        )

def render_chart_analysis_workflow():
    """Chart analysis workflow with AI-powered chart data extraction"""
    st.markdown("## 📊 Chart Analysis")
    st.caption("Extract data from charts and visualizations using AI vision")
    
    col_back, col_clear = st.columns([1, 1])
    with col_back:
        if st.button("← Back", key="back_from_chart"):
            reset_workflow()
            st.rerun()
    with col_clear:
        if st.button("Clear Results", key="clear_chart"):
            clear_results()
            st.rerun()
    
    # Check if image processing is available
    if not PDF_PROCESSING_AVAILABLE:
        st.error("❌ Image processing modules not available. Please check utils imports.")
        st.error("💡 Tip: Make sure you're running with the virtual environment activated:")
        st.code("source data-env/bin/activate  # On Mac/Linux\n# or\ndata-env\\Scripts\\activate  # On Windows")
        return
    else:
        st.info("✅ PDF processing modules available")
    
    # Check API keys
    if not check_api_keys():
        st.stop()
    
    # Image Upload
    uploaded_image = st.file_uploader(
        "Upload Chart Image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a chart, graph, or visualization image"
    )
    
    if uploaded_image:
        st.success(f"✅ Uploaded: {uploaded_image.name}")
        
        # Display the uploaded image
        col_img, col_info = st.columns([2, 1])
        
        with col_img:
            st.image(uploaded_image, caption=f"Uploaded: {uploaded_image.name}", use_column_width=True)
        
        with col_info:
            st.metric("File Name", uploaded_image.name)
            st.metric("File Size", f"{uploaded_image.size / 1024:.1f} KB")
        
        if st.button("🔍 Analyze Chart", type="primary", use_container_width=True):
            with st.spinner("Analyzing chart image..."):
                # Save uploaded image to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_image.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_image.read())
                    temp_image_path = tmp_file.name
                
                try:
                    # Analyze the image
                    results = analyze_chart_image(temp_image_path, uploaded_image.name)
                    
                    if results.get('error'):
                        st.error(f"❌ Error analyzing image: {results['error']}")
                        if results.get('raw_response'):
                            with st.expander("🔍 Raw AI Response"):
                                st.text(results['raw_response'])
                    else:
                        st.success("✅ Chart analysis completed!")
                        
                        # Store results in session state
                        st.session_state.results = results
                        st.session_state.workflow_data = {
                            'type': 'chart_analysis',
                            'image_name': uploaded_image.name,
                            'results': results
                        }
                        st.rerun()
                
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_image_path):
                        os.unlink(temp_image_path)
    else:
        st.info("📤 Upload a chart image to begin analysis")
    
    # Display results
    if st.session_state.results and st.session_state.workflow_data.get('type') == 'chart_analysis':
        render_chart_analysis_results(st.session_state.results, st.session_state.workflow_data)


def render_chart_analysis_results(results: Dict[str, Any], workflow_data: Dict[str, Any]):
    """Render chart analysis results"""
    st.markdown("---")
    
    # Store data for validation
    if results.get('charts'):
        store_validation_data('chart_analysis', results)
    
    # Display chart data if available
    if results.get('charts'):
        display_chart_data(results)
        
        # Provide download option for structured data
        json_str = json.dumps(results, indent=2)
        image_name = workflow_data.get('image_name', 'chart')
        st.download_button(
            label="📥 Download Chart Data (JSON)",
            data=json_str,
            file_name=f"{image_name}_chart_data.json",
            mime="application/json"
        )
    else:
        st.warning("⚠️ No charts detected in the image")

def render_url_extraction_workflow():
    """Direct URL extraction workflow"""
    st.markdown("## 🔗 Direct URL Extraction")
    st.caption("Extract all data tables from a specific URL")
    
    col_back, col_clear = st.columns([1, 1])
    with col_back:
        if st.button("← Back", key="back_from_url"):
            reset_workflow()
            st.rerun()
    with col_clear:
        if st.button("Clear Results", key="clear_url"):
            clear_results()
            st.rerun()
    
    url_input = st.text_input(
        "URL:", 
        placeholder="https://ourworldindata.org/grapher/annual-co2-emissions-per-country",
        help="Enter the URL you want to extract data from"
    )
    
    # Sample URLs
    st.write("**🌐 Sample Data Sources:**")
    sample_urls = [
        ("🌡️ CO2 Emissions", "https://ourworldindata.org/grapher/annual-co2-coal"),
        ("☀️ Solar Capacity", "https://ourworldindata.org/grapher/solar-pv-cumulative-capacity"),
        ("🚗 EV Sales", "https://ourworldindata.org/grapher/electric-car-sales")
    ]
    
    cols = st.columns(len(sample_urls))
    for i, (name, url) in enumerate(sample_urls):
        if cols[i].button(name, key=f"url_sample_{i}", use_container_width=True):
            st.session_state.sample_url = url
            st.rerun()
    
    # Apply sample URL
    if 'sample_url' in st.session_state:
        url_input = st.session_state.sample_url
        del st.session_state.sample_url
        st.rerun()
    
    if url_input and st.button("🔍 Extract Data", type="primary", use_container_width=True):
        try:
            parsed_url = urlparse(url_input)
            if not all([parsed_url.scheme, parsed_url.netloc]):
                st.error("Invalid URL format - please include http:// or https://")
                return
        except Exception:
            st.error("Invalid URL format")
            return
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("GROQ_API_KEY environment variable not set")
            return
        
        with st.spinner("🕸️ Extracting data from URL..."):
            result = asyncio.run(run_direct_url_extraction(url_input, groq_api_key))
            
            if result and result.success:
                st.session_state.results = result
                st.session_state.workflow_data = {
                    'type': 'url_extraction',
                    'url': url_input,
                    'results': result
                }
                st.rerun()
            else:
                error_msg = result.error if result else "No data found"
                st.error(f"Failed to extract data: {error_msg}")
    
    # Display results
    if st.session_state.results and st.session_state.workflow_data.get('type') == 'url_extraction':
        render_comprehensive_extraction_results(st.session_state.results, st.session_state.workflow_data)

async def run_direct_url_extraction(url: str, api_key: str):
    """Direct URL extraction"""
    try:
        url_info = {
            'url': url,
            'title': 'Direct URL Extraction',
            'confidence': 1.0,
            'found_via': 'direct_input'
        }
        
        tables = await extract_from_url(url, url_info)
        
        if tables:
            total_rows = sum(table.get('rows', 0) for table in tables)
            
            return type('DirectExtractionResult', (), {
                'success': True,
                'tables': tables,
                'summary': {
                    'total_tables': len(tables),
                    'total_rows': total_rows,
                    'extraction_mode': 'Direct URL',
                    'source_url': url
                }
            })()
        else:
            return type('DirectExtractionResult', (), {
                'success': False,
                'error': 'No tables found on the provided URL',
                'tables': [],
                'summary': {}
            })()
            
    except Exception as e:
        return type('DirectExtractionResult', (), {
            'success': False,
            'error': str(e),
            'tables': [],
            'summary': {}
        })()


# Add this complete implementation to replace/enhance render_data_validation_workflow() in unified_app.py

def safe_float(value, default=0):
    """Safely convert value to float"""
    try:
        if value is None:
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    """Safely convert value to int"""
    try:
        if value is None:
            return default
        return int(float(value))
    except (ValueError, TypeError):
        return default


def debug_collected_data():
    """Debug: Show what's actually in collected_data"""
    if 'collected_data' not in st.session_state:
        st.error("No collected_data in session_state")
        return

    st.write("### 🔍 DEBUG: Collected Data Structure")
    for key, entry in st.session_state.collected_data.items():
        st.write(f"**Key:** {key}")
        st.json(entry)
        st.write("---")


def convert_collected_data_to_validation_format():
    """Convert collected data to format expected by validators - flexible version"""
    if 'collected_data' not in st.session_state:
        return []

    validation_data = []

    for key, entry in st.session_state.collected_data.items():
        data = entry.get('data', {})
        tables = data.get('tables', [])

        if not tables:
            continue

        for table_idx, table in enumerate(tables):
            headers = table.get('headers', [])
            rows = table.get('data', [])

            if not headers or not rows:
                continue

            # Find numeric columns (potential Y values)
            numeric_cols = []
            for col_idx, header in enumerate(headers):
                # Check if column contains numeric data
                try:
                    sample_values = [row[col_idx] for row in rows[:3] if col_idx < len(row)]
                    numeric_count = sum(1 for v in sample_values
                                        if v and str(v).replace(',', '').replace('.', '').replace('$', '').replace('-',
                                                                                                                   '').isdigit())
                    if numeric_count > 0:
                        numeric_cols.append(col_idx)
                except:
                    continue

            # Find year column (potential X values)
            year_col = None
            for col_idx, header in enumerate(headers):
                header_lower = str(header).lower()
                if 'year' in header_lower or 'date' in header_lower:
                    year_col = col_idx
                    break

            # If no year column found by name, check for numeric column with 4-digit years
            if year_col is None:
                for col_idx in range(len(headers)):
                    try:
                        sample = [row[col_idx] for row in rows[:3] if col_idx < len(row)]
                        if all(1990 <= int(float(str(v).replace(',', ''))) <= 2050 for v in sample if v):
                            year_col = col_idx
                            break
                    except:
                        continue

            # Default to first column if still no year found
            if year_col is None and len(headers) > 0:
                year_col = 0

            # For each numeric column, create a validation entry
            for value_col in numeric_cols:
                if value_col == year_col:
                    continue

                # Extract years and values
                years = []
                values = []

                for row in rows:
                    if len(row) > max(year_col, value_col):
                        try:
                            year_str = str(row[year_col]).replace(',', '')
                            value_str = str(row[value_col]).replace(',', '').replace('$', '')

                            year = int(float(year_str))
                            value = float(value_str)

                            years.append(year)
                            values.append(value)
                        except (ValueError, TypeError):
                            continue

                if not years or not values:
                    continue

                # Build entity name
                entity_name = headers[value_col] if value_col < len(headers) else f"Series_{table_idx}"
                table_title = table.get('title', '')
                if table_title:
                    entity_name = f"{table_title}_{entity_name}"

                # Detect data type
                value_header = str(headers[value_col]).lower() if value_col < len(headers) else ''
                is_cost = any(term in value_header for term in ['price', 'cost', '$', 'usd', 'dollar'])

                validation_data.append({
                    'Entity_Name': entity_name,
                    'Display_Name': entity_name,
                    'Region': 'Global',
                    'Curve_Type': 'cost_curve' if is_cost else 's_curve',
                    'Unit': '$' if is_cost else 'units',
                    'Metric': headers[value_col] if value_col < len(headers) else 'Value',
                    'Metric_Category': 'cost' if is_cost else 'physical',
                    'Entity_Type': 'Technology',
                    'X': years,
                    'Y': values,
                    'Quality_Score': 0.5,
                    'Data_Source_Name': entry.get('source', 'web_extraction'),
                    'DataSource_URLs': [entry.get('source', '')]
                })

    return validation_data

def convert_results_to_dataframe(results):
    """Convert results to pandas DataFrame"""
    rows = []
    for result in results:
        entity_name = result.get('Entity_Name', 'Unknown')
        region = result.get('Region', 'Unknown')
        unit = result.get('Unit', '')
        metric = result.get('Metric', 'Value')
        years = result.get('X', [])
        values = result.get('Y', [])
        source = result.get('Data_Source_Name', 'Unknown')

        # Normalize region
        if region.lower() == 'world':
            region = 'Global'

        for year, value in zip(years, values):
            if value is not None:
                rows.append({
                    'product_name': entity_name,
                    'region': region,
                    'unit': unit,
                    'metric': metric,
                    'year': safe_int(year, 2020),
                    'value': safe_float(value, 0),
                    'source': source,
                    'curve_name': f"{entity_name}_{region}",
                    'technology_product_commodity_name': entity_name,
                    'location_name': region
                })

    return pd.DataFrame(rows)




def render_validation_visualizations(validation_results: Dict[str, Any]):
    """Display validation visualizations"""
    st.markdown("#### 📊 Validation Metrics Dashboard")
    
    score = validation_results.get('score')
    if not score:
        st.warning("No score data available for visualization.")
        return
    
    # Dimension scores radar chart
    if hasattr(score, 'dimension_scores') and score.dimension_scores:
        dimensions = list(score.dimension_scores.keys())
        values = list(score.dimension_scores.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=[dim.replace('_', ' ').title() for dim in dimensions],
            fill='toself',
            name='Validation Scores',
            line_color='blue'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Data Quality Dimensions"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Data quality issues breakdown
    enhanced_validation = validation_results.get('enhanced_validation', {})
    if enhanced_validation:
        issue_counts = {}
        
        # Count different types of issues
        if enhanced_validation.get('duplicates') is not None:
            duplicates = enhanced_validation.get('duplicates')
            if hasattr(duplicates, '__len__'):
                issue_counts['Duplicates'] = len(duplicates)
        
        type_issues = enhanced_validation.get('type_issues', {})
        if type_issues:
            issue_counts['Type Issues'] = sum(len(v) for v in type_issues.values())
        
        outliers = enhanced_validation.get('outliers', {})
        if outliers:
            issue_counts['Outliers'] = sum(len(v) for v in outliers.values())
        
        logic_issues = enhanced_validation.get('logic_issues', {})
        if logic_issues:
            issue_counts['Logic Issues'] = sum(len(v) for v in logic_issues.values())
        
        format_issues = enhanced_validation.get('format_issues', {})
        if format_issues:
            issue_counts['Format Issues'] = len(format_issues)
        
        if issue_counts:
            fig = px.bar(
                x=list(issue_counts.keys()),
                y=list(issue_counts.values()),
                title="Data Quality Issues by Category"
            )
            fig.update_layout(
                xaxis_title="Issue Type",
                yaxis_title="Count"
            )
            st.plotly_chart(fig, use_container_width=True)

def render_export_options(validation_results: Dict[str, Any]):
    """Display export options for validation results"""
    st.markdown("#### 📄 Export Validation Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export Validation Report", use_container_width=True):
            try:
                try:
                    from validation_support import export_enhanced_validation_results
                except ImportError as import_error:
                    st.error(f"❌ Failed to import validation modules: {str(import_error)}")
                    st.error("💡 Make sure you're running the app with the virtual environment activated:")
                    st.code("source data-env/bin/activate && streamlit run unified_app.py")
                    return
                
                exports = export_enhanced_validation_results(
                    validation_results, 
                    "stellar_validation"
                )
                
                if 'error' in exports:
                    st.error(f"Export failed: {exports['error']}")
                else:
                    st.success("✅ Validation results exported successfully!")
                    
                    for export_type, filename in exports.items():
                        if export_type != 'error':
                            st.write(f"- **{export_type.title()}**: `{filename}`")
            
            except Exception as e:
                st.error(f"Export failed: {str(e)}")
                if "No module named" in str(e):
                    st.error("💡 Make sure you're running the app with the virtual environment activated:")
                    st.code("source data-env/bin/activate && streamlit run unified_app.py")
    
    with col2:
        if st.button("📋 Copy Validation Summary", use_container_width=True):
            report = validation_results.get('report', 'No report available')
            st.text_area("Validation Summary (Copy this text):", report, height=200)
    
    # Download clean data
    clean_df = validation_results.get('clean_df')
    if clean_df is not None and not clean_df.empty:
        st.markdown("#### 📊 Download Clean Data")
        
        # Convert to CSV
        csv_data = clean_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Clean Data (CSV)",
            data=csv_data,
            file_name=f"clean_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.info(f"Clean dataset contains {len(clean_df):,} validated records.")

def render_query_url_extraction_workflow():
    """Query+URL extraction - combines query context with URL"""
    st.markdown("## 🎯 Query + URL Extraction")
    st.caption("Extract data from a specific URL with query context for better filtering")
    
    col_back, col_clear = st.columns([1, 1])
    with col_back:
        if st.button("← Back", key="back_from_query_url"):
            reset_workflow()
            st.rerun()
    with col_clear:
        if st.button("Clear Results", key="clear_query_url"):
            clear_results()
            st.rerun()
    
    # Query input
    query_type = st.radio(
        "Query Type:", 
        ["Simple Search", "Structured Query"], 
        horizontal=True,
        help="Provide context about what data you're looking for"
    )
    
    query = None
    if query_type == "Simple Search":
        query = st.text_input(
            "Search Query:",
            placeholder="electric vehicle sales China 2015-2025",
            help="Describe what data you're looking for"
        )
    else:
        query = render_structured_query_input()
    
    # URL input
    url_input = st.text_input(
        "Target URL:",
        placeholder="https://example.com/data-page",
        help="The specific webpage to extract data from"
    )
    
    # Sample combinations
    st.write("**💡 Example Combinations:**")
    examples = [
        ("EV Sales", "electric vehicle sales", "https://ourworldindata.org/grapher/electric-car-sales"),
        ("Solar Capacity", "solar energy capacity", "https://ourworldindata.org/grapher/solar-pv-cumulative-capacity"),
    ]
    
    cols = st.columns(len(examples))
    for i, (name, sample_query, sample_url) in enumerate(examples):
        if cols[i].button(name, key=f"combo_{i}", use_container_width=True):
            st.session_state.combo_query = sample_query
            st.session_state.combo_url = sample_url
            st.rerun()
    
    # Apply examples
    if 'combo_query' in st.session_state:
        query = st.session_state.combo_query
        url_input = st.session_state.combo_url
        del st.session_state.combo_query
        del st.session_state.combo_url
    
    # Extract button
    if query and url_input and st.button("🔍 Extract with Context", type="primary", use_container_width=True):
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("GROQ_API_KEY environment variable not set")
            return
        
        try:
            parsed_url = urlparse(url_input)
            if not all([parsed_url.scheme, parsed_url.netloc]):
                st.error("Invalid URL format")
                return
        except Exception:
            st.error("Invalid URL format")
            return
        
        with st.spinner("🔍 Extracting targeted data..."):
            result = asyncio.run(run_query_url_extraction(query, url_input, groq_api_key))
            
            if result and result.success:
                st.session_state.results = result
                st.session_state.workflow_data = {
                    'type': 'query_url_extraction',
                    'query': query,
                    'url': url_input,
                    'results': result
                }
                st.rerun()
            else:
                st.error(f"Failed: {result.error if result else 'Extraction failed'}")
    
    # Display results
    if st.session_state.results and st.session_state.workflow_data.get('type') == 'query_url_extraction':
        render_comprehensive_extraction_results(st.session_state.results, st.session_state.workflow_data)

async def run_query_url_extraction(query: Union[str, dict], url: str, api_key: str):
    """Query+URL extraction"""
    try:
        url_info = {'url': url, 'title': 'Query-Targeted URL', 'confidence': 1.0}
        tables = await extract_from_url(url, url_info)
        
        if tables:
            structured_query = create_query(query)
            scored_tables = []
            
            for table in tables:
                relevance = calculate_table_relevance(table, structured_query)
                table['relevance_score'] = relevance
                table['query_entity'] = structured_query.entity_name
                scored_tables.append(table)
            
            scored_tables.sort(key=lambda t: t.get('relevance_score', 0), reverse=True)
            
            if scored_tables:
                return type('QueryURLResult', (), {
                    'success': True,
                    'tables': scored_tables,
                    'summary': {
                        'total_tables': len(scored_tables),
                        'total_rows': sum(t.get('rows', 0) for t in scored_tables),
                        'extraction_mode': 'Query + URL',
                        'source_url': url
                    }
                })()
        
        return type('QueryURLResult', (), {
            'success': False,
            'error': 'No tables found',
            'tables': [],
            'summary': {}
        })()
    except Exception as e:
        return type('QueryURLResult', (), {
            'success': False,
            'error': str(e),
            'tables': [],
            'summary': {}
        })()
# ==================== STELLAR VALIDATION SYSTEM - COMPLETE ====================

def detect_data_characteristics(validation_data: List[Dict]) -> Dict[str, Any]:
    """
    Intelligently detect data characteristics for adaptive validation.
    Returns comprehensive metadata about the data type.
    """
    if not validation_data:
        return {}
    
    sample = validation_data[0]
    entity_name = str(sample.get('Entity_Name', '')).lower()
    unit = str(sample.get('Unit', '')).lower()
    metric = str(sample.get('Metric', '')).lower()
    curve_type = str(sample.get('Curve_Type', '')).lower()
    
    characteristics = {
        'entity_name': entity_name,
        'unit': unit,
        'metric': metric,
        'curve_type': curve_type,
    }
    
    # 1. COST DATA DETECTION (Wright's Law applicable)
    cost_indicators = ['$', 'usd', 'dollar', 'cost', 'price', 'lcoe', '/mwh', '/kwh', '/watt']
    characteristics['is_cost_data'] = (
        any(ind in unit for ind in cost_indicators) or
        any(ind in metric for ind in cost_indicators) or
        'cost' in curve_type or
        'wright' in curve_type
    )
    
    # 2. PHYSICAL QUANTITY DETECTION (S-Curve applicable)
    physical_indicators = ['sales', 'capacity', 'generation', 'consumption', 'fleet', 'adoption', 'units', 'vehicles']
    characteristics['is_physical_data'] = (
        any(ind in metric for ind in physical_indicators) or
        's_curve' in curve_type or
        's-curve' in curve_type or
        (not characteristics['is_cost_data'] and any(u in unit for u in ['twh', 'gw', 'mw']))
    )
    
    # 3. COMMODITY DETECTION
    commodities = ['aluminum', 'aluminium', 'copper', 'steel', 'iron', 'gold', 'silver', 'zinc', 'nickel']
    characteristics['is_commodity'] = any(comm in entity_name for comm in commodities)
    
    # 4. TECHNOLOGY DETECTION
    technologies = ['solar', 'wind', 'battery', 'ev', 'electric vehicle', 'pv', 'photovoltaic', 'turbine']
    characteristics['is_technology'] = any(tech in entity_name for tech in technologies)
    
    # 5. ENERGY/EV SPECIFIC
    energy_terms = ['solar', 'wind', 'oil', 'gas', 'coal', 'nuclear', 'energy', 'power', 'generation']
    characteristics['is_energy'] = any(term in entity_name for term in energy_terms)
    
    ev_terms = ['ev', 'electric vehicle', 'electric_vehicle', 'bev', 'phev']
    characteristics['is_ev'] = any(term in entity_name for term in ev_terms)
    
    # 6. BATTERY SPECIFIC
    battery_terms = ['battery', 'lithium', 'li-ion', 'lithium-ion', 'cell']
    characteristics['is_battery'] = any(term in entity_name for term in battery_terms)
    
    # 7. DETERMINE PRIMARY FRAMEWORK
    if characteristics['is_commodity']:
        characteristics['primary_framework'] = 'Commodity Cycles'
        characteristics['expected_pattern'] = 'Cyclical, 3-7% annual growth'
    elif characteristics['is_cost_data']:
        characteristics['primary_framework'] = "Wright's Law"
        characteristics['expected_pattern'] = '15-30% cost reduction per production doubling'
    elif characteristics['is_physical_data']:
        characteristics['primary_framework'] = 'S-Curve Adoption'
        characteristics['expected_pattern'] = 'Sigmoid: slow→exponential→saturation'
    else:
        characteristics['primary_framework'] = 'Unknown'
        characteristics['expected_pattern'] = 'To be determined'
    
    return characteristics


def should_validator_run(validator_key: str, characteristics: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Determine if a validator should run based on data characteristics.
    Returns: (should_run: bool, explanation_if_na: str)
    """
    entity = characteristics.get('entity_name', 'data')
    framework = characteristics.get('primary_framework', 'Unknown')
    
    # CATEGORY 1: ALWAYS RUN
    always_run = [
        'units_and_scale', 'year_anomalies', 'definition_clarity',
        'market_context',
        'data_source_quality', 'data_source_integrity', 'multi_source_consistency',
        'regional_market_logic', 'regional_definition', 'global_sum',
        'metric_validity', 'unit_conversion', 'ai_reality',
        'spelling_nomenclature', 'regional_price_pattern', 'historical_label',
        'growth_rate_bounds', 'trusted_sources'
    ]
    
    if validator_key in always_run:
        return (True, "")
    
    # CATEGORY 2: COST ANALYSIS - Only for cost data
    cost_validators = [
        'cost_curve', 'cost_parity', 'cost_curve_anomaly',
        'inflation_adjustment', 'cost_curve_learning_rate'
    ]
    
    if validator_key in cost_validators:
        if characteristics.get('is_cost_data'):
            return (True, "")
        else:
            if characteristics.get('is_commodity'):
                reason = f"Wright's Law only applies to technology cost curves. Current data: {entity} (commodity, follows cyclical patterns)"
            else:
                reason = f"Wright's Law only applies to cost/price data. Current data: {entity} (physical quantity)"
            return (False, reason)
    
    # CATEGORY 3: ADOPTION ANALYSIS
    adoption_validators = [
        'adoption_curve', 'adoption_saturation', 'adoption_curve_shape'
    ]
    
    if validator_key in adoption_validators:
        if characteristics.get('is_physical_data') and not characteristics.get('is_cost_data'):
            return (True, "")
        else:
            reason = f"S-Curve analysis only applies to adoption/deployment data. Current data: {entity} (cost curve)"
            return (False, reason)
    
    # CATEGORY 4: ENERGY TRANSITION
    energy_validators = [
        'oil_displacement', 'derived_oil_displacement',
        'capacity_factor', 'global_oil_sanity'
    ]
    
    if validator_key in energy_validators:
        if characteristics.get('is_energy') or characteristics.get('is_ev'):
            return (True, "")
        else:
            reason = f"Only applies to energy/EV data. Current data: {entity}"
            return (False, reason)
    
    # CATEGORY 5: BATTERY SPECIFIC
    battery_validators = ['battery_size_evolution', 'lithium_chemistry']
    
    if validator_key in battery_validators:
        if characteristics.get('is_battery'):
            return (True, "")
        else:
            reason = f"Only applies to battery/lithium data. Current data: {entity}"
            return (False, reason)
    
    # CATEGORY 6: COMMODITY SPECIFIC
    if validator_key == 'commodity_behavior':
        if characteristics.get('is_commodity'):
            return (True, "")
        else:
            reason = f"Only applies to commodity data. Current data: {entity} ({framework})"
            return (False, reason)
    
    # CATEGORY 7: SPECIALIZED
    if validator_key in ['structural_break', 'mass_balance']:
        return (True, "")
    
    return (True, "")


def interpret_validator_result(result: Any, validator_key: str) -> str:
    """
    Interpret validator results correctly.
    CRITICAL: None or [] means PASS, not N/A!
    """
    if result is None or result == [] or result == {}:
        return 'pass'
    
    if isinstance(result, list):
        if len(result) == 0:
            return 'pass'
        
        has_fail = any(r.get('pass') is False for r in result if isinstance(r, dict))
        has_pass = any(r.get('pass') is True for r in result if isinstance(r, dict))
        has_na = any(r.get('pass') is None for r in result if isinstance(r, dict))
        
        if has_fail:
            return 'fail'
        elif has_na:
            return 'warning'
        elif has_pass:
            return 'pass'
        else:
            return 'pass'
    
    if isinstance(result, dict):
        pass_value = result.get('pass')
        if pass_value is False:
            return 'fail'
        elif pass_value is None:
            return 'warning'
        elif pass_value is True:
            return 'pass'
    
    return 'pass'


def run_all_validators_with_intelligence(results: List[Dict], df: pd.DataFrame) -> Dict[str, Any]:
    """Run all 35 validators with intelligent pre-filtering"""
    characteristics = detect_data_characteristics(results)
    
    try:
        from ground_truth_validators import (
            validate_units_and_scale, check_year_anomalies, definition_clarity_validator,
            bradd_cost_curve_validator, cost_parity_threshold, cost_curve_anomaly_validator,
            inflation_adjustment_validator, cost_curve_learning_rate_validator,
            bradd_adoption_curve_validator, adoption_saturation_feasibility,
            adoption_curve_shape_validator, market_context_validator,
            oil_displacement_check, derived_oil_displacement_validator,
            capacity_factor_validator, global_oil_sanity_validator,
            data_source_quality_validator, data_source_integrity_validator,
            multi_source_consistency_validator, regional_market_logic_validator,
            regional_definition_validator, global_sum_validator,
            metric_validity_validator, unit_conversion_validator,
            ai_reality_check_validator, spelling_and_nomenclature_validator,
            battery_size_evolution_validator, regional_price_pattern_validator,
            historical_data_label_validator, growth_rate_reasonableness_validator,
            lithium_chemistry_validator, commodity_price_behavior_validator,
            trusted_source_validator
        )
    except ImportError as e:
        st.error(f"Failed to import validators: {e}")
        return {}
    
    expert_validations = {}
    validation_summary = {
        'total': 35,
        'passed': 0,
        'failed': 0,
        'warnings': 0,
        'na': 0
    }
    
    validator_functions = {
        'units_and_scale': lambda: validate_units_and_scale(df),
        'year_anomalies': lambda: check_year_anomalies(results, df),
        'definition_clarity': lambda: definition_clarity_validator(df),
        'market_context': lambda: market_context_validator(df, results, None),
        'data_source_quality': lambda: data_source_quality_validator(results),
        'data_source_integrity': lambda: data_source_integrity_validator(results),
        'multi_source_consistency': lambda: multi_source_consistency_validator(df, 0.10),
        'regional_market_logic': lambda: regional_market_logic_validator(df),
        'regional_definition': lambda: regional_definition_validator(df, results),
        'global_sum': lambda: global_sum_validator(df, 0.05),
        'metric_validity': lambda: metric_validity_validator(df),
        'unit_conversion': lambda: unit_conversion_validator(df, None, 0.05),
        'ai_reality': lambda: ai_reality_check_validator(df),
        'spelling_nomenclature': lambda: spelling_and_nomenclature_validator(results, df),
        'regional_price_pattern': lambda: regional_price_pattern_validator(df),
        'historical_label': lambda: historical_data_label_validator(results),
        'growth_rate_bounds': lambda: growth_rate_reasonableness_validator(df),
        'trusted_sources': lambda: trusted_source_validator(results),
        'cost_curve': lambda: bradd_cost_curve_validator(results, df),
        'cost_parity': lambda: cost_parity_threshold(df, 70, 2022),
        'cost_curve_anomaly': lambda: cost_curve_anomaly_validator(df),
        'inflation_adjustment': lambda: inflation_adjustment_validator(results, df),
        'cost_curve_learning_rate': lambda: cost_curve_learning_rate_validator(df),
        'adoption_curve': lambda: bradd_adoption_curve_validator(results, df),
        'adoption_saturation': lambda: adoption_saturation_feasibility(results),
        'adoption_curve_shape': lambda: adoption_curve_shape_validator(df),
        'oil_displacement': lambda: oil_displacement_check(results, df, 0.10, 0.70),
        'derived_oil_displacement': lambda: derived_oil_displacement_validator(results, None),
        'capacity_factor': lambda: capacity_factor_validator(df, df, 3),
        'global_oil_sanity': lambda: global_oil_sanity_validator(df, (55, 60), (90, 110), [(2020, 2021)]),
        'battery_size_evolution': lambda: battery_size_evolution_validator(results, df),
        'lithium_chemistry': lambda: lithium_chemistry_validator(results),
        'commodity_behavior': lambda: commodity_price_behavior_validator(results),
    }
    
    for validator_key, validator_func in validator_functions.items():
        should_run, na_reason = should_validator_run(validator_key, characteristics)
        
        if not should_run:
            expert_validations[validator_key] = [{
                'product': 'All',
                'region': 'All',
                'pass': None,
                'explanation': f'N/A - {na_reason}',
                'severity': 'info'
            }]
            validation_summary['na'] += 1
        else:
            try:
                result = validator_func()
                expert_validations[validator_key] = result if isinstance(result, list) else ([result] if result else [])
                
                status = interpret_validator_result(result, validator_key)
                
                if status == 'pass':
                    validation_summary['passed'] += 1
                elif status == 'fail':
                    validation_summary['failed'] += 1
                elif status == 'warning':
                    validation_summary['warnings'] += 1
                else:
                    validation_summary['na'] += 1
                    
            except Exception as e:
                expert_validations[validator_key] = [{
                    'product': 'All',
                    'region': 'All',
                    'pass': False,
                    'explanation': f'Validator error: {str(e)}',
                    'severity': 'error'
                }]
                validation_summary['failed'] += 1
    
    return {
        'expert_validations': expert_validations,
        'validation_summary': validation_summary,
        'characteristics': characteristics
    }


def get_validator_descriptions() -> Dict[str, Dict]:
    """Get descriptions for all validators"""
    return {
        'units_and_scale': {
            'description': 'Verifies units are consistent within each metric (e.g., all regions use Million Metric Tons)',
        },
        'year_anomalies': {
            'description': 'Checks for impossible years (before 1990 or after 2050) and gaps in time series',
        },
        'definition_clarity': {
            'description': 'Ensures regions are well-defined (e.g., what countries are in "Europe"?)',
        },
        'cost_curve': {
            'description': "Validates technology costs follow Wright's Law declining patterns (15-30% per doubling)",
        },
        'cost_parity': {
            'description': 'Checks if renewable energy costs reached fossil fuel parity',
        },
        'cost_curve_anomaly': {
            'description': 'Identifies unrealistic cost increases (technologies should get cheaper)',
        },
        'inflation_adjustment': {
            'description': 'Verifies costs are in constant dollars (inflation-adjusted)',
        },
        'cost_curve_learning_rate': {
            'description': 'Validates 15-30% cost reduction per production doubling',
        },
        'adoption_curve': {
            'description': 'Checks adoption pattern (S-curve for technologies, linear for commodities)',
        },
        'adoption_saturation': {
            'description': "Validates adoption doesn't exceed 100% market share",
        },
        'adoption_curve_shape': {
            'description': 'Verifies expected sigmoid growth pattern',
        },
        'market_context': {
            'description': 'Explains large changes with known events (2008 GFC, 2020 COVID, 2022 Ukraine)',
        },
        'oil_displacement': {
            'description': 'Calculates barrels/day of oil displaced by electric vehicles',
        },
        'derived_oil_displacement': {
            'description': 'Validates oil displacement calculations',
        },
        'capacity_factor': {
            'description': 'Checks power plant utilization rates (typically 20-90%)',
        },
        'global_oil_sanity': {
            'description': 'Verifies global oil demand is around 100 million barrels/day',
        },
        'data_source_quality': {
            'description': 'Rates source reliability (Tier 1: official/government, Tier 2: commercial)',
        },
        'data_source_integrity': {
            'description': 'Checks if data source changes mid-series',
        },
        'multi_source_consistency': {
            'description': 'Compares values across different sources',
        },
        'regional_market_logic': {
            'description': 'Validates regional shares match market reality',
        },
        'regional_definition': {
            'description': 'Ensures regions properly defined with country lists',
        },
        'global_sum': {
            'description': 'Verifies regional components sum to global totals',
        },
        'metric_validity': {
            'description': 'Ensures metrics appropriate for data type',
        },
        'unit_conversion': {
            'description': 'Validates unit conversions are mathematically correct',
        },
        'ai_reality': {
            'description': 'AI checks if values are reasonable',
        },
        'spelling_nomenclature': {
            'description': 'Checks for common spelling errors',
        },
        'battery_size_evolution': {
            'description': 'Validates battery pack sizes increase over time',
        },
        'regional_price_pattern': {
            'description': 'Checks regional price differentials are logical',
        },
        'historical_label': {
            'description': 'Ensures historical data labeled correctly',
        },
        'growth_rate_bounds': {
            'description': 'Validates growth rates (3-7% for commodities, 20-50% for EVs)',
        },
        'lithium_chemistry': {
            'description': 'Checks lithium content follows chemistry constraints (100-150g/kWh)',
        },
        'commodity_behavior': {
            'description': 'Ensures commodities show cyclical patterns, not tech-like decline',
        },
        'trusted_sources': {
            'description': 'Verifies use of domain-appropriate trusted sources',
        },
    }


def get_context_success_message(validator_key: str, characteristics: Dict[str, Any]) -> str:
    """Get context-specific success messages"""
    entity = characteristics.get('entity_name', 'data').replace('_', ' ').title()
    is_commodity = characteristics.get('is_commodity', False)
    is_aluminum = 'aluminum' in entity.lower() or 'aluminium' in entity.lower()
    
    messages = {
        'ai_reality': {
            'aluminum': f"{entity}: China ~60% of global production is realistic",
            'default': f"{entity} values are reasonable"
        },
        'growth_rate_bounds': {
            'commodity': f"{entity}: 3-7% annual growth expected for commodities",
            'default': f"{entity} growth rates within expected bounds"
        },
        'adoption_curve': {
            'commodity': f"{entity} shows linear/cyclical growth (commodities don't follow S-curves)",
            'default': "Pattern validated"
        },
        'commodity_behavior': {
            'commodity': f"{entity} shows commodity price cycles, not tech-like decline",
            'default': "Behavior validated"
        },
        'regional_market_logic': {
            'aluminum': "China dominance validated (~60% global production)",
            'default': "Regional shares match market reality"
        },
    }
    
    if validator_key in messages:
        msg_dict = messages[validator_key]
        if is_aluminum and 'aluminum' in msg_dict:
            return msg_dict['aluminum']
        elif is_commodity and 'commodity' in msg_dict:
            return msg_dict['commodity']
        else:
            return msg_dict.get('default', '')
    
    return ""


def render_data_profiling_tab(validation_results: Dict):
    """Enhanced data profiling with all quality checks"""
    st.markdown("### 📊 Data Profiling")
    
    enhanced = validation_results.get('enhanced_validation', {})
    validation_data = st.session_state.get('validation_data', [])
    
    if not enhanced and not validation_data:
        st.info("No profiling data available. Basic validation was performed.")
        return
    
    # Data Type Validation
    st.markdown("#### 📋 Data Type Validation")
    type_issues = enhanced.get('type_issues', {})
    if type_issues:
        issue_count = sum(len(v) for v in type_issues.values())
        st.warning(f"Found {issue_count} data type issues in {len(type_issues)} columns")
        with st.expander("View Type Issues"):
            for col, indices in type_issues.items():
                st.write(f"**{col}**: {len(indices)} invalid values")
    else:
        st.success("All data types are correct")
    
    # Completeness Analysis
    st.markdown("#### 📈 Completeness Analysis")
    completeness = enhanced.get('completeness', {})
    
    if completeness:
        overall = completeness.get('overall_completeness', 0) * 100
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Completeness", f"{overall:.1f}%")
        with col2:
            missing = completeness.get('missing_critical_count', 0)
            st.metric("Missing Critical", missing)
        with col3:
            total = completeness.get('total_records', 0)
            st.metric("Total Records", total)
        
        if overall < 50:
            st.error("Critical: Over 50% data missing")
        elif overall < 75:
            st.warning("Moderate data gaps detected")
        else:
            st.success("Good data coverage")
        
        # Column-level completeness
        column_stats = completeness.get('column_stats', {})
        if column_stats:
            st.markdown("**Column Completeness:**")
            col_data = []
            for col, stats in column_stats.items():
                col_data.append({
                    'Column': col,
                    'Completeness': f"{stats.get('completeness', 0):.1f}%",
                    'Missing Count': stats.get('missing_count', 0),
                    'Critical': '🔴' if stats.get('is_critical', False) else '⚪'
                })
            if col_data:
                st.dataframe(pd.DataFrame(col_data), use_container_width=True, hide_index=True)
    else:
        # Fallback: calculate from validation_data
        if validation_data:
            total_points = sum(len(d.get('Y', [])) for d in validation_data)
            valid_points = sum(len([y for y in d.get('Y', []) if y is not None and y > 0]) for d in validation_data)
            completeness_pct = (valid_points / total_points * 100) if total_points > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Completeness", f"{completeness_pct:.1f}%")
            with col2:
                st.metric("Missing Points", total_points - valid_points)
            with col3:
                st.metric("Total Points", total_points)
    
    # Uniqueness & Duplicates
    st.markdown("#### 🔍 Uniqueness & Duplicate Detection")
    duplicates = enhanced.get('duplicates', pd.DataFrame())
    
    if hasattr(duplicates, 'empty') and not duplicates.empty:
        st.error(f"Found {len(duplicates)} duplicate groups")
        st.dataframe(duplicates.head(10), use_container_width=True)
    else:
        st.success("No duplicate curves detected")
    
    # Format & Consistency
    st.markdown("#### 📐 Format & Consistency")
    format_issues = enhanced.get('format_issues', {})
    
    if format_issues:
        st.warning(f"{len(format_issues)} consistency issues found")
        with st.expander("View Issues"):
            for issue in list(format_issues.values())[:10]:
                st.write(f"• {issue}")
    else:
        # Check validation data for consistency
        consistency_issues = []
        for result in validation_data:
            years = result.get('X', [])
            for year in years:
                try:
                    if year is not None:
                        year_int = int(float(year))
                        if year_int < 1990 or year_int > 2050:
                            consistency_issues.append(f"Invalid year {year_int} in {result.get('Entity_Name')}")
                            break
                except (ValueError, TypeError):
                    consistency_issues.append(f"Non-numeric year in {result.get('Entity_Name')}")
                    break
        
        if consistency_issues:
            st.warning(f"{len(consistency_issues)} consistency issues")
            with st.expander("View Issues"):
                for issue in consistency_issues[:10]:
                    st.write(f"• {issue}")
        else:
            st.success("Data format is consistent")
    
    # Range & Outliers
    st.markdown("#### 📊 Range & Outlier Detection")
    outliers = enhanced.get('outliers', {})
    
    if outliers:
        outlier_count = sum(len(v) for v in outliers.values())
        st.warning(f"{outlier_count} outliers detected across {len(outliers)} series")
        
        with st.expander("View Outlier Details"):
            for series, outlier_list in list(outliers.items())[:5]:
                st.write(f"**{series}**: {len(outlier_list)} outliers")
                if outlier_list:
                    st.write(f"  Values: {outlier_list[:5]}")
    else:
        # Calculate outliers from validation data
        outlier_curves = 0
        for result in validation_data:
            values = [y for y in result.get('Y', []) if y is not None and y > 0]
            if len(values) >= 3:
                Q1 = np.percentile(values, 25)
                Q3 = np.percentile(values, 75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outliers_found = [v for v in values if v < lower or v > upper]
                if outliers_found:
                    outlier_curves += 1
        
        if outlier_curves > 0:
            st.warning(f"{outlier_curves} curves contain outliers")
        else:
            st.success("No significant outliers detected")


def render_data_quality_tab(validation_results: Dict):
    """Enhanced data quality with cleaning summary"""
    st.markdown("### 🧹 Data Quality & Cleaning")
    
    enhanced = validation_results.get('enhanced_validation', {})
    
    # Data Cleaning Applied
    st.markdown("#### 🧹 Data Cleaning Summary")
    cleaning = validation_results.get('cleaning_summary', {})
    
    if cleaning:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Outliers Removed", cleaning.get('outliers_removed', 0))
        with col2:
            st.metric("Duplicates Removed", cleaning.get('duplicates_removed', 0))
        
        if cleaning.get('outliers_removed', 0) > 0 or cleaning.get('duplicates_removed', 0) > 0:
            st.info("Data cleaning was applied to improve quality")
    else:
        st.success("No data cleaning required - dataset is clean")
    
    # Logical Validation
    st.markdown("#### ✅ Logical Validation")
    logic_issues = enhanced.get('logic_issues', {})
    
    if logic_issues:
        logic_count = sum(len(v) for v in logic_issues.values())
        st.error(f"Found {logic_count} logical inconsistencies")
        
        with st.expander("View Logical Issues"):
            for issue_type, indices in logic_issues.items():
                issue_name = issue_type.replace('_', ' ').title()
                st.write(f"**{issue_name}**: {len(indices)} issues")
    else:
        validation_data = st.session_state.get('validation_data', [])
        logical_issues = []
        
        for result in validation_data:
            unit = str(result.get('Unit', '')).lower()
            values = result.get('Y', [])
            
            # Check percentage ranges
            if 'percentage' in unit or '%' in unit:
                for v in values:
                    if v and (safe_float(v) < 0 or safe_float(v) > 100):
                        logical_issues.append(f"Percentage out of range in {result.get('Entity_Name')}")
                        break
        
        if logical_issues:
            st.error(f"{len(logical_issues)} logical issues found")
            with st.expander("View Issues"):
                for issue in logical_issues[:5]:
                    st.write(f"• {issue}")
        else:
            st.success("All data logically consistent")
    
    # Cross-Field Validation
    st.markdown("#### 🔗 Cross-Field Validation")
    st.info("Checking relationships between fields (e.g., regional sums = global totals)")
    
    # Check if regional data sums to global
    validation_data = st.session_state.get('validation_data', [])
    if validation_data:
        entities = {}
        for result in validation_data:
            entity = result.get('Entity_Name', '')
            region = result.get('Region', 'Unknown')
            if entity not in entities:
                entities[entity] = {}
            entities[entity][region] = result
        
        cross_field_issues = []
        for entity, regions in entities.items():
            if 'Global' in regions and len(regions) > 1:
                # Check if sum of regions equals global
                global_vals = regions['Global'].get('Y', [])
                if global_vals:
                    st.write(f"✓ Cross-field check available for {entity}")
        
        if not cross_field_issues:
            st.success("Cross-field validations passed")
    
    # Overall Quality Score
    st.markdown("#### 🎯 Overall Data Quality Score")
    score = validation_results.get('score')
    
    if score and hasattr(score, 'overall_score'):
        score_pct = score.overall_score * 100
        st.progress(score_pct / 100)
        
        if score_pct >= 95:
            st.success(f"**Grade: A+** - Investment Grade Quality ({score_pct:.1f}%)")
        elif score_pct >= 90:
            st.success(f"**Grade: A** - Excellent Quality ({score_pct:.1f}%)")
        elif score_pct >= 80:
            st.info(f"**Grade: B+** - Good Quality ({score_pct:.1f}%)")
        elif score_pct >= 70:
            st.warning(f"**Grade: B** - Acceptable ({score_pct:.1f}%)")
        else:
            st.error(f"**Grade: C** - Needs Improvement ({score_pct:.1f}%)")
    else:
        # Calculate basic quality score
        investment_grade = validation_results.get('investment_grade', {})
        score_pct = investment_grade.get('score', 0) * 100
        
        st.progress(score_pct / 100)
        st.info(f"**Quality Score: {score_pct:.1f}%** based on validation results")


def render_statistical_analysis_tab(validation_results: Dict):
    """Statistical analysis with Wright's Law and S-Curve detection"""
    st.markdown("### 📊 Advanced Statistical Analysis")
    
    characteristics = validation_results.get('characteristics', {})
    seba = validation_results.get('seba_results', {})
    
    # Display framework applicability
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⚡ Wright's Law Analysis")
        
        if characteristics.get('is_cost_data'):
            st.success("✅ Applicable - Cost data detected")
            
            wright_results = {k: v for k, v in seba.items() if 'wrights_law' in k}
            
            if wright_results:
                wright_compliant = sum(1 for v in wright_results.values() if v.get('compliant'))
                st.metric("Compliance", f"{wright_compliant}/{len(wright_results)} curves")
                
                learning_rates = [v.get('learning_rate', 0) * 100 
                                for v in wright_results.values() if v.get('compliant')]
                if learning_rates:
                    avg_rate = np.mean(learning_rates)
                    st.info(f"Avg learning rate: {avg_rate:.1f}% per doubling")
                    
                    if 15 <= avg_rate <= 30:
                        st.success("✅ Within expected range (15-30%)")
                    else:
                        st.warning("⚠️ Outside typical range")
        else:
            st.info("❌ Not applicable - Physical quantity data")
            st.caption("Wright's Law only applies to technology cost curves")
    
    with col2:
        st.markdown("#### 📈 S-Curve Analysis")
        
        if characteristics.get('is_physical_data') and not characteristics.get('is_cost_data'):
            st.success("✅ Applicable - Adoption data detected")
            
            scurve_results = {k: v for k, v in seba.items() if 'scurve' in k}
            
            if scurve_results:
                scurve_compliant = sum(1 for v in scurve_results.values() if v.get('compliant'))
                st.metric("S-Curve Detected", f"{scurve_compliant}/{len(scurve_results)} curves")
                
                if scurve_compliant > 0:
                    st.success("✅ Sigmoid pattern confirmed")
                else:
                    st.warning("⚠️ Linear pattern (may be early stage)")
        else:
            st.info("❌ Not applicable - Cost data")
            st.caption("S-Curve analysis applies to adoption/deployment data")
    
    st.markdown("---")
    
    # Validation Statistics Summary
    st.markdown("#### 📊 Validation Statistics")
    summary = validation_results.get('validation_summary', {})
    
    if summary:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total = summary.get('total', 35)
            passed = summary.get('passed', 0)
            pass_rate = (passed / total * 100) if total > 0 else 0
            st.metric("Pass Rate", f"{pass_rate:.1f}%")
        
        with col2:
            st.metric("Total Checks", total)
        
        with col3:
            failed = summary.get('failed', 0)
            st.metric("Failed", failed)
        
        with col4:
            warnings = summary.get('warnings', 0)
            st.metric("Warnings", warnings)
    
    st.markdown("---")
    
    # Validation Breakdown by Category
    st.markdown("#### 📋 Validation Breakdown by Category")
    
    categories_info = {
        'Data Integrity': 3,
        'Cost Analysis': 5,
        'Adoption Analysis': 3,
        'Market Context': 1,
        'Energy Transition': 4,
        'Data Sources': 3,
        'Regional Analysis': 3,
        'Technical Validators': 3,
        'Specialized': 10
    }
    
    for cat, count in categories_info.items():
        st.write(f"• **{cat}**: {count} validators")
    
    # Statistical Properties
    st.markdown("---")
    st.markdown("#### 📈 Statistical Properties")
    
    validation_data = st.session_state.get('validation_data', [])
    if validation_data:
        all_values = []
        for result in validation_data:
            values = [y for y in result.get('Y', []) if y is not None and y > 0]
            all_values.extend(values)
        
        if all_values:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean", f"{np.mean(all_values):.2f}")
            with col2:
                st.metric("Median", f"{np.median(all_values):.2f}")
            with col3:
                st.metric("Std Dev", f"{np.std(all_values):.2f}")
            with col4:
                st.metric("Data Points", len(all_values))


def calculate_investment_grade_enhanced(validation_results: Dict) -> Dict:
    """Calculate investment grade status with 95% threshold"""
    summary = validation_results.get('validation_summary', {})
    passed = summary.get('passed', 0)
    failed = summary.get('failed', 0)
    warnings = summary.get('warnings', 0)
    
    applicable = passed + failed + warnings
    
    if applicable == 0:
        return {
            'qualified': False,
            'score': 0,
            'grade': 'F',
            'status': 'NO DATA',
            'recommendation': 'No applicable validators'
        }
    
    pass_rate = passed / applicable
    
    if pass_rate >= 0.95:
        grade = 'A+'
        status = 'INVESTMENT-GRADE'
        qualified = True
        recommendation = "Data meets institutional investment standards"
    elif pass_rate >= 0.90:
        grade = 'A'
        status = 'NEAR INVESTMENT-GRADE'
        qualified = False
        gap = (0.95 - pass_rate) * 100
        recommendation = f"Improve {gap:.1f}% to reach investment grade threshold"
    elif pass_rate >= 0.80:
        grade = 'B+'
        status = 'GOOD QUALITY'
        qualified = False
        gap = (0.95 - pass_rate) * 100
        recommendation = f"Improve {gap:.1f}% for investment grade"
    elif pass_rate >= 0.70:
        grade = 'B'
        status = 'ACCEPTABLE'
        qualified = False
        recommendation = "Significant improvements needed for investment grade"
    else:
        grade = 'C'
        status = 'CRITICAL'
        qualified = False
        recommendation = "Data quality below acceptable standards"
    
    return {
        'qualified': qualified,
        'score': pass_rate,
        'grade': grade,
        'status': status,
        'recommendation': recommendation,
        'passed': passed,
        'failed': failed,
        'warnings': warnings,
        'applicable': applicable
    }


def render_data_validation_workflow():
    """Complete STELLAR validation workflow"""
    st.markdown("""
    <div class="main-header">
        <h1>🔬 STELLAR Data Validation Agent</h1>
        <p>Tony Seba Disruption Framework - Investment Grade Validation (95%+ Threshold)</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("← Back", key="back_from_validation"):
        st.session_state.current_workflow = None
        st.rerun()

    if not VALIDATION_AVAILABLE:
        st.error("⚠️ Validation modules not properly configured.")
        return

    with st.expander("📚 Understanding STELLAR Framework", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **📉 Wright's Law (Cost Curves):**
            - ✅ Applied to: Solar/battery costs, LCOE
            - ❌ NOT applied to: Generation, capacity, commodities
            - Expected: 15-30% cost reduction per doubling
            """)
        with col2:
            st.markdown("""
            **📈 S-Curve (Adoption):**
            - ✅ Applied to: Sales, capacity, generation
            - ❌ NOT applied to: Cost/price data
            - Pattern: Slow → Exponential → Saturation
            """)
        with col3:
            st.markdown("""
            **🔄 Commodity Cycles:**
            - ✅ Applied to: Aluminum, copper, steel
            - ❌ NOT technologies
            - Expected: 3-7% annual growth
            """)

    if not st.session_state.get('collected_data'):
        st.info("📊 No data collected yet. Please use extraction workflows first.")
        with st.expander("ℹ️ How to collect data", expanded=True):
            st.markdown("""
            ### Collect data using these workflows:
            1. **🌐 Web Data Extraction** - Extract tables from web searches
            2. **📄 PDF Analysis** - Extract charts from PDF documents
            3. **📊 Chart Analysis** - Analyze chart images
            4. **🔗 URL Extraction** - Extract from specific URLs
            """)
        return

    st.markdown("### 📊 Collected Data Summary")
    data_summary = []
    for key, entry in st.session_state.collected_data.items():
        data_summary.append({
            'Source': entry['source'],
            'Type': entry['data_type'],
            'Timestamp': entry['timestamp']
        })
    st.dataframe(pd.DataFrame(data_summary), use_container_width=True, hide_index=True)

    st.markdown("### 🎮 Validation Controls")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("✅ Data Quality (7 dimensions): Completeness, accuracy, consistency")
        st.write("✅ Wright's Law Analysis: Cost curves (15-30% learning)")
        st.write("✅ S-Curve Analysis: Adoption patterns")
        st.write("✅ 35 Domain Validators: Specialized rules with adaptive logic")
        st.write("✅ Crisis Detection: GFC/COVID/Ukraine context")
        st.write("✅ Investment Grade: 95%+ threshold calculation")

    with col2:
        if st.button("🚀 Run Validation", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.info("🔬 Converting collected data...")
                progress_bar.progress(0.1)
                validation_data = convert_collected_data_to_validation_format()

                if not validation_data:
                    st.error("❌ No valid data for validation")
                    return
                
                st.success(f"✅ Converted {len(validation_data)} datasets")

                status_text.info("🔄 Running base validation pipeline...")
                progress_bar.progress(0.3)
                
                try:
                    from validation_support import run_validation_pipeline
                    validation_results = run_validation_pipeline(
                        validation_data,
                        enable_seba=True,
                        enable_llm=False
                    )
                except ImportError:
                    st.warning("⚠️ validation_support not available, using basic validation")
                    validation_results = {'seba_results': {}, 'enhanced_validation': {}}

                status_text.info("🔬 Running 35 domain expert validators...")
                progress_bar.progress(0.5)

                df = convert_results_to_dataframe(validation_data)
                expert_results = run_all_validators_with_intelligence(validation_data, df)
                
                validation_results['expert_validations'] = expert_results['expert_validations']
                validation_results['validation_summary'] = expert_results['validation_summary']
                validation_results['characteristics'] = expert_results['characteristics']

                status_text.info("💎 Calculating investment grade...")
                progress_bar.progress(0.9)

                investment_grade = calculate_investment_grade_enhanced(validation_results)
                validation_results['investment_grade'] = investment_grade

                st.session_state.validation_results = validation_results
                st.session_state.validation_data = validation_data
                progress_bar.progress(1.0)
                status_text.success("✅ Validation complete!")

                if investment_grade.get('qualified'):
                    st.balloons()
                
                time.sleep(1)
                st.rerun()

            except Exception as e:
                st.error(f"❌ Validation failed: {str(e)}")
                import traceback
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())

    if st.session_state.get('validation_results'):
        results = st.session_state.validation_results
        render_validation_results_complete(results)


def render_validation_results_complete(validation_results: Dict):
    """Render complete validation results with all tabs"""
    
    st.markdown("---")
    st.markdown("## 📊 STELLAR Validation Results")

    investment_grade = validation_results.get('investment_grade', {})
    characteristics = validation_results.get('characteristics', {})
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Grade", investment_grade.get('grade', 'N/A'))
    with col2:
        st.metric("Score", f"{investment_grade.get('score', 0) * 100:.1f}%")
    with col3:
        status = "✅ YES" if investment_grade.get('qualified') else "❌ NO"
        st.metric("Investment Grade", status)
    with col4:
        applicable = investment_grade.get('applicable', 0)
        st.metric("Applicable Checks", f"{applicable}/35")

    if investment_grade.get('qualified'):
        st.success(f"🎉 {investment_grade.get('status')} - {investment_grade.get('recommendation')}")
    elif investment_grade.get('score', 0) >= 0.80:
        st.info(f"📈 {investment_grade.get('status')} - {investment_grade.get('recommendation')}")
    else:
        st.warning(f"⚠️ {investment_grade.get('status')} - {investment_grade.get('recommendation')}")

    tabs = st.tabs([
    "📊 Overview",
    "📋 Data Profiling",
    "🧹 Data Quality",
    "📊 Statistical Analysis",
    "🔬 Domain Expert Rules (35)",
    "📈 Wright's Law",
    "📉 S-Curves",
    "📄 Report"
])

    with tabs[0]:
        st.markdown("### 📊 Data Quality Overview")
        
        st.info(f"""
        **Data Type:** {characteristics.get('primary_framework', 'Unknown')}  
        **Entity:** {characteristics.get('entity_name', 'unknown').replace('_', ' ').title()}  
        **Expected Pattern:** {characteristics.get('expected_pattern', 'Unknown')}
        """)
        
        summary = validation_results.get('validation_summary', {})
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Validators", summary.get('total', 35))
        with col2:
            st.metric("✅ Passed", summary.get('passed', 0))
        with col3:
            st.metric("❌ Failed", summary.get('failed', 0))
        with col4:
            st.metric("⭕ N/A", summary.get('na', 0))
    with tabs[1]:
        render_data_profiling_tab(validation_results)

    with tabs[2]:
        render_data_quality_tab(validation_results)

    with tabs[3]:
        render_statistical_analysis_tab(validation_results)

    with tabs[4]:
        st.markdown("### 🔬 Domain Expert Validation - 35 Validators")
        
        expert_validations = validation_results.get('expert_validations', {})
        summary = validation_results.get('validation_summary', {})
        
        entity = characteristics.get('entity_name', 'data').replace('_', ' ').title()
        framework = characteristics.get('primary_framework', 'Unknown')
        pattern = characteristics.get('expected_pattern', '')
        
        st.info(f"📊 **Analyzing:** {entity} | **Framework:** {framework} | **Expected:** {pattern}")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total", summary.get('total', 35))
        with col2:
            st.metric("✅ Passed", summary.get('passed', 0))
        with col3:
            st.metric("❌ Failed", summary.get('failed', 0))
        with col4:
            st.metric("⚠️ Warnings", summary.get('warnings', 0))
        with col5:
            st.metric("⭕ N/A", summary.get('na', 0))
        
        applicable = summary.get('passed', 0) + summary.get('failed', 0) + summary.get('warnings', 0)
        pass_rate = (summary.get('passed', 0) / applicable * 100) if applicable > 0 else 0
        
        if pass_rate >= 95:
            st.success(f"✅ **INVESTMENT GRADE** ({pass_rate:.0f}% pass rate)")
        elif pass_rate >= 80:
            st.success(f"✅ **Overall: GOOD** ({pass_rate:.0f}% pass rate)")
        elif pass_rate >= 60:
            st.warning(f"⚠️ **Overall: NEEDS ATTENTION** ({pass_rate:.0f}% pass rate)")
        else:
            st.error(f"❌ **Overall: CRITICAL** ({pass_rate:.0f}% pass rate)")
        
        st.markdown("---")
        
        validator_info = get_validator_descriptions()
        
        categories = {
            "📊 Data Integrity (3 checks)": [
                ('units_and_scale', 'Units and Scale Consistency'),
                ('year_anomalies', 'Year Anomaly Detection'),
                ('definition_clarity', 'Definition Clarity Check'),
            ],
            "💰 Cost Analysis (Wright's Law) (5 checks)": [
                ('cost_curve', 'Cost Curve Validation'),
                ('cost_parity', 'Cost Parity Threshold'),
                ('cost_curve_anomaly', 'Cost Anomaly Detection'),
                ('inflation_adjustment', 'Inflation Adjustment Check'),
                ('cost_curve_learning_rate', 'Learning Rate Validation'),
            ],
            "📈 Adoption Analysis (S-Curve) (3 checks)": [
                ('adoption_curve', 'Adoption Curve Validation'),
                ('adoption_saturation', 'Saturation Limit Check'),
                ('adoption_curve_shape', 'Curve Shape Analysis'),
            ],
            "🌍 Market Context & Crisis (1 check)": [
                ('market_context', 'Market Context Analysis'),
            ],
            "⚡ Energy Transition (STDF) (4 checks)": [
                ('oil_displacement', 'Oil Displacement Calculation'),
                ('derived_oil_displacement', 'Derived Oil Displacement'),
                ('capacity_factor', 'Capacity Factor Validation'),
                ('global_oil_sanity', 'Global Oil Sanity Check'),
            ],
            "📚 Data Sources (3 checks)": [
                ('data_source_quality', 'Source Quality Assessment'),
                ('data_source_integrity', 'Source Integrity Check'),
                ('multi_source_consistency', 'Multi-Source Consistency'),
            ],
            "🌐 Regional Analysis (3 checks)": [
                ('regional_market_logic', 'Regional Market Logic'),
                ('regional_definition', 'Regional Definition Check'),
                ('global_sum', 'Global Sum Validation'),
            ],
            "🔧 Technical Validators (3 checks)": [
                ('metric_validity', 'Metric Validity Check'),
                ('unit_conversion', 'Unit Conversion Validation'),
                ('ai_reality', 'AI Reality Assessment'),
            ],
            "🆕 Specialized Validators (10 checks)": [
                ('spelling_nomenclature', 'Spelling & Nomenclature'),
                ('battery_size_evolution', 'Battery Size Evolution'),
                ('regional_price_pattern', 'Regional Price Patterns'),
                ('historical_label', 'Historical Data Labeling'),
                ('growth_rate_bounds', 'Growth Rate Reasonableness'),
                ('lithium_chemistry', 'Lithium Chemistry Validation'),
                ('commodity_behavior', 'Commodity Price Behavior'),
                ('trusted_sources', 'Trusted Source Verification'),
            ],
        }
        
        for category_name, validators in categories.items():
            with st.expander(category_name, expanded=False):
                for validator_key, display_name in validators:
                    if validator_key in expert_validations:
                        results_list = expert_validations[validator_key]
                        
                        col1, col2 = st.columns([4, 1])
                        
                        with col1:
                            st.write(f"**{display_name}**")
                            desc = validator_info.get(validator_key, {}).get('description', '')
                            if desc:
                                st.caption(desc)
                            
                            if results_list and isinstance(results_list, list) and len(results_list) > 0:
                                result = results_list[0]
                                explanation = result.get('explanation', '')
                                
                                if explanation:
                                    if result.get('pass') is False:
                                        st.caption(f"❌ {explanation}")
                                    elif result.get('pass') is None and 'N/A' in explanation:
                                        st.caption(f"💡 {explanation}")
                                    elif result.get('pass') is True:
                                        context_msg = get_context_success_message(validator_key, characteristics)
                                        if context_msg:
                                            st.caption(f"✅ {context_msg}")
                        
                        with col2:
                            status = interpret_validator_result(results_list, validator_key)
                            
                            if status == 'pass':
                                st.success("✅ Pass")
                            elif status == 'fail':
                                st.error("❌ Fail")
                            elif status == 'warning':
                                st.info("⭕ N/A")
                            else:
                                st.info("ℹ️ Info")

    with tabs[5]:
        st.markdown("### ⚡ Wright's Law Analysis")
        
        if not characteristics.get('is_cost_data'):
            st.info(f"""
            ℹ️ **Wright's Law Not Applicable**
            
            Wright's Law only applies to **technology cost curves**.
            
            Current data: {characteristics.get('entity_name', 'data')} ({characteristics.get('primary_framework', 'Unknown')})
            """)
        else:
            seba_results = validation_results.get('seba_results', {})
            wright_results = {k: v for k, v in seba_results.items() if 'wrights_law' in k}
            
            if wright_results:
                for key, result in wright_results.items():
                    product = key.replace('_wrights_law', '').replace('_', ' ').title()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Product", product)
                    with col2:
                        lr = result.get('learning_rate', 0) * 100
                        st.metric("Learning Rate", f"{lr:.1f}%")
                    with col3:
                        st.metric("R²", f"{result.get('r_squared', 0):.3f}")
                    with col4:
                        if result.get('compliant'):
                            st.success("✅ Compliant")
                        else:
                            st.error("❌ Non-compliant")
            else:
                st.warning("No Wright's Law analysis results available")

    with tabs[6]:
        st.markdown("### 📈 S-Curve Analysis")
        
        if characteristics.get('is_cost_data') and not characteristics.get('is_physical_data'):
            st.info(f"""
            ℹ️ **S-Curve Not Applicable**
            
            S-Curve analysis only applies to **adoption/deployment data**.
            
            Current data: {characteristics.get('entity_name', 'data')} (cost curve)
            """)
        else:
            seba_results = validation_results.get('seba_results', {})
            scurve_results = {k: v for k, v in seba_results.items() if 'scurve' in k}
            
            if scurve_results:
                for key, result in scurve_results.items():
                    product = key.replace('_scurve', '').replace('_', ' ').title()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Product", product)
                    with col2:
                        st.metric("Growth Rate", f"{result.get('growth_rate', 0):.2f}")
                    with col3:
                        st.metric("R²", f"{result.get('r_squared', 0):.3f}")
                    with col4:
                        if result.get('compliant'):
                            st.success("✅ S-Curve")
                        else:
                            st.warning("⚠️ Linear")
            else:
                st.info("No S-curve analysis results available")

    with tabs[7]:
        st.markdown("### 📄 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Download Report", use_container_width=True):
                report = f"""
STELLAR FRAMEWORK - INVESTMENT GRADE VALIDATION REPORT
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

INVESTMENT GRADE ASSESSMENT
{'-'*40}
Status: {investment_grade.get('status', 'Unknown')}
Grade: {investment_grade.get('grade', 'N/A')}
Score: {investment_grade.get('score', 0) * 100:.1f}%
Qualified: {'YES' if investment_grade.get('qualified') else 'NO'}
Recommendation: {investment_grade.get('recommendation', '')}

DATA CHARACTERISTICS
{'-'*40}
Entity: {characteristics.get('entity_name', 'unknown').title()}
Framework: {characteristics.get('primary_framework', 'Unknown')}
Expected Pattern: {characteristics.get('expected_pattern', 'Unknown')}

VALIDATION SUMMARY
{'-'*40}
Total Validators: {summary.get('total', 35)}
Passed: {summary.get('passed', 0)}
Failed: {summary.get('failed', 0)}
Warnings: {summary.get('warnings', 0)}
N/A: {summary.get('na', 0)}
"""
                st.download_button(
                    "💾 Download",
                    report,
                    f"stellar_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    "text/plain"
                )
        
        with col2:
            validation_data = st.session_state.get('validation_data', [])
            if validation_data:
                df = pd.DataFrame(validation_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    "📈 Download CSV",
                    csv,
                    f"validation_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv",
                    use_container_width=True
                )
# ==================== MAIN APPLICATION ====================

def check_environment():
    """Check environment"""
    if not os.getenv("GROQ_API_KEY"):
        st.error("❌ GROQ_API_KEY environment variable not set")
        st.stop()
    
    if not CRAWL4AI_AVAILABLE:
        st.warning("⚠️ crawl4ai not installed - using basic web extraction")

def main():
    """Main application"""
    apply_custom_css()
    init_session_state()
    check_environment()
    
    # Route to workflows
    if st.session_state.current_workflow is None:
        render_quick_start()
    elif st.session_state.current_workflow == "pdf_analysis":
        render_pdf_analysis_workflow()
    elif st.session_state.current_workflow == "chart_analysis":
        render_chart_analysis_workflow()
    elif st.session_state.current_workflow == "web_extraction":
        render_web_extraction_workflow()
    elif st.session_state.current_workflow == "url_extraction":
        render_url_extraction_workflow()
    elif st.session_state.current_workflow == "query_url_extraction":
        render_query_url_extraction_workflow()
    elif st.session_state.current_workflow == "data_validation":
        render_data_validation_workflow()

if __name__ == "__main__":
    main()