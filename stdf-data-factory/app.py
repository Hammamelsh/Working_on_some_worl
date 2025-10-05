import streamlit as st
import os
import tempfile
import hashlib
import json
from typing import Dict, Any, List
from utils import process_pdf_time_series
from utils.api_clients import extract_structured_data_from_markdown
from utils.image_processor import convert_image_to_base64
from openai import OpenAI
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Set page config
st.set_page_config(
    page_title="PDF Time Series Analyzer",
    page_icon="📊",
    layout="wide"
)


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
        
        # Auto-detect if data is likely temporal (years, dates)
        is_temporal = False
        if x_values and isinstance(x_values[0], (int, str)):
            # Check if X values look like years or dates
            str_x = str(x_values[0])
            if str_x.isdigit() and len(str_x) == 4 and 1900 <= int(str_x) <= 2100:
                is_temporal = True
            elif any(char in str_x for char in ['-', '/', ':']):
                is_temporal = True
        
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


def display_file_structure(output_dir: str) -> None:
    """Display generated file structure."""
    if os.path.exists(output_dir):
        st.header("📁 Generated Files")
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), output_dir)
                st.write(f"📄 {rel_path}")


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


def display_upload_info(uploaded_file) -> None:
    """Display information about the uploaded file."""
    st.success(f"✅ PDF uploaded successfully: {uploaded_file.name}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("File Name", uploaded_file.name)
    with col2:
        st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")


def analyze_chart_image(image_path: str, image_name: str = None) -> Dict[str, Any]:
    """Analyze a single chart image and return structured data."""
    try:
        # Create image analysis cache directory
        cache_dir = "extracted_data/image_analysis"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate cache file name based on image name
        if image_name:
            # Clean the image name for file system (remove dots, spaces, special chars)
            clean_name = "".join(c for c in image_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            clean_name = clean_name.replace(' ', '_')
        else:
            # Generate hash from image path as fallback
            clean_name = hashlib.md5(image_path.encode()).hexdigest()[:8]
        
        cache_file = os.path.join(cache_dir, f"{clean_name}_analysis.json")
        
        # Debug: Show what we're looking for
        st.info(f"🔍 Looking for cache file: {clean_name}_analysis.json")
        
        # Check if analysis already exists
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_results = json.load(f)
                st.success(f"✅ Using cached analysis from: {os.path.basename(cache_file)}")
                return cached_results
            except Exception as e:
                st.error(f"⚠️ Failed to load cached analysis from {cache_file}: {e}")
                # Continue with fresh analysis
        else:
            # Try to find similar cache files if exact match not found
            if os.path.exists(cache_dir):
                existing_files = [f for f in os.listdir(cache_dir) if f.endswith('_analysis.json')]
                st.info(f"📂 Available cache files: {existing_files}")
                
                # Try to find a close match based on image name similarity
                if image_name:
                    base_name = os.path.splitext(image_name)[0]  # Remove extension
                    # Create a very flexible normalized version for matching
                    normalized_image = ''.join(c.lower() for c in base_name if c.isalnum())
                    st.info(f"🔤 Searching for matches with normalized name: {normalized_image}")
                    
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
                            
                            if common_chars >= similarity_threshold:
                                similar_cache_file = os.path.join(cache_dir, existing_file)
                                try:
                                    with open(similar_cache_file, 'r', encoding='utf-8') as f:
                                        cached_results = json.load(f)
                                    st.success(f"✅ Found similar cached analysis: {existing_file}")
                                    st.info(f"📁 Loaded from: {similar_cache_file}")
                                    return cached_results
                                except Exception as e:
                                    st.error(f"⚠️ Failed to load similar cache {existing_file}: {e}")
                                    continue
        
        # Get API key from environment
        XAI_API_KEY = os.getenv("XAI_API_KEY")
        if not XAI_API_KEY:
            return {"error": "XAI_API_KEY environment variable not set"}
        
        # Initialize OpenAI client for Grok
        client = OpenAI(
            api_key=XAI_API_KEY,
            base_url="https://api.x.ai/v1",
        )
        
        # Convert image to base64
        image_base64 = convert_image_to_base64(image_path)
        if not image_base64:
            return {"error": "Failed to convert image to base64"}
        
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
        
        # Call Grok API
        completion = client.chat.completions.create(
            model="grok-4",
            messages=messages,
        )
        
        # Get response content
        response_content = completion.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            # Remove any markdown code block formatting if present
            if response_content.startswith('```json'):
                response_content = response_content[7:]
            if response_content.endswith('```'):
                response_content = response_content[:-3]
            if response_content.startswith('```'):
                response_content = response_content[3:]
            
            structured_data = json.loads(response_content.strip())
            
            # Save successful analysis to cache file
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(structured_data, f, indent=2, ensure_ascii=False)
                st.success(f"✅ Analysis cached for future use")
            except Exception as e:
                st.warning(f"⚠️ Failed to save analysis cache: {e}")
                # Don't fail the whole process if caching fails
            
            return structured_data
            
        except json.JSONDecodeError as e:
            error_result = {"error": f"Failed to parse JSON response: {e}", "raw_response": response_content}
            # Don't cache failed results
            return error_result
            
    except Exception as e:
        return {"error": str(e)}


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


def main():
    """Main Streamlit application."""
    st.title("📊 PDF Time Series Analyzer")
    st.markdown("Upload a PDF or chart image to extract and analyze time series data from charts and graphs.")
    
    # Check for required environment variables
    if not check_api_keys():
        st.stop()
    
    # Create uploads directory if it doesn't exist
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    
    # Create tabs for different input types
    tab1, tab2 = st.tabs(["📄 PDF Analysis", "📊 Chart Image Analysis"])
    
    with tab1:
        st.header("📤 Upload PDF")
        st.markdown("Upload a PDF document containing multiple charts and graphs.")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF containing charts, graphs, or time series visualizations",
            key="pdf_uploader"
        )
        
        if uploaded_file is not None:
            # Save and display upload info
            pdf_path = save_uploaded_file(uploaded_file, uploads_dir)
            display_upload_info(uploaded_file)
            
            # Process button
            if st.button("🚀 Process PDF", type="primary", use_container_width=True):
                with st.spinner("Processing PDF for time series data..."):
                    
                    # Process PDF with UI feedback
                    results = process_pdf_with_ui_feedback(pdf_path)
                    
                    if results:
                        # Display results
                        display_processing_summary(results)
                        
                        # Display chart data if available
                        if results.get('structured_data'):
                            display_chart_data(results['structured_data'])
                        
                        # Display output directory info
                        output_dir = os.path.join("extracted_data", results['pdf_name'])
                        st.info(f"📁 All results saved to: `{output_dir}/`")
                        
                        # Show file structure
                        display_file_structure(output_dir)
                        
                        # Provide download option for structured data
                        if results.get('structured_data', {}).get('charts'):
                            import json
                            json_str = json.dumps(results['structured_data'], indent=2)
                            st.download_button(
                                label="📥 Download Structured Data (JSON)",
                                data=json_str,
                                file_name=f"{results['pdf_name']}_structured_data.json",
                                mime="application/json"
                            )
    
    with tab2:
        st.header("📊 Upload Chart Image")
        st.markdown("Upload a single chart or graph image for direct analysis.")
        
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a chart, graph, or visualization image",
            key="image_uploader"
        )
        
        if uploaded_image is not None:
            # Display the uploaded image
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(uploaded_image, caption=f"Uploaded: {uploaded_image.name}", use_column_width=True)
            
            with col2:
                st.metric("File Name", uploaded_image.name)
                st.metric("File Size", f"{uploaded_image.size / 1024:.1f} KB")
            
            # Process button for image
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
                            
                            # Display results
                            if results.get('charts'):
                                display_chart_data(results)
                                
                                # Provide download option for structured data
                                import json
                                json_str = json.dumps(results, indent=2)
                                st.download_button(
                                    label="📥 Download Chart Data (JSON)",
                                    data=json_str,
                                    file_name=f"{uploaded_image.name}_chart_data.json",
                                    mime="application/json"
                                )
                            else:
                                st.warning("⚠️ No charts detected in the image")
                    
                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_image_path):
                            os.unlink(temp_image_path)
    
    # Add info section
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        This application processes documents and images to extract time series data from charts and graphs:
        
        ## 📄 PDF Analysis
        1. **PDF Upload**: Upload your PDF file containing charts or graphs
        2. **Image Extraction**: Each page is rendered as a high-resolution image
        3. **AI Analysis**: Images are analyzed using Grok AI to identify time series elements
        4. **Data Extraction**: Structured data is extracted using Groq AI
        5. **Results Display**: View extracted charts, data points, and download JSON results
        
        ## 📊 Chart Image Analysis
        1. **Image Upload**: Upload a single chart or graph image (PNG, JPG, JPEG)
        2. **Direct Analysis**: Image is analyzed immediately using Grok AI
        3. **Instant Results**: View extracted data and download JSON immediately
        
        **Supported Content:**
        - Line charts, bar charts, scatter plots
        - Time series visualizations
        - Any graph with extractable numerical data
        
        **File Storage:**
        - Uploaded PDFs: `uploads/` folder
        - Extracted data: `extracted_data/{pdf_name}/` folder
        """)


if __name__ == "__main__":
    main()